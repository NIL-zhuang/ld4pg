import random
from omegaconf import DictConfig
from contextlib import contextmanager

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from rich.progress import track
# from einops import reduce, rearrange
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import get_scheduler

from ld4pg.config import *
from ld4pg.model.diffusion_transformer import DiffusionTransformer
from ld4pg.module.ema import LitEma
from ld4pg.module.noise_schedule import linear_beta_schedule, cosine_beta_schedule
from ld4pg.module.utils import extract
from ld4pg.optim.optimizer import get_adamW_optimizer
from ld4pg.utils import default


def get_beta_schedule(beta_schedule, timesteps):
    if beta_schedule == 'linear':
        betas = linear_beta_schedule(timesteps)
    elif beta_schedule == 'cosine':
        betas = cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f'unknown beta schedule {beta_schedule}')
    return betas


class GaussianDiffusion(pl.LightningModule):
    def __init__(
            self,
            model: DiffusionTransformer,
            cfg: DictConfig,
            max_seq_len,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l2',
            objective='pred_noise',
            beta_schedule='cosine',
            p2_loss_weight_gamma=0.,
            # p2 loss weight, from https://arxiv.org/abs/2204.00227
            # 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_k=1,
            ddim_sampling_eta=1.,
            use_ema: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.args = cfg

        # sampling related parameters
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps

        enc_dec_model = BartForConditionalGeneration.from_pretrained(cfg.enc_dec_model)
        self.tokenizer = BartTokenizer.from_pretrained(cfg.enc_dec_model)
        self.encoder = enc_dec_model.get_encoder()
        self.decoder = enc_dec_model.get_decoder()
        self.semantic_encoder = self.encoder
        for f_model in [self.encoder, self.decoder, self.semantic_encoder]:
            for p in f_model.parameters():
                p.requires_grad = False

        self.diffusion_model = model
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.diffusion_model, decay=cfg.ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}")

        if self.diffusion_model.conditional and self.diffusion_model.unconditional_prob > 0:
            self.unconditional_bernoulli = torch.distributions.Bernoulli(probs=self.diffusion_model.unconditional_prob)

        self.latent_dim = self.diffusion_model.latent_dim
        self.self_condition = self.diffusion_model.self_condition

        self.max_seq_len = max_seq_len
        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'
        # self.loss_type = loss_type
        self.ddim_sampling_eta = ddim_sampling_eta

        self.learning_rate = cfg.learning_rate
        self.scheduler = cfg.scheduler
        self.warmup_steps = cfg.lr_warmup_steps
        self.train_steps = cfg.num_train_steps

        self.loss_type = loss_type
        # self.loss_fn = self.get_loss_fn()

        self.normalize = cfg.normalize_latent

        self.register_schedule(beta_schedule, p2_loss_weight_gamma, p2_loss_weight_k, timesteps)

    def register_schedule(self, beta_schedule, p2_loss_weight_gamma, p2_loss_weight_k, timesteps):
        betas = get_beta_schedule(beta_schedule, timesteps)
        # alpha_t = PI_{i=1}^{t} (1 - beta_i)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        # calculate p2 re-weighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def diffusion_model_predictions(self, x, mask, t, condition, condition_mask, x_self_cond=None):
        model_output = self.diffusion_model(x, mask, t, condition, condition_mask, x_self_cond)

        pred_noise, x_start = None, None
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

        # return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, condition, condition_mask):
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        # 从高斯噪声中 sample 一个样本, size 是 bsz x max_seqlen x dim
        batch_size = condition.shape[0]
        latent = torch.randn(batch_size, self.max_seq_len, self.latent_dim)
        mask = [[True] * self.max_seq_len for _ in batch_size]

        x_start = None
        for time, time_next in track(time_pairs, description='sampling loop'):
            time_cond = torch.full((batch_size,), time, dtype=torch.long)
            # 通过 self-condition 实现从 T 到 T-1的采样过程
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start = self.diffusion_model_predictions(latent, mask, time_cond, condition, condition_mask, self_cond)

            if time_next < 0:
                # sample 结束，将 latent 表示为对应的 x_start
                latent = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(latent)
            latent = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return latent, mask

    @torch.no_grad()
    def sample(self, condition=None, condition_mask=None):
        # TODO Create mask that controls length
        # TODO Implement for p_sample_loop
        return self.ddim_sample(condition, condition_mask)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # sqrt(a_t)*x + sqrt(1-a_t)*noise
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, mask, t, condition, condition_mask, noise=None):
        # 对 x 归一化
        if self.normalize:
            x_start = normalize_latent(x_start)

        # noise sample 获取时间步为 t 时刻对应模型的噪声强度
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 设置一定概率的 unconditional 生成，也就把对应 condition 序列的 mask 都设置为 0
        # todo 此处可能有问题
        if self.diffusion_model.conditional and self.diffusion_model.unconditional_prob > 0:
            unconditional_mask = self.unconditional_bernoulli.sample([condition.shape[0]]).bool()
            condition_mask[unconditional_mask, :] = 0

        # 在最一开始，我们把self-condition设置成false，所以这里不会执行
        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                pred_noise, pred_x_start = self.diffusion_model_predictions(x_noisy, mask, t, condition, condition_mask, None)
                x_self_cond = pred_x_start.detatch()

        # predict and take gradient step
        pred_noise, pred_x_start = self.diffusion_model_predictions(x_noisy, mask, t, condition, condition_mask, x_self_cond)
        if self.normalize:
            pred_x_start = denormalize_latent(pred_x_start)

        # 计算最终的 loss，只考虑非mask的部分
        # loss: bsz x max_seqlen x hidden
        # mask: bsz x max_seqlen
        loss = self.loss_fn(pred_x_start, x_start, reduction='none')
        loss = torch.masked_select(loss, ~mask.unsqueeze(-1).expand(loss.shape).bool()).mean()
        log_prefix = 'train' if self.training else 'val'
        loss_dict = {}
        loss_dict.update({f"{log_prefix}/loss": loss})
        return loss, loss_dict

    def shared_step(self, batch, batch_idx):
        latent, latent_mask, condition, condition_mask = self.get_input(batch)
        loss, loss_dict = self(latent, latent_mask, condition, condition_mask)
        return loss, loss_dict

    def get_input(self, batch):
        # 将源端句 latent 作为 condition，目标端句作为 paraphrase
        src = batch['source_text_input_ids']
        src_mask = batch['source_text_attention_mask']
        label = batch['labels']
        label_mask = batch['labels_attention_mask']
        label[:, 0] = 0
        latent = self.encoder(label, attention_mask=batch['labels_attention_mask']).last_hidden_state
        condition = self.semantic_encoder(src, attention_mask=batch['source_text_attention_mask']).last_hidden_state
        return latent, label_mask, condition, src_mask

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.diffusion_model.parameters())
            self.model_ema.copy_to(self.diffusion_model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
            try:
                yield None
            finally:
                if self.use_ema:
                    self.model_ema.restore(self.diffusion_model.parameters())
                    if context is not None:
                        print(f"{context}: Restored training weights")

    def forward(self, latent, mask, condition, condition_mask, *args, **kwargs):
        # 时间步采样
        t = torch.randint(0, self.num_timesteps, (latent.shape[0],), device=self.device).long()
        return self.p_losses(latent, mask, t, condition, condition_mask, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, *args, **kwargs) -> None:
        if self.use_ema:
            self.model_ema(self.diffusion_model)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, loss_dict_no_ema = self.shared_step(batch, batch_idx)
        with self.ema_scope():
            loss_ema, loss_dict_ema = self.shared_step(batch, batch_idx)
            loss_dict_ema = {f"{key}_ema": value for key, value in loss_dict_ema.items()}
        self.log_dict(loss_dict_no_ema, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(loss_dict_ema, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        # 这里只优化 diffusion model 和 均值 / 方差
        optimizer = get_adamW_optimizer(
            self.diffusion_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )
        scheduler = get_scheduler(
            name=self.scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.train_steps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'loss_ema'
            }
        }

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))


def normalize_latent(x_start):
    return (x_start - BART_ENCODER_MEAN) / BART_ENCODER_STD


def denormalize_latent(x_start):
    return x_start * BART_ENCODER_STD + BART_ENCODER_MEAN
