import random
from contextlib import contextmanager

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel

from ld4pg.modules.ema import LitEma
from ld4pg.util import default, extract_into_tensor

__conditioning_keys__ = {
    'concat': 'c_concat',
    'crossattn': 'c_crossattn',
    'adm': 'y'
}


class DDPM(pl.LightningModule):
    def __init__(
            self,
            model: DiffusionTransformer,
            encoder: PreTrainedModel,
            cfg: DictConfig,
            max_seq_len=64,
            # enc_dec_model: PreTrainedModel = None,
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
        self.save_hyperparameters(ignore=["model", "encoder"])
        self.cfg = cfg

        # sampling related parameters
        self.num_timesteps = int(timesteps)
        # self.sampling_timesteps = default(sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps

        # enc_dec_model = BartForConditionalGeneration.from_pretrained(cfg.enc_dec_model)
        self.encoder = encoder
        self.semantic_encoder = self.encoder
        for f_model in [self.encoder, self.semantic_encoder]:
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
        # self.self_condition = self.diffusion_model.self_condition
        self.self_condition = False

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

        self.normalize = cfg.normalize_latent
        self.scale_mean = cfg.scale_mean
        self.scale_factor = cfg.scale_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # sqrt(a_t)*x + sqrt(1-a_t)*noise
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, mask, t, condition, condition_mask, noise=None):
        # 对 x 归一化
        if self.normalize:
            x_start = self.normalize_latent(x_start)

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
        pred_noise, pred_x_start = self.diffusion_model_predictions(x_noisy, mask, t, condition, condition_mask)

        # 计算最终的 loss，只考虑非mask的部分
        # loss: bsz x max_seqlen x hidden
        # mask: bsz x max_seqlen
        loss = self.loss_fn(pred_x_start, x_start, reduction='none')
        loss = torch.masked_select(loss, ~mask.unsqueeze(-1).expand(loss.shape).bool()).mean()
        log_prefix = 'train' if self.training else 'val'
        loss_dict = {}
        loss_dict.update({f"{log_prefix}/loss": loss})
        return loss, loss_dict


class LatentDiffusion(DDPM):
    def __init__(self):
        super().__init__()


class DiffusionWrapper(pl.LightningModule):
    def __init__(
            self,
            diffusion_model_config: DictConfig,
            condition_key: str = 'crossattn',
    ):
        super().__init__()
