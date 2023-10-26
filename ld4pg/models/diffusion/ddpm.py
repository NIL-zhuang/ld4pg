from contextlib import contextmanager
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
from einops import repeat
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutput

from ld4pg.config import SAMPLE_STRATEGY
from ld4pg.models.diffusion.ddim import DDIMSampler
from ld4pg.models.diffusion.dpm_solver import DPMSolverSampler
from ld4pg.modules.diffusion_modules.diffusion_model import DenoisingTransformer
from ld4pg.modules.diffusion_modules.util import make_beta_schedule, extract_into_tensor, noise_like
from ld4pg.modules.ema import LitEma
from ld4pg.util import default, disabled_train

DDPM_SAMPLE_STRATEGY = SAMPLE_STRATEGY['beam1']

__conditioning_keys__ = {
    'concat': 'c_concat',
    'crossattn': 'c_crossattn',
    'adm': 'y'
}


class LatentDiffusion(pl.LightningModule):
    def __init__(
            self,
            model_cfg: DictConfig,
            first_stage_model: PreTrainedModel,
            cond_stage_model: PreTrainedModel,
            first_stage_tokenizer: PreTrainedTokenizer,
            condition_key: str = 'crossattn',
            beta_schedule: str = 'linear',
            parameterization: str = 'x0',
            loss_type: str = 'l1',
            timesteps: int = 1000,
            max_seqlen: int = 64,
            use_ema: bool = True,
            scale_factor=1.0,
            scale_mean=0.0,
            learning_rate=1.0e-4,
            unconditional_prob=0.1,
            normalize=True,
            learn_logvar: bool = False,
            l_simple_weight: float = 1.,
            original_elbo_weight: float = 0.,
            v_posterior: float = 0.,
            log_every_t: int = 100,
            sample_strategy=DDPM_SAMPLE_STRATEGY,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['first_stage_model', 'cond_stage_model', 'first_stage_tokenizer'])
        # self.first_stage_model = first_stage_model.eval()
        # note: this is a hack to get the encoder of the first stage model
        self.first_stage_model = first_stage_model
        self.cond_stage_model = self.first_stage_model.get_encoder()
        # self.cond_stage_model = cond_stage_model.eval()
        self.first_stage_tokenizer = first_stage_tokenizer
        for f_model in (self.first_stage_model, self.cond_stage_model):
            f_model.train = disabled_train
            for param in f_model.parameters():
                param.requires_grad = False

        self.model = DiffusionWrapper(model_cfg, condition_key=condition_key)
        self.num_timesteps = timesteps
        self.max_seqlen = max_seqlen
        self.latent_dim = model_cfg.params.latent_dim
        self.parameterization = parameterization
        self.loss_type = loss_type
        self.schedule = beta_schedule

        self.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.clip_denoised = False
        self.normalize = normalize
        self.scale_factor = scale_factor
        self.scale_mean = scale_mean

        self.unconditional_prob = unconditional_prob
        if self.unconditional_prob > 0.:
            self.unconditional_bernoulli = torch.distributions.Bernoulli(probs=self.unconditional_prob)
            self.unconditional_token_emb = nn.Parameter(
                torch.mean(
                    self.first_stage_model.get_encoder().get_input_embeddings().weight,
                    dim=0
                ))

        self.learning_rate = learning_rate

        self.v_posterior = v_posterior
        self.l_simple_weight = l_simple_weight
        self.original_elbo_weight = original_elbo_weight
        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps)

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=0., size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        self.sample_cfg = sample_strategy
        self.log_every_t = log_every_t
        self.valid_print = True

    def register_schedule(
            self,
            beta_schedule: str = 'linear',
            timesteps: int = 1000,
            linear_start: float = 1e-4,
            linear_end: float = 2e-2,
            cosine_s: float = 8e-3
    ):
        betas = make_beta_schedule(
            beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s
        )
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer(
            'posterior_mean_coef1',
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer(
            'posterior_mean_coef2',
            to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

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

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    @torch.no_grad()
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, latent, latent_mask, condition, condition_mask, t,
            clip_denoised: bool = False, return_x0: bool = False,
    ):
        t_in = t
        model_out = self.apply_model(latent, t_in, condition, latent_mask, condition_mask)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(latent, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=latent, t=t)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    def apply_model(self, x_noisy, t, cond, mask, cond_mask, return_ids=False, *args, **kwargs):
        if not isinstance(cond, list):
            cond = [cond]
            cond_mask = [cond_mask]
        key = 'c_concat' if self.model.condition_key == 'concat' else 'c_crossattn'
        cond = {key: cond, f"{key}_mask": cond_mask}

        x_recon = self.model(x_noisy, mask, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        return x_recon

    def p_losses(self, x_start, mask, t, condition, condition_mask, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, condition, mask, condition_mask)

        if self.parameterization == 'x0':
            target = x_start
        elif self.parameterization == 'eps':
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mask)
        prefix = 'train' if self.training else 'val'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})
        loss = self.l_simple_weight * loss.mean()
        loss_vlb = (self.lvlb_weights[t] * loss_simple).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def get_loss(self, pred, target, mask, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target, pred, reduction="None")
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
        if mean:
            loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()
        return loss

    def shared_step(self, batch, batch_idx):
        latent, latent_mask, condition, condition_mask = self.get_input(batch)
        loss, loss_dict = self(latent, latent_mask, condition, condition_mask)
        return loss, loss_dict

    def get_input(self, batch):
        # 将源端句 latent 作为 condition，目标端句作为 paraphrase
        src = batch['source_text_input_ids'].clone()
        src_mask = batch['source_text_attention_mask'].clone()
        label = batch['labels'].clone()
        label[:, 0] = 0
        label_mask = batch['labels_attention_mask'].clone()

        latent, mask = self.get_first_stage_encoding(label, label_mask)
        cond, cond_mask = self.get_conditioning(src, src_mask)
        return latent, mask, cond, cond_mask

    def get_first_stage_encoding(self, encoder_posterior, mask):
        encoder = self.first_stage_model.get_encoder()
        encoder_posterior = encoder(encoder_posterior, attention_mask=mask).last_hidden_state
        if self.normalize:
            encoder_posterior = 1. / self.scale_factor * encoder_posterior
        return encoder_posterior, mask

    def get_conditioning(self, condition, mask):
        encoder = self.cond_stage_model
        # In training stage, drop conditional guidance with unconditional prob
        # replace unconditional guidance with null token sequences
        embeddings = encoder.embed_tokens(condition) * encoder.embed_scale
        if self.training and self.unconditional_prob == 0.:
            unconditional_embedding = repeat(
                self.unconditional_token_emb, 'd -> b s d', b=condition.shape[0],
                s=condition.shape[1]
            )
            unconditional_mask = self.unconditional_bernoulli.sample([condition.shape[0]]).bool()
            embeddings[unconditional_mask] = unconditional_embedding[unconditional_mask]
        condition = encoder(inputs_embeds=embeddings, attention_mask=mask).last_hidden_state
        return condition, mask

    @torch.no_grad()
    def decode_first_stage(self, latent, latent_mask, sample_strategy: dict):
        if self.normalize:
            latent = latent * self.scale_factor
        encoder_output = BaseModelOutput(last_hidden_state=latent.clone())
        samples = self.first_stage_model.generate(
            encoder_outputs=encoder_output,
            attention_mask=latent_mask.clone(),
            **sample_strategy
        )
        texts = [
            self.first_stage_tokenizer.decode(sample, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for sample in samples
        ]
        return texts

    def forward(self, latent, mask, condition, condition_mask, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (latent.shape[0],), device=self.device).long()
        return self.p_losses(latent, mask, t, condition, condition_mask, *args, **kwargs)

    @torch.no_grad()
    def p_sample(
            self, latent, latent_mask, condition, condition_mask, t,
            temperature: float = 1.0, noise_dropout: float = 0.0,
            repeat_noise: bool = False, return_x0=False, clip_denoised=False
    ):
        # TODO: self conditioning
        bsz, seqlen, dim = latent.shape
        outputs = self.p_mean_variance(
            latent, latent_mask, condition, condition_mask, t,
            clip_denoised=clip_denoised, return_x0=return_x0
        )

        noise = noise_like(latent.shape, device=self.device, repeat=repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(bsz, *((1,) * (len(latent.shape) - 1)))

        if return_x0:
            model_mean, model_var, model_logvar, x0 = outputs
            return model_mean + nonzero_mask * (0.5 * model_logvar).exp() * noise, x0
        else:
            model_mean, model_var, model_logvar = outputs
            return model_mean + nonzero_mask * (0.5 * model_logvar).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
            self, condition, condition_mask, latent_mask=None, x_T=None,
            timesteps=None, start_T=None, log_every_t=None,
            return_intermediates=False, verbose=False,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        bsz = condition.shape[0]
        if x_T is None:
            latent = torch.randn((bsz, self.max_seqlen, self.latent_dim), device=self.device)
        else:
            latent = x_T

        # TODO: handle latent mask
        intermediates = [latent]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        for t in iterator:
            ts = torch.full((bsz,), t, dtype=torch.long, device=self.device)
            latent = self.p_sample(latent, latent_mask, condition, condition_mask, ts, clip_denoised=self.clip_denoised)

            if t % log_every_t == 0 or t == timesteps - 1:
                intermediates.append(latent)

        if return_intermediates:
            return latent, intermediates
        return latent

    @torch.no_grad()
    def sample(self, condition, condition_mask, latent_mask, return_intermediates=False, x_T=None, verbose=False,
               timesteps=None, **kwargs):
        return self.p_sample_loop(
            condition, condition_mask, latent_mask,
            timesteps=timesteps, x_T=x_T,
            return_intermediates=return_intermediates, verbose=verbose,
            **kwargs
        )

    @torch.no_grad()
    def sample_log(self, condition, condition_mask, latent_mask=None, batch_size=16, sampler="dpm", steps=20, **kwargs):
        condition = condition[:batch_size]
        condition_mask = condition_mask[:batch_size]
        if latent_mask is not None:
            latent_mask = latent_mask[:batch_size]
        else:
            # todo Handle Latent Mask Sampling
            latent_mask = None

        if not sampler:
            sample, intermediates = self.sample(
                condition, condition_mask, latent_mask,
                return_intermediates=True, **kwargs
            )
        else:
            assert sampler in ['ddim', 'dpm'], f"{sampler} Sampler is not implemented yet."
            if sampler == 'ddim':
                sampler = DDIMSampler(self, schedule=self.schedule)
            elif sampler == 'dpm':
                sampler = DPMSolverSampler(self, schedule=self.schedule)

            model_kwargs = {
                'mask': latent_mask,
                'cond_mask': condition_mask
            }
            sample, intermediates = sampler.sample(
                steps, batch_size, (self.max_seqlen, self.latent_dim),
                condition=condition, model_kwargs=model_kwargs,
                **kwargs
            )
        return sample, intermediates, latent_mask

    def generate_text(
            self, condition, condition_mask, latent_mask=None, batch_size=16, sample_strategy: dict = None, **kwargs
    ):
        if latent_mask is None:
            return []
        sample, intermediates, latent_mask = self.sample_log(
            condition, condition_mask, latent_mask, batch_size, **kwargs
        )
        return self.decode_first_stage(sample, latent_mask, sample_strategy)

    def generate(self, batch, batch_size=16):
        latent, latent_mask, condition, condition_mask = self.get_input(batch)
        return self.generate_text(condition, condition_mask, latent_mask, batch_size, sample_strategy=self.sample_cfg)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        return loss

    def on_validation_start(self) -> None:
        if self.local_rank == 0:
            self.valid_print = True

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss_no_ema, loss_dict_no_ema = self.shared_step(batch, batch_idx)
        with self.ema_scope():
            loss_ema, loss_dict_ema = self.shared_step(batch, batch_idx)
            loss_dict_ema = {f"{key}_ema": value for key, value in loss_dict_ema.items()}
            if self.local_rank == 0 and self.valid_print:
                self.valid_print = False
                texts = self.generate(batch, batch_size=8)
                print('\n'.join(texts))
                print("=" * 20)
        self.log_dict(loss_dict_no_ema, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(loss_dict_ema, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        latent, latent_mask, condition, condition_mask = self.get_input(batch)
        with self.ema_scope():
            texts = self.generate_text(
                condition, condition_mask, latent_mask, batch_size=condition.shape[0],
                verbose=False, sampler='dpm', steps=20, sample_strategy=self.sample_cfg
            )
        return texts

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params += [self.logvar]

        if self.unconditional_prob > 0.:
            params += [self.unconditional_token_emb]

        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
        return opt


class DiffusionWrapper(pl.LightningModule):
    def __init__(
            self,
            dm_cfg: DictConfig,
            condition_key: str = 'crossattn',
    ):
        super().__init__()
        self.condition_key = condition_key
        assert self.condition_key in [None, 'concat', 'crossattn', 'hybrid']

        dm_cfg = dm_cfg.params
        assert dm_cfg.latent_dim % dm_cfg.attention_head_dim == 0, "Latent dim must be divisible by attention head dim"
        self.diffusion_model = DenoisingTransformer(
            tx_dim=dm_cfg.tx_dim,
            latent_dim=dm_cfg.latent_dim,
            tx_depth=dm_cfg.tx_depth,
            heads=dm_cfg.latent_dim // dm_cfg.attention_head_dim,
            # max_seq_len=dm_cfg.max_seq_len,  # 这一行注释比较奇怪，对qqp_base的模型要注释掉
            dropout=dm_cfg.dropout,
            scale_shift=dm_cfg.scale_shift
        )

    def forward(
            self,
            x,
            mask,
            t,
            c_concat: list = None,
            c_concat_mask: list = None,
            c_crossattn: list = None,
            c_crossattn_mask: list = None
    ):
        if self.condition_key is None:
            return self.diffusion_model(x, mask, timesteps=t)
        elif self.condition_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            xc_mask = torch.cat([mask] + c_concat_mask, dim=1)
            out = self.diffusion_model(xc, xc_mask, timesteps=t)
        elif self.condition_key == 'crossattn':
            cc = torch.cat(c_crossattn, dim=1)
            cc_mask = torch.cat(c_crossattn_mask, dim=1)
            out = self.diffusion_model(x, mask, timesteps=t, context=cc, context_mask=cc_mask)
        elif self.condition_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            xc_mask = torch.cat([mask] + c_concat_mask, dim=1)
            cc = torch.cat(c_crossattn, dim=1)
            cc_mask = torch.cat(c_crossattn_mask, dim=1)
            out = self.diffusion_model(xc, xc_mask, timesteps=t, context=cc, context_mask=cc_mask)
        else:
            return NotImplementedError()
        return out
