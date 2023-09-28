import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from ld4pg.models.control_net.controlnet import ControlNetModel
from ld4pg.models.diffusion.ddpm import LatentDiffusion
from ld4pg.util import default


class LDPControlNetPipeline(pl.LightningModule):
    def __init__(
            self,
            model_cfg: DictConfig,
            text_cond_stage_model: PreTrainedModel,
            first_stage_model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            ldp: LatentDiffusion,
            controlnet: ControlNetModel,
            learning_rate: float = 1e-6,
            controlnet_cond_stage_model: PreTrainedModel = None,
            cn_scale_factor: float = 1.0,
            cn_scale_mean: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            'text_cond_stage_model', 'first_stage_model', 'tokenizer',
            'controlnet_cond_stage_model', 'ldp', 'controlnet'
        ])
        self.model_cfg = model_cfg
        self.text_cond_stage_model = text_cond_stage_model
        self.first_stage_model = first_stage_model
        self.tokenizer = tokenizer
        self.ldp = ldp
        self.controlnet = controlnet
        self.controlnet_cond_stage_model = controlnet_cond_stage_model

        self.num_timesteps = ldp.num_timesteps
        self.learning_rate = learning_rate
        self.scale_factor = cn_scale_factor
        self.scale_mean = cn_scale_mean

    def p_losses(
            self,
            x_start,
            mask,
            t,
            condition,
            condition_mask,
            cn_latent,
            cn_mask,
            noise=None,
            *args,
            **kwargs
    ):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, condition, mask, condition_mask, cn_latent, cn_mask)

        if self.ldp.parameterization == 'x0':
            target = x_start
        elif self.ldp.parameterization == 'eps':
            target = noise
        else:
            raise NotImplementedError

        loss_simple = self.ldp.get_loss(model_output, target, mask)
        prefix = 'train' if self.training else 'val'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.ldp.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.ldp.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.ldp.logvar.data.mean()})
        loss = self.ldp.l_simple_weight * loss.mean()
        loss_vlb = (self.ldp.lvlb_weights[t] * loss_simple).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.ldp.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def apply_model(
            self,
            x_noisy,
            t,
            condition,
            mask,
            condition_mask,
            cn_latent,
            cn_mask,
            *args,
            **kwargs
    ):
        """ Get ControlNet control latent variable representation """
        ...
        with self.ldp.ema_scope():
            ldp_recon = self.ldp.apply_model(
                x_noisy, t, condition, mask, condition_mask, return_ids=False,
                *args, **kwargs
            )
        controlnet_recon = self.controlnet.apply_cn_model(
            x_noisy, t, condition, mask, condition_mask, cn_latent, cn_mask, *args, **kwargs
        )
        recon = ldp_recon + controlnet_recon
        return recon

    @torch.no_grad()
    def q_sample(self, x_start, t, noise=None):
        return self.ldp.q_sample(x_start, t, noise)

    def forward(self, latent, mask, condition, condition_mask, cn_latent, cn_mask, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (latent.shape[0],), device=self.device)
        return self.p_losses(latent, mask, t, condition, condition_mask, cn_latent, cn_mask, *args, **kwargs)

    def shared_step(self, batch, batch_idx):
        latent, latent_mask, label, label_mask, cn_input, cn_mask = self.get_input(batch)
        loss, loss_dict = self(latent, latent_mask, label, label_mask, cn_input, cn_mask)
        return loss, loss_dict

    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss_no_ema, loss_dict_no_ema = self.shared_step(batch, batch_idx)
        with self.controlnet.ema_scope():
            loss_ema, loss_dict_ema = self.shared_step(batch, batch_idx)
            loss_dict_ema = {f"{key}_ema": value for key, value in loss_dict_ema.items()}
        self.log_dict(loss_dict_no_ema, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(loss_dict_ema, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def get_input(self, batch):
        src = batch['source_text_input_ids'].clone()
        src_mask = batch['source_text_attention_mask'].clone()
        label = batch['labels'].clone()
        label_mask = batch['labels_attention_mask'].clone()
        cn_input = batch[self.controlnet.additional_input_key].clone()
        cn_mask = batch[self.controlnet.additional_input_mask].clone()

        latent, mask = self.ldp.get_first_stage_encoding(label, label_mask)
        cond, cond_mask = self.ldp.get_conditioning(src, src_mask)
        cn_latent, cn_mask = self.get_controlnet_stage_encoding(cn_input, cn_mask)
        return latent, mask, cond, cond_mask, cn_latent, cn_mask

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.controlnet.parameters())
        if self.controlnet.learn_logvar:
            print("Diffusion model optimizing logvar")
            params += [self.controlnet.logvar]

        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
        return opt

    def get_controlnet_stage_encoding(self, tokens, mask):
        encoder = self.controlnet_cond_stage_model
        encoder_posterior = encoder(tokens, attention_mask=mask).last_hidden_state
        if self.controlnet.normalize:
            encoder_posterior = 1. / self.scale_factor * encoder_posterior
        return encoder_posterior, mask
