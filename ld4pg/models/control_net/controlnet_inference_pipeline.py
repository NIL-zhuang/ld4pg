from typing import *

import torch
from omegaconf import DictConfig

from ld4pg.models.control_net.controlnet_pipeline import LDPControlNetPipeline
from ld4pg.models.diffusion.ddpm import LatentDiffusion
from ld4pg.models.diffusion.dpm_solver import DPMSolverSampler


class LDPControlnetInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            model_cfg: DictConfig,
            control_cfg: DictConfig,
            ldp: LatentDiffusion,
            ldp_controlnet_pipelines: List[LDPControlNetPipeline],
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.ldp = ldp
        self.controlnet_pipelines = ldp_controlnet_pipelines
        self.num_timesteps = ldp.num_timesteps
        self.control_start = control_cfg.start_step
        self.control_end = control_cfg.end_step

        self.alphas_cumprod = self.ldp.alphas_cumprod
        self.betas = self.ldp.betas
        self.parameterization = self.ldp.parameterization

    @torch.no_grad()
    def apply_model(
            self,
            x_noisy,
            t,
            condition,
            mask,
            condition_mask,
            cn_latents: List,
            cn_masks: List,
            *args,
            **kwargs
    ):
        assert len(cn_latents) == len(cn_masks) == len(self.controlnet_pipelines)
        with self.ldp.ema_scope():
            ldp_recon = self.ldp.apply_model(
                x_noisy, t, condition, mask, condition_mask, return_ids=False,
                *args, **kwargs
            )

        recon = ldp_recon
        for cn_latent, cn_mask, controlnet_pipeline in zip(
                cn_latents, cn_masks, self.controlnet_pipelines
        ):
            controlnet_recon = controlnet_pipeline.controlnet.apply_cn_model(
                x_noisy, t, condition, mask, condition_mask, cn_latent, cn_mask, *args, **kwargs
            )
            recon += controlnet_recon
        return recon

    def get_input(self, batch):
        src = batch['source_text_input_ids'].clone()
        src_mask = batch['source_text_attention_mask'].clone()
        label = batch['labels'].clone()
        label_mask = batch['labels_attention_mask'].clone()
        latent, mask = self.ldp.get_first_stage_encoding(label, label_mask)
        cond, cond_mask = self.ldp.get_conditioning(src, src_mask)

        cn_latents = []
        cn_masks = []
        for controlnet_pipeline in self.controlnet_pipelines:
            controlnet = controlnet_pipeline.controlnet
            cn_input = batch[controlnet.additional_input_key].clone()
            cn_mask = batch[controlnet.additional_input_mask].clone()
            cn_latent, cn_mask = controlnet_pipeline.get_controlnet_stage_encoding(cn_input, cn_mask)
            cn_latents.append(cn_latent)
            cn_masks.append(cn_mask)
        return latent, mask, cond, cond_mask, cn_latents, cn_masks

    def sample_log(
            self, condition, condition_mask, latent_mask=None,
            cn_latents: list = None, cn_masks: list = None,
            batch_size=16, sampler='dpm', steps=25,
            **kwargs
    ):
        condition = condition[:batch_size]
        condition_mask = condition_mask[:batch_size]
        if latent_mask is not None:
            latent_mask = latent_mask[:batch_size]
        if sampler == 'dpm':
            sampler = DPMSolverSampler(self, schedule=self.ldp.schedule)
        else:
            raise NotImplementedError

        model_kwargs = {
            'mask': latent_mask,
            'condition_mask': condition_mask,
            'cn_latents': cn_latents,
            'cn_masks': cn_masks
        }
        sample, intermediates = sampler.sample(
            steps, batch_size, (self.ldp.max_seqlen, self.ldp.latent_dim),
            condition=condition, model_kwargs=model_kwargs,
            **kwargs
        )
        return sample, intermediates, latent_mask

    def decode_first_stage(self, latent, latent_mask, sample_strategy: dict):
        return self.ldp.decode_first_stage(latent, latent_mask, sample_strategy)

    def generate_text(
            self, condition, condition_mask, latent_mask=None,
            cn_latents=None, cn_masks=None,
            batch_size=16, sample_strategy: dict = None,
            **kwargs
    ):
        if latent_mask is None:
            return []
        sample, intermediate, latent_mask = self.sample_log(
            condition, condition_mask, latent_mask, cn_latents, cn_masks,
            batch_size, **kwargs
        )
        return self.decode_first_stage(sample, latent_mask, sample_strategy)
