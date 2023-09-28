import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from ld4pg.models.diffusion import LatentDiffusion
from ld4pg.util import default, disabled_train


class ZeroInit(nn.Module):
    """
    Zero init module for the control net.
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Linear(latent_dim, latent_dim, bias=True)
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def forward(self, x):
        return self.net(x)


class ControlNetModel(LatentDiffusion):
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
            sample_strategy=None,
            additional_input_key: str = "",
            additional_input_mask: str = "",
            *args,
            **kwargs,
    ):
        super().__init__(
            model_cfg, first_stage_model, cond_stage_model, first_stage_tokenizer, condition_key,
            beta_schedule, parameterization, loss_type, timesteps, max_seqlen,
            use_ema, scale_factor, scale_mean, learning_rate, unconditional_prob, normalize,
            learn_logvar, l_simple_weight, original_elbo_weight, v_posterior, log_every_t, sample_strategy
        )
        self.save_hyperparameters(ignore=[
            'first_stage_model', 'cond_stage_model', 'first_stage_tokenizer'
        ])
        for f_model in (self.first_stage_model, self.cond_stage_model):
            f_model.train = disabled_train
            for param in f_model.parameters():
                param.requires_grad = False

        self.additional_input_key = additional_input_key
        self.additional_input_mask = additional_input_mask
        self.zero_in = ZeroInit(self.latent_dim)
        self.zero_out = ZeroInit(self.latent_dim)

    def apply_cn_model(
            self,
            x_noisy,
            t,
            condition,
            mask,
            condition_mask,
            cn_latent,
            cn_mask,  # B x T
            *args,
            **kwargs
    ):
        cn_latent_input = self.zero_in(cn_latent)
        latent = x_noisy + cn_latent_input * cn_mask.unsqueeze(-1)
        cn_latent_out = self.apply_model(latent, t, condition, mask, condition_mask, *args, **kwargs)
        output = self.zero_out(cn_latent_out)
        return output
