import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from ld4pg.models.diffusion import DiffusionWrapper, LatentDiffusion
from ld4pg.modules.ema import LitEma


class ControlNetDiffusionWrapper(nn.Module):
    def __init__(self):
        super().__init__()


class ControlNetModel(pl.LightningModule):
    def __init__(
            self,
            model_cfg: DictConfig,
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
            *args,
            **kwargs,
    ):
        super().__init__()
        #### Maybe this doesn't need in ControlNetModel ####
        # self.first_stage_model = first_stage_model.eval()
        # note: this is a hack to get the encoder of the first stage model
        # self.cond_stage_model = self.first_stage_model.get_encoder()
        # self.cond_stage_model = cond_stage_model.eval()
        # self.first_stage_tokenizer = first_stage_tokenizer
        # for f_model in (self.first_stage_model, self.cond_stage_model):
        #     f_model.train = disabled_train
        #     for param in f_model.parameters():
        #         param.requires_grad = False

        self.model = DiffusionWrapper(model_cfg, condition_key=condition_key)
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
                torch.mean(self.first_stage_model.get_encoder().get_input_embeddings().weight, dim=0))

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

    @classmethod
    def from_ld4pg(
            cls,
            cfg: DictConfig,
            ldm: LatentDiffusion,
            controlnet_condition: str = "",
            load_weights_from_ldm: bool = True
    ):
        ...

    def forward(self, latent, mask, condition, condition_mask, *args, **kwargs):
        pass
