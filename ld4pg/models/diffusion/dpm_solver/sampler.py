import torch

from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver

MODEL_TYPES = {
    "x0": "x_start",
    "eps": "noise",
    "v": "v"
}


class DPMSolverSampler(object):
    def __init__(self, model, device=torch.device('cuda'), **kwargs):
        super().__init__()
        self.model = model
        self.device = device
        self.register_buffer("alphas_cumprod", model.alphas_cumprod)

    def register_buffer(self, name, attr):
        attr = attr.clone().detach().to(torch.float32).to(self.model.device)
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(
            self,
            dpm_steps,
            batch_size,
            shape,
            condition=None,
            condition_mask=None,
            latent_mask=None,
            callback=None,
            normals_sequence=None,
            img_callback=None,
            quantize_x0=False,
            eta=0.,
            mask=None,
            x0=None,
            temperature=1.,
            noise_dropout=0.,
            score_corrector=None,
            corrector_kwargs=None,
            verbose=True,
            x_T=None,
            log_every_t=100,
            unconditional_guidance_scale=1.,
            unconditional_conditioning=None,
            # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
            dpm_skip_type="time_uniform",
            dpm_method="multistep",
            dpm_order=2,
            dpm_lower_order_final=True,
            **kwargs
    ):
        if condition is not None:
            if isinstance(condition, dict):
                c_tmp = condition[list(condition.keys())[0]]
                while isinstance(c_tmp, list):
                    c_tmp = c_tmp[0]
                if isinstance(c_tmp, torch.Tensor):
                    cbs = c_tmp.shape[0]
                    if cbs != batch_size:
                        print(f"Warning: Got {cbs} conditioning but batch-size is {batch_size}")
            elif isinstance(condition, list):
                for c_tmp in condition:
                    if c_tmp.shape[0] != batch_size:
                        print(f"Warning: Got {c_tmp.shape[0]} conditioning but batch-size is {batch_size}")
            else:
                if isinstance(condition, torch.Tensor):
                    if condition.shape[0] != batch_size:
                        print(f"Warning: Got {condition.shape[0]} conditioning but batch-size is {batch_size}")

        max_seqlen, latent_dim = shape
        size = (batch_size, max_seqlen, latent_dim)
        if verbose:
            print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {dpm_steps}')

        if x_T is None:
            latent = torch.randn(size, device=self.model.betas.device)
        else:
            latent = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)
        model_fn = model_wrapper(
            self.model.apply_model,
            ns,
            model_type=MODEL_TYPES[self.model.parameterization],
            guidance_type="classifier-free",
            condition=condition,
            model_kwargs={'mask': latent_mask, 'cond_mask': condition_mask},
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )
        dpm_solver = DPM_Solver(model_fn, ns)
        x = dpm_solver.sample(
            latent, steps=dpm_steps,
            skip_type=dpm_skip_type, method=dpm_method, order=dpm_order,
            lower_order_final=dpm_lower_order_final
        )
        return x, []
