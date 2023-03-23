from collections import namedtuple

import torch.nn.functional as F
from einops import rearrange, reduce
from tqdm.auto import tqdm

from diffusion.utils import *

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
EPS = 1e-5


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            *,
            max_seq_len,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            objective='pred_noise',
            beta_schedule='cosine',
            p2_loss_weight_gamma=0.,
            # p2 loss weight, from https://arxiv.org/abs/2204.00227
            # 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_k=1,
            ddim_sampling_eta=1.,
    ):
        super().__init__()

        self.diffusion_model = model
        if self.diffusion_model.class_conditional and self.diffusion_model.unconditional_prob > 0:
            self.class_unconditional_bernoulli = torch.distributions.Bernoulli(probs=self.diffusion_model.unconditional_prob)

        self.latent_dim = self.diffusion_model.latent_dim
        self.self_condition = self.diffusion_model.self_condition

        self.max_seq_len = max_seq_len

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        betas = self.get_beta_schedule(beta_schedule, timesteps)

        # alpha_t = PI_{i=1}^{t} (1 - beta_i)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

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

        register_buffer('latent_mean', torch.tensor([0] * self.latent_dim))
        register_buffer('latent_scale', torch.tensor(1))

    def get_beta_schedule(self, beta_schedule, timesteps):
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        return betas

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def normalize_latent(self, x_start):
        return (x_start - self.latent_mean) / (self.latent_scale + EPS)

    def denormalize_latent(self, x_start):
        return x_start * (self.latent_scale + EPS) + self.latent_mean

    def diffusion_model_predictions(self, x, mask, t, x_self_cond=None, class_id=None):
        model_output = self.diffusion_model(x, mask, t, x_self_cond, class_id=class_id)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, shape, lengths, class_id=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        latent = torch.randn(shape, device=device)
        mask = [[True] * length + [False] * (self.max_seq_len - length) for length in lengths]
        mask = torch.tensor(mask, dtype=torch.bool, device=device)

        x_start = None
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.diffusion_model_predictions(latent, mask, time_cond, self_cond, class_id=class_id)

            if time_next < 0:
                latent = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(latent)

            latent = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return latent, mask

    @torch.no_grad()
    def sample(self, batch_size, length, class_id=None):
        # TODO Create mask that controls length 
        max_seq_len, latent_dim = self.max_seq_len, self.latent_dim
        # TODO Implement for p_sample_loop 

        sample_fn = self.ddim_sample
        return sample_fn((batch_size, max_seq_len, latent_dim), length, class_id)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # sqrt(a_t)*x + sqrt(1-a_t)*noise
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    # TODO handle masking
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, mask, t, class_id, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        # 获取时间步为 t 时刻对应模型的噪声强度
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Perform unconditional generation with some probability
        # 设置一定概率的 unconditional 生成，也就是把对应的 class_id 设置成 None，这里就把 class_id 设置成了 4
        if self.diffusion_model.class_conditional and self.diffusion_model.unconditional_prob > 0:
            class_unconditional_mask = self.class_unconditional_bernoulli.sample(class_id.shape).bool()
            class_id[class_unconditional_mask] = self.diffusion_model.num_classes

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.diffusion_model_predictions(x, mask, t, class_id=class_id).pred_x_start.detach()

        # predict and take gradient step
        predictions = self.diffusion_model_predictions(x, mask, t, x_self_cond, class_id=class_id)

        loss = self.loss_fn(predictions.pred_x_start, x_start, reduction='none')
        loss = rearrange([reduce(loss[i][:torch.sum(mask[i])], 'l d -> 1', 'mean') for i in range(x_start.shape[0])], 'b 1 -> b 1')

        return loss.mean()

    def forward(self, txt_latent, mask, class_id, *args, **kwargs):
        bsz, seqlen, dim = txt_latent.shape
        assert seqlen == self.max_seq_len, f'length must be {self.max_seq_len}'
        # 时间步采样
        t = torch.randint(0, self.num_timesteps, (bsz,), device=txt_latent.device).long()
        return self.p_losses(txt_latent, mask, t, class_id, *args, **kwargs)
