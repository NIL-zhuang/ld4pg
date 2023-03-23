from torch.optim import AdamW
from ld4pg.optim.lion_optim import Lion


def separate_weight_decay_params(params):
    # Exclude affine params in norms (e.g. LayerNorm, GroupNorm, etc.) and bias terms
    no_wd_params = [param for param in params if param.ndim < 2]
    wd_params = [param for param in params if param not in set(no_wd_params)]
    return wd_params, no_wd_params


def get_adamW_optimizer(params, lr=3e-4, betas=(0.9, 0.99), weight_decay=0.01):
    params = list(params)
    wd_params, no_wd_params = separate_weight_decay_params(params)

    param_groups = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=betas)


def get_lion_optimizer(params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01):
    params = list(params)
    wd_params, no_wd_params = separate_weight_decay_params(params)

    param_groups = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]
    return Lion(param_groups, lr=lr, weight_decay=weight_decay, betas=betas)
