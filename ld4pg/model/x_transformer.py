from functools import partial

import pytorch_lightning as pl
import torch.nn as nn

from ld4pg.config import *
from ld4pg.module import Attention
from ld4pg.module.ffn import FeedForward
from ld4pg.module.norm import ScaleNorm, RMSNorm, Scale
from ld4pg.module.pos_bias import RelativePositionBias, DynamicPositionBias, AlibiPositionalBias, LearnedAlibiPositionalBias
from ld4pg.module.pos_emb import FixedPositionalEmbedding, RotaryEmbedding
from ld4pg.module.residual import TimeConditionedResidual, GRUGatingResidual, Residual
from ld4pg.module.utils import ShiftTokens, deepnorm_init
from ld4pg.utils import groupby_prefix_and_trim, default, exists, not_equals, equals, cast_tuple


class LDEncoder(pl.LightningModule):
    def __init__(
            self,
            dim,
            depth,
            heads=8,
            causal=False,
            cross_attend=False,
            only_cross=False,
            use_scalenorm=False,
            use_rmsnorm=False,
            alibi_pos_bias=False,
            alibi_num_heads=None,
            alibi_learned=False,
            rel_pos_bias=False,
            rel_pos_num_buckets=32,
            rel_pos_max_distance=128,
            dynamic_pos_bias=False,
            dynamic_pos_bias_log_distance=False,
            dynamic_pos_bias_mlp_depth=2,
            dynamic_pos_bias_norm=False,
            position_infused_attn=False,
            rotary_pos_emb=False,
            rotary_emb_dim=None,
            custom_layers=None,
            sandwich_coef=None,
            par_ratio=None,
            residual_attn=False,
            cross_residual_attn=False,
            macaron=False,
            pre_norm=True,
            gate_residual=False,
            scale_residual=False,
            scale_residual_constant=1.,
            deepnorm=False,
            shift_tokens=0,
            sandwich_norm=False,
            zero_init_branch_output=False,
            time_emb_dim=None,
            **kwargs
    ):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim('attn_', kwargs)

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        self.has_pos_emb = position_infused_attn or rel_pos_bias or rotary_pos_emb
        self.pia_pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else None

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim) if rotary_pos_emb else None

        assert not (alibi_pos_bias and rel_pos_bias), 'you can only choose Alibi positional bias or T5 relative positional bias, not both'
        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'

        # relative positional bias
        # 这里选择的是 T5 Relative Position Bias
        self.rel_pos = None
        if rel_pos_bias:
            self.rel_pos = RelativePositionBias(scale=dim_head ** 0.5, causal=causal, heads=heads, num_buckets=rel_pos_num_buckets,
                                                max_distance=rel_pos_max_distance)
        elif dynamic_pos_bias:
            self.rel_pos = DynamicPositionBias(dim=dim // 4, heads=heads, log_distance=dynamic_pos_bias_log_distance,
                                               depth=dynamic_pos_bias_mlp_depth, norm=dynamic_pos_bias_norm)
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, 'number of ALiBi heads must be less than the total number of heads'
            alibi_pos_klass = LearnedAlibiPositionalBias if alibi_learned else AlibiPositionalBias
            self.rel_pos = alibi_pos_klass(heads=alibi_num_heads)

        # determine deepnorm and residual scale

        if deepnorm:
            assert scale_residual_constant == 1, 'scale residual constant is being overridden by deep norm settings'
            pre_norm = sandwich_norm = False
            scale_residual = True
            scale_residual_constant = (2 * depth) ** 0.25

        assert not (not pre_norm and sandwich_norm), 'sandwich norm cannot be used when not using prenorm'
        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        self.cross_attend = cross_attend

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, 'zero_init_output': True}
            ff_kwargs = {**ff_kwargs, 'zero_init_output': True}

        # calculate layer block order
        # 实际的功能就是 layer_types = default_block * depth
        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        self.scale_shift = exists(time_emb_dim)

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(zip(self.layer_types, shift_tokens)):
            is_last_layer = ind == (len(self.layer_types) - 1)

            if layer_type == 'a':
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            if self.scale_shift and layer_type in ['f']:
                residual = TimeConditionedResidual(time_emb_dim, dim)
            else:
                residual_fn = GRUGatingResidual if gate_residual else Residual
                residual = residual_fn(dim, scale_residual=scale_residual, scale_residual_constant=scale_residual_constant)

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm or (self.scale_shift and layer_type in ['f']) else None
            post_main_norm = norm_fn() if not pre_norm and not is_last_layer else None

            norms = nn.ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm
            ])

            self.layers.append(nn.ModuleList([
                norms,
                layer,
                residual
            ]))

        if deepnorm:
            init_gain = (8 * depth) ** -0.25
            deepnorm_init(self, init_gain)

    def forward(
            self,
            x,
            mask=None,
            context=None,
            context_mask=None,
            time_emb=None,
            mems=None,
            attn_mask=None,
            return_hiddens=False
    ):
        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross_attend is set to True'

        hidden = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        # 不存在这玩意
        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(list(map(lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1], mems)))
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length, x.device)

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            # is_last = ind == (len(self.layers) - 1)
            if layer_type == 'a':
                if return_hiddens:
                    hidden.append(x)
                layer_mem = mems.pop(0) if mems else None

            residual = x

            pre_branch_norm, post_branch_norm, post_main_norm = norm
            if exists(pre_branch_norm):
                x = pre_branch_norm(x)

            out = None
            if layer_type == 'a':
                out, inter = block(x, mask=mask, attn_mask=attn_mask, sinusoidal_emb=self.pia_pos_emb, rel_pos=self.rel_pos,
                                   rotary_pos_emb=rotary_pos_emb, prev_attn=prev_attn, mem=layer_mem)
            elif layer_type == 'c':
                out, inter = block(x, context=context, mask=mask, context_mask=context_mask, prev_attn=prev_cross_attn)
            elif layer_type == 'f':
                out = block(x)

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            if self.scale_shift and layer_type in ['f']:
                x = residual_fn(out, residual, time_emb)
            else:
                x = residual_fn(out, residual)

            if layer_type in ('a', 'c') and return_hiddens:
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hidden,
                attn_intermediates=intermediates
            )
            return x, intermediates

        return x
