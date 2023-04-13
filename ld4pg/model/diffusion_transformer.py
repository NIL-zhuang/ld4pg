import torch
import torch.nn as nn
from einops import rearrange

from ld4pg.modules.diffusion_modules.x_transformers import Encoder
from ld4pg.module.pos_emb import SinusoidalPositionalEmbedding, AbsolutePositionalEmbedding
from ld4pg.utils import init_zero_


class DiffusionTransformer(nn.Module):
    """
    一个由Attention Block构成的Transformer
    通过 cross attention 把 condition 作为条件引导模型

    Bidirectional Pre-LN transformer
    12 layers, hidden_dim = 768
    learnable absolute positional encodings
    T5 relative positional biases
    GeGLU activations
    """

    def __init__(
            self,
            tx_dim,
            tx_depth,
            heads,
            latent_dim=None,
            max_seq_len=64,
            self_condition=False,
            dropout=0.1,
            scale_shift=False,
            conditional=True,
            unconditional_prob=0,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.self_condition = self_condition
        self.scale_shift = scale_shift
        self.conditional = conditional
        self.unconditional_prob = unconditional_prob

        self.max_seq_len = max_seq_len

        # time embeddings
        time_emb_dim = tx_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(tx_dim),
            nn.Linear(tx_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_pos_embed_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, tx_dim)
        )

        self.pos_emb = AbsolutePositionalEmbedding(tx_dim, max_seq_len)

        self.cross = (self_condition or conditional)

        self.ld_encoder = Encoder(
            dim=tx_dim,
            depth=tx_depth,
            heads=heads,
            attn_dropout=dropout,  # dropout post-attention
            ff_dropout=dropout,  # feedforward dropout
            rel_pos_bias=True,
            ff_glu=True,
            cross_attend=self.cross,
            time_emb_dim=tx_dim * 4 if self.scale_shift else None,
        )
        # null embedding 来表示没有self-condition的前一步输入信息
        # if self_condition:
        #     self.null_embedding = nn.Embedding(1, tx_dim)
        #     self.context_proj = nn.Linear(latent_dim, tx_dim)

        self.input_proj = nn.Linear(latent_dim, tx_dim)
        self.norm = nn.LayerNorm(tx_dim)
        self.output_proj = nn.Linear(tx_dim, latent_dim)

        self.latent_mean = nn.Parameter(torch.zeros(latent_dim))
        self.latent_scale = nn.Parameter(torch.zeros(latent_dim))

        init_zero_(self.output_proj)

    def forward(self, x, mask, time, condition=None, condition_mask=None, x_self_cond=None):
        """
        x, condition: input, [bsz x seqlen x dim]
        mask, condition_mask: bool tensor where False indicates masked positions, [bsz x seqlen]
        time: timestep, [bsz]
        """

        time_emb = self.time_mlp(time)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')

        # Bsz x Dim
        pos_emb = self.pos_emb(x)
        time_pos_emb = self.time_pos_embed_mlp(time_emb)
        tx_input = self.input_proj(x) + pos_emb + time_pos_emb

        # 使用 cross attention 来看对应的 content
        context, context_mask = [], []
        # 在 seq2seq 任务中暂时不考虑 self-condition
        # if self.self_condition:
        #     if x_self_cond is None:
        #         null_context = repeat(self.null_embedding.weight, '1 d -> b 1 d', b=x.shape[0])
        #         context.append(null_context)
        #         context_mask.append(torch.tensor([[True] for _ in range(x.shape[0])], dtype=bool, device=x.device))
        #     else:
        #         context.append(self.context_proj(x_self_cond))
        #         context_mask.append(mask)
        if self.conditional:
            assert condition is not None
            context.append(condition)
            context_mask.append(condition_mask)

        # context 包含了 self-condition 和 condition 的信息，拼接为了 Bsz x 2 x Dim 的向量
        context = torch.cat(context, dim=1)
        context_mask = torch.cat(context_mask, dim=1)
        x = self.ld_encoder(tx_input, mask=mask, context=context, context_mask=context_mask, time_emb=time_emb)

        x = self.norm(x)

        return self.output_proj(x)
