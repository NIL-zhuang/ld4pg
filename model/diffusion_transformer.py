from functools import partial

from einops import rearrange, repeat

from diffusion.utils import *
from model.x_transformer import AbsolutePositionalEmbedding, Encoder, string_begins_with, group_dict_by_key


# Helper Functions

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionTransformer(nn.Module):
    """
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
            class_conditional=False,
            num_classes=0,
            class_unconditional_prob=0,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.self_condition = self_condition
        self.scale_shift = scale_shift
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.class_unconditional_prob = class_unconditional_prob

        self.max_seq_len = max_seq_len

        # time embeddings
        sinu_pos_emb = SinusoidalPosEmb(tx_dim)

        time_emb_dim = tx_dim * 4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(tx_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_pos_embed_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, tx_dim)
        )

        self.pos_emb = AbsolutePositionalEmbedding(tx_dim, max_seq_len)

        self.cross = (self_condition or class_conditional)

        self.encoder = Encoder(
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
        if self_condition:
            self.null_embedding = nn.Embedding(1, tx_dim)
            self.context_proj = nn.Linear(latent_dim, tx_dim)
        # 类型信息，其中第 num_classes 表示的就是 无类型
        if self.class_conditional:
            assert num_classes > 0
            self.class_embedding = nn.Embedding(num_classes + 1, tx_dim)

        self.input_proj = nn.Linear(latent_dim, tx_dim)
        self.norm = nn.LayerNorm(tx_dim)
        self.output_proj = nn.Linear(tx_dim, latent_dim)

        init_zero_(self.output_proj)

    def forward(self, x, mask, time, x_self_cond=None, class_id=None):
        """
        x: input, [batch, length, latent_dim]
        mask: bool tensor where False indicates masked positions, [batch, length] 
        time: timestep, [batch]
        """

        time_emb = self.time_mlp(time)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')
        time_pos_emb = self.time_pos_embed_mlp(time_emb)

        # Bsz x Dim
        pos_emb = self.pos_emb(x)
        tx_input = self.input_proj(x) + pos_emb + time_pos_emb

        if self.cross:
            context, context_mask = [], []
            if self.self_condition:
                if x_self_cond is None:
                    null_context = repeat(self.null_embedding.weight, '1 d -> b 1 d', b=x.shape[0])
                    context.append(null_context)
                    context_mask.append(torch.tensor([[True] for _ in range(x.shape[0])], dtype=bool, device=x.device))
                else:
                    context.append(self.context_proj(x_self_cond))
                    context_mask.append(mask)
            if self.class_conditional:
                assert exists(class_id)
                class_emb = self.class_embedding(class_id)
                class_emb = rearrange(class_emb, 'b d -> b 1 d')
                context.append(class_emb)
                context_mask.append(torch.tensor([[True] for _ in range(x.shape[0])], dtype=bool, device=x.device))

            # context 包含了self-condition和class-condition的信息，拼接为了 Bsz x 2 x Dim 的向量
            context = torch.cat(context, dim=1)
            context_mask = torch.cat(context_mask, dim=1)
            x = self.encoder(tx_input, mask=mask, context=context, context_mask=context_mask, time_emb=time_emb)
        else:
            x = self.encoder(tx_input, mask=mask, time_emb=time_emb)

        x = self.norm(x)

        return self.output_proj(x)
