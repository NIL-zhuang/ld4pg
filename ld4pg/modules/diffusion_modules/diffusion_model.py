from einops import rearrange
from torch import nn

from ld4pg.modules.diffusion_modules.x_transformers import AbsolutePositionalEmbedding, Encoder, ScaledSinusoidalEmbedding, init_zero_


class DenoisingTransformer(nn.Module):
    def __init__(
            self,
            tx_dim: int = 768,
            latent_dim: int = 768,
            tx_depth: int = 6,
            heads: int = 12,
            max_seq_len: int = 64,
            dropout: float = 0.1,
            scale_shift: bool = True,
    ):
        super().__init__()

        self.input_proj = nn.Linear(latent_dim, tx_dim)
        self.output_proj = nn.Linear(tx_dim, latent_dim)
        self.norm = nn.LayerNorm(tx_dim)
        init_zero_(self.norm)

        tx_emb_dim = tx_dim * 4
        self.time_mlp = nn.Sequential(
            ScaledSinusoidalEmbedding(tx_dim),
            nn.Linear(tx_dim, tx_emb_dim),
            nn.GELU(),
            nn.Linear(tx_emb_dim, tx_emb_dim)
        )
        self.time_pos_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(tx_emb_dim, tx_dim)
        )

        self.pos_emb = AbsolutePositionalEmbedding(tx_dim, max_seq_len)

        self.encoder = Encoder(
            dim=tx_dim,
            depth=tx_depth,
            heads=heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
            rel_pos_bias=True,
            cross_attend=True,
            time_emb_dim=tx_dim * 4 if scale_shift else None,
        )

    def timestep_embedding(self, timesteps):
        time_emb = self.time_mlp(timesteps)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')
        time_emb = self.time_pos_mlp(time_emb)
        return time_emb

    def position_embedding(self, x):
        pos_emb = self.pos_emb(x)
        return pos_emb

    def forward(self, x, mask, timesteps=None, context=None, context_mask=None):
        """
        Apply the model to an input batch
        Args:
            x: [bsz x seqlen x dim] Tensor of inputs
            mask: [bsz x seqlen] input mask
            timesteps: 1-D batch of timesteps
            context: conditioning plugged in via cross attention
            context_mask: [bsz x seqlen] context mask for cross attention
        Returns:
            a [bsz x seqlen x dim] Tensor of outputs
        """
        time_emb = self.timestep_embedding(timesteps)
        pos_emb = self.position_embedding(x)
        x = self.input_proj(x)
        tx_input = x + time_emb + pos_emb

        tx_output = self.encoder(
            tx_input,
            context=context,
            mask=mask,
            context_mask=context_mask,
            time_emb=time_emb
        )

        output = self.norm(tx_output)
        output = self.output_proj(output)
        return output
