"""Perceiver Resampler for compressing variable-length sequences.

Canonicalized from the executed V3 training architecture
(`scripts/experiments/train_perceiver_v3.py`, the implementation that trained
the retained `perceiver_v2/checkpoint-best` run). The training computation
order is preserved: per layer, cross-attention with separate query/KV
pre-normalization, latent self-attention, then an FFN branch pre-normalized
by its own LayerNorm, all with residual connections, followed by a final
LayerNorm over the latents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverAttention(nn.Module):
    """Multi-head attention with pre-LayerNorm on queries and keys/values.

    Used for both cross-attention (q = latents, kv = input tokens) and
    latent self-attention (q = kv = latents), as in the V3 training script.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate query/KV normalization (V3 training structure)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,            # (B, N_q, dim)
        kv: torch.Tensor,           # (B, N_kv, dim)
        mask: torch.Tensor = None,  # (B, N_kv) bool, True = keep, False = pad
    ) -> torch.Tensor:
        """Attention output (B, N_q, dim)."""
        B, N_q, dim = q.shape
        N_kv = kv.shape[1]

        # Pre-normalization (separate for queries and keys/values)
        q = self.ln_q(q)
        kv = self.ln_kv(kv)

        # Multi-head projections
        Q = self.q_proj(q).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, N_q, N_kv)

        # Optional key-padding mask (True = keep) → (B, 1, 1, N_kv)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask.bool(), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N_q, dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed-forward network: Linear → GELU → Dropout → Linear → Dropout."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverResamplerLayer(nn.Module):
    """Single Perceiver layer: cross-attn → self-attn → FFN, all residual.

    The FFN branch carries its own pre-normalization (`ln_ffn`), matching the
    V3 training script's computation order.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = PerceiverAttention(dim, num_heads, dropout)
        self.self_attn = PerceiverAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, mlp_ratio, dropout)
        self.ln_ffn = nn.LayerNorm(dim)

    def forward(
        self,
        latents: torch.Tensor,      # (B, K, dim)
        tokens: torch.Tensor,       # (B, T, dim)
        token_mask: torch.Tensor = None,  # (B, T) bool, True = keep
    ) -> torch.Tensor:
        """Updated latents (B, K, dim)."""
        # Cross-attention: latents read from input tokens (optional KV mask)
        latents = latents + self.cross_attn(latents, tokens, token_mask)
        # Self-attention: latents communicate with each other
        latents = latents + self.self_attn(latents, latents)
        # FFN with its own pre-normalization
        latents = latents + self.ffn(self.ln_ffn(latents))
        return latents


class PerceiverResampler(nn.Module):
    """Perceiver Resampler: compress variable-length sequences to fixed latents.

    Learned latent queries cross-attend to the input sequence, compressing
    variable-length inputs into a fixed number of latent representations:
    O(num_latents * seq_len) instead of O(seq_len^2).

    Architecture (V3 training structure):
        Input:  (batch_size, seq_len, dim)
        Latents: (num_latents, dim) learned parameters

        For each layer:
            - Cross-attention: latents attend to input (optional input mask)
            - Self-attention: latents attend to themselves
            - Feed-forward (pre-normalized by ln_ffn)

        Output: (batch_size, num_latents, dim)

    Args:
        dim: Dimension of input and latent embeddings
        num_latents: Number of latent query tokens (e.g., 64)
        num_layers: Number of Perceiver layers (e.g., 4)
        num_heads: Number of attention heads (e.g., 8)
        mlp_ratio: FFN expansion ratio (e.g., 4.0)
        dropout: Dropout probability

    The optional ``mask`` argument accepts a (batch_size, seq_len) boolean
    tensor where True marks valid tokens; existing callers that do not pass
    a mask are unaffected.
    """

    def __init__(
        self,
        dim: int = 512,
        num_latents: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim = dim
        self.num_latents = num_latents

        # Learned latent queries (small init scaled by dim, as in V3 training)
        self.latents = nn.Parameter(
            torch.randn(num_latents, dim) * (dim ** -0.5)
        )

        # Perceiver layers
        self.layers = nn.ModuleList(
            [
                PerceiverResamplerLayer(dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.ln_out = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,            # (B, T, dim) input sequence
        mask: torch.Tensor = None,  # (B, T) bool, True = keep (optional)
    ) -> torch.Tensor:
        """Resample input sequence to fixed-size latents.

        Args:
            x: Input sequence (batch_size, seq_len, dim)
            mask: Optional boolean key-padding mask (batch_size, seq_len),
                True = valid token, False = padding.

        Returns:
            Latent representations (batch_size, num_latents, dim)
        """
        batch_size = x.shape[0]

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply Perceiver layers
        for layer in self.layers:
            latents = layer(latents, x, mask)

        # Final normalization
        latents = self.ln_out(latents)

        return latents

    @property
    def num_parameters(self) -> int:
        """Return number of parameters."""
        return sum(p.numel() for p in self.parameters())
