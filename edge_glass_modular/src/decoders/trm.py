"""TRM-inspired primitive layer implementations.

This module provides the reusable transformer primitives (config, RMSNorm,
RoPE, causal attention, MLP, transformer layer) consumed by the QwenVLM
TRM-inspired iterative-refinement path in `src/models/trm_qwen_vlm.py`.

Note: a standalone `TRMDecoder` language model that previously lived here was
retired — it was never recursive and had no execution evidence in any retained
experiment (see AUDIT_LEDGER.md, TRM reconciliation audit).
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TRMConfig:
    """Configuration for TRM decoder."""

    vocab_size: int = 32000
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    intermediate_dim: int = 2048
    max_seq_len: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    use_rope: bool = True
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute sin/cos for max sequence length
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin embeddings for sequence length."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys.

    Args:
        q: Queries (batch_size, num_heads, seq_len, head_dim)
        k: Keys (batch_size, num_heads, seq_len, head_dim)
        cos: Cosine embeddings (seq_len, head_dim)
        sin: Sine embeddings (seq_len, head_dim)

    Returns:
        Rotated queries and keys
    """
    # Reshape for rotation
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    # Apply rotation
    q_embed = (q * cos.unsqueeze(0).unsqueeze(0)) + (rotate_half(q) * sin.unsqueeze(0).unsqueeze(0))
    k_embed = (k * cos.unsqueeze(0).unsqueeze(0)) + (rotate_half(k) * sin.unsqueeze(0).unsqueeze(0))

    return q_embed, k_embed


class TRMAttention(nn.Module):
    """Multi-head self-attention with optional RoPE."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.use_rope = config.use_rope

        assert self.hidden_dim % self.num_heads == 0

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim, max_seq_len=config.max_seq_len, theta=config.rope_theta
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if enabled
        if self.use_rope:
            cos, sin = self.rotary_emb(seq_len)
            q, k = apply_rotary_emb(q, k, cos, sin)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if attention_mask is None:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1
            ).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        else:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        # Output projection
        output = self.o_proj(attn_output)

        return output


class TRMMLP(nn.Module):
    """MLP feed-forward network."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.fc2 = nn.Linear(config.intermediate_dim, config.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TRMLayer(nn.Module):
    """Single TRM transformer layer."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.attention = TRMAttention(config)
        self.mlp = TRMMLP(config)
        self.norm1 = RMSNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.norm2 = RMSNorm(config.hidden_dim, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
