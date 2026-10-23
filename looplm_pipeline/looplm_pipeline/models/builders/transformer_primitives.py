import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, elementwise_affine: bool = True):
        super().__init__()
        self.eps = float(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        y = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = y.to(input_dtype)
        if self.weight is not None:
            y = y * self.weight.to(dtype=input_dtype)
        return y


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope_scaling(
    inv_freq: torch.Tensor,
    rope_scaling: Optional[Dict[str, Any]],
) -> torch.Tensor:
    if not rope_scaling:
        return inv_freq

    rope_type = str(rope_scaling.get("rope_type", rope_scaling.get("type", ""))).lower()
    factor = float(rope_scaling.get("factor", 1.0))
    if factor == 1.0:
        return inv_freq

    if rope_type == "llama3":
        old_context_len = float(rope_scaling.get("original_max_position_embeddings", 8192))
        low_freq_factor = float(rope_scaling.get("low_freq_factor", 1.0))
        high_freq_factor = float(rope_scaling.get("high_freq_factor", 4.0))
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * math.pi / inv_freq

        inv_freq_scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth = smooth.clamp(0.0, 1.0)
        inv_freq_smoothed = (1 - smooth) * (inv_freq / factor) + smooth * inv_freq
        medium = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        return torch.where(medium, inv_freq_smoothed, inv_freq_scaled)

    if rope_type in {"linear", ""}:
        return inv_freq / factor

    return inv_freq


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    theta: float,
    rope_scaling: Optional[Dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        float(theta) ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    inv_freq = _apply_rope_scaling(inv_freq, rope_scaling)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(hidden_states: torch.Tensor, num_groups: int) -> torch.Tensor:
    if num_groups == 1:
        return hidden_states
    return hidden_states.repeat_interleave(num_groups, dim=1)


class RotaryCausalSelfAttention(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // self.num_heads))
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = float(getattr(config, "rope_theta", 1000000.0))
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.dropout = float(getattr(config, "attention_dropout", 0.0))
        self.use_qk_norm = bool(getattr(config, "use_qk_norm", False))
        bias = bool(getattr(config, "attention_bias", False))

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias)
        if self.use_qk_norm:
            eps = float(getattr(config, "rms_norm_eps", 1.0e-6))
            self.q_norm = RMSNorm(self.head_dim, eps, elementwise_affine=True)
            self.k_norm = RMSNorm(self.head_dim, eps, elementwise_affine=True)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = _build_rope_cache(
            seq_len,
            self.head_dim,
            device=x.device,
            dtype=q.dtype,
            theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        k = _repeat_kv(k, self.num_key_value_groups)
        v = _repeat_kv(v, self.num_key_value_groups)

        causal_mask = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool).tril()
        attn_mask = causal_mask[None, None, :, :]
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            attn_mask = attn_mask & key_mask

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(y)


class SwiGLUMLP(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        bias = bool(getattr(config, "mlp_bias", False))
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
