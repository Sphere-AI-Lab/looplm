import json
import math
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from looplm_pipeline.models.attention import AttentionBackendReport, attach_attention_backend_report
from looplm_pipeline.models.builders.modeling_outputs import CausalLMOutput
from looplm_pipeline.models.builders.transformer_primitives import (
    RMSNorm,
    RotaryCausalSelfAttention,
    SwiGLUMLP,
)


LoopFormerCausalLMOutput = CausalLMOutput


class LoopFormerConfig(SimpleNamespace):
    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class LoopFormerBlock(nn.Module):
    def __init__(self, config: LoopFormerConfig):
        super().__init__()
        hidden_size = int(config.hidden_size)
        eps = float(getattr(config, "rms_norm_eps", 1.0e-6))
        self.input_layernorm = RMSNorm(hidden_size, eps, elementwise_affine=False)
        self.self_attn = RotaryCausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps, elementwise_affine=False)
        self.mlp = SwiGLUMLP(config)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True),
        )

    def reset_loopformer_modulation(self) -> None:
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate_attn, gate_mlp, scale_attn, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
        x = x + gate_attn.unsqueeze(1) * self.self_attn(
            self.input_layernorm(x) * (1 + scale_attn.unsqueeze(1)),
            attention_mask=attention_mask,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self.post_attention_layernorm(x) * (1 + scale_mlp.unsqueeze(1))
        )
        return x


class LoopFormerSharedLayers(nn.Module):
    def __init__(self, config: LoopFormerConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [LoopFormerBlock(config) for _ in range(int(config.num_hidden_layers))]
        )

    def reset_loopformer_modulation(self) -> None:
        for layer in self.layers:
            layer.reset_loopformer_modulation()

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, c, attention_mask=attention_mask)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = int(frequency_embedding_size)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class LoopFormerForCausalLM(nn.Module):
    def __init__(self, config: LoopFormerConfig):
        super().__init__()
        self.config = config
        self.max_model_loops = int(getattr(config, "max_model_loops", 8))
        hidden_size = int(config.hidden_size)
        vocab_size = int(config.vocab_size)
        eps = float(getattr(config, "rms_norm_eps", 1.0e-6))
        time_embedding_size = int(getattr(config, "time_embedding_size", 256))

        self.model = nn.ModuleDict(
            {
                "embed_tokens": nn.Embedding(vocab_size, hidden_size),
                "layers": LoopFormerSharedLayers(config),
                "norm": RMSNorm(hidden_size, eps, elementwise_affine=True),
            }
        )
        self.time_embedder = TimestepEmbedder(hidden_size, time_embedding_size)
        self.dt_embedder = TimestepEmbedder(hidden_size, time_embedding_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if bool(getattr(config, "tie_word_embeddings", False)):
            self.lm_head.weight = self.model["embed_tokens"].weight

    def post_init_weights(self) -> None:
        self.model["layers"].reset_loopformer_modulation()
        if bool(getattr(self.config, "tie_word_embeddings", False)):
            self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model["embed_tokens"].weight

    def _default_loop_steps(self) -> List[float]:
        return [1.0 / float(self.max_model_loops)] * self.max_model_loops

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loop_steps: Optional[List[float]] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_: Any,
    ) -> LoopFormerCausalLMOutput | tuple:
        if loop_steps is None:
            loop_steps = self._default_loop_steps()
        if len(loop_steps) < 1:
            raise ValueError("loop_steps must contain at least one step size.")

        x = self.model["embed_tokens"](input_ids)
        hidden_states = [] if output_hidden_states else None
        ti = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for dt in loop_steps:
            dt_base = torch.ones_like(ti) * float(dt)
            c = self.time_embedder(ti) + self.dt_embedder(dt_base)
            x = self.model["layers"](x, c, attention_mask=attention_mask)
            if hidden_states is not None:
                hidden_states.append(x)
            ti = ti + float(dt)

        x = self.model["norm"](x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = LoopFormerCausalLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=x,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
        )
        if return_dict:
            return output
        return tuple(v for v in (loss, logits, x, output.hidden_states) if v is not None)

    def save_pretrained(self, output_dir: str, safe_serialization: bool = True) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2, sort_keys=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))


def build_loopformer_config(
    model_cfg: Dict[str, Any],
    *,
    family: str,
    architectures: list[str],
    default_rms_norm_eps: float,
    default_rope_theta: float,
    default_use_qk_norm: bool,
) -> LoopFormerConfig:
    loop_cfg = dict(model_cfg.get("loopformer") or {})
    max_model_loops = int(loop_cfg.get("max_model_loops", loop_cfg.get("max_loops", 8)))
    shared_depth = int(loop_cfg.get("shared_depth", model_cfg.get("num_hidden_layers", 3)))
    config_payload = dict(model_cfg)
    config_payload["family"] = family
    config_payload["num_hidden_layers"] = shared_depth
    config_payload["max_model_loops"] = max_model_loops
    config_payload["time_embedding_size"] = int(loop_cfg.get("time_embedding_size", 256))
    config_payload.setdefault("head_dim", int(model_cfg["hidden_size"]) // int(model_cfg["num_attention_heads"]))
    config_payload.setdefault("rms_norm_eps", default_rms_norm_eps)
    config_payload.setdefault("rope_theta", default_rope_theta)
    config_payload.setdefault("attention_dropout", 0.0)
    config_payload.setdefault("attention_bias", False)
    config_payload.setdefault("mlp_bias", False)
    config_payload.setdefault("use_qk_norm", default_use_qk_norm)
    config_payload.setdefault("tie_word_embeddings", False)
    config_payload.setdefault("use_cache", False)
    config_payload.setdefault("architectures", architectures)
    config_payload["loopformer"] = loop_cfg
    return LoopFormerConfig(**config_payload)


def build_loopformer_model(
    model_cfg: Dict[str, Any],
    *,
    model_cls: type[LoopFormerForCausalLM],
    family: str,
    architectures: list[str],
    default_rms_norm_eps: float,
    default_rope_theta: float,
    default_use_qk_norm: bool,
    applied_via: str,
):
    from looplm_pipeline.models.attention import resolve_model_torch_dtype

    cfg = build_loopformer_config(
        model_cfg,
        family=family,
        architectures=architectures,
        default_rms_norm_eps=default_rms_norm_eps,
        default_rope_theta=default_rope_theta,
        default_use_qk_norm=default_use_qk_norm,
    )
    model = model_cls(cfg)
    raw_model_torch_dtype = model_cfg.get("torch_dtype")
    model_dtype = resolve_model_torch_dtype(raw_model_torch_dtype)
    if model_dtype is not None:
        model.to(dtype=model_dtype)
    attach_attention_backend_report(
        model,
        AttentionBackendReport(
            requested=model_cfg.get("attn_implementation"),
            attempted=["sdpa"],
            selected="sdpa",
            actual="sdpa",
            applied_via=applied_via,
        ),
    )
    return model
