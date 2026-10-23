import json
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional

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


class BaseLoopConfig(SimpleNamespace):
    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class BaseLoopBlock(nn.Module):
    def __init__(self, config: BaseLoopConfig):
        super().__init__()
        hidden_size = int(config.hidden_size)
        eps = float(getattr(config, "rms_norm_eps", 1.0e-6))
        self.input_layernorm = RMSNorm(hidden_size, eps, elementwise_affine=True)
        self.self_attn = RotaryCausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps, elementwise_affine=True)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), attention_mask=attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class BaseLoopSharedLayers(nn.Module):
    def __init__(self, config: BaseLoopConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [BaseLoopBlock(config) for _ in range(int(config.num_hidden_layers))]
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x


class BaseLoopForCausalLM(nn.Module):
    def __init__(self, config: BaseLoopConfig):
        super().__init__()
        self.config = config
        self.max_model_loops = int(getattr(config, "max_model_loops", 1))
        if self.max_model_loops < 1:
            raise ValueError("max_model_loops must be >= 1.")

        hidden_size = int(config.hidden_size)
        vocab_size = int(config.vocab_size)
        eps = float(getattr(config, "rms_norm_eps", 1.0e-6))

        self.model = nn.ModuleDict(
            {
                "embed_tokens": nn.Embedding(vocab_size, hidden_size),
                "layers": BaseLoopSharedLayers(config),
                "norm": RMSNorm(hidden_size, eps, elementwise_affine=True),
            }
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if bool(getattr(config, "tie_word_embeddings", False)):
            self.tie_weights()

    def post_init_weights(self) -> None:
        if bool(getattr(self.config, "tie_word_embeddings", False)):
            self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model["embed_tokens"].weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_: Any,
    ) -> CausalLMOutput | tuple:
        x = self.model["embed_tokens"](input_ids)
        hidden_states = [] if output_hidden_states else None
        for _ in range(self.max_model_loops):
            x = self.model["layers"](x, attention_mask=attention_mask)
            if hidden_states is not None:
                hidden_states.append(x)

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

        output = CausalLMOutput(
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


def build_baseloop_config(
    model_cfg: Dict[str, Any],
    *,
    family: str,
    architectures: list[str],
    default_rms_norm_eps: float,
    default_rope_theta: float,
    default_use_qk_norm: bool,
) -> BaseLoopConfig:
    baseloop_cfg = dict(model_cfg.get("baseloop") or {})
    shared_depth = int(
        baseloop_cfg.get("shared_depth", model_cfg.get("shared_depth", model_cfg.get("num_hidden_layers", 1)))
    )
    max_model_loops = int(
        baseloop_cfg.get("max_model_loops", model_cfg.get("max_model_loops", 1))
    )
    if shared_depth < 1:
        raise ValueError("baseloop.shared_depth must be >= 1.")
    if max_model_loops < 1:
        raise ValueError("baseloop.max_model_loops must be >= 1.")

    baseloop_cfg["shared_depth"] = shared_depth
    baseloop_cfg["max_model_loops"] = max_model_loops

    config_payload = dict(model_cfg)
    config_payload["family"] = family
    config_payload["num_hidden_layers"] = shared_depth
    config_payload["max_model_loops"] = max_model_loops
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
    config_payload["baseloop"] = baseloop_cfg
    return BaseLoopConfig(**config_payload)


def build_baseloop_model(
    model_cfg: Dict[str, Any],
    *,
    model_cls: type[BaseLoopForCausalLM],
    family: str,
    architectures: list[str],
    default_rms_norm_eps: float,
    default_rope_theta: float,
    default_use_qk_norm: bool,
    applied_via: str,
):
    from looplm_pipeline.models.attention import resolve_model_torch_dtype

    cfg = build_baseloop_config(
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
