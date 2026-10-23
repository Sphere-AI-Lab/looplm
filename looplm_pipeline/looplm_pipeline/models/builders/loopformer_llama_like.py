from typing import Any, Dict

from looplm_pipeline.models.builders.loopformer_common import (
    LoopFormerConfig,
    LoopFormerForCausalLM,
    build_loopformer_config,
    build_loopformer_model,
)


class LoopFormerLlamaForCausalLM(LoopFormerForCausalLM):
    pass


LoopFormerLlamaConfig = LoopFormerConfig


def _build_config(model_cfg: Dict[str, Any]) -> LoopFormerConfig:
    return build_loopformer_config(
        model_cfg,
        family="loopformer_llama_like",
        architectures=["LoopFormerLlamaForCausalLM"],
        default_rms_norm_eps=1.0e-5,
        default_rope_theta=10000.0,
        default_use_qk_norm=False,
    )


def build_loopformer_llama_like(model_cfg: Dict[str, Any]):
    return build_loopformer_model(
        model_cfg,
        model_cls=LoopFormerLlamaForCausalLM,
        family="loopformer_llama_like",
        architectures=["LoopFormerLlamaForCausalLM"],
        default_rms_norm_eps=1.0e-5,
        default_rope_theta=10000.0,
        default_use_qk_norm=False,
        applied_via="custom_loopformer_llama_like",
    )
