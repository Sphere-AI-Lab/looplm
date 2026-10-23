from typing import Any, Dict

from looplm_pipeline.models.builders.baseloop_common import (
    BaseLoopConfig,
    BaseLoopForCausalLM,
    build_baseloop_config,
    build_baseloop_model,
)


class BaseLoopLlamaForCausalLM(BaseLoopForCausalLM):
    pass


BaseLoopLlamaConfig = BaseLoopConfig


def _build_config(model_cfg: Dict[str, Any]) -> BaseLoopConfig:
    return build_baseloop_config(
        model_cfg,
        family="baseloop_llama_like",
        architectures=["BaseLoopLlamaForCausalLM"],
        default_rms_norm_eps=1.0e-5,
        default_rope_theta=10000.0,
        default_use_qk_norm=False,
    )


def build_baseloop_llama_like(model_cfg: Dict[str, Any]):
    return build_baseloop_model(
        model_cfg,
        model_cls=BaseLoopLlamaForCausalLM,
        family="baseloop_llama_like",
        architectures=["BaseLoopLlamaForCausalLM"],
        default_rms_norm_eps=1.0e-5,
        default_rope_theta=10000.0,
        default_use_qk_norm=False,
        applied_via="custom_baseloop_llama_like",
    )
