from contextlib import nullcontext
from typing import Dict

from looplm_pipeline.models.builders.baseloop_llama_like import build_baseloop_llama_like
from looplm_pipeline.models.builders.loopformer_llama_like import build_loopformer_llama_like
from looplm_pipeline.models.init_weights import init_model_weights

try:
    from transformers.modeling_utils import no_init_weights
except ImportError:  # pragma: no cover
    no_init_weights = None


BUILDERS = {
    "baseloop_llama_like": build_baseloop_llama_like,
    "loopformer_llama_like": build_loopformer_llama_like,
}


def build_model_from_config(
    model_cfg: Dict,
    *,
    init_weights: bool = True,
    disable_hf_init: bool = False,
):
    family = model_cfg["family"]
    if family not in BUILDERS:
        supported = ", ".join(sorted(BUILDERS))
        raise ValueError(f"Unsupported model family: {family}. Supported: {supported}")

    init_ctx = (
        no_init_weights()
        if disable_hf_init and no_init_weights is not None
        else nullcontext()
    )
    with init_ctx:
        model = BUILDERS[family](model_cfg)

    if init_weights:
        init_std = float(model_cfg.get("init_std", model_cfg.get("initializer_range", 0.02)))
        init_model_weights(model, std=init_std)
        if hasattr(model, "post_init_weights"):
            model.post_init_weights()
    return model
