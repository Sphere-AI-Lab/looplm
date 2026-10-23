from __future__ import annotations

import os
import sys
import unittest

_MISSING_TORCH = ""
try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None
    _MISSING_TORCH = f"torch is required for loop model tests: {exc}"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if not _MISSING_TORCH:
    from looplm_pipeline.models.registry import build_model_from_config
    from looplm_pipeline.train.loss import compute_pretrain_loss
else:
    build_model_from_config = None
    compute_pretrain_loss = None


def _tiny_config(family: str) -> dict:
    cfg = {
        "name": f"tiny_{family}",
        "family": family,
        "hf_model_type": "llama",
        "architectures": ["TinyForCausalLM"],
        "model_type": "llama_loop",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "max_position_embeddings": 16,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_implementation": "sdpa",
        "mlp_bias": False,
        "rms_norm_eps": 1.0e-5,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "use_cache": False,
        "torch_dtype": "float32",
    }
    if family == "baseloop_llama_like":
        cfg["baseloop"] = {"shared_depth": 2, "max_model_loops": 3}
    elif family == "loopformer_llama_like":
        cfg["loopformer"] = {"shared_depth": 2, "max_model_loops": 3, "time_embedding_size": 16}
    else:
        raise ValueError(f"bad family: {family}")
    return cfg


def _batch() -> dict:
    input_ids = torch.randint(0, 32, (2, 6), dtype=torch.long)
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": torch.ones_like(input_ids),
    }


@unittest.skipIf(bool(_MISSING_TORCH), _MISSING_TORCH)
class TestLoopModels(unittest.TestCase):
    def test_baseloop_causal_lm_loss_backpropagates(self):
        torch.manual_seed(1)
        model = build_model_from_config(_tiny_config("baseloop_llama_like"))
        result = compute_pretrain_loss(model, _batch(), {"type": "causal_lm"})
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_loopformer_self_distill_loss_backpropagates(self):
        torch.manual_seed(2)
        model = build_model_from_config(_tiny_config("loopformer_llama_like"))
        result = compute_pretrain_loss(
            model,
            _batch(),
            {
                "type": "loopformer_self_distill",
                "max_model_loops": 3,
                "short_ce_weight": 0.1,
                "consistency_weight": 0.1,
            },
        )
        self.assertTrue(torch.isfinite(result.loss))
        self.assertIn("loss_long", result.metrics)
        self.assertIn("loss_short", result.metrics)
        result.loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
