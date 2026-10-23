from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_MISSING_YAML = ""
try:
    import yaml  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover
    _MISSING_YAML = f"PyYAML is required for config integrity tests: {exc}"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from looplm_pipeline.utils.exp_config import load_merged_config


@unittest.skipIf(bool(_MISSING_YAML), _MISSING_YAML)
class TestConfigIntegrity(unittest.TestCase):
    def test_all_experiment_configs_are_minimal_looplm_configs(self):
        exp_paths = sorted((ROOT / "configs" / "experiments").glob("*.yaml"))
        self.assertEqual(len(exp_paths), 8)

        seen = set()
        for path in exp_paths:
            cfg = load_merged_config(str(path))
            seen.add(path.name)
            self.assertIn(cfg["model"]["family"], {"baseloop_llama_like", "loopformer_llama_like"})
            self.assertIn(cfg["optim"]["name"], {"adamw", "muon"})
            self.assertEqual(cfg["tokenize"]["seq_len"], 2048)
            train = cfg["train"]
            self.assertEqual(
                train["global_batch_size"],
                8 * train["micro_batch_size"] * train["grad_accum_steps"],
                path.name,
            )
            loss_type = cfg["train"].get("loss", {}).get("type", "causal_lm")
            if cfg["model"]["family"] == "loopformer_llama_like":
                self.assertEqual(loss_type, "loopformer_self_distill")
            else:
                self.assertEqual(loss_type, "causal_lm")

        expected = {
            "llama3_1_480m_baseloop_adamw.yaml",
            "llama3_1_480m_baseloop_muon.yaml",
            "llama3_1_480m_loopformer_adamw.yaml",
            "llama3_1_480m_loopformer_muon.yaml",
            "llama3_1_760m_baseloop_adamw.yaml",
            "llama3_1_760m_baseloop_muon.yaml",
            "llama3_1_760m_loopformer_adamw.yaml",
            "llama3_1_760m_loopformer_muon.yaml",
        }
        self.assertEqual(seen, expected)


if __name__ == "__main__":
    unittest.main()
