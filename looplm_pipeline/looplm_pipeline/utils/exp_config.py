"""Load experiment yaml plus referenced configs (with optional data_sources overlay)."""
import os
from typing import Any, Dict


def resolve_exp_config_path(exp_cfg_path: str, target: str) -> str:
    if os.path.isabs(target):
        return target
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, target)


def deep_merge_dicts(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overlay into base (overlay wins for leaf values)."""
    out = dict(base)
    for key, val in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge_dicts(out[key], val)
        else:
            out[key] = val
    return out


def load_merged_config(exp_cfg_path: str) -> Dict[str, Any]:
    from looplm_pipeline.utils.io import load_yaml

    exp_cfg = load_yaml(exp_cfg_path)
    model_cfg = load_yaml(resolve_exp_config_path(exp_cfg_path, exp_cfg["model_config"]))
    model_overlay_key = "model_overlay_config"
    if exp_cfg.get(model_overlay_key):
        overlay = load_yaml(resolve_exp_config_path(exp_cfg_path, exp_cfg[model_overlay_key]))
        model_cfg = deep_merge_dicts(model_cfg, overlay)
    if exp_cfg.get("model_overrides"):
        model_cfg = deep_merge_dicts(model_cfg, exp_cfg["model_overrides"])

    data_sources = load_yaml(resolve_exp_config_path(exp_cfg_path, exp_cfg["data_sources_config"]))
    overlay_key = "data_sources_overlay_config"
    if exp_cfg.get(overlay_key):
        overlay = load_yaml(resolve_exp_config_path(exp_cfg_path, exp_cfg[overlay_key]))
        data_sources = deep_merge_dicts(data_sources, overlay)

    return {
        "exp": exp_cfg,
        "model": model_cfg,
        "data_sources": data_sources,
        "tokenize": load_yaml(resolve_exp_config_path(exp_cfg_path, exp_cfg["tokenize_config"])),
        "fsdp": deep_merge_dicts(
            load_yaml(resolve_exp_config_path(exp_cfg_path, exp_cfg["fsdp_config"])),
            exp_cfg.get("fsdp_overrides", {}),
        ),
        "optim": load_yaml(resolve_exp_config_path(exp_cfg_path, exp_cfg["optim_config"])),
        "schedule": load_yaml(resolve_exp_config_path(exp_cfg_path, exp_cfg["schedule_config"])),
        "train": exp_cfg["train"],
    }
