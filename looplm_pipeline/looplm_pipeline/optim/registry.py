from typing import Dict, List, Tuple

import torch

from looplm_pipeline.optim.adamw import build_adamw
from looplm_pipeline.optim.muon import build_muon


# eta_stable(N,D) ~= 1e-4 * 38.4588 * N^{-0.2219} * D^{-0.3509}
# N, D are measured in billions (B).
_LR_STABLE_BASE = 1e-4
_LR_PAPER_C = 38.4588
_LR_PAPER_ALPHA_N = 0.2219
_LR_PAPER_ALPHA_D = 0.3509


def compute_stable_lr(ref_params_b: float, ref_tokens_b: float) -> float:
    if ref_params_b <= 0.0 or ref_tokens_b <= 0.0:
        raise ValueError("ref_params_b and ref_tokens_b must be > 0.")
    return (
        _LR_STABLE_BASE
        * _LR_PAPER_C
        * (ref_params_b ** (-_LR_PAPER_ALPHA_N))
        * (ref_tokens_b ** (-_LR_PAPER_ALPHA_D))
    )


def _fill_lr_from_ref_if_missing(optim_cfg: Dict) -> Dict:
    cfg = dict(optim_cfg)
    if cfg.get("lr", None) is not None:
        return cfg
    ref_params_b = cfg.get("ref_params")
    ref_tokens_b = cfg.get("ref_tokens")
    if ref_params_b is None or ref_tokens_b is None:
        raise ValueError("optim config must provide lr or both ref_params and ref_tokens.")
    cfg["lr"] = compute_stable_lr(float(ref_params_b), float(ref_tokens_b))
    return cfg


def build_optimizer(params, optim_cfg: Dict):
    params_list = list(params)
    named_params: List[Tuple[str, torch.nn.Parameter]] = []
    plain_params: List[torch.nn.Parameter] = []
    if params_list and isinstance(params_list[0], tuple):
        named_params = [(str(name), p) for name, p in params_list]
        plain_params = [p for _, p in named_params]
    else:
        plain_params = params_list

    name = str(optim_cfg["name"]).lower()
    if name == "adamw":
        return build_adamw(plain_params, _fill_lr_from_ref_if_missing(optim_cfg))
    if name == "muon":
        return build_muon(named_params if named_params else plain_params, _fill_lr_from_ref_if_missing(optim_cfg))
    raise ValueError(f"Unsupported optimizer: {name}. Supported: adamw, muon")
