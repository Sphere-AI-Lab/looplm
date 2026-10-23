import inspect
from typing import Dict, Iterable

import torch


def build_adamw(params: Iterable, cfg: Dict):
    params = list(params)
    optim_kwargs = {
        "lr": float(cfg["lr"]),
        "betas": tuple(cfg.get("betas", [0.9, 0.95])),
        "eps": float(cfg.get("eps", 1.0e-8)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
    }

    sig = inspect.signature(torch.optim.AdamW)
    if "fused" in sig.parameters and bool(cfg.get("use_fused", True)):
        has_cuda_param = any(getattr(p, "is_cuda", False) for p in params if p.requires_grad)
        if has_cuda_param:
            optim_kwargs["fused"] = True

    return torch.optim.AdamW(params, **optim_kwargs)
