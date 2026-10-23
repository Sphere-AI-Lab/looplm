import math
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


_NS_COEFFS: Dict[str, Tuple[Tuple[float, float, float], ...]] = {
    "KJ": (
        (3.4445, -4.775, 2.0315),
        (3.4445, -4.775, 2.0315),
        (3.4445, -4.775, 2.0315),
        (3.4445, -4.775, 2.0315),
        (3.4445, -4.775, 2.0315),
    ),
}

_NS_ALIASES = {"kj": "KJ"}


def _canonical_ns_method(ns_method: str) -> str:
    key = str(ns_method).strip()
    canonical = _NS_ALIASES.get(key.lower(), key)
    if canonical not in _NS_COEFFS:
        supported = ", ".join(sorted(_NS_COEFFS))
        raise ValueError(f"Unsupported ns_method={ns_method!r}. This minimal pipeline supports: {supported}")
    return canonical


def _get_ns_coeffs(ns_method: str, step_idx: int) -> Tuple[float, float, float]:
    coeffs = _NS_COEFFS[_canonical_ns_method(ns_method)]
    return coeffs[min(int(step_idx), len(coeffs) - 1)]


def zeropower_via_newtonschulz5(
    grad_matrix: torch.Tensor,
    steps: int,
    ns_method: str = "KJ",
    epsilon: float = 1e-7,
) -> torch.Tensor:
    if grad_matrix.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(grad_matrix.shape)}")
    steps_i = int(steps)
    if steps_i < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    x = grad_matrix.to(dtype=torch.float32)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.transpose(0, 1)
    x = x / (x.norm() + epsilon)
    x = x.to(dtype=torch.bfloat16)

    for step_idx in range(steps_i):
        a, b, c = _get_ns_coeffs(ns_method, step_idx)
        gram = x @ x.transpose(0, 1)
        x = float(a) * x + (float(b) * gram + float(c) * (gram @ gram)) @ x

    if transposed:
        x = x.transpose(0, 1)
    return x


class Muon(torch.optim.Optimizer):
    """Muon for matrix parameters, with AdamW fallback for embeddings, heads and vectors."""

    def __init__(
        self,
        *,
        lr: float,
        wd: float,
        muon_params: Sequence[torch.nn.Parameter],
        adamw_params: Sequence[torch.nn.Parameter],
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_method: str = "KJ",
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1.0e-8,
    ):
        params = list(muon_params) + list(adamw_params)
        if not params:
            raise ValueError("Muon got an empty parameter list.")
        defaults = {
            "lr": float(lr),
            "wd": float(wd),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "ns_steps": int(ns_steps),
            "ns_method": str(ns_method),
            "adamw_betas": tuple(float(x) for x in adamw_betas),
            "adamw_eps": float(adamw_eps),
        }
        super().__init__(params, defaults)
        muon_param_ids = {id(p) for p in muon_params}
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["use_muon"] = id(p) in muon_param_ids

    @staticmethod
    def _adjust_lr_for_muon(lr: float, param_shape: torch.Size) -> float:
        rows, cols = param_shape[:2]
        return float(lr) * (0.2 * math.sqrt(float(max(rows, cols))))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["wd"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            ns_method = str(group["ns_method"])
            beta1, beta2 = group["adamw_betas"]
            eps = float(group["adamw_eps"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if state.get("use_muon", False):
                    grad_matrix = grad if grad.ndim == 2 else grad.view(grad.size(0), -1)
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad_matrix)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad_matrix)
                    update_source = grad_matrix.add(buf, alpha=momentum) if nesterov else buf
                    orth_update = zeropower_via_newtonschulz5(
                        update_source,
                        steps=ns_steps,
                        ns_method=ns_method,
                    ).to(dtype=p.dtype)
                    if grad.ndim != 2:
                        orth_update = orth_update.view_as(grad)
                    adjusted_lr = self._adjust_lr_for_muon(lr, p.shape)
                    p.mul_(1.0 - lr * wd)
                    p.add_(orth_update, alpha=-adjusted_lr)
                    continue

                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(grad)
                    state["moment2"] = torch.zeros_like(grad)

                state["step"] += 1
                step = state["step"]
                m = state["moment1"]
                v = state["moment2"]
                m.lerp_(grad, 1.0 - beta1)
                v.lerp_(grad.square(), 1.0 - beta2)
                normed_grad = m / (v.sqrt() + eps)
                bias_correction1 = 1.0 - (beta1**step)
                bias_correction2 = 1.0 - (beta2**step)
                scale = bias_correction1 / math.sqrt(bias_correction2)

                p.mul_(1.0 - lr * wd)
                p.add_(normed_grad, alpha=-(lr / scale))

        return loss


def _normalize_named_params(params: Iterable) -> List[Tuple[str, torch.nn.Parameter]]:
    items = list(params)
    if not items:
        return []
    first = items[0]
    if isinstance(first, tuple) and len(first) == 2 and isinstance(first[1], torch.nn.Parameter):
        return [(str(name), p) for name, p in items]
    return [("", p) for p in items]


def _should_use_muon(name: str, param: torch.nn.Parameter, exclude_keywords: Sequence[str]) -> bool:
    if not param.requires_grad:
        return False
    if param.ndim < 2:
        return False
    lname = name.lower()
    return not any(keyword in lname for keyword in exclude_keywords)


def build_muon(params: Iterable, cfg: Dict):
    unsupported = {
        "step_adaptive_muon",
        "ns_steps_by_layer_type",
        "ns_schedule_by_layer_type",
        "ns_schedule_by_shape",
        "ns_precondition",
        "norm_mode",
    }
    present = sorted(key for key in unsupported if key in cfg)
    if present:
        raise ValueError(f"Unsupported Muon options in minimal pipeline: {present}")

    named_params = _normalize_named_params(params)
    exclude_keywords = list(
        cfg.get(
            "exclude_name_keywords",
            ["embed", "embedding", "lm_head", "output", "classifier"],
        )
    )
    muon_params: List[torch.nn.Parameter] = []
    adamw_params: List[torch.nn.Parameter] = []
    for name, p in named_params:
        if _should_use_muon(name, p, exclude_keywords):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    if not muon_params:
        return torch.optim.AdamW(
            adamw_params,
            lr=float(cfg["lr"]),
            betas=tuple(cfg.get("adamw_betas", [0.9, 0.95])),
            eps=float(cfg.get("adamw_eps", 1.0e-8)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )

    return Muon(
        lr=float(cfg["lr"]),
        wd=float(cfg.get("weight_decay", 0.0)),
        muon_params=muon_params,
        adamw_params=adamw_params,
        momentum=float(cfg.get("momentum", 0.95)),
        nesterov=bool(cfg.get("nesterov", True)),
        ns_steps=int(cfg.get("ns_steps", 5)),
        ns_method=str(cfg.get("ns_method", "KJ")),
        adamw_betas=tuple(cfg.get("adamw_betas", [0.9, 0.95])),
        adamw_eps=float(cfg.get("adamw_eps", 1.0e-8)),
    )
