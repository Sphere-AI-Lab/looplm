from typing import Any, Dict, List


def lr_multiplier(
    step: int,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    total_steps: int,
    name: str = "wsd",
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0.")
    if warmup_steps < 0 or stable_steps < 0 or decay_steps < 0:
        raise ValueError("warmup_steps/stable_steps/decay_steps must be >= 0.")

    # Keep boundary behavior stable when step exceeds configured schedule.
    s = min(max(step, 0), total_steps)

    if name == "wsd":
        if warmup_steps + stable_steps + decay_steps != total_steps:
            raise ValueError("warmup_steps + stable_steps + decay_steps must equal total_steps.")

        # Warmup: linear increase to 1.0 * eta_stable.
        if s <= warmup_steps:
            if warmup_steps == 0:
                return 1.0
            return float(s) / float(warmup_steps)

        # Stable stage: hold eta_stable.
        if s <= warmup_steps + stable_steps:
            return 1.0

        # Decay: linear drop from 1.0 to 0.1.
        if decay_steps == 0:
            return 0.1
        decay_progress = float(s - warmup_steps - stable_steps) / float(decay_steps)
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        return 1.0 - 0.9 * decay_progress

    if name == "ws":
        if decay_steps != 0:
            raise ValueError("For ws, decay_steps must be 0.")
        if warmup_steps + stable_steps != total_steps:
            raise ValueError("warmup_steps + stable_steps must equal total_steps for ws.")

        if s <= warmup_steps:
            if warmup_steps == 0:
                return 1.0
            return float(s) / float(warmup_steps)
        return 1.0

    if name == "cosine_decay":
        import math

        if warmup_steps >= total_steps:
            raise ValueError("For cosine_decay, warmup_steps must be < total_steps.")

        # Warmup: linear increase to peak LR.
        if s <= warmup_steps:
            if warmup_steps == 0:
                return 1.0
            return float(s) / float(warmup_steps)

        # Cosine decay from 1.0 to 0.1.
        decay_progress = float(s - warmup_steps) / float(total_steps - warmup_steps)
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return 0.1 + 0.9 * cosine

    raise ValueError(f"Unsupported scheduler: {name}")


def apply_lr_for_step(
    optimizer: Any,
    schedule_cfg: Dict,
    base_lrs: List[float],
    step: int,
    *,
    total_steps: int,
) -> None:
    """Set param group learning rates for this step from schedule_cfg and base_lrs."""
    name = str(schedule_cfg.get("name", "wsd"))
    if name == "constant_lr":
        if "lr" not in schedule_cfg:
            raise ValueError("constant_lr schedule requires key 'lr'.")
        lr_const = float(schedule_cfg["lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr_const
        return
    mult = lr_multiplier(
        step=step,
        warmup_steps=int(schedule_cfg.get("warmup_steps", 0)),
        stable_steps=int(schedule_cfg.get("stable_steps", 0)),
        decay_steps=int(schedule_cfg.get("decay_steps", 0)),
        total_steps=total_steps,
        name=name,
    )
    for i, group in enumerate(optimizer.param_groups):
        group["lr"] = base_lrs[i] * mult
