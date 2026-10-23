import random
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
        ignore_index=-100,
    )


def causal_lm_loss_per_sequence(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    bsz, seqlen, vocab = shift_logits.shape
    loss_tok = F.cross_entropy(
        shift_logits.view(bsz * seqlen, vocab),
        shift_labels.view(bsz * seqlen),
        reduction="none",
        ignore_index=-100,
    ).view(bsz, seqlen)

    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].to(loss_tok.dtype)
        denom = shift_mask.sum(dim=1).clamp_min(1.0)
        return (loss_tok * shift_mask).sum(dim=1) / denom

    valid = (shift_labels != -100).to(loss_tok.dtype)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (loss_tok * valid).sum(dim=1) / denom


@dataclass
class PretrainLossResult:
    loss: torch.Tensor
    metrics: Dict[str, torch.Tensor | float] = field(default_factory=dict)
    loss_seq: Optional[torch.Tensor] = None


def get_pretrain_loss_type(train_or_loss_cfg: Optional[Dict]) -> str:
    cfg = train_or_loss_cfg or {}
    if "loss" in cfg and isinstance(cfg.get("loss"), dict):
        cfg = cfg["loss"]
    return str(cfg.get("type", "causal_lm")).strip().lower()


def sample_short_trajectory(max_loops: int) -> list[float]:
    max_loops = int(max_loops)
    if max_loops < 2:
        raise ValueError("max_loops must be >= 2")

    allowed = [x / max_loops for x in range(1, max_loops + 1)]
    min_step = allowed[0]
    length = random.randint(1, max_loops - 1)
    trajectory = []
    remaining = 1.0
    remaining_steps = length

    for _ in range(length - 1):
        cap = remaining - (remaining_steps - 1) * min_step
        choices = [step for step in allowed if step <= cap + 1e-12]
        step = random.choice(choices)
        trajectory.append(float(step))
        remaining -= step
        remaining_steps -= 1

    trajectory.append(float(remaining))
    return trajectory


def _masked_hidden_mse(
    teacher_hidden: torch.Tensor,
    student_hidden: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    per_token = (teacher_hidden.detach().float() - student_hidden.float()).pow(2).mean(dim=-1)
    if attention_mask is None:
        return per_token.mean()
    mask = attention_mask.to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)


def _full_loop_steps(max_loops: int) -> list[float]:
    max_loops = int(max_loops)
    return [1.0 / float(max_loops)] * max_loops


def _loopformer_self_distill_loss(model, batch: Dict[str, torch.Tensor], loss_cfg: Dict) -> PretrainLossResult:
    max_loops = int(loss_cfg.get("max_model_loops", loss_cfg.get("max_loops", 8)))
    short_ce_weight = float(loss_cfg.get("short_ce_weight", 0.1))
    consistency_weight = float(loss_cfg.get("consistency_weight", 0.1))

    long_steps = _full_loop_steps(max_loops)
    short_steps = sample_short_trajectory(max_loops)
    model_kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    long_out = model(**model_kwargs, loop_steps=long_steps)
    short_out = model(**model_kwargs, loop_steps=short_steps)
    long_loss = causal_lm_loss(long_out.logits, batch["labels"])
    short_loss = causal_lm_loss(short_out.logits, batch["labels"])
    consistency_loss = _masked_hidden_mse(
        teacher_hidden=long_out.last_hidden_state,
        student_hidden=short_out.last_hidden_state,
        attention_mask=batch.get("attention_mask", None),
    )
    loss = long_loss + short_ce_weight * short_loss + consistency_weight * consistency_loss
    return PretrainLossResult(
        loss=loss,
        metrics={
            "loss_long": long_loss.detach(),
            "loss_short": short_loss.detach(),
            "loss_consistency": consistency_loss.detach(),
            "loopformer_short_steps": float(len(short_steps)),
            "loopformer_short_total_dt": float(sum(short_steps)),
        },
    )


def compute_pretrain_loss(model, batch: Dict[str, torch.Tensor], loss_cfg: Optional[Dict] = None) -> PretrainLossResult:
    cfg = dict(loss_cfg or {})
    loss_type = get_pretrain_loss_type(cfg)
    if loss_type in {"causal_lm", "ce", "cross_entropy"}:
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask", None))
        return PretrainLossResult(loss=causal_lm_loss(outputs.logits, batch["labels"]))
    if loss_type == "loopformer_self_distill":
        return _loopformer_self_distill_loss(model, batch, cfg)
    raise ValueError(f"Unsupported pretraining loss type: {loss_type}")


def compute_validation_loss(model, batch: Dict[str, torch.Tensor], loss_cfg: Optional[Dict] = None) -> torch.Tensor:
    cfg = dict(loss_cfg or {})
    loss_type = get_pretrain_loss_type(cfg)
    if loss_type == "loopformer_self_distill":
        max_loops = int(cfg.get("max_model_loops", cfg.get("max_loops", 8)))
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            loop_steps=_full_loop_steps(max_loops),
        )
        return causal_lm_loss(outputs.logits, batch["labels"])
    return compute_pretrain_loss(model, batch, cfg).loss
