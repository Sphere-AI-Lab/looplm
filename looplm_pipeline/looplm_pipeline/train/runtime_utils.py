import json
import time

import torch

from looplm_pipeline.train.checkpoint import save_checkpoint
from looplm_pipeline.train.loss import causal_lm_loss, compute_validation_loss
from looplm_pipeline.utils.dist import is_dist, is_main_process


def _format_duration(seconds: float) -> str:
    """Format duration seconds as HH:MM:SS."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _maybe_save_emergency_checkpoint(engine, optimizer, output_dir: str, step: int, reason: str) -> None:
    # All ranks must enter save_checkpoint. Diagnostic snapshots live separately
    # so --resume never selects a snapshot from an incomplete optimizer step.
    save_checkpoint(
        ckpt_dir=f"{output_dir}/checkpoints/emergency",
        model=engine.model,
        optimizer=optimizer,
        step=step,
        extra_state={"emergency_reason": reason},
    )


def _all_ranks_finite(is_finite: torch.Tensor, device: torch.device) -> bool:
    flag = is_finite.detach().to(device=device, dtype=torch.int32)
    if is_dist():
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item())


def _dist_every_rank_has_batch(has_batch: bool, device: torch.device) -> bool:
    flag = torch.tensor([1 if has_batch else 0], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(int(flag.item()) > 0)


def _run_validation(
    engine,
    val_dataloader,
    loss_cfg=None,
    *,
    max_batches: int = 0,
    log_interval_batches: int = 0,
    log_interval_seconds: float = 30.0,
    step: int | None = None,
) -> float:
    """Run one pass over validation set, return mean loss (reduced across ranks)."""
    model = engine.model
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    max_batches = max(0, int(max_batches or 0))
    log_interval_batches = max(0, int(log_interval_batches or 0))
    log_interval_seconds = max(0.0, float(log_interval_seconds))
    start_time = time.perf_counter()
    last_log_time = start_time

    if is_main_process():
        print(
            json.dumps(
                {
                    "step": step,
                    "event": "validation_start",
                    "max_batches": max_batches,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    with torch.no_grad():
        val_iter = iter(val_dataloader)
        while True:
            if max_batches > 0 and total_batches >= max_batches:
                break
            try:
                batch = next(val_iter)
                has_batch = True
            except StopIteration:
                batch = None
                has_batch = False

            if is_dist():
                has_batch = _dist_every_rank_has_batch(has_batch, engine.device)
            if not has_batch:
                break

            batch = {k: v.to(engine.device, non_blocking=True) for k, v in batch.items()}
            with engine.autocast_context():
                if loss_cfg is None:
                    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    loss = causal_lm_loss(outputs.logits, batch["labels"])
                else:
                    loss = compute_validation_loss(model, batch, loss_cfg)
            n = batch["input_ids"].numel()
            total_loss += loss.item() * n
            total_tokens += n
            total_batches += 1

            now = time.perf_counter()
            due_to_batches = log_interval_batches > 0 and total_batches % log_interval_batches == 0
            due_to_time = log_interval_seconds > 0.0 and (now - last_log_time) >= log_interval_seconds
            if is_main_process() and (due_to_batches or due_to_time):
                elapsed = now - start_time
                print(
                    json.dumps(
                        {
                            "step": step,
                            "event": "validation_progress",
                            "validation_batches": total_batches,
                            "validation_tokens_local": int(total_tokens),
                            "elapsed_time_seconds": float(elapsed),
                            "seconds_per_validation_batch": float(elapsed / max(1, total_batches)),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
                last_log_time = now

    model.train()
    totals_t = torch.tensor(
        [total_loss, float(total_tokens), float(total_batches)],
        dtype=torch.float64,
        device=engine.device,
    )
    if is_dist():
        torch.distributed.all_reduce(totals_t)
    agg_loss = totals_t[0].item()
    agg_tokens = totals_t[1].item()
    agg_batches = int(totals_t[2].item())
    mean_loss = agg_loss / max(agg_tokens, 1)
    if is_main_process():
        print(
            json.dumps(
                {
                    "step": step,
                    "event": "validation_done",
                    "validation_loss": float(mean_loss),
                    "validation_batches": int(agg_batches),
                    "validation_tokens": int(agg_tokens),
                    "elapsed_time_seconds": float(time.perf_counter() - start_time),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    return mean_loss
