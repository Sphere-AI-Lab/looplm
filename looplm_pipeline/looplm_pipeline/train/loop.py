import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch

from looplm_pipeline.optim.schedulers import apply_lr_for_step
from looplm_pipeline.train.checkpoint import save_checkpoint
from looplm_pipeline.train.loss import compute_pretrain_loss, get_pretrain_loss_type
from looplm_pipeline.train.runtime_utils import _all_ranks_finite, _format_duration, _maybe_save_emergency_checkpoint, _run_validation
from looplm_pipeline.utils.dist import barrier, get_world_size, is_main_process
from looplm_pipeline.utils.logging import JsonlLogger


def _next_batch_with_restart(dataloader, data_iter) -> Tuple[Dict[str, torch.Tensor], object]:
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(dataloader)
        try:
            return next(data_iter), data_iter
        except StopIteration as exc:
            raise RuntimeError("Training dataloader produced no batches.") from exc


def _skip_dataloader_microbatches(dataloader, data_iter, microbatches_to_skip: int) -> object:
    remaining = int(microbatches_to_skip)
    if remaining <= 0:
        return data_iter
    if is_main_process():
        print(f"Resume data position: skipping {remaining} already-consumed microbatch(es).", flush=True)
    last_log = time.perf_counter()
    skipped = 0
    while skipped < remaining:
        _, data_iter = _next_batch_with_restart(dataloader, data_iter)
        skipped += 1
        if is_main_process() and skipped < remaining:
            now = time.perf_counter()
            if (now - last_log) >= 30.0:
                print(f"Resume data position: skipped {skipped}/{remaining} microbatch(es).", flush=True)
                last_log = now
    if is_main_process():
        print(f"Resume data position: skipped {remaining} microbatch(es).", flush=True)
    return data_iter


def _build_loop_train_state(
    *,
    step: int,
    grad_accum_steps: int,
    tokens_seen_total: int,
    total_elapsed_seconds: float,
    base_lrs: List[float],
    checkpoint_metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    state = {
        "version": 1,
        "step": int(step),
        "microbatches_consumed": int(step) * int(grad_accum_steps),
        "grad_accum_steps": int(grad_accum_steps),
        "tokens_seen_total": int(tokens_seen_total),
        "elapsed_time_seconds": float(total_elapsed_seconds),
        "base_lrs": [float(lr) for lr in base_lrs],
        "controller_state": {},
    }
    if checkpoint_metadata:
        state.update(dict(checkpoint_metadata))
    return state


def _mean_metric_values(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    return {name: float(sum(values) / len(values)) for name, values in metrics.items() if values}


def run_pretrain_loop(
    engine,
    optimizer: torch.optim.Optimizer,
    dataloader,
    logger: JsonlLogger,
    train_cfg: Dict,
    schedule_cfg: Dict,
    output_dir: str,
    start_step: int = 0,
    val_dataloader: Optional[object] = None,
    resume_state: Optional[Dict[str, Any]] = None,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    resume_state = dict(resume_state or {})
    model = engine.model
    model.train()
    if hasattr(optimizer, "_global_step"):
        optimizer._global_step = int(start_step)

    pretrain_loss_cfg = train_cfg.get("loss", {})
    pretrain_loss_type = get_pretrain_loss_type(pretrain_loss_cfg)
    max_steps = int(train_cfg["max_steps"])
    grad_accum_steps = int(train_cfg["grad_accum_steps"])
    log_every = int(train_cfg.get("log_interval", 10))
    save_every = int(train_cfg.get("save_checkpoint_interval", 1000))
    validation_interval = int(train_cfg.get("validation_interval", 0))
    validation_max_batches = int(train_cfg.get("validation_max_batches", train_cfg.get("max_validation_batches", 0)))
    validation_log_interval_batches = int(train_cfg.get("validation_log_interval_batches", 0))
    validation_log_interval_seconds = float(train_cfg.get("validation_log_interval_seconds", 30.0))
    world_size = get_world_size()
    schedule_total_steps = int(schedule_cfg.get("total_steps", max_steps))

    saved_base_lrs = resume_state.get("base_lrs", None)
    if isinstance(saved_base_lrs, list) and len(saved_base_lrs) == len(optimizer.param_groups):
        base_lrs = [float(lr) for lr in saved_base_lrs]
    else:
        base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    total_elapsed_seconds = float(resume_state.get("elapsed_time_seconds", 0.0) or 0.0)
    tokens_seen_total = int(resume_state.get("tokens_seen_total", 0) or 0)
    tokens_per_sec_window: Deque[float] = deque(maxlen=100)
    run_peak_mem_allocated = 0.0
    run_peak_mem_reserved = 0.0

    data_iter = iter(dataloader)
    step = int(start_step)
    default_resume_microbatches = step * grad_accum_steps
    resume_microbatches = int(resume_state.get("microbatches_consumed", default_resume_microbatches) or 0)
    if step > 0 and step < max_steps and resume_microbatches > 0:
        data_iter = _skip_dataloader_microbatches(dataloader, data_iter, resume_microbatches)

    while step < max_steps:
        step_start_time = time.perf_counter()
        step_setup_start_time = step_start_time
        if engine.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(engine.device)
        engine.zero_grad(optimizer)
        step_setup_seconds = time.perf_counter() - step_setup_start_time

        running_loss = 0.0
        tokens_this_step = 0
        step_loss_metrics: Dict[str, List[float]] = {}

        grad_accumulation_start_time = time.perf_counter()
        for accum_idx in range(grad_accum_steps):
            batch, data_iter = _next_batch_with_restart(dataloader, data_iter)
            batch = {k: v.to(engine.device, non_blocking=True) for k, v in batch.items()}
            tokens_this_step += batch["input_ids"].numel() * world_size
            sync_gradients = accum_idx == (grad_accum_steps - 1)

            with engine.grad_sync_context(sync_gradients=sync_gradients):
                with engine.autocast_context():
                    loss_result = compute_pretrain_loss(model, batch, pretrain_loss_cfg)
                    base_loss = loss_result.loss
                    for metric_name, metric_value in loss_result.metrics.items():
                        if isinstance(metric_value, torch.Tensor):
                            metric_value = float(metric_value.detach().float().cpu())
                        step_loss_metrics.setdefault(metric_name, []).append(float(metric_value))

                loss = base_loss / grad_accum_steps
                if not _all_ranks_finite(torch.isfinite(loss), engine.device):
                    reason = f"non_finite_loss detected across ranks at step={step + 1}"
                    _maybe_save_emergency_checkpoint(engine, optimizer, output_dir, step, reason)
                    raise RuntimeError(reason)
                running_loss += float(loss.detach().cpu())
                engine.backward(loss)
        grad_accumulation_seconds = time.perf_counter() - grad_accumulation_start_time

        optimizer_pre_step_start_time = time.perf_counter()
        next_step = step + 1
        apply_lr_for_step(optimizer, schedule_cfg, base_lrs, next_step, total_steps=schedule_total_steps)

        gradients_finite = torch.ones((), dtype=torch.bool, device=engine.device)
        for p in model.parameters():
            if p.grad is None:
                continue
            gradients_finite &= torch.isfinite(p.grad).all()
        if not _all_ranks_finite(gradients_finite, engine.device):
            reason = f"non_finite_gradients detected across ranks at step={next_step}"
            _maybe_save_emergency_checkpoint(engine, optimizer, output_dir, step, reason)
            raise RuntimeError(reason)
        optimizer_pre_step_seconds = time.perf_counter() - optimizer_pre_step_start_time

        optimizer_step_start_time = time.perf_counter()
        engine.optimizer_step(optimizer)
        step = next_step
        optimizer_step_seconds = time.perf_counter() - optimizer_step_start_time
        optimizer_post_step_seconds = 0.0

        elapsed = time.perf_counter() - step_start_time
        timed_step_seconds = step_setup_seconds + grad_accumulation_seconds + optimizer_pre_step_seconds + optimizer_step_seconds
        timing_gap_seconds = float(elapsed - timed_step_seconds)
        total_elapsed_seconds += elapsed
        tokens_seen_total += tokens_this_step
        step_tokens_per_sec = float(tokens_this_step / max(elapsed, 1e-6))
        tokens_per_sec_window.append(step_tokens_per_sec)
        tokens_per_sec_avg_100 = float(sum(tokens_per_sec_window) / max(1, len(tokens_per_sec_window)))

        peak_mem_allocated = None
        peak_mem_reserved = None
        if engine.device.type == "cuda":
            peak_mem_allocated = float(torch.cuda.max_memory_allocated(engine.device))
            peak_mem_reserved = float(torch.cuda.max_memory_reserved(engine.device))
            run_peak_mem_allocated = max(run_peak_mem_allocated, peak_mem_allocated)
            run_peak_mem_reserved = max(run_peak_mem_reserved, peak_mem_reserved)

        should_log = (step % log_every == 0) or (step == max_steps)
        if should_log and is_main_process():
            avg_seconds_per_step = total_elapsed_seconds / float(max(1, step))
            remaining_steps = max(0, max_steps - step)
            eta_seconds = remaining_steps * avg_seconds_per_step
            log_entry = {
                "step": step,
                "loss": float(running_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "tokens_per_sec": step_tokens_per_sec,
                "tokens_per_sec_avg_100": tokens_per_sec_avg_100,
                "seconds_per_step": elapsed,
                "avg_seconds_per_step": float(avg_seconds_per_step),
                "elapsed_time_seconds": float(total_elapsed_seconds),
                "elapsed_time_hms": _format_duration(total_elapsed_seconds),
                "step_setup_seconds": float(step_setup_seconds),
                "grad_accumulation_seconds": float(grad_accumulation_seconds),
                "optimizer_pre_step_seconds": float(optimizer_pre_step_seconds),
                "optimizer_step_seconds": float(optimizer_step_seconds),
                "optimizer_post_step_seconds": float(optimizer_post_step_seconds),
                "timed_step_seconds": float(timed_step_seconds),
                "timing_gap_seconds": float(timing_gap_seconds),
                "eta_seconds": float(eta_seconds),
                "eta_hms": _format_duration(eta_seconds),
                "tokens_seen_total": int(tokens_seen_total),
                "epoch_like_progress": float(step / max(1, max_steps)),
            }
            if peak_mem_allocated is not None and peak_mem_reserved is not None:
                mb = 1024.0 * 1024.0
                log_entry.update(
                    {
                        "max_memory_allocated_mb": peak_mem_allocated / mb,
                        "max_memory_reserved_mb": peak_mem_reserved / mb,
                        "run_peak_memory_allocated_mb": run_peak_mem_allocated / mb,
                        "run_peak_memory_reserved_mb": run_peak_mem_reserved / mb,
                    }
                )
            log_entry.update(_mean_metric_values(step_loss_metrics))
            logger.log(log_entry)

        should_run_validation = (
            validation_interval > 0
            and val_dataloader is not None
            and ((step % validation_interval == 0) or (step == max_steps))
        )
        if should_run_validation:
            validation_kwargs = {
                "max_batches": validation_max_batches,
                "log_interval_batches": validation_log_interval_batches,
                "log_interval_seconds": validation_log_interval_seconds,
                "step": step,
            }
            if pretrain_loss_type == "causal_lm":
                val_loss = _run_validation(engine, val_dataloader, **validation_kwargs)
            else:
                val_loss = _run_validation(engine, val_dataloader, loss_cfg=pretrain_loss_cfg, **validation_kwargs)
            if is_main_process():
                logger.log(
                    {
                        "step": step,
                        "validation_loss": float(val_loss),
                        "validation_max_batches": int(validation_max_batches),
                    }
                )

        should_save_checkpoint = (step % save_every == 0) or (step == max_steps)
        if should_save_checkpoint:
            train_state = _build_loop_train_state(
                step=step,
                grad_accum_steps=grad_accum_steps,
                tokens_seen_total=tokens_seen_total,
                total_elapsed_seconds=total_elapsed_seconds,
                base_lrs=base_lrs,
                checkpoint_metadata=checkpoint_metadata,
            )
            save_checkpoint(
                ckpt_dir=f"{output_dir}/checkpoints",
                model=model,
                optimizer=optimizer,
                step=step,
                extra_state={"loss": running_loss, "train_state": train_state},
                rank_local_state={"train_state": train_state},
            )

    barrier()
