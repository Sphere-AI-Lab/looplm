import argparse
import json
import os
from typing import Dict

from looplm_pipeline.models.attention import format_attention_backend_report, get_attention_backend_report
from looplm_pipeline.models.registry import build_model_from_config
from looplm_pipeline.optim.registry import compute_stable_lr
from looplm_pipeline.train.checkpoint import load_latest_checkpoint
from looplm_pipeline.train.dataloader import build_dataloader
from looplm_pipeline.train.fsdp_engine import TrainEngine
from looplm_pipeline.train.loop import run_pretrain_loop
from looplm_pipeline.utils.dist import cleanup_distributed, get_rank, get_world_size, init_distributed, is_dist, is_main_process
from looplm_pipeline.utils.exp_config import load_merged_config
from looplm_pipeline.utils.hashing import stable_hash
from looplm_pipeline.utils.io import dump_yaml, ensure_dir
from looplm_pipeline.utils.logging import JsonlLogger
from looplm_pipeline.utils.rng import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoopLM pretraining entrypoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint under output_dir/checkpoints.")
    parser.add_argument("--grad-accum-steps", type=int, help="Override gradient accumulation for a different world size.")
    return parser.parse_args()


def validate_global_batch_size(train_cfg: Dict, world_size: int) -> None:
    global_batch_size = int(train_cfg["global_batch_size"])
    micro_batch_size = int(train_cfg["micro_batch_size"])
    grad_accum_steps = int(train_cfg["grad_accum_steps"])
    if min(global_batch_size, micro_batch_size, grad_accum_steps, world_size) <= 0:
        raise ValueError("Batch sizes, grad_accum_steps and world_size must all be positive.")
    expected = micro_batch_size * grad_accum_steps * world_size
    if global_batch_size != expected:
        raise ValueError(
            "Invalid batch config: "
            f"global_batch_size={global_batch_size}, but "
            f"micro_batch_size({micro_batch_size}) * "
            f"grad_accum_steps({grad_accum_steps}) * world_size({world_size}) = {expected}."
        )


def build_resume_metadata(cfg: Dict, world_size: int) -> Dict:
    train_cfg = cfg["train"]
    data_payload = {
        "data_sources": cfg["data_sources"],
        "tokenize": cfg["tokenize"],
        "train_data_loader": {
            "drop_last": True,
            "micro_batch_size": int(train_cfg["micro_batch_size"]),
            "grad_accum_steps": int(train_cfg["grad_accum_steps"]),
            "num_workers": int(train_cfg.get("num_workers", 0)),
            "prefetch_factor": int(train_cfg.get("prefetch_factor", 2)),
            "persistent_workers": bool(train_cfg.get("persistent_workers", int(train_cfg.get("num_workers", 0)) > 0)),
            "dataloader_seed": train_cfg.get("dataloader_seed", cfg["data_sources"].get("dataloader_seed", None)),
        },
        "world_size": int(world_size),
    }
    resume_payload = {
        "model": cfg["model"],
        "optim": cfg["optim"],
        "schedule": cfg["schedule"],
        "fsdp": cfg["fsdp"],
        "train": cfg["train"],
        "data": data_payload,
        "world_size": int(world_size),
    }
    return {
        "resume_fingerprint": stable_hash(json.dumps(resume_payload, sort_keys=True, ensure_ascii=True, default=str)),
        "resume_fingerprint_payload": resume_payload,
        "data_pipeline_fingerprint": stable_hash(json.dumps(data_payload, sort_keys=True, ensure_ascii=True, default=str)),
        "data_pipeline_fingerprint_payload": data_payload,
    }


def resolve_resume_state(extra_state: Dict, current_resume_metadata: Dict) -> Dict:
    rank_local_state = extra_state.get("rank_local_state", {})
    resume_state = {}
    if isinstance(rank_local_state, dict) and isinstance(rank_local_state.get("train_state"), dict):
        resume_state = dict(rank_local_state["train_state"])
    elif isinstance(extra_state.get("train_state"), dict):
        resume_state = dict(extra_state["train_state"])

    for key, label in (
        ("resume_fingerprint", "training config"),
        ("data_pipeline_fingerprint", "data pipeline"),
    ):
        saved = resume_state.get(key)
        current = current_resume_metadata.get(key)
        if saved and current and saved != current:
            raise ValueError(
                f"Cannot reproducibly resume: {label} fingerprint changed. "
                f"checkpoint={saved}, current={current}."
            )
    return resume_state


def _write_run_start_files(cfg: Dict, output_dir: str, resume: bool) -> None:
    dump_yaml(f"{output_dir}/resolved_config.yaml", cfg)
    with open(f"{output_dir}/run_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_name": cfg["exp"]["experiment_name"],
                "package": "looplm_pipeline",
                "resume": bool(resume),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    if not resume:
        optim_cfg = cfg.get("optim", {})
        stable_lr = None
        if optim_cfg.get("ref_params") is not None and optim_cfg.get("ref_tokens") is not None:
            stable_lr = compute_stable_lr(float(optim_cfg["ref_params"]), float(optim_cfg["ref_tokens"]))
        with open(f"{output_dir}/metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "step": 0,
                        "hyperparams": {
                            "model": cfg["model"].get("name"),
                            "optimizer": optim_cfg.get("name"),
                            "lr": optim_cfg.get("lr", stable_lr),
                            "stable_lr_from_ref": stable_lr,
                            "schedule": cfg["schedule"].get("name"),
                            "loss": cfg["train"].get("loss", {"type": "causal_lm"}),
                        },
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def _train(args: argparse.Namespace) -> None:
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    cfg = load_merged_config(args.config)
    if args.grad_accum_steps is not None:
        cfg["train"]["grad_accum_steps"] = args.grad_accum_steps
    validate_global_batch_size(cfg["train"], world_size)
    resume_metadata = build_resume_metadata(cfg, world_size)
    seed_everything(int(cfg["exp"].get("seed", 42)) + rank)

    output_dir = cfg["exp"]["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(f"{output_dir}/checkpoints")

    if is_main_process():
        _write_run_start_files(cfg, output_dir, resume=args.resume)

    model = build_model_from_config(cfg["model"])
    if is_main_process():
        attn_report = get_attention_backend_report(model)
        print(f"Attention backend: {format_attention_backend_report(attn_report)}")

    engine = TrainEngine(model=model, fsdp_cfg=cfg["fsdp"], use_dist=is_dist(), local_rank=local_rank)
    optimizer = engine.build_optimizer(cfg["optim"])

    start_step = 0
    resume_state: Dict = {}
    if args.resume:
        start_step, extra_state = load_latest_checkpoint(
            ckpt_dir=f"{output_dir}/checkpoints",
            model=engine.model,
            optimizer=optimizer,
            map_location="cpu",
        )
        if start_step <= 0 and not extra_state:
            raise FileNotFoundError(f"--resume was requested but no checkpoint was found under {output_dir}/checkpoints")
        resume_state = resolve_resume_state(extra_state, resume_metadata)

    train_dataloader, val_dataloader = build_dataloader(
        data_cfg=cfg["data_sources"],
        tokenize_cfg=cfg["tokenize"],
        train_cfg=cfg["train"],
        rank=rank,
        world_size=world_size,
    )

    logger = JsonlLogger(f"{output_dir}/metrics.jsonl")
    run_pretrain_loop(
        engine=engine,
        optimizer=optimizer,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        logger=logger,
        train_cfg=cfg["train"],
        schedule_cfg=cfg["schedule"],
        output_dir=output_dir,
        start_step=start_step,
        resume_state=resume_state,
        checkpoint_metadata=resume_metadata,
    )

    if is_main_process():
        print("Training finished.")


def main() -> None:
    args = parse_args()
    try:
        init_distributed()
        _train(args)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
