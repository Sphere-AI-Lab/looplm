import os
import tempfile
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)

from looplm_pipeline.utils.io import ensure_dir
from looplm_pipeline.utils.rng import get_rng_state, set_rng_state


def _checkpoint_model(model: torch.nn.Module) -> torch.nn.Module:
    # Keep FSDP's state-dict hooks; only DDP can be safely unwrapped.
    return model.module if isinstance(model, DDP) else model


def _full_state_context(model: torch.nn.Module, *, rank0_only: bool):
    if not isinstance(model, FSDP):
        return nullcontext()
    # CPU FSDP can reuse the all-gather buffer for successive optimizer moments.
    # Disabling CPU offload when already on CPU makes FSDP clone those views,
    # preventing exp_avg from being overwritten by exp_avg_sq during collection.
    offload_optimizer = next(model.parameters()).device.type != "cpu"
    return FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only),
        FullOptimStateDictConfig(offload_to_cpu=offload_optimizer, rank0_only=rank0_only),
    )


def _atomic_save(payload: Dict, path: str) -> None:
    fd, tmp_path = tempfile.mkstemp(prefix=".checkpoint_", suffix=".tmp", dir=os.path.dirname(path))
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _load_checkpoint_payload(path: str, map_location: str):
    # PyTorch >=2.6 defaults torch.load(weights_only=True), which rejects
    # checkpoints containing optimizer/RNG/numpy states.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _ckpt_path(ckpt_dir: str, step: int) -> str:
    return os.path.join(ckpt_dir, f"step_{step:08d}.pt")


def _latest_checkpoint_path(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = [
        os.path.join(ckpt_dir, x)
        for x in os.listdir(ckpt_dir)
        if x.startswith("step_") and x.endswith(".pt")
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _rank_state_for_payload(rank_local_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "rank": int(_get_rank()),
        "rng_state": get_rng_state(),
        "rank_local_state": rank_local_state or {},
    }


def _gather_rank_states(rank_local_state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    local_state = _rank_state_for_payload(rank_local_state)
    if not _is_dist():
        return [local_state]

    gathered: Optional[List[Optional[Dict[str, Any]]]]
    gathered = [None for _ in range(_get_world_size())] if _get_rank() == 0 else None
    dist.gather_object(local_state, object_gather_list=gathered, dst=0)
    if _get_rank() != 0:
        return []
    assert gathered is not None
    return [state for state in gathered if state is not None]


def _rank_states_by_rank(rank_states: object) -> Dict[int, Dict[str, Any]]:
    if isinstance(rank_states, dict):
        out: Dict[int, Dict[str, Any]] = {}
        for key, value in rank_states.items():
            if isinstance(value, dict):
                out[int(key)] = value
        return out
    if isinstance(rank_states, list):
        out = {}
        for value in rank_states:
            if isinstance(value, dict) and "rank" in value:
                out[int(value["rank"])] = value
        return out
    return {}


def _restore_rank_rng_state(payload: Dict[str, Any]) -> None:
    rank_states = _rank_states_by_rank(payload.get("rank_states", {}))
    rank = _get_rank()
    if rank in rank_states and "rng_state" in rank_states[rank]:
        set_rng_state(rank_states[rank]["rng_state"])
        return
    if "rng_state" in payload:
        # Backward-compatible fallback for older rank-0-only checkpoints.
        set_rng_state(payload["rng_state"])


def save_checkpoint(
    ckpt_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    extra_state: Optional[Dict] = None,
    rank_local_state: Optional[Dict[str, Any]] = None,
) -> str:
    ensure_dir(ckpt_dir)
    path = _ckpt_path(ckpt_dir, step)
    rank_states = _gather_rank_states(rank_local_state)
    model = _checkpoint_model(model)
    sharded_optimizer = isinstance(model, FSDP) and not getattr(optimizer, "_looplm_full_parameter_state", False)
    # All ranks participate in FSDP's model/optimizer state collectives.
    with _full_state_context(model, rank0_only=True):
        model_state = model.state_dict()
        optimizer_state = FSDP.optim_state_dict(model, optimizer) if sharded_optimizer else optimizer.state_dict()

    save_error = [None]
    if _get_rank() == 0:
        payload = {
            "model": model_state,
            "optimizer": optimizer_state,
            "optimizer_format": "fsdp_full" if sharded_optimizer else "full",
            "step": step,
            "rng_state": rank_states[0]["rng_state"] if rank_states else get_rng_state(),
            "rank_states": rank_states,
            "extra_state": extra_state or {},
        }
        try:
            _atomic_save(payload, path)
        except Exception as exc:
            if not _is_dist():
                raise
            save_error[0] = f"{type(exc).__name__}: {exc}"
    if _is_dist():
        # Propagate rank-0 IO errors instead of leaving peers at a barrier.
        dist.broadcast_object_list(save_error, src=0)
    if save_error[0] is not None:
        raise RuntimeError(f"Checkpoint save failed: {save_error[0]}")
    return path


def load_latest_checkpoint(
    ckpt_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str = "cpu",
) -> Tuple[int, Dict]:
    latest = _latest_checkpoint_path(ckpt_dir)
    if latest is None:
        return 0, {}
    payload = _load_checkpoint_payload(latest, map_location=map_location)
    model = _checkpoint_model(model)
    sharded_optimizer = isinstance(model, FSDP) and not getattr(optimizer, "_looplm_full_parameter_state", False)
    expected_format = "fsdp_full" if sharded_optimizer else "full"
    if payload.get("optimizer_format", "full") != expected_format:
        raise ValueError("Checkpoint optimizer format does not match the current distributed optimizer.")
    with _full_state_context(model, rank0_only=False):
        model.load_state_dict(payload["model"], strict=True)
        optimizer_state = (
            FSDP.optim_state_dict_to_load(model, optimizer, payload["optimizer"])
            if sharded_optimizer
            else payload["optimizer"]
        )
    optimizer_context = (
        FSDP.summon_full_params(model, writeback=False)
        if isinstance(model, FSDP) and not sharded_optimizer
        else nullcontext()
    )
    with optimizer_context:
        optimizer.load_state_dict(optimizer_state)
    _restore_rank_rng_state(payload)
    step = int(payload.get("step", 0))
    extra_state = dict(payload.get("extra_state", {}))
    rank_states = _rank_states_by_rank(payload.get("rank_states", {}))
    rank_state = rank_states.get(_get_rank(), {})
    if isinstance(rank_state, dict) and isinstance(rank_state.get("rank_local_state"), dict):
        extra_state["rank_local_state"] = rank_state["rank_local_state"]
    return step, extra_state
