from contextlib import nullcontext
from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from looplm_pipeline.optim.registry import build_optimizer

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:  # pragma: no cover
    FSDP = None


def _autocast_dtype(mixed_precision: str):
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return None


class TrainEngine:
    def __init__(self, model: torch.nn.Module, fsdp_cfg: Dict, use_dist: bool, local_rank: int):
        self.fsdp_cfg = fsdp_cfg
        self.grad_clip = float(fsdp_cfg.get("grad_clip_norm", 1.0))
        self.mixed_precision = str(fsdp_cfg.get("mixed_precision", "bf16")).lower()
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if local_rank >= device_count:
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible. "
                    "Check mpirun/torchrun mapping and CUDA_VISIBLE_DEVICES."
                )
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
        model.to(self.device)

        self.is_fsdp = bool(fsdp_cfg.get("enabled", False))
        self.full_parameter_optimizer = False
        if self.is_fsdp:
            if not use_dist:
                raise ValueError("FSDP requires distributed training; launch with torchrun and at least two processes.")
            if FSDP is None:
                raise ImportError("FSDP was requested but torch.distributed.fsdp is unavailable.")
            # Match DDP's rank-0 initialization, including when testing on CPU.
            for tensor in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(tensor.detach(), src=0)
            self.model = FSDP(model, use_orig_params=True, device_id=self.device)
        elif use_dist:
            self.model = DDP(
                model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=False,
            )
        else:
            self.model = model

    def autocast_context(self):
        dtype = _autocast_dtype(self.mixed_precision)
        if torch.cuda.is_available() and dtype is not None:
            return torch.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()

    def grad_sync_context(self, sync_gradients: bool = True):
        if sync_gradients or not hasattr(self.model, "no_sync"):
            return nullcontext()
        return self.model.no_sync()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def build_optimizer(self, optim_cfg: Dict) -> torch.optim.Optimizer:
        self.full_parameter_optimizer = self.is_fsdp and str(optim_cfg["name"]).lower() == "muon"
        context = (
            FSDP.summon_full_params(self.model, writeback=False)
            if self.full_parameter_optimizer
            else nullcontext()
        )
        with context:
            optimizer = build_optimizer(self.model.named_parameters(), optim_cfg)
        # Muon state describes full matrices; AdamW state follows local shards.
        optimizer._looplm_full_parameter_state = self.full_parameter_optimizer
        return optimizer

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.full_parameter_optimizer:
            # Newton-Schulz must operate on full 2D matrices, not 1D shards.
            # Every rank applies the same update and writes back its own shard.
            with FSDP.summon_full_params(self.model, with_grads=True):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                optimizer.step()
        else:
            if self.is_fsdp:
                self.model.clip_grad_norm_(self.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            optimizer.step()

    def zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)
