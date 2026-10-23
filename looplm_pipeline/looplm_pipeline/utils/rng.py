import random
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = {
            "current_device": int(torch.cuda.current_device()),
            "current_device_state": torch.cuda.get_rng_state(),
            "all_device_states": torch.cuda.get_rng_state_all(),
        }
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        cuda_state = state["cuda"]
        if isinstance(cuda_state, dict):
            current_device_state = cuda_state.get("current_device_state")
            if current_device_state is not None:
                torch.cuda.set_rng_state(current_device_state)
                return
            cuda_state = cuda_state.get("all_device_states")
        if isinstance(cuda_state, (list, tuple)):
            if len(cuda_state) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_state)
            elif len(cuda_state) > 0:
                # Older checkpoints may store rank 0's full local GPU RNG list.
                # In 1-GPU-per-node resume, fall back to the first visible device.
                torch.cuda.set_rng_state(cuda_state[0])
