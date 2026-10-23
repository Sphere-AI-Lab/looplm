from typing import Dict, List

import torch


def causal_lm_collate(features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.tensor([x["input_ids"] for x in features], dtype=torch.long)
    labels = torch.tensor([x["labels"] for x in features], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
