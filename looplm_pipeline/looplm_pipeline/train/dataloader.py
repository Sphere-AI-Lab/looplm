import inspect
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from looplm_pipeline.data.collate import causal_lm_collate
from looplm_pipeline.data.readers import iter_rows
from looplm_pipeline.data.tokenize import PackedTokenizer


def _iter_shuffled_rows(source: Iterator[Dict], buffer_size: int, seed: int) -> Iterator[Dict]:
    """Streaming shuffle via a fixed-size buffer (seeded RNG)."""
    rng = random.Random(seed)
    buf: List[Dict] = []
    for row in source:
        buf.append(row)
        if len(buf) >= buffer_size:
            j = rng.randrange(len(buf))
            yield buf.pop(j)
    rng.shuffle(buf)
    for row in buf:
        yield row


class PackedDataset(IterableDataset):
    """Iterable dataset that reads JSONL or Parquet, packs tokens to fixed seq_len."""

    def __init__(
        self,
        data_path: str,
        text_key: str,
        id_key: str,
        tokenizer_name_or_path: str,
        seq_len: int,
        rank: int,
        world_size: int,
        add_eos: bool = True,
        *,
        shuffle_buffer_size: int = 0,
        shuffle_seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.text_key = text_key
        self.id_key = id_key
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.add_eos = add_eos
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_seed = shuffle_seed

    def _shuffle_stream_seed(self, worker_id: int) -> int:
        return int(self.shuffle_seed) + self.rank * 1_000_003 + int(worker_id)

    def _iter_rows_for_partition(self) -> Iterator[Dict]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        base = iter_rows(
            self.data_path,
            text_key=self.text_key,
            id_key=self.id_key,
            rank=self.rank,
            world_size=self.world_size,
            worker_id=worker_id,
            num_workers=num_workers,
        )
        if self.shuffle_buffer_size > 0:
            yield from _iter_shuffled_rows(
                base,
                self.shuffle_buffer_size,
                self._shuffle_stream_seed(worker_id),
            )
        else:
            yield from base

    def __iter__(self) -> Iterator[Dict]:
        tok = PackedTokenizer(
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            seq_len=self.seq_len,
            add_eos=self.add_eos,
        )
        token_buf = []
        for row in self._iter_rows_for_partition():
            text = str(row.get(self.text_key, ""))
            ids = tok.encode_text(text)
            token_buf.extend(ids)
            while len(token_buf) >= self.seq_len:
                chunk = token_buf[: self.seq_len]
                token_buf = token_buf[self.seq_len :]
                yield {"input_ids": chunk, "labels": chunk.copy()}


def build_dataloader(
    data_cfg: Dict[str, Any],
    tokenize_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train and optional validation dataloaders.

    train_path / validation_path are directories (or single files); all .jsonl and .parquet under them are read.
    If validation_path is set and train_cfg["validation_interval"] > 0, returns (train_dl, val_dl);
    otherwise returns (train_dl, None).

    Training rows are shuffled with a streaming buffer using shuffle_seed / shuffle_buffer_size from data_cfg.
    """
    text_key = data_cfg["text_column"]
    id_key = data_cfg.get("id_column", "sample_id")
    train_path = data_cfg["train_path"]
    validation_path = data_cfg.get("validation_path") or ""
    validation_interval = int(train_cfg.get("validation_interval", 0))
    num_workers = int(train_cfg.get("num_workers", 0))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    persistent_workers = bool(train_cfg.get("persistent_workers", num_workers > 0))

    shuffle_seed = int(data_cfg.get("shuffle_seed", 42))
    shuffle_buffer_size = int(data_cfg.get("shuffle_buffer_size", 10_000))
    dataloader_seed = int(data_cfg.get("dataloader_seed", train_cfg.get("dataloader_seed", shuffle_seed)))

    train_shuffle_buffer = shuffle_buffer_size

    common_dataset_kw = dict(
        text_key=text_key,
        id_key=id_key,
        tokenizer_name_or_path=tokenize_cfg["tokenizer_name_or_path"],
        seq_len=int(tokenize_cfg["seq_len"]),
        add_eos=bool(tokenize_cfg.get("add_eos", True)),
    )

    train_dataset = PackedDataset(
        data_path=train_path,
        rank=rank,
        world_size=world_size,
        shuffle_buffer_size=train_shuffle_buffer,
        shuffle_seed=shuffle_seed,
        **common_dataset_kw,
    )
    train_loader_kwargs = {}
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = persistent_workers
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
    if "in_order" in inspect.signature(DataLoader).parameters:
        train_loader_kwargs["in_order"] = True

    train_generator = torch.Generator()
    train_generator.manual_seed(dataloader_seed + rank * 1_000_003)

    train_dl = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["micro_batch_size"]),
        drop_last=True,
        num_workers=num_workers,
        collate_fn=causal_lm_collate,
        pin_memory=torch.cuda.is_available(),
        generator=train_generator,
        **train_loader_kwargs,
    )

    val_dl: Optional[DataLoader] = None
    if validation_path and validation_interval > 0:
        val_dataset = PackedDataset(
            data_path=validation_path,
            rank=rank,
            world_size=world_size,
            shuffle_buffer_size=0,
            shuffle_seed=shuffle_seed,
            **common_dataset_kw,
        )
        requested_validation_num_workers = int(train_cfg.get("validation_num_workers", 0))
        if requested_validation_num_workers > 0 and rank == 0:
            print(
                "validation_num_workers is currently ignored; validation uses num_workers=0 "
                "so worker subprocess failures do not abort the validation step.",
                flush=True,
            )
        val_loader_kwargs = {}
        if "in_order" in inspect.signature(DataLoader).parameters:
            val_loader_kwargs["in_order"] = True
        val_dl = DataLoader(
            val_dataset,
            batch_size=int(train_cfg["micro_batch_size"]),
            num_workers=0,
            collate_fn=causal_lm_collate,
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(dataloader_seed),
            **val_loader_kwargs,
        )

    return train_dl, val_dl
