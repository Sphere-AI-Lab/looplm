from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from looplm_pipeline.models.builders.modeling_outputs import CausalLMOutput
from looplm_pipeline.models.registry import build_model_from_config
from looplm_pipeline.optim.muon import Muon
from looplm_pipeline.train.checkpoint import _full_state_context, load_latest_checkpoint, save_checkpoint
from looplm_pipeline.train.dataloader import build_dataloader
from looplm_pipeline.train.fsdp_engine import TrainEngine
from looplm_pipeline.train.loop import run_pretrain_loop
from looplm_pipeline.utils.logging import JsonlLogger
from looplm_pipeline.utils.rng import seed_everything


def _model(family):
    cfg = dict(
        name="tiny", family=family + "_llama_like", vocab_size=16,
        hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        rms_norm_eps=1e-5, rope_theta=10000.0, torch_dtype="float32",
    )
    cfg[family] = dict(shared_depth=2, max_model_loops=3, time_embedding_size=16)
    return build_model_from_config(cfg)


def _batches():
    generator = torch.Generator().manual_seed(123)
    batches = []
    for _ in range(5):
        ids = torch.randint(0, 16, (2, 6), generator=generator)
        batches.append(dict(input_ids=ids, labels=ids.clone(), attention_mask=torch.ones_like(ids)))
    return batches


def _train_cfg(family, *, max_steps=4, accum=2):
    loss = {"type": "causal_lm"}
    if family == "loopformer":
        loss = dict(type="loopformer_self_distill", max_model_loops=3,
                    short_ce_weight=0.1, consistency_weight=0.1)
    return dict(max_steps=max_steps, grad_accum_steps=accum, log_interval=max_steps,
                save_checkpoint_interval=2 if max_steps == 4 else 1, loss=loss)


def _run(engine, optimizer, output, cfg, *, start_step=0, resume_state=None):
    run_pretrain_loop(
        engine, optimizer, _batches(), JsonlLogger(str(output / "metrics.jsonl")),
        cfg, dict(name="wsd", total_steps=4, warmup_steps=1, stable_steps=2, decay_steps=1),
        str(output), start_step=start_step, resume_state=resume_state,
    )


def _assert_equal(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f"{path}: tensor differs after resume (shapes {a.shape}, {b.shape})"
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_equal(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_equal(x, y)
    else:
        assert a == b, (a, b)


def _check_resume(output, family, optimizer_name, *, fsdp=False, distributed=False):
    seed_everything(42 + (dist.get_rank() if distributed else 0))
    engine = TrainEngine(_model(family), {"enabled": fsdp}, distributed, 0)
    optim_cfg = dict(name=optimizer_name, lr=0.001, weight_decay=0.1, use_fused=False)
    optimizer = engine.build_optimizer(optim_cfg)
    if optimizer_name == "muon":
        assert isinstance(optimizer, Muon), "Muon silently fell back to AdamW"
        assert any(state["use_muon"] for state in optimizer.state.values())
    cfg = _train_cfg(family)
    _run(engine, optimizer, output, cfg)
    checkpoint = output / "checkpoints" / "step_00000004.pt"
    baseline = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not distributed or dist.get_rank() == 0:
        checkpoint.unlink()
    if distributed:
        dist.barrier()
    # Restore into a fresh model/optimizer, with a distinct initial RNG state.
    seed_everything(999)
    restored = TrainEngine(_model(family), {"enabled": fsdp}, distributed, 0)
    restored_optimizer = restored.build_optimizer(optim_cfg)
    step, extra = load_latest_checkpoint(str(output / "checkpoints"), restored.model, restored_optimizer)
    assert step == 2
    expected = torch.load(output / "checkpoints" / "step_00000002.pt", map_location="cpu", weights_only=False)
    if fsdp:
        with _full_state_context(restored.model, rank0_only=False):
            _assert_equal(expected["model"], restored.model.state_dict())
            if optimizer_name == "adamw":
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                _assert_equal(expected["optimizer"], FSDP.optim_state_dict(restored.model, restored_optimizer))
    state = extra["rank_local_state"]["train_state"]
    _run(restored, restored_optimizer, output, cfg, start_step=step, resume_state=state)
    resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _assert_equal(baseline["model"], resumed["model"])
    _assert_equal(baseline["optimizer"], resumed["optimizer"])
    assert resumed["extra_state"]["train_state"]["tokens_seen_total"] == baseline["extra_state"]["train_state"]["tokens_seen_total"]
    return resumed


class _FaultyModel(torch.nn.Module):
    def __init__(self, fault, inject=True):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16)
        self.fault = fault
        self.inject = inject
        self.calls = 0

    def forward(self, input_ids, attention_mask=None):
        self.calls += 1
        logits = self.head(self.embed(input_ids))
        if self.inject and self.calls == 2:
            if self.fault == "loss":
                logits = logits * float("nan")
            else:
                self.head.weight.register_hook(lambda grad: grad * float("nan"))
        return CausalLMOutput(logits=logits)


def _check_emergency(output, fault, *, distributed=False, fsdp=False):
    # Only rank 0 fails; every rank must still save and exit together.
    inject = not distributed or dist.get_rank() == 0
    engine = TrainEngine(_FaultyModel(fault, inject), {"enabled": fsdp}, distributed, 0)
    optimizer = engine.build_optimizer(dict(name="adamw", lr=0.001, use_fused=False))
    try:
        _run(engine, optimizer, output, _train_cfg("baseloop", max_steps=2, accum=1))
    except RuntimeError as exc:
        assert "non_finite_" in str(exc), str(exc)
    else:
        raise AssertionError("non-finite training did not stop")
    directory = output / "checkpoints"
    healthy = torch.load(directory / "step_00000001.pt", weights_only=False, map_location="cpu")
    diagnostic = torch.load(directory / "emergency" / "step_00000001.pt", weights_only=False, map_location="cpu")
    assert not (directory / "step_00000002.pt").exists()
    assert diagnostic["step"] == 1
    assert len(diagnostic["rank_states"]) == (2 if distributed else 1)
    _assert_equal(healthy["model"], diagnostic["model"])
    # --resume must choose the healthy checkpoint, not the diagnostic snapshot.
    step, extra = load_latest_checkpoint(str(directory), engine.model, optimizer)
    assert step == 1 and "emergency_reason" not in extra


class TestTrainingRuntime(unittest.TestCase):
    def test_training_drops_partial_microbatches_validation_keeps_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "word": 2}, unk_token="<unk>"))
            tokenizer.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>",
                                   eos_token="<eos>", pad_token="<eos>").save_pretrained(root / "tokenizer")
            path = root / "data.jsonl"
            path.write_text(json.dumps({"id": "one", "text": " ".join(["word"] * 23)}) + "\n")
            train, validation = build_dataloader(
                dict(train_path=str(path), validation_path=str(path), text_column="text", id_column="id"),
                dict(tokenizer_name_or_path=str(root / "tokenizer"), seq_len=8, add_eos=True),
                dict(micro_batch_size=2, num_workers=0, validation_interval=1), 0, 1,
            )
            self.assertEqual([len(batch["input_ids"]) for batch in train], [2])
            self.assertEqual([len(batch["input_ids"]) for batch in validation], [2, 1])

    def test_resume_restores_weights_optimizer_and_data_position(self):
        with tempfile.TemporaryDirectory() as directory:
            for family in ["baseloop", "loopformer"]:
                for optimizer in ["adamw", "muon"]:
                    with self.subTest(family=family, optimizer=optimizer):
                        _check_resume(Path(directory) / f"{family}_{optimizer}", family, optimizer)

    def test_nonfinite_loss_and_gradients_preserve_healthy_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            for fault in ["loss", "gradients"]:
                with self.subTest(fault=fault):
                    _check_emergency(Path(directory) / fault, fault)

    def test_failed_save_keeps_existing_checkpoint_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            path = Path(save_checkpoint(directory, model, optimizer, 1))
            before = path.read_bytes()
            with mock.patch("looplm_pipeline.train.checkpoint.torch.save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    save_checkpoint(directory, model, optimizer, 1)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    @unittest.skipUnless(os.environ.get("LOOPLM_RUN_DISTRIBUTED_TESTS") == "1", "opt-in two-process CPU checks")
    def test_distributed_training_resume_and_single_rank_failure(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            port = server.getsockname()[1]
            server.close()
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            result = subprocess.run(
                [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2",
                 "--master_addr", "127.0.0.1", "--master_port", str(port),
                 str(Path(__file__).resolve()), "--distributed-worker", directory],
                env=env, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _distributed_worker(directory):
    dist.init_process_group("gloo", timeout=timedelta(seconds=20))
    try:
        root = Path(directory)
        for fsdp in [False, True]:
            for optimizer in ["adamw", "muon"]:
                output = root / f"fsdp_{fsdp}_{optimizer}"
                if dist.get_rank() == 0:
                    print(f"Checking resume: fsdp={fsdp}, optimizer={optimizer}", flush=True)
                _check_resume(output, "loopformer", optimizer, fsdp=fsdp, distributed=True)
            for fault in ["loss", "gradients"]:
                _check_emergency(root / f"fsdp_{fsdp}_{fault}", fault, distributed=True, fsdp=fsdp)
        if dist.get_rank() == 0:
            print("DDP/FSDP resume and single-rank non-finite regression checks passed.", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--distributed-worker":
        _distributed_worker(sys.argv[2])
    else:
        unittest.main()
