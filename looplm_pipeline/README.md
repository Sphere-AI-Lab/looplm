# looplm_pipeline

Minimal pretraining pipeline for Llama-style BaseLoop and LoopFormer experiments.

## Scope

- Models: `baseloop_llama_like`, `loopformer_llama_like`
- Sizes: `llama3_1_480m`, `llama3_1_760m`
- Optimizers: `AdamW`, default `Muon`
- Training: packed JSONL/Parquet data, DDP/FSDP wrapper, WSD schedule, checkpoint/resume, validation, JSONL metrics

## Install

Use Python 3.10 or newer. Run from the `looplm_pipeline/` directory (all commands below assume this working directory):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The examples below assume eight CUDA GPUs per node and a compatible PyTorch installation.

## Configure Data

Edit:

```text
configs/data/sources.yaml
configs/data/tokenize_llama3_1.yaml
```

Training rows should have a `text` column by default. JSONL example:

```json
{"id": "sample-1", "text": "training text"}
```

Set real train/validation paths and replace the tokenizer placeholder with a local Llama 3.1 tokenizer or a Hugging Face tokenizer ID. Tokens are packed into fixed-length sequences; incomplete sequences and incomplete training microbatches are dropped. Each rank/worker partition needs enough data to produce a full microbatch. Validation keeps incomplete microbatches.

## Run

Single node:

```bash
bash scripts/torchrun_train.sh configs/experiments/llama3_1_760m_loopformer_muon.yaml 8
```

Resume:

```bash
bash scripts/torchrun_train.sh configs/experiments/llama3_1_760m_loopformer_muon.yaml 8 --resume
```

Multi-node uses the standard environment variables:

```bash
export NNODES=2
export NODE_RANK=0  # Use 1 on the second node.
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
bash scripts/torchrun_train.sh configs/experiments/llama3_1_760m_baseloop_muon.yaml 8 --grad-accum-steps 8
```

`global_batch_size` must equal `world_size * micro_batch_size * grad_accum_steps`.

All supplied configs target eight total processes. For other world sizes, set `--grad-accum-steps` so this equality still holds. The two-node BaseLoop example above uses 16 total processes and halves the default accumulation. Use the same override when resuming, and make the data, tokenizer and output directory accessible on every node.

## FSDP and Checkpoints

The default is DDP. To enable FSDP, set `enabled: true` in `configs/train/fsdp.yaml` and launch distributed training. AdamW uses sharded optimizer state. Muon gathers full parameter matrices and gradients for each update and keeps full optimizer state on every rank, so its optimizer memory is not sharded.

Checkpoints contain full model weights and the appropriate optimizer state, plus per-rank RNG/data-position metadata. Resume requires the same training/data configuration and world size. After a non-finite loss or gradient, all ranks save a diagnostic snapshot under `checkpoints/emergency/`; `--resume` selects a regular checkpoint from a completed optimizer step. Checkpoint writes are atomic.

## Tests

```bash
python -m unittest discover -s tests -v
```

Two-process CPU regressions additionally exercise DDP/FSDP resume and failures confined to one rank. They require local socket access and a PyTorch build supporting CPU FSDP:

```bash
LOOPLM_RUN_DISTRIBUTED_TESTS=1 python -m unittest discover -s tests -v
```

## Experiment Configs

```text
configs/experiments/llama3_1_480m_baseloop_adamw.yaml
configs/experiments/llama3_1_480m_baseloop_muon.yaml
configs/experiments/llama3_1_480m_loopformer_adamw.yaml
configs/experiments/llama3_1_480m_loopformer_muon.yaml
configs/experiments/llama3_1_760m_baseloop_adamw.yaml
configs/experiments/llama3_1_760m_baseloop_muon.yaml
configs/experiments/llama3_1_760m_loopformer_adamw.yaml
configs/experiments/llama3_1_760m_loopformer_muon.yaml
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
