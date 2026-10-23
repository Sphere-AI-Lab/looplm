#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: scripts/torchrun_train.sh CONFIG NPROC_PER_NODE [extra train args]}
NPROC_PER_NODE=${2:-8}
shift 2

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  -m looplm_pipeline.cli.train \
  --config "${CONFIG}" \
  "$@"
