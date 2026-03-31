#!/usr/bin/env bash
# Launch DiLoCo training on M1 Max with 2 gloo workers.
# Usage: ./scripts/run_m1_experiment.sh configs/m1_baseline.yaml
#
# Runs torchrun with 2 processes on a single node using gloo backend (CPU).
# Suitable for M1 Max with 32GB RAM using gpt-neo (~10M params).

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml>}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

echo "=== DiLoCo M1 Experiment ==="
echo "Config: $CONFIG"
echo "Workers: 2 (gloo, CPU)"
echo ""

# Disable MPS (Metal) — we want pure CPU for reproducibility
export PYTORCH_ENABLE_MPS_FALLBACK=0

# WandB — set to offline if no API key
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Note: WANDB_API_KEY not set — running in offline mode"
    export WANDB_MODE=offline
fi

mkdir -p checkpoints

uv run torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    diloco_training/training/start_training.py \
    --config "$CONFIG"
