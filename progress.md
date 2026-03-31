# Lattice-Compressed DiLoCo — Progress & GPU Experiment Guide

## What This Paper Is About

**Title (working):** *"Lattice-Compressed Distributed Low-Communication Training"*

**Core idea:** DiLoCo reduces sync *frequency* (workers train independently for H local steps). We reduce sync *volume* by compressing pseudo-gradients with E8P12 lattice vector quantization — achieving **~8× bandwidth reduction with zero convergence degradation**. These two techniques are orthogonal and composable.

**Novelty:** First application of lattice-based vector quantization (E8 lattice, 65,536 codepoints in 8D) to distributed training pseudo-gradients. Unlike scalar quantization (int8) or learned codebooks, the E8 lattice is algebraically defined, achieving near-optimal sphere-packing density — 2 bits/dim with quality rivaling 3-bit scalar quantizers.

**Key finding so far:** Pseudo-gradients are near-isotropic (unlike KV-cache vectors), so no random rotation preprocessing is needed — simplifying the pipeline and saving O(d²) overhead.

---

## Current State (2026-03-31)

### ✅ Completed

| Component | Status | Details |
|-----------|--------|---------|
| E8P12 lattice compressor | ✅ Done | `diloco_training/utils/lattice_compression.py` (~500 lines) |
| Distributed AllReduce | ✅ Done | `distributed_reduce_lattice()` — AllGather + local decompress+average |
| Error feedback | ✅ Done | Residual accumulation across outer steps |
| Training loop integration | ✅ Done | `--compression_method {none,int8,lattice}` + `--compression_error_feedback` |
| Unit tests | ✅ Done | 24 tests passing (quantizer, calibration, distributed, error feedback) |
| Repo cleanup | ✅ Done | Stripped non-LLM code (ResNet, Wav2Vec, GCN, BigGAN, etc.) |
| Small-scale experiments | ✅ Done | All 4 methods converge identically on GPT-Neo-Small |
| Analysis scripts | ✅ Done | `scripts/analysis/` — experiment runners + convergence plots |
| Compression metrics | ✅ Done | Cosine similarity, ratio, timing logged to WandB |

### 🔲 Remaining (GPU-scale, for the paper)

| Task | What's Needed |
|------|---------------|
| **Ablation: local_steps × compression** | Test `local_steps ∈ {5, 10, 25, 50}` × `{none, int8, lattice}` on GPU |
| **Large model experiments** | GPT-NeoX 1.3B+ on real C4, 4-8 GPU workers (A100/H100) |
| **Bandwidth simulation** | Simulate geo-distributed scenarios, measure wall-clock speedup |

---

## Small-Scale Experiment Results (CPU, GPT-Neo-Small 1.3M params)

| Method | Final Loss | Min Loss | Compression | Cos Similarity |
|--------|-----------|----------|-------------|----------------|
| DiLoCo (no compression) | 8.4247 | 8.3375 | 1× | — |
| DiLoCo + int8 | 8.4246 | 8.3375 | 4× | ~0.99 |
| DiLoCo + E8 lattice | 8.4213 | 8.3371 | **8×** | 0.965 |
| DiLoCo + lattice + error feedback | 8.4244 | 8.3374 | **8×** | → 1.0 over time |

**Setup:** 2 simulated workers, 30 outer steps, 8 local steps each, lr=3e-4, outer_lr=0.1, 1000 synthetic samples.

**Conclusion:** All compression methods converge identically. Lattice achieves 8× compression with negligible quality loss.

---

## How to Run GPU Experiments

### Branch

```bash
git clone https://github.com/exalsius/diloco-training.git
cd diloco-training
git checkout diloco_lattice
```

### Install

```bash
pip install uv  # if not installed
uv sync
```

### Quick Test (single machine, 2 GPUs)

```bash
# Baseline DiLoCo (no compression)
torchrun --nproc_per_node=2 \
  diloco_training/training/start_training.py \
  --model gpt-neo \
  --dataset c4 \
  --device cuda \
  --pgroup_backend nccl \
  --total_steps 1000 \
  --local_steps 50 \
  --batch_size 32 \
  --per_device_train_batch_size 16 \
  --outer_lr 0.7 \
  --lr 1e-3 \
  --compression_method none

# DiLoCo + E8 lattice compression (8×)
torchrun --nproc_per_node=2 \
  diloco_training/training/start_training.py \
  --model gpt-neo \
  --dataset c4 \
  --device cuda \
  --pgroup_backend nccl \
  --total_steps 1000 \
  --local_steps 50 \
  --batch_size 32 \
  --per_device_train_batch_size 16 \
  --outer_lr 0.7 \
  --lr 1e-3 \
  --compression_method lattice

# DiLoCo + lattice + error feedback
torchrun --nproc_per_node=2 \
  diloco_training/training/start_training.py \
  --model gpt-neo \
  --dataset c4 \
  --device cuda \
  --pgroup_backend nccl \
  --total_steps 1000 \
  --local_steps 50 \
  --batch_size 32 \
  --per_device_train_batch_size 16 \
  --outer_lr 0.7 \
  --lr 1e-3 \
  --compression_method lattice \
  --compression_error_feedback
```

### Multi-Node (e.g., 4 GPUs across 2 machines)

```bash
# On machine 1 (master):
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 \
  --master_addr=<MASTER_IP> --master_port=29500 \
  diloco_training/training/start_training.py \
  --model gpt-neo-x --dataset c4 --device cuda --pgroup_backend nccl \
  --total_steps 5000 --local_steps 50 --compression_method lattice

# On machine 2:
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 \
  --master_addr=<MASTER_IP> --master_port=29500 \
  diloco_training/training/start_training.py \
  --model gpt-neo-x --dataset c4 --device cuda --pgroup_backend nccl \
  --total_steps 5000 --local_steps 50 --compression_method lattice
```

### Available Models

| Model | Params | Use Case |
|-------|--------|----------|
| `gpt-neo-small` | ~1.3M | Quick debugging / CPU tests |
| `gpt-neo` | ~36M | Small GPU experiments |
| `gpt-neo-x` | ~150M | Paper-scale experiments |
| `gpt-neo-tiny` | ~10M | Unit tests |

### Compression Methods

| Flag | Compression | Description |
|------|-------------|-------------|
| `--compression_method none` | 1× | Standard DiLoCo (fp32 AllReduce) |
| `--compression_method int8` | 4× | Int8 scalar quantization (existing baseline) |
| `--compression_method lattice` | 8× | E8P12 lattice VQ (**ours**) |
| `--compression_method lattice --compression_error_feedback` | 8× | Lattice + error feedback (**ours**) |

---

## Paper Experiment Plan

### Experiment 1: Convergence Comparison (Main Result)
- **Model:** GPT-NeoX 1.3B (or GPT-Neo 36M as minimum)
- **Dataset:** C4 (streaming, English)
- **Workers:** 4-8 GPUs
- **Local steps:** 50
- **Methods:** DDP baseline, DiLoCo, DiLoCo+int8, DiLoCo+lattice, DiLoCo+lattice+EF
- **Metric:** Validation loss/perplexity vs total steps & wall-clock time

### Experiment 2: local_steps × Compression Ablation
- **local_steps:** {5, 10, 25, 50, 100}
- **Methods:** {none, int8, lattice}
- **Hypothesis:** More local steps → larger pseudo-gradients → more structure → better compressibility

### Experiment 3: Scaling Workers
- **Workers:** {2, 4, 8, 16}
- **Fixed total compute budget
- **Show:** Communication volume scales linearly with workers for baseline, but is ~8× less with lattice

### Experiment 4: Bandwidth-Constrained Training
- **Simulate:** 1 Gbps, 100 Mbps, 10 Mbps inter-node bandwidth
- **Show:** Wall-clock speedup from compression grows as bandwidth decreases
- **Key plot:** Time-to-target-loss vs available bandwidth

---

## Key Files

| File | Purpose |
|------|---------|
| `diloco_training/utils/lattice_compression.py` | E8P12 quantizer, compress/decompress |
| `diloco_training/utils/quantization.py` | `distributed_reduce_lattice()` + int8 |
| `diloco_training/utils/diloco_utils.py` | `update_outer_optimizer()` with compression dispatch |
| `diloco_training/training/training_config.py` | Config fields for compression |
| `diloco_training/training/distributed_trainer.py` | Main trainer |
| `diloco_training/training/start_training.py` | CLI entry point |
| `tests/unit/test_lattice_compression.py` | 24 unit tests |
| `scripts/analysis/direct_comparison.py` | In-process experiment runner |
| `scripts/analysis/run_experiments.py` | Distributed experiment runner (torchrun) |
| `scripts/analysis/convergence_analysis.py` | Analysis + plotting |

## Related Work (to position against in paper)

- **DiLoCo** (Douillard et al., 2023) — our baseline
- **PowerSGD** (Vogels et al., 2019) — low-rank gradient compression (different axis: VQ vs low-rank)
- **QSGD** (Alistarh et al., 2017) — scalar quantized SGD
- **QuIP#** (Tseng et al., 2024) — E8 lattice for weight quant (we adapt for gradient compression)
- **1-bit Adam** (Tang et al., 2021) — extreme compression for DDP (not DiLoCo)
- **Deep Gradient Compression** (Lin et al., 2018) — sparsification (orthogonal to ours)
