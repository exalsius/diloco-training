#!/usr/bin/env python3
"""
Direct comparison of DiLoCo compression methods.

Trains a small GPT-Neo model with 2 simulated DiLoCo workers and
captures loss at each outer step for: none, int8, lattice, lattice+EF.

This avoids WandB and subprocess overhead — runs everything in-process.
"""

import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diloco_training.models.gpt_neo import get_small_gpt_neo
from diloco_training.utils.lattice_compression import (
    compress_tensor,
    decompress_tensor,
)
from diloco_training.utils.quantization import quantize_tensor, dequantize_tensor


def make_synthetic_data(vocab_size, seq_len, n_samples, seed=42):
    """Generate synthetic token sequences for training."""
    rng = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (n_samples, seq_len), generator=rng)
    return tokens


def train_step(model, batch, device="cpu"):
    """Single forward + backward pass, returns loss."""
    input_ids = batch[:, :-1].to(device)
    labels = batch[:, 1:].to(device)
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    return loss.item()


def run_diloco_experiment(
    compression_method="none",
    error_feedback=False,
    n_workers=2,
    local_steps=10,
    outer_steps=10,
    lr=3e-4,
    outer_lr=0.1,
    seed=42,
    vocab_size=4096,
    seq_len=64,
    n_samples=1000,
    batch_size=8,
):
    """Run a complete DiLoCo experiment with the given compression method."""
    torch.manual_seed(seed)

    # Create model and data
    config, base_model = get_small_gpt_neo()
    data = make_synthetic_data(vocab_size, seq_len, n_samples, seed=seed)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    data_iter = iter(loader)

    # Create n_workers copies of the model (simulating distributed workers)
    workers = []
    for w in range(n_workers):
        worker_model = copy.deepcopy(base_model)
        worker_optim = torch.optim.AdamW(worker_model.parameters(), lr=lr)
        workers.append({"model": worker_model, "optim": worker_optim})

    # Global (outer) model parameters
    global_params = [p.clone().detach() for p in base_model.parameters()]
    outer_optim = torch.optim.SGD([torch.nn.Parameter(p) for p in global_params], lr=outer_lr)

    # Error feedback buffers (per worker, per param)
    ef_buffers = [[torch.zeros_like(p) for p in base_model.parameters()] for _ in range(n_workers)]

    results = {"losses": [], "outer_losses": [], "method": compression_method, "ef": error_feedback}

    for outer_step in range(outer_steps):
        worker_losses = []

        # Phase 1: Local training on each worker
        for w_idx, worker in enumerate(workers):
            model = worker["model"]
            optim = worker["optim"]
            model.train()

            step_losses = []
            for _ in range(local_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                optim.zero_grad()
                loss = train_step(model, batch)
                optim.step()
                step_losses.append(loss)

            worker_losses.append(sum(step_losses) / len(step_losses))

        avg_loss = sum(worker_losses) / len(worker_losses)
        results["losses"].append(avg_loss)

        # Phase 2: Compute pseudo-gradients and compress
        pseudo_grads_all = []  # [worker][param] = compressed or raw grad

        for w_idx, worker in enumerate(workers):
            model = worker["model"]
            pseudo_grads = []

            for p_idx, (local_p, global_p) in enumerate(
                zip(model.parameters(), global_params)
            ):
                pg = local_p.data - global_p  # pseudo-gradient

                if compression_method == "lattice":
                    if error_feedback:
                        pg = pg + ef_buffers[w_idx][p_idx]

                    if pg.numel() >= 8:
                        compressed = compress_tensor(pg)
                        reconstructed = decompress_tensor(compressed)
                        if error_feedback:
                            ef_buffers[w_idx][p_idx] = pg - reconstructed
                        pseudo_grads.append(reconstructed)
                    else:
                        if error_feedback:
                            ef_buffers[w_idx][p_idx].zero_()
                        pseudo_grads.append(pg.clone())

                elif compression_method == "int8":
                    if pg.numel() >= 4:
                        q, qmin, qmax = quantize_tensor(pg)
                        reconstructed = dequantize_tensor(q, qmin, qmax)
                        pseudo_grads.append(reconstructed)
                    else:
                        pseudo_grads.append(pg.clone())

                else:  # none
                    pseudo_grads.append(pg.clone())

            pseudo_grads_all.append(pseudo_grads)

        # Phase 3: Average pseudo-gradients across workers
        outer_optim.zero_grad()
        for p_idx, outer_p in enumerate(outer_optim.param_groups[0]["params"]):
            avg_grad = torch.stack([pseudo_grads_all[w][p_idx] for w in range(n_workers)]).mean(dim=0)
            outer_p.grad = avg_grad

        # Phase 4: Outer optimizer step
        outer_optim.step()

        # Update global params
        for p_idx, outer_p in enumerate(outer_optim.param_groups[0]["params"]):
            global_params[p_idx] = outer_p.data.clone().detach()

        # Phase 5: Broadcast updated params to all workers
        for worker in workers:
            for local_p, global_p in zip(worker["model"].parameters(), global_params):
                local_p.data.copy_(global_p)

        # Evaluate on a batch
        with torch.no_grad():
            try:
                eval_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                eval_batch = next(data_iter)
            input_ids = eval_batch[:, :-1]
            labels = eval_batch[:, 1:]
            outputs = workers[0]["model"](input_ids=input_ids, labels=labels)
            eval_loss = outputs.loss.item()

        results["outer_losses"].append(eval_loss)

        print(
            f"  Outer step {outer_step+1:3d}/{outer_steps} | "
            f"train_loss={avg_loss:.4f} | eval_loss={eval_loss:.4f}"
        )

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outer_steps", type=int, default=15)
    parser.add_argument("--local_steps", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/comparison.json")
    args = parser.parse_args()

    experiments = [
        {"name": "DiLoCo (no compression)", "compression_method": "none", "error_feedback": False},
        {"name": "DiLoCo + int8 (~4×)", "compression_method": "int8", "error_feedback": False},
        {"name": "DiLoCo + E8 lattice (~8×)", "compression_method": "lattice", "error_feedback": False},
        {"name": "DiLoCo + E8 lattice + EF (~8×)", "compression_method": "lattice", "error_feedback": True},
    ]

    all_results = {}
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"  {exp['name']}")
        print(f"{'='*60}")
        start = time.time()
        result = run_diloco_experiment(
            compression_method=exp["compression_method"],
            error_feedback=exp["error_feedback"],
            outer_steps=args.outer_steps,
            local_steps=args.local_steps,
        )
        elapsed = time.time() - start
        result["elapsed"] = elapsed
        all_results[exp["name"]] = result
        print(f"  Completed in {elapsed:.1f}s")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Method':<35} {'Final Loss':>12} {'Min Loss':>12} {'Time':>8}")
    print(f"{'-'*70}")
    for name, res in all_results.items():
        final = res["outer_losses"][-1]
        best = min(res["outer_losses"])
        print(f"{name:<35} {final:>12.4f} {best:>12.4f} {res['elapsed']:>7.1f}s")
    print(f"{'='*70}")

    # Generate plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        colors = {"none": "#2196F3", "int8": "#FF9800", "lattice": "#4CAF50"}
        styles = {False: "-", True: "--"}

        for name, res in all_results.items():
            method = res["method"]
            ef = res["ef"]
            color = colors.get(method, "#999")
            ls = styles.get(ef, "-")
            steps = list(range(1, len(res["outer_losses"]) + 1))

            ax1.plot(steps, res["outer_losses"], color=color, linestyle=ls, linewidth=2, label=name, marker="o", markersize=4)
            ax2.plot(steps, res["losses"], color=color, linestyle=ls, linewidth=2, label=name, marker="s", markersize=4)

        ax1.set_xlabel("Outer Step")
        ax1.set_ylabel("Eval Loss")
        ax1.set_title("Convergence: Eval Loss vs Outer Step")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Outer Step")
        ax2.set_ylabel("Train Loss (avg per outer step)")
        ax2.set_title("Training Loss vs Outer Step")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.suptitle("DiLoCo + Lattice Compression: GPT-Neo-Small (1.3M params, 2 workers)", fontsize=13)
        fig.tight_layout()

        fig_path = output_path.parent / "convergence_comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure saved to {fig_path}")

    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    main()
