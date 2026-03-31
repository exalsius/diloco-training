#!/usr/bin/env python3
"""
Run DiLoCo compression experiments: baseline, int8, lattice, lattice+EF.

Runs training with gpt-neo (small, ~10M params) on C4 dataset using 2
gloo workers on CPU. Captures loss curves per outer step and writes
results to CSV for analysis.

Usage:
    uv run python scripts/analysis/run_experiments.py
    uv run python scripts/analysis/run_experiments.py --total_steps 500 --local_steps 25
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "diloco_baseline",
        "compression_method": "none",
        "compression_error_feedback": False,
        "description": "DiLoCo baseline (no compression)",
    },
    {
        "name": "diloco_int8",
        "compression_method": "int8",
        "compression_error_feedback": False,
        "description": "DiLoCo + int8 quantization (~4x)",
    },
    {
        "name": "diloco_lattice",
        "compression_method": "lattice",
        "compression_error_feedback": False,
        "description": "DiLoCo + E8 lattice compression (~8x)",
    },
    {
        "name": "diloco_lattice_ef",
        "compression_method": "lattice",
        "compression_error_feedback": True,
        "description": "DiLoCo + E8 lattice + error feedback (~8x)",
    },
]


def run_experiment(
    exp: dict,
    model: str,
    dataset: str,
    total_steps: int,
    local_steps: int,
    batch_size: int,
    per_device_batch_size: int,
    lr: float,
    outer_lr: float,
    warmup_steps: int,
    seed: int,
    nproc: int,
    port: int,
    output_dir: Path,
) -> dict:
    """Run a single experiment and return results."""
    name = exp["name"]
    print(f"\n{'='*60}")
    print(f"  {exp['description']}")
    print(f"  Method: {exp['compression_method']}, EF: {exp['compression_error_feedback']}")
    print(f"{'='*60}")

    checkpoint_path = output_dir / f"checkpoints/{name}.pth"
    # Remove old checkpoints for this experiment (they include rank/optim suffixes)
    import glob as globmod
    for old_ckpt in globmod.glob(str(output_dir / f"checkpoints/{name}*")):
        os.remove(old_ckpt)
    # Also remove default checkpoint.pth files that may interfere
    for old_default in globmod.glob("checkpoint*"):
        os.remove(old_default)

    cmd = [
        "uv", "run", "torchrun",
        f"--nnodes=1",
        f"--nproc_per_node={nproc}",
        f"--master_addr=127.0.0.1",
        f"--master_port={port}",
        "diloco_training/training/start_training.py",
        "--model", model,
        "--dataset", dataset,
        "--local_steps", str(local_steps),
        "--total_steps", str(total_steps),
        "--per_device_train_batch_size", str(per_device_batch_size),
        "--batch_size", str(batch_size),
        "--device", "cpu",
        "--pgroup_backend", "gloo",
        "--lr", str(lr),
        "--outer_lr", str(outer_lr),
        "--warmup_steps", str(warmup_steps),
        "--compression_method", exp["compression_method"],
        "--checkpoint_path", str(checkpoint_path),
        "--checkpoint_interval", str(total_steps + 1),  # don't checkpoint mid-run
        "--seed", str(seed),
        "--wandb_project_name", "diloco-lattice-compression",
        "--wandb_group", name,
    ]

    if exp["compression_error_feedback"]:
        cmd.append("--compression_error_feedback")

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
            env=env,
        )
        elapsed = time.time() - start_time

        # Parse loss from output
        losses = []
        for line in result.stdout.split("\n") + result.stderr.split("\n"):
            # Look for outer optimizer sync lines or loss logs
            if "outer_loss" in line.lower() or "Outer optimizer sync" in line:
                pass  # metrics logged to wandb
            if "Training step" in line and "loss:" in line:
                try:
                    loss_str = line.split("loss:")[1].strip().split()[0].rstrip(",")
                    losses.append(float(loss_str))
                except (IndexError, ValueError):
                    pass

        return {
            "name": name,
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "elapsed_seconds": elapsed,
            "stdout_lines": len(result.stdout.split("\n")),
            "stderr_lines": len(result.stderr.split("\n")),
            "errors": result.stderr[-500:] if result.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "status": "timeout",
            "returncode": -1,
            "elapsed_seconds": 3600,
        }
    except Exception as e:
        return {
            "name": name,
            "status": "error",
            "returncode": -1,
            "elapsed_seconds": time.time() - start_time,
            "errors": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run DiLoCo compression experiments")
    parser.add_argument("--model", default="gpt-neo", help="Model name")
    parser.add_argument("--dataset", default="c4", help="Dataset name")
    parser.add_argument("--total_steps", type=int, default=200, help="Total training steps")
    parser.add_argument("--local_steps", type=int, default=25, help="Local steps between syncs")
    parser.add_argument("--batch_size", type=int, default=16, help="Total batch size")
    parser.add_argument("--per_device_batch_size", type=int, default=8, help="Per device batch size")
    parser.add_argument("--lr", type=float, default=4e-4, help="Inner learning rate")
    parser.add_argument("--outer_lr", type=float, default=0.7, help="Outer learning rate")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nproc", type=int, default=2, help="Number of workers")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--experiments", nargs="*", default=None,
                       help="Run specific experiments by name (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    exps = EXPERIMENTS
    if args.experiments:
        exps = [e for e in EXPERIMENTS if e["name"] in args.experiments]

    print("=" * 60)
    print("DiLoCo Lattice Compression Experiments")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Total steps: {args.total_steps}")
    print(f"Local steps: {args.local_steps}")
    print(f"Workers: {args.nproc}")
    print(f"Experiments: {[e['name'] for e in exps]}")
    print()

    results = []
    base_port = 29510

    for i, exp in enumerate(exps):
        result = run_experiment(
            exp,
            model=args.model,
            dataset=args.dataset,
            total_steps=args.total_steps,
            local_steps=args.local_steps,
            batch_size=args.batch_size,
            per_device_batch_size=args.per_device_batch_size,
            lr=args.lr,
            outer_lr=args.outer_lr,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            nproc=args.nproc,
            port=base_port + i,
            output_dir=output_dir,
        )
        results.append(result)

        status_icon = "✓" if result["status"] == "success" else "✗"
        print(f"\n{status_icon} {result['name']}: {result['status']} ({result['elapsed_seconds']:.1f}s)")
        if result.get("errors"):
            print(f"  Error: {result['errors'][:200]}")

    # Write summary
    summary_file = output_dir / "experiment_summary.csv"
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "status", "elapsed_seconds", "returncode"])
        writer.writeheader()
        writer.writerows([{k: r.get(k) for k in ["name", "status", "elapsed_seconds", "returncode"]} for r in results])

    print(f"\n{'='*60}")
    print(f"Summary saved to: {summary_file}")
    print(f"WandB logs in: wandb/")
    print(f"\nTo analyze results:")
    print(f"  uv run python scripts/analysis/convergence_analysis.py")


if __name__ == "__main__":
    main()
