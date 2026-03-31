#!/usr/bin/env python3
"""
Convergence analysis for DiLoCo + Lattice Compression experiments.

Reads WandB offline runs or CSV logs, compares training curves across
compression methods (none, int8, lattice, lattice+EF).

Usage:
    python scripts/analysis/convergence_analysis.py --wandb_dir wandb/
    python scripts/analysis/convergence_analysis.py --csv results/

Outputs:
    - Loss vs step curves (PNG)
    - Perplexity vs step curves (PNG)
    - Bandwidth vs quality tradeoff (PNG)
    - Summary statistics table (stdout + CSV)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("pandas required: pip install pandas")
    sys.exit(1)

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.2)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def load_wandb_runs(wandb_dir: str) -> Dict[str, pd.DataFrame]:
    """Load metrics from WandB offline run directories."""
    runs = {}
    wandb_path = Path(wandb_dir)

    for run_dir in sorted(wandb_path.glob("offline-run-*")):
        # Read config to determine compression method
        config_file = run_dir / "files" / "config.yaml"
        wandb_meta = run_dir / "files" / "wandb-metadata.json"
        history_file = run_dir / "files" / "wandb-history.jsonl"

        if not history_file.exists():
            # Try alternative location
            history_files = list(run_dir.glob("**/wandb-history.jsonl"))
            if not history_files:
                continue
            history_file = history_files[0]

        # Parse run name from config or directory
        run_name = run_dir.name
        compression_method = "unknown"

        if config_file.exists():
            try:
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                cm = config.get("compression_method", {})
                if isinstance(cm, dict):
                    compression_method = cm.get("value", "none")
                else:
                    compression_method = str(cm)
                ef = config.get("compression_error_feedback", {})
                if isinstance(ef, dict):
                    ef = ef.get("value", False)
                if ef:
                    compression_method += "+ef"
            except Exception:
                pass

        # Parse history
        records = []
        with open(history_file) as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue

        if records:
            df = pd.DataFrame(records)
            run_key = f"{compression_method}_{run_dir.name[-8:]}"
            runs[run_key] = df

    return runs


def load_csv_results(csv_dir: str) -> Dict[str, pd.DataFrame]:
    """Load metrics from CSV files."""
    runs = {}
    csv_path = Path(csv_dir)

    for csv_file in sorted(csv_path.glob("*.csv")):
        df = pd.read_csv(csv_file)
        run_name = csv_file.stem
        runs[run_name] = df

    return runs


def plot_loss_curves(
    runs: Dict[str, pd.DataFrame],
    output_dir: str,
    loss_col: str = "training/outer_loss",
    step_col: str = "real_step",
):
    """Plot training loss vs step for all runs."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {
        "none": "#2196F3",
        "int8": "#FF9800",
        "lattice": "#4CAF50",
        "lattice+ef": "#E91E63",
    }

    for run_name, df in runs.items():
        if loss_col not in df.columns or step_col not in df.columns:
            # Try alternative column names
            if "loss" in df.columns:
                loss_col_actual = "loss"
            else:
                continue
            step_col_actual = step_col if step_col in df.columns else "_step"
        else:
            loss_col_actual = loss_col
            step_col_actual = step_col

        mask = df[loss_col_actual].notna()
        steps = df.loc[mask, step_col_actual]
        losses = df.loc[mask, loss_col_actual]

        # Determine color from run name
        color = "#999999"
        label = run_name
        for method, c in colors.items():
            if method in run_name.lower():
                color = c
                label = method.upper()
                break

        ax.plot(steps, losses, label=label, color=color, alpha=0.8, linewidth=2)

    ax.set_xlabel("Outer Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("DiLoCo Training Loss: Compression Method Comparison")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path / 'loss_curves.png'}")


def plot_perplexity_curves(
    runs: Dict[str, pd.DataFrame],
    output_dir: str,
    ppl_col: str = "training/outer_perplexity",
    step_col: str = "real_step",
):
    """Plot perplexity vs step for all runs."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {
        "none": "#2196F3",
        "int8": "#FF9800",
        "lattice": "#4CAF50",
        "lattice+ef": "#E91E63",
    }

    for run_name, df in runs.items():
        if ppl_col not in df.columns:
            continue

        mask = df[ppl_col].notna()
        steps = df.loc[mask, step_col] if step_col in df.columns else df.loc[mask].index
        ppls = df.loc[mask, ppl_col]

        color = "#999999"
        label = run_name
        for method, c in colors.items():
            if method in run_name.lower():
                color = c
                label = method.upper()
                break

        ax.plot(steps, ppls, label=label, color=color, alpha=0.8, linewidth=2)

    ax.set_xlabel("Outer Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("DiLoCo Perplexity: Compression Method Comparison")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir)
    fig.savefig(output_path / "perplexity_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path / 'perplexity_curves.png'}")


def plot_bandwidth_tradeoff(output_dir: str):
    """Plot compression ratio vs convergence quality tradeoff."""
    methods = ["None (fp32)", "Int8", "E8 Lattice", "E8 Lattice + EF"]
    compression_ratios = [1.0, 4.0, 8.0, 8.0]
    # These are theoretical/measured cosine similarities from our unit tests
    cos_sims = [1.0, 0.99, 0.965, 0.98]
    bits_per_param = [32, 8, 4, 4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    # Left: Compression ratio bar chart
    bars = ax1.bar(methods, compression_ratios, color=colors, alpha=0.8, edgecolor="black")
    ax1.set_ylabel("Compression Ratio (×)")
    ax1.set_title("Bandwidth Reduction")
    for bar, ratio in zip(bars, compression_ratios):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{ratio:.0f}×",
            ha="center",
            fontweight="bold",
        )
    ax1.set_ylim(0, 10)

    # Right: Bits per parameter vs cosine similarity
    ax2.scatter(bits_per_param, cos_sims, c=colors, s=200, zorder=5, edgecolors="black")
    for i, method in enumerate(methods):
        ax2.annotate(
            method,
            (bits_per_param[i], cos_sims[i]),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=9,
        )
    ax2.set_xlabel("Bits per Parameter")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Compression Quality vs Bandwidth")
    ax2.set_xlim(0, 35)
    ax2.set_ylim(0.9, 1.01)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("DiLoCo Pseudo-Gradient Compression: Bandwidth–Quality Tradeoff", fontsize=14)
    fig.tight_layout()

    output_path = Path(output_dir)
    fig.savefig(output_path / "bandwidth_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path / 'bandwidth_tradeoff.png'}")


def print_summary_table(runs: Dict[str, pd.DataFrame]):
    """Print summary statistics for each run."""
    print("\n" + "=" * 80)
    print("CONVERGENCE SUMMARY")
    print("=" * 80)

    headers = ["Method", "Final Loss", "Min Loss", "Final PPL", "Steps", "Bytes Sent (MB)"]
    print(f"{'Method':<20} {'Final Loss':>12} {'Min Loss':>12} {'Final PPL':>12} {'Steps':>8} {'MB Sent':>12}")
    print("-" * 80)

    for run_name, df in runs.items():
        loss_col = "training/outer_loss"
        ppl_col = "training/outer_perplexity"
        bytes_col = "comm/total_bytes_sent_mb"

        if loss_col in df.columns:
            losses = df[loss_col].dropna()
            final_loss = losses.iloc[-1] if len(losses) > 0 else float("nan")
            min_loss = losses.min() if len(losses) > 0 else float("nan")
        else:
            final_loss = min_loss = float("nan")

        if ppl_col in df.columns:
            ppls = df[ppl_col].dropna()
            final_ppl = ppls.iloc[-1] if len(ppls) > 0 else float("nan")
        else:
            final_ppl = float("nan")

        steps = len(df)

        if bytes_col in df.columns:
            total_bytes = df[bytes_col].dropna().sum()
        else:
            total_bytes = float("nan")

        print(
            f"{run_name:<20} {final_loss:>12.4f} {min_loss:>12.4f} "
            f"{final_ppl:>12.2f} {steps:>8d} {total_bytes:>12.2f}"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="DiLoCo Compression Convergence Analysis")
    parser.add_argument("--wandb_dir", type=str, default="wandb/", help="WandB offline runs directory")
    parser.add_argument("--csv", type=str, default=None, help="CSV results directory (alternative to wandb)")
    parser.add_argument("--output_dir", type=str, default="results/figures/", help="Output directory for figures")
    args = parser.parse_args()

    if args.csv:
        runs = load_csv_results(args.csv)
    else:
        runs = load_wandb_runs(args.wandb_dir)

    if not runs:
        print(f"No runs found in {args.csv or args.wandb_dir}")
        print("Run experiments first, then analyze results.")
        # Still generate the theoretical bandwidth tradeoff plot
        plot_bandwidth_tradeoff(args.output_dir)
        return

    print(f"Found {len(runs)} runs")
    for name, df in runs.items():
        print(f"  {name}: {len(df)} records, columns: {list(df.columns)[:5]}...")

    plot_loss_curves(runs, args.output_dir)
    plot_perplexity_curves(runs, args.output_dir)
    plot_bandwidth_tradeoff(args.output_dir)
    print_summary_table(runs)


if __name__ == "__main__":
    main()
