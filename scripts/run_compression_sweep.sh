#!/usr/bin/env bash
# Run the full compression sweep on M1 Max.
# Launches 4 experiments sequentially: baseline → int8 → lattice → lattice+EF

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CONFIGS=(
    "configs/m1_baseline.yaml"
    "configs/m1_int8.yaml"
    "configs/m1_lattice.yaml"
    "configs/m1_lattice_ef.yaml"
)

echo "=== DiLoCo Compression Sweep ==="
echo "Configs: ${CONFIGS[*]}"
echo ""

for cfg in "${CONFIGS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: $cfg"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ./scripts/run_m1_experiment.sh "$cfg"
    echo ""
    echo "✓ Completed: $cfg"
    echo ""
done

echo "=== All experiments completed ==="
