#!/bin/bash
set -euo pipefail

# ============================================================================
# IV Regression Simulation Batch Runner
#
# Runs a set of experiments and saves results under results/<timestamp>/.
# Each experiment gets its own subfolder with config.json, results.npz,
# summary.txt, and comparison.png.
#
# Usage:
#   chmod +x run_simulation.sh
#   ./run_simulation.sh
# ============================================================================

# Common settings
EPOCHS=50
REPEATS=10
DX=4
DZ=8

# Timestamped output root
TIMESTAMP=$(date +%m%d-%H%M)
OUTROOT="results/${TIMESTAMP}"

echo "============================================"
echo "  IV Regression Batch Simulation"
echo "  Output: ${OUTROOT}/"
echo "  Epochs: ${EPOCHS}  |  Repeats: ${REPEATS}"
echo "============================================"



# ============================================================================
# Experiment 1: quadratic — TOSG + OSTG + SLIM variants comparison
# ============================================================================
python run_simulation.py \
    --model quadratic --dx 4 --dz 8 \
    --epochs ${EPOCHS} --repeats ${REPEATS} \
    --algo all \
    --slim-configs "16,16,identity;8,8,identity;1,1,identity" \
    --outdir "${OUTROOT}/quadratic_dx4_dz8"


python run_simulation.py \
    --model quadratic --dx 8 --dz 16 \
    --epochs ${EPOCHS} --repeats ${REPEATS} \
    --algo all \
    --slim-configs "16,16,identity;8,8,identity;1,1,identity" \
    --outdir "${OUTROOT}/quadratic_dx8_dz16"