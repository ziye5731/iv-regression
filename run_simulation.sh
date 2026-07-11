#!/bin/bash
set -euo pipefail

# ============================================================================
# IV Regression Simulation Runner
#
# Reads ALL settings from experiment_config.py.
# Edit experiment_config.py to change any parameter, then run this script.
#
# Usage:
#   chmod +x run_simulation.sh
#   ./run_simulation.sh
# ============================================================================

python run_simulation.py
