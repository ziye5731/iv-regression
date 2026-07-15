"""
experiment_config.py -- Centralized configuration for IV regression experiments.

ALL tunable parameters and experiment settings are defined here.
Edit this file to change any hyperparameter; run_simulation.py reads from here.
This file is copied to the results directory for reproducibility.
"""

# ============================================================================
# 1. DGP  (data generating process)
# ============================================================================
DGP_MODE = "tosg_paper"

# --- tosg_paper ---
#   z     ~ N(0, I)
#   h     ~ N(1, I)
#   eps_x ~ N(0, I),  eps_y ~ N(0, 1)
#   x = phi(gamma*^T z) + c * (h + eps_x)
#   y = theta*^T x      + c * (h_1 + eps_y)
DGP_D_X = 8
DGP_D_Z = 16
DGP_TOSG_NOISE_C = 1.0
DGP_TOSG_PHI_FUNC = "quadratic"     # "linear" or "quadratic"

# ============================================================================
# 2. Algorithms
# ============================================================================
ALGO_LIST = ["tosg", "otsg", "slim", "dcov"]

# --- TOSG ---
ALGO_TOSG_LR = 0.01
ALGO_TOSG_LR_DECAY = 0.5

# --- OTSG ---
ALGO_OTSG_THETA_LR = 0.01
ALGO_OTSG_THETA_LR_DECAY = 0.5
ALGO_OTSG_GAMMA_LR = 0.01
ALGO_OTSG_GAMMA_LR_DECAY = 0.5

# --- SLIM ---
ALGO_SLIM_LR = 0.01
ALGO_SLIM_LR_DECAY = 0.5
ALGO_SLIM_CONFIGS = [
    (8,  8,  "identity"),
    (1,  1,  "identity"),
]

# --- DCOV ---
ALGO_DCOV_LR = 0.1
ALGO_DCOV_LR_DECAY = 0.5
ALGO_DCOV_B = 4

# ============================================================================
# 3. Other
# ============================================================================
SEED = 1
N_ITERATIONS = int(5e5)
N_REPEATS = 5
VERBOSE_EVERY = int(1e5)
RESUME_FROM = None
OUTDIR = None
SAVE_PLOT = None
X_AXIS_SCALE = "log"
