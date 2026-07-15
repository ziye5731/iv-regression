"""
experiment_config.py -- Centralized configuration for IV regression experiments.

ALL tunable parameters and experiment settings are defined here.
Edit this file to change any hyperparameter; run_simulation.py reads from here.
This file is copied to the results directory for reproducibility.
"""

# ============================================================================
# 1. Model
# ============================================================================
MODEL = "poly3"                 # "linear", "quadratic", "poly2", "poly3"

# ============================================================================
# 2. Data dimensions
# ============================================================================
D_X = 4                         # dimension of explanatory variable x
D_Z = 8                         # dimension of instrument z

# ============================================================================
# 3. Data distribution parameters
# ============================================================================
MEAN_Z = 0.0                    # mean of instrument z (per component)
SIGMA_Z = 1.0                   # std  of instrument z (per component)
SIGMA_C = 0.5                   # std of confounding term c
SIGMA_Y = 0.3                   # std of noise eps_y
SIGMA_X = 0.3                   # std of noise eps_x (per component)

# ============================================================================
# 4. Random seed
# ============================================================================
SEED = 1

# ============================================================================
# 5. Training
# ============================================================================
N_ITERATIONS = int(5e5)           # iterations per run (same for all algorithms)
N_REPEATS = 10                  # number of independent runs

# ============================================================================
# 6. Algorithm selection
# ============================================================================
# List of algorithms to run.  Valid entries:
#   "tosg", "ostg", "dcov", "slim" (all SLIM_CONFIGS variants),
#   or "all" (runs everything)
ALGORITHMS = ["tosg", "ostg", "dcov"]

# ============================================================================
# 7. TOSG-IVaR hyperparameters
# ============================================================================
TOSG_LR = 0.01                  # initial learning rate
TOSG_LR_DECAY = 0.5             # decay exponent: alpha_t = lr / t^decay

# ============================================================================
# 8. OSTG-IVaR hyperparameters
# ============================================================================
OSTG_THETA_LR = 0.01            # learning rate for theta
OSTG_THETA_LR_DECAY = 0.5
OSTG_GAMMA_LR = 0.01            # learning rate for gamma (first-stage)
OSTG_GAMMA_LR_DECAY = 0.5

# ============================================================================
# 9. First-Order SLIM hyperparameters
# ============================================================================
SLIM_LR = 0.01                  # initial learning rate
SLIM_LR_DECAY = 0.5             # decay exponent

# Multiple SLIM variants for side-by-side comparison.
# Each entry: (B_M, B_m, W_type), where
#   B_M    = batch size for Jacobian estimate M̃
#   B_m    = batch size for moment estimate m̃
#   W_type = weighting matrix type: "identity" or "random"
SLIM_CONFIGS = [
    (8,  8,  "identity"),
    (1,  1,  "identity"),
]

# ============================================================================
# 10. Distance Covariance Optimization (DCO) hyperparameters
# ============================================================================
DCOV_LR = 0.01                  # initial learning rate
DCOV_LR_DECAY = 0.5             # decay exponent
DCOV_B = 64                     # batch size for distance covariance estimate

# ============================================================================
# 11. Output
# ============================================================================
OUTDIR = None                   # None = auto-generate timestamped directory
SAVE_PLOT = None                # None = save as comparison.png in OUTDIR
X_AXIS_SCALE = "log"            # "log" or "linear" for convergence plots
