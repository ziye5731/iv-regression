"""
experiment_config.py -- Centralized configuration for IV regression experiments.

ALL tunable parameters and experiment settings are defined here.
Edit this file to change any hyperparameter; run_simulation.py reads from here.
This file is copied to the results directory for reproducibility.
"""

# ============================================================================
# 1. Model
# ============================================================================
MODEL = "linear"                 # "linear", "quadratic", "poly2", "poly3"

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
SEED = 42

# ============================================================================
# 5. Training
# ============================================================================
N_EPOCHS = 100                  # number of epochs
N_SAMPLES = 5000                # samples per epoch
N_REPEATS = 10                  # number of independent runs

# ============================================================================
# 6. Algorithm selection
# ============================================================================
# Which algorithms to run: "tosg", "ostg", "slim", "both" (= tosg + slim), "all"
ALGORITHMS = "all"

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
    (16, 16, "identity"),
    (8,  8,  "identity"),
    (1,  1,  "identity"),
]

# ============================================================================
# 10. Output
# ============================================================================
OUTDIR = None                   # None = auto-generate timestamped directory
SAVE_PLOT = None                # None = save as comparison.png in OUTDIR
