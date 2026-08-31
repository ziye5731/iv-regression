"""
experiment_config.py -- Centralized configuration for IV regression experiments.

ALL tunable parameters and experiment settings are defined here.
Edit this file to change any hyperparameter; run_simulation.py reads from here.
This file is copied to the results directory for reproducibility.
"""

# ============================================================================
# 1. DGP  (data generating process)
# ============================================================================
# Modes: "tosg", "otsg", "deepgmm", "quadratic"
DGP_MODE = "quadratic"

# --- tosg ---
#   z     ~ N(0, I)
#   h     ~ N(1, I)
#   eps_x ~ N(0, I),  eps_y ~ N(0, 1)
#   x = phi(gamma*^T z) + c * (h + eps_x)
#   y = theta*^T x      + c * (h_1 + eps_y)
DGP_TOSG_D_X = 8
DGP_TOSG_D_Z = 16
DGP_TOSG_NOISE_C = 1.0
DGP_TOSG_PHI_FUNC = "quadratic"     # "linear", "quadratic", "sin", "tanh", "relu", "sigmoid", "cubic"
DGP_TOSG_GAMMA_SCALE = 1.0          # scales gamma* (IV strength; larger → stronger instruments)

# --- otsg ---
#   eps ~ N(0, sigma_eps^2 I),  nu ~ N(rho * eps_1, 0.25)
#   x = gamma*^T z + eps
#   y = theta*^T x + nu
DGP_OTSG_D_X = 8
DGP_OTSG_D_Z = 16
DGP_OTSG_SIGMA_EPS = 0.5          # 0.5 or 1.0
DGP_OTSG_RHO = 1.0                # 1.0 or 4.0
DGP_OTSG_GAMMA_SCALE = 1.0       # scales gamma* (IV strength; larger → stronger instruments)

# --- quadratic ---
#   eps_x ~ N(0, (1-rho) I_dx),
#   z ~ N(0, I_dz),
#   c ~ N(0, rho I_dx)
#   x = gamma*^T z + c + eps_x
#   y = g(theta*; x) + 1^T c + eps_y
DGP_QUADRATIC_D_X = 4
DGP_QUADRATIC_D_Z = 8
DGP_QUADRATIC_RHO = 0.5
DGP_QUADRATIC_GAMMA_SCALE = 1.0  # scales gamma* (IV strength; larger → stronger instruments)

# --- deepgmm ---
#   z = (z1, z2) ~ Unif([-3, 3]^2)
#   eps ~ N(0,1), gamma, delta ~ N(0, 0.1)
#   x = z1 + eps + gamma
#   y = h*(x) + eps + delta
DGP_DEEPGMM_H_STAR = "sin"        # "step", "abs", "linear", "sin"
DGP_DEEPGMM_HIDDEN_SIZES = [64,32]  # MLP hidden layers
DGP_DEEPGMM_IV_STRENGTH = 1.0    # scales z1 coefficient (IV strength; larger → stronger instruments)

# ============================================================================
# 2. Algorithms
# ============================================================================
ALGO_LIST = ["tosg","otsg","sieve"]

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

# --- DCOV3 ---
ALGO_DCOV3_LR = 0.01
ALGO_DCOV3_LR_DECAY = 0.5

# --- DCOV4 ---
ALGO_DCOV4_LR = 0.01
ALGO_DCOV4_LR_DECAY = 0.5

# --- SieveGMM ---
ALGO_SIEVE_LR = 0.1
ALGO_SIEVE_LR_DECAY = 0.5
ALGO_SIEVE_DEGREE = 2          # polynomial sieve degree (1 or 2)
ALGO_SIEVE_BASIS = "poly"      # "poly" (monomials) or "hermite" (orthonormal for N(0,I))
ALGO_SIEVE_B = 1               # moment mini-batch size per step
ALGO_SIEVE_REG = 1e-2          # preconditioner regularization lambda
ALGO_SIEVE_CLIP = 10.0         # cap on per-step parameter displacement
ALGO_SIEVE_W_TYPE = "diag"     # "diag" (inverse-variance) or "identity"
ALGO_SIEVE_EMA = 0.0           # >0 -> EMA rate for preconditioner; 0 -> running average

# ============================================================================
# 3. Other
# ============================================================================
SEED = 10
N_ITERATIONS = int(1e8)
N_REPEATS = 1
VERBOSE_EVERY = int(1e5)
HISTORY_EVERY = None          # record training history every N iterations
RESUME_FROM = None
OUTDIR = None
SAVE_PLOT = None
X_AXIS_SCALE = "log"  # 'linear', 'log', 'symlog', 'asinh', 'logit', 'function', 'functionlog'

N_JOBS = 4                         # parallel algos (> 1 uses multiprocessing)
EARLY_STOP_THRESHOLD = 0.0         # stop when param error change < this
EARLY_STOP_PATIENCE = 0            # how many checks before stopping

