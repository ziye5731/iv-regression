"""
experiment_config.py -- Centralized configuration for IV regression experiments.

ALL tunable parameters and experiment settings are defined here.
Edit this file to change any hyperparameter; run_simulation.py reads from here.
This file is copied to the results directory for reproducibility.
"""

# ============================================================================
# 1. DGP  (data generating process)
# ============================================================================
# Modes: "tosg", "otsg", "deepgmm", "quadratic", "logistic"
DGP_MODE = "logistic"

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
#   y = g(theta*; x) + (1/sqrt(d_x)) 1^T c + eps_y
DGP_QUADRATIC_D_X = 4
DGP_QUADRATIC_D_Z = 8
DGP_QUADRATIC_RHO = 0.5
DGP_QUADRATIC_GAMMA_SCALE = 1.0  # scales gamma* (IV strength; larger → stronger instruments)
# MSE-contrast knobs (defaults = README DGP):
#   THETA_SCALE ↓  → less saturated link → MSE more sensitive to theta*
#   NOISE_EPS_Y ↓  → lower irreducible MSE floor
#   C_COEF scales the (1/sqrt(d_x))1^T c term (drives both noise floor and endogeneity)
DGP_QUADRATIC_THETA_SCALE = 1.0
DGP_QUADRATIC_NOISE_EPS_Y = 1.0
DGP_QUADRATIC_C_COEF = 1.0

# --- logistic ---
#   eps_x ~ N(0, (1-rho) I_dx),
#   z ~ N(0, I_dz),
#   c ~ N(0, rho I_dx)
#   x = gamma*^T z + c + eps_x
#   y = sigmoid(theta*^T x) + (1/sqrt(d_x)) 1^T c + eps_y
DGP_LOGISTIC_D_X = 4
DGP_LOGISTIC_D_Z = 8
DGP_LOGISTIC_RHO = 0.5
DGP_LOGISTIC_GAMMA_SCALE = 1.0  # scales gamma* (IV strength; larger → stronger instruments)
# MSE-contrast knobs (defaults = README DGP): see quadratic section for meaning.
# Recommended for a visible MSE gap: THETA_SCALE≈0.1-0.3, GAMMA_SCALE≈1-3, NOISE_EPS_Y≈0.1
DGP_LOGISTIC_THETA_SCALE = 1.0
DGP_LOGISTIC_NOISE_EPS_Y = 1.0
DGP_LOGISTIC_C_COEF = 1.0

# --- expiv (exponential / log-link IV) ---
#   x = gamma*^T z + c + eps_x
#   y = exp(theta*^T x) + (1/sqrt(d_x)) 1^T c + eps_y
# INDEX_SCALE = target std of the linear index theta*^T x (theta* is
#   auto-normalised so exp() stays well scaled). 0.5 is stable for all
#   algorithms; >= 0.75 is harder and can make fixed-step SGD diverge.
DGP_EXPIV_D_X = 4
DGP_EXPIV_D_Z = 8
DGP_EXPIV_RHO = 0.5
DGP_EXPIV_GAMMA_SCALE = 1.0
DGP_EXPIV_INDEX_SCALE = 0.5
DGP_EXPIV_NOISE_EPS_Y = 1.0
DGP_EXPIV_C_COEF = 1.0

# --- probit (probit / normal-CDF link; y continuous, see dgp.py) ---
#   y = Phi(theta*^T x) + (1/sqrt(d_x)) 1^T c + eps_y
# INDEX_SCALE = target std of theta*^T x (larger -> saturated/flat objective).
DGP_PROBIT_D_X = 4
DGP_PROBIT_D_Z = 8
DGP_PROBIT_RHO = 0.5
DGP_PROBIT_GAMMA_SCALE = 1.0
DGP_PROBIT_INDEX_SCALE = 1.0
DGP_PROBIT_NOISE_EPS_Y = 1.0
DGP_PROBIT_C_COEF = 1.0

# --- sine (periodic, non-convex) ---
#   y = a*sin(x_1 + phi) + b^T x_2: + int + (1/sqrt(d_x)) 1^T c + eps_y
# d_theta = d_x + 2 (amplitude, phase, linear coefs, intercept).
DGP_SINE_D_X = 4
DGP_SINE_D_Z = 8
DGP_SINE_RHO = 0.5
DGP_SINE_GAMMA_SCALE = 1.0
DGP_SINE_THETA_SCALE = 1.0
DGP_SINE_NOISE_EPS_Y = 1.0
DGP_SINE_C_COEF = 1.0

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
ALGO_LIST = ["sieve","tosg","otsg"]

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

# --- Sieve1 (online Sieve-SGMM: diagonal inverse-variance weighting) ---
ALGO_SIEVE_LR = 0.1
ALGO_SIEVE_LR_DECAY = 0.5
ALGO_SIEVE_DEGREE = 2          # polynomial sieve degree (1 or 2)
ALGO_SIEVE_BASIS = "poly"      # "poly" (monomials) or "hermite" (orthonormal for N(0,I))
ALGO_SIEVE_B = 1               # moment mini-batch size per step
ALGO_SIEVE_REG = 1e-2          # preconditioner regularization lambda
ALGO_SIEVE_CLIP = 10.0         # cap on per-step parameter displacement
ALGO_SIEVE_W_TYPE = "diag"     # "diag" (inverse-variance) or "identity"
ALGO_SIEVE_EMA = 0.0           # >0 -> EMA rate for preconditioner; 0 -> running average

# --- Sieve2 (full covariance weighting, projection, Polyak-Ruppert averaging) ---
ALGO_SIEVE2_LR = 0.1
ALGO_SIEVE2_LR_DECAY = 0.6    # step-size exponent a in (1/2, 1)
ALGO_SIEVE2_DEGREE = 2         # polynomial sieve degree (1 or 2)
ALGO_SIEVE2_BASIS = "poly"     # "poly" or "hermite" (basis includes intercept)
ALGO_SIEVE2_B = 1              # moment mini-batch size per step
ALGO_SIEVE2_REG = 1e-2         # ridge lambda_0
ALGO_SIEVE2_REG_DECAY = 0.25   # lambda_t = lambda_0 / t^{reg_decay} -> 0
ALGO_SIEVE2_CLIP = 10.0        # cap on per-step parameter displacement
ALGO_SIEVE2_W_TYPE = "full"    # "full" (Omega^{-1}), "diag", "identity"
ALGO_SIEVE2_EMA = 0.0          # >0 -> EMA rate for preconditioner; 0 -> running average
ALGO_SIEVE2_PROJ_RADIUS = 10.0 # projection radius (<=0 disables)
ALGO_SIEVE2_AVERAGE = True     # Polyak-Ruppert averaging

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

N_JOBS = 3                         # parallel algos (> 1 uses multiprocessing)
EARLY_STOP_THRESHOLD = 0.0         # stop when param error change < this
EARLY_STOP_PATIENCE = 0            # how many checks before stopping

