"""
experiment_config.py -- Centralized configuration for IV regression experiments.

ALL tunable parameters and experiment settings are defined here.
Edit this file to change any hyperparameter; run_simulation.py reads from here.
This file is copied to the results directory for reproducibility.
"""

# ============================================================================
# 1. DGP  (data generating process)
# ============================================================================
# Modes: "tosg", "otsg", "deepgmm", "quadratic", "logistic", "expiv", "probit", "sine"
#        or a composite "<structural>-<first_stage>" name
#        (first item = y-x model, second item = x-z model), e.g.
#        "quadratic-linear", "quadratic-sin", "logistic-tanh", "probit-relu".
#        First stages: linear, quadratic, sin, tanh, relu, sigmoid, cubic.
#        A plain name is the same as "<structural>-linear".
DGP_MODE = "quadratic-linear"

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
DGP_EXPIV_GAMMA_SCALE = 0.1
DGP_EXPIV_INDEX_SCALE = 0.5
DGP_EXPIV_NOISE_EPS_Y = 0.1
DGP_EXPIV_C_COEF = 0.5

# --- probit (probit / normal-CDF link; y continuous, see dgp.py) ---
#   y = Phi(theta*^T x) + (1/sqrt(d_x)) 1^T c + eps_y
# INDEX_SCALE = target std of theta*^T x (larger -> saturated/flat objective).
DGP_PROBIT_D_X = 4
DGP_PROBIT_D_Z = 8
DGP_PROBIT_RHO = 0.5
DGP_PROBIT_GAMMA_SCALE = 0.2
DGP_PROBIT_INDEX_SCALE = 1.0
DGP_PROBIT_NOISE_EPS_Y = 1.0
DGP_PROBIT_C_COEF = 1.0

# --- sine (periodic, non-convex) ---
#   y = a*sin(x_1 + phi) + b^T x_2: + int + (1/sqrt(d_x)) 1^T c + eps_y
# d_theta = d_x + 2 (amplitude, phase, linear coefs, intercept).
DGP_SINE_D_X = 2
DGP_SINE_D_Z = 4
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
ALGO_LIST = ["gmmexp","tosg","otsg"]

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
ALGO_SIEVE_LR = 0.01
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

# --- Sieve3 (two-batch sieve GMM with identity weighting + Polyak-Ruppert averaging) ---
ALGO_SIEVE3_LR = 0.01          # conservative: W=I, no preconditioner (raise for well-scaled DGPs)
ALGO_SIEVE3_LR_DECAY = 0.5     # step-size exponent a: alpha_t = lr0 * t^{-a}
ALGO_SIEVE3_DEGREE = 2         # sieve degree (1 or 2); need p >= d_theta to identify
ALGO_SIEVE3_BASIS = "hermite"  # "hermite" (orthonormal for N(0,I)) or "poly"
ALGO_SIEVE3_B_M = 1            # Jacobian mini-batch size
ALGO_SIEVE3_B_m = 1            # moment mini-batch size
ALGO_SIEVE3_CLIP = 10.0        # cap on per-step parameter displacement
ALGO_SIEVE3_AVERAGE = True     # Polyak-Ruppert averaging

# --- GMM ablation experiments (algorithm name: "gmmexp") ---
# Any hyperparameter below may be a scalar or a LIST; LISTs are expanded into
# the full cross-product (one run per combination), so a single
# ALGO_LIST = ["gmmexp"] entry can run the whole grid.
# Result labels encode every axis:
#   gmmexp_{h|p}{degrees}_{precond}_{i|invS}_{run|batch}[_B{B_M}m{B_m}]
#   (h = Hermite family, p = polynomial; digits = the included degrees)
#
#   family  : "herm" (orthonormal Hermite, degrees 0-3) | "poly" (monomials, 0-2)
#   basis   : the DEGREE SET of the sieve.  A flat list of ints is ONE basis;
#             a list of tuples/lists means SEVERAL bases:
#               [1]            -> psi(z) = z                ("lin")
#               [1, 2]         -> z, H_2, z_i z_j          (old "herm2")
#               [0, 1, 2]      -> the above PLUS H_0 = 1    (herm2 + intercept)
#               [1, 2, 3]      -> adds H_3 terms
#               [(1,), (1,2), (0,1,2)]  -> three separate runs
#             NOTE: orthonormal Hermite blocks have E[psi psi^T] = I, and the
#             constant H_0 is NOT included unless 0 is in the degree set.
#   precond : "gd" (plain gradient) | "nt" (Newton-type preconditioning)
#   w_type  : "identity" (W = I)    | "inv_var" (W = diag(1/(S+lambda)):
#                                              divide by S -> adaptive step)
#   m_source: "running" (op. Jacobian = running average of past batches)
#             "batch"   (op. Jacobian = the current batch)
# NOTE: LR must be paired with w_type -- "identity" needs a small LR, "inv_var"
# needs a much larger one (W ~ 1/S is small while theta is still far away).
# Recommended: keep ALGO_GMMEXP_W_TYPE scalar and run the grid once per value.
# Reproducing the existing algorithms (also set LR / LR_DECAY / AVERAGE):
#   First-Order SLIM : family="herm", basis=[1],   precond="gd", w_type="identity",
#                      m_source="batch",   B_M=8, B_m=8, LR=0.01, AVERAGE=False
#   Sieve3           : family="herm", basis=[1,2], precond="gd", w_type="identity",
#                      m_source="batch",   B_M=1, B_m=1, LR=0.01, AVERAGE=True
#   Sieve1           : family="poly", basis=[1,2], precond="nt", w_type="inv_var",
#                      m_source="running", B_M=1, B_m=1, LR=0.01, AVERAGE=False
ALGO_GMMEXP_FAMILY = "herm"            # "herm" | "poly"
ALGO_GMMEXP_BASIS = [(1), (0, 1), (0, 1, 2), (0,1,2,3)]          # one degree set, or [(1,), (1,2), (0,1,2)] for several
ALGO_GMMEXP_PRECOND = ["nt"]
ALGO_GMMEXP_W_TYPE = ["identity"]    # "identity" | "inv_var"
ALGO_GMMEXP_M_SOURCE = ["running"]   # "running" | "batch"
ALGO_GMMEXP_B_M = 1                # scalar or list
ALGO_GMMEXP_B_m = 1                # scalar or list
ALGO_GMMEXP_LR = 0.01              # None -> per-precond default (gd: 1e-3, nt: 0.1)
ALGO_GMMEXP_LR_DECAY = 0.5
ALGO_GMMEXP_CLIP = 10.0            # cap on per-step parameter displacement
ALGO_GMMEXP_REG = 1e-2             # ridge for the weighting / preconditioner
ALGO_GMMEXP_AVERAGE = False         # Polyak-Ruppert averaging

# ============================================================================
# 3. Other
# ============================================================================
SEED = 10
N_ITERATIONS = int(5e6)
N_REPEATS = 10
VERBOSE_EVERY = int(1e5)
HISTORY_EVERY = None          # record training history every N iterations
RESUME_FROM = None
OUTDIR = None
SAVE_PLOT = None
X_AXIS_SCALE = "log"  # 'linear', 'log', 'symlog', 'asinh', 'logit', 'function', 'functionlog'

N_JOBS = 3                         # parallel algos (> 1 uses multiprocessing)
EARLY_STOP_THRESHOLD = 0.0         # stop when param error change < this
EARLY_STOP_PATIENCE = 0            # how many checks before stopping

