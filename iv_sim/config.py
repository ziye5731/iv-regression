"""
config.py -- Simulation parameter configuration.

All tunable parameters are centralized here for easy management.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .models import BaseModel, linear_model, MLPModel


@dataclass
class SimulationConfig:
    """Simulation configuration for IV regression.

    Supports multiple DGP modes:
      - "tosg":      TOSG paper DGP (two-sample oracle with phi nonlinearity)
      - "otsg":      OTSG paper DGP (one-sample with endogeneity via rho)
      - "deepgmm":   DeepGMM DGP (unknown h*; use MLP model)
      - "quadratic": quadratic structural form g(theta; x)
      - "logistic":  sigmoid structural form g(theta; x)

    TOSG DGP:
        z     ~ N(0, I)
        h     ~ N(1, I)
        eps_x ~ N(0, I),  eps_y ~ N(0, 1)
        x     = phi(gamma*^T z) + noise_c * (h + eps_x)
        y     = theta*^T x      + noise_c * (h_1 + eps_y)

    OTSG DGP:
        eps   ~ N(0, sigma_eps^2 I)
        nu    ~ N(rho * eps_1, 0.25)
        x     = gamma*^T z + eps
        y     = theta*^T x + nu

    DeepGMM DGP:
        z     = (z1, z2) ~ Unif([-3, 3]^2)
        eps   ~ N(0,1), gamma, delta ~ N(0, 0.1)
        x     = z1 + eps + gamma
        y     = h*(x) + eps + delta
    """

    # --- DGP mode ---
    dgp_mode: str = "tosg"   # "tosg", "otsg", "deepgmm", "quadratic", "logistic"

    # --- Data dimensions ---
    d_x: int = 5          # dimension of x
    d_z: int = 5          # dimension of z

    # --- Random seed ---
    seed: int = 42

    # --- True parameters (auto-generated) ---
    theta_star: np.ndarray = None   # (d_x, 1) for linear, or (d_theta, 1) for MLP
    gamma_star: np.ndarray = None   # (d_z, d_x)

    # --- TOSG DGP hyperparameters ---
    noise_c: float = 0.5           # noise scale c
    phi_func: str = "linear"       # first-stage nonlinearity: "linear", "quadratic", "sin", "tanh", "relu", "sigmoid", "cubic"

    # --- OTSG DGP hyperparameters ---
    otsg_sigma_eps: float = 0.5    # std of eps in first stage
    otsg_rho: float = 1.0          # endogeneity strength

    # --- DeepGMM DGP hyperparameters ---
    deepgmm_h_star: str = "abs"    # h* type: "step", "abs", "linear", "sin"
    deepgmm_hidden_sizes: list = field(default_factory=lambda: [64, 32])  # MLP hidden layers

    # --- IV strength ---
    gamma_scale: float = 1.0           # scales gamma_star for TOSG, OTSG, Quadratic, Logistic
    deepgmm_iv_strength: float = 1.0   # scales z1 coefficient for DeepGMM

    # --- Structural-noise / scaling knobs (defaults reproduce the README DGPs) ---
    noise_eps_y: float = 1.0            # std of the exogenous outcome noise eps_y
    c_coef: float = 1.0                 # coefficient on (1/sqrt(d_x)) 1^T c in y (endogeneity)
    first_stage: str = "linear"         # x = phi(gamma*^T z) + c + eps_x; see FIRST_STAGE_FUNCS
    quadratic_theta_scale: float = 1.0  # scales theta* in the Quadratic DGP
    logistic_theta_scale: float = 1.0   # scales theta* in the Logistic DGP
    expiv_index_scale: float = 1.0      # target std of theta*^T x in the ExpIV DGP
    probit_index_scale: float = 1.0     # target std of theta*^T x in the Probit DGP
    sine_theta_scale: float = 1.0       # scales theta* in the Sine DGP

    # --- Algorithm parameters ---
    tosg_lr: float = 0.01
    tosg_lr_decay: float = 0.5

    slim_lr: float = 0.01
    slim_lr_decay: float = 0.5
    slim_B_M: int = 1
    slim_B_m: int = 1
    slim_W_type: str = "identity"

    otsg_theta_lr: float = 0.01
    otsg_theta_lr_decay: float = 0.5
    otsg_gamma_lr: float = 0.01
    otsg_gamma_lr_decay: float = 0.5

    dcov_lr: float = 0.01
    dcov_lr_decay: float = 0.5
    dcov_B: int = 64

    dcov3_lr: float = 0.01
    dcov3_lr_decay: float = 0.5

    dcov4_lr: float = 0.01
    dcov4_lr_decay: float = 0.75

    sieve_lr: float = 0.1
    sieve_lr_decay: float = 0.5
    sieve_degree: int = 2          # polynomial sieve degree (1 or 2)
    sieve_basis: str = "poly"      # "poly" (monomials) or "hermite" (orthonormal for N(0,I))
    sieve_B: int = 1               # moment mini-batch size per step
    sieve_reg: float = 1e-2        # preconditioner regularization lambda
    sieve_clip: float = 10.0       # cap on per-step parameter displacement
    sieve_W_type: str = "diag"     # "diag" (inverse-variance) or "identity"
    sieve_ema: float = 0.0         # >0 -> EMA rate for preconditioner; 0 -> running average

    # --- Sieve2 (online Sieve-SGMM with full covariance weighting + averaging) ---
    sieve2_lr: float = 0.1
    sieve2_lr_decay: float = 0.75    # step-size exponent a in (1/2, 1)
    sieve2_degree: int = 2           # polynomial sieve degree (1 or 2)
    sieve2_basis: str = "poly"       # "poly" or "hermite" (basis includes intercept)
    sieve2_B: int = 1                # moment mini-batch size per step
    sieve2_reg: float = 1e-2         # ridge lambda_0
    sieve2_reg_decay: float = 0.25   # lambda_t = lambda_0 / t^{reg_decay} -> 0
    sieve2_clip: float = 10.0        # cap on per-step parameter displacement
    sieve2_W_type: str = "full"      # "full" (Omega^{-1}), "diag", "identity"
    sieve2_ema: float = 0.0          # >0 -> EMA rate for preconditioner; 0 -> running average
    sieve2_proj_radius: float = 10.0 # projection radius (<=0 disables)
    sieve2_average: bool = True      # Polyak-Ruppert averaging

    # --- Sieve3 (two-batch sieve GMM with identity weighting + averaging) ---
    sieve3_lr: float = 1e-3          # conservative: W=I, no preconditioner
    sieve3_lr_decay: float = 0.5     # step-size exponent a: alpha_t = lr0 * t^{-a}
    sieve3_degree: int = 2           # sieve degree (1 or 2)
    sieve3_basis: str = "hermite"    # "hermite" (orthonormal for N(0,I)) or "poly"
    sieve3_B_M: int = 1              # Jacobian mini-batch size
    sieve3_B_m: int = 1              # moment mini-batch size
    sieve3_clip: float = 10.0        # cap on per-step parameter displacement
    sieve3_average: bool = True      # Polyak-Ruppert averaging

    # --- GMM ablation experiments (algorithm name: "gmmexp") ---
    gmmexp_family: str = "herm"           # "herm" (orthonormal) | "poly" (monomials)
    gmmexp_basis: tuple = (0, 1, 2)       # degree set: (1,)=z, (1,2)=herm2, (0,1,2)=herm2+intercept
    gmmexp_precond: str = "gd"            # "gd" (gradient) | "nt" (preconditioned)
    gmmexp_w_type: str = "identity"       # "identity" (W=I) | "inv_var" (=1/(S+lam))
    gmmexp_m_source: str = "running"      # "running" (past avg) | "batch" (current)
    gmmexp_lr: float | None = None        # None -> per-precond default (gd 1e-3, nt 0.1)
    gmmexp_lr_decay: float | None = None  # None -> 0.5
    gmmexp_B_M: int = 1                   # Jacobian mini-batch size
    gmmexp_B_m: int = 1                   # moment mini-batch size
    gmmexp_clip: float = 10.0             # cap on per-step parameter displacement
    gmmexp_reg: float = 1e-2              # ridge for weighting / preconditioner
    gmmexp_average: bool = True           # Polyak-Ruppert averaging

    # --- Training ---
    n_iterations: int = 100000
    verbose_every: int = 50000
    history_every: int = 1   # record history every N steps to save memory
    history_metric_every: int = 1  # record metrics every N steps for saved history
    start_iteration: int = 0  # starting iteration count for resumed training

    # --- Repetition ---
    n_repeats: int = 10

    # --- Log-spaced checkpoints (used when X_AXIS_SCALE = "log") ---
    history_checkpoints: list = field(default_factory=list)  # exact steps to record; overrides history_even if non-empty

    # --- Early stopping ---
    early_stop_threshold: float = 0.0   # stop when param error changes < threshold
    early_stop_patience: int = 0        # number of checks to wait before stopping

    # --- Non-field: set after __post_init__ ---
    has_known_model: bool = True  # True → param error meaningful; False → MLP

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        from .dgp import get_dgp
        dgp = get_dgp(self.dgp_mode)
        dgp.setup_model(self, rng)
        self.has_known_model = dgp.has_known_model

    @property
    def d_theta(self) -> int:
        """Number of parameters (derived from model)."""
        return self.model.param_dim(self.d_x)

