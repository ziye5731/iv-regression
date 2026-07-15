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

    Supports three DGP modes:
      - "tosg":    TOSG paper DGP (two-sample oracle with phi nonlinearity)
      - "otsg":    OTSG paper DGP (one-sample with endogeneity via rho)
      - "deepgmm": DeepGMM DGP (unknown h*; use MLP model)

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
    dgp_mode: str = "tosg"   # "tosg", "otsg", "deepgmm"

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
    phi_func: str = "linear"       # first-stage nonlinearity: "linear" or "quadratic"

    # --- OTSG DGP hyperparameters ---
    otsg_sigma_eps: float = 0.5    # std of eps in first stage
    otsg_rho: float = 1.0          # endogeneity strength

    # --- DeepGMM DGP hyperparameters ---
    deepgmm_h_star: str = "abs"    # h* type: "step", "abs", "linear", "sin"
    deepgmm_hidden_sizes: list = field(default_factory=lambda: [64, 32])  # MLP hidden layers

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

    # --- Training ---
    n_iterations: int = 100000
    verbose_every: int = 50000

    # --- Repetition ---
    n_repeats: int = 10

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

