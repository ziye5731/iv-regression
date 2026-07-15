"""
dgp.py -- Data Generating Process abstraction.

Each DGP subclass encapsulates everything that differs between DGPs:
  - parameter configuration (reading from experiment_config)
  - data generator creation
  - model setup (LinearModel vs MLP)
  - display / summary formatting
  - whether the structural form is known (→ param error is meaningful)

This eliminates scattered `if dgp_mode == "xxx"` branches throughout the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import SimulationConfig


# ============================================================================
# Abstract base
# ============================================================================

class BaseDGP(ABC):
    """Abstract base for a Data Generating Process.

    Attributes:
        dgp_mode: string identifier matching experiment_config.DGP_MODE.
        has_known_model: True if g(theta;x) is a known parametric form
            (→ ||theta - theta*|| is meaningful). False for MLP / nonparametric.
    """

    dgp_mode: str = ""
    has_known_model: bool = True

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @abstractmethod
    def configure(self, config: SimulationConfig, cfg) -> None:
        """Read mode-specific hyperparams from experiment_config into config."""
        ...

    @abstractmethod
    def setup_model(self, config: SimulationConfig, rng: np.random.Generator) -> None:
        """Create the structural model (LinearModel / MLPModel) and true params."""
        ...

    # ------------------------------------------------------------------
    # Data generator
    # ------------------------------------------------------------------

    @abstractmethod
    def create_generator(self, config: SimulationConfig, seed: int | None = None):
        """Return the appropriate data generator for this DGP."""
        ...

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @property
    def param_error_label(self) -> str:
        """Label for the per-step metric printed during training."""
        return "||theta - theta*||"

    def compute_param_metric(self, theta: np.ndarray, config: SimulationConfig) -> float:
        """Compute the per-step metric (param error or param norm)."""
        return float(np.linalg.norm(theta - config.theta_star))

    def compute_pred_mse(self, theta: np.ndarray, config: SimulationConfig,
                         generator, n_test: int = 500) -> float:
        """Quick prediction MSE on a fresh batch (for verbose progress)."""
        _, x_test, y_test = generator.generate_batch(n_test)
        y_pred = config.model.predict(theta, x_test)
        return float(np.mean((y_pred - y_test) ** 2))

    def startup_lines(self, config: SimulationConfig) -> list[str]:
        """Lines printed in the startup banner."""
        return [
            f"  DGP:        {self.dgp_mode.upper()}",
            f"  Dimensions: d_x = {config.d_x}, d_z = {config.d_z}, "
            f"d_theta = {config.d_theta}",
        ]

    def summary_dgp_line(self, config: SimulationConfig) -> str:
        """One-line DGP description for summary.txt."""
        return f"DGP:         {self.dgp_mode.upper()}"

    def summary_dim_line(self, config: SimulationConfig) -> str:
        """Dimension line for summary.txt."""
        return (f"Dimensions:  d_x={config.d_x}, d_z={config.d_z}, "
                f"d_theta={config.d_theta}")

    def summary_header(self) -> str:
        """Table header for summary.txt."""
        return (f"{'Algorithm':<22} {'Median':>10} {'Mean+/-Std':>18} "
                f"{'Pred MSE (med)':>16}")

    def summary_row(self, algo_name: str, res: dict) -> str:
        """Table row for one algorithm in summary.txt."""
        label = algo_name.upper()
        pe_med = res["param_error_median"][-1]
        pe_mean = res["param_error_mean"][-1]
        pe_std = res["param_error_std"][-1]
        pm_med = res["pred_mse_median"][-1]
        return (f"{label:<22} {pe_med:>8.4f}   "
                f"{pe_mean:>8.4f}+/-{pe_std:.4f}   {pm_med:>12.4f}")


# ============================================================================
# TOSG
# ============================================================================

class TOSGDGP(BaseDGP):
    """TOSG paper DGP.

        z     ~ N(0, I)
        h     ~ N(1, I)
        eps_x ~ N(0, I),  eps_y ~ N(0, 1)
        x     = phi(gamma*^T z) + noise_c * (h + eps_x)
        y     = theta*^T x      + noise_c * (h_1 + eps_y)
    """

    dgp_mode = "tosg"

    def configure(self, config, cfg):
        config.d_x = getattr(cfg, "DGP_TOSG_D_X", getattr(cfg, "DGP_D_X", 5))
        config.d_z = getattr(cfg, "DGP_TOSG_D_Z", getattr(cfg, "DGP_D_Z", 5))
        config.noise_c = getattr(cfg, "DGP_TOSG_NOISE_C", 0.5)
        config.phi_func = getattr(cfg, "DGP_TOSG_PHI_FUNC", "linear")

    def setup_model(self, config, rng):
        from .models import linear_model
        config.model = linear_model
        if config.theta_star is None:
            config.theta_star = config.model.true_params(rng, config.d_x)
        if config.gamma_star is None:
            config.gamma_star = rng.normal(0, 1, size=(config.d_z, config.d_x))

    def create_generator(self, config, seed=None):
        from .data_generator import IVDataGenerator
        return IVDataGenerator(config, seed=seed)

    def summary_dgp_line(self, config):
        return (f"DGP:         TOSG, phi={config.phi_func}, "
                f"noise_c={config.noise_c}")


# ============================================================================
# OTSG
# ============================================================================

class OTSGDGP(BaseDGP):
    """OTSG paper DGP.

        eps   ~ N(0, sigma_eps^2 I)
        nu    ~ N(rho * eps_1, 0.25)
        x     = gamma*^T z + eps
        y     = theta*^T x + nu
    """

    dgp_mode = "otsg"

    def configure(self, config, cfg):
        config.d_x = getattr(cfg, "DGP_OTSG_D_X", getattr(cfg, "DGP_D_X", 5))
        config.d_z = getattr(cfg, "DGP_OTSG_D_Z", getattr(cfg, "DGP_D_Z", 5))
        config.otsg_sigma_eps = getattr(cfg, "DGP_OTSG_SIGMA_EPS", 0.5)
        config.otsg_rho = getattr(cfg, "DGP_OTSG_RHO", 1.0)

    def setup_model(self, config, rng):
        from .models import linear_model
        config.model = linear_model
        if config.theta_star is None:
            config.theta_star = config.model.true_params(rng, config.d_x)
        if config.gamma_star is None:
            config.gamma_star = rng.normal(0, 1, size=(config.d_z, config.d_x))

    def create_generator(self, config, seed=None):
        from .data_generator import OTSGDataGenerator
        return OTSGDataGenerator(config, seed=seed)

    def summary_dgp_line(self, config):
        return (f"DGP:         OTSG, sigma_eps={config.otsg_sigma_eps}, "
                f"rho={config.otsg_rho}")


# ============================================================================
# DeepGMM
# ============================================================================

class DeepGMMDGP(BaseDGP):
    """DeepGMM DGP (unknown structural form → MLP).

        z     = (z1, z2) ~ Unif([-3, 3]^2)
        eps   ~ N(0,1),  gamma, delta ~ N(0, 0.1)
        x     = z1 + eps + gamma
        y     = h*(x) + eps + delta

    h* ∈ {step, abs, linear, sin}.
    Dimensions are fixed: d_x = 1, d_z = 2.
    Model is MLP (unknown structural form).
    """

    dgp_mode = "deepgmm"
    has_known_model = False

    def configure(self, config, cfg):
        # Dimensions are fixed by this DGP -- set in setup_model()
        config.deepgmm_h_star = getattr(cfg, "DGP_DEEPGMM_H_STAR", "abs")
        config.deepgmm_hidden_sizes = list(
            getattr(cfg, "DGP_DEEPGMM_HIDDEN_SIZES", [64, 32])
        )

    def setup_model(self, config, rng):
        from .models import MLPModel
        config.d_x = 1
        config.d_z = 2
        config.model = MLPModel(hidden_sizes=config.deepgmm_hidden_sizes)
        config.theta_star = np.zeros((config.model.param_dim(config.d_x), 1))
        config.gamma_star = np.zeros((config.d_z, config.d_x))

    def create_generator(self, config, seed=None):
        from .data_generator import DeepGMMDataGenerator
        gen = DeepGMMDataGenerator(config, seed=seed)
        gen.set_model(config.model)
        return gen

    # ------------------------------------------------------------------
    # Display (MLP-specific -- no param error)
    # ------------------------------------------------------------------

    @property
    def param_error_label(self) -> str:
        return "||theta||"

    def compute_param_metric(self, theta, config):
        return float(np.linalg.norm(theta))

    def startup_lines(self, config):
        return [
            f"  DGP:        DeepGMM, h* = {config.deepgmm_h_star}",
            f"  Dimensions: d_x = {config.d_x}, d_z = {config.d_z}",
            f"  Model:      MLP({config.deepgmm_hidden_sizes}), "
            f"#params = {config.d_theta}",
        ]

    def summary_dgp_line(self, config):
        return f"DGP:         DeepGMM (h*={config.deepgmm_h_star})"

    def summary_dim_line(self, config):
        return (f"Dimensions:  d_x={config.d_x}, d_z={config.d_z}, "
                f"MLP params={config.d_theta}")

    def summary_header(self):
        return f"{'Algorithm':<22} {'Pred MSE (med)':>16}"

    def summary_row(self, algo_name, res):
        label = algo_name.upper()
        pm_med = res["pred_mse_median"][-1]
        return f"{label:<22} {pm_med:>16.6f}"


# ============================================================================
# Registry
# ============================================================================

_DGP_REGISTRY: dict[str, BaseDGP] = {
    "tosg":      TOSGDGP(),
    "tosg_paper": TOSGDGP(),
    "otsg":      OTSGDGP(),
    "otsg_paper": OTSGDGP(),
    "deepgmm":   DeepGMMDGP(),
}


def get_dgp(mode: str) -> BaseDGP:
    """Look up the DGP descriptor by mode string.

    Raises ValueError for unknown modes.
    """
    if mode not in _DGP_REGISTRY:
        raise ValueError(
            f"Unknown DGP mode: '{mode}'. "
            f"Valid modes: {list(_DGP_REGISTRY.keys())}"
        )
    return _DGP_REGISTRY[mode]
