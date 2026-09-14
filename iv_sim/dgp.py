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


def _index_std_scale(theta: np.ndarray, gamma_star: np.ndarray,
                     d_x: int, target_std: float,
                     first_stage: str = "linear",
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Rescale theta so that Var(theta^T x) == target_std^2.

    Under x = phi(gamma_star^T z) + c + eps_x with Var(c) + Var(eps_x) = I the
    covariance of x is gamma_star^T gamma_star + I when phi is the identity;
    for a nonlinear first stage it is estimated by Monte Carlo.  Used by the
    exponential / probit DGPs to keep the linear index -- and therefore
    exp(.) / Phi(.) -- well scaled regardless of d_x, d_z and first stage.
    """
    if first_stage == "linear":
        Sigma_x = gamma_star.T @ gamma_star + np.eye(d_x)
    else:
        from .data_generator import apply_first_stage
        rng = rng if rng is not None else np.random.default_rng(0)
        n = 20000
        z = rng.normal(0, 1, size=(n, gamma_star.shape[0]))
        x = (apply_first_stage(z @ gamma_star, first_stage)
             + rng.normal(0, 1, size=(n, d_x)))
        x = x - x.mean(axis=0, keepdims=True)
        Sigma_x = (x.T @ x) / n
    var_idx = float(theta.ravel() @ Sigma_x @ theta.ravel())
    if not np.isfinite(var_idx) or var_idx <= 0.0:
        return theta
    return theta * (target_std / np.sqrt(var_idx))


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
        # Flatten both to avoid (d_x,) - (d_x, 1) → (d_x, d_x) broadcasting bug
        return float(np.linalg.norm(theta.ravel() - config.theta_star.ravel()))

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
            f"  IV strength: gamma_scale = {config.gamma_scale}",
        ]

    def summary_dgp_line(self, config: SimulationConfig) -> str:
        """One-line DGP description for summary.txt."""
        return (f"DGP:         {self.dgp_mode.upper()}, "
                f"gamma_scale={config.gamma_scale}")

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
        config.gamma_scale = getattr(cfg, "DGP_TOSG_GAMMA_SCALE", 1.0)

    def setup_model(self, config, rng):
        from .models import linear_model
        config.model = linear_model
        if config.theta_star is None:
            config.theta_star = config.model.true_params(rng, config.d_x)
        if config.gamma_star is None:
            config.gamma_star = rng.normal(0, 1, size=(config.d_z, config.d_x))
        config.gamma_star *= config.gamma_scale

    def create_generator(self, config, seed=None):
        from .data_generator import IVDataGenerator
        return IVDataGenerator(config, seed=seed)

    def summary_dgp_line(self, config):
        return (f"DGP:         TOSG, phi={config.phi_func}, "
                f"noise_c={config.noise_c}, "
                f"gamma_scale={config.gamma_scale}")


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
        config.gamma_scale = getattr(cfg, "DGP_OTSG_GAMMA_SCALE", 1.0)

    def setup_model(self, config, rng):
        from .models import linear_model
        config.model = linear_model
        if config.theta_star is None:
            config.theta_star = config.model.true_params(rng, config.d_x)
        if config.gamma_star is None:
            config.gamma_star = rng.normal(0, 1, size=(config.d_z, config.d_x))
        config.gamma_star *= config.gamma_scale

    def create_generator(self, config, seed=None):
        from .data_generator import OTSGDataGenerator
        return OTSGDataGenerator(config, seed=seed)

    def summary_dgp_line(self, config):
        return (f"DGP:         OTSG, sigma_eps={config.otsg_sigma_eps}, "
                f"rho={config.otsg_rho}, "
                f"gamma_scale={config.gamma_scale}")


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
        config.deepgmm_iv_strength = getattr(cfg, "DGP_DEEPGMM_IV_STRENGTH", 1.0)

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

    def get_h_star(self, config):
        """Get the true structural function h* for visualization.

        Returns the h_star callable from the data generator.
        """
        from .data_generator import _make_h_star
        h_type = getattr(config, "deepgmm_h_star", "abs")
        return _make_h_star(h_type)

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
            f"  IV strength: iv_strength = {config.deepgmm_iv_strength}",
        ]

    def summary_dgp_line(self, config):
        return (f"DGP:         DeepGMM (h*={config.deepgmm_h_star}), "
                f"iv_strength={config.deepgmm_iv_strength}")

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


class QuadraticDGP(BaseDGP):
    """Quadratic DGP from README.

    Draw:
        eps_x ~ N(0, (1-rho) I_dx),
        z ~ N(0, I_dz),
        c ~ N(0, rho I_dx)

    x = gamma*^T z + c + eps_x
    y = g(theta*; x) + (1/sqrt(d_x)) * 1^T c + eps_y
    where g is the QuadraticModel in `iv_sim.models`.
    """

    dgp_mode = "quadratic"

    def configure(self, config, cfg):
        config.d_x = getattr(cfg, "DGP_QUADRATIC_D_X", getattr(cfg, "DGP_D_X", 5))
        config.d_z = getattr(cfg, "DGP_QUADRATIC_D_Z", getattr(cfg, "DGP_D_Z", 5))
        config.quadratic_rho = getattr(cfg, "DGP_QUADRATIC_RHO", 0.5)
        config.gamma_scale = getattr(cfg, "DGP_QUADRATIC_GAMMA_SCALE", 1.0)
        config.quadratic_theta_scale = getattr(
            cfg, "DGP_QUADRATIC_THETA_SCALE", 1.0)
        config.noise_eps_y = getattr(cfg, "DGP_QUADRATIC_NOISE_EPS_Y", 1.0)
        config.c_coef = getattr(cfg, "DGP_QUADRATIC_C_COEF", 1.0)

    def setup_model(self, config, rng):
        from .models import QuadraticModel
        config.model = QuadraticModel()
        if config.theta_star is None:
            config.theta_star = config.model.true_params(rng, config.d_x)
            config.theta_star = config.theta_star * getattr(
                config, "quadratic_theta_scale", 1.0)
        if config.gamma_star is None:
            config.gamma_star = rng.normal(0, 1, size=(config.d_z, config.d_x))
        config.gamma_star *= config.gamma_scale

    def create_generator(self, config, seed=None):
        from .data_generator import QuadraticDataGenerator
        gen = QuadraticDataGenerator(config, seed=seed)
        return gen


class LogisticDGP(BaseDGP):
    """Logistic DGP from README.

    Draw:
        eps_x ~ N(0, (1-rho) I_dx),
        z ~ N(0, I_dz),
        c ~ N(0, rho I_dx)

    x = gamma*^T z + c + eps_x
    y = g(theta*; x) + (1/sqrt(d_x)) * 1^T c + eps_y
    where g is the LogisticModel (sigmoid) in `iv_sim.models`.
    """

    dgp_mode = "logistic"

    def configure(self, config, cfg):
        config.d_x = getattr(cfg, "DGP_LOGISTIC_D_X", getattr(cfg, "DGP_D_X", 5))
        config.d_z = getattr(cfg, "DGP_LOGISTIC_D_Z", getattr(cfg, "DGP_D_Z", 5))
        config.logistic_rho = getattr(cfg, "DGP_LOGISTIC_RHO", 0.5)
        config.gamma_scale = getattr(cfg, "DGP_LOGISTIC_GAMMA_SCALE", 1.0)
        config.logistic_theta_scale = getattr(
            cfg, "DGP_LOGISTIC_THETA_SCALE", 1.0)
        config.noise_eps_y = getattr(cfg, "DGP_LOGISTIC_NOISE_EPS_Y", 1.0)
        config.c_coef = getattr(cfg, "DGP_LOGISTIC_C_COEF", 1.0)

    def setup_model(self, config, rng):
        from .models import LogisticModel
        config.model = LogisticModel()
        if config.theta_star is None:
            config.theta_star = config.model.true_params(rng, config.d_x)
            config.theta_star = config.theta_star * getattr(
                config, "logistic_theta_scale", 1.0)
        if config.gamma_star is None:
            config.gamma_star = rng.normal(0, 1, size=(config.d_z, config.d_x))
        config.gamma_star *= config.gamma_scale

    def create_generator(self, config, seed=None):
        from .data_generator import LogisticDataGenerator
        gen = LogisticDataGenerator(config, seed=seed)
        return gen

    def summary_dgp_line(self, config):
        return (f"DGP:         Logistic, rho={config.logistic_rho}, "
                f"gamma_scale={config.gamma_scale}, "
                f"theta_scale={getattr(config, 'logistic_theta_scale', 1.0)}, "
                f"sigma_eps_y={getattr(config, 'noise_eps_y', 1.0)}, "
                f"c_coef={getattr(config, 'c_coef', 1.0)}")


# ============================================================================
# ExpIV  (exponential / log-link)
# ============================================================================

class ExpIVDGP(BaseDGP):
    """Exponential (log-link) IV DGP.

        eps_x ~ N(0, (1-rho) I_dx),  z ~ N(0, I_dz),  c ~ N(0, rho I_dx)
        x = gamma*^T z + c + eps_x
        y = exp(theta*^T x) + c_coef * (1/sqrt(d_x)) 1^T c + eps_y

    The exponential mean is the canonical Poisson / log-link specification
    used by log-link IV estimators (Mullahy 1997; Windmeijer & Santos Silva
    1997).  theta* is auto-normalised so the linear index has a target
    standard deviation (DGP_EXPIV_INDEX_SCALE, default 0.5); values >= 0.75
    make the problem much harder and can make fixed-step SGD diverge.
    """

    dgp_mode = "expiv"

    def configure(self, config, cfg):
        config.d_x = getattr(cfg, "DGP_EXPIV_D_X", 4)
        config.d_z = getattr(cfg, "DGP_EXPIV_D_Z", 8)
        config.expiv_rho = getattr(cfg, "DGP_EXPIV_RHO", 0.5)
        config.gamma_scale = getattr(cfg, "DGP_EXPIV_GAMMA_SCALE", 1.0)
        config.expiv_index_scale = getattr(cfg, "DGP_EXPIV_INDEX_SCALE", 0.5)
        config.noise_eps_y = getattr(cfg, "DGP_EXPIV_NOISE_EPS_Y", 1.0)
        config.c_coef = getattr(cfg, "DGP_EXPIV_C_COEF", 1.0)

    def setup_model(self, config, rng):
        from .models import ExponentialModel
        config.model = ExponentialModel()
        if config.gamma_star is None:
            config.gamma_star = rng.normal(0, 1, size=(config.d_z, config.d_x))
            config.gamma_star *= config.gamma_scale
        if config.theta_star is None:
            raw = config.model.true_params(rng, config.d_x)
            config.theta_star = _index_std_scale(
                raw, config.gamma_star, config.d_x,
                float(getattr(config, "expiv_index_scale", 0.5)),
                first_stage=getattr(config, "first_stage", "linear"),
                rng=rng)

    def create_generator(self, config, seed=None):
        from .data_generator import ExponentialDataGenerator
        return ExponentialDataGenerator(config, seed=seed)

    def summary_dgp_line(self, config):
        return (f"DGP:         ExpIV (exp link), rho={config.expiv_rho}, "
                f"gamma_scale={config.gamma_scale}, "
                f"index_scale={getattr(config, 'expiv_index_scale', 0.5)}, "
                f"sigma_eps_y={getattr(config, 'noise_eps_y', 1.0)}, "
                f"c_coef={getattr(config, 'c_coef', 1.0)}")


# ============================================================================
# Probit  (probit link)
# ============================================================================

class ProbitDGP(BaseDGP):
    """Probit-link IV DGP.

        x = gamma*^T z + c + eps_x
        y = Phi(theta*^T x) + c_coef * (1/sqrt(d_x)) 1^T c + eps_y

    Phi is the standard normal CDF.  The outcome is kept continuous
    (regression form) so that E[g(theta*;x) - y | z] = 0 holds and the IV
    algorithms target theta*; a genuinely binary outcome would need
    control-function / MLE methods outside this framework.
    """

    dgp_mode = "probit"

    def configure(self, config, cfg):
        config.d_x = getattr(cfg, "DGP_PROBIT_D_X", 4)
        config.d_z = getattr(cfg, "DGP_PROBIT_D_Z", 8)
        config.probit_rho = getattr(cfg, "DGP_PROBIT_RHO", 0.5)
        config.gamma_scale = getattr(cfg, "DGP_PROBIT_GAMMA_SCALE", 1.0)
        config.probit_index_scale = getattr(cfg, "DGP_PROBIT_INDEX_SCALE", 1.0)
        config.noise_eps_y = getattr(cfg, "DGP_PROBIT_NOISE_EPS_Y", 1.0)
        config.c_coef = getattr(cfg, "DGP_PROBIT_C_COEF", 1.0)

    def setup_model(self, config, rng):
        from .models import ProbitModel
        config.model = ProbitModel()
        if config.gamma_star is None:
            config.gamma_star = rng.normal(0, 1, size=(config.d_z, config.d_x))
            config.gamma_star *= config.gamma_scale
        if config.theta_star is None:
            raw = config.model.true_params(rng, config.d_x)
            config.theta_star = _index_std_scale(
                raw, config.gamma_star, config.d_x,
                float(getattr(config, "probit_index_scale", 1.0)),
                first_stage=getattr(config, "first_stage", "linear"),
                rng=rng)

    def create_generator(self, config, seed=None):
        from .data_generator import ProbitDataGenerator
        return ProbitDataGenerator(config, seed=seed)

    def summary_dgp_line(self, config):
        return (f"DGP:         Probit (Phi link), rho={config.probit_rho}, "
                f"gamma_scale={config.gamma_scale}, "
                f"index_scale={getattr(config, 'probit_index_scale', 1.0)}, "
                f"sigma_eps_y={getattr(config, 'noise_eps_y', 1.0)}, "
                f"c_coef={getattr(config, 'c_coef', 1.0)}")


# ============================================================================
# Sine  (periodic, non-convex)
# ============================================================================

class SineDGP(BaseDGP):
    """Periodic (non-convex) IV DGP.

        x = gamma*^T z + c + eps_x
        y = theta*_0 sin(x_1 + theta*_1) + sum_{j>=2} theta*_j x_j
            + theta*_int + c_coef * (1/sqrt(d_x)) 1^T c + eps_y

    The phase enters through a sine, so the objective is non-convex and
    multimodal -- a global-convergence stress test (the same family as
    DeepGMM's h*(x) = sin(x)).  d_theta = d_x + 2.
    """

    dgp_mode = "sine"

    def configure(self, config, cfg):
        config.d_x = getattr(cfg, "DGP_SINE_D_X", 4)
        config.d_z = getattr(cfg, "DGP_SINE_D_Z", 8)
        config.sine_rho = getattr(cfg, "DGP_SINE_RHO", 0.5)
        config.gamma_scale = getattr(cfg, "DGP_SINE_GAMMA_SCALE", 1.0)
        config.sine_theta_scale = getattr(cfg, "DGP_SINE_THETA_SCALE", 1.0)
        config.noise_eps_y = getattr(cfg, "DGP_SINE_NOISE_EPS_Y", 1.0)
        config.c_coef = getattr(cfg, "DGP_SINE_C_COEF", 1.0)

    def setup_model(self, config, rng):
        from .models import SineModel
        config.model = SineModel()
        if config.theta_star is None:
            config.theta_star = config.model.true_params(rng, config.d_x)
            config.theta_star = config.theta_star * getattr(
                config, "sine_theta_scale", 1.0)
        if config.gamma_star is None:
            config.gamma_star = rng.normal(0, 1, size=(config.d_z, config.d_x))
        config.gamma_star *= config.gamma_scale

    def create_generator(self, config, seed=None):
        from .data_generator import SineDataGenerator
        return SineDataGenerator(config, seed=seed)

    def summary_dgp_line(self, config):
        return (f"DGP:         Sine (periodic), rho={config.sine_rho}, "
                f"gamma_scale={config.gamma_scale}, "
                f"theta_scale={getattr(config, 'sine_theta_scale', 1.0)}, "
                f"sigma_eps_y={getattr(config, 'noise_eps_y', 1.0)}, "
                f"c_coef={getattr(config, 'c_coef', 1.0)}")


# register Quadratic
_DGP_REGISTRY["quadratic"] = QuadraticDGP()

# register Logistic
_DGP_REGISTRY["logistic"] = LogisticDGP()

# register ExpIV / Probit / Sine
_DGP_REGISTRY["expiv"] = ExpIVDGP()
_DGP_REGISTRY["probit"] = ProbitDGP()
_DGP_REGISTRY["sine"] = SineDGP()


# ============================================================================
# Composite modes:  "<structural>-<first_stage>"   e.g. "quadratic-sin"
# ============================================================================

class FirstStageVariant(BaseDGP):
    """Structural DGP whose first stage x = phi(gamma*^T z) + c + eps_x is
    nonlinear.

    Created by get_dgp() for composite mode names such as ``quadratic-sin``
    (structural part ``quadratic``, first stage ``sin``) or
    ``logistic-tanh``.  The first item of the name is the y-x model, the second
    is the x-z model.  Everything except configure() is delegated to the
    wrapped structural DGP.
    """

    def __init__(self, base: BaseDGP, first_stage: str):
        self._base = base
        self.first_stage = first_stage
        self.dgp_mode = f"{base.dgp_mode}-{first_stage}"
        self.has_known_model = base.has_known_model

    def __getattr__(self, name):
        # Delegate everything not defined here (param_error_label,
        # startup_lines, summary_* overrides, get_h_star, ...) to the wrapped
        # structural DGP.
        base = object.__getattribute__(self, "_base")
        return getattr(base, name)

    def configure(self, config, cfg):
        self._base.configure(config, cfg)
        config.first_stage = self.first_stage
        # keep OTSG's first-stage model aligned with the DGP
        config.phi_func = self.first_stage

    def setup_model(self, config, rng):
        self._base.setup_model(config, rng)

    def create_generator(self, config, seed=None):
        return self._base.create_generator(config, seed=seed)

    def startup_lines(self, config):
        lines = list(self._base.startup_lines(config))
        if lines:
            lines[0] = f"  DGP:        {self.dgp_mode}"
        lines.append(
            f"  First stage: x = {self.first_stage}(gamma*^T z) + c + eps_x")
        return lines

    def summary_dgp_line(self, config):
        return (f"{self._base.summary_dgp_line(config)} "
                f"| first_stage={self.first_stage}")


# Structural DGPs whose x-z relation can be swapped for a nonlinear one.
_FIRST_STAGE_STRUCTS = ("tosg", "quadratic", "logistic", "expiv", "probit",
                        "sine")


def get_dgp(mode: str) -> BaseDGP:
    """Look up the DGP descriptor by mode string.

    Plain modes are listed in `_DGP_REGISTRY`.  Composite names of the form
    "<structural>-<first_stage>" (e.g. "quadratic-sin", "probit-relu") build a
    copy of the structural DGP with a nonlinear first stage.

    Raises ValueError for unknown modes.
    """
    from .data_generator import FIRST_STAGE_FUNCS
    key = str(mode).lower()
    if key in _DGP_REGISTRY:
        return _DGP_REGISTRY[key]
    if "-" in key:
        struct, first_stage = key.rsplit("-", 1)
        if first_stage in FIRST_STAGE_FUNCS and struct in _FIRST_STAGE_STRUCTS:
            return FirstStageVariant(_DGP_REGISTRY[struct], first_stage)
    raise ValueError(
        f"Unknown DGP mode: '{mode}'. Plain modes: {list(_DGP_REGISTRY.keys())}; "
        f"composite modes '<structural>-<first_stage>' with structural in "
        f"{list(_FIRST_STAGE_STRUCTS)} and first_stage in "
        f"{list(FIRST_STAGE_FUNCS)}."
    )
