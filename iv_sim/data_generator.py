from __future__ import annotations

"""
data_generator.py -- Data generation for IV regression.

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
    eps   ~ N(0, 1),  gamma, delta ~ N(0, 0.1)
    x     = z1 + eps + gamma
    y     = h*(x) + eps + delta
    h* in {step, abs, linear, sin}
"""

import numpy as np
from numpy.random import Generator
from typing import Callable

from .config import SimulationConfig
from .models import linear_model


class IVDataGenerator:
    """IV regression data generator following the TOSG paper DGP."""

    def __init__(self, config: SimulationConfig, seed: int | None = None):
        self.config = config
        actual_seed = seed if seed is not None else config.seed
        self.rng: Generator = np.random.default_rng(actual_seed)
        self.theta_star = config.theta_star
        self.gamma_star = config.gamma_star
        self.model = linear_model

    def _phi(self, s: np.ndarray) -> np.ndarray:
        f = self.config.phi_func
        if f == "linear":
            return s
        elif f == "quadratic":
            return s ** 2
        elif f == "sin":
            return np.sin(s)
        elif f == "tanh":
            return np.tanh(s)
        elif f == "relu":
            return np.maximum(0, s)
        elif f == "sigmoid":
            return 1.0 / (1.0 + np.exp(-s))
        elif f == "cubic":
            return s ** 3
        else:
            raise ValueError(f"Unknown phi_func: {self.config.phi_func}")

    def generate_batch(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d_z, d_x, c = self.config.d_z, self.config.d_x, self.config.noise_c
        z = self.rng.normal(0, 1, size=(n, d_z))
        h = self.rng.normal(1, 1, size=(n, d_x))
        ex = self.rng.normal(0, 1, size=(n, d_x))
        ey = self.rng.normal(0, 1, size=(n, 1))
        x = self._phi(z @ self.gamma_star) + c * (h + ex)
        y = x @ self.theta_star + c * (h[:, :1] + ey)
        return z, x, y

    def generate_pair(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = np.atleast_2d(z)
        n, d_x, c = z.shape[0], self.config.d_x, self.config.noise_c
        h1, h2 = self.rng.normal(1, 1, size=(n, d_x)), self.rng.normal(1, 1, size=(n, d_x))
        ex1, ex2 = self.rng.normal(0, 1, size=(n, d_x)), self.rng.normal(0, 1, size=(n, d_x))
        ey1, ey2 = self.rng.normal(0, 1, size=(n, 1)), self.rng.normal(0, 1, size=(n, 1))
        x1 = self._phi(z @ self.gamma_star) + c * (h1 + ex1)
        y1 = x1 @ self.theta_star + c * (h1[:, :1] + ey1)
        x2 = self._phi(z @ self.gamma_star) + c * (h2 + ex2)
        y2 = x2 @ self.theta_star + c * (h2[:, :1] + ey2)
        return x1, y1, x2, y2

    def generate_online(self):
        d_z, d_x, c = self.config.d_z, self.config.d_x, self.config.noise_c
        while True:
            z = self.rng.normal(0, 1, size=(1, d_z))
            h = self.rng.normal(1, 1, size=(1, d_x))
            ex = self.rng.normal(0, 1, size=(1, d_x))
            ey = self.rng.normal(0, 1, size=(1, 1))
            x = self._phi(z @ self.gamma_star) + c * (h + ex)
            y = x @ self.theta_star + c * (h[:, :1] + ey)
            yield z, x, y

    def reset_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# OTSG Data Generator
# ---------------------------------------------------------------------------

class OTSGDataGenerator:
    """IV regression data generator following the OTSG paper DGP.

    DGP:
        eps   ~ N(0, sigma_eps^2 I_dx)
        nu    ~ N(rho * eps_1, 0.25)
        x     = gamma*^T z + eps
        y     = theta*^T x + nu

    where eps_1 is the first coordinate of eps.
    """

    def __init__(self, config: SimulationConfig, seed: int | None = None):
        self.config = config
        actual_seed = seed if seed is not None else config.seed
        self.rng: Generator = np.random.default_rng(actual_seed)
        self.theta_star = config.theta_star
        self.gamma_star = config.gamma_star
        self.model = linear_model
        self.sigma_eps = getattr(config, "otsg_sigma_eps", 0.5)
        self.rho = getattr(config, "otsg_rho", 1.0)

    def generate_batch(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d_z, d_x = self.config.d_z, self.config.d_x
        z = self.rng.normal(0, 1, size=(n, d_z))
        eps = self.rng.normal(0, self.sigma_eps, size=(n, d_x))
        nu = self.rng.normal(self.rho * eps[:, :1], 0.5)  # std=0.5 → var=0.25
        x = z @ self.gamma_star + eps
        y = x @ self.theta_star + nu
        return z, x, y

    def generate_pair(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate two conditionally independent (x,y) pairs given z."""
        z = np.atleast_2d(z)
        n, d_x = z.shape[0], self.config.d_x
        eps1 = self.rng.normal(0, self.sigma_eps, size=(n, d_x))
        eps2 = self.rng.normal(0, self.sigma_eps, size=(n, d_x))
        nu1 = self.rng.normal(self.rho * eps1[:, :1], 0.5)
        nu2 = self.rng.normal(self.rho * eps2[:, :1], 0.5)
        x1 = z @ self.gamma_star + eps1
        y1 = x1 @ self.theta_star + nu1
        x2 = z @ self.gamma_star + eps2
        y2 = x2 @ self.theta_star + nu2
        return x1, y1, x2, y2

    def generate_online(self):
        """Infinite generator yielding one (z, x, y) per step."""
        d_z, d_x = self.config.d_z, self.config.d_x
        while True:
            z = self.rng.normal(0, 1, size=(1, d_z))
            eps = self.rng.normal(0, self.sigma_eps, size=(1, d_x))
            nu = self.rng.normal(self.rho * eps[:, :1], 0.5)
            x = z @ self.gamma_star + eps
            y = x @ self.theta_star + nu
            yield z, x, y

    def reset_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# DeepGMM Data Generator
# ---------------------------------------------------------------------------

def _make_h_star(h_type: str) -> Callable[[np.ndarray], np.ndarray]:
    """Build the h*(x) function for DeepGMM DGP."""
    if h_type == "step":
        return lambda x: (x > 0).astype(float)
    elif h_type == "abs":
        return lambda x: np.abs(x)
    elif h_type == "linear":
        return lambda x: x
    elif h_type == "sin":
        return lambda x: np.sin(x)
    else:
        raise ValueError(f"Unknown h_star type: {h_type}")


class DeepGMMDataGenerator:
    """IV regression data generator following the DeepGMM DGP.

    DGP:
        z     = (z1, z2) ~ Unif([-3, 3]^2)   → d_z = 2
        eps   ~ N(0, 1),  gamma, delta ~ N(0, 0.1)
        x     = z1 + eps + gamma                → d_x = 1 (scalar)
        y     = h*(x) + eps + delta

    h* is one of: step, abs, linear, sin.
    The model g(theta; x) is an MLP (unknown structural form).
    """

    def __init__(self, config: SimulationConfig, seed: int | None = None):
        self.config = config
        actual_seed = seed if seed is not None else config.seed
        self.rng: Generator = np.random.default_rng(actual_seed)
        h_type = getattr(config, "deepgmm_h_star", "abs")
        self.h_star = _make_h_star(h_type)
        self.iv_strength = getattr(config, "deepgmm_iv_strength", 1.0)
        # For DeepGMM, model is set externally (MLP), not linear_model
        self.model = None

    def set_model(self, model):
        """Set the structural model (MLP) after construction."""
        self.model = model

    def generate_batch(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate batch of (z, x, y)."""
        # z ~ Unif([-3, 3]^2)
        z = self.rng.uniform(-3, 3, size=(n, 2))
        eps = self.rng.normal(0, 1, size=(n, 1))
        gamma = self.rng.normal(0, 0.1, size=(n, 1))
        delta = self.rng.normal(0, 0.1, size=(n, 1))
        # x = iv_strength * z1 + eps + gamma  (scalar)
        x = self.iv_strength * z[:, :1] + eps + gamma
        # y = h*(x) + eps + delta
        y = self.h_star(x) + eps + delta
        return z, x, y

    def generate_pair(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate two conditionally independent (x,y) pairs given z.

        Both pairs share the same z1 instrument but have independent noise.
        """
        z = np.atleast_2d(z)
        n = z.shape[0]
        z1 = z[:, :1]  # (n, 1)
        eps1 = self.rng.normal(0, 1, size=(n, 1))
        eps2 = self.rng.normal(0, 1, size=(n, 1))
        gamma1 = self.rng.normal(0, 0.1, size=(n, 1))
        gamma2 = self.rng.normal(0, 0.1, size=(n, 1))
        delta1 = self.rng.normal(0, 0.1, size=(n, 1))
        delta2 = self.rng.normal(0, 0.1, size=(n, 1))
        x1 = self.iv_strength * z1 + eps1 + gamma1
        y1 = self.h_star(x1) + eps1 + delta1
        x2 = self.iv_strength * z1 + eps2 + gamma2
        y2 = self.h_star(x2) + eps2 + delta2
        return x1, y1, x2, y2

    def generate_online(self):
        """Infinite generator yielding one (z, x, y) per step."""
        while True:
            z = self.rng.uniform(-3, 3, size=(1, 2))
            eps = self.rng.normal(0, 1, size=(1, 1))
            gamma = self.rng.normal(0, 0.1, size=(1, 1))
            delta = self.rng.normal(0, 0.1, size=(1, 1))
            x = self.iv_strength * z[:, :1] + eps + gamma
            y = self.h_star(x) + eps + delta
            yield z, x, y

    def reset_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_data_generator(config: SimulationConfig, seed: int | None = None):
    """Create the appropriate data generator for the active DGP.

    Delegates to the DGP descriptor so there is no per-mode branching here.
    """
    from .dgp import get_dgp
    return get_dgp(config.dgp_mode).create_generator(config, seed=seed)


# ---------------------------------------------------------------------------
# First-stage nonlinearity  x = phi(gamma*^T z) + c + eps_x
# ---------------------------------------------------------------------------

FIRST_STAGE_FUNCS = ("linear", "quadratic", "sin", "tanh", "relu",
                     "sigmoid", "cubic")


def apply_first_stage(s: np.ndarray, first_stage: str) -> np.ndarray:
    """Apply the first-stage nonlinearity phi elementwise.

    Args:
        s: linear index gamma*^T z, shape (n, d_x).
        first_stage: one of FIRST_STAGE_FUNCS.

    Returns:
        phi(s), same shape as s.
    """
    if first_stage == "linear":
        return s
    if first_stage == "quadratic":
        return s ** 2
    if first_stage == "sin":
        return np.sin(s)
    if first_stage == "tanh":
        return np.tanh(s)
    if first_stage == "relu":
        return np.maximum(0.0, s)
    if first_stage == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(s, -500.0, 500.0)))
    if first_stage == "cubic":
        return s ** 3
    raise ValueError(
        f"Unknown first stage '{first_stage}'. Available: {list(FIRST_STAGE_FUNCS)}")


class QuadraticDataGenerator:
    """Data generator for the Quadratic DGP described in README.

    DGP:
        eps_x ~ N(0, (1-rho) I_dx),
        z ~ N(0, I_dz),
        c ~ N(0, rho I_dx)
        x = gamma*^T z + c + eps_x
        y = g(theta*; x) + (1/sqrt(d_x)) * 1^T c + eps_y

    Here g(theta; x) is implemented by `QuadraticModel` in `iv_sim.models`.
    """

    def __init__(self, config: SimulationConfig, seed: int | None = None):
        self.config = config
        actual_seed = seed if seed is not None else config.seed
        self.rng: Generator = np.random.default_rng(actual_seed)
        self.theta_star = config.theta_star
        self.gamma_star = config.gamma_star
        from .models import QuadraticModel
        self.model = QuadraticModel()
        self.rho = getattr(config, "quadratic_rho", 0.5)
        # Structural-noise / endogeneity knobs (defaults reproduce the README DGP).
        self.noise_eps_y = getattr(config, "noise_eps_y", 1.0)
        self.c_coef = getattr(config, "c_coef", 1.0)
        # First-stage map: x = phi(gamma*^T z) + c + eps_x ("linear" by default;
        # set by composite DGP modes such as "quadratic-sin").
        self.first_stage = getattr(config, "first_stage", "linear")
        # Normalize the aggregate confounder:  Var(1^T c / sqrt(d_x)) = rho is
        # then independent of d_x, so the endogeneity strength does not grow
        # with the dimension (and is comparable across DGPs).
        self.c_scale = self.c_coef / np.sqrt(self.config.d_x)

    def generate_batch(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d_z, d_x = self.config.d_z, self.config.d_x
        z = self.rng.normal(0, 1, size=(n, d_z))
        eps_x = self.rng.normal(0, np.sqrt(1.0 - self.rho), size=(n, d_x))
        c = self.rng.normal(0, np.sqrt(self.rho), size=(n, d_x))
        eps_y = self.rng.normal(0, self.noise_eps_y, size=(n, 1))
        x = apply_first_stage(z @ self.gamma_star, self.first_stage) + c + eps_x
        # g(theta; x) is produced by the QuadraticModel
        y = (self.model.predict(self.theta_star, x)
             + self.c_scale * np.sum(c, axis=1, keepdims=True) + eps_y)
        return z, x, y

    def generate_pair(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = np.atleast_2d(z)
        n = z.shape[0]
        eps_x1 = self.rng.normal(0, np.sqrt(1.0 - self.rho), size=(n, self.config.d_x))
        eps_x2 = self.rng.normal(0, np.sqrt(1.0 - self.rho), size=(n, self.config.d_x))
        c1 = self.rng.normal(0, np.sqrt(self.rho), size=(n, self.config.d_x))
        c2 = self.rng.normal(0, np.sqrt(self.rho), size=(n, self.config.d_x))
        eps_y1 = self.rng.normal(0, self.noise_eps_y, size=(n, 1))
        eps_y2 = self.rng.normal(0, self.noise_eps_y, size=(n, 1))
        x1 = apply_first_stage(z @ self.gamma_star, self.first_stage) + c1 + eps_x1
        x2 = apply_first_stage(z @ self.gamma_star, self.first_stage) + c2 + eps_x2
        y1 = (self.model.predict(self.theta_star, x1)
              + self.c_scale * np.sum(c1, axis=1, keepdims=True) + eps_y1)
        y2 = (self.model.predict(self.theta_star, x2)
              + self.c_scale * np.sum(c2, axis=1, keepdims=True) + eps_y2)
        return x1, y1, x2, y2

    def generate_online(self):
        while True:
            z = self.rng.normal(0, 1, size=(1, self.config.d_z))
            eps_x = self.rng.normal(0, np.sqrt(1.0 - self.rho), size=(1, self.config.d_x))
            c = self.rng.normal(0, np.sqrt(self.rho), size=(1, self.config.d_x))
            eps_y = self.rng.normal(0, self.noise_eps_y, size=(1, 1))
            x = apply_first_stage(z @ self.gamma_star, self.first_stage) + c + eps_x
            y = (self.model.predict(self.theta_star, x)
                 + self.c_scale * np.sum(c, axis=1, keepdims=True) + eps_y)
            yield z, x, y

    def reset_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)


class LogisticDataGenerator(QuadraticDataGenerator):
    """Data generator for the Logistic DGP described in README.

    DGP:
        eps_x ~ N(0, (1-rho) I_dx),
        z ~ N(0, I_dz),
        c ~ N(0, rho I_dx)
        x = gamma*^T z + c + eps_x
        y = g(theta*; x) + (1/sqrt(d_x)) * 1^T c + eps_y

    where g(theta; x) = sigmoid(theta^T x) is implemented by
    `LogisticModel` in `iv_sim.models`.  The sampling logic is identical to
    the Quadratic DGP (same first stage / confounder / noise structure), so it
    is inherited from QuadraticDataGenerator; only the structural model and the
    rho config attribute differ.
    """

    def __init__(self, config: SimulationConfig, seed: int | None = None):
        super().__init__(config, seed=seed)
        from .models import LogisticModel
        self.model = LogisticModel()
        self.rho = getattr(config, "logistic_rho", 0.5)


# ---------------------------------------------------------------------------
# ExpIV / Probit / Sine data generators
#
# All three reuse the Quadratic first-stage / confounder / noise structure:
#     x = gamma*^T z + c + eps_x
#     y = g(theta*; x) + c_coef * (1/sqrt(d_x)) 1^T c + eps_y
# Only the structural model g(theta; x) differs, so the sampling logic is
# inherited from QuadraticDataGenerator.
# ---------------------------------------------------------------------------

class ExponentialDataGenerator(QuadraticDataGenerator):
    """Data generator for the ExpIV (exponential / log-link) DGP.

        y = exp(theta*^T x) + c_coef * (1/sqrt(d_x)) 1^T c + eps_y
    """

    def __init__(self, config: SimulationConfig, seed: int | None = None):
        super().__init__(config, seed=seed)
        from .models import ExponentialModel
        self.model = ExponentialModel()
        self.rho = getattr(config, "expiv_rho", 0.5)


class ProbitDataGenerator(QuadraticDataGenerator):
    """Data generator for the Probit (probit-link) DGP.

        y = Phi(theta*^T x) + c_coef * (1/sqrt(d_x)) 1^T c + eps_y

    The outcome is continuous (regression form), see the model docstring.
    """

    def __init__(self, config: SimulationConfig, seed: int | None = None):
        super().__init__(config, seed=seed)
        from .models import ProbitModel
        self.model = ProbitModel()
        self.rho = getattr(config, "probit_rho", 0.5)


class SineDataGenerator(QuadraticDataGenerator):
    """Data generator for the Sine (periodic, non-convex) DGP.

        y = theta*_0 sin(x_1 + theta*_1) + sum_{j>=2} theta*_j x_j
            + theta*_int + c_coef * (1/sqrt(d_x)) 1^T c + eps_y
    """

    def __init__(self, config: SimulationConfig, seed: int | None = None):
        super().__init__(config, seed=seed)
        from .models import SineModel
        self.model = SineModel()
        self.rho = getattr(config, "sine_rho", 0.5)

