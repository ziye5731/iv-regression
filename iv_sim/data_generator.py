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
        # x = z1 + eps + gamma  (scalar)
        x = z[:, :1] + eps + gamma
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
        x1 = z1 + eps1 + gamma1
        y1 = self.h_star(x1) + eps1 + delta1
        x2 = z1 + eps2 + gamma2
        y2 = self.h_star(x2) + eps2 + delta2
        return x1, y1, x2, y2

    def generate_online(self):
        """Infinite generator yielding one (z, x, y) per step."""
        while True:
            z = self.rng.uniform(-3, 3, size=(1, 2))
            eps = self.rng.normal(0, 1, size=(1, 1))
            gamma = self.rng.normal(0, 0.1, size=(1, 1))
            delta = self.rng.normal(0, 0.1, size=(1, 1))
            x = z[:, :1] + eps + gamma
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
