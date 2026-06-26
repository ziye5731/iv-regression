from __future__ import annotations

"""
data_generator.py -- Data generation for IV regression simulation.

Data generating process (model-agnostic):
    y = g(theta*; x) + c + eps_y
    x = gamma*^T z   + c + eps_x

Noise distributions:
    z   ~ N(0, sigma_z^2 I)
    c   ~ N(0, sigma_c^2)
    eps_y ~ N(0, sigma_y^2)
    eps_x ~ N(0, sigma_x^2 I)

The confounding term c creates endogeneity: x is correlated
with the composite error (c + eps_y).  The instrument z is
uncorrelated with both c and the noises, enabling identification.
"""

import numpy as np
from numpy.random import Generator

from .config import SimulationConfig


class IVDataGenerator:
    """IV regression data generator.

    Generates (z, x, y) triples following the DGP above.
    Supports batch generation, online streaming, and conditionally-
    independent paired sampling (needed by TOSG-IVaR).

    Attributes:
        config: SimulationConfig with model, noise, and dimension specs.
        rng: NumPy random generator.
    """

    def __init__(self, config: SimulationConfig, seed: int | None = None):
        """
        Args:
            config: simulation configuration.
            seed: independent seed for this generator (uses config.seed if None).
        """
        self.config = config
        actual_seed = seed if seed is not None else config.seed
        self.rng: Generator = np.random.default_rng(actual_seed)

        # True parameters
        self.theta_star = config.theta_star   # (d_theta, 1)
        self.gamma_star = config.gamma_star   # (d_z, d_x)
        self.model = config.model             # structural model g

    # ------------------------------------------------------------------
    # Low-level sampling
    # ------------------------------------------------------------------

    def _sample_z(self, n: int) -> np.ndarray:
        """Sample instruments z ~ N(0, sigma_z^2 I).  Shape: (n, d_z)."""
        return self.rng.normal(
            0, self.config.sigma_z, size=(n, self.config.d_z)
        )

    def _sample_c(self, n: int) -> np.ndarray:
        """Sample confounder c ~ N(0, sigma_c^2).  Shape: (n, 1)."""
        return self.rng.normal(
            0, self.config.sigma_c, size=(n, 1)
        )

    def _sample_noise_y(self, n: int) -> np.ndarray:
        """Sample true noise eps_y ~ N(0, sigma_y^2).  Shape: (n, 1)."""
        return self.rng.normal(
            0, self.config.sigma_y, size=(n, 1)
        )

    def _sample_noise_x(self, n: int) -> np.ndarray:
        """Sample true noise eps_x ~ N(0, sigma_x^2 I).  Shape: (n, d_x)."""
        return self.rng.normal(
            0, self.config.sigma_x, size=(n, self.config.d_x)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_batch(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a batch of n samples (z, x, y).

        Returns:
            z: (n, d_z) instruments
            x: (n, d_x) covariates
            y: (n, 1)   response
        """
        z = self._sample_z(n)
        c = self._sample_c(n)
        noise_y = self._sample_noise_y(n)
        noise_x = self._sample_noise_x(n)

        # x = gamma*^T z + c + eps_x
        x = z @ self.gamma_star + c + noise_x

        # y = g(theta*; x) + c + eps_y
        y = self.model.predict(self.theta_star, x) + c + noise_y

        return z, x, y

    def generate_pair(
        self, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Given z, generate two conditionally independent (x, y) pairs.

        This matches the TOSG-IVaR requirement: for a single z,
        independently sample two (x, y) pairs (c and noises are
        drawn independently).

        Args:
            z: instrument, shape (1, d_z) or (d_z,).

        Returns:
            x1, y1, x2, y2: two conditionally independent observations.
        """
        z = np.atleast_2d(z)  # ensure (1, d_z)
        n = z.shape[0]

        c1 = self._sample_c(n)
        c2 = self._sample_c(n)
        ny1 = self._sample_noise_y(n)
        ny2 = self._sample_noise_y(n)
        nx1 = self._sample_noise_x(n)
        nx2 = self._sample_noise_x(n)

        x1 = z @ self.gamma_star + c1 + nx1
        y1 = self.model.predict(self.theta_star, x1) + c1 + ny1

        x2 = z @ self.gamma_star + c2 + nx2
        y2 = self.model.predict(self.theta_star, x2) + c2 + ny2

        return x1, y1, x2, y2

    def generate_online(self):
        """Online generator: yield one sample at a time.

        Yields:
            (z, x, y): each of shape (1, d).
        """
        while True:
            z = self._sample_z(1)      # (1, d_z)
            c = self._sample_c(1)      # (1, 1)
            ny = self._sample_noise_y(1)
            nx = self._sample_noise_x(1)
            x = z @ self.gamma_star + c + nx
            y = self.model.predict(self.theta_star, x) + c + ny
            yield z, x, y

    def reset_seed(self, seed: int):
        """Reset the random seed."""
        self.rng = np.random.default_rng(seed)
