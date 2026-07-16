"""
models.py -- Model definitions for IV regression.

Models:
    LinearModel:      g(theta; x) = theta^T x   (known parametric form)
    MLPModel:         g(theta; x) = MLP(x)       (unknown structural form)
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseModel(ABC):
    """Abstract base class for the structural equation g(theta; x)."""

    @abstractmethod
    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate g(theta; x)."""
        ...

    @abstractmethod
    def gradient(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute nabla_theta g(theta; x)."""
        ...

    @abstractmethod
    def init_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        """Sample random initial theta."""
        ...

    @abstractmethod
    def true_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        """Generate true theta* for data generation."""
        ...

    @abstractmethod
    def param_dim(self, d_x: int) -> int:
        """Return d_theta given input dimension."""
        ...


class LinearModel(BaseModel):
    """Linear structural equation: g(theta; x) = theta^T x."""

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        theta_2d = np.atleast_2d(theta.reshape(-1, 1))
        return x @ theta_2d

    def gradient(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        _ = theta
        return x

    def init_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        return rng.normal(0, 0.1, size=(d_x, 1))

    def true_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        return rng.normal(0, 1.0, size=(d_x, 1))

    def param_dim(self, d_x: int) -> int:
        return d_x


linear_model = LinearModel()


class LinearFirstStage:
    """First-stage model matching DGP phi function.

    - phi_func = "linear":   h(gamma; z) = z @ gamma
    - phi_func = "quadratic": h(gamma; z) = (z @ gamma)^2
    gamma has shape (d_z, d_x).
    """

    def __init__(self, phi_func: str = "linear"):
        self.phi_func = phi_func

    def predict(self, gamma: np.ndarray, z: np.ndarray) -> np.ndarray:
        s = z @ gamma
        f = self.phi_func
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
            raise ValueError(f"Unknown phi_func: {self.phi_func}")

    def gamma_update(self, z: np.ndarray, residual: np.ndarray,
                     gamma: np.ndarray) -> np.ndarray:
        # General form: gamma_update = z.T @ (phi'(s) * residual)
        f = self.phi_func
        s = z @ gamma
        if f == "linear":
            phi_prime = np.ones_like(s)
        elif f == "quadratic":
            phi_prime = 2.0 * s
        elif f == "sin":
            phi_prime = np.cos(s)
        elif f == "tanh":
            phi_prime = 1.0 - np.tanh(s) ** 2
        elif f == "relu":
            phi_prime = (s > 0).astype(float)
        elif f == "sigmoid":
            sig = 1.0 / (1.0 + np.exp(-s))
            phi_prime = sig * (1.0 - sig)
        elif f == "cubic":
            phi_prime = 3.0 * (s ** 2)
        else:
            raise ValueError(f"Unknown phi_func: {self.phi_func}")

        return z.T @ (phi_prime * residual)

    @staticmethod
    def init_params(rng: np.random.Generator, d_z: int, d_x: int,
                    scale: float = 0.1) -> np.ndarray:
        return rng.normal(0, scale, size=(d_z, d_x))


# ---------------------------------------------------------------------------
# MLP Model (for DeepGMM DGP -- unknown structural form)
# ---------------------------------------------------------------------------

class MLPModel(BaseModel):
    """Two-hidden-layer MLP: g(theta; x) = W3 @ ReLU(W2 @ ReLU(W1 @ x + b1) + b2) + b3.

    Parameters theta is a flat vector of all weights and biases.
    Architecture: d_x → hidden_sizes[0] → hidden_sizes[1] → 1.

    Gradient is computed via manual backpropagation.
    """

    def __init__(self, hidden_sizes: Optional[List[int]] = None):
        """
        Args:
            hidden_sizes: list of two hidden layer sizes. Default: [64, 32].
        """
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [64, 32]
        if len(self.hidden_sizes) != 2:
            raise ValueError("MLPModel expects exactly 2 hidden layer sizes.")

    # ------------------------------------------------------------------
    # Parameter packing / unpacking
    # ------------------------------------------------------------------

    def _shapes(self, d_x: int) -> dict:
        """Return shapes of each parameter block."""
        h1, h2 = self.hidden_sizes
        return {
            "W1": (h1, d_x),
            "b1": (h1, 1),
            "W2": (h2, h1),
            "b2": (h2, 1),
            "W3": (1, h2),
            "b3": (1, 1),
        }

    def _unpack(self, theta: np.ndarray, d_x: int) -> dict:
        """Unflatten theta into parameter dict.

        Args:
            theta: (d_theta, 1) or (d_theta,) flat parameter vector.
            d_x: input dimension.

        Returns:
            dict with keys "W1", "b1", "W2", "b2", "W3", "b3".
        """
        theta = np.atleast_1d(theta).ravel()
        shapes = self._shapes(d_x)
        idx = 0
        params = {}
        for name, shape in shapes.items():
            size = int(np.prod(shape))
            params[name] = theta[idx:idx + size].reshape(shape)
            idx += size
        return params

    def _pack(self, params: dict) -> np.ndarray:
        """Flatten parameter dict into a (d_theta, 1) vector."""
        flat_parts = [params[name].ravel() for name in ["W1", "b1", "W2", "b2", "W3", "b3"]]
        return np.concatenate(flat_parts).reshape(-1, 1)

    # ------------------------------------------------------------------
    # Parameter dimensions
    # ------------------------------------------------------------------

    def param_dim(self, d_x: int) -> int:
        shapes = self._shapes(d_x)
        return sum(int(np.prod(s)) for s in shapes.values())

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Forward pass: g(theta; x).

        Args:
            theta: (d_theta, 1) or (d_theta,).
            x: (n, d_x).

        Returns:
            (n, 1) predictions.
        """
        params = self._unpack(theta, x.shape[1])
        # Layer 1
        z1 = x @ params["W1"].T + params["b1"].T   # (n, h1)
        a1 = np.maximum(0, z1)                       # ReLU
        # Layer 2
        z2 = a1 @ params["W2"].T + params["b2"].T   # (n, h2)
        a2 = np.maximum(0, z2)                       # ReLU
        # Output
        out = a2 @ params["W3"].T + params["b3"].T  # (n, 1)
        return out

    # ------------------------------------------------------------------
    # Gradient (manual backprop)
    # ------------------------------------------------------------------

    def gradient(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute ∇_theta g(theta; x) via backpropagation.

        Args:
            theta: (d_theta, 1) or (d_theta,).
            x: (n, d_x).

        Returns:
            (n, d_theta) gradient of output w.r.t. all parameters.
        """
        n = x.shape[0]
        params = self._unpack(theta, x.shape[1])
        W1, b1 = params["W1"], params["b1"]
        W2, b2 = params["W2"], params["b2"]
        W3, b3 = params["W3"], params["b3"]

        # --- Forward pass (caching intermediates) ---
        z1 = x @ W1.T + b1.T          # (n, h1)
        a1 = np.maximum(0, z1)         # (n, h1)
        z2 = a1 @ W2.T + b2.T          # (n, h2)
        a2 = np.maximum(0, z2)         # (n, h2)
        # output: a2 @ W3.T + b3.T

        # --- Backward pass (per-sample) ---
        # d_out/d_b3 = 1
        grad_b3 = np.ones((n, 1))                              # (n, 1)

        # d_out/d_W3 = a2  (outer: for each sample, grad is a2^T)
        grad_W3 = a2                                            # (n, h2)

        # d_out/d_a2 = W3^T
        d_a2 = W3.T                                             # (h2, 1)
        # d_out/d_z2 = d_a2 ⊙ ReLU'(z2)
        d_z2 = d_a2.T * (z2 > 0).astype(float)                 # (n, h2)

        # d_out/d_b2 = d_z2
        grad_b2 = d_z2                                          # (n, h2)

        # d_out/d_W2: for each sample, outer(d_z2_row, a1_row)
        grad_W2 = d_z2[:, :, np.newaxis] * a1[:, np.newaxis, :]  # (n, h2, h1)
        grad_W2 = grad_W2.reshape(n, -1)                         # (n, h2*h1)

        # d_out/d_a1 = d_z2 @ W2
        d_a1 = d_z2 @ W2                                        # (n, h1)
        # d_out/d_z1 = d_a1 ⊙ ReLU'(z1)
        d_z1 = d_a1 * (z1 > 0).astype(float)                    # (n, h1)

        # d_out/d_b1 = d_z1
        grad_b1 = d_z1                                          # (n, h1)

        # d_out/d_W1: for each sample, outer(d_z1_row, x_row)
        grad_W1 = d_z1[:, :, np.newaxis] * x[:, np.newaxis, :]  # (n, h1, d_x)
        grad_W1 = grad_W1.reshape(n, -1)                         # (n, h1*d_x)

        # Concatenate all gradients in canonical order
        grad = np.concatenate([grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3], axis=1)
        return grad  # (n, d_theta)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        """He (Kaiming) initialization for ReLU MLP."""
        h1, h2 = self.hidden_sizes
        W1 = rng.normal(0, np.sqrt(2.0 / d_x), size=(h1, d_x))
        b1 = np.zeros((h1, 1))
        W2 = rng.normal(0, np.sqrt(2.0 / h1), size=(h2, h1))
        b2 = np.zeros((h2, 1))
        W3 = rng.normal(0, np.sqrt(2.0 / h2), size=(1, h2))
        b3 = np.zeros((1, 1))
        return self._pack({"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3})

    def true_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        """No true parameters for MLP -- return zeros placeholder."""
        return np.zeros((self.param_dim(d_x), 1))
