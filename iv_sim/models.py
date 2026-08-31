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


class QuadraticModel(BaseModel):
    """Quadratic structural model:

    Parameterization (theta flattened):
      - first d_x entries: linear coefficient b (shape d_x)
      - remaining entries: upper-triangular entries of symmetric matrix A

    Prediction: g(theta; x) = x @ b + x^T A x  (returns shape (n,1)).
    """

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta.reshape(-1, 1)).ravel()
        d_x = x.shape[1]
        b = theta[:d_x].reshape(-1, 1)  # (d_x, 1)
        A = self._vec_to_mat(theta[d_x:], d_x)
        linear = x @ b  # (n,1)
        quad = np.sum((x @ A) * x, axis=-1, keepdims=True)  # (n,1)
        return linear + quad

    def gradient(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        d_x = x.shape[1]
        # gradient w.r.t. b is x (n, d_x)
        grad_b = x  # (n, d_x)
        # gradient w.r.t packed upper-triangular A entries
        # build per-sample outer products x_i x_j
        grad_full = np.einsum("ni,nj->nij", x, x)  # (n, d_x, d_x)
        grad_packed_list = []
        for i in range(d_x):
            for j in range(i, d_x):
                if i == j:
                    grad_packed_list.append(grad_full[:, i, j:j+1])
                else:
                    # off-diagonal contributes twice due to symmetry
                    grad_packed_list.append(2.0 * grad_full[:, i, j:j+1])
        grad_packed = np.concatenate(grad_packed_list, axis=-1)  # (n, n_packed)
        return np.concatenate([grad_b, grad_packed], axis=-1)

    def init_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        # initialize linear coeffs small, and diagonal of A ~ 1/d_x scale
        b = rng.normal(0, 0.1, size=(d_x, 1))
        # upper-triangular packed entries
        n_packed = d_x * (d_x + 1) // 2
        A_packed = np.zeros((n_packed, 1))
        return np.vstack([b, A_packed])

    def true_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        # Construct a true parameter that corresponds roughly to
        # g(x) = (x + s)^T (x + s) = x^T x + 2 s^T x + const
        # We encode A = I and b = 2*s.
        s = rng.normal(0, 1.0, size=(d_x, 1))
        b = 2.0 * s  # linear term
        A = np.eye(d_x)
        # pack upper-triangular entries
        packed = []
        for i in range(d_x):
            for j in range(i, d_x):
                packed.append(A[i, j])
        packed = np.array(packed).reshape(-1, 1)
        return np.vstack([b, packed])

    def param_dim(self, d_x: int) -> int:
        return d_x + d_x * (d_x + 1) // 2

    @staticmethod
    def _vec_to_mat(vec, d_x):
        vec = np.atleast_1d(vec.ravel())
        A = np.zeros((d_x, d_x))
        idx = 0
        for i in range(d_x):
            for j in range(i, d_x):
                A[i, j] = vec[idx]
                if i != j:
                    A[j, i] = A[i, j]
                idx += 1
        return A



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
    """Two-hidden-layer MLP with Layer Normalization.
    
    Architecture: d_x → [LN] → tanh(W1@x+b1) → [LN] → tanh(W2@a1+b2) → W3@a2+b3
    Layer Norm is applied before each activation, with learnable scale (gamma) and shift (beta).
    """

    def __init__(self, hidden_sizes: Optional[List[int]] = None):
        """
        Args:
            hidden_sizes: list of two hidden layer sizes. Default: [128, 64].
        """
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [128, 64]
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
            "gamma_ln1": (h1, 1),  # Layer norm scale for layer 1
            "beta_ln1": (h1, 1),   # Layer norm shift for layer 1
            "W2": (h2, h1),
            "b2": (h2, 1),
            "gamma_ln2": (h2, 1),  # Layer norm scale for layer 2
            "beta_ln2": (h2, 1),   # Layer norm shift for layer 2
            "W3": (1, h2),
            "b3": (1, 1),
        }

    def _unpack(self, theta: np.ndarray, d_x: int) -> dict:
        """Unflatten theta into parameter dict.

        Args:
            theta: (d_theta, 1) or (d_theta,) flat parameter vector.
            d_x: input dimension.

        Returns:
            dict with parameter names.
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
        flat_parts = [params[name].ravel() for name in 
                     ["W1", "b1", "gamma_ln1", "beta_ln1",
                      "W2", "b2", "gamma_ln2", "beta_ln2",
                      "W3", "b3"]]
        return np.concatenate(flat_parts).reshape(-1, 1)

    # ------------------------------------------------------------------
    # Layer Normalization helper
    # ------------------------------------------------------------------

    @staticmethod
    def _layer_norm(z: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Apply layer normalization: gamma * (z - mean) / (std + eps) + beta.
        
        Args:
            z: (n, d) pre-activation.
            gamma: (d, 1) scale.
            beta: (d, 1) shift.
            eps: small value for numerical stability.
            
        Returns:
            (n, d) normalized output.
        """
        mean = np.mean(z, axis=1, keepdims=True)  # (n, 1)
        var = np.var(z, axis=1, keepdims=True)    # (n, 1)
        z_norm = (z - mean) / np.sqrt(var + eps)  # (n, d)
        return z_norm * gamma.T + beta.T          # (n, d)

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
        z1 = x @ params["W1"].T + params["b1"].T           # (n, h1)
        z1_ln = self._layer_norm(z1, params["gamma_ln1"], params["beta_ln1"])  # (n, h1)
        a1 = np.tanh(z1_ln)                                # (n, h1)
        # Layer 2
        z2 = a1 @ params["W2"].T + params["b2"].T          # (n, h2)
        z2_ln = self._layer_norm(z2, params["gamma_ln2"], params["beta_ln2"])  # (n, h2)
        a2 = np.tanh(z2_ln)                                # (n, h2)
        # Output
        out = a2 @ params["W3"].T + params["b3"].T         # (n, 1)
        return out

    # ------------------------------------------------------------------
    # Gradient (manual backprop)
    # ------------------------------------------------------------------

    def gradient(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute ∇_theta g(theta; x) via backpropagation with layer norm.

        Args:
            theta: (d_theta, 1) or (d_theta,).
            x: (n, d_x).

        Returns:
            (n, d_theta) gradient of output w.r.t. all parameters.
        """
        eps = 1e-6
        n = x.shape[0]
        params = self._unpack(theta, x.shape[1])
        W1, b1, gamma_ln1, beta_ln1 = params["W1"], params["b1"], params["gamma_ln1"], params["beta_ln1"]
        W2, b2, gamma_ln2, beta_ln2 = params["W2"], params["b2"], params["gamma_ln2"], params["beta_ln2"]
        W3, b3 = params["W3"], params["b3"]

        # --- Forward pass with layer norm ---
        z1 = x @ W1.T + b1.T                       # (n, h1)
        mean1 = np.mean(z1, axis=1, keepdims=True)
        var1 = np.var(z1, axis=1, keepdims=True)
        z1_norm = (z1 - mean1) / np.sqrt(var1 + eps)
        z1_ln = z1_norm * gamma_ln1.T + beta_ln1.T
        a1 = np.tanh(z1_ln)                        # (n, h1)

        z2 = a1 @ W2.T + b2.T                      # (n, h2)
        mean2 = np.mean(z2, axis=1, keepdims=True)
        var2 = np.var(z2, axis=1, keepdims=True)
        z2_norm = (z2 - mean2) / np.sqrt(var2 + eps)
        z2_ln = z2_norm * gamma_ln2.T + beta_ln2.T
        a2 = np.tanh(z2_ln)                        # (n, h2)

        # --- Backward pass ---
        grad_b3 = np.ones((n, 1))
        grad_W3 = a2

        d_a2 = W3.T
        d_z2_ln = d_a2.T * (1.0 - a2 ** 2)

        # Gradient w.r.t. gamma_ln2, beta_ln2
        grad_gamma_ln2 = z2_norm * d_z2_ln
        grad_beta_ln2 = d_z2_ln

        # Backprop through layer norm for layer 2
        d_z2_norm = d_z2_ln * gamma_ln2.T
        std2 = np.sqrt(var2 + eps)
        d_var2 = np.sum(d_z2_norm * (z2 - mean2) * -0.5 * (std2 ** -3), axis=1, keepdims=True)
        d_mean2 = np.sum(d_z2_norm * (-1.0 / std2), axis=1, keepdims=True) + d_var2 * np.sum(-2 * (z2 - mean2), axis=1, keepdims=True) / z2.shape[1]
        d_z2 = d_z2_norm / std2 + d_var2 * 2 * (z2 - mean2) / z2.shape[1] + d_mean2 / z2.shape[1]

        grad_b2 = d_z2
        grad_W2 = d_z2[:, :, np.newaxis] * a1[:, np.newaxis, :]
        grad_W2 = grad_W2.reshape(n, -1)

        d_a1 = d_z2 @ W2
        d_z1_ln = d_a1 * (1.0 - a1 ** 2)

        # Gradient w.r.t. gamma_ln1, beta_ln1
        grad_gamma_ln1 = z1_norm * d_z1_ln
        grad_beta_ln1 = d_z1_ln

        # Backprop through layer norm for layer 1
        d_z1_norm = d_z1_ln * gamma_ln1.T
        std1 = np.sqrt(var1 + eps)
        d_var1 = np.sum(d_z1_norm * (z1 - mean1) * -0.5 * (std1 ** -3), axis=1, keepdims=True)
        d_mean1 = np.sum(d_z1_norm * (-1.0 / std1), axis=1, keepdims=True) + d_var1 * np.sum(-2 * (z1 - mean1), axis=1, keepdims=True) / z1.shape[1]
        d_z1 = d_z1_norm / std1 + d_var1 * 2 * (z1 - mean1) / z1.shape[1] + d_mean1 / z1.shape[1]

        grad_b1 = d_z1
        grad_W1 = d_z1[:, :, np.newaxis] * x[:, np.newaxis, :]
        grad_W1 = grad_W1.reshape(n, -1)

        # Concatenate all gradients
        grad = np.concatenate([grad_W1, grad_b1, grad_gamma_ln1, grad_beta_ln1,
                              grad_W2, grad_b2, grad_gamma_ln2, grad_beta_ln2,
                              grad_W3, grad_b3], axis=1)
        return grad

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        """Xavier initialization for tanh MLP with layer norm."""
        h1, h2 = self.hidden_sizes
        W1 = rng.normal(0, np.sqrt(1.0 / d_x), size=(h1, d_x))
        b1 = np.zeros((h1, 1))
        gamma_ln1 = np.ones((h1, 1))
        beta_ln1 = np.zeros((h1, 1))
        W2 = rng.normal(0, np.sqrt(1.0 / h1), size=(h2, h1))
        b2 = np.zeros((h2, 1))
        gamma_ln2 = np.ones((h2, 1))
        beta_ln2 = np.zeros((h2, 1))
        W3 = rng.normal(0, np.sqrt(1.0 / h2), size=(1, h2))
        b3 = np.zeros((1, 1))
        return self._pack({
            "W1": W1, "b1": b1, "gamma_ln1": gamma_ln1, "beta_ln1": beta_ln1,
            "W2": W2, "b2": b2, "gamma_ln2": gamma_ln2, "beta_ln2": beta_ln2,
            "W3": W3, "b3": b3
        })

    def true_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        """No true parameters for MLP -- return zeros placeholder."""
        return np.zeros((self.param_dim(d_x), 1))
