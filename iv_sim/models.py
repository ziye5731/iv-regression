"""
models.py -- Model definitions for IV regression.

Defines the structural equation g(theta; x) and its gradient,
as well as the first-stage mapping h(gamma; z).

Models follow a common interface so algorithms work generically
with any model (linear or nonlinear).

Supported models:
    - LinearModel:   g(theta; x) = theta^T x
    - QuadraticModel: g(theta; x) = x^T A x + b^T x
    - PolyModel:      g(theta; x) = sum of monomials up to given degree
"""

import numpy as np
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement
from math import comb
from typing import Optional


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    """Abstract base class for the structural equation g(theta; x).

    Subclasses must implement:
        - predict(theta, x):  evaluate g at given theta, x
        - gradient(theta, x): compute nabla_theta g(theta; x)
        - init_params(rng, d_x): sample a random initial theta
        - true_params(rng, d_x): generate a "true" theta* for data generation
        - param_dim(d_x): return number of parameters given input dimension
    """

    @abstractmethod
    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate g(theta; x).

        Args:
            theta: parameter vector of shape (d_theta,) or (d_theta, 1).
            x: input of shape (..., d_x).

        Returns:
            predicted values of shape (..., 1).
        """
        ...

    @abstractmethod
    def gradient(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute nabla_theta g(theta; x).

        Args:
            theta: parameter vector.
            x: input of shape (n, d_x).

        Returns:
            gradient matrix of shape (n, d_theta).
        """
        ...

    @abstractmethod
    def init_params(
        self, rng: np.random.Generator, d_x: int
    ) -> np.ndarray:
        """Sample a random parameter vector for training initialization.

        Args:
            rng: NumPy random generator.
            d_x: input dimension.

        Returns:
            parameter vector of shape (d_theta, 1).
        """
        ...

    @abstractmethod
    def true_params(
        self, rng: np.random.Generator, d_x: int
    ) -> np.ndarray:
        """Generate a "true" parameter vector theta*.

        Used by the data generator to define the ground-truth model.

        Args:
            rng: NumPy random generator.
            d_x: input dimension.

        Returns:
            theta* of shape (d_theta, 1).
        """
        ...

    @abstractmethod
    def param_dim(self, d_x: int) -> int:
        """Return the number of parameters for a given input dimension.

        Args:
            d_x: input dimension.

        Returns:
            d_theta.
        """
        ...


# ---------------------------------------------------------------------------
# Linear model
# ---------------------------------------------------------------------------

class LinearModel(BaseModel):
    """Linear structural equation: g(theta; x) = theta^T x."""

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        theta_2d = np.atleast_2d(theta.reshape(-1, 1))  # (d, 1)
        return x @ theta_2d

    def gradient(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        # Linear model: gradient = x (independent of theta)
        _ = theta
        return x

    def init_params(
        self, rng: np.random.Generator, d_x: int
    ) -> np.ndarray:
        return rng.normal(0, 0.1, size=(d_x, 1))

    def true_params(
        self, rng: np.random.Generator, d_x: int
    ) -> np.ndarray:
        return rng.normal(0, 1.0, size=(d_x, 1))

    def param_dim(self, d_x: int) -> int:
        return d_x


# ---------------------------------------------------------------------------
# Quadratic model
# ---------------------------------------------------------------------------

class QuadraticModel(BaseModel):
    """Quadratic structural equation.

        g(theta; x) = b^T x + x^T A x

    where theta = [b; vec_upper(A)] with A symmetric.
    Total parameters: d_x + d_x*(d_x+1)/2.
    """

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        d_x = x.shape[-1]
        b = theta[:d_x].reshape(-1, 1)            # (d_x, 1)
        A = self._vec_to_mat(theta[d_x:], d_x)     # (d_x, d_x)

        linear = x @ b                              # (n, 1)
        quad = np.sum((x @ A) * x, axis=-1, keepdims=True)  # (n, 1)
        return linear + quad

    def gradient(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        d_x = x.shape[-1]
        n = x.shape[0]

        # grad w.r.t b: x
        grad_b = x  # (n, d_x)

        # grad w.r.t packed A:
        # For symmetric A, d/dA_{ij} of x^T A x = x_i * x_j (i==j)
        # or 2*x_i*x_j (i!=j). We pack upper triangle.
        grad_full = np.einsum("ni,nj->nij", x, x)  # (n, d_x, d_x)

        grad_packed_list = []
        for i in range(d_x):
            for j in range(i, d_x):
                if i == j:
                    grad_packed_list.append(grad_full[:, i, j:j+1])
                else:
                    grad_packed_list.append(2.0 * grad_full[:, i, j:j+1])
        grad_packed = np.concatenate(grad_packed_list, axis=-1)  # (n, n_A)

        return np.concatenate([grad_b, grad_packed], axis=-1)  # (n, d_theta)

    def init_params(
        self, rng: np.random.Generator, d_x: int
    ) -> np.ndarray:
        d = self.param_dim(d_x)
        return rng.normal(0, 0.1, size=(d, 1))

    def true_params(
        self, rng: np.random.Generator, d_x: int
    ) -> np.ndarray:
        """Generate a sparse-ish true parameter."""
        d = self.param_dim(d_x)
        theta = np.zeros((d, 1))
        # Linear part
        theta[:d_x, 0] = rng.normal(0, 1.0, size=d_x)
        # Quadratic part: only a few non-zero entries
        n_quad = d - d_x
        if n_quad > 0:
            nz = max(1, n_quad // 3)
            idx = rng.choice(n_quad, size=nz, replace=False)
            theta[d_x + idx, 0] = rng.normal(0, 0.5, size=nz)
        return theta

    def param_dim(self, d_x: int) -> int:
        return d_x + d_x * (d_x + 1) // 2

    @staticmethod
    def _vec_to_mat(vec: np.ndarray, d_x: int) -> np.ndarray:
        """Unpack upper-triangle vector into symmetric matrix."""
        vec = np.atleast_1d(vec.ravel())  # ensure 1-D flat
        A = np.zeros((d_x, d_x))
        idx = 0
        for i in range(d_x):
            for j in range(i, d_x):
                A[i, j] = float(vec[idx])
                if i != j:
                    A[j, i] = A[i, j]
                idx += 1
        return A


# ---------------------------------------------------------------------------
# Neural network (single hidden layer)
# ---------------------------------------------------------------------------

class NeuralNetModel(BaseModel):
    """Single-hidden-layer neural network with tanh activation.

        g(theta; x) = w^T tanh(W x + b) + c

    where theta = [W_flat; b; w; c].
    Hidden dim H = d_x (same as input for simplicity).
    Total params: d_x*H + H + H + 1.
    """

    def __init__(self, hidden_dim: Optional[int] = None):
        self._hidden_dim = hidden_dim

    def _H(self, d_x: int) -> int:
        return self._hidden_dim if self._hidden_dim is not None else max(2, d_x)

    def _unpack(self, theta: np.ndarray, d_x: int):
        H = self._H(d_x)
        th = theta.ravel()
        W_flat = th[:d_x * H].reshape(H, d_x)       # (H, d_x)
        b_vec = th[d_x * H : d_x * H + H].reshape(H, 1)  # (H, 1)
        w_vec = th[d_x * H + H : d_x * H + 2*H].reshape(H, 1)  # (H, 1)
        c_val = th[-1]                                # scalar
        return W_flat, b_vec, w_vec, c_val

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        d_x = x.shape[-1]
        W, b, w, c = self._unpack(theta, d_x)
        hidden = np.tanh(x @ W.T + b.T)  # (n, H)
        return hidden @ w + c            # (n, 1)

    def gradient(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        d_x = x.shape[-1]
        H = self._H(d_x)
        n = x.shape[0]
        W, b, w, c = self._unpack(theta, d_x)

        z = x @ W.T + b.T                     # (n, H)
        tanh_z = np.tanh(z)                    # (n, H)
        dtanh = 1.0 - tanh_z ** 2              # (n, H)
        delta = dtanh * w.T                    # (n, H) weighted by w

        # Grad w.r.t W_ij: delta[:,j] * x[:,i]
        grad_W = np.einsum("nh,ni->nhi", delta, x).reshape(n, -1)  # (n, H*d_x)
        # Grad w.r.t b_j: delta
        grad_b = delta                                          # (n, H)
        # Grad w.r.t w_j: tanh_z
        grad_w = tanh_z                                         # (n, H)
        # Grad w.r.t c: 1
        grad_c = np.ones((n, 1))                                # (n, 1)

        return np.concatenate([grad_W, grad_b, grad_w, grad_c], axis=-1)

    def init_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        H = self._H(d_x)
        d = self.param_dim(d_x)
        return rng.normal(0, 0.1, size=(d, 1))

    def true_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        H = self._H(d_x)
        d = self.param_dim(d_x)
        # Xavier-like init scaled to reasonable magnitude
        theta = np.zeros((d, 1))
        # W: d_x*H
        s = np.sqrt(2.0 / (d_x + H))
        theta[:d_x * H, 0] = rng.normal(0, s, size=d_x * H)
        # b: H
        theta[d_x * H:d_x * H + H, 0] = rng.normal(0, 0.5, size=H)
        # w: H
        theta[d_x * H + H:d_x * H + 2*H, 0] = rng.normal(0, 1.0 / np.sqrt(H), size=H)
        # c
        theta[-1, 0] = rng.normal(0, 0.5)
        return theta

    def param_dim(self, d_x: int) -> int:
        H = self._H(d_x)
        return d_x * H + 2 * H + 1


# ---------------------------------------------------------------------------
# Sinusoidal model
# ---------------------------------------------------------------------------

class SinusoidalModel(BaseModel):
    """Sum of sinusoidal components.

        g(theta; x) = Σ_{k=1}^K a_k * sin(b_k^T x + c_k)

    where K = min(5, d_x+1).  Each component has d_x + 2 params.
    Total: K * (d_x + 2).
    """

    def _K(self, d_x: int) -> int:
        return min(5, d_x + 1)

    def _unpack(self, theta: np.ndarray, d_x: int):
        K = self._K(d_x)
        th = theta.ravel()
        # params per component: a_k (scalar), b_k (d_x,), c_k (scalar)
        per = d_x + 2
        a_list = th[0::per]                    # K scalars
        b_mat = np.zeros((K, d_x))
        c_list = np.zeros(K)
        for k in range(K):
            b_mat[k] = th[k * per + 1 : k * per + 1 + d_x]
            c_list[k] = th[k * per + 1 + d_x]
        return a_list, b_mat, c_list

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        d_x = x.shape[-1]
        K = self._K(d_x)
        a_list, b_mat, c_list = self._unpack(theta, d_x)
        result = np.zeros((x.shape[0], 1))
        for k in range(K):
            result += a_list[k] * np.sin(x @ b_mat[k:k+1].T + c_list[k])
        return result

    def gradient(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        d_x = x.shape[-1]
        K = self._K(d_x)
        n = x.shape[0]
        a_list, b_mat, c_list = self._unpack(theta, d_x)
        grads = []
        for k in range(K):
            arg = (x @ b_mat[k:k+1].T + c_list[k]).ravel()  # (n,)
            cos_arg = np.cos(arg).reshape(-1, 1)             # (n, 1)
            sin_arg = np.sin(arg).reshape(-1, 1)             # (n, 1)
            # d/da_k: sin(arg)
            grad_ak = sin_arg                                # (n, 1)
            # d/db_k: a_k * cos(arg) * x
            grad_bk = a_list[k] * cos_arg * x                 # (n, d_x)
            # d/dc_k: a_k * cos(arg)
            grad_ck = a_list[k] * cos_arg                     # (n, 1)
            grads.append(np.concatenate([grad_ak, grad_bk, grad_ck], axis=-1))
        return np.concatenate(grads, axis=-1)                 # (n, K*(d_x+2))

    def init_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        d = self.param_dim(d_x)
        return rng.normal(0, 0.1, size=(d, 1))

    def true_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        K = self._K(d_x)
        d = self.param_dim(d_x)
        theta = np.zeros((d, 1))
        per = d_x + 2
        for k in range(K):
            theta[k * per, 0] = rng.normal(0, 1.0)           # a_k
            theta[k * per + 1 : k * per + 1 + d_x, 0] = \
                rng.normal(0, 1.0 / np.sqrt(d_x), size=d_x)  # b_k
            theta[k * per + 1 + d_x, 0] = rng.normal(0, 1.0) # c_k
        return theta

    def param_dim(self, d_x: int) -> int:
        return self._K(d_x) * (d_x + 2)


# ---------------------------------------------------------------------------
# Sigmoid (logistic) model
# ---------------------------------------------------------------------------

class SigmoidModel(BaseModel):
    """Sigmoid / logistic function.

        g(theta; x) = a / (1 + exp(-(b^T x + c)))

    Total params: d_x + 2  (b vec, c scalar, a scale).
    """

    def _unpack(self, theta: np.ndarray, d_x: int):
        th = theta.ravel()
        b_vec = th[:d_x].reshape(d_x, 1)
        c_val = th[d_x]
        a_val = th[d_x + 1]
        return b_vec, c_val, a_val

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        d_x = x.shape[-1]
        b_vec, c_val, a_val = self._unpack(theta, d_x)
        z = x @ b_vec + c_val  # (n, 1)
        return a_val / (1.0 + np.exp(-z))

    def gradient(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        d_x = x.shape[-1]
        b_vec, c_val, a_val = self._unpack(theta, d_x)
        z = x @ b_vec + c_val
        sig = 1.0 / (1.0 + np.exp(-z))          # (n, 1)
        dsig = sig * (1.0 - sig)                 # (n, 1)

        # d/db: a * dsig * x
        grad_b = a_val * dsig * x                # (n, d_x)
        # d/dc: a * dsig
        grad_c = a_val * dsig                    # (n, 1)
        # d/da: sig
        grad_a = sig                             # (n, 1)

        return np.concatenate([grad_b, grad_c, grad_a], axis=-1)

    def init_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        d = self.param_dim(d_x)
        return rng.normal(0, 0.1, size=(d, 1))

    def true_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        d = self.param_dim(d_x)
        theta = np.zeros((d, 1))
        theta[:d_x, 0] = rng.normal(0, 0.5 / np.sqrt(d_x), size=d_x)  # b
        theta[d_x, 0] = rng.normal(0, 0.2)                              # c
        theta[d_x + 1, 0] = rng.normal(0, 2.0)                          # a (scale)
        return theta

    def param_dim(self, d_x: int) -> int:
        return d_x + 2


# ---------------------------------------------------------------------------
# Polynomial model (all monomials up to given degree)
# ---------------------------------------------------------------------------

class PolyModel(BaseModel):
    """Polynomial structural equation with all monomials up to `degree`.

        g(theta; x) = sum_{|alpha| <= degree} theta_alpha * x^alpha

    where alpha is a multi-index.  degree=1 gives the linear model.
    """

    def __init__(self, degree: int = 2):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        design = self._design_matrix(x)  # (n, d_theta)
        theta_2d = np.atleast_2d(theta.reshape(-1, 1))
        return design @ theta_2d

    def gradient(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        # Linear in theta, so gradient = design matrix
        _ = theta
        return self._design_matrix(x)

    def init_params(
        self, rng: np.random.Generator, d_x: int
    ) -> np.ndarray:
        d = self.param_dim(d_x)
        return rng.normal(0, 0.1, size=(d, 1))

    def true_params(
        self, rng: np.random.Generator, d_x: int
    ) -> np.ndarray:
        d = self.param_dim(d_x)
        theta = np.zeros((d, 1))
        # Linear part fully specified
        theta[:d_x, 0] = rng.normal(0, 1.0, size=d_x)
        # Higher-order: sparse
        if d > d_x:
            n_higher = d - d_x
            nz = max(1, n_higher // 4)
            idx = rng.choice(n_higher, size=nz, replace=False)
            theta[d_x + idx, 0] = rng.normal(0, 0.3, size=nz)
        return theta

    def param_dim(self, d_x: int) -> int:
        # Number of monomials of degree 0..k in d_x variables:
        #   sum_{r=0}^k C(d_x + r - 1, r) = C(d_x + k, k)
        return comb(d_x + self.degree, self.degree)

    def _design_matrix(self, x: np.ndarray) -> np.ndarray:
        """Build the monomial design matrix.

        Columns correspond to all monomials up to self.degree,
        including the constant term (degree-0).
        """
        n, d_x = x.shape
        columns = []
        for deg in range(self.degree + 1):
            for alpha in combinations_with_replacement(range(d_x), deg):
                col = np.ones(n)
                for var_idx in alpha:
                    col = col * x[:, var_idx]
                columns.append(col.reshape(-1, 1))
        return np.concatenate(columns, axis=-1)  # (n, d_theta)


# ---------------------------------------------------------------------------
# First-stage model
# ---------------------------------------------------------------------------

class LinearFirstStage:
    """First-stage model: h(gamma; z) = gamma^T z.

    Maps instruments z to explanatory variables x.
    Used both for data generation and as the trainable first-stage
    in OTSG.

    gamma has shape (d_z, d_x).
    """

    @staticmethod
    def predict(gamma: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute h(gamma; z) = z @ gamma.

        Args:
            gamma: (d_z, d_x).
            z: (..., d_z).

        Returns:
            Predicted x, shape (..., d_x).
        """
        return z @ gamma

    @staticmethod
    def gradient(gamma: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute nabla_gamma h(gamma; z) w.r.t. gamma.

        For h = z @ gamma, the gradient w.r.t gamma (flattened) is a
        matrix of shape (d_x, d_z * d_x).  For the OTSG update we need
        nabla_gamma h(gamma; z)^T * (h - x), which simplifies to
        z^T @ (h - x) of shape (d_z, d_x).

        This method returns the Jacobian-vector-product ready form:
        for each sample, it returns an outer product that, when
        multiplied by the residual vector, gives the gamma update.

        Args:
            gamma: (d_z, d_x) — not used (linear model).
            z: (n, d_z).

        Returns:
            A tensor-like representation.  For the OTSG update,
            the caller will compute: z.T @ residual  → (d_z, d_x).
            We return z so the caller can do the multiplication.

        We provide a convenience helper `gamma_update` instead.
        """
        _ = gamma
        return z  # (n, d_z) — caller computes z.T @ residual

    @staticmethod
    def gamma_update(
        z: np.ndarray, residual: np.ndarray
    ) -> np.ndarray:
        """Compute the gamma update direction: z^T @ residual.

        This is nabla_gamma h^T * (h(gamma; z) - x).

        Args:
            z: (n, d_z).
            residual: (n, d_x) = h(gamma; z) - x.

        Returns:
            Update matrix of shape (d_z, d_x).
        """
        return z.T @ residual

    @staticmethod
    def init_params(
        rng: np.random.Generator, d_z: int, d_x: int, scale: float = 0.1
    ) -> np.ndarray:
        """Randomly initialize gamma for training.

        Args:
            rng: NumPy random generator.
            d_z: instrument dimension.
            d_x: covariate dimension.
            scale: std of initialization.

        Returns:
            gamma of shape (d_z, d_x).
        """
        return rng.normal(0, scale, size=(d_z, d_x))


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Factory functions for models with constructor arguments
def _make_poly2():
    return PolyModel(degree=2)


def _make_poly3():
    return PolyModel(degree=3)


_MODEL_REGISTRY: dict[str, BaseModel] = {
    "linear": LinearModel(),
    "quadratic": QuadraticModel(),
    "poly2": _make_poly2(),
    "poly3": _make_poly3(),
    "nn": NeuralNetModel(),
    "sin": SinusoidalModel(),
    "sigmoid": SigmoidModel(),
}


def get_model(name: str) -> BaseModel:
    """Get a model instance by name.

    Args:
        name: model name, e.g. "linear", "quadratic", "poly2", "poly3".

    Returns:
        A BaseModel instance.

    Raises:
        ValueError: if the name is not registered.
    """
    name_lower = name.lower()
    if name_lower not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name_lower]
