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
    in OSTG-IVaR.

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
        matrix of shape (d_x, d_z * d_x).  For the OSTG update we need
        nabla_gamma h(gamma; z)^T * (h - x), which simplifies to
        z^T @ (h - x) of shape (d_z, d_x).

        This method returns the Jacobian-vector-product ready form:
        for each sample, it returns an outer product that, when
        multiplied by the residual vector, gives the gamma update.

        Args:
            gamma: (d_z, d_x) — not used (linear model).
            z: (n, d_z).

        Returns:
            A tensor-like representation.  For the OSTG update,
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
