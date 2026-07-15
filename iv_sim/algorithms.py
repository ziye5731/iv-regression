from __future__ import annotations

"""
algorithms.py -- Optimization algorithms for IV regression.

Implements the two IV regression algorithms from the README:

1. TOSG-IVaR (Two-Sample Oracle Stochastic Gradient IV Regression)
   - Source: Chen et al. 2024 (https://arxiv.org/abs/2405.19463)
   - Update rule:
     theta_{t+1} = theta_t - alpha_{t+1} (g(theta_t; x_t) - y_t)
                   * nabla_theta g(theta_t; x_t')

2. First-Order SLIM (Stochastic Linearized Instrumental-variable Method)
   - Source: Chen et al. 2025 (https://arxiv.org/abs/2510.20996)
   - Update rule:
     theta_{t+1} = theta_t - alpha_{t+1} (g(theta_t; x_{t,1}) - y_{t,1})
                   * nabla_theta g(theta_t; x_{t,2}) * z_{t,1}^T W z_{t,2}
"""

import numpy as np
from numpy.random import Generator
from itertools import permutations

from .config import SimulationConfig
from .data_generator import IVDataGenerator
from .models import BaseModel, LinearFirstStage


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseIVAlgorithm:
    """Abstract base class for IV regression algorithms.

    Subclasses must implement _step() defining the per-iteration update.
    """

    def __init__(
        self,
        config: SimulationConfig,
        model: BaseModel | None = None,
        seed: int | None = None,
        init_theta: np.ndarray | None = None,
    ):
        """
        Args:
            config: simulation configuration.
            model: structural model instance (defaults to config.model).
            seed: independent random seed.
            init_theta: initial theta (if None, random init). Used for resume.
        """
        self.config = config
        self.model = model if model is not None else config.model
        self.rng: Generator = np.random.default_rng(
            seed if seed is not None else config.seed
        )
        if init_theta is not None:
            self.theta = init_theta.copy()
        else:
            self.theta = self.model.init_params(self.rng, config.d_x)
        self.t = 0  # iteration counter

    @property
    def samples_per_step(self) -> int:
        """Number of (z, x, y) samples consumed per training step.

        Used to equalize the total sample budget across algorithms
        (not iterations) for fair comparison.
        """
        raise NotImplementedError

    def _learning_rate(self) -> float:
        """Compute the current learning rate alpha_t.

        Decay schedule: alpha_t = lr0 / t^decay
        """
        self.t += 1
        lr0 = self._get_lr0()
        decay = self._get_lr_decay()
        return lr0 / (self.t ** decay)

    def _get_lr0(self) -> float:
        """Subclass override: return initial learning rate."""
        raise NotImplementedError

    def _get_lr_decay(self) -> float:
        """Subclass override: return learning rate decay exponent."""
        raise NotImplementedError

    def _step(
        self, generator: IVDataGenerator, alpha: float
    ) -> tuple[np.ndarray, float]:
        """Perform one parameter update step.

        Args:
            generator: data generator.
            alpha: current learning rate.

        Returns:
            (updated theta, current loss).
        """
        raise NotImplementedError

    def train(
        self, generator: IVDataGenerator, n_iter: int, verbose: bool = False,
        verbose_every: int = 5000,
    ) -> list[dict]:
        """Train for n_iter steps.

        Args:
            generator: online data generator.
            n_iter: number of iterations.
            verbose: whether to print progress.
            verbose_every: print interval (iterations).

        Returns:
            history: list of {'step': int, 'theta': ndarray,
                    'loss': float, 'lr': float}.
        """
        history = []
        max_theta_norm = 1e6  # safety cap to prevent overflow
        for _ in range(n_iter):
            # Compute learning rate (increments t)
            lr = self._learning_rate()
            # Perform one update
            theta_new, loss = self._step(generator, lr)
            # Clip parameter norm for numerical stability
            th_norm = float(np.linalg.norm(theta_new))
            if th_norm > max_theta_norm:
                theta_new = theta_new * (max_theta_norm / th_norm)
            self.theta = theta_new
            history.append({
                "step": self.t,
                "theta": self.theta.copy(),
                "loss": loss,
                "lr": lr,
            })
            if verbose and self.t % verbose_every == 0:
                from .dgp import get_dgp
                dgp = get_dgp(self.config.dgp_mode)
                metric = dgp.compute_param_metric(self.theta, self.config)
                label = dgp.param_error_label
                mse = dgp.compute_pred_mse(self.theta, self.config, generator)
                print(
                    f"  [step {self.t:6d}] loss={loss:.6f}, "
                    f"{label}={metric:.6f}, MSE={mse:.6f}, lr={lr:.6f}"
                )
        return history


# ---------------------------------------------------------------------------
# TOSG-IVaR
# ---------------------------------------------------------------------------

class TOSGIVaR(BaseIVAlgorithm):
    """TOSG-IVaR algorithm.

    Update rule (Algorithm 1, Chen et al. 2024):
        theta_{t+1} = theta_t - alpha_{t+1} (g(theta_t; x_t) - y_t)
                      * nabla_theta g(theta_t; x_t')

    where x_t and x_t' are two conditionally independent draws
    given the same instrument z_t.
    """

    def _get_lr0(self) -> float:
        return self.config.tosg_lr

    @property
    def samples_per_step(self) -> int:
        return 2  # two conditionally independent (x,y) pairs per step

    def _get_lr_decay(self) -> float:
        return self.config.tosg_lr_decay

    def _step(
        self, generator: IVDataGenerator, alpha: float
    ) -> tuple[np.ndarray, float]:
        """TOSG-IVaR single-step update.

        1. Draw one z from the online stream
        2. Given z, independently sample two (x, y) pairs
        3. Predict with x_t, compute gradient with x_t'
        """
        # Draw one z
        z, _, _ = next(generator.generate_online())

        # Two conditionally independent (x, y) pairs given z
        x1, y1, x2, y2 = generator.generate_pair(z)

        # Prediction error: g(theta_t; x_t) - y_t
        pred = self.model.predict(self.theta, x1)  # (1, 1)
        error = pred - y1                            # (1, 1)

        # Gradient: nabla_theta g(theta_t; x_t')
        grad = self.model.gradient(self.theta, x2)  # (1, d_theta)

        # Update with gradient clipping for numerical stability
        err_val = float(error.item())
        err_val = np.clip(err_val, -1e4, 1e4)
        theta_new = self.theta - alpha * err_val * grad.T  # (d_theta, 1)
        loss = err_val ** 2
        loss = err_val ** 2

        return theta_new, loss


# ---------------------------------------------------------------------------
# First-Order SLIM
# ---------------------------------------------------------------------------

class FirstOrderSLIM(BaseIVAlgorithm):
    """First-Order SLIM algorithm.

    General update rule (Algorithm 1, Chen et al. 2025):

        theta_{t+1} = theta_t - alpha_{t+1} * M̃_{B_M}(theta_t)^T * W * m̃_{B_m}(theta_t)

    where

        M̃_{B_M}(theta) = (1/B_M) * Σ_{i=1}^{B_M} ∇_theta g(theta; x_i) * z_i^T
        m̃_{B_m}(theta) = (1/B_m) * Σ_{j=1}^{B_m} z_j * (g(theta; x_j) - y_j)

    and W is a positive-definite weighting matrix of shape (d_z, d_z).

    In the streaming setting (B_M = B_m = 1):

        theta_{t+1} = theta_t - alpha_{t+1} (g(theta_t; x_{t,1}) - y_{t,1})
                      * nabla_theta g(theta_t; x_{t,2}) * z_{t,1}^T W z_{t,2}
    """

    def __init__(
        self,
        config: SimulationConfig,
        model: BaseModel | None = None,
        seed: int | None = None,
        W: np.ndarray | None = None,
        B_M: int | None = None,
        B_m: int | None = None,
        W_type: str | None = None,
        init_theta: np.ndarray | None = None,
    ):
        """
        Args:
            config: simulation configuration.
            model: structural model instance.
            seed: independent random seed.
            W: explicit weighting matrix (d_z, d_z). Overrides W_type.
            B_M: batch size for Jacobian estimate M̃. Defaults to config.slim_B_M.
            B_m: batch size for moment estimate m̃. Defaults to config.slim_B_m.
            W_type: "identity", "random", or "custom". Defaults to config.slim_W_type.
            init_theta: initial theta (None = random).
        """
        super().__init__(config, model, seed, init_theta=init_theta)

        # --- Batch sizes ---
        self.B_M = B_M if B_M is not None else getattr(config, "slim_B_M", 1)
        self.B_m = B_m if B_m is not None else getattr(config, "slim_B_m", 1)

        # --- Weighting matrix W ---
        if W is not None:
            self.W = W
        else:
            w_type = W_type if W_type is not None else getattr(
                config, "slim_W_type", "identity"
            )
            self.W = self._build_W(config, w_type, seed)

    def _build_W(
        self, config: SimulationConfig, w_type: str, seed: int | None
    ) -> np.ndarray:
        """Construct a positive-definite weighting matrix.

        Args:
            config: simulation configuration.
            w_type: "identity", "random", or "custom".
            seed: random seed for reproducibility.

        Returns:
            W of shape (d_z, d_z), positive-definite.
        """
        d = config.d_z
        if w_type == "identity":
            return np.eye(d)
        elif w_type == "random":
            # Random symmetric positive-definite via A A^T + eps * I
            rng = np.random.default_rng(seed if seed is not None else config.seed)
            A = rng.normal(0, 1, size=(d, d))
            W_raw = A @ A.T
            # Normalize so that trace(W) = d (same scale as identity)
            W_raw *= d / np.trace(W_raw)
            return W_raw + 0.1 * np.eye(d)
        else:
            raise ValueError(f"Unknown W_type: '{w_type}'. "
                             f"Use 'identity' or 'random'.")

    def _get_lr0(self) -> float:
        return self.config.slim_lr

    @property
    def samples_per_step(self) -> int:
        return self.B_M + self.B_m

    def _get_lr_decay(self) -> float:
        return self.config.slim_lr_decay

    def _step(
        self, generator: IVDataGenerator, alpha: float
    ) -> tuple[np.ndarray, float]:
        """First-Order SLIM single-step update.

        General batch formulation:

            M̃ = (1/B_M) Σ z_i * ∇_theta g(theta; x_i)^T    (d_z, d_theta)
            m̃ = (1/B_m) Σ z_j * (g(theta; x_j) - y_j)       (d_z, 1)
            theta_new = theta - alpha * M̃^T @ W @ m̃          (d_theta, 1)

        Uses batched data generation and batched model evaluation
        for efficiency: one generate_batch + two model calls per step.
        """
        B_total = self.B_M + self.B_m

        # --- 1. Generate all data at once (batch) ---
        z_all, x_all, y_all = generator.generate_batch(B_total)

        # Split into M-batch (Jacobian) and m-batch (moment)
        z_M = z_all[:self.B_M]          # (B_M, d_z)
        x_M = x_all[:self.B_M]          # (B_M, d_x)
        z_m = z_all[self.B_M:]          # (B_m, d_z)
        x_m = x_all[self.B_M:]          # (B_m, d_x)
        y_m = y_all[self.B_M:]          # (B_m, 1)

        # --- 2. Batch model evaluation (2 calls total) ---
        # Jacobian: ∇g(θ; x_M)  shape (B_M, d_theta)
        grad_M = self.model.gradient(self.theta, x_M)
        # M̃ = (1/B_M) * Z_M^T @ grad_M = (d_z, B_M) @ (B_M, d_theta)
        M_tilde = (z_M.T @ grad_M) / self.B_M          # (d_z, d_theta)

        # Predictions and errors for moment estimate
        pred_m = self.model.predict(self.theta, x_m)   # (B_m, 1)
        err_m = pred_m - y_m                            # (B_m, 1)

        # m̃ = (1/B_m) * Z_m^T @ err_m = (d_z, B_m) @ (B_m, 1)
        m_tilde = (z_m.T @ err_m) / self.B_m            # (d_z, 1)

        # --- 3. Update ---
        # M̃^T @ W @ m̃: (d_theta, d_z) @ (d_z, d_z) @ (d_z, 1) = (d_theta, 1)
        update = M_tilde.T @ self.W @ m_tilde            # (d_theta, 1)

        # Clip for numerical stability
        update_norm = float(np.linalg.norm(update))
        max_update = 1e4
        if update_norm > max_update:
            update = update * (max_update / update_norm)

        theta_new = self.theta - alpha * update
        loss = float(np.mean(err_m ** 2))

        return theta_new, loss


# ---------------------------------------------------------------------------
# OTSG
# ---------------------------------------------------------------------------

class OTSGIVaR(BaseIVAlgorithm):
    """OTSG (One-Sample Two-Stage Gradient IV Regression).

    Update rule (Algorithm 2, Chen et al. 2024):
        theta_{t+1} = theta_t
            - alpha_{t+1} (g(theta_t; h(gamma_t; z_t)) - y_t)
              * nabla_theta g(theta_t; h(gamma_t; z_t))
        gamma_{t+1} = gamma_t
            - beta_{t+1} nabla_gamma h(gamma_t; z_t)^T
              * (h(gamma_t; z_t) - x_t)

    Uses only one data pair (z_t, x_t, y_t) per step.  The first-stage
    model h replaces the unobserved x with h(gamma_t; z_t), similar to
    two-stage least squares (2SLS).
    """

    def __init__(
        self,
        config: SimulationConfig,
        model: BaseModel | None = None,
        seed: int | None = None,
        init_theta: np.ndarray | None = None,
    ):
        super().__init__(config, model, seed, init_theta=init_theta)
        # First-stage model matches DGP phi_func
        phi = getattr(config, "phi_func", "linear")
        self.first_stage = LinearFirstStage(phi_func=phi)
        self.gamma = self.first_stage.init_params(
            self.rng, config.d_z, config.d_x
        )

    def _get_lr0(self) -> float:
        return self.config.otsg_theta_lr

    @property
    def samples_per_step(self) -> int:
        return 1  # one (z, x, y) observation per step

    def _get_lr_decay(self) -> float:
        return self.config.otsg_theta_lr_decay

    def _gamma_learning_rate(self) -> float:
        """Compute current gamma learning rate beta_t.

        Uses a separate decay schedule for the first-stage update.
        """
        beta0 = self.config.otsg_gamma_lr
        decay = self.config.otsg_gamma_lr_decay
        return beta0 / (self.t ** decay)

    def _step(
        self, generator: IVDataGenerator, alpha: float
    ) -> tuple[np.ndarray, float]:
        """OTSG single-step update.

        1. Draw one (z, x, y)
        2. Predict x_hat = h(gamma_t; z_t)
        3. Update theta using g(theta_t; x_hat) and its gradient
        4. Update gamma using first-stage residual: (x_hat - x_t)
        """
        # One sample
        z, x, y = next(generator.generate_online())

        # First-stage prediction: x_hat = h(gamma_t; z_t)
        x_hat = self.first_stage.predict(self.gamma, z)  # (1, d_x)

        # --- Theta update ---
        # Prediction error using x_hat
        pred = self.model.predict(self.theta, x_hat)      # (1, 1)
        error = pred - y                                    # (1, 1)

        # Gradient of g w.r.t theta at x_hat
        grad = self.model.gradient(self.theta, x_hat)      # (1, d_theta)

        # Update theta
        err_val = float(error.item())
        err_val = np.clip(err_val, -1e4, 1e4)
        theta_new = self.theta - alpha * err_val * grad.T  # (d_theta, 1)
        loss = err_val ** 2

        # --- Gamma update ---
        beta = self._gamma_learning_rate()
        residual = x_hat - x                                # (1, d_x)
        gamma_update = self.first_stage.gamma_update(
            z, residual, self.gamma)                        # (d_z, d_x)
        # Clip
        gu_norm = float(np.linalg.norm(gamma_update))
        if gu_norm > 1e4:
            gamma_update = gamma_update * (1e4 / gu_norm)
        self.gamma = self.gamma - beta * gamma_update

        return theta_new, loss


# ---------------------------------------------------------------------------
# Distance Covariance Optimization (DCO)
# ---------------------------------------------------------------------------

class DistanceCovOpt(BaseIVAlgorithm):
    """Distance Covariance Optimization for IV regression.

    Minimizes a centered pairwise-distance objective that drives z and
    residual r(θ) = y - g(θ; x) toward independence.  For a batch of B:

        D_{ij} = ||z_i - z_j||,   R2_{ij} = (r_i - r_j)^2
        F = mean(D * R2) - mean(D) * mean(R2)

    This is zero when z ⟂ r.  The gradient uses smooth squared
    differences (no sign function), making it amenable to SGD.

    Reference: Székely et al. (2007), Annals of Statistics.
    """

    def __init__(
        self,
        config: SimulationConfig,
        model: BaseModel | None = None,
        seed: int | None = None,
        B: int | None = None,
        init_theta: np.ndarray | None = None,
    ):
        """
        Args:
            config: simulation configuration.
            model: structural model instance.
            seed: independent random seed.
            B: batch size for dCov estimate. Defaults to config.dcov_B.
            init_theta: initial theta (None = random).
        """
        super().__init__(config, model, seed, init_theta=init_theta)
        self.B = B if B is not None else getattr(config, "dcov_B", 32)

    def _get_lr0(self) -> float:
        return self.config.dcov_lr

    @property
    def samples_per_step(self) -> int:
        return self.B

    def _get_lr_decay(self) -> float:
        return self.config.dcov_lr_decay

    def _step(
        self, generator: IVDataGenerator, alpha: float
    ) -> tuple[np.ndarray, float]:
        """DCO single-step update.

        Minimizes a centered pairwise-distance objective:
            F = mean(D * R2) - mean(D) * mean(R2)

        where D_{ij} = ||z_i - z_j|| and R2_{ij} = (r_i - r_j)^2.
        This is zero when z and r are independent and uses smooth
        (non-sign) gradients.
        """
        B = self.B

        # --- 1. Sample batch ---
        z, x, y = generator.generate_batch(B)

        # --- 2. Pairwise z-distance ---
        z_diff = z[:, None, :] - z[None, :, :]
        D_raw = np.linalg.norm(z_diff, axis=-1)           # (B, B)

        # --- 3. Residuals and squared differences ---
        pred = self.model.predict(self.theta, x)           # (B, 1)
        r_flat = (pred - y).ravel()                        # (B,)
        r_diff = r_flat[:, None] - r_flat[None, :]         # (B, B)
        R2 = r_diff ** 2                                   # (B, B)

        # --- 4. Centered gradient ---
        # ∂F/∂θ = (2/B²) * ∇g^T @ [ (D - mean(D)) ⊙ (r_i - r_j) ] @ 1 · 2
        # (The factor of 2 comes from ∂(r_i - r_j)^2/∂θ)
        D_mean = D_raw.mean()
        D_centered = D_raw - D_mean                        # (B, B)
        # Weight each sample by centered distance * residual diff
        weights = D_centered * r_diff                      # (B, B)
        w_sum = weights.sum(axis=1, keepdims=True)         # (B, 1)

        grad_g = self.model.gradient(self.theta, x)        # (B, d_theta)
        grad_F = (4.0 / (B * B)) * (grad_g.T @ w_sum)      # (d_theta, 1)

        # --- 5. Gradient normalization ---
        gnorm = float(np.linalg.norm(grad_F))
        if gnorm > 1e-8:
            grad_F = grad_F / gnorm

        # --- 6. Loss: centered distance objective (for monitoring) ---
        loss = np.mean(D_raw * R2) - D_mean * R2.mean()

        theta_new = self.theta - alpha * grad_F

        return theta_new, loss


# ---------------------------------------------------------------------------
# DCOV3  (Order-3 Distance Covariance Optimization)
# ---------------------------------------------------------------------------

class DCOV3(BaseIVAlgorithm):
    """DCOV3: Order-3 Distance Covariance Optimization.

    Uses triplets of i.i.d. samples to construct an unbiased stochastic
    gradient estimator for the distance covariance objective
    F(θ) = dCov²(z, Y - g(θ; x)).

    Each step draws 3 samples {(z_i, x_i, y_i)}_{i=1}^3, then
    symmetrizes the kernel v_t(θ; i,j,k) over all 3! = 6 permutations:

        v_t(i,j,k) = (||z_i-z_j|| - 2||z_i-z_k|| + δ̂_{t-1})
                     · sgn(ε_i - ε_j)
                     · (∇g(θ; x_j) - ∇g(θ; x_i))

    where ε_i = y_i - g(θ; x_i), and δ̂_t is a running estimate
    of E||z - z'|| updated online via

        δ̂_t = (t-1)/t · δ̂_{t-1} + 1/t · (1/3) Σ_{i<j} ||z_i - z_j||.
    """

    def __init__(
        self,
        config: SimulationConfig,
        model: BaseModel | None = None,
        seed: int | None = None,
        init_theta: np.ndarray | None = None,
    ):
        super().__init__(config, model, seed, init_theta=init_theta)
        self.delta_hat = 0.0  # running estimate of E||z - z'||

    def _get_lr0(self) -> float:
        return getattr(self.config, "dcov3_lr", self.config.dcov_lr)

    @property
    def samples_per_step(self) -> int:
        return 3

    def _get_lr_decay(self) -> float:
        return getattr(self.config, "dcov3_lr_decay", self.config.dcov_lr_decay)

    def _step(
        self, generator: IVDataGenerator, alpha: float
    ) -> tuple[np.ndarray, float]:
        """DCOV3 single-step update with order-3 symmetrized gradient."""

        # --- 1. Draw 3 i.i.d. samples ---
        z, x, y = generator.generate_batch(3)          # (3, d_z), (3, d_x), (3, 1)

        # --- 2. Precompute residuals & gradients ---
        residuals = (self.model.predict(self.theta, x) - y).ravel()  # (3,)
        grads = self.model.gradient(self.theta, x)                    # (3, d_theta)

        # --- 3. Pairwise z-distances ---
        z_diff = z[:, None, :] - z[None, :, :]          # (3, 3, d_z)
        z_dists = np.linalg.norm(z_diff, axis=-1)       # (3, 3)

        # --- 4. Pairwise z-distances for δ̂ update ---
        pair_mean = (z_dists[0, 1] + z_dists[0, 2] + z_dists[1, 2]) / 3.0

        # --- 5. Symmetrize over all 6 permutations ---
        # Use δ̂_{t-1} (NOT yet updated) as the formula specifies.
        perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2),
                 (1, 2, 0), (2, 0, 1), (2, 1, 0)]

        grad_F = np.zeros_like(self.theta)                # (d_theta, 1)
        for i, j, k in perms:
            z_term = z_dists[i, j] - 2.0 * z_dists[i, k] # + self.delta_hat
            sgn_term = np.sign(residuals[i] - residuals[j])
            grad_term = grads[j] - grads[i]              # (d_theta,)
            v = z_term * sgn_term * grad_term.reshape(-1, 1)
            grad_F += v

        grad_F /= 6.0                                     # average over 6 perms

        # --- 6. Online update of δ̂ (AFTER gradient—uses δ̂_{t-1} above) ---
        # δ̂_t = (t-1)/t · δ̂_{t-1} + 1/t · (1/3) Σ_{i<j} ||z_i - z_j||
        self.delta_hat = ((self.t - 1) / self.t) * self.delta_hat + \
                         (1.0 / self.t) * pair_mean

        # --- 7. Monitoring loss: centred pairwise dCov objective ---
        r_diff = residuals[:, None] - residuals[None, :]  # (3, 3)
        R2 = r_diff ** 2
        D_mean = z_dists.mean()
        loss = np.mean(z_dists * R2) - D_mean * R2.mean()

        theta_new = self.theta - alpha * grad_F

        return theta_new, loss


# ---------------------------------------------------------------------------
# DCOV4  (Order-4 U-statistic Distance Covariance Optimization)
# ---------------------------------------------------------------------------

class DCOV4(BaseIVAlgorithm):
    """DCOV4: Order-4 U-statistic gradient for dCov².

    Uses 4 i.i.d. samples per step and symmetrizes the kernel over all
    4! = 24 permutations.  Unlike DCOV3, this does NOT require an online
    estimate of δ = E||z - z'||—the constant term is handled internally
    by the cross-terms between independent sample indices.

    Kernel (before symmetrisation):

        v(i,j,k,l) =  ||z_i - z_j|| · sgn(ε_i - ε_j) · (∇g_j - ∇g_i)
                    + ||z_i - z_j|| · sgn(ε_k - ε_l) · (∇g_l - ∇g_k)
                    - ||z_i - z_j|| · sgn(ε_i - ε_k) · (∇g_k - ∇g_i)
                    - ||z_i - z_k|| · sgn(ε_i - ε_j) · (∇g_j - ∇g_i)

    ∇̂F = (1 / 4!) Σ_{(i,j,k,l)!} v(i,j,k,l)
    θ_t = θ_{t-1} - α_t · ∇̂F
    """

    def __init__(
        self,
        config: SimulationConfig,
        model: BaseModel | None = None,
        seed: int | None = None,
        init_theta: np.ndarray | None = None,
    ):
        super().__init__(config, model, seed, init_theta=init_theta)

    def _get_lr0(self) -> float:
        return getattr(self.config, "dcov4_lr", self.config.dcov_lr)

    @property
    def samples_per_step(self) -> int:
        return 4

    def _get_lr_decay(self) -> float:
        return getattr(self.config, "dcov4_lr_decay", self.config.dcov_lr_decay)

    def _step(
        self, generator: IVDataGenerator, alpha: float
    ) -> tuple[np.ndarray, float]:
        """DCOV4 single-step update with order-4 symmetrised gradient."""

        # --- 1. Draw 4 i.i.d. samples ---
        z, x, y = generator.generate_batch(4)

        # --- 2. Precompute residuals, gradients, pairwise quantities ---
        residuals = (self.model.predict(self.theta, x) - y).ravel()    # (4,)
        grads = self.model.gradient(self.theta, x)                     # (4, d_theta)

        z_diff = z[:, None, :] - z[None, :, :]                         # (4, 4, d_z)
        z_dists = np.linalg.norm(z_diff, axis=-1)                      # (4, 4)

        # sgn[a,b] = sign(ε_a - ε_b),  gdiff[a,b] = ∇g_b - ∇g_a
        sgn = np.sign(residuals[:, None] - residuals[None, :])          # (4, 4)
        gdiff = grads[None, :, :] - grads[:, None, :]                   # (4, 4, d_theta)

        # --- 3. All 24 permutations ---
        perms = list(permutations(range(4)))

        grad_F = np.zeros_like(self.theta)
        for i, j, k, l in perms:
            v1 = z_dists[i, j] * sgn[i, j] * gdiff[i, j]               # (d_theta,)
            v2 = z_dists[i, j] * sgn[k, l] * gdiff[k, l]
            v3 = -z_dists[i, j] * sgn[i, k] * gdiff[i, k]
            v4 = -z_dists[i, k] * sgn[i, j] * gdiff[i, j]
            grad_F += (v1 + v2 + v3 + v4).reshape(-1, 1)

        grad_F /= 24.0

        # --- 4. Monitoring loss ---
        r_diff = residuals[:, None] - residuals[None, :]                # (4, 4)
        R2 = r_diff ** 2
        D_mean = z_dists.mean()
        loss = np.mean(z_dists * R2) - D_mean * R2.mean()

        theta_new = self.theta - alpha * grad_F

        return theta_new, loss


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

_ALGO_REGISTRY = {
    "tosg": TOSGIVaR,
    "tosg_ivar": TOSGIVaR,
    "otsg": OTSGIVaR,
    "otsg_ivar": OTSGIVaR,
    "slim": FirstOrderSLIM,
    "first_order_slim": FirstOrderSLIM,
    "dcov": DistanceCovOpt,
    "dco": DistanceCovOpt,
    "distance_cov": DistanceCovOpt,
    "dcov3": DCOV3,
    "dcov4": DCOV4,
}


def get_algorithm(name: str):
    """Get an algorithm class by name.

    Args:
        name: algorithm name, e.g. "tosg" or "slim".

    Returns:
        Algorithm class.

    Raises:
        ValueError: if the name is not registered.
    """
    name_lower = name.lower()
    if name_lower not in _ALGO_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{name}'. Available: {list(_ALGO_REGISTRY.keys())}"
        )
    return _ALGO_REGISTRY[name_lower]
