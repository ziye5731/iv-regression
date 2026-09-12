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
# Numba JIT helpers (optional; gracefully fall back if not available)
# ---------------------------------------------------------------------------

_HAS_NUMBA = False
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    njit = lambda f=None, **kwargs: (f if f is not None else (lambda x: x))
    prange = range


if _HAS_NUMBA:
    @njit
    def _dcov3_numba_grad(residuals, grads, z_dists, delta_hat, d_theta):
        """Numba-compiled DCOV3 gradient: 6-permutation symmetrisation."""
        grad_F = np.zeros((d_theta, 1))
        perms = np.array([
            [0, 1, 2], [0, 2, 1], [1, 0, 2],
            [1, 2, 0], [2, 0, 1], [2, 1, 0],
        ], dtype=np.int64)
        for p_idx in range(6):
            i, j, k = perms[p_idx]
            z_term = z_dists[i, j] - 2.0 * z_dists[i, k] + delta_hat
            sgn_term = 1.0 if residuals[i] > residuals[j] else (
                -1.0 if residuals[i] < residuals[j] else 0.0)
            for d in range(d_theta):
                grad_term = grads[j, d] - grads[i, d]
                grad_F[d, 0] += z_term * sgn_term * grad_term
        grad_F /= 6.0
        return grad_F

    @njit
    def _dcov4_numba_grad(residuals, grads, z_dists, d_theta):
        """Numba-compiled DCOV4 gradient: 24-permutation symmetrisation."""
        grad_F = np.zeros((d_theta, 1))
        perms_list = np.array([
            [0,1,2,3],[0,1,3,2],[0,2,1,3],[0,2,3,1],[0,3,1,2],[0,3,2,1],
            [1,0,2,3],[1,0,3,2],[1,2,0,3],[1,2,3,0],[1,3,0,2],[1,3,2,0],
            [2,0,1,3],[2,0,3,1],[2,1,0,3],[2,1,3,0],[2,3,0,1],[2,3,1,0],
            [3,0,1,2],[3,0,2,1],[3,1,0,2],[3,1,2,0],[3,2,0,1],[3,2,1,0],
        ], dtype=np.int64)
        for p_idx in range(24):
            i, j, k, l = perms_list[p_idx]
            v1 = z_dists[i, j] * _sgn(residuals[i], residuals[j])
            v2 = z_dists[i, j] * _sgn(residuals[k], residuals[l])
            v3 = -z_dists[i, j] * _sgn(residuals[i], residuals[k])
            v4 = -z_dists[i, k] * _sgn(residuals[i], residuals[j])
            for d in range(d_theta):
                gdiff_ij = grads[j, d] - grads[i, d]
                gdiff_kl = grads[l, d] - grads[k, d]
                gdiff_ik = grads[k, d] - grads[i, d]
                grad_F[d, 0] += (v1 * gdiff_ij + v2 * gdiff_kl
                                  + v3 * gdiff_ik + v4 * gdiff_ij)
        grad_F /= 24.0
        return grad_F

    @njit
    def _sgn(a, b):
        if a > b:
            return 1.0
        elif a < b:
            return -1.0
        return 0.0


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
        start_iter: int = 0,
    ):
        """
        Args:
            config: simulation configuration.
            model: structural model instance (defaults to config.model).
            seed: independent random seed.
            init_theta: initial theta (if None, random init). Used for resume.
            start_iter: starting iteration count for resumed training.
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
        self.t = max(0, start_iter)  # iteration counter

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

    def _history_theta(self, theta: np.ndarray) -> np.ndarray:
        """Return the theta to record in the training history.

        Defaults to the raw iterate; subclasses may override this to record
        a derived estimator (e.g. a Polyak–Ruppert average).
        """
        return theta

    def train(
        self, generator: IVDataGenerator, n_iter: int, verbose: bool = False,
        verbose_every: int = 5000, history_every: int = 1,
        history_checkpoints: list[int] | None = None,
        early_stop_threshold: float = 0.0,
        early_stop_patience: int = 0,
    ) -> list[dict]:
        """Train for n_iter steps.

        Args:
            generator: online data generator.
            n_iter: number of iterations.
            verbose: whether to print progress.
            verbose_every: print interval (iterations).
            history_every: record history every N iterations (ignored if
                history_checkpoints is provided).
            history_checkpoints: if provided, record history only at these
                exact step numbers (log-spaced for log-scale plotting).
            early_stop_threshold: if > 0, stop when ||theta - theta*||
                changes by less than this over early_stop_patience checks.
            early_stop_patience: number of checks to wait before stopping.

        Returns:
            history: list of {'step': int, 'theta': ndarray,
                    'loss': float, 'lr': float}.
        """
        history = []
        max_theta_norm = 1e6  # safety cap to prevent overflow
        if history_checkpoints is not None:
            checkpoint_set = set(history_checkpoints)
        else:
            checkpoint_set = None
            history_every = max(1, history_every)
            max_history_records = 100000
            estimated_records = (n_iter + history_every - 1) // history_every
            if estimated_records > max_history_records:
                history_every = max(1, int(np.ceil(n_iter / max_history_records)))

        # Early stopping state
        use_early_stop = early_stop_threshold > 0 and early_stop_patience > 0
        if use_early_stop and hasattr(self.config, 'theta_star') and self.config.theta_star is not None:
            theta_star_1d = self.config.theta_star.ravel()
            _check_interval = max(1000, verbose_every // 10)
            _patience_counter = 0
            _prev_param_err = float('inf')
        else:
            use_early_stop = False
            _check_interval = 0

        # Local variable bindings for speed
        _step = self._step
        _learning_rate = self._learning_rate
        _theta = self.theta
        _verbose = verbose
        _ve = verbose_every
        _norm_check_mod = 1000  # only check norm every N steps

        for i in range(n_iter):
            # Compute learning rate (increments t)
            lr = _learning_rate()
            # Perform one update
            theta_new, loss = _step(generator, lr)
            # Clip parameter norm (sparsely to reduce overhead)
            if i % _norm_check_mod == 0:
                th_norm = float(np.linalg.norm(theta_new))
                if th_norm > max_theta_norm:
                    theta_new = theta_new * (max_theta_norm / th_norm)
            self.theta = theta_new
            _theta = theta_new
            # Decide whether to record this step
            if checkpoint_set is not None:
                record_now = self.t in checkpoint_set or i == n_iter - 1
            else:
                record_now = (self.t % history_every == 0 or i == n_iter - 1)
            if record_now:
                history.append({
                    "step": self.t,
                    "theta": self._history_theta(_theta).ravel().copy(),
                    "loss": loss,
                    "lr": lr,
                })
            if _verbose and self.t % _ve == 0:
                from .dgp import get_dgp
                dgp = get_dgp(self.config.dgp_mode)
                metric = dgp.compute_param_metric(_theta, self.config)
                label = dgp.param_error_label
                mse = dgp.compute_pred_mse(_theta, self.config, generator)
                print(
                    f"  [step {self.t:6d}] loss={loss:.6f}, "
                    f"{label}={metric:.6f}, MSE={mse:.6f}, lr={lr:.6f}"
                )
            # Early stopping check
            if use_early_stop and self.t % _check_interval == 0:
                cur_err = float(np.linalg.norm(_theta.ravel() - theta_star_1d))
                if abs(_prev_param_err - cur_err) < early_stop_threshold:
                    _patience_counter += 1
                    if _patience_counter >= early_stop_patience:
                        if _verbose or True:
                            print(f"  >> Early stop at step {self.t}: "
                                  f"param error={cur_err:.6f}, "
                                  f"delta={abs(_prev_param_err - cur_err):.2e}")
                        # Ensure final step is recorded
                        if not record_now:
                            history.append({
                                "step": self.t,
                                "theta": self._history_theta(_theta).ravel().copy(),
                                "loss": loss,
                                "lr": lr,
                            })
                        break
                else:
                    _patience_counter = 0
                _prev_param_err = cur_err

        if not history:
            history.append({
                "step": self.t,
                "theta": self._history_theta(_theta).ravel().copy(),
                "loss": float("nan"),
                "lr": float("nan"),
            })
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
        start_iter: int = 0,
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
            start_iter: starting iteration count for resumed training.
        """
        super().__init__(config, model, seed, init_theta=init_theta,
                         start_iter=start_iter)

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
        start_iter: int = 0,
    ):
        super().__init__(config, model, seed, init_theta=init_theta,
                         start_iter=start_iter)
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

    Minimizes the true distance covariance objective that drives z and
    residual r(θ) = y - g(θ; x) toward independence.  For a batch of B:

        D_{ij} = ||z_i - z_j||,   |r|_{ij} = |r_i - r_j|
        F = mean(D · |r|) - mean(D) · mean(|r|)

    F = 0  ⇔  z ⟂ r.  The gradient uses sgn(r_i - r_j).

    Reference: Székely et al. (2007), Annals of Statistics.
    """

    def __init__(
        self,
        config: SimulationConfig,
        model: BaseModel | None = None,
        seed: int | None = None,
        B: int | None = None,
        init_theta: np.ndarray | None = None,
        start_iter: int = 0,
    ):
        """
        Args:
            config: simulation configuration.
            model: structural model instance.
            seed: independent random seed.
            B: batch size for dCov estimate. Defaults to config.dcov_B.
            init_theta: initial theta (None = random).
            start_iter: starting iteration count for resumed training.
        """
        super().__init__(config, model, seed, init_theta=init_theta,
                         start_iter=start_iter)
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
        R2 = np.abs(r_diff)                                 # |r_i - r_j|  (true dCov)

        # --- 4. Centered gradient ---
        # ∇_θ |r_i - r_j| = sgn(r_i - r_j)·(∇g_i - ∇g_j)
        # ∇F = (2/B²) Σ_ij (D_ij - D̄) · sgn(r_i - r_j) · ∇g_i   [by symmetry]
        D_mean = D_raw.mean()
        D_centered = D_raw - D_mean                        # (B, B)
        weights = D_centered * np.sign(r_diff)              # (B, B)
        w_sum = weights.sum(axis=1, keepdims=True)         # (B, 1)

        grad_g = self.model.gradient(self.theta, x)        # (B, d_theta)
        grad_F = (2.0 / (B * B)) * (grad_g.T @ w_sum)      # (d_theta, 1)

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
        start_iter: int = 0,
    ):
        super().__init__(config, model, seed, init_theta=init_theta,
                         start_iter=start_iter)
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
        residuals = (y - self.model.predict(self.theta, x)).ravel()  # (3,)
        grads = self.model.gradient(self.theta, x)                    # (3, d_theta)

        # --- 3. Pairwise z-distances ---
        z_diff = z[:, None, :] - z[None, :, :]          # (3, 3, d_z)
        z_dists = np.linalg.norm(z_diff, axis=-1)       # (3, 3)

        # --- 4. Pairwise z-distances for δ̂ update ---
        pair_mean = (z_dists[0, 1] + z_dists[0, 2] + z_dists[1, 2]) / 3.0

        # --- 5. Symmetrize over all 6 permutations ---
        # Use δ̂_{t-1} (NOT yet updated) as the formula specifies.
        if _HAS_NUMBA:
            d_theta = grads.shape[1]
            grad_F = _dcov3_numba_grad(residuals, grads, z_dists,
                                        self.delta_hat, d_theta)
        else:
            perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2),
                     (1, 2, 0), (2, 0, 1), (2, 1, 0)]
            grad_F = np.zeros_like(self.theta)            # (d_theta, 1)
            for i, j, k in perms:
                z_term = z_dists[i, j] - 2.0 * z_dists[i, k] + self.delta_hat
                sgn_term = np.sign(residuals[i] - residuals[j])
                grad_term = grads[j] - grads[i]          # (d_theta,)
                v = z_term * sgn_term * grad_term.reshape(-1, 1)
                grad_F += v
            grad_F /= 6.0                                 # average over 6 perms

        # --- 6. Online update of δ̂ (AFTER gradient—uses δ̂_{t-1} above) ---
        # δ̂_t = (t-1)/t · δ̂_{t-1} + 1/t · (1/3) Σ_{i<j} ||z_i - z_j||
        self.delta_hat = ((self.t - 1) / self.t) * self.delta_hat + \
                         (1.0 / self.t) * pair_mean

        # --- 7. Monitoring loss: centred pairwise dCov objective ---
        r_diff = residuals[:, None] - residuals[None, :]  # (3, 3)
        R2 = np.abs(r_diff)
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
        start_iter: int = 0,
    ):
        super().__init__(config, model, seed, init_theta=init_theta,
                         start_iter=start_iter)

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
        residuals = (y-self.model.predict(self.theta, x)).ravel()    # (4,)
        grads = self.model.gradient(self.theta, x)                     # (4, d_theta)

        z_diff = z[:, None, :] - z[None, :, :]                         # (4, 4, d_z)
        z_dists = np.linalg.norm(z_diff, axis=-1)                      # (4, 4)

        # sgn[a,b] = sign(ε_a - ε_b),  gdiff[a,b] = ∇g_b - ∇g_a
        sgn = np.sign(residuals[:, None] - residuals[None, :])          # (4, 4)
        gdiff = grads[None, :, :] - grads[:, None, :]                   # (4, 4, d_theta)

        # --- 3. All 24 permutations ---
        if _HAS_NUMBA:
            d_theta = grads.shape[1]
            grad_F = _dcov4_numba_grad(residuals, grads, z_dists, d_theta)
        else:
            perms = list(permutations(range(4)))
            grad_F = np.zeros_like(self.theta)
            for i, j, k, l in perms:
                v1 = z_dists[i, j] * sgn[i, j] * gdiff[i, j]           # (d_theta,)
                v2 = z_dists[i, j] * sgn[k, l] * gdiff[k, l]
                v3 = -z_dists[i, j] * sgn[i, k] * gdiff[i, k]
                v4 = -z_dists[i, k] * sgn[i, j] * gdiff[i, j]
                grad_F += (v1 + v2 + v3 + v4).reshape(-1, 1)
            grad_F /= 24.0

        # --- 4. Monitoring loss ---
        r_diff = residuals[:, None] - residuals[None, :]                # (4, 4)
        R2 = np.abs(r_diff)
        D_mean = z_dists.mean()
        loss = np.mean(z_dists * R2) - D_mean * R2.mean()

        theta_new = self.theta - alpha * grad_F

        return theta_new, loss


# ---------------------------------------------------------------------------
# Sieve1  (online preconditioned GMM with a sieve of instrument functions)
# ---------------------------------------------------------------------------

def _sieve_poly_features(
    z: np.ndarray, degree: int, include_intercept: bool = False
) -> np.ndarray:
    """Polynomial sieve features of z up to `degree`.

    degree = 1 -> [z_1, ..., z_dz]
    degree = 2 -> degree-1 features + [z_i z_j for i <= j]

    Only degree 1 and 2 are implemented (degree 3 is available through the
    orthonormal Hermite basis, "herm3").  If include_intercept is True, a
    constant column of ones is prepended.

    Returns (B, p).
    """
    if degree not in (1, 2):
        raise ValueError(
            f"_sieve_poly_features supports degree 1 or 2, got {degree}")
    d_z = z.shape[1]
    feats = []
    if include_intercept:
        feats.append(np.ones((z.shape[0], 1)))
    feats.append(z)
    if degree >= 2:
        cols = []
        for i in range(d_z):
            for j in range(i, d_z):
                cols.append((z[:, i] * z[:, j]).reshape(-1, 1))
        feats.append(np.concatenate(cols, axis=1))
    return np.concatenate(feats, axis=1)


def _sieve_hermite_features(
    z: np.ndarray, degree: int, include_intercept: bool = False
) -> np.ndarray:
    """Orthonormal (probabilists') Hermite sieve features of z, z ~ N(0, I).

    Tensor-product basis up to `degree` (supports degree 1, 2 and 3):

        degree 1: H_1(z_i) = z_i
        degree 2: H_2(z_i) = (z_i^2 - 1)/sqrt(2), and
                  H_1(z_i) H_1(z_j) = z_i z_j  (i < j)
        degree 3: H_3(z_i) = (z_i^3 - 3 z_i)/sqrt(6),
                  H_2(z_i) H_1(z_j)  (i != j), and
                  H_1(z_i) H_1(z_j) H_1(z_k) = z_i z_j z_k  (i < j < k)

    All blocks are orthonormal and mutually orthogonal, i.e.
    E[psi psi^T] = I.  If include_intercept is True, a constant column of ones
    is prepended (the constant H_0 = 1 is normally excluded).

    Returns (B, p).
    """
    d_z = z.shape[1]
    feats = []
    if include_intercept:
        feats.append(np.ones((z.shape[0], 1)))
    feats.append(z)
    if degree >= 2:
        cols = []
        for i in range(d_z):
            cols.append(((z[:, i] ** 2 - 1.0) / np.sqrt(2.0)).reshape(-1, 1))
        for i in range(d_z):
            for j in range(i + 1, d_z):
                cols.append((z[:, i] * z[:, j]).reshape(-1, 1))
        feats.append(np.concatenate(cols, axis=1))
    if degree >= 3:
        cols = []
        # H_3(z_i) = (z_i^3 - 3 z_i)/sqrt(6)
        for i in range(d_z):
            cols.append(((z[:, i] ** 3 - 3.0 * z[:, i])
                         / np.sqrt(6.0)).reshape(-1, 1))
        # H_2(z_i) * H_1(z_j),  i != j
        for i in range(d_z):
            h2 = (z[:, i] ** 2 - 1.0) / np.sqrt(2.0)
            for j in range(d_z):
                if j == i:
                    continue
                cols.append((h2 * z[:, j]).reshape(-1, 1))
        # z_i z_j z_k,  i < j < k
        for i in range(d_z):
            for j in range(i + 1, d_z):
                for k in range(j + 1, d_z):
                    cols.append((z[:, i] * z[:, j] * z[:, k]).reshape(-1, 1))
        feats.append(np.concatenate(cols, axis=1))
    return np.concatenate(feats, axis=1)


class Sieve1(BaseIVAlgorithm):
    """Sieve1: online preconditioned GMM with a sieve of instrument functions.

    Minimizes

        Q(theta) = (1/2) m(theta)^T W m(theta),
        m(theta) = E[ psi(z) * (g(theta; x) - y) ],

    using a single (z, x, y) stream sample per step - no two-sample oracle and
    no first-stage nuisance model.  Each step:

        1. evaluate the current-sample moment  m_t = psi(z_t) (g(theta_t;x_t)-y_t)
        2. build a Newton-style preconditioner from *past* samples only

               A_t = (Mbar^T W_t Mbar + lambda_t I)^{-1} Mbar^T W_t,

           where Mbar is the running average of psi(z) grad_theta g(theta;x)^T
           and W_t is diagonal inverse-variance weighting.
        3. update  theta_{t+1} = theta_t - alpha_{t+1} A_t m_t.

    Because A_t depends only on F_{t-1}, E[A_t m_t | F_{t-1}] = E[A_t] m(theta_t),
    so the update is (conditionally) unbiased and drives m(theta) to 0.

    The sieve basis psi is a fixed set of instrument functions of z; a growing
    polynomial / Hermite basis approximates the Chamberlain optimal instrument
    E[grad g | z] as degree -> infinity (Newey 1990; Chen 2007), without ever
    estimating a nuisance model.
    """

    def __init__(
        self,
        config: SimulationConfig,
        model: BaseModel | None = None,
        seed: int | None = None,
        degree: int | None = None,
        basis: str | None = None,
        B: int | None = None,
        reg: float | None = None,
        clip: float | None = None,
        W_type: str | None = None,
        ema: float | None = None,
        init_theta: np.ndarray | None = None,
        start_iter: int = 0,
    ):
        super().__init__(config, model, seed, init_theta=init_theta,
                         start_iter=start_iter)
        self.degree = degree if degree is not None else getattr(
            config, "sieve_degree", 2)
        self.basis = basis if basis is not None else getattr(
            config, "sieve_basis", "poly")
        self.B = B if B is not None else getattr(config, "sieve_B", 1)
        self.reg = reg if reg is not None else getattr(config, "sieve_reg", 1e-2)
        self.clip = clip if clip is not None else getattr(config, "sieve_clip", 10.0)
        self.w_type = W_type if W_type is not None else getattr(
            config, "sieve_W_type", "diag")
        self.ema = ema if ema is not None else getattr(config, "sieve_ema", 0.0)

        if self.degree not in (1, 2):
            raise ValueError(f"Sieve1 supports degree 1 or 2, got {self.degree}")
        if self.basis not in ("poly", "hermite"):
            raise ValueError(
                f"Unknown sieve basis '{self.basis}' (use 'poly' or 'hermite')")

        self._p = self._feature_dim(config.d_z)
        self._M_bar = np.zeros((self._p, config.d_theta))
        self._S_bar = np.zeros(self._p)
        self._warned_rank = False

    # ------------------------------------------------------------------
    # Sieve / preconditioner helpers
    # ------------------------------------------------------------------

    def _feature_dim(self, d_z: int) -> int:
        if self.degree == 1:
            return d_z
        # degree 2: linear + all symmetric quadratic monomials
        return d_z + d_z * (d_z + 1) // 2

    def _features(self, z: np.ndarray) -> np.ndarray:
        if self.basis == "hermite":
            return _sieve_hermite_features(z, self.degree)
        return _sieve_poly_features(z, self.degree)

    def _precond_rate(self) -> float:
        """Update rate for the running preconditioner statistics."""
        if self.ema and self.ema > 0.0:
            return self.ema
        return 1.0 / max(1, self.t)

    def _weight_vector(self) -> np.ndarray:
        """Diagonal weighting vector (p,) for the moment quadratic form."""
        if self.w_type == "identity":
            return np.ones(self._p)
        # diagonal inverse-variance weighting: W = diag(1 / E[psi^2 eps^2])
        return 1.0 / (self._S_bar + self.reg)

    # ------------------------------------------------------------------
    # BaseIVAlgorithm interface
    # ------------------------------------------------------------------

    def _get_lr0(self) -> float:
        return getattr(self.config, "sieve_lr", 0.1)

    @property
    def samples_per_step(self) -> int:
        return self.B

    def _get_lr_decay(self) -> float:
        return getattr(self.config, "sieve_lr_decay", 0.5)

    def _step(
        self, generator: IVDataGenerator, alpha: float
    ) -> tuple[np.ndarray, float]:
        """Sieve1 single-step update (B fresh samples for the moment)."""
        B = self.B
        z, x, y = generator.generate_batch(B)

        psi = self._features(z)                        # (B, p)
        pred = self.model.predict(self.theta, x)       # (B, 1)
        eps = pred - y                                  # (B, 1)
        grad = self.model.gradient(self.theta, x)      # (B, d_theta)

        # -- current-sample moment (mini-batch mean)
        m_t = (psi.T @ eps) / B                         # (p, 1)

        # -- preconditioner from past samples only (=> unbiased given F_{t-1})
        W_vec = self._weight_vector()                   # (p,)
        WM = self._M_bar * W_vec[:, None]               # (p, d)
        MtWM = self._M_bar.T @ WM                       # (d, d)
        MtW = WM.T                                      # (d, p)
        d, p = MtW.shape
        if d <= p:
            # Levenberg-style damping scaled with the moment-matrix trace
            reg_eff = self.reg * max(1.0, float(np.trace(MtWM)) / d)
            A = np.linalg.solve(MtWM + reg_eff * np.eye(d), MtW)   # (d, p)
        else:
            # Under-identified moment system (e.g. MLP): fall back to M^T W.
            if not self._warned_rank:
                print(f"  [Sieve1] d_theta={d} > p={p}; using gradient "
                      f"direction (no Newton inverse).")
                self._warned_rank = True
            scale = max(1.0, float(np.trace(MtWM)) / d)
            A = MtW / scale

        update = A @ m_t                               # (d, 1)

        # -- numerical safety: cap the per-step parameter displacement
        un = float(np.linalg.norm(update))
        if un > self.clip:
            update = update * (self.clip / un)

        theta_new = self.theta - alpha * update
        loss = float(np.mean(eps ** 2))

        # -- update running statistics with the current sample (after theta)
        J = (psi.T @ grad) / B                          # (p, d)
        beta = self._precond_rate()
        self._M_bar = (1.0 - beta) * self._M_bar + beta * J
        psi2_eps2 = ((psi * psi).T @ (eps * eps)) / B   # (p, 1)  E[psi^2 eps^2]
        self._S_bar = (1.0 - beta) * self._S_bar + beta * psi2_eps2.ravel()

        return theta_new, loss


# ---------------------------------------------------------------------------
# Sieve2  (online Sieve-SGMM: full covariance weighting, projection, averaging)
# ---------------------------------------------------------------------------

class Sieve2(BaseIVAlgorithm):
    """Sieve2: online Sieve-SGMM with full-moment-covariance weighting.

    Refinement of Sieve1 adding the standard stochastic-approximation
    machinery needed to make the theory hold (see README):

      - instrument basis includes the intercept  psi(z) = (1, psi_1, ...)
      - full moment-covariance weighting  W = Omega^{-1},
        Omega = E[ q q^T ],  q = psi(z) (y - g(theta; x))
      - step-size exponent a in (1/2, 1)  (default t^{-0.75})
      - projection onto a compact set  Pi_Theta
      - Polyak–Ruppert (tail) averaging of the iterates
      - vanishing ridge  lambda_t = lambda_0 / t^{reg_decay} -> 0

    Update (descent form; the '+' goes with q = psi(y - g)):

        theta_t = Pi_Theta[ theta_{t-1} + gamma_t A_{t-1} q_t ],
        A_{t-1} = (Jbar^T W Jbar + lambda_t I)^{-1} Jbar^T W,
        Jbar    = running average of psi(z) grad g(theta; x)^T.

    Since q = -m (m = psi(g - y), Sieve1's moment) and Jbar = E[psi grad g^T],
    this is identical to Sieve1's theta - alpha A m.  A_{t-1} uses past
    samples only, so E[A_{t-1} q_t | F_{t-1}] = A_{t-1} m_K(theta_{t-1}).
    """

    def __init__(
        self,
        config: SimulationConfig,
        model: BaseModel | None = None,
        seed: int | None = None,
        degree: int | None = None,
        basis: str | None = None,
        B: int | None = None,
        reg: float | None = None,
        reg_decay: float | None = None,
        clip: float | None = None,
        W_type: str | None = None,
        ema: float | None = None,
        proj_radius: float | None = None,
        average: bool | None = None,
        init_theta: np.ndarray | None = None,
        start_iter: int = 0,
    ):
        super().__init__(config, model, seed, init_theta=init_theta,
                         start_iter=start_iter)
        self.degree = degree if degree is not None else getattr(
            config, "sieve2_degree", 2)
        self.basis = basis if basis is not None else getattr(
            config, "sieve2_basis", "poly")
        self.B = B if B is not None else getattr(config, "sieve2_B", 1)
        self.reg = reg if reg is not None else getattr(
            config, "sieve2_reg", 1e-2)
        self.reg_decay = reg_decay if reg_decay is not None else getattr(
            config, "sieve2_reg_decay", 0.25)
        self.clip = clip if clip is not None else getattr(
            config, "sieve2_clip", 10.0)
        self.w_type = W_type if W_type is not None else getattr(
            config, "sieve2_W_type", "full")
        self.ema = ema if ema is not None else getattr(config, "sieve2_ema", 0.0)
        self.proj_radius = proj_radius if proj_radius is not None else getattr(
            config, "sieve2_proj_radius", 10.0)
        self.average = average if average is not None else getattr(
            config, "sieve2_average", True)

        if self.degree not in (1, 2):
            raise ValueError(
                f"Sieve2 supports degree 1 or 2, got {self.degree}")
        if self.basis not in ("poly", "hermite"):
            raise ValueError(
                f"Unknown sieve basis '{self.basis}' (use 'poly' or 'hermite')")
        if self.w_type not in ("full", "diag", "identity"):
            raise ValueError(
                f"Unknown sieve W_type '{self.w_type}' "
                f"(use 'full', 'diag' or 'identity')")

        self.include_intercept = True
        self._p = self._feature_dim(config.d_z)
        self._M_bar = np.zeros((self._p, config.d_theta))
        self._Omega_bar = np.zeros((self._p, self._p))
        self._warned_rank = False
        self.theta_bar = self.theta.copy()   # Polyak–Ruppert average
        self._avg_count = 0

    # ------------------------------------------------------------------
    # Sieve / preconditioner helpers
    # ------------------------------------------------------------------

    def _feature_dim(self, d_z: int) -> int:
        p = d_z  # linear terms
        if self.degree >= 2:
            if self.basis == "hermite":
                p += d_z + d_z * (d_z - 1) // 2   # H_2 + cross terms
            else:
                p += d_z * (d_z + 1) // 2         # symmetric quadratics
        return p + 1  # intercept column

    def _features(self, z: np.ndarray) -> np.ndarray:
        if self.basis == "hermite":
            return _sieve_hermite_features(
                z, self.degree, include_intercept=self.include_intercept)
        return _sieve_poly_features(
            z, self.degree, include_intercept=self.include_intercept)

    def _precond_rate(self) -> float:
        if self.ema and self.ema > 0.0:
            return self.ema
        return 1.0 / max(1, self.t)

    def _ridge(self) -> float:
        """Vanishing ridge: lambda_t = lambda_0 / t^{reg_decay} -> 0."""
        return self.reg / (self.t ** self.reg_decay)

    def _weight_matrix(self) -> np.ndarray:
        """Weighting matrix W (p, p), built from past samples only."""
        lam = self._ridge()
        if self.w_type == "identity":
            return np.eye(self._p)
        if self.w_type == "diag":
            diag = np.diag(self._Omega_bar)
            return np.diag(1.0 / (diag + lam))
        # full: W = (Omega + lambda I)^{-1}
        return np.linalg.solve(self._Omega_bar + lam * np.eye(self._p),
                               np.eye(self._p))

    def _history_theta(self, theta: np.ndarray) -> np.ndarray:
        """Record the Polyak–Ruppert average when averaging is enabled."""
        return self.theta_bar if self.average else theta

    # ------------------------------------------------------------------
    # BaseIVAlgorithm interface
    # ------------------------------------------------------------------

    def _get_lr0(self) -> float:
        return getattr(self.config, "sieve2_lr", 0.1)

    @property
    def samples_per_step(self) -> int:
        return self.B

    def _get_lr_decay(self) -> float:
        return getattr(self.config, "sieve2_lr_decay", 0.75)

    def _step(
        self, generator: IVDataGenerator, alpha: float
    ) -> tuple[np.ndarray, float]:
        """Sieve2 single-step update (B fresh samples for the moment)."""
        B = self.B
        z, x, y = generator.generate_batch(B)

        psi = self._features(z)                       # (B, p)
        pred = self.model.predict(self.theta, x)      # (B, 1)
        eps = y - pred                                 # (B, 1)  y - g
        grad = self.model.gradient(self.theta, x)     # (B, d_theta)

        # -- current-sample moment (q_t = psi^T (y - g) / B) ---
        q_t = (psi.T @ eps) / B                        # (p, 1)

        # -- preconditioner from past samples only (predictable given F_{t-1})
        W = self._weight_matrix()                      # (p, p)
        WM = W @ self._M_bar                           # (p, d)
        MtWM = self._M_bar.T @ WM                      # (d, d)
        MtW = WM.T                                     # (d, p)
        d, p = MtW.shape
        lam = self._ridge()
        if d <= p:
            reg_eff = lam * max(1.0, float(np.trace(MtWM)) / d)
            A = np.linalg.solve(MtWM + reg_eff * np.eye(d), MtW)   # (d, p)
        else:
            if not self._warned_rank:
                print(f"  [Sieve2] d_theta={d} > p={p}; using gradient "
                      f"direction (no Newton inverse).")
                self._warned_rank = True
            scale = max(1.0, float(np.trace(MtWM)) / d)
            A = MtW / scale

        update = A @ q_t                               # (d, 1)

        # -- numerical safety: cap the per-step displacement
        un = float(np.linalg.norm(update))
        if un > self.clip:
            update = update * (self.clip / un)

        # -- descent step (see docstring for the sign with q = psi(y - g))
        theta_new = self.theta + alpha * update

        # -- projection onto the compact set {||theta|| <= proj_radius}
        if self.proj_radius and self.proj_radius > 0:
            nrm = float(np.linalg.norm(theta_new))
            if nrm > self.proj_radius:
                theta_new = theta_new * (self.proj_radius / nrm)

        loss = float(np.mean(eps ** 2))

        # -- update running statistics with the current sample (after theta)
        J = (psi.T @ grad) / B                         # (p, d)  E[psi grad g^T]
        outer = (psi * eps).T @ (psi * eps) / B        # (p, p)  E[psi psi^T eps^2]
        beta = self._precond_rate()
        self._M_bar = (1.0 - beta) * self._M_bar + beta * J
        self._Omega_bar = (1.0 - beta) * self._Omega_bar + beta * outer

        # -- Polyak–Ruppert averaging
        if self.average:
            self._avg_count += 1
            w = 1.0 / self._avg_count
            self.theta_bar = (1.0 - w) * self.theta_bar + w * theta_new

        return theta_new, loss


# ---------------------------------------------------------------------------
# Sieve3  (two-batch sieve GMM with identity weighting + Polyak-Ruppert avg)
# ---------------------------------------------------------------------------

class Sieve3(BaseIVAlgorithm):
    """Sieve3: two-batch sieve-GMM update with identity weighting.

    See README.  Objective

        F(theta) = E[(Y - g(theta;x)) psi_K(z)^T] W
                   E[psi_K(z) (Y - g(theta;x))],

    whose gradient is (up to constants)

        grad F(theta)
          = E[ psi_K(z1)^T W psi_K(z2) (g(theta;x2) - Y2)
               grad_theta g(theta;x1) ].

    With the orthonormal Hermite basis (and no heteroskedasticity) the optimal
    weighting is W = I, yielding the two-batch stochastic update

        theta_{t+1} = theta_t
                      - alpha_{t+1} M~_{B_M}(theta_t)^T m~_{B_m}(theta_t),

        M~_{B_M}(theta) = (1/B_M) sum_i psi_K(z_i) grad_theta g(theta;x_i)^T,
        m~_{B_m}(theta) = (1/B_m) sum_j psi_K(z_j) (g(theta;x_j) - y_j).

    The Polyak-Ruppert average of the iterates is also recorded:

        theta_bar_{t+1} = (t-1)/t theta_bar_t + (1/t) theta_{t+1}.

    The sieve basis psi_K(z) is a fixed set of instrument functions of z (no
    intercept); no two-sample oracle and no first-stage nuisance model are
    required.  Each step consumes B_M + B_m fresh samples.
    """

    def __init__(
        self,
        config: SimulationConfig,
        model: BaseModel | None = None,
        seed: int | None = None,
        degree: int | None = None,
        basis: str | None = None,
        B_M: int | None = None,
        B_m: int | None = None,
        clip: float | None = None,
        average: bool | None = None,
        init_theta: np.ndarray | None = None,
        start_iter: int = 0,
    ):
        """
        Args:
            config: simulation configuration.
            model: structural model instance.
            seed: independent random seed.
            degree: sieve degree (1 or 2). Defaults to config.sieve3_degree.
            basis: "hermite" (orthonormal for N(0,I)) or "poly".
                Defaults to config.sieve3_basis.
            B_M: batch size for the Jacobian estimate M~.
                Defaults to config.sieve3_B_M.
            B_m: batch size for the moment estimate m~.
                Defaults to config.sieve3_B_m.
            clip: cap on per-step parameter displacement. Defaults to
                config.sieve3_clip.
            average: Polyak-Ruppert averaging of the iterates.
                Defaults to config.sieve3_average.
            init_theta: initial theta (None = random).
            start_iter: starting iteration count for resumed training.
        """
        super().__init__(config, model, seed, init_theta=init_theta,
                         start_iter=start_iter)
        self.degree = degree if degree is not None else getattr(
            config, "sieve3_degree", 2)
        self.basis = basis if basis is not None else getattr(
            config, "sieve3_basis", "hermite")
        self.B_M = B_M if B_M is not None else getattr(config, "sieve3_B_M", 1)
        self.B_m = B_m if B_m is not None else getattr(config, "sieve3_B_m", 1)
        self.clip = clip if clip is not None else getattr(
            config, "sieve3_clip", 10.0)
        self.average = average if average is not None else getattr(
            config, "sieve3_average", True)

        if self.degree not in (1, 2):
            raise ValueError(
                f"Sieve3 supports degree 1 or 2, got {self.degree}")
        if self.basis not in ("poly", "hermite"):
            raise ValueError(
                f"Unknown sieve basis '{self.basis}' (use 'poly' or 'hermite')")
        if self.B_M < 1 or self.B_m < 1:
            raise ValueError(
                f"Sieve3 batch sizes must be >= 1, got B_M={self.B_M}, "
                f"B_m={self.B_m}")

        self.include_intercept = False
        self._p = self._feature_dim(config.d_z)
        self.theta_bar = self.theta.copy()   # Polyak-Ruppert average
        self._avg_count = 0

    # ------------------------------------------------------------------
    # Sieve helpers
    # ------------------------------------------------------------------

    def _feature_dim(self, d_z: int) -> int:
        p = d_z  # linear terms
        if self.degree >= 2:
            if self.basis == "hermite":
                p += d_z + d_z * (d_z - 1) // 2   # H_2(z_i) + cross terms
            else:
                p += d_z * (d_z + 1) // 2         # symmetric quadratics
        return p

    def _features(self, z: np.ndarray) -> np.ndarray:
        if self.basis == "hermite":
            return _sieve_hermite_features(
                z, self.degree, include_intercept=self.include_intercept)
        return _sieve_poly_features(
            z, self.degree, include_intercept=self.include_intercept)

    # ------------------------------------------------------------------
    # BaseIVAlgorithm interface
    # ------------------------------------------------------------------

    def _get_lr0(self) -> float:
        return getattr(self.config, "sieve3_lr", 0.01)

    def _get_lr_decay(self) -> float:
        return getattr(self.config, "sieve3_lr_decay", 0.5)

    @property
    def samples_per_step(self) -> int:
        return self.B_M + self.B_m

    def _history_theta(self, theta: np.ndarray) -> np.ndarray:
        """Record the Polyak-Ruppert average when averaging is enabled."""
        return self.theta_bar if self.average else theta

    def _step(
        self, generator: IVDataGenerator, alpha: float
    ) -> tuple[np.ndarray, float]:
        """Sieve3 single-step update (B_M + B_m fresh samples)."""
        B_M, B_m = self.B_M, self.B_m
        z_all, x_all, y_all = generator.generate_batch(B_M + B_m)

        # Split into the Jacobian (M) batch and the moment (m) batch
        z_M, x_M = z_all[:B_M], x_all[:B_M]
        z_m, x_m, y_m = z_all[B_M:], x_all[B_M:], y_all[B_M:]

        # M~_{B_M} = (1/B_M) sum_i psi(z_i) grad g(theta; x_i)^T   (p, d)
        psi_M = self._features(z_M)
        grad_M = self.model.gradient(self.theta, x_M)
        M_tilde = (psi_M.T @ grad_M) / B_M

        # m~_{B_m} = (1/B_m) sum_j psi(z_j) (g(theta; x_j) - y_j)   (p, 1)
        psi_m = self._features(z_m)
        pred_m = self.model.predict(self.theta, x_m)
        err_m = pred_m - y_m
        m_tilde = (psi_m.T @ err_m) / B_m

        # theta - alpha * M~^T W m~  with W = I
        update = M_tilde.T @ m_tilde                     # (d, 1)

        # Numerical safety: cap the per-step parameter displacement
        un = float(np.linalg.norm(update))
        if un > self.clip:
            update = update * (self.clip / un)

        theta_new = self.theta - alpha * update
        loss = float(np.mean(err_m ** 2))

        # Polyak-Ruppert averaging of the iterates
        if self.average:
            self._avg_count += 1
            w = 1.0 / self._avg_count
            self.theta_bar = (1.0 - w) * self.theta_bar + w * theta_new

        return theta_new, loss


# ---------------------------------------------------------------------------
# GMMExp  --  generic two-batch GMM estimator for ablation studies
# ---------------------------------------------------------------------------

class GMMExp(BaseIVAlgorithm):
    """Generic two-batch online GMM estimator used for ablation studies.

    This single algorithm implements a family of two-batch online GMM
    estimators used for ablation studies.  Three factors vary, and everything
    else is held fixed so the factors are isolated:

      basis      : "lin"   -> psi(z) = z                  (no sieve)
                   "herm1"/"herm2"/"herm3" -> orthonormal Hermite sieve,
                   "poly1"/"poly2" -> polynomial sieve
      precond    : "gd"    -> direction = M_op^T W m_t     (plain gradient)
                   "nt"    -> direction = (M_op^T W M_op + lam I)^{-1}
                                          M_op^T W m_t    (Newton-type)
      w_type     : "identity" -> W = I
                   "inv_var"  -> W = diag(1 / (diag(Om_bar) + lam))
                                 (Sieve1's weighting: dividing by S makes
                                  W an adaptive step-size normaliser)

    plus one structural switch:

      m_source   : "running" -> M_op = Mbar (running average of past batches,
                                              Sieve1/Sieve2 style)
                   "batch"   -> M_op = the current Jacobian batch
                                              (SLIM / Sieve3 style)

    Fixed design decisions (identical for every variant):

      * one step draws ``B_M + B_m`` fresh samples, split into a Jacobian batch
        (``B_M``) and a moment batch (``B_m``), so every variant consumes the
        same number of samples per step (see ``samples_per_step``);
      * ``Mbar`` (p x d) and ``Om_bar`` (p x p, ``E[psi psi^T eps^2]``) are
        running averages of *past* batches only and are updated *after* the
        parameter step, so the update direction is ``F_{t-1}``-measurable and
        (conditionally) unbiased;
      * ``gd`` and ``nt`` reuse the very same ``M_op`` and ``W``; the only
        difference between them is the Newton matrix inverse;
      * ``W`` is used exactly as computed - no rescaling - so its magnitude
        (which carries the adaptive step size in ``inv_var``) is preserved;
      * Polyak-Ruppert averaging is switchable (``average``).

    Any of basis / precond / w_type / m_source / B_M / B_m may be set to a LIST
    in the config (``ALGO_GMMEXP_*``); the runner then enumerates the full
    cross-product, so a single ``ALGO_LIST = ["gmmexp"]`` entry expands into one
    run per combination (result labels:
    ``gmmexp_{basis}_{precond}_{i|invS}``).

    Reproduction of the existing algorithms (also set lr and average):
      First-Order SLIM : basis="lin",   precond="gd", w_type="identity",
                         m_source="batch",   B_M=B_m=8, average=False
      Sieve3           : basis="herm2", precond="gd", w_type="identity",
                         m_source="batch",   B_M=B_m=1, average=True
      Sieve1           : basis="poly2", precond="nt", w_type="inv_var",
                         m_source="running", B_M=B_m=1, average=False
    """

    # --- factor defaults (overridden by kwargs / ALGO_GMMEXP_* config) ---
    basis: str = "lin"
    precond: str = "gd"
    w_type: str = "identity"
    m_source: str = "running"

    # encoded basis -> (feature kind, sieve degree)
    _BASIS_MAP = {
        "lin":   ("lin", 1),
        "herm1": ("herm", 1),
        "herm2": ("herm", 2),
        "herm3": ("herm", 3),
        "poly1": ("poly", 1),
        "poly2": ("poly", 2),
    }

    def __init__(
        self,
        config: SimulationConfig,
        model: BaseModel | None = None,
        seed: int | None = None,
        basis: str | None = None,
        precond: str | None = None,
        w_type: str | None = None,
        m_source: str | None = None,
        B_M: int | None = None,
        B_m: int | None = None,
        clip: float | None = None,
        reg: float | None = None,
        average: bool | None = None,
        init_theta: np.ndarray | None = None,
        start_iter: int = 0,
    ):
        super().__init__(config, model, seed, init_theta=init_theta,
                         start_iter=start_iter)
        self.basis = basis if basis is not None else getattr(
            config, "gmmexp_basis", self.basis)
        self.precond = precond if precond is not None else getattr(
            config, "gmmexp_precond", self.precond)
        self.w_type = w_type if w_type is not None else getattr(
            config, "gmmexp_w_type", self.w_type)
        self.m_source = m_source if m_source is not None else getattr(
            config, "gmmexp_m_source", self.m_source)
        if self.basis not in self._BASIS_MAP:
            raise ValueError(f"Unknown basis '{self.basis}'; "
                             f"valid: {sorted(self._BASIS_MAP)}")
        self.basis_kind, self.degree = self._BASIS_MAP[self.basis]
        self.B_M = B_M if B_M is not None else getattr(config, "gmmexp_B_M", 1)
        self.B_m = B_m if B_m is not None else getattr(config, "gmmexp_B_m", 1)
        self.clip = clip if clip is not None else getattr(
            config, "gmmexp_clip", 10.0)
        self.reg = reg if reg is not None else getattr(
            config, "gmmexp_reg", 1e-2)
        self.average = (average if average is not None
                        else getattr(config, "gmmexp_average", True))

        if self.precond not in ("gd", "nt"):
            raise ValueError(f"Unknown precond '{self.precond}'")
        if self.w_type not in ("identity", "inv_var"):
            raise ValueError(f"Unknown w_type '{self.w_type}' "
                             f"(use 'identity' or 'inv_var')")
        if self.m_source not in ("running", "batch"):
            raise ValueError(f"Unknown m_source '{self.m_source}' "
                             f"(use 'running' or 'batch')")
        if self.B_M < 1 or self.B_m < 1:
            raise ValueError("GMMExp batch sizes must be >= 1")

        self._p = self._feature_dim(config.d_z)
        self._M_bar = np.zeros((self._p, config.d_theta))
        self._Om_bar = np.zeros((self._p, self._p))
        self.theta_bar = self.theta.copy()   # Polyak-Ruppert average
        self._avg_count = 0
        self._warned_rank = False

    # ------------------------------------------------------------------
    # Sieve / weighting helpers
    # ------------------------------------------------------------------

    def _feature_dim(self, d_z: int) -> int:
        if self.basis_kind == "lin" or self.degree == 1:
            return d_z
        if self.basis_kind == "herm":
            p = d_z + d_z + d_z * (d_z - 1) // 2          # H_1 + H_2 + cross
            if self.degree >= 3:
                p += d_z + d_z * (d_z - 1)                # H_3 + H_2*H_1
                p += d_z * (d_z - 1) * (d_z - 2) // 6     # triple products
            return p
        return d_z + d_z * (d_z + 1) // 2             # linear + quadratics

    def _features(self, z: np.ndarray) -> np.ndarray:
        if self.basis_kind == "lin" or self.degree == 1:
            return z                                  # no sieve: psi(z) = z
        if self.basis_kind == "herm":
            return _sieve_hermite_features(z, self.degree,
                                           include_intercept=False)
        return _sieve_poly_features(z, self.degree, include_intercept=False)

    def _weight_matrix(self) -> np.ndarray:
        """W built from PAST running statistics only (measurable w.r.t. F_{t-1}).

        "identity": W = I - no weighting (reproduces SLIM / Sieve3).
        "inv_var" : W = diag(1 / (diag(Om_bar) + lam)) - Sieve1's weighting.
                    Its magnitude is deliberately NOT rescaled: dividing the
                    direction by the moment variance is what turns W into an
                    adaptive step-size normaliser (W ~ 1/E[eps^2]).
        """
        if self.w_type == "identity":
            return np.eye(self._p)
        return np.diag(1.0 / (np.diag(self._Om_bar) + self.reg))

    def _tag(self) -> str:
        return f"gmmexp {self.basis}/{self.precond}/{self.w_type}/{self.m_source}"

    # ------------------------------------------------------------------
    # BaseIVAlgorithm interface
    # ------------------------------------------------------------------

    def _get_lr0(self) -> float:
        override = getattr(self.config, "gmmexp_lr", None)
        if override is not None:
            return float(override)
        # Newton-type steps are well scaled (like Sieve1); plain gradient steps
        # need a conservative step (like Sieve3).
        return 1e-3 if self.precond == "gd" else 0.1

    def _get_lr_decay(self) -> float:
        override = getattr(self.config, "gmmexp_lr_decay", None)
        return float(override) if override is not None else 0.5

    @property
    def samples_per_step(self) -> int:
        return self.B_M + self.B_m

    def _history_theta(self, theta: np.ndarray) -> np.ndarray:
        """Record the Polyak-Ruppert average when averaging is enabled."""
        return self.theta_bar if self.average else theta

    def _step(
        self, generator: IVDataGenerator, alpha: float
    ) -> tuple[np.ndarray, float]:
        """GMMExp single-step update (B_M + B_m fresh samples)."""
        B_M, B_m = self.B_M, self.B_m
        z_all, x_all, y_all = generator.generate_batch(B_M + B_m)
        z_M, x_M = z_all[:B_M], x_all[:B_M]
        z_m, x_m, y_m = z_all[B_M:], x_all[B_M:], y_all[B_M:]

        # Jacobian batch: used to update the running Mbar, and directly as the
        # operator when m_source == "batch" (SLIM / Sieve3).
        psi_M = self._features(z_M)
        grad_M = self.model.gradient(self.theta, x_M)
        J_batch = (psi_M.T @ grad_M) / B_M                # (p, d)

        # Moment batch
        psi_m = self._features(z_m)
        err_m = self.model.predict(self.theta, x_m) - y_m  # (B_m, 1)
        m_t = (psi_m.T @ err_m) / B_m                      # (p, 1)

        # Direction: identical M_op and W for "gd" and "nt"; the only
        # difference is whether the Newton matrix inverse is applied.
        # M_op is the running past average by default, or the current batch
        # when m_source == "batch" (SLIM / Sieve3).
        M_op = J_batch if self.m_source == "batch" else self._M_bar
        W = self._weight_matrix()                          # (p, p)
        MtW = M_op.T @ W                                   # (d, p)

        if self.precond == "nt":
            MtWM = MtW @ M_op                              # (d, d)
            d, p = MtWM.shape
            if d <= p:
                reg_eff = self.reg * max(1.0, float(np.trace(MtWM)) / d)
                A = np.linalg.solve(MtWM + reg_eff * np.eye(d), MtW)
                direction = A @ m_t                        # (d, 1)
            else:
                if not self._warned_rank:
                    print(f"  [{self._tag()}] d_theta={d} > p={p}; using "
                          f"gradient direction (no Newton inverse).")
                    self._warned_rank = True
                scale = max(1.0, float(np.trace(MtWM)) / d)
                direction = (MtW @ m_t) / scale
        else:
            direction = MtW @ m_t                          # (d, 1)

        # Numerical safety: cap the per-step parameter displacement
        un = float(np.linalg.norm(direction))
        if un > self.clip:
            direction = direction * (self.clip / un)

        theta_new = self.theta - alpha * direction
        loss = float(np.mean(err_m ** 2))

        # Running statistics from past batches only (update AFTER the step)
        beta = 1.0 / max(1, self.t)
        self._M_bar = (1.0 - beta) * self._M_bar + beta * J_batch
        outer = (psi_m * err_m).T @ (psi_m * err_m) / B_m   # E[psi psi^T eps^2]
        self._Om_bar = (1.0 - beta) * self._Om_bar + beta * outer

        # Polyak-Ruppert averaging of the iterates
        if self.average:
            self._avg_count += 1
            w = 1.0 / self._avg_count
            self.theta_bar = (1.0 - w) * self.theta_bar + w * theta_new

        return theta_new, loss


# SieveGMM is kept as an alias of Sieve1 for backwards compatibility.
SieveGMM = Sieve1


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
    "sieve1": Sieve1,
    "sieve": Sieve1,
    "sievegmm": Sieve1,
    "sieve_gmm": Sieve1,
    "sieve2": Sieve2,
    "sieve2_gmm": Sieve2,
    "sieve3": Sieve3,
    "sieve3_gmm": Sieve3,
    "gmmexp": GMMExp,
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
