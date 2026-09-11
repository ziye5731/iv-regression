"""
metrics.py -- Evaluation metrics for IV regression.

Common metrics for assessing parameter estimation quality:
- Parameter error (||theta_hat - theta*||_2)
- Relative error (||theta_hat - theta*||_2 / ||theta*||_2)
- Prediction MSE (on a held-out test set)
"""

import numpy as np

from .config import SimulationConfig
from .data_generator import IVDataGenerator


def parameter_error(theta_hat: np.ndarray, theta_star: np.ndarray) -> float:
    """Compute L2 parameter error ||theta_hat - theta*||_2.

    Args:
        theta_hat: estimated parameter.
        theta_star: true parameter.

    Returns:
        L2 norm error.
    """
    # Flatten both to 1D to avoid (d_x,) - (d_x, 1) → (d_x, d_x) broadcasting bug
    return float(np.linalg.norm(theta_hat.ravel() - theta_star.ravel()))


def relative_error(theta_hat: np.ndarray, theta_star: np.ndarray) -> float:
    """Compute relative L2 error ||theta_hat - theta*||_2 / ||theta*||_2.

    Args:
        theta_hat: estimated parameter.
        theta_star: true parameter.

    Returns:
        Relative error.
    """
    denom = np.linalg.norm(theta_star)
    if denom < 1e-12:
        return float(np.linalg.norm(theta_hat))
    return float(np.linalg.norm(theta_hat - theta_star) / denom)


def prediction_mse(
    theta: np.ndarray,
    generator: IVDataGenerator,
    n_test: int = 2000,
) -> float:
    """Compute prediction MSE on an independent test set.

    The test data is generated with the same DGP (including endogeneity),
    so this measures the overall structural-equation error.

    Args:
        theta: model parameter.
        generator: data generator (used for fresh test data).
        n_test: test set size.

    Returns:
        MSE value.
    """
    _, x_test, y_test = generator.generate_batch(n_test)
    y_pred = generator.model.predict(theta, x_test)
    mse = float(np.mean((y_pred - y_test) ** 2))
    return mse


def evaluate_history(
    history: list[dict],
    config: SimulationConfig,
    generator: IVDataGenerator,
    n_test: int = 2000,
    step_every: int = 1,
) -> dict[str, np.ndarray]:
    """Extract metric sequences from training history.

    For each recorded step in history, compute parameter error and
    prediction MSE. Only records with step % step_every == 0 are kept,
    plus the final step.

    Args:
        history: training history from algorithm.train().
        config: simulation config.
        generator: independent test data generator.
        n_test: test set size.
        step_every: record metrics every N iterations.

    Returns:
        {
            'steps': step numbers (M,),
            'param_error': L2 parameter errors (M,),
            'pred_mse': prediction MSEs (M,),
            'loss': training loss / objective values (M,),
        }
    """
    theta_star = config.theta_star

    steps = []
    param_errors = []
    pred_mses = []
    losses = []

    for record in history:
        if step_every > 1 and record["step"] % step_every != 0:
            if record["step"] != history[-1]["step"]:
                continue
        steps.append(record["step"])
        param_errors.append(parameter_error(record["theta"], theta_star))
        pred_mses.append(prediction_mse(record["theta"], generator, n_test))
        losses.append(float(record.get("loss", np.nan)))

    return {
        "steps": np.array(steps),
        "param_error": np.array(param_errors),
        "pred_mse": np.array(pred_mses),
        "loss": np.array(losses),
    }


def aggregate_repeats(
    all_histories: list[list[dict]],
    config: SimulationConfig,
    generator: IVDataGenerator,
    n_test: int = 2000,
    step_every: int = 1,
) -> dict[str, np.ndarray]:
    """Aggregate metrics across repeated runs (mean ± std).

    Args:
        all_histories: list of histories, one per repeat.
        config: simulation config.
        generator: test data generator.
        n_test: test set size.
        step_every: record metrics every N iterations.

    Returns:
        {
            'steps': common steps (M,),
            'param_error_mean': (M,), 'param_error_std': (M,),
            'param_error_median': (M,), 'param_error_q25': (M,), 'param_error_q75': (M,),
            'pred_mse_mean': (M,), 'pred_mse_std': (M,),
            'pred_mse_median': (M,), 'pred_mse_q25': (M,), 'pred_mse_q75': (M,),
            'loss_mean': (M,), 'loss_std': (M,),
            'loss_median': (M,), 'loss_q25': (M,), 'loss_q75': (M,),
        }
    """
    # Align to shortest history
    min_len = min(len(h) for h in all_histories)
    aligned = [h[:min_len] for h in all_histories]

    evals = [
        evaluate_history(h, config, generator, n_test, step_every)
        for h in aligned
    ]

    common_steps = evals[0]["steps"]

    param_err = np.array([e["param_error"] for e in evals])  # (n_repeats, M)
    pred_mse = np.array([e["pred_mse"] for e in evals])
    loss = np.array([e["loss"] for e in evals])              # (n_repeats, M)

    return {
        "steps": common_steps,
        "param_error_mean": param_err.mean(axis=0),
        "param_error_std": param_err.std(axis=0),
        "param_error_median": np.median(param_err, axis=0),
        "param_error_q25": np.quantile(param_err, 0.25, axis=0),
        "param_error_q75": np.quantile(param_err, 0.75, axis=0),
        "pred_mse_mean": pred_mse.mean(axis=0),
        "pred_mse_std": pred_mse.std(axis=0),
        "pred_mse_median": np.median(pred_mse, axis=0),
        "pred_mse_q25": np.quantile(pred_mse, 0.25, axis=0),
        "pred_mse_q75": np.quantile(pred_mse, 0.75, axis=0),
        # Loss is algorithm-specific and may be NaN or non-positive (e.g. the
        # centered distance-covariance objective), so use NaN-aware statistics.
        "loss_mean": np.nanmean(loss, axis=0),
        "loss_std": np.nanstd(loss, axis=0),
        "loss_median": np.nanmedian(loss, axis=0),
        "loss_q25": np.nanquantile(loss, 0.25, axis=0),
        "loss_q75": np.nanquantile(loss, 0.75, axis=0),
    }
