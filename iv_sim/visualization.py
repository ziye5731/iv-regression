from __future__ import annotations

"""
visualization.py -- Plotting utilities for IV regression simulations.

Plots include:

1. Parameter error convergence curves (with std shading)
2. Prediction MSE convergence curves (with std shading)
3. Training loss / objective convergence curves (with std shading)
4. Multi-algorithm comparison plots
"""

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

COLORS = {
    # TOSG/OTSG/DCOV: warm family
    "tosg": "#FF9646",       # orange
    "tosg_ivar": "#FF9646",
    "otsg": "#D45C5C",       # dark red
    "otsg_ivar": "#810000",
    "dcov": "#FF4081",       # purple
    "dco": "#FF4081",
    "distance_cov": "#7B1FA2",
    "dcov3": "#DF94FF",
    "dcov4": "#7B1FA2",    # magenta / 桃红色
    "sieve": "#088122",
    "sieve1": "#088122",
    "sieve2": "#B5FF35",
    "sieve3": "#00B0FF",     # light blue
    # SLIM default fallback
    "slim": "#2979FF",       # bright blue
    "first_order_slim": "#2979FF",
}

# Palette for SLIM variants: cool, spread across blue-cyan-teal-green-lime-indigo
_SLIM_PALETTE = [
    "#2979FF",  # bright blue
    "#1A237E",  # dark indigo
    "#61EDFF",  # cyan
    "#3BFF41",  # green
    "#009688",  # teal
    "#1F4500",
    "#B5FF35",  # light blue
    "#DD00FF",
]

LABELS = {
    "tosg": "TOSG-IVaR",
    "tosg_ivar": "TOSG-IVaR",
    "otsg": "OTSG-IVaR",
    "otsg_ivar": "OTSG-IVaR",
    "dcov": "DCOV",
    "dco": "DCOV",
    "distance_cov": "DCOV",
    "dcov3": "DCOV3",
    "dcov4": "DCOV4",
    "slim": "First-Order SLIM",
    "first_order_slim": "First-Order SLIM",
    "sieve": "Sieve1",
    "sieve1": "Sieve1",
    "sieve2": "Sieve2",
    "sieve3": "Sieve3",
}

# Ablation palette for the gmmexp cross-product runs
_GMMEXP_PALETTE = [plt.get_cmap("tab20")(i / 20.0) for i in range(18)]
_gmmexp_variant_counter: dict[str, int] = {}

FIG_SIZE = (10, 5)
DPI = 120

# Track SLIM variant index for color assignment
_slim_variant_counter: dict[str, int] = {}


def _get_color_and_label(algo_name: str) -> tuple[str, str]:
    """Get color and display label for an algorithm.

    For SLIM variants (names starting with 'slim_'), auto-assigns
    distinct colors from a palette.
    """
    key = algo_name.lower()
    if key in COLORS:
        return COLORS[key], LABELS.get(key, algo_name)
    # SLIM variant: slim_B{M}_m{m}_{w}
    if key.startswith("slim_"):
        if key not in _slim_variant_counter:
            _slim_variant_counter[key] = len(_slim_variant_counter)
        idx = _slim_variant_counter[key]
        color = _SLIM_PALETTE[idx % len(_SLIM_PALETTE)]
        # Parse: slim_B8_m8_id → "SLIM(B=8, m=8, id)"
        parts = algo_name[5:].split("_")
        formatted = []
        for p in parts:
            if len(p) > 1 and p[0].isalpha() and p[1:].isdigit():
                formatted.append(f"{p[0]}={p[1:]}")
            else:
                formatted.append(p)
        label = "SLIM(" + ", ".join(formatted) + ")"
        return color, label
    # gmmexp_{basis}_{precond}_{weight}[_B{BM}m{Bm}] ablation runs
    if key.startswith("gmmexp"):
        if key not in _gmmexp_variant_counter:
            _gmmexp_variant_counter[key] = len(_gmmexp_variant_counter)
        idx = _gmmexp_variant_counter[key]
        suffix = key[len("gmmexp"):].strip("_")
        label = f"GMMEXP({suffix.replace('_', ',')})" if suffix else "GMMEXP"
        return _GMMEXP_PALETTE[idx % len(_GMMEXP_PALETTE)], label
    return "#333333", algo_name


# ---------------------------------------------------------------------------
# Single-algorithm convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(
    results: dict[str, np.ndarray],
    algo_name: str = "",
    figsize: tuple = FIG_SIZE,
    dpi: int = DPI,
    save_path: str | None = None,
    use_quantile: bool = True,
):
    """Plot parameter error and prediction MSE convergence for one algorithm.

    Args:
        results: dictionary from aggregate_repeats().
        algo_name: algorithm name (for legend).
        figsize: figure size.
        dpi: resolution.
        save_path: path to save the figure (None = don't save).
        use_quantile: if True, shade IQR (25-75); else shade mean±std.
    """
    color, label = _get_color_and_label(algo_name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # Decide which band to plot
    if use_quantile and "param_error_q25" in results:
        lo1, hi1 = results["param_error_q25"], results["param_error_q75"]
        lo2, hi2 = results["pred_mse_q25"], results["pred_mse_q75"]
        band_label = "IQR (25–75)"
    else:
        lo1 = results["param_error_mean"] - results["param_error_std"]
        hi1 = results["param_error_mean"] + results["param_error_std"]
        lo2 = results["pred_mse_mean"] - results["pred_mse_std"]
        hi2 = results["pred_mse_mean"] + results["pred_mse_std"]
        band_label = "mean ± std"

    # Use median for the line if available, else mean
    line1 = results.get("param_error_median", results["param_error_mean"])
    line2 = results.get("pred_mse_median", results["pred_mse_mean"])

    # --- Left: parameter L2 error ---
    ax1.plot(
        results["steps"], line1,
        color=color, linewidth=1.5, label=label,
    )
    ax1.fill_between(
        results["steps"], lo1, hi1,
        color=color, alpha=0.15, label=band_label,
    )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"Parameter error $\|\hat{\theta} - \theta^*\|_2$")
    ax1.set_title("Parameter estimation error")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Right: prediction MSE ---
    ax2.plot(
        results["steps"], line2,
        color=color, linewidth=1.5, label=label,
    )
    ax2.fill_between(
        results["steps"], lo2, hi2,
        color=color, alpha=0.15, label=band_label,
    )
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Prediction MSE")
    ax2.set_title("Prediction MSE convergence")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Multi-algorithm comparison plot
# ---------------------------------------------------------------------------

def plot_comparison(
    all_results: dict[str, dict[str, np.ndarray]],
    figsize: tuple = (18, 5),
    dpi: int = DPI,
    save_path: str | None = None,
    use_quantile: bool = True,
    x_scale: str = "log",
    title: str = "Same iterations",
):
    """Plot convergence curves for multiple algorithms side-by-side.

    Args:
        all_results: {algo_name: aggregate_repeats() result}.
        figsize: figure size.
        dpi: resolution.
        save_path: path to save the figure.
        use_quantile: if True, shade IQR; else shade mean±std.
        x_scale: "log" or "linear" for x-axis scale.
        title: subtitle for the figure.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    for algo_name, results in all_results.items():
        color, label = _get_color_and_label(algo_name)

        # Choose band and line
        if use_quantile and "param_error_q25" in results:
            lo1, hi1 = results["param_error_q25"], results["param_error_q75"]
            lo2, hi2 = results["pred_mse_q25"], results["pred_mse_q75"]
        else:
            lo1 = results["param_error_mean"] - results["param_error_std"]
            hi1 = results["param_error_mean"] + results["param_error_std"]
            lo2 = results["pred_mse_mean"] - results["pred_mse_std"]
            hi2 = results["pred_mse_mean"] + results["pred_mse_std"]
        # Use median for the central curve if available
        line1 = results.get("param_error_median", results["param_error_mean"])
        line2 = results.get("pred_mse_median", results["pred_mse_mean"])

        # Parameter error
        ax1.plot(
            results["steps"], line1,
            color=color, linewidth=1.5, label=label,
        )
        ax1.fill_between(
            results["steps"], lo1, hi1,
            color=color, alpha=0.1,
        )

        # Prediction MSE
        ax2.plot(
            results["steps"], line2,
            color=color, linewidth=1.5, label=label,
        )
        ax2.fill_between(
            results["steps"], lo2, hi2,
            color=color, alpha=0.1,
        )

        # Training loss / objective (algorithm-specific; skip if unavailable)
        if "loss_mean" in results or "loss_median" in results:
            if use_quantile and "loss_q25" in results:
                lo3, hi3 = results["loss_q25"], results["loss_q75"]
            else:
                lo3 = results["loss_mean"] - results["loss_std"]
                hi3 = results["loss_mean"] + results["loss_std"]
            line3 = results.get("loss_median", results["loss_mean"])
            ax3.plot(
                results["steps"], line3,
                color=color, linewidth=1.5, label=label,
            )
            ax3.fill_between(
                results["steps"], lo3, hi3,
                color=color, alpha=0.1,
            )

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"Parameter error $\|\hat{\theta} - \theta^*\|_2$")
    ax1.set_title(f"Parameter error ({title})")
    ax1.set_yscale("log")
    ax1.set_xscale(x_scale)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Prediction MSE")
    ax2.set_title(f"Prediction MSE ({title})")
    ax2.set_yscale("log")
    ax2.set_xscale(x_scale)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Loss")
    ax3.set_title(f"Loss ({title})")
    # symlog (not log) so non-positive objectives such as DCOV stay visible
    ax3.set_yscale("symlog", linthresh=1e-3)
    ax3.set_xscale(x_scale)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Same-samples comparison plot
# ---------------------------------------------------------------------------

def plot_comparison_by_samples(
    all_results: dict[str, dict[str, np.ndarray]],
    samples_per_step: dict[str, int],
    figsize: tuple = (18, 5),
    dpi: int = DPI,
    save_path: str | None = None,
    use_quantile: bool = True,
    x_scale: str = "log",
):
    """Plot convergence vs *samples seen* (not iterations).

    Each algorithm is truncated to the minimum sample budget across all
    methods.  Batch methods that see more samples per step are interpolated
    onto the common sample-count grid for fair visual comparison.

    Args:
        all_results: {algo_name: aggregate_repeats() result}.
        samples_per_step: {algo_name: int} samples consumed per iteration.
        figsize, dpi, save_path, use_quantile: same as plot_comparison.
        x_scale: "log" or "linear" for x-axis scale.
    """
    if not samples_per_step:
        return  # nothing to compare

    # Minimum samples per step (baseline: online, 1 sample/step)
    min_sp = min(samples_per_step.values())

    # Determine the common sample-count grid.
    # All algorithms ran for the same total number of iterations and recorded
    # history at the same steps, so the shared sample budget is the one of the
    # cheapest method:  max_samples = (final recorded step) * min_sp.
    # NOTE: the budget must use the *final step value*, not the number of
    # recorded points.  When history is stored at log-spaced checkpoints
    # (X_AXIS_SCALE = "log") these differ by orders of magnitude, and using
    # len(steps) wrongly collapses the whole plot to the first few samples.
    first_key = next(iter(all_results))
    steps_common = all_results[first_key]["steps"]
    max_samples = int(steps_common[-1]) * min_sp
    # Common grid: the baseline's sample count at each recorded step.
    # Works both for dense histories and for log-spaced checkpoints.
    common_grid = steps_common.astype(float) * min_sp

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    for algo_name, results in all_results.items():
        color, label = _get_color_and_label(algo_name)
        sp = samples_per_step.get(algo_name, 1)

        # Sample counts at each step: [sp, 2*sp, ..., n_iter*sp]
        sample_counts = results["steps"].astype(float) * sp
        # How many steps to keep (up to max_samples)
        keep = sample_counts <= max_samples
        if not np.any(keep):
            continue
        sample_counts = sample_counts[keep]

        # Choose band and line
        if use_quantile and "param_error_q25" in results:
            lo1_raw = results["param_error_q25"][keep]
            hi1_raw = results["param_error_q75"][keep]
            lo2_raw = results["pred_mse_q25"][keep]
            hi2_raw = results["pred_mse_q75"][keep]
        else:
            lo1_raw = (results["param_error_mean"] - results["param_error_std"])[keep]
            hi1_raw = (results["param_error_mean"] + results["param_error_std"])[keep]
            lo2_raw = (results["pred_mse_mean"] - results["pred_mse_std"])[keep]
            hi2_raw = (results["pred_mse_mean"] + results["pred_mse_std"])[keep]
        # Use median for the raw line values when plotting by samples if available
        line1_raw = results.get("param_error_median", results["param_error_mean"])[keep]
        line2_raw = results.get("pred_mse_median", results["pred_mse_mean"])[keep]

        # Loss (algorithm-specific; skip if unavailable)
        has_loss = "loss_mean" in results or "loss_median" in results
        if has_loss:
            if use_quantile and "loss_q25" in results:
                lo3_raw = results["loss_q25"][keep]
                hi3_raw = results["loss_q75"][keep]
            else:
                lo3_raw = (results["loss_mean"] - results["loss_std"])[keep]
                hi3_raw = (results["loss_mean"] + results["loss_std"])[keep]
            line3_raw = results.get("loss_median", results["loss_mean"])[keep]

        # Interpolate to common grid
        if sp == min_sp:
            line1, line2 = line1_raw, line2_raw
            lo1, hi1 = lo1_raw, hi1_raw
            lo2, hi2 = lo2_raw, hi2_raw
            if has_loss:
                line3, lo3, hi3 = line3_raw, lo3_raw, hi3_raw
            x_vals = sample_counts
        else:
            line1 = np.interp(common_grid, sample_counts, line1_raw)
            line2 = np.interp(common_grid, sample_counts, line2_raw)
            lo1 = np.interp(common_grid, sample_counts, lo1_raw)
            hi1 = np.interp(common_grid, sample_counts, hi1_raw)
            lo2 = np.interp(common_grid, sample_counts, lo2_raw)
            hi2 = np.interp(common_grid, sample_counts, hi2_raw)
            if has_loss:
                line3 = np.interp(common_grid, sample_counts, line3_raw)
                lo3 = np.interp(common_grid, sample_counts, lo3_raw)
                hi3 = np.interp(common_grid, sample_counts, hi3_raw)
            x_vals = common_grid

        ax1.plot(x_vals, line1, color=color, linewidth=1.5, label=label)
        ax1.fill_between(x_vals, lo1, hi1, color=color, alpha=0.1)

        ax2.plot(x_vals, line2, color=color, linewidth=1.5, label=label)
        ax2.fill_between(x_vals, lo2, hi2, color=color, alpha=0.1)

        if has_loss:
            ax3.plot(x_vals, line3, color=color, linewidth=1.5, label=label)
            ax3.fill_between(x_vals, lo3, hi3, color=color, alpha=0.1)

    ax1.set_xlabel("Samples seen")
    ax1.set_ylabel(r"Parameter error $\|\hat{\theta} - \theta^*\|_2$")
    ax1.set_title("Parameter error (same samples)")
    ax1.set_yscale("log")
    ax1.set_xscale(x_scale)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Samples seen")
    ax2.set_ylabel("Prediction MSE")
    ax2.set_title("Prediction MSE (same samples)")
    ax2.set_yscale("log")
    ax2.set_xscale(x_scale)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel("Samples seen")
    ax3.set_ylabel("Loss")
    ax3.set_title("Loss (same samples)")
    # symlog (not log) so non-positive objectives such as DCOV stay visible
    ax3.set_yscale("symlog", linthresh=1e-3)
    ax3.set_xscale(x_scale)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(
    all_results: dict[str, dict[str, np.ndarray]],
    config=None,
):
    """Print a final metrics summary table.

    Delegates formatting to the DGP descriptor so the table layout
    adapts automatically (e.g. MSE-only for DeepGMM / MLP).

    Args:
        all_results: {algo_name: aggregate_repeats() result}.
        config: optional SimulationConfig; used to look up the DGP descriptor.
    """
    from .dgp import get_dgp

    if config is not None:
        dgp = get_dgp(config.dgp_mode)
    else:
        # Fallback: use TOSG (default layout)
        dgp = get_dgp("tosg")

    print("\n" + "=" * 80)
    print(dgp.summary_header())
    print("-" * 80)
    for algo_name, results in all_results.items():
        print(dgp.summary_row(algo_name, results))
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# MSE-only comparison plot (for DeepGMM -- unknown structural form)
# ---------------------------------------------------------------------------

def plot_comparison_mse_only(
    all_results: dict[str, dict[str, np.ndarray]],
    samples_per_step: dict[str, int] | None = None,
    figsize: tuple = (8, 5),
    dpi: int = DPI,
    save_path: str | None = None,
    use_quantile: bool = True,
    x_scale: str = "log",
    title: str = "Same iterations",
    by_samples: bool = False,
):
    """Plot prediction MSE convergence only (no parameter error).

    Used for DeepGMM DGP where the true structural form is unknown and
    parameter distance to truth is meaningless.

    Args:
        all_results: {algo_name: aggregate_repeats() result}.
        samples_per_step: if by_samples=True, {algo_name: int} samples per iter.
        figsize: figure size.
        dpi: resolution.
        save_path: path to save the figure.
        use_quantile: if True, shade IQR; else shade mean±std.
        x_scale: "log" or "linear" for x-axis scale.
        title: subtitle for the figure.
        by_samples: if True, x-axis is samples seen (not iterations).
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    if by_samples and samples_per_step:
        min_sp = min(samples_per_step.values())
        first_key = next(iter(all_results))
        steps_common = all_results[first_key]["steps"]
        # Budget = (final recorded step) * min_sp.  Use the step *value*, not
        # len(steps): with log-spaced checkpoints len(steps) << last step and
        # the previous code collapsed the plot to the first few samples.
        max_samples = int(steps_common[-1]) * min_sp
        common_grid = steps_common.astype(float) * min_sp
    else:
        by_samples = False

    for algo_name, results in all_results.items():
        color, label = _get_color_and_label(algo_name)

        if by_samples and samples_per_step:
            sp = samples_per_step.get(algo_name, 1)
            sample_counts = results["steps"].astype(float) * sp
            keep = sample_counts <= max_samples
            if not np.any(keep):
                continue
            sample_counts = sample_counts[keep]
            line_raw = results.get("pred_mse_median", results["pred_mse_mean"])[keep]
            if use_quantile and "pred_mse_q25" in results:
                lo_raw = results["pred_mse_q25"][keep]
                hi_raw = results["pred_mse_q75"][keep]
            else:
                lo_raw = (results["pred_mse_mean"] - results["pred_mse_std"])[keep]
                hi_raw = (results["pred_mse_mean"] + results["pred_mse_std"])[keep]
            if sp == min_sp:
                line, lo, hi = line_raw, lo_raw, hi_raw
                x_vals = sample_counts
            else:
                line = np.interp(common_grid, sample_counts, line_raw)
                lo = np.interp(common_grid, sample_counts, lo_raw)
                hi = np.interp(common_grid, sample_counts, hi_raw)
                x_vals = common_grid
        else:
            if use_quantile and "pred_mse_q25" in results:
                lo = results["pred_mse_q25"]
                hi = results["pred_mse_q75"]
            else:
                lo = results["pred_mse_mean"] - results["pred_mse_std"]
                hi = results["pred_mse_mean"] + results["pred_mse_std"]
            line = results.get("pred_mse_median", results["pred_mse_mean"])
            x_vals = results["steps"]

        ax.plot(x_vals, line, color=color, linewidth=1.5, label=label)
        ax.fill_between(x_vals, lo, hi, color=color, alpha=0.1)

    if by_samples:
        ax.set_xlabel("Samples seen")
    else:
        ax.set_xlabel("Iteration")
    ax.set_ylabel("Prediction MSE")
    ax.set_title(f"Prediction MSE ({title})")
    ax.set_yscale("log")
    ax.set_xscale(x_scale)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# DeepGMM: h* vs h comparison plot
# ---------------------------------------------------------------------------

def plot_h_star_vs_h(
    h_star_func,
    model,
    theta_map: dict[str, np.ndarray],
    x_range: tuple[float, float] = (-4, 4),
    n_points: int = 500,
    figsize: tuple = (10, 6),
    dpi: int = DPI,
    save_path: str | None = None,
):
    """Plot true h* versus learned h for multiple algorithms.

    Args:
        h_star_func: callable that computes the true y = h*(x).
        model: the MLP model with a predict method.
        theta_map: mapping from algorithm label to final theta parameters.
        x_range: (x_min, x_max) for plotting.
        n_points: number of points to evaluate.
        figsize: figure size.
        dpi: resolution.
        save_path: path to save the figure (None = don't save).
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x_vals = np.linspace(x_range[0], x_range[1], n_points).reshape(-1, 1)
    y_true = h_star_func(x_vals)
    ax.plot(x_vals, y_true, color="#2979FF", linewidth=2.5, label="$h^*$ (true)")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = iter(colors)
    for algo_name, theta in theta_map.items():
        color = next(color_cycle, None)
        if color is None:
            color_cycle = iter(colors)
            color = next(color_cycle)
        y_pred = model.predict(theta.reshape(-1, 1), x_vals)
        ax.plot(
            x_vals,
            y_pred,
            color=color,
            linewidth=1.8,
            linestyle="-",
            alpha=0.85,
            label=f"{algo_name}",
        )

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title("True vs Learned Structural Function (DeepGMM)", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()
