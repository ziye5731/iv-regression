from __future__ import annotations

"""
visualization.py -- Plotting utilities for IV regression simulations.

Plots include:

1. Parameter error convergence curves (with std shading)
2. Prediction MSE convergence curves (with std shading)
3. Multi-algorithm comparison plots
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
    "otsg": "#810000",       # dark red
    "otsg_ivar": "#810000",
    "dcov": "#7B1FA2",       # purple
    "dco": "#7B1FA2",
    "distance_cov": "#7B1FA2",
    "dcov3": "#DF94FF",      
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
    "otsg": "OTSG",
    "otsg_ivar": "OTSG",
    "dcov": "DCOV",
    "dco": "DCOV",
    "distance_cov": "DCOV",
    "dcov3": "DCOV3",
    "dcov4": "DCOV4",
    "slim": "First-Order SLIM",
    "first_order_slim": "First-Order SLIM",
}

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
    figsize: tuple = (12, 5),
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

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
    figsize: tuple = (12, 5),
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

    # Determine the common sample-count grid
    # All algorithms ran for the same iterations, so each has the same
    # number of history points.  The baseline (min_sp) sees:
    #   max_samples = n_iter * min_sp
    first_key = next(iter(all_results))
    n_iter = len(all_results[first_key]["steps"])
    max_samples = n_iter * min_sp
    # Common grid: every `min_sp` samples (matching the baseline's steps)
    common_grid = np.arange(min_sp, max_samples + 1, min_sp, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

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
        line1_raw = results.get("param_error_median", results["param_error_mean"])[keep]
        line2_raw = results.get("pred_mse_median", results["pred_mse_mean"])[keep]

        # Interpolate to common grid
        if sp == min_sp:
            line1 = line1_raw
            line2 = line2_raw
            lo1, hi1 = lo1_raw, hi1_raw
            lo2, hi2 = lo2_raw, hi2_raw
            x_vals = sample_counts
        else:
            line1 = np.interp(common_grid, sample_counts, line1_raw)
            line2 = np.interp(common_grid, sample_counts, line2_raw)
            lo1 = np.interp(common_grid, sample_counts, lo1_raw)
            hi1 = np.interp(common_grid, sample_counts, hi1_raw)
            lo2 = np.interp(common_grid, sample_counts, lo2_raw)
            hi2 = np.interp(common_grid, sample_counts, hi2_raw)
            x_vals = common_grid

        ax1.plot(x_vals, line1, color=color, linewidth=1.5, label=label)
        ax1.fill_between(x_vals, lo1, hi1, color=color, alpha=0.1)

        ax2.plot(x_vals, line2, color=color, linewidth=1.5, label=label)
        ax2.fill_between(x_vals, lo2, hi2, color=color, alpha=0.1)

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
        n_iter = len(all_results[first_key]["steps"])
        max_samples = n_iter * min_sp
        common_grid = np.arange(min_sp, max_samples + 1, min_sp, dtype=float)
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
