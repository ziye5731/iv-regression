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
    "tosg": "#2196F3",    # blue
    "tosg_ivar": "#2196F3",
    "ostg": "#4CAF50",    # green
    "ostg_ivar": "#4CAF50",
    "slim": "#FF5722",    # orange
    "first_order_slim": "#FF5722",
}

# Palette for multiple SLIM variants (B_M, B_m, W combos)
_SLIM_PALETTE = [
    "#FF5722",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#FF9800",  # amber
    "#E91E63",  # pink
    "#3F51B5",  # indigo
    "#8BC34A",  # light green
    "#795548",  # brown
]

LABELS = {
    "tosg": "TOSG-IVaR",
    "tosg_ivar": "TOSG-IVaR",
    "ostg": "OSTG-IVaR",
    "ostg_ivar": "OSTG-IVaR",
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
):
    """Plot convergence curves for multiple algorithms side-by-side.

    Args:
        all_results: {algo_name: aggregate_repeats() result}.
        figsize: figure size.
        dpi: resolution.
        save_path: path to save the figure.
        use_quantile: if True, shade IQR; else shade mean±std.
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
    ax1.set_title("Parameter error comparison")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Prediction MSE")
    ax2.set_title("Prediction MSE comparison")
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
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(
    all_results: dict[str, dict[str, np.ndarray]],
):
    """Print a final metrics summary table with mean±std and median.

    Args:
        all_results: {algo_name: aggregate_repeats() result}.
    """
    print("\n" + "=" * 80)
    print(f"{'Algorithm':<22} {'Median':>10} {'Mean±Std':>18} {'Pred MSE (median)':>18}")
    print("-" * 80)
    for algo_name, results in all_results.items():
        final_median = results["param_error_median"][-1]
        final_mean = results["param_error_mean"][-1]
        final_std = results["param_error_std"][-1]
        final_mse_median = results["pred_mse_median"][-1]
        label = LABELS.get(algo_name.lower(), algo_name)
        print(
            f"{label:<22} "
            f"{final_median:>8.4f}   "
            f"{final_mean:>8.4f}±{final_std:.4f}   "
            f"{final_mse_median:>12.4f}"
        )
    print("=" * 80 + "\n")
