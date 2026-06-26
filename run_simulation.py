#!/usr/bin/env python3
from __future__ import annotations

"""
run_simulation.py -- Main entry point for IV regression simulations.

Runs TOSG-IVaR and First-Order SLIM comparison experiments.

Usage:
    python run_simulation.py                         # default (linear)
    python run_simulation.py --help                  # show all options
    python run_simulation.py --model quadratic        # quadratic model
    python run_simulation.py --model poly2 --dim 3    # polynomial degree-2
    python run_simulation.py --epochs 200 --repeats 20
    python run_simulation.py --algo tosg --model linear
    python run_simulation.py --save results/comp.png
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np

from iv_sim.config import SimulationConfig
from iv_sim.data_generator import IVDataGenerator
from iv_sim.algorithms import TOSGIVaR, FirstOrderSLIM, OSTGIVaR
from iv_sim.metrics import aggregate_repeats
from iv_sim.visualization import plot_comparison, print_summary_table


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single_experiment(
    config: SimulationConfig,
    algo_name: str,
    seed: int,
    slim_kwargs: Optional[dict] = None,
) -> list[dict]:
    """Run one complete training run.

    Args:
        config: simulation configuration.
        algo_name: algorithm name ("tosg", "ostg", or "slim").
        seed: random seed for this run.
        slim_kwargs: optional dict with B_M, B_m, W_type for SLIM.

    Returns:
        Training history list.
    """
    # Independent data generator for this run
    generator = IVDataGenerator(config, seed=seed)

    # Instantiate algorithm
    if algo_name.lower() in ("tosg", "tosg_ivar"):
        algo = TOSGIVaR(config, seed=seed + 1000)
    elif algo_name.lower() in ("ostg", "ostg_ivar"):
        algo = OSTGIVaR(config, seed=seed + 1000)
    elif algo_name.lower() in ("slim", "first_order_slim"):
        sk = slim_kwargs or {}
        algo = FirstOrderSLIM(
            config,
            seed=seed + 1000,
            B_M=sk.get("B_M", config.slim_B_M),
            B_m=sk.get("B_m", config.slim_B_m),
            W_type=sk.get("W_type", config.slim_W_type),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    # Train
    total_iter = config.n_epochs * config.n_samples
    history = algo.train(generator, total_iter, verbose=False)

    return history


# ---------------------------------------------------------------------------
# Repeated experiments
# ---------------------------------------------------------------------------

def run_repeated_experiment(
    config: SimulationConfig,
    algo_name: str,
    n_repeats: int,
    slim_kwargs: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """Run multiple independent repeats and aggregate results.

    Args:
        config: simulation configuration.
        algo_name: algorithm name.
        n_repeats: number of independent runs.
        slim_kwargs: optional dict with B_M, B_m, W_type for SLIM.

    Returns:
        aggregate_repeats() result.
    """
    all_histories = []
    for i in range(n_repeats):
        seed = config.seed + i * 100
        print(f"  Repeat {i + 1}/{n_repeats} (seed={seed})...")
        history = run_single_experiment(config, algo_name, seed, slim_kwargs)
        all_histories.append(history)

    # Independent evaluation generator
    eval_generator = IVDataGenerator(config, seed=config.seed + 99999)

    return aggregate_repeats(all_histories, config, eval_generator, skip=50)


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------

def _save_results(
    outdir: str,
    config: SimulationConfig,
    all_results: dict[str, dict[str, np.ndarray]],
    elapsed: float,
    args: argparse.Namespace,
    slim_configs: list[dict],
):
    """Save experiment config, metrics, and summary to outdir.

    Writes:
        - config.json    : experiment parameters (human-readable)
        - results.npz    : aggregated metric arrays
        - summary.txt    : text summary of final results
        - comparison.png : convergence plot
    """
    # --- config.json ---
    config_dict = {
        "model": config.model_name,
        "d_x": config.d_x,
        "d_z": config.d_z,
        "d_theta": config.d_theta,
        "sigma_z": config.sigma_z,
        "sigma_c": config.sigma_c,
        "sigma_y": config.sigma_y,
        "sigma_x": config.sigma_x,
        "n_epochs": config.n_epochs,
        "n_samples": config.n_samples,
        "total_iterations": config.n_epochs * config.n_samples,
        "n_repeats": config.n_repeats,
        "tosg_lr": config.tosg_lr,
        "tosg_lr_decay": config.tosg_lr_decay,
        "slim_lr": config.slim_lr,
        "slim_lr_decay": config.slim_lr_decay,
        "slim_B_M": config.slim_B_M,
        "slim_B_m": config.slim_B_m,
        "slim_W_type": config.slim_W_type,
        "seed": config.seed,
        "algorithms_run": args.algo,
        "slim_configs": [
            {"B_M": sc["B_M"], "B_m": sc["B_m"], "W_type": sc["W_type"]}
            for sc in slim_configs
        ],
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # --- results.npz ---
    npz_kwargs = {}
    for algo_name, res in all_results.items():
        for key, arr in res.items():
            npz_kwargs[f"{algo_name}_{key}"] = arr
    np.savez(os.path.join(outdir, "results.npz"), **npz_kwargs)

    # --- summary.txt ---
    lines = []
    lines.append("=" * 60)
    lines.append("IV Regression Simulation Summary")
    lines.append("=" * 60)
    lines.append(f"Timestamp:   {config_dict['timestamp']}")
    lines.append(f"Model:       {config_dict['model']}")
    lines.append(f"Dimensions:  d_x={config_dict['d_x']}, d_z={config_dict['d_z']}, "
                 f"d_theta={config_dict['d_theta']}")
    lines.append(f"Noise:       sigma_z={config_dict['sigma_z']}, "
                 f"sigma_c={config_dict['sigma_c']}, "
                 f"sigma_y={config_dict['sigma_y']}, "
                 f"sigma_x={config_dict['sigma_x']}")
    lines.append(f"Training:    {config_dict['n_epochs']} epochs x "
                 f"{config_dict['n_samples']} samples")
    lines.append(f"Repeats:     {config_dict['n_repeats']}")
    lines.append(f"Elapsed:     {config_dict['elapsed_seconds']} s")
    lines.append("-" * 60)
    lines.append(f"{'Algorithm':<22} {'Median':>10} {'Mean±Std':>18} {'Pred MSE (med)':>16}")
    lines.append("-" * 60)
    for algo_name, res in all_results.items():
        label = algo_name.upper()
        pe_med = res["param_error_median"][-1]
        pe_mean = res["param_error_mean"][-1]
        pe_std = res["param_error_std"][-1]
        pm_med = res["pred_mse_median"][-1]
        lines.append(
            f"{label:<22} {pe_med:>8.4f}   "
            f"{pe_mean:>8.4f}±{pe_std:.4f}   "
            f"{pm_med:>12.4f}"
        )
    lines.append("=" * 60)
    text = "\n".join(lines)
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)

def _parse_slim_configs(raw: Optional[str]) -> list:
    """Parse --slim-configs string into list of SLIM parameter dicts.

    Format: "B_M,B_m,W_type;B_M,B_m,W_type;..."
    Example: "8,8,identity;4,4,random;1,1,identity"

    Returns:
        List of dicts with keys B_M, B_m, W_type, and a generated label.
        Returns a single default config if raw is None.
    """
    if raw is None:
        return []

    configs = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        items = [x.strip() for x in part.split(",")]
        if len(items) != 3:
            raise ValueError(
                f"Invalid SLIM config '{part}': expected 'B_M,B_m,W_type'"
            )
        try:
            B_M = int(items[0])
            B_m = int(items[1])
        except ValueError:
            raise ValueError(
                f"Invalid batch sizes in '{part}': B_M and B_m must be integers"
            )
        W_type = items[2]
        # Generate a readable label
        label = f"slim_B{B_M}_m{B_m}_{W_type[:2]}"
        configs.append({"B_M": B_M, "B_m": B_m, "W_type": W_type, "label": label})
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="IV Regression Simulation (TOSG-IVaR vs First-Order SLIM)"
    )
    parser.add_argument(
        "--model", type=str, default="linear",
        choices=["linear", "quadratic", "poly2", "poly3"],
        help="structural model g(theta; x) (default: linear)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="number of epochs (overrides default)",
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="samples per epoch (overrides default)",
    )
    parser.add_argument(
        "--repeats", type=int, default=None,
        help="number of independent repeats (overrides default)",
    )
    parser.add_argument(
        "--dim", type=int, default=None,
        help="shortcut: set both d_x and d_z to this value",
    )
    parser.add_argument(
        "--dx", type=int, default=None,
        help="dimension of explanatory variable x (overrides --dim)",
    )
    parser.add_argument(
        "--dz", type=int, default=None,
        help="dimension of instrument z (overrides --dim)",
    )
    parser.add_argument(
        "--algo", type=str, default="both",
        choices=["tosg", "ostg", "slim", "both", "all"],
        help="algorithm(s) to run (default: both = tosg+slim)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="save comparison plot (e.g. results/comparison.png)",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="output directory: saves config, results (npz), summary, and plot",
    )
    parser.add_argument(
        "--slim-BM", type=int, default=None,
        help="SLIM batch size for Jacobian estimate M̃ (default: 1)",
    )
    parser.add_argument(
        "--slim-Bm", type=int, default=None,
        help="SLIM batch size for moment estimate m̃ (default: 1)",
    )
    parser.add_argument(
        "--slim-W", type=str, default=None,
        choices=["identity", "random"],
        help="SLIM weighting matrix type (default: identity)",
    )
    parser.add_argument(
        "--slim-configs", type=str, default=None,
        help=(
            "Multiple SLIM configs for side-by-side comparison. "
            "Format: 'BM,Bm,W_type;BM,Bm,W_type;...' "
            "Example: '8,8,identity;4,4,random'"
        ),
    )
    args = parser.parse_args()

    # --- Build configuration ---
    config = SimulationConfig(model_name=args.model)

    # Command-line overrides
    if args.epochs is not None:
        config.n_epochs = args.epochs
    if args.samples is not None:
        config.n_samples = args.samples
    if args.repeats is not None:
        config.n_repeats = args.repeats
    if args.slim_BM is not None:
        config.slim_B_M = args.slim_BM
    if args.slim_Bm is not None:
        config.slim_B_m = args.slim_Bm
    if args.slim_W is not None:
        config.slim_W_type = args.slim_W
    # Dimension overrides: --dx/--dz take priority over --dim
    if args.dim is not None:
        config.d_x = args.dim
        config.d_z = args.dim
    if args.dx is not None:
        config.d_x = args.dx
    if args.dz is not None:
        config.d_z = args.dz
    # Re-generate true parameters if dimensions changed from defaults
    if args.dim is not None or args.dx is not None or args.dz is not None:
        rng = np.random.default_rng(config.seed)
        config.theta_star = config.model.true_params(rng, config.d_x)
        config.gamma_star = rng.normal(0, 1, size=(config.d_z, config.d_x))

    # Parse SLIM configs
    slim_configs = _parse_slim_configs(args.slim_configs)
    # If --slim-configs not given but slim is selected, use CLI overrides / defaults
    if not slim_configs and args.algo in ("slim", "both", "all"):
        slim_configs = [{
            "B_M": config.slim_B_M,
            "B_m": config.slim_B_m,
            "W_type": config.slim_W_type,
            "label": "slim",
        }]

    # --- Print experiment info ---
    print("=" * 60)
    print("  IV Regression Simulation")
    print("=" * 60)
    print(f"  Model:      {config.model_name}")
    print(f"  Dimensions: d_x = {config.d_x}, d_z = {config.d_z}, d_theta = {config.d_theta}")
    print(f"  Noise:      sigma_z={config.sigma_z}, sigma_c={config.sigma_c}, "
          f"sigma_y={config.sigma_y}, sigma_x={config.sigma_x}")
    total_iter = config.n_epochs * config.n_samples
    print(f"  Training:   {config.n_epochs} epochs x {config.n_samples} samples "
          f"= {total_iter} iterations")
    print(f"  Repeats:    {config.n_repeats}")

    algo_labels = []
    if args.algo in ("tosg", "both", "all"):
        algo_labels.append("TOSG-IVaR")
    if args.algo in ("ostg", "all"):
        algo_labels.append("OSTG-IVaR")
    for sc in slim_configs:
        algo_labels.append(
            f"SLIM(B_M={sc['B_M']}, B_m={sc['B_m']}, W={sc['W_type']})"
        )
    print(f"  Algorithms: {', '.join(algo_labels)}")
    print("-" * 60)

    # --- Run experiments ---
    t_start = time.time()
    all_results = {}

    if args.algo in ("tosg", "both", "all"):
        print("\n[*] Running TOSG-IVaR...")
        all_results["tosg"] = run_repeated_experiment(
            config, "tosg", config.n_repeats
        )

    if args.algo in ("ostg", "all"):
        print("\n[*] Running OSTG-IVaR...")
        all_results["ostg"] = run_repeated_experiment(
            config, "ostg", config.n_repeats
        )

    for sc in slim_configs:
        label = sc["label"]
        print(f"\n[*] Running SLIM (B_M={sc['B_M']}, B_m={sc['B_m']}, "
              f"W={sc['W_type']})...")
        all_results[label] = run_repeated_experiment(
            config, "slim", config.n_repeats,
            slim_kwargs={"B_M": sc["B_M"], "B_m": sc["B_m"], "W_type": sc["W_type"]},
        )

    t_elapsed = time.time() - t_start
    print(f"\nElapsed: {t_elapsed:.1f} seconds")

    # --- Output results ---
    print_summary_table(all_results)

    # Plot
    plot_path = args.save
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        plot_path = plot_path or os.path.join(args.outdir, "comparison.png")
    plot_comparison(all_results, save_path=plot_path)

    # --- Save all results to output directory ---
    if args.outdir:
        _save_results(args.outdir, config, all_results, t_elapsed, args, slim_configs)
        print(f"\nResults saved to: {args.outdir}/")


if __name__ == "__main__":
    main()
