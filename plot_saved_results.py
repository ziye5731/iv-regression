#!/usr/bin/env python3
"""Plot saved simulation metrics from a results.npz file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_results(path: Path) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    algos: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        if "_" not in key:
            continue
        algo, metric = key.split("_", 1)
        algos.setdefault(algo, {})[metric] = data[key]
    return algos


def prettify_algo_name(name: str) -> str:
    return name.upper()


def plot_metric(
    algos: dict[str, dict[str, np.ndarray]],
    metric: str,
    out_file: Path,
    title: str,
    show_quartiles: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for algo, values in sorted(algos.items()):
        steps = values.get("steps")
        mean = values.get(f"{metric}_mean")
        if steps is None or mean is None:
            continue
        label = prettify_algo_name(algo)
        ax.plot(steps, mean, label=label, linewidth=2)
        std = values.get(f"{metric}_std")
        if std is not None:
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
        elif show_quartiles:
            q25 = values.get(f"{metric}_q25")
            q75 = values.get(f"{metric}_q75")
            if q25 is not None and q75 is not None:
                ax.fill_between(steps, q25, q75, alpha=0.2)
        plotted += 1
    if plotted == 0:
        raise ValueError(f"No metric '{metric}' found in results file")
    ax.set_xlabel("steps")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file)
    print(f"Saved plot: {out_file}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot saved metric history from a results.npz file."
    )
    parser.add_argument(
        "results_path",
        type=Path,
        help="Path to the saved results.npz file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively after saving.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for plot images.",
    )
    args = parser.parse_args()

    results_path = args.results_path
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    out_dir = args.output_dir or results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    algos = load_results(results_path)
    if not algos:
        raise ValueError(f"No algorithms found in {results_path}")

    plot_metric(
        algos,
        metric="param_error",
        out_file=out_dir / f"{results_path.stem}_param_error.png",
        title="Parameter Error vs Steps",
    )
    plot_metric(
        algos,
        metric="pred_mse",
        out_file=out_dir / f"{results_path.stem}_pred_mse.png",
        title="Prediction MSE vs Steps",
    )
    if any("loss_mean" in values for values in algos.values()):
        plot_metric(
            algos,
            metric="loss",
            out_file=out_dir / f"{results_path.stem}_loss.png",
            title="Loss vs Steps",
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
