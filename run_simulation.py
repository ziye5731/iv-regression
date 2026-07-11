#!/usr/bin/env python3
from __future__ import annotations

"""
run_simulation.py -- Main entry point for IV regression simulations.

Reads ALL hyperparameters from experiment_config.py (or a user-specified config).
No command-line arguments except --config to point to an alternative config file.

The config file is copied to the output directory for reproducibility.

Usage:
    python run_simulation.py                          # reads experiment_config.py
    python run_simulation.py --config my_config.py     # reads my_config.py
"""

import importlib.util
import os
import shutil
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np

from iv_sim.config import SimulationConfig
from iv_sim.data_generator import IVDataGenerator
from iv_sim.algorithms import TOSGIVaR, FirstOrderSLIM, OTSGIVaR, DistanceCovOpt
from iv_sim.metrics import aggregate_repeats
from iv_sim.visualization import plot_comparison, plot_comparison_by_samples, print_summary_table


def _load_config_module(config_path: str):
    """Load a Python config file as a module and return it."""
    spec = importlib.util.spec_from_file_location("_experiment_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_simulation_config(cfg) -> SimulationConfig:
    """Construct a SimulationConfig from the experiment config module."""
    config = SimulationConfig(model_name=cfg.MODEL)
    config.d_x = cfg.D_X
    config.d_z = cfg.D_Z
    config.mean_z = cfg.MEAN_Z
    config.sigma_z = cfg.SIGMA_Z
    config.sigma_c = cfg.SIGMA_C
    config.sigma_y = cfg.SIGMA_Y
    config.sigma_x = cfg.SIGMA_X
    config.seed = cfg.SEED
    config.n_iterations = cfg.N_ITERATIONS
    config.n_repeats = cfg.N_REPEATS
    config.verbose_every = getattr(cfg, "VERBOSE_EVERY", 50000)
    config.tosg_lr = cfg.TOSG_LR
    config.tosg_lr_decay = cfg.TOSG_LR_DECAY
    config.otsg_theta_lr = cfg.OTSG_THETA_LR
    config.otsg_theta_lr_decay = cfg.OTSG_THETA_LR_DECAY
    config.otsg_gamma_lr = cfg.OTSG_GAMMA_LR
    config.otsg_gamma_lr_decay = cfg.OTSG_GAMMA_LR_DECAY
    config.slim_lr = cfg.SLIM_LR
    config.slim_lr_decay = cfg.SLIM_LR_DECAY
    config.dcov_lr = cfg.DCOV_LR
    config.dcov_lr_decay = cfg.DCOV_LR_DECAY
    config.dcov_B = cfg.DCOV_B
    rng = np.random.default_rng(config.seed)
    config.theta_star = config.model.true_params(rng, config.d_x)
    config.gamma_star = rng.normal(0, 1, size=(config.d_z, config.d_x))
    return config


def _parse_slim_configs(cfg) -> list[dict]:
    """Convert SLIM_CONFIGS tuples to dict list with labels."""
    configs = []
    for B_M, B_m, W_type in cfg.SLIM_CONFIGS:
        label = f"slim_B{B_M}_m{B_m}_{W_type[:2]}"
        configs.append({"B_M": B_M, "B_m": B_m, "W_type": W_type, "label": label})
    return configs


def run_single_experiment(
    config: SimulationConfig,
    algo_name: str,
    seed: int,
    slim_kwargs: Optional[dict] = None,
    init_theta: Optional[np.ndarray] = None,
) -> list[dict]:
    """Run one complete training run."""
    generator = IVDataGenerator(config, seed=seed)
    if algo_name.lower() in ("tosg", "tosg_ivar"):
        algo = TOSGIVaR(config, seed=seed + 1000, init_theta=init_theta)
    elif algo_name.lower() in ("otsg", "otsg_ivar"):
        algo = OTSGIVaR(config, seed=seed + 1000, init_theta=init_theta)
    elif algo_name.lower() in ("slim", "first_order_slim"):
        sk = slim_kwargs or {}
        algo = FirstOrderSLIM(
            config, seed=seed + 1000,
            B_M=sk.get("B_M", config.slim_B_M),
            B_m=sk.get("B_m", config.slim_B_m),
            W_type=sk.get("W_type", config.slim_W_type),
            init_theta=init_theta,
        )
    elif algo_name.lower() in ("dcov", "dco", "distance_cov"):
        algo = DistanceCovOpt(config, seed=seed + 1000, init_theta=init_theta)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    # Same iterations for all algorithms
    n_iter = config.n_iterations
    ve = config.verbose_every
    do_verbose = bool(ve)  # False/None/0 → silent
    history = algo.train(generator, n_iter, verbose=do_verbose,
                         verbose_every=ve if do_verbose else 5000)
    return history


def run_repeated_experiment(
    config: SimulationConfig,
    algo_name: str,
    n_repeats: int,
    slim_kwargs: Optional[dict] = None,
    init_thetas: Optional[list[np.ndarray]] = None,
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """Run multiple independent repeats and aggregate results.

    Returns:
        (aggregated_results, final_thetas) where final_thetas[i] is the
        final theta from repeat i, used for checkpoint/resume.
    """
    all_histories = []
    final_thetas = []
    t0_repeats = time.time()
    for i in range(n_repeats):
        seed = config.seed + i * 100
        init_th = init_thetas[i] if init_thetas is not None else None
        eta_str = ""
        if i > 0:
            elapsed = time.time() - t0_repeats
            avg_per_repeat = elapsed / i
            remaining = avg_per_repeat * (n_repeats - i)
            if remaining < 60:
                eta_str = f" [ETA: {remaining:.0f}s]"
            elif remaining < 3600:
                eta_str = f" [ETA: {remaining/60:.0f}m {remaining%60:.0f}s]"
            else:
                h = int(remaining // 3600)
                m = int((remaining % 3600) // 60)
                eta_str = f" [ETA: {h}h {m}m]"
        print(f"  Repeat {i + 1}/{n_repeats} (seed={seed})...{eta_str}")
        history = run_single_experiment(config, algo_name, seed, slim_kwargs,
                                         init_theta=init_th)
        all_histories.append(history)
        final_thetas.append(history[-1]["theta"].copy())
    eval_generator = IVDataGenerator(config, seed=config.seed + 99999)
    return aggregate_repeats(all_histories, config, eval_generator, skip=50), final_thetas


def _save_results(
    outdir: str,
    config: SimulationConfig,
    all_results: dict[str, dict[str, np.ndarray]],
    all_thetas: dict[str, list[np.ndarray]],
    elapsed: float,
    algo_names: list[str],
    slim_configs: list[dict],
    config_file_path: str,
):
    """Save results: config.py (copy), results.npz, summary.txt."""
    shutil.copy2(config_file_path, os.path.join(outdir, "config.py"))
    npz_kwargs = {}
    for algo_name, res in all_results.items():
        for key, arr in res.items():
            npz_kwargs[f"{algo_name}_{key}"] = arr
    # Save per-repeat final thetas (for resume)
    for algo_name, thetas in all_thetas.items():
        for i, th in enumerate(thetas):
            npz_kwargs[f"{algo_name}_theta_{i}"] = th
    np.savez(os.path.join(outdir, "results.npz"), **npz_kwargs)
    lines = []
    lines.append("=" * 60)
    lines.append("IV Regression Simulation Summary")
    lines.append("=" * 60)
    lines.append(f"Timestamp:   {datetime.now().isoformat()}")
    lines.append(f"Model:       {config.model_name}")
    lines.append(f"Dimensions:  d_x={config.d_x}, d_z={config.d_z}, d_theta={config.d_theta}")
    lines.append(f"Z dist:      N({config.mean_z}, {config.sigma_z}^2 I)")
    lines.append(f"Noise:       sigma_c={config.sigma_c}, sigma_y={config.sigma_y}, sigma_x={config.sigma_x}")
    lines.append(f"Training:    {config.n_iterations} iterations")
    lines.append(f"Repeats:     {config.n_repeats}")
    lines.append(f"Algorithms:  {', '.join(algo_names)}")
    lines.append(f"Elapsed:     {round(elapsed, 1)} s")
    lines.append("-" * 60)
    lines.append(f"{'Algorithm':<22} {'Median':>10} {'Mean+/-Std':>18} {'Pred MSE (med)':>16}")
    lines.append("-" * 60)
    for algo_name, res in all_results.items():
        label = algo_name.upper()
        pe_med = res["param_error_median"][-1]
        pe_mean = res["param_error_mean"][-1]
        pe_std = res["param_error_std"][-1]
        pm_med = res["pred_mse_median"][-1]
        lines.append(f"{label:<22} {pe_med:>8.4f}   {pe_mean:>8.4f}+/-{pe_std:.4f}   {pm_med:>12.4f}")
    lines.append("=" * 60)
    text = "\n".join(lines)
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


def main():
    config_path = os.path.join(os.path.dirname(__file__), "experiment_config.py")
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "--config":
        config_path = argv[1]
    elif len(argv) >= 1 and argv[0].startswith("--"):
        print("Usage: python run_simulation.py [--config <path>]", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(config_path):
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    config_path = os.path.abspath(config_path)
    print(f"Loading config: {config_path}")
    cfg = _load_config_module(config_path)
    config = build_simulation_config(cfg)
    slim_configs = _parse_slim_configs(cfg)

    # --- Resume logic ---
    resume_from = getattr(cfg, "RESUME_FROM", None)
    old_results = None
    if resume_from is not None:
        resume_dir = os.path.abspath(resume_from)
        print(f"\n  Resuming from: {resume_dir}")
        # Load old config
        old_cfg_path = os.path.join(resume_dir, "config.py")
        if not os.path.isfile(old_cfg_path):
            print(f"  ERROR: config.py not found in {resume_dir}", file=sys.stderr)
            sys.exit(1)
        old_cfg = _load_config_module(old_cfg_path)
        config = build_simulation_config(old_cfg)
        # Override N_ITERATIONS with the NEW value (additional iterations)
        config.n_iterations = cfg.N_ITERATIONS
        # Load old results for thetas
        old_npz_path = os.path.join(resume_dir, "results.npz")
        if os.path.isfile(old_npz_path):
            old_results = dict(np.load(old_npz_path, allow_pickle=True))
            print(f"  Loaded {len(old_results)} keys from results.npz")
        else:
            print(f"  WARNING: results.npz not found; starting from scratch")

    outdir = cfg.OUTDIR
    if outdir is None:
        outdir = os.path.join("results", datetime.now().strftime("%m%d-%H%M"))
    save_plot = cfg.SAVE_PLOT
    # --- Resolve algorithm selection ---
    algo_list = cfg.ALGORITHMS
    if isinstance(algo_list, str):
        algo_list = [algo_list]  # backward compat: single string

    algo_names = []
    run_tosg = "tosg" in algo_list or "all" in algo_list
    run_otsg = "otsg" in algo_list or "all" in algo_list
    run_dcov = "dcov" in algo_list or "all" in algo_list
    run_slim = "slim" in algo_list or "all" in algo_list

    if run_tosg:
        algo_names.append("TOSG-IVaR")
    if run_otsg:
        algo_names.append("OTSG-IVaR")
    if run_dcov:
        algo_names.append("DCOV")
    if run_slim:
        for sc in slim_configs:
            algo_names.append(f"SLIM(B_M={sc['B_M']}, B_m={sc['B_m']}, W={sc['W_type']})")
    print("=" * 60)
    print("  IV Regression Simulation")
    print("=" * 60)
    print(f"  Model:      {config.model_name}")
    print(f"  Dimensions: d_x = {config.d_x}, d_z = {config.d_z}, d_theta = {config.d_theta}")
    print(f"  Z dist:     N({config.mean_z}, {config.sigma_z}^2 I)")
    print(f"  Noise:      sigma_c={config.sigma_c}, sigma_y={config.sigma_y}, sigma_x={config.sigma_x}")
    print(f"  Training:   {config.n_iterations} iterations per run")
    print(f"  Repeats:    {config.n_repeats}")
    print(f"  Algorithms: {', '.join(algo_names)}")
    print("-" * 60)

    # --- Calibration: quick run to estimate total time ---
    def _format_eta(remaining):
        if remaining < 60:
            return f"{remaining:.0f}s"
        elif remaining < 3600:
            return f"{remaining/60:.0f}m {remaining%60:.0f}s"
        else:
            h = int(remaining // 3600)
            m = int((remaining % 3600) // 60)
            return f"{h}h {m}m"

    CALIB_ITERS = min(config.n_iterations, 500)
    print(f"\n  Calibrating (quick {CALIB_ITERS}-iter test per algorithm)...")
    calib_times = {}  # algo_label -> seconds per iter

    def _calibrate_one(algo_label, algo_name, slim_kw=None):
        t0 = time.time()
        run_single_experiment(config, algo_name, seed=config.seed + 99999,
                              slim_kwargs=slim_kw)
        dt = time.time() - t0
        calib_times[algo_label] = dt / CALIB_ITERS

    # Temporarily override n_iterations for calibration
    orig_n_iter = config.n_iterations
    config.n_iterations = CALIB_ITERS
    calib_labels = []

    if run_tosg:
        _calibrate_one("TOSG-IVaR", "tosg")
        calib_labels.append("TOSG-IVaR")
    if run_otsg:
        _calibrate_one("OTSG-IVaR", "otsg")
        calib_labels.append("OTSG-IVaR")
    if run_dcov:
        _calibrate_one("DCOV", "dcov")
        calib_labels.append("DCOV")
    if run_slim:
        for sc in slim_configs:
            lbl = f"SLIM(B={sc['B_M']},m={sc['B_m']})"
            _calibrate_one(lbl, "slim", {"B_M": sc["B_M"], "B_m": sc["B_m"],
                                          "W_type": sc["W_type"]})
            calib_labels.append(lbl)

    config.n_iterations = orig_n_iter  # restore

    # --- Compute total estimate ---
    total_est = 0.0
    for lbl in calib_labels:
        per_iter = calib_times.get(lbl, 0)
        per_run = per_iter * orig_n_iter
        total_est += per_run * config.n_repeats
    print(f"  Estimated total time: {_format_eta(total_est)} "
          f"(~{datetime.fromtimestamp(time.time() + total_est).strftime('%H:%M')} finish)\n")

    t_start = time.time()
    all_results = {}
    all_thetas = {}

    # Helper: extract saved thetas from old_results (resume)
    def _get_init_thetas(algo_key):
        if old_results is None:
            return None
        thetas = []
        for i in range(config.n_repeats):
            k = f"{algo_key}_theta_{i}"
            if k in old_results:
                thetas.append(old_results[k])
            else:
                return None  # incomplete
        return thetas if len(thetas) == config.n_repeats else None

    # Count total algorithm runs for overall ETA
    total_runs = 0
    if run_tosg: total_runs += 1
    if run_otsg: total_runs += 1
    if run_dcov: total_runs += 1
    if run_slim: total_runs += len(slim_configs)
    runs_done = 0

    # --- Print initial scope ---
    total_repeats = total_runs * config.n_repeats
    print(f"\n  Total: {total_runs} algorithm(s) x {config.n_repeats} repeats "
          f"= {total_repeats} runs")

    if run_tosg:
        print("\n[*] Running TOSG-IVaR...")
        all_results["tosg"], all_thetas["tosg"] = run_repeated_experiment(
            config, "tosg", config.n_repeats, init_thetas=_get_init_thetas("tosg"))
        runs_done += 1
        if runs_done < total_runs:
            elapsed = time.time() - t_start
            avg = elapsed / runs_done
            rem = avg * (total_runs - runs_done)
            finish_ts = datetime.fromtimestamp(time.time() + rem)
            print(f"  [Overall: {_format_eta(elapsed)} elapsed, "
                  f"{_format_eta(rem)} remaining → done ~{finish_ts.strftime('%H:%M')}]")
    if run_otsg:
        print("\n[*] Running OTSG-IVaR...")
        all_results["otsg"], all_thetas["otsg"] = run_repeated_experiment(
            config, "otsg", config.n_repeats, init_thetas=_get_init_thetas("otsg"))
        runs_done += 1
        if runs_done < total_runs:
            elapsed = time.time() - t_start
            avg = elapsed / runs_done
            rem = avg * (total_runs - runs_done)
            finish_ts = datetime.fromtimestamp(time.time() + rem)
            print(f"  [Overall: {_format_eta(elapsed)} elapsed, "
                  f"{_format_eta(rem)} remaining → done ~{finish_ts.strftime('%H:%M')}]")
    if run_dcov:
        print("\n[*] Running DCOV...")
        all_results["dcov"], all_thetas["dcov"] = run_repeated_experiment(
            config, "dcov", config.n_repeats, init_thetas=_get_init_thetas("dcov"))
        runs_done += 1
        if runs_done < total_runs:
            elapsed = time.time() - t_start
            avg = elapsed / runs_done
            rem = avg * (total_runs - runs_done)
            finish_ts = datetime.fromtimestamp(time.time() + rem)
            print(f"  [Overall: {_format_eta(elapsed)} elapsed, "
                  f"{_format_eta(rem)} remaining → done ~{finish_ts.strftime('%H:%M')}]")
    if run_slim:
        for sc in slim_configs:
            label = sc["label"]
            print(f"\n[*] Running SLIM (B_M={sc['B_M']}, B_m={sc['B_m']}, "
                  f"W={sc['W_type']})...")
            all_results[label], all_thetas[label] = run_repeated_experiment(
                config, "slim", config.n_repeats,
                slim_kwargs={"B_M": sc["B_M"], "B_m": sc["B_m"],
                             "W_type": sc["W_type"]},
                init_thetas=_get_init_thetas(label),
            )
            runs_done += 1
            if runs_done < total_runs:
                elapsed = time.time() - t_start
                avg = elapsed / runs_done
                rem = avg * (total_runs - runs_done)
                finish_ts = datetime.fromtimestamp(time.time() + rem)
                print(f"  [Overall: {_format_eta(elapsed)} elapsed, "
                      f"{_format_eta(rem)} remaining → done ~{finish_ts.strftime('%H:%M')}]")
    t_elapsed = time.time() - t_start
    print(f"\nElapsed: {t_elapsed:.1f} seconds")
    print_summary_table(all_results)

    x_scale = getattr(cfg, "X_AXIS_SCALE", "log")

    # Build samples_per_step map for same-samples plot
    sp_map = {}
    for algo_name, results in all_results.items():
        if algo_name == "tosg":
            sp_map[algo_name] = 2
        elif algo_name == "otsg":
            sp_map[algo_name] = 1
        elif algo_name == "dcov":
            sp_map[algo_name] = config.dcov_B
        elif algo_name.startswith("slim_"):
            parts = algo_name.split("_")
            bm_val = int(parts[1][1:]) if len(parts) > 1 else 1
            bm2_val = int(parts[2][1:]) if len(parts) > 2 else 1
            sp_map[algo_name] = bm_val + bm2_val
        else:
            sp_map[algo_name] = 1

    os.makedirs(outdir, exist_ok=True)

    # Plot 1: same iterations
    plot_path1 = (os.path.join(outdir, "comparison.png")
                  if save_plot is None else save_plot)
    plot_comparison(all_results, save_path=plot_path1, x_scale=x_scale,
                    title="same iterations")

    # Plot 2: same samples
    plot_path2 = (os.path.join(outdir, "comparison_samples.png")
                  if save_plot is None
                  else save_plot.replace(".png", "_samples.png"))
    plot_comparison_by_samples(all_results, sp_map, save_path=plot_path2,
                                x_scale=x_scale)

    _save_results(outdir, config, all_results, all_thetas, t_elapsed,
                  algo_names, slim_configs, config_path)
    print(f"\nResults saved to: {outdir}/")


if __name__ == "__main__":
    main()
