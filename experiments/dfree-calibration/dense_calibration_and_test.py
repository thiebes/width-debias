"""Dense, parallel, D-free calibration and test.

What this does, per profile model:

1. Build a *dense* D-free calibration: 100 log-spaced CNR_0 values from 2 to
   100, with 5000 noisy realisations per cell. The lookup is the
   concatenation of all per-frame (CNR, var_bias) data from those sweeps,
   which puts substantially more data points at low per-frame CNR than the
   default 7-point sweep. ``dense_calibration`` caches it under
   ``output/dense-calibration/``; ``correction-floor/per_profile_floor.py``
   uses the same cached calibration.

2. Evaluate the four schemes (ols, inverse_variance, subtract, mas)
   on *random* log-distributed CNR_0 values, 1000 runs per cell. Random
   test CNR_0 values let us see the bias curve as a continuous function
   rather than at the calibration points.

3. Report mean(D_hat) / D_true binned in deciles of log-CNR_0 and, for the
   corrected schemes, the signed deviation pooled per decile and overall with
   its SE (evaluation and propagated calibration error). The overall mean
   |mean(D_hat)/D_true - 1| is kept for comparison with earlier tables, next to
   the value Monte Carlo error alone would give (sqrt(2/pi) x the combined SE,
   averaged over points): a mean of absolute values is not zero for an
   unbiased estimator.

Seeds are offset by 10^6 per shape (calibration, test, and the random test
CNR_0 draws), so the shapes see independent noise.

Both phases parallelise across CNR_0 via ProcessPoolExecutor, with progress
reports every 10% completion. Wall time per shape is about 2 h for the
calibration (once; cached) and under 1 h for the test on 8 cores.

Usage::

    .venv/Scripts/python.exe experiments/dfree-calibration/dense_calibration_and_test.py [MODEL ...]

MODEL is ``gaussian``, ``voigt``, ``generalized_gaussian:1.5`` or
``generalized_gaussian:4`` (default: all four). The script runs its own process
pool, so WIDTH_DEBIAS_WORKERS has no effect on it.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias import Experiment, SCHEMES, model_from_spec, spec_tag  # noqa: E402
from width_debias.fit import fit_D  # noqa: E402
from width_debias.montecarlo import run_ensemble  # noqa: E402
from width_debias.bias_lookup import BiasLookup, frame_stats  # noqa: E402
from width_debias.weights import calibration_D_se, msd_and_weights  # noqa: E402


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
D_TRUE = 0.02              # sigma_0^2 / tau; L_D / FWHM_0 ~ 0.085
CNR0_RANGE = (2.0, 100.0)  # log-spaced range
N_CALIB_POINTS = 100        # log-spaced calibration CNR_0 values
N_TEST_POINTS = 100         # random log-distributed test CNR_0 values
N_RUNS_CALIB = 5000
N_RUNS_TEST = 1000
CORRECTED = ("subtract", "mas")

SHAPES = ("gaussian", "voigt", "generalized_gaussian:1.5", "generalized_gaussian:4")
CAL_DIR = Path(__file__).resolve().parents[2] / "output" / "dense-calibration"

# Worker count: leave 1 core for the OS
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)


def shape_offset(model_name: str) -> int:
    """Seed offset per shape, so the shapes see independent noise."""
    if model_name not in SHAPES:
        sys.exit(f"unknown shape {model_name!r}; expected one of {SHAPES}")
    return 10 ** 6 * SHAPES.index(model_name)


# --------------------------------------------------------------------------
# Workers (top-level for pickling)
# --------------------------------------------------------------------------
def _calib_one(args):
    """Run a single-CNR_0 calibration ensemble and return the FrameStats."""
    cnr0, exp, n_runs, seed, model_name = args
    model = model_from_spec(model_name)
    # Already inside this script's pool: one process per ensemble.
    ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed, model=model, workers=1)
    return frame_stats(ens)


def _test_one(args):
    """Run a single-CNR_0 evaluation ensemble across all schemes.

    Per scheme: (mean, median, SD of D/D_true, finite count, mean calibration
    SE / D_true; NaN for the uncorrected schemes).
    """
    cnr0, exp, n_runs, seed, model_name, trunc = args
    model = model_from_spec(model_name)
    ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed, model=model, workers=1)
    out = {}
    times = exp.times
    for scheme in SCHEMES:
        ratios, cal = [], []
        for r in range(ens.n_runs):
            msd, w = msd_and_weights(scheme, ens.variance[r], ens.cnr_frame, trunc)
            d = fit_D(times, msd, w)
            if d.ok and np.isfinite(d.D):
                ratios.append(d.D / exp.D)
                if scheme in CORRECTED:
                    cal.append(calibration_D_se(times, msd, w, ens.cnr_frame, trunc,
                                                ens.variance[r]) / exp.D)
        ratios = np.asarray(ratios)
        if ratios.size > 1:
            out[scheme] = (float(ratios.mean()), float(np.median(ratios)),
                           float(ratios.std(ddof=1)), int(ratios.size),
                           float(np.nanmean(cal)) if cal else np.nan)
        else:
            out[scheme] = (np.nan, np.nan, np.nan, int(ratios.size), np.nan)
    return cnr0, out


# --------------------------------------------------------------------------
# Parallel orchestrators with progress reporting
# --------------------------------------------------------------------------
def _parallel_with_progress(worker, args_list, label):
    """Submit args_list to worker via ProcessPoolExecutor; report every 10%."""
    n = len(args_list)
    milestones = max(1, n // 10)
    done = 0
    results = [None] * n
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(worker, args): i for i, args in enumerate(args_list)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % milestones == 0 or done == n:
                pct = 100.0 * done / n
                dt = time.time() - t0
                eta = dt * (n - done) / max(done, 1)
                print(f"    [{label}] {done}/{n} ({pct:3.0f}%)  "
                      f"elapsed {dt:5.0f}s  ETA {eta:4.0f}s",
                      flush=True)
    return results


def dense_calibration(model_name: str) -> BiasLookup:
    """The dense D-free calibration for one shape, built once and cached."""
    seed = shape_offset(model_name) + 1000
    path = CAL_DIR / f"dense_{spec_tag(model_name)}_runs{N_RUNS_CALIB}_seed{seed}.npz"
    if path.exists():
        print(f"  Dense calibration: cached ({path.name})", flush=True)
        return BiasLookup.load(path)
    cnr0_calib = np.logspace(np.log10(CNR0_RANGE[0]), np.log10(CNR0_RANGE[1]),
                             N_CALIB_POINTS)
    print(f"  Dense calibration: {N_CALIB_POINTS} log-spaced CNR_0 in "
          f"[{CNR0_RANGE[0]}, {CNR0_RANGE[1]}], {N_RUNS_CALIB} runs/cell, "
          f"seeds {seed} + i ({N_WORKERS} workers)", flush=True)
    cal_exp = Experiment(D=0.0)         # D-free calibration
    cal_args = [(float(c), cal_exp, N_RUNS_CALIB, seed + i, model_name)
                for i, c in enumerate(cnr0_calib)]
    trunc = BiasLookup.from_stats(_parallel_with_progress(_calib_one, cal_args, "calib"))
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    trunc.save(path)
    return trunc


# --------------------------------------------------------------------------
# Per-shape driver
# --------------------------------------------------------------------------
def _pooled(dev: np.ndarray, eval_se: np.ndarray, cal_se: np.ndarray) -> str:
    """Inverse-variance pooled deviation; calibration SE as one shared systematic (upper bound)."""
    weight = 1.0 / (eval_se ** 2 + cal_se ** 2)
    weight /= weight.sum()
    p = float(np.sum(weight * dev))
    pe = float(np.sqrt(np.sum(weight ** 2 * eval_se ** 2)))
    pc = float(np.sum(weight * cal_se))
    return f"{p:+.4f} +/- {pe:.4f} (eval) +/- {pc:.4f} (cal); {p / np.hypot(pe, pc):+.2f} SE"


def run_shape(model_name: str) -> None:
    offset = shape_offset(model_name)
    print(f"\n{'=' * 70}\n=== {model_name}  (n_workers = {N_WORKERS}) ===\n"
          f"{'=' * 70}")
    t0 = time.time()
    trunc = dense_calibration(model_name)
    print(f"  Calibration ready [{time.time() - t0:.0f}s]; "
          f"lookup spans per-frame CNR {trunc.cnr_grid[0]:.3f} -> "
          f"{trunc.cnr_grid[-1]:.1f} ({trunc.cnr_grid.size} data points)")

    # ---------- Test ----------
    rng = np.random.default_rng(20260521 + offset)
    log_lo, log_hi = np.log10(CNR0_RANGE[0]), np.log10(CNR0_RANGE[1])
    cnr0_test = np.sort(10.0 ** rng.uniform(log_lo, log_hi, N_TEST_POINTS))
    test_exp = Experiment(D=D_TRUE)
    print(f"\n  Test: {N_TEST_POINTS} random log-distributed CNR_0 in "
          f"[{CNR0_RANGE[0]}, {CNR0_RANGE[1]}], {N_RUNS_TEST} runs/cell, "
          f"seeds {offset + 50000} + i")
    test_args = [(float(c), test_exp, N_RUNS_TEST, offset + 50000 + i, model_name, trunc)
                 for i, c in enumerate(cnr0_test)]
    t0 = time.time()
    test_out = _parallel_with_progress(_test_one, test_args, "test")
    print(f"  Test done [{time.time() - t0:.0f}s]")

    # ---------- Report ----------
    stat = {s: np.array([test_out[i][1][s] for i in range(N_TEST_POINTS)]) for s in SCHEMES}
    mean = {s: v[:, 0] for s, v in stat.items()}
    eval_se = {s: v[:, 2] / np.sqrt(v[:, 3]) for s, v in stat.items()}
    cal_se = {s: v[:, 4] for s, v in stat.items()}
    test_cnr0 = np.array([c for c, _ in test_out])
    log_edges = np.linspace(log_lo, log_hi, 11)   # 10 bins
    bin_idx = np.clip(np.searchsorted(log_edges[1:-1], np.log10(test_cnr0)), 0, 9)

    print("\n  Decile report (mean(D_hat) / D_true, averaged within each "
          "log-CNR_0 decile):")
    print(f"  {'CNR_0 range':<22}{'n':>4}  "
          f"{'OLS':>10}{'inv-var':>10}{'subtract':>10}{'mas':>10}")
    for b in range(10):
        mask = bin_idx == b
        if not mask.any():
            continue
        edge_lo, edge_hi = 10 ** log_edges[b], 10 ** log_edges[b + 1]
        print(f"  [{edge_lo:5.2f}, {edge_hi:5.2f}]  {int(mask.sum()):>4}  "
              + "".join(f"{np.nanmean(mean[s][mask]):>10.3f}"
                        for s in ("ols", "inverse_variance", "subtract", "mas")))

    for scheme in CORRECTED:
        print(f"\n  {scheme}: pooled mean(D_hat)/D_true - 1 per decile "
              "(inverse combined variance; calibration SE as one shared systematic)")
        for b in range(10):
            mask = (bin_idx == b) & np.isfinite(mean[scheme])
            if mask.any():
                print(f"    [{10 ** log_edges[b]:5.2f}, {10 ** log_edges[b + 1]:5.2f}]  "
                      + _pooled(mean[scheme][mask] - 1.0, eval_se[scheme][mask],
                                cal_se[scheme][mask]))
        good = np.isfinite(mean[scheme])
        print("    all points    "
              + _pooled(mean[scheme][good] - 1.0, eval_se[scheme][good], cal_se[scheme][good]))

    print("\n  Overall mean |mean(D_hat)/D_true - 1| across all test points "
          "(and the value Monte Carlo error alone gives, corrected schemes):")
    for scheme in SCHEMES:
        v = np.abs(mean[scheme] - 1.0)
        good = np.isfinite(v)
        line = f"    {scheme:>20}: {v[good].mean():.3f}  (median {np.median(v[good]):.3f})"
        if scheme in CORRECTED:
            null = np.sqrt(2.0 / np.pi) * np.hypot(eval_se[scheme], cal_se[scheme])
            line += f"  null {np.nanmean(null):.3f}"
        print(line)


def main() -> None:
    print("Dense parallel D-free calibration and test")
    print(f"  D_true = {D_TRUE} sigma_0^2/tau   (L_D/FWHM_0 ~ "
          f"{np.sqrt(2 * D_TRUE) / 2.355:.3f})")
    print(f"  workers = {N_WORKERS}")
    t0 = time.time()
    specs = sys.argv[1:] or list(SHAPES)
    for spec in specs:
        shape_offset(spec)              # fail fast on a bad spec
    for name in specs:
        run_shape(name)
    print(f"\n[total wall time: {(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
