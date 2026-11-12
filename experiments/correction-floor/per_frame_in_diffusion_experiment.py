"""Per-frame residual-bias diagnostic with the *dense* calibration.

The earlier per-frame diagnostic (`per_frame_diagnostic.py`) used the sparse
calibration (7 log-spaced CNR_0, 400 runs/cell) -- the same calibration
`poc_demo.py` uses. This script reruns the diagnostic with the dense
calibration matching `dense_calibration_and_test.py` (100 log-spaced
CNR_0 in [2, 100], 1000 runs/cell). The point is to test whether the
per-frame residual at the deepest frames shrinks substantially, i.e.
whether the residuals seen in the sparse-calibration diagnostic were
calibration-budget-limited or something more fundamental.

Three possible outcomes:
* residuals shrink substantially -- the correction works deeper than CNR ~ 0.7
  with adequate calibration;
* residuals shrink moderately -- there is a bias-vs-calibration-budget
  tradeoff, characterised here;
* residuals stay similar -- there is a model-misspecification floor of the
  ``subtract`` correction that more calibration cannot fix.

Wall time estimate: ~2 hours for all three shapes (Voigt is the slow one).

Usage::

    .venv/Scripts/python.exe scripts/per_frame_dense.py
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias import (Experiment, model_from_spec,  # noqa: E402
                           per_frame_diagnostic)
from width_debias.montecarlo import run_ensemble  # noqa: E402
from width_debias.bias_lookup import BiasLookup, frame_stats  # noqa: E402


# Match dense_calibration_and_test.py exactly for an apples-to-apples test.
CNR0_RANGE = (2.0, 100.0)
N_CALIB = 100
N_RUNS_CAL = 1000
# Per-frame diagnostic config
D_TRUE = 0.02
CNR0_REPORT = (5.0, 8.0, 13.0, 30.0)
N_RUNS_DIAG = 4000
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)


def _calib_one(args):
    cnr0, exp, n_runs, seed, model_name = args
    model = model_from_spec(model_name)
    ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed, model=model)
    return frame_stats(ens)


def dense_calibrate(model_name: str) -> BiasLookup:
    cal_exp = Experiment(D=0.0)
    cnr0_grid = np.logspace(np.log10(CNR0_RANGE[0]),
                             np.log10(CNR0_RANGE[1]),
                             N_CALIB)
    args_list = [(float(c), cal_exp, N_RUNS_CAL, 1000 + i, model_name)
                 for i, c in enumerate(cnr0_grid)]
    n = len(args_list)
    results = [None] * n
    done = 0
    milestones = max(1, n // 10)
    t0 = time.time()
    print(f"  Dense calibration ({model_name}, {N_CALIB} CNR_0, "
          f"{N_RUNS_CAL} runs/cell, n_workers={N_WORKERS})",
          flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_calib_one, args): i
                   for i, args in enumerate(args_list)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % milestones == 0 or done == n:
                pct = 100.0 * done / n
                dt = time.time() - t0
                eta = dt * (n - done) / max(done, 1)
                print(f"    [cal] {done}/{n} ({pct:3.0f}%)  "
                      f"elapsed {dt:5.0f}s  ETA {eta:4.0f}s",
                      flush=True)
    return BiasLookup.from_stats(results)


def diagnostic_report(model_name: str, trunc: BiasLookup) -> None:
    pmodel = model_from_spec(model_name)
    exp = Experiment(D=D_TRUE)
    print(f"\n=== {model_name}: per-frame diagnostic with DENSE calibration ===",
          flush=True)
    for cnr0 in CNR0_REPORT:
        fn = per_frame_diagnostic(exp, cnr0, trunc, scheme="subtract",
                                  n_runs=N_RUNS_DIAG, seed=30000,
                                  profile_model=pmodel)
        mas = per_frame_diagnostic(exp, cnr0, trunc, scheme="mas",
                                   n_runs=N_RUNS_DIAG, seed=30000,
                                   profile_model=pmodel)
        print(f"\n  CNR_0 = {cnr0:g}  "
              f"(per-frame CNR {fn.cnr[0]:.2f} -> {fn.cnr[-1]:.2f})",
              flush=True)
        print(f"  {'frame':>5}{'CNR':>8}{'sig2_true':>11}"
              f"{'E[raw]':>10}{'E[FN]':>10}{'resid_FN':>11}"
              f"{'SE_FN':>9}"
              f"{'E[mas]':>10}{'resid_mas':>11}{'SE_mas':>9}",
              flush=True)
        for f in range(exp.n_frames):
            print(f"  {f:>5}{fn.cnr[f]:>8.2f}{fn.true_variance[f]:>11.4f}"
                  f"{fn.mean_raw[f]:>10.4f}"
                  f"{fn.mean_corrected[f]:>10.4f}{fn.residual[f]:>+11.4f}"
                  f"{fn.residual_se[f]:>9.4f}"
                  f"{mas.mean_corrected[f]:>10.4f}{mas.residual[f]:>+11.4f}"
                  f"{mas.residual_se[f]:>9.4f}",
                  flush=True)
        print(f"  residual range: FN [{np.nanmin(fn.residual):+.4f}, "
              f"{np.nanmax(fn.residual):+.4f}] sigma_0^2;  "
              f"mas [{np.nanmin(mas.residual):+.4f}, "
              f"{np.nanmax(mas.residual):+.4f}] sigma_0^2",
              flush=True)


def main() -> None:
    print(f"Per-frame diagnostic with DENSE calibration "
          f"(D = {D_TRUE} sigma_0^2/tau, L_D/FWHM_0 ~ "
          f"{np.sqrt(2 * D_TRUE) / 2.355:.3f})", flush=True)
    t_start = time.time()
    # Optional model specs; "generalized_gaussian:<p>" sets the shape. The
    # default list is the committed run, whose bare "generalized_gaussian" is
    # p = 2 (a Gaussian profile).
    specs = sys.argv[1:] or ["gaussian", "voigt", "generalized_gaussian"]
    for spec in specs:
        model_from_spec(spec)          # fail fast on a bad spec
    for name in specs:
        t0 = time.time()
        trunc = dense_calibrate(name)
        print(f"  Calibration done [{time.time() - t0:.0f}s]", flush=True)
        diagnostic_report(name, trunc)
    print(f"\n[total {(time.time() - t_start) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
