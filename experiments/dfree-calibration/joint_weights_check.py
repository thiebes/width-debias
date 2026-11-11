"""Score the joint (CNR, sigma^2) lookup's slope weights: own fitted width against width-averaged.

A joint lookup can read the weight 1 / Var(sigma_hat^2) at each frame's own
fitted variance, or average it over the variance grid. The own-width read ties
a frame's weight to that frame's error: a frame that scatters high reads a
larger Var(sigma_hat^2) and a smaller weight, so the slope fit leans on the
frames that scattered low, a bias at the O(1/CNR^2) order the subtraction
removes. ``msd_and_weights`` uses the width-averaged weight when no key is
given; the own-width arm rebuilds the other read here.

Both arms apply the same subtraction (each frame keyed on its own fitted
variance, as ``msd_and_weights`` does without a key) to the same ensembles, so
they differ only by the weights. Gaussian fits; the calibration (seed 7) and
the evaluation ensembles (seeds 20000 + i) are independent.

    .venv/Scripts/python.exe experiments/dfree-calibration/joint_weights_check.py

Set ``WIDTH_DEBIAS_WORKERS=0`` to use every core (a few minutes).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias import (BiasLookup, Experiment, calibrate_cnr_sigma2,  # noqa: E402
                          fit_D, msd_and_weights, run_ensemble)

CAL_CNR = np.geomspace(0.25, 100.0, 16)
CAL_VARIANCE = np.linspace(0.8, 2.0, 7)
CAL_RUNS = 500
CAL_SEED = 7
D_VALUES = (0.02, 0.1)
CNR0_VALUES = (8.0, 13.0, 21.0, 34.0)
EVAL_RUNS = 600
EVAL_SEED = 20000


def own_width_weights(model: BiasLookup, variance_row: np.ndarray,
                      cnr: np.ndarray) -> np.ndarray:
    """1 / Var(sigma_hat^2) read at each frame's own fitted variance."""
    rv = model.rel_variance(cnr, variance_row)
    return 1.0 / np.clip(rv, np.nanmax(rv) * 1e-12, None)


def main() -> None:
    grid = calibrate_cnr_sigma2(cnr_grid=CAL_CNR, variance_grid=CAL_VARIANCE,
                                n_runs=CAL_RUNS, seed=CAL_SEED)
    model = BiasLookup.from_calibration(grid)
    print(f"Joint calibration: {CAL_CNR.size} CNR x {CAL_VARIANCE.size} variance "
          f"cells (CNR {CAL_CNR[0]:g} to {CAL_CNR[-1]:g}, sigma^2 "
          f"{CAL_VARIANCE[0]:g} to {CAL_VARIANCE[-1]:g}), {CAL_RUNS} runs per "
          f"cell, seed {CAL_SEED}; Gaussian fits")
    print(f"Evaluation: {EVAL_RUNS} runs per cell, seeds {EVAL_SEED} + i; subtract "
          "keyed on each frame's own fitted variance in both arms")
    print("mean D/D_true +/- its evaluation SE (SD over realisations / sqrt(n))\n")
    print(f"{'D':>6} {'CNR_0':>6}   {'own-width weights':>20}   {'width-averaged':>20}")
    for d_true in D_VALUES:
        exp = Experiment(D=d_true)
        for i, cnr0 in enumerate(CNR0_VALUES):
            ens = run_ensemble(exp, cnr0, n_runs=EVAL_RUNS, seed=EVAL_SEED + i)
            ratios = {"own": [], "averaged": []}
            for row in ens.variance:
                msd, w_averaged = msd_and_weights("subtract", row, ens.cnr_frame, model)
                w_own = own_width_weights(model, row, ens.cnr_frame)
                for arm, w in (("own", w_own), ("averaged", w_averaged)):
                    fit = fit_D(exp.times, msd, w)
                    if fit.ok:
                        ratios[arm].append(fit.D / d_true)
            cells = []
            for arm in ("own", "averaged"):
                r = np.asarray(ratios[arm])
                cells.append(f"{r.mean():8.3f} +/- {r.std(ddof=1) / np.sqrt(r.size):6.3f}")
            print(f"{d_true:6.2f} {cnr0:6g}   {cells[0]:>20}   {cells[1]:>20}", flush=True)


if __name__ == "__main__":
    main()
