"""Five-way method comparison: bias and precision of D vs CNR_0.

Compares, on identical simulated data:

* Gaussian-fit estimator + OLS
* Gaussian-fit estimator + inverse-variance weighting
* Gaussian-fit estimator + noise-bias correction (hard subtraction)
* Gaussian-fit estimator + MAS-style correction (gated subtraction)
* Moment estimator + inverse-variance weighting (the non-truncated route)

The noise-bias correction removes the bias but can overcorrect in the
deepest noise. Two alternatives target that: the MAS-style gated subtraction
(it switches off rather than overshooting), and the moment estimator (it has
no wall to fold against, so it needs no correction at all -- at the cost of
higher variance). The table reports both bias and spread so the
bias-variance trade-off is visible.

Run from the repository root::

    python scripts/compare_methods.py

Writes ``output/method_comparison.png``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias import (Experiment, GaussianModel, MomentModel,  # noqa: E402
                            evaluate, sweep_stats)
from width_debias.bias_lookup import BiasLookup  # noqa: E402

EXP = Experiment(sigma0=1.0, D=0.02, tau=1.0, n_frames=12, t_max=3.0)
CNR0_GRID = np.array([5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 90.0])
N_RUNS = 400

# (column label, profile-model, scheme key)
COLUMNS = [
    ("gauss+OLS",      "gaussian", "ols"),
    ("gauss+inv-var",  "gaussian", "inverse_variance"),
    ("gauss+folded",   "gaussian", "subtract"),
    ("gauss+MAS",      "gaussian", "mas"),
    ("moment+inv-var", "moment",   "inverse_variance"),
]


def _run(model, seed_cal, seed_eval):
    stats = sweep_stats(EXP, CNR0_GRID, n_runs=N_RUNS, seed=seed_cal, model=model)
    trunc = BiasLookup.from_stats(stats)
    return evaluate(EXP, CNR0_GRID, trunc, n_runs=N_RUNS, seed=seed_eval,
                    profile_model=model)


def main() -> None:
    print("width-debias method comparison")
    print(f"  experiment: D={EXP.D}, n_frames={EXP.n_frames}, {N_RUNS} runs/CNR_0\n")

    print("Running Gaussian-fit estimator ...")
    gauss = _run(GaussianModel(), seed_cal=1, seed_eval=20000)
    print("Running moment estimator ...")
    moment = _run(MomentModel(), seed_cal=1, seed_eval=20000)
    results = {"gaussian": gauss, "moment": moment}

    bias = {label: results[m][s].mean_bias for label, m, s in COLUMNS}
    spread = {label: results[m][s].spread for label, m, s in COLUMNS}
    labels = [c[0] for c in COLUMNS]

    # --- bias table ------------------------------------------------------
    print("\n  <D_est> / D_true   (1.000 = unbiased)")
    print("  CNR_0 " + "".join(f"{lab:>15}" for lab in labels))
    for i, c0 in enumerate(CNR0_GRID):
        print(f"  {c0:5.0f} " + "".join(f"{bias[lab][i]:15.3f}" for lab in labels))

    print("\n  mean |bias-1|  (lower = less biased)")
    for lab in labels:
        b = bias[lab]
        print(f"    {lab:<16} {np.nanmean(np.abs(b - 1.0)):.3f}")

    print("\n  mean spread  std(D_est/D_true)  (lower = more precise)")
    for lab in labels:
        print(f"    {lab:<16} {np.nanmean(spread[lab]):.3f}")

    print("\n  Reading: MAS gives the lowest bias and removes the deep-regime")
    print("  overshoot that folded-normal shows at CNR_0 = 5. The moment")
    print("  estimator is unbiased per frame but its profile-ratio form is")
    print("  unstable -- a near-zero denominator produces wild, often negative")
    print("  D. Removing the wall needs a stable non-truncated estimator")
    print("  (CVE-style, on raw displacements), not an intensity-profile moment.")

    _plot(bias)


def _plot(bias) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[matplotlib not installed -- skipping figure]")
        return

    # The figure shows the four Gaussian-fit schemes on a log axis. The moment
    # estimator goes negative (unstable) and cannot be drawn on log y; its
    # failure is reported in the printed table instead.
    styles = {
        "gauss+OLS":     ("tab:gray",   "o-"),
        "gauss+inv-var": ("tab:orange", "o-"),
        "gauss+folded":  ("tab:green",  "o-"),
        "gauss+MAS":     ("tab:blue",   "o-"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for lab, (color, ls) in styles.items():
        ax.plot(CNR0_GRID, bias[lab], ls, color=color, label=lab)
    ax.axhline(1.0, ls="--", c="k", lw=1, label="unbiased")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("initial CNR$_0$")
    ax.set_ylabel("$\\langle D_{est}\\rangle\\,/\\,D_{true}$")
    ax.set_title("Bias in $D$: MAS-gated correction vs the alternatives")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs("output", exist_ok=True)
    out = os.path.join("output", "method_comparison.png")
    fig.savefig(out, dpi=150)
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    main()
