"""Before/after check of the slope-fit weights: frame 0 weighted vs frame 0 dropped.

``msd_and_weights`` now weights every frame, t = 0 included, by
1 / Var(sigma_hat_i^2), which equals fitting the corrected widths against time
(``crlb-precision/efficiency_check.py`` found it reaches the Cramer-Rao bound).
The former weights were 1 / (Var_i + Var_0) with frame 0 at weight 0. This
script compares the two on cached Monte Carlo, with no new simulation:

1. calibration SE: the 16 Gaussian calibration draws of
   ``calibration_se_check.py`` applied to its fixed evaluation set; observed
   variance of each cell mean over the draws / predicted (calibration_D_se),
   now that frame 0's own calibration error enters the slope;
2. bias: the four-shape evaluation sets of ``efficiency_check.py`` (1000 runs
   per cell); |mean D/D_in - 1| in units of the combined SE (evaluation and
   calibration), and the number of cells beyond 2;
3. precision: spread of D/D_in over realizations, new / former;
4. coverage: fraction of realizations whose fit_D 95 % interval (covariance
   SE combined with the calibration SE) contains D_in.

Frames below per-frame CNR 1 are dropped, as in the other validations.

    .venv/Scripts/python.exe experiments/dfree-calibration/slope_fit_check.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "crlb-precision"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias import Experiment, BiasLookup  # noqa: E402
from width_debias.bias_lookup import frame_stats  # noqa: E402
from width_debias.weights import calibration_D_se, msd_and_weights  # noqa: E402

import calibration_se_check as cse  # noqa: E402
import efficiency_check as eff  # noqa: E402

FLOOR = 1.0
VARIANTS = ("former", "current")


def weights(variant: str, var0: np.ndarray, cnr: np.ndarray,
            model: BiasLookup) -> np.ndarray:
    """Slope weights for one cell (the exact key makes them the same per row)."""
    if variant == "current":
        _, w = msd_and_weights("subtract", var0, cnr, model)
    else:
        v = model.frame_variance(cnr)
        w = 1.0 / (v + v[0])
        w[0] = 0.0
    return np.where(cnr >= FLOOR, w, 0.0)


def cell(model, var, cnr, d_true, variant):
    """Per-realization D/D_in, covariance SE, and calibration SE (all / D_in)."""
    times = Experiment(D=d_true).times
    w = weights(variant, var[0], cnr, model)
    good = w > 0
    corrected = var - model.var_bias(cnr)
    msd = corrected - corrected[:, [0]]
    tg, wg = times[good], w[good]
    sw, swt, swtt = wg.sum(), (wg * tg).sum(), (wg * tg * tg).sum()
    denom = sw * swtt - swt ** 2
    c = wg * (sw * tg - swt) / denom
    ratio = msd[:, good] @ c / 2.0 / d_true
    d_se = np.sqrt(sw / denom) / 2.0 / d_true
    cal = calibration_D_se(times, msd[0], w, cnr, model) / d_true
    return ratio, d_se, cal


def main() -> None:
    # 1. Calibration SE over the 16 Gaussian draws.
    ks = [k for k in range(cse.N_CAL) if cse._cal_path(k).exists()]
    models = [BiasLookup.from_stats([frame_stats(e) for e in cse._ensembles(k)])
              for k in ks]
    with np.load(cse.OUT / f"eval_runs{cse.EVAL_RUNS}.npz") as z:
        evar, ecnr = z["variance"], z["cnr"]
    nd, nc = len(cse.D_VALUES), len(cse.CNR0_VALUES)
    print(f"1. Calibration SE: {len(ks)} Gaussian draws, {cse.EVAL_RUNS} runs per "
          "cell; observed / predicted variance of the cell mean, mean over D")
    print("            " + "".join(f"{c:>7g}" for c in cse.CNR0_VALUES) + "    mean")
    for variant in VARIANTS:
        means = np.full((len(ks), nd, nc), np.nan)
        pred = np.full((len(ks), nd, nc), np.nan)
        for j, m in enumerate(models):
            for di, d in enumerate(cse.D_VALUES):
                for ci in range(nc):
                    r, _, s = cell(m, evar[di, ci], ecnr[di, ci], d, variant)
                    means[j, di, ci], pred[j, di, ci] = r.mean(), s
        ratio = means.var(axis=0, ddof=1) / np.mean(pred ** 2, axis=0)
        print(f"  {variant:9s} " + "".join(f"{v:7.2f}" for v in ratio.mean(axis=0))
              + f"  {ratio.mean():6.2f}")
    print(f"  (each per-CNR_0 entry +- about {np.sqrt(2 / (len(ks) - 1) / nd):.2f} "
          "if the D values were independent; they share the calibration)")

    # 2-4. Bias, precision, coverage on the four-shape evaluation sets.
    head = "            " + "".join(f"{c:>7g}" for c in eff.CNR0_VALUES)
    totals = {v: [0, 0] for v in VARIANTS}
    for spec in eff.SPECS:
        cal = eff.calibration(spec)
        with np.load(eff._eval_path(spec)) as z:
            evar, ecnr = z["variance"], z["cnr"]
        nd, nc = len(eff.D_VALUES), len(eff.CNR0_VALUES)
        res = {v: np.full((nd, nc, 4), np.nan) for v in VARIANTS}
        for variant in VARIANTS:
            for di, d in enumerate(eff.D_VALUES):
                for ci in range(nc):
                    r, d_se, s = cell(cal, evar[di, ci], ecnr[di, ci], d, variant)
                    comb = np.hypot(r.std(ddof=1) / np.sqrt(r.size), s)
                    cover = np.mean(np.abs(r - 1.0) <= 1.96 * np.hypot(d_se, s))
                    res[variant][di, ci] = (r.mean(), (r.mean() - 1.0) / comb,
                                            r.std(ddof=1), cover)
            z = np.abs(res[variant][..., 1])
            totals[variant][0] += int(np.sum(z > 2))
            totals[variant][1] += z.size
        print(f"\n=== {spec} (rows D, columns CNR_0)")
        for label, k, fmt in (("mean D/D_in", 0, "7.3f"),
                              ("(mean - 1) / combined SE", 1, "7.1f"),
                              ("fit_D 95 % interval coverage", 3, "7.2f")):
            for variant in VARIANTS:
                print(f"[{variant}] {label}\n{head}")
                for di, d in enumerate(eff.D_VALUES):
                    print(f"  {d:6.3f}    "
                          + "".join(f"{v:{fmt}}" for v in res[variant][di, :, k]))
        print(f"spread of D/D_in, current / former\n{head}")
        for di, d in enumerate(eff.D_VALUES):
            print(f"  {d:6.3f}    " + "".join(
                f"{v:7.2f}" for v in res["current"][di, :, 2] / res["former"][di, :, 2]))
    print("\nCells beyond 2 combined SE (all shapes, D, CNR_0): "
          + ", ".join(f"{v} {a} of {b}" for v, (a, b) in totals.items())
          + f" (about {0.0455 * totals['current'][1]:.1f} expected by chance)")


if __name__ == "__main__":
    main()
