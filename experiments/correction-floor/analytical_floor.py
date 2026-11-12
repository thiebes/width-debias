"""Locate the analytical floor of the noise-bias correction.

In the model of docs/DERIVATION.md the calibrated `var_bias(CNR)` is the variance
of the width estimator, which follows a localization-CRLB scaling at high
CNR:

    var_bias(CNR)  ~=  A / CNR^2

where `A` is set by the experimental geometry (pixel spacing, true width,
profile shape). At low CNR the fit's actual distribution stops being a
folded normal -- the estimator's bounds and the likelihood surface's
structure take over -- and the empirical `var_bias(CNR)` diverges from
the power-law prediction. The crossover marks the **analytical floor**:
below it, no amount of calibration with simple subtraction can debias the
fit, because the underlying model has broken.

This script (a) builds a modest calibration so the var_bias curve covers
deep CNR cleanly, (b) takes A from the exact Fisher information of the width
fit (``width_debias.fisher.leading_order_terms``: var_bias = Var(sigma_hat)
to leading order, the Cramer-Rao value), with the Monte Carlo estimator
spread at high CNR as a check, (c) pools the per-frame var_bias cells into
log-CNR bins with standard errors, and (d) reports the CNR at which a resolved
bin's ratio to A / CNR^2 departs from 1 by more than ``DEPART_Z`` of its SE,
walking down from high CNR. Above that CNR the leading-order term describes
the bias within the Monte Carlo resolution; below it the higher-order terms
matter. (An earlier version asked only whether the ratio left a factor-2
band, which hides departures of tens of per cent of the bias.) The Voigt
fit's second-order bias is 0.91 of the Fisher value, so its ratio tends to
0.91, not 1, where that difference is resolved.

The script is light: 30 log-spaced CNR_0 calibration points, 1000 runs per
cell. Single var_bias cells are noise-dominated above CNR ~ 3 (the bias there
is below 1e-2 sigma_0^2). An earlier version fitted A to those unbinned
high-CNR cells after dropping the negative ones; that overestimated A about
five-fold (0.37 against about 0.06 sigma_0^2), and a loop that exited at an
out-of-band top cell reported the lowest grid CNR (0.1) as the floor.

Seeds are offset by 10^6 per shape, and start at 3000 so they are disjoint from
the dense calibration's (``dfree-calibration/dense_calibration_and_test.py``).

Usage::

    .venv/Scripts/python.exe experiments/correction-floor/analytical_floor.py [MODEL ...]

MODEL is ``gaussian``, ``voigt``, or ``generalized_gaussian:<p>``; the default
runs Gaussian, Voigt, and the generalized-Gaussian at p = 1.5 and p = 4.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias import (Experiment, leading_order_terms, model_from_spec,  # noqa: E402
                           spec_tag)
from width_debias.montecarlo import run_ensemble  # noqa: E402
from width_debias.bias_lookup import BiasLookup, frame_stats  # noqa: E402


CNR0_RANGE = (2.0, 100.0)
N_CALIB = 30
N_RUNS_CAL = 1000
SPREAD_CNR_MIN = 8.0      # Monte Carlo A from the estimator spread on cells at CNR >= this
N_BINS = 24               # log-CNR bins pooling the per-frame calibration cells
RESOLVE_Z = 3.0           # a bin is resolved if |mean var_bias| > this many SE
DEPART_Z = 3.0            # a resolved bin departs if |ratio - 1| > this many SE;
                          # 3, not 2, because the walk stops at the first of
                          # about ten resolved bins (2 SE would stop on noise
                          # about one time in three)
SHAPES = ("gaussian", "voigt", "generalized_gaussian:1.5", "generalized_gaussian:4")
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)


def _calib_one(args):
    """One calibration cell: per-frame stats plus the per-frame Var(sigma_hat^2)."""
    cnr0, exp, n_runs, seed, model_name = args
    model = model_from_spec(model_name)
    # Already inside this script's pool: one process per ensemble.
    ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed, model=model, workers=1)
    return frame_stats(ens), np.nanvar(ens.variance, axis=0, ddof=1)


def modest_calibrate(model_name: str):
    """Return the calibration lookup and, per cell, (CNR, Var(sigma_hat^2))."""
    cal_exp = Experiment(D=0.0)
    cnr0_grid = np.logspace(np.log10(CNR0_RANGE[0]),
                             np.log10(CNR0_RANGE[1]),
                             N_CALIB)
    offset = 10 ** 6 * SHAPES.index(model_name)
    args_list = [(float(c), cal_exp, N_RUNS_CAL, offset + 3000 + i, model_name)
                 for i, c in enumerate(cnr0_grid)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_calib_one, args): i
                   for i, args in enumerate(args_list)}
        results = [None] * len(args_list)
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    stats = [r[0] for r in results]
    spread_cnr = np.concatenate([s.cnr for s in stats])
    spread_var = np.concatenate([r[1] for r in results])
    return BiasLookup.from_stats(stats), spread_cnr, spread_var


def _bin_log_cnr(cnr: np.ndarray, vb: np.ndarray) -> tuple[np.ndarray, ...]:
    """Mean var_bias and its standard error in log-spaced CNR bins.

    One calibration cell's var_bias is a Monte Carlo mean over N_RUNS_CAL fits;
    above CNR ~ 3 its noise exceeds the A/CNR^2 signal, so single cells cannot
    test the model there. Pooling neighbouring cells and taking the standard
    error from their scatter gives a resolved estimate where one exists, and
    says so where it does not. Negative cells stay in: they are the lower half
    of the noise, and dropping them biases every high-CNR bin upward.
    """
    edges = np.logspace(np.log10(cnr.min()), np.log10(cnr.max()) + 1e-9,
                        N_BINS + 1)
    idx = np.digitize(cnr, edges) - 1
    centers, means, ses = [], [], []
    for b in range(N_BINS):
        sel = idx == b
        if sel.sum() < 3:
            continue
        centers.append(float(np.exp(np.mean(np.log(cnr[sel])))))
        means.append(float(vb[sel].mean()))
        ses.append(float(vb[sel].std(ddof=1) / np.sqrt(sel.sum())))
    return np.array(centers), np.array(means), np.array(ses)


def find_floor(model_name: str, trunc: BiasLookup,
               spread_cnr: np.ndarray, spread_var: np.ndarray) -> dict:
    """Take A from the Fisher information; locate where var_bias departs from A/CNR^2.

    E[sigma_hat^2] = sigma^2 + Var(sigma_hat) for any unbiased width estimate,
    so to leading order var_bias = Var(sigma_hat), the width's Cramer-Rao
    variance, A / CNR^2 with A from the noise-free profile model alone. The
    Monte Carlo check: for a small, near-normal width error,
    Var(sigma_hat^2) = 4 sigma^2 Var(sigma_hat), and with the calibration at
    D = 0 the true variance is sigma_0^2 = 1 in every frame, so
    A_MC = CNR^2 Var(sigma_hat^2) / 4 at high CNR (>= SPREAD_CNR_MIN), where the
    spread is measured to a few per cent per cell. The prediction A/CNR^2 is
    then tested against the binned var_bias at lower CNR.
    """
    model = model_from_spec(model_name)
    A = float(leading_order_terms(model, Experiment().x, 1.0).bias_coef)
    ok = np.isfinite(spread_cnr) & np.isfinite(spread_var) \
        & (spread_cnr >= SPREAD_CNR_MIN)
    a_cells = spread_cnr[ok] ** 2 * spread_var[ok] / 4.0
    A_mc = float(np.median(a_cells))
    A_spread = float(np.percentile(a_cells, 84) - np.percentile(a_cells, 16)) / 2

    good = np.isfinite(trunc.cnr_grid) & np.isfinite(trunc.varbias_grid) \
        & (trunc.cnr_grid > 0)
    cnr, vb, se = _bin_log_cnr(trunc.cnr_grid[good], trunc.varbias_grid[good])
    pred = A / cnr ** 2
    ratio = vb / pred
    ratio_se = se / pred
    # Resolved = the bin mean differs from zero by more than RESOLVE_Z SE, in
    # either sign: a significantly negative var_bias (the p = 4 fit at low CNR)
    # is a departure from the model, not noise to skip.
    resolved = np.abs(vb) > RESOLVE_Z * se
    # A bin departs from the leading-order term if its ratio differs from 1 by
    # more than DEPART_Z of its own standard errors.
    in_band = np.abs(ratio - 1.0) <= DEPART_Z * ratio_se

    # Cross-check: weighted least-squares A from the resolved bins at CNR >= 1.
    fit = resolved & (vb > 0) & (cnr >= 1.0)
    A_bias = np.nan
    if fit.sum() >= 2:
        g, w = cnr[fit] ** -2.0, se[fit] ** -2.0
        A_bias = float(np.sum(w * g * vb[fit]) / np.sum(w * g * g))

    # Walk down from the highest resolved bin. The floor is the lowest bin
    # reached before the first resolved bin that departs from 1; that bin is
    # the breakdown. Unresolved bins neither set nor break the floor. If the
    # term holds to the bottom of the grid, the breakdown is not located and
    # the lowest bin is an upper bound on it.
    floor, breakdown, reached = np.nan, np.nan, False
    for i in range(len(cnr) - 1, -1, -1):
        if not resolved[i]:
            continue
        if not in_band[i]:
            reached, breakdown = True, float(cnr[i])
            break
        floor = float(cnr[i])

    return {
        "model": spec_tag(model_name),
        "A": A,
        "A_mc": A_mc,
        "A_spread": A_spread,
        "A_bias": A_bias,
        "floor_cnr": floor,
        "breakdown_cnr": breakdown,
        "floor_reached": reached,
        "in_band": in_band,
        "cnr": cnr,
        "var_bias_empirical": vb,
        "var_bias_se": se,
        "var_bias_predicted": pred,
        "ratio": ratio,
        "ratio_se": ratio_se,
        "resolved": resolved,
    }


def report(result: dict) -> None:
    print(f"\n=== {result['model']} ===")
    print(f"  A from the Fisher information (leading_order_terms): "
          f"{result['A']:.4f} sigma_0^2")
    print(f"  A from the estimator spread (CNR >= {SPREAD_CNR_MIN}, check): "
          f"{result['A_mc']:.4f} sigma_0^2  (ratio Fisher / MC "
          f"{result['A'] / result['A_mc']:.3f}; cell-to-cell half-width "
          f"{result['A_spread']:.4f})")
    print(f"  A from the resolved var_bias bins at CNR >= 1 (cross-check): "
          f"{result['A_bias']:.4f} sigma_0^2")
    if result["floor_reached"]:
        print(f"  Departure: var_bias / (A/CNR^2) differs from 1 by > {DEPART_Z} SE "
              f"at CNR ~ {result['breakdown_cnr']:.2f}; lowest agreeing resolved "
              f"bin above it: CNR ~ {result['floor_cnr']:.2f}")
    else:
        print(f"  No departure located: every resolved bin agrees with 1 within "
              f"{DEPART_Z} SE; the departure is below CNR ~ {result['floor_cnr']:.2f}")
    print(f"\n  {'CNR':>7}  {'empirical':>10}  {'SE':>8}  {'predicted':>10}"
          f"  {'ratio':>6}  {'+/-':>5}")
    for c, e, s, p, r, rs, ok, inb in zip(
            result["cnr"], result["var_bias_empirical"], result["var_bias_se"],
            result["var_bias_predicted"], result["ratio"], result["ratio_se"],
            result["resolved"], result["in_band"]):
        if not ok:
            flag = "  (unresolved)"
        elif inb:
            flag = ""
        else:
            flag = "  <<<"
        print(f"  {c:7.3f}  {e:10.4f}  {s:8.4f}  {p:10.4f}  {r:6.2f}  "
              f"{rs:5.2f}{flag}")


def main() -> None:
    print(f"Analytical floor of the noise-bias correction")
    print(f"  Calibration: {N_CALIB} log-spaced CNR_0 in "
          f"{CNR0_RANGE}, {N_RUNS_CAL} runs/cell")
    print(f"  A = CNR^2 Var(sigma_hat) from the Fisher information; Monte Carlo check "
          f"CNR^2 Var(sigma_hat^2) / 4, median over cells at CNR >= {SPREAD_CNR_MIN}")
    print(f"  var_bias pooled into {N_BINS} log-CNR bins; a bin is resolved if "
          f"|mean| exceeds {RESOLVE_Z} SE")
    print(f"  Departure criterion: resolved empirical/predicted differs from 1 "
          f"by > {DEPART_Z} SE")
    print(f"  Seeds: 10^6 x shape index + 3000 + i\n")

    specs = sys.argv[1:] or list(SHAPES)
    for spec in specs:
        if spec not in SHAPES:
            sys.exit(f"unknown shape {spec!r}; expected one of {SHAPES}")
    t_start = time.time()
    for name in specs:
        t0 = time.time()
        trunc, spread_cnr, spread_var = modest_calibrate(name)
        result = find_floor(name, trunc, spread_cnr, spread_var)
        report(result)
        print(f"  [{spec_tag(name)}: {time.time()-t0:.0f}s]")
    print(f"\n[total {(time.time()-t_start)/60:.1f} min]")


if __name__ == "__main__":
    main()
