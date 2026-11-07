"""Head-to-head: the noise-bias correction vs an unbounded width fit.

The expert panel's open question: is the correction even needed? An estimator
that admits negative variance has no wall at zero to fold against, so it should
be unbiased without any correction -- at some cost in variance. This script
quantifies that trade-off against the derived correction, on identical data.

The unbounded comparator (:class:`UnboundedGaussianModel`) is the *same*
shape-matched Gaussian, fitted in the precision parameter ``b = 1 / sigma^2``
so the fitted variance ``1 / b`` can take any sign and the wall at zero is
genuinely gone (a variance parameterisation cannot do this -- ``sigma^2 = 0``
is an uncrossable singularity). It is the sharpest possible challenger because,
being shape-matched, it is near-CRLB efficient -- unlike the shape-general
unbounded route (the moment estimator), which is carried here as the
catastrophic reference and is the only unbounded option for the Voigt /
generalized-Gaussian shapes (a signed precision has no clean meaning for their
scale parameter, so the parametric unbounded fit is Gaussian-specific).

Two levels:

* **Per-frame (the mechanism).** Isolated single profiles at sigma^2 = sigma_0^2
  across a sweep of per-frame CNR (no diffusion, no decay). Reports the mean
  fitted variance (bias) and its SD (precision) for the bounded and unbounded
  fits.
* **D-level (the deliverable).** A full diffusion experiment across a CNR_0
  sweep. Reports <D>/D_true (bias) and SD(D/D_true) (precision) for the bounded
  fit + OLS (naive), the bounded fit + folded-normal (the method), the
  unbounded fit + OLS (no correction), and the moment estimator + OLS.

All variances in sigma_0^2; D ratios are dimensionless.

Set ``WIDTH_DEBIAS_WORKERS=0`` to use all cores.

Usage::

    .venv/Scripts/python.exe scripts/unbounded_fit_comparison.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias import (Experiment, GaussianModel, MomentModel,  # noqa: E402
                            UnboundedGaussianModel, calibrate_d_free, evaluate,
                            run_ensemble)
from width_debias.fit import fit_D  # noqa: E402
from width_debias.weights import msd_and_weights  # noqa: E402

# --- Per-frame (isolated single-profile) sweep -----------------------------
PF_CNR_GRID = np.logspace(np.log10(0.3), np.log10(30.0), 18)
PF_N_RUNS = 15_000

# --- D-level sweep ---------------------------------------------------------
EXP = Experiment(sigma0=1.0, D=0.02, tau=1.0, n_frames=12, t_max=3.0)
D_CNR0_GRID = np.array([4.0, 6.0, 9.0, 13.0, 21.0, 34.0, 55.0, 90.0])
D_N_RUNS = 2500

# Calibration for the noise-bias correction (D-free, standard geometry).
CAL_CNR0_GRID = np.logspace(np.log10(2.0), np.log10(100.0), 50)
CAL_N_RUNS = 1500


# --- Per-frame statistics --------------------------------------------------
def isolated_stats(cnr: float, model, n_runs: int, seed: int):
    """Mean, SD, negative-fraction and NaN-fraction of the fitted variance for
    isolated profiles at sigma^2 = sigma_0^2 and per-frame CNR ``cnr``.

    A no-decay (large tau), no-diffusion (D=0) experiment puts every frame at
    the same per-frame CNR, so one ``run_ensemble`` column is a clean isolated
    single-profile sample -- and it inherits the parallel machinery.
    """
    exp = Experiment(sigma0=1.0, D=0.0, tau=1.0e9, n_frames=2, t_max=1.0)
    ens = run_ensemble(exp, float(cnr), n_runs=n_runs, seed=seed, model=model)
    v = ens.variance[:, 0]
    finite = np.isfinite(v)
    vf = v[finite]
    nan_frac = float(np.mean(~finite))
    if vf.size < 2:
        return np.nan, np.nan, np.nan, nan_frac
    neg_frac = float(np.mean(vf < 0.0))
    return float(vf.mean()), float(vf.std(ddof=1)), neg_frac, nan_frac


def per_frame_sweep():
    print("Per-frame mechanism: isolated profiles at sigma^2 = 1 sigma_0^2")
    print(f"  {PF_N_RUNS} fits per CNR cell\n")
    header = (f"  {'CNR':>7}{'bnd mean':>10}{'bnd SD':>9}"
              f"{'unb mean':>10}{'unb SD':>9}{'unb neg':>9}{'unb NaN':>9}")
    print(header)
    rows = []
    for i, cnr in enumerate(PF_CNR_GRID):
        bm, bsd, _, _ = isolated_stats(cnr, GaussianModel(), PF_N_RUNS,
                                       seed=7000 + i)
        um, usd, uneg, unan = isolated_stats(cnr, UnboundedGaussianModel(),
                                             PF_N_RUNS, seed=7000 + i)
        rows.append((cnr, bm, bsd, um, usd, uneg, unan))
        print(f"  {cnr:7.2f}{bm:10.4f}{bsd:9.4f}{um:10.4f}{usd:9.4f}"
              f"{uneg:9.4f}{unan:9.4f}", flush=True)
    return rows


# --- D-level statistics ----------------------------------------------------
def d_ols(exp: Experiment, grid: np.ndarray, profile_model, n_runs: int,
          seed: int):
    """<D>/D_true, its median, and SD under OLS for one profile model.

    OLS needs no truncation model (it neither corrects points nor reweights),
    so this isolates the estimator's own bias and precision.
    """
    grid = np.asarray(grid, float)
    mean = np.full(grid.size, np.nan)
    median = np.full(grid.size, np.nan)
    spread = np.full(grid.size, np.nan)
    for i, cnr0 in enumerate(grid):
        ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed + i,
                           model=profile_model)
        ratios = []
        for r in range(ens.n_runs):
            msd, w = msd_and_weights("ols", ens.variance[r], ens.cnr_frame,
                                     None)
            d = fit_D(exp.times, msd, w)
            if d.ok and np.isfinite(d.D):
                ratios.append(d.D / exp.D)
        ratios = np.asarray(ratios, float)
        if ratios.size:
            mean[i] = ratios.mean()
            median[i] = np.median(ratios)
            spread[i] = ratios.std()
    return mean, median, spread


def d_level_sweep():
    print("\nD-level head-to-head: full diffusion experiment")
    print(f"  D = {EXP.D} sigma_0^2/tau (L_D/FWHM_0 ~ 0.085), "
          f"{D_N_RUNS} runs/CNR_0\n")

    print(f"  Building D-free folded-normal calibration "
          f"({CAL_CNR0_GRID.size} CNR_0, {CAL_N_RUNS} runs/cell)...", flush=True)
    t0 = time.time()
    trunc = calibrate_d_free(CAL_CNR0_GRID, n_runs=CAL_N_RUNS, seed=1,
                             model=GaussianModel())
    print(f"  Calibration done [{time.time() - t0:.0f}s]", flush=True)

    # Bounded fit: OLS (naive) and folded-normal (the method), from one sweep.
    print("  Bounded Gaussian fit (OLS + folded-normal)...", flush=True)
    bounded = evaluate(EXP, D_CNR0_GRID, trunc, n_runs=D_N_RUNS, seed=20000,
                       profile_model=GaussianModel())
    print("  Unbounded Gaussian fit (OLS)...", flush=True)
    unb_mean, unb_med, unb_sd = d_ols(EXP, D_CNR0_GRID,
                                      UnboundedGaussianModel(), D_N_RUNS,
                                      seed=20000)
    print("  Moment estimator (OLS)...", flush=True)
    mom_mean, mom_med, mom_sd = d_ols(EXP, D_CNR0_GRID, MomentModel(),
                                      D_N_RUNS, seed=20000)

    series = {
        "bounded+OLS":    (bounded["ols"].mean_bias, bounded["ols"].spread),
        "bounded+folded": (bounded["subtract"].mean_bias,
                           bounded["subtract"].spread),
        "unbounded+OLS":  (unb_mean, unb_sd),
        "moment+OLS":     (mom_mean, mom_sd),
    }

    print("\n  <D_est>/D_true   (1.000 = unbiased)")
    print("  CNR_0 " + "".join(f"{k:>16}" for k in series))
    for i, c0 in enumerate(D_CNR0_GRID):
        print(f"  {c0:5.0f} " + "".join(f"{series[k][0][i]:16.3f}"
                                        for k in series))
    print("\n  SD(D_est/D_true)   (precision; lower = tighter)")
    print("  CNR_0 " + "".join(f"{k:>16}" for k in series))
    for i, c0 in enumerate(D_CNR0_GRID):
        print(f"  {c0:5.0f} " + "".join(f"{series[k][1][i]:16.3f}"
                                        for k in series))
    return series


# --- Figure ----------------------------------------------------------------
SPAIN_CVD = {
    "red":    "#D55E00",
    "gold":   "#E69F00",
    "blue":   "#0072B2",
    "green":  "#009E73",
    "purple": "#CC79A7",
}
inch = 1 / 2.54


def make_figure(pf_rows, d_series):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[matplotlib not installed -- skipping figure]")
        return

    fs = 8
    fig, axes = plt.subplots(2, 2, layout="constrained",
                             figsize=(17 * inch, 14 * inch))
    for ax in axes.flat:
        ax.tick_params(axis="both", which="both", direction="in",
                       top=True, right=True, labelsize=fs, length=4, width=1)
    fig.patch.set_facecolor("w")
    fig.patch.set_alpha(1)

    cnr = np.array([r[0] for r in pf_rows])
    bm = np.array([r[1] for r in pf_rows])
    bsd = np.array([r[2] for r in pf_rows])
    um = np.array([r[3] for r in pf_rows])
    usd = np.array([r[4] for r in pf_rows])

    # Panel A: per-frame bias (mean fitted variance vs CNR).
    axA = axes[0, 0]
    axA.axhline(1.0, ls="--", c="k", lw=1)
    axA.plot(cnr, bm, "-o", color=SPAIN_CVD["red"], lw=2, ms=3,
             label="bounded fit")
    axA.plot(cnr, um, "-s", color=SPAIN_CVD["green"], lw=2, ms=3,
             label="unbounded fit")
    axA.set_xscale("log")
    axA.set_xlabel("per-frame CNR", fontsize=fs)
    axA.set_ylabel(r"$\mathrm{E}[\hat{\sigma}^2]\,/\,\sigma_0^2$", fontsize=fs)
    axA.legend(frameon=False, fontsize=fs)

    # Panel B: per-frame precision (SD of fitted variance vs CNR).
    axB = axes[0, 1]
    axB.plot(cnr, bsd, "-o", color=SPAIN_CVD["red"], lw=2, ms=3,
             label="bounded fit")
    axB.plot(cnr, usd, "-s", color=SPAIN_CVD["green"], lw=2, ms=3,
             label="unbounded fit")
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel("per-frame CNR", fontsize=fs)
    axB.set_ylabel(r"$\mathrm{SD}[\hat{\sigma}^2]\,/\,\sigma_0^2$", fontsize=fs)
    axB.legend(frameon=False, fontsize=fs)

    cnr0 = D_CNR0_GRID
    # The moment estimator is omitted from the figure: its D is negative and
    # ~10^5 off scale (catastrophic), so on the shared log axes it cannot be
    # drawn and would squash the three width-fit curves. Its numbers are in the
    # printed table and the caption note instead (cf. moment_estimator_failure.py).
    d_style = {
        "bounded+OLS":    (SPAIN_CVD["red"], "-o"),
        "bounded+folded": (SPAIN_CVD["blue"], "-o"),
        "unbounded+OLS":  (SPAIN_CVD["green"], "-s"),
    }

    # Panel C: D bias vs CNR_0.
    axC = axes[1, 0]
    axC.axhline(1.0, ls="--", c="k", lw=1)
    for label, (color, ls) in d_style.items():
        axC.plot(cnr0, d_series[label][0], ls, color=color, lw=2, ms=3,
                 label=label)
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.set_xlabel(r"initial CNR$_0$", fontsize=fs)
    axC.set_ylabel(r"$\langle D_{\mathrm{est}}\rangle\,/\,D_{\mathrm{true}}$",
                   fontsize=fs)
    axC.legend(frameon=False, fontsize=fs)

    # Panel D: D precision vs CNR_0.
    axD = axes[1, 1]
    for label, (color, ls) in d_style.items():
        axD.plot(cnr0, d_series[label][1], ls, color=color, lw=2, ms=3,
                 label=label)
    axD.set_xscale("log")
    axD.set_yscale("log")
    axD.set_xlabel(r"initial CNR$_0$", fontsize=fs)
    axD.set_ylabel(r"$\mathrm{SD}(D_{\mathrm{est}}/D_{\mathrm{true}})$",
                   fontsize=fs)
    axD.legend(frameon=False, fontsize=fs)

    os.makedirs("output", exist_ok=True)
    out = os.path.join("output", "unbounded_fit_comparison.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nFigure written to {out}")


def save_results(pf_rows, d_series):
    """Persist the computed arrays so the figure can be re-rendered (or restyled)
    without repeating the 60-minute Monte Carlo."""
    os.makedirs("output", exist_ok=True)
    out = os.path.join("output", "unbounded_fit_comparison.npz")
    payload = {"pf_rows": np.array(pf_rows, float), "cnr0": D_CNR0_GRID}
    for k, (bias, sd) in d_series.items():
        key = k.replace("+", "_")
        payload[f"{key}__bias"] = np.asarray(bias, float)
        payload[f"{key}__sd"] = np.asarray(sd, float)
    np.savez(out, **payload)
    print(f"Results saved to {out}")


def main() -> None:
    t0 = time.time()
    pf_rows = per_frame_sweep()
    d_series = d_level_sweep()
    save_results(pf_rows, d_series)
    make_figure(pf_rows, d_series)
    print(f"\n[total {(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
