"""Does the reported confidence interval on D actually cover the truth?

The validation so far checks the *mean* <D_hat>/D_true, not whether a per-
measurement confidence interval on D_hat covers D_true at its nominal rate. This
script does the interval check, on the de-biased estimator (coverage is only
meaningful once the estimator is unbiased; ``inverse_variance`` is shown as the
uncorrected contrast, whose interval covers less as its bias grows against the
spread of a single measurement).

The standard covariance interval ``D_hat +/- 1.96 * D_se`` (``D_se`` from the
weighted-least-squares slope covariance, ``fit.fit_D``) treats the MSD points
as independent. They are not: every ``MSD_i = sigma_hat^2_i - sigma_hat^2_0``
shares the fitted t=0 width. With the library's per-frame weights (every
frame, t = 0 included, weighted by 1 / Var(sigma_hat^2_i)) the shared term only
shifts the intercept, so it does not widen the slope interval; the check is
whether the interval then covers at the nominal rate. (Because the weights are
calibrated and deterministic in CNR, this interval has the *same* width for
every measurement at a given CNR_0 -- it does not adapt to the data.)

For each CNR_0 the script reports, over many realisations:

* empirical coverage of the 0.95 covariance interval (fraction containing
  D_true), for ``subtract`` (de-biased) and ``inverse_variance`` (biased), with
  its binomial SE;
* the narrowness ratio ``D_se / SD(D_hat)`` -- below 1 means too narrow --
  with an SE from batch means over sub-ensembles;
* the skew of D_hat (its sampling distribution is right-skewed at low CNR, so a
  symmetric interval is doubly wrong there);
* the coverage of a parametric bootstrap interval that resamples each frame's
  ``sigma_hat^2`` (including the shared t=0 frame) and takes a percentile
  interval, which also follows the skew. It uses 2000 resamples: with B
  resamples and interpolated percentile bounds, even an exact bootstrap covers
  only about 0.95 (B - 1) / (B + 1), which is 0.940 at B = 200 and 0.949 at
  B = 2000.

The per-frame variances used to resample are measured from the ensemble here, to
isolate the methodological question (given correct per-frame variances, does the
covariance interval under-cover, and does the bootstrap fix it); in practice they
come from each frame's own fit covariance or the calibration.

The SEs are conditional on the one calibration: ``D_se`` and the weights come
from the shared lookup, so its Monte Carlo error moves every cell's interval
together and is not in the pooled SEs.

Run from the repository root::

    .venv/Scripts/python.exe experiments/coverage-interval/coverage_check.py

Set ``WIDTH_DEBIAS_WORKERS=0`` to use all cores. Reuses the standard-geometry
D-free Gaussian calibration cached by ``subdiffusive_alpha.py``; builds it if
absent. Outputs ``output/coverage_check.{png,txt}``.
"""
from __future__ import annotations

import os
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from width_debias import Experiment, GaussianModel, calibrate_d_free  # noqa: E402
from width_debias.montecarlo import run_ensemble  # noqa: E402
from width_debias.bias_lookup import BiasLookup  # noqa: E402
from width_debias.weights import msd_and_weights  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[2] / "output"
OUTPUT.mkdir(exist_ok=True)
CAL_PATH = OUTPUT / "calibration_subdiffusive.npz"   # shared standard-geometry cache

# DICE palette; see plot_style.md.
COV_BLUE = "#003f7f"       # covariance interval (DICE blue)
BOOT_GOLD = "#f7941e"      # bootstrap interval (DICE gold)
IV_GREEN = "#0cce6b"       # inverse-variance, the biased contrast (DICE green)
NOMINAL_GREY = "#999999"
INCH = 1 / 2.54
FONT = 9

D_TRUE = 0.02
GEOMETRY = dict(sigma0=1.0, tau=1.0, half_window=6.0, n_pixels=241,
                n_frames=12, t_max=3.0)
# 13 log-spaced CNR_0 and 8000 runs per cell: the narrowness ratio's relative
# SE is about 1 / sqrt(2 N_RUNS) = 0.008, small enough to read a trend in it.
CNR0_SWEEP = tuple(np.round(np.geomspace(5.0, 90.0, 13), 1))
N_RUNS = 8000
N_BOOT = 2000              # 200 would cap even an exact bootstrap at 0.940
N_BATCH = 8                # sub-ensembles for the batch-means SEs
BOOT_CHUNK = 200           # realisations per bootstrap block (bounds memory)
POOL_FROM_CNR0 = 21.0      # pool the narrowness where every frame has CNR >= 1
Z = 1.959964               # two-sided 0.95 normal quantile
CAL_CNR0 = np.geomspace(2.0, 100.0, 60)
N_RUNS_CAL = 1500
SEED = 90000


def load_calibration() -> BiasLookup:
    """Load the shared D-free Gaussian calibration; build and cache if absent."""
    if CAL_PATH.exists() and "framevar_grid" in np.load(CAL_PATH).files:
        d = np.load(CAL_PATH)
        return BiasLookup(cnr_grid=d["cnr_grid"], p_grid=d["p_grid"],
                               relvar_grid=d["relvar_grid"],
                               varbias_grid=d["varbias_grid"],
                               framevar_grid=d["framevar_grid"])
    trunc = calibrate_d_free(CAL_CNR0, n_runs=N_RUNS_CAL, model=GaussianModel(),
                             **GEOMETRY)
    np.savez(CAL_PATH, cnr_grid=trunc.cnr_grid, p_grid=trunc.p_grid,
             relvar_grid=trunc.relvar_grid, varbias_grid=trunc.varbias_grid,
             framevar_grid=trunc.framevar_grid)
    return trunc


def wls_slope(msd, t, w) -> tuple[np.ndarray, float]:
    """Vectorised weighted-LS slope over the last axis; returns (D, D_se).

    ``msd`` has frames on its last axis (any leading shape). ``D_se`` is the
    covariance SE of D and is a scalar -- it depends only on the (deterministic)
    weights and times, not on the data, so every measurement at one CNR_0 shares
    it. Mirrors :func:`width_debias.fit.fit_D`.
    """
    g = w > 0
    tg, wg = t[g], w[g]
    sw = wg.sum(); swt = (wg * tg).sum(); swtt = (wg * tg * tg).sum()
    denom = sw * swtt - swt ** 2
    swm = msd[..., g] @ wg
    swtm = msd[..., g] @ (wg * tg)
    slope = (sw * swtm - swt * swm) / denom
    return slope / 2.0, float(np.sqrt(sw / denom) / 2.0)


def _skew(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    d = a - a.mean(); s2 = (d ** 2).mean()
    return float((d ** 3).mean() / s2 ** 1.5) if s2 > 0 else 0.0


def coverage_at(cnr0: float, trunc: BiasLookup, seed: int) -> dict:
    """Coverage of the covariance, bootstrap, and inverse-variance intervals."""
    exp = Experiment(D=D_TRUE, **GEOMETRY)
    ens = run_ensemble(exp, cnr0, n_runs=N_RUNS, seed=seed, model=GaussianModel())
    t = exp.times
    cnr = ens.cnr_frame
    var_bias = trunc.var_bias(cnr)                       # (n_frames,), deterministic
    _, w = msd_and_weights("subtract", ens.variance[0], cnr, trunc)

    # Point estimates per realisation (de-biased and biased).
    corrected = ens.variance - var_bias
    msd_sub = corrected - corrected[:, [0]]
    D_sub, D_se = wls_slope(msd_sub, t, w)               # D_se scalar (shared)
    msd_iv = ens.variance - ens.variance[:, [0]]
    D_iv, _ = wls_slope(msd_iv, t, w)

    half = Z * D_se
    hit_cov = np.abs(D_sub - D_TRUE) <= half
    hit_iv = np.abs(D_iv - D_TRUE) <= half
    sd = float(np.std(D_sub[np.isfinite(D_sub)]))

    # Ensemble-measured per-frame variance Var(sigma_hat^2_i), the bootstrap's
    # resampling scale.
    vfr = np.nanvar(ens.variance, axis=0, ddof=1)

    # Parametric bootstrap: resample each frame's sigma_hat^2 (incl. the shared
    # t=0 frame), re-correct, refit; percentile interval over the resamples.
    # Blocks of BOOT_CHUNK realisations keep the (runs, N_BOOT, frames) array small.
    rng = np.random.default_rng(seed + 7)
    hit_boot = np.empty(N_RUNS, dtype=bool)
    for start in range(0, N_RUNS, BOOT_CHUNK):
        block = ens.variance[start:start + BOOT_CHUNK]
        samp = (block[:, None, :]
                + rng.standard_normal((block.shape[0], N_BOOT, exp.n_frames))
                * np.sqrt(vfr))
        cb = samp - var_bias
        D_b, _ = wls_slope(cb - cb[:, :, [0]], t, w)     # (block, N_BOOT)
        lo, hi = np.percentile(D_b, [2.5, 97.5], axis=1)
        hit_boot[start:start + block.shape[0]] = (lo <= D_TRUE) & (D_TRUE <= hi)

    # Batch means over N_BATCH sub-ensembles give the SE of the narrowness.
    batches = np.array_split(np.arange(N_RUNS), N_BATCH)
    narrow_b = [D_se / np.std(D_sub[b]) for b in batches]

    cover_cov, cover_cov_se = _binom(hit_cov)
    cover_iv, cover_iv_se = _binom(hit_iv)
    cover_boot, cover_boot_se = _binom(hit_boot)
    return dict(cnr0=cnr0, min_cnr=float(cnr.min()), D_se=D_se, sd=sd,
                narrowness=D_se / sd if sd > 0 else np.nan,
                narrowness_se=float(np.std(narrow_b, ddof=1) / np.sqrt(N_BATCH)),
                skew=_skew(D_sub), median_sub=float(np.median(D_sub)) / D_TRUE,
                median_iv=float(np.median(D_iv)) / D_TRUE,
                cover_cov=cover_cov, cover_cov_se=cover_cov_se,
                cover_iv=cover_iv, cover_iv_se=cover_iv_se,
                cover_boot=cover_boot, cover_boot_se=cover_boot_se)


def _binom(hit: np.ndarray) -> tuple[float, float]:
    """Fraction of hits and its binomial SE."""
    p = float(np.mean(hit))
    return p, float(np.sqrt(p * (1.0 - p) / hit.size))


def _normal_coverage(ratio: float) -> float:
    """Coverage of D_hat +/- Z * D_se for a normal D_hat with D_se = ratio * SD."""
    return float(erf(Z * ratio / sqrt(2.0)))


def pooled(rows: list[dict]) -> list[str]:
    """Pooled coverages and narrowness, computed here rather than by hand.

    The coverages are plain means over the cells (each cell has the same number
    of runs), with SE sqrt(sum se^2) / n. The narrowness is pooled where every
    frame keeps per-frame CNR >= 1 (CNR0 >= POOL_FROM_CNR0), weighted by its
    inverse batch-mean variance, beside the plain mean; the normal coverage it
    implies is 2 Phi(Z * ratio) - 1. All of them are conditional on the one
    calibration (module docstring).
    """
    lines = []
    for key, label in (("cover_cov", "covariance, subtract"),
                       ("cover_boot", "bootstrap, subtract"),
                       ("cover_iv", "covariance, inverse_variance")):
        p = np.array([r[key] for r in rows])
        se = np.array([r[key + "_se"] for r in rows])
        lines.append(f"pooled coverage over {len(rows)} cells, {label}: "
                     f"{p.mean():.4f} +/- {np.sqrt((se ** 2).sum()) / len(rows):.4f}")
    sel = [r for r in rows if r["cnr0"] >= POOL_FROM_CNR0]
    ratio = np.array([r["narrowness"] for r in sel])
    w = 1.0 / np.array([r["narrowness_se"] for r in sel]) ** 2
    r_w, r_w_se = float((w * ratio).sum() / w.sum()), float(1.0 / np.sqrt(w.sum()))
    r_plain = float(ratio.mean())
    r_plain_se = float(ratio.std(ddof=1) / np.sqrt(ratio.size))
    lines += [
        f"pooled narrowness from CNR0 {POOL_FROM_CNR0:g} up ({len(sel)} cells): "
        f"{r_w:.4f} +/- {r_w_se:.4f} (inverse-variance weights), "
        f"{r_plain:.4f} +/- {r_plain_se:.4f} (plain mean)",
        f"normal coverage implied by the weighted narrowness: {_normal_coverage(r_w):.4f}",
    ]
    return lines


def write_report(rows: list[dict], path: Path) -> None:
    lines = [
        "=" * 78,
        "Coverage / interval check -- does the reported CI on D cover the truth?",
        "=" * 78,
        f"D_true = {D_TRUE} sigma0^2/tau, Gaussian, standard geometry; "
        f"{N_RUNS} runs/cell, {N_BOOT} bootstrap.",
        f"calibration: D-free (calibrate_d_free, D = 0), Gaussian, {len(CAL_CNR0)} "
        f"CNR0 in [{CAL_CNR0[0]:g}, {CAL_CNR0[-1]:g}] x {N_RUNS_CAL} runs, seeds 1 + i "
        f"(shared cache {CAL_PATH.name}); evaluation seeds {SEED} + 100 k for "
        "CNR0 index k, bootstrap seed + 7.",
        "0.95 intervals. Coverage = fraction containing D_true.",
        "subtract is de-biased; inverse_variance is the uncorrected contrast.",
        "median sub/iv = median D_hat / D_true for subtract / inverse_variance.",
        "narrowness = D_se/SD(D_hat) (<1 = covariance CI too narrow).",
        f"SEs: batch means over {N_BATCH} sub-ensembles (narrowness); binomial "
        "(coverages; cov SE is the covariance interval's, the others are alike).",
        "SEs are conditional on the one calibration, whose Monte Carlo error is "
        "shared by every cell.",
        "",
        "   CNR0 | minCNR | median sub/iv |  SD/D |  narrow +/- SE   |  skew | "
        "cover cov/boot/iv (cov SE)",
        "  (cover nominal = 0.95)",
    ]
    for r in rows:
        lines.append(
            f"  {r['cnr0']:5.1f} | {r['min_cnr']:6.2f} | "
            f"   {r['median_sub']:.2f}/{r['median_iv']:.2f}  | "
            f"{r['sd'] / D_TRUE:5.2f} |  {r['narrowness']:5.3f} +/- {r['narrowness_se']:.3f} | "
            f"{r['skew']:5.2f} | "
            f"{r['cover_cov']:.3f}/{r['cover_boot']:.3f}/{r['cover_iv']:.3f} "
            f"({r['cover_cov_se']:.3f})")
    lines += [""] + pooled(rows)
    report = "\n".join(lines)
    print(report)
    path.write_text(report + "\n", encoding="utf-8")


def make_figure(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    cnr0 = np.array([r["cnr0"] for r in rows])

    def col(key):
        return np.array([r[key] for r in rows])

    fig, (axA, axB) = plt.subplots(1, 2, layout="constrained",
                                   figsize=(18.0 * INCH, 8.0 * INCH))
    axA.axhline(0.95, color=NOMINAL_GREY, lw=1.0, ls=":", label="nominal 0.95")
    for key, mk, color, label in [
        ("cover_cov", "o-", COV_BLUE, "covariance CI, subtract"),
        ("cover_boot", "s-", BOOT_GOLD, "bootstrap CI, subtract"),
        ("cover_iv", "^-", IV_GREEN, "covariance CI, inverse-variance"),
    ]:
        axA.errorbar(cnr0, col(key), yerr=col(key + "_se"), fmt=mk, color=color,
                     lw=2, ms=4, capsize=2, label=label)
    # Zoom on the nominal band, widened if any point falls lower; room at the
    # bottom for the legend and at the top for the panel letter.
    lowest = min((col(k) - 2 * col(k + "_se")).min()
                 for k in ("cover_cov", "cover_boot", "cover_iv"))
    axA.set_ylim(min(0.86, lowest - 0.06), 0.98)
    axA.set_ylabel("empirical coverage of the 0.95 interval", fontsize=FONT)

    axB.axhline(1.0, color=NOMINAL_GREY, lw=1.0, ls=":", label="SE = SD")
    axB.errorbar(cnr0, col("narrowness"), yerr=col("narrowness_se"), fmt="o-",
                 color=COV_BLUE, lw=2, ms=4, capsize=2,
                 label=r"covariance SE / SD(D)")
    axB.set_ylabel("covariance SE / realised SD (ratio)", fontsize=FONT)
    # Room below the curve for the legend, and above 1 for the panel letter.
    lowest_b = (col("narrowness") - 2 * col("narrowness_se")).min()
    axB.set_ylim(min(0.86, lowest_b - 0.04), 1.06)

    for letter, ax in zip("ab", (axA, axB)):
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 90])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$\mathrm{CNR}_0$", fontsize=FONT)
        ax.tick_params(axis="both", which="both", direction="in", top=False,
                       right=False, labelsize=FONT)
        ax.legend(frameon=False, fontsize=FONT, loc="lower right", handlelength=1.5)
        ax.set_facecolor("white")
        ax.text(0.03, 0.97, f"({letter})", transform=ax.transAxes, va="top",
                fontsize=FONT, fontweight="bold")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure: {path.name}")


def main() -> None:
    print(f"Loading/building D-free calibration ...")
    trunc = load_calibration()
    print(f"Coverage sweep: {len(CNR0_SWEEP)} CNR0 x {N_RUNS} runs "
          f"x {N_BOOT} bootstrap ...")
    rows = [coverage_at(float(c), trunc, SEED + 100 * i)
            for i, c in enumerate(CNR0_SWEEP)]
    write_report(rows, OUTPUT / "coverage_check.txt")
    make_figure(rows, OUTPUT / "coverage_check.png")


if __name__ == "__main__":
    main()
