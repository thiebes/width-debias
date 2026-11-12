"""Does the noise-bias correction de-bias the anomalous exponent alpha?

The MSD curve of a disordered film grows as ``MSD(t) = 2 D t^alpha``; the
exponent ``alpha`` is the disorder signature the field reads off (``alpha < 1``
subdiffusive, the disordered-nanocrystal-solid case). The noise bias does
not just inflate D -- because the per-frame contrast-to-noise falls as the
population decays, the late, low-CNR frames are lifted more than the early ones,
which tilts the MSD curve toward a steeper apparent exponent. So a naive fit
reads alpha too high (toward the ballistic alpha = 2), and the question is
whether the per-frame correction recovers the true alpha, not only D.

This is the clean single-population synthetic version of the result the
disordered-nanocrystal-solid bed shows on real model maps (naive alpha = 2.0
against a true alpha ~ 0.87). The bed is :class:`AnomalousExperiment`: a single
Gaussian (or other shape) population whose true variance follows
``sigma0^2 + 2 D t^alpha``, with single-exponential decay and white noise.

Design. Each exponent is compared at a *matched diffusion length over the
window*: D is chosen per alpha so the end-point MSD ``2 D t_max^alpha`` is the
same (here ``MSD_END = 0.12 sigma0^2``, i.e. ``L_D / FWHM_0 ~ 0.15``, the
nanocrystal-solid regime), so only the curve's shape -- the exponent -- differs
between panels, not the amount of spreading. ``alpha = 1`` lands exactly on the
project's canonical ``D = 0.02 sigma0^2/tau`` case. Because the simulated peak
tracks decay only (the project's CNR convention), the per-frame CNR is
``cnr0 * exp(-t/tau)`` for every alpha and D, so the noise is identical across
panels and the comparison is fair.

Three analyses per operating point:

* **naive** -- the field-standard fit: raw fitted variances, all frames, power
  law ``2 D t^alpha``.
* **naive + cut** -- the raw variances fitted over only the frames above the
  per-profile CNR floor, with no subtraction. It separates the two
  interventions the corrected arm combines.
* **corrected** -- the ``subtract`` scheme: subtraction of the calibrated per-frame
  ``var_bias(CNR)``, then the same power-law fit over only the frames above the
  per-profile CNR floor (the design rule: keep ``cnr0 * exp(-t/tau) >~ 1``,
  i.e. ``cnr0 >~ exp(t_max/tau) ~ 20`` for ``t_max = 3 tau``).

Each mean-trajectory exponent carries a Monte Carlo standard error from
resampling the realisations; quote the bias only where it exceeds about two of
these.

The calibration is D-free (built at D = 0, so it is also alpha-free) and
geometry-matched to the bed; it is cached to ``output/`` after the first build.

Run from the repository root::

    .venv/Scripts/python.exe scripts/subdiffusive_alpha.py [profile_model] [--quick]

Set ``WIDTH_DEBIAS_WORKERS=0`` to use all cores for the Monte Carlo.
Outputs ``output/subdiffusive_alpha[_<model>].{png,txt}``.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scipy.optimize import curve_fit  # noqa: E402

from width_debias import AnomalousExperiment, calibrate_d_free  # noqa: E402
from width_debias.models import MODELS  # noqa: E402
from width_debias.montecarlo import run_ensemble  # noqa: E402
from width_debias.bias_lookup import BiasLookup  # noqa: E402
from width_debias.weights import corrected_variance  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[2] / "output"
OUTPUT.mkdir(exist_ok=True)

# Spain CVD (Okabe-Ito) semantic pairing: blue = corrected (tracks truth),
# red = naive (biased); truth is drawn in black. See plot_style.md.
CORR_BLUE = "#0072B2"
NAIVE_RED = "#D55E00"
TRUTH_BLACK = "#000000"
FLOOR_GREY = "#999999"
INCH = 1 / 2.54
FONT = 9

# True exponents: subdiffusive (the disorder case), Fickian control, mild
# superdiffusive (the Saha bed spans both directions).
ALPHA_TRUE = (0.6, 0.8, 1.0, 1.2)
MSD_END = 0.12               # matched end-point MSD (sigma0^2); L_D/FWHM_0 ~ 0.15
CNR0_EVAL = (5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 90.0)
FLOOR_CNR = 1.0              # per-profile working floor (nc_solid bed)
ALPHA_BOUNDS = (0.2, 2.0)    # power-law fit bounds on the exponent

# D-free calibration grid (geometry-matched; built once, cached).
CAL_CNR0 = np.geomspace(2.0, 100.0, 60)
N_RUNS_CAL = 1500
N_RUNS_EVAL = 2000
N_BOOT = 200                 # resamples for the mean-trajectory exponent SE
EVAL_SEED = 20000            # independent of the calibration seed (invariant)


def d_for_alpha(alpha: float, t_max: float) -> float:
    """D that fixes the end-point MSD at ``MSD_END`` for this exponent.

    ``2 D t_max^alpha = MSD_END`` holds the diffusion length over the window
    constant across exponents, so the panels differ only in curve shape.
    """
    return MSD_END / (2.0 * t_max ** alpha)


def fit_msd_powerlaw(t: np.ndarray, var: np.ndarray,
                     mask: np.ndarray | None = None) -> tuple[float, float]:
    """Field-standard ``MSD = 2 D t^alpha`` fit; returns ``(D, alpha)``.

    MSD uses the fitted t=0 width (``msd = var - var[0]``, the project
    invariant). The fit runs over ``t > 0`` with NaN frames dropped; ``mask``
    (frames above the CNR floor) implements the discard-sub-floor-frames rule.
    """
    msd = var - var[0]
    keep = (t > 0) & np.isfinite(msd)
    if mask is not None:
        keep = keep & mask
    if keep.sum() < 3:
        return np.nan, np.nan
    try:
        popt, _ = curve_fit(
            lambda tt, d, a: 2.0 * d * tt ** a, t[keep], msd[keep],
            p0=[max(msd[keep][-1] / (2.0 * t[keep][-1]), 1e-6), 1.0],
            bounds=([0.0, ALPHA_BOUNDS[0]], [np.inf, ALPHA_BOUNDS[1]]),
            maxfev=20000)
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        return np.nan, np.nan


def build_or_load_calibration(model, geometry: dict, cal_cnr0: np.ndarray,
                              n_runs_cal: int, path: Path) -> BiasLookup:
    """D-free, geometry-matched 1-D calibration; cached after the first build.

    The lookup is built at D = 0, so it is keyed on per-frame CNR alone and is
    independent of both the application's D and its anomalous exponent.
    """
    if path.exists() and "framevar_grid" in np.load(path).files:
        d = np.load(path)
        return BiasLookup(cnr_grid=d["cnr_grid"], p_grid=d["p_grid"],
                               relvar_grid=d["relvar_grid"],
                               varbias_grid=d["varbias_grid"],
                               framevar_grid=d["framevar_grid"])
    trunc = calibrate_d_free(cal_cnr0, n_runs=n_runs_cal, model=model,
                             **geometry)
    np.savez(path, cnr_grid=trunc.cnr_grid, p_grid=trunc.p_grid,
             relvar_grid=trunc.relvar_grid, varbias_grid=trunc.varbias_grid,
             framevar_grid=trunc.framevar_grid)
    return trunc


@dataclass
class CellSamples:
    """Recovered exponents at one (alpha_true, cnr0) point.

    The bias is read from the *mean trajectory* E[MSD(t)] over realisations:
    the correction removes the systematic per-frame bias, so the mean corrected
    trajectory has the true shape and its fitted exponent is the de-biased
    estimate (this is the nc_solid bed's convention). Fitting the exponent per
    realisation and averaging instead would fold in the power-law fit's own
    per-measurement skew -- at this small diffusion length the single-shot
    exponent is broad and bound-clipped, so its mean is pulled high even with no
    noise bias. That per-measurement spread is reported separately, as the
    precision band, not as the bias.
    """

    alpha_true: float
    cnr0: float
    frames_kept: float          # frames surviving the floor cut (deterministic in CNR)
    a_naive_traj: float         # exponent from the mean raw trajectory, all frames
    a_naive_cut_traj: float     # exponent from the mean raw trajectory + floor cut
    a_corr_traj: float          # exponent from the mean corrected trajectory + cut
    a_naive_se: float           # Monte Carlo standard errors of the three
    a_naive_cut_se: float       #   mean-trajectory exponents (resampling the
    a_corr_se: float            #   realisations with replacement)
    d_naive_traj: float         # D / D_true from the mean raw trajectory
    d_corr_traj: float          # D / D_true from the mean corrected trajectory
    a_corr_samples: np.ndarray  # per-realisation corrected exponent (precision band)


def recover_alpha(alpha_true: float, cnr0: float, model,
                  trunc: BiasLookup, geometry: dict,
                  n_runs: int, seed: int, floor: float) -> CellSamples:
    """Recover the exponent at one operating point, naive vs corrected.

    Returns the mean-trajectory exponents (the bias estimate) plus the
    per-realisation corrected exponents (the precision picture).
    """
    exp = AnomalousExperiment(D=d_for_alpha(alpha_true, geometry["t_max"]),
                              alpha=alpha_true, **geometry)
    ens = run_ensemble(exp, cnr0, n_runs=n_runs, seed=seed, model=model)
    work = ens.cnr_frame >= floor
    t = exp.times

    corrected = np.array([
        corrected_variance("subtract", ens.variance[r], ens.cnr_frame, trunc)
        for r in range(ens.n_runs)])

    # Bias: fit the mean trajectory (naive over all frames, corrected over the
    # frames above the per-profile CNR floor). The naive + cut arm applies the
    # floor cut without the subtraction, so naive -> naive + cut isolates the
    # cut and naive + cut -> corrected isolates the subtraction; below the
    # design floor, where the cut drops frames, naive -> corrected mixes both.
    dn, a_naive_traj = fit_msd_powerlaw(t, np.nanmean(ens.variance, axis=0))
    _, a_naive_cut_traj = fit_msd_powerlaw(t, np.nanmean(ens.variance, axis=0),
                                           mask=work)
    dc, a_corr_traj = fit_msd_powerlaw(t, np.nanmean(corrected, axis=0), mask=work)

    # Monte Carlo standard error of each mean-trajectory exponent: refit the
    # mean of realisations drawn with replacement. The per-realisation s.d. /
    # sqrt(n) would understate it, because the exponent of the mean trajectory
    # is a nonlinear functional of the realisations (nc_solid convention).
    rng = np.random.default_rng(seed + 7)
    boot = np.full((N_BOOT, 3), np.nan)
    for b in range(N_BOOT):
        idx = rng.integers(0, ens.n_runs, ens.n_runs)
        raw_b = np.nanmean(ens.variance[idx], axis=0)
        boot[b, 0] = fit_msd_powerlaw(t, raw_b)[1]
        boot[b, 1] = fit_msd_powerlaw(t, raw_b, mask=work)[1]
        boot[b, 2] = fit_msd_powerlaw(t, np.nanmean(corrected[idx], axis=0),
                                      mask=work)[1]
    a_naive_se, a_naive_cut_se, a_corr_se = np.nanstd(boot, axis=0, ddof=1)

    # Precision: the corrected exponent a single measurement would report.
    a_corr_samples = []
    for r in range(ens.n_runs):
        _, ac = fit_msd_powerlaw(t, corrected[r], mask=work)
        if np.isfinite(ac):
            a_corr_samples.append(ac)

    return CellSamples(
        alpha_true=alpha_true, cnr0=cnr0, frames_kept=float(work.sum()),
        a_naive_traj=a_naive_traj, a_naive_cut_traj=a_naive_cut_traj,
        a_corr_traj=a_corr_traj, a_naive_se=float(a_naive_se),
        a_naive_cut_se=float(a_naive_cut_se), a_corr_se=float(a_corr_se),
        d_naive_traj=dn / exp.D, d_corr_traj=dc / exp.D,
        a_corr_samples=np.array(a_corr_samples))


def run_sweep(model, trunc: BiasLookup, geometry: dict,
              cnr0_grid, n_runs: int, floor: float) -> list[CellSamples]:
    cells = []
    for ai, alpha in enumerate(ALPHA_TRUE):
        for ci, cnr0 in enumerate(cnr0_grid):
            seed = EVAL_SEED + 1000 * ai + ci
            cells.append(recover_alpha(alpha, float(cnr0), model, trunc,
                                       geometry, n_runs, seed, floor))
    return cells


def _summary(a: np.ndarray) -> tuple[float, float, float, float]:
    """(mean, median, 25th, 75th) of a sample, NaN-safe for an empty sample."""
    if a.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    return (float(np.mean(a)), float(np.median(a)),
            float(np.percentile(a, 25)), float(np.percentile(a, 75)))


def write_report(cells: list[CellSamples], model_name: str, n_frames: int,
                 n_runs: int, t_max: float, floor: float, design_floor: float,
                 path: Path, calibration: str) -> None:
    """Table per exponent plus the headline figure of merit.

    The figure of merit mirrors the project's metric (lower residual bias in
    the working regime is better): the mean absolute exponent error
    ``|alpha_recovered - alpha_true|`` over the operating points at or above the
    design floor, naive against corrected.
    """
    by_alpha = {a: [c for c in cells if c.alpha_true == a] for a in ALPHA_TRUE}
    lines = [
        "=" * 78,
        f"Subdiffusive-alpha recovery -- subtract, {model_name} fit",
        "=" * 78,
        f"bed: AnomalousExperiment, MSD = 2 D t^alpha, matched end-point "
        f"MSD = {MSD_END} sigma0^2 (L_D/FWHM_0 ~ 0.15)",
        f"per-frame CNR = cnr0 * exp(-t/tau); floor cut at CNR >= {floor}; "
        f"design floor cnr0 >= exp(t_max/tau) = {design_floor:.1f}",
        f"{n_runs} runs/cell; bias = exponent of the mean trajectory; "
        "1-shot = per-realisation corrected exponent (precision)",
        calibration,
        f"evaluation seeds {EVAL_SEED} + 1000 a + c (exponent index a, CNR0 index c); "
        f"resampling seed + 7, {N_BOOT} resamples",
        "",
    ]
    for alpha, cells_a in by_alpha.items():
        lines.append("-" * 78)
        lines.append(f"=== true alpha = {alpha:.2f}   (D = "
                     f"{d_for_alpha(alpha, t_max):.4g} sigma0^2/tau^alpha) ===")
        lines.append("  cnr0 | frames | a naive (SE)  | a naive+cut (SE) | "
                     "a corr (SE)   | a corr 1-shot med[IQR] | D corr/D")
        for c in cells_a:
            _, med, lo, hi = _summary(c.a_corr_samples)
            lines.append(
                f"  {c.cnr0:5.0f} | {c.frames_kept:4.0f}/{n_frames} | "
                f"{c.a_naive_traj:5.2f} ({c.a_naive_se:4.2f}) | "
                f"{c.a_naive_cut_traj:5.2f} ({c.a_naive_cut_se:4.2f})     | "
                f"{c.a_corr_traj:5.2f} ({c.a_corr_se:4.2f}) | "
                f"{med:5.2f} [{lo:4.2f},{hi:4.2f}]      | {c.d_corr_traj:7.2f}")
        lines.append("")

    # Figures of merit over the working regime (cnr0 >= design floor). There the
    # cut drops no frame, so naive + cut equals naive and naive -> corrected is
    # the subtraction alone. Cells have independent seeds, so their errors
    # combine independently; the SEs are Monte Carlo over realisations and leave
    # out the calibration's own error, which all cells share. A cell whose
    # exponent sits at a fit bound is left out of the pooled value: resampling
    # returns the bound almost every time, so its SE collapses toward zero and
    # an inverse-variance weight would let that one cell set the pooled value.
    lines.append("=" * 78)
    lines.append(f"Signed exponent error, exponent(mean trajectory) - alpha_true, "
                 f"pooled over cnr0 >= {design_floor:.0f} (the design floor) with "
                 "inverse-variance weights: value (SE) [z] {cells pooled}")
    lines.append(f"(cells with the exponent within 0.01 of a fit bound "
                 f"{ALPHA_BOUNDS} are left out; SE excludes the shared "
                 "calibration error)")
    lines.append("  true alpha | naive                     | naive+cut                 | corrected")
    for alpha, cells_a in by_alpha.items():
        work_cells = [c for c in cells_a if c.cnr0 >= design_floor]

        def pooled(a_attr, se_attr):
            a = np.array([getattr(c, a_attr) for c in work_cells])
            se = np.array([getattr(c, se_attr) for c in work_cells])
            off_bound = ((a > ALPHA_BOUNDS[0] + 0.01)
                         & (a < ALPHA_BOUNDS[1] - 0.01))
            ok = np.isfinite(a) & np.isfinite(se) & (se > 0) & off_bound
            if not ok.any():
                return f"{'all at bound':>21} {{0}}"
            w = 1.0 / se[ok] ** 2
            value = np.sum(w * (a[ok] - alpha)) / w.sum()
            value_se = 1.0 / np.sqrt(w.sum())
            return (f"{value:+6.3f} ({value_se:5.3f}) [{value / value_se:+5.1f}] "
                    f"{{{int(ok.sum())}}}")

        lines.append(f"  {alpha:9.2f}  | {pooled('a_naive_traj', 'a_naive_se')} | "
                     f"{pooled('a_naive_cut_traj', 'a_naive_cut_se')} | "
                     f"{pooled('a_corr_traj', 'a_corr_se')}")
    lines.append("")
    lines.append(f"Mean |exponent(mean trajectory) - alpha_true| over cnr0 >= "
                 f"{design_floor:.0f}, with its Monte Carlo SE and the value Monte "
                 "Carlo error alone gives (mean of sqrt(2/pi) x SE; about 0.8 SE, "
                 "not 0, for an unbiased estimator):")
    lines.append("  true alpha | naive                 | naive+cut             | corrected")
    for alpha, cells_a in by_alpha.items():
        work_cells = [c for c in cells_a if c.cnr0 >= design_floor]

        def merit(a_attr, se_attr):
            err = np.nanmean([abs(getattr(c, a_attr) - alpha) for c in work_cells])
            ses = np.array([getattr(c, se_attr) for c in work_cells])
            se = np.sqrt(np.nansum(ses ** 2)) / len(work_cells)
            null = np.sqrt(2.0 / np.pi) * np.nanmean(ses)
            return f"{err:5.3f} ({se:5.3f}) null {null:5.3f}"

        lines.append(f"  {alpha:9.2f}  | {merit('a_naive_traj', 'a_naive_se')} | "
                     f"{merit('a_naive_cut_traj', 'a_naive_cut_se')} | "
                     f"{merit('a_corr_traj', 'a_corr_se')}")
    report = "\n".join(lines)
    print(report)
    path.write_text(report)


def make_figure(cells: list[CellSamples], model_name: str,
                design_floor: float, path: Path) -> None:
    """Recovered alpha vs CNR0, one panel per true exponent (small multiples).

    Mean recovered alpha is the bias line; the shaded band is the inter-quartile
    spread (per-measurement precision, which widens as the contrast falls). The
    truth line and the design-floor marker frame each panel.

    matplotlib is imported here, not at module top, so the Monte Carlo's spawned
    worker processes (which re-import this script on Windows) do not pay the
    pyplot import cost on every spawn.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    n = len(ALPHA_TRUE)
    cnr0_ticks = sorted({c.cnr0 for c in cells})
    fig, axes = plt.subplots(1, n, layout="constrained", sharey=True,
                             figsize=(6.0 * n * INCH, 7.0 * INCH))
    for ax, alpha in zip(axes, ALPHA_TRUE):
        cells_a = [c for c in cells if c.alpha_true == alpha]
        cnr0 = np.array([c.cnr0 for c in cells_a])
        # Precision band: the inter-quartile spread of the corrected exponent a
        # single measurement would report (wide at small diffusion length).
        band = np.array([_summary(c.a_corr_samples)[2:] for c in cells_a])
        ax.fill_between(cnr0, band[:, 0], band[:, 1], color=CORR_BLUE, alpha=0.15,
                        label="corrected, single-measurement IQR")
        # Bias lines: the exponent of the mean trajectory.
        ax.plot(cnr0, [c.a_naive_traj for c in cells_a], "o-", color=NAIVE_RED,
                lw=2, ms=4, label="naive (all frames)")
        ax.plot(cnr0, [c.a_corr_traj for c in cells_a], "s-", color=CORR_BLUE,
                lw=2, ms=4, label="corrected + floor cut")
        ax.axhline(alpha, color=TRUTH_BLACK, lw=1.2, ls="--", label="true exponent")
        ax.axvline(design_floor, color=FLOOR_GREY, lw=1.0, ls=":",
                   label=r"design floor (CNR$_0 \approx 20$)")
        ax.set_xscale("log")
        ax.set_xticks(cnr0_ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", which="both", direction="in",
                       top=True, right=True, labelsize=FONT)
        ax.set_ylim(0.3, 2.05)
        ax.set_xlabel(r"CNR$_0$", fontsize=FONT)
        ax.text(0.5, 0.96, rf"true $\alpha$ = {alpha:.1f}", ha="center",
                va="top", fontsize=FONT, transform=ax.transAxes)
    axes[0].set_ylabel(r"recovered exponent  $\alpha$", fontsize=FONT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=FONT,
               loc="outside lower center", ncol=3)
    fig.patch.set_facecolor("w")
    fig.patch.set_alpha(1)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?", default="gaussian",
                        choices=("gaussian", "voigt", "generalized_gaussian"),
                        help="profile model (default gaussian)")
    parser.add_argument("--quick", action="store_true",
                        help="small budget for a science sanity check")
    parser.add_argument("--floor", type=float, default=FLOOR_CNR,
                        help=f"per-frame CNR floor cut (default {FLOOR_CNR}); "
                             "the calibration is unaffected, so floor variants "
                             "reuse the cached calibration")
    args = parser.parse_args()

    model = MODELS[args.model]()
    geometry = dict(sigma0=1.0, tau=1.0, half_window=6.0, n_pixels=241,
                    n_frames=12, t_max=3.0, amplitude0=1.0)
    design_floor = float(np.exp(geometry["t_max"] / geometry["tau"]))

    suffix = "" if args.model == "gaussian" else f"_{args.model}"
    cal_cnr0, n_runs_cal = CAL_CNR0, N_RUNS_CAL
    cnr0_grid, n_runs = CNR0_EVAL, N_RUNS_EVAL
    cal_tag = ""
    if args.quick:
        cal_cnr0, n_runs_cal = np.geomspace(2.0, 100.0, 18), 300
        cnr0_grid, n_runs = (8.0, 21.0, 55.0), 300
        cal_tag = "_quick"
    # The calibration is floor-independent, so it is shared across floor variants;
    # only the reports/figures carry a floor tag when the floor is non-default.
    floor_tag = "" if args.floor == FLOOR_CNR else f"_floor{args.floor:g}"
    cal_path = OUTPUT / f"calibration_subdiffusive{suffix}{cal_tag}.npz"
    txt_path = OUTPUT / f"subdiffusive_alpha{suffix}{floor_tag}.txt"
    png_path = OUTPUT / f"subdiffusive_alpha{suffix}{floor_tag}.png"

    print(f"Building/loading D-free calibration ({args.model}, "
          f"{len(cal_cnr0)} CNR0 x {n_runs_cal} runs) ...")
    trunc = build_or_load_calibration(model, geometry, cal_cnr0, n_runs_cal,
                                      cal_path)

    print(f"Sweeping {len(ALPHA_TRUE)} exponents x {len(cnr0_grid)} CNR0 "
          f"x {n_runs} runs (floor CNR >= {args.floor}) ...")
    cells = run_sweep(model, trunc, geometry, cnr0_grid, n_runs, args.floor)

    calibration = (f"calibration: D-free (calibrate_d_free, D = 0), {args.model}, "
                   f"{len(cal_cnr0)} CNR0 in [{cal_cnr0[0]:g}, {cal_cnr0[-1]:g}] x "
                   f"{n_runs_cal} runs, seeds 1 + i (cache {cal_path.name})")
    write_report(cells, args.model, geometry["n_frames"], n_runs,
                 geometry["t_max"], args.floor, design_floor, txt_path, calibration)
    make_figure(cells, args.model, design_floor, png_path)


if __name__ == "__main__":
    main()
