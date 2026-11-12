"""Validate the per-frame correction on Marc Melendez's Mn diffusion maps.

Marc supplied noiseless diffusion maps and variance curves for four Mn-doping
cases (undoped, 0.51%, 0.81%, 4.04% Mn) from the Magdaleno et al. 2025
reaction-diffusion model. The maps are multi-population: free excitons and two
trapped populations of different width emit independently, so the
photoluminescence (PL) is a linear sum of population terms, each weighted by
its emission rate per carrier (q_i ν_i in ``diffusion_map.py``; for the 0.51% Mn
defaults 0.12, 0.0022, and 0.10 /ns for free, Mn-trapped, and defect-trapped).

The reference is ``case.var_true``, the second moment of the PL profile. It is
emission-weighted: each population counts by its emitted intensity, not by its
carrier number, so it equals the carrier second moment only when every
population has the same emission rate per carrier, which is not the case here.
Reaching the carrier moment needs the forward model (as on the Saha bed); this
script scores the width-extraction stage against the PL moment only. The
profile is a mixture, so no single peaked shape matches it. Two paths are
reported:

  * **Moment estimator.** The second moment, with no lower bound on the width,
    closes the shape gap to the PL moment by construction (its noiseless value
    equals ``case.var_true``). It is not corrected here; its cost is variance.
    On these maps that cost is severe -- under realistic noise the cross-seed
    SD reaches order 1e5 μm² and the NaN fraction about 0.2, so the bare moment
    is not usable here. The practical path to the carrier distribution for this
    mixture is the forward fit of the reaction-diffusion model (paper-scope, as
    on the Saha bed), not the bare moment.
  * **Single-shape fits + correction.** A Gaussian, Voigt, or
    generalized-Gaussian fit, each with its joint (CNR, sigma^2) calibration on
    Marc's geometry. The generalized-Gaussian arm is
    ``GeneralizedGaussianModel()`` at its default p = 2, a Gaussian profile
    fitted by scale (its calibration was also built at p = 2), so it differs
    from the Gaussian arm only in its fit bounds and calibration. Scored two
    ways: ``|corrected - noiseless fit|`` isolates the noise correction;
    ``|corrected - var_true|`` adds the shape gap to the PL moment, which the
    correction does not close. The noiseless-fit-vs-var_true shape gap is
    tabulated across estimators.

Run from the repository root::

    .venv/Scripts/python.exe experiments/mn-trapping/marc_diffmap_validation.py

Outputs ``output/marc_diffmap_validation.{png,txt}``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from width_debias import CalibrationGrid  # noqa: E402
from width_debias.fit import fit_variance as fit_gaussian  # noqa: E402
from width_debias.models import (GeneralizedGaussianModel,  # noqa: E402
                                   MomentModel, VoigtModel)
from width_debias.bias_lookup import BiasLookup  # noqa: E402
from width_debias.weights import corrected_variance  # noqa: E402

from marc_data import load_all  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[2] / "output"
OUTPUT.mkdir(exist_ok=True)

DICE_BLUE = "#003f7f"
DICE_GOLD = "#f7941e"
DICE_GREEN = "#0cce6b"

VOIGT_GAMMA = 0.01        # must match the Voigt calibration build
CNR_GRID = (10.0, 20.0, 50.0, 100.0, 200.0)
N_SEEDS = 15
FLOOR_CNR = 1.0           # per-frame CNR above which the correction is reliable

# (key, label, colour, calibration file, fit function) -- one per estimator.
ESTIMATORS = [
    ("gaussian", "Gaussian", DICE_BLUE,
     "calibration_marc_gaussian_filtered.npz", fit_gaussian),
    ("voigt", "Voigt", DICE_GOLD,
     "calibration_marc_voigt_filtered.npz",
     VoigtModel(gamma=VOIGT_GAMMA).fit_variance),
    ("gengauss", "gen. Gaussian (p = 2)", DICE_GREEN,
     "calibration_marc_gengauss_filtered.npz",
     GeneralizedGaussianModel().fit_variance),
]


def fit_each_row(x: np.ndarray, profiles: np.ndarray, fit_fn) -> np.ndarray:
    """Variance fit of every row with ``fit_fn``; NaN where the fit fails."""
    out = np.full(profiles.shape[0], np.nan)
    for i in range(profiles.shape[0]):
        r = fit_fn(x, profiles[i])
        if getattr(r, "ok", True) and np.isfinite(r.variance):
            out[i] = r.variance
    return out


def validation_pass(case, trunc: BiasLookup, fit_fn, cnr0: float,
                    n_seeds: int):
    """Per-frame raw and corrected variance, averaged over seeds."""
    peak0 = float(case.diffmap[0].max())
    noise_sigma = peak0 / cnr0
    cnr_frame = case.diffmap.max(axis=1) / noise_sigma

    n_frames = case.diffmap.shape[0]
    raw = np.full((n_seeds, n_frames), np.nan)
    cor = np.full((n_seeds, n_frames), np.nan)
    for s in range(n_seeds):
        rng = np.random.default_rng(int(cnr0) * 1000 + s)
        noisy = case.diffmap + rng.normal(0.0, noise_sigma, case.diffmap.shape)
        var_hat = fit_each_row(case.x, noisy, fit_fn)
        raw[s] = var_hat
        cor[s] = corrected_variance("subtract", var_hat, cnr_frame, trunc)
    return (np.nanmean(raw, axis=0), np.nanmean(cor, axis=0), cnr_frame)


def _residual(recovered, truth, mask=None) -> float:
    d = np.abs(recovered - truth)
    if mask is not None:
        d = d[mask]
    return float(np.nanmean(d))


def main() -> None:
    cases = load_all()
    lines = [
        "=" * 72,
        "Marc Mn diffusion-map validation -- reference: PL second moment",
        "(var_true, emission-weighted; not the carrier moment).",
        "Moment estimator closes the shape gap to the PL moment; single-shape",
        "fits + correction = noise-debiased but shape-gap-limited.",
        "Calibration: joint (CNR, sigma^2) grid per estimator (named below).",
        "=" * 72,
        f"n_seeds per CNR_0: {N_SEEDS} (seed int(CNR_0) * 1000 + s);  "
        f"reliable regime: per-frame CNR >= {FLOOR_CNR}",
        "Noise correction = |corrected - noiseless fit| (isolates distortion 1).",
        "Shape gap = mean|noiseless fit - PL 2nd moment| per frame.",
        "PL residual = |corrected - var_true| (noise and shape gap together).",
        "",
    ]

    # noiseless_fits[est_key][case.name] = noiseless-fit trajectory
    noiseless_fits: dict = {}
    shape_gaps: dict = {}          # (est_key, case.name) -> scalar gap

    for key, label, _color, cal_file, fit_fn in ESTIMATORS:
        cal_path = OUTPUT / cal_file
        if not cal_path.exists():
            lines.append(f"### {label}: calibration {cal_file} not found "
                         f"-- skipped\n")
            continue
        trunc = BiasLookup.from_calibration(CalibrationGrid.load(cal_path))
        noiseless_fits[key] = {}
        lines.append("#" * 72)
        lines.append(f"### Estimator: {label}   (calibration: {cal_file})")
        lines.append("#" * 72)
        for case in cases:
            nf = fit_each_row(case.x, case.diffmap, fit_fn)   # noiseless fit
            noiseless_fits[key][case.name] = nf
            gap = _residual(nf, case.var_true)
            shape_gaps[(key, case.name)] = gap
            lines.append(f"-- {case.name}: noiseless fit "
                         f"[{np.nanmin(nf):.4g}, {np.nanmax(nf):.4g}] μm², "
                         f"shape gap {gap:.4g} μm² "
                         f"(PL 2nd-moment max {case.var_true.max():.4g})")
            lines.append("   CNR_0 | min CNR | frames>=1 | raw/cor vs noiseless "
                         "fit | cor vs PL moment")
            for cnr0 in CNR_GRID:
                raw, cor, cnr_frame = validation_pass(
                    case, trunc, fit_fn, cnr0, N_SEEDS)
                work = cnr_frame >= FLOOR_CNR
                n_work = int(work.sum())
                r_raw = _residual(raw, nf, work)
                r_cor = _residual(cor, nf, work)
                r_cor_truth = _residual(cor, case.var_true, work)
                lines.append(f"   {cnr0:5.0f} | {cnr_frame.min():7.2f} | "
                             f"{n_work:4d}/{cnr_frame.size} | "
                             f"{r_raw:8.4g} / {r_cor:8.4g} | {r_cor_truth:8.4g}")
            lines.append("")

    # --- MomentModel: the estimator with no lower bound --------------------
    # The second central moment equals the PL moment by construction, so it
    # closes the shape gap -- but it is not corrected here and pays in noise
    # variance, which the x^2 weighting amplifies in the tails. We report the
    # cross-seed SD (instability) and the NaN fraction (frames where the
    # baseline-subtracted total goes <= 0, i.e. unusable) as the cost of that
    # gap-closing.
    MOMENT_KEY, MOMENT_LABEL, MOMENT_COLOR = "moment", "second moment", "#555555"
    moment = MomentModel()
    noiseless_fits[MOMENT_KEY] = {}
    lines.append("#" * 72)
    lines.append(f"### Estimator: {MOMENT_LABEL}  -- closes the shape gap to the"
                 " PL moment (no lower bound, no correction)")
    lines.append("   Scored against var_true (PL moment); cost is variance"
                 " (cross-seed SD, NaN frac).")
    lines.append("#" * 72)
    for case in cases:
        nm = fit_each_row(case.x, case.diffmap, moment.fit_variance)
        noiseless_fits[MOMENT_KEY][case.name] = nm
        shape_gaps[(MOMENT_KEY, case.name)] = _residual(nm, case.var_true)
        lines.append(f"-- {case.name}: noiseless moment "
                     f"[{np.nanmin(nm):.4g}, {np.nanmax(nm):.4g}] μm², "
                     f"shape gap {shape_gaps[(MOMENT_KEY, case.name)]:.4g} μm²")
        lines.append("   CNR_0 | min CNR | frames>=1 | mean|noisy - var_true| | "
                     "cross-seed SD | NaN frac")
        n_frames = case.diffmap.shape[0]
        peak0 = float(case.diffmap[0].max())
        for cnr0 in CNR_GRID:
            noise_sigma = peak0 / cnr0
            cnr_frame = case.diffmap.max(axis=1) / noise_sigma
            work = cnr_frame >= FLOOR_CNR
            noisy = np.full((N_SEEDS, n_frames), np.nan)
            for s in range(N_SEEDS):
                rng = np.random.default_rng(int(cnr0) * 1000 + s)
                prof = case.diffmap + rng.normal(0.0, noise_sigma,
                                                 case.diffmap.shape)
                noisy[s] = fit_each_row(case.x, prof, moment.fit_variance)
            resid = _residual(np.nanmean(noisy, axis=0), case.var_true, work)
            sd = (float(np.nanmean(np.nanstd(noisy, axis=0)[work]))
                  if work.any() else float("nan"))
            nan_frac = (float(np.mean(np.isnan(noisy[:, work])))
                        if work.any() else float("nan"))
            lines.append(f"   {cnr0:5.0f} | {cnr_frame.min():7.2f} | "
                         f"{int(work.sum()):4d}/{cnr_frame.size} | "
                         f"{resid:8.4g} | {sd:8.4g} | {nan_frac:6.3f}")
        lines.append("")

    # --- Shape-gap comparison table (the headline) ------------------------
    present = [e for e in ESTIMATORS if e[0] in noiseless_fits]
    table_cols = [(e[0], e[1]) for e in present]
    if MOMENT_KEY in noiseless_fits:
        table_cols.append((MOMENT_KEY, MOMENT_LABEL))
    if table_cols:
        lines.append("=" * 72)
        lines.append("SHAPE GAP (μm²) -- mean|noiseless fit - PL 2nd moment|,"
                     " per estimator")
        lines.append("=" * 72)
        header = "  case            " + "".join(f"{lbl[:21]:>23}"
                                                for _k, lbl in table_cols)
        lines.append(header)
        for case in cases:
            row = f"  {case.name:14s}" + "".join(
                f"{shape_gaps[(k, case.name)]:23.4g}" for k, _lbl in table_cols)
            lines.append(row)
        lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUTPUT / "marc_diffmap_validation.txt").write_text(report + "\n",
                                                        encoding="utf-8")

    # --- Figure: shape gap per estimator across the doping series ---------
    if present:
        inch = 1 / 2.54
        font = 8
        fig, axes = plt.subplots(2, 2, figsize=(17 * inch, 12.5 * inch),
                                 sharex=True, layout="constrained")
        fig.patch.set_alpha(0)
        for letter, ax, case in zip("abcd", axes.ravel(), cases):
            ax.set_facecolor("white")
            ax.plot(case.t, case.var_true, color="k", lw=1.6, ls=":",
                    label="PL 2nd moment")
            for key, label, color, _f, _fn in present:
                # The p = 2 fit equals the Gaussian one; dash it so both show.
                ls = (0, (4, 3)) if key == "gengauss" else "-"
                ax.plot(case.t, noiseless_fits[key][case.name], color=color,
                        lw=2, ls=ls, label=f"{label} fit")
            if MOMENT_KEY in noiseless_fits:
                ax.plot(case.t, noiseless_fits[MOMENT_KEY][case.name],
                        color=MOMENT_COLOR, lw=1.5, ls="--",
                        label="moment estimator")
            ax.text(0.03, 0.95, f"({letter})", transform=ax.transAxes,
                    va="top", fontsize=font, fontweight="bold")
            ax.text(0.12, 0.95, case.name, transform=ax.transAxes,
                    va="top", fontsize=font)
            ax.tick_params(direction="in", which="both", top=False,
                           right=False, labelsize=font)
            ax.set_ylim(0, max(case.var_true.max() * 1.2, 0.03))
        axes[1, 1].legend(fontsize=font, loc="lower right", frameon=False)
        for ax in axes[1, :]:
            ax.set_xlabel(r"$t\,(\mathrm{ns})$", fontsize=font)
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$\sigma^2(t)\,(\mu\mathrm{m}^2)$", fontsize=font)
        path = OUTPUT / "marc_diffmap_validation.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"\nSaved figure: {path.name}")


if __name__ == "__main__":
    main()
