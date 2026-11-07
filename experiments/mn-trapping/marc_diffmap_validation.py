"""Validate the per-frame correction on Marc Melendez's Mn diffusion maps.

Marc supplied noiseless diffusion maps and ground-truth variance curves for four
Mn-doping cases (undoped, 0.51%, 0.81%, 4.04% Mn) from the Magdaleno et al. 2025
reaction-diffusion model. The maps are multi-population: a free-exciton
population and two trapped populations of different width emit independently, so
the photoluminescence is a linear sum of population terms and its second moment
is an intensity-weighted average -- the apparent (effective) transport.

Per docs/VALIDATION_DESIGN.md the truth is the carrier (transport) second
moment, supplied here as ``case.var_true``. Because the emission is linear in
the populations, that recorded second moment equals the carrier moment, so no
forward-model stage is needed (unlike the nonlinear Saha bed); the
width-extraction stage alone can reach it -- but only with the right estimator.
The profile is a mixture, so no single peaked shape matches it, and the
framework routes this bed to the moment estimator. Two paths are reported:

  * **Moment estimator -- the in-principle carrier path.** The non-truncated
    second moment closes the shape gap by construction (its noiseless value
    equals ``case.var_true``), so it is scored directly against the carrier
    moment. It takes no noise-bias correction (it is linear in the data, so
    unbiased to leading order); the cost is
    variance. On these maps that cost is severe -- under realistic noise the
    cross-seed SD reaches order 1e4-1e5 um^2 and the NaN fraction ~0.2, so the
    bare moment is not usable here. The practical carrier-recovery path for this
    mixture is therefore the forward fit of the reaction-diffusion model
    (paper-scope, as on the nonlinear Saha bed), not the bare moment.
  * **Single-shape fits + folded-normal.** A Gaussian, Voigt, or
    generalized-Gaussian fit, each with its joint (CNR, sigma^2) calibration on
    Marc's geometry. Scored two ways: ``|corrected - noiseless fit|`` isolates
    the noise (truncation) correction, which it removes; ``|corrected -
    var_true|`` shows the residual shape gap to the carrier moment, which the
    correction does not close. The noiseless-fit-vs-var_true shape gap is
    tabulated across estimators.

Run from the repository root::

    .venv/Scripts/python.exe scripts/marc_diffmap_validation.py

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
    ("gengauss", "generalized-Gaussian", DICE_GREEN,
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
    """Per-frame raw and folded-normal-corrected variance, averaged over seeds."""
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
        "Marc Mn diffusion-map validation -- carrier-moment truth (var_true)",
        "Moment estimator = carrier-recovery path; single-shape fits + folded",
        "= noise-debiased but shape-gap-limited.  (joint 2-D calibration)",
        "=" * 72,
        f"n_seeds per CNR_0: {N_SEEDS};  reliable regime: per-frame "
        f"CNR >= {FLOOR_CNR}",
        "Noise correction = |corrected - noiseless fit| (isolates distortion 1).",
        "Shape gap = mean|noiseless fit - true 2nd moment (carrier)| per frame.",
        "Carrier residual = |corrected - var_true| (total error vs the truth).",
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
                         f"[{np.nanmin(nf):.4g}, {np.nanmax(nf):.4g}] um^2, "
                         f"shape gap {gap:.4g} um^2 "
                         f"(true 2nd-moment max {case.var_true.max():.4g})")
            lines.append("   CNR_0 | min CNR | frames>=1 | raw/cor vs noiseless "
                         "fit | cor vs carrier moment")
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

    # --- MomentModel: the non-truncated estimator -------------------------
    # The second central moment IS the true-transport width by construction, so
    # it closes the shape gap -- but it takes no noise-bias correction (it is
    # linear in the data) and pays in noise variance, which the x^2 weighting amplifies in
    # the tails. We report the cross-seed SD (instability) and the NaN fraction
    # (frames where the baseline-subtracted total goes <= 0, i.e. unusable) as
    # the cost of that gap-closing.
    MOMENT_KEY, MOMENT_LABEL, MOMENT_COLOR = "moment", "second moment", "#9467bd"
    moment = MomentModel()
    noiseless_fits[MOMENT_KEY] = {}
    lines.append("#" * 72)
    lines.append(f"### Estimator: {MOMENT_LABEL}  -- the carrier-recovery path"
                 " (NON-truncated, no folded correction)")
    lines.append("   Scored against var_true (carrier moment); cost is variance"
                 " (cross-seed SD, NaN frac).")
    lines.append("#" * 72)
    for case in cases:
        nm = fit_each_row(case.x, case.diffmap, moment.fit_variance)
        noiseless_fits[MOMENT_KEY][case.name] = nm
        shape_gaps[(MOMENT_KEY, case.name)] = _residual(nm, case.var_true)
        lines.append(f"-- {case.name}: noiseless moment "
                     f"[{np.nanmin(nm):.4g}, {np.nanmax(nm):.4g}] um^2, "
                     f"shape gap {shape_gaps[(MOMENT_KEY, case.name)]:.4g} um^2")
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
        lines.append("SHAPE GAP (um^2) -- mean|noiseless fit - true 2nd moment|,"
                     " per estimator")
        lines.append("=" * 72)
        header = "  case            " + "".join(f"{lbl[:12]:>14}"
                                                for _k, lbl in table_cols)
        lines.append(header)
        for case in cases:
            row = f"  {case.name:14s}" + "".join(
                f"{shape_gaps[(k, case.name)]:14.4g}" for k, _lbl in table_cols)
            lines.append(row)
        lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUTPUT / "marc_diffmap_validation.txt").write_text(report)

    # --- Figure: shape gap per estimator across the doping series ---------
    if present:
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True)
        for ax, case in zip(axes.ravel(), cases):
            ax.plot(case.t, case.var_true, color="k", lw=1.6, ls=":",
                    label="true 2nd moment")
            for key, label, color, _f, _fn in present:
                ax.plot(case.t, noiseless_fits[key][case.name], color=color,
                        lw=1.7, label=f"{label} fit")
            if MOMENT_KEY in noiseless_fits:
                ax.plot(case.t, noiseless_fits[MOMENT_KEY][case.name],
                        color=MOMENT_COLOR, lw=1.5, ls="--",
                        label="second moment")
            ax.set_title(case.name, fontsize=10)
            ax.tick_params(direction="in", which="both", top=True, right=True)
            ax.set_ylim(0, max(case.var_true.max() * 1.2, 0.03))
            ax.grid(True, alpha=0.2)
        axes[0, 0].legend(fontsize=7, loc="upper left")
        for ax in axes[1, :]:
            ax.set_xlabel("time  (ns)")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"variance  $\sigma^2(t)$  (um$^2$)")
        fig.suptitle("Estimator shape gap on Marc's diffusion maps "
                     "(noiseless fits vs true 2nd moment)", fontsize=12)
        fig.tight_layout()
        path = OUTPUT / "marc_diffmap_validation.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"\nSaved figure: {path.name}")


if __name__ == "__main__":
    main()
