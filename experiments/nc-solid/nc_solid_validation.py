"""Validate the noise-bias correction on the Solari disordered-NC-solid maps.

This is the poster's headline regime on real model transport: CH3NH3PbBr3
nanocrystal-solid diffusion maps (Marc Melendez's runs of the Solari et al.
hopping model, made for this work; see `nc_data`), L_D/FWHM ~ 0.14.
The maps track position only, so `nc_data` (this folder)
imposes the population decay `N0 exp(-t/tau)` (paper lifetime tau ~ 7 ns) and
projects each 2-D frame to a 1-D profile.

First pass uses the Gaussian fit and the NC-solid-geometry calibration
(`calibration_nc_solid_gaussian.npz`, built at this pixel grid). The paper fits
a Voigt to each slice, so a Voigt pass is the natural refinement once this
validates the mechanics.

Per docs/VALIDATION_DESIGN.md the reference is the carrier (true) second moment
`var_true`. The NC-solid emission is linear in carrier density -- the imposed
decay is a per-frame scalar, so it sets the per-frame contrast-to-noise without
changing the spatial moment -- so the recorded-profile second moment equals the
carrier moment, and the near-Gaussian shape makes the noiseless Gaussian fit
match it (a negligible shape gap, reported here as a check). D and alpha are
scored against the carrier second moment; the noise-bias correction removes
the per-frame noise bias.

The calibration file is a ``CalibrationGrid`` over (CNR, sigma^2), so the
lookup is the experimental 2-D path, keyed on each frame's own fitted variance;
the recommended path is the 1-D D-free lookup. The maps are used raw
(``load_all()``; ``denoise=False``), with their finite-walker sampling noise;
the de-noised maps are the bed of ``nc_solid_alpha_vs_cnr.py`` and the figure.
Each exponent carries a bootstrap SE over the noise realisations.

Run from the repository root::

    .venv/Scripts/python.exe experiments/nc-solid/nc_solid_validation.py

Outputs ``output/nc_solid_validation.{png,txt}``.
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
from scipy.optimize import curve_fit  # noqa: E402

from width_debias import CalibrationGrid  # noqa: E402
from width_debias.fit import fit_variance as fit_gaussian  # noqa: E402
from width_debias.bias_lookup import BiasLookup  # noqa: E402
from width_debias.weights import corrected_variance  # noqa: E402

from nc_data import load_all  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[2] / "output"
OUTPUT.mkdir(exist_ok=True)
CAL_PATH = OUTPUT / "calibration_nc_solid_gaussian.npz"

DICE_BLUE = "#003f7f"
DICE_GOLD = "#f7941e"

CNR_GRID = (3.0, 5.0, 10.0, 20.0, 50.0, 100.0)
FIG_CNR0 = 20.0
N_SEEDS = 400        # noise realisations per (map, CNR_0); 15 left the exponent unresolved
N_BOOT = 200         # resamples of the realisations for the exponent SE
FLOOR_CNR = 1.0


def fit_msd_powerlaw(t: np.ndarray, var: np.ndarray,
                     mask: np.ndarray | None = None) -> tuple[float, float]:
    """Field-standard MSD = 2 D t^alpha fit; returns (D, alpha).

    MSD uses the fitted t=0 width: ``msd = var - var[0]`` (per the project
    invariant). Fit over t > 0; NaN frames are dropped. If ``mask`` is given
    (e.g. frames above the per-profile CNR floor), only those frames are fit --
    this is the "discard sub-floor frames" design rule.
    """
    msd = var - var[0]
    m = (t > 0) & np.isfinite(msd)
    if mask is not None:
        m = m & mask
    if m.sum() < 3:
        return np.nan, np.nan
    try:
        popt, _ = curve_fit(
            lambda tt, d, a: 2.0 * d * tt ** a, t[m], msd[m],
            p0=[max(msd[m][-1] / (2.0 * t[m][-1]), 1e-6), 1.0],
            bounds=([0.0, 0.2], [np.inf, 2.0]), maxfev=20000)
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        return np.nan, np.nan


def fit_each_row(x: np.ndarray, profiles: np.ndarray) -> tuple[np.ndarray, int]:
    """Fitted variance per row, and the number of failed fits.

    A failed fit keeps its fallback value, as the calibration's Monte Carlo
    does: dropping it here but not there would compare two different
    estimators, and the dropped fits are the heavy right tail that carries the
    bias.
    """
    out = np.full(profiles.shape[0], np.nan)
    failed = 0
    for i in range(profiles.shape[0]):
        r = fit_gaussian(x, profiles[i])
        failed += not getattr(r, "ok", True)
        if np.isfinite(r.variance):
            out[i] = r.variance
    return out, failed


def validation_pass(case, trunc: BiasLookup, cnr0: float, n_seeds: int,
                    case_index: int):
    """Per-realisation raw and corrected trajectories at one CNR_0.

    Seeds are offset by 10^6 per map, so the maps see independent noise.
    """
    peak0 = float(case.profiles[0].max())
    noise_sigma = peak0 / cnr0
    cnr_frame = case.profiles.max(axis=1) / noise_sigma
    n_frames = case.profiles.shape[0]
    raw = np.full((n_seeds, n_frames), np.nan)
    cor = np.full((n_seeds, n_frames), np.nan)
    failed = 0
    for s in range(n_seeds):
        rng = np.random.default_rng(10 ** 6 * case_index + int(cnr0) * 1000 + s)
        noisy = case.profiles + rng.normal(0.0, noise_sigma, case.profiles.shape)
        var_hat, n_failed = fit_each_row(case.x, noisy)
        failed += n_failed
        raw[s] = var_hat
        cor[s] = corrected_variance("subtract", var_hat, cnr_frame, trunc)
    return raw, cor, cnr_frame, failed


def _alpha_se(t: np.ndarray, per_seed: np.ndarray, mask, rng) -> float:
    """SE of the mean-trajectory exponent: refit the mean of resampled realisations."""
    n = per_seed.shape[0]
    boot = [fit_msd_powerlaw(t, np.nanmean(per_seed[rng.integers(0, n, n)], axis=0),
                             mask=mask)[1] for _ in range(N_BOOT)]
    return float(np.nanstd(boot, ddof=1))


def _residual(recovered, truth, mask=None) -> float:
    d = np.abs(recovered - truth)
    if mask is not None:
        d = d[mask]
    return float(np.nanmean(d))


def main() -> None:
    if not CAL_PATH.exists():
        print(f"Calibration not found: {CAL_PATH.name}. Build it with "
              "scripts/build_calibration.py (tag nc_solid_gaussian) first.")
        return
    trunc = BiasLookup.from_calibration(CalibrationGrid.load(CAL_PATH))
    cases = load_all()
    if not cases:
        print("No NC-solid maps found on the pshare; nothing to validate.")
        return

    lines = [
        "=" * 70,
        "Disordered NC-solid validation (Solari et al.) -- folded-normal",
        "Gaussian fit; reference = carrier 2nd moment (var_true); decay tau = "
        f"{cases[0].tau_ns:.0f} ns",
        "=" * 70,
        f"n_seeds per CNR_0: {N_SEEDS};  reliable regime: per-frame "
        f"CNR >= {FLOOR_CNR}",
        "",
    ]
    fig_rows = {}
    rng = np.random.default_rng(0)
    for case_index, case in enumerate(cases):
        nf, _ = fit_each_row(case.x, case.profiles)           # noiseless fit
        gap = _residual(nf, case.var_true)
        D_true, a_true = fit_msd_powerlaw(case.t, case.var_true)  # carrier truth
        D_nf, a_nf = fit_msd_powerlaw(case.t, nf)             # noiseless-fit check
        lines.append("-" * 70)
        lines.append(f"=== {case.name} ===   noiseless fit "
                     f"[{np.nanmin(nf):.4g}, {np.nanmax(nf):.4g}] um^2, "
                     f"shape gap {gap:.4g} um^2")
        lines.append(f"  carrier-moment truth MSD fit:  D = {D_true:.4g} "
                     f"um^2/ns, alpha = {a_true:.3f}")
        lines.append(f"  noiseless Gaussian-fit MSD:    D = {D_nf:.4g} um^2/ns, "
                     f"alpha = {a_nf:.3f}  (shape-gap check vs truth)")
        lines.append("  naive = all-frames raw fit;  method = corrected + "
                     "sub-floor frames dropped (CNR >= 1)")
        lines.append("  CNR_0 | min CNR | n>=1 | D naive/method | "
                     "alpha naive (SE) / method (SE) | failed fits  "
                     "(truth D={:.3g}, alpha={:.2f})".format(D_true, a_true))
        for cnr0 in CNR_GRID:
            raw_s, cor_s, cnr_frame, failed = validation_pass(case, trunc, cnr0, N_SEEDS,
                                                              case_index)
            raw, cor = np.nanmean(raw_s, axis=0), np.nanmean(cor_s, axis=0)
            work = cnr_frame >= FLOOR_CNR
            D_naive, a_naive = fit_msd_powerlaw(case.t, raw)          # all, raw
            D_meth, a_meth = fit_msd_powerlaw(case.t, cor, mask=work)  # cut+cor
            se_naive = _alpha_se(case.t, raw_s, None, rng)
            se_meth = _alpha_se(case.t, cor_s, work, rng)
            lines.append(f"  {cnr0:5.0f} | {cnr_frame.min():7.2f} | "
                         f"{int(work.sum()):2d}/{cnr_frame.size} | "
                         f"{D_naive:.3g}/{D_meth:.3g} | "
                         f"{a_naive:.3f} ({se_naive:.3f}) / {a_meth:.3f} ({se_meth:.3f}) | "
                         f"{failed} of {raw_s.size}")
            if cnr0 == FIG_CNR0:
                fig_rows[case.name] = (cnr_frame, raw, cor, nf)
        lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUTPUT / "nc_solid_validation.txt").write_text(report)

    # --- Figure: recovered vs reference variance per map at FIG_CNR0 ------
    fig, axes = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 3.6),
                             squeeze=False)
    for ax, case in zip(axes[0], cases):
        cnr_frame, raw, cor, nf = fig_rows[case.name]
        work = cnr_frame >= FLOOR_CNR
        ax.plot(case.t, case.var_true, color="k", lw=1.7,
                label="carrier 2nd moment (truth)")
        ax.plot(case.t, nf, color="#888888", lw=1.0, ls=":",
                label="noiseless Gaussian fit")
        ax.plot(case.t, raw, ".", color=DICE_GOLD, ms=4, alpha=0.7,
                label="raw fit")
        ax.plot(case.t, cor, ".", color=DICE_BLUE, ms=4, label="corrected")
        if work.any() and not work.all():
            ax.axvline(case.t[work][-1], color="#bbbbbb", ls="--", lw=1.0)
        ax.set_title(f"{case.name}", fontsize=10)
        ax.set_xlabel("time  (ns)")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.grid(True, alpha=0.2)
    axes[0][0].set_ylabel(r"variance  $\sigma^2(t)$  (um$^2$)")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Correction on Solari NC-solid maps  (CNR$_0$ = "
                 f"{FIG_CNR0:.0f}, tau = {cases[0].tau_ns:.0f} ns)", fontsize=12)
    fig.tight_layout()
    path = OUTPUT / "nc_solid_validation.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure: {path.name}")


if __name__ == "__main__":
    main()
