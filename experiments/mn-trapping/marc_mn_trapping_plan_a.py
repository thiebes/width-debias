"""Plan A validation on Marc Melendez's reaction-diffusion model.

Per-frame variance recovery on the noiseless diffusion map of the 0.51% Mn
sample from Magdaleno et al. 2025 (ACS Nanosci. Au). For each frame in
Marc's noiseless diffmap, ask whether the corrected fitted
variance matches the noiseless fit.

This version uses the **joint (CNR, sigma^2) calibration** loaded from
disk: ``output/calibration_marc_gaussian_filtered.npz`` built by
``scripts/build_calibration.py`` with Marc's pixel geometry. The
correction is the ``subtract`` scheme with fixed-point iteration on the
2-D lookup (handled automatically by ``BiasLookup.var_bias`` when a
joint calibration is in place).

Run from the repo root::

    .venv/Scripts/python.exe experiments/mn-trapping/marc_mn_trapping_plan_a.py

Outputs ``output/marc_mn_trapping_plan_a.{png,txt}``.

Test B (Voigt fit on PSF-convolved diffmap) is gated on the existence of
``output/calibration_marc_voigt_filtered.npz``; it is skipped if that
calibration hasn't been built yet.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from width_debias import CalibrationGrid  # noqa: E402
from width_debias.fit import fit_variance as fit_gaussian  # noqa: E402
from width_debias.models import VoigtModel  # noqa: E402
from width_debias.bias_lookup import BiasLookup  # noqa: E402
from width_debias.weights import corrected_variance  # noqa: E402

from diffusion_map import (  # noqa: E402
    Parameters, diffusion, times as time_axis, x_axis,
)


# Physical defaults (Magdaleno et al. 2025, 0.51% Mn sample)
PAR = Parameters()

# Optical PSF (NA = 1.3 oil objective, lambda ~ 590 nm):
# diffraction-limited FWHM ~ 1.22 lambda / (2 NA) ~ 0.28 um, Gaussian
# sigma ~ FWHM / 2.355 ~ 0.12 um.
PSF_SIGMA = 0.12      # um, Gaussian convolution kernel
VOIGT_GAMMA = 0.01    # um, fixed Lorentzian HWHM in the Voigt fit

# CNR_0 sweep and per-CNR averaging budget
CNR_GRID = np.array([5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
N_SEEDS = 20

OUTPUT = Path(__file__).resolve().parents[2] / "output"
OUTPUT.mkdir(exist_ok=True)
CAL_GAUSSIAN_PATH = OUTPUT / "calibration_marc_gaussian_filtered.npz"
CAL_VOIGT_PATH = OUTPUT / "calibration_marc_voigt_filtered.npz"


def gaussian_psf_convolve(diffmap: np.ndarray, dx: float,
                          sigma_psf: float) -> np.ndarray:
    """Convolve each row with a Gaussian PSF of std ``sigma_psf``."""
    half = int(np.ceil(4 * sigma_psf / dx))
    kx = np.arange(-half, half + 1) * dx
    kernel = np.exp(-0.5 * (kx / sigma_psf) ** 2)
    kernel /= kernel.sum()
    out = np.empty_like(diffmap)
    for i in range(diffmap.shape[0]):
        out[i] = np.convolve(diffmap[i], kernel, mode="same")
    return out


def fit_each_row(x: np.ndarray, profiles: np.ndarray, fit_fn) -> np.ndarray:
    """Apply ``fit_fn(x, row)`` to every row of ``profiles``; collect variances."""
    out = np.full(profiles.shape[0], np.nan)
    for i in range(profiles.shape[0]):
        r = fit_fn(x, profiles[i])
        if getattr(r, "ok", True) and np.isfinite(r.variance):
            out[i] = r.variance
    return out


def per_frame_cnr_known(noiseless: np.ndarray, noise_sigma: float) -> np.ndarray:
    """Per-frame CNR computed from the noiseless peak and known noise level."""
    return noiseless.max(axis=1) / noise_sigma


def validation_pass(noiseless: np.ndarray, x_arr: np.ndarray, fit_fn,
                    trunc_model: BiasLookup, cnr0: float, n_seeds: int,
                    peak0: float, seed_base: int = 0):
    """Per-frame raw and corrected variance over n_seeds.

    The BiasLookup carries the joint (CNR, sigma^2) lookup; calling
    ``corrected_variance("subtract", ...)`` invokes the fixed-point
    iteration in ``model.var_bias`` automatically.
    """
    n_frames = noiseless.shape[0]
    noise_sigma = peak0 / cnr0
    cnr_frame = per_frame_cnr_known(noiseless, noise_sigma)

    raw = np.full((n_seeds, n_frames), np.nan)
    cor = np.full((n_seeds, n_frames), np.nan)
    for s in range(n_seeds):
        rng = np.random.default_rng(seed_base + s)
        noisy = noiseless + rng.normal(0.0, noise_sigma, noiseless.shape)
        var_hat = fit_each_row(x_arr, noisy, fit_fn)
        raw[s] = var_hat
        cor[s] = corrected_variance("subtract", var_hat, cnr_frame,
                                    trunc_model)

    return (np.nanmean(raw, axis=0), np.nanmean(cor, axis=0), cnr_frame)


def _load_trunc_model(path: Path, label: str) -> BiasLookup | None:
    if not path.exists():
        print(f"  ({label}) calibration not found at {path} -- skipping")
        return None
    cal = CalibrationGrid.load(path)
    print(f"  ({label}) loaded calibration {path.name}")
    print(f"    cnr grid: {cal.cnr_grid.size} log-spaced "
          f"[{cal.cnr_grid[0]:.2f}, {cal.cnr_grid[-1]:.1f}]")
    print(f"    sigma^2 grid: {cal.variance_grid.size} linear-in-sigma "
          f"[{cal.variance_grid[0]:.4g}, {cal.variance_grid[-1]:.4g}] um^2")
    print(f"    n_runs/cell: {cal.n_runs}, "
          f"model: {cal.model_name}, "
          f"pixel grid: {cal.pixel_x.size if cal.pixel_x is not None else '?'}")
    return BiasLookup.from_calibration(cal)


def main():
    print("=" * 70)
    print("Marc Mn-trapping Plan A: per-frame variance recovery")
    print("(joint 2-D calibration; subtractive subtract + fixed-point iter)")
    print("=" * 70)

    print("\nBuilding noiseless diffusion map (Marc, 0.51% Mn defaults)...")
    diffmap = diffusion(PAR)
    t_arr = time_axis(PAR)
    x_arr = x_axis(PAR)
    dx = PAR.L / PAR.nbins
    peak0_raw = float(diffmap[0].max())
    print(f"  diffmap shape: {diffmap.shape}, dx = {dx:.4f} um, "
          f"peak at t=0: {peak0_raw:.4f}")

    # ============================================================
    # Test A -- Gaussian fit on as-is diffmap
    # ============================================================
    print("\n" + "-" * 70)
    print("Test A: Gaussian fit on as-is diffmap")
    print("-" * 70)
    trunc_a = _load_trunc_model(CAL_GAUSSIAN_PATH, "Test A")
    results_a = None
    nf_var_a = None
    if trunc_a is not None:
        print("  Noiseless Gaussian fit per frame...")
        nf_var_a = fit_each_row(x_arr, diffmap, fit_gaussian)
        print(f"  noiseless fit: min = {np.nanmin(nf_var_a):.4g}, "
              f"max = {np.nanmax(nf_var_a):.4g}, "
              f"final = {nf_var_a[-1]:.4g} um^2")
        results_a = {}
        for cnr0 in CNR_GRID:
            print(f"  CNR_0 = {cnr0:.0f}: {N_SEEDS} seeds...")
            results_a[cnr0] = validation_pass(
                diffmap, x_arr, fit_gaussian, trunc_a, cnr0, N_SEEDS,
                peak0_raw, seed_base=int(cnr0) * 1000,
            )

    # ============================================================
    # Test B -- Voigt fit on PSF-convolved diffmap
    # ============================================================
    print("\n" + "-" * 70)
    print(f"Test B: Voigt fit on PSF-convolved diffmap "
          f"(PSF sigma = {PSF_SIGMA} um, Voigt gamma = {VOIGT_GAMMA} um)")
    print("-" * 70)
    trunc_b = _load_trunc_model(CAL_VOIGT_PATH, "Test B")
    results_b = None
    nf_var_b = None
    if trunc_b is not None:
        print("  Convolving diffmap with Gaussian PSF...")
        diffmap_psf = gaussian_psf_convolve(diffmap, dx, PSF_SIGMA)
        peak0_psf = float(diffmap_psf[0].max())
        print(f"  PSF-convolved peak at t=0: {peak0_psf:.4f}")
        voigt = VoigtModel(gamma=VOIGT_GAMMA)
        print("  Noiseless Voigt fit per frame...")
        nf_var_b = fit_each_row(x_arr, diffmap_psf, voigt.fit_variance)
        print(f"  noiseless fit: min = {np.nanmin(nf_var_b):.4g}, "
              f"max = {np.nanmax(nf_var_b):.4g}, "
              f"final = {nf_var_b[-1]:.4g} um^2")
        results_b = {}
        for cnr0 in CNR_GRID:
            print(f"  CNR_0 = {cnr0:.0f}: {N_SEEDS} seeds...")
            results_b[cnr0] = validation_pass(
                diffmap_psf, x_arr, voigt.fit_variance, trunc_b, cnr0,
                N_SEEDS, peak0_psf, seed_base=int(cnr0) * 1000 + 7,
            )

    # ============================================================
    # Plot
    # ============================================================
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available -- skipping figure)")
    else:
        active = [(label, results, truth, ylabel)
                  for label, results, truth, ylabel in [
                      ("Test A: Gaussian, as-is diffmap",
                       results_a, nf_var_a,
                       r"fitted $\sigma^2$ ($\mu$m$^2$)"),
                      (f"Test B: Voigt on PSF-convolved diffmap "
                       f"(PSF $\\sigma$ = {PSF_SIGMA} um)",
                       results_b, nf_var_b,
                       r"fitted $\sigma_G^2$ ($\mu$m$^2$)"),
                  ] if results is not None]
        if active:
            fig, axes = plt.subplots(1, len(active), figsize=(7 * len(active), 5),
                                      constrained_layout=True, squeeze=False)
            colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(CNR_GRID)))
            for ax, (label, results, truth, ylabel) in zip(axes[0], active):
                ax.plot(t_arr, truth, "k-", lw=2.5, label="noiseless fit",
                        zorder=10)
                for color, cnr0 in zip(colors, CNR_GRID):
                    raw, cor, _ = results[cnr0]
                    ax.plot(t_arr, raw, color=color, ls=":", lw=0.7, alpha=0.4,
                            label=f"raw, CNR$_0$={cnr0:.0f}")
                    ax.plot(t_arr, cor, color=color, ls="-", lw=1.3,
                            label=f"corrected, CNR$_0$={cnr0:.0f}")
                ax.set_xlabel("time (ns)")
                ax.set_ylabel(ylabel)
                ax.set_title(label)
                ax.legend(fontsize=6, ncol=2, loc="upper right")
                ax.grid(alpha=0.3)
                ax.set_ylim(-0.01, max(np.nanmax(truth) * 1.5, 0.15))
            fig.suptitle("Marc Mn-trapping: joint-calibration subtract\n"
                         "black: noiseless truth, solid: corrected, dotted: raw")
            fig_path = OUTPUT / "marc_mn_trapping_plan_a.png"
            fig.savefig(fig_path, dpi=150)
            print(f"\nFigure saved: {fig_path}")

    # ============================================================
    # Text summary
    # ============================================================
    lines = []
    lines.append("=" * 70)
    lines.append("Marc Mn-trapping Plan A: per-frame variance recovery")
    lines.append("(joint 2-D calibration; subtractive subtract)")
    lines.append("=" * 70)
    lines.append(f"diffmap: {PAR.nbins} pixels over {PAR.L} um, "
                 f"{int(PAR.tmax / PAR.dt)} frames over {PAR.tmax} ns")
    lines.append(f"reference: Magdaleno et al. 2025 (0.51% Mn, Voigt fit)")
    lines.append(f"n_seeds per CNR_0: {N_SEEDS}")
    lines.append("")
    for label, results, truth in [
        ("Test A -- Gaussian fit, as-is diffmap", results_a, nf_var_a),
        ("Test B -- Voigt fit on PSF-convolved diffmap", results_b, nf_var_b),
    ]:
        lines.append(f"=== {label} ===")
        if results is None:
            lines.append("  (calibration not built; skipped)")
            lines.append("")
            continue
        lines.append(f"  noiseless-fit range: "
                     f"[{np.nanmin(truth):.4g}, {np.nanmax(truth):.4g}] um^2  "
                     f"(final: {truth[-1]:.4g})")
        lines.append("  CNR_0  |  min CNR  |  mean |residual|  (um^2):  "
                     "raw     |   corrected")
        for cnr0 in CNR_GRID:
            raw, cor, cnr_frame = results[cnr0]
            mar_raw = float(np.nanmean(np.abs(raw - truth)))
            mar_cor = float(np.nanmean(np.abs(cor - truth)))
            lines.append(f"  {cnr0:>5.0f}  |  {cnr_frame.min():>7.2f}  |  "
                         f"{mar_raw:>23.4g}  |  {mar_cor:>11.4g}")
        lines.append("")

    text = "\n".join(lines)
    text_path = OUTPUT / "marc_mn_trapping_plan_a.txt"
    text_path.write_text(text)
    print(f"Summary saved: {text_path}")
    print()
    print(text)


if __name__ == "__main__":
    main()
