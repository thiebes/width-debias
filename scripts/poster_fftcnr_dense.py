"""Dense CNR-recovery sweep for the poster's fft-cnr panel (Panel 4).

An external simulation study of fft-cnr ran this sweep on a
coarse six-point CNR grid ([2, 5, 10, 20, 50, 100]); the poster's ratio plot
(recovered / true CNR) reads as six connected dots. This reproduces the same
experiment on a dense log-spaced CNR grid so the median curve and its
interquartile band are smooth, and writes a poster-specific npz so the
external six-point result is left untouched.

The clean profiles match the external sweep exactly: a generalized-Gaussian family
``amp * exp(-(|x - center| / scale) ** p)`` for p = 2 (Gaussian), 1.3 (heavy
tail), and 4 (flat top), plus a true Voigt (Gaussian core + Lorentzian wings)
that sits outside the generalized-Gaussian family and so tests genuine model
mismatch. Each profile is estimated through ``fit_model="generalized_gaussian"``
-- the path width-debias uses -- and the true CNR is amplitude / noise_std.

A noise mode is selected on the command line. ``white`` (the default) adds the
constant-sigma Gaussian noise the calibration is built on; ``shot`` uses the
project's photon-counting :class:`width_debias.noise.ShotNoise` (pure Poisson,
no read floor), whose variance follows the local signal. Under shot noise the
true CNR is the peak SNR -- peak counts / sqrt(peak counts) -- so the frame-0
peak holds ``cnr**2`` counts and the target CNR is recovered by construction;
the test asks whether fft-cnr's single, spatially-averaged noise estimate reads
that peak CNR back, since the noise is now concentrated under the peak.

Run from the repository root::

    .venv/Scripts/python.exe scripts/poster_fftcnr_dense.py [white|shot]

Output: ``output/gengauss_cnr_sweep_{dense,shot}.npz`` (drop-in for
``poster_figures.figure_fftcnr``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import voigt_profile

from fft_cnr import fft_cnr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from width_debias.noise import ShotNoise  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
OUTPUT.mkdir(exist_ok=True)

_SHOT = ShotNoise(read=0.0, poisson=True)   # photon-limited, no read floor

# Geometry and shape parameters -- identical to the external sweep.
N = 512
AMP = 1.0                      # peak amplitude; noise_std = AMP / true_cnr
SCALE = 25.0                   # generalized-Gaussian scale parameter (pixels)
VOIGT_SIGMA_G = 18.0           # Voigt Gaussian width
VOIGT_GAMMA_L = 10.0           # Voigt Lorentzian half-width
SHAPES = ["Gaussian (p=2)", "heavy-tail (p=1.3)", "flat-top (p=4)", "Voigt"]

# Dense log-spaced CNR grid (six points in the external sweep).
TRUE_CNR = np.logspace(np.log10(2.0), np.log10(100.0), 30)
N_TRIALS = 500


def make_clean(shape: str) -> np.ndarray:
    """Noise-free, peak-normalized profile of a shape, as the external sweep defines it."""
    x = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    if shape == "Gaussian (p=2)":
        return AMP * np.exp(-(np.abs(x - center) / SCALE) ** 2.0)
    if shape == "heavy-tail (p=1.3)":
        return AMP * np.exp(-(np.abs(x - center) / SCALE) ** 1.3)
    if shape == "flat-top (p=4)":
        return AMP * np.exp(-(np.abs(x - center) / SCALE) ** 4.0)
    if shape == "Voigt":
        vp = voigt_profile(x - center, VOIGT_SIGMA_G, VOIGT_GAMMA_L)
        return AMP * vp / vp.max()
    raise ValueError(shape)


def _noisy_profile(clean: np.ndarray, cnr: float, mode: str,
                   rng: np.random.Generator) -> np.ndarray:
    """One noisy realisation of ``clean`` at true CNR ``cnr`` under ``mode``."""
    if mode == "white":
        return clean + rng.normal(0.0, AMP / cnr, N)
    # Shot: treat the single profile as a one-frame stack; ShotNoise scales it so
    # the peak holds cnr**2 counts (peak SNR = cnr) and returns amplitude units.
    return _SHOT.sample(clean[None, :], cnr, rng)[0]


def sweep(mode: str) -> dict:
    """Median/IQR recovered-CNR array per shape, generalized-Gaussian path."""
    out = {}
    for shape in SHAPES:
        clean = make_clean(shape)
        gg = np.full((TRUE_CNR.size, N_TRIALS), np.nan)
        for i, cnr in enumerate(TRUE_CNR):
            for t in range(N_TRIALS):
                rng = np.random.default_rng(1000 * i + t)
                prof = _noisy_profile(clean, cnr, mode, rng)
                gg[i, t] = fft_cnr(prof, fit_model="generalized_gaussian").cnr
        out[shape] = gg
        med = np.nanmedian(gg, axis=1) / TRUE_CNR
        print(f"  {shape:20s} median ratio span: "
              f"{med.min():.3f} -> {med.max():.3f}")
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "white"
    if mode not in ("white", "shot"):
        raise SystemExit(f"noise mode must be white or shot, got {mode!r}")
    print(f"Dense fft-cnr CNR sweep [{mode}] (N={N}, {TRUE_CNR.size} CNR points, "
          f"{N_TRIALS} trials/cell)")
    res = sweep(mode)
    suffix = "dense" if mode == "white" else "shot"
    npz = OUTPUT / f"gengauss_cnr_sweep_{suffix}.npz"
    np.savez(
        npz,
        true_cnr=TRUE_CNR,
        shapes=np.array(SHAPES),
        **{f"gg_{i}": res[s] for i, s in enumerate(SHAPES)},
    )
    print(f"Saved {npz.name}")


if __name__ == "__main__":
    main()
