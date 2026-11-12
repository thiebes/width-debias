"""Diagnose the high-spread overcorrection of the ``subtract`` scheme on the Saha bed.

At low CNR_0 and high spread, the correction pulls the recovered alpha below the
noise-free-fit reference. The shape-mismatch explanation is ruled out (the Saha
emission profile is near-Gaussian, beta ~ 1.93), so this script tests the next
hypothesis: the 2-D calibration over-estimates the per-frame variance bias at
the low-CNR late frames and the correction therefore over-subtracts.

Per frame at several operating points it compares two quantities:

* B_actual  -- the real noise-induced Gaussian-fit bias, measured by Monte Carlo
               on the actual Saha profile: E[sigma_hat^2] - sigma^2_ref, where
               sigma^2_ref is the noise-free fit variance.
* B_calib   -- what the correction actually subtracts: the mean over the same
               realisations of model.var_bias(cnr, sigma_hat^2) (the 2-D lookup
               with its fixed-point inversion, exactly as `corrected_variance`
               calls it).

The gap B_calib - B_actual is the mean of a paired difference over the same
realisations, so its SE is the SD of that difference over sqrt(n). A frame is
flagged only where the gap is resolved (|z| > 2); a fixed threshold would flag
Monte Carlo noise at low CNR and miss small resolved gaps at high CNR. The SE
leaves out the calibration's own Monte Carlo error.

The per-frame floor that the real-data scripts apply (FLOOR = 1.3) is read from
the summary: walking down from the highest per-frame CNR, the first frame whose
gap is resolved at 3 SE, and the lowest clean frame above it.

If B_calib > B_actual at the low-CNR late frames, the overcorrection is a
calibration-accuracy problem (deep-CNR MC noise), pointing the fix at a denser /
higher-run-count low-CNR calibration rather than a shape-matched estimator.

    .venv/Scripts/python.exe experiments/saha-reaction-diffusion/diagnose_overcorrection.py

Writes ``output/saha-reaction-diffusion/diagnose_overcorrection.txt``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, _HERE)

import nonfickian_validation as nf  # noqa: E402
from width_debias.fit import fit_variance  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "output" / "saha-reaction-diffusion"
OPERATING_POINTS = ((2000.0, 6.0), (2000.0, 12.0), (2000.0, 25.0), (2000.0, 50.0))
N_MC = 4000
FLOOR_Z = 3.0


def diagnose(A: float, cnr0: float, lines: list[str], frames: list[tuple],
             power: int = 2, n_mc: int = N_MC, seed: int = 0) -> None:
    rg = nf.build_regime(A, power)
    model = nf.load_model()
    if model is None:
        sys.exit(f"no calibration at {nf.CALIBRATION_PATH}")
    noise_sigma = rg.peak[0] / cnr0
    cnr_frame = rg.peak / noise_sigma
    rng = np.random.default_rng(seed)

    lines.append(f"A={A:.0f}, CNR_0={cnr0:g}, emission power={power}, {n_mc} MC fits/frame, "
                 f"seed {seed}")
    lines.append(" frame  t(ns)   cnr  sigma2_ref  B_actual  B_calib      gap      SE      z"
                 "   verdict")
    for f in range(nf.N_FRAMES):
        ref = rg.gauss_var[f]
        clean = rg.clean[f]
        fitted = np.empty(n_mc)
        subtracted = np.empty(n_mc)
        for r in range(n_mc):
            noisy = clean + rng.normal(0.0, noise_sigma, size=clean.shape)
            v = fit_variance(nf.PIXEL_X, noisy).variance
            fitted[r] = v
            subtracted[r] = model.var_bias(float(cnr_frame[f]), v)
        b_actual = float(fitted.mean()) - ref
        b_calib = float(subtracted.mean())
        gap = b_calib - b_actual
        se = float(np.std(subtracted - fitted, ddof=1) / np.sqrt(n_mc))
        z = gap / se
        verdict = "over" if z > 2 else ("under" if z < -2 else "ok")
        frames.append((float(cnr_frame[f]), gap, se, z))
        lines.append(f"  {f:2d}   {rg.times[f]:5.2f}  {cnr_frame[f]:5.2f}   {ref:.4f}"
                     f"   {b_actual:+.4f}  {b_calib:+.4f}  {gap:+.4f}  {se:.4f}  {z:+5.1f}"
                     f"   {verdict}")
    lines.append("")


def main() -> None:
    lines: list[str] = []
    frames: list[tuple] = []
    for i, (A, cnr0) in enumerate(OPERATING_POINTS):
        diagnose(A, cnr0, lines, frames, seed=i)
        print("\n".join(lines[-(nf.N_FRAMES + 3):]), flush=True)
    cnr, gap, se, z = (np.array(c) for c in zip(*frames))
    lines.append("Summary over every frame of every operating point (sigma^2 in um^2):")
    # Walk down from the highest per-frame CNR to the first frame whose gap is
    # resolved at FLOOR_Z SE in either direction. With about 50 frames a 2 SE
    # rule stops on noise (one frame in twenty), so the walk uses 3 SE.
    order = np.argsort(cnr)[::-1]
    first = next((i for i in order if abs(z[i]) > FLOOR_Z), None)
    if first is None:
        lines.append(f"  no frame has a gap beyond {FLOOR_Z:g} SE")
    else:
        clean = cnr > cnr[first]
        kind = "over" if z[first] > 0 else "under"
        lines.append(f"  first resolved gap walking down in CNR (|z| > {FLOOR_Z:g}): {kind}-"
                     f"subtraction at per-frame CNR {cnr[first]:.2f} (gap {gap[first]:+.4f}, "
                     f"z {z[first]:+.1f})")
        lines.append(f"  above it: {int(clean.sum())} frames from CNR {cnr[clean].min():.2f} up, "
                     f"gaps {gap[clean].min():+.4f} to {gap[clean].max():+.4f}, "
                     f"|z| <= {np.abs(z[clean]).max():.1f}")
        lines.append(f"  per-frame floor implied: between CNR {cnr[first]:.2f} and "
                     f"{cnr[clean].min():.2f} (the scripts use 1.3)")
    below = cnr < 1.0
    lines.append(f"  frames below per-frame CNR 1 ({int(below.sum())}) are below the floor and "
                 "dropped in every application; their gaps are not a target")
    report = "\n".join(lines)
    print("\n".join(lines[-6:]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnose_overcorrection.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
