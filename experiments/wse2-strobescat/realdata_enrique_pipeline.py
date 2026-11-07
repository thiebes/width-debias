"""Reproduce Enrique's diffusion fit exactly, then apply our bias correction.

Caveat: the calibration subtracted here is built for this repo's bounded
Gaussian width fit, but the widths it is applied to come from Enrique's own fit,
a different estimator with its own bias at low CNR. The corrected values from
this script are therefore not a valid correction of his widths; the
estimator-matched pipeline is ``realdata_diagnostics.py``.

Enrique's pipeline (``diff_eval_app.py`` ``DiffusionAnalysis`` / the merged-array
D fit) takes his per-time fitted widths, merges the interleaved sweeps onto one
sorted time axis, and fits

    MSD(t) = sigma^2(t) - sigma^2(0) = 2 D (t - t0)         (forced through origin)

over ``t in [t0, time_fit]`` with inverse-variance weights from his fit-error
``errorsquare = |2 sigma * sigma_err|``. His stored ``time_fit = 2 ns`` and
``cutoff = 0`` for every measurement here -- so his "clean" D is the early,
high-CNR regime (which is exactly why it looks clean: the noise bias is
small there).

This script:
1. reproduces that D from his stored ``sigmas`` / ``err_sigmas`` / ``time_arr``
   (his pipeline output -- nothing re-fit),
2. applies our correction by subtracting ``var_bias(CNR) * sigma0^2`` from each
   frame's sigma^2 (CNR from fft-cnr on his profiles; var_bias from the
   white D-free calibration at the measurement geometry), then refits with his
   exact recipe.

Everything is in his um^2 units (his pipeline's sigma(um), which is not tied to
the 0.060 um pixel, ``UM_PER_PX``). The headline is the corrected/his ratio, which is unit-free.
Even a small-but-real, correctable bias on a clean high-CNR measurement is the
point.

Usage::

    .venv/Scripts/python.exe scripts/realdata_enrique_pipeline.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from realdata_diagnostics import (MEASUREMENTS, FIT_HALF_SIGMA,  # noqa: E402
                                  load_measurement, per_frame_cnr,
                                  build_or_load_calibration, decide_noise_law)
from width_debias.fit import fit_variance  # noqa: E402

WIP = os.path.join(os.path.dirname(__file__), "diagnostics")
RED = "#c60b1e"
BLUE = "#0077c8"
GRAY = "0.6"
INCH = 1 / 2.54
LABELS = ["wse2_betterpower_5MHz_0p022", "wse2_250606_0p025",
          "wse2_250722_0p043", "mixed_tmdc_26_32_250716",
          "mixed_tmdc_comparison_all_0", "mixed_tmdc_pureS_250701",
          "mixed_tmdc_50_50_250702", "wse2_250224_0p020"]


def _his_D(t, var, errsq, t0, time_fit):
    """Enrique's fit: MSD = 2 D (t - t0) forced through the origin, inverse-
    variance weighted (w = 1/errorsquare^2), over t in [t0, time_fit]."""
    msd = var - var[0]
    sel = t <= time_fit + 1e-9
    x = (t[sel] - t0)
    y = msd[sel]
    w = 1.0 / np.maximum(errsq[sel], 1e-12) ** 2
    # forced-through-origin weighted slope: minimise sum w (y - m x)^2
    m = float(np.sum(w * y * x) / np.sum(w * x * x))
    return m / 2.0, msd, sel


def process(name):
    label, folder = MEASUREMENTS[name]
    m = load_measurement(folder)
    t, profiles, npix, center = m["t"], m["profiles"], m["npix"], m["center"]
    sig_um, err_um = m["sigma_um"], m["errsigma_um"]
    d = np.load(m["analysis_path"], allow_pickle=True).flat[0]
    time_fit = float(d.get("time_fit", 2.0))

    # geometry + per-frame CNR (windowed), same as the diagnostic pipeline
    rough = float(np.sqrt(fit_variance(np.arange(npix, dtype=float),
                                       profiles[0]).variance))
    half = max(6, int(round(FIT_HALF_SIGMA * rough)))
    lo, hi = max(0, center - half), min(npix, center + half)
    win = profiles[:, lo:hi]
    cnr, _, shot, amp, gain, read = per_frame_cnr(win)
    noise_law = decide_noise_law(gain, read, amp)
    trunc = build_or_load_calibration(rough, hi - lo, noise_law)

    # correction in his um^2 units: var_bias(CNR) is in sigma0^2 units; multiply
    # by his sigma^2(t0) (his t0 width squared) to land in um^2.
    sig0_um2 = float(sig_um[0] ** 2)
    var_his = sig_um ** 2
    var_corr = var_his - trunc.var_bias(cnr) * sig0_um2
    errsq = np.abs(2.0 * sig_um * err_um)               # his MSD-point error
    t0 = float(t[0])

    D_his, msd_his, sel = _his_D(t, var_his, errsq, t0, time_fit)
    D_corr, msd_corr, _ = _his_D(t, var_corr, errsq, t0, time_fit)
    ratio = D_corr / D_his if D_his else float("nan")
    cnr_fit = cnr[sel]
    _plot(label, t, msd_his, msd_corr, errsq, sel, t0, D_his, D_corr, time_fit)
    return dict(label=label, time_fit=time_fit, nfit=int(sel.sum()),
                cnr_lo=float(np.nanmin(cnr_fit)), cnr_hi=float(np.nanmax(cnr_fit)),
                D_his=D_his, D_corr=D_corr, ratio=ratio)


def _plot(label, t, msd_his, msd_corr, errsq, sel, t0, D_his, D_corr, time_fit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = 9
    # large figure, small markers, so the fit-range data is easy to read
    fig, ax = plt.subplots(layout="constrained", figsize=(20 * INCH, 14 * INCH))
    ax.tick_params(axis="both", which="both", direction="in", top=True,
                   right=True, labelsize=fs, length=4, width=1)
    fig.patch.set_facecolor("w"); fig.patch.set_alpha(1)
    ts, mh, mc, eh = t[sel], msd_his[sel], msd_corr[sel], errsq[sel]
    # data points only; the D values are reported as numbers (no fit lines, which
    # would visually emphasise how small the correction is)
    ax.errorbar(ts, mh, yerr=eh, fmt="o", mfc="none", mec=RED, ecolor=RED,
                ms=3, capsize=1.5, lw=0.7,
                label=f"his $\\sigma^2$:  D = {D_his:.4f} $\\mu$m$^2$/ns")
    ax.errorbar(ts, mc, yerr=eh, fmt="o", color=BLUE, ecolor=BLUE,
                ms=2.5, capsize=1.5, lw=0.7,
                label=f"corrected:  D = {D_corr:.4f} $\\mu$m$^2$/ns  "
                      f"(ratio {D_corr/D_his:.3f})")
    yl = np.concatenate([mh - eh, mh + eh])
    pad = 0.08 * (yl.max() - yl.min() + 1e-9)
    ax.set_ylim(yl.min() - pad, yl.max() + pad)
    ax.set_xlim(ts.min() - 0.1, ts.max() + 0.1)
    ax.set_xlabel("time after pump (ns)", fontsize=fs)
    ax.set_ylabel(r"MSD $\sigma^2(t)-\sigma^2(0)$ ($\mu$m$^2$)", fontsize=fs)
    ax.legend(frameon=False, fontsize=fs)
    out = os.path.join(WIP, label, "msd_enrique_pipeline.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    return out


def main():
    rows = [process(n) for n in LABELS]
    print(f"\n  Enrique's pipeline (D over t <= time_fit, forced origin, "
          f"inv-var) vs our correction; D in um^2/ns (his units):\n")
    print(f"  {'D_his':>9}{'D_corr':>9}{'ratio':>7}{'nfit':>5}{'cnr_lo':>7}"
          f"{'cnr_hi':>7}  measurement")
    for r in rows:
        print(f"  {r['D_his']:>9.4f}{r['D_corr']:>9.4f}{r['ratio']:>7.3f}"
              f"{r['nfit']:>5d}{r['cnr_lo']:>7.1f}{r['cnr_hi']:>7.1f}  {r['label']}")


if __name__ == "__main__":
    main()
