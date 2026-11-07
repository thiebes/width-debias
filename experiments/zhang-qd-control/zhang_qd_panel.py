"""Real-data panel: the noise-bias correction on Zhang et al. (2022) QD-solid
femtosecond exciton transport.

Source data are the published figure worksheets exported from the manuscript
Origin project (Zhang et al., "Ultrafast exciton transport at early times in
quantum dot solids", Nat. Mater. 21, 533-539, 2022), placed read-only in
this folder:

* ``Figure 1d <delay>.csv`` -- the raw 64x64 differential ON/OFF transient-
  absorption images of the OA (oleic-acid-capped PbS) film at five pump-probe
  delays (0, 60, 120, 260, 400 fs). These carry the *noise*, so the per-frame
  CNR is measured here with ``fft-cnr`` on a line-cut band through the spot.
* ``Figure 2a.csv`` -- the published MSD(t) = sigma^2(t) - sigma^2(0) for all
  five PbS films over the full window (61 delays to 4100 fs). The OA column is
  the authors' own fitted widths; these supply the naive MSD used below (their
  fit, our CNR).
* ``Figure 4a-c.csv`` -- MSD(t) of the CdSe / CdS / CsPbI3 "no transport"
  control films. Their MSD scatters around zero; its RMS is the measurement
  noise floor the fast-regime PbS signal must clear.

Why this dataset is a scope test rather than a correction target: the reported
fast-regime diffusivity (Dfast ~ 99-280 cm^2/s for the PbS films) is anomalously
large, and it is read from the noisy earliest frames. The noise bias would
inflate a fitted width at low CNR -- so the question is whether the anomalous
Dfast is partly a noise-bias artifact. Two facts answer it:

* The per-frame CNR is moderate (9-24) and the noise is white (the camera
  differential has a constant read floor, gain ~ 0), so ``var_bias(CNR)`` is
  below 0.01 of sigma0^2.
* The CNR *rises* with delay (the TA bleach builds up over the first few hundred
  fs faster than the spot dilutes), so the bias is largest at t=0 -- the
  subtracted reference. A bias that shrinks with time adds a *negative*-sloping
  term to MSD(t): it would suppress Dfast, not inflate it. Correcting nudges
  Dfast slightly up.

So the headline is the corrected/naive ratio (~1) and the sign: this is a real
measurement where the method correctly reports that no correction is needed, and
where dCNR/dt sets the sign of the bias-induced D error.

Widths/MSD are in the published nm^2; sigma0^2 = (172 nm)^2 from the t=0 width.
D is in cm^2/s (MSD in nm^2 over time in fs: D = slope/2, 1 nm^2/fs = 10 cm^2/s).

Set ``WIDTH_DEBIAS_WORKERS=1`` for the (cached) calibration build.

Usage::

    .venv/Scripts/python.exe scripts/zhang_qd_panel.py
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import fft_cnr  # noqa: E402
from width_debias import GaussianModel, calibrate_d_free  # noqa: E402
from width_debias.fit import fit_D, fit_variance  # noqa: E402
from width_debias.bias_lookup import BiasLookup  # noqa: E402

RE = os.path.dirname(__file__)
CALIB_CACHE = os.path.join("output", "calibration_zhang_qd.npz")

DELAYS = [0, 60, 120, 260, 400]          # fs; the Fig 1d image delays
SIGMA0_NM = 172.0                        # t=0 spot width (paper); sigma0^2 ref
BAND = 2                                 # half-height of summed central band (px)
FAST_MAX_FS = 260                        # fast-regime fit window upper edge (fs)
FLOOR = 1.3                              # per-frame CNR over-subtraction floor
CAL_CNR0_GRID = np.logspace(np.log10(1.0), np.log10(150.0), 24)
CAL_N_RUNS = 500
NM2_PER_FS_TO_CM2_S = 10.0               # 1 nm^2/fs = 10 cm^2/s


def _img(delay):
    p = os.path.join(RE, f"Figure 1d {delay}fs.csv")
    return np.loadtxt(p, delimiter=",", encoding="utf-8-sig")


def _msd_curve(fname, col=1):
    """Read a (time fs, MSD nm^2) column pair from an exported figure CSV."""
    rows = list(csv.reader(open(os.path.join(RE, fname), encoding="utf-8-sig")))
    t, m = [], []
    for r in rows[3:]:
        if len(r) > col and r[0].strip() and r[col].strip():
            t.append(float(r[0])); m.append(float(r[col]))
    return np.array(t), np.array(m)


def per_frame_cnr_and_width():
    """Per-frame CNR (fft-cnr) and bounded width (px^2) on the OA images.

    Returns the t=0 spot width in px (the calibration geometry) and the
    per-delay CNR; the widths themselves are taken from the published Fig 2a.
    """
    cnr, sig0_px = [], None
    for d in DELAYS:
        img = _img(d)
        j0, _ = np.unravel_index(np.argmax(gaussian_filter(img, 2.0)), img.shape)
        band = img[max(0, j0 - BAND):j0 + BAND + 1, :].sum(0)
        band = band - np.median(band)
        xs = np.arange(band.size, dtype=float)
        cnr.append(float(fft_cnr.fft_cnr(band, fit_model="generalized_gaussian").cnr))
        if d == 0:
            sig0_px = float(np.sqrt(fit_variance(xs, band).variance))
    return float(sig0_px), np.array(cnr), img.shape[1]


def build_or_load_calibration(sigma0_px, npix):
    if os.path.exists(CALIB_CACHE):
        z = np.load(CALIB_CACHE)
        if (int(z["npix"]) == npix and abs(float(z["sigma0_px"]) - sigma0_px) < 1.5
                and "framevar_grid" in z.files):
            print(f"  loaded cached calibration {CALIB_CACHE}", flush=True)
            return BiasLookup(cnr_grid=z["cnr_grid"], p_grid=z["p_grid"],
                                   relvar_grid=z["relvar_grid"],
                                   varbias_grid=z["varbias_grid"],
                                   framevar_grid=z["framevar_grid"])
    half_window = (npix / 2.0) / sigma0_px
    trunc = calibrate_d_free(
        CAL_CNR0_GRID, sigma0=1.0, half_window=half_window, n_pixels=npix,
        n_frames=12, t_max=3.0, n_runs=CAL_N_RUNS, seed=1,
        model=GaussianModel(), noise_model=None)
    os.makedirs("output", exist_ok=True)
    np.savez(CALIB_CACHE, cnr_grid=trunc.cnr_grid, p_grid=trunc.p_grid,
             relvar_grid=trunc.relvar_grid, varbias_grid=trunc.varbias_grid,
             framevar_grid=trunc.framevar_grid,
             sigma0_px=sigma0_px, npix=npix)
    print(f"  built + cached calibration -> {CALIB_CACHE}", flush=True)
    return trunc


def _Dfit(t, msd):
    """Fickian D (cm^2/s) from MSD(t): slope/2, MSD nm^2 vs t fs."""
    d = fit_D(t, msd, np.ones_like(t))
    return d.D * NM2_PER_FS_TO_CM2_S


def main():
    t0 = time.time()
    print("Measuring per-frame CNR on the OA Fig 1d images ...", flush=True)
    sig0_px, cnr, npix = per_frame_cnr_and_width()
    print(f"  {npix}px frames; t0 spot {sig0_px:.2f}px; "
          f"CNR {cnr.min():.1f}..{cnr.max():.1f} (rises with delay)", flush=True)

    print("Calibration (white) ...", flush=True)
    trunc = build_or_load_calibration(sig0_px, npix)
    print(f"  ready [{time.time()-t0:.0f}s]", flush=True)

    # Published OA MSD at the five image delays (their fit, our CNR).
    t_oa, m_oa = _msd_curve("Figure 2a.csv")
    msd5 = np.array([m_oa[np.argmin(np.abs(t_oa - d))] for d in DELAYS])
    delays = np.array(DELAYS, float)
    sigma0_sq = SIGMA0_NM ** 2

    # Additive variance bias per frame (nm^2) and corrected MSD. The correction
    # subtracts var_bias from each sigma^2; in the MSD it is the *difference*
    # var_bias(t) - var_bias(0) that moves the points.
    vb = trunc.var_bias(cnr) * sigma0_sq            # nm^2
    msd5_corr = msd5 - (vb - vb[0])
    keep = cnr >= FLOOR

    fast = delays <= FAST_MAX_FS
    D_naive = _Dfit(delays[fast], msd5[fast])
    D_corr = _Dfit(delays[fast], msd5_corr[fast])
    # naive Dfast on the dense published grid, for cross-check with the paper
    dense = (t_oa <= FAST_MAX_FS)
    D_naive_dense = _Dfit(t_oa[dense], m_oa[dense])

    # Control noise floor: MSD RMS of the no-transport films (nm^2).
    floors = {}
    for fn, nm in [("Figure 4a.csv", "CdSe"), ("Figure 4b.csv", "CdS"),
                   ("Figure 4c.csv", "CsPbI3")]:
        _, mc = _msd_curve(fn)
        floors[nm] = float(np.sqrt(np.mean(mc ** 2)))

    print("\n  per-frame (MSD in nm^2; sigma0^2 = %.0f nm^2):" % sigma0_sq)
    print(f"  {'t_fs':>6}{'CNR':>7}{'MSD_naive':>11}{'var_bias':>10}{'MSD_corr':>11}")
    for i, d in enumerate(DELAYS):
        print(f"  {d:>6}{cnr[i]:>7.1f}{msd5[i]:>11.0f}{vb[i]:>10.1f}{msd5_corr[i]:>11.0f}")

    print(f"\n  Dfast over t <= {FAST_MAX_FS} fs (cm^2/s):")
    print(f"    naive (5 image frames)   {D_naive:8.1f}")
    print(f"    naive (dense Fig 2a)     {D_naive_dense:8.1f}   (paper OA Dfast ~ 280)")
    print(f"    corrected                {D_corr:8.1f}")
    print(f"    ratio corrected/naive    {D_corr / D_naive:8.4f}")
    print(f"\n  control MSD noise floor (RMS, nm^2): "
          + ", ".join(f"{k} {v:.0f}" for k, v in floors.items()))
    print(f"  OA fast-regime signal at 60 fs: {msd5[1]:.0f} nm^2 "
          f"(~{msd5[1]/np.mean(list(floors.values())):.0f}x the floor)")

    _figure(t_oa, m_oa, delays, msd5, msd5_corr, cnr, keep, floors,
            D_naive, D_corr)
    print(f"\n[total {(time.time()-t0)/60:.1f} min]")


def _figure(t_oa, m_oa, delays, msd5, msd5_corr, cnr, keep, floors,
            D_naive, D_corr):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[matplotlib missing]")
        return
    SPAIN = {"red": "#D55E00", "blue": "#0072B2", "green": "#009E73",
             "grey": "0.6"}
    inch = 1 / 2.54; fs = 8
    fig, ax = plt.subplots(1, 3, layout="constrained",
                           figsize=(24 * inch, 7 * inch))
    for a in ax:
        a.tick_params(axis="both", which="both", direction="in", top=True,
                      right=True, labelsize=fs, length=4, width=1)
    fig.patch.set_facecolor("w"); fig.patch.set_alpha(1)

    # (a) fast-regime MSD: dense published + 5 corrected points + Dfast lines
    a = ax[0]
    fast = t_oa <= 500
    a.plot(t_oa[fast], m_oa[fast], "-", color=SPAIN["grey"], lw=1,
           label="published MSD (OA)")
    a.plot(delays, msd5, "o", color=SPAIN["red"], ms=5, label="naive (image frames)")
    a.plot(delays, msd5_corr, "s", color=SPAIN["blue"], ms=4, label="corrected")
    tt = np.linspace(0, 260, 30)
    a.plot(tt, 2 * (D_naive / 10.0) * tt, "-", color=SPAIN["red"], lw=1.2)
    a.plot(tt, 2 * (D_corr / 10.0) * tt, "--", color=SPAIN["blue"], lw=1.2)
    a.set_xlim(-10, 460)
    a.set_xlabel("delay (fs)", fontsize=fs)
    a.set_ylabel(r"MSD $=\sigma^2(t)-\sigma^2(0)$ (nm$^2$)", fontsize=fs)
    a.legend(frameon=False, fontsize=fs, loc="lower right")

    # (b) per-frame CNR rises with delay; floor band
    a = ax[1]
    a.plot(delays, cnr, "o-", color=SPAIN["green"], ms=5, lw=1)
    a.axhspan(0.5, FLOOR, color=SPAIN["red"], alpha=0.12)
    a.axhline(FLOOR, ls="--", c=SPAIN["red"], lw=1)
    a.set_xlabel("delay (fs)", fontsize=fs)
    a.set_ylabel("per-frame CNR", fontsize=fs)
    a.set_ylim(0, cnr.max() * 1.15)

    # (c) control noise floor vs OA signal
    a = ax[2]
    names = list(floors); vals = [floors[n] for n in names]
    a.bar(range(len(names)), vals, color=SPAIN["grey"], width=0.6)
    a.axhline(msd5[1], ls="-", c=SPAIN["red"], lw=1.5)
    a.text(len(names) - 0.5, msd5[1], " OA signal @60 fs",
           color=SPAIN["red"], fontsize=fs, va="bottom", ha="right")
    a.set_xticks(range(len(names))); a.set_xticklabels(names, fontsize=fs)
    a.set_ylabel(r"MSD noise floor / signal (nm$^2$)", fontsize=fs)
    a.set_yscale("log")

    os.makedirs("output", exist_ok=True)
    out = os.path.join("output", "zhang_qd_panel.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Figure written to {out}")


if __name__ == "__main__":
    main()
