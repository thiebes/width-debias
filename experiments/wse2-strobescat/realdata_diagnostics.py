"""Per-measurement noise-bias diagnostics for Enrique's diffusion scans.

A review tool that processes one diffusion time-series end to end and writes
a complete diagnostic set (manifest, naive/corrected MSD plots, an all-frame
overlay montage, and per-frame overlays for the noisiest frames).

What it does for one measurement
--------------------------------
1. Reads Enrique's processed ``analysis_[...]_.npy`` (his per-frame reduced 1-D
   ``profiles`` and his fitted widths ``sigmas(um)`` / ``err_sigmas``). His
   profiles are used (line-cut band through the spot, physical time), re-oriented
   by correlation with a bright-frame reference (``_orient_to_reference``). The
   widths are this repo's own bounded Gaussian fit of each profile in a fixed
   window, so the data estimator matches the calibration's; his widths enter
   only through their relative fit error.
2. Measures per-frame CNR and the noise law with ``fft-cnr``
   (``estimate_noise_model=True``): the white/spectral CNR with its CI, the
   shot-corrected CNR, and the fitted ``gain`` / ``read`` decomposition. The
   noise law (white vs shot) selects the matched calibration.
3. Builds (and caches) a D-free ``var_bias(CNR)`` calibration at the measurement
   geometry and the matched noise law, then subtracts ``var_bias(CNR) * sigma0^2``
   per frame (the noise-bias correction, ``docs/DERIVATION.md``).

Units (important)
-----------------
The pixel is ``UM_PER_PX`` = 0.060 um at the sample for every scan. Do not derive the scale from his files: the ``X(um)`` array steps by
0.0036 = 0.060^2 (the pixel size applied twice when the file was written), and
the ratio of his fitted ``sigmas(um)`` to our px fit is not a calibration (its
per-scan median ranges from 0.065 to 0.089 um/px and depends on the fit model
and window). The diagnostics here report **px^2**; multiply by ``UM_PER_PX**2``
for um^2. The naive per-point error uses his scale-free relative error
``err_sigma/sigma``.

Caveats surfaced for review (not silently handled)
--------------------------------------------------
* The even/odd ON/OFF parity oscillation is present in his widths (the
  interleaved fine-delay batch reads ~2x wider); all frames are kept and the
  oscillation is shown, never smoothed.
* The corrected per-point SE currently carries only the naive fit SE; the
  calibration's MC error on ``var_bias`` is not propagated, and its size has
  not been measured for these calibrations.
* ``var_bias(CNR)`` is scaled by the frame-0 width (``sigma0^2``) for every
  frame, inside a window fixed at frame 0, although sigma^2 grows several-fold
  over the retained frames on some scans; the D-free calibration is valid only
  while sigma^2 stays near sigma0^2.
* The CNR key is the ``fft-cnr`` estimate; its confidence interval is recorded
  in the manifest but not propagated.

Usage::

    .venv/Scripts/python.exe experiments/wse2-strobescat/realdata_diagnostics.py mapbbr_250730
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import fft_cnr  # noqa: E402
from width_debias import GaussianModel, calibrate_d_free  # noqa: E402
from width_debias.fit import fit_D, fit_variance  # noqa: E402
from width_debias.bias_lookup import BiasLookup  # noqa: E402

# outputs live with the experiment (this folder's diagnostics/ subtree); not
# tracked, because they are derived from unpublished scans
WIP = os.path.join(os.path.dirname(__file__), "diagnostics")
CALIB_DIR = os.path.join(WIP, "_calib")

# --- measurement registry: name -> (label, folder) ---------------------------
# The canonical (most-merged) analysis_*_.npy in the folder is used; partial
# scan-index files are Enrique's exploration, not separate measurements.
MEASUREMENTS = {
    "wse2_250224_0p020": (
        "WSe2_Power_var_250224_0p020",
        r"P:\Lab\Enrique\lab\MultiChannelData\WSe2\Power_var\250224\0.020",
    ),
    "mixed_tmdc_comparison_all_0": (
        "mixed_tmdc_comparison_all_0",
        r"P:\Lab\Enrique\lab\MultiChannelData\mixed_tmdc\comparison_all\0",
    ),
    "mixed_tmdc_26_32_250716": (
        "mixed_tmdc_26_32_250716",
        r"P:\Lab\Enrique\lab\MultiChannelData\mixed_tmdc\26_32\250716",
    ),
    "mixed_tmdc_pureS_250701": (
        "mixed_tmdc_pureS_250701",
        r"P:\Lab\Enrique\lab\MultiChannelData\mixed_tmdc\pureS\250701",
    ),
    "mixed_tmdc_50_50_250702": (
        "mixed_tmdc_50_50_250702",
        r"P:\Lab\Enrique\lab\MultiChannelData\mixed_tmdc\50_50\250702",
    ),
    # clean WSe2 diffusers surfaced by the de-parity triage (both sweeps agree,
    # high tstat, many low-CNR frames)
    "wse2_betterpower_5MHz_0p022": (
        "WSe2_Better_power_5MHz_0p022",
        r"P:\Lab\Enrique\lab\MultiChannelData\WSe2\Better_power\5MHz\0.022",
    ),
    "wse2_250606_0p025": (
        "WSe2_Power_var_250606_0p025",
        r"P:\Lab\Enrique\lab\MultiChannelData\WSe2\Power_var\250606\0.025",
    ),
    "wse2_250722_0p043": (
        "WSe2_Power_var_250722_0p043",
        r"P:\Lab\Enrique\lab\MultiChannelData\WSe2\Power_var\250722\0.043",
    ),
    "mapbbr_250730": (
        "MAPbBr3_strobe_MA_250730",
        r"P:\Lab\Enrique\lab\TPLM _TSM\MA\250730",
    ),
}

UM_PER_PX = 0.060      # pixel size at the sample (um), every scan.
#                        Not X(um), not sigma(um)/sigma(px).
FLOOR = 1.3            # per-frame CNR floor: below it var_bias ~ A/CNR^2
#                        over-subtracts (corrected variance can go negative).
FIT_HALF_SIGMA = 4.0   # fit window half-width in units of sigma0. A few sigma of
#                        baseline keeps the Gaussian fit stable; a much wider
#                        window lets it wander onto noise at low CNR (and would
#                        inflate the calibrated var_bias unphysically).
CAL_CNR0_GRID = np.logspace(np.log10(1.0), np.log10(150.0), 24)
CAL_N_RUNS = 500
CAL_N_FRAMES = 12
CAL_T_MAX = 3.0

# plot_style spanish_flag: red = salient (the biased naive), blue = reference
# (the corrected value), fern green for the third series.
RED = "#c60b1e"
BLUE = "#0077c8"
GREEN = "#4a7729"
GRAY = "0.6"
INCH = 1 / 2.54


# ---------------------------------------------------------------------------
# Data loading and reduction (his profiles / his fits)
# ---------------------------------------------------------------------------
def _find_canonical(folder: str) -> str:
    """The most-merged analysis_*_.npy in a folder (excluding the _dataset maps)."""
    import glob
    fs = [f for f in glob.glob(os.path.join(folder, "analysis_*_.npy"))
          if not f.endswith("_dataset.npy")]
    if not fs:
        raise FileNotFoundError(f"no analysis_*_.npy in {folder}")
    return max(fs, key=lambda f: os.path.basename(f).count(",") + 1)


def _orient_to_reference(profiles: np.ndarray) -> np.ndarray:
    """Baseline-subtract every frame and orient it by *correlation* to the
    bright-frame reference bump.

    The stroboSCAT differential's sign flips frame to frame (the pump-on/off
    parity), so the spot is a positive feature in some frames and negative in
    others. A per-frame min/max test mis-orients the noisy frames (it latches
    onto whichever tail is larger), which scrambles the width series. Instead,
    build a positive reference bump from the brightest (early) frames and flip
    each frame to correlate positively with it -- a consistent sign rule.
    """
    ref = profiles[:4].mean(0)
    ref = ref - np.median(ref)
    if abs(ref.min()) > abs(ref.max()):
        ref = -ref                      # reference is a positive bump
    refc = ref - ref.mean()
    out = []
    for p in profiles:
        pp = p - np.median(p)
        s = np.sign(np.dot(pp - pp.mean(), refc)) or 1.0
        out.append(s * pp)
    return np.array(out)


def load_measurement(folder: str):
    """Load Enrique's per-frame profiles and fitted widths, sorted by time.

    Returns a dict with: ``t`` (ns), ``profiles`` (oriented, n_frames x npix),
    ``sigma_um`` / ``errsigma_um`` (his fitted width and its fit error, kept for
    reference only; the px->um scale is ``UM_PER_PX``), ``npix``, ``center``
    (common spot column), ``analysis_path``.
    """
    analysis_path = _find_canonical(folder)
    d = np.load(analysis_path, allow_pickle=True).flat[0]
    t = np.asarray(d["time_arr(ns)"], float)
    prof = np.asarray(d["profiles"], float)
    sig_um = np.asarray(d["sigmas(um)"], float)
    err_um = np.asarray(d["err_sigmas"], float)
    # sweep (batch) labels from the acquisition-order time resets: each measurement
    # is a merge of interleaved delay sweeps, each with its own sigma^2(0). Detect
    # before sorting, then carry the label through the time-sort.
    batch = np.zeros(t.size, int)
    b = 0
    for i in range(1, t.size):
        if t[i] < t[i - 1]:
            b += 1
        batch[i] = b
    order = np.argsort(t)
    t, prof, sig_um, err_um, batch = (t[order], prof[order], sig_um[order],
                                      err_um[order], batch[order])
    prof = _orient_to_reference(prof)
    early = prof[:4].mean(0)
    center = int(np.argmax(gaussian_filter1d(early, 2)))
    return {
        "t": t, "profiles": prof, "sigma_um": sig_um, "errsigma_um": err_um,
        "npix": prof.shape[1], "center": center, "batch": batch,
        "analysis_path": analysis_path,
    }


def per_frame_cnr(profiles: np.ndarray):
    """Per-frame white/shot CNR + noise decomposition via fft-cnr.

    Returns arrays: white CNR, white CNR CI (half-width), shot CNR, amplitude,
    gain, read.
    """
    n = len(profiles)
    cnr = np.full(n, np.nan)
    cnr_ci = np.full(n, np.nan)
    shot = np.full(n, np.nan)
    amp = np.full(n, np.nan)
    gain = np.full(n, np.nan)
    read = np.full(n, np.nan)
    for i, p in enumerate(profiles):
        est = fft_cnr.fft_cnr(p, fit_model="generalized_gaussian",
                              estimate_noise_model=True)
        cnr[i] = float(est.cnr)
        ci = getattr(est, "cnr_ci95", None)
        if ci is not None:
            cnr_ci[i] = 0.5 * (float(ci[1]) - float(ci[0]))
        amp[i] = float(est.amplitude)
        nm = est.noise_model
        if nm is not None:
            gain[i] = float(getattr(nm, "gain", np.nan))
            read[i] = float(getattr(nm, "read", np.nan))
            shot[i] = float(nm.peak_snr(est.amplitude))
    return cnr, cnr_ci, shot, amp, gain, read


def decide_noise_law(gain: np.ndarray, read: np.ndarray, amp: np.ndarray) -> str:
    """Classify the noise law from fft-cnr's ``var = gain*signal + read^2`` fit.

    stroboSCAT differential frames are read/floor dominated (white): the fitted
    ``gain`` sits at the noise level (it flips sign frame to frame) and the shot
    variance at the peak (``gain*amp``) stays below the read variance
    (``read^2``). Call it ``shot`` only when the gain is robustly positive
    across frames *and* the peak shot variance actually exceeds the read
    variance -- otherwise the gain is consistent with zero and the data is
    white."""
    g = float(np.nanmedian(gain))
    frac_pos = float(np.nanmean(gain > 0))
    shot_var = g * float(np.nanmedian(amp))
    read_var = float(np.nanmedian(read)) ** 2
    if g > 0 and frac_pos > 0.7 and shot_var > read_var:
        return "shot"
    return "white"


# ---------------------------------------------------------------------------
# Calibration (cached, geometry- and noise-law-matched)
# ---------------------------------------------------------------------------
def build_or_load_calibration(sigma0_px: float, npix: int, noise_law: str):
    os.makedirs(CALIB_DIR, exist_ok=True)
    tag = f"{noise_law}_npix{npix}_sig0{sigma0_px:.2f}"
    cache = os.path.join(CALIB_DIR, f"calib_{tag}.npz")
    if os.path.exists(cache) and "framevar_grid" in np.load(cache).files:
        z = np.load(cache)
        print(f"  loaded cached calibration {cache}", flush=True)
        return BiasLookup(cnr_grid=z["cnr_grid"], p_grid=z["p_grid"],
                               relvar_grid=z["relvar_grid"],
                               varbias_grid=z["varbias_grid"],
                               framevar_grid=z["framevar_grid"])
    half_window = (npix / 2.0) / sigma0_px
    noise_model = None  # white. (Shot would pass width_debias.noise.ShotNoise.)
    if noise_law == "shot":
        from width_debias.noise import ShotNoise
        noise_model = ShotNoise(read=1.0, poisson=True)
    print(f"  building calibration ({noise_law}, npix={npix}, "
          f"half_window={half_window:.1f}) ...", flush=True)
    trunc = calibrate_d_free(
        CAL_CNR0_GRID, sigma0=1.0, half_window=half_window, n_pixels=npix,
        n_frames=CAL_N_FRAMES, t_max=CAL_T_MAX, n_runs=CAL_N_RUNS, seed=1,
        model=GaussianModel(), noise_model=noise_model)
    np.savez(cache, cnr_grid=trunc.cnr_grid, p_grid=trunc.p_grid,
             relvar_grid=trunc.relvar_grid, varbias_grid=trunc.varbias_grid,
             framevar_grid=trunc.framevar_grid,
             sigma0_px=sigma0_px, npix=npix)
    print(f"  built + cached -> {cache}", flush=True)
    return trunc


# ---------------------------------------------------------------------------
# Per-frame overlay drawing helpers
# ---------------------------------------------------------------------------
def _gauss(x, center, var, amp):
    return amp * np.exp(-0.5 * (x - center) ** 2 / var)


def _overlay(ax, prof, center, var_naive, var_corr, halfwin, fs):
    """Draw a profile with the naive and corrected Gaussians overlaid."""
    npix = prof.size
    lo, hi = max(0, center - halfwin), min(npix, center + halfwin)
    x = np.arange(lo, hi)
    amp = float(prof[center])
    ax.plot(x, prof[lo:hi], ".", color=GRAY, ms=3)
    xx = np.linspace(lo, hi, 200)
    ax.plot(xx, _gauss(xx, center, var_naive, amp), "-", color=RED, lw=1.2)
    if var_corr > 0:
        ax.plot(xx, _gauss(xx, center, var_corr, amp), "-", color=BLUE, lw=1.2)


def truncate_degraded_tail(t, msd, se, keep, k=3.0, min_keep=4,
                           early_frac=0.4, tail_frac=0.5):
    """Truncate the time series where the late tail *sustainedly* departs the
    MSD trend, discarding that frame and every later one.

    The late-time signal degrades monotonically (the spot fades into noise), so
    the goal is to cut the degraded tail -- not to react to an isolated early
    parity blip. The rule:

    1. Fit the inverse-variance MSD line over the CNR-kept frames (the early,
       high-weight frames dominate, so the line tracks the clean trend).
    2. Set the residual scale from the *early* clean frames only (a robust
       1.4826*MAD over the first ``early_frac`` of the time span), so the scale
       reflects the trend's natural scatter, not the noisy tail. Mark every
       frame with ``|residual| > k*scale`` an outlier (symmetric: high or low).
    3. Truncate at the earliest frame that is itself an outlier *and* whose
       suffix (it and all later kept frames) is at least ``tail_frac`` outliers
       -- i.e. the point from which the series stays off-trend. An isolated
       outlier surrounded by on-trend frames does not trigger a cut.

    Returns a boolean ``retained`` mask (a subset of ``keep``) and the cut time
    (``np.nan`` if nothing was truncated).
    """
    idx = np.where(keep)[0]
    if idx.size < min_keep:
        return keep.copy(), float("nan")
    w = 1.0 / np.maximum(se[idx], 1e-9) ** 2
    d = fit_D(t[idx], msd[idx], w)
    resid = msd[idx] - (d.intercept + 2.0 * d.D * t[idx])
    # robust scale from the early clean frames (at least min_keep of them)
    t0, t1 = t[idx[0]], t[idx[-1]]
    early = t[idx] <= t0 + early_frac * (t1 - t0)
    if int(early.sum()) < min_keep:
        early = np.zeros(idx.size, bool)
        early[:min_keep] = True
    r_e = resid[early]
    mad = float(np.median(np.abs(r_e - np.median(r_e))))
    scale = 1.4826 * mad if mad > 0 else (float(np.std(r_e)) or float(np.std(resid)) or 1.0)
    outlier = np.abs(resid) > k * scale
    # earliest frame that starts a majority-outlier tail
    cut_pos = None
    for i in range(min_keep, idx.size):
        if outlier[i] and float(np.mean(outlier[i:])) >= tail_frac:
            cut_pos = i
            break
    retained = keep.copy()
    cut_time = float("nan")
    if cut_pos is not None:
        retained = np.zeros_like(keep)
        retained[idx[:cut_pos]] = True
        cut_time = float(t[idx[cut_pos]])
    return retained, cut_time


# ---------------------------------------------------------------------------
# Main per-measurement processing
# ---------------------------------------------------------------------------
def process(name: str):
    t_start = time.time()
    label, folder = MEASUREMENTS[name]
    outdir = os.path.join(WIP, label)
    framesdir = os.path.join(outdir, "frames")
    os.makedirs(framesdir, exist_ok=True)

    m = load_measurement(folder)
    t, profiles, npix, center = m["t"], m["profiles"], m["npix"], m["center"]
    sig_um, err_um = m["sigma_um"], m["errsigma_um"]
    n = t.size
    print(f"[{label}] {os.path.basename(m['analysis_path'])}", flush=True)
    print(f"  {n} frames, t={t.min():.2f}..{t.max():.2f} ns, npix={npix}, "
          f"spot column {center}", flush=True)

    # Spot size from the t=0 (earliest, brightest) frame, then a centred
    # +/- FIT_HALF_SIGMA * sigma0 window. Both the width fit and the per-frame
    # CNR are measured on this window. Restricting to the spot is essential:
    # the full 100-px line-cut carries low-frequency background/fringe structure
    # away from the spot that corrupts fft-cnr's spectral CNR at low signal
    # (it then fits a noise excursion, not the peak -- a real peak can read
    # CNR < 1 and a flat frame CNR > 3). It also keeps the bounded width fit
    # stable. The naive width is this px fit -- the standard bounded Gaussian
    # estimator the correction acts on -- so the calibration estimator and the
    # data estimator match. (His sigmas(um) are kept in the manifest for
    # reference; widths are reported in px^2, um^2 = px^2 * UM_PER_PX**2.)
    rough_sigma0 = float(np.sqrt(fit_variance(np.arange(npix, dtype=float),
                                              profiles[0]).variance))
    half = max(6, int(round(FIT_HALF_SIGMA * rough_sigma0)))
    lo, hi = max(0, center - half), min(npix, center + half)
    win = profiles[:, lo:hi]
    win_x = np.arange(hi - lo, dtype=float)
    npix_win = hi - lo

    # per-frame CNR + noise law, measured on the spot window
    cnr, cnr_ci, shot, amp, gain, read = per_frame_cnr(win)
    noise_law = decide_noise_law(gain, read, amp)
    print(f"  fit/CNR window +/-{half} px ({npix_win} px); "
          f"CNR(white) {np.nanmin(cnr):.1f}..{np.nanmax(cnr):.1f}; "
          f"noise law: {noise_law} "
          f"(median gain {np.nanmedian(gain):.1e}, read {np.nanmedian(read):.1e})",
          flush=True)

    var_naive = np.array([fit_variance(win_x, p).variance for p in win])
    sigma0_px2 = float(var_naive[0])
    sigma0_px = float(np.sqrt(sigma0_px2))
    # naive SE on sigma^2 from Enrique's scale-free relative fit error
    # (d(sig^2)/sig^2 = 2 dsig/sig); his absolute um scale cancels in the ratio.
    se_naive = var_naive * 2.0 * (err_um / sig_um)
    print(f"  sigma0 = {sigma0_px:.2f} px (sigma0^2 = {sigma0_px2:.2f} px^2)",
          flush=True)

    trunc = build_or_load_calibration(sigma0_px, npix_win, noise_law)
    vb = trunc.var_bias(cnr) * sigma0_px2            # px^2
    var_corr = var_naive - vb
    se_corr = se_naive.copy()                        # var_bias MC error deferred
    keep = cnr >= FLOOR

    # MSD = sigma^2(t) - sigma^2(t0); D = slope/2. Naive is well-defined over all
    # frames; the *corrected* fit is only valid above the floor (below it
    # var_bias ~ A/CNR^2 over-subtracts). On top of the floor, the late tail is
    # truncated where the corrected MSD sustainedly departs the trend (a
    # degraded noisy tail, not an isolated early blip); the headline fit is over
    # the retained frames.
    #
    # De-parity: each measurement is a merge of interleaved delay sweeps, each
    # with its own sigma^2(0). A single global baseline injects a spurious slope
    # from the inter-sweep offset, so the MSD is baselined per sweep (each sweep
    # re-zeroed to its own earliest frame). A single global offset is
    # slope-invariant; only this per-sweep re-zeroing legitimately affects D.
    batch = m["batch"]
    base_n = np.zeros_like(var_naive)
    base_c = np.zeros_like(var_corr)
    for bb in set(batch.tolist()):
        sel = batch == bb
        base_n[sel] = var_naive[sel][0]
        base_c[sel] = var_corr[sel][0]
    msd_naive = var_naive - base_n
    msd_corr = var_corr - base_c
    retained, cut_time = truncate_degraded_tail(t, msd_corr, se_corr, keep)
    w_ols = np.ones(n)
    w_iv = 1.0 / np.maximum(se_naive, 1e-9) ** 2
    w_iv_k = w_iv * keep
    w_iv_r = w_iv * retained
    w_ols_r = retained.astype(float)
    d_n_ols = fit_D(t, msd_naive, w_ols)          # naive, all frames
    d_n_iv = fit_D(t, msd_naive, w_iv)
    d_n_iv_k = fit_D(t, msd_naive, w_iv_k)        # naive, kept (CNR floor only)
    d_c_iv_k = fit_D(t, msd_corr, w_iv_k)         # corrected, kept (CNR floor only)
    d_n_ols_r = fit_D(t, msd_naive, w_ols_r)      # naive, retained (headline)
    d_n_iv_r = fit_D(t, msd_naive, w_iv_r)
    d_c_ols_r = fit_D(t, msd_corr, w_ols_r)       # corrected, retained (headline)
    d_c_iv_r = fit_D(t, msd_corr, w_iv_r)

    _write_manifest(outdir, t, cnr, cnr_ci, shot, gain, read, noise_law,
                    var_naive, se_naive, vb, var_corr, se_corr, keep, retained)
    _plot_msd(os.path.join(outdir, "msd_naive.png"), t, msd_naive, se_naive,
              keep, retained, d_n_iv_r, d_n_iv, "all frames", cut_time, "naive")
    _plot_msd(os.path.join(outdir, "msd_corrected.png"), t, msd_corr, se_corr,
              keep, retained, d_c_iv_r, d_c_iv_k, f"CNR>={FLOOR}, no outlier cut",
              cut_time, "corrected")
    _montage(os.path.join(outdir, "montage.png"), t, profiles, center,
             var_naive, var_corr, cnr, keep, retained)
    # per-frame overlays for the noisiest frames: the below-floor (flagged)
    # frames if any, else the three lowest-CNR frames (to exhibit the format)
    flagged = np.where(~keep)[0]
    examples = flagged if flagged.size else np.argsort(cnr)[:3]
    for i in examples:
        _frame_png(os.path.join(framesdir, f"frame_{i:02d}_t{t[i]:+.2f}.png"),
                   t[i], profiles[i], center, var_naive[i], var_corr[i],
                   cnr[i], cnr_ci[i], shot[i], se_naive[i], se_corr[i],
                   noise_law, bool(keep[i]))

    n_kept = int(keep.sum())
    n_ret = int(retained.sum())
    summary = {
        "name": name, "label": label, "noise_law": noise_law, "n_frames": n,
        "n_kept": n_kept, "n_retained": n_ret, "cut_time": cut_time,
        "cnr_lo": float(np.nanmin(cnr)), "cnr_hi": float(np.nanmax(cnr)),
        # headline: naive vs corrected on the retained frames (CNR floor + the
        # truncation at the first wildly-off-trend frame)
        "D_naive_ols_r": d_n_ols_r.D, "D_naive_iv_r": d_n_iv_r.D,
        "D_corr_ols_r": d_c_ols_r.D, "D_corr_iv_r": d_c_iv_r.D,
        # reference: kept (CNR floor only, no outlier cut) and all-frames
        "D_naive_iv_k": d_n_iv_k.D, "D_corr_iv_k": d_c_iv_k.D,
        "D_naive_iv": d_n_iv.D, "sigma0_px": sigma0_px,
    }
    _print_table(t, cnr, var_naive, vb, var_corr, keep, retained)
    cut_s = "none" if not np.isfinite(cut_time) else f"t >= {cut_time:.2f} ns"
    print(f"\n  floor CNR >= {FLOOR}: {n_kept}/{n} kept; outlier truncation "
          f"({cut_s}): {n_ret}/{n} retained")
    print(f"  D (px^2/ns), RETAINED frames:  naive OLS {d_n_ols_r.D:.3f} "
          f"inv-var {d_n_iv_r.D:.3f}   corrected OLS {d_c_ols_r.D:.3f} "
          f"inv-var {d_c_iv_r.D:.3f}")
    if d_n_iv_r.D:
        print(f"  ratio corrected/naive (inv-var, retained): "
              f"{d_c_iv_r.D / d_n_iv_r.D:.3f}")
    print(f"  reference (CNR-kept, no outlier cut): naive inv-var "
          f"{d_n_iv_k.D:.3f}  corrected inv-var {d_c_iv_k.D:.3f}")
    print(f"  outputs -> {outdir}")
    print(f"  [{(time.time() - t_start) / 60:.1f} min]")
    _update_index(summary)
    return summary


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _write_manifest(outdir, t, cnr, cnr_ci, shot, gain, read, noise_law,
                    var_naive, se_naive, vb, var_corr, se_corr, keep, retained):
    path = os.path.join(outdir, "manifest.csv")
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["time_ns", "cnr_white", "cnr_white_ci", "cnr_shot",
                     "gain", "read", "noise_law", "var_naive_px2",
                     "se_naive_px2", "var_bias_px2", "var_corr_px2",
                     "se_corr_px2", "keep_flag", "retained_flag"])
        for i in range(t.size):
            wr.writerow([f"{t[i]:.3f}", f"{cnr[i]:.4f}", f"{cnr_ci[i]:.4f}",
                         f"{shot[i]:.4f}", f"{gain[i]:.4e}", f"{read[i]:.4e}",
                         noise_law, f"{var_naive[i]:.4f}", f"{se_naive[i]:.4f}",
                         f"{vb[i]:.4f}", f"{var_corr[i]:.4f}",
                         f"{se_corr[i]:.4f}", int(keep[i]), int(retained[i])])
    print(f"  manifest -> {path}", flush=True)


def _new_ax(w_cm, h_cm, fs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(layout="constrained",
                           figsize=(w_cm * INCH, h_cm * INCH))
    ax.tick_params(axis="both", which="both", direction="in", top=True,
                   right=True, labelsize=fs, length=4, width=1)
    fig.patch.set_facecolor("w")
    fig.patch.set_alpha(1)
    return fig, ax


def _plot_msd(path, t, msd, se, keep, retained, d_head, d_ref, ref_label,
              cut_time, which):
    """MSD vs time with the headline (retained) slope fit.

    Three classes of point are drawn distinctly: retained (filled, used for the
    headline fit), kept-but-cut (open squares -- past the first-outlier
    truncation), and below-floor (gray, capped off-scale). A dashed vertical
    line marks the truncation time. The y-axis is scaled to the retained data so
    the diffusion signal stays readable. ``d_ref`` is a faint reference fit
    (all-frames for the naive panel; CNR-kept-without-the-outlier-cut for the
    corrected panel) so the effect of the truncation is visible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = 8
    fig, ax = _new_ax(9, 7, fs)
    r = retained
    cut = keep & ~retained               # kept but removed by outlier truncation
    below = ~keep                        # below the CNR floor
    base = r if r.any() else keep
    # y-range from the retained point *values* (not their error bars, so a single
    # large error bar does not flatten the trend); bars may clip at the edge
    lo = float(np.nanmin(msd[base]))
    hi = float(np.nanmax(msd[base]))
    pad = 0.30 * (hi - lo + 1e-9)
    ylo, yhi = lo - pad, hi + pad
    tt = np.linspace(t.min(), t.max(), 50)
    ax.plot(tt, d_ref.intercept + 2 * d_ref.D * tt, "--", color=GREEN, lw=1.3,
            alpha=0.6, label=f"{ref_label}: D={d_ref.D:.2f}")
    ax.plot(tt, d_head.intercept + 2 * d_head.D * tt, "-", color=RED, lw=2.0,
            label=f"retained inv-var: D={d_head.D:.2f}")
    ax.errorbar(t[r], msd[r], yerr=se[r], fmt="o", color=RED, ms=4,
                capsize=2, lw=1, label=f"{which} MSD (retained)")
    if cut.any():
        mc = np.clip(msd[cut], ylo + 0.02 * (yhi - ylo), yhi)
        ax.plot(t[cut], mc, "s", mfc="none", mec="#d55e00", ms=5,
                label="cut (degraded tail)")
    if below.any():
        mb = np.clip(msd[below], ylo + 0.02 * (yhi - ylo), yhi)
        off = msd[below] < ylo
        ax.plot(t[below][~off], mb[~off], "o", mfc="none", mec=GRAY, ms=4,
                label=f"CNR < {FLOOR} (flagged)")
        if off.any():
            ax.plot(t[below][off], mb[off], "v", color=GRAY, ms=6)
    if np.isfinite(cut_time):
        ax.axvline(cut_time, ls=":", color="0.4", lw=1)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("time after pump (ns)", fontsize=fs)
    ax.set_ylabel(r"MSD $\sigma^2(t)-\sigma^2(0)$ (px$^2$)", fontsize=fs)
    ax.legend(frameon=False, fontsize=fs - 1)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  {which} MSD -> {path}", flush=True)


def _montage(path, t, profiles, center, var_naive, var_corr, cnr, keep, retained):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = t.size
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    halfwin = max(12, int(3 * np.sqrt(np.nanmax(var_naive))))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.2 * INCH * 2,
                                                  nrow * 2.6 * INCH * 2),
                             layout="constrained")
    fs = 6
    for i in range(nrow * ncol):
        ax = axes.flat[i]
        if i >= n:
            ax.axis("off")
            continue
        _overlay(ax, profiles[i], center, var_naive[i], var_corr[i], halfwin, fs)
        ax.tick_params(labelsize=fs - 1, direction="in")
        # tag: "*" below CNR floor, "x" cut after the first outlier
        tag = "" if retained[i] else (" x" if keep[i] else " *")
        ax.set_title(f"t={t[i]:+.2f} ns  CNR={cnr[i]:.1f}{tag}", fontsize=fs)
    fig.patch.set_facecolor("w")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  montage -> {path}", flush=True)


def _frame_png(path, t_i, prof, center, var_naive, var_corr, cnr_i, cnr_ci,
               shot_i, se_n, se_c, noise_law, keep_i):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    halfwin = max(15, int(3.5 * np.sqrt(max(var_naive, var_corr, 1.0))))
    fig, axes = plt.subplots(1, 3, figsize=(24 * INCH, 7 * INCH),
                             layout="constrained")
    fs = 8
    titles = ["naive", "corrected", "both"]
    for ax, ttl in zip(axes, titles):
        npix = prof.size
        lo, hi = max(0, center - halfwin), min(npix, center + halfwin)
        x = np.arange(lo, hi)
        amp = float(prof[center])
        ax.plot(x, prof[lo:hi], ".", color=GRAY, ms=4)
        xx = np.linspace(lo, hi, 200)
        if ttl in ("naive", "both"):
            ax.plot(xx, _gauss(xx, center, var_naive, amp), "-", color=RED,
                    lw=1.5, label="naive")
        if ttl in ("corrected", "both") and var_corr > 0:
            ax.plot(xx, _gauss(xx, center, var_corr, amp), "-", color=BLUE,
                    lw=1.5, label="corrected")
        ax.tick_params(direction="in", top=True, right=True, labelsize=fs)
        ax.set_xlabel("position (px)", fontsize=fs)
        ax.legend(frameon=False, fontsize=fs)
    axes[0].set_ylabel("differential signal (a.u.)", fontsize=fs)
    fig.suptitle(
        f"t={t_i:+.2f} ns   CNR(white)={cnr_i:.1f}+/-{cnr_ci:.1f}  "
        f"CNR(shot)={shot_i:.1f}  noise={noise_law}\n"
        rf"$\sigma^2_{{naive}}$={var_naive:.2f}+/-{se_n:.2f} px$^2$   "
        rf"$\sigma^2_{{corr}}$={var_corr:.2f}+/-{se_c:.2f} px$^2$   "
        f"{'KEPT' if keep_i else 'BELOW FLOOR'}", fontsize=fs)
    fig.patch.set_facecolor("w")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _print_table(t, cnr, var_naive, vb, var_corr, keep, retained):
    print(f"\n  {'t_ns':>7}{'CNR':>7}{'var_naive':>11}{'var_bias':>10}"
          f"{'var_corr':>10}{'flag':>6}")
    for i in range(t.size):
        flag = "" if retained[i] else ("  x" if keep[i] else "  *")
        print(f"  {t[i]:>7.2f}{cnr[i]:>7.1f}{var_naive[i]:>11.2f}"
              f"{vb[i]:>10.2f}{var_corr[i]:>10.2f}{flag:>6}")


def _update_index(s):
    path = os.path.join(WIP, "index.md")
    header = ("# Real-data noise-bias diagnostics\n\n"
              "Per-measurement diagnostics of the noise-bias correction on "
              "Enrique's diffusion scans. Widths in px^2 (1 px = 0.060 um at the "
              "sample, so 1 px^2 = 0.0036 um^2). D in px^2/ns. The naive and "
              "corrected D are the "
              "inverse-variance fits over the **retained** frames: the MSD is "
              "baselined per sweep (de-parity: each interleaved delay sweep "
              "re-zeroed to its own t0), then frames below the per-frame CNR floor "
              "are dropped, then the late tail is truncated where the corrected MSD "
              "sustainedly departs the trend (k=3 robust, majority-outlier tail; "
              "early scale, so isolated early blips do not trigger it). "
              "Frames (kept/retained) reports both cuts.\n\n"
              "| Measurement | Noise law | Frames (kept/ret) | CNR range | cut at | "
              "D naive (ret) | D corrected (ret) | ratio | Folder |\n"
              "| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |\n")
    ratio = (s["D_corr_iv_r"] / s["D_naive_iv_r"]) if s["D_naive_iv_r"] else float("nan")
    cut = "-" if not np.isfinite(s["cut_time"]) else f"{s['cut_time']:.1f} ns"
    row = (f"| {s['label']} | {s['noise_law']} | "
           f"{s['n_frames']}/{s['n_kept']}/{s['n_retained']} | "
           f"{s['cnr_lo']:.1f}-{s['cnr_hi']:.1f} | {cut} | "
           f"{s['D_naive_iv_r']:.3f} | {s['D_corr_iv_r']:.3f} | {ratio:.2f} | "
           f"[{s['label']}]({s['label']}/) |\n")
    existing = ""
    if os.path.exists(path):
        with open(path) as f:
            existing = f.read()
    if not existing.startswith("# Real-data"):
        existing = header
    # replace an existing row for this label, else append
    lines = existing.splitlines(keepends=True)
    lines = [ln for ln in lines if f"| {s['label']} |" not in ln]
    with open(path, "w") as f:
        f.writelines(lines)
        f.write(row)
    print(f"  index -> {path}", flush=True)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "mapbbr_250730"
    process(name)


if __name__ == "__main__":
    main()
