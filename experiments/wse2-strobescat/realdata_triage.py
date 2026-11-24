"""Re-triage all analysis-bearing measurements with de-parity and per-sweep D.

Every measurement is a merge of interleaved delay sweeps (detected by time
resets in the acquisition order), each with its own sigma^2(0). The previous
triage baselined to a single global t0, which injects a spurious slope. This
triage:

* refits each frame's width on a +/- 4 sigma0 window (consistent estimator),
  measures the per-frame CNR on that window (fft-cnr),
* applies an analytic white var_bias proxy (0.4/CNR^2 * sigma0^2) for the
  correction (no Monte Carlo -- this is a ranking screen, not the final fit),
* de-parities: each sweep re-zeroed to its own t0,
* fits D three ways -- de-parity merged, and each sweep separately,
* scores a clean diffuser as: de-parity-merged D > 0 with good fit quality,
  BOTH sweeps positive and mutually consistent, and reaching the low-CNR regime.

Writes ``triage_deparity.csv`` to the WIP area and prints the ranked head.

Usage::

    .venv/Scripts/python.exe scripts/realdata_triage.py
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import fft_cnr  # noqa: E402
from width_debias.fit import fit_D, fit_variance  # noqa: E402
from collaborator_data import stroboscat_scans_dir  # noqa: E402

ROOT = str(stroboscat_scans_dir())   # the scan directory (DATASETS.md), read only
OUT = os.path.join(os.path.dirname(__file__), "diagnostics", "triage_deparity.csv")
FLOOR = 1.3
A_WHITE = 0.4
MIN_SWEEP = 4          # min kept frames to fit a sweep's D
AGREE_RATIO = 2.5      # sweeps "agree" if max|D|/min|D| below this (both > 0)


def _orient(profiles):
    ref = profiles[:4].mean(0) - np.median(profiles[:4].mean(0))
    if abs(ref.min()) > abs(ref.max()):
        ref = -ref
    refc = ref - ref.mean()
    out = []
    for p in profiles:
        pp = p - np.median(p)
        out.append((np.sign(np.dot(pp - pp.mean(), refc)) or 1.0) * pp)
    return np.array(out)


def _ivfit(t, msd, se, mask):
    w = (1.0 / np.maximum(se, 1e-9) ** 2) * mask
    if mask.sum() < MIN_SWEEP:
        return None
    return fit_D(t, msd, w)


def _process(folder):
    fs = [f for f in glob.glob(os.path.join(folder, "analysis_*_.npy"))
          if not f.endswith("_dataset.npy")]
    f = max(fs, key=lambda x: os.path.basename(x).count(",") + 1)
    a = np.load(f, allow_pickle=True).flat[0]
    t = np.asarray(a["time_arr(ns)"], float)
    prof = np.asarray(a["profiles"], float)
    sig = np.asarray(a["sigmas(um)"], float)
    err = np.asarray(a["err_sigmas"], float)
    if prof.ndim != 2 or prof.shape[0] != t.size or t.size < 8:
        return None
    # batches from the acquisition-order time resets, before sorting
    batch_acq, b = np.zeros(t.size, int), 0
    for i in range(1, t.size):
        if t[i] < t[i - 1]:
            b += 1
        batch_acq[i] = b
    o = np.argsort(t)
    t, prof, sig, err, batch = t[o], prof[o], sig[o], err[o], batch_acq[o]
    prof = _orient(prof)
    npix = prof.shape[1]
    center = int(np.argmax(gaussian_filter1d(prof[:4].mean(0), 2)))
    s0 = fit_variance(np.arange(npix, dtype=float), prof[0]).variance
    half = max(6, int(round(4.0 * np.sqrt(s0))))
    lo, hi = max(0, center - half), min(npix, center + half)
    xs = np.arange(hi - lo, dtype=float)
    var = np.array([fit_variance(xs, p[lo:hi]).variance for p in prof])
    cnr = np.array([float(fft_cnr.fft_cnr(p[lo:hi],
                    fit_model="generalized_gaussian").cnr) for p in prof])
    sig0_px2 = float(var[0])
    keep = cnr >= FLOOR
    se = var * 2.0 * (err / np.where(sig == 0, np.nan, sig))
    vb = A_WHITE / np.maximum(cnr, 1e-6) ** 2 * sig0_px2
    vc = var - vb                                   # corrected (analytic proxy)
    # de-parity: each sweep re-zeroed to its own t0
    base = np.zeros_like(vc)
    for bb in set(batch):
        sel = batch == bb
        base[sel] = vc[sel][0]
    msd_dp = vc - base
    d_dp = _ivfit(t, msd_dp, se, keep)
    # per-sweep
    dsw = {}
    for bb in sorted(set(batch)):
        sel = batch == bb
        msd_s = np.where(sel, vc - base, 0.0)
        dsw[bb] = _ivfit(t, msd_s, se, keep & sel)
    ds = [dsw[bb].D for bb in sorted(dsw) if dsw[bb] is not None]
    both_pos = len(ds) >= 2 and all(d > 0 for d in ds)
    agree = both_pos and (max(ds) / max(min(ds), 1e-9) <= AGREE_RATIO)
    tstat = (d_dp.D / d_dp.D_se) if (d_dp and d_dp.D_se and
             np.isfinite(d_dp.D_se) and d_dp.D_se > 0) else np.nan
    return dict(
        nfr=t.size, nkept=int(keep.sum()), nbatch=len(set(batch)),
        cnrlo=float(np.nanmin(cnr)), cnrhi=float(np.nanmax(cnr)),
        sig0=float(np.sqrt(sig0_px2)),
        D_dp=(d_dp.D if d_dp else np.nan),
        D_s0=(ds[0] if len(ds) > 0 else np.nan),
        D_s1=(ds[1] if len(ds) > 1 else np.nan),
        both_pos=int(both_pos), agree=int(agree), tstat=tstat,
        nbias=int(np.sum(keep & (cnr < 6))))


def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"No scan directory at {ROOT} (WIDTH_DEBIAS_DATA_ROOT, or else data/). See DATASETS.md.")
    files = [f for f in glob.glob(os.path.join(ROOT, "**", "analysis_*_.npy"),
                                  recursive=True) if not f.endswith("_dataset.npy")]
    bydir = {}
    for f in files:
        d = os.path.dirname(f)
        n = os.path.basename(f).count(",") + 1
        if d not in bydir or n > bydir[d][1]:
            bydir[d] = (f, n)
    rows = []
    for d in sorted(bydir):
        try:
            r = _process(d)
        except Exception:
            r = None
        if r is None:
            continue
        r["meas"] = os.path.relpath(d, ROOT)
        rows.append(r)

    def clean(r):
        return (np.isfinite(r["D_dp"]) and r["D_dp"] > 0 and r["agree"]
                and np.isfinite(r["tstat"]) and r["tstat"] > 3 and r["nkept"] >= 8
                and r["nbias"] >= 2)
    rows.sort(key=lambda r: (clean(r), r["tstat"] if np.isfinite(r["tstat"])
                             else -1), reverse=True)
    cols = ["meas", "nfr", "nkept", "nbatch", "cnrlo", "cnrhi", "sig0",
            "D_dp", "D_s0", "D_s1", "both_pos", "agree", "tstat", "nbias"]
    with open(OUT, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r[c] for c in cols})
    n_clean = sum(clean(r) for r in rows)
    print(f"{len(rows)} measurements; {n_clean} pass the clean-diffuser screen; "
          f"wrote {OUT}\n")
    print(f"{'D_dp':>6}{'D_s0':>7}{'D_s1':>7}{'agree':>6}{'tstat':>6}{'kept':>5}"
          f"{'bias':>5}{'cnrlo':>6}{'cnrhi':>7}{'sig0':>6}  measurement")
    for r in rows[:30]:
        print(f"{r['D_dp']:>6.2f}{r['D_s0']:>7.2f}{r['D_s1']:>7.2f}"
              f"{r['agree']:>6d}{r['tstat']:>6.1f}{r['nkept']:>5d}{r['nbias']:>5d}"
              f"{r['cnrlo']:>6.1f}{r['cnrhi']:>7.1f}{r['sig0']:>6.1f}  {r['meas']}")


if __name__ == "__main__":
    main()
