"""Early-time MSD plots with a manual per-measurement time cutoff.

Reads the per-measurement ``manifest.csv`` produced by
``realdata_diagnostics.py`` (so no Monte Carlo is rerun -- the corrected
variances are already in the manifest), keeps the CNR-floor frames at or below a
manual cutoff time, re-baselines the MSD so the first retained point sits at
zero, and fits the naive and corrected MSD with an inverse-variance line over
that early window.

At these short cutoffs the per-frame CNR is high, so ``var_bias`` is small and
the naive and corrected points/fits nearly coincide -- the plot reports the
clean early-regime D (and the naive/corrected agreement is a consistency check);
the noise-bias effect lives in the late low-CNR frames that the cutoff
removes.

Usage::

    .venv/Scripts/python.exe scripts/realdata_early_cut_plots.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from width_debias.fit import fit_D  # noqa: E402
from realdata_diagnostics import MEASUREMENTS, _find_canonical  # noqa: E402

WIP = os.path.join(os.path.dirname(__file__), "diagnostics")

# measurement label -> manual cutoff time (ns): keep CNR-floor frames at t <= cut
CUTOFFS = {
    "mixed_tmdc_26_32_250716": 2.0,
    "mixed_tmdc_50_50_250702": 1.0,
    "mixed_tmdc_comparison_all_0": 1.0,
    "mixed_tmdc_pureS_250701": 1.0,
    "WSe2_Power_var_250224_0p020": 1.0,
}

# All five measurements are two-sweep merges (a coarse and a fine delay sweep,
# detected by the single time reset in the acquisition order), each sweep with
# its own sigma^2(0). A single global baseline is slope-invariant and would not
# change D; re-zeroing each merged sweep to its own t0 is the legitimate
# multi-dataset offset, and it does change D (it removes the spurious slope the
# inter-sweep baseline difference otherwise injects).
DEPARITY = set(CUTOFFS)

GOLD = "#f7941e"      # naive
BLUE = "#0072B2"      # corrected
INCH = 1 / 2.54


def _read_manifest(label):
    path = os.path.join(WIP, label, "manifest.csv")
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["time_ns"]) for r in rows])
    vn = np.array([float(r["var_naive_px2"]) for r in rows])
    sn = np.array([float(r["se_naive_px2"]) for r in rows])
    vc = np.array([float(r["var_corr_px2"]) for r in rows])
    sc = np.array([float(r["se_corr_px2"]) for r in rows])
    keep = np.array([int(r["keep_flag"]) for r in rows]).astype(bool)
    cnr = np.array([float(r["cnr_white"]) for r in rows])
    return t, vn, sn, vc, sc, keep, cnr


def _fit(t, msd, se):
    return fit_D(t, msd, 1.0 / np.maximum(se, 1e-9) ** 2)


def _label_to_folder(label):
    for lbl, folder in MEASUREMENTS.values():
        if lbl == label:
            return folder
    raise KeyError(label)


def _batch_baseline(label, t_sorted, var_sorted):
    """Per even/odd-batch sigma^2(0): each acquisition batch (detected by the
    time reset in the original acquisition order) gets the sigma^2 of its own
    earliest-time frame as the MSD baseline. Returns the baseline per frame
    (aligned to the time-sorted manifest order)."""
    a = np.load(_find_canonical(_label_to_folder(label)),
                allow_pickle=True).flat[0]
    t_orig = np.asarray(a["time_arr(ns)"], float)
    batch_of, b = {}, 0
    for i, tt in enumerate(t_orig):
        if i > 0 and t_orig[i] < t_orig[i - 1]:
            b += 1
        batch_of[round(float(tt), 3)] = b
    batches = np.array([batch_of[round(float(tt), 3)] for tt in t_sorted])
    baseline = np.zeros_like(var_sorted)
    for bb in set(batches):
        sel = batches == bb
        baseline[sel] = var_sorted[np.where(sel)[0][0]]   # earliest in batch
    return baseline


def plot_one(label, cutoff):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, vn, sn, vc, sc, keep, cnr = _read_manifest(label)
    deparity = label in DEPARITY
    if deparity:
        # each even/odd batch gets its own sigma^2(0) baseline (computed over the
        # full series, so the cut does not move the baseline)
        base_n = _batch_baseline(label, t, vn)
        base_c = _batch_baseline(label, t, vc)
        msd_n_all, msd_c_all = vn - base_n, vc - base_c
    else:
        # single baseline at the first frame
        msd_n_all, msd_c_all = vn - vn[0], vc - vc[0]
    m = keep & (t <= cutoff + 1e-9)            # CNR-floor frames up to the cutoff
    t_m = t[m]
    msd_n, msd_c = msd_n_all[m], msd_c_all[m]
    if not deparity:                            # re-zero to the first kept point
        msd_n, msd_c = msd_n - msd_n[0], msd_c - msd_c[0]
    sn_m, sc_m = sn[m], sc[m]
    d_n = _fit(t_m, msd_n, sn_m)
    d_c = _fit(t_m, msd_c, sc_m)

    fs = 8
    fig, ax = plt.subplots(layout="constrained", figsize=(9 * INCH, 7 * INCH))
    ax.tick_params(axis="both", which="both", direction="in", top=True,
                   right=True, labelsize=fs, length=4, width=1)
    fig.patch.set_facecolor("w")
    fig.patch.set_alpha(1)
    tt = np.linspace(t_m.min(), t_m.max(), 50)
    ax.plot(tt, d_n.intercept + 2 * d_n.D * tt, "--", color=GOLD, lw=1.5,
            label=f"naive fit: D={d_n.D:.2f}")
    ax.plot(tt, d_c.intercept + 2 * d_c.D * tt, "-", color=BLUE, lw=1.5,
            label=f"corrected fit: D={d_c.D:.2f}")
    ax.errorbar(t_m, msd_n, yerr=sn_m, fmt="o", mfc="none", mec=GOLD,
                ecolor=GOLD, ms=5, capsize=2, lw=1, label="naive MSD")
    ax.errorbar(t_m, msd_c, yerr=sc_m, fmt="o", color=BLUE, ms=4, capsize=2,
                lw=1, label="corrected MSD")
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    ax.set_xlabel("time after pump (ns)", fontsize=fs)
    ax.set_ylabel(r"MSD $\sigma^2(t)-\sigma^2(0)$ (px$^2$)", fontsize=fs)
    ax.legend(frameon=False, fontsize=fs - 1)
    out = os.path.join(WIP, label, f"msd_earlycut_{cutoff:g}ns.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    ratio = d_c.D / d_n.D if d_n.D else float("nan")
    print(f"  {label}: cut<={cutoff:g}ns, {int(m.sum())} pts, "
          f"CNR {cnr[m].min():.1f}-{cnr[m].max():.1f}  "
          f"D naive {d_n.D:.3f}  corrected {d_c.D:.3f}  ratio {ratio:.3f}",
          flush=True)
    print(f"    -> {out}", flush=True)


def main():
    for label, cut in CUTOFFS.items():
        plot_one(label, cut)


if __name__ == "__main__":
    main()
