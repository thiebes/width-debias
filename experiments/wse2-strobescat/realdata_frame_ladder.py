"""A CNR ladder of individual profile frames: benefits and limitations.

Generates clean single-panel profile overlays (raw 1-D profile + naive Gaussian
+ corrected Gaussian) for a hand-picked set of frames spanning the per-frame CNR
range, to show where the noise-bias correction helps and where it fails.
Reads widths from the per-measurement ``manifest.csv`` (px^2) and the profile
from the analysis file; nothing is re-fit.

Usage::

    .venv/Scripts/python.exe scripts/realdata_frame_ladder.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from realdata_diagnostics import MEASUREMENTS, load_measurement  # noqa: E402

WIP = os.path.join(os.path.dirname(__file__), "diagnostics")
OUTDIR = os.path.join(WIP, "_frame_ladder")
RED = "#c60b1e"
BLUE = "#0077c8"
GRAY = "0.55"
INCH = 1 / 2.54

# (registry name, target time ns, short caption) -- a ladder from high to low CNR
LADDER = [
    ("wse2_250722_0p043", -0.00, "high CNR: no bias, correction leaves it alone"),
    ("wse2_250224_0p020", 2.00, "moderate CNR: small bias, small correction"),
    ("mixed_tmdc_50_50_250702", 1.40, "near floor: large useful correction"),
    ("mixed_tmdc_pureS_250701", 6.80, "at the floor: strong correction, limit of use"),
    ("mixed_tmdc_26_32_250716", 9.80, "below floor: over-subtraction (fails)"),
]


def _manifest_row(label, t_target):
    rows = list(csv.DictReader(open(os.path.join(WIP, label, "manifest.csv"))))
    ts = np.array([float(r["time_ns"]) for r in rows])
    i = int(np.argmin(np.abs(ts - t_target)))
    r = rows[i]
    return (float(r["time_ns"]), float(r["cnr_white"]),
            float(r["var_naive_px2"]), float(r["se_naive_px2"]),
            float(r["var_corr_px2"]))


def _gauss(x, c, var, amp):
    return amp * np.exp(-0.5 * (x - c) ** 2 / var)


def make_frame(name, t_target, caption, idx):
    label, folder = MEASUREMENTS[name]
    m = load_measurement(folder)
    t, profiles, center = m["t"], m["profiles"], m["center"]
    i = int(np.argmin(np.abs(t - t_target)))
    prof = profiles[i]
    t_ns, cnr, var_n, se_n, var_c = _manifest_row(label, t_target)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = 9
    half = int(max(14, 4.0 * np.sqrt(max(var_n, 1.0))))
    lo, hi = max(0, center - half), min(prof.size, center + half)
    x = np.arange(lo, hi)
    amp = float(prof[center])
    xx = np.linspace(lo, hi, 300)
    fig, ax = plt.subplots(layout="constrained", figsize=(13 * INCH, 10 * INCH))
    ax.tick_params(axis="both", which="both", direction="in", top=True,
                   right=True, labelsize=fs, length=4, width=1)
    fig.patch.set_facecolor("w"); fig.patch.set_alpha(1)
    ax.plot(x, prof[lo:hi], "o", color=GRAY, ms=3, label="profile")
    ax.plot(xx, _gauss(xx, center, var_n, amp), "-", color=RED, lw=1.6,
            label=f"naive  $\\sigma^2$={var_n:.1f} px$^2$")
    if var_c > 0:
        ax.plot(xx, _gauss(xx, center, var_c, amp), "-", color=BLUE, lw=1.6,
                label=f"corrected  $\\sigma^2$={var_c:.1f} px$^2$")
    else:
        ax.plot([], [], " ", label=f"corrected  $\\sigma^2$={var_c:.1f} "
                                   f"px$^2$ (negative)")
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel("position (px)", fontsize=fs)
    ax.set_ylabel("differential signal (a.u.)", fontsize=fs)
    ax.set_title(f"CNR = {cnr:.1f}   (t = {t_ns:+.2f} ns,  {label})\n{caption}",
                 fontsize=fs)
    ax.legend(frameon=False, fontsize=fs, loc="upper right")
    out = os.path.join(OUTDIR, f"{idx}_cnr{cnr:04.1f}_{label}.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"  CNR {cnr:5.1f}  naive {var_n:6.1f}  corr {var_c:7.1f}  "
          f"{name} t={t_ns:+.2f}  -> {os.path.basename(out)}", flush=True)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for k, (name, t, cap) in enumerate(LADDER, 1):
        make_frame(name, t, cap, k)
    print(f"\n  ladder -> {OUTDIR}")


if __name__ == "__main__":
    main()
