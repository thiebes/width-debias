"""Reduce CsPbBr3 single-crystal diffusion CSVs to per-frame profiles.

Takes a folder of raw acquisition CSVs (one per repeat; each is spatial-position
rows by TCSPC-time-bin columns) and produces the per-frame spatial profiles the
correction operates on, plus the per-frame fitted variance and CNR.

The reduction (pulse merge, peak ordering, time binning, spatial centering and
averaging across repeats) mirrors Upasana's `1D Diffusion evaluation` pipeline so
the result matches theirs; the per-frame width fit and CNR use this project's
own `fit_variance` and `fft-cnr`.

Output: an ``.npz`` with ``t_ns``, ``x_um``, ``profiles`` (n_frames, n_pixels),
``variance_um2`` and ``cnr`` (n_frames,), plus the geometry the calibration must
match (``half_window_um``, ``max_var``). Feed it to ``apply_correction.py``.

Example::

    python experiments/cspbbr-strobescat/reduce.py \
        --data "data/cspbbr_sample_1/13.47 uJ_Fluence_19_May_2026" \
        --n-files 3 --max-var 6.25 --out output/cspbbr_13p47.npz

(``data/`` is the default data root; with ``WIDTH_DEBIAS_DATA_ROOT`` set, the
folder is under that root.) ``--max-var`` must equal the calibration's ``--max-var`` (the fit's variance cap
must be identical in calibration and application).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from width_debias.fit import fit_variance        # noqa: E402
try:
    from width_debias.cnr import per_frame_cnr
    HAVE_FFT_CNR = True
except Exception:                                  # pragma: no cover
    HAVE_FFT_CNR = False

# Picoharp-330 detection optics constant (matches the source pipeline).
LENS_MAGNIFICATION = 3.636363633333333


# --- raw CSV loader (handles the trailing comma) ----------------------------
def load_csv(path: str, ncols: int) -> np.ndarray:
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip(",\n").split(",")
            rows.append(np.array(parts[:ncols], dtype=float))
    return np.array(rows)


# --- source reduction functions (faithful to the lab pipeline) --------------
def merge_pulses(data, time_resolution_ns, divider, n_pulses, max_rep_rate):
    rep_rate = max_rep_rate / divider
    bins_between = int(((1 / rep_rate) * 1e9) / time_resolution_ns)
    merged = []
    for row in data:
        counts = row[:bins_between]
        for k in range(1, n_pulses):
            counts = counts + row[bins_between * k:bins_between * (k + 1)]
        merged.append(counts)
    return merged


def constant_bin(x, n):
    """Average consecutive groups of ``n`` samples along axis 0."""
    out = []
    i = 0
    while i < len(x):
        out.append(np.mean(x[i:i + n], axis=0))
        i += n
    return np.array(out)


def reduce_folder(data_dir, n_files, ncols, tbin, cut_time):
    meta_files = sorted(glob.glob(os.path.join(data_dir, "*metadata.json")))
    with open(meta_files[0]) as f:
        md = json.load(f)
    time_resolution_ns = md["TimeResolution"] * 1e-3
    divider = md["laserDivider"]
    steps = md["steps"]
    objective = md["objective"]
    max_rep_rate = md["MaxRepRate"] * 1e6
    xmin, xmax = md["x_range"]
    ymin, ymax = md["y_range"]
    apd_range = np.linalg.norm(np.array([xmin, ymin]) - np.array([xmax, ymax]))
    spatial_res_mm = (apd_range / (LENS_MAGNIFICATION * objective)) / steps

    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))[:n_files]
    if not csvs:
        raise SystemExit(f"no CSVs found in {data_dir}")

    binned = []
    for c in csvs:
        arr = load_csv(c, ncols)[1:]                 # drop header row
        merged = merge_pulses(arr, time_resolution_ns, divider, 1, max_rep_rate)
        merged = np.array(merged)
        peak = int(np.argmax(merged.sum(axis=0)))    # order so the peak is t=0
        merged = np.roll(merged, -peak, axis=1)
        binned.append(constant_bin(merged.T, tbin))  # (n_tframes, n_pixels)

    t_uncut = np.arange(binned[0].shape[0]) * tbin * time_resolution_ns
    t_uncut = t_uncut - t_uncut.min()
    lo, hi = [int(np.abs(t_uncut - t).argmin()) for t in (0.0, cut_time)]

    # spatial centering + averaging across repeats
    min_w = 10_000
    centered = []
    for b in binned:
        win = b[lo:hi]
        c0 = int(np.argmax(win[:5].mean(axis=0)))    # peak column at early time
        rm = c0 * 2 - win.shape[1]
        if rm > 0:
            cen = np.delete(win, range(abs(rm)), axis=1)
        elif rm < 0:
            cen = np.delete(win, np.array(range(abs(rm))) + c0 * 2, axis=1)
        else:
            cen = win
        min_w = min(min_w, cen.shape[1])
        centered.append(cen)
    merged = np.mean([c[:, :min_w] for c in centered], axis=0)

    x_um = (np.arange(merged.shape[1]) - int(np.argmax(merged.sum(axis=0)))) \
        * spatial_res_mm * 1e3
    t_ns = t_uncut[lo:hi] - t_uncut[lo]
    return t_ns, x_um, merged, len(csvs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder of raw acquisition CSVs")
    p.add_argument("--n-files", type=int, default=3,
                   help="number of repeat acquisitions to merge (default 3)")
    p.add_argument("--ncols", type=int, default=32768, help="time bins per CSV")
    p.add_argument("--tbin", type=int, default=70, help="time-binning width")
    p.add_argument("--cut-time", type=float, default=20.0, help="ns window to keep")
    p.add_argument("--max-var", type=float, default=6.25,
                   help="fit variance cap (um^2); MUST equal the calibration's "
                        "--max-var")
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    t_ns, x_um, profiles, n_used = reduce_folder(
        args.data, args.n_files, args.ncols, args.tbin, args.cut_time)
    print(f"reduced {n_used} acquisitions -> {profiles.shape[0]} frames x "
          f"{profiles.shape[1]} pixels, half-window {np.max(np.abs(x_um)):.3f} um")

    variance = np.array([fit_variance(x_um, row, max_var=args.max_var).variance
                         for row in profiles])
    if HAVE_FFT_CNR:
        cnr = per_frame_cnr(profiles)
    else:
        edge = max(3, profiles.shape[1] // 8)
        floor = np.array([np.std(np.r_[r[:edge], r[-edge:]]) for r in profiles])
        peak = profiles.max(axis=1) - np.median(profiles[:, :edge])
        cnr = peak / np.where(floor > 0, floor, np.nan)
        print("WARNING: fft-cnr not installed; using a crude edge-std CNR")

    np.savez(args.out, t_ns=t_ns, x_um=x_um, profiles=profiles,
             variance_um2=variance, cnr=cnr,
             half_window_um=np.array([float(np.max(np.abs(x_um)))]),
             max_var=np.array([args.max_var]), n_files=np.array([n_used]))
    print(f"saved {args.out}")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(t_ns, cnr, "o-"); ax[0].axhline(1.3, color="r", ls="--")
        ax[0].set_xlabel("time (ns)"); ax[0].set_ylabel("per-frame CNR")
        ax[1].plot(t_ns, np.sqrt(np.clip(variance, 0, None)), "o-")
        ax[1].set_xlabel("time (ns)"); ax[1].set_ylabel("sigma (um)")
        ax[2].plot(t_ns, variance - variance[0], "o-")
        ax[2].set_xlabel("time (ns)"); ax[2].set_ylabel("MSD (um^2)")
        fig.tight_layout()
        fig.savefig(os.path.splitext(args.out)[0] + "_diagnostic.png", dpi=110)
        print("saved diagnostic figure")


if __name__ == "__main__":
    main()
