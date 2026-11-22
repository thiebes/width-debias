"""Poster hero figure (Panel 4, the pivot): fft-cnr recovers the true per-frame
contrast-to-noise ratio.

The correction takes the per-frame CNR as its only data input, so the whole scheme
rests on that CNR being measurable from the recorded profile alone. This panel is
that guarantee: over 1000 noisy Gaussian realisations at each of six true CNR
values, the fft-cnr estimate (median plus interquartile / 5-95 spread) sits on the
1:1 line across window sizes N = 128-1024. The message is robustness -- every window
size recovers the true CNR -- so the four N series share one hue and separate by
alpha (plot_style "related series"), reading as a single diagonal-hugging story.

Data provenance: the estimates come from an external simulation study of fft-cnr
(Gaussian 1-D profiles). The full 1000-trial array is 2.3 MB and
is not committed here; this script reduces it to a tiny per-(N, CNR) summary
(`fft_cnr_recovery_summary.npz`, committed) and plots from that. Delete the summary
to force a re-reduction from the source, which is read from the directory that
WIDTH_DEBIAS_EXTERNAL_RESULTS names. See DATASETS.md.

Run:
    .venv/Scripts/python.exe experiments/fft-cnr-recovery/build_fft_cnr_recovery_figure.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
# The full-trial source array is external; its directory comes from the
# environment (see DATASETS.md).
_EXTERNAL = os.environ.get("WIDTH_DEBIAS_EXTERNAL_RESULTS")
SOURCE = Path(_EXTERNAL) / "case1_gaussian_1d.npz" if _EXTERNAL else None
SUMMARY = HERE / "fft_cnr_recovery_summary.npz"
FIGURE = HERE / "fft_cnr_recovery.png"

inch = 1 / 2.54
# Spanish-flag palette (plot_style.md). One story (window size) -> one hue ranked
# by alpha; blue is the reference/signal role. The 1:1 line is a neutral guide.
FLAG_BLUE = "#0077c8"
LIMS = (1.0, 150.0)


def _reduce_source() -> dict:
    """Collapse the 1000-trial source array to per-(N, CNR) median and spread.

    Only the median and the 25/75 and 5/95 percentiles are needed for the panel,
    so the committed artifact is ~120 floats rather than the 2.3 MB source. Infinite
    estimates (a zero-noise trial divides by zero) are dropped before the
    percentiles, matching the source notebook.
    """
    d = np.load(SOURCE)
    true_cnr = d["true_cnr_values"].astype(float)
    n_values = d["N_values"].astype(int)
    est = d["results_template"]["est_cnr"].astype(float)      # (N, CNR, trials)
    est = np.where(np.isfinite(est), est, np.nan)
    return dict(
        true_cnr=true_cnr,
        n_values=n_values,
        median=np.nanmedian(est, axis=2),
        q25=np.nanpercentile(est, 25, axis=2),
        q75=np.nanpercentile(est, 75, axis=2),
        p05=np.nanpercentile(est, 5, axis=2),
        p95=np.nanpercentile(est, 95, axis=2),
        n_trials=int(d["n_trials"]),
    )


def _load_summary() -> dict:
    """Return the committed per-(N, CNR) summary, reducing from source if absent."""
    if not SUMMARY.exists():
        if SOURCE is None or not SOURCE.exists():
            raise FileNotFoundError(
                f"Neither the committed summary ({SUMMARY.name}) nor the source "
                f"array is present; set WIDTH_DEBIAS_EXTERNAL_RESULTS to the "
                f"directory that holds it. See DATASETS.md.")
        summ = _reduce_source()
        np.savez(SUMMARY, **summ)
        print(f"Reduced {SOURCE.name} -> {SUMMARY.name}")
        return summ
    d = np.load(SUMMARY)
    return {k: d[k] for k in d.files}


def build() -> Path:
    """fft-cnr estimate vs true CNR, all window sizes hugging the 1:1 line."""
    s = _load_summary()
    true_cnr = s["true_cnr"]
    n_values = s["n_values"]
    fs = 11

    fig, ax = plt.subplots(layout="constrained", figsize=(9.5 * inch, 9.0 * inch))

    # 1:1 reference: a thin neutral guide the estimates should sit on.
    ax.plot(LIMS, LIMS, ls=(0, (4, 3)), color="#555555", lw=1.0, zorder=1)

    # One hue, alpha ranked faint -> full with N (more samples = tighter recovery).
    alphas = np.linspace(0.35, 1.0, len(n_values))
    for i, (N, a) in enumerate(zip(n_values, alphas)):
        # interquartile envelope, very light: overlapping N form a soft blue band
        # around the diagonal that reads as "tight recovery" without cluttering.
        ax.fill_between(true_cnr, s["q25"][i], s["q75"][i],
                        color=FLAG_BLUE, alpha=0.10, lw=0, zorder=2)
        ax.plot(true_cnr, s["median"][i], marker="o", ms=4, color=FLAG_BLUE,
                alpha=a, lw=2.0, zorder=3, label=f"N = {N}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(LIMS)
    ax.set_ylim(LIMS)
    ax.set_aspect("equal")
    ax.set_xlabel("true CNR", fontsize=fs)
    ax.set_ylabel("fft-cnr estimate", fontsize=fs)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xticks([1, 10, 100])
    ax.set_yticks([1, 10, 100])
    ax.tick_params(axis="both", which="both", direction="in",
                   top=False, right=False, labelsize=fs)
    ax.legend(fontsize=fs, loc="upper left", frameon=False, handlelength=1.2)

    # Transparent margin, opaque plot area (plot_style rule 8) so the panel drops
    # onto the poster background without a white halo.
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("white")
    fig.savefig(FIGURE, dpi=300)
    plt.close(fig)
    return FIGURE


if __name__ == "__main__":
    print(build())
