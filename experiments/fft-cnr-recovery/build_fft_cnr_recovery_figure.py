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
(Gaussian 1-D profiles, fft-cnr 0.4.0). The full 1000-trial array is 2.3 MB and is
not committed here; this script reduces it to a tiny per-(N, CNR) summary
(`fft_cnr_recovery_summary.npz`, committed) and plots from that. The summary
records the fft-cnr version and the SHA-256 of the array it was reduced from, and
whenever the array is available the script checks it against that hash, so a
changed source cannot leave the summary stale unnoticed. Delete the summary to
force a re-reduction from the source, which is read from the directory that
WIDTH_DEBIAS_EXTERNAL_RESULTS names. See DATASETS.md.

Run:
    .venv/Scripts/python.exe experiments/fft-cnr-recovery/build_fft_cnr_recovery_figure.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import warnings
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
SOURCE_NAME = "case1_gaussian_1d.npz"
# The fft-cnr release that produced the source estimates; the source does not
# record it, so change this together with the source.
FFT_CNR_VERSION = "0.4.0"
SUMMARY = HERE / "fft_cnr_recovery_summary.npz"
FIGURE = HERE / "fft_cnr_recovery.png"

inch = 1 / 2.54
# Spanish-flag palette (plot_style.md). One story (window size) -> one hue ranked
# by alpha; blue is the reference/signal role. The 1:1 line is a neutral guide.
FLAG_BLUE = "#0077c8"
LIMS = (1.0, 150.0)


def _source_path(required: bool) -> Path | None:
    """The source array in the WIDTH_DEBIAS_EXTERNAL_RESULTS directory.

    Read when needed rather than at import. With ``required`` an unset variable
    or a missing file raises, naming which; otherwise either gives None.
    """
    directory = os.environ.get("WIDTH_DEBIAS_EXTERNAL_RESULTS")
    if not directory:
        if required:
            raise FileNotFoundError(
                f"WIDTH_DEBIAS_EXTERNAL_RESULTS is not set; set it to the directory that "
                f"holds {SOURCE_NAME}. See DATASETS.md.")
        return None
    path = Path(directory).expanduser().resolve() / SOURCE_NAME
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"{SOURCE_NAME} not found in {path.parent} "
                "(WIDTH_DEBIAS_EXTERNAL_RESULTS). See DATASETS.md.")
        return None
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reduce_source(source: Path) -> dict:
    """Collapse the 1000-trial source array to per-(N, CNR) median and spread.

    Only the median and the 25/75 and 5/95 percentiles are needed for the panel,
    so the committed artifact is ~120 floats rather than the 2.3 MB source. Infinite
    estimates (a zero-noise trial divides by zero) are dropped before the
    percentiles, matching the source notebook. The fft-cnr version and the source's
    SHA-256 are stored with the numbers, so the summary names what it came from.
    """
    d = np.load(source)
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
        fft_cnr_version=FFT_CNR_VERSION,
        source_sha256=_sha256(source),
    )


def _load_summary() -> dict:
    """Return the committed per-(N, CNR) summary, reducing from source if absent.

    When the summary exists and the source is also available, the source's hash
    is checked against the one the summary records, and a mismatch warns.
    """
    if not SUMMARY.exists():
        source = _source_path(required=True)
        summ = _reduce_source(source)
        np.savez(SUMMARY, **summ)
        print(f"Reduced {SOURCE_NAME} -> {SUMMARY.name} (fft-cnr {FFT_CNR_VERSION})")
        return summ
    d = np.load(SUMMARY)
    summ = {k: d[k] for k in d.files}
    source = _source_path(required=False)
    if source is not None and _sha256(source) != str(summ["source_sha256"]):
        warnings.warn(f"{SOURCE_NAME} differs from the array {SUMMARY.name} was reduced "
                      "from; delete the summary and rerun to rebuild it, and if the "
                      "source comes from another fft-cnr release, first update "
                      "FFT_CNR_VERSION and the version the caption, README and "
                      "DATASETS.md quote", stacklevel=2)
    return summ


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
