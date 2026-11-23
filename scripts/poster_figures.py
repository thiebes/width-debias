"""Assemble the poster's figures by importing each panel's builder from its home.

The panel render code lives with its figure -- schematics in `illustrations/`, data
panels with their experiment under `experiments/` -- and this module is the
orchestrator that calls them. The only panels defined here are the two fft-cnr
recovery figures. Their sweeps come from `scripts/poster_fftcnr_dense.py`, and
Panel 4 falls back to an external simulation study's sweep, so they have no
width-debias experiment to live in. Shared poster primitives (palette, axis
style) come from `scripts/poster_style.py`.

Run from the repository root::

    .venv/Scripts/python.exe scripts/poster_figures.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(__file__)
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)                                  # poster_style
for _rel in ("illustrations/measurement-schematic",
             "illustrations/wall-fold-schematic",
             "experiments/crlb-precision",
             "experiments/dfree-calibration",
             "experiments/correction-floor"):
    sys.path.insert(0, os.path.join(_ROOT, *_rel.split("/")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from poster_style import ROLE1, ROLE3, ROLE5, save_poster, style as _style  # noqa: E402

# Panel builders, each imported from its home (single source of truth).
from build_measurement_schematic import build as figure_schematic_measurement  # noqa: E402
from build_wall_fold_schematic import build as figure_schematic_wall_fold  # noqa: E402
from render_crlb_panel import build as figure_crlb  # noqa: E402
from render_bias_panels import build_bias_curves, build_corrected_bias  # noqa: E402
from render_floor_panel import build as figure_floor  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
OUTPUT.mkdir(exist_ok=True)

# The Panel 4 CNR-recovery figure restyles an external simulation study's sweep
# of fft-cnr over profile shapes. Prefer the dense poster-local sweep
# (scripts/poster_fftcnr_dense.py) when present, so the ratio curve is smooth;
# otherwise fall back to the external six-point file, read from the directory
# that WIDTH_DEBIAS_EXTERNAL_RESULTS names.
_FFTCNR_DENSE = OUTPUT / "gengauss_cnr_sweep_dense.npz"
_FFTCNR_EXTERNAL_NAME = "case11_gengauss_cnr_sweep.npz"


def _fftcnr_sweep() -> Path | None:
    """The Panel 4 sweep: the dense local file, else the external one; None if neither.

    The variable is read here, when the panel needs it, and the note names the
    files looked for and where.
    """
    if _FFTCNR_DENSE.exists():
        return _FFTCNR_DENSE
    directory = os.environ.get("WIDTH_DEBIAS_EXTERNAL_RESULTS")
    if not directory:
        print(f"  ({_FFTCNR_DENSE.name} not in output/ and WIDTH_DEBIAS_EXTERNAL_RESULTS "
              "is not set -- skipping Panel 4 fft-cnr)")
        return None
    path = Path(directory).expanduser().resolve() / _FFTCNR_EXTERNAL_NAME
    if not path.exists():
        print(f"  ({_FFTCNR_DENSE.name} not in output/ and {_FFTCNR_EXTERNAL_NAME} not "
              f"found in {path.parent} -- skipping Panel 4 fft-cnr)")
        return None
    return path


# Sweep shape name -> (poster label, role colour); shared by the fft-cnr figures.
_FFTCNR_STYLE = {
    "Gaussian (p=2)": ("Gaussian", ROLE1),
    "heavy-tail (p=1.3)": ("gen-Gaussian (heavy tail)", ROLE3),
    "flat-top (p=4)": ("gen-Gaussian (flat top)", "#888888"),
    "Voigt": ("Voigt", ROLE5),
}


def figure_fftcnr(npz_path: Path | None) -> Path | None:
    """Panel 4 (CNR-centric spine): fft-cnr recovers CNR across profile shapes.

    Restyles the fft-cnr shape sweep (generalized-Gaussian fit
    path -- the one width-debias uses) into the poster palette: the recovery
    ratio (estimated / true CNR) vs true CNR, median + interquartile band, one
    line per shape, centred on 1.0. The three poster-primary shapes carry roles
    1-3; the flat-top is a neutral secondary reference.
    """
    if npz_path is None:
        return None
    if not npz_path.exists():
        print(f"  (fft-cnr sweep {npz_path} not found -- skipping Panel 4 fft-cnr)")
        return None
    d = np.load(npz_path, allow_pickle=True)
    true_cnr = d["true_cnr"]
    shapes = list(d["shapes"])

    # Plot the recovery ratio (estimated / true) rather than estimated vs true:
    # on the diagonal plot every shape collapses onto the 1:1 line and the
    # shape-dependent spread is invisible. The ratio centres on 1.0 and pulls the
    # interquartile bands apart, so the accuracy across shapes reads at a glance.
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    lo, hi = true_cnr.min() * 0.85, true_cnr.max() * 1.15
    ax.axhline(1.0, color="#666666", ls="--", lw=1.0, label="exact recovery")
    for i, shape in enumerate(shapes):
        gg = d[f"gg_{i}"]                       # (n_cnr, n_trials)
        ratio = gg / true_cnr[:, None]          # estimated / true, per trial
        med = np.nanmedian(ratio, axis=1)
        q1 = np.nanpercentile(ratio, 25, axis=1)
        q3 = np.nanpercentile(ratio, 75, axis=1)
        label, color = _FFTCNR_STYLE.get(shape, (shape, "#333333"))
        ax.fill_between(true_cnr, q1, q3, color=color, alpha=0.15)
        ax.semilogx(true_cnr, med, "o-", ms=4, lw=1.7, color=color, label=label)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"true contrast-to-noise  CNR")
    ax.set_ylabel(r"recovered CNR / true CNR")
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    _style(ax)
    fig.tight_layout()
    path = OUTPUT / "poster_panel4_fftcnr.png"
    save_poster(fig, path)
    plt.close(fig)
    return path


def figure_fftcnr_white_vs_shot(white_npz: Path, shot_npz: Path) -> Path | None:
    """Compare fft-cnr CNR recovery under white vs shot (Poisson) noise.

    Two panels share the CNR axis. Under white noise the recovery ratio sits at
    1.0 (the noise is spatially uniform, so the spectral noise estimate equals
    the peak noise). Under shot noise the variance follows the local signal and
    concentrates under the peak, so fft-cnr's window-averaged noise sits below
    the peak noise and the reported CNR overestimates the true peak SNR by a
    profile-dependent factor (~2.6-3.2x). The split is the subject of fft-cnr
    issue #1; here it is shown, not corrected.
    """
    if not (white_npz.exists() and shot_npz.exists()):
        print("  (white and/or shot fft-cnr sweep missing -- skipping comparison)")
        return None

    fig, (axW, axS) = plt.subplots(2, 1, figsize=(4.6, 5.4), sharex=True)
    for ax, npz, title in ((axW, white_npz, "white noise"),
                           (axS, shot_npz, "shot noise (Poisson)")):
        d = np.load(npz, allow_pickle=True)
        true_cnr = d["true_cnr"]
        ax.axhline(1.0, color="#666666", ls="--", lw=1.0)
        for i, shape in enumerate(list(d["shapes"])):
            ratio = d[f"gg_{i}"] / true_cnr[:, None]
            med = np.nanmedian(ratio, axis=1)
            q1 = np.nanpercentile(ratio, 25, axis=1)
            q3 = np.nanpercentile(ratio, 75, axis=1)
            label, color = _FFTCNR_STYLE.get(shape, (shape, "#333333"))
            ax.fill_between(true_cnr, q1, q3, color=color, alpha=0.15)
            ax.semilogx(true_cnr, med, "o-", ms=3.5, lw=1.6, color=color,
                        label=label)
        ax.set_ylabel("recovered / true CNR")
        ax.text(0.03, 0.93, title, transform=ax.transAxes, fontsize=9,
                va="top")
        _style(ax)
    axW.set_ylim(0.8, 1.2)
    axS.axhline(0.0, color="none")          # keep the y=1 reference in view
    axS.text(0.03, 0.07, "fft-cnr reads peak/(window-averaged noise),\n"
             "not the peak SNR -- a ~3x offset", transform=axS.transAxes,
             fontsize=7.5, va="bottom", color="#555")
    axS.legend(fontsize=7, loc="center right", frameon=False)
    axS.set_xlabel(r"true contrast-to-noise  CNR")
    fig.tight_layout()
    path = OUTPUT / "fftcnr_white_vs_shot.png"
    save_poster(fig, path)
    plt.close(fig)
    return path


def main() -> None:
    print("Producing poster figures...")
    print(f"  Panel 1 schematic (measurement):   {figure_schematic_measurement().name}")
    wall_fold_paths = figure_schematic_wall_fold()
    print(f"  Panel 2A schematic (wall + fold):  "
          f"{', '.join(p.name for p in wall_fold_paths)}")
    p3 = build_bias_curves()
    if p3:
        print(f"  Panel 3 (bias curves):             {p3.name}")
    print(f"  Panel 4 (precision floor):         {figure_crlb().name}")
    p4c = figure_fftcnr(_fftcnr_sweep())
    if p4c:
        print(f"  Panel 4 (fft-cnr CNR recovery):    {p4c.name}")
    p5 = build_corrected_bias()
    if p5:
        print(f"  Panel 5 (corrected bias):          {p5.name}")
    pf = figure_floor()
    if pf:
        print(f"  Panel 5 inset (per-profile floor): {pf.name}")
    print("Done.")


if __name__ == "__main__":
    main()
