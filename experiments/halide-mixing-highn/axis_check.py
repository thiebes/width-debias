"""Do the two projections of the high-N Br-I maps agree, and do they carry the halved 2-D moment?

``br_highn_data.load_case`` fits the x projection (``prof_x``) and takes the
carrier reference as half the walkers' 2-D second moment
(``variance.x_<frac>.avg.dat``). Halving assumes equal second moments along
the two axes, and comparing a fit of the profile with that reference assumes
the collapsed profile carries the walkers' moment. This script tests both and
writes ``axis_check.txt`` here. It changes nothing the experiment reads.

Per Br fraction, at seven frames, the table gives for ``prof_x`` and ``prof_y``:
the centroid; the binned second moment M about the profile's own centroid;
M_x / M_y; (M_x + M_y) / V, with V the 2-D moment; M / (V/2) per axis; the
excess kurtosis; and the larger of the two fractions of the profile in the
outermost tenth of the window on each side, which shows whether the window
clips the moments. Summary lines per fraction give the extremes over all 601
frames and the time of each, since most frames are not printed.

The moments are taken on the bin centres ``-L/2 + (k + 1/2) L/n`` (bin width
L/n), with Sheppard's bin-width terms subtracted (h^2/12 from the second
moment; h^2/2 times the binned second moment, less 7 h^4/240, from the
fourth). (M_x + M_y) / V then compares the binned profiles with the walkers'
own moment and needs no assumption about the axes; at frame 0 it is 1 only for
this choice, and on the files' axis or without the Sheppard term it would not
be. M / (V/2) adds the assumption that halving makes. The files' x axis is
``linspace(-L/2, L/2, n)``, whose spacing L/(n - 1) is n/(n - 1) of the bin
width. Every second moment on that axis, the one the width fits use, is larger
by (n/(n - 1))^2 = 1.0033 for n = 600, and the fitted widths and the D of the
noise-free fit carry that factor.

The shape gap per axis: the D of the bounded Gaussian fit of the noise-free
profile over the usable band at CNR0 15, divided by D_ref, as
``render_perframe_figure.py`` computes it for ``prof_x`` (same stride, floor,
fit, cap and power-law fit). Both axes use the x projection's usable band, so
the two D values differ only by the profile. The same power-law fit of the
binned moments M(t) over that band gives a D that carries no profile shape, so
the fitted D over the moment D is the shape's part, times the axis factor the
fit carries. The line also gives the failed fits and the largest fitted
variance as a fraction of the fit's cap.

Sampling scale: for N independent walkers, the binned variance along one axis
has a relative SE of sqrt((2 + k) / N), with k the excess kurtosis, so M_x / M_y
carries sqrt(2 (2 + k) / N). That is a lower bound. The centroids drift by many
times their walker-only SE, sqrt(M / N), so the 16 realizations of the halide
disorder set the scatter, and the averaged files cannot show how large it is:
the table does not test isotropy. (M_x + M_y) / V has no walker-sampling term
if V and the maps come from the same runs, yet it can still differ from 1 if
the two average the instances differently. If V averages each instance's
variance about its own centroid while the maps pool the instances, the profiles
also carry the spread of the instance centroids. That is a hypothesis; the
files do not record which.

Run from the repository root (needs the collapsed maps and variance files in
the data root; about a minute)::

    .venv/Scripts/python.exe experiments/halide-mixing-highn/axis_check.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "halide-mixing-brI-legacy"))
sys.path.insert(0, os.path.dirname(__file__))

from br_halide_highn_validation import (FLOOR_CNR, MAX_VAR, STRIDE, fit_each_row,  # noqa: E402
                                        subsample)
from br_halide_validation import fit_msd_powerlaw  # noqa: E402
from br_highn_data import CASES, MAP_DIR, available, load_case  # noqa: E402

FRAME_TIMES_NS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
CNR0_SHAPE_GAP = 15.0         # the initial CNR of the per-frame table
EDGE_FRACTION = 0.1           # outermost tenth of the window on each side
OUT_TXT = Path(__file__).with_name("axis_check.txt")


def _collapsed(frac: str):
    return np.load(MAP_DIR / f"x_{frac}" / f"collapsed_1d.x_{frac}.npz")


def binned_moments(centres: np.ndarray, width: float, profiles: np.ndarray) -> dict[str, np.ndarray]:
    """Centroid, Sheppard-corrected second moment and excess kurtosis of each row."""
    weights = profiles / profiles.sum(axis=1, keepdims=True)
    centroid = weights @ centres
    offset = centres[None, :] - centroid[:, None]
    m2_binned = (weights * offset ** 2).sum(axis=1)
    m4_binned = (weights * offset ** 4).sum(axis=1)
    m2 = m2_binned - width ** 2 / 12.0
    m4 = m4_binned - width ** 2 / 2.0 * m2_binned + 7.0 * width ** 4 / 240.0
    return {"centroid": centroid, "m2": m2, "kurtosis": m4 / m2 ** 2 - 3.0}


def edge_fraction(profiles: np.ndarray) -> np.ndarray:
    """Larger of the two fractions of each row in the outermost tenth of the window."""
    n_edge = int(round(EDGE_FRACTION * profiles.shape[1]))
    total = profiles.sum(axis=1)
    return np.maximum(profiles[:, :n_edge].sum(axis=1), profiles[:, -n_edge:].sum(axis=1)) / total


def _largest_departure(values: np.ndarray, t: np.ndarray) -> str:
    """The largest |value - 1| over the frames, and when it occurs."""
    i = int(np.argmax(np.abs(values - 1.0)))
    return f"{abs(values[i] - 1.0):.4f} at {t[i]:.3f} ns"


def band_d_ratios(frac: str, prof_x: np.ndarray, prof_y: np.ndarray, m2_x: np.ndarray,
                  m2_y: np.ndarray) -> str:
    """D / D_ref over the usable band, per axis, from the noise-free fit and from M(t)."""
    case_x = load_case(frac)
    envelope = np.exp(-case_x.t / case_x.tau_ns)[:, None]
    if not np.array_equal(case_x.profiles, prof_x * envelope):
        sys.exit(f"x = {frac}: the decay envelope here differs from load_case's")
    case_y = replace(case_x, profiles=prof_y * envelope)
    case_x, case_y = subsample(case_x, STRIDE), subsample(case_y, STRIDE)
    noise = case_x.profiles[0].max() / CNR0_SHAPE_GAP
    usable = case_x.profiles.max(axis=1) / noise >= FLOOR_CNR
    t = case_x.t
    d_ref, _ = fit_msd_powerlaw(t, case_x.var_true)
    fits = [fit_each_row(case.x, case.profiles) for case in (case_x, case_y)]
    d_fit = [fit_msd_powerlaw(t, fit, mask=usable)[0] / d_ref for fit in fits]
    d_moment = [fit_msd_powerlaw(t, m2[::STRIDE], mask=usable)[0] / d_ref for m2 in (m2_x, m2_y)]
    failed = sum(int(np.isnan(fit[usable]).sum()) for fit in fits)
    largest = max(float(np.nanmax(fit[usable])) for fit in fits) / MAX_VAR
    return (f"  usable band at CNR0 {CNR0_SHAPE_GAP:g} ({int(usable.sum())} frames, t <= "
            f"{t[usable].max():.2f} ns), D / D_ref: noise-free Gaussian fit prof_x {d_fit[0]:.3f},"
            f" prof_y {d_fit[1]:.3f}; binned moment prof_x {d_moment[0]:.3f}, prof_y"
            f" {d_moment[1]:.3f}; {failed} failed fits, largest fitted variance {largest:.2f} of"
            " the cap")


def check_fraction(frac: str) -> list[str]:
    """Table rows, all-frame extremes and the band's D ratios for one Br fraction."""
    data = _collapsed(frac)
    box, n_bins, n_walkers = float(data["box_nm"]), int(data["n_bins"]), int(data["n_walkers"])
    if not np.array_equal(data["x"], np.linspace(-box / 2.0, box / 2.0, n_bins)):
        sys.exit(f"x = {frac}: the x axis is not linspace(-L/2, L/2, n); the header's factor does not apply")
    width = box / n_bins
    centres = -box / 2.0 + (np.arange(n_bins) + 0.5) * width
    t = data["t"]
    half_v = load_case(frac).var_true
    mx = binned_moments(centres, width, data["prof_x"])
    my = binned_moments(centres, width, data["prof_y"])
    edge_x, edge_y = edge_fraction(data["prof_x"]), edge_fraction(data["prof_y"])
    ratio_xy = mx["m2"] / my["m2"]
    sum_over_v = (mx["m2"] + my["m2"]) / (2.0 * half_v)
    x_over_half, y_over_half = mx["m2"] / half_v, my["m2"] / half_v

    rows = [f"\nx = {frac} ({n_walkers} walkers)"]
    for t_ns in FRAME_TIMES_NS:
        i = int(round(t_ns / float(data["tsamp_ns"])))
        rows.append(
            f"  {t[i]:5.2f} {mx['centroid'][i]:+8.2f} {my['centroid'][i]:+8.2f}"
            f" {mx['m2'][i]:9.1f} {my['m2'][i]:9.1f} {half_v[i]:9.1f}"
            f" {ratio_xy[i]:7.4f} {sum_over_v[i]:7.4f} {x_over_half[i]:7.4f} {y_over_half[i]:7.4f}"
            f" {mx['kurtosis'][i]:+7.3f} {my['kurtosis'][i]:+7.3f}"
            f" {edge_x[i]:8.1e} {edge_y[i]:8.1e}")
    kurt = np.concatenate([mx["kurtosis"], my["kurtosis"]])
    sampling_xy = np.sqrt(2.0 * (2.0 + max(0.0, kurt.max())) / n_walkers)
    rows.append(
        f"  all {t.size} frames: |M_x/M_y - 1| <= {_largest_departure(ratio_xy, t)}"
        f" (walker sampling alone about {sampling_xy:.4f}, a lower bound);"
        f" |(M_x+M_y)/V - 1| <= {_largest_departure(sum_over_v, t)}")
    rows.append(
        f"  all {t.size} frames: |M_x/(V/2) - 1| <= {_largest_departure(x_over_half, t)};"
        f" |M_y/(V/2) - 1| <= {_largest_departure(y_over_half, t)};"
        f" excess kurtosis {kurt.min():+.3f} to {kurt.max():+.3f};"
        f" edge fraction <= {max(edge_x.max(), edge_y.max()):.1e}")
    rows.append(band_d_ratios(frac, data["prof_x"], data["prof_y"], mx["m2"], my["m2"]))
    return rows


def _header(fracs: list[str]) -> list[str]:
    """The table header, with the bin width and axis factor of the files read."""
    geometries = {(float(_collapsed(frac)["box_nm"]), int(_collapsed(frac)["n_bins"]))
                  for frac in fracs}
    if len(geometries) != 1:
        sys.exit(f"the Br fractions have different windows or bins: {sorted(geometries)}")
    box, n_bins = geometries.pop()
    width = box / n_bins
    return [
        "=" * 76,
        "High-N Br-I: the x and y projections against the halved 2-D moment (axis_check.py)",
        f"Moments on the bin centres (bin width {width:g} nm), about each profile's own centroid,",
        f"with Sheppard's bin-width terms subtracted ({width ** 2 / 12.0:.2f} nm^2 from M). V: the"
        " walkers' 2-D",
        "second moment (variance.x_<frac>.avg.dat); V/2 is the experiment's reference.",
        f"On the files' x axis, which the fits use, every M is larger by ({n_bins}/{n_bins - 1})^2"
        f" = {(n_bins / (n_bins - 1)) ** 2:.4f}.",
        "Edge: the larger fraction of the profile in the outermost tenth of the window.",
        "=" * 76,
        "  t(ns)   cen_x    cen_y      M_x       M_y       V/2   Mx/My (Mx+My)/V"
        " Mx/(V/2) My/(V/2)  kurt_x  kurt_y   edge_x   edge_y",
        "          (nm)     (nm)     (nm^2)    (nm^2)    (nm^2)",
    ]


def main() -> None:
    fracs = [frac for frac in CASES if available(frac)]
    if not fracs:
        sys.exit(f"No high-N collapsed maps found in {MAP_DIR} (WIDTH_DEBIAS_DATA_ROOT, or else data/). See DATASETS.md.")
    lines = _header(fracs)
    for frac in fracs:
        lines += check_fraction(frac)
    report = "\n".join(lines)
    print(report)
    OUT_TXT.write_text(report + "\n")


if __name__ == "__main__":
    main()
