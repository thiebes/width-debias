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
clips the moments. A summary line per fraction gives the extremes over all 601
frames.

The moments are taken on the bin centres ``-L/2 + (k + 1/2) L/n`` (bin width
L/n = 5 nm), with Sheppard's bin-width terms subtracted (h^2/12 from the second
moment; h^2/2 times the second moment, less 7 h^4/240, from the fourth).
(M_x + M_y) / V then compares the binned profiles with the walkers' own moment
and needs no assumption about the axes. M / (V/2) adds the assumption that
halving makes. The files' x axis is ``linspace(-L/2, L/2, n)``, whose spacing
L/(n - 1) is n/(n - 1) of the bin width. Every second moment on that axis, the
one the width fits use, is larger by (n/(n - 1))^2 = 1.0033 for n = 600, and
the fitted widths and the D of the noise-free fit carry that factor.

The shape gap per axis: the D of the bounded Gaussian fit of the noise-free
profile over the usable band at CNR0 15, divided by D_ref, as
``render_perframe_figure.py`` computes it for ``prof_x`` (same stride, floor,
fit, cap and power-law fit). Both axes use the x projection's usable band, so
the two D values differ only by the profile.

Sampling scale: for N independent walkers, the binned variance along one axis
has a relative SE of sqrt((2 + k) / N), with k the excess kurtosis. M_x / M_y
then carries sqrt(2 (2 + k) / N) and M / (V/2) carries sqrt((2 + k) / (2 N)).
(M_x + M_y) / V comes from the same walkers and has no sampling term. The
16 realizations of the halide disorder add scatter that the averaged files
cannot show, so these are lower bounds.

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

from br_halide_highn_validation import FLOOR_CNR, STRIDE, fit_each_row, subsample  # noqa: E402
from br_halide_validation import fit_msd_powerlaw  # noqa: E402
from br_highn_data import CASES, MAP_DIR, available, load_case  # noqa: E402

FRAME_TIMES_NS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
CNR0_SHAPE_GAP = 15.0         # the initial CNR of the per-frame table
EDGE_FRACTION = 0.1           # outermost tenth of the window on each side
OUT_TXT = Path(__file__).with_name("axis_check.txt")


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


def noise_free_d_ratios(frac: str, prof_x: np.ndarray, prof_y: np.ndarray) -> tuple[float, float, int]:
    """D of the noise-free fit of prof_x and prof_y over the usable band, each over D_ref."""
    case_x = load_case(frac)
    envelope = np.exp(-case_x.t / case_x.tau_ns)[:, None]
    if not np.array_equal(case_x.profiles, prof_x * envelope):
        sys.exit(f"x = {frac}: the decay envelope here differs from load_case's")
    case_y = replace(case_x, profiles=prof_y * envelope)
    case_x, case_y = subsample(case_x, STRIDE), subsample(case_y, STRIDE)
    noise = case_x.profiles[0].max() / CNR0_SHAPE_GAP
    usable = case_x.profiles.max(axis=1) / noise >= FLOOR_CNR
    d_ref, _ = fit_msd_powerlaw(case_x.t, case_x.var_true)
    d_x, _ = fit_msd_powerlaw(case_x.t, fit_each_row(case_x.x, case_x.profiles), mask=usable)
    d_y, _ = fit_msd_powerlaw(case_y.t, fit_each_row(case_y.x, case_y.profiles), mask=usable)
    return d_x / d_ref, d_y / d_ref, int(usable.sum())


def check_fraction(frac: str) -> tuple[list[str], dict[str, float]]:
    """Table rows and all-frame extremes for one Br fraction."""
    data = np.load(MAP_DIR / f"x_{frac}" / f"collapsed_1d.x_{frac}.npz")
    box, n_bins, n_walkers = float(data["box_nm"]), int(data["n_bins"]), int(data["n_walkers"])
    if not np.array_equal(data["x"], np.linspace(-box / 2.0, box / 2.0, n_bins)):
        sys.exit(f"x = {frac}: the x axis is not linspace(-L/2, L/2, n); the header's factor does not apply")
    width = box / n_bins
    centres = -box / 2.0 + (np.arange(n_bins) + 0.5) * width
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
            f"  {data['t'][i]:5.2f} {mx['centroid'][i]:+8.2f} {my['centroid'][i]:+8.2f}"
            f" {mx['m2'][i]:9.1f} {my['m2'][i]:9.1f} {half_v[i]:9.1f}"
            f" {ratio_xy[i]:7.4f} {sum_over_v[i]:7.4f} {x_over_half[i]:7.4f} {y_over_half[i]:7.4f}"
            f" {mx['kurtosis'][i]:+7.3f} {my['kurtosis'][i]:+7.3f}"
            f" {edge_x[i]:8.1e} {edge_y[i]:8.1e}")
    kurt = np.concatenate([mx["kurtosis"], my["kurtosis"]])
    extremes = {
        "ratio_xy": float(np.max(np.abs(ratio_xy - 1.0))),
        "sum_over_v": float(np.max(np.abs(sum_over_v - 1.0))),
        "x_over_half": float(np.max(np.abs(x_over_half - 1.0))),
        "y_over_half": float(np.max(np.abs(y_over_half - 1.0))),
        "kurt_min": float(kurt.min()), "kurt_max": float(kurt.max()),
        "edge": float(max(edge_x.max(), edge_y.max())),
        "sampling_xy": float(np.sqrt(2.0 * (2.0 + max(0.0, kurt.max())) / n_walkers)),
    }
    rows.append(
        f"  all {data['t'].size} frames: |M_x/M_y - 1| <= {extremes['ratio_xy']:.4f}"
        f" (walker sampling alone about {extremes['sampling_xy']:.4f});"
        f" |(M_x+M_y)/V - 1| <= {extremes['sum_over_v']:.4f};"
        f" |M_x/(V/2) - 1| <= {extremes['x_over_half']:.4f};"
        f" |M_y/(V/2) - 1| <= {extremes['y_over_half']:.4f};"
        f" excess kurtosis {extremes['kurt_min']:+.3f} to {extremes['kurt_max']:+.3f};"
        f" edge fraction <= {extremes['edge']:.1e}")
    d_x, d_y, n_usable = noise_free_d_ratios(frac, data["prof_x"], data["prof_y"])
    rows.append(f"  noise-free fit D / D_ref over the usable band ({n_usable} frames at CNR0"
                f" {CNR0_SHAPE_GAP:g}): prof_x {d_x:.3f}, prof_y {d_y:.3f}")
    return rows, extremes


def main() -> None:
    fracs = [frac for frac in CASES if available(frac)]
    if not fracs:
        sys.exit(f"No high-N collapsed maps found in {MAP_DIR} (WIDTH_DEBIAS_DATA_ROOT, or else data/). See DATASETS.md.")
    lines = [
        "=" * 76,
        "High-N Br-I: the x and y projections against the halved 2-D moment (axis_check.py)",
        "Moments on the bin centres (bin width 5 nm), about each profile's own centroid,",
        "with Sheppard's bin-width terms subtracted (2.08 nm^2 from M). V: the walkers' 2-D",
        "second moment (variance.x_<frac>.avg.dat); V/2 is the experiment's reference.",
        "On the files' x axis, which the fits use, every M is larger by (600/599)^2 = 1.0033.",
        "Edge: the larger fraction of the profile in the outermost tenth of the window.",
        "=" * 76,
        "  t(ns)   cen_x    cen_y      M_x       M_y       V/2   Mx/My (Mx+My)/V"
        " Mx/(V/2) My/(V/2)  kurt_x  kurt_y   edge_x   edge_y",
        "          (nm)     (nm)     (nm^2)    (nm^2)    (nm^2)",
    ]
    for frac in fracs:
        rows, _ = check_fraction(frac)
        lines += rows
    report = "\n".join(lines)
    print(report)
    OUT_TXT.write_text(report + "\n")


if __name__ == "__main__":
    main()
