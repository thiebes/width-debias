"""Load Marc Melendez's diffusion maps and variance curves (provided 2026-06).

For each Mn-doping case Marc supplied two files in ``data/``:

  * ``diffmap_<case>.dat`` -- an ``(n_t, n_x)`` matrix of noiseless PL intensity
    (250 time frames x 250 spatial bins).
  * ``PL_var_<case>.dat`` -- a reduced table whose third column (index 2) is the
    spatial variance sigma^2(t), i.e. the second central moment of the PL
    profile. Confirmed to match each diffmap's own second moment to machine
    precision. It is emission-weighted (each population counts by its emitted
    intensity), so it is the reference for the recorded profile's width, not
    the carrier second moment.

Geometry follows the Magdaleno et al. 2025 (0.51% Mn) reaction-diffusion model:
``L = 5 um`` over ``nbins = 250`` (dx = 0.0201 um), ``tmax = 25 ns`` over
``dt = 0.1 ns``, initial width ``sigma0 = 0.1 um``. The four cases span weak to
strong trapping (undoped -> 4.04% Mn); higher doping suppresses spreading, so
the final variance falls from ~0.138 um^2 (undoped) to ~0.023 um^2 (4.04%).

The spatial axis is centred at zero; the per-frame Gaussian fit finds the centre
itself, so the origin convention does not affect the recovered variance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"

L = 5.0          # box length [um]
NBINS = 250      # spatial bins
TMAX = 25.0      # simulation time [ns]
DT = 0.1         # sampling step [ns]
SIGMA0 = 0.1     # initial width [um]

# Display label -> file-name stem.
CASES = {
    "undoped": "undoped",
    "0.51% Mn": "0.51_Mn_doped",
    "0.81% Mn": "0.81_Mn_doped",
    "4.04% Mn": "4.04_Mn_doped",
}


@dataclass
class MarcCase:
    """One doping case: geometry, noiseless map, and PL-moment variance."""
    name: str
    x: np.ndarray          # spatial axis (centred), um
    t: np.ndarray          # time axis, ns
    diffmap: np.ndarray    # (n_t, n_x) noiseless PL intensity
    var_true: np.ndarray   # (n_t,) spatial variance, um^2


def load_case(label: str) -> MarcCase:
    """Load a case by display label (e.g. ``"0.51% Mn"``) or file stem."""
    stem = CASES.get(label, label)
    diffmap = np.loadtxt(DATA / f"diffmap_{stem}.dat")
    pv = np.loadtxt(DATA / f"PL_var_{stem}.dat")
    n_x = diffmap.shape[1]
    x = np.linspace(-L / 2.0, L / 2.0, n_x)
    return MarcCase(name=label, x=x, t=pv[:, 0], diffmap=diffmap,
                    var_true=pv[:, 2])


def load_all() -> list[MarcCase]:
    """Load every doping case in increasing-doping order."""
    return [load_case(label) for label in CASES]
