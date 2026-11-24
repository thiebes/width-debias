"""Load the high-statistics Br-I halide-mixing maps (400k-walker average).

Marc Meléndez's original maps are histograms of only 20,000 walkers, so each
frame carries Poisson sampling noise that, read off the peak, sets an intrinsic
contrast-to-noise floor (~30) right inside the CNR range we want to test. To get
a clean ground-truth profile with no shape assumption, Marc's simulator was
re-run at high walker count and the independent instances averaged (the run
procedure is in ``experiments/halide-mixing-brI-legacy/rerun/README.md``). The
files are read from the ``halide_mixing_highn/`` folder of the data root
(``scripts/data_root.py``). The 2-D averaged map was
collapsed to 1-D on the run machine (``collapsed_1d.x_<frac>.npz``: the
``prof_x``/``prof_y`` projections plus axes and metadata), so only the small npz
is read here -- never the multi-GB map.

Geometry: 600 bins over a 3000 nm window (5 nm/bin), 601 frames at 5 ps to 3.0
ns, initial 1-D sigma ~25 nm. The wide box leaves the spot well inside even the
fastest fraction (last-frame +/-7.6 sigma at x=0.25), so the binned moment is not
clipped -- unlike the legacy 150 nm maps in :mod:`br_data`.

The carrier (transport) truth is the particle second moment in
``variance.x_<frac>.avg.dat`` (the 2-D trace var_x + var_y, computed from the
walkers' unwrapped positions and so free of any binning or box truncation);
halving it gives the isotropic 1-D truth, the same convention as
:mod:`br_data`. The collapsed profiles' binned 1-D moment agrees with this to
~1% (a small Sheppard bin-width offset), confirming the projection.

Two modelling points the validation depends on (shared with :mod:`br_data`):

  * **Imposed decay.** The maps are per-frame normalised, so a real transient-PL
    measurement's decaying integral is absent. Each frame is multiplied by
    ``N0 * exp(-t/tau)`` to set the per-frame contrast-to-noise. Over the 3 ns
    window a 2D-perovskite lifetime (tau ~ 1 ns) decays ~20x, which -- together
    with the diffusive spreading -- drives the late-frame per-frame CNR down
    through ~1, the regime the correction is built for. (The legacy 0.1 ns
    window never reached it.)
  * **Clean truth.** With 400,000 walkers the map's own sampling noise is small
    (intrinsic peak CNR ~50 for this geometry), so the imposed measurement noise
    dominates across the CNR sweep up to ~50; above that the map's own noise
    starts to limit, which is why the sweep is capped there.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from data_root import halide_mixing_highn_dir  # noqa: E402

# Representative 2D-perovskite exciton lifetime (ns); over the 3 ns window the
# decay drives the per-frame contrast-to-noise falloff.
DEFAULT_TAU_NS = 1.0


@dataclass
class BrCase:
    """One Br-fraction map: geometry, decay-weighted 1-D profiles, carrier truth.

    Mirrors the dataclass of the legacy ``experiments/halide-mixing-brI-legacy/br_data``
    loader (identical field names), so the validation pipeline consumes either
    interchangeably; defined here to keep this study folder self-contained.
    """
    name: str
    x: np.ndarray          # spatial axis (centred), nm
    t: np.ndarray          # time axis, ns
    profiles: np.ndarray   # (n_frames, n_x) 1-D profiles after the decay envelope
    var_true: np.ndarray   # (n_frames,) carrier 1-D variance, nm^2 (particle, un-clipped)
    tau_ns: float


HIGHN = halide_mixing_highn_dir()   # one x_<frac>/ folder per Br fraction
NBINS = 600
NFRAMES = 601
TSAMP_NS = 0.005      # 5 ps per frame
L_NM = 3000.0         # diffmap window length (nm)
CASES = ("0.25", "0.5", "0.75")   # Br fraction x

# Initial 1-D spot width (nm); the simulator's ``sigma`` parameter, confirmed by
# the frame-0 binned moment (~628 nm^2 -> 25.06 nm). Used to set the calibration
# window in sigma units.
SIGMA0_HINT_NM = 25.0

# Intrinsic peak contrast-to-noise of the 400k-walker average for this geometry
# (~sqrt(N_total * f_peak), f_peak ~ 0.006). Above this the map's own sampling
# noise competes with the imposed noise, so validation sweeps should cap CNR_0
# at or below it.
INTRINSIC_PEAK_CNR = 50.0


def _case_dir(frac: str) -> Path:
    return HIGHN / f"x_{frac}"


def available(frac: str) -> bool:
    """True when the collapsed profiles for this fraction are present."""
    return (_case_dir(frac) / f"collapsed_1d.x_{frac}.npz").exists()


def load_case(frac: str, tau_ns: float = DEFAULT_TAU_NS, n0: float = 1.0) -> BrCase:
    """Load one Br fraction from the collapsed npz, with the decay envelope applied.

    Uses the ``prof_x`` projection (transport is isotropic to <1%, so the choice
    of axis is immaterial). The truth is ``variance.avg.dat / 2``; both are
    keyed on the same frame times.
    """
    folder = _case_dir(frac)
    d = np.load(folder / f"collapsed_1d.x_{frac}.npz")
    x = d["x"]
    t = d["t"]
    prof = d["prof_x"] * (n0 * np.exp(-t / tau_ns))[:, None]   # impose decay envelope
    varf = np.loadtxt(folder / f"variance.x_{frac}.avg.dat")
    var_true = varf[:, 1] / 2.0                                # 2-D trace -> isotropic 1-D
    return BrCase(name=f"Br x={frac}", x=x, t=t, profiles=prof,
                  var_true=var_true, tau_ns=tau_ns)


def load_all(tau_ns: float = DEFAULT_TAU_NS) -> list[BrCase]:
    """Load every Br fraction whose collapse is present, in increasing-x order."""
    return [load_case(frac, tau_ns=tau_ns) for frac in CASES if available(frac)]
