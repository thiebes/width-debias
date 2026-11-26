"""Load the high-statistics Br-I halide-mixing maps (400k-walker average).

Marc Meléndez's original maps are histograms of only 20,000 walkers, so each
frame carries Poisson sampling noise that, read off the peak, sets an intrinsic
contrast-to-noise floor (~30) right inside the CNR range we want to test. To get
a clean ground-truth profile with no shape assumption, Marc's simulator was
re-run at high walker count and the independent instances averaged (the run
procedure is in ``experiments/halide-mixing-brI-legacy/rerun/README.md``). The
files are read from the ``halide_mixing_highn/`` folder of the data root
(``scripts/collaborator_data.py``). The 2-D averaged map was collapsed to 1-D on
the run machine (``collapsed_1d.x_<frac>.npz``: the ``prof_x``/``prof_y``
projections plus axes and metadata), so only that small npz and the variance file
are read here -- never the multi-GB map.

Geometry: 600 bins over a 3000 nm window (5 nm/bin), 601 frames at 5 ps to 3.0
ns, initial 1-D sigma ~25 nm. The wide box leaves the spot well inside even the
fastest fraction (last-frame +/-7.6 sigma at x=0.25), so the binned moment is not
clipped -- unlike the legacy 150 nm maps in :mod:`br_data`.

The carrier (transport) truth is the particle second moment V in
``variance.x_<frac>.avg.dat`` (the 2-D trace var_x + var_y, computed from the
walkers' unwrapped positions and so free of any binning or box truncation);
halving it gives the isotropic 1-D truth, the same convention as
:mod:`br_data`. That rests on two assumptions: the second moments along the
two axes are equal, so V/2 is the 1-D truth, and the collapsed 1-D profile
carries the walkers' moment along its axis. ``axis_check.py`` compares the
profiles with both (``axis_check.txt``). With M the binned second moment of
each projection about its own centroid, on the bin centres and with
Sheppard's bin-width term subtracted, M/(V/2) is within 0.0123 of 1 on each
axis over all 601 frames. (M_x + M_y)/V, which needs no assumption about the
axes, departs from 1 by up to 0.0108, growing with time, which walker sampling
cannot produce. The files' x axis, which the fits use, spaces
the bins by 600/599 of their 5 nm width, so every second moment on it is
larger by a factor 1.0033.

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
from collaborator_data import halide_mixing_highn_dir  # noqa: E402

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


MAP_DIR = halide_mixing_highn_dir()
NBINS = 600
NFRAMES = 601
TSAMP_NS = 0.005      # 5 ps per frame
L_NM = 3000.0         # diffmap window length (nm)
CASES = ("0.25", "0.5", "0.75")   # Br fraction x

# Initial 1-D spot width (nm); the simulator's ``sigma`` parameter, confirmed by
# the frame-0 binned moments, 622.1 to 625.4 nm^2 on the bin centres
# (``axis_check.txt``). Used to set the calibration window in sigma units.
SIGMA0_HINT_NM = 25.0

# Intrinsic peak contrast-to-noise of the 400k-walker average for this geometry
# (~sqrt(N_total * f_peak), f_peak ~ 0.006). Above this the map's own sampling
# noise competes with the imposed noise, so validation sweeps should cap CNR_0
# at or below it.
INTRINSIC_PEAK_CNR = 50.0


def _case_dir(frac: str) -> Path:
    return MAP_DIR / f"x_{frac}"


def available(frac: str) -> bool:
    """True when the collapsed profiles for this fraction are present."""
    return (_case_dir(frac) / f"collapsed_1d.x_{frac}.npz").exists()


def load_case(frac: str, tau_ns: float = DEFAULT_TAU_NS, n0: float = 1.0) -> BrCase:
    """Load one Br fraction from the collapsed npz, with the decay envelope applied.

    Uses the ``prof_x`` projection. The ratio of its binned second moment to
    that of ``prof_y`` is within 0.0143 of 1. Walker sampling alone gives about
    0.003, a lower bound, since the 16 realizations of the disorder add scatter
    the averaged files cannot show. Over the usable band at CNR0 15, the
    Gaussian fit of the noise-free ``prof_y`` gives a D higher than that of
    ``prof_x`` by 0.017, 0.034 and 0.024 of D_ref (x = 0.25, 0.5, 0.75); the
    binned moments account for 0.004, -0.001 and 0.012 of those, and these
    files cannot tell whether the difference comes from the transport or from
    the finite set of realizations (``axis_check.txt``). The truth is
    ``variance.avg.dat / 2``; both are keyed on the same frame times.
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
