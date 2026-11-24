"""Load Marc Melendez's Br-I halide-mixing diffusion maps (read only, from the data root).

Source: a 2D-perovskite exciton-diffusion Monte Carlo on a halide-mixing energy
landscape (Br fraction x), the regime of "halide mixing inhibits exciton
transport in two-dimensional perovskites" (the folder's Biblio). Halide disorder
suppresses transport far below the clean-layer diffusivity (D0 ~ 0.02 um^2/ns):
the effective D is ~0.003 um^2/ns and the spreading is subdiffusive
(apparent alpha ~ 0.88), so this is a heterogeneous-transport stress test on a
material system distinct from the Mn maps and the disordered nanocrystal solids.

For each Br fraction Marc supplied (in ``BrI_x_<frac>/``):

  * ``diffmap.x_<frac>.dat`` -- 21 frames of 250x250 exciton-density maps, stacked
    as consecutive 250-row blocks, over 150x150 nm^2 (0.6 nm/bin), sampled every
    5 ps (tsamp = 0.005 ns) to tmax = 0.1 ns. Each frame is normalised to unit
    integral (the simulation tracks position only -- no recombination).
  * ``variance.x_<frac>.dat`` -- per frame, ``(time_ns, variance)``. The variance
    is the *particle* (carrier) second moment, summed over both axes
    (var_x + var_y). Transport is isotropic here (var_x = var_y to <1%), so the
    1-D carrier variance is half of it. This is the carrier-transport truth, and
    being particle-based it is free of the box-truncation that clips the binned
    map's own moment (below).

Two modelling points the validation depends on:

  * **Imposed decay.** The maps are per-frame normalised, so a real transient-PL
    measurement's decaying integral is absent. We multiply each frame by
    ``N0 * exp(-t/tau)`` to set the per-frame contrast-to-noise. Over the 0.1 ns
    window a 2D-perovskite lifetime (tau ~ 1 ns) decays only ~10%, so the CNR
    falloff is dominated by diffusive spreading (the unit-integral 1-D peak drops
    as 1/sqrt(variance)); the result is insensitive to the exact tau.
  * **Narrow box.** The profile reaches only ~+/-2.3 sigma of the +/-75 nm window
    by the last frame, so the binned second moment clips ~15% of the variance at
    late times and reads the apparent alpha / D_eff low. A bounded width fit
    extrapolates the tails and recovers the particle variance (the carrier
    truth); the moment estimator does not. That is why the truth here is the
    particle ``variance.dat``, not the map's own moment.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from collaborator_data import halide_mixing_dir  # noqa: E402

MAP_DIR = halide_mixing_dir()
L_NM = 150.0          # box length (nm); the parameters file labels it "um" in error
NBINS = 250
NFRAMES = 21
TSAMP_NS = 0.005      # 5 ps per frame
CASES = ("0.25", "0.5", "0.75")   # Br fraction x

# Representative 2D-perovskite exciton lifetime (ns). Over the 0.1 ns window the
# decay is weak, so the validation is insensitive to this value.
DEFAULT_TAU_NS = 1.0


@dataclass
class BrCase:
    """One Br-fraction map: geometry, decay-weighted 1-D profiles, carrier truth."""
    name: str
    x: np.ndarray          # spatial axis (centred), nm
    t: np.ndarray          # time axis, ns
    profiles: np.ndarray   # (n_frames, n_x) 1-D profiles after the decay envelope
    var_true: np.ndarray   # (n_frames,) carrier 1-D variance, nm^2 (particle, un-clipped)
    tau_ns: float


def load_case(frac: str, tau_ns: float = DEFAULT_TAU_NS, n0: float = 1.0) -> BrCase:
    """Load one Br fraction (``"0.25"`` / ``"0.5"`` / ``"0.75"``) with decay applied."""
    folder = MAP_DIR / f"BrI_x_{frac}"
    cube = np.loadtxt(folder / f"diffmap.x_{frac}.dat").reshape(NFRAMES, NBINS, NBINS)
    varf = np.loadtxt(folder / f"variance.x_{frac}.dat")
    x = np.linspace(-L_NM / 2.0, L_NM / 2.0, NBINS)
    t = varf[:, 0]
    prof = cube.sum(axis=1)                              # project one axis -> (t, x)
    prof = prof * (n0 * np.exp(-t / tau_ns))[:, None]    # impose decay envelope
    var_true = varf[:, 1] / 2.0                          # 2-D trace -> isotropic 1-D
    return BrCase(name=f"Br x={frac}", x=x, t=t, profiles=prof,
                  var_true=var_true, tau_ns=tau_ns)


def load_all(tau_ns: float = DEFAULT_TAU_NS) -> list[BrCase]:
    """Load every Br fraction present in the data root, in increasing-x order."""
    return [load_case(frac, tau_ns=tau_ns) for frac in CASES
            if (MAP_DIR / f"BrI_x_{frac}" / f"diffmap.x_{frac}.dat").exists()]
