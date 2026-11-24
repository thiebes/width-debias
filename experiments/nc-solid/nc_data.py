"""Load Marc Melendez's disordered-NC-solid diffusion maps (read only, from the data root).

Model: the exciton-hopping simulation of Solari et al., "Exciton Transport in
Disordered Perovskite Nanocrystal Solids" (CH3NH3PbBr3 NC films, C8/C12/C16
ligands). Marc Melendez made these diffmap files for this work; they are not
the runs behind the paper's Fig. 5, which used only variance curves, and its
deposit (doi:10.21950/A0AB3H) has no maps. The small map's 2-D variance grows
with exponent 0.85, like the deposited C16 run (0.84), but about 25% faster, so
it is a separate run of the same model. Small / medium / large are the C16 / C12
/ C8 structures (mean NC radius 3.7 / 5.3 / 7.4 nm). The files are read from
the ``simulated_map/`` folder of the data root (``scripts/data_root.py``) and
never written.

Each `diffmap_<tag>.dat` is 20 frames of 250x250 exciton-concentration maps over
a 1500x1500 nm^2 area (6 nm/pixel), 1 ns/frame. The maps track exciton position
only -- there is NO recombination, and the concentration is per-frame
normalized. To model a real transient-PL measurement we impose the population
decay envelope `N0 * exp(-t/tau)` (the paper's thin-film exciton lifetime is
tau ~ 6-8 ns overall, with per-sample TPLM tau_1/e ~ 6 / 14 / 18 ns for
C16 / C12 / C8). The decay is a per-frame scalar, so it sets the amplitude --
hence the per-frame contrast-to-noise -- without changing the spatial variance.

The pipeline fits a 1-D width, so we project the 2-D map to 1-D by summing over
y. The paper itself fits each time slice to a Voigt profile; the spatial second
moment we report here is the transport ground truth (sigma0 ~ 200 nm spot,
MSD growth ~2000-5000 nm^2 over the run, L_D/FWHM ~ 0.14 -- the headline
nanocrystal-solid regime).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from data_root import simulated_map_dir  # noqa: E402

MAP_DIR = simulated_map_dir()
L_UM = 1.5            # box length (um); 1500 nm
NBINS = 250
NFRAMES = 20
DT_NS = 1.0
CASES = ("small", "medium", "large")

# Representative thin-film exciton lifetime (ns), from Solari et al. A single
# value validates the mechanics; per-sample tau (C16/C12/C8 ~ 6/14/18 ns for
# small/medium/large) is the refinement.
DEFAULT_TAU_NS = 7.0


@dataclass
class NCCase:
    """One disordered-NC-solid map: geometry, decay-weighted 1-D profiles, truth."""
    name: str
    x: np.ndarray          # spatial axis (centred), um
    t: np.ndarray          # time axis, ns
    profiles: np.ndarray   # (n_frames, n_x) 1-D profiles after the decay envelope
    var_true: np.ndarray   # (n_frames,) spatial variance, um^2
    tau_ns: float


def _denoise_row(x: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, float]:
    """Gaussian fit to one frame, taken as the bed's truth for that frame.

    We do not claim the real profile is Gaussian: fitting a Gaussian to each frame
    and treating it as the truth defines a realistic-width Gaussian test bed (its
    width and amplitude are the maps', its ground truth is Gaussian by
    construction). It also sets aside the per-bin Poisson sampling noise of the
    ~50,000-walker histograms, which read through ``max`` would inflate the
    per-frame contrast-to-noise (the peak appears to *rise* with spreading instead
    of falling as 1/sigma). The reconstruction keeps the row's integral, so the
    population is conserved and the peak falls as 1/sigma. Returns
    ``(gaussian_row, sigma2)``. (The name ``denoise`` is kept for the call sites;
    the operation is "fit a Gaussian and take it as truth", not shape recovery.)
    """
    tot = float(row.sum())
    p = row / tot
    c0 = float((p * x).sum())
    s0 = float(np.sqrt(((x - c0) ** 2 * p).sum()))

    def model(xx, c, s):
        e = np.exp(-0.5 * ((xx - c) / s) ** 2)
        return e / e.sum()

    try:
        (c, s), _ = curve_fit(model, x, p, p0=[c0, s0], maxfev=20000)
    except (RuntimeError, ValueError):
        c, s = c0, s0
    return model(x, c, s) * tot, float(s) ** 2


def load_case(tag: str, tau_ns: float = DEFAULT_TAU_NS, n0: float = 1.0,
              denoise: bool = False) -> NCCase:
    """Load one map by tag (``small`` / ``medium`` / ``large``), with decay applied.

    With ``denoise=True`` each frame's projection is replaced by its smooth
    Gaussian reconstruction (see :func:`_denoise_row`) before the decay envelope,
    so the clean signal is the continuum density and the only noise in the Monte
    Carlo is the measurement noise added downstream. ``var_true`` is then the
    de-noised Gaussian width; otherwise it is the raw projection's second moment.
    """
    cube = np.loadtxt(MAP_DIR / f"diffmap_{tag}.dat").reshape(NFRAMES, NBINS, NBINS)
    x = np.linspace(-L_UM / 2.0, L_UM / 2.0, NBINS)
    t = np.arange(NFRAMES) * DT_NS
    prof = cube.sum(axis=1)                              # project y -> (t, x)
    if denoise:
        smooth = np.empty_like(prof)
        var = np.empty(NFRAMES)
        for f in range(NFRAMES):
            smooth[f], var[f] = _denoise_row(x, prof[f])
        prof = smooth
    prof = prof * (n0 * np.exp(-t / tau_ns))[:, None]    # impose decay envelope
    if not denoise:
        tot = prof.sum(axis=1)
        xbar = (prof * x).sum(1) / tot
        var = (prof * (x[None, :] - xbar[:, None]) ** 2).sum(1) / tot
    return NCCase(name=tag, x=x, t=t, profiles=prof, var_true=var, tau_ns=tau_ns)


def load_all(tau_ns: float = DEFAULT_TAU_NS, denoise: bool = False) -> list[NCCase]:
    """Load every map present in the data root, in small/medium/large order."""
    return [load_case(tag, tau_ns=tau_ns, denoise=denoise) for tag in CASES
            if (MAP_DIR / f"diffmap_{tag}.dat").exists()]
