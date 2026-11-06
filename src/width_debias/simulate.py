"""Generation of synthetic 1-D profile sequences for Monte Carlo validation.

A population of excited states is modelled as a 1-D Gaussian that broadens by
Fickian diffusion and decays by a single exponential lifetime.  White noise is
added at the level of the raw spatial profile -- the only level at which it can
be added correctly, because the fitting hierarchy reshapes the error
distribution downstream (Thiebes & Grumstrup, JCP 2024).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gaussian(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    """A 1-D Gaussian profile."""
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


@dataclass(frozen=True)
class Experiment:
    """Parameters of a single simulated diffusion measurement.

    Lengths are in arbitrary spatial units; the initial spot width ``sigma0``
    sets the scale.  The nanocrystal-solid regime targeted here is
    ``L_D / FWHM_0`` of order 0.05--0.2 (e.g. a 30--80 nm diffusion length
    against a ~400 nm diffraction-limited spot).
    """

    sigma0: float = 1.0          # initial Gaussian width (sets the length scale)
    D: float = 0.02              # diffusion coefficient (variance units / time)
    tau: float = 1.0             # exponential lifetime
    amplitude0: float = 1.0      # initial peak amplitude
    mean: float = 0.0            # profile centre
    half_window: float = 6.0     # spatial axis spans [-half_window, +half_window] * sigma0
    n_pixels: int = 241          # spatial samples
    n_frames: int = 12           # number of time points
    t_max: float = 3.0           # last time point (in units of tau)

    @property
    def x(self) -> np.ndarray:
        return np.linspace(-self.half_window * self.sigma0,
                            self.half_window * self.sigma0, self.n_pixels)

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.t_max * self.tau, self.n_frames)

    @property
    def true_variance(self) -> np.ndarray:
        """Ground-truth profile variance at each frame: sigma0^2 + 2 D t."""
        return self.sigma0 ** 2 + 2.0 * self.D * self.times

    @property
    def true_msd(self) -> np.ndarray:
        """Ground-truth mean-squared displacement: 2 D t."""
        return 2.0 * self.D * self.times

    @property
    def amplitude(self) -> np.ndarray:
        """Peak amplitude at each frame (exponential decay only).

        The simulated profiles hold this peak as they spread (see
        :func:`noise_free_series`), and the noise floor is constant, so the
        per-frame CNR falls with the decay alone.
        """
        return self.amplitude0 * np.exp(-self.times / self.tau)


@dataclass(frozen=True)
class AnomalousExperiment(Experiment):
    """A diffusion measurement with anomalous (non-Fickian) spreading.

    The variance grows as a power law, ``sigma^2(t) = sigma0^2 + 2 D t^alpha``,
    rather than the Fickian ``alpha = 1``. The exponent alpha is the disorder
    signature the field reads from the MSD curve: ``alpha < 1`` is subdiffusive
    (energetic disorder / trapping, as in disordered nanocrystal solids),
    ``alpha > 1`` superdiffusive. ``alpha = 1`` reproduces :class:`Experiment`
    exactly, which is the consistency check the validation leans on.

    Only the two truth properties change; the geometry (``x``, ``times``), the
    single-exponential decay (``amplitude``), and the CNR bookkeeping are
    inherited unchanged, so the whole Monte Carlo / fit / weighting pipeline
    runs on this bed with no other change. The noise bias lifts the
    low-CNR late frames more than the high-CNR early ones, which tilts the MSD
    curve toward a steeper apparent exponent -- so the bias contaminates alpha,
    not only D, and this bed is how the correction's recovery of alpha is
    scored.

    ``D`` carries units of ``sigma0^2 / tau^alpha`` here (so ``2 D t^alpha`` is
    a variance for ``t`` in units of ``tau``); it coincides with the Fickian
    ``sigma0^2 / tau`` only at ``alpha = 1``. To compare exponents at a matched
    diffusion length over the window, choose ``D`` per alpha to fix the
    end-point MSD ``2 D t_max^alpha`` (see
    ``experiments/subdiffusive-alpha/subdiffusive_alpha.py``).
    """

    alpha: float = 1.0           # MSD ~ t^alpha; 1 = Fickian, <1 subdiffusive

    @property
    def true_variance(self) -> np.ndarray:
        """Ground-truth profile variance: sigma0^2 + 2 D t^alpha."""
        return self.sigma0 ** 2 + 2.0 * self.D * self.times ** self.alpha

    @property
    def true_msd(self) -> np.ndarray:
        """Ground-truth mean-squared displacement: 2 D t^alpha."""
        return 2.0 * self.D * self.times ** self.alpha


@dataclass(frozen=True)
class MultiExpExperiment(Experiment):
    """A measurement whose population decays with a sum of exponentials.

    Real photoluminescence rarely decays as a single exponential: a fast
    surface/trap channel plus a slow bulk channel give a bi-exponential, and a
    distribution of trap depths gives a stretched-exponential-like curve. The
    peak amplitude here is a normalised sum of exponentials,

        amplitude(t) = amplitude0 * sum_i f_i exp(-t/tau_i) / sum_i f_i,

    so amplitude(0) = amplitude0 and CNR_0 keeps its meaning. The diffusion law
    is the inherited Fickian one; only the decay (hence the per-frame CNR
    trajectory) changes.

    The point of this bed is that the correction keys on the
    *per-frame CNR*, not on the decay law: the per-frame residual bias is the
    same function of per-frame CNR whichever decay produced it, so multi-
    exponential decay does not invalidate the correction. It does change *which*
    per-frame CNR appears at *which* frame, so any result indexed by the initial
    CNR_0 (rather than the per-frame CNR) is decay-law-specific.

    ``tau`` (inherited) still sets the time axis (``times`` span
    ``0..t_max*tau``); ``taus`` are the decay lifetimes in the same units. A
    single component ``fractions=(1.0,), taus=(tau,)`` reproduces
    :class:`Experiment` exactly.
    """

    fractions: tuple = (1.0,)    # component weights (need not sum to 1; normalised)
    taus: tuple = (1.0,)         # component lifetimes, same units as ``tau``

    def __post_init__(self) -> None:
        if len(self.fractions) != len(self.taus):
            raise ValueError("fractions and taus must have the same length")

    @property
    def amplitude(self) -> np.ndarray:
        """Peak amplitude at each frame: normalised sum of exponentials."""
        f = np.asarray(self.fractions, float)
        tau_i = np.asarray(self.taus, float)
        decay = (f[:, None] * np.exp(-self.times[None, :] / tau_i[:, None])).sum(0)
        return self.amplitude0 * decay / f.sum()


def noise_sigma_for_cnr0(exp: Experiment, cnr0: float) -> float:
    """Noise standard deviation that yields the requested initial CNR.

    CNR_0 is defined as the initial peak amplitude divided by the noise RMS.
    """
    return exp.amplitude0 / cnr0


def cnr_per_frame(exp: Experiment, noise_sigma: float) -> np.ndarray:
    """Per-frame contrast-to-noise ratio of an area-conserving profile.

    The peak falls with the decay and, as ``sigma0 / sigma(t)``, with diffusive
    spreading. The simulated profiles hold the peak at ``amplitude`` instead
    (see :func:`noise_free_series`), so this is not the CNR of a simulated
    frame; for that, use ``Ensemble.cnr_frame`` from
    :func:`~width_debias.montecarlo.run_ensemble`, which the calibration is
    keyed on.
    """
    peak = exp.amplitude * (exp.sigma0 / np.sqrt(exp.true_variance))
    return peak / noise_sigma


def noise_free_series(exp: Experiment) -> np.ndarray:
    """Stack of noise-free profiles, shape (n_frames, n_pixels)."""
    x = exp.x
    return np.array([
        gaussian(x, a, exp.mean, np.sqrt(v))
        for a, v in zip(exp.amplitude, exp.true_variance)
    ])


def noisy_series(exp: Experiment, noise_sigma: float,
                 rng: np.random.Generator) -> np.ndarray:
    """One noisy realisation of the full profile sequence."""
    clean = noise_free_series(exp)
    return clean + rng.normal(0.0, noise_sigma, size=clean.shape)
