"""Generation of synthetic 1-D profile sequences for Monte Carlo validation.

A population of excited states is modelled as a 1-D Gaussian that broadens by
Fickian diffusion and decays by a single exponential lifetime.  By default the
peak follows the decay alone, an approximation for small diffusion; with
``Experiment.spread_dims`` set it also falls as the profile spreads, so the
population changes only by the decay.  White noise is
added at the level of the raw spatial profile -- the only level at which it can
be added correctly, because the fitting hierarchy reshapes the error
distribution downstream (Thiebes & Grumstrup, JCP 2024).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfcx

_SQRT2 = np.sqrt(2.0)


def gaussian(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    """A 1-D Gaussian profile."""
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def unit_peak_height(width: np.ndarray, psf_hwhm: float = 0.0) -> np.ndarray:
    """Peak height of the unit-area profile of Gaussian width ``width``, times sqrt(2 pi).

    ``1 / width`` for the Gaussian, and for the generalized Gaussian at a fixed
    shape exponent; for a Gaussian of width ``width`` convolved with a
    Lorentzian of half-width ``psf_hwhm`` (the Voigt profile),
    ``erfcx(psf_hwhm / (width sqrt 2)) / width``. The constant factor cancels
    in the ratios the spreading peak takes.
    """
    width = np.asarray(width, float)
    return erfcx(psf_hwhm / (width * _SQRT2)) / width if psf_hwhm > 0 else 1.0 / width


@dataclass(frozen=True)
class Experiment:
    """Parameters of a single simulated diffusion measurement.

    Lengths are in arbitrary spatial units; the initial spot width ``sigma0``
    sets the scale.  The nanocrystal-solid regime targeted here is
    ``L_D / FWHM_0`` of order 0.05--0.2 (e.g. a 30--80 nm diffusion length
    against a ~400 nm diffraction-limited spot).

    By default (``spread_dims=None``) the peak follows the decay law alone: the
    simulated profiles hold it as they spread, so the integral of a profile
    grows with its width, an approximation for small diffusion.
    ``spread_dims`` selects a peak that also falls as the profile spreads, at a
    population that changes only by the decay::

        amplitude(t) = decay_amplitude(t) * h(sigma_x(t)) / h(sigma0)
                       * [h(sigma_y(t)) / h(sigma0)] ** (spread_dims - 1)

    with ``sigma_x(t) ** 2`` the ``true_variance`` and h(s) the peak height of
    the unit-area profile of width s (:func:`unit_peak_height`).
    ``spread_dims=1`` is a profile summed over the other axis, whose integral is
    conserved apart from the decay. ``spread_dims=2`` is a cut through the
    centre of a spot that also spreads along y, with
    ``sigma_y(t) ** 2 = sigma0 ** 2 + (sigma_x(t) ** 2 - sigma0 ** 2) * D_y / D``.
    h(s) is ``1 / s`` for the Gaussian and the generalized Gaussian at a fixed
    shape exponent. For the Voigt profile it depends on the Lorentzian
    half-width ``psf_hwhm``, which must equal the ``gamma`` of the
    :class:`~width_debias.models.VoigtModel` that draws the profiles, and 0 for
    every other model (:func:`~width_debias.montecarlo.run_ensemble` checks
    this). The factor is 1 at t = 0, so ``amplitude0`` and the initial CNR keep
    their meaning, and 1 at every t when D = 0, so a D = 0 calibration is the
    same under either peak.
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
    spread_dims: int | None = None   # None: decay alone; 1 summed profile, 2 centre cut
    D_y: float | None = None     # diffusion coefficient along y for spread_dims = 2; None means D
    psf_hwhm: float = 0.0        # Lorentzian half-width of a Voigt profile, for the spreading peak

    def __post_init__(self) -> None:
        if self.spread_dims not in (None, 1, 2):
            raise ValueError(f"spread_dims must be None, 1 or 2, not {self.spread_dims!r}")
        if self.psf_hwhm < 0:
            raise ValueError(f"psf_hwhm must be >= 0, not {self.psf_hwhm}")

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
    def decay_amplitude(self) -> np.ndarray:
        """Peak amplitude at each frame under the decay law alone: a single exponential.

        A subclass with another decay law overrides this property, not
        ``amplitude``, so that ``spread_dims`` still applies.
        """
        return self.amplitude0 * np.exp(-self.times / self.tau)

    @property
    def spread_factor(self) -> np.ndarray:
        """Factor by which spreading lowers the peak at each frame; ones without ``spread_dims``."""
        if self.spread_dims is None:
            return np.ones(self.n_frames)
        var_x = np.asarray(self.true_variance, float)
        h0 = unit_peak_height(np.array([self.sigma0]), self.psf_hwhm)[0]
        factor = unit_peak_height(np.sqrt(var_x), self.psf_hwhm) / h0
        if self.spread_dims == 2:
            d_y = self.D if self.D_y is None else self.D_y
            ratio = d_y / self.D if self.D > 0 else 0.0
            var_y = self.sigma0 ** 2 + (var_x - self.sigma0 ** 2) * ratio
            factor = factor * unit_peak_height(np.sqrt(var_y), self.psf_hwhm) / h0
        return factor

    @property
    def amplitude(self) -> np.ndarray:
        """Peak amplitude of the simulated profile at each frame.

        By default this is ``decay_amplitude``: the profiles hold the peak as
        they spread (see :func:`noise_free_series`), and the noise floor is
        constant, so the per-frame CNR falls with the decay alone. With
        ``spread_dims`` set it is ``decay_amplitude * spread_factor``, and the
        per-frame CNR falls with the spreading as well.
        """
        if self.spread_dims is None:
            return self.decay_amplitude
        return self.decay_amplitude * self.spread_factor


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
    single-exponential decay (``decay_amplitude``), and the CNR bookkeeping are
    inherited unchanged, so the whole Monte Carlo / fit / weighting pipeline
    runs on this bed with no other change. A spreading peak (``spread_dims``)
    follows this bed's ``true_variance``. The noise bias lifts the
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
    decay of the peak here is a normalised sum of exponentials,

        decay_amplitude(t) = amplitude0 * sum_i f_i exp(-t/tau_i) / sum_i f_i,

    so amplitude(0) = amplitude0 and CNR_0 keeps its meaning; a spreading peak
    (``spread_dims``) multiplies it by the inherited ``spread_factor``. The
    diffusion law is the inherited Fickian one; only the decay (hence the
    per-frame CNR trajectory) changes.

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
        super().__post_init__()
        if len(self.fractions) != len(self.taus):
            raise ValueError("fractions and taus must have the same length")

    @property
    def decay_amplitude(self) -> np.ndarray:
        """Peak amplitude at each frame under the decay alone: normalised sum of exponentials."""
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
    """Per-frame contrast-to-noise ratio of a profile whose peak falls as it spreads.

    With ``exp.spread_dims`` set, this is the peak of the simulated frames,
    ``exp.amplitude``, over the noise RMS; for white noise and a profile
    centred on a pixel it equals ``Ensemble.cnr_frame`` from
    :func:`~width_debias.montecarlo.run_ensemble`, which the calibration is
    keyed on. With the default peak (the decay alone), the simulated profiles
    hold the peak as they spread (see :func:`noise_free_series`), and this
    returns instead the CNR of a Gaussian profile summed over the other axis,
    whose peak falls also as ``sigma0 / sigma(t)``: the CNR of the frames that
    ``spread_dims=1`` would simulate, not of the frames ``exp`` simulates.
    """
    if exp.spread_dims is not None:
        return exp.amplitude / noise_sigma
    peak = exp.amplitude * (exp.sigma0 / np.sqrt(exp.true_variance))
    return peak / noise_sigma


def noise_free_series(exp: Experiment) -> np.ndarray:
    """Stack of noise-free Gaussian profiles, shape (n_frames, n_pixels).

    Each frame's peak is ``exp.amplitude``: by default the decay alone, so a
    profile holds its peak as it spreads; with ``exp.spread_dims`` set, a peak
    that also falls with the spreading.
    """
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
