"""Profile models: how each frame's spatial profile is shaped and fitted.

The truncation-at-zero bias is not specific to a Gaussian fit -- it afflicts
*any* non-negative width parameter. This module makes the profile model
pluggable so the boundary diagnostic, the Monte Carlo, and the weighting can be
run for each of:

* :class:`GaussianModel` -- the baseline; a pure Gaussian profile.
* :class:`VoigtModel` -- a diffusing Gaussian convolved with a *fixed*
  Lorentzian instrument PSF. This is the model Prins' group uses (Solari et
  al. 2026 fit Voigt profiles). The diffusive width is the Gaussian component
  sigma_G; the Lorentzian HWHM gamma is a fixed instrument parameter.
* :class:`GeneralizedGaussianModel` -- a profile with a tunable shape exponent
  p (p = 2 Gaussian, p < 2 heavy-tailed, p > 2 flat-topped), matching the
  generalized-Gaussian family fitted by the ``fft-cnr`` package. This is the
  handle for non-Gaussian transport (e.g. subdiffusive nanocrystal solids).

Every model exposes the same two-method interface used by ``montecarlo``:

* ``noise_free_series(exp)`` -- the clean profile stack for an Experiment.
* ``fit_variance(x, y)``     -- fit one noisy profile, return its diffusive
  variance as a :class:`~width_debias.fit.FrameFit`.

For every model the *diffusive variance* (the quantity that grows as
sigma^2(t) = sigma_0^2 + 2 D t and enters MSD(t) = variance(t) - variance(0))
is the same physical quantity; only the profile shape and the estimator differ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma as _gamma
from scipy.special import voigt_profile as _voigt

from .fit import FrameFit, _unusable_frame, fit_variance as _fit_gaussian_variance
from .simulate import Experiment, gaussian, noise_free_series as _gaussian_series


# --------------------------------------------------------------------------
# Gaussian -- the baseline model (delegates to width_debias.fit)
# --------------------------------------------------------------------------
class GaussianModel:
    """Pure Gaussian profile; the baseline model.

    ``max_var`` is the optional upper bound on the fitted variance, forwarded to
    :func:`width_debias.fit.fit_variance`. The default (``None``) uses the
    ``span^2`` bound (the full field of view); pass a physical cap ``(G * sigma0) ** 2`` when
    the output window is far wider than the spot, so the low-CNR fit cannot
    wander out to the window scale. It must match between calibration and
    application (see :func:`width_debias.fit.fit_variance`).
    """

    name = "gaussian"

    def __init__(self, max_var: float | None = None):
        self.max_var = max_var

    def noise_free_series(self, exp: Experiment) -> np.ndarray:
        return _gaussian_series(exp)

    def fit_variance(self, x: np.ndarray, y: np.ndarray) -> FrameFit:
        return _fit_gaussian_variance(x, y, max_var=self.max_var)


# --------------------------------------------------------------------------
# Voigt -- diffusing Gaussian convolved with a fixed Lorentzian PSF
# --------------------------------------------------------------------------
def _voigt_peaknorm(x: np.ndarray, amp: float, mean: float,
                     sigma_g: float, gamma: float) -> np.ndarray:
    """Voigt profile scaled so that ``amp`` is the peak height."""
    prof = _voigt(x - mean, sigma_g, gamma)
    peak = _voigt(np.array([0.0]), sigma_g, gamma)[0]
    return amp * prof / peak


@dataclass
class VoigtModel:
    """Diffusing Gaussian (variance sigma_G^2) convolved with a fixed
    Lorentzian PSF of half-width-at-half-maximum ``gamma``.

    The Voigt fit deconvolves the fixed instrument Lorentzian from the
    diffusing Gaussian, so the recovered, truncation-prone width is the
    Gaussian component sigma_G; MSD(t) = sigma_G^2(t) - sigma_G^2(0).
    """

    gamma: float = 0.3          # Lorentzian HWHM, in sigma_0 units (fixed PSF)
    fit_gamma: bool = False     # if True, gamma is a free fit parameter too
    name: str = "voigt"

    def noise_free_series(self, exp: Experiment) -> np.ndarray:
        x = exp.x
        return np.array([
            _voigt_peaknorm(x, a, exp.mean, np.sqrt(v), self.gamma)
            for a, v in zip(exp.amplitude, exp.true_variance)
        ])

    def fit_variance(self, x: np.ndarray, y: np.ndarray) -> FrameFit:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if (unusable := _unusable_frame(x, y)) is not None:
            return unusable
        span = float(x[-1] - x[0])
        sg_min = span / x.size           # ~ one pixel
        sg_max = 0.45 * span
        sg0 = max(min(span / 8.0, sg_max), 2 * sg_min)
        amp0 = float(np.max(np.abs(y))) or 1.0
        mean0 = float(x[int(np.argmax(y))])

        if self.fit_gamma:
            def model(xx, amp, mean, sg, gam):
                return _voigt_peaknorm(xx, amp, mean, sg, gam)
            p0 = [amp0, mean0, sg0, self.gamma]
            lo = [0.0, x[0], sg_min, sg_min]
            hi = [2 * amp0, x[-1], sg_max, sg_max]
        else:
            def model(xx, amp, mean, sg):
                return _voigt_peaknorm(xx, amp, mean, sg, self.gamma)
            p0 = [amp0, mean0, sg0]
            lo = [0.0, x[0], sg_min]
            hi = [2 * amp0, x[-1], sg_max]

        try:
            popt, _ = curve_fit(model, x, y, p0=p0, bounds=(lo, hi), maxfev=10000)
            sigma_g = float(popt[2])
            amp = float(popt[0])
            ok = True
        except (RuntimeError, ValueError):
            sigma_g, amp, ok = sg0, amp0, False

        at_wall = sigma_g <= 2 * sg_min
        return FrameFit(variance=sigma_g ** 2, amplitude=amp,
                        at_wall=bool(at_wall), ok=ok)


# --------------------------------------------------------------------------
# Generalized Gaussian -- tunable shape exponent (matches fft-cnr's family)
# --------------------------------------------------------------------------
def _genstd_var_factor(p: float) -> float:
    """Variance of exp(-0.5 |z|^p) as a density, in units of the scale s^2.

    For a profile A * exp(-0.5 |(x-c)/s|^p), the second central moment is
    s^2 * 2^(2/p) * Gamma(3/p) / Gamma(1/p).  Equals 1 at p = 2 (Gaussian).
    """
    return float(2.0 ** (2.0 / p) * _gamma(3.0 / p) / _gamma(1.0 / p))


def _gen_gaussian(x: np.ndarray, amp: float, mean: float,
                  scale: float, shape: float) -> np.ndarray:
    """Generalized-Gaussian profile, peak-normalised to ``amp``."""
    z = np.abs((x - mean) / scale)
    return amp * np.exp(-0.5 * z ** shape)


@dataclass
class GeneralizedGaussianModel:
    """Profile ``A exp(-0.5 |(x-c)/s|^p)`` with shape exponent ``p``.

    p = 2 recovers the Gaussian; p < 2 is heavy-tailed (sharp peak, fat
    tails); p > 2 is flat-topped.  This is the profile family fitted by
    ``fft-cnr`` -- the handle for non-Gaussian transport.  The diffusive
    variance is s^2 * factor(p); the fit recovers the scale s with p held
    fixed at the known transport shape.
    """

    shape: float = 2.0          # p
    name: str = "generalized_gaussian"

    @property
    def var_factor(self) -> float:
        return _genstd_var_factor(self.shape)

    def _scale_for_variance(self, variance: np.ndarray) -> np.ndarray:
        return np.sqrt(np.asarray(variance, float) / self.var_factor)

    def noise_free_series(self, exp: Experiment) -> np.ndarray:
        x = exp.x
        scales = self._scale_for_variance(exp.true_variance)
        return np.array([
            _gen_gaussian(x, a, exp.mean, s, self.shape)
            for a, s in zip(exp.amplitude, scales)
        ])

    def fit_variance(self, x: np.ndarray, y: np.ndarray) -> FrameFit:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if (unusable := _unusable_frame(x, y)) is not None:
            return unusable
        span = float(x[-1] - x[0])
        s_min = span / x.size
        s_max = 0.45 * span
        s0 = max(min(span / 8.0, s_max), 2 * s_min)
        amp0 = float(np.max(np.abs(y))) or 1.0
        mean0 = float(x[int(np.argmax(y))])

        def model(xx, amp, mean, scale):
            return _gen_gaussian(xx, amp, mean, scale, self.shape)

        try:
            popt, _ = curve_fit(
                model, x, y, p0=[amp0, mean0, s0],
                bounds=([0.0, x[0], s_min], [2 * amp0, x[-1], s_max]),
                maxfev=10000,
            )
            scale = float(popt[2])
            amp = float(popt[0])
            ok = True
        except (RuntimeError, ValueError):
            scale, amp, ok = s0, amp0, False

        at_wall = scale <= 2 * s_min
        variance = scale ** 2 * self.var_factor
        return FrameFit(variance=variance, amplitude=amp,
                        at_wall=bool(at_wall), ok=ok)


# --------------------------------------------------------------------------
# Moment estimator -- the non-truncated alternative ("remove the wall")
# --------------------------------------------------------------------------
@dataclass
class MomentModel:
    """Non-truncated, moment-based width estimator.

    Instead of a bounded curve fit, the variance is the discrete second
    central moment of the (background-subtracted) profile. That estimator is
    linear in the data: it can return a *negative* variance and does not fold
    at zero. Its sampling distribution is therefore approximately symmetric
    and, to leading order, unbiased -- so no bias correction is needed (cf.
    Vestergaard's covariance-based estimator: unbiased *because* it admits
    negative values). The cost is higher variance, since the x^2 weighting
    amplifies tail noise.

    The estimator assumes the profile is centred near x = 0, which the
    simulation guarantees; a production implementation would track the
    centroid. ``noise_free_series`` reuses the Gaussian profile -- the moment
    estimator is the *estimator* under test, applied to ordinary Gaussian
    data.
    """

    name: str = "moment"
    edge_frac: float = 0.1       # fraction of pixels each side used for baseline

    def noise_free_series(self, exp: Experiment) -> np.ndarray:
        return _gaussian_series(exp)

    def fit_variance(self, x: np.ndarray, y: np.ndarray) -> FrameFit:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        n = x.size
        edge = max(1, int(self.edge_frac * n))
        baseline = 0.5 * (y[:edge].mean() + y[-edge:].mean())
        ys = y - baseline                       # may be negative (noise)

        total = float(ys.sum())
        amp = float(np.nanmax(y))
        if not np.isfinite(total) or total <= 0.0:
            # Negative/zero total signal: a noise-dominated frame, unusable.
            return FrameFit(variance=np.nan, amplitude=amp,
                            at_wall=False, ok=False)

        mean = float(np.sum(ys * x) / total)
        variance = float(np.sum(ys * (x - mean) ** 2) / total)
        # No wall: the moment estimator is not bounded below.
        return FrameFit(variance=variance, amplitude=amp,
                        at_wall=False, ok=True)


# --------------------------------------------------------------------------
# Unbounded Gaussian -- the same shape-matched fit with the wall removed
# --------------------------------------------------------------------------
def _gaussian_precision(x: np.ndarray, mu: float, b: float,
                        amp: float) -> np.ndarray:
    """Gaussian parameterised by its *precision* ``b = 1 / sigma^2``.

    ``b > 0`` is the ordinary peaked Gaussian; ``b = 0`` is a flat line; ``b <
    0`` is convex (largest at the window edges). Unlike a variance
    parameterisation -- which has an essential singularity at ``sigma^2 = 0``
    that the optimiser cannot cross, so "removing the lower bound" on the
    variance is illusory -- the precision passes *smoothly* through zero, so
    the fit genuinely has no wall. The exponent is clipped from above so the
    convex ``b < 0`` branch cannot overflow.
    """
    return amp * np.exp(np.minimum(-0.5 * b * (x - mu) ** 2, 50.0))


class UnboundedGaussianModel:
    """Gaussian width fit with the positivity wall removed (precision-fitted).

    This is the head-to-head comparator for the folded-normal correction: the
    *same* shape-matched Gaussian, fitted in the precision parameter
    ``b = 1 / sigma^2`` so the fitted variance ``1 / b`` can take any sign. Its
    sampling distribution is therefore not folded at zero, which is the
    standard premise behind "an unconstrained fit needs no bias correction".
    The question it answers is "is the correction even needed?".

    The precision is bounded symmetrically to ``+/- 1 / var_min`` (var_min ~ one
    pixel squared), with the reported variance ``1 / b`` clipped to ``+/-
    span^2`` near ``b = 0``. The variance parameterisation is deliberately
    avoided: ``sigma^2 = 0`` is an essential singularity of the profile, an
    implicit wall the optimiser will not cross, so a variance-bounded "unbounded"
    fit cannot actually reach negative values.

    Finding (``experiments/estimator-comparison/unbounded_fit_comparison.py``):
    from per-frame CNR ~1.5 up this fit is identical to the bounded one, because
    the wall never binds. Both still carry the same upward bias (mean fitted
    variance 1.038, 1.016, 1.010 sigma_0^2 at CNR 1.52, 2.0, 2.6, against 1.0),
    which the correction removes: there the wall question is moot, the
    correction is not. In the bias-prone band (per-frame CNR < ~1.5)
    removing the wall only *partly* debiases the mean (e.g. at CNR 0.5 the mean
    fitted variance falls from ~2.4 to ~1.6 sigma_0^2, still well above the true
    1.0) and explodes the variance (the ``1 / b`` tail), because even without a
    wall a nonlinear shape-matched fit overwhelmingly returns a positive width:
    a localised positive noise bump fits better than a global convex trend, so
    negative variances stay rare. The unbounded fit is therefore not a
    substitute for the correction. The estimator is Gaussian-specific (a signed
    precision has no clean meaning for the Voigt or generalized-Gaussian shape);
    the shape-general unbounded counterpart is the linear, catastrophically
    noisy :class:`MomentModel`, which is the one estimator that does robustly
    admit negative variances.
    """

    name = "unbounded_gaussian"

    def noise_free_series(self, exp: Experiment) -> np.ndarray:
        return _gaussian_series(exp)

    def fit_variance(self, x: np.ndarray, y: np.ndarray) -> FrameFit:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if (unusable := _unusable_frame(x, y)) is not None:
            return unusable
        span = float(x[-1] - x[0])

        var_max = span ** 2
        var_min = (span / x.size) ** 2
        b_max = 1.0 / var_min               # ~ a one-pixel spike
        b0 = 1.0 / (span / 4.0) ** 2        # precision of the broad guess

        margin = 0.3 * span
        central = (x >= x[0] + margin) & (x <= x[-1] - margin)
        idx = np.where(central)[0]
        if idx.size:
            mu0 = float(x[idx][int(np.argmax(np.abs(y[idx])))])
        else:
            mu0 = float(x[int(np.argmax(np.abs(y)))])
        amp_max = float(np.max(np.abs(y))) or 1.0

        p0 = [mu0, b0, amp_max]
        lower = [x[0] + margin, -b_max, 0.0]
        upper = [x[-1] - margin, b_max, 2.0 * amp_max]
        try:
            popt, _ = curve_fit(_gaussian_precision, x, y, p0=p0,
                                bounds=(lower, upper), maxfev=5000)
            b, amp = float(popt[1]), float(popt[2])
            ok = True
            var = float(np.clip(1.0 / b, -var_max, var_max)) if b != 0.0 \
                else var_max
        except (RuntimeError, ValueError):
            var, amp, ok = (span / 4.0) ** 2, amp_max, False

        # No wall: the precision passes smoothly through zero, so there is no
        # lower bound to be pinned against.
        return FrameFit(variance=var, amplitude=amp, at_wall=False, ok=ok)


# Convenience registry for scripts that select a model by name.
MODELS = {
    "gaussian": GaussianModel,
    "voigt": VoigtModel,
    "generalized_gaussian": GeneralizedGaussianModel,
    "moment": MomentModel,
    "unbounded_gaussian": UnboundedGaussianModel,
}


def model_from_spec(spec: str):
    """Build a profile model from a ``name`` or ``name:shape`` string.

    ``"generalized_gaussian"`` alone gives the default shape p = 2, which is a
    Gaussian profile fitted by scale -- it does not test a non-Gaussian shape.
    A validation of the generalized-Gaussian family must name the exponent,
    e.g. ``"generalized_gaussian:1.5"`` (heavy-tailed) or
    ``"generalized_gaussian:4"`` (flat-topped). Scripts pass the spec string
    (not the model) to worker processes, so it must rebuild the same model.
    """
    name, _, shape = spec.partition(":")
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    if not shape:
        return MODELS[name]()
    if name != "generalized_gaussian":
        raise ValueError(f"only generalized_gaussian takes a shape, got {spec!r}")
    return MODELS[name](shape=float(shape))


def spec_tag(spec: str) -> str:
    """File-name tag for a model spec: ``generalized_gaussian:1.5`` -> ``..._p1.5``."""
    name, _, shape = spec.partition(":")
    return f"{name}_p{float(shape):g}" if shape else name
