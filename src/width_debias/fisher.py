"""Fisher information of the width fit, and the leading-order bias it implies.

The fitted variance of a profile, sigma_hat^2, is the square of a noisy width.
If the width estimate is unbiased to second order (E[sigma_hat] = sigma), then

    E[sigma_hat^2] = sigma^2 + Var(sigma_hat),

so the additive bias the ``subtract`` scheme removes is, to leading order in
1/CNR^2, the width's own estimation variance. For an efficient fit that variance
is the Cramer-Rao value, which follows from the noise-free profile model alone:
no Monte Carlo is needed for the leading term. The second-order bias of the
nonlinear least-squares fit, evaluated numerically on this grid, confirms the
premise for scale families (the Gaussian and the generalized-Gaussian at fixed
p give var_bias = Var(sigma_hat) exactly); for the Voigt model with a fixed
Lorentzian it is 0.91 of it, a difference the calibration's smoothed residual
absorbs (see ``BiasLookup.from_cells``).

Everything here works on the model's own ``noise_free_series`` through a stand-in
for :class:`~width_debias.simulate.Experiment`, so every profile model is
supported without model-specific derivatives. Each frame is parameterised by its
diffusive variance v, with peak amplitude and centre as nuisance parameters.

At a fixed pixel grid and peak-defined CNR, the 1-D Gaussian gives
Var(sigma_hat) = (2/sqrt(pi)) sigma dx / CNR^2: the bias grows as sigma, not as
sigma^2, which is what :func:`leading_order_terms` tabulates for width scaling.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .simulate import Experiment, noise_sigma_for_cnr0


def _profile_stack(model, x: np.ndarray, variance: np.ndarray,
                   amplitude: np.ndarray, mean: float = 0.0) -> np.ndarray:
    """Noise-free profiles at the given per-frame variances on grid ``x``."""
    stand_in = SimpleNamespace(x=np.asarray(x, float), mean=mean,
                               amplitude=np.asarray(amplitude, float),
                               true_variance=np.asarray(variance, float))
    return model.noise_free_series(stand_in)


def profile_jacobian(model, x: np.ndarray, variance: float, *,
                     amplitude: float = 1.0, mean: float = 0.0,
                     rel_step: float = 1e-4) -> np.ndarray:
    """Derivatives of one noise-free profile, shape (3, n_pix): d/dv, d/dA, d/dc.

    Central differences in v and the centre c; the amplitude derivative is the
    profile over its amplitude, exact because every model is linear in its peak.
    """
    v = np.array([float(variance)])
    amp = np.array([float(amplitude)])
    hv = rel_step * v[0]
    hc = rel_step * np.sqrt(v[0])
    dv = (_profile_stack(model, x, v + hv, amp, mean)
          - _profile_stack(model, x, v - hv, amp, mean))[0] / (2 * hv)
    dc = (_profile_stack(model, x, v, amp, mean + hc)
          - _profile_stack(model, x, v, amp, mean - hc))[0] / (2 * hc)
    profile = _profile_stack(model, x, v, amp, mean)[0]
    return np.stack([dv, profile / amplitude, dc])


def var_of_fitted_variance(model, x: np.ndarray, variance: float,
                           noise_var: float | np.ndarray, *,
                           amplitude: float = 1.0, mean: float = 0.0,
                           estimator: str = "ls") -> float:
    """Leading-order Var(sigma_hat^2) of one frame's width fit.

    ``noise_var`` is a scalar (white noise) or a per-pixel vector. ``"ml"`` is
    the Cramer-Rao bound, (J W J^T)^-1 with W = 1/noise_var. ``"ls"`` is the
    covariance of the unweighted least-squares fit every model here uses,
    (J J^T)^-1 J Sigma J^T (J J^T)^-1; under white noise the two coincide, and
    under shot noise only the sandwich describes the fit actually run.
    """
    jac = profile_jacobian(model, x, variance, amplitude=amplitude, mean=mean)
    noise = np.broadcast_to(np.asarray(noise_var, float), jac.shape[1:])
    if estimator == "ml":
        return float(np.linalg.inv((jac / noise) @ jac.T)[0, 0])
    if estimator == "ls":
        bread = np.linalg.inv(jac @ jac.T)
        meat = (jac * noise) @ jac.T
        return float((bread @ meat @ bread)[0, 0])
    raise ValueError(f"unknown estimator {estimator!r}; expected 'ls' or 'ml'")


@dataclass(frozen=True)
class LeadingOrderTerms:
    """CNR-free leading-order coefficients of the width fit at one geometry.

    White noise and peak-defined CNR: at per-frame CNR c and the calibration
    width, var_bias ~ ``bias_coef / c^2`` and Var(sigma_hat^2) ~
    ``framevar_coef / c^2``, in the squared units of the pixel grid.
    ``bias_scale[j]`` is the leading var_bias at ``variance_grid[j]`` over its
    value at ``variance_cal`` and equal CNR (None when not tabulated).
    """

    variance_cal: float
    bias_coef: float
    framevar_coef: float
    variance_grid: np.ndarray | None = None
    bias_scale: np.ndarray | None = None


def leading_order_terms(model, x: np.ndarray, variance_cal: float,
                        variance_grid: np.ndarray | None = None) -> LeadingOrderTerms:
    """Leading-order bias and spread coefficients, optionally with a width-scale table.

    Evaluated at CNR 1 (unit peak, unit noise), so the coefficients carry the
    1/CNR^2 dependence implicitly. The width table uses the same pixel grid as
    the calibration, so it includes the window truncation a wide profile meets.
    The profile sits at the grid's midpoint, as the calibration cells place it.
    """
    x = np.asarray(x, float)
    centre = 0.5 * float(x[0] + x[-1])
    framevar_cal = var_of_fitted_variance(model, x, variance_cal, 1.0, mean=centre)
    bias_cal = framevar_cal / (4.0 * variance_cal)
    if variance_grid is None:
        return LeadingOrderTerms(variance_cal=float(variance_cal),
                                 bias_coef=bias_cal, framevar_coef=framevar_cal)
    variance_grid = np.asarray(variance_grid, float)
    bias = np.array([var_of_fitted_variance(model, x, v, 1.0, mean=centre) / (4.0 * v)
                     for v in variance_grid])
    return LeadingOrderTerms(variance_cal=float(variance_cal),
                             bias_coef=bias_cal, framevar_coef=framevar_cal,
                             variance_grid=variance_grid,
                             bias_scale=bias / bias_cal)


# --------------------------------------------------------------------------
# Whole-sequence Fisher information
# --------------------------------------------------------------------------
def sequence_fisher_information(model, exp: Experiment,
                                noise_sigma: float) -> np.ndarray:
    """Fisher matrix for theta = (sigma_0^2, D, {A_i}, {c_i}), white noise.

    The profile depends on (sigma_0^2, D) only through each frame's variance
    v_i = sigma_0^2 + 2 D t_i, so d mu_i / d sigma_0^2 = d mu_i / d v_i and
    d mu_i / d D = 2 t_i d mu_i / d v_i. Every model is peak-normalised, so
    d mu_i / d A_i = mu_i / A_i.
    """
    v, t, amp = exp.true_variance, exp.times, exp.amplitude
    n = exp.n_frames
    hv = 1e-4 * v
    dv = (_profile_stack(model, exp.x, v + hv, amp)
          - _profile_stack(model, exp.x, v - hv, amp)) / (2 * hv[:, None])
    hc = 1e-4 * exp.sigma0
    dc = (_profile_stack(model, exp.x, v, amp, hc)
          - _profile_stack(model, exp.x, v, amp, -hc)) / (2 * hc)
    mu = _profile_stack(model, exp.x, v, amp)
    fim = np.zeros((2 + 2 * n, 2 + 2 * n))
    for i in range(n):
        cols = np.zeros((2 + 2 * n, exp.x.size))
        cols[0] = dv[i]
        cols[1] = 2.0 * t[i] * dv[i]
        cols[2 + i] = mu[i] / amp[i]
        cols[2 + n + i] = dc[i]
        fim += cols @ cols.T
    return fim / noise_sigma ** 2


def per_frame_crlb(model, exp: Experiment, noise_sigma: float) -> np.ndarray:
    """Var(sigma_hat_i^2) floor for each frame alone (A_i, c_i as nuisances)."""
    return np.array([
        var_of_fitted_variance(model, exp.x, v, noise_sigma ** 2,
                               amplitude=a, estimator="ml")
        for v, a in zip(exp.true_variance, exp.amplitude)
    ])


def crlb_D(model, exp: Experiment, cnr0: float) -> float:
    """Cramer-Rao bound on D from the whole profile sequence."""
    fim = sequence_fisher_information(model, exp, noise_sigma_for_cnr0(exp, cnr0))
    return float(np.linalg.inv(fim)[1, 1])
