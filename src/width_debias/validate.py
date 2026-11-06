"""End-to-end validation: does the folded-normal correction reduce D bias?

The validation has two stages, deliberately on independent random seeds:

1. **Calibrate** -- run Monte Carlo ensembles across a CNR sweep and build a
   :class:`~width_debias.bias_lookup.BiasLookup` mapping CNR to the
   truncation fraction, the MSD-point variance, and the additive variance bias.
2. **Evaluate** -- run fresh ensembles, fit D under each scheme using only the
   per-frame CNR plus the calibrated model, and report ``<D_est> / D_true`` as
   a function of CNR_0.

A scheme that works drives the bias curve toward 1 at low CNR, where OLS and
the standard inverse-variance scheme leave D inflated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit import fit_D
from .montecarlo import run_ensemble
from .simulate import Experiment
from .bias_lookup import BiasLookup, frame_stats
from .weights import (SCHEMES, calibration_D_se, corrected_variance,
                      msd_and_weights)


def sweep_stats(exp: Experiment, cnr0_grid: np.ndarray, n_runs: int = 400,
                seed: int = 1, model=None, noise_model=None) -> list:
    """Per-frame statistics from a Monte Carlo CNR sweep.

    Returns one :class:`~width_debias.bias_lookup.FrameStats` per CNR_0.  This
    is the raw material both for the truncation model and for per-frame
    diagnostics, so callers can reuse it instead of resampling.
    ``model`` selects the profile model (see :mod:`width_debias.models`);
    ``noise_model`` selects the noise law (see :mod:`width_debias.noise`,
    ``None`` = white).
    """
    stats = []
    for i, cnr0 in enumerate(cnr0_grid):
        ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed + i,
                           model=model, noise_model=noise_model)
        stats.append(frame_stats(ens))
    return stats


def calibrate(exp: Experiment, cnr0_grid: np.ndarray, n_runs: int = 400,
              seed: int = 1, model=None, noise_model=None) -> BiasLookup:
    """Build a truncation model from a Monte Carlo CNR sweep.

    The calibration uses ``exp.D`` to generate synthetic profile sequences, so
    the lookup is implicitly conditioned on that D. For a D-independent
    calibration the recommended approach is :func:`calibrate_d_free`, which is
    a thin wrapper that calls this function with ``D = 0``.
    """
    return BiasLookup.from_stats(
        sweep_stats(exp, cnr0_grid, n_runs=n_runs, seed=seed, model=model,
                    noise_model=noise_model))


def calibrate_d_free(cnr0_grid: np.ndarray, *, sigma0: float = 1.0,
                     tau: float = 1.0, half_window: float = 6.0,
                     n_pixels: int = 241, n_frames: int = 12,
                     t_max: float = 3.0, amplitude0: float = 1.0,
                     n_runs: int = 400, seed: int = 1,
                     model=None, noise_model=None) -> BiasLookup:
    """Build a truncation model **without reference to the application's D**.

    Generates the calibration profile sequence at ``D = 0``, so every frame's
    true variance is ``sigma_0^2`` and the resulting 1-D ``var_bias(CNR)``
    lookup is keyed on per-frame CNR alone. The slope fit in
    :func:`width_debias.weights.msd_and_weights` consumes it like any 1-D
    lookup.

    Use this whenever the application's D is unknown (real data), variable
    across measurements, or itself heterogeneous (a mixed-D population). In
    the low-D regime where the truncation bias bites, the per-frame variance
    barely changes from ``sigma_0^2`` over the measurement window, so the
    bias calibrated at ``sigma^2 = sigma_0^2`` is the bias the application
    actually sees. Tested by Monte Carlo at D = 0.005 to 0.1 sigma_0^2/tau on
    four profile shapes (``experiments/dfree-calibration``): with the
    calibration's own Monte Carlo error included, 4 of 96 cells lie beyond two
    combined standard errors of the reference. Quote a corrected D with the
    combined SE, the fit's own and
    :func:`~width_debias.weights.calibration_D_se`.

    All parameters except ``cnr0_grid`` describe the experimental geometry
    (pixel grid, frame count, lifetime); the calibration must match the
    geometry of the data it will be applied to. Lengths are in the unit of
    ``sigma0``: with the default ``sigma0 = 1``, the fitted variances passed to
    the lookup must be in units of sigma_0^2. The lookup records the pixel grid
    (``pixel_x``), the profile model (``model_spec``) and the lowest calibrated
    CNR (``cnr_floor``), so an application can check its data against them.
    """
    from .calibration import _describe_model
    from .models import GaussianModel

    cal_exp = Experiment(sigma0=sigma0, D=0.0, tau=tau, amplitude0=amplitude0,
                         half_window=half_window, n_pixels=n_pixels,
                         n_frames=n_frames, t_max=t_max)
    lookup = calibrate(cal_exp, cnr0_grid, n_runs=n_runs, seed=seed, model=model,
                       noise_model=noise_model)
    lookup.pixel_x = cal_exp.x
    lookup.model_spec = _describe_model(GaussianModel() if model is None else model)
    lookup.cnr_floor = float(lookup.cnr_grid[0])
    return lookup


def calibrate_fisher(cnr_grid: np.ndarray | None = None, *, sigma0: float = 1.0,
                     half_window: float = 6.0, n_pixels: int = 241,
                     model=None, n_runs: int | np.ndarray = 300, seed: int = 1,
                     width_scaling: bool = False, knee_cnr: float = 6.0,
                     workers: int | None = None) -> BiasLookup:
    """Fisher-anchored D-free calibration: exact leading term, smoothed residual.

    Like :func:`calibrate_d_free`, this calibrates at sigma^2 = sigma_0^2 and
    never uses the application's D. The leading-order bias, the width's Fisher
    variance, comes from the profile model and pixel grid with no Monte Carlo
    error; single-frame Monte Carlo cells over ``cnr_grid`` (default 84
    log-spaced points over CNR 0.25 to 90) supply only the higher-order
    residual, smoothed across all cells (:meth:`BiasLookup.from_cells`).
    ``width_scaling`` adds a width table so that a variance key rescales the
    lookup to each frame's width; it is off by default, so the model's
    behaviour matches the D-free lookup's interface exactly. The geometry
    arguments must match the data, as for :func:`calibrate_d_free`.

    Experimental, and not the recommended path: in its pre-registered test
    against :func:`calibrate_d_free` its predicted calibration SE was too small
    on the Voigt and p = 4 profiles (``experiments/fisher-calibration``).
    """
    from .calibration import calibrate_cells
    from .fisher import leading_order_terms
    from .models import GaussianModel

    if model is None:
        model = GaussianModel()
    if cnr_grid is None:
        cnr_grid = np.geomspace(0.25, 90.0, 84)
    pixel_x = Experiment(sigma0=sigma0, half_window=half_window, n_pixels=n_pixels).x
    variance_cal = sigma0 ** 2
    cells = calibrate_cells(cnr_grid, model=model, pixel_x=pixel_x,
                            variance_cal=variance_cal, n_runs=n_runs, seed=seed,
                            workers=workers)
    variance_grid = variance_cal * np.geomspace(0.25, 16.0, 41) if width_scaling else None
    terms = leading_order_terms(model, pixel_x, variance_cal, variance_grid)
    return BiasLookup.from_cells(cells, terms, knee_cnr=knee_cnr)


@dataclass
class SchemeResult:
    """Bias curve for one weighting scheme across the CNR_0 sweep."""

    scheme: str
    cnr0: np.ndarray
    mean_bias: np.ndarray     # <D_est> / D_true
    median_bias: np.ndarray   # median(D_est) / D_true
    spread: np.ndarray        # std(D_est / D_true)
    # Mean calibration SE of D_est / D_true (weights.calibration_D_se): the
    # lookup's own Monte Carlo error, which ``spread / sqrt(n_runs)`` omits.
    # NaN for schemes that subtract nothing or a model without SE grids.
    cal_se: np.ndarray | None = None


@dataclass
class FrameDiagnostic:
    """Per-frame residual bias of a width-fitting correction.

    For ``subtract`` and ``mas`` the final D estimate is a deterministic
    linear functional of the corrected per-frame variances (the slope-fit
    weights are deterministic in CNR), so the per-frame residual
    ``mean_corrected - true_variance`` fully determines the bias in D. This
    diagnostic exposes that residual frame by frame, which localises where a
    correction over- or under-shoots -- information the collapsed D ratio
    hides. See ``docs/DERIVATION.md``.

    All variances and standard errors are in units of sigma_0^2.

    Standard errors on the means scale roughly as ``SD(sigma_hat^2)/CNR_frame``
    -- they grow sharply at low per-frame CNR because the fit's noise scales
    as ``1/CNR``. Interpret any reported residual against ``residual_se`` to
    see whether it is statistically resolved.
    """

    scheme: str
    cnr0: float
    cnr: np.ndarray             # per-frame CNR
    times: np.ndarray           # frame times
    true_variance: np.ndarray   # sigma^2_true(t)
    mean_raw: np.ndarray        # E[sigma_hat^2] -- raw fitted, uncorrected
    mean_corrected: np.ndarray  # E[sigma_tilde^2] -- after the scheme's correction
    residual: np.ndarray        # mean_corrected - true_variance
    mean_raw_se: np.ndarray     # standard error of mean_raw (= SD/sqrt(n))
    mean_corrected_se: np.ndarray  # standard error of mean_corrected
    residual_se: np.ndarray     # = mean_corrected_se (true_variance is exact)
    n_runs_used: np.ndarray     # finite-fit count contributing to each frame


def per_frame_diagnostic(exp: Experiment, cnr0: float, model: BiasLookup,
                         scheme: str = "subtract", n_runs: int = 2000,
                         seed: int = 30000,
                         profile_model=None) -> FrameDiagnostic:
    """Per-frame residual bias of a correction scheme at a given CNR_0.

    Runs a Monte Carlo ensemble at ``cnr0``, applies the ``scheme``'s
    per-frame correction to every realisation, and reports, frame by frame,
    how well the corrected width ``E[sigma_tilde^2(t)]`` recovers
    ``sigma^2_true(t)``.

    ``scheme`` should be a width-fitting scheme (``subtract``, ``mas``;
    or ``ols`` / ``inverse_variance`` to see the uncorrected raw bias).
    """
    ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed,
                       model=profile_model)
    n_frames = exp.n_frames
    corrected = np.full((ens.n_runs, n_frames), np.nan)
    for r in range(ens.n_runs):
        corrected[r] = corrected_variance(scheme, ens.variance[r],
                                          ens.cnr_frame, model)

    mean_raw = np.full(n_frames, np.nan)
    mean_corr = np.full(n_frames, np.nan)
    mean_raw_se = np.full(n_frames, np.nan)
    mean_corr_se = np.full(n_frames, np.nan)
    n_used = np.zeros(n_frames, dtype=int)
    for f in range(n_frames):
        raw = ens.variance[:, f]
        raw = raw[np.isfinite(raw)]
        if raw.size:
            mean_raw[f] = float(raw.mean())
            if raw.size > 1:
                mean_raw_se[f] = float(raw.std(ddof=1)) / np.sqrt(raw.size)
        cor = corrected[:, f]
        cor = cor[np.isfinite(cor)]
        n_used[f] = cor.size
        if cor.size:
            mean_corr[f] = float(cor.mean())
            if cor.size > 1:
                mean_corr_se[f] = float(cor.std(ddof=1)) / np.sqrt(cor.size)

    true_var = exp.true_variance
    return FrameDiagnostic(scheme=scheme, cnr0=float(cnr0),
                           cnr=ens.cnr_frame, times=exp.times,
                           true_variance=true_var, mean_raw=mean_raw,
                           mean_corrected=mean_corr,
                           residual=mean_corr - true_var,
                           mean_raw_se=mean_raw_se,
                           mean_corrected_se=mean_corr_se,
                           residual_se=mean_corr_se,
                           n_runs_used=n_used)


def evaluate(exp: Experiment, cnr0_grid: np.ndarray, model: BiasLookup,
             n_runs: int = 400, seed: int = 20000,
             profile_model=None, noise_model=None) -> dict[str, SchemeResult]:
    """Fit D under every scheme and return the bias curves.

    ``model`` is the calibrated :class:`BiasLookup`; ``profile_model`` is
    the profile model (Gaussian / Voigt / generalized-Gaussian) and must match
    the one used for calibration. ``noise_model`` selects the noise law (see
    :mod:`width_debias.noise`, ``None`` = white); to test robustness to a
    noise-law mismatch, pass a ``model`` calibrated under one law and a
    ``noise_model`` of another.
    """
    times = exp.times
    cnr0_grid = np.asarray(cnr0_grid, float)
    n = cnr0_grid.size
    out = {s: {"mean": np.full(n, np.nan),
               "median": np.full(n, np.nan),
               "spread": np.full(n, np.nan),
               "cal_se": np.full(n, np.nan)} for s in SCHEMES}

    for i, cnr0 in enumerate(cnr0_grid):
        ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed + i,
                           model=profile_model, noise_model=noise_model)
        for scheme in SCHEMES:
            ratios, cal = [], []
            for r in range(ens.n_runs):
                msd, w = msd_and_weights(scheme, ens.variance[r],
                                         ens.cnr_frame, model)
                d = fit_D(times, msd, w)
                if d.ok and np.isfinite(d.D):
                    ratios.append(d.D / exp.D)
                    if scheme in ("subtract", "mas"):
                        cal.append(calibration_D_se(times, msd, w, ens.cnr_frame,
                                                    model, ens.variance[r]) / exp.D)
            ratios = np.asarray(ratios, float)
            if ratios.size:
                out[scheme]["mean"][i] = ratios.mean()
                out[scheme]["median"][i] = np.median(ratios)
                out[scheme]["spread"][i] = ratios.std()
            if cal:
                out[scheme]["cal_se"][i] = float(np.nanmean(cal))

    return {s: SchemeResult(scheme=s, cnr0=cnr0_grid,
                            mean_bias=out[s]["mean"],
                            median_bias=out[s]["median"],
                            spread=out[s]["spread"],
                            cal_se=out[s]["cal_se"])
            for s in SCHEMES}
