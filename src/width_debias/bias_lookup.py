"""Per-frame statistics of the fitted widths, and the CNR-keyed bias lookup.

For each frame, the Monte Carlo ensemble gives the sampling distribution of the
fitted MSD increment.  Three quantities summarise that distribution:

* ``trunc_fraction`` -- the fraction of realisations with MSD <= 0, the share
  of fitted increments the noise puts at or below zero.
* ``rel_variance``   -- the variance of the MSD point, the input the *standard*
  inverse-variance scheme already uses.
* ``skew``           -- the sample skewness of the MSD point, which a
  symmetric-error model of the slope fit ignores.

All three are monotone in per-frame CNR.  :class:`BiasLookup` packages
them against CNR so a weighting scheme can look them up from a per-frame CNR
alone (measured, on real data, with the ``fft-cnr`` package).

The class supports **two calibration modes** with the same API:

* **1-D** (recommended when built at D = 0): built from :func:`frame_stats`
  via :meth:`BiasLookup.from_stats`. Looks up ``var_bias`` and the
  per-frame width variance from CNR alone.
  :func:`~width_debias.validate.calibrate_d_free` builds it at D = 0, so the
  lookup carries no reference to the application's D;
  :func:`~width_debias.validate.calibrate` builds it at a given D (legacy,
  D-aware) -- see ``docs/DERIVATION.md``.
* **2-D, joint (CNR, sigma^2)** (experimental): built from a
  :class:`~width_debias.calibration.CalibrationGrid` via
  :meth:`BiasLookup.from_calibration`. Looks up ``var_bias`` and
  per-frame ``rel_variance`` from (CNR, sigma^2) jointly. Its low-CNR
  cells need far more Monte Carlo runs than the 1-D path, and keying each frame
  on its own fitted width under-corrects when the width grows several-fold
  (see :meth:`var_bias`); see :mod:`width_debias.calibration`.

The selection between the two is automatic: pass a ``variance`` argument to
:meth:`var_bias` or :meth:`rel_variance` to trigger the 2-D lookup when the
2-D table is present; otherwise the 1-D fallback runs. The two read
``variance`` on different scales. :meth:`var_bias` (and :meth:`var_bias_se`,
:meth:`var_bias_factor`) take an observed fitted variance, or a key on that
scale, and find the true width it implies by a fixed-point iteration;
:meth:`rel_variance` and :meth:`frame_variance` take a true width, which
:meth:`implied_true_variance` computes from an observed one.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from .montecarlo import Ensemble

# Version of the archive layout :meth:`BiasLookup.save` writes. Version 1 is the
# first versioned layout; archives without the key predate it.
LOOKUP_FORMAT_VERSION = 1

# Archive keys from width-debias 0.4.0 that name a field since renamed.
_RENAMED_FIELDS = {"cnr_floor": "lowest_calibrated_cnr"}


def _resolve_npz_path(path) -> Path:
    """The file ``np.savez`` wrote for ``path``: it appends ``.npz`` when missing."""
    given = Path(os.fspath(path))
    if given.is_file() or given.suffix == ".npz":
        return given
    return given.with_name(given.name + ".npz")


@dataclass
class FrameStats:
    """Per-frame summary of the MSD sampling distribution."""

    cnr: np.ndarray              # per-frame CNR
    trunc_fraction: np.ndarray   # P(MSD <= 0) per frame
    at_wall_fraction: np.ndarray # fraction of fits pinned at the width wall
    rel_variance: np.ndarray     # Var(MSD) per frame
    skew: np.ndarray             # sample skewness of MSD per frame
    mean_msd: np.ndarray         # mean fitted MSD per frame
    true_msd: np.ndarray         # ground-truth MSD per frame
    var_bias: np.ndarray         # E[sigma_hat^2] - sigma^2_true per frame
    #                              (the additive variance bias; see DERIVATION.md)
    var_bias_se: np.ndarray | None = None  # Monte Carlo SE of var_bias per frame
    frame_variance: np.ndarray | None = None  # Var(sigma_hat^2) per frame


def _skew(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if a.size < 3:
        return np.nan
    d = a - a.mean()
    s2 = (d ** 2).mean()
    if s2 == 0:
        return 0.0
    return float((d ** 3).mean() / s2 ** 1.5)


def frame_stats(ens: Ensemble) -> FrameStats:
    """Reduce a Monte Carlo ensemble to per-frame distribution statistics."""
    n_frames = ens.exp.n_frames
    trunc = np.full(n_frames, np.nan)
    rel_var = np.full(n_frames, np.nan)
    skew = np.full(n_frames, np.nan)
    mean_msd = np.full(n_frames, np.nan)
    var_bias = np.full(n_frames, np.nan)
    var_bias_se = np.full(n_frames, np.nan)
    frame_var = np.full(n_frames, np.nan)
    true_variance = ens.exp.true_variance

    for f in range(n_frames):
        col = ens.msd[:, f]
        col = col[np.isfinite(col)]
        if col.size:
            trunc[f] = float(np.mean(col <= 0.0))
            rel_var[f] = float(np.var(col, ddof=1)) if col.size > 1 else np.nan
            skew[f] = _skew(col)
            mean_msd[f] = float(np.mean(col))
        # Additive variance bias E[sigma_hat^2] - sigma^2_true (DERIVATION.md s.2).
        vcol = ens.variance[:, f]
        vcol = vcol[np.isfinite(vcol)]
        if vcol.size:
            var_bias[f] = float(np.mean(vcol)) - float(true_variance[f])
        if vcol.size > 1:
            # The lookup's own Monte Carlo error; propagated into D by
            # weights.calibration_D_se.
            var_bias_se[f] = float(np.std(vcol, ddof=1) / np.sqrt(vcol.size))
            # The width variance of this frame alone, a function of its own
            # CNR. rel_variance above also carries this ensemble's t = 0 frame,
            # which belongs to the calibration's CNR_0, not the application's.
            frame_var[f] = float(np.var(vcol, ddof=1))

    at_wall_frac = np.nanmean(ens.at_wall.astype(float), axis=0)
    return FrameStats(cnr=ens.cnr_frame, trunc_fraction=trunc,
                      at_wall_fraction=at_wall_frac, rel_variance=rel_var,
                      skew=skew, mean_msd=mean_msd, true_msd=ens.exp.true_msd,
                      var_bias=var_bias, var_bias_se=var_bias_se,
                      frame_variance=frame_var)


def _bilinear(x_grid: np.ndarray, y_grid: np.ndarray, table: np.ndarray,
              x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of ``table[i, j]`` defined on ``x_grid`` x ``y_grid``.

    ``x_grid`` and ``y_grid`` are 1-D ascending; ``table`` has shape
    ``(x_grid.size, y_grid.size)``. Queries outside the grid are clipped to the
    boundary value (i.e. nearest-edge extrapolation in each axis).
    """
    x_arr = np.atleast_1d(np.asarray(x, float))
    y_arr = np.atleast_1d(np.asarray(y, float))

    xc = np.clip(x_arr, x_grid[0], x_grid[-1])
    yc = np.clip(y_arr, y_grid[0], y_grid[-1])

    i = np.clip(np.searchsorted(x_grid, xc) - 1, 0, x_grid.size - 2)
    j = np.clip(np.searchsorted(y_grid, yc) - 1, 0, y_grid.size - 2)

    x_lo = x_grid[i]; x_hi = x_grid[i + 1]
    y_lo = y_grid[j]; y_hi = y_grid[j + 1]
    fx = (xc - x_lo) / np.maximum(x_hi - x_lo, 1e-30)
    fy = (yc - y_lo) / np.maximum(y_hi - y_lo, 1e-30)

    v00 = table[i, j]
    v01 = table[i, j + 1]
    v10 = table[i + 1, j]
    v11 = table[i + 1, j + 1]
    out = (v00 * (1 - fx) * (1 - fy)
           + v10 * fx * (1 - fy)
           + v01 * (1 - fx) * fy
           + v11 * fx * fy)
    # Return scalar if scalar inputs were passed.
    if np.ndim(x) == 0 and np.ndim(y) == 0:
        return float(out[0])
    return out


def _interp_se(x, xp: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """SE of ``np.interp(x, xp, fp)`` when each ``fp[k]`` has independent SE ``sp[k]``.

    The interpolant is ``(1 - f) fp[k] + f fp[k+1]``, so its SE is
    ``sqrt((1 - f)^2 sp[k]^2 + f^2 sp[k+1]^2)``; outside the grid it is the
    edge point's SE, matching np.interp's edge clamping.
    """
    xa = np.clip(np.atleast_1d(np.asarray(x, float)), xp[0], xp[-1])
    k = np.clip(np.searchsorted(xp, xa) - 1, 0, xp.size - 2)
    f = (xa - xp[k]) / np.maximum(xp[k + 1] - xp[k], 1e-30)
    out = np.sqrt((1 - f) ** 2 * sp[k] ** 2 + f ** 2 * sp[k + 1] ** 2)
    return float(out[0]) if np.ndim(x) == 0 else out


def _interp_matrix(x: np.ndarray, xp: np.ndarray) -> np.ndarray:
    """Weights W with ``W @ fp == np.interp(x, xp, fp)``, edges clamped as np.interp does."""
    xa = np.clip(np.atleast_1d(np.asarray(x, float)), xp[0], xp[-1])
    k = np.clip(np.searchsorted(xp, xa) - 1, 0, xp.size - 2)
    f = (xa - xp[k]) / np.maximum(xp[k + 1] - xp[k], 1e-30)
    weights = np.zeros((xa.size, xp.size))
    rows = np.arange(xa.size)
    weights[rows, k] = 1.0 - f
    weights[rows, k + 1] = f
    return weights


def _fixed_point_bias(sigma2_obs: np.ndarray, bias_at, var_lo: float,
                      var_hi: float, n_iter: int):
    """Bias read at the true variance implied by the observed one, v = v_obs - b(v).

    The lookup is calibrated against the true variance but only the biased
    observed one is known, so a few fixed-point steps refine the key. The
    iterate is clipped to the calibrated range each step (no extrapolation).
    """
    sigma2_true = sigma2_obs.copy()
    for _ in range(int(n_iter)):
        bias = bias_at(np.clip(sigma2_true, var_lo, var_hi))
        sigma2_true = sigma2_obs - bias
    return sigma2_obs - sigma2_true


def _bilinear_se(x_grid: np.ndarray, y_grid: np.ndarray, se_table: np.ndarray,
                 x, y) -> np.ndarray:
    """SE of :func:`_bilinear` when each table cell has an independent SE."""
    xc = np.clip(np.atleast_1d(np.asarray(x, float)), x_grid[0], x_grid[-1])
    yc = np.clip(np.atleast_1d(np.asarray(y, float)), y_grid[0], y_grid[-1])
    i = np.clip(np.searchsorted(x_grid, xc) - 1, 0, x_grid.size - 2)
    j = np.clip(np.searchsorted(y_grid, yc) - 1, 0, y_grid.size - 2)
    fx = (xc - x_grid[i]) / np.maximum(x_grid[i + 1] - x_grid[i], 1e-30)
    fy = (yc - y_grid[j]) / np.maximum(y_grid[j + 1] - y_grid[j], 1e-30)
    var = (((1 - fx) * (1 - fy)) ** 2 * se_table[i, j] ** 2
           + (fx * (1 - fy)) ** 2 * se_table[i + 1, j] ** 2
           + ((1 - fx) * fy) ** 2 * se_table[i, j + 1] ** 2
           + (fx * fy) ** 2 * se_table[i + 1, j + 1] ** 2)
    out = np.sqrt(var)
    return float(out[0]) if (np.ndim(x) == 0 and np.ndim(y) == 0) else out


@dataclass
class BiasLookup:
    """Per-frame CNR (+ optional variance) -> (p, rel_variance, var_bias).

    Two calibration modes share one API; see module docstring.

    Attributes
    ----------
    cnr_grid, p_grid, relvar_grid, varbias_grid : the 1-D lookup arrays
        (always present; the 2-D path also fills them with the variance-grid
        mean of each row so 1-D queries still work).
    variance_grid, varbias_2d, relvar_2d : the 2-D lookups (``None`` for a
        1-D calibration).
    varbias_se_grid, varbias_se_2d : Monte Carlo SE of each var_bias point
        (``None`` for a model built or saved without it). They give the
        calibration's own error, which :func:`~width_debias.weights.calibration_D_se`
        propagates into D.
    framevar_grid : per-frame ``Var(sigma_hat^2)`` on ``cnr_grid`` (1-D
        models from :meth:`from_stats`; ``None`` for models built or saved
        without it, which weight by the ``relvar_grid`` MSD-point variance).
        See :meth:`frame_variance`.
    varbias_cov_factor : (n_grid, p) factor F of the covariance F F^T of the
        smoothed ``varbias_grid`` (models from :meth:`from_cells`). Smoothed
        values are correlated, so the calibration SE of D needs the full
        covariance, not per-point SEs.
    width_variance_grid, width_scale_grid, calibration_variance : the
        leading-order bias at a true variance over its value at
        ``calibration_variance`` and equal CNR (:meth:`from_cells` with a width
        table). When present, a variance passed to :meth:`var_bias` rescales
        the lookup to that width; see :meth:`_width_lookup_point`.
    lowest_calibrated_cnr : the lowest CNR the calibration covers, for
        provenance. Every lookup takes the edge value outside ``cnr_grid``.
        It is not a floor below which the correction fails; that floor is the
        ``min_cnr`` of :func:`~width_debias.weights.msd_and_weights`.
    pixel_x, model_spec : the calibration's pixel grid and profile model, for
        provenance; an application must match both. :meth:`from_cells` and
        :func:`~width_debias.validate.calibrate_d_free` record these and
        ``lowest_calibrated_cnr``; other constructors leave them ``None``, and
        the match is then the caller's to keep. Nothing checks them
        automatically.
    format_version : the archive layout a loaded model was saved with
        (``None`` for a model built in memory or saved before versioning).
    """

    cnr_grid: np.ndarray
    p_grid: np.ndarray
    relvar_grid: np.ndarray
    varbias_grid: np.ndarray
    # Optional 2-D extensions, set by from_calibration().
    variance_grid: np.ndarray | None = None
    varbias_2d: np.ndarray | None = None
    relvar_2d: np.ndarray | None = None
    varbias_se_grid: np.ndarray | None = None
    varbias_se_2d: np.ndarray | None = None
    framevar_grid: np.ndarray | None = None
    # Optional Fisher-anchored extensions, set by from_cells().
    varbias_cov_factor: np.ndarray | None = None
    width_variance_grid: np.ndarray | None = None
    width_scale_grid: np.ndarray | None = None
    calibration_variance: float | None = None
    lowest_calibrated_cnr: float | None = None
    pixel_x: np.ndarray | None = None
    model_spec: str | None = None
    format_version: int | None = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_stats(cls, stats_list: list[FrameStats]) -> "BiasLookup":
        """1-D calibration from a CNR sweep of full experiments (``calibrate_d_free``'s path)."""
        cnr = np.concatenate([s.cnr for s in stats_list])
        p = np.concatenate([s.trunc_fraction for s in stats_list])
        rv = np.concatenate([s.rel_variance for s in stats_list])
        vb = np.concatenate([s.var_bias for s in stats_list])
        good = (np.isfinite(cnr) & np.isfinite(p) & np.isfinite(rv)
                & np.isfinite(vb) & (cnr > 0))
        order = np.argsort(cnr[good])
        se = fv = None
        if all(s.var_bias_se is not None for s in stats_list):
            se = np.concatenate([s.var_bias_se for s in stats_list])[good][order]
        if all(s.frame_variance is not None for s in stats_list):
            fv = np.concatenate([s.frame_variance for s in stats_list])[good][order]
        return cls(cnr_grid=cnr[good][order],
                   p_grid=p[good][order],
                   relvar_grid=rv[good][order],
                   varbias_grid=vb[good][order],
                   varbias_se_grid=se,
                   framevar_grid=fv)

    @classmethod
    def from_calibration(cls, calib) -> "BiasLookup":
        """2-D, D-free calibration from a :class:`CalibrationGrid`.

        Stores the 2-D lookup directly; also computes a 1-D marginal (mean
        over the variance grid) so 1-D queries still work as a fallback.
        ``p_grid`` is set to zeros: the truncation-fraction quantity is a
        relic of the 1-D mode and is unused by ``subtract`` / ``mas``.
        """
        cnr = np.asarray(calib.cnr_grid, float)
        var_g = np.asarray(calib.variance_grid, float)
        vb2 = np.asarray(calib.var_bias, float)
        rv2 = np.asarray(calib.rel_variance, float)
        # 1-D marginals: mean over the variance grid, for the no-variance path.
        vb_1d = np.nanmean(vb2, axis=1)
        rv_1d = np.nanmean(rv2, axis=1)
        p_1d = np.zeros_like(cnr)
        # Per-cell SE of the mean bias from the cell's sample variance. A grid
        # built with a filter threshold drops the fits above it from the bias
        # mean, and this unfiltered spread then makes the SE slightly
        # conservative.
        se2 = np.sqrt(rv2 / max(int(calib.n_runs), 1))
        n_var = np.sum(np.isfinite(se2), axis=1)
        se_1d = np.sqrt(np.nansum(se2 ** 2, axis=1)) / np.maximum(n_var, 1)
        order = np.argsort(cnr)
        return cls(
            cnr_grid=cnr[order],
            p_grid=p_1d[order],
            relvar_grid=rv_1d[order],
            varbias_grid=vb_1d[order],
            variance_grid=var_g,
            varbias_2d=vb2[order, :],
            relvar_2d=rv2[order, :],
            varbias_se_grid=se_1d[order],
            varbias_se_2d=se2[order, :],
        )

    @classmethod
    def from_cells(cls, cells, terms, *, knee_cnr: float = 6.0,
                   asymptote: float | None = 1.0, n_grid: int = 256,
                   cnr_max: float = 1e3) -> "BiasLookup":
        """Fisher-anchored, smoothed 1-D calibration from single-frame cells.

        ``cells`` is a :class:`~width_debias.calibration.CalibrationCells`;
        ``terms`` the :class:`~width_debias.fisher.LeadingOrderTerms` of the
        same model and pixel grid. The bias is written var_bias = L(CNR)
        rho(CNR): L = ``terms.bias_coef`` / CNR^2 is the leading-order bias (the
        width's Fisher variance), and rho, the ratio of the Monte Carlo bias to
        L, is smoothed over every cell with a P-spline that equals
        ``asymptote`` above ``knee_cnr``.

        The asymptote is fixed (1 by default) because the Monte Carlo cannot
        pin it: at high CNR the mean bias is only about 0.06/CNR of one fit's
        spread, so an affordable budget leaves the high-CNR ratio uncertain by
        more than its own size. 1 is the exact leading order for scale families
        (Gaussian, generalized-Gaussian); the second-order bias of the Voigt fit
        is 0.91 of the Fisher variance, a difference the free spline absorbs
        below the knee and that is below 2e-4 sigma_0^2 above it.
        ``asymptote=None`` fits it instead. Pooling replaces one cell's Monte
        Carlo error with the error of a curve fitted to all cells; that error is
        correlated along the grid, so its covariance factor is stored for
        :func:`~width_debias.weights.calibration_D_se`. Lookups below the lowest
        cell clamp to it (``lowest_calibrated_cnr``); nothing is extrapolated.
        """
        from .smoothing import pspline_fit

        if cells.noise_model != "gaussian":
            raise NotImplementedError("the Fisher leading term is implemented for "
                                      "white noise only")
        good = (np.isfinite(cells.mean_bias) & np.isfinite(cells.bias_sd)
                & (cells.n_finite > 1))
        cnr = cells.cnr[good]
        u = np.log(cnr)
        n_fits = cells.n_finite[good]
        knee = float(np.log(knee_cnr))
        leading = terms.bias_coef / cnr ** 2

        fit_asymptote = asymptote is None
        offset = 0.0 if fit_asymptote else float(asymptote)

        # A cell mean's SE is its RMS spread over sqrt(n). Deep-CNR spreads are
        # heavy-tailed, so one cell's sample spread is too noisy to weight by:
        # smooth the sample variance relative to its Fisher value. A smooth of
        # its log (the pilot) estimates a geometric mean, which sits below the
        # arithmetic one where the tail is heavy (it understated the variance of
        # the cell means near CNR 1 two- to fourfold); it only sets the weights
        # and a floor for the smooth of the variance itself.
        framevar_ratio = cells.bias_sd[good] ** 2 / (terms.framevar_coef / cnr ** 2)
        pilot = np.exp(pspline_fit(u, np.log(framevar_ratio), np.ones_like(u)).evaluate(u))
        spread = pspline_fit(u, framevar_ratio, pilot * np.sqrt(2.0 / (n_fits - 1)))
        cell_var = np.maximum(spread.evaluate(u), pilot) * terms.framevar_coef / cnr ** 2
        ratio_se = np.sqrt(cell_var / n_fits) / leading
        ratio_fit = pspline_fit(u, cells.mean_bias[good] / leading - offset,
                                ratio_se, knee=knee, intercept=fit_asymptote)
        # The weights need Var(sigma_hat^2) only to leading accuracy: smooth its
        # log ratio to the Fisher value (which the fit attains at high CNR, so
        # the log asymptote is 0), with the normal-theory SE of a log sample
        # variance.
        framevar_fit = pspline_fit(u, np.log(framevar_ratio),
                                   np.sqrt(2.0 / (n_fits - 1)), knee=knee,
                                   intercept=fit_asymptote)

        grid = np.geomspace(cnr.min(), cnr_max, n_grid)
        grid_u = np.log(grid)
        grid_leading = terms.bias_coef / grid ** 2
        design = ratio_fit.design(grid_u)
        eigval, eigvec = np.linalg.eigh(ratio_fit.coef_cov)
        factor = grid_leading[:, None] * (design @ (eigvec * np.sqrt(np.clip(eigval, 0.0, None))))
        framevar = terms.framevar_coef / grid ** 2 * np.exp(framevar_fit.evaluate(grid_u))
        return cls(cnr_grid=grid, p_grid=np.zeros_like(grid),
                   relvar_grid=framevar,
                   varbias_grid=grid_leading * (offset + design @ ratio_fit.coef),
                   varbias_se_grid=np.linalg.norm(factor, axis=1),
                   framevar_grid=framevar,
                   varbias_cov_factor=factor,
                   width_variance_grid=terms.variance_grid,
                   width_scale_grid=terms.bias_scale,
                   calibration_variance=terms.variance_cal,
                   lowest_calibrated_cnr=float(cnr.min()),
                   pixel_x=cells.pixel_x,
                   model_spec=cells.model_spec)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    @property
    def is_2d(self) -> bool:
        return self.variance_grid is not None and self.varbias_2d is not None

    @property
    def is_width_scaled(self) -> bool:
        return (self.width_scale_grid is not None
                and self.width_variance_grid is not None
                and self.calibration_variance is not None)

    @property
    def is_width_aware(self) -> bool:
        """Whether a width argument changes the lookup (2-D, or 1-D with a width table)."""
        return self.is_2d or self.is_width_scaled

    def implied_true_variance(self, cnr, variance, n_iter: int = 4) -> np.ndarray:
        """The true width an observed fitted variance (or a key) implies.

        ``variance - var_bias(cnr, variance)``, the point the fixed-point
        iteration of :meth:`var_bias` reads the bias at. The weights
        (:meth:`rel_variance`, :meth:`frame_variance`) and the calibration SE
        are read at the same point, so they take it from here.
        """
        variance = np.asarray(variance, float)
        return variance - self.var_bias(cnr, variance, n_iter)

    def p(self, cnr) -> np.ndarray:
        """Interpolated truncation fraction at the given per-frame CNR."""
        return np.interp(cnr, self.cnr_grid, self.p_grid,
                         left=self.p_grid[0], right=self.p_grid[-1])

    def var_bias(self, cnr, variance=None, n_iter: int = 4):
        """Interpolated additive variance bias.

        ``E[sigma_hat^2] - sigma^2_true``. To leading order it is the width's
        estimation variance Var(sigma_hat) (``docs/DERIVATION.md``,
        :mod:`width_debias.fisher`); subtracting it per frame removes the bias
        the MSD slope would otherwise inherit.

        With a 2-D calibration **and** a ``variance`` argument (the observed
        raw fitted variance), looks up the bias at (CNR, sigma^2_true)
        jointly; otherwise falls back to the 1-D lookup on CNR alone.

        **Subtlety: the lookup is calibrated against the true variance, but
        the application only knows the biased observed variance.** Querying
        directly at the observed value evaluates the bias at the wrong point,
        which distorts the per-frame correction (especially badly at low CNR,
        where the bias is largest). The fix is a short fixed-point iteration:
        starting from the observed variance, subtract the lookup at the
        current estimate to refine an estimate of the true variance, and
        re-query. Three or four iterations suffice; the bias used for the
        correction is the lookup at the converged estimate.

        **Choice of key.** ``variance`` is the key that stands in for the
        unknown true sigma^2, and any array can be passed. Keying on each
        frame's own sigma_hat^2 (the usual call) under-corrects when sigma^2
        grows several-fold over the measurement: at low CNR a frame whose fit
        scatters high reads the lookup near the grid edge, and the mean is
        over-subtracted. On the high-N Br-I bed (sigma^2 growing ~64x) this
        left D at 0.89 to 0.96 of the reference at CNR_0 15 to 20 on two of
        the three maps (0.99 on the fastest-spreading one), where a
        measurement-level key (a robust line through the measurement's own
        sigma_hat^2) matched the exact key to 0.003. That key assumes a form
        for sigma^2(t), so it is not the default here; see
        ``experiments/halide-mixing-highn/key_sensitivity.py`` and
        ``smoothed_key`` in ``br_halide_highn_perframe.py`` for a worked
        example and its limits (below CNR_0 ~15 no key determined D).

        With a width table (:meth:`from_cells`) and a ``variance``, the 1-D
        lookup is rescaled to the key's width (:meth:`_width_lookup_point`),
        with the same fixed-point iteration. Because the leading bias scales as
        sigma, not sigma^2, a key error moves it about half as much as in the
        2-D lookup. A model with neither ignores ``variance``.
        """
        if variance is None or not self.is_width_aware:
            return np.interp(cnr, self.cnr_grid, self.varbias_grid,
                             left=self.varbias_grid[0],
                             right=self.varbias_grid[-1])
        if self.is_2d:
            def bias_at(sigma2):
                return _bilinear(self.cnr_grid, self.variance_grid,
                                 self.varbias_2d, cnr, sigma2)
            var_lo, var_hi = self.variance_grid[0], self.variance_grid[-1]
        else:
            def bias_at(sigma2):
                scale, cnr_eq = self._width_lookup_point(cnr, sigma2)
                return scale * np.interp(cnr_eq, self.cnr_grid, self.varbias_grid)
            var_lo, var_hi = self.width_variance_grid[0], self.width_variance_grid[-1]
        return _fixed_point_bias(np.asarray(variance, float), bias_at,
                                 float(var_lo), float(var_hi), n_iter)

    def _width_lookup_point(self, cnr, variance):
        """Scale q and equivalent CNR at which the calibration curve is read.

        The leading bias at true variance v is r(v) A_cal / CNR^2, with r the
        width table. Reading the calibrated curve at c_eq = CNR sqrt(q / r),
        q = v / v_cal, and multiplying by q reproduces that exactly, and puts
        the higher-order residual at the CNR of equal relative width precision,
        Var(sigma_hat) / sigma^2, which is what it depends on in the continuum
        limit. ``variance`` is clipped to the table.
        """
        variance = np.clip(np.asarray(variance, float), self.width_variance_grid[0],
                           self.width_variance_grid[-1])
        ratio = np.interp(variance, self.width_variance_grid, self.width_scale_grid)
        scale = variance / self.calibration_variance
        return scale, np.asarray(cnr, float) * np.sqrt(scale / ratio)

    def var_bias_factor(self, cnr, variance=None, n_iter: int = 4) -> np.ndarray:
        """Rows M with Cov(looked-up var_bias) = M M^T (needs ``varbias_cov_factor``).

        The interpolation weights of the lookup applied to the grid's
        covariance factor, rescaled by the width table when one is used.
        """
        cnr_arr = np.atleast_1d(np.asarray(cnr, float))
        if variance is None or not self.is_width_scaled:
            return _interp_matrix(cnr_arr, self.cnr_grid) @ self.varbias_cov_factor
        sigma2_true = self.implied_true_variance(
            cnr_arr, np.atleast_1d(np.asarray(variance, float)), n_iter)
        scale, cnr_eq = self._width_lookup_point(cnr_arr, sigma2_true)
        return scale[:, None] * (_interp_matrix(cnr_eq, self.cnr_grid)
                                 @ self.varbias_cov_factor)

    def var_bias_cov(self, cnr, variance=None, n_iter: int = 4) -> np.ndarray:
        """Covariance of the looked-up biases at several frames.

        Smoothed calibrations correlate neighbouring lookups; a raw per-cell
        lookup does not, so without a covariance factor this is diagonal.
        """
        if self.varbias_cov_factor is None:
            return np.diag(np.atleast_1d(self.var_bias_se(cnr, variance, n_iter)) ** 2)
        rows = self.var_bias_factor(cnr, variance, n_iter)
        return rows @ rows.T

    def var_bias_se(self, cnr, variance=None, n_iter: int = 4):
        """Monte Carlo SE of the looked-up :meth:`var_bias`.

        The calibration is a finite Monte Carlo, so every subtracted bias
        carries an error the measurement's own SE does not see. At low CNR it
        dominates: two 300-run calibrations of the same sweep differ by 0.22
        in D/D_ref at D = 0.02 sigma_0^2/tau, CNR_0 = 34, against an
        evaluation SE of 0.04. The SE is propagated through the same
        interpolation weights as the lookup (independent cells assumed; for
        the D-free 1-D grid each point is its own ensemble frame, the frames of
        one ensemble are uncorrelated, and the per-point SE matches the spread
        over 16 independent calibrations). With
        a 2-D model and a ``variance``, it is evaluated at the same
        fixed-point estimate of sigma^2_true that :meth:`var_bias` uses.
        A smoothed model (:meth:`from_cells`) takes the SE from its covariance
        factor, which also covers the correlation between grid neighbours.
        Returns NaN when the model carries no SE (a model built or saved
        without it).
        """
        cnr_arr = np.asarray(cnr, float)
        if self.varbias_cov_factor is not None:
            se = np.linalg.norm(self.var_bias_factor(cnr_arr, variance, n_iter), axis=1)
            return float(se[0]) if np.ndim(cnr_arr) == 0 else se
        if variance is None or not self.is_2d:
            if self.varbias_se_grid is None:
                return np.full(np.shape(cnr_arr), np.nan) if np.ndim(cnr_arr) else np.nan
            return _interp_se(cnr_arr, self.cnr_grid, self.varbias_se_grid)
        if self.varbias_se_2d is None:
            return np.full(np.shape(cnr_arr), np.nan) if np.ndim(cnr_arr) else np.nan
        sigma2_true = np.clip(self.implied_true_variance(cnr_arr, variance, n_iter),
                              float(self.variance_grid[0]),
                              float(self.variance_grid[-1]))
        return _bilinear_se(self.cnr_grid, self.variance_grid,
                            self.varbias_se_2d, cnr_arr, sigma2_true)

    def frame_variance(self, cnr, variance=None):
        """Interpolated per-frame ``Var(sigma_hat^2)`` at the given CNR (1-D).

        With a width table and an explicit ``variance`` (a true width), the
        lookup is rescaled to that width (Var(sigma_hat^2) = 4 v Var(sigma_hat)
        to leading order, so the same table serves).
        :func:`~width_debias.weights.msd_and_weights` passes the true width a
        measurement-level key implies, never a frame's own fitted variance: a
        weight that depends on its own point's noise biases the slope at the
        order being corrected.

        The slope weights are ``1 / Var(sigma_hat_i^2)`` at the application's
        own frame CNR, frame 0 included (see
        :func:`~width_debias.weights.msd_and_weights`). The 1-D
        ``relvar_grid`` holds a different quantity, the MSD-point variance
        ``Var(sigma_hat_i^2) + Var(sigma_hat_0^2)`` measured inside each
        calibration ensemble, whose t = 0 frame sits at the calibration's
        CNR_0. Neighbouring grid points come from different ensembles, so that
        lookup is jagged, and each ensemble's own t = 0 point reads exactly 0:
        a frame whose CNR lay next to a calibration CNR_0 drew 5 to 50 times
        its proper weight. Returns ``None`` when the model carries no such grid.
        """
        if self.framevar_grid is None:
            return None
        if variance is None or not self.is_width_scaled:
            return np.interp(cnr, self.cnr_grid, self.framevar_grid,
                             left=self.framevar_grid[0], right=self.framevar_grid[-1])
        scale, cnr_eq = self._width_lookup_point(cnr, variance)
        return scale ** 2 * np.interp(cnr_eq, self.cnr_grid, self.framevar_grid)

    def rel_variance(self, cnr, variance=None):
        """Interpolated variance of the (fitted-variance) point.

        With a 1-D calibration from :meth:`from_stats` this is the MSD-point
        variance ``Var(sigma_hat^2(t) - sigma_hat^2(0))`` empirically measured
        per CNR; from :meth:`from_cells` it is the per-frame ``Var(sigma_hat^2)``
        (a copy of ``framevar_grid``). With a 2-D calibration and a
        ``variance`` (a true width) it is the per-frame ``Var(sigma_hat^2)`` at
        that width; without one, its average over the variance grid.
        :func:`~width_debias.weights.msd_and_weights` weights by it, read at a
        key's true width or averaged, never at a frame's own fitted variance.
        """
        if variance is None or not self.is_2d:
            return np.interp(cnr, self.cnr_grid, self.relvar_grid,
                             left=self.relvar_grid[0],
                             right=self.relvar_grid[-1])
        return _bilinear(self.cnr_grid, self.variance_grid, self.relvar_2d,
                         cnr, variance)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path) -> None:
        """Save every field that is set to one ``.npz`` archive.

        Unset fields are omitted rather than written empty: an empty
        ``variance_grid`` would read back as a 2-D model, and scripts that pass
        an archive's keys straight to the constructor keep working. Like
        ``np.savez``, a path without the ``.npz`` suffix gets one; :meth:`load`
        accepts the path as given here.
        """
        arrays = {f.name: np.asarray(getattr(self, f.name)) for f in fields(self)
                  if getattr(self, f.name) is not None}
        arrays["format_version"] = np.asarray(LOOKUP_FORMAT_VERSION)
        np.savez_compressed(str(path), **arrays)

    @classmethod
    def load(cls, path) -> "BiasLookup":
        """Load a model saved with :meth:`save` (or any archive of its fields).

        A 1-D lookup saved without ``framevar_grid`` loads with a warning:
        :func:`~width_debias.weights.msd_and_weights` then uses the MSD-point
        weights, with frame 0 at weight 0.
        """
        known = {f.name for f in fields(cls)}
        values = {}
        with np.load(_resolve_npz_path(path)) as archive:
            for key in archive.files:
                name = _RENAMED_FIELDS.get(key, key)
                if name in known:
                    values[name] = (archive[key].item() if archive[key].ndim == 0
                                    else archive[key])
        version = values.get("format_version")
        if version is not None and int(version) > LOOKUP_FORMAT_VERSION:
            raise ValueError(f"{path} has archive format {version}; this version of "
                             f"width-debias reads up to {LOOKUP_FORMAT_VERSION}")
        model = cls(**values)
        if not model.is_2d and model.framevar_grid is None:
            warnings.warn(f"{path} has no framevar_grid, so msd_and_weights uses the "
                          "MSD-point weights (frame 0 at weight 0) for this lookup",
                          stacklevel=2)
        return model
