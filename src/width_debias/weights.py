"""Schemes for the MSD -> D linear fit.

Four schemes are compared:

* ``ols``               -- ordinary least squares; every frame weighted equally.
* ``inverse_variance``  -- the field-standard scheme, w_i = 1 / Var(sigma_hat_i^2).
                           It corrects for the unequal noise across frames but
                           leaves the bias intact.
* ``subtract``          -- the derived scheme and the recommended one. It
                           subtracts the additive variance bias s^2 from each
                           frame before forming the MSD, then weights by
                           inverse variance.
* ``mas``               -- a Plaszczynski-MAS-style variant of ``subtract``.
                           The subtraction is smoothly gated so it vanishes,
                           rather than overshooting, when a frame is noise
                           dominated -- it cannot drive the corrected variance
                           below zero. The gate also switches the correction
                           off below an initial CNR of about 4, where ``mas``
                           leaves the raw inflation in place.

The fitted variance carries an additive upward bias equal to the width's
estimation variance (``docs/DERIVATION.md`` derives it from a folded-normal
model), and that bias -- not extra scatter -- is what inflates D. A
reweighting of the slope fit cannot remove a mean bias present in every point;
only the per-frame subtraction in ``subtract`` / ``mas`` can. They differ
from ``inverse_variance`` *only* by that subtraction, so the comparison
isolates the effect of the bias correction.

Every scheme takes the *per-frame CNR* as input. In the Monte Carlo validation
the CNR is known exactly; on real data it is measured from each profile with
the ``fft-cnr`` package (see :mod:`width_debias.cnr`).

The ``model`` (a :class:`~width_debias.bias_lookup.BiasLookup`) can be
a 1-D calibration (recommended: :func:`~width_debias.validate.calibrate_d_free`,
built at D = 0, so it carries no reference to the application's D) or the
experimental joint (CNR, sigma^2) calibration from
:func:`~width_debias.calibration.calibrate_cnr_sigma2`. With the joint
calibration the per-frame raw fitted variance is also passed into the lookup,
so the correction is width-aware.
"""
from __future__ import annotations

import warnings

import numpy as np

from .bias_lookup import BiasLookup

SCHEMES = ("ols", "inverse_variance", "subtract", "mas")


class CalibrationRangeWarning(UserWarning):
    """A frame's CNR lies outside the lookup's calibrated range.

    The lookup takes its edge value there instead of extrapolating, so the
    subtracted bias and the weight of that frame are the edge values. Extend
    the calibration's CNR grid, or leave such frames out with ``min_cnr``.
    """


def _warn_outside_calibration(cnr: np.ndarray, model: BiasLookup) -> None:
    lo, hi = float(model.cnr_grid[0]), float(model.cnr_grid[-1])
    cnr = cnr[np.isfinite(cnr)]
    if cnr.size and (cnr.min() < lo or cnr.max() > hi):
        # The message names only the model's range, so a loop over many
        # realisations warns once per lookup, not once per call.
        warnings.warn(CalibrationRangeWarning(
            f"per-frame CNR outside the calibrated range [{lo:.3g}, {hi:.3g}]; "
            "the lookup takes its edge value there"), stacklevel=3)


def corrected_variance(scheme: str, variance_row: np.ndarray,
                       cnr_frame: np.ndarray,
                       model: BiasLookup,
                       key_variance: np.ndarray | None = None) -> np.ndarray:
    """Apply a scheme's per-frame bias correction to the fitted variances.

    Returns the corrected per-frame variances sigma_tilde^2(t). For ``ols``
    and ``inverse_variance`` this is the input unchanged -- those schemes do
    not correct the points, only the weights. For ``subtract`` it is the
    hard subtraction ``sigma_hat^2 - var_bias(CNR)``; for ``mas`` the smoothly
    gated subtraction (see module docstring).

    This is the single place the bias correction is applied.
    :func:`msd_and_weights` calls it to form the MSD, and
    :func:`width_debias.validate.per_frame_diagnostic` calls it to report
    the per-frame residual bias. Because the slope fit is linear in these
    corrected variances and the weights are deterministic in CNR, the
    per-frame residual ``E[sigma_tilde^2] - sigma^2_true`` fully determines
    the bias in the final D (see ``docs/DERIVATION.md``).

    ``key_variance`` is the width at which a width-aware model (2-D, or 1-D
    with a width table) reads the bias; by default each frame's own fitted
    variance, refined by the lookup's fixed-point iteration. A model with no
    width dependence ignores it.
    """
    variance_row = np.asarray(variance_row, float)
    cnr = np.asarray(cnr_frame, float)
    key = variance_row if key_variance is None else np.asarray(key_variance, float)
    if scheme in ("ols", "inverse_variance"):
        return variance_row
    if scheme == "subtract":
        return variance_row - model.var_bias(cnr, key)
    if scheme == "mas":
        # Plaszczynski-MAS-style gated subtraction. The gate (1 - exp(-u)),
        # u = sigma_hat^2 / s^2, is ~1 when the frame is well above the noise
        # (full s^2 subtraction, matching subtract) and ~0 when it is
        # noise dominated (no subtraction). corrected = s^2 * (u - 1 + e^-u)
        # is >= 0 for all u >= 0 -- the correction can never overshoot.
        s2 = model.var_bias(cnr, key)
        hi = np.nanmax(s2)
        floor = hi * 1e-6 if np.isfinite(hi) and hi > 0 else 1e-12
        s2 = np.clip(s2, floor, None)
        u = np.clip(variance_row, 0.0, None) / s2
        return variance_row - s2 * (1.0 - np.exp(-u))
    raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")


def msd_and_weights(scheme: str, variance_row: np.ndarray,
                    cnr_frame: np.ndarray, model: BiasLookup,
                    key_variance: np.ndarray | None = None,
                    min_cnr: float | None = None):
    """Return ``(msd, weights)`` for one realisation's MSD-vs-t fit.

    Parameters
    ----------
    scheme : one of :data:`SCHEMES`.
    variance_row : array, shape (n_frames,)
        The fitted variances sigma_hat^2(t) for a single realisation.
    cnr_frame : array, shape (n_frames,)
        Per-frame CNR (known in Monte Carlo; measured with ``fft-cnr`` on real
        data).
    model : calibrated :class:`~width_debias.bias_lookup.BiasLookup`.
    key_variance : optional array, shape (n_frames,)
        A measurement-level width key on the scale of the fitted variances, for
        a width-aware lookup; see Notes.
    min_cnr : optional per-frame CNR floor. Frames below it get weight 0, so
        :func:`~width_debias.fit.fit_D` and :func:`calibration_D_se` leave
        them out. Off by default; the validations use 1 (1.3 on measured data),
        below which the correction no longer holds (README, "On real data").
        This is not the lookup's ``lowest_calibrated_cnr``, the edge of its
        grid, which usually lies well below the floor. A scheme that consults
        the lookup warns (:class:`CalibrationRangeWarning`) when a frame it
        keeps lies outside the calibrated CNR range.

    Notes
    -----
    ``subtract`` subtracts the calibrated additive variance bias
    ``model.var_bias(CNR, sigma_hat^2)`` from each frame before differencing.
    ``mas`` subtracts the same bias through a smooth gate that switches the
    correction off in deep noise (so the corrected variance never crosses
    zero). ``ols`` and ``inverse_variance`` leave the variances uncorrected.

    The weighted schemes weight every frame, t = 0 included, by
    ``1 / Var(sigma_hat_i^2)``. The MSD points all share ``sigma_hat^2(0)``,
    so they are not independent; but a constant shift does not change a
    free-intercept slope, so fitting ``msd = corrected - corrected[0]`` with
    these weights is the same as fitting the corrected widths against time,
    whose points are independent. That fit reaches the Cramer-Rao bound on D
    (0.95 to 1.10 x sqrt(CRLB) from CNR_0 21 on four shapes;
    ``experiments/crlb-precision/efficiency_check.py``), and ``fit_D``'s
    covariance SE is then the right one. MSD-point weights,
    ``1 / (Var_i + Var_0)`` with frame 0 at weight 0, treat the points as
    independent and leave the best-measured frame out of the slope: 1.30 to
    1.43 x sqrt(CRLB). ``ols`` keeps unit weights on the MSD points with frame
    0 at weight 0, and a 1-D model saved without a ``framevar_grid`` uses the
    MSD-point weights (it has no per-frame variance to weight by).

    ``key_variance`` is a measurement-level stand-in for each frame's fitted
    width, smoothed across the measurement (for example a robust line through
    sigma_hat^2(t)). For a width-aware lookup (2-D, or 1-D with a width table)
    it keys the bias, and the weights are read at the true width it implies,
    ``key - var_bias(CNR, key)``, the point the bias is read at. Without it
    the bias is keyed on each frame's own fitted variance and the weights do
    not depend on width (a 2-D lookup uses its width-averaged 1-D
    ``rel_variance``). A weight read at a frame's own noisy width would
    correlate with that frame's error and bias the slope at O(1/CNR^2), the
    order being corrected: on a joint calibration with Gaussian fits it left
    D/D_true at 0.39 to 0.96 over CNR_0 8 to 34, against 0.88 to 1.09 for the
    width-averaged weight
    (``experiments/dfree-calibration/joint_weights_check.py``). A bias keyed
    on the frame's own width errs only at O(1/CNR^4).
    """
    variance_row = np.asarray(variance_row, float)
    cnr = np.asarray(cnr_frame, float)
    if variance_row.ndim != 1 or cnr.shape != variance_row.shape:
        # A 2-D batch would be differenced against its first row, not each
        # realisation's own t = 0 frame.
        raise ValueError("msd_and_weights takes one realisation: variance_row and "
                         f"cnr_frame of shape (n_frames,), got {variance_row.shape} "
                         f"and {cnr.shape}")
    if key_variance is not None and np.shape(key_variance) != variance_row.shape:
        raise ValueError("key_variance must match variance_row, shape "
                         f"{variance_row.shape}; got {np.shape(key_variance)}")

    # --- bias correction ------------------------------------------------
    corrected = corrected_variance(scheme, variance_row, cnr, model, key_variance)
    msd = corrected - corrected[0]

    # --- weights --------------------------------------------------------
    # 1 / Var(sigma_hat_i^2), looked up at this measurement's own frame CNR:
    # the 2-D calibration's rel_variance or the 1-D frame_variance, at the
    # key's true width when a width-aware lookup has a key. A 1-D model
    # without a frame_variance grid falls back to rel_variance, the MSD-point
    # variance measured inside each calibration ensemble (jagged in CNR; see
    # BiasLookup.frame_variance), with frame 0 at weight 0.
    if scheme == "ols":
        w = np.ones_like(cnr)
        per_frame = False
    else:
        width = None
        if key_variance is not None and model.is_width_aware:
            width = model.implied_true_variance(cnr, key_variance)
        per_frame = model.is_2d or model.framevar_grid is not None
        if model.is_2d:
            rv = model.rel_variance(cnr, width)
        elif per_frame:
            rv = model.frame_variance(cnr, width)
        else:
            rv = model.rel_variance(cnr)
        hi = np.nanmax(rv)
        tiny = hi * 1e-12 if np.isfinite(hi) and hi > 0 else 1e-12
        w = 1.0 / np.clip(rv, tiny, None)

    w = np.asarray(w, float).copy()
    if not per_frame:
        w[0] = 0.0   # the MSD origin, which these weights cannot place
    kept = np.ones(cnr.shape, bool) if min_cnr is None else cnr >= min_cnr
    w = np.where(kept, w, 0.0)
    if scheme != "ols":
        _warn_outside_calibration(cnr[kept], model)
    return msd, w


def calibration_D_se(times: np.ndarray, msd: np.ndarray, weights: np.ndarray,
                     cnr_frame: np.ndarray, model: BiasLookup,
                     variance_row: np.ndarray | None = None,
                     key_variance: np.ndarray | None = None) -> float:
    """SE of a corrected D from the calibration's own Monte Carlo error.

    :func:`~width_debias.fit.fit_D` is a weighted straight-line fit, so its
    slope is a fixed linear combination of the MSD points, ``b = sum c_i m_i``
    with ``c_i = w_i (S_w t_i - S_wt) / (S_w S_wtt - S_wt^2)``. A calibration
    error ``e_i`` in frame i's subtracted bias moves ``m_i`` by
    ``-(e_i - e_0)``. Since ``sum c_i = 0`` over the fitted frames, the common
    ``e_0`` shift cancels (the intercept absorbs it), and frame 0's own error
    enters through ``c_0`` when frame 0 is fitted. With independent
    calibration cells, summing over the fitted frames,

        Var_cal(D) = (1/4) sum_i c_i^2 s_i^2,   s_i = model.var_bias_se(CNR_i).

    Pass the same ``times``, ``msd``, and ``weights`` given to ``fit_D`` so the
    frame set matches, and ``variance_row`` (the raw fitted variances) when the
    model is a joint (CNR, sigma^2) calibration. Combine with the fit's own SE
    as ``sqrt(D_se**2 + calibration_D_se**2)``. This is exact for
    ``subtract``; for ``mas`` it ignores the gate's dependence on the bias
    and is approximate. The weights' own dependence on the calibration is
    second order and not included. Sixteen independent calibrations on the
    fft-cnr paper's grid confirm the prediction: observed / predicted variance
    0.98 over 18 draws with the MSD-point weights
    (``experiments/dfree-calibration/calibration_se_check.py``, where the
    weight term is below 0.06 of it from CNR_0 5 up and about 0.3 of it at
    CNR_0 3), and 0.89 over 16 draws with frame 0 weighted
    (``slope_fit_check.py``). Returns NaN when the model has no SE.

    A smoothed model (:meth:`~width_debias.bias_lookup.BiasLookup.from_cells`)
    correlates the biases of neighbouring frames, so the sum uses the full
    covariance, ``Var_cal(D) = (1/4) c^T M M^T c`` with M from
    :meth:`~width_debias.bias_lookup.BiasLookup.var_bias_factor`.
    ``key_variance`` is the width key passed to ``msd_and_weights``, if any.
    A width-aware lookup (2-D, or 1-D with a width table) needs
    ``variance_row`` or ``key_variance``, and raises when neither is given.
    """
    t = np.asarray(times, float)
    m = np.asarray(msd, float)
    w = np.asarray(weights, float)
    cnr = np.asarray(cnr_frame, float)
    if variance_row is None and key_variance is None and model.is_width_aware:
        # The correction read this lookup at each frame's width; without it the
        # SE would come from the width-averaged grid and understate the error.
        raise ValueError("a width-aware lookup needs variance_row (or the "
                         "key_variance given to msd_and_weights)")
    good = np.isfinite(t) & np.isfinite(m) & np.isfinite(w) & (w > 0)
    if good.sum() < 2:
        return np.nan
    tg, wg = t[good], w[good]
    sw, swt, swtt = wg.sum(), (wg * tg).sum(), (wg * tg * tg).sum()
    denom = sw * swtt - swt ** 2
    if denom == 0:
        return np.nan
    c = wg * (sw * tg - swt) / denom
    key = variance_row if key_variance is None else key_variance
    key = None if key is None else np.asarray(key, float)[good]
    if model.varbias_cov_factor is not None:
        # Smoothed lookups are correlated: Var_cal(D) = (1/4) c^T M M^T c.
        rows = model.var_bias_factor(cnr[good], key)
        return float(np.linalg.norm(c @ rows) / 2.0)
    s = np.asarray(model.var_bias_se(cnr[good], key), float)
    if not np.all(np.isfinite(s)):
        return np.nan
    return float(np.sqrt(np.sum(c ** 2 * s ** 2)) / 2.0)
