"""Per-frame width estimation and the MSD -> D linear fit.

This is the fitting hierarchy that the weighting scheme acts on:

1. Each spatial profile is fit with a Gaussian to estimate its variance.
2. The MSD curve, MSD(t) = sigma^2(t) - sigma^2(0), is fit linearly; the
   slope is 2 D.

A width fitted to a noisy profile scatters about the true width, so its
square, the fitted variance, exceeds the true variance on average by the
width's estimation variance. Carried through the MSD slope, that bias inflates
D. The variance is bounded below (about one pixel squared), but above a
per-frame CNR of about 1.5 the bound does not bind and plays no part in the
bias; below that, noise can throw a fit to a broad profile and the fitted
variance becomes right-skewed.

The fitting here mirrors the validated DICE engine (Thiebes & Grumstrup, JCP
2024): the variance is fit directly with a wide lower-to-upper bound, and a
fit that fails to converge falls back to a fixed broad guess rather than being
discarded -- discarding the failures would quietly delete the upper tail that
carries the bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


def _gaussian_var(x: np.ndarray, mu: float, var: float, amp: float) -> np.ndarray:
    """Gaussian parameterised by its variance (so the fit estimates sigma^2)."""
    return amp * np.exp(-0.5 * (x - mu) ** 2 / var)


@dataclass(frozen=True)
class FrameFit:
    variance: float       # fitted Gaussian variance, sigma_hat^2
    amplitude: float      # fitted peak amplitude
    at_wall: bool         # True if the variance was pinned near the lower bound
    ok: bool              # True if the optimiser converged


def _unusable_frame(x: np.ndarray, y: np.ndarray) -> FrameFit | None:
    """NaN for a profile with a non-finite pixel, else ``None``.

    Such a profile is unusable, not a failed fit: the broad fallback guess
    would turn it into a finite variance that every downstream step trusts,
    whereas NaN propagates and :func:`fit_D` drops the frame.
    """
    if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
        return None
    return FrameFit(variance=np.nan, amplitude=np.nan, at_wall=False, ok=False)


def fit_variance(x: np.ndarray, y: np.ndarray,
                 max_var: float | None = None) -> FrameFit:
    """Fit a single noisy profile with a Gaussian; return its variance.

    The variance is bounded to ``[~1 pixel^2, var_max]``.  A failed fit falls
    back to the broad initial guess (the DICE convention) so that the heavy
    upper tail of the variance distribution is preserved, not silently pruned.

    ``max_var`` sets the upper bound on the fitted variance. The default
    (``None``) uses ``span^2`` -- the full field of view, so the
    fit may return a width as large as the whole window. That is too loose when
    the output window is sized far wider than the spot: at low CNR the optimiser
    walks out to the bound on noise, the fitted-variance distribution becomes
    heavy-tailed, and any mean-based bias calibration built on it inflates and
    destabilises. Pass an explicit physical cap -- e.g. ``(G * sigma0) ** 2`` for
    a maximum credible width growth ``G = sigma_max / sigma0`` -- to bound the
    wandering near the real width. The bound must be **identical** in the
    calibration and the application fit, or the ``var_bias`` subtraction is
    mismatched.

    A profile with a non-finite pixel returns NaN (see :func:`_unusable_frame`).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if (unusable := _unusable_frame(x, y)) is not None:
        return unusable
    n = x.size
    span = float(x[-1] - x[0])

    var_min = (span / n) ** 2          # ~ one pixel squared
    var_max = span ** 2 if max_var is None else float(max_var)
    # Broad initial guess / fallback; clamp inside the bounds so an explicit
    # (tighter) max_var cannot leave p0 outside curve_fit's bounds.
    var0 = min(max((span / 4.0) ** 2, 4.0 * var_min), 0.9 * var_max)

    # Seed the centre from the peak within the central 60% of the window.
    margin = 0.3 * span
    central = (x >= x[0] + margin) & (x <= x[-1] - margin)
    idx = np.where(central)[0]
    if idx.size:
        mu0 = float(x[idx][int(np.argmax(np.abs(y[idx])))])
    else:
        mu0 = float(x[int(np.argmax(np.abs(y)))])
    amp_max = float(np.max(np.abs(y))) or 1.0
    amp0 = amp_max

    p0 = [mu0, var0, amp0]
    lower = [x[0] + margin, var_min, 0.0]
    upper = [x[-1] - margin, var_max, 2.0 * amp_max]
    try:
        popt, _ = curve_fit(_gaussian_var, x, y, p0=p0,
                            bounds=(lower, upper), maxfev=5000)
        var, amp = float(popt[1]), float(popt[2])
        ok = True
    except (RuntimeError, ValueError):
        var, amp, ok = var0, amp0, False

    at_wall = var <= 2.0 * var_min
    return FrameFit(variance=var, amplitude=amp, at_wall=bool(at_wall), ok=ok)


def fit_series_variance(x: np.ndarray, profiles: np.ndarray) -> list[FrameFit]:
    """Fit every frame in a profile sequence."""
    return [fit_variance(x, row) for row in profiles]


@dataclass(frozen=True)
class DFit:
    D: float              # estimated diffusion coefficient (slope / 2)
    intercept: float
    ok: bool
    D_se: float = np.nan  # covariance-based standard error of D (= SE_slope / 2)


def fit_D(times: np.ndarray, msd: np.ndarray, weights: np.ndarray) -> DFit:
    """Weighted linear fit of MSD(t); the slope is 2 D.

    ``weights`` are the per-point fit weights (w_i, proportional to 1/variance
    in the standard scheme).  Frames with zero or non-finite weight are
    dropped.  This is the single point where every weighting scheme enters.

    ``D_se`` is the textbook weighted-least-squares standard error of the slope,
    halved (D = slope / 2): ``Var(slope) = Sw / (Sw*Swtt - Swt^2)``. It is exact
    when the weights are the true inverse point-variances *and the points are
    independent*. The MSD points share the fitted ``sigma_hat^2(0)``, but with
    frame 0 weighted (as :func:`~width_debias.weights.msd_and_weights` does
    for the weighted schemes) the slope equals that of the independent widths,
    so the SE applies. With frame 0 at weight 0 the shared offset makes it too
    narrow (the interval ``experiments/coverage-interval/coverage_check.py``
    probes). The D-free calibration's per-frame variances are measured at
    sigma_0^2, so they understate the later frames' variance when D is large;
    combined with the calibration SE the 95 % interval covered 0.93 to 0.96 of
    runs for D <= 0.02 sigma_0^2/tau and 0.89 to 0.94 at D = 0.1
    (``experiments/dfree-calibration/slope_fit_check.py``). For OLS-style unit
    weights it omits the residual-variance scaling and is not a calibrated SE.
    """
    t = np.asarray(times, float)
    m = np.asarray(msd, float)
    w = np.asarray(weights, float)

    good = np.isfinite(t) & np.isfinite(m) & np.isfinite(w) & (w > 0)
    if good.sum() < 2:
        return DFit(D=np.nan, intercept=np.nan, ok=False)

    t, m, w = t[good], m[good], w[good]
    # Weighted least squares for a straight line m = a + b t.
    sw = w.sum()
    swt = (w * t).sum()
    swm = (w * m).sum()
    swtt = (w * t * t).sum()
    swtm = (w * t * m).sum()
    denom = sw * swtt - swt ** 2
    if denom == 0:
        return DFit(D=np.nan, intercept=np.nan, ok=False)
    slope = (sw * swtm - swt * swm) / denom
    intercept = (swm - slope * swt) / sw
    var_slope = sw / denom
    d_se = float(np.sqrt(var_slope) / 2.0) if var_slope > 0 else np.nan
    return DFit(D=float(slope / 2.0), intercept=float(intercept), ok=True,
                D_se=d_se)
