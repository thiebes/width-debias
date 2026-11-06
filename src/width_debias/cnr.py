"""Per-frame CNR estimation -- the input the whole weighting scheme rests on.

For Monte Carlo validation the per-frame CNR is known exactly (it is set by the
simulation; see ``Ensemble.cnr_frame`` from
:func:`width_debias.montecarlo.run_ensemble`).  For *real*
measurements it must be estimated from each spatial profile, with no repeated
frames and no separate background region available.  That is exactly what the
``fft-cnr`` package does: it separates slowly varying signal from rapid noise
in the spatial frequency domain and returns a CNR estimate per 1-D profile.

This module wraps ``fft-cnr`` behind a single function and degrades gracefully
when the package is not installed, so the simulation-only path has no hard
dependency on it. Estimate the CNR on a window of about +/-4 sigma_0 around the
spot: background and fringe structure in a full line cut corrupts the spectral
estimate at low signal.

The calibration is keyed on the exact CNR, and the estimate is not an unbiased
stand-in for it near the floor: it reads slightly low at CNR 1 to 2 and about
twice the exact value below CNR 0.5, and its error is correlated with the
frame's width error. Keying the correction on the per-frame estimate is
therefore reliable only where every fitted frame stays well above the floor
(CNR_0 about 34 at D = 0.02 sigma_0^2/tau in the reference geometry). See
``experiments/wse2-strobescat/README.md`` for the domain and its evidence.
Use fft-cnr 0.3.3 or later (the first PyPI release with commit 69cd361), which
refuses degenerate generalized-Gaussian fits below CNR 1 instead of reporting
runaway CNRs.
"""
from __future__ import annotations

import numpy as np

try:  # optional dependency -- only needed for real experimental data
    from fft_cnr import fft_cnr as _fft_cnr
    HAVE_FFT_CNR = True
except ImportError:  # pragma: no cover - exercised only without the package
    _fft_cnr = None
    HAVE_FFT_CNR = False


def per_frame_cnr(profiles: np.ndarray,
                  fit_model: str = "generalized_gaussian",
                  **fft_cnr_kwargs) -> np.ndarray:
    """Estimate the CNR of every row of ``profiles`` with ``fft-cnr``.

    The rows are passed as given: window them to about +/-4 sigma_0 around the
    spot first (slice the stack, or pass fft-cnr's ``roi``), as the validated
    real-data path does (module docstring).

    Parameters
    ----------
    profiles : array, shape (n_frames, n_pixels)
        Stack of 1-D spatial profiles, one per time point.
    fit_model : str
        The ``fft-cnr`` amplitude estimator. Defaults to
        ``"generalized_gaussian"`` because this project's primary profiles are
        Voigt and generalized-Gaussian, and the amplitude (hence CNR) must be
        estimated without shape bias. A fixed Gaussian template -- or fft-cnr's
        default peak method -- is accurate on a true Gaussian but biased on the
        heavy-tailed / flat-topped shapes that bracket Voigt and
        generalized-Gaussian: the fft-cnr shape-sensitivity study (case 3) puts
        a Gaussian fit 0.05-0.17 of the amplitude off across that kurtosis
        range while the generalized-Gaussian fit stays within ~0.01, with the
        Gaussian falling
        out as the p=2 special case. Because the truncation bias scales as
        ~A/CNR^2, a biased CNR feeds straight into a biased correction, so the
        shape-robust estimator matters here. The generalized-Gaussian fit
        returns a finite confidence interval (it falls back to the peak method
        if the fit fails), so there is no missing-CI pathology.
    **fft_cnr_kwargs
        Passed to ``fft_cnr.fft_cnr`` for every row (for example ``roi``,
        ``window`` or ``rng``).

    Returns
    -------
    cnr : array, shape (n_frames,)

    Raises
    ------
    ImportError
        If the optional ``fft-cnr`` dependency is not installed.  Install it
        with ``pip install width-debias[cnr]``.
    """
    if not HAVE_FFT_CNR:
        raise ImportError(
            "fft-cnr is required for CNR estimation from real data; "
            "install it with `pip install width-debias[cnr]`."
        )
    profiles = np.atleast_2d(np.asarray(profiles, float))
    return np.array(
        [float(_fft_cnr(row, fit_model=fit_model, **fft_cnr_kwargs).cnr)
         for row in profiles])
