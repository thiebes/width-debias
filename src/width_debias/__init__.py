"""width-debias: correcting the noise bias in fitted profile widths.

A width fitted to a noisy profile scatters about the true width, so the fitted
variance exceeds the true one on average by the width's estimation variance,
which grows as the per-frame CNR falls. In a diffusion measurement the CNR
falls frame by frame, so the bias rises over time, leaks through the MSD fit,
and inflates the extracted diffusion coefficient D. Standard inverse-variance
weighting corrects the unequal noise across frames but cannot remove a mean
bias from a slope fit.

The package subtracts a calibrated bias ``var_bias(CNR)`` from each frame's
fitted variance, then weights by inverse variance; the MSD and D corrections
follow from the corrected widths. ``docs/DERIVATION.md`` derives the additive
form from a folded-normal model of the fitted width. The per-frame CNR (from
the ``fft-cnr`` package) is the input.

Typical use::

    import numpy as np
    from width_debias import Experiment, calibrate_d_free, evaluate
    from width_debias.fit import fit_D
    from width_debias.weights import msd_and_weights

    cnr0 = np.array([5, 8, 13, 21, 34, 55, 90])
    lookup = calibrate_d_free(cnr0)          # D-free lookup var_bias(CNR)

    # Monte Carlo bias curves <D_est>/D_ref for ols, inverse_variance,
    # subtract, and mas
    results = evaluate(Experiment(D=0.02), cnr0, lookup)

    # A measurement: fitted widths in units of sigma_0^2 and per-frame CNR
    msd, w = msd_and_weights("subtract", variance, cnr, lookup,
                             min_cnr=1.3)    # drop frames below the floor
    D = fit_D(times, msd, w).D

``subtract`` is the recommended scheme and :func:`calibrate_d_free` the
recommended calibration. The README states the conditions for real data: a
calibration that matches the data's geometry and units, and the per-frame CNR
floor.
"""
from .simulate import (AnomalousExperiment, Experiment, MultiExpExperiment,
                       cnr_per_frame, noisy_series)
from .models import (GaussianModel, GeneralizedGaussianModel, MODELS,
                     MomentModel, UnboundedGaussianModel, VoigtModel,
                     model_from_spec, spec_tag)
from .montecarlo import Ensemble, run_ensemble
from .bias_lookup import FrameStats, BiasLookup, frame_stats
from .weights import (SCHEMES, CalibrationRangeWarning, calibration_D_se,
                      corrected_variance, msd_and_weights)
from .validate import (FrameDiagnostic, SchemeResult, calibrate,
                       calibrate_d_free, calibrate_fisher, evaluate,
                       per_frame_diagnostic, sweep_stats)
from .calibration import (CalibrationCells, CalibrationGrid, calibrate_cells,
                          calibrate_cnr_sigma2)
from .fisher import LeadingOrderTerms, leading_order_terms

__version__ = "0.4.0"

__all__ = [
    "Experiment", "AnomalousExperiment", "MultiExpExperiment",
    "cnr_per_frame", "noisy_series",
    "GaussianModel", "VoigtModel", "GeneralizedGaussianModel", "MomentModel",
    "UnboundedGaussianModel", "MODELS", "model_from_spec", "spec_tag",
    "Ensemble", "run_ensemble",
    "FrameStats", "BiasLookup", "frame_stats",
    "SCHEMES", "msd_and_weights", "corrected_variance", "calibration_D_se",
    "CalibrationRangeWarning",
    "SchemeResult", "calibrate", "calibrate_d_free", "calibrate_fisher",
    "evaluate", "sweep_stats",
    "FrameDiagnostic", "per_frame_diagnostic",
    "CalibrationGrid", "calibrate_cnr_sigma2",
    "CalibrationCells", "calibrate_cells",
    "LeadingOrderTerms", "leading_order_terms",
]
