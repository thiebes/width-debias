# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-11-11

Fixes from the review of 0.4.0. The recommended path (`calibrate_d_free`, then
`msd_and_weights("subtract")`, then `fit_D`) gives the same results for valid
input: a probe of the paths the fft-cnr paper used gives 105 of 105 outputs
byte-identical to 0.4.0, and a test pins the path's outputs to the 0.4.0
values.

### Fixed

- The joint (CNR, sigma^2) and width-scaled lookups read their slope weights at
  the true width a key implies, or average them over width when no key is
  given. They read them at each frame's own fitted variance, a weight tied to
  that frame's error, which biases the slope at the order being corrected. On a
  500-run joint calibration with Gaussian fits (D 0.02 and 0.1 sigma_0^2/tau,
  initial CNR 8 to 34), corrected D/D_true was 0.39 to 0.96 with the former
  weights and is 0.88 to 1.09 (evaluation SE 0.004 to 0.12;
  `experiments/dfree-calibration/joint_weights_check.py`).
- `calibrate_cnr_sigma2` fits every cell with the model instance passed, so the
  shape exponent of a generalized-Gaussian model, a Voigt `gamma`, or a
  `max_var` is the one calibrated. The model was rebuilt from its name, which
  calibrated `GeneralizedGaussianModel(shape=1.5)` as p = 2. Passing
  `model_kwargs` keeps the former rebuild.
- `calibrate_cnr_sigma2` keeps every fit in a cell's bias mean, as the
  application keeps a failed fit's fallback value. Pass
  `filter_threshold=5 * variance_grid.max()` to reproduce a grid built with the
  former default.
- Calibration profiles (`calibrate_cells`, `calibrate_cnr_sigma2`) and the
  leading-order Fisher terms are placed at the pixel grid's midpoint. On a grid
  that does not run from minus its end to its end (one from 0 to 12), the
  calibration profile had a quarter of the true variance and the Fisher terms
  were centred on the grid's edge, so both gave a wrong bias. `per_frame_crlb`
  and `sequence_fisher_information` place the profile at the experiment's
  centre (`Experiment.mean`), which they took as 0.
- A profile with a non-finite pixel or grid point gives a NaN variance, an
  unusable frame, in every model, not the broad fallback width.
- `BiasLookup.load` accepts the path given to `save` (`save("cal")` writes
  `cal.npz`, and a directory named `cal` does not shadow it), and every
  calibration loader closes its archive; an open archive kept the file locked
  on Windows.
- `calibration_D_se` raises for a width-aware lookup given neither
  `variance_row` nor `key_variance`, where it took the SE from the
  width-averaged grid.

### Added

- `msd_and_weights(..., min_cnr=...)` sets the weight of frames below a
  per-frame CNR to zero, the floor the validations apply.
- `CalibrationRangeWarning`: `msd_and_weights` warns when a frame it keeps lies
  outside the lookup's calibrated CNR range, where the lookup takes its edge
  value.
- `calibrate_d_free` records the calibration's pixel grid, profile model, and
  lowest CNR (`pixel_x`, `model_spec`, `lowest_calibrated_cnr`).
- `BiasLookup.implied_true_variance` gives the true width an observed fitted
  variance or key implies, the point at which `var_bias`, the slope weights,
  and the calibration SE are read; `BiasLookup.is_width_aware` says whether a
  width argument changes the lookup.
- `BiasLookup` archives carry a format version (`format_version`). `load`
  refuses a newer format, and warns when a 1-D lookup has no `framevar_grid`
  (its weights then fall back to the MSD-point variance).
- `per_frame_cnr` passes extra keyword arguments to `fft_cnr.fft_cnr` (for
  example `roi`).
- The package root exports `fit_D`, `DFit`, `per_frame_cnr`, `WhiteNoise`, and
  `ShotNoise`.

### Changed

- `BiasLookup.cnr_floor` is `lowest_calibrated_cnr`, so it is not mistaken for
  the correction floor (`min_cnr`); `load` reads archives that use either name.
- `msd_and_weights` takes one realisation and refuses a 2-D batch, which it
  differenced against the batch's first row, and a `key_variance` of another
  shape than `variance_row`.
- `calibrate_cnr_sigma2` refuses a `noise_model` other than `"gaussian"` or
  `"poisson"` (any other value ran white noise), and its default worker count
  follows `WIDTH_DEBIAS_WORKERS` like `run_ensemble` (it used every core but
  one).
- `scripts/build_calibration.py` keeps every fit in the bias mean unless
  `--filter-threshold` is given, and passes the configured model to
  `calibrate_cnr_sigma2`. The NC-solid calibration preset passes the filter it
  ran with, so a rebuild reproduces its grid.
- CI runs on `ubuntu-24.04` with setup-uv v10.3.2 and the PyPI publish action
  v1.14.3, installs the package non-editable for the tests, and fails a
  release whose tag does not match `__version__`.

### Documentation

- Docstrings, docs, and experiment READMEs describe the bias as the noise bias
  of the fitted width, its estimation variance, not a truncation or wall
  effect.
- The tests check what the release notes promise: the exact subtraction and
  calibrated bias, the recommended path's outputs against 0.4.0, the saved
  weights, worker-count independence, failed fits kept in the bias, and
  closed archives. They import the installed package.

## [0.4.0] - 2026-11-06

The first release.

### Added

- `BiasLookup.frame_variance` (with `framevar_grid` in cached 1-D
  calibrations) gives the per-frame width variance used for the slope weights.
- `width_debias.fisher` computes the leading-order bias of the fitted variance,
  the width's Fisher-information variance, from the profile model and pixel
  grid alone (`leading_order_terms`, `var_of_fitted_variance` with Cramer-Rao
  or least-squares-sandwich forms). The whole-sequence Fisher matrix and
  `crlb_D` move here from `experiments/crlb-precision`.
- `calibrate_fisher` builds a D-free 1-D calibration from that exact leading
  term and a P-spline smooth of the higher-order residual over single-frame
  Monte Carlo cells (`calibrate_cells`, `CalibrationCells`,
  `BiasLookup.from_cells`). The smoothed lookup carries the sampling
  covariance of the smooth as a factor (`BiasLookup.var_bias_factor`,
  `var_bias_cov`), and `calibration_D_se` uses it. With `width_scaling=True` a width table rescales
  the bias to each frame's width (it grows as sigma at a fixed pixel grid and
  peak CNR). It is not yet the recommended path.
- `msd_and_weights`, `corrected_variance`, and `calibration_D_se` take an
  optional `key_variance`, a measurement-level width key. A 1-D model with a
  width table scales its weights only with such a key; a joint (CNR, sigma^2)
  model keys its weights on each frame's own fitted variance.
- `BiasLookup.save` and `BiasLookup.load`.

### Changed

- The package is named for what it corrects, the noise bias in each fitted
  profile width, from which the MSD and D corrections follow: the distribution
  `folded-debias` is `width-debias` (import `width_debias`), the
  `folded_normal` scheme is `subtract`, the `truncation` module and
  `TruncationModel` are `bias_lookup` and `BiasLookup`, and
  `FOLDED_DEBIAS_WORKERS` is `WIDTH_DEBIAS_WORKERS`. In the regime where the
  correction holds, the fit's lower bound does not bind, and the bias is the
  width's estimation variance; the folded-normal model is one derivation of it.
- The weighted schemes weight every frame, t = 0 included, by
  `1 / Var(sigma_hat_i^2)`. The MSD fit is then the same as fitting the
  corrected widths against time and reaches the Cramer-Rao bound on D (0.98 to
  1.08 x sqrt(CRLB) from initial CNR 21 on four profile shapes). A 1-D model
  saved without `framevar_grid` keeps the MSD-point weights with frame 0 at
  weight 0.
- The `cnr` extra requires `fft-cnr>=0.3.3`, the first release that refuses
  degenerate generalized-Gaussian fits below CNR 1.
- `license` uses the SPDX expression form (`"MIT"`) with `license-files`.

### Documentation

- The README gives the installation from PyPI, a worked example on a
  measurement, and the conditions for real data: a calibration that matches the
  data's geometry and units, the per-frame CNR floor, and the narrower range
  over which an estimated CNR holds.
- `docs/DERIVATION.md` states the leading-order bias as growing with sigma at a
  fixed pixel grid and peak CNR, matching `width_debias.fisher`.

## [0.3.0] - 2026-10-21

Not tagged or published; 0.4.0 is the first release. The code at commit
2836e6c, on which the fft-cnr paper's cases 27 to 30 were first run.

### Added

- Profile models (`GaussianModel`, `VoigtModel`, `GeneralizedGaussianModel`,
  `MomentModel`, `UnboundedGaussianModel`), the Monte Carlo ensemble
  (`run_ensemble`), and the MSD slope fit (`fit_D`, with the covariance
  standard error `DFit.D_se`).
- Simulation beds: `Experiment`, `AnomalousExperiment` (subdiffusive spreading),
  and `MultiExpExperiment` (multi-exponential decay), with the noise laws in
  `folded_debias.noise` (`WhiteNoise`, `ShotNoise`).
- `evaluate` (bias curves per scheme over a CNR sweep), `per_frame_diagnostic`,
  and `cnr.per_frame_cnr` (per-frame CNR from measured profiles through
  `fft-cnr`).
- Four slope-fit schemes in `msd_and_weights`: `ols`, `inverse_variance`,
  `folded_normal`, and `mas`. `folded_normal` subtracts the calibrated per-frame
  variance bias `var_bias(CNR)` and weights each MSD point by its inverse
  variance, with frame 0 at weight 0.
- `calibrate_d_free`, the D-free 1-D calibration: the standard sweep at D = 0,
  so the lookup is keyed on per-frame CNR alone. `calibrate_cnr_sigma2` builds
  the experimental joint (CNR, sigma^2) lookup.
- `calibration_D_se` (in `weights`) propagates the calibration's own Monte Carlo
  error into the corrected D, through the slope fit's linear map.
  `TruncationModel.var_bias_se` gives the standard error of each looked-up bias,
  `frame_stats` records it, and `SchemeResult.cal_se` reports it per cell. The
  propagated value matches the scatter across 16 independent calibrations
  (observed/predicted 0.98).
- `model_from_spec` and `spec_tag` build a profile model from a string such as
  `"generalized_gaussian:1.5"`, so scripts and worker processes can name the
  generalized-Gaussian shape exponent.
- Packaging: the version is defined once, in `folded_debias.__version__`;
  published dependencies carry compatible upper bounds, and `uv.lock` records
  the environment that produced the committed results. MIT license.

### Documentation

- `TruncationModel.var_bias` and `calibrate_cnr_sigma2` document the choice of
  key for the joint (CNR, sigma^2) lookup: keying each frame on its own fitted
  variance under-corrects when the width grows several-fold over the
  measurement.
