# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Experiment.spread_dims` lets the simulated peak fall as the profile
  spreads, at a population that changes only by the decay: 1 for a profile
  summed over the other axis, 2 for a cut through the centre of a spot that
  also spreads along y (`D_y`, default `D`). For the Voigt profile the peak
  depends on `psf_hwhm`, which must equal the `VoigtModel`'s `gamma` (0 for
  every other model); every built-in model's `noise_free_series`,
  `run_ensemble` and the Fisher bounds refuse a mismatch
  (`models.check_spreading_peak`). The three fields are keyword-only, so the
  positional fields of `AnomalousExperiment` and `MultiExpExperiment` keep
  their order, and settings the peak cannot honour (`spread_dims=True`, a
  negative `D_y`, `D_y` without a centre cut, `psf_hwhm` without
  `spread_dims`) raise `ValueError`. The default (`None`) keeps the peak at
  the decay alone, so every output is unchanged. The option leaves the peak
  unchanged at t = 0 and at every t when D = 0, and draws the same white noise
  from the same seed. `Experiment.decay_amplitude` and
  `Experiment.spread_factor` give the two factors of the peak (a subclass with
  its own decay law overrides `decay_amplitude`, as `MultiExpExperiment`
  does), and with the option on `cnr_per_frame` returns the CNR of the
  simulated frames.
- `experiments/dfree-calibration/validate_d_zero.py` takes `--spread-dims`,
  to evaluate with a peak that also falls as the profile spreads, and
  `--cal-runs`, to set the calibration runs per CNR_0 point apart from the
  evaluation runs. A falling peak takes the late frames to a lower per-frame
  CNR than the D = 0 calibration reaches, so each window names, for each
  `--spread-dims`, CNR_0 points that extend its calibration grid down (for
  the short window CNR_0 1.8 for a summed profile and 1.6 for a centre cut),
  and the script stops before calibrating if a frame kept at CNR >= 1 would
  still lie below the calibrated range. Each run saves its cell statistics
  and its lookup as `.npz` files, and `--report` rebuilds a report from saved
  statistics (a test rebuilds the committed ones). Reports print the SEs and
  the spread to five decimals and the mean to four, so a large calibration's
  SE at large D is not printed as 0.000. The falling-peak tables for both
  windows (summed profile, 5000-run calibration) are committed with their
  statistics and lookups. The corrected
  fit (CNR >= 1) has 6 of 140 cells beyond 2 combined SE in the standard
  window and 7 of 96 in the short window, and its pooled bias is within 0.8
  combined SE on every shape; the naive all-frames fit has 45 of 140 and 43 of
  96. With the decay alone, at 300 and 2000 calibration runs, the corrected
  fit has 4 of 140 and 5 of 96 and the naive fit 46 of 140 and 41 of 96; the
  smaller calibration gives the corrected fit a larger combined SE, so its
  counts are not a like-for-like comparison.
- `experiments/halide-mixing-brI-legacy/rerun/collapse_diffmaps.py` collapses an
  averaged 2-D map of the high-N Br-I re-run into the 1-D profile file that
  `experiments/halide-mixing-highn/` reads. It refuses to replace an existing
  profile file, to collapse a map whose frame count is not tmax / tsamp + 1,
  and to record instance counts that the map does not hold (missing instances,
  copies of one realisation, or a variance file that is not their mean). With
  `--check` it compares chosen frames with an existing profile file, and the
  averaged variance file with the kept instances' variance files. The script
  that made the profile files is not kept, and `collapse_check.txt` shows that
  the averaged maps that are kept, from a first run with only 5 distinct
  realisations per fraction, do not reproduce them, so the script's profiles
  cannot be verified against them. `tests/test_collapse_diffmaps.py` pins the
  file's keys, axes and grid, the refusals and the variance comparison on small
  hand-built instances averaged by `average_diffmaps.py`. The top-up scripts
  `rerun_unique.sh` and `rerun_all_unique.sh`, which brought the re-run to 16
  distinct realisations per fraction, are added, `rerun_unique.sh` with one
  comment reworded. The rerun README and the high-N `DATASETS.md` describe the
  full chain and which of its inputs are not kept.
- `experiments/halide-mixing-highn/axis_check.py` compares the x and y
  projections of the high-N Br-I maps with the halved 2-D moment that the
  experiment takes as its reference, and writes `axis_check.txt`: per Br
  fraction and at seven frames, each projection's centroid, binned second
  moment, excess kurtosis and window-edge fraction; the extremes over all
  frames with their times; and, over the usable band at CNR0 15, the D/D_ref
  of each axis from the noise-free fit and from the binned moments, with the
  fit checks. With M the binned second moment of each projection about its
  own centroid, on the bin centres, and V the walkers' 2-D moment, M/(V/2) is
  within 0.0123 of 1 on each axis and M_x/M_y within 0.0143 of 1; walker sampling alone gives about 0.003, a
  lower bound, since the 16 disorder realizations add scatter the averaged
  files cannot show, so the table does not test isotropy. (M_x + M_y)/V
  departs from 1 by up to 0.0108, growing with time. The noise-free fit of
  `prof_y` gives D/D_ref 1.009, 0.992 and 0.984 against 0.992, 0.958 and 0.960
  for `prof_x`; the binned moments, whose D is within 0.006 of D_ref on both
  axes, account for 0.004, -0.001 and 0.012 of that difference and the
  profile shape for the rest, and these files cannot
  tell whether it comes from the transport or from the finite set of
  realizations. The loader still reads `prof_x`, and no result of the
  experiment changes. Its docstrings had stated without a table that the axes
  agree to within one part in a hundred: the projections differ by up to
  0.0143, more than that, and the isotropy of the transport is not tested.
  They had also given the binned moment's offset from the reference to bin
  width alone; part of it is the files' x axis, whose spacing is 600/599 of
  the bin width. Both now give the table's values.

### Changed

- Experiment scripts read collaborator data from a data root instead of fixed
  shared-drive paths: the folder `WIDTH_DEBIAS_DATA_ROOT` names (`~` and
  relative paths are resolved), or else `data/` at the repository root, which
  git ignores. `scripts/collaborator_data.py` holds the folders
  (`stroboscat_scans/`, `simulated_map/`, `halide_mixing/`,
  `halide_mixing_highn/`, `cspbbr_sample_1/`). The loaders in `nc-solid`,
  `halide-mixing-brI-legacy`, `halide-mixing-highn` and `wse2-strobescat` read
  their folders there, and `cspbbr-strobescat/run_all.py` takes its
  `--sample-root` default from it. `nc_data.load_case` and `load_all` take a
  keyword-only `map_dir` that overrides the folder. With each folder mapped to
  its old location the scripts resolve the same folders, files and keys in the
  same order, so no result changes. When data is missing, the scripts name the
  folder they searched and the variable that sets it. Each set's `DATASETS.md`
  and README and the repository map name the data-root folders, and
  `.gitignore` ignores `data/`. `tests/test_data_root.py` checks how the root
  resolves, the folder names, and that no tracked text file names a drive
  path.

### Fixed

- `experiments/halide-mixing-brI-legacy/rerun/run_instances.sh` starts instance k
  after k times a gap (2 s by default, an optional fifth argument), because the
  simulator seeds from the wall clock at one-second resolution. Its random 0 to
  4 s stagger gave at most five distinct realisations when the instances
  started together. When the instances are done, it lists one instance per
  distinct complete variance file, by hash, in `unique_instances.txt`, as the
  top-up scripts do, and names the incomplete ones. When fewer than asked are
  distinct it prints the `rerun_unique.sh` command first, and it prints the
  averaging command only for distinct instances; its closing hint averaged
  every instance, copies included. It also exports the walker count to the
  instances, which it did not: each instance's parameter file got an `N` line
  with no value unless the caller had exported it. It refuses a folder that
  already holds `inst_*` folders, a gap that is not a whole number of seconds,
  and a parameter file that does not give both tmax and tsamp.
  `tests/test_run_instances.py` runs it with a stub simulator.
- `experiments/halide-mixing-highn/br_halide_highn_validation.py`, the test of
  the 1-D D-free lookup on the high-N halide bed, caps the fitted width at 12
  sigma_0 = 300 nm in the calibration and in the application, as the rest of
  the set does. With the fit's default bound, the 3000 nm window, low-CNR
  calibration fits wandered toward the window, the lookup's bias reached about
  340,000 nm^2 at per-frame CNR 1, and the corrected arm collapsed. The script
  also leaves out frames below per-frame CNR 1.3, divides each arm's D by the
  noise-free fit's D over the same frames, gives SEs of D and of the residuals
  from 200 seeds and from 200 draws of the lookup's own error, prints an arm
  whose power-law exponent ends at a fit bound as "at bound", and counts
  failed fits. A residual is a frame's fitted variance, averaged over the
  seeds, minus the variance fitted to the same frame's noise-free profile,
  averaged over a band of frames. The exponent bounds are defined once, as
  `ALPHA_BOUNDS` beside `fit_msd_powerlaw` in
  `experiments/halide-mixing-brI-legacy/br_halide_validation.py`, and the
  at-bound marks read them there. It adds the corrected fit restricted to
  per-frame CNR >= 3 beside the naive one, and its header gives the lookup at
  sigma_0, with its propagated SE, next to its leading term, the width's
  Fisher variance on the calibration grid. It applies the correction through
  `corrected_variance` and
  caches the lookup with `BiasLookup.save`, under a name that carries the run
  count. Its table is committed as `br_halide_highn_validation.txt`. The set
  README said this lookup under-corrects on the bed because the width grows
  26- to 61-fold; the table shows that, read down to the floor, it
  over-subtracts. Between per-frame CNR 1.3 and 3 its residual is -4066 to
  -3548 nm^2 at initial CNR 15 and 20, and its exponent sits at the fit's 0.2
  bound on 4 of those 6 cells, because the lookup at sigma_0 is 14 to 132
  times its leading term between per-frame CNR 3 and 1.3. Kept to per-frame
  CNR >= 3, it gives D at 0.96 to 1.02 of the noise-free fit's.
  `br_halide_highn_perframe.py` imports the cap, the bounded fit and the
  per-frame CNR floor from it, with unchanged fits.

### Documentation

- The README's "On real data" list and the `calibrate_d_free` and
  `calibration_D_se` docstrings say that the calibration's own error is shared
  by every measurement made with a lookup and does not average down. A mean of
  N measurements carries the full calibration SE, combined once in quadrature
  with the statistical error of the mean (the mean of the per-measurement
  values is an upper bound), and the calibration runs per initial-CNR point
  should be well above N. The D-free calibration README gives the calibration
  SE against the single-measurement spread of D divided by the square root of
  the calibration runs, summed over the cells of each table: at most about 1,
  close to 1 at the smallest D and lower where the width grows more, since the
  lookup's error does not grow with the width, and lower again (0.80 to 0.91)
  with a peak that also falls as the profile spreads, which moves the frames
  between the calibrated CNRs. The `--n-runs` entry of
  `docs/calibration_runs.md` says that the joint lookup's error is shared in
  the same way and links the README, whose numbers are for the 1-D lookup.
  `docs/DERIVATION.md`, `docs/REPO_MAP.md` and the D-free calibration README
  say the calibration SE is at most about the evaluation SE at equal runs,
  where they said about equal, and the first two say the calibration runs
  should be well above the number of measurements averaged, where they said
  exceed.
- The high-N halide README, `br_halide_highn_perframe.py`,
  `key_sensitivity.py`, `docs/DERIVATION.md` and the `BiasLookup.var_bias`
  docstring give the width growth over the fitted frames, 11- to 24-fold at
  initial CNR 15 and 20 (13- to 24-fold at 20), where they said about 64-fold.
  The README and `br_halide_highn_perframe.py` say that at a given CNR the
  bias grows in proportion to sigma to leading order, where they said with
  sigma^2. The README, the per-frame caption, `br_halide_highn_perframe.py`,
  `key_sensitivity.py` and the opening of the `render_perframe_figure.py`
  docstring call the offset between the noise-free fit and the carrier
  reference the shape gap: these profiles are linear in the carrier density,
  so there is no emission-to-carrier gap. The per-frame figure's legend, its
  table line and the colour note that describes them keep the label "carrier
  gap". The set README gives the 1-D lookup's result its own section and says
  that the bounded width fit, its cap and the per-frame CNR floor live in
  `br_halide_highn_validation.py`, which `br_halide_highn_perframe.py`
  imports and the set's other scripts read through it.
- Scripts that read an external simulation study's result files
  (`experiments/fft-cnr-recovery/build_fft_cnr_recovery_figure.py`,
  `experiments/dfree-calibration/calibration_se_check.py` and
  `scripts/poster_figures.py`) take the directory from
  `WIDTH_DEBIAS_EXTERNAL_RESULTS`, which has no default, and name the file and
  the directory when a file they need is missing. `scripts/poster_fftcnr_dense.py`
  writes its sweeps as `output/gengauss_cnr_sweep_{dense,shot}.npz`, the names
  `scripts/poster_figures.py` reads.
- The fft-cnr recovery summary and figure are rebuilt from fft-cnr 0.4.0
  estimates: in every (N, CNR) cell the median over 1000 trials lies between
  0.989 and 1.006 of the true CNR. The summary records the fft-cnr version and
  the SHA-256 of its source array, and the build script checks that hash
  whenever the source is available.
- `tests/test_quoted_numbers.py` recomputes validation numbers that the
  README, `docs/DERIVATION.md`, `docs/REPO_MAP.md`, set READMEs, captions and
  library docstrings restate, from the committed tables that produced them,
  and checks each copy it lists; a copy it does not list is not checked. It
  also keeps superseded values out of tracked documents other than the
  changelog. A rerun that changes a table fails the test until each listed
  copy, and its snippet in the test, are updated.
- The high-N halide README, `docs/DERIVATION.md` and the `BiasLookup`
  docstring give the measurement-level (smoothed) key's agreement with the
  exact key as within 0.004 from initial CNR 15, and the iterated key's as
  within 0.002, as `key_sensitivity.txt` gives them; they said 0.003.
- The `bias_lookup` module docstring, `BiasLookup.var_bias`, the
  `calibrate_cnr_sigma2` docstring, `docs/calibration_runs.md` and
  `docs/DERIVATION.md` say that keying the joint (CNR, sigma^2) lookup on each
  frame's own fitted variance over-corrects when the width grows several-fold,
  as `key_sensitivity.txt` shows: at initial CNR 15 and 20 on the two
  slower-spreading high-N halide maps, that key leaves D at 0.892 to 0.957 of
  the reference, against 0.972 to 0.985 with the exact key, and its residuals
  below CNR 2 are negative. They said it under-corrects. `BiasLookup.var_bias`
  and `docs/DERIVATION.md` give that key's D on the fastest-spreading map as
  0.99 to 1.00, where they said 0.99, and the high-N halide per-frame caption
  names the own-width key as the exception to D not depending on the key from
  initial CNR 15.
- The validation numbers in the README, `docs/DERIVATION.md`, and the
  docstrings of `calibrate_d_free` and `msd_and_weights` come from the
  regenerated experiment sets, whose seeds are independent per profile shape:
  4 of 140 cells beyond two combined standard errors, keeping frames with
  per-frame CNR >= 1 (5 of 140 with every frame), in place of the 4 of 96 the
  `calibrate_d_free` docstring quoted from an earlier run on seeds shared
  across shapes; 0.95 to 1.10 x sqrt(CRLB) from initial CNR 21 (formerly 0.98
  to 1.08); and a per-profile CNR floor of 0.89 to 1.51 depending on shape.
- `docs/DERIVATION.md` builds the model on the fitted width's estimation
  variance; the bounded and unbounded fits agree above a per-frame CNR of
  about 1.5, and the bound itself carries negligible mass above about 1 (the
  `fit` module docstring says the same). Its
  floor section counts the lookup's own error and sets the tolerance
  relative to the end-point MSD.
- The README, `docs/DERIVATION.md`, and the `calibrate_d_free` and
  `UnboundedGaussianModel` docstrings state the floor rule in full, the
  criterion (two combined standard errors) behind the deciles where `subtract`
  reads low, the frame filter behind "4 of 140", and the unbounded fit's mean
  and spread at per-frame CNR 0.52 as the regenerated tables give them.
- The experiment tables and figures name the corrected arm `subtract`, the
  scheme's name, instead of "folded-normal". Regenerated with the current
  library, their numbers are unchanged except in the unbounded-fit comparison
  and the noise-law tables.
- The unbounded-fit comparison and the noise-law tables use the per-frame slope
  weights in their corrected arms: half the former spread at initial CNR 34 in
  the unbounded-fit comparison, and under shot noise with a matched
  calibration a mean error of 0.025 against 0.046. The read-floor figure places
  read 0 at its true position on a symmetric-logarithmic axis.
- The coverage check runs 13 initial CNR values at 8000 runs each, with
  standard errors and pooled values printed by the script. The `fit_D`
  covariance interval covers 0.944 at D = 0.02 sigma_0^2/tau (nominal 0.95),
  because the D-free lookup's per-frame variances are calibrated at the
  initial width while the width grows, as the `fit_D` docstring notes. A
  parametric bootstrap with 2000 resamples and per-frame variances measured
  over the ensemble covers 0.951; its resampled D is exactly normal, so its
  coverage tests those per-frame variances, not the interval's form. The
  narrowness is pooled as a plain mean over the cells whose every frame has
  per-frame CNR >= 1 (0.967 of the realised spread), beside the coverage
  measured in those cells (0.942).
- The method comparison commits its table (`method_comparison.txt`), and its
  figure keeps the legacy D-aware calibration, which its caption states. Its
  script prints both corrected schemes' mean |bias - 1| and their largest
  per-cell difference from initial CNR 8 up, instead of naming the lower one,
  whose order depends on the draw, and its docstring matches the table.
- The nc-solid table header says that frames below the per-frame CNR floor
  are dropped from the fit, where the Mn table only scores above it. The
  read-floor script names its white-calibration and white-noise results
  apart, and the unbounded-fit script gives its correct run path.
- The Mn-trapping bed names its reference as the emission-weighted PL second
  moment, not the carrier moment, and its generalized-Gaussian arm as p = 2.
  Where every frame has per-frame CNR >= 1.8, the raw and corrected widths lie
  within 0.0053 square micrometres of the noiseless fit, so the bed bounds the
  correction's error rather than resolving it.
- The p = 2 proof-of-concept figure and table from 2026-06 are removed: a
  Gaussian profile under the generalized-Gaussian fitter's tighter
  `var_max`, run with the earlier labels and the legacy calibration path,
  and superseded by the p = 1.5 and p = 4 runs.
- The `MomentModel` docstring no longer calls the moment estimator unbiased to
  leading order: it is a ratio of two sums linear in the data, taken about an
  estimated centroid, so its bias is of order 1/CNR^2, the same order as a
  fitted width's.

## [0.4.1] - 2026-11-11

Fixes from the review of 0.4.0. The recommended path (`calibrate_d_free`, then
`msd_and_weights("subtract")`, then `fit_D`) gives the same results for valid
input: a probe of that path and the calls around it gives 105 of 105 outputs
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
2836e6c, on which an external simulation study first ran.

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
