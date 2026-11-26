# width-debias

**Correction of the noise bias in fitted profile widths, for
mean-squared-displacement (MSD) diffusion measurements.**

A width fitted to a noisy profile scatters about the true width, and squaring
it to form the variance turns that scatter into a bias. On average the fitted
variance exceeds the true one by the width's estimation variance,
`E[sigma_hat^2] = sigma^2 + Var(sigma_hat)`. To leading order Var(sigma_hat) is
the Cramér-Rao value, which for a Gaussian profile at a fixed pixel spacing `dx`
is `(2/sqrt(pi)) sigma dx / CNR^2`, so the bias grows as the contrast-to-noise
ratio (CNR) falls. In a transient measurement the CNR falls frame by frame as
the population decays, so the bias rises over time, tilts the MSD curve upward,
and inflates the diffusion coefficient D. In simulation (Gaussian profile,
D = 0.02 sigma_0^2/tau, initial CNR 34) an unweighted fit returns 1.20 times the
reference D. Inverse-variance weighting corrects the unequal noise across
frames but cannot remove a mean bias present in every point.

The package corrects each profile's fitted width, and the MSD and D follow. It
subtracts a Monte Carlo calibrated bias `var_bias(CNR)` from each frame's fitted
variance before the slope fit, keyed on the per-frame CNR; in the same
simulation the corrected fit returns 1.01 times the reference. The calibration
is built at D = 0, so it does not require a prior value of D. Removing the
fit's lower bound on the width does not help: above a per-frame CNR of about
1.5 the bounded and unbounded fits are identical, and both carry the full bias.
Below a per-frame CNR of about 1 the fits spread into a broad upper tail and the
correction no longer holds (see [On real data](#on-real-data)). The derivation
of the additive form is in
[`docs/DERIVATION.md`](https://github.com/thiebes/width-debias/blob/v0.4.1/docs/DERIVATION.md).

## Install

```bash
pip install width-debias            # core: NumPy and SciPy
pip install "width-debias[cnr]"     # adds fft-cnr, for per-frame CNR from measured profiles
```

To run the experiments and tests, install from a clone of the
[repository](https://github.com/thiebes/width-debias):

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e .[dev]           # core + matplotlib + pytest + fft-cnr
```

`uv.lock` in the repository records the exact environment that produced the
committed results.

## Quick start

```python
import numpy as np
from width_debias import Experiment, calibrate_d_free, evaluate

cnr0 = np.array([5, 8, 13, 21, 34, 55, 90])
lookup = calibrate_d_free(cnr0, n_runs=300)       # D-free lookup var_bias(CNR)

exp = Experiment(D=0.02)                          # sigma_0^2/tau; L_D/FWHM_0 ~ 0.085
results = evaluate(exp, cnr0, lookup, n_runs=300)

r = results["subtract"]
se = np.hypot(r.spread / np.sqrt(300), r.cal_se)  # evaluation + calibration SE
print(r.mean_bias, se)                            # <D_est>/D_ref per CNR_0
```

On a measurement, the per-frame CNR comes from `per_frame_cnr` (which wraps
`fft-cnr`), the corrected MSD and weights from
`msd_and_weights("subtract", ...)`, the slope from `fit_D`, and the
calibration's contribution to the uncertainty from `calibration_D_se`:

```python
import numpy as np
from width_debias import (calibrate_d_free, calibration_D_se, fit_D,
                          msd_and_weights, per_frame_cnr)

# times (n_frames,); profiles (n_frames, n_pixels) cut to about +/-4 sigma_0
# around the spot; variance (n_frames,) the fitted widths in px^2;
# sigma0_px the t = 0 width in pixels
n_pixels = profiles.shape[1]
lookup = calibrate_d_free(cnr0, half_window=(n_pixels / 2) / sigma0_px,
                          n_pixels=n_pixels)     # the data's grid, in units of sigma_0
cnr = per_frame_cnr(profiles)
msd, w = msd_and_weights("subtract", variance / sigma0_px**2, cnr, lookup,
                         min_cnr=1.3)            # drop frames below the floor
fit = fit_D(times, msd, w)
D = fit.D * sigma0_px**2                         # px^2 per unit time
se = np.hypot(fit.D_se, calibration_D_se(times, msd, w, cnr, lookup)) * sigma0_px**2
```

## On real data

The validation keys the lookup on the exact per-frame CNR, uses a calibration
built for the same geometry as the data, and drops frames below the floor. On a
measurement each of these is a step for the user. The library records the
calibration's geometry but does not check it against the data.

- **Match the calibration to the data.** `var_bias` is a variance in the length
  unit of the calibration, and its size depends on the pixel grid, the profile
  model and its fit bound, and the noise law. Work in units of the measured
  t = 0 width sigma_0: divide the pixel axis by sigma_0 and the fitted
  variances by sigma_0^2, build the calibration with the data's pixel count and
  half-window in those units
  (`calibrate_d_free(cnr0, half_window=(n_pixels / 2) / sigma0_px, n_pixels=n_pixels, model=..., noise_model=...)`),
  and multiply the fitted D and its SE by sigma_0^2 at the end. A calibration in
  units of sigma_0 applied to widths in micrometres subtracts a bias that is
  off by the square of the unit ratio (16 for sigma_0 = 0.25 μm).
- **Use enough calibration runs.** The lookup's own Monte Carlo error is
  shared by every measurement made with it, so it does not show in their
  scatter and does not average down. For one measurement, combine
  `calibration_D_se` with the fit's SE in quadrature, as in the example above.
  For the mean of N measurements made with one lookup, combine the calibration
  SE once, in quadrature, with the statistical error of the mean, not divided
  by the square root of N; the mean of the per-measurement `calibration_D_se`
  values is an upper bound on it. The quick start does this for the mean of
  300 runs (`r.cal_se` is the mean of the per-run values), with `n_runs` equal
  to N, so there the two terms are about equal. The calibration SE of D falls
  as one over the square root of `n_runs`, the runs per initial-CNR point (400
  by default). In the validation it is at most about `s / sqrt(n_runs)`, with
  `s` the spread of D over single measurements: summed over the cells, their
  ratio is 0.92 to 0.98 per shape and window where every frame is read at a
  calibrated CNR, and 0.80 to 0.91 where the frames fall between calibrated
  CNRs, as a measurement's frames almost always do
  (`experiments/dfree-calibration`). With `n_runs` equal to N the calibration
  SE can be as large as the statistical error of the mean, so choose `n_runs`
  well above the number of measurements to be averaged: at 4N the combined SE
  of the mean is at most about 1.12 times its statistical error, so the
  default suits means of up to about 100 measurements.
- **Drop frames below the floor.** The correction holds above a per-frame CNR
  of about 1 (0.89 to 1.51 depending on shape). The validations set the weight of
  frames below a per-frame CNR of 1.0 or 1.3 to zero before `fit_D`;
  `msd_and_weights(..., min_cnr=...)` does the same, and applies no floor
  unless asked. A CNR outside the calibrated grid takes the value at the
  nearest grid edge, and `msd_and_weights` then issues a
  `CalibrationRangeWarning`.
- **The estimated CNR holds over a narrower range.** With the CNR estimated
  from each frame by `fft-cnr` instead of known, the estimate is biased near
  the floor (it does not scatter evenly about the exact value) and its error is
  correlated with the width error, so the corrected mean D holds in simulation
  only where every fitted frame stays well above the floor. How far above
  depends on the profile shape and the diffusion length.
  See
  [`experiments/wse2-strobescat`](https://github.com/thiebes/width-debias/tree/master/experiments/wse2-strobescat).
- **Estimate the CNR on the spot.** Cut each profile to about ±4 sigma_0 around
  the spot before calling `per_frame_cnr`. Background and fringe structure in a
  full line cut corrupts the spectral CNR estimate at low signal.

## What the validation shows

- Above a per-frame CNR of about 1.5 the bounded and unbounded fits agree:
  they return the same widths and the same biased D
  (`experiments/estimator-comparison`). Below a per-frame CNR of about 1,
  where the bound carries appreciable mass, the bounded fit reflects rather
  than clips at it (`experiments/boundary-diagnostic`).
- With the calibration's own Monte Carlo error included, 4 of 140 cells lie
  beyond two combined standard errors of the reference, about 6 expected by
  chance (Gaussian, Voigt, and generalized-Gaussian p = 1.5 and p = 4
  profiles; D = 0.005 to 0.1 sigma_0^2/tau; initial CNR 5 to 90; frames with
  per-frame CNR >= 1), and the corrected D pooled over each shape's cells is
  within 0.4 combined standard errors of the reference. The uncorrected fit
  has 46 of 140. One calibration serves every cell of a shape, so the count is
  indicative rather than a test against a binomial null. At small D and low
  initial CNR the combined SE is large, so those cells constrain the bias only
  loosely (`experiments/dfree-calibration`).
- The correction holds above a per-frame CNR of about 1 (0.89 to 1.51 depending
  on shape: walking down from high CNR, the deepest CNR reached while every
  cell satisfies |residual| + 2 combined SE <= 0.3 x 2 D t_max, which is
  0.036 sigma_0^2 at D = 0.02 sigma_0^2/tau), where the leading-order bias model
  `var_bias = A / CNR^2` itself breaks (`experiments/correction-floor`). With
  single-exponential decay to 3 tau, every frame stays above the floor from an
  initial CNR of about 20.
- It corrects the anomalous exponent of subdiffusive transport as well as D
  (`experiments/subdiffusive-alpha`, `experiments/nc-solid`).

[`docs/REPO_MAP.md`](https://github.com/thiebes/width-debias/blob/v0.4.1/docs/REPO_MAP.md)
lists every experiment set with its result, and each set's README gives the
details and how to run it.

## Profile models

The bias affects any non-negative width estimator, so the profile model is
pluggable (`width_debias.models`): `GaussianModel`, `VoigtModel` (a diffusing
Gaussian convolved with a fixed Lorentzian point-spread function), and
`GeneralizedGaussianModel` (shape exponent p; p = 2 is the Gaussian).
`model_from_spec("generalized_gaussian:1.5")` builds a model from a spec
string. The experiment scripts in the repository take a model spec; name the
generalized-Gaussian shape explicitly:

```bash
python experiments/poc-bias-correction/poc_demo.py voigt
python experiments/boundary-diagnostic/diagnose_boundary.py generalized_gaussian:1.5
```

## Tests

```bash
pytest
```

`tests/test_quoted_numbers.py` checks the copies it lists of the validation
numbers quoted in this README, the docs, the set READMEs and the docstrings
against the committed tables they come from; a copy it does not list is not
checked. When you quote a number outside its set folder, add the copy to a
claim there, so a rerun that changes the table fails the tests until the copy
is updated.

## Layout

| Path | Role |
| ------- | ------- |
| `src/width_debias/simulate.py` | profile geometry: diffusion, decay, noise |
| `src/width_debias/noise.py` | noise laws: white, shot, Poisson, read floor |
| `src/width_debias/models.py` | profile models and `model_from_spec` |
| `src/width_debias/fit.py` | per-frame width fit; MSD to D linear fit |
| `src/width_debias/montecarlo.py` | Monte Carlo ensembles |
| `src/width_debias/bias_lookup.py` | per-frame statistics; `BiasLookup` lookups and their SE |
| `src/width_debias/calibration.py` | single-frame calibration cells; joint (CNR, sigma^2) calibration |
| `src/width_debias/fisher.py` | Fisher-information width variance, leading-order bias, CRLB on D |
| `src/width_debias/smoothing.py` | P-spline smoother for the calibration residual |
| `src/width_debias/weights.py` | the four schemes; calibration SE of D |
| `src/width_debias/validate.py` | calibration and evaluation sweeps |
| `src/width_debias/cnr.py` | per-frame CNR from measured profiles via `fft-cnr` |
| `experiments/` | one folder per experiment set (see `docs/REPO_MAP.md`) |
| `illustrations/` | schematic figures |
| `scripts/` | shared calibration tooling and figure rendering |

## Related work

- **DICE**: Monte Carlo engine for optical diffusion measurements (Thiebes and
  Grumstrup, *J. Chem. Phys.* 2024).
- **fft-cnr**: single-profile CNR estimation
  ([PyPI](https://pypi.org/project/fft-cnr/)).

## License

MIT, see [`LICENSE`](https://github.com/thiebes/width-debias/blob/v0.4.1/LICENSE).
