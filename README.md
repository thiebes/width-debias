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
simulation the corrected fit returns 1.00 times the reference. The calibration
is built at D = 0, so it does not require a prior value of D. Removing the
fit's lower bound on the width does not help: above a per-frame CNR of about
1.5 the bounded and unbounded fits are identical, and both carry the full bias.
Below a per-frame CNR of about 1 the fits spread into a broad upper tail and the
correction no longer holds (see [On real data](#on-real-data)). The derivation,
which reaches the same additive form from a folded-normal model of the width
estimate, is in
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

On a measurement, the per-frame CNR comes from `width_debias.cnr.per_frame_cnr`
(which wraps `fft-cnr`), the corrected MSD and weights from
`width_debias.weights.msd_and_weights("subtract", ...)`, the slope from
`width_debias.fit.fit_D`, and the calibration's contribution to the
uncertainty from `width_debias.weights.calibration_D_se`:

```python
import numpy as np
from width_debias import calibrate_d_free
from width_debias.cnr import per_frame_cnr
from width_debias.fit import fit_D
from width_debias.weights import calibration_D_se, msd_and_weights

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
- **Drop frames below the floor.** The correction holds above a per-frame CNR
  of about 1 (0.7 to 1.5 depending on shape). The validations set the weight of
  frames below a per-frame CNR of 1.0 or 1.3 to zero before `fit_D`;
  `msd_and_weights(..., min_cnr=...)` does the same, and applies no floor
  unless asked. A CNR outside the calibrated grid takes the value at the
  nearest grid edge, and `msd_and_weights` then issues a
  `CalibrationRangeWarning`.
- **The estimated CNR holds over a narrower range.** With the CNR estimated
  from each frame by `fft-cnr` instead of known, the corrected mean D holds in
  simulation only where every fitted frame stays well above the floor: from an
  initial CNR of about 34 at D = 0.02 sigma_0^2/tau for the Gaussian profile
  (about 55 for p = 1.5 and 13 for p = 4), and from about 21 at D >= 0.05.
  Below that, the pooled Gaussian mean at D = 0.02 over initial CNR 8 to 90 is
  1.9 of the reference (standard error 0.4). See
  [`experiments/wse2-strobescat`](https://github.com/thiebes/width-debias/tree/v0.4.1/experiments/wse2-strobescat).
- **Estimate the CNR on the spot.** Cut each profile to about ±4 sigma_0 around
  the spot before calling `per_frame_cnr`. Background and fringe structure in a
  full line cut corrupts the spectral CNR estimate at low signal.

## What the validation shows

- Above a per-frame CNR of about 1.5 the fit's lower bound does not bind: the
  bounded and unbounded fits return the same widths and the same biased D
  (`experiments/estimator-comparison`). Below it, the bounded fit reflects
  rather than clips at the bound (`experiments/boundary-diagnostic`).
- With the calibration's own Monte Carlo error included, 4 of 96 cells lie
  beyond two combined standard errors of the reference (Gaussian, Voigt, and
  generalized-Gaussian p = 1.5 and p = 4 profiles; D = 0.005, 0.02, and 0.1
  sigma_0^2/tau; initial CNR 3 to 90; `slope_fit_check.txt`). The cells share
  noise seeds across shapes and one calibration per shape, so the count is
  indicative rather than a test against a binomial null. At small D and low
  initial CNR the combined SE is large, so those cells constrain the bias only
  loosely (`experiments/dfree-calibration`).
- The correction holds above a per-frame CNR of about 1 (0.7 to 1.5 depending on
  shape), where the leading-order bias model `var_bias = A / CNR^2` itself
  breaks (`experiments/correction-floor`). With single-exponential decay to
  3 tau, every frame stays above the floor from an initial CNR of about 20.
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
