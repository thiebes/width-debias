# coverage-interval

## Purpose

Test whether the reported confidence interval on D covers D_true at the nominal
rate, in addition to the mean being unbiased. Compares the covariance-based CI, a parametric bootstrap, and an
analytical attribution for the de-biased `subtract` estimator.

## Contents

- `coverage_check.py` -- sweeps CNR0 and reports empirical 0.95 coverage of
  the covariance interval and a parametric bootstrap interval for `subtract`
  (with the uncorrected `inverse_variance` as a contrast), the ratio
  D_se / SD(D̂), and the pooled values, with SEs.
- `narrowness_attribution.py` -- at three CNR0, compares each frame's measured
  Var(σ̂²) with the lookup's and predicts D_se / SD(D̂) from the width's growth
  alone, to explain why the covariance interval runs slightly narrow.

## Data

None. Fully synthetic. Reuses the cached `calibration_subdiffusive.npz`.

## Run

```bash
.venv/Scripts/python.exe experiments/coverage-interval/coverage_check.py
.venv/Scripts/python.exe experiments/coverage-interval/narrowness_attribution.py
```

Set `WIDTH_DEBIAS_WORKERS` (the coverage sweep fits 13 CNR0 × 8000 runs, about
25 min with four workers, and its 2000 bootstrap resamples per run add to that;
the attribution takes a few minutes serial).
Outputs to repo-root `output/`; committed snapshots are here.

## Key result + figure

D = 0.02 σ₀²/τ, Gaussian, 13 log-spaced CNR₀ from 5 to 90, 8000 runs per
cell. SEs come from batch means over 8 sub-ensembles (the width ratio) and
from the binomial (the coverages, 0.0024 per cell). The pooled values are
printed by `coverage_check.py`. Every SE is conditional on the one
calibration: its Monte Carlo error moves every cell's interval together and is
not included.

With the library slope weights (per-frame Var(σ̂²) from the D-free lookup,
frame 0 included), the covariance interval covers D_ref 0.939 to 0.952 of the
time, 0.9438 ± 0.0007 pooled over the 13 cells, against the nominal 0.95. The
shortfall is small but resolved. D_se is 0.969 ± 0.003 of the realised
SD(D̂), pooled from CNR₀ 21 up with inverse-variance weights (0.967 ± 0.003
as a plain mean; 0.944 to 1.003 per cell over the whole sweep), which for a
normal D̂ implies a coverage of 0.942.

The cause is the width (`narrowness_attribution.py`). The lookup's per-frame
variances are calibrated at σ₀², while here the width grows to σ² = 1.12 σ₀²
by t = 3τ, and at a fixed CNR Var(σ̂²) grows about as σ³. The measured
per-frame variance exceeds the lookup's by (σ²/σ₀²)^k with k = 1.36 ± 0.13
over the frames with per-frame CNR ≥ 1 (heteroscedasticity-robust SE; 0.13
assuming equal scatter). The σ³ scaling alone predicts
D_se / SD(D̂) of 0.964 to 0.971, and the correlation between frames is at
most 0.05. The shortfall therefore grows with the width's growth, 2 D t_max;
the `fit_D` docstring notes the same effect at D = 0.1 σ₀²/τ. A width-scaled
lookup (`frame_variance` with a width key) is the corresponding fix; it is
not tested here.

The parametric bootstrap (2000 resamples) covers 0.945 to 0.955, 0.9505 ±
0.0007 pooled, at the nominal rate. It resamples each frame with the
per-frame variance measured over the ensemble, which includes the width's
growth, so it supports the attribution above; a single measurement does not
have those variances, and the bootstrap here is a check, not a deployable
interval. (With 200 resamples and percentile bounds, even an exact bootstrap
would cover only about 0.940.) The shared σ̂²₀ is common-mode: with frame 0
weighted it shifts the intercept and leaves the slope, so it does not widen
the interval. The inverse-variance contrast covers 0.936 to 0.953, because at
this D its bias
(median 1.05 to 1.57 of D_ref at CNR₀ 5 to 21) is small against the spread of
a single measurement (SD 0.82 to 4.61 of D_ref). The D-free calibration is the
shared `calibration_subdiffusive.npz` (Gaussian, 60 CNR₀ × 1500 runs), named
with the seeds in the table header.

Figure: `coverage_check.png`, caption `coverage_check_caption.md`; tables
`coverage_check.txt`, `narrowness_attribution.txt`.

## Depends on

`width_debias` (`fit.fit_D` returns `D_se`). No sibling-folder helpers.

## See also

`docs/DERIVATION.md` (the noise-bias correction).
