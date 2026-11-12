# coverage-interval

## Purpose

Test whether the reported confidence interval on D covers D_true at the nominal
rate, in addition to the mean being unbiased. Compares the covariance-based CI, a parametric bootstrap, and an
analytical attribution for the de-biased `subtract` estimator.

## Contents

- `coverage_check.py` -- sweeps CNR0 and reports empirical 0.95 coverage for
  `subtract` (and the biased `inverse_variance` as the failing contrast),
  plus the `corr_inflation = sqrt(Var_full/Var_diag)` attribution, with SEs.
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

Set `WIDTH_DEBIAS_WORKERS` (the coverage sweep, 13 CNR0 × 8000 runs, takes
about 25 min with four workers; the attribution takes a few minutes serial).
Outputs to repo-root `output/`; committed snapshots are here.

## Key result + figure

D = 0.02 σ₀²/τ, Gaussian, 13 log-spaced CNR₀ from 5 to 90, 8000 runs per
cell. SEs come from batch means over 8 sub-ensembles (the width ratios) and
from the binomial (the coverages, 0.0024 per cell).

With the library slope weights (per-frame Var(σ̂²) from the D-free lookup,
frame 0 included), the covariance interval covers D_ref 0.939 to 0.952 of the
time, 0.944 ± 0.0007 pooled over the 13 cells, against the nominal 0.95. The
shortfall is small but resolved. D_se is 0.969 ± 0.002 of the realised
SD(D̂), pooled from CNR₀ 21 up (0.944 to 1.003 per cell over the whole
sweep), which for a normal D̂ implies a coverage of 0.943.

The cause is the width (`narrowness_attribution.py`). The lookup's per-frame
variances are calibrated at σ₀², while here the width grows to σ² = 1.12 σ₀²
by t = 3τ, and at a fixed CNR Var(σ̂²) grows about as σ³. The measured
per-frame variance exceeds the lookup's by (σ²/σ₀²)^k with k = 1.36 ± 0.13
over the frames with per-frame CNR ≥ 1. The σ³ scaling alone predicts
D_se / SD(D̂) of 0.964 to 0.971, and the correlation between frames is at
most 0.05. The shortfall therefore grows with the width's growth, 2 D t_max;
the `fit_D` docstring notes the same effect at D = 0.1 σ₀²/τ. A width-scaled
lookup (`frame_variance` with a width key) is the corresponding fix; it is
not tested here.

The parametric bootstrap (200 resamples) covers 0.937 to 0.949, no better.
The shared σ̂²₀ is common-mode: it shifts the intercept and leaves the slope,
so it does not widen the interval (inflation 0.864 to 0.892, below 1). The
inverse-variance contrast covers 0.936 to 0.953, because at this D its bias
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
