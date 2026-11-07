# coverage-interval

## Purpose

Test whether the reported confidence interval on D covers D_true at the nominal
rate, in addition to the mean being unbiased. Compares the covariance-based CI, a parametric bootstrap, and an
analytical attribution for the de-biased folded-normal estimator.

## Contents

- `coverage_check.py` -- sweeps CNR0 and reports empirical 95% coverage for
  `subtract` (and the biased `inverse_variance` as the failing contrast),
  plus the `corr_inflation = sqrt(Var_full/Var_diag)` attribution.

## Data

None. Fully synthetic. Reuses the cached `calibration_subdiffusive.npz`.

## Run

```bash
.venv/Scripts/python.exe experiments/coverage-interval/coverage_check.py
```

Set `WIDTH_DEBIAS_WORKERS=0`; ~5 min. Outputs to repo-root `output/`; committed
snapshots are here.

## Key result + figure

With the library slope weights (per-frame Var(σ̂²) from the D-free lookup,
frame 0 included), the covariance interval covers D_ref 0.94 to 0.95 of the
time at every CNR₀ from 5 to 90 (2000 runs per cell, nominal 0.95). Its width
matches the realised spread (mean D_se / SD(D̂) 0.94 to 1.00), and the
parametric bootstrap covers 0.94 to 0.95. The shared σ̂²₀ is common-mode: it
shifts the intercept and leaves the slope, so it does not widen the interval
(inflation 0.86 to 0.89, below 1). The inverse-variance contrast also covers
0.93 to 0.95, because at this D (0.02 σ₀²/τ) its bias (1.10 to 1.65 of D_ref
at CNR₀ 5 to 21) is small against the spread of a single measurement (0.84 to
4.75 of D_ref). The D-free calibration is the shared
`calibration_subdiffusive.npz` (Gaussian, 60 CNR₀ × 1500 runs), named with
the seeds in the table header. Figure: `coverage_check.png`; table
`coverage_check.txt`.

## Depends on

`width_debias` (`fit.fit_D` returns `D_se`). No sibling-folder helpers.

## See also

`docs/DERIVATION.md` (the noise-bias correction).
