# subdiffusive-alpha

## Purpose

Clean single-population synthetic test that the correction de-biases the
anomalous exponent alpha, not only D, on the `AnomalousExperiment` bed
(`sigma^2(t) = sigma0^2 + 2 D t^alpha`).

## Contents

- `subdiffusive_alpha.py [profile_model] [--quick] [--floor F]` -- sweeps true
  alpha in {0.6, 0.8, 1.0, 1.2} x CNR0 and fits `MSD = 2 D t^alpha`, comparing
  the naive all-frames fit against `subtract` + the per-frame CNR floor cut.

## Data

None. Fully synthetic. Caches `calibration_subdiffusive[_<model>].npz` (also
reused by multiexp-decay and coverage-interval).

## Run

```bash
.venv/Scripts/python.exe experiments/subdiffusive-alpha/subdiffusive_alpha.py
```

Set `WIDTH_DEBIAS_WORKERS=0`; about 40 min on 4 workers when the calibration
is built, about 15 min on all cores with it cached. Outputs to
repo-root `output/`; committed snapshots are here.

## Key result + figure

The naive fit pins at the ballistic α = 2.0 for CNR₀ ≲ 13, and for α = 1.2
also at CNR₀ 21. The working regime is CNR₀ 21 to 90, above the design floor
exp(t_max/τ) ≈ 20 (D-free Gaussian calibration of 60 CNR₀ × 1500 runs; 2000
runs per cell; Monte Carlo SE from 200 resamples of the realisations). There,
the signed exponent error of the mean trajectory, α − α_ref pooled over the
cells with inverse-variance weights, is:

| α_ref | Naive | z | Cells | Corrected | z | Cells |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 0.6 | +0.032 ± 0.009 | +3.8 | 4 | +0.004 ± 0.008 | +0.5 | 4 |
| 0.8 | +0.024 ± 0.011 | +2.2 | 4 | −0.011 ± 0.010 | −1.0 | 4 |
| 1.0 | +0.065 ± 0.014 | +4.7 | 4 | +0.019 ± 0.014 | +1.4 | 4 |
| 1.2 | +0.081 ± 0.016 | +4.9 | 3 | +0.049 ± 0.016 | +3.0 | 4 |

The floor cut drops no frame at CNR₀ ≥ 21, so the naive + cut arm equals the
naive arm there and the comparison isolates the subtraction. A cell whose
exponent lies within 0.01 of a fit bound (0.2 or 2.0) is left out of the pooled
value: resampling returns the bound almost every time, so its SE collapses and
an inverse-variance weight would let that one cell set the result (the naive
α = 1.2 cell at CNR₀ 21 reads 2.00 ± 0.01). The SEs leave out the calibration's
own error, which all cells share. The corrected α = 1.2 residual
(+0.049 ± 0.016, z = 3.0) is therefore resolved against the evaluation noise
and not yet tested against the calibration error.

The mean corrected |α − α_ref| over the same cells is 0.030 ± 0.020,
0.018 ± 0.022, 0.046 ± 0.023, and 0.100 ± 0.041 for α = 0.6, 0.8, 1.0, and 1.2,
against 0.025, 0.029, 0.033, and 0.052 from Monte Carlo error alone (about 0.8
SE for an unbiased estimator). The naive values are 0.18 to 0.30 (Monte Carlo
error alone 0.025 to 0.039). Corrected D is 0.84 to 1.05 of D_ref there. Below
CNR₀ 21 the table's naive + cut arm separates the cut from the subtraction
(for α = 0.6 the cut alone moves the naive exponent from 2.0 to 0.92 to 1.13).
Figure: `subdiffusive_alpha.png`; table `subdiffusive_alpha.txt`.

## Depends on

`width_debias` (`AnomalousExperiment`). No sibling-folder helpers.

## See also

`experiments/nc-solid/` (the same exponent bias on a realistic transport bed).
