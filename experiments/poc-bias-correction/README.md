# poc-bias-correction

## Purpose

The core proof-of-concept: show the low-CNR noise bias in the extracted
diffusion coefficient D, and that the noise-bias correction removes it. One
two-panel figure per profile model (truncation diagnostic; bias curves
`mean(D_hat)/D_true` vs CNR0 for OLS, inverse-variance, and folded-normal).

## Contents

- `poc_demo.py [profile_model]` -- runs the two-panel proof-of-concept for a
  single profile model (`gaussian` default, or `voigt` / `generalized_gaussian`).

## Data

None. Fully synthetic (`width_debias` Monte Carlo); no pshare datasets.

## Run

```bash
.venv/Scripts/python.exe experiments/poc-bias-correction/poc_demo.py
.venv/Scripts/python.exe experiments/poc-bias-correction/poc_demo.py voigt
.venv/Scripts/python.exe experiments/poc-bias-correction/poc_demo.py generalized_gaussian:1.5
.venv/Scripts/python.exe experiments/poc-bias-correction/poc_demo.py generalized_gaussian:4
```

The generalized-Gaussian takes its shape exponent in the spec; the bare
`generalized_gaussian` is p = 2, a Gaussian profile.

Set `WIDTH_DEBIAS_WORKERS=0` (all cores) to speed up the Monte Carlo. Outputs
are written to the repo-root `output/` directory; the committed snapshots in this
folder are the current results.

## Key result + figure

D = 0.02 σ₀²/τ, 400 runs per CNR₀. The calibration is D-free
(`calibrate_d_free` at D = 0, 7 CNR₀ points × 400 runs), with calibration and
evaluation seeds disjoint and offset by 10⁶ per shape. Each table gives the
folded-normal cell's combined SE (evaluation and calibration). No folded-normal
cell lies beyond 2 combined SE on any shape (0 of 7 each). Inverse-variance
weighting leaves 1 (Gaussian), 2 (Voigt), 4 (p = 1.5), and 0 (p = 4) of 7 cells
beyond 2 evaluation SE. The mean `|mean(D_hat)/D_ref - 1|` over the sweep is
0.291 folded-normal against 0.368 inverse-variance (Gaussian), 0.093 against
0.279 (Voigt), 0.081 against 0.261 (p = 1.5), and 0.043 against 0.043 (p = 4);
Monte Carlo error alone would give 0.143, 0.099, 0.118, and 0.047 for
folded-normal. The Gaussian mean is set by the CNR₀ 5 cell, 2.67 ± 0.86
combined SE (1.9 SE from 1). OLS is inflated 46-fold at CNR₀ 5
for the Gaussian. The p = 4 OLS fit reads low at CNR₀ ≤ 21 (0.13 at CNR₀ 5),
and the p = 4 bias is small enough that inverse-variance weighting alone
leaves no cell beyond 2 SE.

Tables: `poc_demo_gaussian.txt`, `poc_demo_voigt.txt`,
`poc_demo_generalized_gaussian_p1.5.txt`, `poc_demo_generalized_gaussian_p4.txt`.
`poc_demo_generalized_gaussian.txt` and its figure are a p = 2 run (a Gaussian
profile with the generalized-Gaussian fitter's tighter `var_max`), included for
reference. Figures: `poc_demo*.png`.

## Depends on

`width_debias` (`model_from_spec`, `Experiment`, `calibrate_d_free`,
`evaluate`, `sweep_stats`, `BiasLookup`). No sibling-folder helpers.

## See also

`docs/DERIVATION.md` (folded-normal derivation).
