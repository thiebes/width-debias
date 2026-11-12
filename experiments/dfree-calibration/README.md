# dfree-calibration

## Purpose

Establish that the bias calibration is **D-free**: built at `D = 0`, keyed on
per-frame CNR alone, so the application's true D is never used. Includes the
dense calibration and the four-scheme test on the four validation shapes that
back the corrected-bias curves.

## Contents

- `validate_d_zero.py [--short-window] [profile_model ...]` -- D-independence
  test: one calibration built at `D = 0`, applied across
  `D in {0.005, 0.01, 0.02, 0.05, 0.1}` on Gaussian / Voigt / generalized-Gaussian
  (`generalized_gaussian:1.5`, `generalized_gaussian:4`; the bare name is
  p = 2). Every cell is scored under five arms on the same realizations
  (corrected with all frames or with frames below per-frame CNR 1 dropped;
  naive inverse-variance with cuts at per-frame CNR 3, at 2, or no cut). Prints
  each cell's evaluation and calibration SE, the pooled bias per D and per
  shape, the RMS error of one measurement and of the mean of 10 and 100, and
  the number of measurements averaged before the bias exceeds the error of
  their mean. `--short-window` runs a dim-sample geometry (window 0.5τ,
  CNR₀ 2 to 12, D 0.02 to 0.2 σ₀²/τ).
- `dense_calibration_and_test.py [MODEL ...]` -- dense (100 log-spaced CNR0)
  D-free calibration then a random log-distributed test, all four schemes
  (`ols`, `inverse_variance`, `subtract`, `mas`). The default runs the
  four validation shapes. The calibration takes about 2 h per shape (cached
  under `output/dense-calibration/`), the test under 1 h on 7 workers.
- `calibration_se_check.py generate|analyse` -- tests `calibration_D_se`
  against 16 independent calibrations on the fft-cnr paper's grid (48 CNR0,
  1500 runs) applied to one fixed evaluation set; separates the bias and
  weight terms, checks per-point SE and within-ensemble correlation, and ranks
  the paper's three seeds. `generate` is about 5 h on 8 cores (cached in
  `output/calibration-se-check/`).
- `weights_check.py` -- compares two 1-D slope weightings (per-frame variance
  against the MSD-point lookup) on the cached draws of
  `calibration_se_check.py`: mean, spread over realizations, and calibration
  SE per cell. Minutes once the cache exists.
- `joint_weights_check.py` -- scores the joint (CNR, sigma^2) lookup's slope
  weights: read at each frame's own fitted variance, against averaged over the
  variance grid (what `msd_and_weights` uses without a width key), on one
  joint calibration and fresh evaluation ensembles. A few minutes with
  `WIDTH_DEBIAS_WORKERS=0`.
- `slope_fit_check.py` -- compares the library slope weights (every frame,
  t = 0 included, by 1 / Var(sigma_hat_i^2)) with frame 0 at weight 0 on the cached draws and on the four-shape evaluation sets of
  `crlb-precision/efficiency_check.py`: calibration SE, bias in combined SE,
  spread, and interval coverage.
- `render_bias_panels.py` -- renders the bias figures from this set's
  `dense_calibration_and_test.txt` (the poster-time run: Gaussian, Voigt, and
  p = 2, former slope weights): `poster_panel3_bias.png` (the two standard
  fits fail) and `poster_panel5_corrected_bias.png` (corrected, three shapes).
  Uses `scripts/poster_style.py`, whose decile parser reads a shape header only
  when the shape name has no `:p` suffix.

## Data

None. Fully synthetic.

## Run

```bash
.venv/Scripts/python.exe experiments/dfree-calibration/validate_d_zero.py
.venv/Scripts/python.exe experiments/dfree-calibration/validate_d_zero.py --short-window
.venv/Scripts/python.exe experiments/dfree-calibration/dense_calibration_and_test.py
```

Set `WIDTH_DEBIAS_WORKERS=0` for `validate_d_zero.py` (about 20 min standard,
1 h short window, on 4 workers). The dense run uses its own process pool.
Outputs to repo-root `output/`; committed snapshots are in this folder.

## Key result + figure

`validate_d_zero`, standard window (t_max = 3τ). Each shape has one D-free
calibration (7 CNR₀ points × 300 runs) and 300 evaluation runs per cell, with
seeds offset by 10⁶ per shape and the library slope weights. A cell's deviation
is judged against its combined SE: the evaluation SE plus, for the corrected
arms, the calibration's propagated SE (`calibration_D_se`). Cells beyond 2
combined SE, out of 35 per shape:

| Arm | Gaussian | Voigt | p = 1.5 | p = 4 | Total of 140 |
| ------- | ------- | ------- | ------- | ------- | ------- |
| corrected, all frames | 2 | 1 | 0 | 2 | 5 |
| corrected, CNR ≥ 1 | 1 | 2 | 0 | 1 | 4 |
| naive, CNR ≥ 3 | 1 | 3 | 5 | 2 | 11 |
| naive, CNR ≥ 2 | 3 | 5 | 6 | 3 | 17 |
| naive, all frames | 11 | 14 | 17 | 4 | 46 |

About 6 of 140 are expected beyond 2 SE by chance. One calibration serves all
35 cells of a shape, so the cells are correlated and the counts are only
indicative. Pooled over the 35 cells (inverse combined variance, with the
calibration SE added as one fully correlated systematic, an upper bound), the
corrected (CNR ≥ 1)
D/D_ref − 1 is +0.0015 ± 0.0050 (Gaussian), −0.0011 ± 0.0059 (Voigt),
+0.0021 ± 0.0068 (p = 1.5), and +0.0011 ± 0.0029 (p = 4), each within 0.4
combined SE. The naive all-frames fit pools to +0.0042 ± 0.0017 (2.5 SE),
+0.0048 ± 0.0020 (2.4 SE), +0.0091 ± 0.0023 (4.0 SE), and +0.0005 ± 0.0010
(0.5 SE). The pooled values weight the large-D cells most, because their SE is
smallest; per D, the naive all-frames bias at D = 0.005 σ₀²/τ is
+0.070 ± 0.034 (Gaussian) and +0.14 ± 0.04 (p = 1.5). The p = 4 naive bias is
not resolved in this window. The small-D, low-CNR₀ cells have combined SE of
several D_ref (up to 2.3 at D = 0.005, CNR₀ 5 for p = 1.5), so they bound the
bias only loosely.

For one measurement, the RMS error √(bias² + spread²) of each naive arm,
divided by that of the corrected (CNR ≥ 1) arm, is:

| Naive arm | CNR₀ 5 | CNR₀ 8 to 21 | CNR₀ 34 to 90 |
| ------- | ------- | ------- | ------- |
| CNR ≥ 3 | 2.3 to 3.3 | 0.75 to 1.62 | 0.99 to 1.09 |
| CNR ≥ 2 | 1.07 to 1.42 | 0.71 to 1.17 | 0.99 to 1.02 |
| all frames | 0.91 to 3.77 | 0.95 to 1.04 | 0.99 to 1.01 |

The two lowest ratios belong to one p = 1.5 cell (D = 0.005, CNR₀ 21) whose
corrected mean reads 1.63 ± 0.45. Above 1.04 the all-frames ratio occurs only
for Gaussian at CNR₀ 5 (1.04 to 3.77). A single measurement is dominated by its
spread, so the correction and the uncut naive fit are close there. Averaging
shrinks the spread term (√(bias² + spread²/N) for the mean of N measurements);
bias² is estimated as (D/D_ref − 1)² − evaluation SE², floored at 0. Where the
naive all-frames bias is resolved, it exceeds the statistical error of the mean
after 15 to 91 averaged measurements (spread²/bias², 46 cells). The corrected
arm resolves a bias in 4 cells (27 to 46 measurements), fewer than the 6
expected by chance.

At equal runs the calibration SE of a cell is about its evaluation SE (at 300
runs each, for example 0.193 against 0.195 for p = 1.5 at D = 0.02, CNR₀ 8).
The calibration error is shared by every measurement made with that
calibration, so it does not average down. The mean of N measurements carries
the full calibration SE, and the calibration runs per CNR₀ point should be well
above the number of measurements averaged to keep it below the statistical
error of the mean. With 300 runs the mean-of-100 comparison already shows the
calibration error. The naive-to-corrected RMS ratio there spans 0.63 to 5.3,
and the ratios below 1 are consistent with this calibration draw's error in the
corrected arm.

`validate_d_zero --short-window` (window 0.5τ, CNR₀ 2 to 12, D 0.02 to
0.2 σ₀²/τ; per shape one calibration of 6 CNR₀ points × 2000 runs and 2000
evaluation runs per cell, 24 cells). Every frame keeps 0.61 to 1 of CNR₀, so the
CNR ≥ 1 cut drops nothing and the two corrected arms coincide. The corrected
fit has 0, 0, 3, and 2 cells beyond 2 combined SE (Gaussian, Voigt, p = 1.5,
p = 4; 5 of 96) and pools to +0.0005 ± 0.0121, −0.0010 ± 0.0145,
+0.0033 ± 0.0158, and +0.0053 ± 0.0069, each within 0.8 combined SE. The naive
all-frames fit has 10, 12, 11, and 8 (41 of 96) and pools to +0.0135 (3.8 SE),
+0.0222 (5.2 SE), +0.0290 (6.0 SE), and +0.0066 (3.2 SE). The cuts leave no
fit at CNR₀ 2 (both) or 3 (cut at 3). Where they keep a fit but drop frames
(CNR₀ 3 and 4 for the cut at 2, CNR₀ 4 for the cut at 3), their
single-measurement RMS error is 1.00 to 1.43 and 1.86 to 2.04 times the
corrected one; from CNR₀ 6 they drop no frame. For one measurement the
corrected and naive all-frames fits have the same RMS error (ratio 1.00 to
1.01). The naive bias exceeds the error of the mean after 47 to 652 averaged
measurements. On p = 4 the corrected D reads low at CNR₀ 3 and 4 for
D ≤ 0.05 σ₀²/τ (0.65 to 0.87, up to 2.4 combined SE).

`calibration_se_check` (`calibration_se_check.txt`; 16 calibration draws of
1500 runs on the fft-cnr paper's grid, library slope weights): the propagated
calibration SE holds. The observed / predicted variance of the cell mean,
averaged over D, is 0.57 to 1.27 per CNR₀ (about ±0.37 expected per cell) and
0.89 overall; per-point `var_bias` variance matches to 0.91 to 1.07; the weight
term is at most 0.04 of the prediction from CNR₀ 5 up (0.07 and 0.22 at CNR₀ 2
and 3). Over the 560 three-draw subsets the mean ratio has median 0.90 and a
5th to 95th percentile range of 0.49 to 1.31. The fft-cnr paper's seed triple
(1, 101, 201) gives 1.31 on this evaluation set; 0.09 of the 816 triples from
all 18 draws lie at or above it.

`weights_check` (`weights_check.txt`; the same 16 draws, 300 evaluation runs
per cell): the library per-frame weights narrow the spread of D/D_in over
realizations to 0.36 to 0.78 of the MSD-point-lookup weights (the fallback,
frame 0 at weight 0) in every cell, and shift cell means by at most 2.3
combined SE (mean over draws), with no consistent sign.

`joint_weights_check` (`joint_weights_check.txt`; one joint calibration of
16 CNR × 7 σ² cells × 500 runs with Gaussian fits, seed 7; 600 evaluation runs
per cell at D 0.02 and 0.1 σ₀²/τ, CNR₀ 8 to 34): with the weights read at each
frame's own fitted variance, the corrected D/D_true is 0.39 to 0.96; with the
width-averaged weights it is 0.88 to 1.09 (evaluation SE 0.004 to 0.12; the
calibration SE is not included). Both arms subtract the same bias, so the
difference is the weights alone. The own-width arm reads low in every cell, by
up to 0.61 at D = 0.02 σ₀²/τ, CNR₀ 13; the width-averaged arm lies within 2.1
evaluation SE of 1 in every cell.

`slope_fit_check` (`slope_fit_check.txt`): weighting frame 0 narrows the
spread of D/D_in to 0.42 to 0.77 of the frame-0-at-weight-0 values across the
four shapes; 2 of 96 cells lie beyond 2 combined SE with either weighting (4.4
expected by chance); the calibration SE holds (observed / predicted 0.89, and
0.90 with frame 0 at weight 0); the nominal 0.95 `fit_D` interval covers 0.89
to 0.96 of realizations.

Dense sweep (`dense_calibration_and_test_four_shapes.txt`; cached 5000-run
calibration over 100 CNR₀ cells, then 100 random log-distributed test CNR₀ in
[2, 100] × 1000 runs per shape, seeds offset by 10⁶ per shape). Pooled over all
100 test points, with the calibration SE as one shared systematic, the
`subtract` D/D_ref − 1 is +0.0012 ± 0.0037 (Gaussian), +0.0016 ± 0.0043
(Voigt), +0.0026 ± 0.0046 (p = 1.5), and −0.0004 ± 0.0022 (p = 4), each within
0.6 combined SE. In the CNR₀ 2 to 3 decile the corrected D reads low on Voigt
(decile mean 0.49; pooled −0.56, 2.8 combined SE) and p = 4 (0.57; −0.41, 5.1
SE). The last frames there sit near per-frame CNR 0.1, far below the floor,
where the subtraction over-corrects. MAS reads high below CNR₀ about 6.5 on
Gaussian, Voigt, and p = 1.5 (pooled +0.13 to +2.7), and on p = 4 it reads low
(−0.73 at CNR₀ 3 to 4.4, −1.83 at CNR₀ 2 to 3), so MAS fails in both directions
depending on the shape. `subtract` is the default. The overall mean
|⟨D̂⟩/D_ref − 1|, `subtract` against MAS, is 0.080 against 0.174 (Gaussian),
0.111 against 0.112 (Voigt), 0.084 against 0.199 (p = 1.5), and 0.099 against
0.363 (p = 4); Monte Carlo error alone would give 0.095, 0.090, 0.074, and
0.049.

Tables: `validate_d_zero_{gaussian,voigt,genGauss_p1.5,genGauss_p4}.txt`,
`validate_d_zero_short_{gaussian,voigt,genGauss_p1.5,genGauss_p4}.txt`,
`dense_calibration_and_test_four_shapes.txt`, `calibration_se_check.txt`,
`weights_check.txt`, `joint_weights_check.txt`, `slope_fit_check.txt`. `validate_d_zero_genGauss.txt` is a
p = 2 draw kept for reference. `dense_calibration_and_test.txt` (Gaussian,
Voigt, p = 2) and `dense_calibration_and_test_genGauss_p1.5_p4.txt` are the
earlier dense run with the former slope weights; the first is the source of the
poster panels. The three shapes in `poster_panel5_corrected_bias.png` are
Gaussian, Voigt, and p = 2.

## Depends on

`width_debias`. No sibling-folder helpers.

## See also

`docs/DERIVATION.md` section 8 (calibration without knowledge of D).
