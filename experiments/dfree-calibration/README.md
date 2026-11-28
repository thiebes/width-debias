# dfree-calibration

## Purpose

Establish that the bias calibration is **D-free**: built at `D = 0`, keyed on
per-frame CNR alone, so the application's true D is never used. Includes the
dense calibration and the four-scheme test on the four validation shapes that
back the corrected-bias curves.

## Contents

- `validate_d_zero.py [--short-window] [--spread-dims {1,2}] [--cal-runs N]
  [profile_model ...]` -- D-independence test: one calibration built at
  `D = 0`, applied across
  `D in {0.005, 0.01, 0.02, 0.05, 0.1}` on Gaussian / Voigt / generalized-Gaussian
  (`generalized_gaussian:1.5`, `generalized_gaussian:4`; the bare name is
  p = 2). Every cell is scored under five arms on the same realizations
  (corrected with all frames or with frames below per-frame CNR 1 dropped;
  naive inverse-variance with cuts at per-frame CNR 3, at 2, or no cut). Prints
  each cell's evaluation and calibration SE, the pooled bias per D and per
  shape, the RMS error of one measurement and of the mean of 10 and 100, and
  the number of measurements averaged before the bias exceeds the error of
  their mean. `--short-window` runs a dim-sample geometry (window 0.5τ,
  CNR₀ 2 to 12, D 0.02 to 0.2 σ₀²/τ). `--spread-dims 1` (or 2) evaluates with
  a peak that also falls as the profile spreads (a summed profile, or a cut
  through the centre; the short window then adds CNR₀ 1.8, or 1.6, to its
  calibration), and `--cal-runs N` sets the calibration runs per CNR₀
  point apart from the evaluation runs. Each run saves its cell statistics and
  its lookup as `.npz` files beside its report, and `--report NPZ ...`
  rebuilds reports from saved statistics without rerunning.
- `dense_calibration_and_test.py [MODEL ...]` -- dense (100 log-spaced CNR0)
  D-free calibration then a random log-distributed test, all four schemes
  (`ols`, `inverse_variance`, `subtract`, `mas`). The default runs the
  four validation shapes. The calibration takes about 2 h per shape (cached
  under `output/dense-calibration/`), the test under 1 h on 7 workers.
- `calibration_se_check.py generate|analyse` -- tests `calibration_D_se`
  against 16 independent calibrations on a 48-point grid (CNR0 1.5 to 150,
  1500 runs) applied to one fixed evaluation set; separates the bias and
  weight terms, checks per-point SE and within-ensemble correlation, and, when
  `WIDTH_DEBIAS_EXTERNAL_RESULTS` names a directory that holds two external
  calibrations on the same grid (seeds 101 and 201), ranks the triple they form
  with draw 0 (seed 1). `generate` is about 5 h on 8 cores (cached in
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

Synthetic. `calibration_se_check.py analyse` can also read two external
calibrations (seeds 101 and 201) from the directory that
`WIDTH_DEBIAS_EXTERNAL_RESULTS` names. The committed `calibration_se_check.txt`
was written with them; without them its external-triple section is skipped,
with a note on stderr.

## Run

```bash
.venv/Scripts/python.exe experiments/dfree-calibration/validate_d_zero.py
.venv/Scripts/python.exe experiments/dfree-calibration/validate_d_zero.py --short-window
.venv/Scripts/python.exe experiments/dfree-calibration/validate_d_zero.py --spread-dims 1 --cal-runs 5000
.venv/Scripts/python.exe experiments/dfree-calibration/validate_d_zero.py --short-window --spread-dims 1 --cal-runs 5000
.venv/Scripts/python.exe experiments/dfree-calibration/validate_d_zero.py --report experiments/dfree-calibration/validate_d_zero_spread1_cal5000_gaussian.npz
.venv/Scripts/python.exe experiments/dfree-calibration/dense_calibration_and_test.py
```

Set `WIDTH_DEBIAS_WORKERS=0` for `validate_d_zero.py` (about 20 min standard,
1 h short window, on 4 workers; the two falling-peak runs took about 30 min
together on 16 workers). The dense run uses its own process pool.
Outputs to repo-root `output/`; committed snapshots are in this folder.
`--report` takes the cell statistics (`.npz`), not the `_lookup.npz` files.

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

At equal runs the calibration SE of a cell is at most about its evaluation SE
(at 300 runs each, for example 0.193 against 0.195 for p = 1.5 at D = 0.02,
CNR₀ 8).
The calibration error is shared by every measurement made with that
calibration, so it does not average down. The mean of N measurements carries
the full calibration SE, and the calibration runs per CNR₀ point should be well
above the number of measurements averaged to keep it below the statistical
error of the mean (see the ratio after the short-window results). With 300 runs
the mean-of-100 comparison already shows the calibration error. The
naive-to-corrected RMS ratio there spans 0.63 to 5.3, and the ratios below 1
are consistent with this calibration draw's error in the corrected arm.

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

In both windows, over the cells of the corrected arm (CNR ≥ 1), the calibration
SE times the square root of the calibration runs per CNR₀ point, divided by the
spread of D/D_ref over single measurements, each summed over the cells, is 0.92
to 0.98 per shape and window. Summed, the larger cells, which the tables print
with the most digits, set the ratio. Over the row at the smallest D it is 0.96
to 0.99 in the standard window and 0.93 to 0.99 in the short, and at the
largest D 0.82 to 0.89 and 0.88 to 0.95. In these runs the calibration SE of
D, in absolute units, is the same at every D: every D has the same per-frame
CNRs, because the simulated profiles keep their peak height as they spread,
and the lookup, built at D = 0, is read by the per-frame CNR alone, so each
frame falls exactly at a CNR the lookup was calibrated at. The spread of
single measurements grows with the width, which grows more at larger D. In
general, at a given CNR the scatter of a fitted width grows with the width,
while the lookup's error is that of the width at σ₀, and a frame between two
calibrated CNRs reads an interpolation of two independent values, whose error
is smaller. So the calibration SE is at most about the single-measurement
spread divided by the square root of the calibration runs.
At a ratio of 1, the mean of N measurements has a combined SE of √2 times its
statistical error with N calibration runs per point and 1.12 times with 4N;
where the ratio is below 1 the factors are smaller.

Falling peak (`--spread-dims 1 --cal-runs 5000`, both windows). In the runs
above the peak falls with the decay alone. Here it also falls as the summed
profile spreads, by σ₀/σ, so the late frames reach a lower per-frame CNR. The
calibration stays at D = 0, where the profile does not spread, and has 5000
runs per CNR₀ point; the evaluation cells, runs, and seeds are those above. In
the short window the calibration grid gains CNR₀ 1.8, so that every frame lies
inside the calibrated range (lowest frame 1.107 to 1.128 over the shapes,
lowest calibrated 1.092). In the standard window the arms without a cut read
the lookup's edge value for frames below per-frame CNR 0.249 (lowest frame
0.197 to 0.206). Cells beyond 2 combined SE, per shape (Gaussian, Voigt,
p = 1.5, p = 4) and in total:

| Arm | Standard window | Short window |
| ------- | ------- | ------- |
| corrected, all frames | 2, 2, 0, 2 (6 of 140) | 1, 0, 4, 2 (7 of 96) |
| corrected, CNR ≥ 1 | 2, 2, 0, 2 (6 of 140) | 1, 0, 4, 2 (7 of 96) |
| naive, CNR ≥ 3 | 1, 3, 5, 3 (12 of 140) | 1, 3, 4, 2 (10 of 64) |
| naive, CNR ≥ 2 | 4, 4, 6, 3 (17 of 140) | 5, 5, 8, 4 (22 of 80) |
| naive, all frames | 9, 15, 17, 4 (45 of 140) | 10, 12, 13, 8 (43 of 96) |

By chance about 6 cells are expected beyond 2 SE in the standard window and
about 4 in the short window. The corrected counts are not a like-for-like
comparison with those of the decay-alone runs above: their calibrations have
300 (standard) and 2000 (short) runs per CNR₀ point against 5000 here, so here
the combined SE is smaller and a cell of the same bias lies more SE from 1.
The naive arms carry no calibration SE. The corrected (CNR ≥ 1)
D/D_ref − 1, pooled over a shape's cells, is −0.0004 ± 0.0021 (Gaussian),
−0.0017 ± 0.0026 (Voigt), +0.0021 ± 0.0029 (p = 1.5), and −0.0004 ± 0.0012
(p = 4) in the standard window, and −0.0013 ± 0.0077, −0.0020 ± 0.0090,
+0.0068 ± 0.0101, and +0.0018 ± 0.0044 in the short window, each within 0.8
combined SE. Pooled per D it is within 1.6 combined SE on every shape in both
windows. The pooled biases are of the same order as with the decay alone (at
most 0.0068 against 0.0053); their combined SE is smaller because the
calibration has more runs. The calibration SE times the square root of the
calibration runs, over the single-measurement spread, summed over the cells as
above, is 0.89 to 0.91 per shape in the standard window and 0.80 to 0.85 in
the short window: below 1, as with the decay alone, and read from tables that
print both to five decimals. It is lower than with the decay alone even at the
smallest D, where the width hardly grows, because the falling peak moves every
frame after t = 0 between the CNRs at which the lookup was calibrated. Unlike
the decay-alone runs above, these runs read the lookup between its calibrated
CNRs, as `calibration_se_check` and the dense sweep also do. The naive
all-frames fit pools to +0.0046 (2.5 SE),
+0.0054 (2.5 SE), +0.0100 (4.1 SE), and +0.0006 (0.5 SE) in the standard
window, and +0.0165 (4.4 SE), +0.0247 (5.5 SE), +0.0344 (6.8 SE), and +0.0071
(3.3 SE) in the short window. On p = 4 in the short window the corrected D
again reads low at CNR₀ 3 and 4 for D ≤ 0.05 σ₀²/τ (0.59 to 0.92, up to 2.7
combined SE). The D = 0 lookup, read at each frame's CNR, therefore corrects D
under a falling peak as it does under the decay alone.

`calibration_se_check` (`calibration_se_check.txt`; 16 calibration draws of
1500 runs on a 48-point CNR₀ grid, library slope weights): the propagated
calibration SE holds. The observed / predicted variance of the cell mean,
averaged over D, is 0.57 to 1.27 per CNR₀ (about ±0.37 expected per cell),
and 0.89 overall in `slope_fit_check.txt` (its current row); per-point `var_bias` variance matches to 0.91 to 1.07; the weight
term is at most 0.04 of the prediction from CNR₀ 5 up (0.07 and 0.22 at CNR₀ 2
and 3). Over the 560 three-draw subsets the mean ratio has median 0.90 and a
5th to 95th percentile range of 0.49 to 1.31, so a calibration error estimated
from three draws can fall far from its expected value. The triple formed by
draw 0 (seed 1) and two external calibrations on the same grid (seeds 101 and
201) gives 1.31 on this evaluation set; 0.09 of the 816 triples from all 18
draws lie at or above it.

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
`validate_d_zero_short_{gaussian,voigt,genGauss_p1.5,genGauss_p4}.txt`, the
falling-peak tables
`validate_d_zero{,_short}_spread1_cal5000_{gaussian,voigt,generalized_gaussian_p1.5,generalized_gaussian_p4}.txt`
with their cell statistics (`.npz`) and lookups (`_lookup.npz`), computed on
another machine and rebuilt from the statistics with `--report`,
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
