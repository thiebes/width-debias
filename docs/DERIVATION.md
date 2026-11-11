# Derivation of the MSD bias and its correction

This note derives why the diffusion coefficient is overestimated and what the
correct fix is. A width fitted to a noisy profile scatters about the true
width; squaring it adds the width's estimation variance to the fitted
variance. In the regime where the correction works (per-frame CNR above about
1) the fit's lower bound does not bind: the bounded and unbounded fits return
the same widths and the same biased D (`experiments/estimator-comparison`).
Below it, `experiments/boundary-diagnostic/diagnose_boundary.py` found the
fitted-variance distribution strictly interior and right-skewed with no
delta-mass at the lower bound for the Gaussian, Voigt, and generalized-Gaussian
p = 1.5 and p = 4 fits (a pile-up of 0.05 to 0.06 of fits at the bound appears
only at CNR ≈ 1 for Voigt and p = 1.5) — **reflection, not clipping**. The
model below covers both regimes, and the reflection does not change its
result.

## 1. Model

At each frame *i* the analysis fits a profile and reports a width. Write the
width estimate as `X_i` with

    X_i ~ N(σ_i, s_i²),

where `σ_i` is the true width at frame *i* and `s_i² = Var(σ̂_i)` is the
width's estimation variance, a decreasing function of the per-frame CNR (at
high CNR the Cramér–Rao value; Section 3). The fitted **variance** is the
square of a Gaussian,

    σ̂²_i = X_i².

A fitted width cannot be negative. Where `X_i < 0` has appreciable probability
(per-frame CNR below about 1), the fit reports `|X_i|`: it reflects that mass
rather than clipping it. Reflection leaves the square unchanged,
`|X_i|² = X_i²`, so the moments below hold with or without the bound; the
folded-normal model and the unbounded one give the same fitted variance.
`σ̂²_i / s_i²` is a non-central χ²₁ with non-centrality `(σ_i/s_i)²` —
continuous, strictly positive, right-skewed, the shape the diagnostic saw.

## 2. Moments

For `X ~ N(σ, s²)`, using `E[X²] = σ² + s²` and `E[X⁴] = σ⁴ + 6σ²s² + 3s⁴`:

    E[σ̂²]   = σ² + s²
    Var[σ̂²] = E[X⁴] − E[X²]² = 4σ²s² + 2s⁴ = 2s²(2σ² + s²)

**The fitted variance carries an additive upward bias equal to the estimation
noise variance `s²`.** This is the width analogue of the squared-magnitude
corrections used for Rician magnitude data and for polarization amplitudes; the
specific references are leads in `prior_art_map.md` and are not yet audited in
`citations_audit.md`, so none is cited here.

## 3. The scale of the bias, and the truncation fraction

To leading order in 1/CNR the width's estimation variance is the Cramér–Rao
value, `s_i² = A(σ_i) / CNR_i²`, where `A` follows from the Fisher information
of the profile model on the pixel grid (`width_debias.fisher`). For a 1-D
Gaussian at a fixed pixel spacing `dx` and peak-defined CNR, `s² ∝ σ dx / CNR²`.
The Monte Carlo spread of the fitted width reproduces the Fisher `A` to 0.986
to 1.006 on four shapes (Section 9). Because the CNR falls frame by frame,
`s_i²` grows with time (Section 4). Below per-frame CNR about 1.2 the
higher-order terms take over and the bias exceeds `A / CNR²` (Section 9); a
calibration against matched simulations carries them, which is why the
package calibrates `var_bias(CNR)` rather than using `A / CNR²` alone.

`BiasLookup` also reports a dimensionless readout of the width's
signal-to-noise `α_i = σ_i / s_i`: the truncation fraction, the mass of the
latent estimate below zero,

    p_i = P(X_i < 0) = Φ(−α_i),     so   α_i = −Φ⁻¹(p_i).

In terms of `p`:

- relative bias of σ̂²:      `s²/σ² = 1/α²              = 1 / Φ⁻¹(p)²`
- relative variance of σ̂²:  `Var[σ̂²]/σ⁴ = 2(2α²+1)/α⁴`, with `α = −Φ⁻¹(p)`

So the *shape* of the bias and variance is fixed by `p` alone. The *scale* is
not: `p` is dimensionless, the bias `s²` has units of variance, and the scale
comes from the per-frame CNR. That is why the weighting takes the per-frame CNR
as its input, not `p`. In the correctable regime `p` is negligible: at leading
order the Gaussian fit at `σ = σ_0` has `α ≈ 4.2` at per-frame CNR 1
(`p ≈ 10⁻⁵`), so the bound matters only below CNR 1, where the higher-order
terms also take over.

## 4. Why the bias reaches D, and why a weight alone cannot remove it

The MSD point is `MSD_i = σ̂²_i − σ̂²_0`, so

    E[MSD_i] = (σ_i² − σ_0²) + (s_i² − s_0²) = MSD_true,i + (s_i² − s_0²).

Because CNR falls frame by frame, `s_i` grows with *i*; the bias `s_i² − s_0²`
is an increasing function of time. An increasing additive offset on `MSD(t)`
tilts the fitted slope upward — `D` is overestimated, the more so at low CNR.

A reweighting of the MSD-vs-t fit **cannot** remove this. Weights change which
points dominate the slope; they cannot subtract a mean offset that is present
in every point. Down-weighting late frames trades the bias for variance but
never eliminates it. The bias must be **subtracted**, not weighted away. This
is the central result: the standard inverse-variance scheme is structurally
unable to fix the bias, however the weights are chosen.

## 5. The derived method

1. **Bias-correct each frame.** Subtract the estimation-noise variance:
   `ṽ²_i = σ̂²_i − s_i²`, so `E[ṽ²_i] = σ_i²`. Build the MSD from the corrected
   variances: `MSD_i = ṽ²_i − ṽ²_0`.
2. **Weight by inverse variance.** `w_i = 1 / Var[σ̂²_i]` on every frame,
   t = 0 included, with `Var[σ̂²] = 2s²(2σ² + s²)`. The MSD points share
   `σ̂²_0`, but a constant shift does not change a free-intercept slope, so
   fitting `MSD_i = σ̃²_i − σ̃²_0` with these weights is the same as fitting the
   corrected widths against time, whose points are independent. That fit
   reaches the Cramér–Rao bound on D (0.95 to 1.10 × √CRLB from CNR₀ 21 on four
   shapes; `experiments/crlb-precision/efficiency_check.py`). Subtracting a
   calibrated, deterministic bias does not change the variance, so the weight is
   the ordinary inverse-variance weight; the subtraction is what removes the
   bias.

In `width_debias` this is the `subtract` scheme. It differs from
`inverse_variance` *only* by step 1, so comparing the two isolates the effect
of the bias correction.

## 6. Implementation

The per-frame noise variance `s_i²` is obtained by calibration: a Monte Carlo
CNR sweep gives `E[σ̂²] − σ²_true` as a function of CNR — the `var_bias(CNR)`
model in `BiasLookup`. The model of Section 1 is what licenses this:
it tells us the bias is *additive*, *positive*, and a function of the per-frame
CNR, so a one-dimensional `var_bias(CNR)` calibration is the right object to
build. Calibration and evaluation use independent random seeds. On real data
the same calibration is done against simulations matched to the measurement
(simulation-based debiasing).

## 7. Two alternatives, tested

The `subtract` scheme removes the *calibrated* `var_bias(CNR)`. When that
calibration is imperfect — it conflates frames of equal CNR but different true
width, and the low-CNR mean is heavy-tailed — the hard subtraction can
overshoot and push `D` below the reference in the deepest noise. Two routes were
tested against this.

**MAS-gated subtraction (`mas` scheme).** Following Plaszczynski's Modified
Asymptotic estimator, the subtraction is passed through a smooth gate:

    ṽ²_i = σ̂²_i − s_i² · (1 − exp(−σ̂²_i / s_i²)).

Writing `u = σ̂²/s²`, the corrected value is `s²·(u − 1 + e^−u)`, which is ≥ 0
for all `u ≥ 0` and increasing — the correction **cannot drive the variance
negative**. The gate is ≈ 1 when the frame is well above the noise (full `s²`
subtraction, matching `subtract`) and → 0 when it is noise dominated (the
frame is left uncorrected rather than overshot). On the narrow `CNR₀ ∈ [5,90]`
sweep (`experiments/estimator-comparison/moment_estimator_failure.py`), `mas`
gave the lowest bias of any
scheme — mean |⟨D⟩/D_ref − 1| = 0.089 vs `subtract` 0.116 — and
removed the deep-regime overshoot.

**On the wider `CNR₀ ∈ [2,100]` sweep the ranking reverses.**
At low CNR₀ the MAS gate switches the correction off where the noise bias is
large. On `experiments/dfree-calibration/dense_calibration_and_test.py`'s
100-point sweep on four shapes (a cached 5000-run calibration over 100 CNR₀
cells, then 100 random test CNR₀ × 1000 runs per shape, seeds offset by 10⁶
per shape; `dense_calibration_and_test_four_shapes.txt`), the `subtract` bias
pooled over all 100 test points, with the calibration SE as one shared
systematic, is within 0.6 combined SE of zero on every shape. MAS reads high
below CNR₀ about 6.5 on Gaussian, Voigt, and p = 1.5 (pooled +0.13 to +2.7)
and low on p = 4 (−0.73 at CNR₀ 3 to 4.4, −1.83 at CNR₀ 2 to 3), so it fails
in a direction set by the shape. `subtract` reads low only in the CNR₀ 2 to 3
decile, on Voigt (decile mean 0.49) and p = 4 (0.57), where the last frames
sit near per-frame CNR 0.1, far below the floor of Section 9. The overall mean
|⟨D̂⟩/D_ref − 1|, `subtract` against MAS, is 0.080 against 0.174 (Gaussian),
0.111 against 0.112 (Voigt), 0.084 against 0.199 (p = 1.5), and 0.099 against
0.363 (p = 4); Monte Carlo error alone would give 0.095, 0.090, 0.074, and
0.049. For typical measurements either scheme works. The case for `subtract`
is the failure mode: its failures are confined to frames far below the floor,
while MAS fails over a wider CNR₀ range and in either direction.
**`subtract` is the recommended scheme.** MAS is available in the package for
comparison.

**Non-truncated moment estimator (`MomentModel`).** The alternative is to
remove the wall entirely: estimate the variance from the discrete second
moment of the profile, which is linear in the data, admits negative values,
and so needs no correction (cf. Vestergaard's covariance-based estimator,
unbiased *because* it allows negative values). It **fails here**: the moment
is a ratio, second moment over total intensity, and the denominator passes
through zero at low CNR, producing wild, often negative `D` (mean bias ~10³,
enormous spread). Removing the wall is sound in principle, but it needs a
*stable* non-truncated estimator — the covariance-based estimator achieves
this on raw displacement increments, which an intensity-profile MSD pipeline
does not have. An intensity-profile moment is not a viable substitute.

## 8. Calibration without knowledge of D

The calibration step measures `s²(CNR)` empirically by running synthetic
profile sequences and recording the empirical mean of `σ̂² − σ²_true` per
per-frame CNR. A naive implementation of that step (`calibrate(exp, ...)`)
takes the calibration experiment's `D` as an input — methodologically
awkward, since the goal is to estimate `D` from real data where `D` is
unknown, and meaningless on heterogeneous transport where no single `D`
describes the dynamics.

The fix is small. The bias `s²(CNR)` depends most strongly on the per-frame
CNR (as `1/CNR²`); its dependence on the width enters through `A(σ)`, which
grows as `σ` at a fixed pixel grid and peak CNR (Section 3). In the low-`D`
regime where the noise bias bites, `σ²(t)` barely changes from `σ_0²` across
the measurement window, so
calibrating at `σ² = σ_0²` (i.e. `D = 0`) captures the bias the application
actually sees. `calibrate_d_free(cnr0_grid, ...)` is the thin wrapper that
runs the standard calibration sweep at `D = 0`; the lookup it returns is
keyed on per-frame CNR alone and is reused unchanged across any application
`D`.

Tested by Monte Carlo at `D ∈ {0.005, 0.01, 0.02, 0.05, 0.1}` on Gaussian,
Voigt, and generalized-Gaussian p = 1.5 and p = 4 profiles
(`experiments/dfree-calibration/validate_d_zero.py`, 300-run calibration and
300 evaluation runs per cell).

The calibration is itself a finite Monte Carlo, and its error is common to
every measurement corrected with it, so it does not average away over runs.
`weights.calibration_D_se` propagates it into D through the slope fit's
linear map (`BiasLookup.var_bias_se` gives each looked-up bias's SE;
with `D̂ = ½ Σ cᵢ mᵢ` and `Σ cᵢ = 0`, `Var_cal(D) = ¼ Σ cᵢ² sᵢ²`; the common
t = 0 shift cancels into the intercept, and frame 0's own error enters through
`c₀` because frame 0 is weighted). Checked against 16 independent calibration
draws applied to one fixed evaluation set: observed / predicted variance of the
cell mean 0.89 with the per-frame slope weights
(`experiments/dfree-calibration/calibration_se_check.txt`). At equal runs the
calibration and evaluation terms are about equal, and the calibration term does
not shrink with more evaluation runs. It is shared by every measurement made
with the calibration, so the mean of N measurements carries all of it; the
calibration runs per CNR₀ point should exceed the number of measurements
averaged.

With both terms in the cell SE and the per-frame slope weights
(`validate_d_zero.py`; seeds offset by 10⁶ per shape), keeping frames with
per-frame CNR ≥ 1, 4 of 140 cells (four shapes; 5 D from 0.005 to 0.1 σ₀²/τ;
CNR₀ 5 to 90) lie beyond two combined SE, about 6 expected by chance. Pooled
over a shape's 35 cells, with the calibration SE as one shared systematic, the
corrected D/D_ref − 1 is within 0.4 combined SE of zero on every shape; the
uncorrected all-frames fit has 46 of 140 cells beyond 2 SE and pools 2.4 to
4.0 SE high on Gaussian, Voigt, and p = 1.5. In a short window (t_max = 0.5τ,
CNR₀ 2 to 12) the counts are 5 of 96 corrected and 41 of 96 uncorrected. One
calibration serves all cells of a shape, so these counts are indicative, not a
test against a binomial null. The small-D, low-CNR cells
have large combined SE and so constrain the bias only loosely. `D = 0.005` has
`L_D = √(2Dτ) ≈ 0.042 FWHM`, so its slope is near the precision limit of any
estimator at these CNRs. A low-CNR application should report the combined SE,
and a tighter claim needs a larger calibration budget.

The D = 0 lookup assumes the width changes little over the measurement. When
it grows several-fold, the bias at a given CNR changes with it, and a joint
`var_bias(CNR, σ²)` calibration (`width_debias.calibration.calibrate_cnr_sigma2`)
is needed. Its lookup requires a stand-in for the unknown true `σ²`. Keying
each frame on its own fitted `σ̂²` under-corrects at low CNR, because a frame
whose fit scatters high reads the lookup near the grid edge; a
measurement-level key (a robust line through the measurement's own `σ̂²`) avoids
that but assumes the form of `σ²(t)`. On the high-N Br-I bed (`σ²` growing about
64-fold) the per-frame key leaves D at 0.89 to 0.96 of the reference at CNR₀ 15
to 20 on the two slower-spreading maps (0.99 on the third), the measurement-level key matches the exact key to 0.003, and below
CNR₀ 15 no key determines D (`experiments/halide-mixing-highn/key_sensitivity.py`).
The joint calibration also needs several thousand runs per cell at low CNR.

## 9. Where the correction works: the per-profile floor

Because the slope fit is a deterministic linear functional of the corrected
per-frame variances (the weights are deterministic in CNR), the bias in
`D̂` is fully determined by the per-frame residuals `E[σ̃²(t)] − σ²_true(t)`.
The cleanest characterisation is therefore an isolated single-profile
sweep: generate many noisy profiles at a chosen per-profile CNR (no
diffusion, no decay, no frame indexing), apply the calibrated
`var_bias(CNR)` subtraction, and measure the residual mean with its MC
standard error.

`experiments/correction-floor/per_profile_floor.py` runs this sweep with 50,000 isolated fits
per CNR cell at `σ²_true = σ_0²`, corrected with the dense D-free lookup (100
CNR₀ cells of 5000 runs, seeds independent of the test cells and offset by 10⁶
per shape). Each cell's residual carries two errors: the test cell's Monte
Carlo error and the lookup's own error at that CNR. For Gaussian profiles
(`per_profile_floor_gaussian.txt`):

| per-profile CNR | E[raw] (σ_0²) | residual (σ_0²) | combined SE (σ_0²) |
| ------- | ------- | ------- | ------- |
| 1.96 | 1.016 | −0.001 | 0.004 |
| 1.51 | 1.033 | −0.001 | 0.006 |
| 1.16 | 1.084 | +0.006 | 0.010 |
| 0.89 | 1.238 | +0.013 | 0.028 |
| 0.69 | 1.710 | −0.041 | 0.087 |
| 0.53 | 2.680 | −0.181 | 0.168 |

From CNR 0.9 up the residual is within 2.2 combined SE on every shape. Below
it the Gaussian residual stays within 1.8 combined SE down to CNR 0.05, and the
other shapes reach 2.5 to 3.5 SE only in single cells below CNR 0.12; but the
lookup's own SE there (0.03 to 0.38 σ_0² for the Gaussian) is too large for a
small residual to show that the correction works. The floor is
therefore set by a tolerance: walking down from high CNR, it is the deepest
per-profile CNR reached while every cell satisfies |residual| + 2 combined SE
≤ tolerance. The tolerance is a fraction of the end-point MSD `2 D t_max` of
the default experiment (`t_max = 3τ`, D in σ_0²/τ):

| Tolerance (σ_0²) | Basis | Gaussian | Voigt | p = 1.5 | p = 4 |
| ------- | ------- | ------- | ------- | ------- | ------- |
| 0.003 | 0.1 × 2 D t_max, D = 0.005 | 7.27 | 9.45 | 12.3 | 3.31 |
| 0.009 | 0.3 × 2 D t_max, D = 0.005 | 1.96 | 3.31 | 3.31 | 1.96 |
| 0.012 | 0.1 × 2 D t_max, D = 0.02 | 1.51 | 2.55 | 2.55 | 1.51 |
| 0.036 | 0.3 × 2 D t_max, D = 0.02 | 1.16 | 1.16 | 1.51 | 0.89 |
| 0.060 | 0.1 × 2 D t_max, D = 0.1 | 1.16 | 0.89 | 1.16 | 0.14 |
| 0.180 | 0.3 × 2 D t_max, D = 0.1 | 0.89 | 0.065 | 0.89 | 0.050 |
| 0.010 | fixed tolerance | 1.96 | 2.55 | 3.31 | 1.96 |

The headline floor is the 0.036 σ_0² row: per-profile CNR 1.16 (Gaussian),
1.16 (Voigt), 1.51 (p = 1.5), and 0.89 (p = 4). A tight tolerance reports the
calibration budget as much as the correction: near CNR 2 the lookup's SE is
about 0.004 σ_0², so two combined SE nearly fill a 0.01 σ_0² tolerance before
any residual is counted.

Severity differs more than location. Uncorrected, the Gaussian fit inflates
the most at deep CNR (E[raw] 6.4 to 8.7 σ_0² below CNR 0.25), Voigt and
p = 1.5 reach 1.6 and 2.3 σ_0², and the flat-topped p = 4 fit reads *low*
(E[raw] 0.53 σ_0² at CNR 0.05), so for p = 4 the correction adds to the
variance there. (The table `per_profile_floor_genGauss.txt` is the p = 2
generalized-Gaussian fit, a Gaussian profile under that fitter's tighter
bound, run with an earlier 1000-run lookup and floor rule.)

### Analytical handle on where the model breaks

The model predicts its own bias without a fit. With `σ̂ = σ + ε` and
`ε ~ N(0, s²)`, `E[σ̂²] = σ² + s²`, so `var_bias = s² = Var(σ̂)`, the width's
estimation variance. At high CNR `s² = A / CNR²`, and `A` follows from the
Fisher information of the width fit on the calibration's pixel grid
(`width_debias.fisher.leading_order_terms`), with no Monte Carlo error.
`experiments/correction-floor/analytical_floor.py` predicts
`var_bias = A / CNR²` and compares it with the calibrated var_bias pooled into
log-CNR bins with standard errors (30 CNR₀ cells of 1000 runs). As a check, the
Monte Carlo spread `CNR² Var(σ̂²) / 4` at CNR ≥ 8 reproduces each Fisher A:

| shape | A (σ_0², Fisher) | Fisher / Monte Carlo | departure (> 3 SE) | below it |
| ------- | ------- | ------- | ------- | ------- |
| Gaussian | 0.0564 | 0.986 | CNR ≈ 1.18 | bias 2.0 × A/CNR² at 1.18, up to 8.9 × at 0.48 |
| Voigt | 0.0812 | 0.998 | CNR ≈ 0.85 | 1.45 to 1.62 × at 0.48 to 0.85 |
| generalized-Gaussian p = 1.5 | 0.1007 | 1.006 | CNR ≈ 1.18 | 1.7 to 3.6 × at 0.48 to 1.18 |
| generalized-Gaussian p = 4 | 0.0191 | 1.001 | CNR ≈ 1.18 | negative (−1.4 ± 0.2 × at 1.18) |

So the leading-order model breaks near CNR 1, at the same place as the
per-profile working floor. The flat-topped p = 4 fit departs differently: its
low-CNR bias is negative (−0.35 σ_0² at CNR 0.12), and the fit underestimates
the width instead of inflating it.

Below the breakdown the calibration still captures the empirical mean, so a
subtraction can remain unbiased in expectation if the calibration resolves
that mean; the per-profile tables show the residual stays small but that a
5000-run lookup does not resolve it to a useful tolerance there. Fitting A to
the calibration's mean bias instead of taking it from the Fisher information
is noise-limited, because above CNR about 3 the bias is below the cells' Monte
Carlo noise.

### Design recommendation that follows

A measurement is bias-correctable end-to-end if **every frame's
per-profile CNR exceeds the working floor**. Equivalently, the deepest
frame's CNR — `CNR₀ · exp(−t_max/τ)` — must satisfy

    CNR₀ · exp(−t_max/τ)  ≳  1.

For the typical `t_max = 3 τ` this means `CNR₀ ≳ exp(3) ≈ 20`. If
experimental constraints prevent that, the late frames must be discarded
or replaced with denser early-time sampling so the slope fit is
dominated by frames inside the working regime.

## 10. A multiplicative variant, not implemented

Two cases fit the Section 8 lookup least well. In the first, the width grows
substantially over the measurement, so the bias calibrated at `σ_0²` is the
wrong size for the later frames. In the second, a fraction of low-CNR fits peg
at the fitter's upper bound `var_max`, and that tail pulls the calibrated mean
upward by an amount that scales with `var_max`. A robust multiplicative variant
addresses both. It summarises each calibration cell by its median and median
absolute deviation instead of the mean and standard deviation, and it divides
each fitted variance by `1 + g(CNR)`, with `g = bias_cal(CNR) / σ_0²`, instead
of subtracting the bias.

The division assumes that the bias is proportional to `σ²_true`. The Fisher
calculation does not support that assumption: at a fixed pixel grid and
peak-defined CNR the leading-order bias grows as `σ` (Section 3;
`width_debias.fisher`), so the division over-scales the bias by `σ/σ_0` when
the width grows. The package does not implement the variant. For a width that
grows over the measurement it provides the width-scaled 1-D lookup
(`calibrate_fisher(..., width_scaling=True)`) and the joint `var_bias(CNR, σ²)`
lookup of Section 8; both are experimental.

## 11. Scope and limitations

- **Subdiffusive transport (`MSD ∝ t^α`, α < 1).** The correction
  de-biases α as well as D: `experiments/subdiffusive-alpha/subdiffusive_alpha.py` on the
  `AnomalousExperiment` bed, over CNR₀ 21 to 90 (2000 runs per cell, Monte
  Carlo SE from resampling the realisations), the signed corrected error
  α − α_ref pooled over the cells is +0.004 ± 0.008, −0.011 ± 0.010,
  +0.019 ± 0.014, and +0.049 ± 0.016 for α = 0.6, 0.8, 1.0, and 1.2, against
  +0.024 to +0.081 naive (z 2.2 to 4.9). The SEs leave out the shared
  calibration error, so the α = 1.2 residual (z = 3.0) is not yet tested
  against it. Below CNR₀ 21 the naive fit pins at the ballistic α = 2.0. The
  exponent is read from the mean trajectory (§4's bias carries into the
  slope's curvature, not only its slope).
- **Decay law.** The correction subtracts
  `var_bias(per-frame CNR)`, so it depends on the per-frame CNR, not on how the
  population decays. `experiments/multiexp-decay/multiexp_decay.py` (the `MultiExpExperiment` bed)
  confirms the per-frame residual collapses across single- and bi-exponential
  decay for per-frame CNR ≥ 1 (across-law spread ≤ 0.012 σ₀²) and corrected D
  stays within 0.93 to 1.06 of D_ref over CNR₀ 8 to 90 for three decay laws.
  Thresholds quoted in the *initial* CNR₀ (the §9 design
  rule `CNR₀ ≳ exp(t_max/τ)`) are single-exponential-specific; the invariant
  form is the per-frame `CNR ≳ 1` of §9.
- **Wall pile-up at CNR ≈ 1.** The Voigt and generalized-Gaussian p = 1.5 fits
  pile up at the lower bound in 0.05 to 0.06 of fits at CNR ≈ 1; a small
  censored term on top of the noise-bias correction may be needed for the
  deepest-noise frames.
- **Not tested:** heterogeneous transport with site-dependent mobility (an
  effective `D` from a mixed population), and a trapped-exciton background
  (a non-zero `B(x)` added to each frame).
