# Folded-normal derivation of the MSD bias and its correction

This note derives, from a folded-normal model of the fitted width, why the
diffusion coefficient is overestimated and what the correct fix is. The model
is the one the boundary diagnostic selected:
`experiments/boundary-diagnostic/diagnose_boundary.py` found the
fitted-variance distribution strictly interior and right-skewed with no
delta-mass at the lower bound for the Gaussian, Voigt, and generalized-Gaussian
p = 1.5 and p = 4 fits (a wall pile-up of 0.05 to 0.06 of fits appears only at
CNR ≈ 1 for Voigt and p = 1.5) — **reflection, not clipping**.

## 1. Model

At each frame *i* the analysis fits a profile and reports a width. A fitted
width cannot be negative. Reflection at that wall means the latent width
estimate behaves as the absolute value of a Gaussian: writing the latent
estimate as `X_i`, the reported width is `|X_i|` with

    X_i ~ N(σ_i, s_i²),

where `σ_i` is the true width at frame *i* and `s_i` is the per-frame
estimation noise (a decreasing function of the per-frame CNR). The fitted
**variance** is therefore the square of a Gaussian,

    σ̂²_i = |X_i|² = X_i².

Equivalently `σ̂²_i / s_i²` is a non-central χ²₁ with non-centrality `(σ_i/s_i)²`
— continuous, strictly positive, right-skewed, exactly the shape the diagnostic
saw.

## 2. Moments

For `X ~ N(σ, s²)`, using `E[X²] = σ² + s²` and `E[X⁴] = σ⁴ + 6σ²s² + 3s⁴`:

    E[σ̂²]   = σ² + s²
    Var[σ̂²] = E[X⁴] − E[X²]² = 4σ²s² + 2s⁴ = 2s²(2σ² + s²)

**The fitted variance carries an additive upward bias equal to the estimation
noise variance `s²`.** This is the width analogue of the squared-magnitude
corrections used for Rician magnitude data and for polarization amplitudes; the
specific references are leads in `prior_art_map.md` and are not yet audited in
`citations_audit.md`, so none is cited here.

## 3. The per-frame truncation fraction

Define the latent width signal-to-noise `α_i = σ_i / s_i`. The truncation
fraction — the mass of the latent estimator below the wall — is

    p_i = P(X_i < 0) = Φ(−α_i),     so   α_i = −Φ⁻¹(p_i).

`p_i` is a clean, dimensionless readout of `α_i`, and on real data it follows
from the per-frame CNR (via `fft-cnr`). In terms of `p`:

- relative bias of σ̂²:      `s²/σ² = 1/α²              = 1 / Φ⁻¹(p)²`
- relative variance of σ̂²:  `Var[σ̂²]/σ⁴ = 2(2α²+1)/α⁴`, with `α = −Φ⁻¹(p)`

So the *shape* of the bias and variance is fixed by `p` alone. The *scale* is
not: `p` is dimensionless, the bias `s²` has units of variance. The absolute
scale must come from the per-frame CNR through a noise model (the Cramér–Rao
bound on the width, `s² ∝ σ dx / CNR²` to leading order for a 1-D Gaussian at a
fixed pixel spacing `dx` and peak-defined CNR; `width_debias.fisher`) or by
calibration against matched simulations. Both `p` and the CNR scale are needed — which is why the
weighting takes the per-frame CNR as its input, not `p` alone.

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
   reaches the Cramér–Rao bound on D (0.98 to 1.08 × √CRLB from CNR₀ 21 on four
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
model in `BiasLookup`. The folded-normal model is what licenses this:
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
At very low CNR₀ (≲ 4) the MAS gate switches the correction off where the
noise bias is large, leaving most of the raw inflation in. On
`experiments/dfree-calibration/dense_calibration_and_test.py`'s 100-point
sweep (5000 calibration runs per cell, 1000 test runs; the committed
`dense_calibration_and_test.txt`), the overall mean |⟨D⟩/D_ref − 1| is,
folded-normal vs MAS: Gaussian 0.119 vs 0.438, Voigt 0.175 vs 0.167, and the
p = 2 generalized-Gaussian fit 0.155 vs 0.151 (that row is a Gaussian
profile); the non-Gaussian rows (`dense_calibration_and_test_genGauss_p1.5_p4.txt`)
give 0.178 vs 0.815 for p = 1.5 and 0.179 vs 0.509 for p = 4. Folded-normal
wins clearly where MAS leaves a large deep-CNR inflation in (Gaussian,
p = 1.5, and p = 4, whose CNR₀ < 4 deciles MAS sends negative); for Voigt and
the p = 2 fit the two tie. The medians are close (0.02 to 0.07 for
both schemes on every shape), so for typical measurements either scheme works. The case for
folded-normal is the failure mode, not the mean: it degrades gracefully in
deep noise (residuals erratic but bounded), while MAS fails with large
positive residuals that grow with CNR depth when its gate switches the
correction off. **Folded-normal is the recommended scheme.** MAS is available
in the package for comparison.

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

The fix is small. The folded-normal bias `s²(CNR)` is dominated by per-frame
CNR; the dependence on `σ²(t)` is weak (the second-order term in the
folded-normal expansion). In the low-`D` regime where the noise bias
bites, `σ²(t)` barely changes from `σ_0²` across the measurement window, so
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
(`experiments/dfree-calibration/slope_fit_check.txt`). At these budgets the
calibration and evaluation terms are comparable, and the calibration term does
not shrink with more evaluation runs.

With both terms in the cell SE and the per-frame slope weights, 4 of 96 cells
(four shapes; D = 0.005, 0.02, 0.1; CNR₀ 3 to 90) lie beyond two combined SE
(`slope_fit_check.txt`). The cells share noise seeds across shapes and one
calibration per shape, so this count is indicative, not a test against a
binomial null. The small-D, low-CNR cells
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
per CNR cell and the dense calibration. For Gaussian profiles at
`σ²_true = σ_0²`:

| per-profile CNR | folded-normal residual (σ_0²) | reading |
| ------- | ------- | ------- |
| ≳ 1 | \|residual\| ≲ 0.01 | correction excellent (≲ 0.01 of σ²_true) |
| 0.7 – 1.0 | up to ≈ 0.05 | useful; a few hundredths of σ²_true |
| 0.5 – 0.7 | 0.1 – 0.3 | unreliable per realisation; mean still toward the reference |
| 0.3 – 0.5 | 0.3 – 0.8 | failing; residuals comparable to σ²_true |
| < 0.3 | ≳ σ²_true | broken |

The same sweep on the other shapes (`per_profile_floor_{voigt,genGauss_p1.5,genGauss_p4}.txt`),
with the floor taken as the deepest CNR reached, walking down, before the
first cell whose residual exceeds both 2 SE and 0.01 σ_0²:

| shape | floor (CNR) | first failure below it |
| ------- | ------- | ------- |
| Gaussian | 1.16 | +0.036 σ_0² at 0.89, −0.34 at 0.69 |
| Voigt | 1.5 | −0.013 at 1.16; within 0.01 again at 0.89 and 0.69; +0.030 at 0.53; +0.34 at 0.41 |
| generalized-Gaussian p = 1.5 | 0.89 | −0.12 at 0.69 |
| generalized-Gaussian p = 4 | 0.69 | +0.013 at 0.53; within 0.012 down to 0.11 |

Severity differs more than location. Uncorrected, the Gaussian fit inflates
the most at deep CNR (E[raw] ≈ 6–8 σ_0² below CNR 0.25), Voigt and p = 1.5
reach 1.5–2.3 σ_0², and the flat-topped p = 4 fit reads *low* (E[raw] 0.55
σ_0² at CNR 0.05), so for p = 4 the correction adds to the variance there. (The table
`per_profile_floor_genGauss.txt` is the p = 2 generalized-Gaussian fit, a
Gaussian profile under that fitter's tighter bound.)

### Analytical handle on where the model breaks

The model predicts its own bias without a fit. With `σ̂ = σ + ε` and
`ε ~ N(0, s²)`, `E[σ̂²] = σ² + s²`, so `var_bias = s² = Var(σ̂)`, and for a
small width error `Var(σ̂²) = 4σ² Var(σ̂)`. Var(σ̂²) is measured to a few
parts in a hundred per calibration cell at any CNR, while the mean bias
above CNR ≈ 3 is below 0.01 σ_0² and noise-dominated in a 300-run cell.
`experiments/correction-floor/analytical_floor.py` therefore takes
`A = CNR² Var(σ̂²)/4` at `CNR ≥ 8`, predicts `var_bias = A / CNR²`, and
compares the prediction with the calibrated var_bias pooled into log-CNR
bins with standard errors:

| shape | A (σ_0²) | model holds down to | resolved departure |
| ------- | ------- | ------- | ------- |
| Gaussian | 0.058 | CNR ≈ 1.2 | CNR ≈ 0.85 (bias 3.8x the prediction) |
| Voigt | 0.082 | CNR ≈ 0.3 | none located above 0.28 |
| generalized-Gaussian p = 1.5 | 0.10 | CNR ≈ 1.2 | CNR ≈ 0.85 (3.0x) |
| generalized-Gaussian p = 4 | 0.019 | CNR ≈ 3.6 | CNR ≈ 0.36 (bias negative) |

The Gaussian A agrees with the 50,000-fit per-profile tables (0.05 to 0.07
σ_0² at CNR 1.5 to 6). So the folded-normal model breaks near CNR 1 for the
Gaussian and p = 1.5 fits, at the same place as the per-profile working
floor. The flat-topped p = 4 fit departs differently: its low-CNR bias is
negative (−0.38 σ_0² at CNR 0.12), and the fit underestimates the width
instead of folding it upward.

Below the breakdown the calibration still captures the empirical mean, so a
subtraction can remain unbiased in expectation if the calibration resolves
that mean, and the per-profile tables show it does not at practical budgets.
Fitting A directly to the high-CNR calibration cells instead of taking it from
the spread overestimates it several-fold, because the bias there is below the
cells' Monte Carlo noise.

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
  `AnomalousExperiment` bed: the mean corrected |α−α_ref| over CNR₀ 21 to 90 (2000 runs per cell, Monte Carlo SE from resampling the realisations) is 0.030 ± 0.020, 0.018 ± 0.022, 0.046 ± 0.023, and 0.100 ± 0.041 for α = 0.6, 0.8, 1.0, and 1.2, against 0.18 to 0.30 naive. The corrected error is consistent
  with zero for α = 0.6 and 0.8 and about 2 SE for α = 1.0 and 1.2. Below
  CNR₀ 20 the naive fit pins at the ballistic α = 2.0. The exponent is read from the mean trajectory (§4's bias
  carries into the slope's curvature, not only its slope).
- **Decay law.** The correction subtracts
  `var_bias(per-frame CNR)`, so it depends on the per-frame CNR, not on how the
  population decays. `experiments/multiexp-decay/multiexp_decay.py` (the `MultiExpExperiment` bed)
  confirms the per-frame residual collapses across single- and bi-exponential
  decay for per-frame CNR ≥ 1 (across-law spread ≤ 0.012 σ₀²) and corrected D
  stays within 0.89–1.11. Thresholds quoted in the *initial* CNR₀ (the §9 design
  rule `CNR₀ ≳ exp(t_max/τ)`) are single-exponential-specific; the invariant
  form is the per-frame `CNR ≳ 1` of §9.
- **Wall pile-up at CNR ≈ 1.** The Voigt and generalized-Gaussian p = 1.5 fits
  pile up at the lower bound in 0.05 to 0.06 of fits at CNR ≈ 1; a small
  censored term on top of the noise-bias correction may be needed for the
  deepest-noise frames.
- **Not tested:** heterogeneous transport with site-dependent mobility (an
  effective `D` from a mixed population), and a trapped-exciton background
  (a non-zero `B(x)` added to each frame).
