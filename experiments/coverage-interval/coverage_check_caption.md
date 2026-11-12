# Caption -- coverage_check.png

Interval check for the corrected diffusion coefficient: Gaussian profile,
D = 0.02 σ₀²/τ, 12 frames to t = 3τ under single-exponential decay, 13
log-spaced CNR₀ from 5 to 90, 8000 runs per CNR₀, slope weights from the D-free
lookup. (a) Fraction of runs whose 0.95 interval contains D_true, for the
covariance interval on the `subtract` estimate (blue), a parametric bootstrap
percentile interval on the same estimate (gold, 200 resamples), and the
covariance interval on the uncorrected inverse-variance estimate (green); error
bars are binomial SEs (0.0024) and the dotted line is the nominal 0.95. The
covariance interval covers 0.944 pooled over the sweep. (b) Two ratios of
standard errors, with batch-mean SEs over 8 sub-ensembles. Blue: the covariance
SE of D over the realised SD of D̂, 0.969 ± 0.002 pooled from CNR₀ 21 up; the
interval runs about 0.03 narrow because the lookup's per-frame variances are
calibrated at σ₀², while the width grows to 1.12 σ₀² by the last frame
(`narrowness_attribution.txt`). Gold: the factor by which the shared frame-0
width would change the slope SE relative to treating the MSD points as
independent; it lies below 1 (0.864 to 0.892) because the shared width shifts
only the intercept.
