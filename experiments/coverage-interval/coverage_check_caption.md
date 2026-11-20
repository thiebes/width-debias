# Caption -- coverage_check.png

Interval check for the corrected diffusion coefficient: Gaussian profile,
D = 0.02 σ₀²/τ, 12 frames to t = 3τ under single-exponential decay, 13
log-spaced CNR₀ from 5 to 90, 8000 runs per CNR₀, slope weights from the D-free
lookup. (a) Fraction of runs whose 0.95 interval contains D_true, for the
covariance interval on the `subtract` estimate (blue), a parametric bootstrap
percentile interval on the same estimate (gold, 2000 resamples, with per-frame
variances measured over the ensemble), and the covariance interval on the
uncorrected inverse-variance estimate (green); error bars are binomial SEs
(0.0024) and the dotted line is the nominal 0.95. Pooled over the sweep, the
covariance interval covers 0.944 and the bootstrap 0.951. (b) The covariance SE
of D over the realised SD of D̂, with batch-mean SEs over 8 sub-ensembles:
0.967 ± 0.003 averaged over the 7 cells whose every frame has per-frame CNR ≥
1 (CNR₀ 21 to 90). The covariance interval runs about 0.03 narrow because the
lookup's per-frame variances are calibrated at σ₀², while the width grows to
1.12 σ₀² by the last frame (`narrowness_attribution.txt`). The bootstrap
resamples with the per-frame variances measured over the ensemble and covers
at the nominal rate, so the shortfall lies in those variances, not in the
interval's form. All SEs are conditional on the one calibration.
