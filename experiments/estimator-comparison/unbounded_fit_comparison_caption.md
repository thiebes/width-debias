# Caption -- unbounded_fit_comparison.png

Removing the positivity bound from the width fit does not remove the noise
bias. (a) Mean and (b) standard deviation of the fitted variance of isolated
Gaussian profiles at σ² = σ₀² against per-frame CNR, 15,000 fits per cell, for
the bounded fit (variance > 0, blue) and an unbounded fit (the Gaussian fitted
in precision 1/σ², so the variance can take either sign, gold). From per-frame
CNR 1.5 up the two fits are identical and both read high (1.038, 1.016, and
1.010 σ₀² at CNR 1.52, 2.0, and 2.6); below that the unbounded fit lowers the
mean (2.03 against 5.63 σ₀² at CNR 0.3) but its SD stays near 10 σ₀². (c) Mean
D̂/D_true and (d) its SD against CNR₀ in the full experiment (D = 0.02 σ₀²/τ,
12 frames to t = 3τ, 2500 runs per CNR₀) for the bounded fit with OLS (blue),
the bounded fit with the `subtract` correction (green; D-free calibration, 50
CNR₀ × 1500 runs), and the unbounded fit with OLS (gold). Both OLS arms read
1.199 at CNR₀ 34. `subtract` reads 1.003 to 1.028 from CNR₀ 21 up, where every
frame has per-frame CNR ≥ 1, and 0.82 to 1.03 below, where the last frames fall
under the floor; its SD is 0.19 to 8.3 against 0.60 to 132 for the bounded OLS
fit. The moment estimator gives a negative mean D at every CNR₀ (down to
−1.2 × 10⁵ D_true at CNR₀ 4) and is left out; its values are in
`unbounded_fit_comparison.txt`.
