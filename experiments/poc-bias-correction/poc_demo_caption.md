# Caption -- poc_demo.png

Proof of concept on a Gaussian profile: D = 0.02 σ₀²/τ (L_D/FWHM₀ = 0.085),
12 frames to t = 3τ under single-exponential decay, 400 runs per CNR₀.
(a) Fraction of MSD points at or below zero and (b) MSD sample skewness against
per-frame CNR, one point per frame of each CNR₀ cell after frame 0 (the MSD
reference, where MSD = 0 by construction). The fraction depends on
the frame's true MSD as well as its CNR, so early frames of high-CNR₀ cells
also sit high; the skewness rises steeply below a per-frame CNR of about 1.
(c) Mean D̂/D_true against CNR₀ for unweighted OLS (blue), inverse-variance
weighting (gold), and the `subtract` correction (green), the last with a
D-free calibration (7 CNR₀ × 400 runs, seeds independent of the evaluation).
The vertical axis is symmetric-logarithmic: linear between −2 and 2,
logarithmic above. OLS reads 46 times D_true at CNR₀ 5. `subtract` lies
within 2 combined SE (evaluation and calibration) of 1 in all 7 cells,
including 2.67 ± 0.86 at CNR₀ 5, where the last frames fall below per-frame
CNR 0.3; inverse-variance weighting leaves 1 of 7 cells beyond 2 SE. Table:
`poc_demo_gaussian.txt`.
