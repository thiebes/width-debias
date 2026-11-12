# Caption -- poc_demo_generalized_gaussian_p4.png

Proof of concept on a flat-topped generalized-Gaussian profile (shape exponent
p = 4, fitted by scale with p fixed): D = 0.02 σ₀²/τ, 12 frames to t = 3τ under
single-exponential decay, 400 runs per CNR₀. (a) Fraction of MSD points at or
below zero and (b) MSD sample skewness against per-frame CNR, one point per
frame of each CNR₀ cell after frame 0 (the MSD reference, where MSD = 0 by
construction); the fraction depends on the frame's true MSD as well
as its CNR. (c) Mean D̂/D_true against CNR₀ for unweighted OLS (blue),
inverse-variance weighting (gold), and the `subtract` correction (green), the
last with a D-free calibration (7 CNR₀ × 400 runs, seeds independent of the
evaluation). The vertical axis is symmetric-logarithmic: linear between −2 and
2, logarithmic above. On this shape the width bias is small and turns negative
below a per-frame CNR of about 1, so OLS reads low rather than high (0.13 at
CNR₀ 5, −0.17 at CNR₀ 8). `subtract` lies within 2 combined SE (evaluation and
calibration) of 1 in all 7 cells, and inverse-variance weighting alone also
leaves none beyond 2 SE. Table: `poc_demo_generalized_gaussian_p4.txt`.
