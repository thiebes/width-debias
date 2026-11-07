# Caption: br_halide_highn_perframe.png

**Figure N.** Noise bias in the per-frame width for three Br-I halide-mixing
maps, Br fraction (a) 0.25, (b) 0.5, (c) 0.75, on high-statistics
(400,000-walker) maps at initial contrast-to-noise CNR₀ = 15 with white noise.
Each point is the mean fitted variance over 800 noisy realizations minus the
Gaussian fit of the same noise-free profile, Δσ², so it isolates the noise
bias; bars are standard errors. Red circles: the naive bounded fit. Blue
squares: the noise-bias correction, with the bias read from a joint (CNR,
σ²) calibration (40 CNR × 26 σ × 4000 runs) keyed on a robust straight line
through each realization's own fitted variances, refitted to the corrected
values. The indigo line is the separate offset between the noise-free fit and
the carrier reference (the second moment of the walker distribution): the
shape and emission-to-carrier gap, which no noise correction removes. Frames
in the shaded band fall below the per-frame CNR floor of 1.3 and are excluded
from the fit and from the plot.

Over the unshaded frames the mean |Δσ²| falls from 124, 107, and 140 nm²
(naive) to 39, 38, and 63 nm² (corrected). The short-window diffusion
coefficient, relative to the D of the noise-free fit, goes from 1.034, 1.049,
and 1.060 (naive) to 1.002, 1.013, and 1.025 (corrected), each with a standard
error of 0.006 to 0.007 from resampling the realizations; the x = 0.75
remainder of 0.025 is resolved. Relative to the carrier reference D_ref the
corrected values are 0.993, 0.971, and 0.984, the difference being the
carrier gap.

CNR₀ = 15 is the lowest initial contrast at which the corrected D does not
depend on the choice of key: a key error of 0.2 of σ² moves D by at most 0.005
there, while at CNR₀ 8 to 10 it can move D by up to 0.3
(`key_sensitivity.py`). The straight-line key assumes the variance grows
linearly in time, which holds for these maps (exponent 0.990 to 0.997).
