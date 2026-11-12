# Caption -- read_floor_sweep.png

Bias of the MSD-slope D against a constant detector read floor added to
Poisson shot noise: Gaussian profile, D = 0.02 σ₀²/τ, 12 frames to t = 3τ under
single-exponential decay. The vertical axis is the mean over 8 CNR₀ from 3 to
60 of |⟨D̂⟩/D_true − 1|, with 1000 evaluation runs per CNR₀; every calibration
is D-free, built from 30 CNR₀ in [2, 80] × 1000 runs. The read floor is in the
same counts as the peak, which holds CNR₀² counts at frame 0, so read 0 is
photon-limited. The horizontal axis is symmetric-logarithmic: linear between 0
and 1 count, logarithmic above. Uncorrected OLS (gold) is 0.19 on
photon-limited data and rises with the floor to 14.4 at 4 counts and 20.7 to
26.0 at 8 to 32 counts, above its value on pure white noise (18.6, gold dotted
line). `subtract` with a calibration built at the same read floor (green) stays
at 0.014 to 0.025 up to 2 counts, rises to 0.12 to 0.13 at 4 and 8 counts, and
to 0.97 to 1.27 at 16 and 32, where the CNR₀ 3 cell starts below per-frame CNR
1 (0.55 and 0.28 at frame 0). `subtract` with the default white-noise
calibration (blue) stays at 0.21 to 0.35 at every read floor, against 0.098
when it is applied to white data (blue dotted line); on photon-limited data it
over-subtracts, and D̂ goes negative at CNR₀ 3 (`shot_noise_validation.txt`).
The metric is unsigned and includes Monte Carlo noise, so an unbiased estimator
does not score zero; the table gives no standard errors. Table:
`read_floor_sweep.txt`.
