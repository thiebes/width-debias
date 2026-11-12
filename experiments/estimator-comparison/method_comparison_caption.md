# Caption -- method_comparison.png

Mean D̂/D_true against CNR₀ for four schemes on the same Gaussian-fit data:
unweighted OLS (indigo), inverse-variance weighting (ochre), the `subtract`
correction (blue), and the MAS gated subtraction (green). D = 0.02 σ₀²/τ, 12
frames to t = 3τ under single-exponential decay, 400 runs per CNR₀; the bias
lookup is calibrated at the evaluation D (the legacy D-aware path, seeds
1 + i against evaluation seeds 20000 + i). OLS reads 46 times D_true at CNR₀ 5.
`subtract` and MAS lie within 0.03 of each other from CNR₀ 8 up (2.68 and 2.76
at CNR₀ 5), and both read 0.77 to 0.80 at CNR₀ 8, where the last frames fall
below per-frame CNR 0.5. The moment estimator, run on the same profiles, gives a
negative mean D at 5 of 7 CNR₀ and cannot be drawn on the logarithmic axis; its
values are in the script's printed table.
