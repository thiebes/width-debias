# Noise-induced bias in the measured transport exponent for disordered nanocrystal-solid diffusion maps

Technical note on the width-debias method applied to the disordered perovskite
nanocrystal-solid simulations (Marc Meléndez's maps from the Solari et al.
hopping model, made for this work; not the runs behind the paper's Fig. 5).
Numbers and figures are produced by [`nc_solid_alpha_vs_cnr.py`](nc_solid_alpha_vs_cnr.py)
(1000 noise realisations per cell); figures are written to `output/`.

## Summary

A bounded width fit followed by a mean-squared-displacement (MSD) slope fit
mis-estimates the transport exponent α when the per-frame contrast-to-noise
ratio (CNR) is low. We build a Gaussian test bed whose per-frame width is taken
from the nanocrystal-solid maps, so its true exponent is subdiffusive (α ≈ 0.88)
and exactly known. Under white measurement noise a standard all-frames fit reads
the motion as ballistic (α at the fit ceiling of 2) until the initial CNR exceeds
roughly 50. The bias direction is fixed: it always
raises the apparent exponent. It can therefore hide subdiffusion but cannot
manufacture it, so a measured subdiffusive exponent is robust to this effect.
The size of the bias is set by the constant, signal-independent part of the
noise: under pure photon-limited (shot) noise it nearly vanishes, and a read
floor of a few counts restores it. The simulation fixes the threshold only for
an assumed noise law; the real measurement's regime is set by its detector and
is measurable from the raw frames.

## A Gaussian test bed from the maps' width trajectory

Three maps (`small`, `medium`, `large`) record 20 frames of a normalised
exciton concentration on a 250 × 250 grid over a 1500 × 1500 nm² area
(6 nm per pixel, 1 ns per frame). They track exciton position only, with no
recombination. To model a transient-photoluminescence measurement we impose the
population decay N₀·exp(−t/τ) with a representative thin-film lifetime τ = 7 ns
(per-sample lifetimes 6 / 14 / 18 ns for the C16 / C12 / C8 ligands), and
project each 2-D frame to a 1-D profile by summing over one axis. The decay is a
per-frame scalar, so it sets the amplitude — hence the per-frame CNR — without
changing the spatial second moment.

We fit a Gaussian to each noiseless frame and take those Gaussians as the truth:
a test bed whose per-frame width σ(t) and amplitude are the maps', but whose
ground truth is Gaussian by construction. This makes no claim that the real
exciton profile is Gaussian — it defines a realistic-width Gaussian bed on which
the noise bias is measured against an exactly known truth. Fitting a Gaussian
also sets aside the per-bin Poisson noise of the 50,000-walker histograms, which
would otherwise inflate the peak (the raw `max` rises with time instead of
falling as 1/σ). The true D and α follow from σ²(t); the diffusion length is
small relative to the spot (L_D / FWHM ≈ 0.14), the regime characteristic of
nanocrystal solids.

| map | true α | true D (μm²/ns) |
| ------- | ------- | ------- |
| small | 0.883 | 1.46 × 10⁻⁴ |
| medium | 0.884 | 1.52 × 10⁻⁴ |
| large | 0.875 | 1.99 × 10⁻⁴ |

All three are subdiffusive in the ground truth, so the subdiffusive transport is
a property of the simulated disorder, not of any fitting choice.

## The bias under white noise, and the CNR threshold

We add white (constant-standard-deviation) Gaussian noise to the decayed
profiles, fit each frame's width with a bounded Gaussian, form
MSD(t) = σ̂²(t) − σ̂²(0), and fit MSD = 2D·t^α with α bounded to [0.2, 2]. The
initial CNR, CNR₀, is the frame-0 peak height over the noise standard deviation;
as the spot decays and spreads, the per-frame CNR falls below CNR₀.

The reported bias is the exponent of the seed-averaged MSD trajectory, i.e. the
exponent that survives averaging many measurements. Averaging removes random
scatter but not this bias: it is a property of the estimator's mean, not of any
one realisation.

Figure 1 (`output/nc_solid_alpha_vs_cnr.png`) shows the apparent exponent
against CNR₀ for the three maps. Below CNR₀ ≈ 12 the fit sits at the ceiling
α = 2 (ballistic). The true subdiffusive value returns only once the deepest
fitted frame keeps a CNR of about 2 to 3, which here corresponds to CNR₀ ≈ 35 to
50 across the three maps. These thresholds are the bed's own: because the truth
is the Gaussian fit and the only noise added is the controlled measurement noise,
they are not limited by the maps' sampling noise. The one caveat is that the bed
is Gaussian — faithful here because the maps are near-Gaussian (kurtosis ≈ 0),
but a strongly non-Gaussian profile could shift them.

| map | CNR₀ at recovery | deepest-frame CNR there | single-shot s.d. of α |
| ------- | ------- | ------- | ------- |
| small | 50 | 3.2 | ±0.59 |
| medium | 50 | 3.2 | ±0.59 |
| large | 35 | 2.2 | ±0.61 |

Recovery is defined as |α_bias − α_true| ≤ 0.05. The last column is the
standard deviation of the exponent across single realisations at the recovery
point; it remains large (a few tenths) until CNR₀ ≈ 100, so a single
low-contrast measurement is imprecise as well as biased.

Three points bound the reading:

- The value α = 2 is the fit's upper bound. The unconstrained fit would return
  α ≥ 2, so "ballistic" here means "pushed to the ceiling and clipped", not a
  measurement of exactly 2.
- The small-diffusion-length regime amplifies the effect. The spatial variance
  grows only from about 0.040 to 0.044 μm² across the run, so even a modest
  late-time bias dominates the slope. A measurement with larger spreading would
  show a smaller fractional effect on α.
- The exponent is more fragile to noise than the diffusion coefficient. The
  exponent needs a deepest-frame CNR of about 2 to 3, above the CNR ≈ 1 at
  which the diffusion coefficient is recoverable, so discarding only the
  sub-CNR-1 frames does not protect α.

The sample lifetime sets how fast the contrast falls and therefore the
threshold. On the `small` map, a 7 ns lifetime requires CNR₀ ≈ 50, but a
14–18 ns lifetime recovers the exponent already at CNR₀ ≈ 20 (deepest-frame CNR
9–12). The longer-lived samples are well protected; the shorter-lived ones are
where the late frames fall into the difficult range.

## The noise law governs the bias

White noise has a constant floor that persists in the low-signal wings of the
profile, where the bounded width fit takes its leverage. Photon-limited (shot)
noise instead follows the signal and fades in the wings. The two laws therefore
differ in how strongly they drive the bias. We compare them on the `small` map
by replacing white noise with shot noise (Poisson counts scaled so the frame-0
peak holds CNR₀² counts, with an optional constant read-noise floor), holding
CNR₀ fixed.

Figure 2 (`output/nc_solid_noise_law.png`) shows the result. Under pure shot
noise (no read floor) the apparent exponent stays near the true value across
the whole range; at the lowest CNR₀ it falls slightly below the truth rather
than above it. A read floor of 3 counts restores the upward bias but mildly,
recovering by CNR₀ ≈ 12; a floor of 10 counts approaches the white worst case.

| noise law | exponent recovered by | behaviour at low CNR₀ |
| ------- | ------- | ------- |
| white (constant floor) | CNR₀ ≈ 35–50 | pinned at the ceiling α = 2 |
| shot, read = 10 counts | CNR₀ ≈ 20–35 | near white |
| shot, read = 3 counts | CNR₀ ≈ 12 | pinned only at the lowest CNR₀ |
| pure shot, read = 0 | CNR₀ ≈ 8 | falls slightly below the true α |

At equal CNR₀ the shot case is more forgiving for two compounding reasons: the
clean wings weaken the bias mechanism, and the shot CNR falls more slowly with
signal (as the square root of the amplitude rather than linearly), so the
deepest frame keeps a higher CNR. The slight dip below the true exponent under
pure shot noise is consistent with the sign change of the single-frame
width-variance bias under photon-limited noise reported in the fft-cnr
validation cases.

This is the established noise-law result for the noise bias, here confirmed
on the nanocrystal-solid maps with true Poisson sampling: the bias is set by the
contrast against the constant noise floor, not by the total noise. Differential
or pump-probe techniques, which ride on a bright constant pedestal, are
effectively white and fully in the bias regime; direct photon-counting imaging
is closer to the shot end, with its read floor and background setting how much
bias appears.

## Direction, and the reading for the measured exponent

The bias raises the apparent exponent in every case where it acts. It can move a
true α ≈ 0.87 toward 1 or up to the ballistic ceiling, but it cannot move a true
exponent downward (the one exception, the mild dip under pure shot noise, also
does not create subdiffusion where there is none). A measured subdiffusive
exponent is therefore robust to this effect: if anything, residual bias means
the true transport is slightly more subdiffusive than the fit reports.

This separates cleanly into two cases. The simulated subdiffusion is not at
issue: the bed's truth is the disordered-transport width trajectory itself, with
no measurement noise, so α ≈ 0.88 is the simulated physics, not a noise artifact.
For an experimental exponent, recovering α < 1 in this
fragile regime implies that the measurement stayed in the high-contrast part of
the data — through some combination of bright signal, a fit window that stops
before the contrast falls, discarding the late low-contrast frames, or operating
near the photon-limited regime where the bias is weak. Which of these applies is
not determined by the simulation.

## Why the bed is built from Gaussian fits, not the raw maps

The maps are histograms of about 50,000 discrete walkers per frame (pixel values
are exact multiples of 1/49984), so each frame carries Poisson sampling noise.
Read through the raw peak, that noise inflates the per-frame CNR: the histogram
`max` rides the sampling noise and rises with time instead of falling as 1/σ.
Fitting a Gaussian to each frame removes that — it sets the bed's truth from the
smooth width and amplitude, and the measurement noise we then add is the only
noise in the study. So the sampling noise does not propagate into the bias
result; it is the reason to fit rather than to read the raw peak.

The cost is the Gaussian assumption. Here it is benign: the maps are
near-Gaussian (generalized-Gaussian shape exponent p ≈ 1.9 to 2.0, kurtosis ≈ 0),
so the fitted Gaussian and the actual profile agree, and the bias measured on the
Gaussian bed is representative. On a strongly non-Gaussian profile the truncation
bias could differ, and the bed would have to be built differently — either from a
shape-matched fit or, more cleanly, from a model run at high walker count (the
re-implemented hopping model in `nc_solid_sim.py` does this without any shape
assumption).

## Limits

The threshold quoted here is specific to an assumed noise law and to these
small-diffusion-length maps. Two measurable quantities would convert it from a
design rule into a statement about a real measurement:

- The per-frame CNR of the actual data, read from the raw frames, which places
  the measurement on the CNR₀ axis of Figure 1.
- The detector's noise law — its read floor and gain — fitted from the raw
  frames via the photon-transfer relation (variance versus mean). This places
  the measurement between the curves of Figure 2 and determines whether the
  bias regime applies at all.

A practical consequence follows for applying the correction to real data. The
width-debias noise-bias correction is calibrated on white noise and keyed
on the spectral CNR. On shot-dominated data both assumptions are off — the
bias-governing CNR is the peak-local member, and a white calibration
over-corrects photon-limited data — so the correct order of operations is to
establish the noise law first and apply the matched correction, rather than
applying the white-calibrated correction blindly.

## Methods

Maps are loaded read-only and processed by
[`nc_data.py`](nc_data.py).
Each frame is a histogram of about 50,000 discrete random walkers (the pixel
values are integer multiples of 1/49984). We fit a Gaussian (centre, width) to
each noiseless projection and rebuild it with the same integral, then take that
Gaussian as the bed's truth. The reconstruction conserves the population and
makes the peak fall as 1/σ with spreading (the raw histogram `max` instead rises,
biasing the per-frame CNR high). Its second moment is the ground truth σ²(t).
Each per-frame width is then fit with the bounded Gaussian estimator
([`width_debias.fit`](../../src/width_debias/fit.py)); the MSD uses the fitted
t = 0 width, MSD = σ̂²(t) − σ̂²(0). The exponent and diffusion coefficient come
from a nonlinear least-squares fit of MSD = 2D·t^α with α ∈ [0.2, 2]. The bias
is the exponent of the mean MSD over 1000 noise realisations; the single-shot
standard deviation is taken across per-realisation exponents. Noise laws are the
pluggable models in [`width_debias.noise`](../../src/width_debias/noise.py):
white noise with standard deviation peak₀/CNR₀, and shot noise with Poisson
counts and an optional read floor, whose per-frame CNR is
peak / √(peak + read²). CNR₀ is the frame-0 peak CNR and means the same starting
contrast across noise laws.

## Figure caption (`output/nc_solid_poster_figure.png`)

The figure shows the noise bias and its correction over the realistic measurement
regime: no MSD experiment is run below CNR₀ ≈ 30, the apparent exponent carries a
residual upward bias well into the contrast people actually use, and the
noise-bias correction recovers the nominal exponent. An inset extends the sweep
to low contrast, where the naive estimate diverges to the ballistic ceiling. One
representative sample is shown (C16, the smallest crystals); the others behave the
same. Produced by
[`nc_solid_poster_figure.py`](nc_solid_poster_figure.py).

> **A noise bias in the measured transport exponent, and its correction.** Red
> points are the apparent mean-squared-displacement exponent α from a standard
> frame-by-frame width fit, plotted against the initial contrast-to-noise CNR₀, for
> exciton transport in a disordered perovskite nanocrystal solid (the C16 /
> hexadecylamine-ligand sample, the smallest crystals). The test bed is built from
> a kinetic Monte Carlo of about 50,000 exciton walkers hopping on a polydisperse
> nanocrystal array (Solari et al.): each frame is projected to a 1-D profile over a
> 1500 nm field (20 frames spanning 20 ns, small-diffusion-length regime
> L_D / FWHM ≈ 0.14), and a Gaussian is fit to each frame and taken as that frame's
> ground truth, giving a realistic-width clean bed whose nominal exponent is
> subdiffusive (α ≈ 0.88, yellow line). This is not a claim that the real profile is
> Gaussian; it sets the bed's truth and removes the maps' walker-sampling noise so
> the only noise is the measurement noise studied here. White measurement noise
> scaled to each CNR₀ is then added, the bounded width is refit, and α is read from
> the MSD averaged over 64,000 independent noisy realisations (each marker is an
> independent Monte Carlo estimate, not a fitted curve; error bars are the bootstrap
> standard error of the exponent). The naive estimate (red) reads about 0.09 too
> high at CNR₀ ≈ 30 and clears only near CNR₀ ≈ 200. The noise-bias correction
> (blue) subtracts from each frame, before the fit, the calibrated per-frame
> variance bias — the folded-normal estimation-noise variance looked up by the
> frame's CNR from a D-free white-noise calibration — and recovers the nominal
> exponent across the measured regime, leaving only a small residual that grows
> toward the lowest contrast (about 0.01 at CNR₀ ≈ 30). The inset extends the sweep
> to low contrast (naive only): below CNR₀ ≈ 30 the apparent exponent climbs toward
> the ballistic ceiling (α = 2), saturating there below CNR₀ ≈ 8 (below the
> per-frame floor the correction over-subtracts erratically, so it is shown only in
> the main panel). Because the bias can only raise the exponent, it masks
> subdiffusion but never creates it — the effect is generic to the low-contrast,
> short-diffusion-length regime, not specific to one material.
