# Saha reaction-diffusion test bed

A Python port of Marc Meléndez's Saha reaction-diffusion model, used as a
realistic non-Fickian test bed for the folded-normal truncation correction. The
model is the spatial extension of the Cutrupi et al. (2026) rate model: a single
charge-pair population diffuses, and at every point the Saha equation sets the
free-charge fraction, so excitons and free charges diffuse at different rates and
the effective diffusivity is density-dependent.

The lifetime fit and the identifiability analysis built on this model (the
Antonella and Carlos free-exciton diffusion paper) are in the separate
`free-exciton-diffusion` repo. Only the correction validation is here.

## Findings

The correction does not improve the recovered exponent on this bed.
`poster_saha_nonfickian.png` (no error bars; its generating script is not in this
repo) shows the uncorrected alpha below the noise-free-fit reference (1.270) by
0.034 at the lowest CNR_0 (about 12) and folded-normal further below it, by
0.055; the same order holds at CNR_0 about 18 and 25, and from CNR_0 40 up both
are within 0.003 of the reference. Two properties limit what the bed
demonstrates:

- The noise bias is modest in the trusted CNR band. The clean correction
  demonstration is the single-population NC-solid bed (where emission is
  proportional to the carrier density, so the bias acts cleanly), not this
  multi-population bed.
- The distortion this bed exposes most clearly, the emission-to-carrier
  nonlinearity, is outside the scope of the correction (it requires a
  forward-model inversion, which the noise correction does not perform) and is
  the subject of the Cutrupi model itself.

The bed supports three technical results:

- The validation framework in [`docs/VALIDATION_DESIGN.md`](../../docs/VALIDATION_DESIGN.md):
  truth is the carrier distribution's second moment; three distortions sit
  between the recorded profile and the carrier distribution (noise, handled by
  the noise-bias correction; the instrument PSF; the emission-to-carrier
  mapping, handled by the forward model); and two workflows (simulation truth is
  input-derived, experimental truth is recovered by forward fit or analytics).
- The recombination and emission terms are bimolecular in the free population,
  so they carry x², not x (checked against Marc's authoritative code).
- The 2-D lookup over-subtracts below a per-frame CNR between 1.13 and 1.37 on
  this bed (`diagnose_overcorrection.txt`: A = 2000, CNR_0 6, 12, 25, 50, 4000
  fits per frame; walking down in CNR, the first gap resolved at 3 SE is
  +0.012 um^2 at CNR 1.13, and the 31 frames from CNR 1.37 up have gaps of
  -0.004 to +0.001 um^2, |z| <= 2.4). This supports the per-frame floor of 1.3
  that the real-data scripts apply. Below CNR 1 the gaps are large in both
  directions, but those frames are dropped in every application.

## Files

- `reaction_diffusion_saha.py` — the model port: the forward simulator. Returns
  the emission, population (carrier), and free-fraction maps over time. The other
  scripts import it. (A copy is also the model foundation of the
  `free-exciton-diffusion` repo.)
- `nonfickian_validation.py` — the validation harness: builds regimes and their
  carrier / emission / fit reference α, runs the estimator paths (Gaussian fit
  plus the non-truncated moment estimator), applies the noise-bias correction,
  flags the CNR floor, and reports how many non-positive MSD points the
  log-log exponent fit leaves out (`nonfickian_validation.txt`, CNR_0 60: 0 to
  1 per regime on the moment path, none on the Gaussian + folded path).
- `diagnose_overcorrection.py` — locates where the 2-D lookup starts to
  over-subtract (compares the measured per-frame fit bias against what the
  calibration subtracts, frame by frame, with a paired SE);
  `diagnose_overcorrection.txt` is its committed output.
- `forward_fit_validation.py` — stage-2 forward-fit validation: recovers the
  carrier transport by fitting the forward model, applying the folded-normal
  correction to the low-CNR frames. Imports the `width_debias` library.
- `distortion_figures.py` — illustration and diagnostic figures: the two gaps
  (carrier vs emission vs fit α) and the variance-trajectory mechanism.
- `nonfickian_gaps.png`, `saha_second_moment_bias.png`, `poster_saha_nonfickian.png`
  — committed figure snapshots for this bed.
- Regenerated figures and calibrations are written to the repo-root `output/`
  (gitignored); the committed snapshots in this folder are the current record.

## Pointers

- Validation framework: [`docs/VALIDATION_DESIGN.md`](../../docs/VALIDATION_DESIGN.md)
- The noise-bias correction: [`docs/DERIVATION.md`](../../docs/DERIVATION.md)
- The lifetime and De/Df diffusion-paper work: the `free-exciton-diffusion` repo.
