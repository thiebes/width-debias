# mn-trapping

## Purpose

Run the correction on Marc Meléndez's Mn-doped reaction-diffusion diffmaps
(Magdaleno et al. 2025), a non-Fickian, multi-population bed, across three
width estimators and the moment estimator. Each estimator's error against the
reference is split into the correction residual (against the noiseless fit)
and the shape gap (noiseless fit against the reference).

The reference is the second moment of the model's photoluminescence (PL)
profile, `PL_var` column 2. It is emission-weighted: free excitons, Mn-trapped,
and defect-trapped populations each count by their emitted intensity (emission
rate per carrier q_i ν_i, 0.12, 0.0022, and 0.10 /ns for the 0.51% Mn
defaults), not by their carrier number. It is therefore the reference for the
recorded profile's width, not the carrier second moment; reaching the carrier
moment needs the forward model.

## Contents

- `marc_data.py` -- loader for the `.dat` diffmaps (reads `data/`).
- `diffusion_map.py` -- the reaction-diffusion model (refactor of Marc's code)
  and PSF-convolution helpers.
- `marc_diffmap_validation.py` -- the correction with Gaussian, Voigt, and
  generalized-Gaussian fits on all four doping cases, plus the `MomentModel`
  comparator; separates correction residual from shape gap. The
  generalized-Gaussian arm is `GeneralizedGaussianModel()` at its default
  p = 2, a Gaussian profile fitted by scale (its calibration was also built at
  p = 2), so it differs from the Gaussian arm only in its fit bounds and
  calibration.
- `marc_mn_trapping_plan_a.py` -- per-frame variance recovery on the 0.51% Mn
  diffmap (Gaussian on the map as given; Voigt on the PSF-convolved map),
  scored against the noiseless fit.

## Data

Collaborator `.dat` diffmaps in the local (gitignored) `data/` dir; see
`DATASETS.md`. Needs the Marc-geometry joint (CNR, σ²) calibrations built by
`scripts/build_calibration.py`
(`calibration_marc_{gaussian,voigt,gengauss}_filtered.npz` in `output/`).

## Run

```bash
.venv/Scripts/python.exe -X utf8 experiments/mn-trapping/marc_diffmap_validation.py
.venv/Scripts/python.exe experiments/mn-trapping/marc_mn_trapping_plan_a.py
```

Outputs to repo-root `output/`; committed snapshots are here. The validation
table is written in UTF-8 (`-X utf8` keeps the console print from failing when
redirected).

## Key result + figure

`marc_diffmap_validation.txt`: white noise at CNR₀ 10 to 200, 15 seeds per
CNR₀ (seed int(CNR₀) × 1000 + s), joint calibration per estimator; residuals
are the mean over frames with per-frame CNR ≥ 1 of |seed-mean width −
reference|.

- **Noise.** Where every frame has per-frame CNR ≥ 1.8 (0.51% and 0.81% Mn at
  CNR₀ 100 and 200, 4.04% Mn from CNR₀ 20), the raw and corrected widths lie
  0.0001 to 0.005 μm² from the noiseless fit, and the correction changes that
  distance by at most 0.0005 μm². At these CNR₀ the noise bias is small
  against the 15-seed scatter, so the bed bounds the correction's error rather
  than resolving it. At 4.04% Mn and CNR₀ 10 every frame is still above CNR 1
  (1.44 to 14), but raw and corrected both lie 0.03 to 0.08 μm² from the
  noiseless fit, and the correction moves that by at most 0.005 μm². At CNR₀ 50
  (0.51% and 0.81% Mn, 189 to 223 of 250 frames above CNR 1) the correction
  lowers the distance by 0.014 to 0.016 μm² (Gaussian, from 0.105 and 0.115)
  and 0.003 to 0.005 μm² (Voigt, from 0.038 and 0.042).
- **Shape gap.** The noiseless single-shape fits lie 0.005 to 0.078 μm² from
  the PL moment (largest for the undoped case, where the profile is most
  mixed). Where every frame has CNR ≥ 1.8, the corrected widths' distance from
  the PL moment matches that gap. The correction does not close it.
- **Moment estimator.** It closes the shape gap to the PL moment (0.007 μm²
  undoped, below 0.0002 μm² doped) but is unusable under noise: its cross-seed
  SD reaches 1.5 × 10⁵ μm² and its NaN fraction 0.19 (4.04% Mn, CNR₀ 10).

The practical path to the carrier distribution on this bed is the forward fit
of the reaction-diffusion model. The undoped case never has every frame above
CNR 1 in this sweep (minimum per-frame CNR 0.31 at CNR₀ 200). Figure:
`marc_diffmap_validation.png` (caption beside it), plus
`marc_mn_trapping_plan_a.png` and the calibration diagnostics.

## Depends on

`width_debias` (`CalibrationGrid`, `fit`, `models`, `bias_lookup`, `weights`).
Local sibling loaders `marc_data.py`, `diffusion_map.py`. The calibrations come
from `scripts/build_calibration.py` (shared tooling).

## See also

`docs/VALIDATION_DESIGN.md` (the recorded-profile moment against the carrier
moment).
