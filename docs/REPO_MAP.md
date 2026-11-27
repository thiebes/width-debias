# Repository map

Every experiment set is one folder under `experiments/`, with its scripts,
result tables, figures, captions, and a `README.md`. This page lists each set
with its main result; the set's README gives the details and how to run it.

## Layout

```text
src/width_debias/   the library
experiments/         one folder per experiment set (below)
illustrations/       one folder per schematic figure (below)
scripts/             shared calibration tooling and figure rendering
docs/                derivation, validation design, calibration how-to, this map
tests/               package tests
output/              regenerated run outputs and cached calibrations (not tracked)
data/                default data root for collaborator data (not tracked)
```

## Experiment sets

### Synthetic method beds

| Set | Scripts | Main result | Figure(s) |
| ------- | ------- | ------- | ------- |
| [poc-bias-correction](../experiments/poc-bias-correction/) | `poc_demo.py` | D-free calibration (400 runs): no `subtract` cell beyond 2 combined SE on any of the four shapes; inverse-variance leaves 1, 2, 4, and 0 of 7 cells beyond 2 SE (Gaussian, Voigt, p = 1.5, p = 4) | `poc_demo*.png` |
| [boundary-diagnostic](../experiments/boundary-diagnostic/) | `diagnose_boundary.py` | the width fit reflects rather than clips at the lower bound, so the additive bias model of `docs/DERIVATION.md` Section 1 applies | `boundary_diagnostic*.png` |
| [dfree-calibration](../experiments/dfree-calibration/) | `validate_d_zero.py`, `dense_calibration_and_test.py`, `calibration_se_check.py`, `weights_check.py`, `joint_weights_check.py`, `slope_fit_check.py`, `render_bias_panels.py` | corrected D: 4 of 140 cells beyond 2 combined SE (standard window) and 5 of 96 (short window), pooled bias within 0.4 combined SE on every shape (0.8 in the short window); naive all frames 46 of 140 and 41 of 96, its bias exceeding the error of the mean after 15 to 91 (standard) or 47 to 652 (short) averaged measurements; with a peak that also falls as the summed profile spreads (5000-run calibration, against 300 and 2000 runs for the decay-alone counts, so the corrected counts are not like for like), corrected D 6 of 140 and 7 of 96, pooled bias within 0.8 combined SE on every shape, naive all frames 45 of 140 and 43 of 96; calibration SE at most about the evaluation SE at equal runs and shared by every measurement, so calibration runs should be well above the number of measurements averaged; propagated calibration SE observed/predicted 0.89 over 16 draws; joint-lookup slope weights averaged over width give corrected D/D_true 0.88 to 1.09, against 0.39 to 0.96 read at each frame's own fitted width; dense sweep pooled bias within 0.6 combined SE on four shapes, MAS fails high or low by shape below CNR0 about 6.5 | `.txt` tables, falling-peak `.npz` cell statistics and lookups, `poster_panel3_bias.png`, `poster_panel5_corrected_bias.png` |
| [correction-floor](../experiments/correction-floor/) | `per_profile_floor.py`, `analytical_floor.py`, `per_frame_in_diffusion_experiment.py`, `render_floor_panel.py` | per-profile floor at tolerance 0.3 x 2 D t_max (D = 0.02 sigma_0^2/tau): CNR 1.16 (Gaussian, Voigt), 1.51 (p = 1.5), 0.89 (p = 4), residual and calibration SE both counted; Fisher A matches Monte Carlo to 0.986-1.006 and the A/CNR^2 model departs at CNR about 1.18 (0.85 for Voigt) | `.txt` tables, `poster_panel5_floor.png` |
| [noise-law](../experiments/noise-law/) | `shot_noise_validation.py`, `read_floor_sweep.py` | the bias is set by the constant (white / read-floor) part of the noise; a calibration must match the noise law | `read_floor_sweep.png`, `fftcnr_white_vs_shot.png` |
| [coverage-interval](../experiments/coverage-interval/) | `coverage_check.py`, `narrowness_attribution.py` | with the per-frame slope weights the covariance interval covers 0.944 pooled over 13 CNR0 x 8000 runs (nominal 0.95); where every frame has per-frame CNR >= 1 its SE is 0.967 of the realised spread because the lookup's per-frame variances are calibrated at sigma_0^2 while the width grows; a parametric bootstrap with the measured per-frame variances covers 0.951 | `coverage_check.png` |
| [estimator-comparison](../experiments/estimator-comparison/) | `moment_estimator_failure.py`, `unbounded_fit_comparison.py` | removing the lower bound from the width fit does not substitute for the correction | `method_comparison.png`, `unbounded_fit_comparison.png` |
| [subdiffusive-alpha](../experiments/subdiffusive-alpha/) | `subdiffusive_alpha.py` | signed corrected exponent error over CNR0 21-90 within 1.4 SE for alpha 0.6-1.0 and +0.049 ± 0.016 for 1.2 (SE without the calibration error); naive +0.024 to +0.081 (z 2.2-4.9) | `subdiffusive_alpha.png` |
| [multiexp-decay](../experiments/multiexp-decay/) | `multiexp_decay.py` | the correction keys on per-frame CNR, not on the decay law (corrected D 0.93-1.06 of D_ref for three decay laws) | `multiexp_decay.png` |
| [crlb-precision](../experiments/crlb-precision/) | `crlb.py`, `efficiency_check.py`, `render_crlb_panel.py` (+ `precision_limit.md`) | sqrt(CRLB)/D is several times D at low CNR; with the library's per-frame width-variance weighting the corrected D reaches 0.95-1.10 x sqrt(CRLB) from CNR0 21 (independent seeds per shape) | `poster_panel4_crlb.png` |
| [closed-form-scaling](../experiments/closed-form-scaling/) | `linearity_test.py` | the mean bias follows the exact Fisher value 1.13/s^2 for s >= 10 (chi^2/dof 0.88-0.97 at four sigma0) | `linearity_test.png` |
| [fisher-calibration](../experiments/fisher-calibration/) | `compare_calibrations.py`, `fisher_check.py`, `diagnose_criterion3.py` | pre-registered switch test: `calibrate_fisher` matches the raw path's accuracy with 0.20-0.32 of its calibration SE but fails the criterion on Voigt and p = 4 (predicted SE too small: per-draw GCV smoothing choice) and p = 1.5 (cells beyond 2 SE), so the raw path stays recommended; Fisher A matches Monte Carlo to 0.998-1.003 on four shapes; width scaling not established | `.txt` tables |
| [fft-cnr-recovery](../experiments/fft-cnr-recovery/) | `build_fft_cnr_recovery_figure.py` (+ committed summary `.npz`) | fft-cnr recovers the true CNR on the 1:1 line from CNR 2 to 100, N = 128-1024 | `fft_cnr_recovery.png` |

### Transport-simulation beds (Marc Meléndez; data per row, see each `DATASETS.md`)

| Set | Scripts | Main result | Data |
| ------- | ------- | ------- | ------- |
| [nc-solid](../experiments/nc-solid/) | `nc_solid_*`, `run_nc_solid_calibration.sh` (+ loaders `nc_data.py`, `nc_solid_sim.py`) | de-noised maps: the naive exponent climbs toward the ballistic limit at low CNR and the corrected exponent recovers the reference 0.88; raw maps: the correction does not rescue CNR0 3 to 10 (both exponents at the fit bound of 2) | data root; committed `.npz` calibration |
| [mn-trapping](../experiments/mn-trapping/) | `marc_diffmap_validation.py`, `marc_mn_trapping_plan_a.py` (+ loaders) | reference is the emission-weighted PL second moment, not the carrier moment; where every frame has CNR >= 1.8 the raw and corrected widths lie within 0.0054 μm² of the noiseless fit, so the bed bounds the correction rather than resolving it; the 0.005 to 0.078 μm² shape gap needs a forward fit, and the moment estimator closes it but is unusable under noise | local `data/` (not tracked) |
| [saha-reaction-diffusion](../experiments/saha-reaction-diffusion/) | `nonfickian_validation.py`, `forward_fit_validation.py`, `reaction_diffusion_saha.py`, `diagnose_overcorrection.py`, `distortion_figures.py` | the correction does not improve the exponent on this bed (both within 0.003 of the reference from CNR0 40 up); the 2-D lookup over-subtracts below per-frame CNR 1.13 to 1.37, which supports the real-data floor of 1.3; a forward fit recovers the carrier transport | none (in-repo port of the model) |
| [halide-mixing-brI-legacy](../experiments/halide-mixing-brI-legacy/) | `br_halide_validation.py` (+ loader); `rerun/`: `run_instances.sh`, `rerun_unique.sh`, `rerun_all_unique.sh`, `average_diffmaps.py`, `collapse_diffmaps.py` (+ `collapse_check.txt`), the high-N re-run chain | mixing suppresses transport; the correction has little to do over the short window | data root |
| [halide-mixing-highn](../experiments/halide-mixing-highn/) | `br_halide_highn_*`, `key_sensitivity.py`, `render_perframe_figure.py`, `axis_check.py` (+ loader) | the width-fit upper bound governs the apparent bias; with a joint (CNR, sigma^2) calibration and a measurement-level key, D is recovered to 0.96-0.99 from CNR0 15 and is not determined below it; a Fisher width table calibrated at the width of the frames the correction acts on (75 nm, post hoc) matches the joint grid to 0.003 in D with 1/25 of its fits, while one calibrated at sigma_0 over-subtracts (D 0.35-0.59 of the noise-free fit's) because the window sets the sigma_0 residual; the 1-D D-free lookup at sigma_0 over-subtracts for the same reason when read down to the 1.3 floor (residual -4066 to -3548 nm^2 between per-frame CNR 1.3 and 3 at CNR0 15 and 20, exponent at the fit's bound on 4 of 6 cells) and, kept to frames with per-frame CNR >= 3, recovers D to 0.96-1.02 at those CNR0; the x and y projections' binned-moment ratio is within 0.0143 of 1 (walker sampling about 0.003, a lower bound), and the noise-free fit's D/D_ref differs between them by 0.017-0.034 (part through the binned moments, about half at x = 0.75); these files cannot tell whether that comes from the transport or from the finite set of disorder realisations | data root (collapsed 1-D) |

### Measured data

| Set | Scripts | Main result | Data |
| ------- | ------- | ------- | ------- |
| [wse2-strobescat](../experiments/wse2-strobescat/) | `render_realdata_panel.py`, `realdata_*`, `render_recovery_demo.py` | MoSSe alloy `26_32/250716`: corrected/naive D 0.95 over the full range and 0.97 for t <= 2 ns | data root (Enrique Arévalo Rodríguez) |
| [cspbbr-strobescat](../experiments/cspbbr-strobescat/) | `reduce.py`, `apply_correction.py`, `run_all.py`, `run_cspbbr_calibration.sh` | end-to-end correction of measured CsPbBr3 diffusion scans | data root (Upasana) |
| [zhang-qd-control](../experiments/zhang-qd-control/) | `zhang_qd_panel.py` (+ committed figure CSVs) | the published quantum-dot Dfast is robust to the correction (high CNR) | committed CSVs |

## Illustrations

Schematics without data input live under `illustrations/`, one folder per
figure with its build script, figure, and caption. See
[`illustrations/README.md`](../illustrations/README.md).

| Figure | Folder |
| ------- | ------- |
| Measurement schematic (spreading, dimming population; CNR falls per frame) | [measurement-schematic](../illustrations/measurement-schematic/) |
| Wall-and-fold schematic (wall at zero, reflected tail, tilted MSD slope) | [wall-fold-schematic](../illustrations/wall-fold-schematic/) |

## Shared tooling

- `scripts/build_calibration.py`, `scripts/plot_calibration.py`: builder and
  diagnostic plots for a geometry-specific joint calibration.
- `scripts/run_calibration.sh`: generic calibration runner, called by the
  per-set runners. See [`calibration_runs.md`](calibration_runs.md).
- `scripts/poster_figures.py`, `scripts/poster_fftcnr_dense.py`,
  `scripts/poster_style.py`: render the summary figures and hold their shared
  palette and axis style.
- `scripts/collaborator_data.py`: the data root and its folders, where the
  loaders read collaborator data.

## Datasets (not tracked)

Each collaborator and measured-data set carries a `DATASETS.md` naming the files
it reads. The scripts read collaborator data from the data root: the folder that
`WIDTH_DEBIAS_DATA_ROOT` names, or else `data/` at the repository root (ignored
by git). `scripts/collaborator_data.py` holds the folder layout.

| Set | Source |
| ------- | ------- |
| nc-solid | data root, `simulated_map/` (Marc Meléndez's maps) |
| mn-trapping | Marc Meléndez's Mn diffmaps, in the set's local `data/` copy |
| saha-reaction-diffusion | none: an in-repo port of Marc Meléndez's Saha model |
| halide-mixing-brI-legacy | data root, `halide_mixing/` (Marc Meléndez's maps) |
| halide-mixing-highn | data root, `halide_mixing_highn/` (collapsed 1-D; made by the `halide-mixing-brI-legacy/rerun/` chain) |
| wse2-strobescat | data root, `stroboscat_scans/` (Enrique Arévalo Rodríguez's scans; outputs in the set's untracked `diagnostics/`) |
| cspbbr-strobescat | data root, `cspbbr_sample_1/` (Upasana's scans) |
| zhang-qd-control | published-paper data (committed CSVs) |
| fft-cnr-recovery | external simulation study (not distributed; location not recorded; committed summary) |

Three scripts read an external simulation study's result files from the
directory that `WIDTH_DEBIAS_EXTERNAL_RESULTS` names, which has no default:
`experiments/fft-cnr-recovery/build_fft_cnr_recovery_figure.py` (to rebuild its
committed summary, and to check the summary's source hash when the file is
available), `experiments/dfree-calibration/calibration_se_check.py
analyse` (an optional section of its table) and `scripts/poster_figures.py` (a
fallback for one panel). Each says when the variable is unset and names the file
when it is missing.

Collaborator data is referenced by pointer only; it is not copied into the
repository.
