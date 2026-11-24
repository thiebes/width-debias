# `scripts/` -- shared tooling and summary figures

Each experiment's own scripts live in its folder under `experiments/` (see
[`docs/REPO_MAP.md`](../docs/REPO_MAP.md)). `scripts/` holds what several sets
share.

All scripts run from the repository root with `.venv/Scripts/python.exe` on
Windows. Outputs go to `output/` with matching base names.

## Collaborator data

- **`data_root.py`** -- where the experiment sets read collaborator data: the
  folder `WIDTH_DEBIAS_DATA_ROOT` names, or else `data/` at the repository root
  (ignored by git), and the fixed folder layout inside it (`stroboscat_scans/`,
  `simulated_map/`, `halide_mixing/`, `halide_mixing_highn/`,
  `cspbbr_sample_1/`). The loaders of `nc-solid`, the two halide-mixing sets,
  `wse2-strobescat` and `cspbbr-strobescat` import it; each set's `DATASETS.md`
  lists the files its folder must hold.

## Calibration tooling

- **`build_calibration.py --tag <name> ...`** -- builds a geometry-specific joint
  `var_bias(CNR, sigma^2)` calibration for an arbitrary experiment (measured or
  simulated): pass the pixel grid, sigma range, CNR range, profile model, and
  Monte Carlo budget. Saves `output/calibration_<tag>.npz`, loadable with
  `CalibrationGrid.load()`. The transport-simulation beds
  (`experiments/mn-trapping/`) use calibrations built this way. Wall time
  ~2.5-4 h for a 100x100-cell, 1000-runs/cell build.
- **`plot_calibration.py <calibration.npz>`** -- diagnostic heatmaps of a saved
  joint calibration (the `var_bias` and `rel_variance` surfaces, with a data
  trajectory overlaid when applicable).
- **`run_calibration.sh`** -- the generic calibration runner the per-set runners
  call (for example `experiments/cspbbr-strobescat/run_cspbbr_calibration.sh`).
  See [`docs/calibration_runs.md`](../docs/calibration_runs.md).

## Summary figures

- **`poster_style.py`** -- shared figure primitives (the colour palette, axis
  style, and the dense-calibration / per-profile-floor report parsers) that the
  per-experiment render scripts import. No figures here.
- **`poster_figures.py`** -- renders the summary figures by importing each
  figure's builder from its home (schematics in `illustrations/`, data figures in
  their experiments). It defines only the two fft-cnr recovery figures, whose
  sweeps come from `poster_fftcnr_dense.py`; Panel 4 falls back to an external
  simulation study's sweep (`WIDTH_DEBIAS_EXTERNAL_RESULTS`).
- **`poster_fftcnr_dense.py`** -- the dense fft-cnr recovery-ratio figure.
