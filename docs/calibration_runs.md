# Building calibrations

A calibration is the Monte Carlo lookup that the correction subtracts: the
per-frame fit bias `var_bias(CNR, sigma^2)` (and `rel_variance`) for a specific
measurement geometry and noise model. It is built once per dataset and reused by
the application.

## The tool

`scripts/build_calibration.py` is the parameterized builder -- pixel grid, sigma
range, CNR range, profile model, noise model, and Monte Carlo budget are all
flags. It writes `output/calibration_<tag>.npz`, loaded by the application with
`CalibrationGrid.load(path)`.

`scripts/run_calibration.sh` runs it on a multi-core box (e.g. WSL): it pins BLAS
to one thread per process (the fits are tiny nonlinear least-squares; BLAS
threading only oversubscribes) and forwards every argument through. It is the
general entry point -- pass any flags `build_calibration.py` accepts.

```bash
PYTHON=.venv-linux/bin/python bash scripts/run_calibration.sh \
  --tag my_run --n-pixels 64 --half-window 2.5 \
  --sigma-min 0.1 --sigma-max 1.5 --cnr-min 0.5 --cnr-max 80 \
  --n-runs 8000 --model gaussian
```

`WORKERS=N` overrides the worker count (default: all logical cores). The output
`.npz` is copied back to the analysis machine's `output/` directory.

## Dataset presets

A preset is a thin wrapper that calls `run_calibration.sh` with one dataset's
fixed parameters, so the run is reproducible by name:

- `run_nc_solid_calibration.sh` -- the NC-solid Gaussian calibration.
- `run_cspbbr_calibration.sh` -- CsPbBr3 single-crystal diffusion, Gaussian.

To add a dataset: copy a preset, set the geometry from your reduced data
(`--n-pixels`, `--half-window` to match the application's spatial grid;
`--sigma-min/max` to cover the true width range; `--cnr-min/max` to span the
measured per-frame CNR), and choose the noise model.

## Choosing the noise model

The calibration must match the detector's noise, because the noise bias is
driven by the part of the noise that survives in the low-signal profile wings.

- **`--noise gaussian`** (default): additive white noise, constant across the
  profile. Use it when a constant floor dominates -- read noise, dark counts,
  background, or the structured/scanning floor of a raster-scanned measurement.
  Parameterized by the per-frame CNR directly (peak / floor), which fft-cnr
  gives from the data.
- **`--noise poisson --gain G --read-noise R --background B`**: the
  photon-transfer model `var = gain*signal + read^2`, with signal-dependent shot
  noise (slope `gain`) plus a constant read floor. Use it for shot-limited
  photon-counting data. Read `gain` and `read` from fft-cnr's photon-transfer
  fit (`estimate_noise_model=True`) on a bright, high-dynamic-range frame; take
  `background` from the profile edges. A white calibration over-corrects
  shot-limited data and can drive D negative; the converse holds, so match the
  model to the data.

How to tell which: run fft-cnr's photon-transfer fit on a bright raw-count
frame. If it reports a clear `gain` with `signal_dependent=True` and a small
floor, the data is shot-limited -- use Poisson. If the floor dominates and
`signal_dependent` is False, the data is white-floor-dominated -- use gaussian.
The CsPbBr3 single-crystal data is the second case (scanned APD; the fit is
floor-dominated), so its preset uses gaussian.

## Key tunables

- `--max-var`: the fit's upper variance bound. The default is the full window
  (`span^2`), which lets low-CNR fits wander to the window scale and inflates the
  apparent bias when the window is far wider than the spot. Pass a physical cap
  above the observed widths but below the window scale. **It must be identical in
  the calibration and the application**, or the `var_bias` subtraction is
  mismatched.
- `--n-runs`: Monte Carlo runs per cell. The 2-D (CNR, sigma^2) lookup needs
  high statistics where the per-cell bias estimate is noisy -- ~8000/cell at the
  low-CNR end. Drop to ~2000 for a quick look.
- Grid sizes (`--n-cnr`, `--n-sigma`): ~100 each is typical; the total fit count
  is `n_cnr * n_sigma * n_runs`.

## Choosing the key

The joint lookup is indexed by the true variance, which a measurement does not
know. When sigma^2 changes little over the measurement, keying each frame on its
own fitted variance (the default) is adequate. When it grows several-fold, that
key under-corrects at low CNR; see "Choice of key" in
`BiasLookup.var_bias` and `experiments/halide-mixing-highn/key_sensitivity.py`.
