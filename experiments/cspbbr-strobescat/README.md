# cspbbr-strobescat -- CsPbBr3 single-crystal diffusion, real-data correction example

End-to-end application of the noise-bias correction to a real measurement:
CsPbBr3 single-crystal transient-PL-microscopy diffusion (Upasana's data, from
the Antonella/Carlos diffusion paper). The late-time MSD inflates as the
per-frame CNR collapses; the correction removes that bias before the D fit. The
real-data figure of this repository is in `experiments/wse2-strobescat/`; this
folder is a second real-data application. Data on the pshare; see `DATASETS.md`.

The calibration uses the gaussian (white, constant-floor) noise model. That
choice is asserted, not measured: `reduce.py` runs no noise-law fit, and a
raster-scanned APD (photon-counting) measurement could be shot-limited instead.

## Limitations (read before quoting a result)

- **No committed result.** No D from this folder has been reported or checked.
- **Noise law not measured.** See above; a shot-noise-limited measurement needs
  a matched calibration (`experiments/noise-law`).
- **No background subtraction before the width fit.** `reduce.py` fits the
  merged profiles directly, and the Gaussian fit has no offset term, so a dark
  count or pedestal widens the fitted late frames in the same direction as the
  noise bias.
- **Experimental calibration path.** `apply_correction.py` uses the joint
  `(CNR, sigma^2)` calibration, which `src/width_debias/calibration.py` marks
  as experimental; it applies no per-frame CNR floor and propagates no
  calibration SE.

## All fluences at once (recommended)

`run_all.py` does the whole thing across every fluence subfolder: reduces each,
derives its geometry from the fresh reduction, builds one calibration per unique
pixel grid (shared across fluences on that grid, since the gaussian lookup is
fluence-independent), applies the correction, and writes a D-vs-fluence summary
(`cspbbr_D_vs_fluence.csv` and `.png`). The heavy calibration honours `--workers`.

```bash
python experiments/cspbbr-strobescat/run_all.py \
  --sample-root "P:/Upasana/CsPbBr_SingleCrystal/Sample_1" \
  --out-dir output/cspbbr --workers 0 --n-runs 8000
```

`--stages` resumes a partial run (e.g. `--stages apply,summary` after the
calibrations are built). The run machine must be able to read the raw data at
`--sample-root`.

## Single fluence (manual)

1. **Reduce** raw acquisition CSVs to per-frame profiles, fitted variance, and
   CNR (mirrors the lab's reduction; per-frame fit and CNR are this project's):

   ```bash
   python experiments/cspbbr-strobescat/reduce.py \
     --data "P:/Upasana/CsPbBr_SingleCrystal/Sample_1/13.47 uJ_Fluence_19_May_2026" \
     --n-files 3 --max-var 6.25 --out output/cspbbr_13p47.npz
   ```

2. **Calibrate** the `(CNR, sigma^2)` bias lookup for this geometry (heavy; run
   on a multi-core Linux or WSL machine; geometry and tunables are documented in
   the script header):

   ```bash
   PYTHON=.venv-linux/bin/python bash experiments/cspbbr-strobescat/run_cspbbr_calibration.sh
   ```

3. **Apply** the correction and compare naive vs corrected D:

   ```bash
   python experiments/cspbbr-strobescat/apply_correction.py \
     --reduced output/cspbbr_13p47.npz \
     --calibration output/calibration_cspbbr_gaussian.npz
   ```

## Consistency requirements

The bias lookup is conditional on the measurement geometry, so the reduction and
the calibration must agree on:

- the **pixel grid** (`--n-pixels`, `--half-window` in the calibration must match
  the reduced profiles' grid; `apply_correction.py` warns if they disagree), and
- the **fit variance cap** (`reduce.py --max-var` must equal the calibration's
  `--max-var`).

Raw data is not committed (it is the collaborators'); point `--data` at the
acquisition folder on the share.
