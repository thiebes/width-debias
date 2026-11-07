# nc-solid

## Purpose

The realistic-transport result: on a Gaussian-width bed built from Marc
Meléndez's Solari et al. disordered nanocrystal-solid exciton-hopping maps, a
standard frame-by-frame Gaussian-width MSD fit inflates the apparent
subdiffusive exponent alpha toward the ballistic ceiling as the per-frame CNR
falls, and the noise-bias correction recovers the nominal exponent.

Quantitative thresholds on this bed are soft (the maps' finite-walker sampling
noise), so the result is stated qualitatively; the precise thresholds come from
the synthetic beds.

## Contents

- `nc_data.py` -- loader for the NC-solid maps (reads the pshare; `denoise=True`
  reconstructs the smooth Gaussian per frame before measurement noise).
- `nc_solid_sim.py` -- the Solari hopping-KMC simulator.
- `nc_solid_sim_run.py` -- runs the KMC simulator (`nc_solid_sim`).
- `nc_solid_alpha_vs_cnr.py` -- naive vs corrected alpha against CNR0 (the core
  compute; reused by the figure script).
- `nc_solid_poster_figure.py` -- the figure (naive vs corrected alpha,
  ballistic-divergence inset). Reuses `nc_solid_alpha_vs_cnr` and `nc_data`.
- `nc_solid_validation.py` -- scores D and alpha against the carrier moment.
- `run_nc_solid_calibration.sh` -- the high-statistics calibration runner.
- `calibration_nc_solid_gaussian.npz` -- the committed high-statistics
  (8000 runs/cell, D=0 white-noise) var_bias calibration the figure uses (output/
  is gitignored, so the version-controlled copy lives here).
- `nc_solid_poster_figure.png` / `.txt` -- the committed figure and per-CNR0 table.
- `nc_solid_poster_figure_caption.md` -- the figure caption.

## Data

NC-solid maps on the pshare (`P:/Marc/shared_data/disordered_perovskite_nanocrystals_solids`);
see `DATASETS.md`. No local copy. The committed calibration `.npz` is the only
large derived artifact kept in-repo.

## Run

```bash
# regenerate the figure (needs the calibration in output/ first):
cp experiments/nc-solid/calibration_nc_solid_gaussian.npz output/
WIDTH_DEBIAS_WORKERS=0 .venv/Scripts/python.exe experiments/nc-solid/nc_solid_poster_figure.py
```

Rebuild the calibration with `run_nc_solid_calibration.sh` on a many-core Linux
or WSL machine (usage in the script header). Other scripts run the same way from
the repo root.

## Key result + figure

Naive alpha reads ~0.09 high at CNR0 = 30 and climbs to the ballistic ceiling
(alpha = 2) below CNR0 ~ 8; the noise-bias correction recovers the nominal
alpha ~ 0.88 across the measured regime (residual ~0.01 at CNR0 = 30). The bias
only raises alpha, so it can mask subdiffusion but cannot manufacture it. Figure:
`nc_solid_poster_figure.png`; table `nc_solid_poster_figure.txt`. That result is
on the de-noised maps (`denoise=True`) at CNR0 30 to 82; the table predates the
change that keeps a failed fit's fallback value in `nc_solid_alpha_vs_cnr.py`
(the raw-map run below has no failed fits at any CNR0).

On the raw maps, with their finite-walker sampling noise
(`nc_solid_validation.py`, `nc_solid_validation.txt`; 400 noise realisations
per CNR0, bootstrap SE on the mean-trajectory exponent; the experimental 2-D
calibration), the correction does not rescue the low-CNR0 cells. At CNR0 3 to
10, and on the large map at 20, the naive and the corrected alpha both sit at
the fit's bound of 2. The whole MSD is only about 0.004 um^2 (noiseless fit
0.040 to 0.044 um^2 over the measurement), so whatever per-frame bias the 2-D
lookup leaves in the late, low-CNR frames is enough to make the mean trajectory
convex; most likely that residual is what pins the exponent. From CNR0 20 up
(50 on the large map) the correction moves alpha by at most 0.045, and both
arms lie -0.4 to +3.2 SE from the carrier truth, mostly above it (for example
0.928 +/- 0.022 corrected against 0.866 at CNR0 100 on the small map, +2.8 SE).
No fit fails at any CNR0 (0 of 8000 per cell), so how failed fits are handled
does not matter on this bed.

## Depends on

`width_debias` (`GaussianModel`, `noise.WhiteNoise`, `truncation`, `validate`,
`weights`). Local sibling modules `nc_data.py`, `nc_solid_sim.py`,
`nc_solid_alpha_vs_cnr.py`. Optional `curved_text` (the figure's curved label;
`pip install -e ../curved-text`).

## See also

`nc_solid_bias_note.md` (technical note), `docs/VALIDATION_DESIGN.md`,
`experiments/subdiffusive-alpha/` (exponent recovery on a synthetic bed).
