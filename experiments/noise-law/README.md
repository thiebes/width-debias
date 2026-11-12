# noise-law

## Purpose

Show that the noise bias is governed by the contrast against the
**constant** (white / read-floor) part of the noise, not the signal-following
shot part: white noise carries a large bias, pure shot noise a much smaller one
of opposite sign, and a read floor of a few counts restores the large bias.

## Contents

- `shot_noise_validation.py` -- noise-bias correction across white / shot /
  Poisson laws at matched peak CNR; correctness (matched calibration removes
  bias) and robustness (does the white calibration transfer?).
- `read_floor_sweep.py` -- raises a constant read floor on shot noise and tracks
  the bias returning.

## Data

None. Fully synthetic (`width_debias.noise` models).

## Run

```bash
.venv/Scripts/python.exe experiments/noise-law/shot_noise_validation.py
.venv/Scripts/python.exe experiments/noise-law/read_floor_sweep.py
```

Set `WIDTH_DEBIAS_WORKERS=0`. The tables print to stdout (committed here as
`shot_noise_validation.txt`, `read_floor_sweep.txt`, run with `SHOT_NCAL=1000
SHOT_NEVAL=1000 SHOT_NGRID=8`, about 1 h and 2.5 h with 4 workers); the
figure goes to repo-root `output/`.

## Key result + figure

Mean `|mean(D_hat)/D_ref - 1|` over CNR_0 3 to 60 (1000 calibration and 1000
evaluation runs per cell), OLS / `subtract` with matched calibration / `subtract` with
the white calibration: white 18.6 / 0.098 / 0.098, shot 0.22 / 0.025 / 0.28,
Poisson 0.19 / 0.024 / 0.27 (`shot_noise_validation.txt`). Under shot and
Poisson noise OLS reads low, not high, at the lowest CNR_0 (−0.01 and 0.14 at
CNR_0 3). Raising a constant read floor on Poisson data brings the bias back
(`read_floor_sweep.txt`): OLS 0.19, 2.06, 6.84, 14.4 at read 0, 1, 2, 4 counts
and 20.7 to 26.0 at 8 to 32, against 18.6 on white noise. The matched `subtract`
correction stays at 0.014 to 0.025 up to read 2, rises to 0.12 to 0.13 at read
4 and 8, and to 0.97 to 1.27 at 16 and 32. The nominal CNR_0 overstates the
per-frame CNR under a read floor (`ShotNoise.cnr_frame` is
peak / sqrt(peak + read²)): at 16 and 32 counts the CNR_0 3 cell starts below
the correction floor (0.55 and 0.28 at frame 0), which is consistent with the
rise; the table does not give per-cell values to confirm it. The
white calibration must not be applied to shot data: it over-subtracts, D goes
negative at CNR_0 3, and its score stays at 0.21 to 0.35 at every read floor,
against 0.098 on white data. The metric is unsigned and includes Monte Carlo
noise, so an unbiased estimator does not score zero; neither table gives
standard errors. Figures: `read_floor_sweep.png` (caption
`read_floor_sweep_caption.md`) and `fftcnr_white_vs_shot.png` (from
`scripts/poster_figures.py`).

## Depends on

`width_debias` (`noise`). No sibling-folder helpers.

## See also

`docs/DERIVATION.md` (the noise-bias correction).
