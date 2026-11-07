# noise-law

## Purpose

Show that the noise bias is governed by the contrast against the
**constant** (white / read-floor) part of the noise, not the signal-following
shot part: white noise carries the bias everywhere, pure shot noise nowhere, and
a read floor of a few counts restores it.

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
SHOT_NEVAL=1000 SHOT_NGRID=8`); the figure goes to repo-root `output/`.

## Key result + figure

Mean `|mean(D_hat)/D_ref - 1|` over CNR_0 3 to 60 (1000 calibration and 1000
evaluation runs per cell), OLS / folded with matched calibration / folded with
the white calibration: white 18.6 / 0.22 / 0.22, shot 0.22 / 0.046 / 0.35,
Poisson 0.19 / 0.036 / 0.34 (`shot_noise_validation.txt`). Raising a constant
read floor on Poisson data brings the bias back (`read_floor_sweep.txt`): OLS
0.19, 2.06, 6.84, 14.4 at read 0, 1, 2, 4 counts and 20.7 to 26.0 at 8 to 32,
against the white limit 18.6; the matched folded correction stays at 0.04 to
0.05 up to read 2 and degrades to 0.24 at read 4 and 1.1 to 1.7 beyond. The
white calibration must not be applied to shot data (over-subtracts, D goes
negative). Figures: `read_floor_sweep.png`, `fftcnr_white_vs_shot.png`.

## Depends on

`width_debias` (`noise`). No sibling-folder helpers.

## See also

`docs/DERIVATION.md` (the noise-bias correction).
