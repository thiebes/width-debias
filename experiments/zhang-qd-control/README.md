# zhang-qd-control

## Purpose

A scope/robustness control: apply the noise-bias correction to Zhang et al.
(2022) QD-solid ultrafast exciton-transport data (Dfast at early times) and show
the result is robust -- the correction barely moves it, because the per-frame CNR
stays well above the floor. This is a control example, not a correction target.

## Contents

- `zhang_qd_panel.py` -- reduces the published figure CSVs and runs the
  correction; reads the `Figure *.csv` files from this folder.
- `Figure 1d <delay>.csv`, `Figure 1e <delay>.csv` -- raw 64x64 differential
  ON/OFF transient-microscopy frames at each pump-probe delay.
- `Figure 2a.csv`, `Figure 2b.csv` -- published MSD(t) curves.
- `Figure 4a-d.csv` -- MSD(t) of the CdSe / CdS / CsPbI3 "no transport" controls.

## Data

The extracted published-figure CSVs are committed here (small real data). The raw
Origin project (`zhang_Manuscript_data.opju`) is the source; it is gitignored
(local only) -- see `DATASETS.md`.

## Run

```bash
.venv/Scripts/python.exe experiments/zhang-qd-control/zhang_qd_panel.py
```

Outputs to repo-root `output/`; the committed snapshot is `zhang_qd_panel.png`.

## Key result + figure

The Zhang QD Dfast is robust to the correction. The per-frame CNR is 9.3 to 24.2
(white noise) and *rises* with pump-probe delay, so the subtracted bias is
largest at t = 0 (228 nm^2, below 0.01 of sigma0^2) and shrinks with delay; the
correction therefore raises Dfast slightly (corrected/naive 1.015 over
t <= 260 fs; 369 -> 375 cm^2/s from the five image frames). At these CNR the
outcome is expected by construction, so this is a scope control, not a test of
the correction. The raw calibration lookup reads -89 and -49 nm^2 at CNR 22 and
24, where the leading-order bias is a few nm^2: Monte Carlo noise in the
lookup, not a physical negative bias. Figure: `zhang_qd_panel.png`.

## Depends on

`width_debias`. The figure CSVs live beside the script (no external loader).

## See also

Source paper: Zhang et
al., "Ultrafast exciton transport at early times in quantum dot solids" (2022).
