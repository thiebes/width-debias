# correction-floor

## Purpose

Characterise where the correction works: the per-profile CNR floor below which
the calibrated subtraction stops recovering the true variance, both in isolation
and inside a full diffusion experiment.

## Contents

- `per_profile_floor.py [profile_model]` -- isolated single-profile sweep
  (50,000 fits per CNR cell); residual `mean(sigma_tilde^2) - sigma^2_true`.
- `analytical_floor.py [MODEL ...]` -- tests the leading-order prediction
  `var_bias = A/CNR^2` against the binned calibration and reports where it
  breaks; `A` comes from the exact Fisher information of the width fit, with
  the Monte Carlo estimator spread at high CNR as a check.
- `per_frame_in_diffusion_experiment.py [profile_model]` -- the per-frame
  `subtract` / MAS corrections inside a 12-frame diffusion experiment.
- `render_floor_panel.py` -- renders the floor figure
  (`poster_panel5_floor.png`) from this set's `per_profile_floor_*.txt`; uses
  `scripts/poster_style.py`.

## Data

None. Fully synthetic.

## Run

```bash
.venv/Scripts/python.exe experiments/correction-floor/per_profile_floor.py
.venv/Scripts/python.exe experiments/correction-floor/analytical_floor.py
.venv/Scripts/python.exe experiments/correction-floor/per_frame_in_diffusion_experiment.py
```

`per_profile_floor.py` runs its own process pool (all cores but one), so
`WIDTH_DEBIAS_WORKERS` has no effect on it; with 7 workers it runs about 1 h
(Gaussian), 2.5 h (p = 4), 3.7 h (Voigt), and 9 h (p = 1.5). Set
`WIDTH_DEBIAS_WORKERS=0` for the other two scripts. Outputs to repo-root
`output/`; committed snapshots are here.

## Key result + figure

The per-profile floor test fits 50,000 profiles per test cell and corrects them
with the dense D-free lookup (100 CNR₀ cells of 5000 runs, the same budget for every shape, seeds
independent of the test cells and offset by 10⁶ per shape). Walking down from
high CNR, the floor is the deepest per-profile CNR reached while every cell
satisfies |residual| + 2 combined SE ≤ tolerance. The combined SE adds the test
cell's Monte Carlo error and the lookup's own error at that CNR, so a cell
counts only when its residual is shown to be inside the tolerance. The
tolerance is a fraction of the end-point MSD 2 D t_max of the default
experiment (t_max = 3τ; D in σ₀²/τ):

| Tolerance (σ₀²) | Basis | Gaussian | Voigt | p = 1.5 | p = 4 |
| ------- | ------- | ------- | ------- | ------- | ------- |
| 0.003 | 0.1 × 2 D t_max, D = 0.005 | 7.27 | 9.45 | 12.3 | 3.31 |
| 0.009 | 0.3 × 2 D t_max, D = 0.005 | 1.96 | 3.31 | 3.31 | 1.96 |
| 0.012 | 0.1 × 2 D t_max, D = 0.02 | 1.51 | 2.55 | 2.55 | 1.51 |
| 0.036 | 0.3 × 2 D t_max, D = 0.02 | 1.16 | 1.16 | 1.51 | 0.89 |
| 0.060 | 0.1 × 2 D t_max, D = 0.1 | 1.16 | 0.89 | 1.16 | 0.14 |
| 0.180 | 0.3 × 2 D t_max, D = 0.1 | 0.89 | 0.065 | 0.89 | 0.050 |
| 0.010 | fixed tolerance | 1.96 | 2.55 | 3.31 | 1.96 |

The headline floor is the 0.036 σ₀² row (0.3 of the end-point MSD at
D = 0.02 σ₀²/τ), at per-profile CNR 1.16 (Gaussian), 1.16 (Voigt), 1.51
(p = 1.5), and 0.89 (p = 4). Under the same walk a fixed 0.01 σ₀² tolerance gives 1.96,
2.55, 3.31, and 1.96. Near CNR 2 the lookup's own SE is about 0.004 σ₀², so two
combined SE nearly fill a 0.01 σ₀² tolerance before any residual is counted.
The floor at a tight tolerance therefore reports the calibration budget as much
as the correction. The residuals themselves stay small to lower CNR: from
per-profile CNR 1.16 up, |residual| is at most 0.006 σ₀² (Gaussian), 0.008 σ₀²
(Voigt), and 0.004 σ₀² (p = 1.5), and p = 4 stays within 0.005 σ₀² from 0.89
up. The p = 4 fit reads low at deep CNR (raw mean 0.53 σ₀² at CNR 0.05).

The leading-order model (`analytical_floor.py`, 30 CNR₀ cells of 1000 runs)
takes `A` from the exact Fisher information of the width fit: 0.0564
(Gaussian), 0.0812 (Voigt), 0.1007 (p = 1.5), and 0.0191 (p = 4) σ₀². The
Monte Carlo estimator spread at CNR ≥ 8 agrees to a Fisher / Monte Carlo ratio
of 0.986 to 1.006. Walking down from high CNR, the ratio of a resolved bin's
`var_bias` to A/CNR² first differs from 1 by more than 3 SE at CNR about 1.18
for Gaussian, p = 1.5, and p = 4, and about 0.85 for Voigt. Below that the bias
exceeds the leading term on Gaussian (ratio 2.0 at CNR 1.18, up to 8.9 at CNR
0.48), Voigt (1.45 to 1.62 at CNR 0.48 to 0.85), and p = 1.5 (1.7 to 3.6 at CNR
0.48 to 1.18); for p = 4 it turns negative (ratio −1.4 ± 0.2 at CNR 1.18).
Design rule: keep every frame's per-profile CNR above about 1 to 1.5.

Result tables: `per_profile_floor_<model>.txt` (`genGauss` is a p = 2 run kept for
reference, with a 1000-run calibration and the earlier floor rule
(|z| > 2 and |residual| > 0.01 σ₀²);
`genGauss_p1.5`, `genGauss_p4`), `analytical_floor.txt`,
`per_frame_in_diffusion_experiment.txt`. The poster figure
`poster_panel5_floor.png` was rendered from the earlier Gaussian, Voigt, and
p = 2 tables (1000-run calibration); `render_floor_panel.py` reads the committed
tables, so a re-render uses the 5000-run Gaussian and Voigt tables above.

## Depends on

`width_debias`. No sibling-folder helpers.

## See also

`docs/DERIVATION.md` section 9.
