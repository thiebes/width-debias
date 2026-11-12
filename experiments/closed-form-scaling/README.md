# closed-form-scaling

## Purpose

Test the dimensional/CRLB scaling claim that the noise bias `bias/sigma^2`
is a function of `s = CNR*sqrt(sigma/dx)` alone, not CNR alone. The test is
the empirical check on the closed-form scaling argument.

## Contents

- `linearity_test.py` -- D = 0 sweeps at four σ₀ on a fixed pixel grid; compares
  the mean bias per cell with the exact Fisher value on that grid and with the
  closed form 1.13/s², in units of each cell's Monte Carlo SE.

## Data

None. Fully synthetic.

## Run

```bash
.venv/Scripts/python.exe experiments/closed-form-scaling/linearity_test.py
```

Set `WIDTH_DEBIAS_WORKERS=0`. Outputs to repo-root `output/`; committed
snapshots are here.

## Key result + figure

The fixed grid has 251 pixels over ±1.5 μm (dx = 0.012 μm), with σ₀ = 0.05,
0.1, 0.2, and 0.5 μm (4.2 to 41.7 pixels per σ₀) and 500 runs per (σ₀, CNR)
cell with independent seeds per σ₀. In units of σ₀², the mean bias follows the
exact Fisher value, the leading-order bias Var(σ̂) of the width fit, wherever
s = CNR·√(σ₀/dx) is at least 10:

| σ₀ (μm) | Cells with s ≥ 10 | χ²/dof | Beyond 2 SE | Cells with s ≥ 5 | χ²/dof | Beyond 2 SE |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 0.05 | 81 | 0.89 | 1 | 104 | 1.10 | 4 |
| 0.1 | 92 | 0.88 | 3 | 116 | 0.95 | 5 |
| 0.2 | 104 | 0.88 | 2 | 128 | 1.05 | 5 |
| 0.5 | 121 | 0.97 | 7 | 144 | 0.99 | 8 |

The bias over σ₀² therefore depends on s alone, falling as 1.13/s². The exact
Fisher value equals the closed form to 1.000 for σ₀ up to 0.2 μm and is 1.004
times it at σ₀ = 0.5 μm, where the half-window is 3 σ₀ and the profile is
truncated at the window edge. Below s of about 5 the mean departs from the
Fisher value by many SE. It reads high by up to several hundred times (at 4.2
pixels per σ₀; tens of times at 8.3 and 16.7) where low-CNR fits wander toward
the window, and it reads low at the lowest s for σ₀ = 0.2 and 0.5 μm, where
the Fisher value diverges and the bounded fit saturates. That is the regime
below the correction floor. In per-frame CNR, s = 10 is CNR 4.9 at 4.2 pixels
per σ₀ and CNR 1.55 at 41.7 pixels per σ₀. The
truncated-normal form λ(−s)/s falls exponentially in s and does not describe
the bias. Figure:
`linearity_test.png`; table `linearity_test.txt`.

## Depends on

`width_debias`. No sibling-folder helpers.

## See also

`docs/DERIVATION.md` section 10 (closed-form scaling).
