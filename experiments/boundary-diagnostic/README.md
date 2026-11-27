# boundary-diagnostic

## Purpose

Decide which boundary model the width fit obeys at the wall at zero: does the
fitted variance **clip** (Tobit-style pile-up at the bound) or **reflect** (fold
into a right-skewed tail)? This decision selects the correction route.

## Contents

- `diagnose_boundary.py [profile_model]` -- one run per profile model; reports
  the fitted-variance distribution shape vs CNR.

## Data

None. Fully synthetic.

## Run

```bash
.venv/Scripts/python.exe experiments/boundary-diagnostic/diagnose_boundary.py
.venv/Scripts/python.exe experiments/boundary-diagnostic/diagnose_boundary.py voigt
.venv/Scripts/python.exe experiments/boundary-diagnostic/diagnose_boundary.py generalized_gaussian:1.5
.venv/Scripts/python.exe experiments/boundary-diagnostic/diagnose_boundary.py generalized_gaussian:4
```

Outputs to repo-root `output/`; committed snapshots are in this folder.

## Key result + figure

The fit **reflects** rather than clips, so the additive bias model of
`docs/DERIVATION.md` (Section 1) applies (not Tobit/clipping). Voigt and the
generalized-Gaussian p = 1.5 show a minor pile-up at the bound only at CNR 1;
the script's overall verdict line reads that single CNR 1 row, so it prints "CLIPPING" for them although
every row from CNR 1.5 up is reflection-like. The flat-topped p = 4 shows no
pile-up; its mean fitted variance is slightly low at CNR 1 (0.98 sigma_0^2).
Tables: `diagnose_boundary_genGauss_p1.5.txt`, `diagnose_boundary_genGauss_p4.txt`.
Figures: `boundary_diagnostic.png`, `boundary_diagnostic_voigt.png`, and
`boundary_diagnostic_generalized_gaussian.png` (a p = 2 run, which is a Gaussian
profile with the generalized-Gaussian fitter's bounds, kept for reference).

## Depends on

`width_debias`. No sibling-folder helpers.

## See also

`docs/DERIVATION.md` section 2.
