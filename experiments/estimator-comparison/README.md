# estimator-comparison

## Purpose

Answer "is the correction even needed, or does a different estimator avoid the
bias?" Compares the noise-bias correction against the non-parametric moment
estimator and an unbounded (positivity-wall-removed) width fit.

## Contents

- `moment_estimator_failure.py` -- five-way comparison; the `MomentModel`
  (second-moment estimator) is unstable at low CNR / low D (wild, often negative
  D).
- `unbounded_fit_comparison.py` -- `subtract` vs an unbounded width fit (the
  Gaussian fitted in precision `b = 1/sigma^2`, so the fitted variance can take
  any sign), with bounded+OLS and the moment estimator as references.

## Data

None. Fully synthetic.

## Run

```bash
.venv/Scripts/python.exe experiments/estimator-comparison/moment_estimator_failure.py
.venv/Scripts/python.exe experiments/estimator-comparison/unbounded_fit_comparison.py
```

Set `WIDTH_DEBIAS_WORKERS=0`. Outputs to repo-root `output/`; committed
snapshots are here.

## Key result + figure

From per-frame CNR ~1.5 up the bounded and unbounded fits are bit-for-bit
identical (the wall never binds), and both carry the same upward bias (mean
fitted variance 1.038, 1.016, 1.010 sigma_0^2 at CNR 1.52, 2.0, 2.6; `<D>/D_ref`
1.199 at CNR0 34 for both). So the bias in that range is not a wall effect, and
removing the wall does not remove it. Below CNR ~1.5, removing the wall only
partly debiases and explodes the variance, because a nonlinear shape-matched
fit intrinsically resists negative variances.

`subtract`, with the library's per-frame slope weights, reads 1.003 to 1.028
of D_ref from CNR0 21 up, where every frame has per-frame CNR >= 1 (within 1.7
evaluation SE; the calibration SE is not included), with SD(D/D_ref) 0.19 to
0.81 against 0.60 to 4.85 for the bounded OLS fit. Below CNR0 21 the last
frames fall under per-frame CNR 1 and it reads 0.82 to 1.03 (0.82 at CNR0 9,
about 4 evaluation SE low). The moment estimator gives a negative mean D at
every CNR0 here; the figures leave it out and the tables carry it.

Figures: `method_comparison.png`, `unbounded_fit_comparison.png`, each with a
`_caption.md`; table `unbounded_fit_comparison.txt`.

## Depends on

`width_debias` (`MomentModel`, `UnboundedGaussianModel`). No sibling-folder
helpers.

## See also

`docs/DERIVATION.md` section 7.
