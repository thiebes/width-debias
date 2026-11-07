# High-N Br-I bed and the width-fit upper-bound result

A self-contained study folder for two coupled results:

1. A clean, high-statistics Br-I halide-mixing test bed (400,000-walker average),
   and
2. the result that the per-frame width fit's **upper bound** -- not the
   contrast-to-noise alone -- governs how large the noise bias appears.

## The upper bound

A per-frame Gaussian width fit that bounds the fitted variance to
`[~1 pixel^2, span^2]`, where `span` is the full analysis window, misbehaves
when the output window is sized far wider than the spot -- here a 3000 nm box
around a 25 nm spot. The `span^2` ceiling lets a low-contrast fit wander out
toward the window scale on noise. Two consequences:

- the **naive** D is inflated 3-5x at low CNR (most of the apparent bias is the
  loose bound, not the truncation), and
- the bias **calibration** built from the same fit explodes (`var_bias` ~ 1e5
  nm^2 at CNR ~1-2), so the correction over-subtracts catastrophically.

The remedy is a physical cap `var_max = (G * sigma0)^2`, with `G` the maximum
credible width growth `sigma_max / sigma0`. For this bed `G = 12` caps sigma at
300 nm, clearing the real ~196 nm late-frame spot with margin while bounding the
noise-wander. The bound must be **identical** in the calibration and the
application fit, or the subtraction is mismatched.

With the bound, the residual noise bias is modest. The figure
(`br_halide_highn_perframe.png`, CNR0 = 15, 800 realizations; table
`br_halide_highn_perframe.txt`) plots each frame's mean fitted variance minus
the Gaussian fit of the noise-free profile, which isolates the noise bias. The
short-window D over the usable band, divided by the D of the noise-free fit
(the noise correction's target) and, in the last column, by D_ref:

| Br fraction | naive | joint grid | Fisher, σ₀ calibration | Fisher, 75 nm calibration (post hoc) | joint grid / D_ref |
| ------- | ------- | ------- | ------- | ------- | ------- |
| 0.25 | 1.034 | 1.002 | 0.594 | 1.002 | 0.993 |
| 0.5 | 1.049 | 1.013 | 0.370 | 1.010 | 0.971 |
| 0.75 | 1.060 | 1.025 | 0.349 | 1.023 | 0.984 |

The SE over the 800 realizations is 0.004 to 0.007 in every column. Each
corrected arm also carries a calibration SE, the spread of D over 200 draws of
its lookup's own Monte Carlo error, which every measurement shares: 0.001 for
the joint grid, 0.024 to 0.034 for the σ₀ Fisher calibration, and 0.003 to
0.004 for the 75 nm one. Every corrected arm uses the iterated
measurement-level key, and the figure shows the joint grid. The x = 0.75
remainder of 0.025 is resolved. The difference between the two joint-grid
columns is the carrier gap.

The legacy 20k-walker maps do not show this effect because their narrow 150 nm
window already bounds the wander. DICE is unaffected: it sizes its field of view
relative to the FWHM, so its effective width bound already scales with the spot.
The problem is specific to analysing fixed wide-window simulation maps with
`span^2`.

## One calibration rescaled by width

The joint grid measures the bias at every width: 40 CNR points × 26 widths ×
4000 runs, 4.2 million fits. The Fisher width table (`fisher_calibration`)
calibrates at one width instead, with 84 CNR cells × 2000 runs (168,000 fits),
and rescales the bias to each frame's width with the exact Fisher variance of
the width fit on this pixel grid. The arm specified before the run calibrates
at σ₀ = 25 nm. It over-subtracts on every map: the signed residual against the
noise-free fit over the usable band is −1180 to −1632 nm² (joint grid +8 to
+53 nm²), and D falls to 0.35 to 0.59 of the noise-free fit's D.

The cause is the window, not the Fisher scaling. At σ₀ in the 3000 nm window,
low-CNR fits wander toward the 300 nm cap, so the bias the dense joint grid
measures near σ₀ (σ ≈ 28 nm) is about 25 to 100 times the Fisher term at CNR
1 to 2.5, against 1 to 2 times at σ ≥ 76 nm. Rescaling the σ₀ calibration
carries that window-set term into the wide late frames, which are the frames
the correction acts on.

A second arm, added after the first failed and therefore post hoc, calibrates
at 75 nm (`FISHER_LATE_SIGMA_NM`). The width was fixed by a rule before the
run: the median noise-free width of the frames with per-frame CNR between the
floor and 3, pooled over the three maps and CNR₀ 8, 15, and 30 (medians per map
and CNR₀ 60 to 123 nm). It matches the joint grid to within 0.003 in D on every
map, with a calibration SE of 0.003 to 0.004 from 1/25 of the joint grid's
fits. On this bed, one calibration rescaled by the Fisher table works when it
is made at the width of the frames that need the correction, and fails when it
is made at a width where the window dominates the low-CNR fits. The rule is
tested on this bed at CNR₀ 15 only.

## Per-frame residual: the CNR-only lookup and the reference offset

With the 1-D D-aware lookup, the per-frame table at CNR0 = 20 shows the mean
|residual| over the usable band rising after correction (132 -> 525, 89 -> 155,
142 -> 260 nm^2 for x = 0.25 / 0.5 / 0.75). Two separate effects produce it.

1. **The 1-D lookup mixes widths.** The D-aware calibration is keyed on CNR
   alone, but here sigma^2 grows ~64x over the usable band and var_bias scales
   with sigma^2. The calibration reaches each CNR along its own peak-decay path,
   while the data's CNR also falls as the spot spreads, so one CNR value pools
   calibration frames from sigma^2 = 625 to ~38,000 nm^2, and `np.interp`
   through those unsmoothed points is erratic (2031, 7773, and 529 nm^2 at CNR
   1.71, 1.56, and 1.48 for x = 0.25). The script defaults to a joint
   (CNR, sigma^2) D-free calibration on the data's pixel grid
   (`cnr_sigma2_calibration`); `daware` reproduces the D-aware table.
2. **The carrier reference is not the noise-free fit.** At CNR 3 to 10, where
   the noise bias is negligible, the naive residual against `var_true` is
   already -50 to -170 nm^2: the Gaussian fit of the noise-free profile sits
   118 to 148 nm^2 below the carrier second moment on average (up to 256 nm^2).
   That is the shape and emission-to-carrier gap, which the noise correction
   does not address, so a residual against `var_true` mixes it in. The clean
   metric for the noise correction is the residual against the noise-free fit.

Against the noise-free fit at CNR0 = 20 (800 seeds), mean |residual| over the
usable band goes 144 -> 70, 94 -> 61, 170 -> 58 nm^2 with the joint calibration
(signed +140 -> +36, +94 -> -6, +170 -> +5), against 144 -> 462, 94 -> 56,
170 -> 211 with the D-aware lookup.

The joint calibration is sensitive to its key as well as to its Monte Carlo
budget. Applied per measurement at each frame's own noisy sigma_hat^2 (with the
fixed-point inversion), the lookup reads near the grid edge at low CNR and
over-subtracts: at x = 0.5, CNR0 = 10, frames between the floor and CNR 2 go
from +815 nm^2 (naive) to -1274 nm^2 (600-run grid), while keying the same grid
on the true sigma^2 gives +550 and a per-measurement smoothed key gives +264
(residuals against the noise-free fit). The smoothed key (`HIGHN_KEY=smoothed`,
`smoothed_key`) is a robust straight line through the measurement's own
sigma_hat^2 above the floor; it assumes sigma^2(t) is linear, which holds for
this bed's normal diffusion, so it is a bed option, not the library default.

With a denser joint grid (40 CNR points, 20 of them below CNR 3, x 26 sigma x
4000 runs; `cal_highn_cnrsig2_g12_n4000_dense.npz`), mean |residual| against
the noise-free fit over the usable band, x = 0.25 / 0.5 / 0.75:

| CNR0 | key | naive -> corrected (nm^2) | signed corrected (nm^2) |
| ------- | ------- | ------- | ------- |
| 20 | per-frame | 144 -> 65, 94 -> 64, 170 -> 72 | +30, -25, -27 |
| 20 | smoothed | 144 -> 70, 94 -> 35, 170 -> 93 | +38, +25, +90 |
| 10 | per-frame | 177 -> 50, 235 -> 453, 270 -> 365 | -19, -431, -338 |
| 10 | smoothed | 177 -> 65, 235 -> 51, 270 -> 110 | +55, +24, +92 |

Short-window D over the usable band, as D_corr / D_ref at CNR0 8 / 15 / 30:
per-frame key 0.85 / 0.99 / 0.99 (x = 0.25), 0.32 / 0.90 / 0.96 (0.5),
0.34 / 0.90 / 0.96 (0.75); smoothed key 1.00 / 0.99 / 0.99, 0.58 / 0.97 / 0.96,
0.63 / 0.98 / 0.97 (naive 1.08 to 1.21 at CNR0 8). So the dense grid with the
smoothed key removes the noise bias per frame and recovers D within 0.96 to
0.99 from CNR0 15; at CNR0 8 the two slower-spreading maps still read low,
because frames near the floor dominate that short window.

The figure uses the dense joint calibration with the iterated key
(`iterated_key_correction`: the straight line refitted to the corrected values
until it stops changing) at CNR0 15, the lowest initial CNR at which D does not
depend on the key.
Tables: `br_halide_highn_perframe_n4000[_smoothed]_cnr0{10,20}.txt` (dense
grid), `br_halide_highn_perframe_cnrsig2_cnr0{10,20}.txt` (600-run grid),
`br_halide_highn_perframe_daware_cnr020.txt` (D-aware lookup).

Key sensitivity (`key_sensitivity.py`, 300 seeds per map and CNR0, dense grid;
table `key_sensitivity.txt`). The fits are shared, so the keys differ only in
the lookup; D is short-window D / D_ref with an SE from resampling the seeds.

- From CNR0 15 the key barely matters. Perturbing the exact key by a factor
  of 0.8 or 1.2 moves D by at most 0.005 (|dD/de| <= 0.018), and the smoothed and
  iterated keys give the same D as the exact key to within 0.003. The
  per-frame key is the exception: it leaves a resolved low bias even here
  (0.90 and 0.89 at CNR0 15, 0.96 at 20, for x = 0.5 and 0.75; SE 0.01).
- The trend keys are not accurate keys. The iterated line reads 0.08 to 0.20
  of the true sigma^2 below it on average, with a scatter of 0.15 to 0.39 of
  sigma^2 per frame and measurement, yet from CNR0 15 that costs at most 0.003
  in D.
- At CNR0 8 to 10 the result depends on the key: dD/de is 0.12 to 0.28,
  even the exact key gives 1.00 to 1.13, and the smoothed and iterated keys
  disagree by up to 0.3 (x = 0.5, CNR0 8: smoothed 0.59, iterated 0.89, exact
  1.00), well beyond their SE of about 0.05. The data do not determine D there
  once the correction is applied, so this bed's D should be quoted from CNR0
  15 only.
- The residual 0.96 to 0.98 of D_ref at CNR0 20 to 30 for x = 0.5 and 0.75
  appears with every key and with the naive fit at CNR0 30, so it is the shape
  and emission-to-carrier gap against the carrier reference, not noise.

## The bed

Marc Melendez's 20k-walker Br-I maps carry Poisson sampling noise that floors the
intrinsic peak CNR ~30. Running his C Brownian-dynamics simulator at 400,000
walkers (16 instances of 25,000, averaged) gives a clean reference (intrinsic
peak CNR ~50). The 2-D averaged map is collapsed to 1-D on the run machine
(`P:/jthiebes/WIP/BrI_halide_highN/x_<frac>/collapsed_1d.x_<frac>.npz`); the
multi-GB maps stay on the desktop, and only the small collapsed file is read here.
The run procedure is in `experiments/halide-mixing-brI-legacy/rerun/README.md`.

Geometry: 600 bins over a 3000 nm window (5 nm/bin), 601 frames at 5 ps to 3.0 ns,
initial 1-D sigma ~25 nm. The carrier (transport) reference is the particle second
moment `variance.avg.dat / 2`. It is **Fickian** -- alpha 0.997 / 0.993 / 0.990
for x = 0.25 / 0.5 / 0.75, with D = 6265 / 2650 / 2352 nm^2/ns -- D falling with
Br fraction confirms that mixing inhibits transport. (The legacy maps' apparent
subdiffusion is a short-window / narrow-box artifact, not real transport.)

Over the 3 ns window the peak drops ~147x, so most late frames fall below CNR 1
and are excluded by the floor; the correction is tested on the usable band (a
shorter effective window, which is the fair test).

## Contents

- `br_highn_data.py` -- loads the collapsed 1-D profiles, the reference from
  `variance.avg.dat / 2`, imposes the tau ~ 1 ns decay envelope.
- `br_halide_highn_validation.py` -- the CNR0 sweep at this geometry; naive vs
  corrected alpha/D. Uses the 1-D D=0 calibration, which **under**-corrects here
  because the width grows 26-61x (the 2-D (CNR, sigma^2) D-free calibration is the
  deployable path).
- `br_halide_highn_perframe.py [cnr_sigma2|daware|fisher|fisher_late] [CNR0]` --
  the per-frame residual (with a signed per-frame table), short-window D, and
  the figure. Default calibration is the joint (CNR, sigma^2) one; `daware` is
  the 1-D D-aware lookup; `fisher` and `fisher_late` are the Fisher width table
  calibrated at σ₀ and at 75 nm. `HIGHN_KEY` selects the key: `perseed`,
  `smoothed`, or `iterated`.
- `render_perframe_figure.py` -- renders the figure and writes
  `br_halide_highn_perframe.txt` (CNR0 = 15, iterated key, 800 realizations;
  the joint grid, which the figure shows, and both Fisher width-table arms, each
  with its calibration SE).
- `key_sensitivity.py` -- the key-sensitivity comparison (`key_sensitivity.txt`).
- `br_halide_highn_perframe.png` -- the figure (CNR0 = 15); caption in
  `br_halide_highn_perframe_caption.md`.

## Depends on (in the library, not here)

The `max_var` bound lives in the shared fit so calibration and application use one
implementation:

- `src/width_debias/fit.py` -- `fit_variance(x, y, max_var=None)`; default
  `None` keeps the `span^2` bound.
- `src/width_debias/models.py` -- `GaussianModel(max_var=...)` forwards it.

The scripts also reuse two pure helpers (`fit_msd_powerlaw`, `fit_each_row`) from
`experiments/halide-mixing-brI-legacy/br_halide_validation.py`.

## Running

Needs the pshare for the collapsed maps. Pin the BLAS threads so the process pool
does not oversubscribe, and use a modest worker count if the machine is busy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WIDTH_DEBIAS_WORKERS=4 \
  .venv/Scripts/python.exe experiments/halide-mixing-highn/br_halide_highn_perframe.py
```

`br_halide_highn_perframe.py` writes its report, figure, and cached
calibrations to the repo's `output/` (not tracked); `render_perframe_figure.py`
writes the committed figure and table into this folder.

## See also

- `experiments/halide-mixing-brI-legacy/` (the legacy 20k-walker bed).
