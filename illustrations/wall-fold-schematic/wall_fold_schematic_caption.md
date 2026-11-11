# Caption -- wall-and-fold schematic

Low contrast bites: a one-directional bias.

Left -- the fold at the wall, in deep noise (per-frame CNR below about 1). A
fitted width cannot be negative, so there the fitted-variance distribution is
bounded below at zero. The forbidden negative tail (variance < 0) folds back
across the wall as a reflected tail; the realizable estimate and the reflected
tail pile up near the wall, so the distribution of fits (their sum) is
right-skewed and its mean sits above the true variance. Above a per-frame CNR
of about 1 the estimate almost never reaches zero and nothing folds, but the
mean still sits above the true variance: squaring a width that scatters about
the true width adds the width's estimation variance. That excess is the
per-frame bias in the regime where the correction is validated; the fold shown
here adds to it only in deep noise.

Right -- the tilted slope. The per-frame bias grows as CNR falls across the late
frames, lifting the late-time mean-squared displacement above the true line. The
inflated fit has a steeper slope, so the extracted D is overestimated. It is a
bias, not scatter: averaging repeated measurements does not remove it.

Pure schematic.
