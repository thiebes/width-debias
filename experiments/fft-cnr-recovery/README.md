# fft-cnr-recovery

**Purpose** -- show that the per-frame CNR the correction consumes is itself
measurable, by recovering the true CNR from noisy profiles with the open-source
fft-cnr estimator. The estimates come from an external simulation study of the
estimator; this set replots them in the repository's figure palette.

**Contents**

- `build_fft_cnr_recovery_figure.py` -- reduces the external full-trial array
  to a small per-(N, CNR) summary and plots the recovered-vs-true CNR figure in
  the Spanish-flag palette.
- `fft_cnr_recovery_summary.npz` -- committed small derived artifact: per-(N, CNR)
  median and 25/75, 5/95 percentiles over the 1000 trials (~120 floats).
- `fft_cnr_recovery.png` + `fft_cnr_recovery_caption.md` -- the figure and caption.

**Data** -- reduced from the external simulation study's full-trial array, which is not
committed. See [`DATASETS.md`](DATASETS.md).

**Run**

```bash
.venv/Scripts/python.exe experiments/fft-cnr-recovery/build_fft_cnr_recovery_figure.py
```

No env vars needed to plot the committed summary. Rebuilding the summary (delete
it first) needs `WIDTH_DEBIAS_EXTERNAL_RESULTS` set to the directory that holds
the source array.

**Key result + figure** -- the fft-cnr 0.4.0 estimate sits on the 1:1 line from
CNR = 2 to 100 across window sizes N = 128-1024: in every cell the median over the
1000 trials lies between 0.989 and 1.006 of the true CNR, the largest departure
being at N = 1024, CNR = 2. Figure `fft_cnr_recovery.png`, caption
`fft_cnr_recovery_caption.md`.

**Depends on** -- no in-repo helpers; reads the external result array only to
rebuild the summary and, when the array is available, to check its hash against
the summary's.

**See also** -- `src/width_debias/cnr.py` (`per_frame_cnr`, the fft-cnr wrapper
the correction uses on real data), `experiments/wse2-strobescat` (the CNR key
on real data), and `docs/VALIDATION_DESIGN.md`.
