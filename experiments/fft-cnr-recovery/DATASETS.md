# Datasets -- fft-cnr-recovery

This set derives its figure from one array of an external simulation study of
the fft-cnr estimator. The array is synthetic, not collaborator data.

| Item | Location | What it is |
| ------- | ------- | ------- |
| Source array | `case1_gaussian_1d.npz` (~2.3 MB), from an external simulation study. It is not distributed with this repository, and its location is deliberately not recorded here. | 1000 fft-cnr 0.4.0 estimates per (window size N, true CNR) for Gaussian 1-D profiles. |
| Committed summary | `fft_cnr_recovery_summary.npz` (this folder) | Per-(N, CNR) median and 25/75, 5/95 percentiles reduced from the source; ~120 floats. It also records the fft-cnr version (0.4.0) and the source array's SHA-256. The figure plots from this. |

The figure plots from the committed summary and needs no environment variable.
To rebuild the summary, set `WIDTH_DEBIAS_EXTERNAL_RESULTS` to the directory
that holds the source array (an absolute path), delete the summary, and run the
build script, which writes a new one. Whenever the source is available, the
script checks its hash against the one the summary records and warns if they
differ.
