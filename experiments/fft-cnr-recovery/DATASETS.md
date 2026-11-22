# Datasets -- fft-cnr-recovery

This set derives its figure from one array of an external simulation study of
the fft-cnr estimator. The array is synthetic, not collaborator data.

| Item | Location | What it is |
| ------- | ------- | ------- |
| Source array | `case1_gaussian_1d.npz` in the directory that `WIDTH_DEBIAS_EXTERNAL_RESULTS` names (not in this repository; ~2.3 MB) | 1000 fft-cnr estimates per (window size N, true CNR) for Gaussian 1-D profiles. |
| Committed summary | `fft_cnr_recovery_summary.npz` (this folder) | Per-(N, CNR) median and 25/75, 5/95 percentiles reduced from the source; ~120 floats. The figure plots from this. |

The build script reads the source only when the committed summary is missing, then
writes the summary; it then needs `WIDTH_DEBIAS_EXTERNAL_RESULTS` set to the
directory that holds the source array.
