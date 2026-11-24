# Datasets -- cspbbr-strobescat

Collaborator data (Upasana; the Antonella/Carlos diffusion paper), read only and
never committed.

## Source

Raw CsPbBr3 single-crystal transient-PL-microscopy acquisition CSVs, organised
by fluence subfolder under the sample root, the `cspbbr_sample_1/` folder of
the data root (`scripts/data_root.py`: the folder `WIDTH_DEBIAS_DATA_ROOT`
names, or else `data/` at the repository root):

```text
cspbbr_sample_1/<fluence folder>/
```

Each fluence subfolder (e.g. `13.47 uJ_Fluence_19_May_2026`) holds one CSV per
repeat (each a spatial-position scan) plus a `*metadata.json`. `run_all.py`
reads the sample root (`--sample-root`, by default this folder) and `reduce.py`
one fluence folder (`--data`).

## Notes

The noise is white (constant-floor) -- a raster-scanned APD measurement whose
photon-transfer fit is floor-dominated -- so the gaussian calibration model
applies.
