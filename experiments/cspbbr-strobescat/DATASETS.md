# Datasets -- cspbbr-strobescat

Collaborator data: Upasana's CsPbBr3 single-crystal scans (the Antonella/Carlos
diffusion paper). They are not part of this repository; place a copy in the
`cspbbr_sample_1/` folder of the data root. The data root is the folder that
`WIDTH_DEBIAS_DATA_ROOT` names, or else `data/` at the repository root;
`scripts/collaborator_data.py` holds its layout.

## Source

Raw transient-PL-microscopy acquisition CSVs, one subfolder per fluence:

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
