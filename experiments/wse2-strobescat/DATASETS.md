# Datasets -- wse2-strobescat

Collaborator data (Enrique Arévalo Rodríguez), read only and never committed.
The scans are not distributed with this repository; a user with a copy of the
lab folder places it as below.

## Source

stroboSCAT (`.npy`) diffusion scans, read from the `stroboscat_scans/` folder of
the data root (`scripts/data_root.py`: the folder `WIDTH_DEBIAS_DATA_ROOT`
names, or else `data/` at the repository root), which holds the experimenter's
lab folder in its own layout:

```text
stroboscat_scans/MultiChannelData/        # Perovskites, WSe2, mixed_tmdc, QDs
```

The diagnostics read his processed `analysis_[...]_.npy` per-folder (his per-frame
`profiles`, `sigmas`, `time_arr`, `time_fit`). Measurements registered in
`realdata_diagnostics.py` (paths under `stroboscat_scans/MultiChannelData/`):

```text
WSe2/Power_var/250224/0.020          WSe2/Better_power/5MHz/0.022
WSe2/Power_var/250606/0.025          WSe2/Power_var/250722/0.043
mixed_tmdc/comparison_all/0          mixed_tmdc/pureS/250701   (MoS2)
mixed_tmdc/26_32/250716              mixed_tmdc/50_50/250702   (MoS2/MoSe2 alloys)
```

and the MAPbBr3 transient-PL scan `stroboscat_scans/TPLM _TSM/MA/250730`.

Reduction / fit follow Enrique's own pipeline
(`MultiChannel/strobe_evaluation_v0.py`, `Diffusion/diff_eval_app.py` -- the
t<=2 ns forced-origin inverse-variance D fit reproduced by
`realdata_enrique_pipeline.py`). The `realdata_triage.py` screen scans every
`analysis_*` folder under `stroboscat_scans/`.

The scripts write all generated outputs under `diagnostics/`, which is not
tracked: it holds values and images derived from the unpublished scans. The
registry includes
WSe2 `Power_var/250224/0.020`, which is degenerate (negative D) under per-sweep
re-zeroing and is not used.

## Notes

stroboSCAT is white / read-floor-dominated, which matches the shipped white
calibration. TPLM/TCSPC data is shot/Poisson and needs a shot-matched
calibration, not the white one -- match the calibration to the technique.
