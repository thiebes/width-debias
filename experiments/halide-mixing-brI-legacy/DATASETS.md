# Datasets -- halide-mixing-brI-legacy

Collaborator data (Marc Meléndez), read only and never committed.

## Source

The loader `br_data.py` reads the maps from the `halide_mixing/` folder of
the data root (`scripts/data_root.py`: the folder `WIDTH_DEBIAS_DATA_ROOT`
names, or else `data/` at the repository root), one folder per Br fraction:

```text
halide_mixing/BrI_x_<frac>/diffmap.x_<frac>.dat
halide_mixing/BrI_x_<frac>/variance.x_<frac>.dat
```

These are the legacy ~20,000-walker kinetic-Monte-Carlo Br-I halide-mixing
diffmaps (three Br fractions: 0.25, 0.5, 0.75); the fractions present are used.
The same folders hold the simulator and its input files, which
`rerun/README.md` copies for a re-run.

## Note

The high-statistics 400,000-walker bed (clean Fickian ground truth) and the
width-fit upper-bound finding live in the separate
`experiments/halide-mixing-highn/` set; see its `DATASETS.md`.
