# Datasets -- high-N Br-I bed

Large data for this experiment is not committed. The loader reads it from the
`halide_mixing_highn/` folder of the data root (`scripts/data_root.py`: the
folder `WIDTH_DEBIAS_DATA_ROOT` names, or else `data/` at the repository root),
one subfolder per Br fraction `x_<frac>`, `<frac>` in `0.25`, `0.5`, `0.75`.
The analysis needs only the first two files below.

| dataset | path under `halide_mixing_highn/` | what it is |
| ------- | ----------- | ---------- |
| collapsed 1-D profiles | `x_<frac>/collapsed_1d.x_<frac>.npz` | the 400,000-walker averaged map projected to 1-D (prof_x/prof_y + axes + metadata). The profiles `br_highn_data.py` reads. ~1 MB each. |
| carrier variance truth | `x_<frac>/variance.x_<frac>.avg.dat` | the particle second moment per frame; the 1-D carrier truth is this / 2. ~30 KB each. |
| averaged 2-D map | `x_<frac>/diffmap.x_<frac>.avg.npy` | the 600x600x601 averaged density map before the 1-D collapse. ~865 MB each; not needed here, never committed. |
| per-instance raw runs | `x_<frac>/inst_*/` | the 16 individual 25,000-walker realizations that were averaged. |
| run config | `x_<frac>/parameters.x_<frac>.dat` | the Brownian-dynamics parameters for this fraction. |

The multi-GB raw simulation outputs (full per-instance diffmaps) remain on the
machine that produced them; only the averaged and collapsed products were kept,
and only the collapsed `.npz` and the variance file are needed by the analysis
here.

Connection to the code: `br_highn_data.py` (`HIGHN`, the `halide_mixing_highn/`
folder) loads `collapsed_1d.x_<frac>.npz` and `variance.x_<frac>.avg.dat` for each
fraction.
