# Datasets -- high-N Br-I bed

Re-runs of Marc Meléndez's Brownian-dynamics Br-I simulator at 400,000 walkers
(16 instances of 25,000, averaged; procedure in
`../halide-mixing-brI-legacy/rerun/README.md`). They are not part of this
repository; place a copy in the `halide_mixing_highn/` folder of the data root.
The data root is the folder that `WIDTH_DEBIAS_DATA_ROOT` names, or else `data/`
at the repository root; `scripts/collaborator_data.py` holds its layout.

## Source

One subfolder per Br fraction `x_<frac>`, `<frac>` in `0.25`, `0.5`, `0.75`.
`br_highn_data.py` reads the first two files below; the others are listed for
completeness.

| dataset | path under `halide_mixing_highn/` | what it is |
| ------- | ----------- | ---------- |
| collapsed 1-D profiles | `x_<frac>/collapsed_1d.x_<frac>.npz` | the 400,000-walker averaged map projected to 1-D (prof_x/prof_y + axes + metadata). ~1 MB each. |
| carrier variance truth | `x_<frac>/variance.x_<frac>.avg.dat` | the particle second moment per frame; the 1-D carrier truth is this / 2. ~30 KB each. |
| averaged 2-D map | `x_<frac>/diffmap.x_<frac>.avg.npy` | the 600x600x601 averaged density map before the 1-D collapse. ~865 MB each. |
| per-instance raw runs | `x_<frac>/inst_*/` | the 16 individual 25,000-walker realizations that were averaged. |
| run config | `x_<frac>/parameters.x_<frac>.dat` | the Brownian-dynamics parameters for this fraction. |

The multi-GB raw simulation outputs (full per-instance diffmaps) remain on the
machine that produced them.
