# Datasets -- high-N Br-I bed

Re-runs of Marc Meléndez's Brownian-dynamics Br-I simulator at 400,000 walkers
(16 distinct realisations of 25,000 walkers, averaged and collapsed to 1-D; the
procedure and what is kept are in `../halide-mixing-brI-legacy/rerun/README.md`,
and that folder holds `collapse_diffmaps.py`, a collapse script written
afterwards and not checked against the profile files). They are not part of this
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
| first-run 2-D map | `x_<frac>/diffmap.x_<frac>.avg.npy` | the first run's averaged density map (600x600x601), a duplication-weighted average over 5 distinct realisations; not the map the collapsed profiles come from. ~865 MB each. |
| per-instance raw runs | `x_<frac>/inst_*/` | the 5 distinct 25,000-walker realisations of the first run, one folder each; the first run's 11 duplicate instances and the 11 added by the top-up are not kept. |
| run config | `x_<frac>/parameters.x_<frac>.dat` | the Brownian-dynamics parameters for this fraction. |

The averaged 2-D maps of the 16-realisation set, the 11 instances added per
fraction, the log of the top-up, and the script that made the collapsed profiles
are not kept, so the collapsed profiles cannot be re-derived from the files
above; the rerun README gives the check (`collapse_check.txt`).
