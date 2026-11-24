# Datasets -- nc-solid

Collaborator data (Marc Meléndez; Solari et al. maps), read only and never
committed.

## Source

The loader `nc_data.py` reads the maps from the `simulated_map/` folder of
the data root (`scripts/data_root.py`: the folder `WIDTH_DEBIAS_DATA_ROOT`
names, or else `data/` at the repository root):

```text
simulated_map/diffmap_small.dat
simulated_map/diffmap_medium.dat
simulated_map/diffmap_large.dat
```

These are the disordered CH3NH3PbBr3 nanocrystal-solid exciton-hopping maps
(small / medium / large; C16 / C12 / C8 ligands, mean NC radius 3.7 / 5.3 /
7.4 nm), runs of the Solari et al. hopping model that Marc made for this work.
They are not the paper's Fig. 5 runs (the paper used only variance
curves; its deposit, doi:10.21950/A0AB3H, has no maps). The maps are
~50,000-walker histograms tracking position only; the loader imposes decay
`N0*exp(-t/tau)` and (with `denoise=True`) reconstructs the smooth Gaussian per
frame before measurement noise. The scripts use whichever of the three maps
are present. None is committed.

## In-repo derived artifact

`calibration_nc_solid_gaussian.npz` (this folder) is the committed
high-statistics var_bias calibration -- the only large derived artifact kept in
git, because the working copy in `output/` is gitignored.
