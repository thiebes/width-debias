# Datasets -- nc-solid

Collaborator data: Marc Meléndez's runs of the Solari et al. exciton-hopping
model. They are not part of this repository; place a copy in the
`simulated_map/` folder of the data root. The data root is the folder that
`WIDTH_DEBIAS_DATA_ROOT` names, or else `data/` at the repository root;
`scripts/collaborator_data.py` holds its layout.

## Source

```text
simulated_map/diffmap_small.dat
simulated_map/diffmap_medium.dat
simulated_map/diffmap_large.dat
```

`nc_data.py` reads these disordered CH3NH3PbBr3 nanocrystal-solid
exciton-hopping maps (small / medium / large; C16 / C12 / C8 ligands, mean NC
radius 3.7 / 5.3 / 7.4 nm), runs of the Solari et al. hopping model that Marc
made for this work. They are not the runs behind Fig. 5 of Solari et al., which
used only variance curves; that paper's deposit, doi:10.21950/A0AB3H, has no
maps. The maps are ~50,000-walker histograms tracking position only; the loader
imposes decay `N0*exp(-t/tau)` and (with `denoise=True`) reconstructs the smooth
Gaussian per frame before measurement noise.

`nc_solid_validation.py` uses whichever of the three maps are present,
`nc_solid_poster_figure.py` needs the small map, and `nc_solid_alpha_vs_cnr.py`
needs all three.

## In-repo derived artifact

`calibration_nc_solid_gaussian.npz` (this folder) is the committed
high-statistics var_bias calibration -- the only large derived artifact kept in
git, because the working copy in `output/` is gitignored.
