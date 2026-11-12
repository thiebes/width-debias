# Datasets -- mn-trapping

Collaborator data (Marc Meléndez). Referenced by pointer only -- never copied
into git, never moved or renamed on the pshare.

## Local working copy

`experiments/mn-trapping/data/` (gitignored) holds Marc's Mn-doped
reaction-diffusion diffmaps, two files per doping case:

- `diffmap_<case>.dat` -- `(n_t, n_x)` matrix of noiseless PL intensity.
- `PL_var_<case>.dat` -- reduced table whose third column (index 2) is the
  second moment `sigma^2(t)` of the PL profile (emission-weighted, not the
  carrier moment).

Cases: `undoped`, `0.51_Mn_doped`, `0.81_Mn_doped`, `4.04_Mn_doped` (Magdaleno
et al. 2025). The loader `marc_data.py` reads from this `data/` dir
(`Path(__file__).parent/"data"`).

## Authoritative source

Marc Meléndez's pshare share (the P: drive).
Do not commit these `.dat` files; do not edit or rename the pshare originals.
