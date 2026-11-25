# Running Marc's Br-I halide simulator at higher statistics

Marc Meléndez's Br-I diffusion maps are histograms of only 20,000 walkers, so
each frame carries Poisson sampling noise that inflates the per-frame
contrast-to-noise when read off the peak. Running the simulator at a much higher
walker count gives a clean ground truth directly, with no profile-shape
assumption. This folder holds the instructions and helper scripts. The run is a
multi-day job on a Linux or WSL machine, separate from the analysis environment.
The averaged 400,000-walker bed it produces is analysed in
`experiments/halide-mixing-highn/`.

## Requirements

`landscapeBD.diffmap` is a 64-bit x86-64 ELF executable, dynamically linked
against only `libc.so.6` and `libm.so.6` (GLIBC ≤ 2.27). It needs no GPU, MPI, or
GSL. It runs unmodified on WSL2 with Ubuntu 20.04 or later (glibc ≥ 2.31, which
is backward compatible). The `stderr.log` from Marc's run shows that the binary
generates the crystal structure, neighbour list, and the Gaussian initial
exciton distribution internally, so the only inputs are the binary, the two
lattice files, and the parameter file.

## Files to copy from the simulation folders

From `halide_mixing/BrI_x_<frac>/` in the data root (for each Br fraction
`<frac>` in `0.25`, `0.5`, `0.75`), copy into a Linux working directory:

| file | role | SHA-256 (frac 0.25) |
| ------- | ------- | ------- |
| `landscapeBD.diffmap` | the simulator (chmod +x after copying) | `cdba07c99cc263fd497ef9bd51bb02c182b5b5395f2784120339e5f4b34ef591` |
| `basis_BrI.dat` | 8-atom perovskite basis | `67dd8210d7c087dfa94a5c9ab80659eec486bc33e87151694f49f3b1c79231a6` |
| `vectors_BrI.dat` | lattice vectors | `185a8189dc900d962236dc519d438a5cde4c1df8e62ff2a3788980f28e0c8262` |
| `parameters.x_<frac>.dat` | run configuration | `10464dab98fadfa5cf47df9fe218004e256b3826c3f3d3efa90a43c0f6f91eab` |

The 195 MB `landscape*.dat` files and the 2 GB `diffmap*.dat` are outputs, not
inputs; do not copy them.

## What the parameters set

`parameters.x_0.25.dat` defines: 20,000 excitons, a 500 x 500 diffmap over a
1500 nm box (3 nm/bin), `dt = 2.5e-7 ns`, `tsamp = 0.005 ns`, `tmax = 3.0 ns`
(601 frames), and the rate constants `D0_I = 0.0204`, `D0_Br = 0.0222 um^2/ns`,
`kT = 0.0255 eV`, `DeltaE = 0.674 eV`, `a_I = 1.25`, `a_Br = 0.8 nm`,
`alpha = 0.05845`, `rcut = 3.5 nm`. The run writes `variance.x_<frac>.dat`
(per-frame particle second moment, the carrier truth), `density.x_<frac>.dat`,
and `diffmap.x_<frac>.dat` (the 2 GB per-frame-normalised map).

## Runtime and strategy

Marc's 20,000-walker, 3 ns run takes about 13 hours on one cluster core. Walkers
are independent trajectories, so runtime scales linearly with `N`: a single
2-million-walker run would take roughly 1,300 hours.

The practical approach is to run many independent instances in parallel and
average their outputs. Each instance is a fresh random realisation (different
halide disorder and different initial sampling), so averaging `K` instances of
`N_inst` walkers gives both `K * N_inst` effective statistics and the
disorder-averaged density, which is the smooth quantity of interest. The
sampling noise of the average falls as `1 / sqrt(K * N_inst)`.

The total walker count follows from the contrast required. The intrinsic 1-D
peak contrast-to-noise is roughly `sqrt(N_total * f_peak)` with `f_peak ~ 0.006`
for this geometry, so:

- `N_total ~ 4e5` (e.g. 20 instances of 20,000) gives intrinsic peak-CNR ~ 50.
- `N_total ~ 1.6e6` (e.g. 80 instances) gives ~ 100, effectively clean across the
  whole CNR sweep.

At 13 h per 20,000-walker instance, 20 instances on a 20-core machine take one
night; 80 instances take several nights or a many-core machine.

Two parameter changes reduce the cost: set `diffmap_nx = diffmap_ny = 250` in
the parameter file (the analysis projects to 1-D, so 250 bins suffice, and the
2 GB output shrinks by a factor of four), and shorten `tmax` to the fitted
window.

## Steps

1. Install WSL2 + Ubuntu on the run machine (`wsl --install`), open the Ubuntu
   shell, and `cd` to a working directory on a disk with room for the outputs
   (each instance writes a diffmap; at 500 bins that is ~2 GB, at 250 bins
   ~0.5 GB).
2. Copy the four files above into that directory and `chmod +x landscapeBD.diffmap`.
3. **Verify randomisation first** (the parameter file has no seed): run two quick
   instances and confirm they differ.
   ```bash
   ./landscapeBD.diffmap parameters.x_0.25.dat && cp variance.x_0.25.dat v1.dat
   ./landscapeBD.diffmap parameters.x_0.25.dat && diff -q v1.dat variance.x_0.25.dat
   ```
   For the check, first edit the parameter file down to `N 200` and `tmax 0.05`
   so each finishes in seconds. If `diff` reports the files are identical, the
   binary uses a fixed seed and averaging adds no statistics. If they differ,
   restore `N` and `tmax` and continue.
4. Run the instances with `run_instances.sh` (see its header for arguments).
   The simulator seeds its random number generator from the wall clock at
   one-second resolution, and this script staggers the starts by 0 to 4 s, so
   instances that start in the same second are identical: at most five
   distinct realisations result, whatever the number of instances.
5. Top up to the number of distinct realisations wanted with
   `rerun_all_unique.sh` (all three fractions in one pool of workers) or
   `rerun_unique.sh` (one fraction). Each starts every new instance in its own
   clock second, counts an instance as distinct by the hash of its variance
   file, and writes the distinct instance directories to
   `unique_instances.txt`.
6. Average the distinct instances with `average_diffmaps.py`, which writes
   `diffmap.x_<frac>.avg.npy` and `variance.x_<frac>.avg.dat`
   (`rerun_all_unique.sh` runs this step itself).
7. Collapse the averaged map to 1-D with `collapse_diffmaps.py`, which writes
   `collapsed_1d.x_<frac>.npz`.
8. Copy back the collapsed file and the averaged variance file, the two files
   `experiments/halide-mixing-highn/br_highn_data.py` reads. The averaged map
   needs no de-noising. `br_data.py` carries the legacy 150 nm geometry and
   does not read these files.

## How the high-N files were made

The files that `experiments/halide-mixing-highn/` reads were made with the
steps above, with the parameter files changed to a 600 x 600 diffmap over a
3000 nm window (5 nm/bin) and 25,000 walkers per instance. A driver script,
`run_all_fractions.sh`, made those changes and ran steps 4 and 6 for the three
fractions; it reads its inputs from a fixed location of the group's storage and
is not included here. `rerun_all_unique.sh` is included unchanged, and
`rerun_unique.sh` with one comment reworded.

1. A first run of 16 instances per fraction gave only 5 distinct realisations
   per fraction (step 4); each other instance was an exact copy of one of the
   five. Its averaged maps are therefore duplication-weighted averages over 5
   realisations.
2. A top-up (step 5) added 11 instances per fraction, giving 16 distinct
   realisations of 25,000 walkers (400,000 walkers per fraction). These were
   averaged (step 6) and collapsed (step 7). Either top-up script produces this
   set; the record of which one ran is not kept.

Kept with the run's files: the parameter files, the 5 distinct realisations of
the first run, the first run's averaged maps, and the collapsed files and
averaged variance files of the 16-realisation set. Not kept: the 11 instances
added per fraction, the averaged 2-D maps of the 16-realisation set, the log of
the top-up, and the script that made the collapsed files. `collapse_diffmaps.py`
was written afterwards to describe that step.

`collapse_check.txt` holds the output of `collapse_diffmaps.py --check`: the
script's collapse of frames 0, 300 and 600 of the first run's averaged maps,
compared with the collapsed files. The grids and window metadata (x, t, window,
bins, fraction, sampling time) agree exactly; the profiles do not. The largest
absolute difference per frame is 1.0e-3 to 5.0e-3 (each profile sums to 1 over
600 bins), 0.013 to 0.229 of the profile's peak, the larger values at frames 300
and 600. The two sides are different averages (5 realisations,
duplication-weighted, against 16 distinct ones), so the collapsed files cannot
be reproduced from the maps that are kept, and the script is not checked
against them. At frames 300 and 600 the fine structure that the 5 shared
realisations leave in both averages matches `prof_x` to the sum over a frame's
rows and `prof_y` to the sum over its columns, the assignment the script uses.
The averaged variance files are not the first run's duplication-weighted mean
of the kept realisations' variance files, as expected for the 16-realisation
set.

## Checks on the run machine

- **Seed / randomisation** (step 3) determines whether averaging is valid.
  Step 3 detects a fixed seed but not two instances that started in the same
  second; check that all instances are distinct (the top-up scripts compare
  the hashes of their variance files).
- **Frame count:** the diffmap can hold a few more frame blocks than
  `variance.dat` has rows (605 vs 601 in the shared file). Take the frame count
  from `tmax / tsamp + 1` and check whether the diffmap carries a header or
  trailer before reshaping.
- **Box-edge clipping:** at 3 ns the spot can approach the 1500 nm box edge;
  check that the last-frame profile stays well inside before using late frames.
