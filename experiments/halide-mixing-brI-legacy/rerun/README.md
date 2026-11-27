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
average their outputs. An instance with its own seed is a fresh random
realisation (different halide disorder and different initial sampling), so
averaging `K` distinct instances of `N_inst` walkers gives both `K * N_inst`
effective statistics and the disorder-averaged density, which is the smooth
quantity of interest. The sampling noise of the average falls as
`1 / sqrt(K * N_inst)`. The simulator takes its seed from the wall clock (step
4), so two instances that start in the same second are copies of one
realisation.

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
2. Make one folder per fraction, `x_<frac>/`, under a common root, and copy
   into each the four files above, `average_diffmaps.py` and
   `collapse_diffmaps.py`; then `chmod +x landscapeBD.diffmap`.
   `rerun_all_unique.sh` takes the root from the environment variable `ROOT`
   (default `~/halide_rerun`) and runs `average_diffmaps.py` from each fraction
   folder. Widen the parameter file's diffmap window as needed; the high-N run
   used `diffmap_Lx` and `diffmap_Ly` 3000 and `diffmap_nx` and `diffmap_ny`
   600 (5 nm/bin). The run scripts set the walker count `N` of each instance
   from their arguments. Steps 3, 4, 6 and 7, and `rerun_unique.sh`, run inside
   a fraction folder; `rerun_all_unique.sh` runs from anywhere.
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
4. Run the instances with `run_instances.sh` (see its header for arguments)
   in a fraction folder that holds no `inst_*` folders yet; the script refuses
   to reuse them. The simulator seeds its random number generator from the wall
   clock at one-second resolution, so instances that start in the same second
   are identical. The script therefore starts instance k after k times the gap
   (2 s by default), so in any wave each instance starts at least one gap after
   the one before it; only the time each takes to copy its files and start can
   still put two in one second. When all are done, the script keeps one
   instance per distinct complete variance file (compared by hash), writes
   them to `unique_instances.txt`, and reports how many there are and which
   instances are incomplete. If all are distinct, go to step 6; if not, it
   prints the `rerun_unique.sh` command for step 5.
5. Top up to the number of distinct realisations wanted with
   `rerun_all_unique.sh` (all three fractions in one pool of workers) or
   `rerun_unique.sh` (one fraction). Each starts every new instance in its own
   clock second, counts an instance as distinct by the hash of its variance
   file, and writes the distinct instance directories to
   `unique_instances.txt`. Both reuse the fraction folder's parameter file and
   keep the instances already there. Their logic is kept as run, so they do
   not check the walker count or the parameter file as `run_instances.sh`
   does. A walker count written as "20,000" goes into each instance's
   parameter file as given. Without tmax their expected row count is 1;
   without tsamp it is 0, so no instance ever counts as complete:
   `rerun_unique.sh` relaunches the full target for up to six waves, and
   `rerun_all_unique.sh` writes an empty `unique_instances.txt`. Neither
   stops when an instance or the averaging fails, and `rerun_all_unique.sh` prints "TOP-UP COMPLETE" even
   then. Before step 7, check the last `x_<frac>: <n> unique` line of each
   fraction, the "averaged <n> instances" line of step 6 (and any "skip" line
   before it), and that every listed instance's `variance.x_<frac>.dat` has
   `tmax / tsamp + 1` rows. `run.err` holds the simulator's set-up messages
   even after a successful run, so read it for an instance that is incomplete.
6. Average the distinct instances with
   `python3 average_diffmaps.py <frac> $(cat unique_instances.txt)`, which
   writes `diffmap.x_<frac>.avg.npy` and `variance.x_<frac>.avg.dat`
   (`rerun_all_unique.sh` runs this step itself).
7. Collapse the averaged map to 1-D with `python3 collapse_diffmaps.py <frac>`,
   which writes `collapsed_1d.x_<frac>.npz`. The script refuses to replace an
   existing profile file, to collapse a map whose frame count is not
   `tmax / tsamp + 1`, and to record instance counts the map does not hold:
   every instance in `unique_instances.txt` must be present with the same `N`
   and a full variance file, no two may have identical variance files (copies
   of one realisation), and the averaged variance file must be the mean of
   theirs.
8. Copy back the collapsed file and the averaged variance file, the two files
   `experiments/halide-mixing-highn/br_highn_data.py` reads. The averaged map
   needs no de-noising. `br_data.py` carries the legacy 150 nm geometry and
   does not read these files.

## How the high-N files were made

The files that `experiments/halide-mixing-highn/` reads were made with the
steps above, with the parameter files changed to a 600 x 600 diffmap over a
3000 nm window (5 nm/bin) and 25,000 walkers per instance. A driver script,
`run_all_fractions.sh`, made those changes, set up the fraction folders, ran
`run_instances.sh` for the three fractions, and averaged all 16 instances of
each fraction, duplicates included, with `average_diffmaps.py`. The driver
reads its inputs from a fixed location of the group's storage and is not
included here. The `run_instances.sh` it ran staggered the starts by a random
0 to 4 s and listed no distinct instances. The copy kept with the run's files
differs from the repository's version of that time in one message and in
exporting the walker count to the instances, which the repository's version
did not do: run as committed, unless the caller had exported the walker count,
it wrote an `N` line with no value into each instance's parameter file. The first run's instances carry N = 25,000.
`rerun_all_unique.sh` is included unchanged, and `rerun_unique.sh` with one
comment reworded; the headers of both describe that earlier `run_instances.sh`.

1. A first run of 16 instances per fraction, all started together, gave only 5
   distinct realisations per fraction: with a random 0 to 4 s stagger the
   instances started in at most five different seconds, and each other instance
   was an exact copy of one of the five. Its averaged maps are therefore
   duplication-weighted averages over 5 realisations.
2. A top-up (step 5) added 11 instances per fraction, giving 16 distinct
   realisations of 25,000 walkers (400,000 walkers per fraction). These were
   averaged (step 6) and collapsed to 1-D by a script that is not kept. Either
   top-up script produces this set; the record of which one ran is not kept.

Kept with the run's files: the parameter files, the 5 distinct realisations of
the first run (its 11 duplicate instances are not kept), the first run's
averaged maps, and the collapsed files and averaged variance files of the
16-realisation set. Not kept: the 11 instances added per fraction, the averaged
2-D maps of the 16-realisation set, the log of the top-up, and the script that
made the collapsed files. `collapse_diffmaps.py` was written afterwards to
describe that step.

`collapse_check.txt` holds the output of `collapse_diffmaps.py --check` for
frames 0, 300 and 600 of each fraction. It compares the script's collapse of the
first run's averaged maps with the collapsed files, and the averaged variance
files with the kept realisations' variance files.

- The key order, the grids and the window metadata (x, t, window, bins, frame
  count, fraction, sampling time) agree exactly.
- The profiles do not. The largest absolute difference per frame is 1.0e-3 to
  5.0e-3 (each profile sums to 1 over 600 bins): 0.013 to 0.021 of the
  profile's peak at frame 0, and 0.105 to 0.229 at frames 300 and 600. The two
  sides are different averages (5 realisations, duplication-weighted, against 16
  distinct ones), so the collapsed files cannot be reproduced from the maps that
  are kept, and the script's profiles cannot be verified against them.
- The 5 shared realisations leave fine structure (second differences) in both
  averages. At frames 300 and 600 the script's `prof_x` correlates with the
  file's `prof_x` (r 0.54 to 0.70) more than with its `prof_y` (r -0.14 to
  0.21), and the script's `prof_y` with the file's `prof_y` (r 0.45 to 0.61)
  more than with its `prof_x` (r 0.03 to 0.44), in every case. So the file's
  `prof_x` sums over a frame's rows and its `prof_y` over the columns, the
  assignment the script uses. At frame 0 the profiles are narrow and the
  correlations do not separate the axes on every fraction.
- The averaged variance files are not the first run's kind of average, a
  duplication-weighted mean of the kept realisations' variance files. Copies
  have identical files, so if they were, the right weighting would reproduce
  them to rounding. Of the 1365 weightings (each of the 5 kept realisations
  counted at least once, 16 in all), the closest still differs by 61 to
  121 nm² (by fraction) in its worst frame.
- They are consistent with a mean of 16 distinct realisations that include the
  5 kept ones, which the top-up scripts keep when they add instances. Per frame,
  z is the difference between the averaged file and the kept realisations'
  mean, divided by its standard error estimated from their spread; the largest
  |z| over the 601 frames is 1.86 to 3.79. This alone does not single out the
  16-realisation set: the frames are strongly correlated and the spread comes
  from 5 realisations, so a largest |z| of this size is ordinary, and a
  duplication-weighted mean can also give a small |z|. The weighting search
  above is what separates the two.

## Checks on the run machine

- **Seed / randomisation** (step 3) determines whether averaging is valid.
  Step 3 detects a fixed seed but not two instances that started in the same
  second; check that all instances are distinct (`run_instances.sh` and the
  top-up scripts compare the hashes of their variance files and list the
  distinct ones in `unique_instances.txt`).
- **Frame count:** the diffmap can hold a few more frame blocks than
  `variance.dat` has rows (605 vs 601 in the shared file). Take the frame count
  from `tmax / tsamp + 1` and check whether the diffmap carries a header or
  trailer before reshaping. `collapse_diffmaps.py` refuses a map with another
  frame count.
- **Box-edge clipping:** at 3 ns the spot can approach the 1500 nm box edge;
  check that the last-frame profile stays well inside before using late frames.
