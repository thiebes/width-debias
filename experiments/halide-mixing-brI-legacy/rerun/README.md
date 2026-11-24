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
5. Average them with `average_diffmaps.py`, which writes a clean
   `diffmap.x_<frac>.avg.npy` and `variance.x_<frac>.avg.dat`.
6. Copy the two averaged files back. The averaged map needs no de-noising. The
   loader must use the run's geometry (bin count, box size, frame count);
   `br_data.py` carries the legacy 150 nm geometry, and
   `experiments/halide-mixing-highn/br_highn_data.py` reads the averaged bed.

## Checks on the run machine

- **Seed / randomisation** (step 3) determines whether averaging is valid.
- **Frame count:** the diffmap can hold a few more frame blocks than
  `variance.dat` has rows (605 vs 601 in the shared file). Take the frame count
  from `tmax / tsamp + 1` and check whether the diffmap carries a header or
  trailer before reshaping.
- **Box-edge clipping:** at 3 ns the spot can approach the 1500 nm box edge;
  check that the last-frame profile stays well inside before using late frames.
