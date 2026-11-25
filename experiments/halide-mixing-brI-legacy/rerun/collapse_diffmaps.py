#!/usr/bin/env python3
"""Collapse an averaged 2-D diffusion map into the 1-D profiles the analysis reads.

The high-N bed is analysed in one dimension (``br_highn_data.py`` reads only the x
and y projections), so each fraction's averaged map, 601 frames of 600 x 600
float32 bins (about 865 MB), is collapsed once into a small file. The map is
normalised per frame, so each projection already sums to 1 and is not
renormalised. A frame's rows are y and its columns x, so ``prof_x`` sums over the
rows. The map is memory-mapped and summed one frame at a time, so memory stays at
about one frame.

The x grid is ``linspace(-L/2, L/2, n)`` for a window L and n bins, the
convention of the other loaders of this simulator's maps (``br_data.py``,
``nc_data.py``); its spacing L/(n - 1) is slightly wider than the bin width L/n.

Usage:
    python3 collapse_diffmaps.py <frac> [<dir>]
    python3 collapse_diffmaps.py <frac> <dir> --check <frame> [<frame> ...]
Reads (from <dir>, default the current directory):
    diffmap.x_<frac>.avg.npy   -- the average written by average_diffmaps.py
    parameters.x_<frac>.dat    -- diffmap window and bins, sampling time, Br fraction
    unique_instances.txt       -- the averaged instance directories, one per line,
                                  as rerun_unique.sh and rerun_all_unique.sh write
                                  it; the walkers per instance are read from the
                                  first one's parameter file
Writes (in the current directory):
    collapsed_1d.x_<frac>.npz  -- x (nm), t (ns), prof_x and prof_y (frames x bins),
                                  and box_nm, n_bins, n_frames, n_unique,
                                  n_per_instance, n_walkers, x_frac, tsamp_ns
With --check nothing is written: the listed frames are collapsed and compared with
<dir>/collapsed_1d.x_<frac>.npz. The grids and window metadata are compared, and
for each frame and profile the script prints whether they are equal and the
largest absolute difference, also as a fraction of the existing profile's peak.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def read_params(path: Path) -> dict[str, str]:
    """Map each parameter name to its value (the comment after it is ignored)."""
    params = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and not parts[0].startswith("#"):
            params[parts[0]] = parts[1]
    return params


def window(params: dict[str, str]) -> tuple[float, int]:
    """(window width in nm, bins per side) of a square diffmap."""
    box_nm = float(params["diffmap_Lx"])   # the parameter file labels it um; it is nm
    n_bins = int(params["diffmap_nx"])
    if (params["diffmap_Ly"], params["diffmap_ny"]) != (params["diffmap_Lx"], params["diffmap_nx"]):
        sys.exit("the diffmap is not square; the collapse assumes one x and y grid")
    return box_nm, n_bins


def project(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(prof_x, prof_y) of one frame whose rows are y and columns x, summed in float64."""
    frame = np.asarray(frame, dtype=np.float64)
    return frame.sum(axis=0), frame.sum(axis=1)


def collapse(cube: np.ndarray, params: dict[str, str], n_unique: int,
             n_per_instance: int) -> dict[str, np.ndarray | float | int]:
    """The arrays and metadata of ``collapsed_1d.x_<frac>.npz``, in the file's order."""
    box_nm, n_bins = window(params)
    n_frames = cube.shape[0]
    if cube.shape[1:] != (n_bins, n_bins):
        sys.exit(f"map frames are {cube.shape[1:]}, the parameter file gives {n_bins} bins")
    prof_x = np.empty((n_frames, n_bins))
    prof_y = np.empty((n_frames, n_bins))
    for k in range(n_frames):
        prof_x[k], prof_y[k] = project(cube[k])
    tsamp_ns = float(params["tsamp"])
    return {
        "x": np.linspace(-box_nm / 2.0, box_nm / 2.0, n_bins),
        "t": np.arange(n_frames) * tsamp_ns,
        "prof_x": prof_x,
        "prof_y": prof_y,
        "box_nm": box_nm,
        "n_bins": n_bins,
        "n_frames": n_frames,
        "n_unique": n_unique,
        "n_per_instance": n_per_instance,
        "n_walkers": n_unique * n_per_instance,
        "x_frac": float(params["x"]),
        "tsamp_ns": tsamp_ns,
    }


def check(frac: str, src: Path, cube: np.ndarray, params: dict[str, str],
          frames: list[int]) -> None:
    """Compare the collapse of ``frames`` with the existing collapsed file in ``src``."""
    old = np.load(src / f"collapsed_1d.x_{frac}.npz", allow_pickle=False)
    box_nm, n_bins = window(params)
    new_x = np.linspace(-box_nm / 2.0, box_nm / 2.0, n_bins)
    new_t = np.array(frames) * float(params["tsamp"])
    print(f"x_{frac}: x equal {np.array_equal(new_x, old['x'])}; "
          f"t equal at the frames {np.array_equal(new_t, old['t'][frames])}; "
          f"box_nm {box_nm:g} / {float(old['box_nm']):g}; n_bins {n_bins} / {int(old['n_bins'])}; "
          f"x_frac {float(params['x']):g} / {float(old['x_frac']):g}; "
          f"tsamp_ns {float(params['tsamp']):g} / {float(old['tsamp_ns']):g}")
    for k in frames:
        cells = []
        for name, new in zip(("prof_x", "prof_y"), project(cube[k])):
            ref = old[name][k]
            diff = np.abs(new - ref).max()
            cells.append(f"{name} equal {np.array_equal(new, ref)}, max |diff| {diff:.2e} "
                         f"({diff / np.abs(ref).max():.3f} of peak)")
        print(f"  frame {k}: " + "; ".join(cells))


def main() -> None:
    args = sys.argv[1:]
    frames = None
    if "--check" in args:
        i = args.index("--check")
        frames = [int(a) for a in args[i + 1:]]
        args = args[:i]
        if len(args) != 2 or not frames:
            sys.exit("usage: collapse_diffmaps.py <frac> <dir> --check <frame> [<frame> ...]")
    if len(args) not in (1, 2):
        sys.exit("usage: collapse_diffmaps.py <frac> [<dir>]")
    frac = args[0]
    src = Path(args[1]) if len(args) == 2 else Path(".")

    params = read_params(src / f"parameters.x_{frac}.dat")
    cube = np.load(src / f"diffmap.x_{frac}.avg.npy", mmap_mode="r")
    if frames is not None:
        check(frac, src, cube, params, frames)
        return

    instances = (src / "unique_instances.txt").read_text().split()
    if not instances:
        sys.exit("unique_instances.txt lists no instances")
    n_per_instance = int(read_params(src / instances[0] / f"parameters.x_{frac}.dat")["N"])
    out = collapse(cube, params, len(instances), n_per_instance)
    np.savez_compressed(f"collapsed_1d.x_{frac}.npz", **out)
    print(f"collapsed {out['n_frames']} frames of {out['n_bins']} x {out['n_bins']} bins "
          f"({len(instances)} instances of {n_per_instance}) -> collapsed_1d.x_{frac}.npz")


if __name__ == "__main__":
    main()
