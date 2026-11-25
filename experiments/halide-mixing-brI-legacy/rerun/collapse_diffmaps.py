#!/usr/bin/env python3
"""Collapse an averaged 2-D diffusion map into the 1-D profile file the high-N analysis reads.

The high-N bed is analysed in one dimension (``br_highn_data.py`` reads
``prof_x``), so each fraction's averaged map, tmax / tsamp + 1 frames of n x n
float32 bins (601 frames of 600 x 600, about 865 MB, in the high-N run), is
collapsed once into a small file. The map is normalised per frame, so each
projection already sums to 1 and is not renormalised. A frame's rows are y and
its columns x, so ``prof_x`` sums over the rows, as ``br_data.py`` does for this
simulator's maps. The map is memory-mapped and summed one frame at a time, so
memory stays at about one frame.

The x grid is ``linspace(-L/2, L/2, n)`` for a window L and n bins, the
convention of ``br_data.py`` (and of the diffmap loaders ``nc_data.py`` and
``marc_data.py``); its spacing L/(n - 1) is slightly wider than the bin width L/n.

The script that made the existing profile files is not kept. This one was
written afterwards to match their grids, metadata and key order; its profiles
are not checked against theirs, because the averaged maps that are kept are a
different average (``collapse_check.txt`` and ``README.md``).

Before writing, the script refuses to
- replace an existing profile file, since those files cannot be re-derived;
- collapse a map whose frame count is not tmax / tsamp + 1, the variance file's
  row count, so that every profile pairs with one variance row;
- record instance counts the map does not hold: every instance listed in
  unique_instances.txt must be present with the same walker count, and
  variance.x_<frac>.avg.dat must be the mean of their variance files.
  average_diffmaps.py writes that file and the map together from one list, so a
  map averaged from another list (all instances of a run, duplicates included)
  fails here.

Usage:
    python3 collapse_diffmaps.py <frac> [<dir>]
    python3 collapse_diffmaps.py <frac> <dir> --check <frame> [<frame> ...]
Reads (from <dir>, default the current directory):
    diffmap.x_<frac>.avg.npy   -- the average written by average_diffmaps.py
    variance.x_<frac>.avg.dat  -- the averaged variance written with it
    parameters.x_<frac>.dat    -- diffmap window and bins, sampling and simulated
                                  time, Br fraction
    unique_instances.txt       -- the averaged instance directories, one per line,
                                  as rerun_unique.sh and rerun_all_unique.sh write
                                  it; each holds its parameter and variance files
Writes (in the current directory):
    collapsed_1d.x_<frac>.npz  -- x (nm), t (ns), prof_x and prof_y (frames x bins),
                                  and box_nm, n_bins, n_frames, n_unique,
                                  n_per_instance, n_walkers, x_frac, tsamp_ns
With --check nothing is written. The listed frames are collapsed by the same code
and compared with <dir>/collapsed_1d.x_<frac>.npz: the key order, the grid and
window metadata, and for each frame and profile whether they are equal, the
largest absolute difference (also as a fraction of the existing profile's peak),
and the correlation of the fine structure (second differences) with each of the
existing file's two profiles, which shows which axis each existing profile sums
over. If <dir> holds instance directories (inst_*), the averaged variance file is
also compared with their variance files (``compare_variance``).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

KEYS = ("x", "t", "prof_x", "prof_y", "box_nm", "n_bins", "n_frames", "n_unique",
        "n_per_instance", "n_walkers", "x_frac", "tsamp_ns")
COUNT_KEYS = ("n_unique", "n_per_instance", "n_walkers")
GEOMETRY_PARAMS = ("diffmap_Lx", "diffmap_Ly", "diffmap_nx", "diffmap_ny", "tsamp", "tmax", "x")


class Geometry(NamedTuple):
    """A square diffmap's window and time grid, from its parameter file."""

    box_nm: float
    n_bins: int
    n_frames: int
    tsamp_ns: float
    x_frac: float


def read_params(path: Path) -> dict[str, str]:
    """Map each parameter name to its first value (the comment after it is ignored)."""
    params: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            params.setdefault(parts[0], parts[1])
    return params


def read_geometry(path: Path) -> Geometry:
    """The window, bins, frame count, sampling time and Br fraction of a square diffmap."""
    params = read_params(path)
    missing = [key for key in GEOMETRY_PARAMS if key not in params]
    if missing:
        raise ValueError(f"{path.name} has no {', '.join(missing)}")
    box_nm = float(params["diffmap_Lx"])   # the parameter file labels it um; it is nm
    n_bins = int(params["diffmap_nx"])
    if (float(params["diffmap_Ly"]), int(params["diffmap_ny"])) != (box_nm, n_bins):
        raise ValueError(f"{path.name}: the diffmap is not square; the collapse assumes one "
                         "x and y grid")
    tsamp_ns = float(params["tsamp"])
    n_frames = round(float(params["tmax"]) / tsamp_ns) + 1
    return Geometry(box_nm, n_bins, n_frames, tsamp_ns, float(params["x"]))


def read_walkers(path: Path) -> int:
    """The walker count N of one instance's parameter file."""
    params = read_params(path)
    if "N" not in params:
        raise ValueError(f"{path} has no N")
    return int(params["N"])


def project(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(prof_x, prof_y) of one frame whose rows are y and columns x, summed in float64."""
    frame = np.asarray(frame, dtype=np.float64)
    return frame.sum(axis=0), frame.sum(axis=1)


def collapse(cube: np.ndarray, geom: Geometry, frames: list[int] | None = None,
             n_unique: int | None = None,
             n_per_instance: int | None = None) -> dict[str, np.ndarray | float | int]:
    """The contents of ``collapsed_1d.x_<frac>.npz`` in KEYS order, for ``frames`` (default all).

    The instance counts are left out when they are not given: the check compares
    the rest with an existing file and cannot know which instances made it.
    """
    expected_shape = (geom.n_frames, geom.n_bins, geom.n_bins)
    if cube.shape != expected_shape:
        raise ValueError(f"the map is {cube.shape}; the parameter file gives {expected_shape} "
                         "(tmax / tsamp + 1 frames of n_bins x n_bins)")
    frames = list(range(geom.n_frames)) if frames is None else list(frames)
    prof_x = np.empty((len(frames), geom.n_bins))
    prof_y = np.empty((len(frames), geom.n_bins))
    for row, k in enumerate(frames):
        prof_x[row], prof_y[row] = project(cube[k])
    counts_given = n_unique is not None and n_per_instance is not None
    contents = {
        "x": np.linspace(-geom.box_nm / 2.0, geom.box_nm / 2.0, geom.n_bins),
        "t": np.array(frames) * geom.tsamp_ns,
        "prof_x": prof_x,
        "prof_y": prof_y,
        "box_nm": geom.box_nm,
        "n_bins": geom.n_bins,
        "n_frames": geom.n_frames,
        "n_unique": n_unique,
        "n_per_instance": n_per_instance,
        "n_walkers": n_unique * n_per_instance if counts_given else None,
        "x_frac": geom.x_frac,
        "tsamp_ns": geom.tsamp_ns,
    }
    return {key: contents[key] for key in KEYS if counts_given or key not in COUNT_KEYS}


def averaged_instances(frac: str, src: Path, geom: Geometry) -> tuple[int, int]:
    """(instances, walkers per instance) of the average in ``src``, checked against it.

    unique_instances.txt names the instances. The counts are returned only if
    every listed instance is present with the same walker count, and the averaged
    variance file has one row per frame and is the mean of their variance files.
    """
    listed = (src / "unique_instances.txt").read_text().split()
    if not listed:
        raise ValueError("unique_instances.txt lists no instances")
    missing = [d for d in listed if not (src / d / f"variance.x_{frac}.dat").is_file()]
    if missing:
        raise ValueError(f"{len(missing)} instance(s) in unique_instances.txt have no variance "
                         f"file, for example {missing[0]}")
    walker_counts = {read_walkers(src / d / f"parameters.x_{frac}.dat") for d in listed}
    if len(walker_counts) != 1:
        raise ValueError(f"the listed instances differ in walker count: {sorted(walker_counts)}")
    averaged = np.loadtxt(src / f"variance.x_{frac}.avg.dat")
    if averaged.shape[0] != geom.n_frames:
        raise ValueError(f"variance.x_{frac}.avg.dat has {averaged.shape[0]} rows; the map has "
                         f"{geom.n_frames} frames")
    mean = np.mean([np.loadtxt(src / d / f"variance.x_{frac}.dat") for d in listed], axis=0)
    if mean.shape != averaged.shape or not np.allclose(averaged, mean, rtol=1e-12, atol=0.0):
        raise ValueError(f"variance.x_{frac}.avg.dat is not the mean of the {len(listed)} listed "
                         "instances' variance files, so the map is not their average")
    return len(listed), walker_counts.pop()


def compare_variance(frac: str, src: Path, geom: Geometry, n_total: int) -> str:
    """How variance.x_<frac>.avg.dat compares with the variance files of the instances in ``src``.

    The kept instances are some of the run's realisations. Two readings of the
    averaged file are tested, on its variance column (nm^2):
    - a mean of ``n_total`` distinct realisations that include the kept ones. It
      then differs from the kept instances' mean only through the others, so per
      frame z = (averaged - kept mean) / SE, with SE from the kept instances'
      spread: (m / n_total) * sd * sqrt(1/m + 1/K) for K kept and m = n_total - K
      others. The largest |z| over the frames is reported.
    - a duplication-weighted mean of the kept instances alone, each counted a
      whole number of times, ``n_total`` in all, as when instances repeat a seed.
      Such a mean is computed from the same values, so the right weighting would
      reproduce the file to rounding. Every weighting is tried, and the smallest
      largest-difference over the frames is reported.
    The SE is estimated from K instances (K - 1 degrees of freedom), so |z| has
    Student's t tails, not normal ones.
    """
    files = sorted(src.glob(f"inst_*/variance.x_{frac}.dat"))
    columns = [np.loadtxt(f)[:, 1] for f in files]
    kept = np.array([c for c in columns if c.shape[0] == geom.n_frames])
    n_kept = len(kept)
    if n_kept < 2 or n_kept >= n_total:
        return f"  variance: {n_kept} complete kept instance(s), no comparison"
    averaged = np.loadtxt(src / f"variance.x_{frac}.avg.dat")[:, 1]
    n_other = n_total - n_kept
    se = (n_other / n_total) * kept.std(axis=0, ddof=1) * np.sqrt(1 / n_other + 1 / n_kept)
    spread = se > 0
    z_max = np.abs((averaged - kept.mean(axis=0))[spread] / se[spread]).max()
    cuts = np.array(list(itertools.combinations(range(1, n_total), n_kept - 1)))
    weights = np.diff(np.column_stack([np.zeros(len(cuts), int), cuts,
                                       np.full(len(cuts), n_total)]), axis=1)
    best_miss = np.abs(weights @ kept / n_total - averaged).max(axis=1).min()
    names = " ".join(f.parent.name for f in files)
    return (f"  variance.x_{frac}.avg.dat against the {n_kept} kept instances ({names}):\n"
            f"    as {n_total} distinct realisations including them: largest |z| {z_max:.2f} "
            f"over {spread.sum()} frames\n"
            f"    as a duplication-weighted mean of them ({len(weights)} weightings summing to "
            f"{n_total}): every weighting differs by at least {best_miss:.0f} nm^2 in some frame")


def check(frac: str, src: Path, cube: np.ndarray, geom: Geometry, frames: list[int]) -> None:
    """Print how the collapse of ``frames`` compares with the existing profile file in ``src``."""
    name = f"collapsed_1d.x_{frac}.npz"
    existing = np.load(src / name, allow_pickle=False)
    collapsed = collapse(cube, geom, frames)
    print(f"x_{frac}: frames {' '.join(map(str, frames))} of diffmap.x_{frac}.avg.npy collapsed "
          f"by collapse_diffmaps.py, against {name} (values: script / file)")
    print(f"  keys in the order the script writes them: {tuple(existing.files) == KEYS}")
    window = "; ".join(f"{key} {collapsed[key]:g} / {existing[key].item():g}"
                       for key in ("box_nm", "n_bins", "n_frames", "x_frac", "tsamp_ns"))
    print(f"  x equal {np.array_equal(collapsed['x'], existing['x'])}; t equal at the frames "
          f"{np.array_equal(collapsed['t'], existing['t'][frames])}; {window}")
    print("  instance counts in the file (not checkable from the map): "
          + ", ".join(f"{key} {existing[key].item()}" for key in COUNT_KEYS))
    for row, k in enumerate(frames):
        for prof in ("prof_x", "prof_y"):
            ours, theirs = collapsed[prof][row], existing[prof][k]
            diff = np.abs(ours - theirs).max()
            with np.errstate(invalid="ignore"):    # nan when a profile has no fine structure
                fine = [np.corrcoef(np.diff(ours, 2), np.diff(existing[other][k], 2))[0, 1]
                        for other in ("prof_x", "prof_y")]
            print(f"  frame {k} {prof}: equal {np.array_equal(ours, theirs)}, max |diff| "
                  f"{diff:.2e} ({diff / np.abs(theirs).max():.3f} of peak); fine structure r "
                  f"{fine[0]:+.2f} with the file's prof_x, {fine[1]:+.2f} with its prof_y")
    if any(src.glob(f"inst_*/variance.x_{frac}.dat")):
        print(compare_variance(frac, src, geom, int(existing["n_unique"])))


def main() -> None:
    args = sys.argv[1:]
    frames = None
    if "--check" in args:
        i = args.index("--check")
        frame_args, args = args[i + 1:], args[:i]
        if len(args) != 2 or not frame_args or not all(a.isdigit() for a in frame_args):
            sys.exit("usage: collapse_diffmaps.py <frac> <dir> --check <frame> [<frame> ...]")
        frames = [int(a) for a in frame_args]
    if len(args) not in (1, 2):
        sys.exit("usage: collapse_diffmaps.py <frac> [<dir>]")
    frac = args[0]
    src = Path(args[1]) if len(args) == 2 else Path(".")

    geom = read_geometry(src / f"parameters.x_{frac}.dat")
    cube = np.load(src / f"diffmap.x_{frac}.avg.npy", mmap_mode="r")
    if frames is not None:
        outside = [k for k in frames if k >= geom.n_frames]
        if outside:
            sys.exit(f"frames {outside} are beyond the last frame, {geom.n_frames - 1}")
        check(frac, src, cube, geom, frames)
        return

    out_path = Path(f"collapsed_1d.x_{frac}.npz")
    if out_path.exists():
        sys.exit(f"{out_path} exists; this script never replaces a profile file")
    n_unique, n_per_instance = averaged_instances(frac, src, geom)
    contents = collapse(cube, geom, n_unique=n_unique, n_per_instance=n_per_instance)
    with open(out_path, "xb") as fh:
        np.savez_compressed(fh, **contents)
    print(f"collapsed {geom.n_frames} frames of {geom.n_bins} x {geom.n_bins} bins "
          f"({n_unique} instances of {n_per_instance}) -> {out_path}")


if __name__ == "__main__":
    main()
