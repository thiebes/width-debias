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
written afterwards to match their grids, metadata and key order. Its profiles
cannot be verified against theirs, because the averaged maps that are kept are a
different average (``collapse_check.txt`` and ``README.md``).

The script refuses to
- replace an existing profile file, since those files cannot be re-derived;
- collapse a map whose frame count is not tmax / tsamp + 1, the variance file's
  row count, so that every profile pairs with one variance row (in --check too);
- record instance counts the map does not hold. Every instance listed in
  unique_instances.txt must be present with the same walker count and a full
  variance file; no two may have identical variance files, the rule by which the
  top-up scripts count distinct realisations; and variance.x_<frac>.avg.dat must
  be the mean of their variance files. average_diffmaps.py writes that file and
  the map together from one list, so a map averaged from another list fails.

Usage:
    python3 collapse_diffmaps.py <frac> [<dir>]
    python3 collapse_diffmaps.py <frac> <dir> --check <frame> [<frame> ...]
Reads (from <dir>, default the current directory):
    diffmap.x_<frac>.avg.npy   -- the average written by average_diffmaps.py
    variance.x_<frac>.avg.dat  -- the averaged variance written with it
    parameters.x_<frac>.dat    -- diffmap window and bins, sampling and simulated
                                  time, Br fraction
    unique_instances.txt       -- the averaged instance directories, one per line,
                                  as run_instances.sh, rerun_unique.sh and
                                  rerun_all_unique.sh write it; each holds its
                                  parameter and variance files
Writes (in the current directory):
    collapsed_1d.x_<frac>.npz  -- x (nm), t (ns), prof_x and prof_y (frames x bins),
                                  and box_nm, n_bins, n_frames, n_unique,
                                  n_per_instance, n_walkers, x_frac, tsamp_ns
With --check nothing is written. The listed frames are collapsed by the same code
and compared with <dir>/collapsed_1d.x_<frac>.npz: the key order, the grid and
window metadata, and for each frame and profile whether they are equal, the
largest absolute difference (also as a fraction of the existing profile's peak),
and the fine-structure correlation with each of the existing file's two profiles
(``fine_structure_r``), which shows which axis each existing profile sums over.
If <dir> holds instance directories (inst_*), the averaged variance file is also
compared with their variance files (``compare_variance``).
"""
from __future__ import annotations

import hashlib
import itertools
import math
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

KEYS = ("x", "t", "prof_x", "prof_y", "box_nm", "n_bins", "n_frames", "n_unique",
        "n_per_instance", "n_walkers", "x_frac", "tsamp_ns")
COUNT_KEYS = ("n_unique", "n_per_instance", "n_walkers")
WINDOW_KEYS = ("box_nm", "n_bins", "n_frames", "x_frac", "tsamp_ns")
GEOMETRY_PARAMS = ("diffmap_Lx", "diffmap_Ly", "diffmap_nx", "diffmap_ny", "tsamp", "tmax", "x")
MAX_WEIGHTINGS = 2_000_000   # above this the duplication-weighting search is skipped


class Geometry(NamedTuple):
    """A square diffmap's window, bins, frame count, sampling time and Br fraction."""

    box_nm: float
    n_bins: int
    n_frames: int
    tsamp_ns: float
    x_frac: float


class VarianceComparison(NamedTuple):
    """The numbers ``compare_variance`` reports (variances in nm^2)."""

    instances: tuple[str, ...]
    n_total: int
    n_weightings: int
    best_miss_nm2: float | None     # None when the search is skipped
    z_max: float | None             # None when no other realisation is left
    n_frames_z: int


def read_params(path: Path) -> dict[str, str]:
    """Map each parameter name to its first value (the comment after it is ignored)."""
    params: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            params.setdefault(parts[0], parts[1])
    return params


def read_geometry(path: Path) -> Geometry:
    """The geometry of a square diffmap, from its parameter file."""
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

    The instance counts are not determined by the map, so they are included only
    when both are given.
    """
    if (n_unique is None) != (n_per_instance is None):
        raise ValueError("give both n_unique and n_per_instance, or neither")
    expected_shape = (geom.n_frames, geom.n_bins, geom.n_bins)
    if cube.shape != expected_shape:
        raise ValueError(f"the map is {cube.shape}; the parameter file gives {expected_shape} "
                         "(tmax / tsamp + 1 frames of n_bins x n_bins)")
    frames = list(range(geom.n_frames)) if frames is None else list(frames)
    prof_x = np.empty((len(frames), geom.n_bins))
    prof_y = np.empty((len(frames), geom.n_bins))
    for row, k in enumerate(frames):
        prof_x[row], prof_y[row] = project(cube[k])
    counts_given = n_unique is not None
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
    every listed instance is present with the same walker count and a full
    variance file, no two variance files are identical, and the averaged variance
    file is the mean of the listed ones.
    """
    listed = (src / "unique_instances.txt").read_text().split()
    if not listed:
        raise ValueError("unique_instances.txt lists no instances")
    files = [src / d / f"variance.x_{frac}.dat" for d in listed]
    missing = [d for d, f in zip(listed, files) if not f.is_file()]
    if missing:
        raise ValueError(f"{len(missing)} instance(s) in unique_instances.txt have no variance "
                         f"file, for example {missing[0]}")
    walker_counts = {read_walkers(src / d / f"parameters.x_{frac}.dat") for d in listed}
    if len(walker_counts) != 1:
        raise ValueError(f"the listed instances differ in walker count: {sorted(walker_counts)}")
    digests = [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
    if len(set(digests)) != len(listed):
        raise ValueError("unique_instances.txt lists copies of one realisation (identical "
                         "variance files), so their number is not the number of realisations")
    variances = [np.loadtxt(f) for f in files]
    short = [d for d, v in zip(listed, variances) if v.shape[0] != geom.n_frames]
    if short:
        raise ValueError(f"{short[0]}'s variance file does not have {geom.n_frames} rows "
                         "(tmax / tsamp + 1)")
    averaged = np.loadtxt(src / f"variance.x_{frac}.avg.dat")
    if averaged.shape[0] != geom.n_frames:
        raise ValueError(f"variance.x_{frac}.avg.dat has {averaged.shape[0]} rows; the map has "
                         f"{geom.n_frames} frames")
    mean = np.mean(variances, axis=0)
    # average_diffmaps.py writes the mean with 18 significant digits; only the order of
    # summation can differ, which moves non-negative values by about 1e-16 relative.
    if mean.shape != averaged.shape or not np.allclose(averaged, mean, rtol=1e-12, atol=0.0):
        raise ValueError(f"variance.x_{frac}.avg.dat is not the mean of the {len(listed)} listed "
                         "instances' variance files, so the map is not their average")
    return len(listed), walker_counts.pop()


def positive_compositions(total: int, parts: int) -> np.ndarray:
    """Every way to write ``total`` as ``parts`` whole numbers of at least 1, one per row.

    Each row comes from choosing parts - 1 cut points among 1 .. total - 1.
    """
    cuts = np.array(list(itertools.combinations(range(1, total), parts - 1)), dtype=int)
    cuts = cuts.reshape(-1, parts - 1)
    edges = np.column_stack([np.zeros(len(cuts), int), cuts, np.full(len(cuts), total)])
    return np.diff(edges, axis=1)


def compare_variance(frac: str, src: Path, geom: Geometry, n_total: int) -> VarianceComparison | None:
    """Test where variance.x_<frac>.avg.dat comes from, against the kept instances in ``src``.

    The question is whether the averaged file is the first run's average, in which
    instances that shared a seed were copies, or the mean of ``n_total`` distinct
    realisations. Only the variance column (nm^2) is used, and only instances with
    a full variance file. ``n_total`` is the realisation count the profile file
    records; the first run is taken to have had as many instances.

    - Duplication-weighted mean: each kept instance counted a whole number of
      times, at least once (each was in the first run), ``n_total`` in all.
      Copies have identical files, so the right weighting would reproduce the
      averaged file to rounding. For each weighting the largest per-frame
      difference is found; the smallest of these is reported.
    - Mean of ``n_total`` distinct realisations that include the K kept ones: it
      then differs from the kept mean only through the m = n_total - K others.
      Per frame, z = (averaged - kept mean) / SE with
      SE = (m / n_total) * sd * sqrt(1/m + 1/K), where sd is the kept instances'
      sample standard deviation, taken to be the others' spread as well. With
      sd from K instances, z has Student's t tails with K - 1 degrees of freedom.
      The largest |z| over the frames is reported. Frames are strongly
      correlated, and a duplication-weighted mean can also give a small |z|, so
      this reading is a consistency check; the weighting search is what
      separates the two.
    Returns None when fewer than two complete instances are kept.
    """
    files = sorted(src.glob(f"inst_*/variance.x_{frac}.dat"))
    pairs = [(f.parent.name, np.loadtxt(f)[:, 1]) for f in files]
    pairs = [(name, v) for name, v in pairs if v.shape[0] == geom.n_frames]
    if len(pairs) < 2 or len(pairs) > n_total:
        return None
    names = tuple(name for name, _ in pairs)
    kept_variances = np.array([v for _, v in pairs])
    n_kept = len(kept_variances)
    averaged = np.loadtxt(src / f"variance.x_{frac}.avg.dat")[:, 1]

    n_weightings = math.comb(n_total - 1, n_kept - 1)
    best_miss = None
    if n_weightings <= MAX_WEIGHTINGS:
        weights = positive_compositions(n_total, n_kept)
        misses = np.abs(weights @ kept_variances / n_total - averaged).max(axis=1)
        best_miss = float(misses.min())

    z_max, n_frames_z = None, 0
    n_other = n_total - n_kept
    if n_other > 0:
        sd = kept_variances.std(axis=0, ddof=1)
        se = (n_other / n_total) * sd * np.sqrt(1 / n_other + 1 / n_kept)
        spread = se > 0
        z = (averaged - kept_variances.mean(axis=0))[spread] / se[spread]
        z_max, n_frames_z = float(np.abs(z).max()), int(spread.sum())
    return VarianceComparison(names, n_total, n_weightings, best_miss, z_max, n_frames_z)


def fine_structure_r(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of the second differences of two profiles (nan if either is smooth).

    Second differences remove a profile's smooth shape and keep its bin-to-bin
    sampling noise. Two averages that share realisations share part of that noise,
    so this correlation shows which profiles come from the same axis.
    """
    with np.errstate(invalid="ignore"):
        return float(np.corrcoef(np.diff(a, 2), np.diff(b, 2))[0, 1])


def format_comparison(frac: str, result: VarianceComparison) -> list[str]:
    """The report lines of a ``compare_variance`` result."""
    lines = [f"  variance.x_{frac}.avg.dat against the {len(result.instances)} kept instances "
             f"({' '.join(result.instances)}):"]
    if result.best_miss_nm2 is None:
        lines.append(f"    duplication-weighted mean: {result.n_weightings} weightings, not searched")
    else:
        lines.append(f"    duplication-weighted mean of them ({result.n_weightings} weightings, each "
                     f"instance at least once, {result.n_total} in all): the closest differs by "
                     f"{result.best_miss_nm2:.3g} nm^2 in its worst frame")
    if result.z_max is not None:
        lines.append(f"    mean of {result.n_total} distinct realisations including them: largest "
                     f"|z| {result.z_max:.2f} over {result.n_frames_z} frames")
    return lines


def check(frac: str, src: Path, cube: np.ndarray, geom: Geometry, frames: list[int]) -> None:
    """Print how the collapse of ``frames`` compares with the existing profile file in ``src``."""
    name = f"collapsed_1d.x_{frac}.npz"
    from_script = collapse(cube, geom, frames)
    with np.load(src / name, allow_pickle=False) as from_file:
        print(f"x_{frac}: frames {' '.join(map(str, frames))} of diffmap.x_{frac}.avg.npy "
              f"collapsed by collapse_diffmaps.py, against {name} (values: script / file)")
        print(f"  keys in the order the script writes them: {tuple(from_file.files) == KEYS}")
        window = "; ".join(
            f"{key} {from_script[key]:g} / {from_file[key].item():g} "
            f"equal {from_script[key] == from_file[key].item()}" for key in WINDOW_KEYS)
        print(f"  x equal {np.array_equal(from_script['x'], from_file['x'])}; t equal at the "
              f"frames {np.array_equal(from_script['t'], from_file['t'][frames])}; {window}")
        print("  instance counts in the file (not checkable from the map): "
              + ", ".join(f"{key} {from_file[key].item()}" for key in COUNT_KEYS))
        file_profs = {prof: from_file[prof] for prof in ("prof_x", "prof_y")}
        for row, k in enumerate(frames):
            for prof in ("prof_x", "prof_y"):
                script_prof, file_prof = from_script[prof][row], file_profs[prof][k]
                diff = np.abs(script_prof - file_prof).max()
                r_x = fine_structure_r(script_prof, file_profs["prof_x"][k])
                r_y = fine_structure_r(script_prof, file_profs["prof_y"][k])
                print(f"  frame {k} {prof}: equal {np.array_equal(script_prof, file_prof)}, "
                      f"max |diff| {diff:.2e} ({diff / np.abs(file_prof).max():.3f} of peak); "
                      f"fine structure r {r_x:+.2f} with the file's prof_x, {r_y:+.2f} with "
                      "its prof_y")
        n_total = int(from_file["n_unique"])
    if any(src.glob(f"inst_*/variance.x_{frac}.dat")):
        result = compare_variance(frac, src, geom, n_total)
        if result is None:
            print("  variance: fewer than two complete kept instances, no comparison")
        else:
            print("\n".join(format_comparison(frac, result)))


def run(args: list[str]) -> None:
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


def main() -> None:
    try:
        run(sys.argv[1:])
    except (OSError, ValueError) as err:   # a refusal or a missing input: one line, no traceback
        sys.exit(f"collapse_diffmaps.py: {err}")


if __name__ == "__main__":
    main()
