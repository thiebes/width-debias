"""collapse_diffmaps.py: the profile file's keys, axes and grid, its refusals, and its check.

The script turns an averaged 2-D map into the file ``br_highn_data.py`` reads,
on the run machine, where the high-N map and its instances live. These tests use
small hand-built instances, averaged by ``average_diffmaps.py`` itself, so that
each settled rule is pinned exactly: the key order, rows as y and columns as x,
the ``linspace`` grid, no renormalisation, the walker count, the refusals that
keep a profile file from being replaced or labelled with instances its map does
not hold, and the variance comparison whose numbers the rerun README quotes.
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
# The source distribution carries the package and its tests, but not the
# repository's experiment scripts that this module checks; skip it there.
if not (ROOT / "experiments").is_dir():
    pytest.skip("needs a full checkout of the repository (experiments/ is absent)",
                allow_module_level=True)
sys.path.insert(0, str(ROOT / "experiments" / "halide-mixing-brI-legacy" / "rerun"))

import average_diffmaps  # noqa: E402
import collapse_diffmaps as cd  # noqa: E402

FRAC = "0.25"
AGREED_KEYS = ("x", "t", "prof_x", "prof_y", "box_nm", "n_bins", "n_frames", "n_unique",
               "n_per_instance", "n_walkers", "x_frac", "tsamp_ns")
PARAMS = """x       0.25        # Fraction of Br atoms
tsamp   0.5         # Sampling time [ns]
tmax    1.0         # Simulation time [ns]
N       {n}         # Number of excitons
diffmap_Lx 12       # Diffmap dimensions [um]
diffmap_Ly 12.0
diffmap_nx 5        # Number of bins in diffmap
diffmap_ny 5
"""


def _cube() -> np.ndarray:
    """Three 5 x 5 frames with different row and column sums and curved profiles."""
    i, j = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")    # i: row (y), j: column (x)
    frames = [(1 + i**2 + 3 * j**3 + k * i * j).astype(np.float32) for k in range(3)]
    frames[2][1, 4] += 50.0    # a peak at row 1, column 4
    return np.array(frames)


def _variance(idx: int, rows: int = 3) -> np.ndarray:
    """A (time, variance) table that differs between instances."""
    t = 0.5 * np.arange(rows)
    return np.column_stack([t, 10.0 * idx + 7.0 * t + idx**2 * t**2])


def _write_instance(src: Path, idx: int, scale: float, walkers: int, rows: int = 3) -> str:
    inst = src / f"inst_{idx:03d}"
    inst.mkdir()
    (inst / f"parameters.x_{FRAC}.dat").write_text(PARAMS.format(n=walkers))
    np.savetxt(inst / f"diffmap.x_{FRAC}.dat", (scale * _cube()).reshape(-1, 5), fmt="%.6f")
    np.savetxt(inst / f"variance.x_{FRAC}.dat", _variance(idx, rows), fmt="%.6f")
    return inst.name


def _average(monkeypatch, src: Path, names: list[str]) -> None:
    """Run average_diffmaps.py in ``src`` over ``names``, as step 6 does."""
    monkeypatch.chdir(src)
    monkeypatch.setattr(sys, "argv", ["average_diffmaps.py", FRAC, *names])
    average_diffmaps.main()


def _bed(tmp_path: Path, monkeypatch, walkers=(7, 7), rows: int = 3) -> Path:
    """A fraction folder as the top-up leaves it: instances, their average, the list.

    The two instances' maps are 0.5 and 1.5 times ``_cube()``, so their average is
    ``_cube()`` exactly.
    """
    src = tmp_path / "x_0.25"
    src.mkdir()
    (src / f"parameters.x_{FRAC}.dat").write_text(PARAMS.format(n=20))
    names = [_write_instance(src, idx, scale, n, rows)
             for idx, (scale, n) in enumerate(zip((0.5, 1.5), walkers), start=1)]
    _average(monkeypatch, src, names)
    (src / "unique_instances.txt").write_text("".join(f"{name}\n" for name in names))
    return src


def _run(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["collapse_diffmaps.py", *args])
    cd.main()


def test_collapse_writes_the_agreed_keys_axes_and_grid(tmp_path, monkeypatch):
    geom = cd.read_geometry(_bed(tmp_path, monkeypatch) / f"parameters.x_{FRAC}.dat")
    cube = _cube()
    out = cd.collapse(cube, geom, n_unique=2, n_per_instance=7)

    assert tuple(out) == AGREED_KEYS
    assert geom == cd.Geometry(box_nm=12.0, n_bins=5, n_frames=3, tsamp_ns=0.5, x_frac=0.25)
    np.testing.assert_array_equal(out["x"], np.linspace(-6.0, 6.0, 5))
    np.testing.assert_array_equal(out["t"], [0.0, 0.5, 1.0])
    for k in range(3):
        np.testing.assert_allclose(out["prof_x"][k], cube[k].astype(float).sum(axis=0))
        np.testing.assert_allclose(out["prof_y"][k], cube[k].astype(float).sum(axis=1))
        assert out["prof_x"][k].sum() == pytest.approx(cube[k].astype(float).sum())   # not renormalised
    without_peak = cube.copy()
    without_peak[2, 1, 4] -= 50.0
    base = cd.collapse(without_peak, geom, n_unique=2, n_per_instance=7)
    np.testing.assert_allclose(out["prof_x"][2] - base["prof_x"][2], [0, 0, 0, 0, 50])   # column is x
    np.testing.assert_allclose(out["prof_y"][2] - base["prof_y"][2], [0, 50, 0, 0, 0])   # row is y
    assert out["n_walkers"] == 14
    assert (out["n_bins"], out["n_frames"], out["box_nm"], out["x_frac"]) == (5, 3, 12.0, 0.25)


def test_collapse_without_counts_leaves_them_out_and_needs_both(tmp_path, monkeypatch):
    geom = cd.read_geometry(_bed(tmp_path, monkeypatch) / f"parameters.x_{FRAC}.dat")
    out = cd.collapse(_cube(), geom, frames=[2, 0])
    assert tuple(out) == tuple(k for k in AGREED_KEYS if k not in cd.COUNT_KEYS)
    np.testing.assert_array_equal(out["t"], [1.0, 0.0])
    assert out["prof_x"].shape == (2, 5)
    with pytest.raises(ValueError, match="both"):
        cd.collapse(_cube(), geom, n_unique=2)


def test_write_then_check_round_trip(tmp_path, monkeypatch, capsys):
    src = _bed(tmp_path, monkeypatch)
    _run(monkeypatch, FRAC)
    with np.load(src / f"collapsed_1d.x_{FRAC}.npz", allow_pickle=False) as written:
        assert tuple(written.files) == AGREED_KEYS
        assert (int(written["n_unique"]), int(written["n_per_instance"])) == (2, 7)
        np.testing.assert_array_equal(written["x"], np.linspace(-6.0, 6.0, 5))
        np.testing.assert_array_equal(written["t"], [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(written["prof_x"][2], _cube()[2].astype(float).sum(axis=0))

    capsys.readouterr()
    _run(monkeypatch, FRAC, str(src), "--check", "0", "2")
    report = capsys.readouterr().out
    assert "keys in the order the script writes them: True" in report
    assert "x equal True; t equal at the frames True" in report
    for key in ("box_nm", "n_bins", "n_frames", "x_frac", "tsamp_ns"):
        assert re.search(rf"{key} \S+ / \S+ equal True", report), key
    assert report.count("equal True, max |diff| 0.00e+00") == 4
    assert "n_unique 2, n_per_instance 7, n_walkers 14" in report
    prof_x_lines = [line for line in report.splitlines() if " prof_x: " in line]
    assert len(prof_x_lines) == 2
    assert all("fine structure r +1.00 with the file's prof_x" in line for line in prof_x_lines)
    assert "the closest differs by 0 nm^2" in report


def test_refuses_to_replace_a_profile_file(tmp_path, monkeypatch):
    src = _bed(tmp_path, monkeypatch)
    target = src / f"collapsed_1d.x_{FRAC}.npz"
    target.write_bytes(b"existing profile file")
    with pytest.raises(SystemExit, match="never replaces"):
        _run(monkeypatch, FRAC)
    assert target.read_bytes() == b"existing profile file"


def test_refuses_copies_of_one_realisation(tmp_path, monkeypatch):
    """The first run's state: duplicates averaged and all listed."""
    src = _bed(tmp_path, monkeypatch)
    shutil.copytree(src / "inst_001", src / "inst_003")
    names = ["inst_001", "inst_002", "inst_003"]
    _average(monkeypatch, src, names)
    (src / "unique_instances.txt").write_text("\n".join(names) + "\n")
    with pytest.raises(SystemExit, match="copies of one realisation"):
        _run(monkeypatch, FRAC)
    assert not (src / f"collapsed_1d.x_{FRAC}.npz").exists()


def test_refuses_a_map_that_is_not_the_listed_instances_average(tmp_path, monkeypatch):
    src = _bed(tmp_path, monkeypatch)
    (src / "unique_instances.txt").write_text("inst_001\n")    # the map averages two
    with pytest.raises(SystemExit, match="not the mean"):
        _run(monkeypatch, FRAC)
    assert not (src / f"collapsed_1d.x_{FRAC}.npz").exists()


def test_refuses_a_variance_file_off_by_a_small_relative_amount(tmp_path, monkeypatch):
    src = _bed(tmp_path, monkeypatch)
    path = src / f"variance.x_{FRAC}.avg.dat"
    np.savetxt(path, np.loadtxt(path) * (1 + 1e-9))
    with pytest.raises(SystemExit, match="not the mean"):
        _run(monkeypatch, FRAC)


def test_refuses_a_listed_instance_that_is_missing(tmp_path, monkeypatch):
    src = _bed(tmp_path, monkeypatch)
    (src / "unique_instances.txt").write_text("inst_001\ninst_002\ninst_003\n")
    with pytest.raises(SystemExit, match="no variance file, for example inst_003"):
        _run(monkeypatch, FRAC)


def test_refuses_instances_with_different_walker_counts(tmp_path, monkeypatch):
    _bed(tmp_path, monkeypatch, walkers=(7, 8))
    with pytest.raises(SystemExit, match="differ in walker count"):
        _run(monkeypatch, FRAC)


def test_refuses_variance_files_without_one_row_per_frame(tmp_path, monkeypatch):
    _bed(tmp_path, monkeypatch, rows=4)
    with pytest.raises(SystemExit, match="does not have 3 rows"):
        _run(monkeypatch, FRAC)


def test_refuses_a_frame_count_other_than_tmax_over_tsamp_plus_one(tmp_path, monkeypatch):
    geom = cd.read_geometry(_bed(tmp_path, monkeypatch) / f"parameters.x_{FRAC}.dat")
    extra_block = np.concatenate([_cube(), _cube()[:1]])
    with pytest.raises(ValueError, match="tmax / tsamp"):
        cd.collapse(extra_block, geom)


def test_read_geometry_needs_a_square_map_and_names_missing_keys(tmp_path):
    path = tmp_path / f"parameters.x_{FRAC}.dat"
    path.write_text(PARAMS.format(n=7).replace("diffmap_ny 5", "diffmap_ny 4"))
    with pytest.raises(ValueError, match="not square"):
        cd.read_geometry(path)
    path.write_text(PARAMS.format(n=7).replace("tmax    1.0", ""))
    with pytest.raises(ValueError, match="has no tmax"):
        cd.read_geometry(path)


def test_positive_compositions():
    np.testing.assert_array_equal(cd.positive_compositions(4, 2), [[1, 3], [2, 2], [3, 1]])
    weights = cd.positive_compositions(16, 5)
    assert weights.shape == (1365, 5)
    assert (weights.sum(axis=1) == 16).all() and weights.min() == 1
    assert len({tuple(w) for w in weights}) == 1365


def _variance_bed(tmp_path: Path, averaged: np.ndarray) -> tuple[Path, cd.Geometry]:
    """Three kept instances' variance files and an averaged variance column."""
    src = tmp_path / "cmp"
    src.mkdir()
    (src / f"parameters.x_{FRAC}.dat").write_text(PARAMS.format(n=7))
    for idx in (1, 2, 3):
        (src / f"inst_{idx:03d}").mkdir()
        np.savetxt(src / f"inst_{idx:03d}" / f"variance.x_{FRAC}.dat", _variance(idx), fmt="%.6f")
    np.savetxt(src / f"variance.x_{FRAC}.avg.dat", np.column_stack([_variance(1)[:, 0], averaged]))
    return src, cd.read_geometry(src / f"parameters.x_{FRAC}.dat")


def test_compare_variance_finds_a_duplication_weighted_mean(tmp_path):
    kept = [_variance(idx)[:, 1] for idx in (1, 2, 3)]
    src, geom = _variance_bed(tmp_path, (1 * kept[0] + 3 * kept[1] + 2 * kept[2]) / 6)
    result = cd.compare_variance(FRAC, src, geom, n_total=6)
    assert result.instances == ("inst_001", "inst_002", "inst_003")
    assert result.n_weightings == 10
    assert result.best_miss_nm2 < 1e-9
    assert result.z_max is not None and result.n_frames_z == 3


def test_compare_variance_rules_out_weighting_for_a_mean_of_distinct_realisations(tmp_path):
    realisations = [_variance(idx)[:, 1] for idx in (1, 2, 3, 7, 8, 9)]
    src, geom = _variance_bed(tmp_path, np.mean(realisations, axis=0))
    result = cd.compare_variance(FRAC, src, geom, n_total=6)
    assert result.best_miss_nm2 > 5.0
    assert np.isfinite(result.z_max)
