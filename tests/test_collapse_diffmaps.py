"""collapse_diffmaps.py: the profile file's keys, axes and grid, and the checks it makes before writing.

The script turns an averaged 2-D map into the file ``br_highn_data.py`` reads,
on the run machine, where the high-N map and its instances live. These tests use
a small hand-built map so that each settled rule can be pinned exactly: the key
order, rows as y and columns as x, the ``linspace`` grid, no renormalisation,
the walker count, and the refusals that keep a profile file from being replaced
or labelled with instances its map does not hold.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "halide-mixing-brI-legacy" / "rerun"))

import collapse_diffmaps as cd  # noqa: E402

FRAC = "0.25"
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


def _bed(tmp_path: Path, walkers=(7, 7)) -> Path:
    """A fraction folder as the top-up scripts leave it: map, averaged variance, instances."""
    src = tmp_path / "x_0.25"
    src.mkdir()
    (src / f"parameters.x_{FRAC}.dat").write_text(PARAMS.format(n=20))
    np.save(src / f"diffmap.x_{FRAC}.avg.npy", _cube())
    variances = []
    for idx, n in enumerate(walkers, start=1):
        inst = src / f"inst_{idx:03d}"
        inst.mkdir()
        (inst / f"parameters.x_{FRAC}.dat").write_text(PARAMS.format(n=n))
        v = np.column_stack([[0.0, 0.5, 1.0], [10.0 + idx, 20.0 + 2 * idx, 30.0 + idx**2]])
        np.savetxt(inst / f"variance.x_{FRAC}.dat", v, fmt="%.6f")
        variances.append(np.loadtxt(inst / f"variance.x_{FRAC}.dat"))
    np.savetxt(src / f"variance.x_{FRAC}.avg.dat", np.mean(variances, axis=0))
    (src / "unique_instances.txt").write_text(
        "".join(f"inst_{idx:03d}\n" for idx in range(1, len(walkers) + 1)))
    return src


def _run(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["collapse_diffmaps.py", *args])
    cd.main()


def test_collapse_writes_the_settled_keys_axes_and_grid(tmp_path):
    geom = cd.read_geometry(_bed(tmp_path) / f"parameters.x_{FRAC}.dat")
    cube = _cube()
    out = cd.collapse(cube, geom, n_unique=2, n_per_instance=7)

    assert tuple(out) == cd.KEYS
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


def test_collapse_without_counts_leaves_them_out(tmp_path):
    geom = cd.read_geometry(_bed(tmp_path) / f"parameters.x_{FRAC}.dat")
    out = cd.collapse(_cube(), geom, frames=[2, 0])
    assert tuple(out) == tuple(k for k in cd.KEYS if k not in cd.COUNT_KEYS)
    np.testing.assert_array_equal(out["t"], [1.0, 0.0])
    assert out["prof_x"].shape == (2, 5)


def test_write_then_check_round_trip(tmp_path, monkeypatch, capsys):
    src = _bed(tmp_path)
    monkeypatch.chdir(src)
    _run(monkeypatch, FRAC)
    written = np.load(src / f"collapsed_1d.x_{FRAC}.npz", allow_pickle=False)
    assert tuple(written.files) == cd.KEYS
    assert (int(written["n_unique"]), int(written["n_per_instance"])) == (2, 7)
    np.testing.assert_array_equal(written["prof_x"][2], _cube()[2].astype(float).sum(axis=0))

    capsys.readouterr()
    _run(monkeypatch, FRAC, str(src), "--check", "0", "2")
    report = capsys.readouterr().out
    assert "keys in the order the script writes them: True" in report
    assert "x equal True; t equal at the frames True" in report
    assert report.count("equal True, max |diff| 0.00e+00") == 4
    assert "n_unique 2, n_per_instance 7, n_walkers 14" in report


def test_refuses_to_replace_a_profile_file(tmp_path, monkeypatch):
    src = _bed(tmp_path)
    monkeypatch.chdir(src)
    target = src / f"collapsed_1d.x_{FRAC}.npz"
    target.write_bytes(b"existing profile file")
    with pytest.raises(SystemExit, match="never replaces"):
        _run(monkeypatch, FRAC)
    assert target.read_bytes() == b"existing profile file"


def test_refuses_a_map_that_is_not_the_listed_instances_average(tmp_path, monkeypatch):
    src = _bed(tmp_path)
    (src / "unique_instances.txt").write_text("inst_001\n")    # the map averages two
    monkeypatch.chdir(src)
    with pytest.raises(ValueError, match="not the mean"):
        _run(monkeypatch, FRAC)
    assert not (src / f"collapsed_1d.x_{FRAC}.npz").exists()


def test_refuses_a_listed_instance_that_is_missing(tmp_path, monkeypatch):
    src = _bed(tmp_path)
    (src / "unique_instances.txt").write_text("inst_001\ninst_002\ninst_003\n")
    monkeypatch.chdir(src)
    with pytest.raises(ValueError, match="no variance file, for example inst_003"):
        _run(monkeypatch, FRAC)


def test_refuses_instances_with_different_walker_counts(tmp_path, monkeypatch):
    src = _bed(tmp_path, walkers=(7, 8))
    monkeypatch.chdir(src)
    with pytest.raises(ValueError, match="differ in walker count"):
        _run(monkeypatch, FRAC)


def test_refuses_a_frame_count_other_than_tmax_over_tsamp_plus_one(tmp_path):
    geom = cd.read_geometry(_bed(tmp_path) / f"parameters.x_{FRAC}.dat")
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
