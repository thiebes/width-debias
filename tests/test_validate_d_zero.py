"""The D-free validation script: its saved statistics, its reports and its range check.

The falling-peak reports in ``experiments/dfree-calibration`` are rebuilt from
their committed statistics (``--report``), so a report and the numbers quoted
from it can be regenerated without the Monte Carlo. These tests pin that chain,
and with it the field names the script writes and reads. The range check runs
on the noise-free beds only, so nothing here runs a Monte Carlo.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

# Imported before the script, which puts src/ first on the path: the package is
# then the one the rest of the session tests (the installed one on CI).
from width_debias import BiasLookup, model_from_spec

ROOT = Path(__file__).resolve().parents[1]
DFREE = ROOT / "experiments" / "dfree-calibration"


def _load_script():
    """Import the script by path, leaving ``sys.path`` as it was."""
    saved_path = list(sys.path)
    spec = importlib.util.spec_from_file_location("validate_d_zero", DFREE / "validate_d_zero.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module      # a dataclass resolves its module by name
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
    return module


vdz = _load_script()
STATISTICS = sorted(path for path in DFREE.glob("validate_d_zero*_spread1_cal5000_*.npz")
                    if not path.name.endswith("_lookup.npz"))


def _lookup_path(statistics: Path) -> Path:
    return statistics.with_name(statistics.stem + "_lookup.npz")


def test_falling_peak_statistics_are_committed_for_both_windows_and_four_shapes():
    assert len(STATISTICS) == 2 * len(vdz.SHAPES)
    assert all(_lookup_path(path).exists() for path in STATISTICS)


@pytest.mark.parametrize("statistics", STATISTICS, ids=lambda path: path.stem)
def test_committed_report_rebuilds_from_its_statistics(statistics, tmp_path):
    """``--report`` on the committed statistics gives the committed report, line for line."""
    copy = tmp_path / statistics.name
    shutil.copyfile(statistics, copy)
    rebuilt = vdz.write_report(copy)
    assert (rebuilt.read_text(encoding="utf-8")
            == statistics.with_suffix(".txt").read_text(encoding="utf-8"))


@pytest.mark.parametrize("statistics", STATISTICS, ids=lambda path: path.stem)
def test_committed_lookup_has_the_lower_edge_its_range_check_used(statistics):
    lookup = BiasLookup.load(_lookup_path(statistics))
    with np.load(statistics) as saved:
        predicted = float(saved["lowest_calibrated"])
    assert lookup.lowest_calibrated_cnr == pytest.approx(predicted, rel=vdz.RANGE_RTOL)


def test_report_names_the_file_it_needs_when_given_a_lookup(tmp_path):
    copy = tmp_path / _lookup_path(STATISTICS[0]).name
    shutil.copyfile(_lookup_path(STATISTICS[0]), copy)
    with pytest.raises(SystemExit, match="_lookup.npz"):
        vdz.write_report(copy)


@pytest.mark.parametrize("spec", vdz.SHAPES)
@pytest.mark.parametrize("spread_dims", [None, 1, 2], ids=["decay", "summed", "centre_cut"])
@pytest.mark.parametrize("geometry", [vdz.STANDARD, vdz.SHORT_WINDOW], ids=["standard", "short"])
def test_every_offered_run_keeps_its_frames_inside_the_calibration(geometry, spread_dims, spec):
    """Each window, peak and shape the script offers passes its own range check."""
    cal_grid, _ = vdz._calibration_grid(geometry, spread_dims)
    lowest_calibrated, lowest_kept, _ = vdz._cnr_coverage(geometry, cal_grid, spread_dims,
                                                          model_from_spec(spec))
    assert not vdz._below_range(lowest_kept, lowest_calibrated)
