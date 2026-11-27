"""run_instances.sh: each instance starts in its own second, gets its N, and only distinct ones are listed.

The script launches the Br-I simulator on the run machine. Here it runs under bash
with a stub simulator that, like the real one, seeds from the wall clock at
one-second resolution: it writes the current second into its variance file. The
tests pin what the rerun README relies on:
- every instance's parameter file carries the requested walker count;
- instances get distinct seconds, so distinct variance files;
- incomplete instances (no file, or a file short of tmax / tsamp + 1 rows) are
  left out and named;
- copies of one realisation are listed once;
- a folder that already holds instance folders, a gap that is not a whole number
  of seconds, and a parameter file without tmax and tsamp are refused.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "halide-mixing-brI-legacy" / "rerun" / "run_instances.sh"
FRAC = "0.25"
PARAMS = """x       0.25        # Fraction of Br atoms
tsamp   0.005       # Sampling time [ns]
tmax    0.01        # Simulation time [ns]
N       20000       # Number of excitons
"""
CLOCK_STUB = """#!/usr/bin/env bash
s=$(date +%s)
case "$PWD" in
  *inst_002) echo "set-up messages" >&2; exit 1 ;;
  *inst_003) printf '0 %s\\n1 %s\\n' "$s" "$s" > variance.x_0.25.dat ;;
  *) printf '0 %s\\n1 %s\\n2 %s\\n' "$s" "$s" "$s" > variance.x_0.25.dat ;;
esac
"""
FIXED_STUB = """#!/usr/bin/env bash
printf '0 42\\n1 42\\n2 42\\n' > variance.x_0.25.dat
"""


def _bash_with_tools() -> bool:
    bash = shutil.which("bash")
    if bash is None:
        return False
    probe = subprocess.run([bash, "-c", "command -v xargs seq nproc sha256sum awk"],
                           capture_output=True)
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _bash_with_tools(),
                                reason="needs bash with xargs, seq, nproc and sha256sum")


def _folder(tmp_path: Path, stub: str, params: str = PARAMS) -> Path:
    folder = tmp_path / "x_0.25"
    folder.mkdir()
    shutil.copy(SCRIPT, folder / "run_instances.sh")
    for name, text in (("landscapeBD.diffmap", stub), (f"parameters.x_{FRAC}.dat", params),
                       ("basis_BrI.dat", ""), ("vectors_BrI.dat", "")):
        (folder / name).write_bytes(text.encode())
    return folder


def _run(folder: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "run_instances.sh", FRAC, "25000", *args], cwd=folder,
                          capture_output=True, text=True, timeout=120)


def test_lists_distinct_complete_instances_and_passes_n(tmp_path):
    folder = _folder(tmp_path, CLOCK_STUB)
    result = _run(folder, "4", "4", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (folder / "unique_instances.txt").read_text() == "inst_001\ninst_004\n"
    assert ("Incomplete (variance file short of 3 rows; see their run.err): inst_002 inst_003"
            in result.stdout)
    assert "./rerun_unique.sh 0.25 25000 4" in result.stdout
    for k in range(1, 5):
        params = (folder / f"inst_{k:03d}" / f"parameters.x_{FRAC}.dat").read_text()
        assert "\nN       25000       # Number of excitons\n" in params


def test_lists_copies_of_one_realisation_once(tmp_path):
    folder = _folder(tmp_path, FIXED_STUB)
    result = _run(folder, "3", "3", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (folder / "unique_instances.txt").read_text() == "inst_001\n"
    assert "Fewer than 3 distinct" in result.stdout
    assert "Or average the 1 distinct instance(s)" in result.stdout


def test_refuses_a_folder_that_already_holds_instances(tmp_path):
    folder = _folder(tmp_path, FIXED_STUB)
    (folder / "inst_002").mkdir()
    result = _run(folder, "3", "3", "1")
    assert result.returncode == 1
    assert "inst_002 already exists" in result.stdout
    assert not (folder / "inst_001").exists()


def test_refuses_a_gap_that_is_not_a_whole_number(tmp_path):
    folder = _folder(tmp_path, FIXED_STUB)
    result = _run(folder, "3", "3", "1.5")
    assert result.returncode == 1
    assert "gap_seconds must be a whole number" in result.stdout


def test_refuses_a_parameter_file_without_tmax_and_tsamp(tmp_path):
    folder = _folder(tmp_path, FIXED_STUB, params="x 0.25\nN 20000\n")
    result = _run(folder, "3", "3", "1")
    assert result.returncode == 1
    assert "gives no tmax and tsamp" in result.stdout
