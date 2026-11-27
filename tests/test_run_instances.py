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
- a folder that already holds inst_* folders, a gap that is not a whole number
  of seconds, and a parameter file without both tmax and tsamp are refused.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The source distribution carries the package and its tests, but not the
# repository's experiment scripts that this module checks; skip it there.
if not (ROOT / "experiments").is_dir():
    pytest.skip("needs a full checkout of the repository (experiments/ is absent)",
                allow_module_level=True)
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
FULL_CLOCK_STUB = """#!/usr/bin/env bash
s=$(date +%s)
printf '0 %s\\n1 %s\\n2 %s\\n' "$s" "$s" "$s" > variance.x_0.25.dat
"""
FIXED_STUB = """#!/usr/bin/env bash
printf '0 42\\n1 42\\n2 42\\n' > variance.x_0.25.dat
"""
SILENT_STUB = """#!/usr/bin/env bash
exit 0
"""


def _bash_with_tools() -> str | None:
    """The bash the tests run, if it has the tools the script uses."""
    bash = shutil.which("bash")
    if bash is None:
        return None
    try:
        probe = subprocess.run([bash, "-c", "command -v xargs seq nproc sha256sum awk compgen"],
                               capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None
    return bash if probe.returncode == 0 else None


BASH = _bash_with_tools()
pytestmark = pytest.mark.skipif(BASH is None,
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
    return subprocess.run([BASH, "run_instances.sh", FRAC, "25000", *args], cwd=folder,
                          capture_output=True, text=True, timeout=120)


def test_every_instance_gets_its_own_second(tmp_path):
    """Six instances 2 s apart are all distinct; a random 0 to 4 s stagger could not be."""
    folder = _folder(tmp_path, FULL_CLOCK_STUB)
    result = _run(folder, "6", "6", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (folder / "unique_instances.txt").read_text() == "".join(
        f"inst_{k:03d}\n" for k in range(1, 7))
    assert "Average them with:" in result.stdout
    assert "rerun_unique.sh" not in result.stdout
    for k in range(1, 7):
        params = (folder / f"inst_{k:03d}" / f"parameters.x_{FRAC}.dat").read_text()
        assert "\nN       25000       # Number of excitons\n" in params


def test_lists_distinct_complete_instances_and_names_the_rest(tmp_path):
    folder = _folder(tmp_path, CLOCK_STUB)
    result = _run(folder, "4", "4", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (folder / "unique_instances.txt").read_text() == "inst_001\ninst_004\n"
    assert ("Incomplete (variance file short of 3 rows; see their run.err): inst_002 inst_003"
            in result.stdout)
    top_up = result.stdout.index("./rerun_unique.sh 0.25 25000 4")
    assert top_up < result.stdout.index("Or average the 2 distinct instance(s)")
    for k in range(1, 5):
        params = (folder / f"inst_{k:03d}" / f"parameters.x_{FRAC}.dat").read_text()
        assert "\nN       25000       # Number of excitons\n" in params


def test_prints_no_average_command_without_a_distinct_instance(tmp_path):
    folder = _folder(tmp_path, SILENT_STUB)
    result = _run(folder, "2", "2", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (folder / "unique_instances.txt").read_text() == ""
    assert "./rerun_unique.sh 0.25 25000 2" in result.stdout
    assert "average_diffmaps.py" not in result.stdout


def test_lists_copies_of_one_realisation_once(tmp_path):
    folder = _folder(tmp_path, FIXED_STUB)
    result = _run(folder, "3", "3", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (folder / "unique_instances.txt").read_text() == "inst_001\n"
    assert "Fewer than 3 distinct" in result.stdout
    assert "Or average the 1 distinct instance(s)" in result.stdout


@pytest.mark.parametrize("existing", ["inst_002", "inst_009"])
def test_refuses_a_folder_that_already_holds_instances(tmp_path, existing):
    folder = _folder(tmp_path, FIXED_STUB)
    (folder / existing).mkdir()
    result = _run(folder, "3", "3", "1")
    assert result.returncode == 1
    assert f"{existing} already exists" in result.stdout
    assert not (folder / "inst_001").exists()


def test_refuses_a_gap_that_is_not_a_whole_number(tmp_path):
    folder = _folder(tmp_path, FIXED_STUB)
    result = _run(folder, "3", "3", "1.5")
    assert result.returncode == 1
    assert "gap_seconds must be a whole number" in result.stdout


@pytest.mark.parametrize("params", ["x 0.25\nN 20000\n", "x 0.25\ntsamp 0.005\nN 20000\n",
                                    "x 0.25\ntmax 0.01\nN 20000\n"])
def test_refuses_a_parameter_file_without_both_tmax_and_tsamp(tmp_path, params):
    folder = _folder(tmp_path, FIXED_STUB, params=params)
    result = _run(folder, "3", "3", "1")
    assert result.returncode == 1
    assert "does not give both tmax and tsamp" in result.stdout
