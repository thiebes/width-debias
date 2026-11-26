"""The data root that the experiment loaders read collaborator data from.

``scripts/collaborator_data.py`` resolves the root and names its folders. The
folder names are shared with other repositories that read the same data, so
they are fixed here. Collaborator data lives outside the repository, so no
tracked text file may name a location on a group drive.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The source distribution carries the package and its tests, but not the
# repository's scripts and documents that this module checks; skip it there.
if not (ROOT / "scripts").is_dir():
    pytest.skip("needs a full checkout of the repository (scripts/ is absent)",
                allow_module_level=True)
sys.path.insert(0, str(ROOT / "scripts"))

import collaborator_data  # noqa: E402

FOLDERS = {
    collaborator_data.stroboscat_scans_dir: "stroboscat_scans",
    collaborator_data.simulated_map_dir: "simulated_map",
    collaborator_data.halide_mixing_dir: "halide_mixing",
    collaborator_data.halide_mixing_highn_dir: "halide_mixing_highn",
    collaborator_data.cspbbr_sample_1_dir: "cspbbr_sample_1",
}
# A drive letter, a colon and a slash or backslash, not preceded by a letter, so
# URLs ("http" + "://") do not match. Drive C is the local disk.
DRIVE_PATH = re.compile(r"(?<![A-Za-z])[A-BD-Z]:[\\/](?=[A-Za-z0-9_ ~.-])")


def test_unset_or_empty_variable_gives_data_at_the_repository_root(monkeypatch):
    monkeypatch.delenv(collaborator_data.ENV_VAR, raising=False)
    assert collaborator_data.data_root() == ROOT / "data"
    monkeypatch.setenv(collaborator_data.ENV_VAR, "")
    assert collaborator_data.data_root() == ROOT / "data"


def test_set_variable_is_expanded_and_resolved(monkeypatch, tmp_path):
    monkeypatch.setenv(collaborator_data.ENV_VAR, str(tmp_path))
    assert collaborator_data.data_root() == tmp_path.resolve()
    monkeypatch.setenv(collaborator_data.ENV_VAR, "~/width-debias-data")
    assert collaborator_data.data_root() == (Path.home() / "width-debias-data").resolve()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(collaborator_data.ENV_VAR, "relative-root")
    assert collaborator_data.data_root() == (tmp_path / "relative-root").resolve()


def test_folder_names_are_fixed(monkeypatch, tmp_path):
    monkeypatch.setenv(collaborator_data.ENV_VAR, str(tmp_path))
    for folder, name in FOLDERS.items():
        assert folder() == tmp_path.resolve() / name, folder.__name__


def test_no_tracked_text_file_names_a_drive_path():
    try:
        listing = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                                 check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a git checkout to list the tracked files")
    hits = []
    for name in filter(None, listing.decode("utf-8").split("\0")):
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue        # binary, or removed from the work tree
        hits += [f"{name}:{number}: {line.strip()}"
                 for number, line in enumerate(text.splitlines(), 1)
                 if DRIVE_PATH.search(line)]
    assert not hits, "drive paths in tracked files:\n" + "\n".join(hits)
