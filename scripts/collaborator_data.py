"""Where the experiment sets read collaborator data: the data root and its folders.

Collaborator data is never committed. The scripts read it from the data root,
the folder that ``WIDTH_DEBIAS_DATA_ROOT`` names (``~`` and relative paths are
resolved), or else ``data/`` at the repository root, which git ignores. The
folder names are fixed, so one copy of the data can serve several repositories
that read it:

  ``stroboscat_scans/``     ``wse2-strobescat``
  ``simulated_map/``        ``nc-solid``
  ``halide_mixing/``        ``halide-mixing-brI-legacy``
  ``halide_mixing_highn/``  ``halide-mixing-highn``
  ``cspbbr_sample_1/``      ``cspbbr-strobescat``

Each set's ``DATASETS.md`` lists the files its folder holds. ``mn-trapping``
keeps its own local ``data/`` copy. Result files of an external simulation
study are not collaborator data; scripts read them from the folder that
``WIDTH_DEBIAS_EXTERNAL_RESULTS`` names, which has no default.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "WIDTH_DEBIAS_DATA_ROOT"
REPO = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """The folder ``WIDTH_DEBIAS_DATA_ROOT`` names, or else ``data/`` at the repository root."""
    value = os.environ.get(ENV_VAR)
    return Path(value).expanduser().resolve() if value else REPO / "data"


def stroboscat_scans_dir() -> Path:
    """The ``stroboscat_scans/`` folder of the data root."""
    return data_root() / "stroboscat_scans"


def simulated_map_dir() -> Path:
    """The ``simulated_map/`` folder of the data root."""
    return data_root() / "simulated_map"


def halide_mixing_dir() -> Path:
    """The ``halide_mixing/`` folder of the data root."""
    return data_root() / "halide_mixing"


def halide_mixing_highn_dir() -> Path:
    """The ``halide_mixing_highn/`` folder of the data root."""
    return data_root() / "halide_mixing_highn"


def cspbbr_sample_1_dir() -> Path:
    """The ``cspbbr_sample_1/`` folder of the data root."""
    return data_root() / "cspbbr_sample_1"
