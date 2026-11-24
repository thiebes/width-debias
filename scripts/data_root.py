"""Where the experiment sets read collaborator data: the data root and its layout.

Collaborator data is never committed. The scripts read it from the data root:
the folder named by the environment variable ``WIDTH_DEBIAS_DATA_ROOT``, or else
``data/`` at the repository root (ignored by git). The folder names below are
fixed, so one copy of the data can serve several repositories that read it.

  ``stroboscat_scans/``    the measured scan folders, each at its path under the
                           experimenter's lab folder (``wse2-strobescat``)
  ``simulated_map/``       the simulated nanocrystal-solid maps
                           ``diffmap_{small,medium,large}.dat`` (``nc-solid``);
                           the scripts use whichever maps are present
  ``halide_mixing/``       the 20,000-walker Br-I maps, one ``BrI_x_<frac>/``
                           folder per Br fraction (``halide-mixing-brI-legacy``)
  ``halide_mixing_highn/`` the 400,000-walker Br-I maps collapsed to 1-D, one
                           ``x_<frac>/`` folder per Br fraction
                           (``halide-mixing-highn``)
  ``cspbbr_sample_1/``     the CsPbBr3 single-crystal scans, one folder per
                           fluence (``cspbbr-strobescat``)

Each set's ``DATASETS.md`` lists the files its folder must hold. The scripts only
read the data root; a folder that is missing or empty means that set's data is
not available on this machine.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "WIDTH_DEBIAS_DATA_ROOT"
REPO = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """The folder ``WIDTH_DEBIAS_DATA_ROOT`` names, or else ``data/`` at the repository root."""
    value = os.environ.get(ENV_VAR)
    return Path(value) if value else REPO / "data"


def stroboscat_scans_dir() -> Path:
    """The measured scan folders, at their paths under the experimenter's lab folder."""
    return data_root() / "stroboscat_scans"


def simulated_map_dir() -> Path:
    """The folder of the simulated nanocrystal-solid maps."""
    return data_root() / "simulated_map"


def halide_mixing_dir() -> Path:
    """The 20,000-walker Br-I maps, one ``BrI_x_<frac>/`` folder per Br fraction."""
    return data_root() / "halide_mixing"


def halide_mixing_highn_dir() -> Path:
    """The 400,000-walker Br-I maps collapsed to 1-D, one ``x_<frac>/`` folder per Br fraction."""
    return data_root() / "halide_mixing_highn"


def cspbbr_sample_dir() -> Path:
    """The CsPbBr3 single-crystal scans, one folder per fluence."""
    return data_root() / "cspbbr_sample_1"
