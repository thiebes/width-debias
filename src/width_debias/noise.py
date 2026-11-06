"""Pluggable measurement-noise models for the Monte Carlo.

The default pipeline adds white, signal-independent Gaussian noise at the
profile level (constant standard deviation set by the requested CNR). That is
the noise law the default calibration is built on. Real transient-
microscopy detectors are dominated by *shot* (photon-counting) noise, whose
variance scales with the local signal, plus a read-noise floor. A correction
calibrated and validated only on white noise has not been tested against the
noise it will actually meet -- so this module makes the noise law pluggable, the
same way :mod:`width_debias.models` makes the profile shape pluggable.

A noise model carries everything needed to (1) turn a clean profile stack into
one noisy realisation and (2) report the per-frame contrast-to-noise ratio that
keys the correction. Both are parameterised by ``cnr0``, the initial peak CNR,
so a single ``cnr0`` sweep means the same thing across noise laws: at frame 0 the
peak contrast-to-noise equals ``cnr0`` in every model.

``run_ensemble(..., noise_model=None)`` draws white noise inline. Pass a model
to swap the
noise law; :class:`WhiteNoise` reproduces the default through this interface, so
a white-vs-shot comparison runs through one code path with only the noise law
changed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _peak0(clean: np.ndarray) -> float:
    """Initial (frame-0) peak height of a clean profile stack."""
    return float(clean[0].max())


@dataclass(frozen=True)
class WhiteNoise:
    """Constant-standard-deviation additive Gaussian noise (the default law).

    The noise RMS is ``peak0 / cnr0`` -- independent of position and time -- so
    the per-frame CNR is just the decaying peak divided by that fixed RMS. Reproduces the inline default in
    :func:`width_debias.montecarlo.run_ensemble`.
    """

    name: str = "white"

    def _sigma(self, clean: np.ndarray, cnr0: float) -> float:
        return _peak0(clean) / cnr0

    def cnr_frame(self, clean: np.ndarray, cnr0: float) -> np.ndarray:
        return clean.max(axis=1) / self._sigma(clean, cnr0)

    def sample(self, clean: np.ndarray, cnr0: float,
               rng: np.random.Generator) -> np.ndarray:
        return clean + rng.normal(0.0, self._sigma(clean, cnr0),
                                  size=clean.shape)


@dataclass(frozen=True)
class ShotNoise:
    """Signal-dependent (shot) noise with an optional constant-noise floor.

    The clean profile is interpreted as a photon-count rate scaled so that the
    frame-0 peak holds ``cnr0**2`` counts; then the per-pixel variance is the
    local count (shot) plus a fixed floor variance ``read**2``. With that
    scaling the frame-0 peak shot-CNR is ``sqrt(counts) = cnr0``, matching the
    white-noise convention so a shared ``cnr0`` sweep is comparable. The
    per-frame CNR is the peak count over its own noise,
    ``peak / sqrt(peak + read**2)``, which falls with decay and spreading -- but
    the noise is now concentrated where the signal is, the property the
    white-noise calibration does not know about.

    The ``read`` floor spans the transport-microscopy regimes on one axis. A
    large floor reproduces the differential / pump-probe case (stroboSCAT,
    transient absorption), where the noise is set by a bright constant probe
    pedestal and is effectively white. ``read = 0`` is the pedestal-free,
    photon-limited case of direct emission imaging.

    ``poisson`` selects true Poisson counting for the signal term instead of its
    Gaussian approximation. This matters in the dim, low-CNR regime (the
    bias-prone late frames) of genuine photon-counting techniques (time-
    correlated single-photon counting, direct PL imaging), where counts are few
    and the Poisson distribution is discrete and asymmetric -- exactly the
    regime a Gaussian approximation misrepresents. The Gaussian form can return
    small negative samples in the deep tail; that is left as-is, because the
    width fit must cope with such samples on real data (clipping them would
    reintroduce a positive-bias floor).
    """

    read: float = 0.0
    poisson: bool = False
    name: str = "shot"

    def _scale(self, clean: np.ndarray, cnr0: float) -> float:
        """Counts-per-unit-amplitude so the frame-0 peak holds cnr0**2 counts."""
        return (cnr0 ** 2) / _peak0(clean)

    def cnr_frame(self, clean: np.ndarray, cnr0: float) -> np.ndarray:
        peak_counts = clean.max(axis=1) * self._scale(clean, cnr0)
        return peak_counts / np.sqrt(peak_counts + self.read ** 2)

    def sample(self, clean: np.ndarray, cnr0: float,
               rng: np.random.Generator) -> np.ndarray:
        scale = self._scale(clean, cnr0)
        counts = np.clip(clean * scale, 0.0, None)
        if self.poisson:
            noisy = rng.poisson(counts).astype(float)
            if self.read > 0.0:
                noisy = noisy + rng.normal(0.0, self.read, size=counts.shape)
        else:
            sigma = np.sqrt(counts + self.read ** 2)
            noisy = counts + rng.normal(0.0, 1.0, size=counts.shape) * sigma
        return noisy / scale
