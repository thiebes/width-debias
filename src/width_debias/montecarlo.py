"""Monte Carlo orchestration: many noisy realisations of one experiment.

Because true D is known here, the ensemble exposes the bias that real
measurements cannot.  The output is a stack of per-frame fitted variances plus
the MSD curves derived from them, ready for the bias lookup and weighting
analysis.

The profile model (Gaussian / Voigt / generalized-Gaussian) is pluggable --
see :mod:`width_debias.models`.  The default is the Gaussian model.

Realisations are independent, so the ensemble is embarrassingly parallel across
runs.  ``run_ensemble`` can fan the per-realisation fits out over worker
processes (see the ``workers`` argument).  Seeding is per realisation via
:class:`numpy.random.SeedSequence`, so the result depends only on
``(seed, n_runs)`` -- it is identical whether the run is serial or parallel,
and independent of the worker count.  The fitting bottleneck is many small
nonlinear least-squares fits, which is CPU- not BLAS-bound; more cores give a
near-linear speedup.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .models import GaussianModel
from .simulate import Experiment, noise_sigma_for_cnr0


@dataclass
class Ensemble:
    """Results of an ``n_runs`` Monte Carlo sweep of one experiment."""

    exp: Experiment
    cnr0: float
    noise_sigma: float
    cnr_frame: np.ndarray        # per-frame CNR (peak / noise RMS), shape (n_frames,)
    variance: np.ndarray         # fitted variances, shape (n_runs, n_frames)
    msd: np.ndarray              # MSD curves, shape (n_runs, n_frames)
    at_wall: np.ndarray          # width-pinned flag, shape (n_runs, n_frames)
    ok: np.ndarray               # fit-converged flag, shape (n_runs, n_frames)
    model_name: str = "gaussian"

    @property
    def true_D(self) -> float:
        return self.exp.D

    @property
    def n_runs(self) -> int:
        return self.variance.shape[0]


def _msd_from_variance(variance: np.ndarray) -> np.ndarray:
    """MSD(t) = sigma^2(t) - sigma^2(0), using the fitted t=0 frame.

    A real measurement never knows the true initial width; it subtracts the
    fitted one.  Doing the same here keeps the simulated bias honest.
    """
    return variance - variance[:, [0]]


def _fit_realisation(model, x: np.ndarray, clean: np.ndarray,
                     noise_sigma: float, n_frames: int,
                     rng: np.random.Generator, noise_model=None,
                     cnr0: float | None = None):
    """Fit one noisy realisation of the clean profile stack.

    This is the single source of truth for per-realisation work, shared by the
    serial and parallel paths so they are guaranteed to behave identically --
    including the per-frame fallback in ``model.fit_variance`` that keeps a
    failed fit's broad guess rather than dropping it (dropping would delete the
    heavy right tail that carries the bias).

    ``noise_model`` (see :mod:`width_debias.noise`) swaps the noise law; when
    it is ``None`` a constant-sigma white-noise draw is used.
    """
    if noise_model is None:
        noisy = clean + rng.normal(0.0, noise_sigma, size=clean.shape)
    else:
        noisy = noise_model.sample(clean, cnr0, rng)
    variance = np.full(n_frames, np.nan)
    at_wall = np.zeros(n_frames, dtype=bool)
    ok = np.zeros(n_frames, dtype=bool)
    for f in range(n_frames):
        ff = model.fit_variance(x, noisy[f])
        variance[f] = ff.variance
        at_wall[f] = ff.at_wall
        ok[f] = ff.ok
    return variance, at_wall, ok


# --------------------------------------------------------------------------
# Parallel worker plumbing.  The clean profile stack is shipped once per
# worker process via the initialiser (not once per realisation), and each task
# carries only its run index and SeedSequence child.
# --------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(model, x, clean, noise_sigma, n_frames,
                 noise_model=None, cnr0=None) -> None:
    _WORKER.update(model=model, x=x, clean=clean,
                   noise_sigma=noise_sigma, n_frames=n_frames,
                   noise_model=noise_model, cnr0=cnr0)


def _run_one(task):
    run_index, seed_seq = task
    rng = np.random.default_rng(seed_seq)
    variance, at_wall, ok = _fit_realisation(
        _WORKER["model"], _WORKER["x"], _WORKER["clean"],
        _WORKER["noise_sigma"], _WORKER["n_frames"], rng,
        _WORKER["noise_model"], _WORKER["cnr0"],
    )
    return run_index, variance, at_wall, ok


def _resolve_workers(workers, n_runs: int) -> int:
    """Resolve the worker count.

    ``workers=None`` (the default) reads ``WIDTH_DEBIAS_WORKERS`` from the
    environment, falling back to serial (1) when it is unset -- so every script
    that routes through ``run_ensemble`` parallelises just by exporting that
    variable, with no code change.  ``workers<=0`` (or the env var set to 0)
    uses every available core.  The count is clamped to ``n_runs``.
    """
    if workers is None:
        env = os.environ.get("WIDTH_DEBIAS_WORKERS")
        workers = int(env) if env else 1
    if workers <= 0:
        workers = os.cpu_count() or 1
    return max(1, min(int(workers), n_runs))


def run_ensemble(exp: Experiment, cnr0: float, n_runs: int = 600,
                 seed: int = 0, model=None, workers=None,
                 noise_model=None) -> Ensemble:
    """Run ``n_runs`` independent noisy realisations of ``exp`` at ``cnr0``.

    ``model`` is a profile model from :mod:`width_debias.models`
    (default :class:`~width_debias.models.GaussianModel`).  The per-frame CNR
    is taken from the actual noise-free profile peak, so it is correct for any
    profile shape.

    ``noise_model`` (see :mod:`width_debias.noise`) swaps the measurement-
    noise law (white vs shot). ``None`` draws constant-sigma white noise; a
    model also supplies the
    per-frame CNR, which for shot noise differs from peak / RMS.

    ``workers`` controls process-level parallelism over realisations; see
    :func:`_resolve_workers`.  The result is independent of the worker count
    because each realisation draws from its own ``SeedSequence`` child.
    """
    if model is None:
        model = GaussianModel()
    noise_sigma = noise_sigma_for_cnr0(exp, cnr0)
    x = exp.x
    n_frames = exp.n_frames

    clean = model.noise_free_series(exp)            # (n_frames, n_pixels)
    if noise_model is None:
        cnr_frame = clean.max(axis=1) / noise_sigma     # peak / noise RMS
    else:
        cnr_frame = noise_model.cnr_frame(clean, cnr0)

    # One independent RNG stream per realisation, deterministic in ``seed`` and
    # order-independent -- so serial and parallel runs agree bit for bit.
    child_seeds = np.random.SeedSequence(seed).spawn(n_runs)

    variance = np.full((n_runs, n_frames), np.nan)
    at_wall = np.zeros((n_runs, n_frames), dtype=bool)
    ok = np.zeros((n_runs, n_frames), dtype=bool)

    n_workers = _resolve_workers(workers, n_runs)
    if n_workers == 1:
        for r in range(n_runs):
            rng = np.random.default_rng(child_seeds[r])
            variance[r], at_wall[r], ok[r] = _fit_realisation(
                model, x, clean, noise_sigma, n_frames, rng,
                noise_model, cnr0)
    else:
        # Chunk so each worker pulls a batch of realisations per round-trip,
        # amortising the inter-process hand-off over many fits.
        chunksize = max(1, n_runs // (n_workers * 4))
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(model, x, clean, noise_sigma, n_frames,
                      noise_model, cnr0),
        ) as pool:
            for r, v_row, w_row, o_row in pool.map(
                    _run_one, list(enumerate(child_seeds)), chunksize=chunksize):
                variance[r] = v_row
                at_wall[r] = w_row
                ok[r] = o_row

    msd = _msd_from_variance(variance)
    return Ensemble(exp=exp, cnr0=cnr0, noise_sigma=noise_sigma,
                    cnr_frame=cnr_frame, variance=variance, msd=msd,
                    at_wall=at_wall, ok=ok, model_name=model.name)
