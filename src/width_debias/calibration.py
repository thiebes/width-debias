"""Single-frame calibration cells and the 2-D per-frame calibration -- experimental.

The recommended D-free calibration is :func:`width_debias.calibrate_d_free`,
a 1-D lookup that runs the CNR-sweep calibration at ``D = 0``. It
captures the bias at ``sigma^2 = sigma_0^2``, which is the relevant region
for the low-D regime where the truncation bias bites (sigma^2(t) barely
changes across the measurement when D is small); see
``experiments/dfree-calibration/validate_d_zero.py`` for its validation.

:func:`calibrate_cells` supplies the single-frame Monte Carlo cells that
:func:`width_debias.calibrate_fisher` smooths. The 2-D calibration
(:func:`calibrate_cnr_sigma2`) is a more general construction kept
for future work: it generates **single-frame** noisy profiles on a
(CNR, variance) grid, so the lookup is keyed on both the per-frame CNR and
the observed width. The intended use is **heterogeneous transport** where
the per-realisation sigma^2 spread is large enough that the 1-D-at-sigma_0^2
approximation breaks down.

**Status: experimental.** In the low-CNR cells (CNR < 1), the Monte Carlo
noise per cell dominates the per-cell bias estimate and the resulting lookup
is too noisy to use without substantial additional sample budget (n_runs
>~ 5000 per cell, plus post-hoc smoothing). The fixed-point inversion in
:meth:`BiasLookup.var_bias` compensates for the lookup-at-biased-sigma^2
issue but cannot rescue the underlying MC noise.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bias_lookup import _resolve_npz_path
from .models import MODELS, GaussianModel
from .simulate import Experiment


@dataclass
class CalibrationGrid:
    """A 2-D lookup of ``var_bias`` and ``rel_variance`` over (CNR, variance).

    Built by :func:`calibrate_cnr_sigma2`. ``var_bias[i, j]`` is the empirical
    ``E[sigma_hat^2] - sigma^2_true`` from ``n_runs`` fits at
    ``cnr_grid[i]`` and ``variance_grid[j]``. ``rel_variance[i, j]`` is the
    sample variance of those fits -- the per-frame ``Var(sigma_hat^2)``.

    The grids are 1-D and ascending; bilinear interpolation between cells is
    handled by :class:`~width_debias.bias_lookup.BiasLookup`.

    Also carries the calibration's pixel grid (``pixel_x``) so saved
    calibrations can be reloaded without re-deriving the geometry the lookup
    is conditional on. Applications must match this pixel grid (in absolute
    units) for the lookup to apply.
    """

    cnr_grid: np.ndarray         # (n_cnr,), ascending
    variance_grid: np.ndarray    # (n_var,), ascending
    var_bias: np.ndarray         # (n_cnr, n_var)
    rel_variance: np.ndarray     # (n_cnr, n_var)
    n_runs: int
    model_name: str
    pixel_x: np.ndarray | None = None    # the calibration's spatial axis
    amplitude: float = 1.0
    noise_model: str = "gaussian"        # "gaussian" (additive) or "poisson"
    background: float = 0.0              # Poisson baseline count floor
    read_noise: float = 0.0             # Gaussian read-noise std (counts)
    gain: float = 1.0                   # photon-transfer slope (shot var/signal)

    def save(self, path: str | os.PathLike) -> None:
        """Save the calibration to a single ``.npz`` archive."""
        np.savez_compressed(
            str(path),
            cnr_grid=self.cnr_grid,
            variance_grid=self.variance_grid,
            var_bias=self.var_bias,
            rel_variance=self.rel_variance,
            pixel_x=(np.asarray(self.pixel_x, float)
                     if self.pixel_x is not None else np.array([])),
            n_runs=np.array([self.n_runs]),
            model_name=np.array([self.model_name]),
            amplitude=np.array([self.amplitude]),
            noise_model=np.array([self.noise_model]),
            background=np.array([self.background]),
            read_noise=np.array([self.read_noise]),
            gain=np.array([self.gain]),
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "CalibrationGrid":
        """Load a calibration previously saved with :meth:`save`."""
        with np.load(_resolve_npz_path(path)) as z:
            px = z["pixel_x"]
            return cls(
                cnr_grid=z["cnr_grid"],
                variance_grid=z["variance_grid"],
                var_bias=z["var_bias"],
                rel_variance=z["rel_variance"],
                n_runs=int(z["n_runs"][0]),
                model_name=str(z["model_name"][0]),
                pixel_x=px if px.size > 0 else None,
                amplitude=float(z["amplitude"][0]),
                noise_model=(str(z["noise_model"][0])
                             if "noise_model" in z.files else "gaussian"),
                background=(float(z["background"][0])
                            if "background" in z.files else 0.0),
                read_noise=(float(z["read_noise"][0])
                            if "read_noise" in z.files else 0.0),
                gain=(float(z["gain"][0]) if "gain" in z.files else 1.0),
            )


def _single_frame_profile(model, pixel_x: np.ndarray, variance: float,
                          amplitude: float) -> np.ndarray:
    """A noise-free single-frame profile on the given pixel grid.

    Constructs a one-frame :class:`Experiment` whose `D=0`, `t_max=0`, and
    `sigma0 = sqrt(variance)`, so ``true_variance[0] = variance`` exactly, then
    asks the profile model for its noise-free series. The pixel grid is set by
    inverting :attr:`Experiment.x` so that the experiment's x array matches
    ``pixel_x``. ``pixel_x`` is assumed symmetric about zero.
    """
    sigma0 = float(np.sqrt(variance))
    x_max = float(np.max(np.abs(pixel_x)))
    half_window = x_max / sigma0
    exp = Experiment(
        sigma0=sigma0, D=0.0, amplitude0=float(amplitude), mean=0.0,
        half_window=half_window, n_pixels=int(pixel_x.size),
        n_frames=1, t_max=0.0,
    )
    return model.noise_free_series(exp)[0]


def _cell_fits(model, pixel_x: np.ndarray, cnr: float, variance: float,
               amplitude: float, n_runs: int, seed, noise_model: str,
               background: float, read_noise: float, gain: float) -> np.ndarray:
    """Fitted variances of ``n_runs`` noisy single-frame profiles at one cell.

    The noise models are described in :func:`_cell_worker`. Every fit is kept,
    including a failed fit's fallback value, as the application keeps it.
    """
    rng = np.random.default_rng(seed)
    fits = np.empty(n_runs)
    if noise_model == "poisson":
        b = float(background)
        rn = float(read_noise)
        g = float(gain)
        base_noise = np.sqrt(g * b + rn * rn)   # shot(gain*signal) + read floor
        peak = float(cnr) * base_noise          # CNR = peak / baseline noise
        clean = _single_frame_profile(model, pixel_x, variance, peak)
        edge = max(3, pixel_x.size // 8)
        for k in range(n_runs):
            noisy = g * rng.poisson((clean + b) / g).astype(float)  # var = g*signal
            if rn > 0:
                noisy += rng.normal(0.0, rn, size=clean.shape)
            base = np.median(np.concatenate([noisy[:edge], noisy[-edge:]]))
            ff = model.fit_variance(pixel_x, noisy - base)
            fits[k] = ff.variance
    else:
        clean = _single_frame_profile(model, pixel_x, variance, amplitude)
        noise_sigma = float(amplitude) / float(cnr)
        for r in range(n_runs):
            noise = rng.normal(0.0, noise_sigma, size=clean.shape)
            ff = model.fit_variance(pixel_x, clean + noise)
            fits[r] = ff.variance
    return fits


def _cell_worker(args):
    """Top-level worker (picklable) for a single (CNR, variance) cell.

    The bias estimate filters out broken-fit outliers (sigma_hat^2 above
    ``filter_threshold``) before computing the mean. The rel_variance
    estimate is computed over all finite fits -- broken cells should report
    high rel_variance so the downstream slope fit can downweight frames
    that land there.

    Two noise models. ``gaussian`` adds zero-mean white noise of standard
    deviation ``amplitude / cnr`` to a clean profile of peak ``amplitude`` --
    the idealised additive-noise regime (signal-independent), kept for the
    simulation studies.

    ``poisson`` is the realistic detector model following the photon-transfer
    relation ``var = gain * signal + read_noise^2``: per pixel,
    ``gain * Poisson((clean_counts + background) / gain) + Normal(0, read_noise)``
    -- shot noise that rides on the signal (slope ``gain``) plus a constant
    Gaussian read/electronic floor. The two are independent, not a fraction:
    ``gain = 1, read_noise = 0`` is pure photon counting (TCSPC/APD); ``gain > 1``
    is an excess-noise/conversion factor; a large ``read_noise`` is
    read-noise-dominated (camera-like). ``gain`` and ``read_noise`` are detector
    constants (estimate them once from fft-cnr's photon-transfer fit on a bright,
    high-dynamic-range frame). The baseline standard deviation is
    ``sqrt(gain * background + read_noise^2)``, so a requested per-frame ``cnr``
    fixes the peak counts at ``cnr * baseline_noise`` (CNR = peak / baseline
    noise). The baseline is estimated from the outer pixels and subtracted
    before the fit, exactly as the application does, so the bias the lookup
    records matches the data's.
    """
    (i, j, cnr, variance, pixel_x, amplitude, n_runs, seed,
     model_name, model_kwargs, filter_threshold, noise_model, background,
     read_noise, gain) = args
    model = MODELS[model_name](**(model_kwargs or {}))
    fits = _cell_fits(model, pixel_x, cnr, variance, amplitude, n_runs, seed,
                      noise_model, background, read_noise, gain)
    good = np.isfinite(fits)
    if good.sum() > 1:
        # rel_variance uses all finite fits (broken fits should drive this up).
        rv = float(np.var(fits[good], ddof=1))
        # var_bias filters out broken fits (sigma_hat^2 above threshold).
        keep = good & (fits < filter_threshold)
        if keep.sum() > 1:
            vb = float(fits[keep].mean() - variance)
        else:
            vb = float("nan")
    else:
        vb = float("nan")
        rv = float("nan")
    return i, j, vb, rv


def calibrate_cnr_sigma2(model=None, *, pixel_x: np.ndarray | None = None,
                         variance_grid: np.ndarray | None = None,
                         sigma_grid: np.ndarray | None = None,
                         cnr_grid: np.ndarray | None = None,
                         amplitude: float = 1.0, n_runs: int = 300,
                         seed: int = 0,
                         n_workers: int | None = None,
                         model_kwargs: dict | None = None,
                         filter_threshold: float | None = None,
                         noise_model: str = "gaussian",
                         background: float = 0.0,
                         read_noise: float = 0.0,
                         gain: float = 1.0,
                         verbose: bool = False) -> CalibrationGrid:
    """Build the joint (CNR, sigma^2) calibration of the fit bias.

    The lookup has two scalar parameter axes -- per-frame CNR and the true
    fitted variance sigma^2 -- producing a surface ``bias(CNR, sigma^2)``.
    The "2D" refers to those two parameter axes (not anything spatial or
    temporal); the calibration's Monte Carlo is static, with the data the
    lookup will be applied to remaining 1-D spatial profiles measured over
    time.

    The lookup needs a key for the unknown true sigma^2. Keying each frame on
    its own fitted sigma_hat^2 under-corrects when sigma^2 grows several-fold
    over the measurement; see :meth:`BiasLookup.var_bias` ("Choice of
    key") before applying this calibration to fast-spreading data.

    Cells are evaluated in parallel via :class:`ProcessPoolExecutor`. Each
    cell's RNG is seeded from a :class:`numpy.random.SeedSequence` spawn so
    the calibration is reproducible irrespective of completion order.

    Parameters
    ----------
    model : profile model instance (see :mod:`width_debias.models`).
        Defaults to :class:`~width_debias.models.GaussianModel`. The
        model's name is recorded; its class and ``model_kwargs`` are used to
        reconstruct the model inside worker processes.
    pixel_x : the spatial axis on which the fit will run. Must match the
        application's pixel grid for the calibration to apply. Defaults to
        the standard :class:`Experiment` grid.
    variance_grid : the true sigma^2 values to span. Mutually exclusive with
        ``sigma_grid``. Defaults to ``np.linspace(0.6, 3.5, 12)`` (in units
        consistent with ``pixel_x``).
    sigma_grid : alternative way to specify the variance axis -- linear-in-
        sigma spacing, which gives denser sampling at small sigma^2 (where
        the bias gradient is largest near the wall at zero). Internally
        converted to ``variance_grid = sigma_grid ** 2``. Pass either this
        or ``variance_grid``, not both.
    cnr_grid : the per-frame CNR values to span. Defaults to log-spaced
        ``np.logspace(np.log10(0.15), np.log10(120), 22)``.
    amplitude : the noise-free peak amplitude. Only its ratio to noise (=
        CNR) affects the bias, so the value is conventional; left at 1.0.
    n_runs : Monte-Carlo runs per grid cell.
    seed : root random seed; each cell gets an independent stream.
    n_workers : number of worker processes. Defaults to ``os.cpu_count() - 1``.
        Set to 1 to run serially.
    model_kwargs : kwargs to reconstruct the model in every cell, serial or
        parallel (e.g. ``{"gamma": 0.01}`` for :class:`VoigtModel`,
        ``{"shape": 1.5}`` for :class:`GeneralizedGaussianModel`). Each cell
        re-instantiates the model from its name and these kwargs, so the
        passed instance's own settings (shape exponent, ``gamma``,
        ``max_var``) are lost unless they are repeated here.
    filter_threshold : per-cell, when computing the mean bias, discard any
        individual fit whose sigma_hat^2 exceeds this threshold. These are
        broken fits pegged at the fit's upper bound; their contribution to
        the mean pollutes the bias estimate at low-CNR cells. Defaults to
        ``5 * variance_grid.max()``, which catches anything near var_max
        without touching legitimate fits inside the calibration's grid.
        rel_variance is computed without the filter so broken cells
        correctly report high spread (downstream slope fit then
        downweights frames at those cells).
    verbose : print per-completion progress every ~10% of cells.

    Returns
    -------
    :class:`CalibrationGrid`
        The 2-D ``var_bias`` and ``rel_variance`` lookups plus the
        calibration's geometry (``pixel_x``, ``amplitude``) and metadata.
    """
    if model is None:
        model = GaussianModel()
    if pixel_x is None:
        pixel_x = Experiment().x
    if variance_grid is not None and sigma_grid is not None:
        raise ValueError("pass either variance_grid or sigma_grid, not both")
    if sigma_grid is not None:
        variance_grid = np.asarray(sigma_grid, float) ** 2
    elif variance_grid is None:
        variance_grid = np.linspace(0.6, 3.5, 12)
    if cnr_grid is None:
        cnr_grid = np.logspace(np.log10(0.15), np.log10(120.0), 22)
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)

    pixel_x = np.asarray(pixel_x, float)
    cnr_grid = np.asarray(cnr_grid, float)
    variance_grid = np.asarray(variance_grid, float)

    if filter_threshold is None:
        filter_threshold = 5.0 * float(variance_grid.max())

    n_cnr = cnr_grid.size
    n_var = variance_grid.size
    var_bias = np.full((n_cnr, n_var), np.nan)
    rel_var = np.full((n_cnr, n_var), np.nan)

    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_cnr * n_var)

    args_list = []
    for i, cnr in enumerate(cnr_grid):
        for j, variance in enumerate(variance_grid):
            args_list.append((
                i, j, float(cnr), float(variance), pixel_x,
                float(amplitude), int(n_runs), child_seeds[i * n_var + j],
                model.name, model_kwargs or {},
                float(filter_threshold), str(noise_model), float(background),
                float(read_noise), float(gain),
            ))

    total = len(args_list)
    milestones = max(1, total // 10)
    done = 0
    t0 = time.time()

    if n_workers == 1:
        for args in args_list:
            i, j, vb, rv = _cell_worker(args)
            var_bias[i, j] = vb
            rel_var[i, j] = rv
            done += 1
            if verbose and (done % milestones == 0 or done == total):
                _progress(done, total, t0)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_cell_worker, args) for args in args_list]
            for fut in as_completed(futures):
                i, j, vb, rv = fut.result()
                var_bias[i, j] = vb
                rel_var[i, j] = rv
                done += 1
                if verbose and (done % milestones == 0 or done == total):
                    _progress(done, total, t0)

    return CalibrationGrid(
        cnr_grid=cnr_grid, variance_grid=variance_grid,
        var_bias=var_bias, rel_variance=rel_var,
        n_runs=int(n_runs), model_name=model.name,
        pixel_x=pixel_x, amplitude=float(amplitude),
        noise_model=str(noise_model), background=float(background),
        read_noise=float(read_noise), gain=float(gain),
    )


@dataclass
class CalibrationCells:
    """Single-frame Monte Carlo cells at one true width, over per-frame CNR.

    Built by :func:`calibrate_cells`. ``mean_bias[k]`` is the unfiltered
    ``E[sigma_hat^2] - variance_cal`` over the finite fits of cell k and
    ``bias_sd[k]`` their standard deviation, so ``bias_sd**2`` is the frame's
    ``Var(sigma_hat^2)``. The cells are the raw material that
    :meth:`~width_debias.bias_lookup.BiasLookup.from_cells` smooths.
    """

    cnr: np.ndarray
    mean_bias: np.ndarray
    bias_sd: np.ndarray
    n_finite: np.ndarray
    n_runs: np.ndarray
    variance_cal: float
    pixel_x: np.ndarray
    model_spec: str
    noise_model: str = "gaussian"
    seed: int = 0

    def save(self, path: str | os.PathLike) -> None:
        """Save the cells to a single ``.npz`` archive."""
        np.savez_compressed(
            str(path), cnr=self.cnr, mean_bias=self.mean_bias,
            bias_sd=self.bias_sd, n_finite=self.n_finite, n_runs=self.n_runs,
            variance_cal=np.array([self.variance_cal]), pixel_x=self.pixel_x,
            model_spec=np.array([self.model_spec]),
            noise_model=np.array([self.noise_model]),
            seed=np.array([self.seed]))

    @classmethod
    def load(cls, path: str | os.PathLike) -> "CalibrationCells":
        """Load cells previously saved with :meth:`save`."""
        with np.load(_resolve_npz_path(path)) as z:
            return cls(cnr=z["cnr"], mean_bias=z["mean_bias"], bias_sd=z["bias_sd"],
                       n_finite=z["n_finite"], n_runs=z["n_runs"],
                       variance_cal=float(z["variance_cal"][0]),
                       pixel_x=z["pixel_x"], model_spec=str(z["model_spec"][0]),
                       noise_model=str(z["noise_model"][0]), seed=int(z["seed"][0]))


def _describe_model(model) -> str:
    """Model class and settings, recorded because the deep-CNR bias depends on them."""
    return f"{type(model).__name__}({vars(model)})"


def _cells_worker(args):
    """Top-level worker (picklable): reduce one cell's fits to its moments."""
    k, model, pixel_x, cnr, variance, n_runs, seed = args
    fits = _cell_fits(model, pixel_x, cnr, variance, 1.0, n_runs, seed,
                      "gaussian", 0.0, 0.0, 1.0)
    good = fits[np.isfinite(fits)]
    if good.size < 2:
        return k, float("nan"), float("nan"), int(good.size)
    return k, float(good.mean() - variance), float(good.std(ddof=1)), int(good.size)


def calibrate_cells(cnr_grid: np.ndarray, *, model=None,
                    pixel_x: np.ndarray | None = None,
                    variance_cal: float = 1.0,
                    n_runs: int | np.ndarray = 300, seed: int = 0,
                    workers: int | None = None,
                    verbose: bool = False) -> CalibrationCells:
    """Single-frame Monte Carlo of the fitted-variance bias over per-frame CNR.

    Each cell fits ``n_runs`` (a scalar, or one count per cell to spend more
    runs where the bias is noisiest) white-noise profiles at the true variance
    ``variance_cal`` and peak CNR ``cnr_grid[k]``. Unlike
    :func:`calibrate_cnr_sigma2`, no outlier filter is applied: the mean is the
    raw mean the application's subtraction must match, as in
    :func:`~width_debias.validate.calibrate_d_free`. Cells are seeded from one
    :class:`numpy.random.SeedSequence`, so the result depends only on
    ``(seed, n_runs)``; ``workers`` follows ``WIDTH_DEBIAS_WORKERS`` like
    :func:`~width_debias.montecarlo.run_ensemble`.
    """
    from .montecarlo import _resolve_workers

    if model is None:
        model = GaussianModel()
    if pixel_x is None:
        pixel_x = Experiment().x
    pixel_x = np.asarray(pixel_x, float)
    cnr_grid = np.asarray(cnr_grid, float)
    runs_per_cell = np.broadcast_to(np.asarray(n_runs, int), cnr_grid.shape).copy()
    child_seeds = np.random.SeedSequence(seed).spawn(cnr_grid.size)
    tasks = [(k, model, pixel_x, float(c), float(variance_cal),
              int(runs_per_cell[k]), child_seeds[k])
             for k, c in enumerate(cnr_grid)]

    n_workers = _resolve_workers(workers, len(tasks))
    t0 = time.time()
    if n_workers == 1:
        results = [_cells_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_cells_worker, tasks))
    if verbose:
        print(f"    [calibrate_cells] {len(tasks)} cells, {int(runs_per_cell.sum())} "
              f"fits in {time.time() - t0:.0f}s ({n_workers} workers)", flush=True)

    _, mean_bias, bias_sd, n_finite = (np.array(column) for column in zip(*results))
    return CalibrationCells(cnr=cnr_grid, mean_bias=mean_bias, bias_sd=bias_sd,
                            n_finite=n_finite, n_runs=runs_per_cell,
                            variance_cal=float(variance_cal), pixel_x=pixel_x,
                            model_spec=_describe_model(model), seed=int(seed))


def _progress(done: int, total: int, t0: float) -> None:
    """Print one progress line: pct, elapsed, ETA."""
    pct = 100.0 * done / total
    dt = time.time() - t0
    eta = dt * (total - done) / max(done, 1)
    print(f"    [calibrate_cnr_sigma2] {done}/{total} ({pct:3.0f}%)  "
          f"elapsed {dt:6.0f}s  ETA {eta:6.0f}s", flush=True)
