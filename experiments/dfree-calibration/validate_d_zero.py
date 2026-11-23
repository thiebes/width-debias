"""Validate the D=0 calibration: a 1-D lookup that does not reference the
application's D, against the simplest competitor.

The simplest D-free calibration is the existing 1-D one, run at *D = 0*
(``calibrate_d_free``): the calibration profiles all have variance sigma_0^2,
the lookup is keyed on per-frame CNR alone, and the application's D never
enters. Calibrate once at D = 0, evaluate at several D values, and check that
the corrected D is unbiased in every cell.

The competitor: most of the naive bias comes from the late, low-CNR frames
(the bias is A / CNR^2 and the CNR decays over the measurement), so dropping
every frame below a stricter cut removes much of it without a calibration, at
the cost of lever arm. Every cell is therefore scored under five arms, on the
same realizations:

* ``corrected, all frames``  -- subtract, every frame (the library default);
* ``corrected, CNR >= 1``    -- subtract, frames below the floor dropped;
* ``naive, CNR >= 3``        -- inverse_variance (the same weights, no
  subtraction), frames below per-frame CNR 3 dropped;
* ``naive, CNR >= 2``        -- the same with the cut at 2: more lever arm,
  more bias in the frames kept (3 was fixed before any result; 2 tests whether
  a tuned cut closes the gap);
* ``naive, all frames``      -- inverse_variance, every frame (reference).

The comparison reports bias and spread over realizations, the RMS error of a
single measurement, sqrt(bias^2 + spread^2), and that of the mean of N
measurements, sqrt(bias^2 + spread^2 / N), for N in ``N_REPLICATES``. A cut that
removes bias but widens the spread can lose on one measurement; averaging
shrinks the spread but not the bias, so a biased arm falls behind as
measurements are averaged. It also reports, where the bias is resolved, how
many measurements must be averaged before the bias exceeds the statistical
error of the mean (spread^2 / bias^2).

``--short-window`` runs a second geometry: a dim or fragile sample measured
only at early times (window 0.5 tau, CNR_0 2 to 12, D 0.02 to 0.2). Every frame
keeps 0.61 to 1 of CNR_0, so no cut can select good frames and the late frames
do not dominate the bias; the question is whether the correction is then the
only way to an unbiased D. Its seeds have their own root (+500,000).

``--spread-dims 1`` (or 2) evaluates with a peak that also falls as the profile
spreads (``Experiment.spread_dims``: 1 a summed profile, 2 a centre cut),
instead of with decay alone. The calibration stays at D = 0, where the profile
does not spread, so its frames are those of the decay-only bed. A falling peak
puts the late frames at a lower per-frame CNR than the calibration reached
(by sigma_0 / sigma for a summed profile), so the lookup would read its edge
value there; each geometry therefore names the CNR_0 points
(``spread_cal_extra``) that extend its calibration grid down, and the script
stops before calibrating if a frame kept at CNR >= 1 would still lie below the
calibrated range. The added points go after the existing ones, so every
existing point keeps its seed.

``--cal-runs N`` sets the calibration runs per CNR_0 apart from the evaluation
runs per cell. Each realization draws from its own child of the cell's seed,
so a calibration of N runs contains the runs of every smaller one on the same
seeds: a 5000-run lookup holds the 300 (or 2000) runs of the default.

Statistics, and why they take this form:

* Seeds are offset by 10^6 per shape (and calibration and evaluation seeds are
  disjoint), so the four shapes see independent noise; earlier tables used the
  same seeds for every shape.
* Each cell's deviation is judged against the combined SE: the evaluation SE
  (spread over sqrt(runs per cell)) and, for the corrected arms, the calibration SE
  propagated from the lookup's own Monte Carlo error (``calibration_D_se``).
  The naive arms subtract nothing, so they carry no calibration SE (their
  weights use the lookup's frame variance, which affects only efficiency).
  The script counts the cells beyond 2 combined SE; with one calibration per
  shape the cells share its error, so the count is indicative, not a binomial
  test.
* The pooled bias per D (over CNR_0) and per shape (over every cell) weights
  cells by inverse combined variance. Its evaluation part is independent
  between cells; its calibration part is one shared systematic, added as if
  fully correlated (sum of weight x cell calibration SE), an upper bound.
* The RMS comparison needs bias^2, but a cell's observed bias carries its own
  sampling noise, which adds the evaluation SE^2 to (D/D_ref - 1)^2 in
  expectation. Left in, that noise counts as bias and grows in weight as
  spread^2 / N shrinks (at N = 100 and 300 runs it is a third of the spread
  term). So bias^2 is estimated as (D/D_ref - 1)^2 - evaluation SE^2, floored
  at 0. For the corrected arms the observed bias also holds this calibration
  draw's error, so the estimate is of bias^2 + calibration SE^2: what a user
  with an independent calibration of the same budget would see.

    $env:WIDTH_DEBIAS_WORKERS = 0
    .venv/Scripts/python.exe experiments/dfree-calibration/validate_d_zero.py
        [--short-window] [--spread-dims {1,2}] [--cal-runs N] [spec ...]
    .venv/Scripts/python.exe experiments/dfree-calibration/validate_d_zero.py --report <npz> [...]

With no spec, runs the four validation shapes. Each shape's cell statistics go
to repo-root ``output/dfree-calibration/validate_d_zero<tag>_<shape>.npz``, its
report to the ``.txt`` beside it, and its calibration to ``..._lookup.npz``
(``BiasLookup.load`` reads it), where ``<tag>`` is ``_short`` for the short
window, then ``_spread<d>`` and ``_cal<N>`` when those options are set.
``--report`` rebuilds the reports from saved ``.npz`` files without rerunning,
so a table can be regenerated from statistics computed on another machine.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias import (Experiment, calibrate_d_free, model_from_spec,  # noqa: E402
                           run_ensemble, spec_tag)
from width_debias.fit import fit_D  # noqa: E402
from width_debias.weights import calibration_D_se, msd_and_weights  # noqa: E402

@dataclass(frozen=True)
class Geometry:
    """Measurement window and grid; calibration and evaluation share the window."""

    suffix: str
    t_max: float                 # tau
    cnr0_grid: np.ndarray
    d_values: tuple
    seed_offset: int
    n_runs: int                  # calibration runs per CNR_0, and evaluation runs per cell
    spread_cal_extra: tuple = ()  # CNR_0 points added to the calibration under a falling peak


# The standard bed: a 3 tau window, so the per-frame CNR falls about 20-fold.
# Its calibration reaches per-frame CNR 0.25, far below the floor, so a falling
# peak needs no extra calibration point.
STANDARD = Geometry("", 3.0, np.array([5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 90.0]),
                    (0.005, 0.01, 0.02, 0.05, 0.1), 0, 300)
# A dim or fragile sample measured only at early times: a 0.5 tau window, so
# every frame keeps 0.61 to 1 of CNR_0 and a cut cannot select good frames.
# D up to 0.2 keeps the diffusion length over the window at 0.06 to 0.19 FWHM.
# One measurement's spread reaches several D at CNR_0 2, so resolving a bias of
# a fraction of D takes 2000 runs per cell (evaluation SE about 0.1 D there).
# Its calibration reaches only per-frame CNR 1.21, where the decay-only bed's
# last frame at CNR_0 2 sits; a summed-profile peak takes that frame to 1.11
# at D = 0.2, so CNR_0 1.8 (reaching 1.09) is added under a falling peak.
SHORT_WINDOW = Geometry("_short", 0.5, np.array([2.0, 3.0, 4.0, 6.0, 8.0, 12.0]),
                        (0.02, 0.05, 0.1, 0.2), 500_000, 2000, (1.8,))

N_REPLICATES = (10, 100)     # for the RMS error of an averaged measurement
FLOOR = 1.0                  # the per-frame CNR below which frames are dropped
ARMS = (  # (scheme, per-frame CNR cut or None, label)
    ("subtract", None, "corrected, all frames"),
    ("subtract", FLOOR, "corrected, CNR >= 1"),
    ("inverse_variance", 3.0, "naive, CNR >= 3"),
    ("inverse_variance", 2.0, "naive, CNR >= 2"),
    ("inverse_variance", None, "naive, all frames"),
)
COMPETITORS = ("naive, CNR >= 3", "naive, CNR >= 2", "naive, all frames")
SHAPES = ("gaussian", "voigt", "generalized_gaussian:1.5", "generalized_gaussian:4")
OUT = Path(__file__).resolve().parents[2] / "output" / "dfree-calibration"


def _seeds(shape_index: int, geometry: Geometry) -> tuple[int, int]:
    """(calibration, evaluation) seed roots; sweeps add 0..6 and 1000 per D."""
    base = 10 ** 6 * shape_index + geometry.seed_offset
    return base + 1, base + 20000


def _evaluation_experiment(d_true: float, geometry: Geometry, spread_dims: int | None,
                           model) -> Experiment:
    """The evaluation bed; a falling peak needs the profile's Lorentzian half-width."""
    if spread_dims is None:
        return Experiment(D=d_true, t_max=geometry.t_max)
    return Experiment(D=d_true, t_max=geometry.t_max, spread_dims=spread_dims,
                      psf_hwhm=getattr(model, "gamma", 0.0))      # VoigtModel only


def _cnr_coverage(geometry: Geometry, cal_grid: np.ndarray, spread_dims: int | None,
                  model) -> tuple[float, float, float]:
    """(lowest calibrated CNR, lowest frame CNR kept at the floor, lowest frame CNR).

    Per-frame CNRs of the noise-free beds: run_ensemble reads the CNR from the
    noise-free peak, so these are the CNRs the lookup is evaluated at.
    """
    cal_exp = Experiment(D=0.0, t_max=geometry.t_max)
    lowest_calibrated = float(cal_grid.min() * np.min(cal_exp.amplitude) / cal_exp.amplitude0)
    frames = np.concatenate([
        cnr0 * exp.amplitude / exp.amplitude0
        for exp in (_evaluation_experiment(d, geometry, spread_dims, model)
                    for d in geometry.d_values)
        for cnr0 in geometry.cnr0_grid])
    return lowest_calibrated, float(frames[frames >= FLOOR].min()), float(frames.min())


def _cell(exp: Experiment, ens, model, scheme: str,
          cut: float | None) -> tuple[float, float, float, float]:
    """(mean D/D_ref, evaluation SE, mean calibration SE, spread) for one cell and arm.

    NaN where fewer than two realizations give a fit (a strict cut can leave a
    low-CNR_0 cell too few frames).
    """
    ratios, cal = [], []
    for r in range(ens.n_runs):
        msd, w = msd_and_weights(scheme, ens.variance[r], ens.cnr_frame, model)
        if cut is not None:
            w = np.where(ens.cnr_frame >= cut, w, 0.0)
        d = fit_D(exp.times, msd, w)
        if d.ok and np.isfinite(d.D):
            ratios.append(d.D / exp.D)
            if scheme == "subtract":
                cal.append(calibration_D_se(exp.times, msd, w, ens.cnr_frame, model,
                                            ens.variance[r]) / exp.D)
    ratios = np.asarray(ratios, float)
    if ratios.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    spread = float(ratios.std(ddof=1))
    return (float(ratios.mean()), spread / np.sqrt(ratios.size),
            float(np.nanmean(cal)) if cal else 0.0, spread)


def _pooled(bias: np.ndarray, eval_se: np.ndarray, cal_se: np.ndarray) -> tuple[float, float, float]:
    """Inverse-variance pooled bias, its evaluation SE, and its shared calibration SE."""
    ok = np.isfinite(bias) & np.isfinite(eval_se)
    bias, eval_se, cal_se = bias[ok], eval_se[ok], cal_se[ok]
    weight = 1.0 / (eval_se ** 2 + cal_se ** 2)
    weight /= weight.sum()
    return (float(np.sum(weight * bias)), float(np.sqrt(np.sum(weight ** 2 * eval_se ** 2))),
            float(np.sum(weight * cal_se)))


def _table(lines: list[str], title: str, values: np.ndarray, fmt: str,
           cnr0_grid: np.ndarray, d_values: np.ndarray) -> None:
    lines.append(title)
    lines.append("  D_true  " + "".join(f"{'CNR0=' + format(c, 'g'):>10}"
                                        for c in cnr0_grid))
    for k, d in enumerate(d_values):
        lines.append(f"  {d:6.4f}  " + "".join(format(v, fmt) for v in values[k]))
    lines.append("")


def _output_stem(spec: str, geometry: Geometry, spread_dims: int | None, cal_runs: int) -> str:
    """Output path without extension; a string, since a tag such as ``_p1.5`` is no suffix."""
    tag = geometry.suffix
    if spread_dims is not None:
        tag += f"_spread{spread_dims}"
    if cal_runs != geometry.n_runs:
        tag += f"_cal{cal_runs}"
    return str(OUT / f"validate_d_zero{tag}_{spec_tag(spec)}")


def run_shape(spec: str, geometry: Geometry, spread_dims: int | None = None,
              cal_runs: int | None = None) -> Path:
    """Calibrate, evaluate every cell under every arm, save the statistics, and report."""
    shape_index = SHAPES.index(spec) if spec in SHAPES else None
    if shape_index is None:
        sys.exit(f"unknown shape {spec!r}; expected one of {SHAPES}")
    model = model_from_spec(spec)
    cal_runs = geometry.n_runs if cal_runs is None else cal_runs
    cal_seed, eval_seed = _seeds(shape_index, geometry)
    grid, d_values = geometry.cnr0_grid, geometry.d_values
    extra = np.array(geometry.spread_cal_extra if spread_dims is not None else (), float)
    cal_grid = np.concatenate([grid, extra])
    lowest_calibrated, lowest_kept, lowest_frame = _cnr_coverage(geometry, cal_grid,
                                                                 spread_dims, model)
    if lowest_kept < lowest_calibrated:
        sys.exit(f"{spec}: a frame kept at CNR >= {FLOOR:g} reaches per-frame CNR "
                 f"{lowest_kept:.3f}, below the calibrated range (from {lowest_calibrated:.3f}); "
                 "extend the geometry's spread_cal_extra")
    stem = _output_stem(spec, geometry, spread_dims, cal_runs)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    trunc = calibrate_d_free(cal_grid, t_max=geometry.t_max, n_runs=cal_runs, seed=cal_seed,
                             model=model)
    # Saved whole (SE and frame-variance grids included), so a costly
    # calibration can be read again without rerunning it.
    trunc.save(Path(stem + "_lookup.npz"))
    print(f"{spec}: D=0 calibration {time.time() - t0:.0f} s", flush=True)

    # (arm, D, CNR_0, statistic): mean, evaluation SE, calibration SE, spread
    res = np.full((len(ARMS), len(d_values), grid.size, 4), np.nan)
    for k, d_true in enumerate(d_values):
        exp = _evaluation_experiment(d_true, geometry, spread_dims, model)
        for i, cnr0 in enumerate(grid):
            ens = run_ensemble(exp, float(cnr0), n_runs=geometry.n_runs,
                               seed=eval_seed + 1000 * k + i, model=model)
            for a, (scheme, cut, _) in enumerate(ARMS):
                res[a, k, i] = _cell(exp, ens, trunc, scheme, cut)
        print(f"{spec}: D = {d_true:g} {time.time() - t0:.0f} s", flush=True)

    np.savez(stem + ".npz", res=res, spec=spec, suffix=geometry.suffix,
             t_max=geometry.t_max, cnr0_grid=grid, d_values=np.array(d_values),
             cal_grid=cal_grid, n_extra=extra.size, cal_runs=cal_runs, n_runs=geometry.n_runs,
             cal_seed=cal_seed, eval_seed=eval_seed,
             spread_dims=0 if spread_dims is None else spread_dims,
             lowest_calibrated=lowest_calibrated, lowest_kept=lowest_kept,
             lowest_frame=lowest_frame, elapsed=time.time() - t0)
    return write_report(Path(stem + ".npz"))


def write_report(npz_path: Path) -> Path:
    """Build a shape's report from its saved cell statistics; write it beside the ``.npz``."""
    z = np.load(npz_path)
    res, spec = z["res"], str(z["spec"])
    grid, d_values, cal_grid = z["cnr0_grid"], z["d_values"], z["cal_grid"]
    n_extra, spread_dims = int(z["n_extra"]), int(z["spread_dims"])
    lines = [
        f"D=0 calibration applied to all D_true values ({spec}; window t_max = "
        f"{float(z['t_max']):g} tau)",
        f"calibrate_d_free, CNR0 {', '.join(f'{c:g}' for c in cal_grid)}, {int(z['cal_runs'])} "
        f"runs, seed {int(z['cal_seed'])}; evaluation {int(z['n_runs'])} runs per cell, seeds "
        f"{int(z['eval_seed'])} + 1000 k + i; library slope weights; every arm uses the same "
        "realizations",
    ]
    if spread_dims:
        kind = "a summed profile" if spread_dims == 1 else "a centre cut"
        lines.append(f"peak falls with spreading (spread_dims = {spread_dims}, {kind}); the D = 0 "
                     "calibration does not spread")
    lowest_calibrated = float(z["lowest_calibrated"])
    if n_extra:
        added = ", ".join(f"{c:g}" for c in cal_grid[-n_extra:])
        lines.append(f"calibration grid extended by CNR0 {added} (the last seeds), so that "
                     f"every frame kept at CNR >= {FLOOR:g} lies inside the calibrated range "
                     f"(lowest kept {float(z['lowest_kept']):.3f}, lowest calibrated "
                     f"{lowest_calibrated:.3f})")
    if float(z["lowest_frame"]) < lowest_calibrated * (1.0 - 1e-12):
        lines.append(f"the arms without a cut read the lookup's edge value for frames below "
                     f"CNR {lowest_calibrated:.3f} (lowest frame {float(z['lowest_frame']):.3f})")
    lines.append("")
    for a, (_, _, label) in enumerate(ARMS):
        mean, eval_se, cal_se, spread = (res[a, ..., j] for j in range(4))
        bias = mean - 1.0
        comb = np.sqrt(eval_se ** 2 + cal_se ** 2)
        z_score = bias / comb
        lines.append("=" * 78)
        lines.append(f"--- {label}")
        lines.append(f"cells beyond 2 combined SE: {int(np.nansum(np.abs(z_score) > 2))} of "
                     f"{int(np.isfinite(z_score).sum())} (one calibration per shape, so the cells "
                     "share its error)")
        lines.append("")
        _table(lines, "mean D/D_ref", mean, "10.3f", grid, d_values)
        _table(lines, "evaluation SE (spread / sqrt(runs per cell))", eval_se, "10.3f",
               grid, d_values)
        if np.nanmax(cal_se, initial=0.0) > 0:
            _table(lines, "calibration SE (propagated lookup error)", cal_se, "10.3f",
                   grid, d_values)
        _table(lines, "(D/D_ref - 1) / combined SE", z_score, "10.2f", grid, d_values)
        _table(lines, "spread of D/D_ref over realizations (one measurement)", spread, "10.3f",
               grid, d_values)
        if not np.isfinite(bias).any():
            lines.append("no cell leaves enough frames for a fit under this arm")
            lines.append("")
            continue
        lines.append("pooled D/D_ref - 1 (inverse combined variance; calibration SE as one "
                     "shared systematic, fully correlated, an upper bound)")
        for k, d in enumerate(d_values):
            if not np.isfinite(bias[k]).any():
                lines.append(f"  D {d:6.4f}: no cell leaves enough frames for a fit")
                continue
            p, pe, pc = _pooled(bias[k], eval_se[k], cal_se[k])
            lines.append(f"  D {d:6.4f}: {p:+.4f} +/- {pe:.4f} (evaluation) +/- {pc:.4f} "
                         f"(calibration); {p / np.hypot(pe, pc):+.2f} combined SE")
        p, pe, pc = _pooled(bias.ravel(), eval_se.ravel(), cal_se.ravel())
        lines.append(f"  all cells: {p:+.4f} +/- {pe:.4f} (evaluation) +/- {pc:.4f} "
                     f"(calibration); {p / np.hypot(pe, pc):+.2f} combined SE")
        lines.append("")

    # Comparison: RMS error of one measurement, sqrt(bias^2 + spread^2), and of
    # the mean of N measurements, sqrt(bias^2 + spread^2 / N), for each N in
    # N_REPLICATES: averaging shrinks the spread but not the bias, so a biased
    # arm falls behind as N grows. bias^2 has the evaluation SE^2 removed (see
    # the module docstring).
    bias_sq = np.maximum((res[..., 0] - 1.0) ** 2 - res[..., 1] ** 2, 0.0)
    labels = [lab for _, _, lab in ARMS]
    ref = labels.index("corrected, CNR >= 1")
    for n_avg, title in ((1, "one measurement"),
                         *((n, f"the mean of {n} measurements") for n in N_REPLICATES)):
        rmse = np.sqrt(bias_sq + res[..., 3] ** 2 / n_avg)
        lines.append("=" * 78)
        lines.append(f"--- comparison: RMS error of {title}, "
                     f"sqrt(bias^2 + spread^2 / {n_avg}), "
                     "bias^2 = max((D/D_ref - 1)^2 - evaluation SE^2, 0)")
        lines.append("")
        for a, (_, _, label) in enumerate(ARMS):
            _table(lines, f"[{label}]", rmse[a], "10.3f", grid, d_values)
        for competitor in COMPETITORS:
            _table(lines, f"ratio of RMS errors ({title}), {competitor} / corrected, CNR >= 1 "
                          "(> 1: the correction is better; nan: the cut leaves no fit)",
                   rmse[labels.index(competitor)] / rmse[ref], "10.2f", grid, d_values)

    # Measurements to average before the bias outweighs the statistical error of
    # their mean, spread^2 / bias^2; only where the bias is resolved, since an
    # unresolved bias gives an unbounded count.
    resolved = (np.abs(res[..., 0] - 1.0)
                > 2.0 * np.sqrt(res[..., 1] ** 2 + res[..., 2] ** 2)) & (bias_sq > 0)
    n_cross = np.where(resolved, res[..., 3] ** 2 / np.where(bias_sq > 0, bias_sq, 1.0), np.nan)
    lines.append("=" * 78)
    lines.append("--- measurements averaged before the bias exceeds the error of their "
                 "mean, spread^2 / bias^2")
    lines.append("(nan: bias within 2 combined SE, or no fit)")
    lines.append("")
    for a, (_, _, label) in enumerate(ARMS):
        _table(lines, f"[{label}]", n_cross[a], "10.0f", grid, d_values)
    lines.append(f"[total: {float(z['elapsed']):.0f} s]")
    report = "\n".join(lines)
    print(report)
    path = Path(str(npz_path).removesuffix(".npz") + ".txt")
    path.write_text(report + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("specs", nargs="*", default=list(SHAPES),
                        help="model specs (default: the four validation shapes)")
    parser.add_argument("--short-window", action="store_true",
                        help="the 0.5 tau window at CNR0 2 to 12")
    parser.add_argument("--spread-dims", type=int, choices=(1, 2),
                        help="evaluate with a peak that falls as the profile spreads")
    parser.add_argument("--cal-runs", type=int,
                        help="calibration runs per CNR0 (default: the evaluation runs per cell)")
    parser.add_argument("--report", nargs="+", type=Path, metavar="NPZ",
                        help="rebuild reports from saved statistics instead of running")
    args = parser.parse_args()
    if args.report:
        for path in args.report:
            write_report(path)
        return
    geometry = SHORT_WINDOW if args.short_window else STANDARD
    for spec in args.specs:
        run_shape(spec, geometry, args.spread_dims, args.cal_runs)


if __name__ == "__main__":
    main()
