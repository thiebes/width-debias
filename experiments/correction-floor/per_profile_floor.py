"""Per-profile floor of the noise-bias correction.

The cleanest test of where the correction works: isolated single profiles
at chosen per-profile CNR, with no diffusion, no decay, no frame indexing.
For each test CNR:

1. Take the dense D-free calibration of var_bias(CNR) (100 log-spaced CNR_0 in
   [2,100], 5000 runs/cell, the same for every shape) from
   ``dfree-calibration/dense_calibration_and_test.py``, which builds it once
   per shape and caches it.
2. Simulate isolated noisy profiles at sigma^2 = sigma_0^2 and the chosen
   per-profile CNR. Many of them (``N_RUNS_TEST`` per cell). Fit each.
3. Apply the calibration's var_bias(CNR_test) to get the corrected
   per-profile variance. Measure the residual mean
   E[sigma_tilde^2] - sigma^2_true. Its SE combines the test cell's Monte Carlo
   error with the lookup's own error at that CNR (``var_bias_se``): the
   subtracted value is itself an estimate, and leaving its error out flags
   calibration noise as a failure of the correction.
4. The floor: walking down from high CNR, the deepest CNR reached while every
   cell's residual is shown to be within a tolerance (|residual| + 2 combined
   SE <= tolerance). The test is one of equivalence because a cell whose error
   is large is not evidence that the correction works there; below CNR 1 the
   lookup's own error alone is 0.1 to 0.4 sigma_0^2. The floor therefore also
   depends on the calibration budget.

The tolerance is set by an application, not fixed in sigma_0^2: a residual
matters in proportion to the MSD it sits on. The floor is reported for
tolerances of 0.1 and 0.3 of the end-point MSD 2 D t_max of the default
experiment (t_max = 3 tau) at D = 0.005, 0.02, 0.1 sigma_0^2/tau, and for the
earlier fixed 0.01 sigma_0^2 for comparison. At D = 0.005 the end-point MSD is
0.03 sigma_0^2, so a fixed 0.01 sigma_0^2 is a third of the whole signal.

If the per-profile correction holds at CNR_test, the per-frame correction
in any application (a diffusion experiment, real data, whatever) holds at
that CNR -- because the correction operates per profile. The diffusion
experiment's slope fit then inherits the per-frame unbiasedness as a
linear consequence.

Seeds are offset by 10^6 per shape, so the four shapes see independent noise.
All variances and standard errors in sigma_0^2.

Usage::

    .venv/Scripts/python.exe experiments/correction-floor/per_profile_floor.py [MODEL]

The script runs its own process pool (all cores but one), so
WIDTH_DEBIAS_WORKERS has no effect on it.

MODEL is ``gaussian``, ``voigt``, ``generalized_gaussian:1.5`` or
``generalized_gaussian:4``. ``--rescore REPORT ...`` recomputes the floor lines
of written reports from their table rows. The report goes to repo-root
``output/correction-floor/per_profile_floor_<model>.txt``.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dfree-calibration"))

from width_debias import Experiment, model_from_spec, spec_tag  # noqa: E402
from width_debias.calibration import _single_frame_profile  # noqa: E402

# The dense calibration is shared with (and cached by) the dense test.
from dense_calibration_and_test import (CNR0_RANGE, N_CALIB_POINTS,  # noqa: E402
                                        N_RUNS_CALIB, dense_calibration, shape_offset)

# --- Test sweep ------------------------------------------------------------
TEST_CNR_RANGE = (0.05, 100.0)
N_TEST_POINTS = 30
N_RUNS_TEST = 50_000              # isolated-profile fits per test CNR
AMP = 1.0                          # peak amplitude convention for the test
SIGMA0 = 1.0                       # all variances in sigma_0^2 units

# --- Tolerances ------------------------------------------------------------
T_MAX = Experiment().times[-1]
TOL_D = (0.005, 0.02, 0.1)         # sigma_0^2 / tau
TOL_FRACTIONS = (0.1, 0.3)         # of the end-point MSD 2 D t_max
FIXED_TOL = 0.01                   # sigma_0^2, the earlier rule

OUT = Path(__file__).resolve().parents[2] / "output" / "correction-floor"
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)


# --- Workers ---------------------------------------------------------------
def _test_one(args):
    """Isolated single-profile MC at a chosen (CNR, sigma^2=1).

    Returns (cnr, mean_raw, mean_corrected, residual, eval_se, cal_se, n_used).
    """
    cnr_test, n_runs, seed, model_name, pixel_x, lookup_cnr, lookup_bias, lookup_se = args
    model = model_from_spec(model_name)
    clean = _single_frame_profile(model, pixel_x, SIGMA0 ** 2, AMP)
    noise_sigma = AMP / cnr_test
    rng = np.random.default_rng(seed)
    fits = np.empty(n_runs)
    for r in range(n_runs):
        noise = rng.normal(0.0, noise_sigma, size=clean.shape)
        ff = model.fit_variance(pixel_x, clean + noise)
        fits[r] = ff.variance
    good = np.isfinite(fits)
    n_used = int(good.sum())
    if n_used < 2:
        return (cnr_test, np.nan, np.nan, np.nan, np.nan, np.nan, n_used)
    raw = fits[good]
    mean_raw = float(raw.mean())
    eval_se = float(raw.std(ddof=1)) / np.sqrt(n_used)
    # The lookup as BiasLookup.var_bias / var_bias_se apply it (clamped
    # at the ends), passed as arrays so the workers need no model object.
    bias = float(np.interp(cnr_test, lookup_cnr, lookup_bias))
    cal_se = float(np.interp(cnr_test, lookup_cnr, lookup_se))
    mean_corr = mean_raw - bias
    return (cnr_test, mean_raw, mean_corr, mean_corr - SIGMA0 ** 2, eval_se, cal_se, n_used)


# --- Driver ---------------------------------------------------------------
def parallel_with_progress(worker, args_list, label):
    n = len(args_list)
    milestones = max(1, n // 10)
    done = 0
    results = [None] * n
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(worker, args): i
                   for i, args in enumerate(args_list)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % milestones == 0 or done == n:
                pct = 100.0 * done / n
                dt = time.time() - t0
                eta = dt * (n - done) / max(done, 1)
                print(f"    [{label}] {done}/{n} ({pct:3.0f}%)  "
                      f"elapsed {dt:5.0f}s  ETA {eta:4.0f}s", flush=True)
    return results


def floor_sweep(model_name, trunc, offset):
    cnr_test_grid = np.logspace(np.log10(TEST_CNR_RANGE[0]),
                                np.log10(TEST_CNR_RANGE[1]),
                                N_TEST_POINTS)
    pixel_x = Experiment().x
    args = [(float(c), N_RUNS_TEST, offset + 50000 + i, model_name, pixel_x,
             trunc.cnr_grid, trunc.varbias_grid, trunc.varbias_se_grid)
            for i, c in enumerate(cnr_test_grid)]
    print(f"  Test sweep: {N_TEST_POINTS} CNR in {TEST_CNR_RANGE}, "
          f"{N_RUNS_TEST} runs/cell ({N_WORKERS} workers)", flush=True)
    return parallel_with_progress(_test_one, args, "test")


def _fails(residual, eval_se, cal_se, tol) -> bool:
    """A cell passes only if its residual is shown to be within ``tol``.

    The test is one of equivalence: |residual| + 2 combined SE <= tol. A rule
    that fails a cell only when its residual is resolved (|z| > 2) passes every
    cell whose error is large, and below CNR 1 the lookup's own error is
    0.1 to 0.4 sigma_0^2, so that rule put the floor at the bottom of the grid.
    """
    return abs(residual) + 2.0 * np.hypot(eval_se, cal_se) > tol


def _floor(rows, tol) -> float | None:
    """Deepest CNR reached, walking down from the top, before the first failing cell."""
    floor = None
    for cnr, _, _, residual, eval_se, cal_se, _ in rows[::-1]:
        if _fails(residual, eval_se, cal_se, tol):
            break
        floor = cnr
    return floor


def _tolerances() -> list[tuple[str, float]]:
    tolerances = [(f"{frac:g} x 2 D t_max, D {d:g}", frac * 2.0 * d * T_MAX)
                  for d in TOL_D for frac in TOL_FRACTIONS]
    tolerances.append(("the earlier fixed rule", FIXED_TOL))
    return tolerances


def _floor_lines(rows) -> list[str]:
    lines = ["  Per-profile floor: deepest CNR reached, walking down from the top, while every "
             "cell has |residual| + 2 combined SE <= tolerance"]
    for label, tol in _tolerances():
        floor = _floor(rows, tol)
        text = f"{floor:.3f}" if floor is not None else "none (the top cell already fails)"
        lines.append(f"    tolerance {tol:7.4f} sigma_0^2 ({label}): CNR ~ {text}")
    return lines


def report(model_name, results, offset) -> str:
    rows = sorted((r for r in results if r is not None), key=lambda r: r[0])
    lines = [
        f"=== {model_name}: per-profile floor ===",
        f"  calibration: {N_CALIB_POINTS} CNR_0 in {CNR0_RANGE}, {N_RUNS_CALIB} runs/cell, "
        f"seeds {offset + 1000} + i; test: {N_RUNS_TEST} fits/cell, seeds {offset + 50000} + i",
        "  variances in sigma_0^2; tests at sigma^2 = 1; SE eval = test-cell Monte Carlo, "
        "SE cal = lookup error at that CNR; |z| uses both",
        f"  {'CNR':>8}{'n':>10}{'E[raw]':>10}{'E[corr]':>10}"
        f"{'resid':>11}{'SE eval':>9}{'SE cal':>9}{'|z|':>7}{'flag':>6}",
    ]
    for cnr, mean_raw, mean_corr, residual, eval_se, cal_se, n_used in rows:
        z = abs(residual) / np.hypot(eval_se, cal_se)
        flag = "<<<" if _fails(residual, eval_se, cal_se, FIXED_TOL) else ""
        lines.append(f"  {cnr:>8.3f}{n_used:>10d}{mean_raw:>10.4f}"
                     f"{mean_corr:>10.4f}{residual:>+11.4f}"
                     f"{eval_se:>9.4f}{cal_se:>9.4f}{z:>7.2f}{flag:>6}")
    lines.append("  (<<< marks cells not shown within the fixed 0.01 sigma_0^2: "
                 "|residual| + 2 combined SE > 0.01)")
    lines.append("")
    lines += _floor_lines(rows)
    return "\n".join(lines)


def _parse_row(line: str):
    """A table row as (cnr, raw, corr, residual, eval_se, cal_se, n), or None."""
    parts = line.split()
    if len(parts) < 8 or not parts[1].isdigit():
        return None
    try:
        cnr, n, raw, corr, res, se_e, se_c, _ = (float(p) for p in parts[:8])
    except ValueError:
        return None
    return cnr, raw, corr, res, se_e, se_c, int(n)


def rescore(path: Path) -> None:
    """Rebuild the floor lines of a written report from its table rows.

    A run that loaded an earlier version of this script wrote its floors under
    that version's rule; the rows (residual and both SEs) do not depend on the
    rule, so the floors are recomputed from them and the floor section replaced.
    """
    text = path.read_text(encoding="utf-8").splitlines()
    rows = sorted((r for r in map(_parse_row, text) if r is not None), key=lambda r: r[0])
    start = next(i for i, line in enumerate(text) if line.lstrip().startswith("Per-profile floor"))
    tail = [line for line in text[start:] if line.startswith("[total")]
    head = text[:start]
    for i, line in enumerate(head):
        row = _parse_row(line)
        if row is not None:
            base = line.rstrip()
            if base.endswith("<<<"):
                base = base[:-3].rstrip()
            flag = "<<<" if _fails(row[3], row[4], row[5], FIXED_TOL) else ""
            head[i] = f"{base}{flag:>6}"
        elif line.lstrip().startswith("(<<<"):
            head[i] = ("  (<<< marks cells not shown within the fixed 0.01 sigma_0^2: "
                       "|residual| + 2 combined SE > 0.01)")
    out = head + _floor_lines(rows) + [""] + tail
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(_floor_lines(rows)))


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--rescore":
        for name in sys.argv[2:]:
            rescore(Path(name))
        return
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
    offset = shape_offset(model_name)
    print(f"Per-profile floor sweep ({spec_tag(model_name)})")
    t0 = time.time()
    trunc = dense_calibration(model_name)
    print(f"  Calibration done [{time.time() - t0:.0f}s]")
    t1 = time.time()
    results = floor_sweep(model_name, trunc, offset)
    print(f"  Test sweep done [{time.time() - t1:.0f}s]")
    text = report(model_name, results, offset) + f"\n\n[total {(time.time() - t0) / 60:.1f} min]"
    print(text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"per_profile_floor_{spec_tag(model_name)}.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
