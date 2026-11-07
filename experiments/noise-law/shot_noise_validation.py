"""Does the noise-bias correction hold across transport-microscopy noise laws?

The poster is not tied to one technique, so the correction must hold across the
noise regimes that different transport-microscopy methods present. Two bases,
anchored by real data on the pshare:

  WHITE / constant floor -- differential, pump-probe imaging (stroboSCAT,
    transient absorption). Measured on real stroboSCAT frames: the noise is set
    by a bright constant probe pedestal (~172 counts, noise ~1.4 counts), so it
    is effectively signal-independent; the signal's own shot is <~20% of the
    variance even at the brightest pixel. This is the law the calibration is
    built on.
  SHOT / Poisson -- photon-counting and direct emission imaging (time-correlated
    single-photon counting, PL imaging). Poisson by construction: variance =
    counts. Faithful low-count Poisson matters in the dim, low-CNR frames where
    the bias bites.

For each noise law this script asks the two questions an expert panel raised:

  CORRECTNESS -- with a calibration built under the SAME law, is the bias
    removed (as it is for white)?
  ROBUSTNESS  -- does the DEFAULT white-noise calibration, applied to data of
    another law, still de-bias D, or must the calibration be rebuilt?

Everything runs through the same audited calibrate/evaluate path (`validate.py`)
with only the noise law swapped (`width_debias.noise`), so the comparison
isolates the noise model. The metric is mean(D_hat)/D_true for the folded-normal
scheme across an initial-CNR sweep; uncorrected OLS is the raw bias to remove.

Run from the repository root (serial by default; set WIDTH_DEBIAS_WORKERS to
parallelise)::

    .venv/Scripts/python.exe scripts/shot_noise_validation.py

Sweep sizes are set by the SHOT_* environment variables below (small defaults
for a quick serial smoke run; raise for the full run).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias.noise import ShotNoise, WhiteNoise  # noqa: E402
from width_debias.simulate import Experiment  # noqa: E402
from width_debias.validate import calibrate_d_free, evaluate  # noqa: E402

GEOM = dict(sigma0=1.0, tau=1.0, half_window=6.0, n_pixels=241,
            n_frames=12, t_max=3.0)
D_TRUE = 0.02                       # sigma_0^2 / tau; L_D / FWHM_0 ~ 0.085

N_CAL = int(os.environ.get("SHOT_NCAL", 120))
N_EVAL = int(os.environ.get("SHOT_NEVAL", 120))
N_EVAL_GRID = int(os.environ.get("SHOT_NGRID", 8))
CAL_CNR0 = np.geomspace(2.0, 80.0, 30)
EVAL_CNR0 = np.geomspace(3.0, 60.0, N_EVAL_GRID)

# The noise laws to cover, each a base of a real technique.
NOISE_CASES = [
    ("white", WhiteNoise(),
     "differential / pump-probe (stroboSCAT, transient absorption): "
     "constant probe-pedestal noise"),
    ("shot", ShotNoise(read=0.0),
     "photon-limited direct imaging: signal-scaling noise (Gaussian shot)"),
    ("poisson", ShotNoise(read=0.0, poisson=True),
     "photon-counting (TCSPC, direct PL): true Poisson, faithful at low counts"),
]


def _mad(curve: np.ndarray) -> float:
    return float(np.nanmean(np.abs(curve - 1.0)))


def main() -> None:
    print("Folded-normal correction across transport-microscopy noise laws")
    print(f"  D_true = {D_TRUE} sigma_0^2/tau")
    print(f"  calibration: {CAL_CNR0.size} CNR_0 in [2, 80], {N_CAL} runs/cell")
    print(f"  evaluation:  {EVAL_CNR0.size} CNR_0 in [3, 60], {N_EVAL} runs/cell")
    print(f"  workers = {os.environ.get('WIDTH_DEBIAS_WORKERS', '(serial)')}")

    # One calibration per noise law (matched). The white calibration is the
    # default the method ships with; it doubles as the robustness reference.
    print("\nBuilding matched calibrations at D = 0 ...")
    calib = {}
    for key, model, _note in NOISE_CASES:
        print(f"  [{key}] ...", flush=True)
        calib[key] = calibrate_d_free(CAL_CNR0, n_runs=N_CAL, noise_model=model,
                                      **GEOM)
    white_cal = calib["white"]

    exp = Experiment(D=D_TRUE, **GEOM)
    rows = {}
    for key, model, note in NOISE_CASES:
        print(f"\nEvaluating noise law: {key}  ({note})", flush=True)
        r_match = evaluate(exp, EVAL_CNR0, calib[key], n_runs=N_EVAL,
                           noise_model=model)
        r_white = evaluate(exp, EVAL_CNR0, white_cal, n_runs=N_EVAL,
                           noise_model=model)
        rows[key] = dict(
            ols=r_match["ols"].mean_bias,
            folded_matched=r_match["subtract"].mean_bias,
            folded_white=r_white["subtract"].mean_bias,
        )

    print("\n" + "=" * 72)
    print("mean(D_hat)/D_true  (1.0 = unbiased)")
    print("=" * 72)
    for key, _model, note in NOISE_CASES:
        r = rows[key]
        print(f"\n[{key}]  {note}")
        print("  CNR_0 | OLS raw | folded (matched calib) | folded (default "
              "white calib)")
        for i, c in enumerate(EVAL_CNR0):
            print(f"  {c:6.1f} | {r['ols'][i]:7.2f} | "
                  f"{r['folded_matched'][i]:22.3f} | {r['folded_white'][i]:.3f}")

    print("\n" + "=" * 72)
    print("mean |bias - 1| over the sweep")
    print("=" * 72)
    print(f"  {'noise law':<10} {'OLS raw':>9} {'folded matched':>16} "
          f"{'folded white-calib':>20}")
    for key, _model, _note in NOISE_CASES:
        r = rows[key]
        print(f"  {key:<10} {_mad(r['ols']):>9.3f} "
              f"{_mad(r['folded_matched']):>16.3f} "
              f"{_mad(r['folded_white']):>20.3f}")
    print("\nReading: 'folded matched' small => the correction works under that "
          "law given a matched calibration (CORRECTNESS).")
    print("'folded white-calib' close to 'folded matched' => the default white "
          "calibration transfers to that law (ROBUSTNESS); far => recalibrate.")
    print("Real techniques: stroboSCAT / transient absorption ~ white; "
          "TCSPC / direct PL imaging ~ shot/Poisson.")


if __name__ == "__main__":
    main()
