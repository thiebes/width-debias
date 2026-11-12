"""Why the covariance SE of D runs about 0.03 narrow (coverage_check, panel b).

``coverage_check.py`` finds ``D_se / SD(D_hat)`` at 0.96 to 0.98 at every
CNR_0 from 21 up, so the 0.95 covariance interval covers about 0.944. Both the
slope weights and ``D_se`` come from the D-free lookup's per-frame
Var(sigma_hat^2), which is calibrated at the frame-0 width (D = 0). In the
diffusing experiment the width grows to sigma^2 = 1.12 sigma_0^2 by t = 3 tau,
and at a fixed per-frame CNR the variance of sigma_hat^2 grows about as
sigma^3: Var(sigma_hat) is proportional to sigma in 1-D on a fixed pixel grid,
and Var(sigma_hat^2) = 4 sigma^2 Var(sigma_hat). The lookup therefore
understates the late frames, which carry the most leverage on the slope, and
``D_se`` comes out too small.

For a few CNR_0 this script reports, from one ensemble each:

* per frame, the measured Var(sigma_hat^2) over the lookup's
  ``frame_variance``, beside (sigma^2 / sigma_0^2)^1.5;
* ``D_se / SD`` three ways: with SD from the weighted-least-squares sandwich
  under the sigma^3 scaling alone (no Monte Carlo noise), from the measured
  per-frame variances alone (diagonal), and realised over the ensemble (with
  its SE, about 0.013 at 3000 runs);
* the exponent k of measured/lookup = (sigma^2 / sigma_0^2)^k, fitted over every
  frame with per-frame CNR >= 1 (below that the calibration's own floor sets
  the ratio).

If the sigma^3 prediction matches the coverage table's pooled narrowness and k
is near 1.5, the cause is the per-frame variance (the width), not correlation
between frames. The single-cell realised values are too noisy to resolve 0.03
on their own; the pooled check is ``coverage_check.txt`` (8000 runs per cell).

Run from the repository root::

    .venv/Scripts/python.exe experiments/coverage-interval/narrowness_attribution.py

A few minutes serial. Reuses the shared D-free calibration through
``coverage_check.load_calibration``. Outputs ``output/narrowness_attribution.txt``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias import Experiment, GaussianModel  # noqa: E402
from width_debias.montecarlo import run_ensemble  # noqa: E402
from coverage_check import (CAL_PATH, D_TRUE, GEOMETRY, OUTPUT,  # noqa: E402
                            load_calibration)

CNR0_CELLS = (10.3, 34.3, 90.0)
N_RUNS = 3000
SEED = 555000              # cell k uses SEED + 1000 k; independent of coverage_check


def attribution_at(cnr0: float, trunc, seed: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-frame variance ratios and the predicted against realised slope SD.

    Returns the report lines, and ln(sigma^2) and ln(measured/lookup) for the
    frames with per-frame CNR >= 1 (the input to the exponent fit).
    """
    exp = Experiment(D=D_TRUE, **GEOMETRY)
    ens = run_ensemble(exp, cnr0, n_runs=N_RUNS, seed=seed, model=GaussianModel())
    t = exp.times
    cnr = ens.cnr_frame
    fv_lookup = trunc.frame_variance(cnr)
    fv_measured = np.nanvar(ens.variance, axis=0, ddof=1)
    v_true = GEOMETRY["sigma0"] ** 2 + 2.0 * D_TRUE * t

    # The deployed fit is a weighted line through every frame (frame 0
    # included), so the slope of MSD = v - v_0 equals the slope of v; its
    # operator row a gives Var(slope) = a C a^T for any covariance C.
    w = 1.0 / fv_lookup
    x = np.column_stack([np.ones_like(t), t])
    xtwx_inv = np.linalg.inv(x.T @ (w[:, None] * x))
    a = (xtwx_inv @ (x.T * w))[1]
    d_se = float(np.sqrt(xtwx_inv[1, 1])) / 2.0
    finite = np.all(np.isfinite(ens.variance), axis=1)
    rows = ens.variance[finite]
    scale = (v_true / GEOMETRY["sigma0"] ** 2) ** 1.5
    sd_scaled = float(np.sqrt(a @ (fv_lookup * scale * a))) / 2.0
    sd_diag = float(np.sqrt(a @ (fv_measured * a))) / 2.0
    sd_real = float(np.std(rows @ a / 2.0, ddof=1))
    narrow_real_se = (d_se / sd_real) / np.sqrt(2.0 * (rows.shape[0] - 1))
    corr = np.corrcoef(rows, rowvar=False) - np.eye(t.size)
    ratio = fv_measured / fv_lookup

    lines = [f"CNR0 {cnr0:g}  ({N_RUNS} runs, seed {seed}; "
             f"{int((~finite).sum())} non-finite realisations dropped)",
             "     t/tau  sigma^2  per-frame CNR  measured/lookup  (sigma^2)^1.5"]
    for i in range(t.size):
        lines.append(f"    {t[i]:6.2f}  {v_true[i]:7.3f}  {cnr[i]:13.2f}  "
                     f"{ratio[i]:15.3f}  {scale[i]:13.3f}")
    lines += [
        f"  D_se from the lookup weights           {d_se:.5f} sigma0^2/tau",
        f"  D_se/SD, sigma^3 scaling of the lookup {d_se / sd_scaled:.3f}  (no Monte Carlo noise)",
        f"  D_se/SD, measured per-frame variances  {d_se / sd_diag:.3f}",
        f"  D_se/SD, realised                      {d_se / sd_real:.3f} +/- {narrow_real_se:.3f}",
        f"  largest |correlation| between frames   {np.max(np.abs(corr)):.3f}",
        "",
    ]
    keep = cnr >= 1.0
    return lines, np.log(v_true[keep] / GEOMETRY["sigma0"] ** 2), np.log(ratio[keep])


def main() -> None:
    trunc = load_calibration()
    lines = [
        "Narrowness attribution -- why D_se / SD(D_hat) is about 0.97",
        f"D_true = {D_TRUE} sigma0^2/tau, Gaussian, standard geometry; slope weights "
        "1/frame_variance from the D-free lookup",
        f"calibration: shared cache {CAL_PATH.name} (coverage_check.load_calibration)",
        "measured/lookup = ensemble Var(sigma_hat^2) / lookup frame_variance at the "
        "frame's CNR",
        "",
    ]
    log_v, log_ratio = [], []
    for k, cnr0 in enumerate(CNR0_CELLS):
        print(f"CNR0 {cnr0:g} ...", flush=True)
        cell_lines, x, y = attribution_at(cnr0, trunc, SEED + 1000 * k)
        lines += cell_lines
        log_v.append(x)
        log_ratio.append(y)

    # ln(measured/lookup) = k ln(sigma^2 / sigma_0^2), through the origin: at the
    # calibration width the lookup should be right. The SE is from the residual
    # scatter, which carries both the ensemble's and the lookup's Monte Carlo error.
    x = np.concatenate(log_v)
    y = np.concatenate(log_ratio)
    k_hat = float(x @ y / (x @ x))
    resid = y - k_hat * x
    k_se = float(np.sqrt(resid @ resid / (x.size - 1) / (x @ x)))
    lines.append(f"Exponent of measured/lookup = (sigma^2/sigma0^2)^k over {x.size} frames "
                 f"with per-frame CNR >= 1: k = {k_hat:.2f} +/- {k_se:.2f} "
                 "(sigma^3 scaling: k = 1.5)")
    report = "\n".join(lines)
    print(report)
    (OUTPUT / "narrowness_attribution.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
