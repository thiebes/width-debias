"""How close does the corrected D come to the Cramer-Rao bound?

``crlb.py`` derives the bound for the Gaussian measurement model and shows that
"fit a width per frame, then fit the slope" loses no information: an unbiased,
efficient width estimate with inverse-CRLB slope weights reaches the bound.
This script measures how far the actual ``subtract`` estimator is from it, on
the four validation shapes (Gaussian, Voigt, generalized-Gaussian p = 1.5 and
p = 4).

The Fisher matrix for a non-Gaussian shape is built numerically
(``width_debias.fisher``) from the model's own ``noise_free_series``
(central differences in each frame's
variance and centre; the amplitude enters linearly). The parameters are those
the width fits estimate: sigma_0^2 and D, plus a per-frame peak amplitude and
centre as nuisances; no baseline. For the Gaussian the numerical matrix is
checked against the analytic one in ``crlb.py``.

Per cell (D, CNR_0) it reports the spread of the corrected D over realizations
divided by sqrt(CRLB), for three arms:

* ``library`` -- subtract with the library's weights (per-frame variance
  from the D-free calibration);
* ``oracle``  -- the same corrected points weighted by the inverse per-frame
  CRLB, the weights the bound itself implies; the gap between the two arms is
  the cost of the weights, the remainder is the cost of the width fit;
* ``robust``  -- the library arm's interquartile range / 1.349, because at low
  CNR a few heavy-tailed fits dominate the SD;
* ``width``   -- the corrected widths fitted against time directly, every frame
  weighted by the library's per-frame variance, intercept free. The library
  instead fits MSD = sigma_hat^2 - sigma_hat^2(0) with weights
  1 / (v_i + v_0) and frame 0 at weight 0, which treats the MSD points as
  independent although they share sigma_hat^2(0) and leaves the best-measured
  frame out of the slope. A constant shift does not change a free-intercept
  slope, so this arm equals the MSD fit with weights 1 / v_i on every frame.
  The library uses these weights, so the ``library`` and ``width`` arms
  coincide with the current package.

Each arm is computed with every frame and with frames below per-frame CNR 1
dropped (the floor the paper uses). A ratio of 1 is efficient. The bound holds
for unbiased estimators, and the corrected D is unbiased within its SE across
this grid (``docs/DERIVATION.md`` section 8), so the comparison is fair.

``generate`` caches the calibrations and evaluation ensembles under
``output/crlb-efficiency/`` (the Gaussian calibration is draw 0 of
``dfree-calibration/calibration_se_check.py`` when that cache exists);
``analyse`` prints the tables. Seeds are offset by 10^6 per shape, so the four
shapes see independent noise; the cache names carry the seed.

    $env:WIDTH_DEBIAS_WORKERS = 0
    .venv/Scripts/python.exe experiments/crlb-precision/efficiency_check.py generate
    .venv/Scripts/python.exe experiments/crlb-precision/efficiency_check.py analyse
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias import (Experiment, BiasLookup, calibrate_d_free,  # noqa: E402
                           model_from_spec, spec_tag)
from width_debias.fisher import (crlb_D, per_frame_crlb,  # noqa: E402
                                  sequence_fisher_information as fisher_information)
from width_debias.montecarlo import run_ensemble  # noqa: E402
from width_debias.simulate import noise_sigma_for_cnr0  # noqa: E402
from width_debias.weights import msd_and_weights  # noqa: E402

from crlb import fisher_information as gaussian_fisher  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "crlb-efficiency"
GAUSS_CAL = ROOT / "output" / "calibration-se-check" / "cal_seed1_runs1500.npz"

SPECS = ("gaussian", "voigt", "generalized_gaussian:1.5", "generalized_gaussian:4")
CAL_GRID = np.geomspace(1.5, 150.0, 48)
CAL_RUNS = 600
D_VALUES = (0.005, 0.02, 0.1)
CNR0_VALUES = (3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 90.0)
EVAL_RUNS = 1000
EVAL_SEED = 810000
FLOOR = 1.0
N_BOOT = 200


# --------------------------------------------------------------------------
# Cached Monte Carlo
# --------------------------------------------------------------------------
def _offset(spec: str) -> int:
    """Seed offset per shape, so the shapes see independent noise (the Gaussian's is 0)."""
    return 10 ** 6 * SPECS.index(spec)


def _cal_path(spec: str) -> Path:
    return OUT / f"cal_{spec_tag(spec)}_runs{CAL_RUNS}_seed{1 + _offset(spec)}.npz"


def _eval_path(spec: str) -> Path:
    return OUT / f"eval_{spec_tag(spec)}_runs{EVAL_RUNS}_seed{EVAL_SEED + _offset(spec)}.npz"


def calibration(spec: str) -> BiasLookup:
    if spec == "gaussian" and GAUSS_CAL.exists():
        # Draw 0 of the calibration-SE check: the paper's seed-1, 1500-run lookup.
        sys.path.insert(0, str(ROOT / "experiments" / "dfree-calibration"))
        from calibration_se_check import _ensembles
        from width_debias.bias_lookup import frame_stats
        return BiasLookup.from_stats([frame_stats(e) for e in _ensembles(0)])
    with np.load(_cal_path(spec)) as z:
        return BiasLookup(**{k: z[k] for k in z.files})


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        model = model_from_spec(spec)
        if not (spec == "gaussian" and GAUSS_CAL.exists()) and not _cal_path(spec).exists():
            t0 = time.time()
            m = calibrate_d_free(CAL_GRID, n_runs=CAL_RUNS, seed=1 + _offset(spec),
                                 model=model)
            np.savez(_cal_path(spec), cnr_grid=m.cnr_grid, p_grid=m.p_grid,
                     relvar_grid=m.relvar_grid, varbias_grid=m.varbias_grid,
                     varbias_se_grid=m.varbias_se_grid, framevar_grid=m.framevar_grid)
            print(f"{spec} calibration: {time.time() - t0:.0f} s", flush=True)
        if not _eval_path(spec).exists():
            t0 = time.time()
            var = np.full((len(D_VALUES), len(CNR0_VALUES), EVAL_RUNS, 12), np.nan)
            cnr = np.full((len(D_VALUES), len(CNR0_VALUES), 12), np.nan)
            for di, d in enumerate(D_VALUES):
                for ci, c0 in enumerate(CNR0_VALUES):
                    ens = run_ensemble(Experiment(D=d), c0, n_runs=EVAL_RUNS,
                                       seed=EVAL_SEED + _offset(spec) + 100 * di + ci,
                                       model=model)
                    var[di, ci], cnr[di, ci] = ens.variance, ens.cnr_frame
            np.savez(_eval_path(spec), variance=var, cnr=cnr)
            print(f"{spec} evaluation: {time.time() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def _slopes(msd: np.ndarray, times: np.ndarray, w: np.ndarray) -> np.ndarray:
    """D for every realization (rows of msd) under one weight vector."""
    good = w > 0
    tg, wg = times[good], w[good]
    sw, swt, swtt = wg.sum(), (wg * tg).sum(), (wg * tg * tg).sum()
    c = wg * (sw * tg - swt) / (sw * swtt - swt ** 2)
    return msd[:, good] @ c / 2.0


def _spread(x: np.ndarray, rng: np.random.Generator):
    """SD and robust SD (IQR / 1.349), each with a bootstrap SE."""
    def both(a):
        q75, q25 = np.percentile(a, [75, 25])
        return np.array([a.std(ddof=1), (q75 - q25) / 1.349])
    est = both(x)
    boot = np.array([both(x[rng.integers(0, x.size, x.size)]) for _ in range(N_BOOT)])
    return est, boot.std(axis=0, ddof=1)


def analyse() -> None:
    rng = np.random.default_rng(0)
    # Check the numerical Fisher matrix against the analytic Gaussian one.
    exp = Experiment(D=0.02)
    ns = noise_sigma_for_cnr0(exp, 8.0)
    num = fisher_information(model_from_spec("gaussian"), exp, ns)
    ana = gaussian_fisher(exp, ns)
    rel = abs(np.linalg.inv(num)[1, 1] / np.linalg.inv(ana)[1, 1] - 1)
    print(f"Gaussian D-CRLB, numerical vs analytic Fisher matrix: relative "
          f"difference {rel:.1e}")
    assert rel < 1e-5

    head = "            " + "".join(f"{c:>9g}" for c in CNR0_VALUES)
    results = {}
    for spec in SPECS:
        if not _eval_path(spec).exists():
            print(f"\n({spec}: no cached evaluation; run generate)")
            continue
        model, cal = model_from_spec(spec), calibration(spec)
        with np.load(_eval_path(spec)) as z:
            evar, ecnr = z["variance"], z["cnr"]
        nd, nc = len(D_VALUES), len(CNR0_VALUES)
        # arms x floor x (D, CNR0) x (value, SE)
        out = np.full((4, 2, nd, nc, 2), np.nan)
        root_crlb = np.full((nd, nc), np.nan)
        mean = np.full((2, nd, nc), np.nan)
        mean_dir = np.full((2, nd, nc), np.nan)
        for di, d in enumerate(D_VALUES):
            exp = Experiment(D=d)
            for ci, c0 in enumerate(CNR0_VALUES):
                root_crlb[di, ci] = np.sqrt(crlb_D(model, exp, c0))
                pf = per_frame_crlb(model, exp, noise_sigma_for_cnr0(exp, c0))
                var, cnr = evar[di, ci], ecnr[di, ci]
                corrected = var - cal.var_bias(cnr)
                msd = corrected - corrected[:, [0]]
                _, w_lib = msd_and_weights("subtract", var[0], cnr, cal)
                w_orc = 1.0 / (pf + pf[0])
                w_orc[0] = 0.0
                for fi, floor in enumerate((0.0, FLOOR)):
                    keep = cnr >= floor
                    if (keep[1:].sum()) < 2:
                        continue
                    d_lib = _slopes(msd, exp.times, np.where(keep, w_lib, 0.0))
                    d_orc = _slopes(msd, exp.times, np.where(keep, w_orc, 0.0))
                    w_dir = np.where(keep, 1.0 / cal.frame_variance(cnr), 0.0)
                    d_dir = _slopes(corrected, exp.times, w_dir)
                    mean[fi, di, ci] = d_lib.mean() / d
                    (sd_l, rob_l), (se_sd_l, se_rob_l) = _spread(d_lib, rng)
                    (sd_o, _), (se_sd_o, _) = _spread(d_orc, rng)
                    out[0, fi, di, ci] = (sd_l, se_sd_l)
                    out[1, fi, di, ci] = (sd_o, se_sd_o)
                    out[2, fi, di, ci] = (rob_l, se_rob_l)
                    (sd_d, _), (se_sd_d, _) = _spread(d_dir, rng)
                    out[3, fi, di, ci] = (sd_d, se_sd_d)
                    mean_dir[fi, di, ci] = d_dir.mean() / d
        out /= root_crlb[None, None, :, :, None]
        results[spec] = out
        print(f"\n=== {spec}: spread of corrected D / sqrt(CRLB) (1 = efficient); "
              f"rows D, columns CNR_0; +- bootstrap SE")
        print(f"sqrt(CRLB) / D\n{head}")
        for di, d in enumerate(D_VALUES):
            print(f"  {d:6.3f}    " + "".join(f"{v:9.3g}" for v in root_crlb[di] / d))
        for fi, label in enumerate(("all frames", f"frames with CNR >= {FLOOR:g}")):
            for ai, arm in enumerate(("library SD", "oracle-weight SD", "library IQR/1.349",
                                      "width-fit SD")):
                print(f"[{label}] {arm}\n{head}")
                for di, d in enumerate(D_VALUES):
                    print(f"  {d:6.3f}    " + "".join(
                        f"{v:5.2f}+{s:<4.2f}" if np.isfinite(v) else f"{'-':>9}"
                        for v, s in out[ai, fi, di]))
            print(f"[{label}] mean D/D_in (library)\n{head}")
            for di, d in enumerate(D_VALUES):
                print(f"  {d:6.3f}    " + "".join(f"{v:9.3f}" for v in mean[fi, di]))
            print(f"[{label}] mean D/D_in (width fit)\n{head}")
            for di, d in enumerate(D_VALUES):
                print(f"  {d:6.3f}    " + "".join(f"{v:9.3f}" for v in mean_dir[fi, di]))
    np.savez(OUT / "efficiency.npz", D_values=D_VALUES, cnr0_values=CNR0_VALUES,
             arms=np.array(["library_sd", "oracle_sd", "library_robust", "width_sd"]),
             **{spec_tag(s): v for s, v in results.items()})


if __name__ == "__main__":
    {"generate": generate, "analyse": analyse}[sys.argv[1] if len(sys.argv) > 1 else "analyse"]()
