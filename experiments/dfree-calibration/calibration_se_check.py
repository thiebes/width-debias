"""Check the calibration SE propagated into D against independent calibration draws.

``weights.calibration_D_se`` predicts how far a corrected D moves when the D-free
calibration is redrawn. Three draws cannot resolve the ratio of the observed
to the predicted spread (the analysis reports how widely three-draw subsets
scatter), so this script tests the prediction with many draws over one grid.

``generate`` builds and caches the raw ensembles:

* ``N_CAL`` independent D-free calibrations on a 48-point grid (CNR_0 from
  1.5 to 150, 1500 runs each); draw k uses base seed ``1 + 1000 k``, so draw 0
  has seed 1 and the seed ranges never overlap;
* one fixed evaluation set (D 0.005 to 0.1, CNR_0 2 to 90, 300 runs per cell).

``analyse`` applies every calibration to the same evaluation realizations
(subtract, exact per-frame CNR key, frames below CNR 1 dropped), so the spread of a cell's mean over the draws is the calibration term
alone. Per cell it reports observed / predicted variance for three arms:

* ``full``    -- the draw's own lookup (what a user applies);
* ``bias``    -- the draw's var_bias with the weights frozen at the pooled lookup
  (the term calibration_D_se propagates);
* ``weights`` -- the pooled var_bias with the draw's own weights.

The slope weights come from the per-frame variance (``framevar_grid``), so every
arm's model carries one; a model built without it falls back to the MSD-point
weights with frame 0 at weight 0, and an arm on that fallback would be compared
with a prediction made for the per-frame weights.

It also checks the per-point SE (variance of each grid point over the draws
against its recorded SE squared) and the correlation of the fitted variances
between frames of one calibration ensemble. With ``WIDTH_DEBIAS_EXTERNAL_RESULTS``
naming a directory that holds an external triple of Gaussian calibrations on the
same grid, it ranks that triple among the draws.

    $env:WIDTH_DEBIAS_WORKERS = 0
    .venv/Scripts/python.exe experiments/dfree-calibration/calibration_se_check.py generate
    .venv/Scripts/python.exe experiments/dfree-calibration/calibration_se_check.py analyse
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias import Experiment, BiasLookup  # noqa: E402
from width_debias.montecarlo import Ensemble, run_ensemble  # noqa: E402
from width_debias.bias_lookup import frame_stats  # noqa: E402
from width_debias.weights import calibration_D_se, msd_and_weights  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "output" / "calibration-se-check"
EXTERNAL_RESULTS = "WIDTH_DEBIAS_EXTERNAL_RESULTS"

CAL_GRID = np.geomspace(1.5, 150.0, 48)
CAL_RUNS = 1500
N_CAL = int(os.environ.get("N_CAL", 16))
D_VALUES = (0.005, 0.01, 0.02, 0.05, 0.1)
CNR0_VALUES = (2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 90.0)
EVAL_RUNS = 300
EVAL_SEED = 700000
FLOOR = 1.0


def _cal_seed(k: int) -> int:
    return 1 + 1000 * k


def _cal_path(k: int) -> Path:
    return OUT / f"cal_seed{_cal_seed(k)}_runs{CAL_RUNS}.npz"


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eval_path = OUT / f"eval_runs{EVAL_RUNS}.npz"
    if not eval_path.exists():
        t0 = time.time()
        var = np.full((len(D_VALUES), len(CNR0_VALUES), EVAL_RUNS, 12), np.nan)
        cnr = np.full((len(D_VALUES), len(CNR0_VALUES), 12), np.nan)
        for di, d in enumerate(D_VALUES):
            for ci, c0 in enumerate(CNR0_VALUES):
                ens = run_ensemble(Experiment(D=d), c0, n_runs=EVAL_RUNS,
                                   seed=EVAL_SEED + 100 * di + ci)
                var[di, ci], cnr[di, ci] = ens.variance, ens.cnr_frame
        np.savez(eval_path, variance=var, cnr=cnr)
        print(f"evaluation set: {time.time() - t0:.0f} s", flush=True)
    cal_exp = Experiment(D=0.0)
    for k in range(N_CAL):
        path = _cal_path(k)
        if path.exists():
            continue
        t0 = time.time()
        var = np.full((CAL_GRID.size, CAL_RUNS, cal_exp.n_frames), np.nan)
        msd = np.full_like(var, np.nan)
        wall = np.zeros(var.shape, bool)
        cnr = np.full((CAL_GRID.size, cal_exp.n_frames), np.nan)
        # Same seeding as validate.sweep_stats, so draw 0 is calibrate_d_free(seed=1).
        for i, c0 in enumerate(CAL_GRID):
            ens = run_ensemble(cal_exp, float(c0), n_runs=CAL_RUNS,
                               seed=_cal_seed(k) + i)
            var[i], msd[i], wall[i], cnr[i] = (ens.variance, ens.msd,
                                               ens.at_wall, ens.cnr_frame)
        np.savez(path, variance=var, msd=msd, at_wall=wall, cnr=cnr)
        print(f"calibration {k} (seed {_cal_seed(k)}): {time.time() - t0:.0f} s",
              flush=True)


def _ensembles(k: int) -> list[Ensemble]:
    cal_exp = Experiment(D=0.0)
    with np.load(_cal_path(k)) as z:
        return [Ensemble(exp=cal_exp, cnr0=float(c0), noise_sigma=1.0 / c0,
                         cnr_frame=z["cnr"][i], variance=z["variance"][i],
                         msd=z["msd"][i], at_wall=z["at_wall"][i],
                         ok=np.ones_like(z["at_wall"][i]))
                for i, c0 in enumerate(CAL_GRID)]


def _cell_means(model: BiasLookup, var: np.ndarray, cnr: np.ndarray,
                d_true: float):
    """Mean D/D_in over a cell's realizations and the mean predicted cal SE."""
    times = Experiment(D=d_true).times
    ratio, se = [], []
    for row in var:
        msd, w = msd_and_weights("subtract", row, cnr, model)
        w = np.where(cnr >= FLOOR, w, 0.0)
        good = np.isfinite(msd) & (w > 0)
        if good.sum() < 2:
            continue
        tg, mg, wg = times[good], msd[good], w[good]
        sw, swt, swtt = wg.sum(), (wg * tg).sum(), (wg * tg * tg).sum()
        slope = (sw * (wg * tg * mg).sum() - swt * (wg * mg).sum()) / (sw * swtt - swt ** 2)
        ratio.append(slope / 2.0 / d_true)
        se.append(calibration_D_se(times, msd, w, cnr, model) / d_true)
    return float(np.mean(ratio)), float(np.mean(se))


def analyse() -> None:
    ks = [k for k in range(N_CAL) if _cal_path(k).exists()]
    print(f"{len(ks)} calibration draws, {CAL_RUNS} runs each")
    models, ens0 = [], None
    for k in ks:
        ens = _ensembles(k)
        if ens0 is None:
            ens0 = ens
        models.append(BiasLookup.from_stats([frame_stats(e) for e in ens]))
    grid = models[0].cnr_grid
    assert all(np.array_equal(m.cnr_grid, grid) for m in models)
    vb = np.array([m.varbias_grid for m in models])
    rv = np.array([m.relvar_grid for m in models])
    se = np.array([m.varbias_se_grid for m in models])

    # 1. Per-point SE: variance of each grid point over the draws against SE^2.
    ratio_pt = vb.var(axis=0, ddof=1) / np.mean(se ** 2, axis=0)
    print("\nPer-point var_bias: observed / predicted variance, by per-frame CNR")
    edges = [0.1, 0.5, 1, 2, 5, 10, 30, 200]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (grid >= lo) & (grid < hi)
        if sel.any():
            print(f"  CNR {lo:5g}-{hi:<5g} n={sel.sum():3d}: {ratio_pt[sel].mean():5.2f}"
                  f"  (expected 1 +/- {np.sqrt(2 / (len(ks) - 1) / sel.sum()):.2f})")
    # Relative error of rel_variance between draws (it sets the weights).
    # The t = 0 points have rel_variance 0 exactly (MSD(0) = 0 by construction).
    t0_pt = np.all(rv == 0.0, axis=0)
    rel_rv = rv.std(axis=0, ddof=1) / np.where(t0_pt, np.nan, rv.mean(axis=0))
    print(f"\nrel_variance relative spread over draws, by per-frame CNR "
          f"({t0_pt.sum()} t = 0 points with rel_variance 0 excluded)")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (grid >= lo) & (grid < hi) & ~t0_pt
        if sel.any():
            print(f"  CNR {lo:5g}-{hi:<5g}: {np.median(rel_rv[sel]):.3f}")

    # 2. Correlation between frames of one calibration ensemble (draw 0).
    offd = []
    for e in ens0:
        c = np.corrcoef(e.variance, rowvar=False)
        offd.append(c[~np.eye(c.shape[0], dtype=bool)])
    offd = np.concatenate(offd)
    print(f"\nWithin-ensemble frame correlation (draw 0): mean {offd.mean():+.4f}, "
          f"SD {offd.std():.4f} (independent: SD ~ {1 / np.sqrt(CAL_RUNS):.4f})")

    # 3. Per-cell spread of the corrected mean over the draws.
    pooled = BiasLookup(cnr_grid=grid, p_grid=models[0].p_grid,
                             relvar_grid=rv.mean(axis=0), varbias_grid=vb.mean(axis=0),
                             varbias_se_grid=np.sqrt(np.mean(se ** 2, axis=0)),
                             framevar_grid=np.mean([m.framevar_grid for m in models], axis=0))
    arms = {
        "full": lambda m: m,
        "bias": lambda m: BiasLookup(cnr_grid=grid, p_grid=m.p_grid,
                                          relvar_grid=pooled.relvar_grid,
                                          varbias_grid=m.varbias_grid,
                                          varbias_se_grid=m.varbias_se_grid,
                                          framevar_grid=pooled.framevar_grid),
        "weights": lambda m: BiasLookup(cnr_grid=grid, p_grid=m.p_grid,
                                             relvar_grid=m.relvar_grid,
                                             varbias_grid=pooled.varbias_grid,
                                             varbias_se_grid=m.varbias_se_grid,
                                             framevar_grid=m.framevar_grid),
    }
    with np.load(OUT / f"eval_runs{EVAL_RUNS}.npz") as z:
        evar, ecnr = z["variance"], z["cnr"]
    nd, nc = len(D_VALUES), len(CNR0_VALUES)
    means = {a: np.full((len(ks), nd, nc), np.nan) for a in arms}
    pred = np.full((len(ks), nd, nc), np.nan)
    for j, m in enumerate(models):
        for a, make in arms.items():
            mm = make(m)
            for di, d in enumerate(D_VALUES):
                for ci in range(nc):
                    mean, s = _cell_means(mm, evar[di, ci], ecnr[di, ci], d)
                    means[a][j, di, ci] = mean
                    if a == "full":
                        pred[j, di, ci] = s
    p2 = np.mean(pred ** 2, axis=0)
    obs = {a: means[a].var(axis=0, ddof=1) for a in arms}
    np.savez(OUT / "analysis.npz", cnr_grid=grid, varbias=vb, relvar=rv, se=se,
             **{f"means_{a}": v for a, v in means.items()}, pred=pred,
             D_values=D_VALUES, cnr0_values=CNR0_VALUES)
    print("\nCell mean D/D_in over draws: observed / predicted variance "
          "(rows D, columns CNR_0)")
    print("          " + "".join(f"{c:>8g}" for c in CNR0_VALUES))
    for a in arms:
        print(f" arm {a}")
        for di, d in enumerate(D_VALUES):
            print(f"  {d:6.3f}  " + "".join(f"{v:8.2f}" for v in obs[a][di] / p2[di]))
    print("\n Mean over D, per CNR_0 (expected 1 +/- "
          f"{np.sqrt(2 / (len(ks) - 1)):.2f} per cell)")
    for a in arms:
        print(f"  {a:8s}" + "".join(f"{v:8.2f}" for v in (obs[a] / p2).mean(axis=0)))
    # How often would three draws read a mean ratio >= 2?
    from itertools import combinations
    trip = []
    for c in combinations(range(len(ks)), 3):
        r = means["full"][list(c)].var(axis=0, ddof=1) / p2
        trip.append(r.mean())
    trip = np.asarray(trip)
    print(f"\n Three-draw subsets ({trip.size}): mean ratio over all cells, median "
          f"{np.median(trip):.2f}, 5-95% {np.percentile(trip, 5):.2f}-"
          f"{np.percentile(trip, 95):.2f}, fraction >= 2.0: {np.mean(trip >= 2.0):.3f}")
    print("\n Predicted cal SE of D/D_in (mean over draws)")
    for di, d in enumerate(D_VALUES):
        print(f"  {d:6.3f}  " + "".join(f"{v:8.3f}" for v in np.sqrt(p2[di])))

    # 4. The t = 0 grid points (rel_variance 0) pull the interpolated weight up
    # for an evaluation frame whose CNR lies next to a calibration CNR_0. This
    # concerns the MSD-point weights, which the per-frame weights replace; they
    # remain the fallback for a model saved without framevar_grid.
    print("\nInterpolated rel_variance with / without the t = 0 points "
          "(fallback MSD-point weights; D = 0.02, frames with CNR >= 1 after t = 0)")
    for ci, c0 in enumerate(CNR0_VALUES):
        c = ecnr[2, ci]
        keep = (c >= FLOOR) & (np.arange(c.size) > 0)
        r = (np.interp(c, grid, rv[0]) / np.interp(c, grid[~t0_pt], rv[0][~t0_pt]))[keep]
        if r.size:
            print(f"  CNR_0 {c0:5g}: min {r.min():.2f} over {r.size} frames")

    external_triple(models, means["full"], p2, ecnr, evar)


def external_triple(models, full_means, p2, ecnr, evar) -> None:
    """Rank an external triple of Gaussian calibrations (seeds 1, 101, 201) among the draws.

    Seed 1 is draw 0 here; the calibrations at seeds 101 and 201 come from an
    external study on the same grid, read from the directory that
    ``WIDTH_DEBIAS_EXTERNAL_RESULTS`` names. They carry ``framevar_grid``, so
    they are scored with the same per-frame weights as the draws here.
    """
    directory = os.environ.get(EXTERNAL_RESULTS)
    if not directory:
        print(f"\n({EXTERNAL_RESULTS} not set; external triple skipped)")
        return
    paths = [Path(directory) / f"case27_calibration_gaussian_seed{s}.npz" for s in (101, 201)]
    if not all(p.exists() for p in paths):
        print(f"\n(external calibrations not found under {EXTERNAL_RESULTS}; "
              "external triple skipped)")
        return
    from itertools import combinations
    grid = models[0].cnr_grid
    se = np.sqrt(np.mean([m.varbias_se_grid ** 2 for m in models], axis=0))
    extra = []
    for p in paths:
        with np.load(p) as d:
            assert np.allclose(d["cnr_grid"], grid)
            if "framevar_grid" not in d:
                print(f"\n({p.name} has no framevar_grid; external triple skipped)")
                return
            m = BiasLookup(cnr_grid=grid, p_grid=d["p_grid"],
                                relvar_grid=d["relvar_grid"],
                                varbias_grid=d["varbias_grid"], varbias_se_grid=se,
                                framevar_grid=d["framevar_grid"])
        extra.append([[_cell_means(m, evar[di, ci], ecnr[di, ci], d_true)[0]
                       for ci in range(len(CNR0_VALUES))]
                      for di, d_true in enumerate(D_VALUES)])
    allm = np.concatenate([full_means, np.asarray(extra)])
    n = allm.shape[0]
    # Draw 0 is seed 1, so the external triple is draws (0, n-2, n-1).
    ratio = allm[[0, n - 2, n - 1]].var(axis=0, ddof=1) / p2
    trip = np.array([(allm[list(c)].var(axis=0, ddof=1) / p2).mean()
                     for c in combinations(range(n), 3)])
    print(f"\nExternal triple (seeds 1, 101, 201) on this evaluation set: mean ratio "
          f"{ratio.mean():.2f}; per CNR_0 (mean over D)")
    print("  " + "".join(f"{v:8.2f}" for v in ratio.mean(axis=0)))
    print(f"  fraction of the {trip.size} triples from all {n} draws at or above it: "
          f"{np.mean(trip >= ratio.mean()):.4f}")
    allr = allm.var(axis=0, ddof=1) / p2
    print(f"  all {n} draws: mean ratio {allr.mean():.2f}; per CNR_0 "
          + " ".join(f"{v:.2f}" for v in allr.mean(axis=0)))


if __name__ == "__main__":
    {"generate": generate, "analyse": analyse}[sys.argv[1] if len(sys.argv) > 1 else "analyse"]()
