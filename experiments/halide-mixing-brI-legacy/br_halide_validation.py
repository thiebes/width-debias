"""Validate the noise-bias correction on Marc's Br-I halide-mixing maps.

A heterogeneous-transport stress test on a material system distinct from the Mn
maps and the disordered nanocrystal solids: excitons diffusing on a halide-mixing
energy landscape in a 2D perovskite (Br fraction x). Halide disorder suppresses
transport an order of magnitude below the clean-layer diffusivity (effective
D ~ 0.003 vs D0 ~ 0.02 um^2/ns) and makes the spreading subdiffusive
(apparent alpha ~ 0.88) -- the "halide mixing inhibits exciton transport" regime.

Per docs/VALIDATION_DESIGN.md the truth is the carrier (particle) second moment,
supplied as ``case.var_true`` from the simulation's own ``variance.dat`` (it is
particle-based, so free of the box-truncation that clips the binned map's moment).
The emission is a single exciton population (linear in density), so the carrier
moment is reachable by the width-extraction stage alone -- but only with a fit.
The +/-75 nm box is narrow (the profile reaches ~+/-2.3 sigma by the last frame),
so the moment estimator clips ~15% of the variance and reads alpha / D_eff low,
while a bounded Gaussian fit extrapolates the tails and recovers the particle
truth. This bed therefore reinforces the project's bounded-fit-over-moment
message and asks the headline question in the disorder-signature dimension: does
the correction recover the apparent subdiffusive alpha, not only D?

Two analyses per Br fraction and CNR_0, scored against the noiseless Gaussian fit
(which recovers the particle truth here) and against the particle truth itself:

  * **naive** -- raw fitted variances, all frames, power law MSD = 2 D t^alpha.
  * **corrected** -- the ``subtract`` scheme: subtraction of the calibrated per-frame
    var_bias(CNR), then the same fit over frames above the per-frame CNR floor.

The Gaussian D-free calibration is built once at this geometry (narrow window) and
cached. The Gaussian fit on a mildly flat-topped profile (shape exponent ~ 2.35)
carries a small shape gap, reported separately; gen-Gaussian / Voigt fits recover
a near-identical alpha.

Run from the repository root (needs the pshare for the maps)::

    .venv/Scripts/python.exe scripts/br_halide_validation.py

Set ``WIDTH_DEBIAS_WORKERS=0`` to use all cores for the calibration.
Outputs ``output/br_halide_validation.{png,txt}``.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from scipy.optimize import curve_fit  # noqa: E402

from width_debias import GaussianModel, calibrate_d_free  # noqa: E402
from width_debias.fit import fit_variance as fit_gaussian  # noqa: E402
from width_debias.models import MomentModel  # noqa: E402
from width_debias.bias_lookup import BiasLookup  # noqa: E402
from width_debias.weights import corrected_variance  # noqa: E402

from br_data import load_all  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[2] / "output"
OUTPUT.mkdir(exist_ok=True)
CAL_PATH = OUTPUT / "calibration_br_halide.npz"

# Spain CVD (Okabe-Ito) semantic pairing: blue = corrected (tracks truth),
# red = naive (biased); truth in black. See plot_style.md.
CORR_BLUE = "#0072B2"
NAIVE_RED = "#D55E00"
TRUTH_BLACK = "#000000"
FLOOR_GREY = "#999999"
INCH = 1 / 2.54
FONT = 9

CNR0_SWEEP = (2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 60.0)
N_SEEDS = 40
FLOOR_CNR = 1.0

# Calibration geometry: matched to the Br-I maps (initial 1-D sigma ~ 24.8 nm,
# 250-bin +/-75 nm window -> a narrow +/-3 sigma window, captured in the D=0
# calibration so the fit's window truncation is in the lookup). The 0.1 ns
# window decays only ~10% (tau ~ 1 ns), so the per-frame CNR falloff is
# spreading-dominated.
SIGMA0_NM = 24.8
CAL_GEOMETRY = dict(sigma0=SIGMA0_NM, tau=1.0, half_window=75.0 / SIGMA0_NM,
                    n_pixels=250, n_frames=21, t_max=0.1)
CAL_CNR0 = np.geomspace(1.5, 250.0, 50)
N_RUNS_CAL = 1500


def fit_msd_powerlaw(t: np.ndarray, var: np.ndarray,
                     mask: np.ndarray | None = None) -> tuple[float, float]:
    """Field-standard MSD = 2 D t^alpha fit; returns (D, alpha) with D in nm^2/ns.

    MSD uses the fitted t=0 width (``var - var[0]``); the fit runs over t > 0 with
    NaN frames dropped, and ``mask`` restricts it to frames above the CNR floor.
    """
    msd = var - var[0]
    keep = (t > 0) & np.isfinite(msd)
    if mask is not None:
        keep = keep & mask
    if keep.sum() < 3:
        return np.nan, np.nan
    try:
        popt, _ = curve_fit(
            lambda tt, d, a: 2.0 * d * tt ** a, t[keep], msd[keep],
            p0=[max(msd[keep][-1] / (2.0 * t[keep][-1]), 1e-3), 1.0],
            bounds=([0.0, 0.2], [np.inf, 2.0]), maxfev=20000)
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        return np.nan, np.nan


def build_or_load_calibration() -> BiasLookup:
    """Gaussian D-free calibration at the Br-I geometry; cached after first build."""
    if CAL_PATH.exists() and "framevar_grid" in np.load(CAL_PATH).files:
        d = np.load(CAL_PATH)
        return BiasLookup(cnr_grid=d["cnr_grid"], p_grid=d["p_grid"],
                               relvar_grid=d["relvar_grid"],
                               varbias_grid=d["varbias_grid"],
                               framevar_grid=d["framevar_grid"])
    trunc = calibrate_d_free(CAL_CNR0, n_runs=N_RUNS_CAL, model=GaussianModel(),
                             **CAL_GEOMETRY)
    np.savez(CAL_PATH, cnr_grid=trunc.cnr_grid, p_grid=trunc.p_grid,
             relvar_grid=trunc.relvar_grid, varbias_grid=trunc.varbias_grid,
             framevar_grid=trunc.framevar_grid)
    return trunc


def fit_each_row(x: np.ndarray, profiles: np.ndarray) -> np.ndarray:
    out = np.full(profiles.shape[0], np.nan)
    for i in range(profiles.shape[0]):
        r = fit_gaussian(x, profiles[i])
        if r.ok and np.isfinite(r.variance):
            out[i] = r.variance
    return out


def validation_pass(case, trunc: BiasLookup, cnr0: float, n_seeds: int):
    """Mean raw and corrected Gaussian-fit variance over seeds."""
    peak0 = float(case.profiles[0].max())
    noise_sigma = peak0 / cnr0
    cnr_frame = case.profiles.max(axis=1) / noise_sigma
    n_frames = case.profiles.shape[0]
    raw = np.full((n_seeds, n_frames), np.nan)
    cor = np.full((n_seeds, n_frames), np.nan)
    for s in range(n_seeds):
        rng = np.random.default_rng(int(cnr0) * 1000 + s)
        noisy = case.profiles + rng.normal(0.0, noise_sigma, case.profiles.shape)
        var_hat = fit_each_row(case.x, noisy)
        raw[s] = var_hat
        cor[s] = corrected_variance("subtract", var_hat, cnr_frame, trunc)
    return np.nanmean(raw, axis=0), np.nanmean(cor, axis=0), cnr_frame


@dataclass
class CaseResult:
    """Per-case recovery: references plus the CNR_0 sweep (computed once)."""
    name: str
    t: np.ndarray
    var_true: np.ndarray
    a_true: float
    a_nf: float           # noiseless Gaussian-fit exponent (the within-fit reference)
    a_nm: float           # noiseless moment exponent (box-clipped, biased low)
    shape_gap: float      # mean |noiseless fit - carrier truth| (nm^2)
    cnr0: np.ndarray
    min_cnr: np.ndarray
    n_work: np.ndarray
    a_naive: np.ndarray
    a_meth: np.ndarray
    d_naive: np.ndarray
    d_meth: np.ndarray


def sweep_case(case, trunc: BiasLookup) -> CaseResult:
    """Run the CNR_0 sweep for one Br fraction (the single source of the numbers)."""
    nf = fit_each_row(case.x, case.profiles)                  # noiseless Gaussian fit
    nm = np.array([MomentModel().fit_variance(case.x, p).variance
                   for p in case.profiles])                   # noiseless moment
    _, a_true = fit_msd_powerlaw(case.t, case.var_true)       # particle (carrier) truth
    _, a_nf = fit_msd_powerlaw(case.t, nf)
    _, a_nm = fit_msd_powerlaw(case.t, nm)
    rows = {k: [] for k in ("min_cnr", "n_work", "a_naive", "a_meth",
                            "d_naive", "d_meth")}
    for cnr0 in CNR0_SWEEP:
        raw, cor, cnr_frame = validation_pass(case, trunc, cnr0, N_SEEDS)
        work = cnr_frame >= FLOOR_CNR
        d_naive, a_naive = fit_msd_powerlaw(case.t, raw)
        d_meth, a_meth = fit_msd_powerlaw(case.t, cor, mask=work)
        rows["min_cnr"].append(float(cnr_frame.min()))
        rows["n_work"].append(int(work.sum()))
        rows["a_naive"].append(a_naive); rows["a_meth"].append(a_meth)
        rows["d_naive"].append(d_naive); rows["d_meth"].append(d_meth)
    return CaseResult(
        name=case.name, t=case.t, var_true=case.var_true, a_true=a_true,
        a_nf=a_nf, a_nm=a_nm,
        shape_gap=float(np.nanmean(np.abs(nf - case.var_true))),
        cnr0=np.array(CNR0_SWEEP), min_cnr=np.array(rows["min_cnr"]),
        n_work=np.array(rows["n_work"]), a_naive=np.array(rows["a_naive"]),
        a_meth=np.array(rows["a_meth"]), d_naive=np.array(rows["d_naive"]),
        d_meth=np.array(rows["d_meth"]))


def write_report(results: list[CaseResult], n_frames: int, path: Path) -> None:
    lines = [
        "=" * 76,
        "Br-I halide-mixing validation -- subtract, Gaussian fit",
        "carrier truth = particle 2nd moment (variance.dat/2); halide disorder",
        "suppresses D ~10x below D0 and makes transport subdiffusive.",
        "=" * 76,
        f"n_seeds per CNR_0: {N_SEEDS};  per-frame CNR floor: {FLOOR_CNR}",
        "naive = raw all-frames fit;  method = corrected + sub-floor frames dropped",
        "alpha read from the mean trajectory.",
        "",
    ]
    for r in results:
        lines.append("-" * 76)
        lines.append(f"=== {r.name} ===  shape gap (fit vs carrier truth) "
                     f"{r.shape_gap:.1f} nm^2")
        lines.append(f"  carrier truth alpha = {r.a_true:.3f};  noiseless fit "
                     f"alpha = {r.a_nf:.3f} ({r.a_nf - r.a_true:+.3f} vs carrier);"
                     f"  noiseless moment alpha = {r.a_nm:.3f} "
                     f"({r.a_nm - r.a_true:+.3f} vs carrier)")
        lines.append("  CNR_0 | min CNR | n>=1 | alpha naive/method | D naive/method")
        for i, c in enumerate(r.cnr0):
            lines.append(f"  {c:5.0f} | {r.min_cnr[i]:7.2f} | "
                         f"{r.n_work[i]:2d}/{n_frames} | "
                         f"{r.a_naive[i]:.3f}/{r.a_meth[i]:.3f} | "
                         f"{r.d_naive[i]:.3g}/{r.d_meth[i]:.3g}")
        lines.append("")
    report = "\n".join(lines)
    print(report)
    path.write_text(report)


def make_figure(results: list[CaseResult], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    n = len(results)
    fig, axes = plt.subplots(1, n, layout="constrained", sharey=True,
                             figsize=(6.0 * n * INCH, 7.0 * INCH), squeeze=False)
    for ax, r in zip(axes[0], results):
        ax.plot(r.cnr0, r.a_naive, "o-", color=NAIVE_RED, lw=2, ms=4,
                label="naive (all frames)")
        ax.plot(r.cnr0, r.a_meth, "s-", color=CORR_BLUE, lw=2, ms=4,
                label="corrected + floor cut")
        ax.axhline(r.a_true, color=TRUTH_BLACK, lw=1.2, ls="--",
                   label="carrier truth")
        ax.set_xscale("log")
        ax.set_xticks(r.cnr0)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"CNR$_0$", fontsize=FONT)
        ax.tick_params(axis="both", which="both", direction="in", top=True,
                       right=True, labelsize=FONT)
        ax.set_ylim(0.3, 2.05)
        ax.text(0.5, 0.96, r.name, ha="center", va="top", fontsize=FONT,
                transform=ax.transAxes)
    axes[0][0].set_ylabel(r"recovered exponent  $\alpha$", fontsize=FONT)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=FONT,
               loc="outside lower center", ncol=3)
    fig.patch.set_facecolor("w")
    fig.patch.set_alpha(1)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure: {path.name}")


def main() -> None:
    cases = load_all()
    if not cases:
        print("No Br-I maps found on the pshare; nothing to validate.")
        return
    print(f"Building/loading Br-I Gaussian D-free calibration "
          f"({len(CAL_CNR0)} CNR0 x {N_RUNS_CAL} runs) ...")
    trunc = build_or_load_calibration()
    n_frames = cases[0].profiles.shape[0]
    results = [sweep_case(case, trunc) for case in cases]
    write_report(results, n_frames, OUTPUT / "br_halide_validation.txt")
    make_figure(results, OUTPUT / "br_halide_validation.png")


if __name__ == "__main__":
    main()
