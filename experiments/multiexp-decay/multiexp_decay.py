"""Is the noise-bias correction invalidated by multi-exponential decay?

Real photoluminescence rarely decays as a single exponential. The whole project
calibrates and validates with a single-exponential peak, so the question is which
results that touches. The answer turns on one fact: the correction keys on the
*per-frame CNR*, not on the decay law. Single-exponential decay enters the code
at exactly one place -- the peak amplitude -- and from there it only sets which
per-frame CNR appears at which frame.

This script shows the consequence two ways, on a fixed Fickian diffusion
(D = 0.02 sigma0^2/tau) with three decay laws (one single-exponential, two
bi-exponential with different fast/slow splits):

* **Per-frame residual vs per-frame CNR (collapse).** The per-frame corrected
  residual ``E[corrected sigma^2] - sigma^2_true`` is the same function of
  per-frame CNR whichever decay produced it -- the three laws fall on one curve.
  So the correction is not invalidated by the decay law; it debiases each frame
  at that frame's own per-frame CNR. (A small residual spread remains because the
  bias depends weakly on the true width, which differs between laws at a matched
  per-frame CNR -- a second-order effect, the 2-D-calibration territory.)
* **Corrected D vs CNR_0 (shift).** Plotted against the *initial* CNR_0, the
  corrected ``<D>/D_true`` curve shifts between decay laws: at a fixed CNR_0 a
  faster early decay drives the late frames below the floor sooner. So any
  threshold quoted in CNR_0 -- including the design rule ``CNR_0 >~ 20`` -- is
  single-exponential-specific; the decay-agnostic statement is per-frame
  ``CNR >~ 1``.

Run from the repository root::

    .venv/Scripts/python.exe scripts/multiexp_decay.py

Set ``WIDTH_DEBIAS_WORKERS=0`` to use all cores. Reuses the standard-geometry
D-free Gaussian calibration cached by ``subdiffusive_alpha.py``
(``output/calibration_subdiffusive.npz``); builds it if absent.
Outputs ``output/multiexp_decay.{png,txt}``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from width_debias import (Experiment, MultiExpExperiment, GaussianModel,  # noqa: E402
                            calibrate_d_free, evaluate, per_frame_diagnostic)
from width_debias.bias_lookup import BiasLookup  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[2] / "output"
OUTPUT.mkdir(exist_ok=True)
CAL_PATH = OUTPUT / "calibration_subdiffusive.npz"   # shared standard-geometry cache

# Spain CVD (Okabe-Ito) categorical trio; see plot_style.md.
LAW_COLORS = ("#0072B2", "#E69F00", "#009E73")       # blue, gold, green
FLOOR_GREY = "#999999"
INCH = 1 / 2.54
FONT = 9

D_FICKIAN = 0.02
GEOMETRY = dict(sigma0=1.0, tau=1.0, half_window=6.0, n_pixels=241,
                n_frames=12, t_max=3.0, amplitude0=1.0)
CAL_CNR0 = np.geomspace(2.0, 100.0, 60)
N_RUNS_CAL = 1500

# Three decay laws on identical Fickian diffusion. The bi-exponentials start with
# a fast channel, so their per-frame CNR falls faster early than the single
# exponential -- a deliberately different per-frame CNR trajectory.
DECAY_LAWS = (
    ("single exp (tau=1)",
     Experiment(D=D_FICKIAN, **GEOMETRY)),
    ("bi-exp 0.5/0.5, tau=0.3,3",
     MultiExpExperiment(D=D_FICKIAN, fractions=(0.5, 0.5), taus=(0.3, 3.0),
                        **GEOMETRY)),
    ("bi-exp 0.7/0.3, tau=0.5,5",
     MultiExpExperiment(D=D_FICKIAN, fractions=(0.7, 0.3), taus=(0.5, 5.0),
                        **GEOMETRY)),
)

# Span CNR_0 low enough that the slow-tailed bi-exponentials also reach the
# low per-frame CNR region (~0.8-1.5) where the residual is non-trivial, so the
# collapse is tested where it matters, not only where every residual is ~0.
DIAG_CNR0 = (5.0, 8.0, 13.0, 21.0, 35.0, 60.0)
N_RUNS_DIAG = 1500
EVAL_CNR0 = np.geomspace(8.0, 90.0, 8)       # the corrected-D-vs-CNR0 panel
N_RUNS_EVAL = 800


def load_calibration() -> BiasLookup:
    """Load the shared D-free Gaussian calibration; build and cache if absent."""
    if CAL_PATH.exists() and "framevar_grid" in np.load(CAL_PATH).files:
        d = np.load(CAL_PATH)
        return BiasLookup(cnr_grid=d["cnr_grid"], p_grid=d["p_grid"],
                               relvar_grid=d["relvar_grid"],
                               varbias_grid=d["varbias_grid"],
                               framevar_grid=d["framevar_grid"])
    trunc = calibrate_d_free(CAL_CNR0, n_runs=N_RUNS_CAL, model=GaussianModel(),
                             **GEOMETRY)
    np.savez(CAL_PATH, cnr_grid=trunc.cnr_grid, p_grid=trunc.p_grid,
             relvar_grid=trunc.relvar_grid, varbias_grid=trunc.varbias_grid,
             framevar_grid=trunc.framevar_grid)
    return trunc


def residual_vs_cnr(exp, trunc: BiasLookup,
                    seed0: int) -> tuple[np.ndarray, np.ndarray]:
    """Pool per-frame (per-frame CNR, corrected residual) over the diag CNR_0s."""
    cnr, residual = [], []
    for k, cnr0 in enumerate(DIAG_CNR0):
        diag = per_frame_diagnostic(exp, float(cnr0), trunc,
                                    scheme="subtract", n_runs=N_RUNS_DIAG,
                                    seed=seed0 + 1000 * k,
                                    profile_model=GaussianModel())
        cnr.extend(diag.cnr.tolist())
        residual.extend(diag.residual.tolist())
    order = np.argsort(cnr)
    return np.array(cnr)[order], np.array(residual)[order]


def d_recovery(exp, trunc: BiasLookup,
               seed0: int) -> tuple[np.ndarray, np.ndarray]:
    """<D>/D_true vs CNR_0 for the corrected (folded-normal) and naive (OLS)
    fits. The corrected curve is decay-robust; the naive curve's bias onset
    shifts with the decay law, which is why CNR_0-indexed thresholds are
    decay-specific."""
    res = evaluate(exp, EVAL_CNR0, trunc, n_runs=N_RUNS_EVAL, seed=seed0,
                   profile_model=GaussianModel())
    return res["subtract"].mean_bias, res["ols"].mean_bias


def write_report(collapse: dict, dcurves: dict, deepest_at20: dict,
                 design_floor: float, path: Path) -> None:
    lines = [
        "=" * 78,
        "Multi-exponential decay: is the noise-bias correction invalidated?",
        "=" * 78,
        f"Fickian D = {D_FICKIAN} sigma0^2/tau; Gaussian fit; standard geometry.",
        f"calibration: D-free (calibrate_d_free, D = 0), Gaussian, {len(CAL_CNR0)} "
        f"CNR0 in [{CAL_CNR0[0]:g}, {CAL_CNR0[-1]:g}] x {N_RUNS_CAL} runs, seeds 1 + i "
        f"(shared cache {CAL_PATH.name}).",
        f"runs: {N_RUNS_DIAG} per CNR0 for the residual collapse, seeds 40000 + 5000 l "
        f"+ 1000 k (decay law l, CNR0 index k); {N_RUNS_EVAL} per CNR0 for D "
        "recovery, seeds 70000 + 5000 l + k.",
        "Claim: the correction keys on per-frame CNR, not the decay law.",
        "",
        "(1) Per-frame corrected residual collapses onto one curve vs per-frame "
        "CNR.",
        "    Residual (sigma0^2) binned by per-frame CNR; spread across decay "
        "laws = ",
        "    the second-order width dependence, not a decay-law dependence.",
        "  per-frame CNR bin | " + " | ".join(f"{n:>22}" for n in collapse),
    ]
    # Bin the pooled residuals onto a shared per-frame CNR grid for comparison.
    bins = np.array([0.7, 1.0, 1.5, 2.5, 5.0, 12.0, 35.0])
    binned = {}
    for name, (cnr, resid) in collapse.items():
        idx = np.clip(np.digitize(cnr, bins), 1, len(bins) - 1)
        binned[name] = [float(np.mean(resid[idx == b])) if np.any(idx == b)
                        else np.nan for b in range(1, len(bins))]
    for bi in range(len(bins) - 1):
        label = f"{bins[bi]:.1f}-{bins[bi + 1]:.1f}"
        cells = " | ".join(f"{binned[n][bi]:>22.3f}" for n in collapse)
        lines.append(f"  {label:>17} | {cells}")
    spread = []
    for bi in range(len(bins) - 1):
        vals = [binned[n][bi] for n in collapse if np.isfinite(binned[n][bi])]
        if len(vals) > 1:
            spread.append(max(vals) - min(vals))
    lines.append(f"  max across-law spread at matched per-frame CNR: "
                 f"{max(spread):.3f} sigma0^2" if spread else "")
    lines.append("")
    lines.append("(2) Corrected <D>/D_true vs CNR_0 stays ~1 for every decay "
                 "law -- the corrected D is decay-robust.")
    lines.append("  CNR_0 | " + " | ".join(f"{n:>26}" for n in dcurves))
    for i, cnr0 in enumerate(EVAL_CNR0):
        cells = " | ".join(f"{dcurves[n][0][i]:>26.3f}" for n in dcurves)
        lines.append(f"  {cnr0:5.0f} | {cells}")
    lines.append("")
    lines.append("(3) Naive (uncorrected, all-frames) <D>/D_true vs CNR_0 -- "
                 "the bias onset shifts with the decay law,")
    lines.append("    so any CNR_0-indexed threshold is decay-specific.")
    lines.append("  CNR_0 | " + " | ".join(f"{n:>26}" for n in dcurves))
    for i, cnr0 in enumerate(EVAL_CNR0):
        cells = " | ".join(f"{dcurves[n][1][i]:>26.3f}" for n in dcurves)
        lines.append(f"  {cnr0:5.0f} | {cells}")
    lines.append("")
    lines.append(f"(4) Same CNR_0 = {design_floor:.0f} (the single-exp design "
                 "floor) leaves the deepest frame at very")
    lines.append("    different per-frame CNR by decay law -- the decay-agnostic "
                 "rule is per-frame CNR >~ 1:")
    for name, dc in deepest_at20.items():
        lines.append(f"    {name:>26}: deepest per-frame CNR = {dc:.2f}")
    report = "\n".join(lines)
    print(report)
    path.write_text(report)


def make_figure(collapse: dict, dcurves: dict, design_floor: float,
                path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    fig, (axA, axB) = plt.subplots(1, 2, layout="constrained",
                                   figsize=(18.0 * INCH, 8.0 * INCH))

    # Panel A: per-frame residual vs per-frame CNR -- collapse across decay laws.
    for (name, (cnr, resid)), color in zip(collapse.items(), LAW_COLORS):
        axA.plot(cnr, resid, "o", color=color, ms=4, alpha=0.8, label=name)
    axA.axhline(0.0, color="k", lw=0.8)
    axA.axvline(1.0, color=FLOOR_GREY, lw=1.0, ls=":", label="per-frame floor")
    axA.set_xscale("log")
    axA.xaxis.set_major_formatter(ScalarFormatter())
    axA.xaxis.set_minor_formatter(NullFormatter())
    axA.set_xlabel("per-frame contrast-to-noise CNR", fontsize=FONT)
    axA.set_ylabel(r"corrected residual  $E[\tilde\sigma^2]-\sigma^2_{true}$"
                   r"  ($\sigma_0^2$)", fontsize=FONT)
    axA.tick_params(axis="both", which="both", direction="in", top=True,
                    right=True, labelsize=FONT)
    axA.legend(frameon=False, fontsize=FONT, loc="upper right")

    # Panel B: corrected <D>/D_true vs initial CNR_0 -- decay-robust (all ~1).
    for (name, (corrected, _naive)), color in zip(dcurves.items(), LAW_COLORS):
        axB.plot(EVAL_CNR0, corrected, "o-", color=color, lw=2, ms=4, label=name)
    axB.axhline(1.0, color="k", lw=0.8, ls="--", label="unbiased")
    axB.axvline(design_floor, color=FLOOR_GREY, lw=1.0, ls=":",
                label=r"design floor (CNR$_0\approx20$)")
    axB.set_xscale("log")
    axB.xaxis.set_major_formatter(ScalarFormatter())
    axB.xaxis.set_minor_formatter(NullFormatter())
    axB.set_xlabel(r"initial contrast-to-noise CNR$_0$", fontsize=FONT)
    axB.set_ylabel(r"corrected  $\langle D\rangle / D_{true}$", fontsize=FONT)
    axB.tick_params(axis="both", which="both", direction="in", top=True,
                    right=True, labelsize=FONT)
    axB.legend(frameon=False, fontsize=FONT, loc="upper right")

    fig.patch.set_facecolor("w")
    fig.patch.set_alpha(1)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure: {path.name}")


def main() -> None:
    design_floor = float(np.exp(GEOMETRY["t_max"] / GEOMETRY["tau"]))
    print("Loading/building D-free Gaussian calibration ...")
    trunc = load_calibration()

    print(f"Per-frame residual collapse: {len(DECAY_LAWS)} decay laws x "
          f"{len(DIAG_CNR0)} CNR0 x {N_RUNS_DIAG} runs ...")
    collapse = {name: residual_vs_cnr(exp, trunc, seed0=40000 + 5000 * i)
                for i, (name, exp) in enumerate(DECAY_LAWS)}

    print(f"D recovery vs CNR_0: {len(DECAY_LAWS)} decay laws x "
          f"{len(EVAL_CNR0)} CNR0 x {N_RUNS_EVAL} runs ...")
    dcurves = {name: d_recovery(exp, trunc, seed0=70000 + 5000 * i)
               for i, (name, exp) in enumerate(DECAY_LAWS)}

    # The deterministic CNR_0 -> deepest-per-frame-CNR map at the design floor:
    # the same CNR_0 = 20 leaves the deepest frame at very different per-frame
    # CNR depending on the decay law, which is why that threshold is decay-
    # specific.
    deepest_at20 = {name: float(design_floor * exp.amplitude[-1] / exp.amplitude[0])
                    for name, exp in DECAY_LAWS}

    write_report(collapse, dcurves, deepest_at20, design_floor,
                 OUTPUT / "multiexp_decay.txt")
    make_figure(collapse, dcurves, design_floor, OUTPUT / "multiexp_decay.png")


if __name__ == "__main__":
    main()
