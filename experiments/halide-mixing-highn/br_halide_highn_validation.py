"""Validate the noise-bias correction on the high-statistics Br-I maps.

The companion to :mod:`br_halide_validation`, run on the clean 400k-walker
re-simulation (``experiments/halide-mixing-highn/br_highn_data.py``) instead of
Marc's original 20k-walker maps. Two things change versus the legacy bed, and
both matter:

  * **Long window.** 601 frames to 3.0 ns (vs 0.1 ns). With the imposed
    tau ~ 1 ns decay the late-frame per-frame CNR now falls through ~1 -- the
    regime the correction is built for. The legacy 0.1 ns window never reached
    it, so it only ever demonstrated transport recovery, never the correction.
  * **Clean truth + wide box.** 400k walkers push the map's own sampling-noise
    floor up to an intrinsic peak CNR ~50, so the imposed measurement noise --
    not the map's graininess -- sets the per-frame CNR across the sweep. And the
    3000 nm box leaves the spot at +/-7.6 sigma (x=0.25) even at 3 ns, so the
    binned moment is not clipped. The legacy bed's narrow-box / moment-clipping
    caveat therefore does not apply here.

The carrier (particle) truth is ``variance.avg.dat / 2``. For each Br fraction
and CNR_0 the naive (raw all-frames) and corrected (``subtract`` scheme,
sub-floor frames dropped only at the slope fit) power-law fits MSD = 2 D t^alpha
are scored against that truth.

Run from the repository root (needs the pshare for the maps)::

    WIDTH_DEBIAS_WORKERS=0 .venv/Scripts/python.exe experiments/halide-mixing-highn/br_halide_highn_validation.py

Outputs ``output/br_halide_highn_validation.{png,txt}`` and caches the matched
D-free calibration to ``output/calibration_br_highn.npz``.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "halide-mixing-brI-legacy"))
sys.path.insert(0, os.path.dirname(__file__))

from width_debias import GaussianModel, calibrate_d_free  # noqa: E402
from width_debias.models import MomentModel  # noqa: E402
from width_debias.bias_lookup import BiasLookup  # noqa: E402
from width_debias.weights import corrected_variance  # noqa: E402

# Pure helpers shared with the legacy bed (no module-global dependencies).
from br_halide_validation import fit_each_row, fit_msd_powerlaw  # noqa: E402

from br_highn_data import (  # noqa: E402
    INTRINSIC_PEAK_CNR, SIGMA0_HINT_NM, load_all)

OUTPUT = Path(__file__).resolve().parents[2] / "output"
OUTPUT.mkdir(exist_ok=True)
CAL_PATH = OUTPUT / "calibration_br_highn.npz"

CORR_BLUE = "#0072B2"
NAIVE_RED = "#D55E00"
TRUTH_BLACK = "#000000"
INCH = 1 / 2.54
FONT = 9

# Sweep capped at the map's intrinsic peak CNR: above it the map's own sampling
# noise competes with the imposed noise and the test stops being controlled.
CNR0_SWEEP = (2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 50.0)
N_SEEDS = 30
FLOOR_CNR = 1.0
# Analyse every STRIDE-th frame: 601 frames is far more than the MSD power law
# needs, and the per-frame Gaussian fits dominate the runtime. Stride 10 leaves
# 61 frames (25 ps spacing) spanning 0..3 ns -- dense for the slope.
STRIDE = 10

# Calibration geometry matched to the high-N maps: initial 1-D sigma ~25 nm,
# 600-bin 3000 nm window -> a wide +/-60 sigma window (no truncation), 5 nm/bin
# (0.2 sigma/pixel) resolution. n_frames/t_max only need to span the per-frame
# CNR range, so they stay light; cnr0_grid carries the lookup.
CAL_GEOMETRY = dict(sigma0=SIGMA0_HINT_NM, tau=1.0,
                    half_window=1500.0 / SIGMA0_HINT_NM,
                    n_pixels=600, n_frames=21, t_max=3.0)
CAL_CNR0 = np.geomspace(1.2, 60.0, 40)
N_RUNS_CAL = 1500


def subsample(case, stride: int):
    """Return a frame-strided shallow copy of a BrCase (faithful, just fewer t)."""
    from dataclasses import replace
    sl = slice(None, None, stride)
    return replace(case, t=case.t[sl], profiles=case.profiles[sl],
                   var_true=case.var_true[sl])


def build_or_load_calibration() -> BiasLookup:
    """Gaussian D-free calibration at the high-N geometry; cached after first build."""
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
    name: str
    t: np.ndarray
    var_true: np.ndarray
    a_true: float
    a_nf: float
    shape_gap: float
    cnr0: np.ndarray
    min_cnr: np.ndarray
    n_work: np.ndarray
    a_naive: np.ndarray
    a_meth: np.ndarray
    d_naive: np.ndarray
    d_meth: np.ndarray
    d_true: float


def sweep_case(case, trunc: BiasLookup) -> CaseResult:
    """Run the CNR_0 sweep for one Br fraction (the single source of the numbers)."""
    nf = fit_each_row(case.x, case.profiles)              # noiseless Gaussian fit
    d_true, a_true = fit_msd_powerlaw(case.t, case.var_true)
    _, a_nf = fit_msd_powerlaw(case.t, nf)
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
        a_nf=a_nf, d_true=d_true,
        shape_gap=float(np.nanmean(np.abs(nf - case.var_true))),
        cnr0=np.array(CNR0_SWEEP), min_cnr=np.array(rows["min_cnr"]),
        n_work=np.array(rows["n_work"]), a_naive=np.array(rows["a_naive"]),
        a_meth=np.array(rows["a_meth"]), d_naive=np.array(rows["d_naive"]),
        d_meth=np.array(rows["d_meth"]))


def write_report(results: list[CaseResult], n_frames: int, path: Path) -> None:
    lines = [
        "=" * 76,
        "Br-I halide-mixing validation -- HIGH-N (400k-walker) maps",
        "subtract, Gaussian fit; carrier truth = particle 2nd moment / 2.",
        "3000 nm box (no clipping), 3.0 ns window (per-frame CNR reaches ~1).",
        "=" * 76,
        f"n_seeds per CNR_0: {N_SEEDS};  frame stride: {STRIDE} ({n_frames} frames);"
        f"  per-frame CNR floor: {FLOOR_CNR}",
        f"CNR_0 sweep capped at the map's intrinsic peak CNR ~{INTRINSIC_PEAK_CNR:.0f}.",
        "naive = raw all-frames fit;  method = corrected + sub-floor frames dropped.",
        "alpha and D read from the mean trajectory; D in nm^2/ns^alpha.",
        "",
    ]
    for r in results:
        lines.append("-" * 76)
        lines.append(f"=== {r.name} ===  shape gap (fit vs carrier truth) "
                     f"{r.shape_gap:.1f} nm^2")
        lines.append(f"  carrier truth: alpha = {r.a_true:.3f}, D = {r.d_true:.4g};"
                     f"  noiseless fit alpha = {r.a_nf:.3f} "
                     f"({r.a_nf - r.a_true:+.3f} vs carrier)")
        lines.append("  CNR_0 | min CNR | n>=floor | alpha naive/method | "
                     "D naive/method")
        for i, c in enumerate(r.cnr0):
            lines.append(f"  {c:5.0f} | {r.min_cnr[i]:7.2f} | "
                         f"{r.n_work[i]:3d}/{n_frames} | "
                         f"{r.a_naive[i]:.3f}/{r.a_meth[i]:.3f} | "
                         f"{r.d_naive[i]:.4g}/{r.d_meth[i]:.4g}")
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
        print("No high-N collapsed maps found on the pshare; nothing to validate.")
        return
    cases = [subsample(c, STRIDE) for c in cases]
    n_frames = cases[0].profiles.shape[0]
    print(f"Loaded {len(cases)} fraction(s): {[c.name for c in cases]};"
          f" {n_frames} frames after stride {STRIDE}.")
    print(f"Building/loading high-N Gaussian D-free calibration "
          f"({len(CAL_CNR0)} CNR0 x {N_RUNS_CAL} runs) ...")
    trunc = build_or_load_calibration()
    results = [sweep_case(case, trunc) for case in cases]
    write_report(results, n_frames, OUTPUT / "br_halide_highn_validation.txt")
    make_figure(results, OUTPUT / "br_halide_highn_validation.png")


if __name__ == "__main__":
    main()
