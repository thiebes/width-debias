"""Apparent vs true MSD exponent on a Gaussian bed built from the NC-solid maps.

Quantifies how a standard bounded-fit MSD-slope analysis mis-reads the transport
exponent alpha (and the diffusion coefficient D) as the per-frame
contrast-to-noise falls -- the regime a real transient-PL measurement enters at
late times.

The bed: a Gaussian is fit to each of Marc's Fig.-5 simulation frames and *taken
as the truth* -- a test bed whose per-frame width and amplitude are the maps' real
subdiffusive transport, but whose ground truth is Gaussian by construction. This
makes no claim that the real profile is Gaussian; it defines a realistic-width
Gaussian bed with an exactly known truth (``nc_data.load_case(denoise=True)``).
It also sets aside the per-bin Poisson noise of the ~50,000-walker histograms,
which would otherwise inflate the per-frame CNR (the raw ``max`` rides the
sampling noise and rises with time instead of falling as 1/sigma). The maps are
near-Gaussian (kurtosis ~ 0), so the bed is faithful.

Story the figure tells: the noise bias lifts the late-time MSD, so it
pushes apparent alpha *up* toward ballistic (alpha = 2). The true subdiffusive
exponent (alpha ~ 0.88) only re-emerges once the deepest usable frame keeps
CNR >~ 2-3, i.e. CNR_0 >~ 35-50. The bias therefore masks subdiffusion, never
manufactures it: a measured alpha < 1 is robust to this effect, possibly
conservative. Per-seed power-law fits give an honest spread (mean +/- s.d. over
seeds) rather than a single fit to the seed-mean trajectory.

Run from the repository root::

    .venv/Scripts/python.exe scripts/nc_solid_alpha_vs_cnr.py

Outputs ``output/nc_solid_alpha_vs_cnr.{png,txt}``.
"""
from __future__ import annotations

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

from width_debias import CalibrationGrid  # noqa: E402
from width_debias.fit import fit_variance as fit_gaussian  # noqa: E402
from width_debias.noise import ShotNoise, WhiteNoise  # noqa: E402
from width_debias.bias_lookup import BiasLookup  # noqa: E402
from width_debias.weights import corrected_variance  # noqa: E402

from nc_data import load_all, load_case  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[2] / "output"
OUTPUT.mkdir(exist_ok=True)
CAL_PATH = OUTPUT / "calibration_nc_solid_gaussian.npz"

PALETTE = {"blue": "#003f7f", "gold": "#f7941e", "green": "#0cce6b"}
CASE_COLOR = {"small": "#003f7f", "medium": "#f7941e", "large": "#0cce6b"}
inch = 1 / 2.54

CNR_GRID = (3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 50.0, 75.0, 100.0)
N_SEEDS = 1000
FLOOR_CNR = 1.0
TAU_SWEEP = (6.0, 14.0, 18.0)   # C16 / C12 / C8 thin-film lifetimes (ns)

# Noise laws for the noise-law comparison (small map). White is the constant-
# floor worst case; pure shot (read=0) has clean wings where the width fit takes
# its leverage, so the noise bias nearly vanishes; a read floor of a few
# counts restores it. The label is what appears in the legend / cache key; the
# colour keys the figure. See experiments/noise-law/ and the fft-cnr paper
# cases 12/13 (the bias-governing CNR is the peak-local member, which
# ShotNoise.cnr_frame returns as peak / sqrt(peak + read^2)).
NOISE_LAWS = (
    ("white", WhiteNoise(), "#003f7f"),
    ("shot, read=0", ShotNoise(read=0.0, poisson=True), "#0cce6b"),
    ("shot, read=3", ShotNoise(read=3.0, poisson=True), "#f7941e"),
    ("shot, read=10", ShotNoise(read=10.0, poisson=True), "#d55e00"),
)


def fit_powerlaw(t, var, mask=None):
    """Field-standard MSD = 2 D t^alpha fit; returns (D, alpha) or (nan, nan)."""
    msd = var - var[0]
    m = (t > 0) & np.isfinite(msd)
    if mask is not None:
        m = m & mask
    if m.sum() < 3:
        return np.nan, np.nan
    try:
        popt, _ = curve_fit(
            lambda tt, d, a: 2.0 * d * tt ** a, t[m], msd[m],
            p0=[max(msd[m][-1] / (2.0 * t[m][-1]), 1e-6), 1.0],
            bounds=([0.0, 0.2], [np.inf, 2.0]), maxfev=20000)
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        return np.nan, np.nan


def fit_each_row(x, profiles):
    """Fitted variance per row. A failed fit keeps its fallback value, as the
    calibration's Monte Carlo does; dropping it here but not there would compare
    two different estimators."""
    out = np.full(profiles.shape[0], np.nan)
    for i in range(profiles.shape[0]):
        r = fit_gaussian(x, profiles[i])
        if np.isfinite(r.variance):
            out[i] = r.variance
    return out


def compute_row(x, t, profiles, cnr0, n_seeds, trunc, noise_model, seed_base=0):
    """Apparent transport at one CNR_0 over n_seeds noise realisations.

    The noise law is pluggable (``noise_model`` is a ``width_debias.noise``
    model). ``cnr_frame`` is the per-frame contrast-to-noise the model reports
    -- for white noise the peak over a constant RMS, for shot noise the
    peak-local member ``peak / sqrt(peak + read^2)`` (the bias-governing CNR per
    the fft-cnr family result). The corrected trajectory uses the white D=0
    calibration, so it is only meaningful under white noise; under shot noise it
    over-corrects and is reported but not relied on.

    Two complementary quantities:

    * the *bias* -- the exponent / D of the seed-averaged MSD trajectory. This
      is what survives averaging many measurements (the bias that inverse-
      variance weighting and replication cannot remove).
    * the single-shot *spread* -- the s.d. of the per-seed exponent, i.e. how
      far one real measurement scatters around that bias.
    """
    cnr_frame = noise_model.cnr_frame(profiles, cnr0)
    work = cnr_frame >= FLOOR_CNR
    a_seed = []                              # per-seed (single-shot) spread
    raw_stack, cor_stack = [], []            # for the averaged trajectory
    for s in range(n_seeds):
        rng = np.random.default_rng(seed_base + int(cnr0) * 100000 + s)
        noisy = noise_model.sample(profiles, cnr0, rng)
        var_hat = fit_each_row(x, noisy)
        cor = corrected_variance("subtract", var_hat, cnr_frame, trunc)
        raw_stack.append(var_hat); cor_stack.append(cor)
        _, aN = fit_powerlaw(t, var_hat)
        a_seed.append(aN)
    raw_mean = np.nanmean(np.vstack(raw_stack), axis=0)
    cor_mean = np.nanmean(np.vstack(cor_stack), axis=0)
    d_bias_n, a_bias_n = fit_powerlaw(t, raw_mean)
    d_bias_c, a_bias_c = fit_powerlaw(t, cor_mean, mask=work)
    return dict(
        cnr0=float(cnr0), min_cnr=float(cnr_frame.min()), n_work=int(work.sum()),
        a_bias_naive=a_bias_n, a_bias_corr=a_bias_c,
        d_bias_naive=d_bias_n, d_bias_corr=d_bias_c,
        a_seed_sd=float(np.nanstd(a_seed)))


# --- process-pool plumbing (one task per (case, CNR_0) cell) ----------------
_TRUNC = None


def _init_worker(cal_path):
    global _TRUNC
    _TRUNC = BiasLookup.from_calibration(CalibrationGrid.load(cal_path))


def _row_worker(packed):
    key, x, t, profiles, cnr0, n_seeds, noise_model, seed_base = packed
    return key, compute_row(x, t, profiles, cnr0, n_seeds, _TRUNC,
                            noise_model, seed_base)


def _n_workers():
    env = os.environ.get("WIDTH_DEBIAS_WORKERS")
    n = int(env) if env not in (None, "") else 1
    if n <= 0:
        n = os.cpu_count() or 1
    return max(1, n)


def run_tasks(tasks):
    """Run (key, x, t, profiles, cnr0, n_seeds, noise_model, seed_base) tasks."""
    n = _n_workers()
    if n == 1:
        _init_worker(CAL_PATH)
        return dict(_row_worker(t) for t in tasks)
    out = {}
    with ProcessPoolExecutor(max_workers=n, initializer=_init_worker,
                             initargs=(CAL_PATH,)) as ex:
        for key, row in ex.map(_row_worker, tasks):
            out[key] = row
    return out


def main():
    if not CAL_PATH.exists():
        print(f"Calibration not found: {CAL_PATH.name}.")
        return
    cases = load_all(denoise=True)
    if not cases:
        print("No NC-solid maps on the pshare.")
        return

    # Main sweep: one task per (case, CNR_0) cell, fanned over processes.
    # Every task carries its noise model and a seed offset; the main / tau
    # sweeps use white noise (so they reproduce the earlier run bit-for-bit).
    white = WhiteNoise()
    tasks = []
    for case in cases:
        for cnr0 in CNR_GRID:
            tasks.append((("main", case.name, cnr0),
                          case.x, case.t, case.profiles, cnr0, N_SEEDS, white, 0))
    # Lifetime sensitivity on the 'small' map, same machinery.
    tau_cases = {tau: load_case("small", tau_ns=tau, denoise=True)
                 for tau in TAU_SWEEP}
    for tau, c in tau_cases.items():
        for cnr0 in (20.0, 50.0):
            tasks.append((("tau", tau, cnr0), c.x, c.t, c.profiles, cnr0,
                          N_SEEDS, white, 0))
    # Noise-law comparison on the 'small' map (tau = 7 ns): white vs shot with a
    # read-floor sweep. The white law reuses the main small-map white result
    # (skip it here) so the two figures agree exactly; each shot law gets its
    # own seed offset so realisations are independent across laws.
    small = next(c for c in cases if c.name == "small")
    for li, (label, model, _color) in enumerate(NOISE_LAWS):
        if label == "white":
            continue
        for cnr0 in CNR_GRID:
            tasks.append((("noise", label, cnr0), small.x, small.t,
                          small.profiles, cnr0, N_SEEDS, model,
                          li * 10_000_000))

    # Cache the Monte Carlo so figure/table edits do not pay the full run.
    # Delete output/nc_solid_alpha_vs_cnr_cache.pkl to force a recompute.
    cache = OUTPUT / "nc_solid_alpha_vs_cnr_cache.pkl"
    meta = dict(n_seeds=N_SEEDS, cnr=tuple(CNR_GRID), tau=tuple(TAU_SWEEP),
                noise=tuple(label for label, _m, _c in NOISE_LAWS),
                scheme="v3-denoise")
    results = None
    if cache.exists():
        blob = pickle.loads(cache.read_bytes())
        if blob.get("meta") == meta:
            results = blob["results"]
            print("Loaded Monte Carlo from cache (delete the .pkl to recompute).")
    if results is None:
        print(f"Running {len(tasks)} cells x {N_SEEDS} seeds on "
              f"{_n_workers()} worker(s)...")
        results = run_tasks(tasks)
        cache.write_bytes(pickle.dumps(dict(meta=meta, results=results)))

    lines = ["=" * 72,
             "Apparent vs true MSD exponent on Solari NC-solid maps (tau = 7 ns)",
             "=" * 72,
             f"n_seeds = {N_SEEDS};  floor = per-frame CNR >= {FLOOR_CNR}",
             ""]
    truth = {}
    sweeps = {}
    for case in cases:
        D_true, a_true = fit_powerlaw(case.t, case.var_true)
        truth[case.name] = (D_true, a_true)
        rows = [results[("main", case.name, cnr0)] for cnr0 in CNR_GRID]
        sweeps[case.name] = rows
        lines.append(f"--- {case.name} ---  truth: D = {D_true:.3g} um^2/ns, "
                     f"alpha = {a_true:.3f}")
        lines.append("  alpha bias = exponent of the seed-averaged MSD (survives "
                     "averaging); +/- = single-shot s.d.")
        lines.append("  CNR_0 | minCNR | n>=1 | alpha bias naive/corr (1-shot sd) "
                     "| D/Dtrue naive/corr")
        for r in rows:
            lines.append(
                f"  {r['cnr0']:5.0f} | {r['min_cnr']:6.2f} | "
                f"{r['n_work']:2d}/{case.t.size} | "
                f"{r['a_bias_naive']:.3f}/{r['a_bias_corr']:.3f} "
                f"(+/-{r['a_seed_sd']:.2f}) | "
                f"{r['d_bias_naive'] / D_true:.2f}/{r['d_bias_corr'] / D_true:.2f}")
        lines.append("")

    # --- lifetime sensitivity on the 'small' map: longer tau holds CNR up ----
    lines.append("-" * 72)
    lines.append("Lifetime sensitivity (small map): deepest-frame CNR and naive "
                 "alpha vs tau")
    lines.append("  tau (ns) | CNR_0 | min CNR | alpha naive   "
                 "(truth alpha = {:.3f})".format(truth["small"][1]))
    for tau in TAU_SWEEP:
        for cnr0 in (20.0, 50.0):
            r = results[("tau", tau, cnr0)]
            lines.append(f"  {tau:7.0f} | {cnr0:5.0f} | {r['min_cnr']:7.2f} | "
                         f"{r['a_bias_naive']:.3f}")
    lines.append("")

    # --- companion threshold table (the numbers the figure cannot show) ------
    tol = 0.05
    lines.append("-" * 72)
    lines.append(f"Recovery threshold per map (tau = 7 ns): smallest CNR_0 with "
                 f"|alpha_bias - alpha_true| <= {tol}")
    lines.append("  map    | true alpha | true D (um^2/ns) | CNR_0* | deepest-"
                 "frame CNR* | 1-shot sd*")
    for name in ("small", "medium", "large"):
        D_true, a_true = truth[name]
        rec = next((r for r in sweeps[name]
                    if abs(r["a_bias_naive"] - a_true) <= tol), None)
        if rec is None:
            lines.append(f"  {name:6s} | {a_true:10.3f} | {D_true:16.3g} | "
                         "  none in grid")
            continue
        lines.append(f"  {name:6s} | {a_true:10.3f} | {D_true:16.3g} | "
                     f"{rec['cnr0']:6.0f} | {rec['min_cnr']:17.1f} | "
                     f"+/-{rec['a_seed_sd']:.2f}")
    lines.append("  * CNR_0 at recovery; deepest usable frame's CNR there; "
                 "single-shot exponent s.d. there.")
    lines.append("")

    # --- noise-law comparison on the 'small' map ----------------------------
    # The white-noise bias above is the constant-floor worst case. Pure shot
    # noise has clean wings, so the noise bias should nearly vanish; a read
    # floor of a few counts restores it. (fft-cnr paper cases 12/13.)
    def noise_row(label, cnr0):
        """Row for a noise law; white reuses the main small-map white result."""
        if label == "white":
            return results[("main", "small", cnr0)]
        return results[("noise", label, cnr0)]

    a_true_small = truth["small"][1]
    lines.append("-" * 72)
    lines.append("Noise-law comparison (small map, tau = 7 ns): naive alpha bias "
                 f"vs CNR_0  (truth alpha = {a_true_small:.3f})")
    header = "  CNR_0 | " + " | ".join(f"{label:>13s}"
                                       for label, _m, _c in NOISE_LAWS)
    lines.append(header)
    for cnr0 in CNR_GRID:
        cells = []
        for label, _m, _c in NOISE_LAWS:
            r = noise_row(label, cnr0)
            cells.append(f"{r['a_bias_naive']:5.2f}({r['min_cnr']:4.1f})")
        lines.append(f"  {cnr0:5.0f} | " + " | ".join(f"{c:>13s}" for c in cells))
    lines.append("  cell = alpha_bias (deepest-frame CNR).  White pins to the "
                 "ballistic ceiling; shot stays near truth until a read floor "
                 "restores the bias.")
    lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUTPUT / "nc_solid_alpha_vs_cnr.txt").write_text(report)

    # --- Figure: apparent exponent vs CNR_0 (single panel) ------------------
    # Tufte pass: one panel (the D-ratio panel contradicted the headline via
    # the alpha-D coupling, so it is reported in the table instead); references
    # are neutral grey so colour means only the data series; no scatter bands
    # (the single-shot s.d. lives in the table); lines labelled in place.
    fig, ax = plt.subplots(layout="constrained",
                           figsize=(9.5 * inch, 8 * inch))
    fs = 9
    a_truth_lo = min(truth[c][1] for c in truth)
    a_truth_hi = max(truth[c][1] for c in truth)

    GREY_REF = "#9a9a9a"
    ax.axhline(2.0, color=GREY_REF, lw=1.0, ls=":")
    ax.text(2.9, 1.97, "ballistic ceiling", fontsize=fs - 1, color="#6b6b6b",
            va="top")
    ax.axhspan(a_truth_lo, a_truth_hi, color=GREY_REF, alpha=0.30)
    ax.text(70, (a_truth_lo + a_truth_hi) / 2 - 0.015, "true $\\alpha$",
            fontsize=fs - 1, color="#5a5a5a", va="top", ha="center")
    for name in ("small", "medium", "large"):
        rows = sweeps[name]
        x = np.array([r["cnr0"] for r in rows])
        y = np.array([r["a_bias_naive"] for r in rows])
        ax.plot(x, y, marker="o", ms=4, lw=2, color=CASE_COLOR[name],
                label=name)
    ax.set_xscale("log")
    ax.set_xlabel(r"initial contrast-to-noise  CNR$_0$", fontsize=fs)
    ax.set_ylabel(r"apparent MSD exponent  $\alpha$", fontsize=fs)
    ax.set_ylim(0.6, 2.05)
    ax.tick_params(axis="both", which="both", direction="in", top=True,
                   right=True, labelsize=fs)
    # small and medium nearly coincide (near-equal true alpha), so end-labels
    # would collide; a minimal frameless key sits in the empty lower-left.
    ax.legend(fontsize=fs - 1, frameon=False, loc="lower left",
              handlelength=1.2)

    fig.patch.set_facecolor("w")
    fig.patch.set_alpha(1)
    path = OUTPUT / "nc_solid_alpha_vs_cnr.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {path.name}")

    # --- Figure 2: noise law governs the bias (small map) -------------------
    # Same axes as figure 1, but the lines are noise laws on one map: the white
    # constant-floor worst case pins to the ceiling, pure shot tracks the truth
    # (clean wings), and a read floor walks it back up toward white.
    fig2, ax2 = plt.subplots(layout="constrained",
                             figsize=(9.5 * inch, 8 * inch))
    ax2.axhline(2.0, color=GREY_REF, lw=1.0, ls=":")
    ax2.text(2.9, 1.97, "ballistic ceiling", fontsize=fs - 1, color="#6b6b6b",
             va="top")
    ax2.axhspan(a_truth_lo, a_truth_hi, color=GREY_REF, alpha=0.30)
    ax2.text(70, (a_truth_lo + a_truth_hi) / 2 - 0.015, "true $\\alpha$",
             fontsize=fs - 1, color="#5a5a5a", va="top", ha="center")
    for label, _model, color in NOISE_LAWS:
        rows = [noise_row(label, cnr0) for cnr0 in CNR_GRID]
        x = np.array([r["cnr0"] for r in rows])
        y = np.array([r["a_bias_naive"] for r in rows])
        ax2.plot(x, y, marker="o", ms=4, lw=2, color=color, label=label)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"initial contrast-to-noise  CNR$_0$", fontsize=fs)
    ax2.set_ylabel(r"apparent MSD exponent  $\alpha$", fontsize=fs)
    ax2.set_ylim(0.6, 2.05)
    ax2.tick_params(axis="both", which="both", direction="in", top=True,
                    right=True, labelsize=fs)
    ax2.legend(fontsize=fs - 1, frameon=False, loc="lower left",
               handlelength=1.2)
    fig2.patch.set_facecolor("w")
    fig2.patch.set_alpha(1)
    path2 = OUTPUT / "nc_solid_noise_law.png"
    fig2.savefig(path2, dpi=300)
    plt.close(fig2)
    print(f"Saved figure: {path2.name}")


if __name__ == "__main__":
    main()
