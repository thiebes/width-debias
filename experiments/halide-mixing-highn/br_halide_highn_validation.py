"""Validate the 1-D D-free lookup on the high-statistics Br-I maps.

The companion to ``halide-mixing-brI-legacy/br_halide_validation.py``, run on
the clean 400k-walker re-simulation (``br_highn_data.py``). The lookup under
test is the package's recommended 1-D path, ``calibrate_d_free``: one
calibration at D = 0 and the initial width sigma_0, read by the per-frame CNR
alone. On this bed the width grows over the fitted frames, and at a given CNR
the bias grows with the width (in proportion to sigma to leading order, at a
fixed pixel pitch; ``docs/DERIVATION.md``). The question is how far a lookup
that ignores the width is off when sigma^2 grows several-fold. The per-frame
study (``br_halide_highn_perframe.py``) tests the width-aware lookups on the
same maps.

Three design points:

  * **Width-fit bound.** The Gaussian width fit is bounded at
    ``max_var = (G * sigma_0)^2`` with G = 12 (sigma capped at 300 nm), in the
    calibration and in the application fit alike. With the default span^2
    bound, the low-CNR fits wander toward the 3000 nm window and the
    calibration's bias explodes (the README's "The upper bound"). The bound,
    the bounded fit and the per-frame CNR floor are defined here;
    ``br_halide_highn_perframe.py`` imports them, and the set's other scripts
    read the fit and the floor through it, so every high-N script fits with
    one bound and one floor.
  * **Fitted frames.** The peak falls about 150-fold over the 3 ns record, so
    only the frames at or above the per-frame CNR floor of 1.3 enter the
    slope fit. Each arm's D is divided by the D of the noise-free fit over
    the same frames, the noise correction's target. The gap between that fit
    and the carrier second moment is profile shape, which no noise
    correction addresses.
  * **Arms.** naive: the raw fits on the frames at or above the floor.
    naive, CNR >= 3: the raw fits on the frames at or above per-frame CNR 3,
    the stricter cut a careful analyst would use instead of a correction.
    corrected: the raw fits minus the lookup's bias, on the same frames as
    naive. corrected, CNR >= 3: the same corrected fits on the frames of the
    stricter cut, where the lookup is 14 times its leading term at CNR 3
    and about twice it from CNR 4, against 110 to 132 times at CNR 1.3 to 2.

The table's header gives the lookup at sigma_0, with its SE, next to its
leading term, the width's Fisher variance on the calibration's pixel grid
(``width_debias.fisher.leading_order_terms``). Below per-frame CNR 3 the
lookup is many times that term: at sigma_0 the low-CNR fits wander toward
the width cap in the wide window, and the wider spots of the late frames do
not.

The power-law fit bounds the exponent to [0.2, 2.0]. An arm whose exponent
ends within 0.01 of a bound is printed "at bound" in place of its D: its D
is then set by the bound, and its resampled SE collapses
(``experiments/subdiffusive-alpha`` leaves such cells out for the same
reason). A frame's residual is its fitted variance, averaged over the
seeds, minus the variance fitted to the same frame's noise-free profile; a
band's residual is its mean over the band's frames, with SEs from the same
seed resamples and lookup draws as D.

Run from the repository root (needs the maps in the data root)::

    WIDTH_DEBIAS_WORKERS=4 .venv/Scripts/python.exe experiments/halide-mixing-highn/br_halide_highn_validation.py

The workers speed up only the first build of the calibration; the
evaluation fits run in one process.

Writes the table ``br_halide_highn_validation.txt`` into this folder, the
figure to ``output/br_halide_highn_validation.png``, and caches the
calibration in ``output/calibration_br_highn_g12_n1500.npz``.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "halide-mixing-brI-legacy"))
sys.path.insert(0, os.path.dirname(__file__))

from width_debias import GaussianModel, calibrate_d_free  # noqa: E402
from width_debias.bias_lookup import BiasLookup  # noqa: E402
from width_debias.fisher import leading_order_terms  # noqa: E402
from width_debias.fit import fit_variance as _fit_gaussian  # noqa: E402
from width_debias.simulate import Experiment  # noqa: E402
from width_debias.weights import corrected_variance  # noqa: E402

# The power-law fit and its exponent bounds, shared with the legacy bed.
from br_halide_validation import ALPHA_BOUNDS, fit_msd_powerlaw  # noqa: E402

from br_highn_data import (  # noqa: E402
    INTRINSIC_PEAK_CNR, MAP_DIR, SIGMA0_HINT_NM, load_all)

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[1] / "output"
OUTPUT.mkdir(exist_ok=True)

# Tighten the width fit's upper bound from the window span^2 to a physical cap
# (G * sigma0)^2: the spot grows up to ~8x in sigma over the 3 ns record, so
# G = 12 clears every real frame (sigma capped at 300 nm) while bounding the
# low-CNR fit's noise-wandering, which the wide 3000 nm window otherwise leaves
# unbounded. Applied identically to the data fit and the calibration (the
# var_bias must come from the same bound).
G_MAX = 12.0
MAX_VAR = (G_MAX * SIGMA0_HINT_NM) ** 2

OUT_TXT = HERE / "br_halide_highn_validation.txt"
OUT_PNG = OUTPUT / "br_halide_highn_validation.png"

# Sweep capped at the map's intrinsic peak CNR: above it the map's own sampling
# noise competes with the imposed noise and the test stops being controlled.
# CNR_0 15 and 20 are the per-frame study's.
CNR0_SWEEP = (2.0, 3.0, 5.0, 8.0, 15.0, 20.0, 30.0, 50.0)
N_SEEDS = 200
N_BOOT = 200
N_CAL_DRAWS = 200
# The project's per-frame CNR floor, from the per-frame study: its joint
# (CNR, sigma^2) lookup over-subtracts below about 1.3.
FLOOR_CNR = 1.3
STRICT_CNR = 3.0      # the stricter cut of the naive competitor
BOUND_TOL = 0.01
# Analyse every STRIDE-th frame: 601 frames is far more than the MSD power law
# needs, and the per-frame Gaussian fits dominate the runtime. Stride 10 leaves
# 61 frames (50 ps spacing) spanning 0..3 ns.
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
CAL_PATH = OUTPUT / f"calibration_br_highn_g{int(G_MAX)}_n{N_RUNS_CAL}.npz"
# Per-frame CNR at which the header compares the lookup with its leading term.
LOOKUP_CNR = (1.3, 2.0, 3.0, 4.0, 5.0)

NAIVE_RED = "#c60b1e"
STRICT_GOLD = "#9E7900"
CORR_BLUE = "#0077c8"
CORR_STRICT_GREEN = "#4a7729"
REF_INDIGO = "#540d6e"
INCH = 1 / 2.54
FONT = 9


def fit_each_row(x, profiles):
    """Bounded Gaussian width fit per frame, with the tightened max_var bound."""
    out = np.full(profiles.shape[0], np.nan)
    for i in range(profiles.shape[0]):
        r = _fit_gaussian(x, profiles[i], max_var=MAX_VAR)
        if r.ok and np.isfinite(r.variance):
            out[i] = r.variance
    return out


def subsample(case, stride: int):
    """Return a frame-strided shallow copy of a BrCase (faithful, just fewer t)."""
    sl = slice(None, None, stride)
    return replace(case, t=case.t[sl], profiles=case.profiles[sl],
                   var_true=case.var_true[sl])


def build_or_load_calibration() -> BiasLookup:
    """Bounded Gaussian D-free calibration at the high-N geometry; cached after first build."""
    pixel_x = Experiment(D=0.0, **CAL_GEOMETRY).x
    if CAL_PATH.exists():
        trunc = BiasLookup.load(CAL_PATH)
        if (trunc.pixel_x is None or trunc.pixel_x.shape != pixel_x.shape
                or not np.allclose(trunc.pixel_x, pixel_x)):
            raise RuntimeError(f"{CAL_PATH.name} was built on a different pixel grid")
    else:
        trunc = calibrate_d_free(CAL_CNR0, n_runs=N_RUNS_CAL,
                                 model=GaussianModel(max_var=MAX_VAR), **CAL_GEOMETRY)
        trunc.save(CAL_PATH)
    if trunc.varbias_se_grid is None:
        raise RuntimeError("the D-free calibration carries no per-node SE")
    return trunc


def seed_fits(case, cnr0: float, n_seeds: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed bounded width fits of the noisy profiles, and the per-frame CNR."""
    noise_sigma = float(case.profiles[0].max()) / cnr0
    cnr_frame = case.profiles.max(axis=1) / noise_sigma
    raw = np.full((n_seeds, cnr_frame.size), np.nan)
    for s in range(n_seeds):
        rng = np.random.default_rng(int(cnr0) * 1000 + s)
        raw[s] = fit_each_row(case.x, case.profiles
                              + rng.normal(0.0, noise_sigma, case.profiles.shape))
    return raw, cnr_frame


def _d(t, traj, mask):
    return fit_msd_powerlaw(t, traj, mask=mask)[0]


def _band_residual(traj, nf, band):
    """Signed mean of ``traj - nf`` over the band's frames (NaN for an empty band)."""
    return float(np.nanmean((traj - nf)[band])) if band.any() else np.nan


def _corrected(raw, cnr_frame, lookup: BiasLookup):
    """Each row of fitted variances minus the lookup's bias, through the library."""
    return np.array([corrected_variance("subtract", row, cnr_frame, lookup)
                     for row in np.atleast_2d(raw)]).reshape(np.shape(raw))


@dataclass
class CellResult:
    """One (map, CNR_0) cell: frames, residuals, and D / D_nf per arm with SEs."""
    cnr0: float
    n_use: int
    t_last: float
    cnr_min: float
    growth_fitted: float
    n_failed: int
    n_fits: int
    resid_naive_hi: float      # signed mean residual vs noise-free fit, CNR >= 3
    resid_naive_hi_se: float
    resid_cor_hi: float
    resid_cor_hi_se: float
    resid_cor_hi_cal: float
    resid_naive_lo: float      # same, floor <= CNR < 3
    resid_naive_lo_se: float
    resid_cor_lo: float
    resid_cor_lo_se: float
    resid_cor_lo_cal: float
    n_strict: int
    ratio_naive: float
    se_naive: float
    ratio_strict: float
    se_strict: float
    ratio_cor: float
    se_cor_seed: float
    se_cor_cal: float
    ratio_cor_strict: float
    se_cor_strict_seed: float
    se_cor_strict_cal: float
    a_nf: float
    a_naive: float
    a_cor: float
    a_nf_strict: float
    a_naive_strict: float
    a_cor_strict: float
    d_nf: float


def score_cell(case, nf, trunc: BiasLookup, cnr0: float, rng, cal_rng) -> CellResult:
    """Score the four arms at one CNR_0 against the noise-free fit's D."""
    t = case.t
    raw, cnr_frame = seed_fits(case, cnr0, N_SEEDS)
    use = cnr_frame >= FLOOR_CNR
    strict = cnr_frame >= STRICT_CNR
    hi, lo = use & strict, use & ~strict
    cor = _corrected(raw, cnr_frame, trunc)
    d_nf, a_nf = fit_msd_powerlaw(t, nf, mask=use)
    d_nf_strict, a_nf_strict = fit_msd_powerlaw(t, nf, mask=strict)

    m_raw, m_cor = np.nanmean(raw, axis=0), np.nanmean(cor, axis=0)
    d_naive, a_naive = fit_msd_powerlaw(t, m_raw, mask=use)
    d_strict, a_naive_strict = fit_msd_powerlaw(t, m_raw, mask=strict)
    d_cor, a_cor = fit_msd_powerlaw(t, m_cor, mask=use)
    d_cor_strict, a_cor_strict = fit_msd_powerlaw(t, m_cor, mask=strict)

    def boot_row(idx):
        b_raw, b_cor = np.nanmean(raw[idx], axis=0), np.nanmean(cor[idx], axis=0)
        return [_d(t, b_raw, use), _d(t, b_raw, strict), _d(t, b_cor, use),
                _d(t, b_cor, strict),
                _band_residual(b_raw, nf, hi), _band_residual(b_cor, nf, hi),
                _band_residual(b_raw, nf, lo), _band_residual(b_cor, nf, lo)]

    boot = np.array([boot_row(rng.integers(0, N_SEEDS, N_SEEDS)) for _ in range(N_BOOT)])
    seed_se = np.nanstd(boot, axis=0, ddof=1)
    # The lookup's own Monte Carlo error: every grid node is an independent
    # calibration cell, so each moves by its own SE. Shared by every
    # measurement read with this lookup, so it does not shrink with the seeds.
    # Both corrected arms and both residual bands read each drawn lookup.
    se_grid = trunc.varbias_se_grid
    cal_draws = []
    for _ in range(N_CAL_DRAWS):
        drawn = replace(trunc, varbias_grid=trunc.varbias_grid
                        + se_grid * cal_rng.standard_normal(se_grid.shape))
        d_cor_drawn = _corrected(m_raw, cnr_frame, drawn)
        cal_draws.append((_d(t, d_cor_drawn, use), _d(t, d_cor_drawn, strict),
                          _band_residual(d_cor_drawn, nf, hi),
                          _band_residual(d_cor_drawn, nf, lo)))
    cal_se = np.nanstd(np.array(cal_draws), axis=0, ddof=1)

    last = int(np.flatnonzero(use).max())
    return CellResult(
        cnr0=cnr0, n_use=int(use.sum()), t_last=float(t[last]),
        cnr_min=float(cnr_frame[use].min()),
        growth_fitted=float(case.var_true[last] / case.var_true[0]),
        n_failed=int(np.isnan(raw[:, use]).sum()), n_fits=int(raw[:, use].size),
        resid_naive_hi=_band_residual(m_raw, nf, hi), resid_naive_hi_se=seed_se[4],
        resid_cor_hi=_band_residual(m_cor, nf, hi), resid_cor_hi_se=seed_se[5],
        resid_cor_hi_cal=float(cal_se[2]),
        resid_naive_lo=_band_residual(m_raw, nf, lo), resid_naive_lo_se=seed_se[6],
        resid_cor_lo=_band_residual(m_cor, nf, lo), resid_cor_lo_se=seed_se[7],
        resid_cor_lo_cal=float(cal_se[3]),
        n_strict=int(strict.sum()),
        ratio_naive=d_naive / d_nf, se_naive=seed_se[0] / d_nf,
        ratio_strict=d_strict / d_nf_strict, se_strict=seed_se[1] / d_nf_strict,
        ratio_cor=d_cor / d_nf, se_cor_seed=seed_se[2] / d_nf,
        se_cor_cal=float(cal_se[0]) / d_nf,
        ratio_cor_strict=d_cor_strict / d_nf_strict,
        se_cor_strict_seed=seed_se[3] / d_nf_strict,
        se_cor_strict_cal=float(cal_se[1]) / d_nf_strict,
        a_nf=a_nf, a_naive=a_naive, a_cor=a_cor, a_nf_strict=a_nf_strict,
        a_naive_strict=a_naive_strict, a_cor_strict=a_cor_strict, d_nf=d_nf)


@dataclass
class CaseResult:
    name: str
    a_ref: float
    d_ref: float
    growth_record: float
    cells: list[CellResult]


def sweep_case(case, trunc: BiasLookup) -> CaseResult:
    """Run the CNR_0 sweep for one Br fraction (the single source of the numbers)."""
    nf = fit_each_row(case.x, case.profiles)              # noise-free Gaussian fit
    d_ref, a_ref = fit_msd_powerlaw(case.t, case.var_true)
    rng, cal_rng = np.random.default_rng(57), np.random.default_rng(58)
    cells = [score_cell(case, nf, trunc, c0, rng, cal_rng) for c0 in CNR0_SWEEP]
    return CaseResult(name=case.name, a_ref=a_ref, d_ref=d_ref,
                      growth_record=float(case.var_true[-1] / case.var_true[0]),
                      cells=cells)


def _f(v, fmt):
    return "n/a" if not np.isfinite(v) else format(v, fmt)


def _at_bound(alpha: float) -> bool:
    """Whether a power-law exponent ended within BOUND_TOL of a fit bound (NaN is not)."""
    lo, hi = ALPHA_BOUNDS
    return bool(alpha <= lo + BOUND_TOL or alpha >= hi - BOUND_TOL)


def _off_bound(ratio: float, alpha: float) -> float:
    """The ratio, or NaN when its exponent sits at a fit bound (for the figure)."""
    return np.nan if _at_bound(alpha) else ratio


def _ratio_text(ratio, alpha, se_seed, se_cal=None) -> str:
    """D / D_nf with its SEs, or "at bound" when the exponent sets D."""
    if _at_bound(alpha):
        return "at bound"
    ses = _f(se_seed, ".3f")
    if se_cal is not None:
        ses += f"; calibration {_f(se_cal, '.3f')}"
    return f"{_f(ratio, '.3f')} ({ses})"


def _resid_text(value, se_seed, se_cal=None) -> str:
    """A band's mean residual in nm^2 with its SEs."""
    ses = _f(se_seed, ".0f")
    if se_cal is not None:
        ses += f"; calibration {_f(se_cal, '.0f')}"
    return f"{_f(value, '+.0f')} ({ses})"


def lookup_lines(trunc: BiasLookup) -> list[str]:
    """The lookup at sigma_0 against its leading term, at a few per-frame CNR."""
    x = trunc.pixel_x
    lead = leading_order_terms(GaussianModel(max_var=MAX_VAR), x, SIGMA0_HINT_NM ** 2)
    lines = [
        f"lookup at sigma_0 against its leading term, the width's Fisher variance "
        f"(2/sqrt(pi)) dx sigma_0 / CNR^2 on the calibration grid: {x.size} pixels "
        f"from {x[0]:g} to {x[-1]:g} nm, pitch {x[1] - x[0]:.3f} nm; width cap "
        f"sigma {G_MAX * SIGMA0_HINT_NM:.0f} nm, variance {MAX_VAR:.0f} nm^2; SE "
        f"propagated from the independent calibration nodes"]
    for c in LOOKUP_CNR:
        vb = float(trunc.var_bias(c))
        se = float(trunc.var_bias_se(c))
        term = lead.bias_coef / c ** 2
        lines.append(f"  per-frame CNR {c:g}: lookup {vb:.0f} nm^2 (SE {se:.0f}), "
                     f"leading term {term:.1f} nm^2, ratio {vb / term:.1f}")
    return lines


def write_report(results: list[CaseResult], n_frames: int, t_max: float,
                 trunc: BiasLookup) -> str:
    x = trunc.pixel_x
    lines = [
        "=" * 76,
        "High-N Br-I validation of the 1-D D-free lookup (calibrated at D = 0 and "
        "sigma_0, read by the per-frame CNR alone; subtract scheme)",
        f"width-fit bound: max_var = (G*sigma0)^2, G = {G_MAX:.0f} (sigma capped at "
        f"{G_MAX * SIGMA0_HINT_NM:.0f} nm), in the calibration and the application fit",
        f"calibration: calibrate_d_free, {len(CAL_CNR0)} CNR0 from {CAL_CNR0[0]:g} to "
        f"{CAL_CNR0[-1]:g} x {N_RUNS_CAL} runs, {CAL_GEOMETRY['n_frames']} frames, "
        f"sigma0 = {SIGMA0_HINT_NM:g} nm, {x.size} pixels over {x[-1] - x[0]:.0f} nm",
        f"evaluation: {N_SEEDS} seeds per CNR0 (seed 1000*CNR0 + s); frame stride "
        f"{STRIDE} ({n_frames} frames, 0 to {t_max:g} ns); per-frame CNR floor {FLOOR_CNR}; "
        f"CNR0 capped at the map's intrinsic peak CNR ~{INTRINSIC_PEAK_CNR:.0f}",
        "D from MSD = 2 D t^alpha fitted to the seed-mean trajectory over the frames "
        "at or above the floor (naive and corrected) or at or above per-frame CNR "
        f"{STRICT_CNR:g} (naive, CNR >= {STRICT_CNR:g}; corrected, CNR >= "
        f"{STRICT_CNR:g}), each divided by the noise-free fit's D over the same frames",
        f"SE of D and of the residuals: (seeds, from {N_BOOT} resamples; calibration, "
        f"from {N_CAL_DRAWS} draws of the lookup's own error); residuals in nm^2: "
        "the seed-mean fitted variance minus the variance fitted to the same frame's "
        "noise-free profile, signed mean over the band's frames",
        f"an arm whose exponent ends within {BOUND_TOL:g} of a fit bound "
        f"({ALPHA_BOUNDS[0]:.1f} or {ALPHA_BOUNDS[1]:.1f}) is printed 'at bound': "
        "the bound then sets its D",
        "carrier reference: particle second moment / 2, fitted over the full record",
        "failed width fits are dropped from the seed means (the calibration keeps them "
        "as the fallback width)",
        *lookup_lines(trunc),
        "=" * 76,
    ]
    for r in results:
        lines += ["", f"{r.name}: carrier reference alpha = {r.a_ref:.3f}, "
                      f"D_ref = {r.d_ref:.4g} nm^2/ns^alpha; carrier variance grows "
                      f"{r.growth_record:.1f}-fold over the {t_max:g} ns record"]
        for c in r.cells:
            cor_strict = _ratio_text(c.ratio_cor_strict, c.a_cor_strict,
                                     c.se_cor_strict_seed, c.se_cor_strict_cal)
            lines += [
                f"  CNR0 {c.cnr0:g}: {c.n_use} frames at or above the floor, t <= "
                f"{c.t_last:.2f} ns, CNR down to {c.cnr_min:.2f}; carrier variance grows "
                f"{c.growth_fitted:.1f}-fold over them; failed fits {c.n_failed} of {c.n_fits}",
                f"    residual, CNR >= {STRICT_CNR:g}: naive "
                f"{_resid_text(c.resid_naive_hi, c.resid_naive_hi_se)}, corrected "
                f"{_resid_text(c.resid_cor_hi, c.resid_cor_hi_se, c.resid_cor_hi_cal)}; "
                f"floor to {STRICT_CNR:g}: naive "
                f"{_resid_text(c.resid_naive_lo, c.resid_naive_lo_se)}, corrected "
                f"{_resid_text(c.resid_cor_lo, c.resid_cor_lo_se, c.resid_cor_lo_cal)}",
                f"    D / D_nf: naive {_ratio_text(c.ratio_naive, c.a_naive, c.se_naive)}; "
                f"naive, CNR >= {STRICT_CNR:g} ({c.n_strict} frames) "
                f"{_ratio_text(c.ratio_strict, c.a_naive_strict, c.se_strict)}; corrected "
                f"{_ratio_text(c.ratio_cor, c.a_cor, c.se_cor_seed, c.se_cor_cal)}; "
                f"corrected, CNR >= {STRICT_CNR:g} {cor_strict}",
                f"    alpha: noise-free fit {_f(c.a_nf, '.3f')}, naive {_f(c.a_naive, '.3f')}, "
                f"corrected {_f(c.a_cor, '.3f')}; D_nf = {_f(c.d_nf, '.4g')} nm^2/ns^alpha",
                f"    alpha, CNR >= {STRICT_CNR:g}: noise-free fit {_f(c.a_nf_strict, '.3f')}, "
                f"naive {_f(c.a_naive_strict, '.3f')}, corrected {_f(c.a_cor_strict, '.3f')}",
            ]
    return "\n".join(lines) + "\n"


def make_figure(results: list[CaseResult], path: Path) -> None:
    """D / D_nf against CNR_0, one panel per map (working figure, not committed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    n = len(results)
    fig, axes = plt.subplots(1, n, layout="constrained", sharey=True,
                             figsize=(6.0 * n * INCH, 7.0 * INCH), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1)
    for k, (ax, r) in enumerate(zip(axes[0], results)):
        c0 = np.array([c.cnr0 for c in r.cells])
        arms = (("naive", NAIVE_RED, "o",
                 [_off_bound(c.ratio_naive, c.a_naive) for c in r.cells],
                 [c.se_naive for c in r.cells]),
                (rf"naive, CNR $\geq$ {STRICT_CNR:g}", STRICT_GOLD, "^",
                 [_off_bound(c.ratio_strict, c.a_naive_strict) for c in r.cells],
                 [c.se_strict for c in r.cells]),
                ("corrected", CORR_BLUE, "s",
                 [_off_bound(c.ratio_cor, c.a_cor) for c in r.cells],
                 [np.hypot(c.se_cor_seed, c.se_cor_cal) for c in r.cells]),
                (rf"corrected, CNR $\geq$ {STRICT_CNR:g}", CORR_STRICT_GREEN, "D",
                 [_off_bound(c.ratio_cor_strict, c.a_cor_strict) for c in r.cells],
                 [np.hypot(c.se_cor_strict_seed, c.se_cor_strict_cal) for c in r.cells]))
        for label, color, marker, y, se in arms:
            ax.errorbar(c0, y, yerr=se, color=color, marker=marker, ms=4, lw=2,
                        capsize=2, label=label)
        ax.axhline(1.0, color=REF_INDIGO, lw=1.2, ls="--", label="noise-free fit")
        ax.set_xscale("log")
        ax.set_xticks(c0)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$\mathrm{CNR}_0$", fontsize=FONT)
        ax.set_facecolor("white")
        ax.tick_params(axis="both", which="both", direction="in", top=False,
                       right=False, labelsize=FONT)
        ax.text(0.96, 0.96, f"({'abc'[k]}) {r.name}", ha="right", va="top",
                fontsize=FONT, fontweight="bold", transform=ax.transAxes)
    axes[0][0].set_ylabel(r"$D / D_\mathrm{nf}$", fontsize=FONT)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=FONT,
               loc="outside lower center", ncol=3)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure: {path}")


def main() -> None:
    cases = load_all()
    if not cases:
        print(f"No high-N collapsed maps found in {MAP_DIR} (WIDTH_DEBIAS_DATA_ROOT, or else data/). See DATASETS.md; "
              "nothing to validate.")
        return
    cases = [subsample(c, STRIDE) for c in cases]
    n_frames = cases[0].profiles.shape[0]
    print(f"Loaded {len(cases)} fraction(s): {[c.name for c in cases]};"
          f" {n_frames} frames after stride {STRIDE}.")
    print(f"Building/loading the bounded high-N D-free calibration "
          f"({len(CAL_CNR0)} CNR0 x {N_RUNS_CAL} runs) ...", flush=True)
    trunc = build_or_load_calibration()
    results = [sweep_case(case, trunc) for case in cases]
    report = write_report(results, n_frames, float(cases[0].t[-1]), trunc)
    print(report)
    OUT_TXT.write_text(report)
    make_figure(results, OUT_PNG)


if __name__ == "__main__":
    main()
