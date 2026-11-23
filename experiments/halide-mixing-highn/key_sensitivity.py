"""How much does the joint-lookup key matter, and what does a wrong key cost?

The joint (CNR, sigma^2) calibration returns the bias at a frame's true width,
which a measurement does not know. The key is the stand-in used for it. This
script compares keys on the high-N Br-I bed, where sigma^2 grows 11- to 24-fold
over the usable band at CNR0 15 and 20, so the key matters:

* ``perframe``   -- each frame's own sigma_hat^2, fixed-point inversion
                    (the library default);
* ``exact``      -- the true sigma^2 (the reference a measurement cannot have);
* ``smoothed``   -- a robust (Theil-Sen) line through the measurement's raw
                    sigma_hat^2 above the floor, fixed-point inversion on the
                    line's value;
* ``iterated``   -- the same line refitted to the corrected sigma_hat^2 until
                    it stops changing, then a direct lookup at the line;
* ``exact x (1 + e)`` -- the true sigma^2 scaled by e = -0.2 ... +0.2, which
                    measures how D and the residual respond to a key error.

The fits are computed once per (map, CNR0) and shared by every key, so the
keys differ only in the lookup. Residuals are against the noise-free fit of
the same profile (the noise correction's own target; the shape gap against the
carrier reference is reported separately in the bed's README). D is the
short-window power-law fit of the mean trajectory over frames above the floor,
as in ``br_halide_highn_perframe.py``, with an SE from resampling the seeds.

The key-error term: the sensitivity dD/de from the perturbed exact keys,
times the spread of the smoothed key's own error (key / sigma^2_true - 1) over
measurements, is the extra uncertainty the smoothed key carries.

Run from the repository root (needs the pshare and the dense calibration)::

    WIDTH_DEBIAS_WORKERS=7 .venv/Scripts/python.exe experiments/halide-mixing-highn/key_sensitivity.py
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "halide-mixing-brI-legacy"))
sys.path.insert(0, os.path.dirname(__file__))

N_SEEDS = int(os.environ.get("KEYSENS_SEEDS", 300))
CNR0_LIST = (8.0, 10.0, 15.0, 20.0, 30.0)
EPS = (-0.2, -0.1, 0.1, 0.2)
N_BOOT = 100
N_ITER = 8
OUTPUT = Path(__file__).resolve().parents[2] / "output"


def _theil_line(t, v, use):
    from scipy.stats import theilslopes
    ok = use & np.isfinite(v)
    if ok.sum() < 3:
        return v.copy()
    slope, icpt, _, _ = theilslopes(v[ok], t[ok])
    return np.where(use, icpt + slope * t, v)


def _raw_fits(args):
    """Noisy fits for one (map, CNR0): the expensive part, done once."""
    frac, cnr0 = args
    import br_halide_highn_perframe as hp
    from br_halide_highn_validation import STRIDE, subsample
    from br_highn_data import load_all
    case = [subsample(c, STRIDE) for c in load_all() if c.name.endswith(frac)][0]
    ns = float(case.profiles[0].max()) / cnr0
    cnrf = case.profiles.max(axis=1) / ns
    raw = np.full((N_SEEDS, cnrf.size), np.nan)
    for s in range(N_SEEDS):
        rng = np.random.default_rng(7919 * int(cnr0) + s)
        raw[s] = hp.fit_each_row(case.x, case.profiles + rng.normal(0.0, ns, case.profiles.shape))
    nf = hp.fit_each_row(case.x, case.profiles)
    return frac, cnr0, case.t, case.var_true, nf, cnrf, raw


def _keys(trunc, t, var_true, cnrf, raw, floor):
    """Corrected variance per seed for every key; plus the smoothed key's error."""
    from width_debias.bias_lookup import _bilinear
    use = cnrf >= floor

    def direct(key):
        k = np.clip(key, trunc.variance_grid[0], trunc.variance_grid[-1])
        return _bilinear(trunc.cnr_grid, trunc.variance_grid, trunc.varbias_2d, cnrf, k)

    out = {"perframe": [], "exact": [], "smoothed": [], "iterated": []}
    out.update({f"exact x (1{e:+.1f})": [] for e in EPS})
    key_err = []
    for r in raw:
        out["perframe"].append(r - trunc.var_bias(cnrf, r))
        out["exact"].append(r - direct(var_true))
        line = _theil_line(t, r, use)
        out["smoothed"].append(r - trunc.var_bias(cnrf, line))
        key = line
        for _ in range(N_ITER):
            new = _theil_line(t, r - direct(key), use)
            if np.allclose(new[use], key[use], rtol=1e-4):
                key = new
                break
            key = new
        out["iterated"].append(r - direct(key))
        key_err.append(np.where(use, key / var_true - 1.0, np.nan))
        for e in EPS:
            out[f"exact x (1{e:+.1f})"].append(r - direct(var_true * (1 + e)))
    return {k: np.array(v) for k, v in out.items()}, np.array(key_err)


def _d_ratio(t, mean_traj, use, d_ref):
    from br_halide_validation import fit_msd_powerlaw
    d, _ = fit_msd_powerlaw(t, mean_traj, mask=use)
    return d / d_ref


def main() -> None:
    import br_halide_highn_perframe as hp
    from br_halide_validation import fit_msd_powerlaw
    from br_halide_highn_validation import STRIDE, subsample
    from br_highn_data import load_all

    if hp.N_RUNS_CAL2 != 4000:
        sys.exit("expects the dense calibration (HIGHN_CAL2_RUNS unset or 4000)")
    case0 = subsample(load_all()[0], STRIDE)
    trunc = hp.cnr_sigma2_calibration(case0.x)

    jobs = [(f, c) for f in ("0.25", "0.5", "0.75") for c in CNR0_LIST]
    workers = int(os.environ.get("WIDTH_DEBIAS_WORKERS") or 1)
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(_raw_fits, jobs))

    rng = np.random.default_rng(31)
    lines = ["=" * 78,
             "Joint-lookup key sensitivity -- high-N Br-I, dense joint calibration",
             f"{N_SEEDS} seeds per (map, CNR0); floor CNR {hp.FLOOR_CNR}; residuals vs "
             "noise-free fit (nm^2); D = short-window D / D_ref (SE from "
             f"{N_BOOT} resamples)",
             "=" * 78]
    sens_rows = []
    for frac, cnr0, t, var_true, nf, cnrf, raw in results:
        use = cnrf >= hp.FLOOR_CNR
        d_ref, _ = fit_msd_powerlaw(t, var_true)
        cor, key_err = _keys(trunc, t, var_true, cnrf, raw, hp.FLOOR_CNR)
        lines.append(f"\n--- x = {frac}, CNR0 = {cnr0:g}: {use.sum()} usable frames, "
                     f"CNR down to {cnrf[use].min():.2f}")
        lines.append(f"  {'key':<18}{'resid CNR>=2':>14}{'resid <2':>11}{'|resid|':>9}"
                     f"{'D/D_ref':>10}{'SE':>7}")
        naive = np.nanmean(raw, axis=0) - nf
        lo, hi = use & (cnrf < 2), use & (cnrf >= 2)
        dn = _d_ratio(t, np.nanmean(raw, axis=0), use, d_ref)
        lines.append(f"  {'naive':<18}{np.nanmean(naive[hi]):>+14.0f}"
                     f"{(np.nanmean(naive[lo]) if lo.any() else np.nan):>+11.0f}"
                     f"{np.nanmean(np.abs(naive[use])):>9.0f}{dn:>10.3f}")
        dvals = {}
        for name, c in cor.items():
            resid = np.nanmean(c, axis=0) - nf
            d = _d_ratio(t, np.nanmean(c, axis=0), use, d_ref)
            boot = [_d_ratio(t, np.nanmean(c[rng.integers(0, len(c), len(c))], axis=0),
                             use, d_ref) for _ in range(N_BOOT)]
            se = float(np.nanstd(boot, ddof=1))
            dvals[name] = d
            lines.append(f"  {name:<18}{np.nanmean(resid[hi]):>+14.0f}"
                         f"{(np.nanmean(resid[lo]) if lo.any() else np.nan):>+11.0f}"
                         f"{np.nanmean(np.abs(resid[use])):>9.0f}{d:>10.3f}{se:>7.3f}")
        # Sensitivity of D to a relative key error, from the perturbed exact keys.
        es = np.array([0.0] + list(EPS))
        ds = np.array([dvals["exact"]] + [dvals[f"exact x (1{e:+.1f})"] for e in EPS])
        slope = float(np.polyfit(es, ds, 1)[0])
        ke = key_err[:, use]
        bias_k, sd_k = float(np.nanmean(ke)), float(np.nanstd(ke))
        # The systematic part of the key error shifts the mean-trajectory D;
        # its scatter over measurements adds to a single measurement's D.
        lines.append(f"  dD/de = {slope:+.3f} (D/D_ref per unit relative key error); "
                     f"iterated key error over usable frames: mean {bias_k:+.3f}, "
                     f"SD {sd_k:.3f}; implied key terms in D/D_ref: systematic "
                     f"{abs(slope * bias_k):.3f}, per measurement {abs(slope) * sd_k:.3f}")
        sens_rows.append((frac, cnr0, slope, bias_k, sd_k))

    lines.append("\nSummary: dD/de and the iterated key's error, per (map, CNR0)")
    lines.append(f"  {'x':>5} {'CNR0':>5} {'dD/de':>8} {'key err mean':>13} {'key err SD':>11}")
    for frac, cnr0, slope, b, sd in sens_rows:
        lines.append(f"  {frac:>5} {cnr0:>5g} {slope:>+8.3f} {b:>+13.3f} {sd:>11.3f}")
    report = "\n".join(lines)
    print(report)
    (OUTPUT / "key_sensitivity.txt").write_text(report)


if __name__ == "__main__":
    main()
