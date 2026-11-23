"""Render the per-frame figure for the high-N Br-I bed, with its results table.

Three panels, one per Br fraction (x = 0.25, 0.5, 0.75), at initial CNR 15,
the lowest initial CNR at which the corrected D does not depend on the
joint-lookup key (``key_sensitivity.py``). Each panel shows, per frame, the mean
fitted variance over the seeds minus the Gaussian fit of the noise-free profile,
for the naive fit and for the corrected fit, with standard errors. That residual
isolates the noise bias, which is what the correction removes. The separate
offset between the noise-free fit and the carrier reference (the shape gap,
which no noise correction addresses) is drawn as its own line.

The correction uses the dense joint (CNR, sigma^2) calibration and the
iterated measurement-level key (``iterated_key_correction``). The table
``br_halide_highn_perframe.txt`` gives the short-window D against the carrier
reference and against the noise-free fit, with standard errors from resampling
the seeds. It also scores two more corrected arms, the Fisher-anchored 1-D
calibration with a width table (``fisher_calibration``, same key), which
rescales one calibration to each frame's width instead of measuring every
width: the pre-specified arm calibrates at sigma_0, and a post hoc arm at the
width of the frames the correction acts on (``FISHER_LATE_SIGMA_NM``), added
after the sigma_0 arm over-subtracted. Each corrected D carries a calibration
SE as well: the spread of D over draws of its lookup's own Monte Carlo error,
which is shared by every measurement and so does not shrink with the seeds.

    .venv/Scripts/python.exe experiments/halide-mixing-highn/render_perframe_figure.py

Needs the pshare (collapsed maps) and the dense calibration in ``output/``
(``output/build_highn_dense_cal.py`` or ``br_halide_highn_perframe.cnr_sigma2_calibration``).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "halide-mixing-brI-legacy"))

CNR0_FIG = 15.0
N_SEEDS_FIG = int(os.environ.get("HIGHN_FIG_SEEDS", 800))
N_BOOT = 200
N_CAL_DRAWS = 200
OUT_PNG = Path(HERE) / "br_halide_highn_perframe.png"
OUT_TXT = Path(HERE) / "br_halide_highn_perframe.txt"
INCH = 1 / 2.54


def _arms():
    """Labels of the corrected arms, in the order of ``_truncs``."""
    import br_halide_highn_perframe as hp
    return ("joint grid", "Fisher, sigma_0 cal",
            f"Fisher, {hp.FISHER_LATE_SIGMA_NM:g} nm cal (post hoc)")


def _truncs(pixel_x):
    """The corrected arms' lookups: joint grid (drawn in the figure), Fisher at
    sigma_0, Fisher at the late-band width."""
    import br_halide_highn_perframe as hp
    return (hp.cnr_sigma2_calibration(pixel_x), hp.fisher_calibration(pixel_x),
            hp.fisher_calibration(pixel_x, sigma_cal_nm=hp.FISHER_LATE_SIGMA_NM))


def _panel(frac):
    """Per-seed naive and corrected variances, and the final keys, for one map at CNR0_FIG.

    The corrected arms share the fits; ``cor``/``key`` hold, per arm (``_arms``),
    the corrected variances and the iterated key.
    """
    import br_halide_highn_perframe as hp
    from br_halide_highn_validation import STRIDE, subsample
    from br_highn_data import load_all
    case = [subsample(c, STRIDE) for c in load_all() if c.name.endswith(frac)][0]
    truncs = _truncs(case.x)
    ns = float(case.profiles[0].max()) / CNR0_FIG
    cnrf = case.profiles.max(axis=1) / ns
    raw = np.full((N_SEEDS_FIG, cnrf.size), np.nan)
    cor = np.full((len(truncs),) + raw.shape, np.nan)
    key = np.full_like(cor, np.nan)
    for s in range(N_SEEDS_FIG):
        rng = np.random.default_rng(7919 * int(CNR0_FIG) + s)
        vh = hp.fit_each_row(case.x, case.profiles + rng.normal(0.0, ns, case.profiles.shape))
        raw[s] = vh
        for a, trunc in enumerate(truncs):
            cor[a, s], key[a, s] = hp.iterated_key_correction(case.t, vh, cnrf, trunc,
                                                              return_key=True)
    nf = hp.fit_each_row(case.x, case.profiles)
    return frac, case.t, case.var_true, nf, cnrf, raw, cor, key


def _d(t, traj, use):
    from br_halide_validation import fit_msd_powerlaw
    d, _ = fit_msd_powerlaw(t, traj, mask=use)
    return d


def _perturbed(trunc, rng):
    """One draw of a lookup from its own Monte Carlo error.

    Joint grid: every node is an independent Monte Carlo cell, so each moves by
    its own SE. Width table: the smoothed sigma_0 curve moves along its
    covariance factor, which carries the correlation between grid points.
    """
    if trunc.is_2d:
        se = np.nan_to_num(trunc.varbias_se_2d)
        return replace(trunc, varbias_2d=trunc.varbias_2d + se * rng.standard_normal(se.shape))
    factor = trunc.varbias_cov_factor
    return replace(trunc, varbias_grid=trunc.varbias_grid
                   + factor @ rng.standard_normal(factor.shape[1]))


def _calibration_se(trunc, t, cnrf, raw, keys, use, rng):
    """SD of the corrected short-window D over draws of the lookup's own error.

    Each measurement's final key is held fixed: a lookup error moves the key by
    about the same amount, which changes the bias only through its slope in
    sigma^2, a second-order effect. The bias is read directly at the key, as
    ``iterated_key_correction`` does.
    """
    import br_halide_highn_perframe as hp
    cnr = np.broadcast_to(cnrf, keys.shape).ravel()
    draws = []
    for _ in range(N_CAL_DRAWS):
        bias = hp.direct_bias(_perturbed(trunc, rng), cnr, keys.ravel()).reshape(keys.shape)
        draws.append(_d(t, np.nanmean(raw - bias, axis=0), use))
    return float(np.nanstd(draws, ddof=1))


def main():
    import br_halide_highn_perframe as hp
    from br_halide_validation import fit_msd_powerlaw

    from br_halide_highn_validation import STRIDE, subsample
    from br_highn_data import load_all

    if hp.N_RUNS_CAL2 != 4000:
        sys.exit("expects the dense calibration (HIGHN_CAL2_RUNS unset or 4000)")
    # Every map shares the pixel grid; build (or load) the Fisher calibrations
    # once here, before the workers would each try to build them.
    pixel_x = subsample(load_all()[0], STRIDE).x
    truncs = _truncs(pixel_x)
    with ProcessPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(_panel, ["0.25", "0.5", "0.75"]))

    rng = np.random.default_rng(57)
    cal_rng = np.random.default_rng(58)
    arms = _arms()
    lines = ["=" * 76,
             f"High-N Br-I per-frame figure: CNR0 = {CNR0_FIG:g}, {N_SEEDS_FIG} seeds, "
             "iterated key; corrected arms: dense joint calibration (the figure) and the "
             f"Fisher-anchored calibration with a width table ({hp.N_RUNS_FISHER} runs/cell), "
             f"calibrated at sigma_0 (pre-specified) and at {hp.FISHER_LATE_SIGMA_NM:g} nm "
             "(post hoc, the median width of the frames at per-frame CNR below 3)",
             f"floor CNR {hp.FLOOR_CNR}; residuals in nm^2 against the noise-free fit; "
             f"D SE: (seeds, from {N_BOOT} resamples; calibration, from {N_CAL_DRAWS} draws "
             "of the lookup's own error)",
             "=" * 76]
    panels = []
    for frac, t, var_true, nf, cnrf, raw, cor, key in results:
        use = cnrf >= hp.FLOOR_CNR
        d_ref, _ = fit_msd_powerlaw(t, var_true)
        d_nf = _d(t, nf, use)
        mean_raw, mean_cor = np.nanmean(raw, axis=0), np.nanmean(cor, axis=1)
        se_raw = np.nanstd(raw, axis=0, ddof=1) / np.sqrt(len(raw))
        se_cor = np.nanstd(cor, axis=1, ddof=1) / np.sqrt(raw.shape[0])
        d_raw = _d(t, mean_raw, use)
        d_cor = [_d(t, m, use) for m in mean_cor]
        boot = np.array([[_d(t, np.nanmean(a[idx], axis=0), use) for a in (raw, *cor)]
                         for idx in (rng.integers(0, len(raw), len(raw))
                                     for _ in range(N_BOOT))])
        seed_se = np.nanstd(boot, axis=0, ddof=1)
        cal_se = [_calibration_se(tr, t, cnrf, raw, k, use, cal_rng)
                  for tr, k in zip(truncs, key)]
        rn = (mean_raw - nf)[use]
        lines += [
            f"\nx = {frac}: {use.sum()} usable frames, t <= {t[use].max():.2f} ns, "
            f"CNR down to {cnrf[use].min():.2f}",
            f"  residual vs noise-free fit, mean |.| and signed (nm^2): naive "
            f"{np.nanmean(np.abs(rn)):.0f}, {np.nanmean(rn):+.0f}",
        ]
        for name, m in zip(arms, mean_cor):
            rc = (m - nf)[use]
            lines.append(f"    {name:<30} {np.nanmean(np.abs(rc)):.0f}, {np.nanmean(rc):+.0f}")
        lines.append(f"  carrier gap (noise-free fit - reference) over the band: mean "
                     f"{np.nanmean((nf - var_true)[use]):+.0f} nm^2")
        for ref_name, d0 in (("D_ref", d_ref), ("D_noisefree", d_nf)):
            lines.append(f"  short-window D / {ref_name}: naive {d_raw / d0:.3f} "
                         f"({seed_se[0] / d0:.3f})")
            for a, name in enumerate(arms):
                lines.append(f"    {name:<30} {d_cor[a] / d0:.3f} ({seed_se[a + 1] / d0:.3f}; "
                             f"calibration {cal_se[a] / d0:.3f})")
        lines.append(f"  D_ref = {d_ref:.4g} nm^2/ns; D of the noise-free fit = {d_nf:.4g} nm^2/ns")
        panels.append((frac, t, cnrf, use, mean_raw - nf, se_raw, mean_cor[0] - nf,
                       se_cor[0], nf - var_true))
    report = "\n".join(lines)
    print(report)
    OUT_TXT.write_text(report + "\n")
    make_residual_figure(panels, OUT_PNG)


def make_residual_figure(panels, path):
    """Per-frame residual against the noise-free fit, naive and corrected.

    Spanish-flag set: flag red = naive (the bias), blue = corrected, indigo =
    the carrier gap, flag yellow = the zero line and the sub-floor band.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    red, blue, indigo, yellow = "#c60b1e", "#0077c8", "#540d6e", "#ffc400"
    font = 9
    xmax = max(p[1][p[3]].max() for p in panels) + 0.1
    # Scale y to the usable frames: the sub-floor frames are excluded from the
    # fit, are not plotted, and are marked only by the shaded band.
    ylim = 1.15 * max(np.nanmax(np.abs(np.concatenate([
        (p[4] + np.sign(p[4]) * p[5])[p[3]], (p[6] + np.sign(p[6]) * p[7])[p[3]],
        p[8][p[3]]]))) for p in panels)
    fig, axes = plt.subplots(1, len(panels), layout="constrained", sharex=True,
                             sharey=True, figsize=(17 * INCH, 6.5 * INCH))
    for i, (ax, (frac, t, cnrf, use, rn, sen, rc, sec, gap)) in enumerate(zip(axes, panels)):
        m = use & (t <= xmax)
        t_floor = t[~use].min() if (~use).any() else xmax
        if t_floor < xmax:
            ax.axvspan(t_floor, xmax, color=yellow, alpha=0.18, lw=0)
        ax.axhline(0.0, color=yellow, lw=1.2, zorder=1)
        ax.plot(t[m], gap[m], "-", color=indigo, lw=2, label="carrier gap", zorder=2)
        ax.errorbar(t[m], rn[m], yerr=sen[m], fmt="o", color=red, ms=3, lw=0.8,
                    label="naive", zorder=3)
        ax.errorbar(t[m], rc[m], yerr=sec[m], fmt="s", color=blue, ms=3, lw=0.8,
                    label="corrected", zorder=4)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-ylim, ylim)
        ax.set_facecolor("white")
        ax.tick_params(axis="both", which="both", direction="in", top=False,
                       right=False, labelsize=font)
        ax.text(0.04, 0.95, f"({'abc'[i]})", ha="left", va="top", fontsize=font,
                fontweight="bold", transform=ax.transAxes)
        ax.set_xlabel(r"$t \, (\text{ns})$", fontsize=font)
    axes[0].set_ylabel(r"$\Delta\sigma^2 \, (\text{nm}^2)$", fontsize=font)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=font, loc="outside lower center", ncol=3)
    fig.patch.set_alpha(0)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure: {path.name}")


if __name__ == "__main__":
    main()
