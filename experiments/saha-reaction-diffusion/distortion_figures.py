"""Diagnostic figures for the Saha bed: the two distortions, made concrete.

These are illustration/diagnostic figures, NOT poster figures. The Saha bed did
not earn a width-debias poster panel (see README.md): the noise bias is
modest in the trusted band, and the emission-to-carrier framing is out of
width-debias's scope and not novel. The figures are kept because they make the
validation framework (docs/VALIDATION_DESIGN.md) concrete.

Two figures:

* mechanism -- for one strong non-Fickian regime at one moderate CNR_0, the
  per-frame variance sigma^2(t): the noise-free reference, the uncorrected mean
  (which lifts above the reference at the low-CNR late frames -- the truncation
  bias), and the folded-normal-corrected mean. Shows what the noise correction
  does, and that on this bed the divergence sits mostly in the excluded
  below-floor region (the reason this is not a poster panel).
* distortion-alphas -- across the density-coupling sweep, three anomalous
  exponents: the carrier (truth), the recorded-emission moment, and the naive
  Gaussian fit of the emission. The vertical separation is the over-statement of
  alpha by a naive analysis (~0.1), split into the emission-to-carrier gap (the
  forward model must invert it) and the within-fit shape gap.

Both are noise-free-referenced; neither claims carrier-alpha recovery on the
Saha bed. Self-contained: depends only on the model port and the shipped
width_debias package, not on the (actively reworked) validation harness.

Style: the plot style guide, Okabe-Ito palette (blue = truth/
reference, gold = emission/intermediate, red = biased/uncorrected, green =
noise-bias correction).

Run from the repository root (the mechanism figure needs output/calibration_saha.npz)::

    .venv/Scripts/python.exe mc_experiments/saha_reaction_diffusion/distortion_figures.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, _HERE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import reaction_diffusion_saha as rd  # noqa: E402
from width_debias.calibration import CalibrationGrid  # noqa: E402
from width_debias.models import GaussianModel  # noqa: E402
from width_debias.bias_lookup import BiasLookup  # noqa: E402
from width_debias.weights import corrected_variance  # noqa: E402

# Okabe-Ito roles (see plot_style.md; match scripts/poster_figures.py).
BLUE = "#0072B2"    # truth / noise-free reference
GOLD = "#E69F00"    # recorded-emission moment (intermediate)
GREEN = "#009E73"   # folded-normal corrected
RED = "#D55E00"     # uncorrected / naive-fit (biased)
INCH = 1.0 / 2.54

PIXEL_X = np.linspace(-4.0, 4.0, 401)
N_FRAMES = 12
CNR_FLOOR = 1.3
CALIBRATION = os.path.join("output", "calibration_saha.npz")
BASE = dict(L=24.0, nbins=960, De=0.005, Df=0.1, nu=0.25, rho=1e-3,
            tmax=3.0, dt=0.001, sigma0_init=0.35)

A_REGIME = 1314.0       # strong non-Fickian regime for the mechanism figure
A_SWEEP = np.geomspace(30.0, 2000.0, 11)   # density-coupling sweep (alphas figure)
EMISSION_POWER = 2      # x^2 emission (paper eq. 9)
MECH_CNR0 = 18.0
N_RUNS = 400
FONT = 9
GAUSS = GaussianModel()


def _alpha(times: np.ndarray, var: np.ndarray) -> float:
    msd = var - var[0]
    keep = (times > 0) & (msd > 0)
    if keep.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(times[keep]), np.log(msd[keep]), 1)[0])


def _moment(x: np.ndarray, profile: np.ndarray) -> float:
    """Second central moment of a non-negative profile on grid x."""
    w = np.clip(profile, 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.nan
    mean = float((x * w).sum() / s)
    return float(((x - mean) ** 2 * w).sum() / s)


def alphas_for(A: float):
    """(alpha_carrier, alpha_emission, alpha_fit) for one regime, noise-free.

    Carrier and emission moments use the native (wide) simulation grid so the
    tails are captured; the Gaussian fit uses the same grid.
    """
    par = rd.Parameters(L=BASE["L"], nbins=BASE["nbins"], De=BASE["De"],
                        Df=BASE["Df"], nu=BASE["nu"], rho=BASE["rho"], A=A,
                        tsamp=BASE["tmax"] / 40.0, dt=BASE["dt"],
                        tmax=BASE["tmax"], free_charge_emission_power=EMISSION_POWER)
    xs = rd.x_axis(par)
    res = rd.simulate(par, np.exp(-xs ** 2 / (2.0 * BASE["sigma0_init"] ** 2)))
    idx = np.linspace(0, len(res.times) - 1, N_FRAMES).astype(int)
    times = res.times[idx]
    carrier = np.array([_moment(xs, res.population[i]) for i in idx])
    emission = np.array([_moment(xs, res.emission[i]) for i in idx])
    fit = np.array([GAUSS.fit_variance(xs, res.emission[i]).variance for i in idx])
    return _alpha(times, carrier), _alpha(times, emission), _alpha(times, fit)


def _new_ax(width_cm: float, height_cm: float):
    fig, ax = plt.subplots(layout="constrained",
                           figsize=(width_cm * INCH, height_cm * INCH))
    ax.tick_params(axis="both", which="both", direction="in", top=True,
                   right=True, labelsize=FONT, length=4, width=1)
    fig.patch.set_facecolor("w")
    fig.patch.set_alpha(1)
    return fig, ax


def _emission_frames(A: float):
    par = rd.Parameters(L=BASE["L"], nbins=BASE["nbins"], De=BASE["De"],
                        Df=BASE["Df"], nu=BASE["nu"], rho=BASE["rho"], A=A,
                        tsamp=BASE["tmax"] / 40.0, dt=BASE["dt"],
                        tmax=BASE["tmax"], free_charge_emission_power=EMISSION_POWER)
    xs = rd.x_axis(par)
    res = rd.simulate(par, np.exp(-xs ** 2 / (2.0 * BASE["sigma0_init"] ** 2)))
    idx = np.linspace(0, len(res.times) - 1, N_FRAMES).astype(int)
    times = res.times[idx]
    clean = np.array([np.interp(PIXEL_X, xs, res.emission[i], left=0.0, right=0.0)
                      for i in idx])
    return times, clean, clean.max(axis=1)


def fig_mechanism() -> None:
    """Variance trajectory: raw lift vs folded recovery at one CNR_0."""
    if not os.path.exists(CALIBRATION):
        print(f"[skip mechanism: build {CALIBRATION} first]")
        return
    model = BiasLookup.from_calibration(CalibrationGrid.load(CALIBRATION))
    times, clean, peak = _emission_frames(A_REGIME)
    ref = np.array([GAUSS.fit_variance(PIXEL_X, r).variance for r in clean])

    noise_sigma = peak[0] / MECH_CNR0
    cnr_frame = peak / noise_sigma
    rng = np.random.default_rng(7)
    raws = np.full((N_RUNS, N_FRAMES), np.nan)
    cors = np.full((N_RUNS, N_FRAMES), np.nan)
    for r in range(N_RUNS):
        noisy = clean + rng.normal(0.0, noise_sigma, size=clean.shape)
        var_row = np.array([GAUSS.fit_variance(PIXEL_X, noisy[f]).variance
                            for f in range(N_FRAMES)])
        raws[r] = var_row
        cors[r] = corrected_variance("subtract", var_row, cnr_frame, model)
    mean_raw = np.nanmean(raws, axis=0)
    mean_cor = np.nanmean(cors, axis=0)

    fig, ax = _new_ax(8.5, 7.0)
    below = cnr_frame < CNR_FLOOR
    if below.any():
        ax.axvspan(times[below][0], times[-1], color="0.85", alpha=0.5, lw=0)
    ax.plot(times, ref, "-", lw=2, color=BLUE, label="noise-free reference")
    ax.plot(times, mean_raw, "o-", lw=2, color=RED, label="uncorrected")
    ax.plot(times, mean_cor, "s-", lw=2, color=GREEN, label="folded-normal")
    ax.set_xlabel("time (ns)", fontsize=FONT)
    ax.set_ylabel(r"variance  $\sigma^2$  (μm$^2$)", fontsize=FONT)
    ax.set_ylim(0.0, float(np.nanmax(ref) * 2.2))
    ax.legend(frameon=False, fontsize=FONT, loc="upper left")
    out = os.path.join("output", "saha_mechanism_trajectory.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"mechanism: CNR_0={MECH_CNR0}, A={A_REGIME:.0f}, "
          f"{int(below.sum())} frames below floor -> {out}")


def fig_distortion_alphas() -> None:
    """Carrier / emission / fit alpha across the sweep (the two gaps)."""
    carr, emis, fit = [], [], []
    for A in A_SWEEP:
        ac, ae, af = alphas_for(float(A))
        carr.append(ac)
        emis.append(ae)
        fit.append(af)
    carr = np.array(carr)
    emis = np.array(emis)
    fit = np.array(fit)

    fig, ax = _new_ax(8.5, 7.0)
    ax.plot(A_SWEEP, carr, "o-", lw=2, color=BLUE, label="carrier (truth)")
    ax.plot(A_SWEEP, emis, "^-", lw=2, color=GOLD, label="emission moment")
    ax.plot(A_SWEEP, fit, "s-", lw=2, color=RED, label="naive Gaussian fit")
    ax.set_xscale("log")
    ax.set_xlabel("Saha density parameter  $A$", fontsize=FONT)
    ax.set_ylabel(r"anomalous exponent  $\alpha$", fontsize=FONT)
    ax.legend(frameon=False, fontsize=FONT, loc="best")
    out = os.path.join("output", "saha_distortion_alphas.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    gap = float(np.nanmax(fit - carr))
    print(f"distortion-alphas: max naive-fit over-statement of carrier alpha "
          f"= {gap:.3f} -> {out}")


def main() -> None:
    fig_mechanism()
    fig_distortion_alphas()


if __name__ == "__main__":
    main()
