"""Where the noise-bias correction acts on a nonlinear, multi-population bed.

This experiment places the Saha reaction-diffusion bed inside the validation
framework of `docs/VALIDATION_DESIGN.md`. The deliverable of an exciton-diffusion
measurement is the *carrier distribution* and its transport, not the recorded
emission profile. Three distortions sit between the two:

1. measurement noise (and the noise bias it puts in any fitted width),
2. the instrument point-spread function,
3. the emission-to-carrier mapping.

The noise-bias correction in `width_debias` removes distortion 1 only. The
Saha emission is nonlinear in carrier density -- E = qe nu (1-x) n + qf rho x^2
n^2 -- so distortion 3 is present and large, and no width estimator plus noise
correction can reach the carrier transport on its own. That is the forward-model
stage's job (the second stage of the pipeline; `profile_likelihood_demo.py` is
that stage made concrete on real reduced-MSD data).

So this script does the width-extraction (first) stage honestly and quantifies
what is left for the forward stage. For each operating point it reports, with no
noise, three trajectories and the two gaps between them:

* carrier second moment, from the simulated population n(y, t) -- the truth;
* recorded-emission second moment, from the emission profile -- the first
  stage's target;
* single-shape (Gaussian) fit of the emission -- a single shape applied to a
  mixture.

The gap (recorded-emission moment) - (carrier moment) is the emission-to-carrier
distortion; only the forward model closes it. The gap (Gaussian fit) -
(recorded-emission moment) is the shape gap; it is a selection issue, removed by
using the moment estimator rather than a single shape, not by the noise
correction.

Under noise it then checks the first stage two ways:

* the moment estimator (non-truncated, no folded-normal) recovers the
  recorded-emission moment where the per-frame contrast-to-noise allows it
  (it is unstable below CNR ~ 3);
* a Gaussian fit plus the noise-bias correction recovers the *noise-free
  Gaussian fit* -- this isolates distortion 1, and is a first-stage diagnostic,
  not a claim about the carrier transport.

Neither first-stage quantity equals the carrier transport; the residual is the
emission-to-carrier gap, reported against the carrier truth.

The Voigt estimator with a Lorentzian instrument point-spread function is not
exercised here. It assumes a single Gaussian convolved with a Lorentzian and is
mismatched to the Saha mixture exactly as the Gaussian estimator is; its
validation belongs on a dedicated single-population point-spread bed, not on this
mixture (see `docs/VALIDATION_DESIGN.md`).

Build the 2-D (CNR, sigma^2) calibration first (needed only for the Gaussian +
folded-normal first-stage diagnostic; the moment path and the gap report need no
calibration)::

    .venv/Scripts/python.exe scripts/build_calibration.py \\
        --tag saha --model gaussian \\
        --n-pixels 401 --half-window 4.0 \\
        --sigma-min 0.15 --sigma-max 1.2 --n-sigma 100 \\
        --cnr-min 0.3 --cnr-max 150 --n-cnr 100 --n-runs 1000

Then run from the repository root::

    .venv/Scripts/python.exe mc_experiments/saha_reaction_diffusion/nonfickian_validation.py

Parameter magnitudes are anchored to the Cutrupi et al. (2026) scales (nu, rho
from Table S2; De/Df to literature 2D-perovskite exciton diffusivities). This is
a method stress-test on a physically grounded transport model, not a claim about
a specific material.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

# Run from a source checkout without installing either package.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, _HERE)

import reaction_diffusion_saha as rd  # noqa: E402
from width_debias.calibration import CalibrationGrid  # noqa: E402
from width_debias.models import GaussianModel, MomentModel  # noqa: E402
from width_debias.bias_lookup import BiasLookup  # noqa: E402
from width_debias.weights import corrected_variance  # noqa: E402

# Physical pixel grid the Saha emission is resampled onto, in um. Matched by the
# calibration build above (n_pixels / half_window identical). The absolute scale
# is free; pixels-per-sigma, not dx, is what the bias depends on. Here
# sigma0 ~ 0.3 um gives ~15 pixels/sigma, and var_max/sigma^2_max = 44 (no pinch).
HALF_WINDOW_UM = 4.0
N_PIXELS = 401
PIXEL_X = np.linspace(-HALF_WINDOW_UM, HALF_WINDOW_UM, N_PIXELS)
N_FRAMES = 12

CALIBRATION_PATH = os.path.join("output", "calibration_saha.npz")

# First-stage estimators. The moment estimator returns the recorded-emission
# second moment directly (no truncation, no calibration); the Gaussian fit is
# the fitted-width estimator the noise-bias correction debiases.
GAUSS = GaussianModel()
MOMENT = MomentModel()

# Correction-reliability floor for the Gaussian + folded-normal diagnostic. The
# diagnostic (diagnose_overcorrection.py) shows the 2-D var_bias matches the
# actual per-frame fit bias for per-frame CNR >~ 1.3 but over-subtracts in the
# band CNR ~ 0.6-1.3; frames below 1.3 are flagged and dropped. The moment
# estimator has its own, separate floor near CNR ~ 3 (variance, not truncation).
CNR_FLOOR = 1.3
MOMENT_CNR_FLOOR = 3.0

EVAL_CNR0 = 60.0          # initial CNR for the correction figures
N_RUNS = 400

# Base Saha parameters (paper-anchored). De/Df contrast drives the non-Fickian
# effect; A is swept to move alpha. Stability needs Df*dt/dx^2 < 0.5.
#
# rho is set at the low edge of the paper's Table S2 range (0.75e-3 to 3e-3
# /ns). At the higher fitted rho the bimolecular term rho*n^2 burns the dense
# core almost instantly, collapsing the emission peak ~25-1000x over the window
# so per-frame CNR cannot be sustained; at rho=1e-3 the decay is
# exciton-dominated (gentle exp(-nu t)), every frame stays above the inversion
# floor, and the non-Fickian signal is actually stronger (diffusion has time to
# act before the signal dies).
BASE = dict(L=24.0, nbins=960, De=0.005, Df=0.1, nu=0.25, rho=1e-3,
            tmax=3.0, dt=0.001, sigma0_init=0.35)
A_SWEEP = np.geomspace(30.0, 2000.0, 11)


@dataclass
class Regime:
    """One operating point: noise-free profiles and reference trajectories.

    Three reference width trajectories, plus the alphas read from each:

    * ``carrier_var`` -- second moment of the simulated population n(y, t).
      This is the truth (the carrier distribution's transport).
    * ``emission_var`` -- second moment of the recorded emission profile. The
      first (width-extraction) stage targets this; it differs from the carrier
      moment by the emission-to-carrier distortion.
    * ``gauss_var`` -- noise-free Gaussian fit of the emission. A single shape
      on a mixture; it differs from ``emission_var`` by the shape gap.
    """

    A: float
    emission_power: int
    times: np.ndarray          # frame times (ns)
    clean: np.ndarray          # (n_frames, n_pixels) noise-free emission (um grid)
    peak: np.ndarray           # per-frame noise-free emission peak (CNR bookkeeping)
    carrier_var: np.ndarray    # truth: population second moment per frame (um^2)
    emission_var: np.ndarray   # recorded-emission second moment per frame (um^2)
    gauss_var: np.ndarray      # noise-free Gaussian-fit variance per frame (um^2)
    alpha_carrier: float       # alpha from the carrier (truth) trajectory
    alpha_emission: float      # alpha from the recorded-emission moment
    alpha_gauss: float         # alpha from the noise-free Gaussian fit
    width_ratio: float         # late / initial emission fitted width


def _alpha(times: np.ndarray, var: np.ndarray) -> float:
    """Anomalous exponent of MSD(t) = var(t) - var(0): slope of log-log fit.

    A non-positive MSD point has no logarithm and is left out; ``_n_dropped``
    counts them so the reports can say how many.
    """
    msd = var - var[0]
    keep = (times > 0) & (msd > 0)
    if keep.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(times[keep]), np.log(msd[keep]), 1)[0])


def _n_dropped(times: np.ndarray, var: np.ndarray) -> int:
    """Frames after t = 0 that ``_alpha`` leaves out because their MSD is not positive."""
    msd = var - var[0]
    return int(np.sum((times > 0) & ~(msd > 0)))


def _moment_on_grid(grid: np.ndarray, profile: np.ndarray) -> float:
    """Second central moment of a (non-negative) profile on an arbitrary grid."""
    w = np.clip(profile, 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.nan
    mean = float((grid * w).sum() / s)
    return float(((grid - mean) ** 2 * w).sum() / s)


def build_regime(A: float, emission_power: int) -> Regime:
    """Run the Saha model at density coupling A and build the reference set.

    No per-regime width normalization: the profiles stay in physical um so the
    single geometry-matched calibration applies to every regime. The carrier
    truth is taken from the population on the simulation's own grid; the
    emission references are taken on PIXEL_X, the grid the fits will run on.
    """
    par = rd.Parameters(L=BASE["L"], nbins=BASE["nbins"], De=BASE["De"],
                        Df=BASE["Df"], nu=BASE["nu"], rho=BASE["rho"], A=A,
                        tsamp=BASE["tmax"] / 40.0, dt=BASE["dt"],
                        tmax=BASE["tmax"],
                        free_charge_emission_power=emission_power)
    xs = rd.x_axis(par)
    res = rd.simulate(par, np.exp(-xs ** 2 / (2.0 * BASE["sigma0_init"] ** 2)))

    idx = np.linspace(0, len(res.times) - 1, N_FRAMES).astype(int)
    times = res.times[idx]
    clean = np.array([np.interp(PIXEL_X, xs, res.emission[i], left=0.0, right=0.0)
                      for i in idx])

    # Truth: carrier (population) second moment, on the simulation's own grid.
    carrier_var = np.array([_moment_on_grid(xs, res.population[i]) for i in idx])
    # Recorded-emission second moment and noise-free Gaussian fit, on PIXEL_X.
    emission_var = np.array([_moment_on_grid(PIXEL_X, row) for row in clean])
    gauss_var = np.array([GAUSS.fit_variance(PIXEL_X, row).variance
                          for row in clean])
    return Regime(
        A=A, emission_power=emission_power, times=times, clean=clean,
        peak=clean.max(axis=1), carrier_var=carrier_var,
        emission_var=emission_var, gauss_var=gauss_var,
        alpha_carrier=_alpha(times, carrier_var),
        alpha_emission=_alpha(times, emission_var),
        alpha_gauss=_alpha(times, gauss_var),
        width_ratio=float(np.sqrt(gauss_var.max() / gauss_var[0])),
    )


def first_stage_trajectories(regime: Regime, cnr0: float,
                             model: BiasLookup | None,
                             n_runs: int = N_RUNS, seed: int = 0):
    """Mean first-stage width trajectories under noise at cnr0.

    Returns a dict of mean trajectories over realisations:

    * ``moment`` -- the non-truncated moment estimator (targets the
      recorded-emission moment; no folded-normal);
    * ``gauss_folded`` -- the Gaussian fit with the noise-bias correction
      (targets the noise-free Gaussian fit; isolates distortion 1). ``None``
      when no calibration is supplied.

    Plus ``cnr_frame`` and the two per-frame floor masks.
    """
    noise_sigma = regime.peak[0] / cnr0
    cnr_frame = regime.peak / noise_sigma
    rng = np.random.default_rng(seed)

    mom = np.full((n_runs, N_FRAMES), np.nan)
    gf = np.full((n_runs, N_FRAMES), np.nan)
    for r in range(n_runs):
        noisy = regime.clean + rng.normal(0.0, noise_sigma, size=regime.clean.shape)
        mom[r] = np.array([MOMENT.fit_variance(PIXEL_X, noisy[f]).variance
                           for f in range(N_FRAMES)])
        if model is not None:
            var_row = np.array([GAUSS.fit_variance(PIXEL_X, noisy[f]).variance
                                for f in range(N_FRAMES)])
            gf[r] = corrected_variance("subtract", var_row, cnr_frame, model)

    out = {
        "moment": np.nanmean(mom, axis=0),
        "gauss_folded": np.nanmean(gf, axis=0) if model is not None else None,
        "cnr_frame": cnr_frame,
        "below_fit_floor": cnr_frame < CNR_FLOOR,
        "below_moment_floor": cnr_frame < MOMENT_CNR_FLOOR,
    }
    return out


def load_model(path: str = CALIBRATION_PATH) -> BiasLookup | None:
    """Load the 2-D Saha calibration, verifying its geometry matches PIXEL_X."""
    if not os.path.exists(path):
        return None
    calib = CalibrationGrid.load(path)
    if calib.pixel_x is None or calib.pixel_x.size != PIXEL_X.size or \
            not np.allclose(calib.pixel_x, PIXEL_X):
        raise ValueError(
            f"calibration geometry in {path} does not match PIXEL_X; "
            "rebuild with --n-pixels 401 --half-window 4.0")
    return BiasLookup.from_calibration(calib)


def gap_report() -> dict[int, list[Regime]]:
    """The two gaps, with no noise: needs no calibration, runs cheaply.

    For each emission power, builds the A-sweep and reports the carrier (truth)
    alpha, the recorded-emission alpha, and the single-shape-fit alpha, so both
    gaps can be read: emission-to-carrier (truth vs emission) and shape
    (emission vs fit).
    """
    out = {}
    for power in (1, 2):
        print(f"\nReference alphas and the two gaps, "
              f"free_charge_emission_power={power} "
              f"({'Marc C code' if power == 1 else 'paper eq. 9'})")
        print("      A   alpha_carrier  alpha_emission  alpha_gaussfit   "
              "emis-carr   shape    width_ratio")
        regimes = [build_regime(float(a), power) for a in A_SWEEP]
        for rg in regimes:
            emis_carr = rg.alpha_emission - rg.alpha_carrier
            shape = rg.alpha_gauss - rg.alpha_emission
            print(f"  {rg.A:7.0f}     {rg.alpha_carrier:7.3f}        "
                  f"{rg.alpha_emission:7.3f}         {rg.alpha_gauss:7.3f}     "
                  f"{emis_carr:+6.3f}   {shape:+6.3f}     {rg.width_ratio:.2f}")
        out[power] = regimes
    return out


def first_stage_report(regimes_by_power: dict[int, list[Regime]],
                       model: BiasLookup | None,
                       cnr0_list: list[float]) -> None:
    """First-stage recovery under noise, scored against each stage's own target.

    The moment estimator is scored against the recorded-emission alpha (its
    target); the Gaussian + folded-normal path against the noise-free
    Gaussian-fit alpha (distortion-1 diagnostic). Both are then compared to the
    carrier truth, whose residual is the emission-to-carrier gap the forward
    stage must close.
    """
    for power, regimes in regimes_by_power.items():
        for ci, cnr0 in enumerate(cnr0_list):
            print(f"\nFirst-stage recovery at CNR_0={cnr0}, "
                  f"free_charge_emission_power={power}")
            print("      A   a_carrier   a_moment(noisy)  a_gauss+folded   "
                  "frames<fit  frames<mom   MSD<=0 dropped (mom, fit)")
            results = []
            for k, rg in enumerate(regimes):
                tr = first_stage_trajectories(rg, cnr0, model,
                                              seed=100 + k + 1000 * ci)
                keep_mom = ~tr["below_moment_floor"]
                keep_fit = ~tr["below_fit_floor"]
                a_mom = (_alpha(rg.times[keep_mom], tr["moment"][keep_mom])
                         if keep_mom.sum() >= 2 else np.nan)
                drop_mom = (_n_dropped(rg.times[keep_mom], tr["moment"][keep_mom])
                            if keep_mom.sum() >= 2 else 0)
                if tr["gauss_folded"] is not None and keep_fit.sum() >= 2:
                    a_gf = _alpha(rg.times[keep_fit], tr["gauss_folded"][keep_fit])
                    drop_fit = _n_dropped(rg.times[keep_fit], tr["gauss_folded"][keep_fit])
                else:
                    a_gf, drop_fit = np.nan, 0
                results.append((rg, tr, a_mom, a_gf))
                print(f"  {rg.A:7.0f}    {rg.alpha_carrier:6.3f}      "
                      f"{a_mom:6.3f}          {a_gf:6.3f}           "
                      f"{int(tr['below_fit_floor'].sum())}          "
                      f"{int(tr['below_moment_floor'].sum())}            "
                      f"{drop_mom}, {drop_fit}")
            if ci == 0:
                _plot_first_stage(power, results, cnr0)


def _plot_first_stage(power, results, cnr0) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # For the lowest, middle, and highest-alpha regimes, show all three
    # reference trajectories and the two first-stage means, so the two gaps and
    # where the first stage lands are visible.
    alphas = np.array([r[0].alpha_carrier for r in results])
    order = np.argsort(alphas)
    picks = [results[order[0]], results[order[len(order) // 2]],
             results[order[-1]]]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), sharey=False)
    for ax, (rg, tr, a_mom, a_gf) in zip(axes, picks):
        t = rg.times
        ax.plot(t, rg.carrier_var, "k-", lw=2, label="carrier moment (truth)")
        ax.plot(t, rg.emission_var, "-", color="tab:blue",
                label="recorded-emission moment")
        ax.plot(t, rg.gauss_var, ":", color="tab:purple",
                label="noise-free Gaussian fit")
        ax.plot(t, tr["moment"], "o", color="tab:cyan", ms=4,
                label="moment estimator (noisy)")
        if tr["gauss_folded"] is not None:
            ax.plot(t, tr["gauss_folded"], "s", color="tab:green", ms=4,
                    label="Gaussian + folded (noisy)")
        ax.set_xlabel("time (ns)")
        ax.set_title(rf"$\alpha_{{carrier}}$={rg.alpha_carrier:.2f}")
    axes[0].set_ylabel(r"variance (um$^2$)")
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"First-stage references and recovery "
                 f"(emission power {power}, CNR$_0$={cnr0:.0f})")
    fig.tight_layout()
    out = os.path.join("output", f"nonfickian_first_stage_power{power}.png")
    os.makedirs("output", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure written to {out}")


def _plot_gaps(regimes_by_power) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[matplotlib not installed -- skipping figures]")
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for power, c in ((1, "tab:green"), (2, "tab:purple")):
        regimes = regimes_by_power[power]
        acar = np.array([rg.alpha_carrier for rg in regimes])
        aemis = np.array([rg.alpha_emission for rg in regimes])
        agf = np.array([rg.alpha_gauss for rg in regimes])
        order = np.argsort(acar)
        ax.plot(acar[order], aemis[order], "o-", color=c,
                label=f"emission moment (power {power})")
        ax.plot(acar[order], agf[order], "s--", color=c, alpha=0.5,
                label=f"Gaussian fit (power {power})")
    lims = ax.get_xlim()
    ax.plot(lims, lims, ls=":", c="grey", lw=1, label="no gap (= carrier truth)")
    ax.set_xlabel(r"carrier (truth) $\alpha$")
    ax.set_ylabel(r"recovered $\alpha$")
    ax.set_title("Emission-to-carrier and shape gaps (before any noise)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs("output", exist_ok=True)
    out = os.path.join("output", "nonfickian_gaps.png")
    fig.savefig(out, dpi=150)
    print(f"\nFigure written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Saha non-Fickian bed in the validation framework: "
                    "carrier-moment truth, the two gaps, first-stage recovery.")
    parser.add_argument("--calibration", default=CALIBRATION_PATH,
                        help="path to the 2-D calibration .npz (Gaussian + "
                             "folded-normal diagnostic only)")
    parser.add_argument("--cnr0", default=str(EVAL_CNR0),
                        help="comma-separated initial CNR values")
    args = parser.parse_args()
    cnr0_list = [float(c) for c in args.cnr0.split(",")]

    print("Saha non-Fickian bed in the validation framework")
    print(f"  Saha model: De={BASE['De']}, Df={BASE['Df']} um^2/ns "
          f"(contrast {BASE['Df'] / BASE['De']:.0f}x), "
          f"nu={BASE['nu']}, rho={BASE['rho']} /ns")
    print(f"  grid: +/-{HALF_WINDOW_UM} um, {N_PIXELS} pixels, {N_FRAMES} frames")
    print("  truth = carrier (population) second moment; emission is nonlinear "
          "in n (x^2 term), so the forward stage is required for carrier alpha")

    regimes_by_power = gap_report()
    _plot_gaps(regimes_by_power)

    model = load_model(args.calibration)
    if model is None:
        print(f"\n[no calibration at {args.calibration}] -- running the "
              "moment-estimator first-stage path only; build the calibration "
              "(see module docstring) to add the Gaussian + folded-normal "
              "diagnostic.")
    else:
        print(f"\nUsing calibration: {args.calibration}")
    first_stage_report(regimes_by_power, model, cnr0_list)


if __name__ == "__main__":
    main()
