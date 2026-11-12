"""Stage two on the Saha bed: recover the carrier transport by forward fit.

The companion script `nonfickian_validation.py` shows the first (width-extraction)
stage: on this nonlinear, multi-population bed no width estimator plus noise
correction reaches the carrier transport, because the recorded emission is a
distorted view of the carrier distribution (the x^2 free-charge term sharpens the
peak). That distortion is the forward-model stage's job. This script is that
stage.

The forward model is the same Saha engine that generates the bed
(`reaction_diffusion_saha.simulate`), so truth and fit share one model and only
the measurement noise and the pipeline are under test. Following Antonella
Cutrupi's fit (`Fit_Simulation_Diff_free_vs_exc_updated.py`): the fit observable
is the emission (diffusion-map) second-moment MSD, and the lifetime parameters
(qe, qf, nu, rho, A) are fixed -- on this synthetic bed they are known inputs, so
only De and Df are fitted. The single-number L-BFGS-B inverse-Hessian error her
script reports is replaced by a profile-likelihood interval (the chi^2 minimised
over the other parameter rises by 1), as in `profile_likelihood_demo.py`.

The framework point the bed makes concrete: a forward fit is unbiased only when
the model's observable matches the measurement's. The model predicts the
emission second moment, so:

* feeding it the moment-estimator measurement (which is the emission second
  moment, approximately unbiased) recovers the true De, Df and the carrier alpha;
* feeding it the Gaussian-fit-plus-correction measurement (which converges to
  the noise-free Gaussian fit, not the emission moment, because the noise-bias
  correction removes the noise bias but not the shape gap) recovers biased De, Df
  -- the shape gap propagates into the transport parameters.

So the moment estimator (or any reduction that returns the second moment) is the
correct first-stage input to this forward fit; a single-shape fit is not, unless
the forward model is changed to predict that same single-shape fit.

Run from the repository root::

    .venv/Scripts/python.exe mc_experiments/saha_reaction_diffusion/forward_fit_validation.py

The Gaussian-plus-correction comparison needs the 2-D calibration
`output/calibration_saha.npz` (see `nonfickian_validation.py`); without it the
moment path and the noise-free check still run.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, _HERE)

import reaction_diffusion_saha as rd  # noqa: E402
from width_debias.calibration import CalibrationGrid  # noqa: E402
from width_debias.models import GaussianModel  # noqa: E402
from width_debias.bias_lookup import BiasLookup  # noqa: E402
from width_debias.weights import corrected_variance  # noqa: E402

# Measurement grid the emission is resampled onto (matches nonfickian_validation
# and the 2-D Saha calibration geometry).
HALF_WINDOW_UM = 4.0
N_PIXELS = 401
PIXEL_X = np.linspace(-HALF_WINDOW_UM, HALF_WINDOW_UM, N_PIXELS)
N_FRAMES = 12

CALIBRATION_PATH = os.path.join("output", "calibration_saha.npz")
CNR_FLOOR = 1.3            # correction reliability floor (per-frame CNR)
MOMENT_CNR_FLOOR = 3.0     # moment-estimator stability floor

GAUSS = GaussianModel()

# Same bed as nonfickian_validation.py. De, Df are the truth the forward fit must
# recover; A moves the non-Fickian strength; emission power 2 is the correct x^2.
BASE = dict(L=24.0, nbins=960, De=0.005, Df=0.1, nu=0.25, rho=1e-3,
            tmax=3.0, dt=0.001, sigma0_init=0.35)
EVAL_CNR0 = 60.0
N_RUNS = 300

# De/Df search grid, bracketing the truth (De=0.005, Df=0.1). Log-spaced so the
# relative resolution is uniform; the profile-likelihood parabola resolves
# intervals finer than a grid step.
DE_GRID = np.geomspace(0.0015, 0.02, 17)
DF_GRID = np.geomspace(0.04, 0.26, 17)


@dataclass
class Bed:
    """Noise-free references for one operating point (A, emission power)."""

    A: float
    emission_power: int
    times: np.ndarray          # frame times (ns)
    clean: np.ndarray          # (n_frames, n_pixels) noise-free emission on PIXEL_X
    peak: np.ndarray           # per-frame noise-free emission peak
    carrier_alpha: float       # truth: alpha from the population second moment
    emission_msd: np.ndarray   # noise-free emission-moment MSD on PIXEL_X (um^2)


def _alpha(times: np.ndarray, msd: np.ndarray) -> float:
    """Anomalous exponent: slope of log MSD vs log t over positive points."""
    keep = (times > 0) & (msd > 0)
    if keep.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(times[keep]), np.log(msd[keep]), 1)[0])


def _moment_on_grid(grid: np.ndarray, profile: np.ndarray) -> float:
    """Second central moment of a (non-negative) profile on an arbitrary grid."""
    w = np.clip(profile, 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.nan
    mean = float((grid * w).sum() / s)
    return float(((grid - mean) ** 2 * w).sum() / s)


def _noisy_second_moment(grid: np.ndarray, profile: np.ndarray) -> float:
    """Second central moment with no baseline subtraction and no clipping.

    This matches Antonella Cutrupi's reduction (her ``variance()``): for
    zero-mean measurement noise and no real background it is approximately
    unbiased, because admitting negatives keeps the noise contribution
    zero-mean. The edge-baseline ``MomentModel`` is the wrong reduction here --
    as the profile spreads into the window edges, the baseline picks up real
    signal and biases the moment low (worse with time), exactly the
    window/baseline sensitivity flagged in docs/VALIDATION_DESIGN.md.
    """
    s = float(profile.sum())
    if s <= 0:
        return np.nan
    mean = float((grid * profile).sum() / s)
    return float(((grid - mean) ** 2 * profile).sum() / s)


def _simulate(De: float, Df: float, A: float, emission_power: int):
    """Run the Saha engine; return (frame times, emission on PIXEL_X, native xs,
    population on xs) sampled to N_FRAMES."""
    par = rd.Parameters(L=BASE["L"], nbins=BASE["nbins"], De=De, Df=Df,
                        nu=BASE["nu"], rho=BASE["rho"], A=A,
                        tsamp=BASE["tmax"] / 40.0, dt=BASE["dt"],
                        tmax=BASE["tmax"],
                        free_charge_emission_power=emission_power)
    xs = rd.x_axis(par)
    res = rd.simulate(par, np.exp(-xs ** 2 / (2.0 * BASE["sigma0_init"] ** 2)))
    idx = np.linspace(0, len(res.times) - 1, N_FRAMES).astype(int)
    emis = np.array([np.interp(PIXEL_X, xs, res.emission[i], left=0.0, right=0.0)
                     for i in idx])
    pop = np.array([res.population[i] for i in idx])
    return res.times[idx], emis, xs, pop


def emission_msd(De: float, Df: float, A: float, emission_power: int) -> np.ndarray:
    """Emission-moment MSD on PIXEL_X for trial (De, Df): the fit observable."""
    _, emis, _, _ = _simulate(De, Df, A, emission_power)
    var = np.array([_moment_on_grid(PIXEL_X, row) for row in emis])
    return np.clip(var - var[0], 0.0, None)


def build_bed(A: float, emission_power: int) -> Bed:
    """Truth references: carrier alpha (population) and the emission-moment MSD."""
    times, emis, xs, pop = _simulate(BASE["De"], BASE["Df"], A, emission_power)
    carrier_var = np.array([_moment_on_grid(xs, row) for row in pop])
    emis_var = np.array([_moment_on_grid(PIXEL_X, row) for row in emis])
    return Bed(
        A=A, emission_power=emission_power, times=times, clean=emis,
        peak=emis.max(axis=1),
        carrier_alpha=_alpha(times, np.clip(carrier_var - carrier_var[0], 0, None)),
        emission_msd=np.clip(emis_var - emis_var[0], 0.0, None),
    )


def stage1_measurements(bed: Bed, cnr0: float, model: BiasLookup | None,
                        n_runs: int = N_RUNS, seed: int = 0) -> dict:
    """Mean first-stage MSD and its MC error from each estimator under noise.

    Returns per-estimator (mean MSD, MSD standard error, per-frame keep mask).
    The moment path returns the emission second moment (its target); the
    Gaussian+subtract path returns the noise-debiased Gaussian fit.
    """
    noise_sigma = bed.peak[0] / cnr0
    cnr_frame = bed.peak / noise_sigma
    rng = np.random.default_rng(seed)

    mom = np.full((n_runs, N_FRAMES), np.nan)
    gf = np.full((n_runs, N_FRAMES), np.nan)
    for r in range(n_runs):
        noisy = bed.clean + rng.normal(0.0, noise_sigma, size=bed.clean.shape)
        mom[r] = np.array([_noisy_second_moment(PIXEL_X, noisy[f])
                           for f in range(N_FRAMES)])
        if model is not None:
            vr = np.array([GAUSS.fit_variance(PIXEL_X, noisy[f]).variance
                           for f in range(N_FRAMES)])
            gf[r] = corrected_variance("subtract", vr, cnr_frame, model)

    def reduce(arr):
        mean = np.nanmean(arr, axis=0)
        n_ok = np.sum(np.isfinite(arr), axis=0)
        se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n_ok, 1))
        msd = np.clip(mean - mean[0], 0.0, None)
        return msd, se

    out = {"cnr_frame": cnr_frame}
    out["moment_msd"], out["moment_se"] = reduce(mom)
    out["moment_keep"] = cnr_frame >= MOMENT_CNR_FLOOR
    if model is not None:
        out["gauss_msd"], out["gauss_se"] = reduce(gf)
        out["gauss_keep"] = cnr_frame >= CNR_FLOOR
    return out


def _model_grid(A: float, emission_power: int):
    """Precompute the emission-moment MSD for every (De, Df) cell once."""
    grid = np.empty((DE_GRID.size, DF_GRID.size, N_FRAMES))
    for i, de in enumerate(DE_GRID):
        for j, df in enumerate(DF_GRID):
            grid[i, j] = emission_msd(de, df, A, emission_power)
    return grid


def _interval(grid: np.ndarray, prof: np.ndarray, chi2_min: float):
    """Profile-likelihood 1-sigma interval: parabola where chi^2 rises by 1."""
    near = (prof - chi2_min) <= 9.0
    if near.sum() < 3:
        k = int(np.argmin(prof))
        near = np.zeros_like(prof, bool)
        near[max(0, k - 1):k + 2] = True
    c2, c1, c0 = np.polyfit(grid[near], prof[near], 2)
    if c2 <= 0:
        return np.nan, np.nan, float(grid[int(np.argmin(prof))])
    hat = -c1 / (2 * c2)
    disc = (c0 - (chi2_min + 1.0)) / c2 + hat ** 2
    if disc <= 0:
        return np.nan, np.nan, hat
    r = np.sqrt(disc)
    return hat - r, hat + r, hat


def forward_fit(model_grid: np.ndarray, msd_meas: np.ndarray,
                sigma: np.ndarray, keep: np.ndarray) -> dict:
    """Chi^2 over (De, Df) against one measurement; profile-likelihood intervals.

    Only frames in ``keep`` (above the estimator's floor) enter the chi^2.
    Returns De/Df point estimates, 1-sigma intervals, and the reduced chi^2.
    """
    sig = np.maximum(sigma, 1e-9)
    diff = (model_grid - msd_meas) / sig
    surf = np.sum((diff ** 2)[:, :, keep], axis=2)
    chi2_min = float(surf.min())
    prof_de = surf.min(axis=1)
    prof_df = surf.min(axis=0)
    de_lo, de_hi, de_hat = _interval(DE_GRID, prof_de, chi2_min)
    df_lo, df_hi, df_hat = _interval(DF_GRID, prof_df, chi2_min)
    dof = max(int(keep.sum()) - 2, 1)
    return dict(de_hat=de_hat, de_lo=de_lo, de_hi=de_hi,
                df_hat=df_hat, df_lo=df_lo, df_hi=df_hi,
                chi2_red=chi2_min / dof)


def load_model(path: str = CALIBRATION_PATH) -> BiasLookup | None:
    if not os.path.exists(path):
        return None
    calib = CalibrationGrid.load(path)
    if calib.pixel_x is None or calib.pixel_x.size != PIXEL_X.size or \
            not np.allclose(calib.pixel_x, PIXEL_X):
        raise ValueError(
            f"calibration geometry in {path} does not match PIXEL_X; "
            "rebuild with --n-pixels 401 --half-window 4.0")
    return BiasLookup.from_calibration(calib)


def _report_fit(label: str, fit: dict, bed: Bed, A: float, power: int) -> None:
    """Print one forward-fit result and the carrier alpha it implies."""
    de, df = fit["de_hat"], fit["df_hat"]
    if np.isfinite(de) and np.isfinite(df):
        car_alpha = _alpha(*_carrier_alpha_inputs(de, df, A, power))
    else:
        car_alpha = np.nan
    print(f"  {label:<26s} De={de:.4f} [{fit['de_lo']:.4f}, {fit['de_hi']:.4f}]   "
          f"Df={df:.4f} [{fit['df_lo']:.4f}, {fit['df_hi']:.4f}]   "
          f"alpha_carrier(fit)={car_alpha:.3f}   chi2_red={fit['chi2_red']:.2f}")


def _carrier_alpha_inputs(De: float, Df: float, A: float, power: int):
    """(times, carrier MSD) at (De, Df) -- for the implied carrier alpha."""
    times, _, xs, pop = _simulate(De, Df, A, power)
    cvar = np.array([_moment_on_grid(xs, row) for row in pop])
    return times, np.clip(cvar - cvar[0], 0.0, None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Saha stage-2 forward fit: recover De/Df and carrier alpha.")
    parser.add_argument("--calibration", default=CALIBRATION_PATH)
    parser.add_argument("--A", default="863", help="comma-separated A values")
    parser.add_argument("--power", type=int, default=2, choices=(1, 2),
                        help="free-charge emission power (2 = correct x^2)")
    parser.add_argument("--cnr0", type=float, default=EVAL_CNR0)
    args = parser.parse_args()
    a_list = [float(a) for a in args.A.split(",")]

    print("Saha stage-2 forward fit (recover the carrier transport)")
    print(f"  truth: De={BASE['De']}, Df={BASE['Df']} um^2/ns; "
          f"emission power {args.power}; CNR_0={args.cnr0}")
    print(f"  De grid [{DE_GRID[0]:.4f}, {DE_GRID[-1]:.4f}], "
          f"Df grid [{DF_GRID[0]:.4f}, {DF_GRID[-1]:.4f}]")

    model = load_model(args.calibration)
    if model is None:
        print(f"  [no calibration at {args.calibration}] -- moment path and "
              "noise-free check only")

    for A in a_list:
        bed = build_bed(A, args.power)
        print(f"\nA={A:.0f}  carrier-truth alpha={bed.carrier_alpha:.3f}  "
              f"emission-MSD alpha={_alpha(bed.times, bed.emission_msd):.3f}")
        grid = _model_grid(A, args.power)
        meas = stage1_measurements(bed, args.cnr0, model)

        # Diagnostic: is the moment path off-truth from bias or variance? Compare
        # the moment-estimator mean MSD to the noise-free emission MSD per frame.
        print("  frame  cnr   emis_MSD   mom_MSD   mom_SE"
              + ("   gauss_MSD" if model is not None else ""))
        for f in range(N_FRAMES):
            line = (f"  {f:5d}  {meas['cnr_frame'][f]:4.1f}  "
                    f"{bed.emission_msd[f]:8.4f}  {meas['moment_msd'][f]:8.4f}  "
                    f"{meas['moment_se'][f]:7.4f}")
            if model is not None:
                line += f"   {meas['gauss_msd'][f]:8.4f}"
            print(line)

        # Noise-free check: fit the exact emission-moment MSD (machinery + the
        # carrier-alpha recovery; should return the truth by self-consistency).
        keep_all = np.ones(N_FRAMES, bool)
        sig_nf = np.maximum(0.01 * bed.emission_msd, 1e-3)
        fit_nf = forward_fit(grid, bed.emission_msd, sig_nf, keep_all)
        _report_fit("noise-free emission", fit_nf, bed, A, args.power)

        # Moment estimator: the emission second moment under noise. Correct
        # observable for this model -> unbiased De/Df.
        fit_mom = forward_fit(grid, meas["moment_msd"], meas["moment_se"],
                              meas["moment_keep"])
        _report_fit("moment (noisy)", fit_mom, bed, A, args.power)

        # Gaussian + subtract: noise-debiased, but the shape gap remains, so
        # the observable does not match the model -> biased De/Df.
        if model is not None:
            fit_gf = forward_fit(grid, meas["gauss_msd"], meas["gauss_se"],
                                 meas["gauss_keep"])
            _report_fit("Gaussian+subtract (noisy)", fit_gf, bed, A, args.power)


if __name__ == "__main__":
    main()
