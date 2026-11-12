"""Where between pure shot and pure white does a real detector sit?

`shot_noise_validation.py` showed the two limits: under constant (white) noise the
MSD-slope bias is huge (D inflated by factors of dozens at low CNR), while under
photon-limited shot noise (read floor = 0) it nearly vanishes, because the clean
wings constrain the width fit. Real photon-counting detectors are in between: a
read-noise / dark-count / background pedestal adds a constant-variance floor that
reintroduces wing noise and brings the bias back. The relevant axis is the read
floor.

This sweep holds the geometry fixed and raises the read floor `r` (constant noise
in count units, added in quadrature to the local shot variance) from 0 upward.
For each `r` it reports, summarised as mean |mean(D_hat)/D_true - 1| over the
CNR_0 sweep:

  OLS raw            -- the uncorrected bias, climbing from ~0 (pure shot) toward
                        the white level as the pedestal grows.
  subtract, matched  -- the correction with a calibration built at the same read
                        floor (CORRECTNESS).
  subtract, white cal -- the shipped white calibration applied to this data
                        (ROBUSTNESS): it over-corrects at small r and is
                        correct only on white data.

The pure-white case is run as the large-r reference. Reading the read floor in
physical terms needs the detector's photon budget: the floor here is in the same
counts as the peak, which holds cnr0**2 counts at frame 0, so a floor of r counts
matters most for the dim, low-CNR frames where peak counts are fewest.

Run from the repository root (set WIDTH_DEBIAS_WORKERS to parallelise)::

    WIDTH_DEBIAS_WORKERS=0 .venv/Scripts/python.exe experiments/noise-law/read_floor_sweep.py

Sweep sizes follow the SHOT_* environment variables (small defaults for a quick
serial smoke run; raise for the full run).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias.noise import ShotNoise, WhiteNoise  # noqa: E402
from width_debias.simulate import Experiment  # noqa: E402
from width_debias.validate import calibrate_d_free, evaluate  # noqa: E402

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "..", "output")

GEOM = dict(sigma0=1.0, tau=1.0, half_window=6.0, n_pixels=241,
            n_frames=12, t_max=3.0)
D_TRUE = 0.02                       # sigma_0^2 / tau; L_D / FWHM_0 ~ 0.085

N_CAL = int(os.environ.get("SHOT_NCAL", 120))
N_EVAL = int(os.environ.get("SHOT_NEVAL", 120))
N_EVAL_GRID = int(os.environ.get("SHOT_NGRID", 8))
CAL_CNR0 = np.geomspace(2.0, 80.0, 30)
EVAL_CNR0 = np.geomspace(3.0, 60.0, N_EVAL_GRID)

# Read floors in count units (0 = photon-limited). The peak holds cnr0**2 counts
# at frame 0, so these span negligible-vs-peak-shot up to pedestal-dominated.
READ_FLOORS = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])


def _mad(curve: np.ndarray) -> float:
    return float(np.nanmean(np.abs(curve - 1.0)))


def _plot(ols: np.ndarray, matched: np.ndarray, white: np.ndarray,
          ols_white: float, subtract_white: float, path: str) -> None:
    """Bias against read floor, three curves, with the white-noise levels marked.

    The read-floor axis is symmetric-logarithmic (linear below 1 count), so
    read = 0 sits at its true place instead of at a stand-in value on a log
    axis. The two dotted lines are the same metric on pure white noise: OLS,
    and `subtract` with the white calibration, the level the white-calibration
    curve would reach if the read floor made the noise white. DICE palette;
    the caption carries the title role.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    blue, gold, green = "#003f7f", "#f7941e", "#0cce6b"
    inch, font = 1 / 2.54, 8
    fig, ax = plt.subplots(figsize=(12 * inch, 9 * inch), layout="constrained")
    fig.patch.set_alpha(0)
    ax.set_facecolor("white")
    ax.plot(READ_FLOORS, ols, "o-", color=gold, lw=2, label="uncorrected (OLS)")
    ax.plot(READ_FLOORS, white, "s-", color=blue, lw=2,
            label="subtract, white calibration")
    ax.plot(READ_FLOORS, matched, "o-", color=green, lw=2,
            label="subtract, matched calibration")
    ax.axhline(ols_white, color=gold, ls=":", lw=1.2)
    ax.axhline(subtract_white, color=blue, ls=":", lw=1.2)
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.5)
    ax.set_yscale("log")
    ax.set_xlim(-0.15, 1.25 * READ_FLOORS[-1])
    values = np.concatenate([ols, matched, white, [ols_white, subtract_white]])
    ax.set_ylim(0.6 * np.nanmin(values), 2.2 * np.nanmax(values))
    ax.text(0.0, ols_white * 1.12, "OLS on white noise", color=gold,
            fontsize=font, ha="left", va="bottom")
    ax.text(1.2 * READ_FLOORS[-1], subtract_white / 1.12,
            "subtract on white noise", color=blue, fontsize=font,
            ha="right", va="top")
    ax.xaxis.set_major_locator(FixedLocator(READ_FLOORS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("read floor (counts)", fontsize=font)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel(r"mean $|\langle\hat D\rangle/D_\mathrm{true}-1|$", fontsize=font)
    ax.tick_params(direction="in", which="both", top=False, right=False,
                   labelsize=font)
    ax.legend(fontsize=font, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.5,
              columnspacing=1.0, borderaxespad=0.2)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    print("MSD-slope bias vs detector read floor (pure shot -> pure white)")
    print(f"  D_true = {D_TRUE} sigma_0^2/tau")
    print(f"  calibration: {CAL_CNR0.size} CNR_0 in [2, 80], {N_CAL} runs/cell")
    print(f"  evaluation:  {EVAL_CNR0.size} CNR_0 in [3, 60], {N_EVAL} runs/cell")
    print(f"  read floors (counts): {READ_FLOORS.tolist()}")
    print(f"  workers = {os.environ.get('WIDTH_DEBIAS_WORKERS', '(serial)')}")

    # The shipped default: a white-noise calibration. Used as the robustness
    # reference (applied to every read-floor case) and as the large-r limit.
    print("\nBuilding the default white calibration at D = 0 ...", flush=True)
    white_cal = calibrate_d_free(CAL_CNR0, n_runs=N_CAL, noise_model=WhiteNoise(),
                                 **GEOM)

    exp = Experiment(D=D_TRUE, **GEOM)
    ols, subtract_matched, subtract_white = [], [], []
    for r in READ_FLOORS:
        model = ShotNoise(read=float(r), poisson=True)
        print(f"\n[read = {r:5.1f}] building matched calibration ...", flush=True)
        cal_r = calibrate_d_free(CAL_CNR0, n_runs=N_CAL, noise_model=model, **GEOM)
        r_match = evaluate(exp, EVAL_CNR0, cal_r, n_runs=N_EVAL, noise_model=model)
        r_white = evaluate(exp, EVAL_CNR0, white_cal, n_runs=N_EVAL,
                           noise_model=model)
        ols.append(_mad(r_match["ols"].mean_bias))
        subtract_matched.append(_mad(r_match["subtract"].mean_bias))
        subtract_white.append(_mad(r_white["subtract"].mean_bias))
        print(f"  OLS raw {ols[-1]:.3f} | subtract matched {subtract_matched[-1]:.3f} "
              f"| subtract white-cal {subtract_white[-1]:.3f}")

    # The pure-white limit, for reference.
    print("\n[white limit] evaluating white data with the white calibration ...",
          flush=True)
    r_w = evaluate(exp, EVAL_CNR0, white_cal, n_runs=N_EVAL,
                   noise_model=WhiteNoise())
    ols_white = _mad(r_w["ols"].mean_bias)
    subtract_white_white = _mad(r_w["subtract"].mean_bias)

    ols = np.array(ols)
    subtract_matched = np.array(subtract_matched)
    subtract_white = np.array(subtract_white)

    print("\n" + "=" * 72)
    print("mean |mean(D_hat)/D_true - 1| over the CNR_0 sweep, vs read floor")
    print("=" * 72)
    print(f"  {'read':>6} {'OLS raw':>10} {'subtract matched':>16} "
          f"{'subtract white-cal':>18}")
    for i, r in enumerate(READ_FLOORS):
        print(f"  {r:>6.1f} {ols[i]:>10.3f} {subtract_matched[i]:>16.3f} "
              f"{subtract_white[i]:>18.3f}")
    print(f"  {'white':>6} {ols_white:>10.3f} {'-':>16} "
          f"{subtract_white_white:>18.3f}")

    path = os.path.join(OUTPUT, "read_floor_sweep.png")
    _plot(ols, subtract_matched, subtract_white, ols_white, subtract_white_white,
          path)
    print(f"\nSaved {os.path.basename(path)}")


if __name__ == "__main__":
    main()
