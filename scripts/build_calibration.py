"""Build a joint (CNR, sigma^2) calibration for an experiment.

The calibration is geometry-specific: pass the pixel grid, sigma range,
CNR range, profile model, and MC budget for whatever experiment (real or
simulated) you want to analyze. The result is saved as a ``.npz`` archive
that the application can later load via ``CalibrationGrid.load(path)``.

Examples
--------
Marc Mn-trapping geometry, Gaussian fit, 100x100 cells, 1000 runs/cell::

    .venv/Scripts/python.exe scripts/build_calibration.py \\
        --tag marc_gaussian \\
        --n-pixels 250 --half-window 2.5 \\
        --sigma-min 0.07 --sigma-max 0.40 --n-sigma 100 \\
        --cnr-min 0.1 --cnr-max 250 --n-cnr 100 \\
        --n-runs 1000 \\
        --model gaussian

Marc geometry with Voigt fit, including a Lorentzian gamma:

    .venv/Scripts/python.exe scripts/build_calibration.py \\
        --tag marc_voigt \\
        --n-pixels 250 --half-window 2.5 \\
        --sigma-min 0.07 --sigma-max 0.40 --n-sigma 100 \\
        --cnr-min 0.1 --cnr-max 250 --n-cnr 100 \\
        --n-runs 1000 \\
        --model voigt --voigt-gamma 0.01

The resulting calibration is saved as ``output/calibration_<tag>.npz`` and
can be loaded by the application with ``CalibrationGrid.load(path)``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from width_debias import MODELS, calibrate_cnr_sigma2  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
OUTPUT.mkdir(exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a joint (CNR, sigma^2) calibration.")
    p.add_argument("--tag", required=True,
                   help="suffix for the output filename "
                        "(output/calibration_<tag>.npz)")
    p.add_argument("--n-pixels", type=int, required=True,
                   help="number of pixels in the spatial grid")
    p.add_argument("--half-window", type=float, required=True,
                   help="half-width of the spatial grid (in absolute units, "
                        "e.g. um for Marc's data). The pixel grid runs from "
                        "-half_window to +half_window with n_pixels samples.")
    p.add_argument("--sigma-min", type=float, required=True,
                   help="lower edge of the sigma axis (linear-in-sigma "
                        "spacing). Absolute units; should match the "
                        "experiment's expected width range.")
    p.add_argument("--sigma-max", type=float, required=True,
                   help="upper edge of the sigma axis. Pick to cover the "
                        "data's sigma_true with margin.")
    p.add_argument("--n-sigma", type=int, default=100,
                   help="number of sigma points (default 100)")
    p.add_argument("--cnr-min", type=float, default=0.3,
                   help="lower edge of the CNR axis (log spacing)")
    p.add_argument("--cnr-max", type=float, default=100.0,
                   help="upper edge of the CNR axis")
    p.add_argument("--n-cnr", type=int, default=100,
                   help="number of CNR points (default 100)")
    p.add_argument("--n-runs", type=int, default=1000,
                   help="Monte Carlo runs per cell (default 1000)")
    p.add_argument("--model", choices=sorted(MODELS.keys()), default="gaussian",
                   help="profile model (default: gaussian)")
    p.add_argument("--voigt-gamma", type=float, default=None,
                   help="Voigt Lorentzian gamma (only used when "
                        "--model=voigt)")
    p.add_argument("--gengauss-shape", type=float, default=None,
                   help="generalized-Gaussian shape exponent p (only used "
                        "when --model=generalized_gaussian)")
    p.add_argument("--n-workers", type=int, default=None,
                   help="parallel workers; 0 uses every core (default: "
                        "WIDTH_DEBIAS_WORKERS, else serial)")
    p.add_argument("--seed", type=int, default=2026,
                   help="root random seed")
    p.add_argument("--amplitude", type=float, default=1.0,
                   help="noise-free peak amplitude (Gaussian noise: only its "
                        "ratio to noise matters, so conventional 1.0)")
    p.add_argument("--noise", choices=["gaussian", "poisson"], default="gaussian",
                   help="per-pixel noise model. gaussian: additive white noise "
                        "of std amplitude/CNR. poisson: photon counting (TCSPC), "
                        "where a constant --background count floor sets the "
                        "baseline shot noise and the peak counts follow from CNR "
                        "= peak / sqrt(background). Use poisson for "
                        "photon-counting data.")
    p.add_argument("--background", type=float, default=0.0,
                   help="Poisson baseline (dark/background) count floor. Sets "
                        "the absolute count scale, which the Poisson bias "
                        "depends on. Estimate it from the data's profile edges. "
                        "Required (> 0) when --noise=poisson.")
    p.add_argument("--read-noise", type=float, default=0.0,
                   help="Gaussian read/electronic noise standard deviation in "
                        "counts, added per pixel on top of Poisson shot noise "
                        "(only used when --noise=poisson). 0 = pure photon "
                        "counting (TCSPC/APD); a value comparable to or larger "
                        "than sqrt(gain*background) is read-noise-dominated "
                        "(camera). Read it from fft-cnr's photon-transfer fit.")
    p.add_argument("--gain", type=float, default=1.0,
                   help="photon-transfer slope (shot variance per unit signal) "
                        "for --noise=poisson; var = gain*signal + read_noise^2. "
                        "1.0 = pure photon counting; >1 is an excess-noise / "
                        "conversion factor. Read it from fft-cnr's "
                        "photon-transfer fit (estimate_noise_model=True).")
    p.add_argument("--max-var", type=float, default=None,
                   help="upper bound on the fitted variance (units of the pixel "
                        "grid). Default keeps the historical span^2. Pass a "
                        "physical cap (G*sigma0)^2 when the window is far wider "
                        "than the spot. MUST match the value the application "
                        "uses, or the var_bias subtraction is mismatched.")
    p.add_argument("--filter-threshold", type=float, default=None,
                   help="per-cell, discard fits whose sigma_hat^2 exceeds "
                        "this when computing the mean bias. The default keeps "
                        "every fit, failed fits' fallback values included, as "
                        "the application keeps them; pass 5 * sigma_max^2 to "
                        "reproduce a grid built with that filter. rel_variance "
                        "is unaffected.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Pixel grid in absolute units, centered at zero.
    pixel_x = np.linspace(-args.half_window, args.half_window, args.n_pixels)

    # Sigma axis linear-spaced in width (denser in sigma^2 near the wall).
    sigma_grid = np.linspace(args.sigma_min, args.sigma_max, args.n_sigma)

    # CNR axis log-spaced.
    cnr_grid = np.logspace(np.log10(args.cnr_min), np.log10(args.cnr_max),
                            args.n_cnr)

    # Build the model; calibrate_cnr_sigma2 passes this instance to every
    # worker, so its settings are the ones calibrated.
    model_settings: dict = {}
    if args.model == "voigt" and args.voigt_gamma is not None:
        model_settings["gamma"] = float(args.voigt_gamma)
    if args.model == "generalized_gaussian" and args.gengauss_shape is not None:
        model_settings["shape"] = float(args.gengauss_shape)
    if args.max_var is not None:
        model_settings["max_var"] = float(args.max_var)
    model = MODELS[args.model](**model_settings)

    if args.noise == "poisson" and args.background <= 0:
        raise SystemExit("--noise=poisson requires --background > 0 "
                         "(the baseline count floor; estimate it from the "
                         "data's profile edges)")

    print(f"Building calibration tag={args.tag!r}")
    print(f"  model     : {args.model}  {model_settings}")
    if args.noise == "poisson":
        base_noise = (args.gain * args.background + args.read_noise ** 2) ** 0.5
        print(f"  noise     : poisson, gain={args.gain:g}, "
              f"background={args.background:g}, read_noise={args.read_noise:g} "
              f"counts  (baseline noise = {base_noise:.3g}, peak = CNR*baseline)")
    else:
        print(f"  noise     : gaussian (std = amplitude/CNR)")
    print(f"  pixel grid: {args.n_pixels} pixels over "
          f"[-{args.half_window}, {args.half_window}]  "
          f"(dx = {2 * args.half_window / (args.n_pixels - 1):.5g})")
    print(f"  sigma     : {args.n_sigma} linear in sigma, "
          f"[{args.sigma_min}, {args.sigma_max}]  "
          f"(sigma^2 from {args.sigma_min**2:.4g} to {args.sigma_max**2:.4g})")
    print(f"  CNR       : {args.n_cnr} log-spaced, "
          f"[{args.cnr_min}, {args.cnr_max}]")
    print(f"  MC budget : {args.n_runs} runs/cell  "
          f"({args.n_cnr * args.n_sigma * args.n_runs:_} total fits)")
    workers = (args.n_workers if args.n_workers is not None
               else os.environ.get("WIDTH_DEBIAS_WORKERS", "1 (serial)"))
    print(f"  workers   : {workers}  (0 = every core)")
    print(f"  seed      : {args.seed}")

    var_max = (2 * args.half_window) ** 2
    print(f"  var_max   : {var_max:.4g}  "
          f"(ratio var_max / sigma^2_max = {var_max / args.sigma_max**2:.1f})")
    if var_max / args.sigma_max ** 2 < 25:
        print(f"  WARNING: var_max / sigma^2_max < 25; fit's upper bound "
              f"may pinch large-sigma cells")

    if args.filter_threshold is None:
        print("  filter    : none; every fit enters the per-cell mean")
    else:
        print(f"  filter    : drop sigma_hat^2 > {args.filter_threshold:.4g} "
              f"from the per-cell mean")

    t0 = time.time()
    cal = calibrate_cnr_sigma2(
        model=model,
        pixel_x=pixel_x,
        sigma_grid=sigma_grid,
        cnr_grid=cnr_grid,
        amplitude=args.amplitude,
        n_runs=args.n_runs,
        seed=args.seed,
        n_workers=args.n_workers,
        filter_threshold=args.filter_threshold,
        noise_model=args.noise,
        background=args.background,
        read_noise=args.read_noise,
        gain=args.gain,
        verbose=True,
    )
    dt = time.time() - t0
    print(f"\nCalibration finished in {dt / 60:.1f} min "
          f"({dt:.0f} s)")
    print(f"  var_bias range:    [{np.nanmin(cal.var_bias):.4g}, "
          f"{np.nanmax(cal.var_bias):.4g}]")
    print(f"  rel_variance range: [{np.nanmin(cal.rel_variance):.4g}, "
          f"{np.nanmax(cal.rel_variance):.4g}]")
    nan_count = int(np.isnan(cal.var_bias).sum())
    if nan_count:
        print(f"  WARNING: {nan_count} / {cal.var_bias.size} cells failed "
              f"(returned NaN)")

    out_path = OUTPUT / f"calibration_{args.tag}.npz"
    cal.save(out_path)
    print(f"\nSaved: {out_path}  "
          f"({out_path.stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
