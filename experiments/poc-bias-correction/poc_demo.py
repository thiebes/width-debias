"""Proof-of-concept demo: the noise bias and the noise-bias correction.

Run from the repository root::

    python experiments/poc-bias-correction/poc_demo.py [MODEL]

MODEL is ``gaussian``, ``voigt``, or ``generalized_gaussian:<p>``.

Produces ``output/poc_demo.png`` (if matplotlib is installed) with two panels:

* (a) the fraction p of MSD points at or below zero and the MSD skewness vs
      per-frame CNR -- the noise in the fitted width grows as the CNR falls;
* (b) the bias <D_est>/D_true vs CNR_0 for OLS, standard inverse-variance, and
      the derived ``subtract`` scheme (see docs/DERIVATION.md).

It also prints a summary table to stdout, so it is useful with or without a
plotting backend.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Allow running from a source checkout without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias import (Experiment, calibrate_d_free, evaluate,  # noqa: E402
                           model_from_spec, spec_tag, sweep_stats)
from width_debias.bias_lookup import BiasLookup  # noqa: E402

# Nanocrystal-solid regime: short diffusion length against the spot size.
# D and tau are chosen so that L_D = sqrt(2 D tau) ~ 0.20 sigma0, i.e.
# L_D / FWHM_0 ~ 0.085 -- the short-diffusion-length corner that disordered
# perovskite nanocrystal solids occupy and where the noise bias bites.
EXP = Experiment(sigma0=1.0, D=0.02, tau=1.0, n_frames=12, t_max=3.0)
CNR0_GRID = np.array([5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 90.0])
N_RUNS = 400
# Seeds are offset by 10^6 per shape so the shapes see independent noise; any
# other spec (such as the bare p = 2 "generalized_gaussian") takes the next offset.
SHAPES = ("gaussian", "voigt", "generalized_gaussian:1.5", "generalized_gaussian:4")


def _profile_model_from_argv():
    """Profile model selected by an optional command-line argument, and its seed offset.

    ``generalized_gaussian:<p>`` sets the shape exponent; the bare
    ``generalized_gaussian`` is p = 2, a Gaussian profile fitted by scale.
    """
    spec = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
    try:
        model = model_from_spec(spec)
    except ValueError as err:
        sys.exit(str(err))
    offset = 10 ** 6 * (SHAPES.index(spec) if spec in SHAPES else len(SHAPES))
    return spec_tag(spec), model, offset


def main() -> None:
    model_name, pmodel, offset = _profile_model_from_argv()
    fwhm = 2.3548 * EXP.sigma0
    ld = np.sqrt(2.0 * EXP.D * EXP.tau)
    print("width-debias POC demo")
    print(f"  profile model: {model_name}")
    print(f"  experiment: D={EXP.D}, tau={EXP.tau}, n_frames={EXP.n_frames}, "
          f"L_D/FWHM_0={ld / fwhm:.3f}")
    print(f"  CNR_0 sweep: {CNR0_GRID}")
    print(f"  {N_RUNS} Monte Carlo runs per CNR_0; seeds: calibration {1 + offset} + i, "
          f"evaluation {20000 + offset} + i\n")

    # Panel (a) diagnostic sweep at the experiment's own D: the MSD sign
    # fraction and skewness need a nonzero true MSD, so this sweep cannot be
    # the D = 0 one. It is not used by the correction.
    print("Diagnostic sweep for panel (a) ...")
    stats = sweep_stats(EXP, CNR0_GRID, n_runs=N_RUNS, seed=1 + offset, model=pmodel)

    # The correction's calibration is D-free: the same sweep at D = 0, so the
    # lookup never uses the D being estimated (docs/DERIVATION.md section 8).
    print("Calibrating (D-free) ...")
    model: BiasLookup = calibrate_d_free(
        CNR0_GRID, sigma0=EXP.sigma0, tau=EXP.tau, half_window=EXP.half_window,
        n_pixels=EXP.n_pixels, n_frames=EXP.n_frames, t_max=EXP.t_max,
        amplitude0=EXP.amplitude0, n_runs=N_RUNS, seed=1 + offset, model=pmodel)

    # Independent evaluation sweep (different seed).
    print("Evaluating schemes ...")
    results = evaluate(EXP, CNR0_GRID, model, n_runs=N_RUNS,
                       seed=20000 + offset, profile_model=pmodel)

    # --- summary table ---------------------------------------------------
    # The combined SE adds the calibration's own Monte Carlo error, which is
    # shared by every run evaluated with this calibration (weights.calibration_D_se).
    print("\n  <D_est> / D_true   (1.000 = unbiased; SE = evaluation and "
          "calibration combined)")
    print("  CNR_0     OLS     inv-var   folded-normal     SE")
    fn = results["subtract"]
    fn_se = np.hypot(fn.spread / np.sqrt(N_RUNS), fn.cal_se)
    for i, c0 in enumerate(CNR0_GRID):
        print(f"  {c0:6.0f}  {results['ols'].mean_bias[i]:7.3f}  "
              f"{results['inverse_variance'].mean_bias[i]:7.3f}  "
              f"{results['subtract'].mean_bias[i]:11.3f}  {fn_se[i]:9.3f}")

    ols = results["ols"].mean_bias
    iv = results["inverse_variance"].mean_bias
    fn = results["subtract"].mean_bias
    f = np.isfinite(ols) & np.isfinite(iv) & np.isfinite(fn)
    print(f"\n  mean |bias-1|  ols={np.nanmean(np.abs(ols[f] - 1)):.2f}  "
          f"inv-var={np.nanmean(np.abs(iv[f] - 1)):.3f}  "
          f"folded-normal={np.nanmean(np.abs(fn[f] - 1)):.3f}  "
          f"(Monte Carlo error alone would give "
          f"{np.sqrt(2 / np.pi) * np.nanmean(fn_se[f]):.3f} for folded-normal)")
    # The verdict is computed, not written in: which scheme leaves a resolved
    # bias depends on the shape (p = 4's raw bias is small enough that
    # inverse-variance weighting alone is close to unbiased).
    z = (fn - 1.0) / fn_se
    iv_se = results["inverse_variance"].spread / np.sqrt(N_RUNS)
    z_iv = (iv - 1.0) / iv_se
    print(f"  folded-normal cells beyond 2 combined SE: {int(np.sum(np.abs(z[f]) > 2))} "
          f"of {int(f.sum())}")
    print(f"  inverse-variance cells beyond 2 evaluation SE: "
          f"{int(np.sum(np.abs(z_iv[f]) > 2))} of {int(f.sum())}")

    _plot(stats, results, model_name)


def _plot(stats, results, model_name) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[matplotlib not installed -- skipping figure; "
              "install with `pip install width-debias[plot]`]")
        return

    cnr = np.concatenate([s.cnr for s in stats])
    p = np.concatenate([s.trunc_fraction for s in stats])
    skew = np.concatenate([s.skew for s in stats])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.3))

    ax0.scatter(cnr, p, s=20, c="tab:red", alpha=0.75)
    ax0.set_xlabel("per-frame CNR")
    ax0.set_ylabel("truncation fraction $p$  =  P(MSD $\\leq$ 0)", color="tab:red")
    ax0.set_xscale("log")
    ax0.set_ylim(-0.02, 0.55)
    ax0.axhline(0.5, ls=":", c="grey", lw=1)
    ax0.tick_params(axis="y", labelcolor="tab:red")
    ax0b = ax0.twinx()
    ax0b.scatter(cnr, skew, s=16, c="tab:blue", alpha=0.5, marker="^")
    ax0b.set_ylabel("MSD sample skewness", color="tab:blue")
    ax0b.tick_params(axis="y", labelcolor="tab:blue")
    ax0.set_title(f"(a)  The wall distorts the width estimator ({model_name})")

    for scheme, color, label in [
        ("ols", "tab:gray", "OLS (unweighted)"),
        ("inverse_variance", "tab:orange", "inverse-variance (standard)"),
        ("subtract", "tab:green", "folded-normal (derived)"),
    ]:
        r = results[scheme]
        ax1.plot(r.cnr0, r.mean_bias, "o-", color=color, label=label)
    ax1.axhline(1.0, ls="--", c="k", lw=1, label="unbiased")
    ax1.set_xlabel("initial CNR$_0$")
    ax1.set_ylabel("$\\langle D_{est}\\rangle\\, /\\, D_{true}$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")   # OLS spans ~40x; log keeps the corrected schemes legible
    ax1.set_title("(b)  Bias in $D$: only the noise-bias correction removes it")
    ax1.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs("output", exist_ok=True)
    suffix = "" if model_name == "gaussian" else f"_{model_name}"
    out = os.path.join("output", f"poc_demo{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    main()
