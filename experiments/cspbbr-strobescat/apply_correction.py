"""Apply the noise-bias correction to a reduced CsPbBr3 fluence.

Loads the per-frame profiles from ``reduce.py`` and a ``(CNR, sigma^2)``
calibration built by ``scripts/build_calibration.py``, then reports the
diffusion coefficient D under each weighting scheme -- the naive fits
(``ols``, ``inverse_variance``) against the bias-corrected ones
(``subtract``, ``mas``). The correction subtracts the per-frame variance
bias ``var_bias(CNR, sigma_hat^2)`` from each frame before the MSD slope fit.

Example::

    python experiments/cspbbr-strobescat/apply_correction.py \
        --reduced output/cspbbr_13p47.npz \
        --calibration output/calibration_cspbbr_gaussian.npz

The reduction's fit cap (``max_var``) must equal the calibration's ``--max-var``,
and the reduction's pixel grid must match the calibration's, or the bias lookup
does not apply (the script warns if the grids disagree).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from width_debias.calibration import CalibrationGrid     # noqa: E402
from width_debias.bias_lookup import BiasLookup       # noqa: E402
from width_debias.weights import msd_and_weights, SCHEMES  # noqa: E402
from width_debias.fit import fit_D                         # noqa: E402

UM2_PER_NS_TO_CM2_PER_S = 10.0      # 1 um^2/ns = 10 cm^2/s


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reduced", required=True, help="reduce.py .npz output")
    p.add_argument("--calibration", required=True,
                   help="calibration .npz (scripts/build_calibration.py)")
    p.add_argument("--json", default=None,
                   help="also write per-scheme D results to this JSON path")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    d = np.load(args.reduced)
    t_ns = d["t_ns"]
    variance = d["variance_um2"]
    cnr = d["cnr"]
    x_um = d["x_um"]

    calib = CalibrationGrid.load(args.calibration)
    model = BiasLookup.from_calibration(calib)

    # geometry consistency checks (the lookup is conditional on the grid)
    if calib.pixel_x is not None:
        if calib.pixel_x.size != x_um.size or not np.isclose(
                np.max(np.abs(calib.pixel_x)), np.max(np.abs(x_um)), rtol=0.02):
            print(f"WARNING: pixel grid mismatch -- calibration "
                  f"{calib.pixel_x.size} px / half-window "
                  f"{np.max(np.abs(calib.pixel_x)):.3f}; data {x_um.size} px / "
                  f"{np.max(np.abs(x_um)):.3f}. The bias lookup may not apply.")
    print(f"calibration: model={calib.model_name}, noise={calib.noise_model}, "
          f"{calib.n_runs} runs/cell")
    print(f"data: {variance.size} frames, CNR {np.nanmin(cnr):.2f}-"
          f"{np.nanmax(cnr):.1f}\n")

    results = {}
    for scheme in SCHEMES:
        msd, w = msd_and_weights(scheme, variance, cnr, model)
        dfit = fit_D(t_ns, msd, w)
        d_cm2s = dfit.D * UM2_PER_NS_TO_CM2_PER_S
        se_cm2s = dfit.D_se * UM2_PER_NS_TO_CM2_PER_S
        results[scheme] = (d_cm2s, se_cm2s, msd)
        print(f"  {scheme:18s} D = {d_cm2s:.4f} +/- {se_cm2s:.4f} cm^2/s")

    if args.json:
        import json
        payload = {s: {"D_cm2s": results[s][0], "D_se_cm2s": results[s][1]}
                   for s in SCHEMES}
        payload["cnr_min"] = float(np.nanmin(cnr))
        payload["cnr_max"] = float(np.nanmax(cnr))
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)

    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        naive = results["inverse_variance"][2]
        corr = results["subtract"][2]
        ax.plot(t_ns, naive, "o", color="0.5", label="naive (inverse-variance)")
        ax.plot(t_ns, corr, "o", color="C0", label="corrected (folded-normal)")
        for scheme, color in (("inverse_variance", "0.5"), ("subtract", "C0")):
            d_um2ns = results[scheme][0] / UM2_PER_NS_TO_CM2_PER_S
            ax.plot(t_ns, 2 * d_um2ns * t_ns, "-", color=color)
        ax.set_xlabel("time (ns)"); ax.set_ylabel("MSD (um^2)")
        ax.legend()
        fig.tight_layout()
        out = os.path.splitext(args.reduced)[0] + "_corrected.png"
        fig.savefig(out, dpi=110)
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
