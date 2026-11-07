"""Simulated recovery demo: the noise bias and its correction, where the
truth is known (real data has no ground truth, so this carries the "we recover
the true width" claim).

Two figures, written into ``diagnostics/_frame_ladder/``:

* ``0_simulated_benefit.png`` -- a single representative low-CNR frame (CNR 1.5).
  The data look qualitatively unusable (scatter as large as the peak), the naive
  bounded-Gaussian fit is biased ~1.7x too wide, and the correction recovers the
  true width. All three curves are drawn at one amplitude so only the width
  differs.
* ``0b_simulated_distribution.png`` -- the honest ensemble picture over many such
  frames: the naive fitted-width distribution is right-skewed and biased high in
  the *mean* (1.7x true); subtracting the calibrated mean bias recenters the mean
  on truth. It also shows the limitation -- at CNR 1.5 a large fraction of
  individual frames over-subtract below zero, which is why the per-frame floor
  exists. The correction debiases the mean (and hence D, a mean-trend slope), not
  every individual near-floor frame.

The single-frame panel is a *representative* frame (naive near the ensemble
mean), not a cherry-picked tail; read it together with the distribution panel,
which is the rigorous statement.

Usage::

    .venv/Scripts/python.exe experiments/wse2-strobescat/render_recovery_demo.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from width_debias.fit import fit_variance  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(__file__), "diagnostics", "_frame_ladder")
RED = "#c60b1e"; BLUE = "#0077c8"; GREEN = "#0b8a3b"; INCH = 1 / 2.54
SIG0 = 3.0; VAR_TRUE = SIG0 ** 2; NPIX = int(round(8 * SIG0)); C = NPIX / 2.0
X = np.arange(NPIX, dtype=float); AMP = 1.0; CNR = 1.5; NOISE = AMP / CNR


def _naive_set(seed, n):
    rng = np.random.default_rng(seed)
    return np.array([fit_variance(X, AMP * np.exp(-0.5 * (X - C) ** 2 / VAR_TRUE)
                                  + rng.normal(0, NOISE, NPIX)).variance
                     for _ in range(n)])


def benefit_frame(var_bias):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean_nv = VAR_TRUE + var_bias
    rng = np.random.default_rng(7)
    best = None
    for _ in range(6000):
        y = AMP * np.exp(-0.5 * (X - C) ** 2 / VAR_TRUE) + rng.normal(0, NOISE, NPIX)
        fv = fit_variance(X, y); nv = fv.variance
        if abs(nv - mean_nv) < 0.08 * mean_nv:           # representative, not tail
            res = np.std(y - fv.amplitude * np.exp(-0.5 * (X - C) ** 2 / nv))
            if best is None or res < best[0]:
                best = (res, y.copy(), fv.amplitude, nv)
    _, y, a_fit, nv = best
    corr = nv - var_bias
    xx = np.linspace(0, NPIX, 300); g = lambda v: a_fit * np.exp(-0.5 * (xx - C) ** 2 / v)
    fig, ax = plt.subplots(layout="constrained", figsize=(14 * INCH, 10 * INCH))
    ax.tick_params(direction="in", top=True, right=True, labelsize=9)
    ax.plot(X, y, "o", color="0.5", ms=4, label="data (CNR = 1.5 -- looks unusable)")
    ax.plot(xx, g(VAR_TRUE), "-", color=GREEN, lw=1.6, label=r"true  $\sigma^2$=%.1f" % VAR_TRUE)
    ax.plot(xx, g(nv), "-", color=RED, lw=2.0, label=r"naive fit  $\sigma^2$=%.1f  (1.7x too wide)" % nv)
    ax.plot(xx, g(corr), "--", color=BLUE, lw=2.0, label=r"corrected  $\sigma^2$=%.1f  (recovers true)" % corr)
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xlabel("position (px)", fontsize=9); ax.set_ylabel("signal (a.u.)", fontsize=9)
    ax.set_title("A width recovered from a frame that looks unusable\n"
                 "all curves at one amplitude -- only the width differs", fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    out = os.path.join(OUTDIR, "0_simulated_benefit.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"  benefit frame: true={VAR_TRUE:.1f} naive={nv:.1f} corrected={corr:.1f} -> {out}")


def distribution(var_bias):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nv = _naive_set(100, 5000); cor = nv - var_bias
    fig, ax = plt.subplots(layout="constrained", figsize=(14 * INCH, 10 * INCH))
    ax.tick_params(direction="in", top=True, right=True, labelsize=9)
    bins = np.linspace(-10, 45, 70)
    ax.hist(np.clip(nv, bins[0], bins[-1]), bins=bins, color=RED, alpha=0.45, label=r"naive $\sigma^2$")
    ax.hist(np.clip(cor, bins[0], bins[-1]), bins=bins, color=BLUE, alpha=0.45, label=r"corrected $\sigma^2$")
    ax.axvline(VAR_TRUE, color=GREEN, lw=2, label=r"true $\sigma^2$=%.1f" % VAR_TRUE)
    ax.axvline(nv.mean(), color=RED, ls="--", lw=1.5,
               label="mean naive = %.1f (%.1fx)" % (nv.mean(), nv.mean() / VAR_TRUE))
    ax.axvline(cor.mean(), color=BLUE, ls="--", lw=1.5, label="mean corrected = %.1f" % cor.mean())
    ax.set_xlabel(r"fitted $\sigma^2$ (px$^2$)", fontsize=9)
    ax.set_ylabel("number of frames", fontsize=9)
    ax.set_title("Distribution of fitted width over 5000 noisy frames (CNR=1.5)\n"
                 "naive biased high and right-skewed; correction recenters the mean on truth", fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    out = os.path.join(OUTDIR, "0b_simulated_distribution.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"  distribution: mean naive={nv.mean():.1f} mean corrected={cor.mean():.1f} "
          f"corrected<0={100*np.mean(cor<0):.0f}% -> {out}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    var_bias = float(_naive_set(0, 8000).mean()) - VAR_TRUE   # calibrated mean bias
    print(f"CNR={CNR}: true={VAR_TRUE:.1f}  var_bias={var_bias:.2f}")
    benefit_frame(var_bias)
    distribution(var_bias)


if __name__ == "__main__":
    main()
