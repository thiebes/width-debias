"""Run the full CsPbBr3 correction pipeline across all fluences.

Stages, for every fluence subfolder under ``--sample-root``:

1. **reduce**    -- raw CSVs -> per-frame profiles, variance, CNR (reduce.py).
2. **calibrate** -- build the (CNR, sigma^2) bias lookup. For the gaussian model
   the lookup depends only on the pixel grid and the CNR/sigma axes, not on the
   fluence, so fluences that share a pixel grid share one calibration spanning
   their combined CNR/sigma range (no redundant 96M-fit runs). Each calibration's
   geometry and axes are derived from the fresh reductions.
3. **apply**     -- subtract the per-frame bias and fit corrected-vs-naive D.
4. **summary**   -- collect D(fluence) for every scheme into a CSV and a plot.

Heavy step is `calibrate`; it honours WIDTH_DEBIAS_WORKERS via `--workers`. Pick stages
with `--stages` to resume (e.g. `--stages apply,summary` after a calibration run).

Example::

    python experiments/cspbbr-strobescat/run_all.py \
        --out-dir output/cspbbr --workers 16 --n-runs 8000

``--sample-root`` defaults to the ``cspbbr_sample_1/`` folder of the data root
(``scripts/collaborator_data.py``); the run machine must be able to read it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from collaborator_data import cspbbr_sample_1_dir  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = sys.executable
REDUCE = str(HERE / "reduce.py")
APPLY = str(HERE / "apply_correction.py")
BUILD = str(REPO / "scripts" / "build_calibration.py")


def label_and_fluence(folder_name: str):
    """A filesystem-safe label and the fluence (uJ) parsed from a folder name."""
    m = re.match(r"\s*([0-9]+\.?[0-9]*)", folder_name)
    fluence = float(m.group(1)) if m else float("nan")
    label = re.sub(r"[^0-9A-Za-z]+", "_", folder_name).strip("_")
    return label, fluence


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample-root", default=str(cspbbr_sample_1_dir()),
                   help="folder containing the per-fluence subfolders (default: "
                        "cspbbr_sample_1/ in the data root)")
    p.add_argument("--pattern", default="*uJ*",
                   help="glob for fluence subfolders (default '*uJ*')")
    p.add_argument("--out-dir", default="output/cspbbr")
    p.add_argument("--workers", type=int, default=0,
                   help="calibration worker processes (0 = all cores)")
    p.add_argument("--n-files", type=int, default=3,
                   help="repeat acquisitions to merge per fluence")
    p.add_argument("--n-runs", type=int, default=8000)
    p.add_argument("--n-cnr", type=int, default=120)
    p.add_argument("--n-sigma", type=int, default=100)
    p.add_argument("--max-var", type=float, default=6.25)
    p.add_argument("--stages", default="reduce,calibrate,apply,summary",
                   help="comma list of stages to run")
    args = p.parse_args()

    stages = set(s.strip() for s in args.stages.split(","))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if not os.path.isdir(args.sample_root):
        raise SystemExit(f"No sample root at {args.sample_root} (WIDTH_DEBIAS_DATA_ROOT, or else data/). See DATASETS.md.")
    folders = sorted(glob.glob(os.path.join(args.sample_root, args.pattern)))
    folders = [f for f in folders if os.path.isdir(f)]
    if not folders:
        raise SystemExit(f"no fluence folders matching {args.pattern!r} in {args.sample_root}")
    fluences = [(f, *label_and_fluence(os.path.basename(f.rstrip("/\\"))))
                for f in folders]
    print(f"{len(fluences)} fluences: "
          + ", ".join(f"{lab}({fl:g})" for _, lab, fl in fluences))

    # 1. reduce -----------------------------------------------------------
    reduced = {}
    for folder, label, _ in fluences:
        npz = out / f"{label}.npz"
        if "reduce" in stages:
            print(f"\n[reduce] {label}")
            run([PY, REDUCE, "--data", folder, "--n-files", str(args.n_files),
                 "--max-var", str(args.max_var), "--out", str(npz), "--no-plot"])
        reduced[label] = npz

    # geometry table + group by pixel grid --------------------------------
    geom = {}
    print("\n[geometry]")
    for _, label, fl in fluences:
        d = np.load(reduced[label])
        npix = int(d["profiles"].shape[1])
        hw = float(d["half_window_um"][0])
        cnr = d["cnr"]; sig = np.sqrt(np.clip(d["variance_um2"], 0, None))
        g = dict(npix=npix, hw=hw, cnr=(float(np.nanmin(cnr)), float(np.nanmax(cnr))),
                 sig=(float(np.nanmin(sig)), float(np.nanmax(sig))), fluence=fl)
        geom[label] = g
        print(f"  {label:24s} {npix}px hw={hw:.3f}um "
              f"CNR {g['cnr'][0]:.2f}-{g['cnr'][1]:.1f} sigma {g['sig'][0]:.2f}-{g['sig'][1]:.2f}")

    groups = {}
    for label, g in geom.items():
        groups.setdefault((g["npix"], round(g["hw"], 2)), []).append(label)
    print(f"\n{len(groups)} unique pixel grid(s) -> {len(groups)} calibration(s)")

    # 2. calibrate (one per unique grid, spanning the group's ranges) ------
    grid_calib = {}
    for (npix, hw), labels in groups.items():
        tag = f"cspbbr_{npix}px_{hw:.2f}"
        # build_calibration.py always writes to the repo's output/ directory.
        calib = REPO / "output" / f"calibration_{tag}.npz"
        grid_calib[(npix, hw)] = calib
        cnr_lo = max(0.4, min(geom[l]["cnr"][0] for l in labels) * 0.8)
        cnr_hi = max(geom[l]["cnr"][1] for l in labels) * 1.2
        sig_lo = max(0.1, min(geom[l]["sig"][0] for l in labels) * 0.8)
        sig_hi = max(geom[l]["sig"][1] for l in labels) * 1.15
        if "calibrate" in stages:
            print(f"\n[calibrate] grid {tag}  ({len(labels)} fluences)")
            run([PY, BUILD, "--tag", tag, "--n-pixels", str(npix),
                 "--half-window", f"{hw:.4f}",
                 "--sigma-min", f"{sig_lo:.4f}", "--sigma-max", f"{sig_hi:.4f}",
                 "--n-sigma", str(args.n_sigma),
                 "--cnr-min", f"{cnr_lo:.4f}", "--cnr-max", f"{cnr_hi:.4f}",
                 "--n-cnr", str(args.n_cnr), "--n-runs", str(args.n_runs),
                 "--model", "gaussian", "--max-var", str(args.max_var),
                 "--n-workers", str(args.workers)])

    # 3. apply ------------------------------------------------------------
    for _, label, _ in fluences:
        g = geom[label]
        calib = grid_calib[(g["npix"], round(g["hw"], 2))]
        if "apply" in stages:
            print(f"\n[apply] {label}")
            run([PY, APPLY, "--reduced", str(reduced[label]),
                 "--calibration", str(calib),
                 "--json", str(out / f"{label}_result.json"), "--no-plot"])

    # 4. summary ----------------------------------------------------------
    if "summary" in stages:
        rows = []
        for _, label, fl in fluences:
            rj = out / f"{label}_result.json"
            if not rj.exists():
                continue
            r = json.load(open(rj))
            rows.append((fl, label, r))
        rows.sort(key=lambda x: (np.isnan(x[0]), x[0]))
        schemes = ["ols", "inverse_variance", "subtract", "mas"]
        csv = out / "cspbbr_D_vs_fluence.csv"
        with open(csv, "w") as fh:
            fh.write("fluence_uJ,label," + ",".join(
                f"{s}_D_cm2s,{s}_se" for s in schemes) + "\n")
            for fl, label, r in rows:
                fh.write(f"{fl},{label}," + ",".join(
                    f"{r[s]['D_cm2s']:.5f},{r[s]['D_se_cm2s']:.5f}"
                    for s in schemes) + "\n")
        print(f"\nwrote {csv}")

        try:
            import matplotlib.pyplot as plt
            x = [fl for fl, _, _ in rows]
            fig, ax = plt.subplots(figsize=(6, 5))
            for s, c in (("inverse_variance", "0.5"), ("subtract", "C0")):
                y = [r[s]["D_cm2s"] for _, _, r in rows]
                e = [r[s]["D_se_cm2s"] for _, _, r in rows]
                ax.errorbar(x, y, yerr=e, fmt="o-", color=c,
                            label="naive" if s == "inverse_variance" else "corrected")
            ax.set_xlabel("fluence (uJ)"); ax.set_ylabel("D (cm^2/s)")
            ax.set_xscale("log"); ax.legend()
            fig.tight_layout()
            fig.savefig(out / "cspbbr_D_vs_fluence.png", dpi=120)
            print(f"wrote {out / 'cspbbr_D_vs_fluence.png'}")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
