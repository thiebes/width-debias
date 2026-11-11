#!/usr/bin/env bash
# Rebuild the NC-solid Gaussian (CNR, sigma^2) calibration at high statistics.
#
# The poster figure's noise-bias correction reads var_bias from
# output/calibration_nc_solid_gaussian.npz, a Monte Carlo lookup on a
# (CNR, sigma^2) grid. The committed calibration is 1000 runs/cell, whose MC
# noise imprints a reproducible wiggle on the corrected-alpha series. This
# rebuild raises it to 8000 runs/cell (~2.8x less grid noise) and closes two
# coverage gaps (CNR 100 -> 250; sigma-max 0.28 -> 0.35). ~96M small fits.
#
# Run on a multi-core Linux machine (or WSL).
#   PYTHON=.venv-linux/bin/python bash experiments/nc-solid/run_nc_solid_calibration.sh
# Override the worker count with WORKERS=N (default: all logical cores).
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root (this script lives in experiments/<set>/)

# The fits are tiny nonlinear least-squares; BLAS threading does not help and
# one BLAS thread per worker times all cores oversubscribes and can exhaust
# memory. Pin to one thread per process.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PY="${PYTHON:-python3}"
WORKERS="${WORKERS:-$(nproc)}"

# --filter-threshold 0.6125 (5 * sigma_max^2) is the filter this preset ran
# with, so a rebuild reproduces its grid; drop it for a grid that keeps every
# fit in the per-cell bias mean, as calibrate_cnr_sigma2 does by default.
echo "Building NC-solid calibration: $WORKERS workers, python = $PY"
"$PY" scripts/build_calibration.py \
  --tag nc_solid_gaussian \
  --n-pixels 250 --half-window 0.75 \
  --sigma-min 0.15 --sigma-max 0.35 --n-sigma 100 \
  --cnr-min 1 --cnr-max 250 --n-cnr 120 \
  --n-runs 8000 \
  --model gaussian \
  --filter-threshold 0.6125 \
  --n-workers "$WORKERS"

echo
echo "Done. Output: output/calibration_nc_solid_gaussian.npz"
echo "Copy that file back to the analysis machine's output/ directory."
