#!/usr/bin/env bash
#
# Launch K independent instances of Marc's Br-I simulator in parallel, each a
# fresh random realisation, so their diffmaps can be averaged into a clean,
# high-statistics map. Run under WSL2 / Linux from a directory holding the
# binary and inputs (see README.md). Verify randomisation (README step 3) first.
#
# Usage:
#   ./run_instances.sh <frac> <N_per_instance> <K_instances> [max_parallel]
# Example (20 instances of 20,000 walkers, all cores):
#   ./run_instances.sh 0.25 20000 20
#
set -euo pipefail

FRAC="${1:?usage: run_instances.sh <frac e.g. 0.25> <N_per_instance> <K> [max_parallel]}"
NINST="${2:?N per instance, e.g. 20000}"
K="${3:?number of instances, e.g. 20}"
JOBS="${4:-$(nproc)}"

BIN="landscapeBD.diffmap"
BASE="parameters.x_${FRAC}.dat"
for f in "$BIN" "basis_BrI.dat" "vectors_BrI.dat" "$BASE"; do
    [ -f "$f" ] || { echo "missing required file: $f (copy it from the Br-I simulation folder; see README.md)"; exit 1; }
done
chmod +x "$BIN"

run_one () {
    local k d
    k="$1"
    d=$(printf "inst_%03d" "$k")
    mkdir -p "$d"
    cp "$BIN" "basis_BrI.dat" "vectors_BrI.dat" "$d/"
    # per-instance parameter file with N overridden; comment after N is preserved
    sed "s/^N[[:space:]]\+[0-9]\+/N       ${NINST}/" "$BASE" > "$d/${BASE}"
    # small random stagger so a time-seeded RNG differs between instances
    sleep $(( RANDOM % 5 ))
    ( cd "$d" && "./${BIN}" "${BASE}" >run.out 2>run.err )
    echo "  finished $d"
}
export -f run_one
export BIN BASE

echo "Launching $K instance(s) of N=$NINST (frac $FRAC), up to $JOBS in parallel..."
seq 1 "$K" | xargs -I{} -P "$JOBS" bash -c 'run_one "$@"' _ {}
echo "All instances done. Average with:"
echo "    python3 average_diffmaps.py $FRAC inst_*"
