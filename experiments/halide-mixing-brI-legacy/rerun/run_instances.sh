#!/usr/bin/env bash
#
# Launch K independent instances of Marc's Br-I simulator in parallel, so their
# diffmaps can be averaged into a clean, high-statistics map. Run under WSL2 /
# Linux from a fraction folder holding the binary and inputs (see README.md).
# Verify randomisation (README step 3) first.
#
# landscapeBD.diffmap seeds its random number generator from the wall clock at
# one-second resolution, so instances that start in the same second are copies of
# one realisation. Each instance therefore starts after its own offset, GAP seconds
# apart (instance k waits k * GAP s), so the instances of the first wave start in
# different seconds. With more instances than parallel slots, a later instance
# starts when a slot frees plus its offset, and two could still share a second;
# the distinct count below catches that.
#
# When all instances are done, an instance counts as a distinct realisation only
# if its variance file is complete and differs from every other one (by SHA-256,
# as the top-up scripts count). The distinct instances are written to
# unique_instances.txt, one per line, and only they should be averaged; if there
# are fewer than K, top up with rerun_unique.sh (README step 5).
#
# Usage:
#   ./run_instances.sh <frac> <N_per_instance> <K_instances> [max_parallel] [gap_seconds]
# Example (20 instances of 20,000 walkers, all cores, starts 2 s apart):
#   ./run_instances.sh 0.25 20000 20
#
set -euo pipefail

FRAC="${1:?usage: run_instances.sh <frac e.g. 0.25> <N_per_instance> <K> [max_parallel] [gap_s]}"
NINST="${2:?N per instance, e.g. 20000}"
K="${3:?number of instances, e.g. 20}"
JOBS="${4:-$(nproc)}"
GAP="${5:-2}"

BIN="landscapeBD.diffmap"
BASE="parameters.x_${FRAC}.dat"
for f in "$BIN" "basis_BrI.dat" "vectors_BrI.dat" "$BASE"; do
    [ -f "$f" ] || { echo "missing required file: $f (copy it from halide_mixing/BrI_x_${FRAC}/ in the data root; see README.md)"; exit 1; }
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
    sleep $(( k * GAP ))     # its own start second -> its own clock-second seed
    ( cd "$d" && "./${BIN}" "${BASE}" >run.out 2>run.err )
    echo "  finished $d"
}
export -f run_one
export BIN BASE NINST GAP

echo "Launching $K instance(s) of N=$NINST (frac $FRAC), up to $JOBS in parallel, starts ${GAP} s apart..."
seq 1 "$K" | xargs -I{} -P "$JOBS" bash -c 'run_one "$@"' _ {}

# A complete variance file has tmax / tsamp + 1 rows; equal files mean one seed.
EXPROWS=$(awk '$1=="tmax"{t=$2} $1=="tsamp"{s=$2} END{if(s>0)printf "%d", t/s+1; else print 0}' "$BASE")
: > unique_instances.txt
declare -A seen=()
incomplete=0
for d in $(printf 'inst_%03d\n' $(seq 1 "$K")); do
    n=$(wc -l 2>/dev/null < "$d/variance.x_${FRAC}.dat" || echo 0)
    if [ "$EXPROWS" -le 0 ] || [ "$n" -lt "$EXPROWS" ]; then
        incomplete=$(( incomplete + 1 )); continue
    fi
    h=$(sha256sum "$d/variance.x_${FRAC}.dat" | cut -d' ' -f1)
    if [ -z "${seen[$h]:-}" ]; then seen["$h"]="$d"; echo "$d" >> unique_instances.txt; fi
done
distinct=$(wc -l < unique_instances.txt)
echo "All instances done: $distinct distinct realisation(s) of $K; $incomplete incomplete (see their run.err)."
echo "Distinct instances written to unique_instances.txt. Average them with:"
echo "    python3 average_diffmaps.py $FRAC \$(cat unique_instances.txt)"
if [ "$distinct" -lt "$K" ]; then
    echo "Fewer than $K distinct; top up with rerun_unique.sh (README step 5)."
fi
