#!/usr/bin/env bash
#
# Launch K independent instances of Marc's Br-I simulator in parallel, so their
# diffmaps can be averaged into a clean, high-statistics map. Run under WSL2 /
# Linux from a fraction folder holding the binary and inputs (see README.md),
# with no inst_001 .. inst_K folders yet; to add instances to an existing set, use
# rerun_unique.sh. Verify randomisation (README step 3) first.
#
# landscapeBD.diffmap seeds its random number generator from the wall clock at
# one-second resolution, so instances that start in the same second are copies of
# one realisation. Instance k therefore waits k * GAP seconds before it starts.
# xargs starts the instances in order, so in any wave instance k starts at least
# (k - j) * GAP seconds after instance j < k. Only the time each instance takes to
# copy its files and start can still put two in one second; the distinct count
# below catches that.
#
# When all instances are done, the script keeps one instance per distinct
# complete variance file (tmax / tsamp + 1 rows), the lowest index among copies,
# compared by SHA-256 as the top-up scripts compare them. It writes those
# instances to unique_instances.txt, one per line; average only them. If there are
# fewer than K, it prints the rerun_unique.sh command that tops the set up.
#
# Usage (the gap comes after max_parallel, so a call with four arguments keeps
# its meaning; rerun_unique.sh takes them in the other order):
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

[[ "$GAP" =~ ^[1-9][0-9]*$ ]] || { echo "gap_seconds must be a whole number of seconds, 1 or more (got '$GAP')"; exit 1; }

BIN="landscapeBD.diffmap"
BASE="parameters.x_${FRAC}.dat"
for f in "$BIN" "basis_BrI.dat" "vectors_BrI.dat" "$BASE"; do
    [ -f "$f" ] || { echo "missing required file: $f (copy it from halide_mixing/BrI_x_${FRAC}/ in the data root; see README.md)"; exit 1; }
done

# A complete variance file has tmax / tsamp + 1 rows; equal files mean one seed.
EXPROWS=$(awk '$1=="tmax"{t=$2} $1=="tsamp"{s=$2} END{if(s>0)printf "%d", t/s+1; else print 0}' "$BASE")
[ "$EXPROWS" -gt 0 ] || { echo "$BASE gives no tmax and tsamp, so a complete variance file cannot be recognised"; exit 1; }

# Refuse to reuse instance folders: a variance file left by an earlier run would
# be counted as this run's realisation.
for d in $(printf 'inst_%03d\n' $(seq 1 "$K")); do
    [ ! -e "$d" ] || { echo "$d already exists; run in a fraction folder without inst_001 .. inst_$(printf '%03d' "$K"), or add instances with rerun_unique.sh"; exit 1; }
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
    echo "  finished $d (start offset $(( k * GAP )) s)"
}
export -f run_one
export BIN BASE NINST GAP

echo "Launching $K instance(s) of N=$NINST (frac $FRAC), up to $JOBS in parallel, starts ${GAP} s apart..."
seq 1 "$K" | xargs -I{} -P "$JOBS" bash -c 'run_one "$@"' _ {}

: > unique_instances.txt
declare -A seen=()
incomplete=()
for d in $(printf 'inst_%03d\n' $(seq 1 "$K")); do
    n=$(wc -l 2>/dev/null < "$d/variance.x_${FRAC}.dat" || echo 0)
    if [ "$n" -lt "$EXPROWS" ]; then incomplete+=("$d"); continue; fi
    h=$(sha256sum "$d/variance.x_${FRAC}.dat" | cut -d' ' -f1)
    if [ -z "${seen[$h]:-}" ]; then seen["$h"]="$d"; echo "$d" >> unique_instances.txt; fi
done
distinct=$(wc -l < unique_instances.txt)
echo "All instances done: $distinct distinct realisation(s) of $K, written to unique_instances.txt."
if [ "${#incomplete[@]}" -gt 0 ]; then
    echo "Incomplete (variance file short of $EXPROWS rows; see their run.err): ${incomplete[*]}"
fi
if [ "$distinct" -lt "$K" ]; then
    echo "Fewer than $K distinct. Top up to $K (it rewrites unique_instances.txt and prints the average command):"
    echo "    ./rerun_unique.sh $FRAC $NINST $K"
fi
if [ "$distinct" -eq "$K" ]; then
    echo "Average them with:"
    echo "    python3 average_diffmaps.py $FRAC \$(cat unique_instances.txt)"
elif [ "$distinct" -gt 0 ]; then
    echo "Or average the $distinct distinct instance(s) as they are with:"
    echo "    python3 average_diffmaps.py $FRAC \$(cat unique_instances.txt)"
fi
