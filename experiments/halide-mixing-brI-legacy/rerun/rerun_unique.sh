#!/usr/bin/env bash
#
# Top up Marc's Br-I re-run to a target number of *unique* random realisations,
# fixing the seed-collision flaw in run_instances.sh.
#
# landscapeBD.diffmap seeds its RNG only from the wall clock at 1-second
# resolution, so the original `sleep $((RANDOM % 5))` stagger produced at most 5
# distinct seeds no matter how many instances were launched: a 16-instance run
# collapsed to 5 unique realisations, each duplicated. Here every instance starts
# in its own clock-second via a deterministic, monotonically increasing pre-launch
# delay (GAP seconds apart), so K launches give K distinct seeds. The instances
# still run concurrently; the stagger offsets only their start, not their ~13 h
# runtime, so the extra wall-clock cost is just (launched-1)*GAP seconds.
#
# Run under WSL2/Linux from a fraction working directory that already holds the
# binary, the lattice files, the (window-widened) parameter file, and the existing
# inst_* directories -- i.e. ~/halide_rerun/x_<frac> after run_all_fractions.sh.
# The parameter file is reused in place (this does not re-stage from the
# simulation folders), so the widened diffmap window is preserved. New instances
# are numbered after the highest existing index; existing directories are left
# untouched, so duplicate instances from a previous run are kept on disk but
# excluded from the unique count and from the averaging list this script writes.
#
# Usage:
#   ./rerun_unique.sh <frac> <N_per_instance> <target_unique> [gap_seconds] [max_parallel]
# Example (top up x_0.25 to 16 unique realisations of 25,000 walkers):
#   ./rerun_unique.sh 0.25 25000 16
# All three fractions:
#   for f in 0.25 0.5 0.75; do \
#       ( cd ~/halide_rerun/x_$f && ~/halide_rerun/rerun_unique.sh "$f" 25000 16 ); \
#   done
#
set -uo pipefail
shopt -s nullglob

FRAC="${1:?usage: rerun_unique.sh <frac> <N_per_instance> <target_unique> [gap_s] [max_parallel]}"
NINST="${2:?N per instance, e.g. 25000}"
TARGET="${3:?target number of UNIQUE instances, e.g. 16}"
GAP="${4:-2}"
JOBS="${5:-$(nproc)}"

BIN="landscapeBD.diffmap"
BASE="parameters.x_${FRAC}.dat"
for f in "$BIN" "basis_BrI.dat" "vectors_BrI.dat" "$BASE"; do
    [ -f "$f" ] || { echo "missing required file: $f"; exit 1; }
done
chmod +x "$BIN"

# Expected variance rows = frames = tmax/tsamp + 1, read from the parameter file.
# An instance only counts once its variance file is complete; this guards against
# partial directories left by an interrupted run being miscounted as realisations.
EXPROWS=$(awk '$1=="tmax"{t=$2} $1=="tsamp"{s=$2} END{if(s>0)printf "%d", t/s+1; else print 0}' "$BASE")
complete () {
    local n
    n=$(wc -l < "$1/variance.x_${FRAC}.dat" 2>/dev/null || echo 0)
    [ "$EXPROWS" -gt 0 ] && [ "$n" -ge "$EXPROWS" ]
}

# A realisation's identity: the per-frame variance is a deterministic function of
# the seed, so equal variance == identical run (verified against the full 2.8 GB
# diffmap). Cheap to hash, unlike the diffmap itself.
vhash () { sha256sum "$1/variance.x_${FRAC}.dat" 2>/dev/null | cut -d' ' -f1; }

count_unique () {
    local d h; declare -A s=()
    for d in inst_*; do
        [ -d "$d" ] || continue
        complete "$d" || continue
        h=$(vhash "$d"); [ -n "$h" ] && s["$h"]=1
    done
    echo "${#s[@]}"
}

# one representative directory per unique hash (lowest index wins, sorted)
unique_reps () {
    local d h; declare -A s=()
    for d in $(printf '%s\n' inst_* | sort); do
        [ -d "$d" ] || continue
        complete "$d" || continue
        h=$(vhash "$d"); [ -n "$h" ] || continue
        if [ -z "${s[$h]:-}" ]; then s["$h"]="$d"; echo "$d"; fi
    done
}

maxidx=0
for d in inst_*; do
    [ -d "$d" ] || continue
    idx=$((10#${d#inst_})); (( idx > maxidx )) && maxidx=$idx
done

run_one () {                 # $1 = instance index, $2 = pre-launch offset (s)
    local idx off d
    idx="$1"; off="$2"
    d=$(printf 'inst_%03d' "$idx")
    rm -rf "$d"; mkdir -p "$d"
    cp "$BIN" 'basis_BrI.dat' 'vectors_BrI.dat' "$d/"
    sed "s/^N[[:space:]]\+[0-9]\+/N       ${NINST}/" "$BASE" > "$d/${BASE}"
    sleep "$off"             # distinct per instance -> distinct clock-second seed
    ( cd "$d" && "./${BIN}" "${BASE}" >run.out 2>run.err )
    echo "  finished $d (start offset ${off}s)"
}

have=$(count_unique)
echo "x_${FRAC}: $have unique realisation(s) present; target $TARGET (gap ${GAP}s, up to $JOBS parallel)  ($(date))"

launch=0
wave=0
while (( have < TARGET )); do
    need=$(( TARGET - have ))
    wave=$(( wave + 1 ))
    echo "wave $wave: launching $need instance(s) ..."
    running=0
    for ((j=0; j<need; j++)); do
        maxidx=$(( maxidx + 1 ))
        launch=$(( launch + 1 ))
        run_one "$maxidx" "$(( launch * GAP ))" &
        running=$(( running + 1 ))
        if (( running >= JOBS )); then wait -n; running=$(( running - 1 )); fi
    done
    wait
    have=$(count_unique)
    echo "wave $wave done: $have unique now."
    if (( wave >= 6 )); then
        echo "stopping after $wave waves without reaching $TARGET; the binary's"
        echo "seed entropy may be coarser than 1 s -- ask Marc for a seed option."
        break
    fi
done

unique_reps > unique_instances.txt
echo "x_${FRAC}: $have unique realisation(s) on disk  ($(date))"
echo "wrote $(wc -l < unique_instances.txt) representative dir(s) to unique_instances.txt"
echo "average the clean unique set with:"
echo "    python3 average_diffmaps.py ${FRAC} \$(cat unique_instances.txt)"
