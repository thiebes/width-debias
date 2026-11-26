#!/usr/bin/env bash
#
# Global-pool top-up to TARGET unique realisations per Br fraction, packing all
# cores with no per-fraction idle tail. Replaces the seed-collision behaviour of
# run_instances.sh (its `sleep $((RANDOM % 5))` stagger gave at most 5 distinct
# clock-second seeds, so a 16-instance run collapsed to 5 unique realisations).
#
# landscapeBD.diffmap is single-threaded (links only libc/libm) and seeds its RNG
# from the wall clock at 1-second resolution. Here every launched instance starts
# in its own clock-second via a deterministic, global, monotonic pre-launch delay
# (GAP seconds apart), so each realisation gets a distinct seed.
#
# Scheduling: the run box is an AMD Ryzen 7 5700X (8 physical cores, 16 SMT
# threads, ~15 GB to WSL, ~0.9 GB per instance, so cores are the limit). For this
# CPU-bound work, saturating all 16 logical threads maximises aggregate
# throughput, so the default parallelism is 16 -- the proven-safe width from the
# original run. All three fractions share one pool, so cores stay full across
# fraction boundaries.
#
# Reuses each fraction's existing (window-widened) parameter file in place, so new
# instances match the existing ones exactly (substrate 250x250 unit cells,
# N walkers, 600x600 diffmap over a 3000 nm window, dt/tsamp/tmax). Existing
# complete instances are kept and counted; only the shortfall to TARGET is run.
# An instance counts only once its variance file has the full frame count, so
# partial directories from an interrupted run are ignored, not miscounted.
#
# Usage:
#   ./rerun_all_unique.sh [N_per_instance] [target_unique] [gap_s] [parallel]
# Defaults: 25000 16 2 16   (-> 11 new instances per fraction, ~39 h total)
#
# Launch detached so it survives the shell:
#   setsid nohup ./rerun_all_unique.sh > ~/halide_rerun/rerun_all_unique.log 2>&1 &
#   tail -f ~/halide_rerun/rerun_all_unique.log
#
set -uo pipefail
shopt -s nullglob

NINST="${1:-25000}"
TARGET="${2:-16}"
GAP="${3:-2}"
PAR="${4:-16}"
ROOT="${ROOT:-$HOME/halide_rerun}"
FRACS=(0.25 0.5 0.75)

# expected variance rows = frames = tmax/tsamp + 1, from a fraction's param file
exprows () { awk '$1=="tmax"{t=$2} $1=="tsamp"{s=$2} END{if(s>0)printf "%d",t/s+1; else print 0}' "$1"; }
vhash ()   { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

# build the global job list: "<frac> <new_index>" for each instance still needed
JOBS=()
for f in "${FRACS[@]}"; do
    wd="$ROOT/x_$f"
    base="$wd/parameters.x_$f.dat"
    [ -f "$base" ] || { echo "missing $base -- skipping x_$f"; continue; }
    er=$(exprows "$base")
    declare -A seen=()
    maxidx=0
    for d in "$wd"/inst_*; do
        [ -d "$d" ] || continue
        idx=$((10#$(basename "$d" | sed 's/inst_//')))
        (( idx > maxidx )) && maxidx=$idx
        n=$(wc -l < "$d/variance.x_$f.dat" 2>/dev/null || echo 0)
        if [ "$er" -gt 0 ] && [ "$n" -ge "$er" ]; then
            h=$(vhash "$d/variance.x_$f.dat"); [ -n "$h" ] && seen["$h"]=1
        fi
    done
    have=${#seen[@]}
    need=$(( TARGET - have ))
    (( need < 0 )) && need=0
    echo "x_$f: $have unique present, need $need more (next index inst_$(printf '%03d' $((maxidx+1))))"
    for ((j=0; j<need; j++)); do
        maxidx=$(( maxidx + 1 ))
        JOBS+=("$f $maxidx")
    done
    unset seen
done

total=${#JOBS[@]}
echo "global pool: $total instance(s) of N=$NINST, parallelism $PAR, gap ${GAP}s  ($(date))"
[ "$total" -eq 0 ] && { echo "all fractions already at $TARGET unique; nothing to run"; }

# one instance: stage a clean dir, override N, stagger the start, run the binary
run_job () {
    local f idx off d wd base
    f="$1"; idx="$2"; off="$3"
    wd="$ROOT/x_$f"; base="parameters.x_$f.dat"
    d=$(printf 'inst_%03d' "$idx")
    cd "$wd" || return 1
    rm -rf "$d"; mkdir -p "$d"
    cp landscapeBD.diffmap basis_BrI.dat vectors_BrI.dat "$d/"
    chmod +x "$d/landscapeBD.diffmap"
    sed "s/^N[[:space:]]\+[0-9]\+/N       ${NINST}/" "$base" > "$d/$base"
    sleep "$off"          # distinct per instance -> distinct clock-second seed
    ( cd "$d" && ./landscapeBD.diffmap "$base" >run.out 2>run.err )
    echo "  done x_$f/$d (start offset ${off}s)"
}
export -f run_job
export ROOT NINST

# assign each job a global, monotonic start offset, then run the pool
if [ "$total" -gt 0 ]; then
    seq=0
    for entry in "${JOBS[@]}"; do
        seq=$(( seq + 1 ))
        printf '%s %d\n' "$entry" "$(( seq * GAP ))"
    done | xargs -L1 -P "$PAR" bash -c 'run_job "$@"' _
    echo "pool finished ($(date))"
fi

# de-duplicate each fraction and re-average the clean unique set
echo "=== de-duplicate and re-average ==="
for f in "${FRACS[@]}"; do
    wd="$ROOT/x_$f"
    base="$wd/parameters.x_$f.dat"
    [ -f "$base" ] || continue
    er=$(exprows "$base")
    declare -A seen=()
    : > "$wd/unique_instances.txt"
    for d in $(printf '%s\n' "$wd"/inst_* | sort); do
        [ -d "$d" ] || continue
        n=$(wc -l < "$d/variance.x_$f.dat" 2>/dev/null || echo 0)
        [ "$er" -gt 0 ] && [ "$n" -ge "$er" ] || continue
        h=$(vhash "$d/variance.x_$f.dat"); [ -n "$h" ] || continue
        if [ -z "${seen[$h]:-}" ]; then seen["$h"]="$d"; basename "$d" >> "$wd/unique_instances.txt"; fi
    done
    u=$(wc -l < "$wd/unique_instances.txt")
    echo "x_$f: $u unique -> $(tr '\n' ' ' < "$wd/unique_instances.txt")"
    if command -v python3 >/dev/null && python3 -c 'import numpy' 2>/dev/null; then
        ( cd "$wd" && python3 average_diffmaps.py "$f" $(cat unique_instances.txt) )
    else
        echo "  numpy missing; average later: cd $wd && python3 average_diffmaps.py $f \$(cat unique_instances.txt)"
    fi
    unset seen
done
echo "TOP-UP COMPLETE ($(date))"
