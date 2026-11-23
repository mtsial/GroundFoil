#!/bin/bash
# A check of the one irregular sweep of sweep_peaks.sh: goe775 at 5 deg and 10 m/s (Re = 1 million),
# whose downforce jumps by up to 15% from one ride height to the next although every run is
# steady. Two kinds of runs, with the dataset's settings otherwise:
#   fine/     the heights of the jumps (h/c 0.155-0.382) on the fine mesh of the mesh study
#             (--refine 1.4142, cells sqrt(2) smaller): if the jumps go, the default mesh caused them;
#   default/  heights between them (h/c 0.170-0.220) on the default mesh: if the downforce flips
#             between levels from one height to the next, the steady solution is not unique there.
# Run inside the Docker container:
#     cd /work/ground_effect
#     bash sweep_goe775_check.sh
# 10 runs, JOBS at a time (default 4), about 25 minutes; samples in $OUT/<mesh>/h<height>/, output in
# $LOGS/<mesh>_h<height>.log, work folders worker510-513. Runs that have a sample are skipped.
# Then, on Windows: ride_height_map.py --sweeps data\sweep_goe775_check compares them.
set -u
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
WORKER=${WORKER:-510}
OUT=${OUT:-data/sweep_goe775_check}
AIRFOIL=${AIRFOIL:-../data/airfoil_database_test_smooth/goe775.dat}
LOGS=logs/$(basename "$OUT")
if [ ! -f "$AIRFOIL" ]; then
    echo "missing $AIRFOIL"
    exit 1
fi
mkdir -p "$LOGS"

# mesh | h/c | options: the fine-mesh runs (the longest) first
JOBLIST="
fine|0.155|--refine 1.4142
fine|0.194|--refine 1.4142
fine|0.243|--refine 1.4142
fine|0.305|--refine 1.4142
fine|0.382|--refine 1.4142
default|0.170|
default|0.180|
default|0.200|
default|0.210|
default|0.220|
"

run_one() {  # h/c, worker, run folder, log, then the options
    local h=$1 worker=$2 dir=$3 log=$4
    shift 4
    python3 groundGen.py --airfoil "$AIRFOIL" --alpha 5 --U 10 --h "$h" --iterations 1000 --max-iterations 3000 \
        "$@" --worker "$worker" --out "$dir" > "$log" 2>&1
    echo "[$(date +%H:%M)] finished $(basename "$(dirname "$dir")") $(basename "$dir"):" \
         "$(grep -m1 -E 'Cl = ' "$log" | sed 's/^ *//' | cut -c1-40)" \
         "$(grep -m1 -E '^    (converged|NOT converged)' "$log" | sed 's/^ *//' | cut -c1-40)"
}

declare -a PIDS
for ((s = 0; s < JOBS; s++)); do
    PIDS[s]=0
done
echo "[$(date +%H:%M)] goe775 check: $(grep -c '|' <<< "$JOBLIST") runs, $JOBS at a time"
while IFS='|' read -r mesh h opts; do
    [ -z "$mesh" ] && continue
    run="h0${h#0.}"
    dir="$OUT/$mesh/$run"
    if compgen -G "$dir/*.npz" > /dev/null || compgen -G "$dir/not_converged/*.npz" > /dev/null; then
        echo "[$(date +%H:%M)] $mesh $run has a sample already: skipped"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
        wait -n
    done
    running=" $(jobs -rp | tr '\n' ' ') "
    slot=0
    while [[ "$running" == *" ${PIDS[slot]} "* ]]; do
        slot=$((slot + 1))
    done
    echo "[$(date +%H:%M)] start $mesh $run (worker $((WORKER + slot)))"
    run_one "$h" $((WORKER + slot)) "$dir" "$LOGS/${mesh}_$run.log" $opts &
    PIDS[slot]=$!
done <<< "$JOBLIST"
wait
echo "=== goe775 check finished: $(ls "$OUT"/*/*/*.npz 2>/dev/null | wc -l) of $(grep -c '|' <<< "$JOBLIST") runs steady ==="
