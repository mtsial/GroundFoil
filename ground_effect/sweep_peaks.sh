#!/bin/bash
# Ride-height sweeps of six test airfoils, which no model has been trained on, to check where the
# force networks put the peak of the downforce (ride_height_map.py). Each airfoil at 2, 5 and
# 8 deg and 30 m/s (Re = 3 million), at the 12 ride heights of sweep_naca4412.sh (h/c 0.05 to
# 0.6, spaced evenly on a log scale); then two of them at 5 deg and 10 and 50 m/s (Re = 1 and 5
# million), for the effect of the Reynolds number. The dataset's settings otherwise (dataset.sh).
# The airfoils span the thickness, at little and at much camber (ride_height_map.py's numbers):
#     little camber: ag09 (thickness 4.9%, camber 1.9%), n64012a (12%, symmetric), goe775 (21%, 0.2%)
#     much camber:   goe265 (7.7%, camber 7.3%), goe07k (11%, 4.6%), goe626 (16.7%, 5.1%)
# ag09 is one where the networks see the downforce still rising at h/c = 0.05, goe07k one where
# they put the peak closer to the ground as the incidence rises (most airfoils: the other way).
# Run inside the Docker container:
#     cd /work/ground_effect
#     bash sweep_peaks.sh
#
# 264 runs, JOBS at a time (default 4): about 5.5 hours for the 216 runs at 30 m/s, which come
# first, then about 1 hour for the other speeds. Each run has its sample in
# $OUT/<airfoil>_a<alpha>[_U<speed>]/h<height>/ (not_converged/ in it if Cl does not become
# steady) and its output in $LOGS/<sweep>_h<height>.log. The runs share JOBS work folders
# (work/worker500 to 503 by default), one run in each at a time. Runs that already have a sample
# are skipped, so after an interruption the same command goes on where it stopped. To stop:
# Ctrl+C stops this script but not the runs already started, so then type exit to close the
# container, which stops them (their samples are lost, the saved ones stay).
# Then, on Windows, ride_height_map.py --sweeps data\sweep_peaks compares the sweeps with the networks.
set -u
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
WORKER=${WORKER:-500}
OUT=${OUT:-data/sweep_peaks}
AIRFOIL_DIR=${AIRFOIL_DIR:-../data/airfoil_database_test_smooth}
LOGS=logs/$(basename "$OUT")

AIRFOILS="ag09 n64012a goe775 goe265 goe07k goe626"
ALPHAS="8 5 2"  # whole degrees (they name the folders); the highest first: their runs take longest
U_MAIN=30
RE_AIRFOILS="goe265 goe775"
RE_SPEEDS="10 50"  # at 5 deg
HEIGHTS="0.050 0.063 0.079 0.098 0.123 0.155 0.194 0.243 0.305 0.382 0.479 0.600"  # three decimals each

for f in $AIRFOILS $RE_AIRFOILS; do
    if [ ! -f "$AIRFOIL_DIR/$f.dat" ]; then
        echo "missing $AIRFOIL_DIR/$f.dat: the smoothed test airfoils (smooth_airfoils.py, see the README)"
        exit 1
    fi
done
mkdir -p "$LOGS"

# sweep | airfoil | alpha | speed | h/c, one line per run: the lowest heights (the longest runs)
# first, the sweeps at 30 m/s before the other speeds
JOBLIST=$(
    for h in $HEIGHTS; do
        for a in $ALPHAS; do
            for f in $AIRFOILS; do
                printf "%s_a%02d|%s|%s|%s|%s\n" "$f" "$a" "$f" "$a" "$U_MAIN" "$h"
            done
        done
    done
    for h in $HEIGHTS; do
        for U in $RE_SPEEDS; do
            for f in $RE_AIRFOILS; do
                printf "%s_a05_U%s|%s|5|%s|%s\n" "$f" "$U" "$f" "$U" "$h"
            done
        done
    done
)

run_one() {  # airfoil, alpha, speed, h/c, worker, run folder, log
    python3 groundGen.py --airfoil "$AIRFOIL_DIR/$1.dat" --alpha "$2" --U "$3" --h "$4" \
        --iterations 1000 --max-iterations 3000 --worker "$5" --out "$6" > "$7" 2>&1
    echo "[$(date +%H:%M)] finished $(basename "$(dirname "$6")") $(basename "$6"):" \
         "$(grep -m1 -E 'done in|skipping' "$7" | sed 's/^ *//')" \
         "$(grep -m1 -E '^    (converged|NOT converged)' "$7" | sed 's/^ *//' | cut -c1-60)"
}

declare -a PIDS  # the run in each work folder (0: none yet)
for ((s = 0; s < JOBS; s++)); do
    PIDS[s]=0
done
echo "[$(date +%H:%M)] peak sweeps: $(grep -c '|' <<< "$JOBLIST") runs, $JOBS at a time" \
     "(work folders worker$WORKER to worker$((WORKER + JOBS - 1)))"
while IFS='|' read -r sweep f a U h; do
    [ -z "$sweep" ] && continue
    run="h0${h#0.}"  # e.g. 0.050 -> h0050
    dir="$OUT/$sweep/$run"
    if compgen -G "$dir/*.npz" > /dev/null || compgen -G "$dir/not_converged/*.npz" > /dev/null; then
        echo "[$(date +%H:%M)] $sweep $run has a sample already: skipped"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
        wait -n  # wait until one of the running runs has finished
    done
    running=" $(jobs -rp | tr '\n' ' ') "
    slot=0  # the first work folder whose run has finished
    while [[ "$running" == *" ${PIDS[slot]} "* ]]; do
        slot=$((slot + 1))
    done
    echo "[$(date +%H:%M)] start $sweep $run (worker $((WORKER + slot)))"
    run_one "$f" "$a" "$U" "$h" $((WORKER + slot)) "$dir" "$LOGS/${sweep}_$run.log" &
    PIDS[slot]=$!
done <<< "$JOBLIST"
wait
echo "=== peak sweeps finished: $(ls "$OUT"/*/*/*.npz 2>/dev/null | wc -l) of $(grep -c '|' <<< "$JOBLIST")" \
     "runs gave a sample (Cl steady) ==="
