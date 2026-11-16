#!/bin/bash
# Ride-height sweep of the NACA 4412, an airfoil the U-Net has never seen (it is neither among
# the training airfoils nor among the test airfoils; groundGen.py builds it from its formula).
# Upside down at 5 deg and 30 m/s (Re = 3 million), the middle of the training ranges, at 12
# ride heights from h/c 0.05 to 0.6 (spaced evenly on a log scale, as the training heights are
# drawn) and in free air, with the dataset's settings (dataset.sh). The samples are then
# compared with the U-Net's predictions (Cl against h/c, and the fields). Run inside the
# Docker container:
#     cd /work/ground_effect
#     bash sweep_naca4412.sh
#
# The 13 runs are independent, so they run JOBS at a time (default 4), each in its own
# work/workerN folder (400-412), with its sample in $OUT/<name>/ (not_converged/ in it if Cl
# does not become steady) and its output in $LOGS/<name>.log. Runs that already have a sample
# are skipped, so after an interruption the same command goes on where it stopped.
# About 20 minutes with 4 at a time. Then, on Windows, sweep_report.py makes the table and the figure.
set -u
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
OUT=${OUT:-data/sweep_naca4412}
LOGS=logs/$(basename "$OUT")
mkdir -p "$LOGS"

# the dataset's settings, at one angle and speed
WING="--airfoil naca4412 --alpha 5 --U 30 --iterations 1000 --max-iterations 3000"

# name | worker | options. The lowest heights (the longest runs) come first.
JOBLIST="
h0050|400|--h 0.050
h0063|401|--h 0.063
h0079|402|--h 0.079
h0098|403|--h 0.098
h0123|404|--h 0.123
h0155|405|--h 0.155
h0194|406|--h 0.194
h0243|407|--h 0.243
h0305|408|--h 0.305
h0382|409|--h 0.382
h0479|410|--h 0.479
h0600|411|--h 0.600
free|412|--free-air
"

run_one() {  # name, worker, then the options for groundGen.py
    local name=$1 worker=$2
    shift 2
    python3 groundGen.py $WING "$@" --worker "$worker" --out "$OUT/$name" > "$LOGS/$name.log" 2>&1
    echo "[$(date +%H:%M)] finished $name: $(grep -m1 -E 'done in|skipping' "$LOGS/$name.log" | sed 's/^ *//')" \
         "$(grep -m1 -E '^    (converged|NOT converged)' "$LOGS/$name.log" | sed 's/^ *//' | cut -c1-60)"
}

echo "[$(date +%H:%M)] NACA 4412 ride-height sweep: $(grep -c '|' <<< "$JOBLIST") runs, $JOBS at a time"
while IFS='|' read -r name worker opts; do
    [ -z "$name" ] && continue
    if compgen -G "$OUT/$name/*.npz" > /dev/null || compgen -G "$OUT/$name/not_converged/*.npz" > /dev/null; then
        echo "[$(date +%H:%M)] $name has a sample already: skipped"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
        wait -n  # wait until one of the running jobs has finished
    done
    echo "[$(date +%H:%M)] start $name"
    run_one "$name" "$worker" $opts &
done <<< "$JOBLIST"
wait
echo "=== sweep finished: $(ls "$OUT"/*/*.npz 2>/dev/null | wc -l) of 13 runs gave a sample (Cl steady) ==="
