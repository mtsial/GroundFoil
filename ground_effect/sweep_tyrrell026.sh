#!/bin/bash
# Ride-height sweep of the Tyrrell 026 wing, the wind-tunnel case of validation_zerihan.sh, at
# the conditions of the dataset, so that the U-Net (which has never seen this airfoil) can be
# compared with the CFD. The U-Net is trained at 10-50 m/s (chord Reynolds number 1-5 million
# with our 1 m chord), so this runs at 10 m/s (Re 1 million): within its range and the closest
# to the tunnel's 462,000. As in the tunnel: incidence 3.45 deg to the chord line, the ride
# height set with the wing at 2.45 deg (--h-alpha 2.45), the nine heights of the tests from
# h/c 0.067 to 0.671 (0.671 is above the dataset's 0.6), and free air. Otherwise the dataset's
# settings (dataset.sh): the default mesh, 1000 iterations, continued up to 3000 while Cl
# settles. Needs data/airfoils/tyrrell026.dat (make_tyrrell026.py; not in this repository).
# Run inside the Docker container:
#     cd /work/ground_effect
#     bash sweep_tyrrell026.sh
# Then, on Windows, sweep_report.py --sweep data\sweep_tyrrell026 makes the table and the figure.
#
# The 10 runs are independent, so they run JOBS at a time (default 4), each in its own
# work/workerN folder (420-429), with its sample in $OUT/<name>/ (not_converged/ in it if Cl
# does not become steady) and its output in $LOGS/<name>.log. Runs that already have a sample
# are skipped, so after an interruption the same command goes on where it stopped.
# About 30 minutes with 4 at a time.
set -u
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
AIRFOIL=${AIRFOIL:-../data/airfoils/tyrrell026.dat}
OUT=${OUT:-data/sweep_tyrrell026}
LOGS=logs/$(basename "$OUT")
mkdir -p "$LOGS"
if [ ! -f "$AIRFOIL" ]; then
    echo "missing $AIRFOIL: run make_tyrrell026.py first"
    exit 1
fi

# the tunnel's incidence and ride-height setting, the dataset's settings, at 10 m/s
WING="--airfoil $AIRFOIL --alpha 3.45 --h-alpha 2.45 --U 10 --iterations 1000 --max-iterations 3000"

# name | worker | options. The lowest heights (the longest runs) come first.
JOBLIST="
h0067|420|--h 0.067
h0090|421|--h 0.090
h0112|422|--h 0.112
h0134|423|--h 0.134
h0179|424|--h 0.179
h0224|425|--h 0.224
h0313|426|--h 0.313
h0448|427|--h 0.448
h0671|428|--h 0.671
free|429|--free-air
"

run_one() {  # name, worker, then the options for groundGen.py
    local name=$1 worker=$2
    shift 2
    python3 groundGen.py $WING "$@" --worker "$worker" --out "$OUT/$name" > "$LOGS/$name.log" 2>&1
    echo "[$(date +%H:%M)] finished $name: $(grep -m1 -E 'done in|skipping' "$LOGS/$name.log" | sed 's/^ *//')" \
         "$(grep -m1 -E '^    (converged|NOT converged)' "$LOGS/$name.log" | sed 's/^ *//' | cut -c1-60)"
}

echo "[$(date +%H:%M)] Tyrrell 026 ride-height sweep: $(grep -c '|' <<< "$JOBLIST") runs, $JOBS at a time"
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
echo "=== sweep finished: $(ls "$OUT"/*/*.npz 2>/dev/null | wc -l) of 10 runs gave a sample (Cl steady) ==="
