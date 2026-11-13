#!/bin/sh
# Runs the benchmark queue: the lines of <main>/data/bench/queue.txt, in order, each a module and its arguments
# (e.g. "scripts.bench --task aime_2026 --run 0"). A line that ends well is noted in done.txt and never run again;
# lines may be added to queue.txt while this runs. While <main>/logs/pause exists nothing runs and the GPU is free
# (a running line ends at its next item and is taken up again afterwards).
# It is meant to run from a frozen copy of the code (a git worktree with its own engine DLL), so that work on the
# main tree does not change what a benchmark runs on:
#   sh scripts/bench_queue.sh <main tree>
MAIN=$1
[ -d "$MAIN" ] || { echo "usage: sh scripts/bench_queue.sh <main tree>"; exit 1; }
cd "$(dirname "$0")/.." || exit 1
P="$MAIN/../.venv/Scripts/python.exe"
Q="$MAIN/data/bench/queue.txt"
D="$MAIN/data/bench/done.txt"
L="$MAIN/logs/bench_queue.log"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 SCLMD_BENCH="$MAIN/data/bench" SCLMD_PAUSE="$MAIN/logs/pause"
mkdir -p "$MAIN/logs"
touch "$D"
fails=0
while :; do
    while [ -e "$MAIN/logs/pause" ]; do sleep 5; done
    line=$(grep -v '^#' "$Q" | grep -v '^[[:space:]]*$' | grep -vxFf "$D" | head -1)
    [ -z "$line" ] && break
    echo "== $line ($(date '+%m-%d %H:%M:%S'))" >> "$L"
    # shellcheck disable=SC2086
    "$P" -m $line >> "$L" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "$line" >> "$D"
        fails=0
    elif [ $rc -eq 75 ]; then
        echo "== paused ($(date '+%m-%d %H:%M:%S'))" >> "$L"
    else
        fails=$((fails + 1))
        echo "== failed with $rc, attempt $fails ($(date '+%m-%d %H:%M:%S'))" >> "$L"
        sleep 20
        if [ $fails -ge 3 ]; then  # given up: noted, and the queue goes on
            echo "$line" >> "$D.failed"
            echo "$line" >> "$D"
            fails=0
        fi
    fi
done
echo "== queue empty ($(date '+%m-%d %H:%M:%S'))" >> "$L"
