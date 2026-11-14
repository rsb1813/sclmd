#!/bin/sh
# One GPU job at a time, for several sessions working side by side (the model takes 12 to 14 GiB of the card's 16):
#   sh /h/projects/sclm/sclmd/scripts/gpu.sh <label> <command ...>
# waits for the lock (a folder in the main copy's logs/), runs the command, and gives the lock back.
#   GPU_WAIT   seconds to wait for the lock at most (default 200); when it is still held: exit 75, try again
#   GPU_LIMIT  seconds the command may run (default 340); it is stopped after that (exit 124)
# Keep GPU_WAIT + GPU_LIMIT under the 600 s a tool call may take, so that a call is never cut while it holds the lock.
# exit 76: the lock was free but something else holds the card's memory (not started; do not stop other programs).
LOCK=/h/projects/sclm/sclmd/logs/gpu.lock
label=$1
shift
waited=0
while ! mkdir "$LOCK" 2>/dev/null; do
    pid=$(cat "$LOCK/pid" 2>/dev/null)
    since=$(cat "$LOCK/since" 2>/dev/null)
    now=$(date +%s)
    # a holder that is gone (its shell no longer runs), or a lock older than 30 minutes, is given up
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then rm -rf "$LOCK"; continue; fi
    if [ -n "$since" ] && [ $((now - since)) -gt 1800 ]; then rm -rf "$LOCK"; continue; fi
    if [ "$waited" -ge "${GPU_WAIT:-200}" ]; then
        echo "[gpu] busy after ${waited} s (held by: $(cat "$LOCK/label" 2>/dev/null)); try again"
        exit 75
    fi
    sleep 5
    waited=$((waited + 5))
done
echo $$ > "$LOCK/pid"
echo "$label" > "$LOCK/label"
date +%s > "$LOCK/since"
trap 'rm -rf "$LOCK"' EXIT INT TERM
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$used" ] && [ "$used" -gt 2500 ]; then
    sleep 20  # a job that has just ended may still be giving its memory back
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ "$used" -gt 2500 ]; then
        echo "[gpu] ${used} MiB of the card are held by another program: not started"
        exit 76
    fi
fi
[ "$waited" -gt 0 ] && echo "[gpu] waited ${waited} s"
timeout -k 10 "${GPU_LIMIT:-340}" "$@"
code=$?
[ "$code" -eq 124 ] && echo "[gpu] stopped after ${GPU_LIMIT:-340} s"
exit $code
