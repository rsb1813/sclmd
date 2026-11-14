#!/bin/sh
# One GPU job at a time, in a waiting line, for several sessions working side by side (the model takes 12 to 14 GiB of
# the card's 16):
#   sh /h/projects/sclm/sclmd/scripts/gpu.sh <label> <command ...>
# takes a ticket in the waiting line (logs/gpu.queue), waits for its turn, takes the lock (the folder logs/gpu.lock),
# runs the command and gives the lock back.
#   Order: class first, then the ticket's time. Classes: 1 high (labels verify*, fix*, main*: work that ends something,
#   or GPU_PRIORITY=high), 2 normal, 3 low (GPU_PRIORITY=low). A ticket that has waited GPU_AGE seconds (default 900)
#   counts one class higher, so that nobody waits for ever.
#   A ticket is kept when a call gives up (exit 75 or 76): calling again with the same label within 20 minutes of giving
#   up takes the same place in the line (the ticket file's time is when its caller last waited; the name keeps the time
#   it was taken, by which the line is ordered). A ticket whose caller is not waiting right now is passed over (the
#   card does not wait for it) but keeps its place for when it calls again.
#   GPU_WAIT   seconds to wait at most (default 200); then exit 75 (still busy): call again
#   GPU_LIMIT  seconds the command may run (default 340); it is stopped after that (exit 124)
# Keep GPU_WAIT + GPU_LIMIT under the 600 s a tool call may take, so that a call is never cut while it holds the lock.
# exit 76: another program (a game) holds the card's memory (not started; do not stop other programs; call again).
# exit 77: the owner has dropped this brief (logs/exp/cancelled lists the labels).
#   sh gpu.sh status     the job on the card, the waiting line, the hold
#   sh gpu.sh hold       no new job starts (the running one finishes; the line waits); the same as: type nul > logs\gpu.hold
#   sh gpu.sh go         starts again (del logs\gpu.hold)
# Tests: GPU_ROOT is the folder of the lock, the line and the hold (default the main copy's logs/); GPU_FAKE_USED is the
# card's used MiB instead of asking nvidia-smi.
ROOT=${GPU_ROOT:-/h/projects/sclm/sclmd/logs}
LOCK=$ROOT/gpu.lock
Q=$ROOT/gpu.queue
HOLD=$ROOT/gpu.hold
AGE=${GPU_AGE:-900}
TTL=${GPU_TICKET_TTL:-1200}
T0=$(( $(date +%s) - SECONDS ))

used_mib() {
    if [ -n "$GPU_FAKE_USED" ]; then echo "$GPU_FAKE_USED"; return; fi
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '\r '
}

holder() {  # "label, N s of M s" of the job on the card, or "nobody"
    if [ -d "$LOCK" ]; then
        h_label=$(cat "$LOCK/label" 2>/dev/null)
        h_since=$(cat "$LOCK/since" 2>/dev/null)
        h_limit=$(cat "$LOCK/limit" 2>/dev/null)
        echo "${h_label:-?}, $(( T0 + SECONDS - ${h_since:-$((T0 + SECONDS))} )) s of ${h_limit:-?} s"
    else
        echo "nobody"
    fi
}

# The line: one empty file per ticket, named <class>@<ticket time>@<waiting pid>@<label>; only its owner writes it.
# scan sets: head (the best ticket whose caller is waiting now), pos and n (my place among all tickets, and their count).
scan() {
    now=$((T0 + SECONDS))
    head= ; hkey= ; hpid= ; pos=1 ; n=0
    for f in "$Q"/*@*@*@*; do
        [ -e "$f" ] || continue
        b=${f##*/}
        r=${b%%@*}; x=${b#*@}; t=${x%%@*}; x=${x#*@}; p=${x%%@*}
        case "$r$t$p" in *[!0-9]*) rm -f "$f"; continue ;; esac
        if [ "$p" != "$$" ] && ! kill -0 "$p" 2>/dev/null; then
            # its caller is away: given up TTL seconds after it last waited (not after the ticket was taken, which would
            # throw out exactly the tickets that have waited longest)
            seen=$(stat -c %Y "$f" 2>/dev/null || echo 0)
            if [ $((now - seen)) -gt "$TTL" ]; then rm -f "$f"; fi
            alive=0
        else
            alive=1
        fi
        [ -e "$f" ] || continue
        [ $((now - t)) -ge "$AGE" ] && [ "$r" -gt 1 ] && r=$((r - 1))
        key=$((r * 10000000000 + t))
        n=$((n + 1))
        if [ "$f" != "$mine" ] && { [ "$key" -lt "$mkey" ] || { [ "$key" -eq "$mkey" ] && [ "$p" -lt "$$" ]; }; }; then
            pos=$((pos + 1))
        fi
        if [ $alive -eq 1 ] && { [ -z "$head" ] || [ "$key" -lt "$hkey" ] || { [ "$key" -eq "$hkey" ] && [ "$p" -lt "$hpid" ]; }; }; then
            head=$f; hkey=$key; hpid=$p
        fi
    done
}

if [ $# -eq 1 ]; then
    case "$1" in
    status|--status)
        mkdir -p "$Q"
        echo "[gpu] on the card: $(holder)"
        [ -e "$HOLD" ] && echo "[gpu] HOLD: no new job starts ($HOLD; sh gpu.sh go to start again)"
        now=$((T0 + SECONDS))
        lines=
        for f in "$Q"/*@*@*@*; do
            [ -e "$f" ] || continue
            b=${f##*/}
            r=${b%%@*}; x=${b#*@}; t=${x%%@*}; x=${x#*@}; p=${x%%@*}; l=${x#*@}
            state=waiting
            kill -0 "$p" 2>/dev/null || state="away (keeps its place)"
            e=$r
            [ $((now - t)) -ge "$AGE" ] && [ "$r" -gt 1 ] && e=$((r - 1))
            case $r in 1) c=high ;; 2) c=normal ;; *) c=low ;; esac
            [ "$e" != "$r" ] && c="$c, raised for waiting"
            lines="$lines$((e * 10000000000 + t)) $l ($c), ticket $((now - t)) s ago, $state
"
        done
        if [ -n "$lines" ]; then
            echo "[gpu] waiting line:"
            printf %s "$lines" | sort -n | awk '{ $1 = ""; printf "  %d.%s\n", NR, $0 }'
        else
            echo "[gpu] waiting line: empty"
        fi
        exit 0 ;;
    hold|--hold)
        : > "$HOLD"
        echo "[gpu] hold on: no new job starts until sh gpu.sh go (the running one finishes: $(holder))"
        exit 0 ;;
    go|--go)
        rm -f "$HOLD"
        echo "[gpu] hold off"
        exit 0 ;;
    esac
fi

label=$1
shift
if [ -z "$label" ] || [ $# -eq 0 ]; then
    echo "usage: sh gpu.sh <label> <command ...> | status | hold | go"
    exit 2
fi
# a brief the owner has dropped: its engineer is told at the next GPU job (logs/exp/cancelled lists the labels)
if grep -qxF "$label" "$ROOT/exp/cancelled" 2>/dev/null; then
    echo "[gpu] CANCELLED BY THE OWNER: the brief '$label' has been dropped. Do not run any more GPU jobs. Stop work now and hand in your result as it is (verdict: unfinished, with what you had measured so far)."
    exit 77
fi

case "$label" in verify*|fix*|main*) rank=1 ;; *) rank=2 ;; esac
case "$GPU_PRIORITY" in high) rank=1 ;; normal) rank=2 ;; low) rank=3 ;; esac
safe=$(printf %s "$label" | tr -c 'A-Za-z0-9._-' '_')
mkdir -p "$Q"

# my ticket: the place of an earlier call with this label whose caller is gone, or a new one
t=
for f in "$Q"/*@*@*@"$safe"; do
    [ -e "$f" ] || continue
    b=${f##*/}; x=${b#*@}; ot=${x%%@*}; x=${x#*@}; op=${x%%@*}
    if ! kill -0 "$op" 2>/dev/null; then
        seen=$(stat -c %Y "$f" 2>/dev/null || echo 0)
        if [ $((T0 + SECONDS - seen)) -gt "$TTL" ]; then rm -f "$f"; continue; fi
        t=$ot
        mine="$Q/$rank@$t@$$@$safe"
        if mv -f "$f" "$mine" 2>/dev/null; then touch "$mine"; else t=; fi
        break
    fi
done
if [ -z "$t" ]; then
    t=$((T0 + SECONDS))
    mine="$Q/$rank@$t@$$@$safe"
    : > "$mine"
fi
mkey=$((rank * 10000000000 + t))

s0=$SECONDS
waited=0
vram=
last_head=
free_n=0
while :; do
    [ -e "$mine" ] || : > "$mine"
    if [ -d "$LOCK" ]; then
        # a holder that is gone (its shell no longer runs), or a lock older than 30 minutes, is given up
        pid=$(cat "$LOCK/pid" 2>/dev/null)
        since=$(cat "$LOCK/since" 2>/dev/null)
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then rm -rf "$LOCK"; continue; fi
        if [ -n "$since" ] && [ $((T0 + SECONDS - since)) -gt 1800 ]; then rm -rf "$LOCK"; continue; fi
    elif [ ! -e "$HOLD" ]; then
        scan
        if [ "$head" != "$mine" ]; then
            # a head that leaves the card free for about 18 s is not really waiting (its pid has been given to another
            # program): its ticket is taken out of the line. A caller that is in fact waiting puts it back, same place.
            if [ "$head" = "$last_head" ]; then free_n=$((free_n + 1)); else last_head=$head; free_n=1; fi
            if [ "$free_n" -ge 6 ]; then rm -f "$head"; free_n=0; fi
        elif mkdir "$LOCK" 2>/dev/null; then
            echo $$ > "$LOCK/pid"
            echo "$label" > "$LOCK/label"
            echo $((T0 + SECONDS)) > "$LOCK/since"
            echo "${GPU_LIMIT:-340}" > "$LOCK/limit"
            trap 'rm -rf "$LOCK"' EXIT INT TERM
            vram=
            used=$(used_mib)
            if [ -n "$used" ] && [ "$used" -gt 2500 ]; then
                sleep 20  # a job that has just ended may still be giving its memory back
                used=$(used_mib)
            fi
            if [ -n "$used" ] && [ "$used" -gt 2500 ]; then
                # another program holds the card: give the lock back, keep the place, wait on
                vram=$used
                rm -rf "$LOCK"
                trap - EXIT INT TERM
            else
                rm -f "$mine"
                break
            fi
        fi
    fi
    waited=$((SECONDS - s0))
    if [ "$waited" -ge "${GPU_WAIT:-200}" ]; then
        touch "$mine"  # the time it last waited: its place is kept for TTL seconds from now
        scan
        keep="your place in line is kept: $pos of $n (call again with the same label within 20 min)"
        if [ -n "$vram" ] && [ ! -d "$LOCK" ]; then
            echo "[gpu] ${vram} MiB of the card are held by another program: not started; $keep"
            exit 76
        fi
        if [ -e "$HOLD" ]; then
            echo "[gpu] busy after ${waited} s (the owner holds the card: logs/gpu.hold; on it: $(holder)); $keep; try again"
        else
            echo "[gpu] busy after ${waited} s (held by: $(holder)); $keep; try again"
        fi
        exit 75
    fi
    sleep 3
done
[ "$waited" -gt 0 ] && echo "[gpu] waited ${waited} s"
timeout -k 10 "${GPU_LIMIT:-340}" "$@"
code=$?
[ "$code" -eq 124 ] && echo "[gpu] stopped after ${GPU_LIMIT:-340} s"
exit $code
