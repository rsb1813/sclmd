#!/bin/sh
# The allocation's measurements on the model's own text (data/own_v1, scripts/own_text.py): one matrix type at 4 bits,
# one block of ten layers at 4 bits, the head at 4 bits throughout. About a minute and a half each on the GPU.
# A run whose report is already in data/ is skipped; while logs/pause exists no new run starts.
cd "$(dirname "$0")/.." || exit 1
P=../.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
run() {
    [ -e "data/own_v1_codes_v1_$1.json" ] && return
    while [ -e logs/pause ]; do sleep 5; done
    echo "== $1 ($(date +%H:%M:%S))" >> logs/sweep_own.log
    $P -m scripts.eval_codes --set own_v1 --plan "data/plans/$1.json" >> logs/sweep_own.log 2>&1
}
run c6_head8
run c8_head8
for m in down o up gate k q v; do run "h8_${m}8"; done
for b in 30 40 20 50 10 0; do run "h8_l${b}-$((b + 9))_8"; done
echo "== done ($(date +%H:%M:%S))" >> logs/sweep_own.log
