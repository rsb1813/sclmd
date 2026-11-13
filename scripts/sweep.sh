#!/bin/sh
# The measurements the allocation is made from, one after another (each about three minutes on the GPU):
# the head alone at both rates, everything at 4 bits, one matrix type at 4 bits, one block of ten layers at 4 bits.
# A run whose report is already in data/ is skipped. While logs/pause exists, no new run starts (the GPU is wanted
# for something else); remove the file to go on.
cd "$(dirname "$0")/.." || exit 1
P=../.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
run() {
    tag=$1
    shift
    [ -e "data/codes_v1_$tag.json" ] && return
    while [ -e logs/pause ]; do sleep 5; done
    echo "== $tag ($(date +%H:%M:%S))" >> logs/sweep.log
    $P -m scripts.eval_codes "$@" >> logs/sweep.log 2>&1
}
run head8 --head 8
run head6 --head 6
run l0-59_c8 --layers 0-59 --rate 8
for m in up gate down q k v o; do run "c6_${m}8" --plan "data/plans/c6_${m}8.json"; done
for b in 0 10 20 30 40 50; do run "c6_l${b}-$((b + 9))_8" --plan "data/plans/c6_l${b}-$((b + 9))_8.json"; done
echo "== done ($(date +%H:%M:%S))" >> logs/sweep.log
