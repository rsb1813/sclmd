# How fast a prompt is read (the tokens a second of feed(), which is what the chat does with a prompt), and that what
# it leaves is the same: the last pass's logits and outputs and all keys and values, summed up as integers
# (scripts/bits_check.py), so two builds can be compared:
#   python -m scripts.prefill_check --save base       (SCLMD_DLL picks an older build)
#   python -m scripts.prefill_check --against base
# --server: the server's engine (every matrix at 3 bits, 32 rows a pass, a context of 8,192) in place of the chat's.
import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from sclmd import ROOT, ckpt
from sclmd import engine as E
from sclmd import model as M
from sclmd import plan as P
from sclmd.stream import cap_vram
from scripts.bits_check import FOLDER, bits


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--keyless", action="store_true", help="the full-attention layers keep their values only (Engine: keyless)")
    ap.add_argument("--server", action="store_true")
    ap.add_argument("--rows", type=int, default=0, help="rows of a pass (0: 16, the chat's; 32 with --server)")
    ap.add_argument("--lengths", default="", help="prompt lengths to read, with commas")
    ap.add_argument("--capacity", type=int, default=0, help="the context the engine is made for (0: 16,384 for the chat's, 8,192 for the server's)")
    ap.add_argument("--reserve", type=float, default=0.4)
    ap.add_argument("--save", default="")
    ap.add_argument("--against", default="")
    args = ap.parse_args()
    cap_vram(margin=int(args.reserve * 2**30))
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    cfg = M.Config.load(ckpt("orig"))
    if args.server:
        plan, head, capacity, rows = P.uniform(cfg, 6), 8, args.capacity or 8192, args.rows or 32
        lengths = [int(x) for x in (args.lengths or "512,2048,6000").split(",")]
    else:
        plan, head = P.read(os.path.join(ROOT, "data", "plans", "fit_own_v1_11.9.json"))
        capacity, rows = args.capacity or 16384, args.rows or 16
        lengths = [int(x) for x in (args.lengths or "512,2048,6000,12000").split(",")]
    eng = E.Engine.load(args.name, plan, head or 8, capacity, rows=rows, keyless=args.keyless)
    with open(os.path.join(FOLDER, "doc.txt"), encoding="utf-8") as f:
        text = tok.encode(f.read(), add_special_tokens=False)
    text = text * (1 + max(lengths) // len(text))  # a prompt longer than the document goes round it again
    kind = "server" if args.server else "chat"
    print(f"[prefill] the {kind}'s engine, {rows} rows a pass, context {capacity}; the library: {os.path.abspath(E.DLL)}", flush=True)
    sums = []
    for n in lengths:
        eng.crop(0)
        eng.feed(text[:64], keep="last")  # warm
        eng.crop(0)
        torch.cuda.synchronize()
        t = time.perf_counter()
        logits, hidden = eng.feed(text[:n], keep="last")
        torch.cuda.synchronize()
        s = time.perf_counter() - t
        tail = (n - 1) % rows + 1  # the rows of the last pass
        assert logits.shape[0] == tail and hidden.shape[0] == tail
        sums.append([n] + bits(logits[-1]) + bits(hidden) + [x for group in eng.kv_all.values() for t2 in group if t2.dtype == torch.int8 for x in bits(t2)])
        print(f"[prefill] {n:6d} tokens in {s:6.2f} s = {n / s:7.1f} tok/s", flush=True)
    name = f"prefill_{kind}_{rows}" + (f"_{capacity}" if args.capacity else "") + ("_keyless" if args.keyless else "")
    if args.save:
        with open(os.path.join(FOLDER, f"{name}_{args.save}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(sums, f)
        print(f"[prefill] stored as {name}_{args.save}", flush=True)
    if args.against:
        with open(os.path.join(FOLDER, f"{name}_{args.against}.json"), encoding="utf-8") as f:
            old = json.load(f)
        if sums != old:
            print(f"[prefill] OTHER BITS than {args.against}: lengths {[a[0] for a, b in zip(sums, old) if a != b]}", flush=True)
            raise SystemExit(1)
        print(f"[prefill] THE SAME BITS as {args.against}: the last logits, the last pass's outputs and all keys and values, at every length", flush=True)


if __name__ == "__main__":
    main()
