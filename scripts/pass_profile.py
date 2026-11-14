# Where the main model's pass spends its time on the real model, by kind of kernel (the engine waits for the GPU after
# every launch while this runs, so the sum is more than a pass takes: the shares are what to read).
import argparse
import ctypes as C
import json
import os
import time

import torch
from transformers import AutoTokenizer

from sclmd import DATA, ckpt
from sclmd import engine as E
from sclmd import model as M
from sclmd.stream import cap_vram
from scripts.reference import load_set

KINDS = ("input", "q k v", "heads", "scores", "softmax", "attend", "o", "gate up", "down", "head")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--plan")
    ap.add_argument("--rate", type=int, default=6)
    ap.add_argument("--head", type=int, default=8)
    ap.add_argument("--at", default="64,2048", help="context lengths to measure at")
    ap.add_argument("--tokens", default="1,7")
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--rows", type=int, default=8, help="the rows the engine is made for (8, 16 or 32)")
    ap.add_argument("--capacity", type=int, default=8192)
    ap.add_argument("--reserve", type=float, default=0.75, help="GiB of the free VRAM left to other programs")
    E.add_int8_option(ap)
    E.add_kv4_option(ap)
    ap.add_argument("--keyless", action="store_true", help="the full-attention layers keep their values only (Engine: keyless)")
    args = ap.parse_args()
    E.use_int8_option(args)
    E.use_kv4_option(args)
    cap_vram(margin=int(args.reserve * 2**30))
    cfg = M.Config.load(ckpt("orig"))
    if args.plan:
        with open(args.plan, encoding="utf-8") as f:
            p = json.load(f)
        plan, head = {int(k): v for k, v in p["layers"].items()}, p.get("head") or args.head
    else:
        plan, head = {i: {m: args.rate for m in M.LINEARS} for i in range(cfg.n_layers)}, args.head
    eng = E.Engine.load(args.name, plan, head, args.capacity, rows=args.rows, keyless=args.keyless)
    if os.path.exists(os.path.join(DATA, "eval_v1", "tokens.npz")):
        filler = load_set("eval_v1")[0].reshape(-1).tolist()
    else:  # a checkout without the evaluation sets: the document of scripts/speed_check.py
        with open(os.path.join(DATA, "speed_check", "doc.txt"), encoding="utf-8") as f:
            filler = AutoTokenizer.from_pretrained(ckpt("orig")).encode(f.read(), add_special_tokens=False)
    assert eng.lib.sclmd_profile(None) == len(KINDS)
    for at in [int(a) for a in args.at.split(",")]:
        while eng.length < at:
            eng.feed((filler * (1 + at // len(filler)))[eng.length : min(at, eng.length + 4096)], keep="last")
        for tokens in [int(t) for t in args.tokens.split(",")]:
            def passes(n):
                for _ in range(n):
                    eng.feed(filler[:tokens])
                    eng.crop(at)
            passes(3)
            torch.cuda.synchronize()
            t = time.perf_counter()
            passes(args.passes)
            torch.cuda.synchronize()
            plain = (time.perf_counter() - t) / args.passes * 1000
            times = (C.c_double * len(KINDS))()
            eng.lib.sclmd_profile(times)
            passes(args.passes)
            eng.lib.sclmd_profile(None)
            ms = [v / args.passes * 1000 for v in times]
            print(f"at {at}, {tokens} token(s): a pass {plain:.1f} ms; with a wait after every launch {sum(ms):.1f} ms = "
                  + ", ".join(f"{k} {v:.2f}" for k, v in zip(KINDS, ms)), flush=True)


if __name__ == "__main__":
    main()
