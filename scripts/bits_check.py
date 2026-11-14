# Whether a build of the engine computes the same bits as another: a fixed text goes through the real model in passes
# of several widths, with the drafter's guesses on the way, and every output is summed up as integers (logits, last
# outputs, the drafter's logits, at the end all keys and values). A change meant to cost nothing gives the same sums:
#   set SCLMD_DLL=<the old build>   and   python -m scripts.bits_check --save base
#   python -m scripts.bits_check --against base                    (the new build)
import argparse
import hashlib
import json
import os

import torch
from transformers import AutoTokenizer

from sclmd import ROOT, ckpt
from sclmd import engine as E
from sclmd import plan as P
from sclmd.stream import cap_vram

FOLDER = os.path.join(ROOT, "data", "speed_check")


def bits(t: torch.Tensor) -> list:
    """Two sums over a tensor's bits (plain, and weighted by where the element is): any changed bit shows."""
    flat = t.contiguous().view(-1)
    if flat.dtype in (torch.float32, torch.float16):
        flat = flat.view(torch.int32 if flat.dtype == torch.float32 else torch.int16)
    plain = weighted = 0
    for a in range(0, flat.numel(), 1 << 22):  # a piece at a time: the keys and values are gigabytes
        v = flat[a : a + (1 << 22)].to(torch.int64)
        w = (torch.arange(v.numel(), device=v.device, dtype=torch.int64) + a) % 8191 + 1
        plain, weighted = plain + int(v.sum()), (weighted + int((v * w).sum())) % (1 << 61)
    return [plain, weighted]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--plan", default=os.path.join(ROOT, "data", "plans", "fit_own_v1_11.9.json"))
    ap.add_argument("--capacity", type=int, default=16384)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--reserve", type=float, default=0.4)
    ap.add_argument("--upto", type=int, default=13000, help="positions the text is read to")
    ap.add_argument("--save", default="")
    ap.add_argument("--against", default="")
    E.add_int8_option(ap)
    ap.add_argument("--keyless", action="store_true", help="the full-attention layers keep their values only (Engine: keyless)")
    args = ap.parse_args()
    E.use_int8_option(args)
    cap_vram(margin=int(args.reserve * 2**30))
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    plan, head = P.read(args.plan)
    eng = E.Engine.load(args.name, plan, head or 8, args.capacity, rows=args.rows, keyless=args.keyless)
    drafter = E.EngineDrafter.load(eng, ckpt("drafter"), args.name, depth=6)
    with open(os.path.join(FOLDER, "doc.txt"), encoding="utf-8") as f:
        text = tok.encode(f.read(), add_special_tokens=False)
    print(f"[bits] the library: {os.path.abspath(E.DLL)}", flush=True)
    sums, stops = [], {96, 1504, 6000, 12000}
    at = 0
    while at < args.upto:
        nxt = min((s for s in stops if s > at), default=args.upto)
        while at < nxt:  # whole passes of the engine's rows
            n = min(args.rows, nxt - at)
            logits, hn = eng.run([(0, text[at : at + n])])
            sums.append(["pass", at, n] + bits(logits) + bits(hn))
            at += n
        if at in stops:
            # as generation goes: narrow passes that are taken back, and the drafter's guesses from the last output
            last = hn[-1].clone()
            for n in (1, 3, 7, 12):
                logits, hn = eng.run([(0, text[at : at + n])])
                sums.append(["narrow", at, n] + bits(logits) + bits(hn))
                eng.crop(at)
            token, hidden = text[at], last
            for step in range(6):
                logits, hidden = drafter.step(eng.embed([token])[0], hidden, at)
                token = int(logits.argmax())
                sums.append(["guess", at, step, token] + bits(logits) + bits(hidden))
    for kind, group in eng.kv_all.items():
        for name, t in zip(("keys", "values", "key scales", "value scales") if len(group) == 4 else ("values", "value scales"), group):
            sums.append([f"{kind} {name}"] + bits(t))
    digest = hashlib.sha1(json.dumps(sums).encode()).hexdigest()[:16]
    print(f"[bits] {len(sums)} outputs up to position {at}: {digest}", flush=True)
    if args.save:
        with open(os.path.join(FOLDER, f"bits_{args.save}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(sums, f)
        print(f"[bits] stored as bits_{args.save}", flush=True)
    if args.against:
        with open(os.path.join(FOLDER, f"bits_{args.against}.json"), encoding="utf-8") as f:
            old = json.load(f)
        differ = [k for k, (a, b) in enumerate(zip(sums, old)) if a != b]
        if len(sums) != len(old) or differ:
            print(f"[bits] OTHER BITS than {args.against}: {len(differ)} of {len(old)} outputs differ, the first: {sums[differ[0]][:4] if differ else 'the counts'}", flush=True)
            raise SystemExit(1)
        print(f"[bits] THE SAME BITS as {args.against}: all {len(sums)} outputs", flush=True)


if __name__ == "__main__":
    main()
