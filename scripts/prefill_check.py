# How fast a prompt is read (the tokens a second of feed(), which is what the chat does with a prompt), and that what
# it leaves is the same: the last pass's logits and outputs and all keys and values, summed up as integers
# (scripts/bits_check.py), so two builds can be compared:
#   python -m scripts.prefill_check --save base       (SCLMD_DLL picks an older build)
#   python -m scripts.prefill_check --against base
# --server: the server's engine (every matrix at 3 bits, 32 rows a pass, a context of 8,192) in place of the chat's.
# --kv4 sums every tensor of the keys and values (the 4-bit codes and every scale, under names of their own).
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


def by_position(eng: E.Engine, n: int) -> list:
    """--core --kv4: the sliding layers' keys and values (int8, the 4-bit codes and every scale) of sequence 0's last
    `window` positions after n tokens, gathered by position (their slots depend on the ring, which grows with the rows
    of the passes), the 4-bit layers' first tile (int8) where kv4_heads_k keeps it: its positions 8..15 in their own
    tile, which nothing writes over, and 0..7 in the ring tile 0 while they are in the window."""
    t = M.SLIDING
    if t not in eng.kv_all:
        return []
    cfg, group = eng.cfg, eng.kv_all[t]
    hd, kvh, ring = cfg.head_dim[t], cfg.kv_heads[t], eng.ring[t]
    pos = torch.arange(max(0, n - cfg.window), n, device=eng.device)
    plain, four = (group[:4], group[4:]) if eng.kv4_set else (group, ())
    out = [x[:, eng.slots(t, pos).long()] for x in plain]
    if four:
        later = pos[pos >= E.TILE]
        out += [x[:, eng.slots(t, later).long()] for x in four]
        first = [p for p in range(min(n, E.TILE)) if p >= 8 or p >= n - cfg.window]
        if first:
            own = eng.tiles[t][0]
            tile = torch.tensor([own[0] if p < 8 else own[ring] for p in first], device=eng.device)
            sub = torch.tensor(first, device=eng.device) & 7
            out += [codes.view(codes.shape[0], -1, 8, kvh, hd)[:, tile, sub] for codes in four[:2]]  # int8 rows in the codes' bytes
            out += [scale[:, tile * E.TILE + sub] for scale in four[2:]]
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--keyless", action="store_true", help="the full-attention layers keep their values only (Engine: keyless)")
    ap.add_argument("--server", action="store_true")
    ap.add_argument("--rows", type=int, default=0, choices=(0,) + E.ROWS, help="rows of a pass (0: 16, the chat's; 32 with --server)")
    ap.add_argument("--core", action="store_true", help="sum up only what does not depend on the rows of a pass (the last row's logits and "
                    "output, the full-attention layers' keys and values), under a name without the rows: so that passes of other widths can be compared")
    ap.add_argument("--lengths", default="", help="prompt lengths to read, with commas")
    ap.add_argument("--capacity", type=int, default=0, help="the context the engine is made for (0: 16,384 for the chat's, 8,192 for the server's)")
    ap.add_argument("--reserve", type=float, default=0.4)
    ap.add_argument("--repeat", type=int, default=1, help="reads of each length: every read is timed and printed, then the median; "
                    "the bits must be the same every time")
    ap.add_argument("--save", default="")
    ap.add_argument("--against", default="")
    E.add_int8_option(ap)
    E.add_kv4_option(ap)
    args = ap.parse_args()
    int8 = E.use_int8_option(args)
    kv4 = E.use_kv4_option(args)
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
    print(f"[prefill] the {kind}'s engine, {rows} rows a pass, context {capacity}" + (f", int8 products (sigma {int8:g})" if int8 else "") + (", 4-bit sliding keys and values (--kv4)" if kv4 else "")
          + f"; the library: {os.path.abspath(E.DLL)}", flush=True)
    sums, medians = [], []
    for n in lengths:
        times, mine = [], None
        for _ in range(args.repeat):
            eng.crop(0)
            eng.feed(text[: max(64, rows)], keep="last")  # warm (with a pass as wide as the engine's)
            eng.crop(0)
            # what the warm-up and longer reads before left past this prompt is not summed (the prompt's rows never
            # read it): without this, a prompt shorter than the warm-up, or after a longer one, gave OTHER BITS
            for group in eng.kv_all.values():
                for t2 in group:
                    t2.zero_()
            torch.cuda.synchronize()
            t = time.perf_counter()
            logits, hidden = eng.feed(text[:n], keep="last")
            torch.cuda.synchronize()
            s = time.perf_counter() - t
            times.append(s)
            tail = (n - 1) % rows + 1  # the rows of the last pass
            assert logits.shape[0] == tail - E.logits_first(tail, 0) and hidden.shape[0] == tail
            # --core: the full-attention layers' keys and values only (a sliding layer's ring has more tiles for wider passes,
            # so its slots are others; what it holds reaches the last row's output and the next layers' keys)
            kv = {t: g for t, g in eng.kv_all.items() if not args.core or t == M.FULL}
            # --kv4: every tensor, the 4-bit codes (uint8) and every scale too, and with --core the sliding layers' by
            # position (new baselines: without --kv4 the sums are the int8 tensors' as before)
            kept = [t2 for group in kv.values() for t2 in group if kv4 or t2.dtype == torch.int8] + (by_position(eng, n) if kv4 and args.core else [])
            got = [n] + bits(logits[-1]) + bits(hidden[-1:] if args.core else hidden) + [x for t2 in kept for x in bits(t2)]
            print(f"[prefill] {n:6d} tokens in {s:6.2f} s = {n / s:7.1f} tok/s", flush=True)
            if mine is not None and got != mine:
                print(f"[prefill] OTHER BITS on another read of the same {n} tokens", flush=True)
                raise SystemExit(1)
            mine = got
        sums.append(mine)
        medians.append(sorted(times)[len(times) // 2])
        if args.repeat > 1:
            print(f"[prefill] {n:6d} tokens: median of {args.repeat} reads {medians[-1]:.3f} s = {n / medians[-1]:7.1f} tok/s", flush=True)
    if len(lengths) > 1:
        print(f"[prefill] all the lengths: {sum(lengths)} tokens in {sum(medians):.2f} s = {sum(lengths) / sum(medians):7.1f} tok/s"
              + (" (medians)" if args.repeat > 1 else ""), flush=True)
    name = (f"prefill_{kind}_" + ("core" if args.core else str(rows)) + (f"_{capacity}" if args.capacity else "") + ("_keyless" if args.keyless else "")
            + (f"_int8_{int8:g}" if int8 else "") + ("_kv4" if kv4 else ""))
    what = ("the last row's logits and output and the full-attention layers' keys and values" if args.core
            else "the last logits, the last pass's outputs and all keys and values")
    if kv4:
        what += (" (with their scales and the 4-bit codes" + ("; the sliding layers' of the last window's positions, by position)" if args.core else ")"))
    if args.save:
        with open(os.path.join(FOLDER, f"{name}_{args.save}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(sums, f)
        print(f"[prefill] stored as {name}_{args.save}", flush=True)
    if args.against:
        with open(os.path.join(FOLDER, f"{name}_{args.against}.json"), encoding="utf-8") as f:
            old = {a[0]: a for a in json.load(f)}  # by length: a run of other lengths compares the ones both have
        other = [a[0] for a in sums if a[0] in old and a != old[a[0]]]
        missing = [a[0] for a in sums if a[0] not in old]
        if other:
            print(f"[prefill] OTHER BITS than {args.against}: lengths {other}", flush=True)
            raise SystemExit(1)
        if missing:
            print(f"[prefill] {args.against} has no read of the lengths {missing}: not compared", flush=True)
            raise SystemExit(1)
        print(f"[prefill] THE SAME BITS as {args.against}: {what}, at every length", flush=True)


if __name__ == "__main__":
    main()
