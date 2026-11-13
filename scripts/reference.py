# Runs a checkpoint over a token set one layer at a time and stores the last layer's outputs (and what the drafter reads).
# With --ckpt orig this is the reference all loss is measured against; any other checkpoint is scored against it.
import argparse
import json
import os
import time

import numpy as np
import torch

from sclmd import DATA, ckpt
from sclmd import model as M
from sclmd import quant as Q
from sclmd.checkpoint import Checkpoint
from sclmd.stream import Stream, cap_vram, compare


def load_set(name: str):
    z = np.load(os.path.join(DATA, name, "tokens.npz"))
    return torch.from_numpy(z["tokens"].astype(np.int64)), z["category"], [str(x) for x in z["names"]], torch.from_numpy(z["target"])


def by_category(result: dict, category, names) -> dict:
    """Per category and over all sequences: each measure averaged over the sequences, weighted by their targets."""
    w = np.array(result["targets"])

    def mean(rows):
        ww = w[rows]
        out = {k: float(np.nansum(np.array(v)[rows] * ww) / ww.sum()) for k, v in result.items() if k != "targets"}
        return out | {"targets": int(ww.sum())}

    out = {name: mean([i for i, c in enumerate(category) if c == ci]) for ci, name in enumerate(names)}
    out["all"] = mean(list(range(len(w))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="eval_v1")
    ap.add_argument("--ckpt", default="orig", help="orig or qat")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--quant", help="q4_0: every matrix of the layers rounded to llama.cpp's Q4_0 grid (no fitting)")
    ap.add_argument("--against", default="orig", help="the checkpoint whose stored outputs are the reference")
    args = ap.parse_args()
    cap_vram()
    ids, category, names, target = load_set(args.set)
    ck = Checkpoint(ckpt(args.ckpt))
    weights, dtype = None, None
    if args.quant == "q4_0":
        weights = lambda i, w: {k: (Q.q4_0(v).to(v.dtype) if k in M.LINEARS else v) for k, v in w.items()}  # noqa: E731
    elif args.quant == "fp32":  # the same weights in float32 arithmetic: how much of any KL is bf16's rounding
        dtype = torch.float32
    elif args.quant:
        raise SystemExit(f"unknown --quant {args.quant}")
    name = args.ckpt + (f"_{args.quant}" if args.quant else "")
    t = time.time()
    out = Stream(ck, batch=args.batch).run(ids, weights=weights, log=lambda s: print(s, flush=True), dtype=dtype)
    seconds = time.time() - t
    hidden = out["hidden"]
    if name == args.ckpt or args.quant == "fp32":  # kept: the drafter replay and later comparisons read them
        torch.save({"hidden": hidden.cpu(), "shared": out["shared"]}, os.path.join(DATA, f"{args.set}_{name}.pt"))

    del out
    hidden = hidden.cpu()
    torch.cuda.empty_cache()
    embed, norm = ck.embed("cuda"), ck.norm("cuda")
    if dtype is not None:
        embed, norm = embed.to(dtype), norm.to(dtype)
    if name == args.against:
        ref_hidden, ref_embed, ref_norm = hidden, embed, norm
    else:
        ref = Checkpoint(ckpt(args.against))
        ref_hidden = torch.load(os.path.join(DATA, f"{args.set}_{args.against}.pt"), weights_only=True)["hidden"]
        ref_embed, ref_norm = ref.embed("cuda"), ref.norm("cuda")
    result = compare(ck.cfg, ref_hidden, hidden, ref_norm, ref_embed, ids, target, test_norm=norm, test_embed=embed)
    report = {"set": args.set, "ckpt": name, "against": args.against, "tokens": int(ids.numel()), "seconds": round(seconds, 1),
              "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "categories": by_category(result, category, names)}
    with open(os.path.join(DATA, f"{args.set}_{name}_vs_{args.against}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
