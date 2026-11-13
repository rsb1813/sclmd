# Runs a checkpoint over a token set one layer at a time and stores the last layer's outputs (and what the drafter reads).
# With --ckpt orig this is the reference all loss is measured against; any other checkpoint is scored against it.
import argparse
import json
import os
import time

import numpy as np
import torch

from sclmd import DATA, ckpt
from sclmd.checkpoint import Checkpoint
from sclmd.stream import Stream, cap_vram, compare


def load_set(name: str):
    z = np.load(os.path.join(DATA, name, "tokens.npz"))
    return torch.from_numpy(z["tokens"].astype(np.int64)), z["category"], [str(x) for x in z["names"]]


def by_category(result: dict, category, names) -> dict:
    out = {}
    for ci, name in enumerate(names):
        rows = [i for i, c in enumerate(category) if c == ci]
        out[name] = {k: float(np.mean([v[i] for i in rows])) for k, v in result.items()}
    out["all"] = {k: float(np.mean(v)) for k, v in result.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="eval_v1")
    ap.add_argument("--ckpt", default="orig", help="orig or qat")
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    cap_vram()
    ids, category, names = load_set(args.set)
    ck = Checkpoint(ckpt(args.ckpt))
    t = time.time()
    out = Stream(ck, batch=args.batch).run(ids, log=lambda s: print(s, flush=True))
    seconds = time.time() - t
    hidden = out["hidden"]
    path = os.path.join(DATA, f"{args.set}_{args.ckpt}.pt")
    torch.save({"hidden": hidden.cpu(), "shared": out["shared"]}, path)

    embed, norm = ck.embed("cuda"), ck.norm("cuda")
    if args.ckpt == "orig":
        ref_hidden, ref_embed, ref_norm = hidden, embed, norm
    else:
        ref = Checkpoint(ckpt("orig"))
        ref_hidden = torch.load(os.path.join(DATA, f"{args.set}_orig.pt"), weights_only=True)["hidden"]
        ref_embed, ref_norm = ref.embed("cuda"), ref.norm("cuda")
    result = compare(ck.cfg, ref_hidden, hidden, ref_norm, ref_embed, ids, test_norm=norm, test_embed=embed)
    report = {"set": args.set, "ckpt": args.ckpt, "tokens": int(ids.numel()), "seconds": round(seconds, 1),
              "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "categories": by_category(result, category, names)}
    with open(os.path.join(DATA, f"{args.set}_{args.ckpt}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
