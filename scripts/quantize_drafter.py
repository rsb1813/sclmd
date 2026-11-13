# Fits pair codes for the drafter's matrices, at 3 and at 4 bits a weight.
# The drafter only guesses: whatever it loses shows as fewer guesses kept, never in the text. As codes it is a quarter
# of the bytes to read at every guess and 0.65 GiB less VRAM.
# Its inputs come from replaying it over the main model's stored outputs (as scripts/draft_replay.py does), six
# guesses deep, the later ones following the text's real tokens.
import argparse
import os
import time

import torch

from sclmd import DATA, ckpt
from sclmd import codes as C
from sclmd import model as M
from sclmd import quant as Q
from sclmd.checkpoint import Checkpoint
from sclmd.drafter import Drafter
from sclmd.stream import cap_vram
from scripts.draft_replay import DEPTH
from scripts.reference import load_set

DEV = "cuda"
INPUT_OF = {"q": "attn_in", "o": "o_in", "gate": "mlp_in", "up": "mlp_in", "down": "down_in"}


class Inputs:
    """H = X^T X of every input the drafter's matrices get: (where, name) -> [K, K]."""

    def __init__(self):
        self.h, self.rows = {}, 0

    def __call__(self, where, name: str, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        key = (where, name)
        if key not in self.h:
            self.h[key] = torch.zeros(x.shape[-1], x.shape[-1], device=DEV)
        self.h[key].addmm_(x.t(), x)
        if key == ("pre", "in"):
            self.rows += x.shape[0]


def fit_rows(w: torch.Tensor, factor: tuple, bits: int, rows: int = 16384) -> tuple[dict, float]:
    """C.fit over a matrix of any height, a part of the rows at a time with one codebook -> (codes, scales, book), error."""
    cols = w.shape[1]
    if w.shape[0] <= rows:
        f = C.fit(w.to(DEV).float(), factor, bits)
        return {k: f[k].cpu() for k in ("codes", "scales", "book")}, f["error"]
    sample = torch.randperm(w.shape[0], generator=torch.Generator().manual_seed(0))[:rows]
    book = C.pair_codebook(C.rotation(cols, DEV).rotate(w[sample].to(DEV).float()), bits)
    codes, scales, lost, energy = [], [], 0.0, 0.0
    for a in range(0, w.shape[0], rows):
        f = C.fit(w[a : a + rows].to(DEV).float(), factor, bits, book=book)
        codes.append(f["codes"].cpu())
        scales.append(f["scales"].cpu())
        lost += f["error"] * f["energy"]
        energy += f["energy"]
    return {"codes": torch.cat(codes), "scales": torch.cat(scales), "book": book.float().cpu()}, lost / energy


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1", help="folder under C:\\models\\sclmd-codes")
    ap.add_argument("--set", default="eval_v1", help="the set whose stored outputs (data/<set>_orig.pt) the drafter is replayed over")
    ap.add_argument("--bits", default="6,8")
    ap.add_argument("--stride", type=int, default=2, help="every how many places are used")
    ap.add_argument("--places", type=int, default=128, help="places drafted together")
    args = ap.parse_args()
    cap_vram()
    t0 = time.time()
    ids, _, _, target = load_set(args.set)
    stored = torch.load(os.path.join(DATA, f"{args.set}_orig.pt"), weights_only=True)
    ck = Checkpoint(ckpt("orig"))
    cfg, embed, norm = ck.cfg, ck.embed(DEV), ck.norm(DEV)
    scale = M.embed_scale(cfg, embed.dtype)
    drafter = Drafter.load(ckpt("drafter"), DEV)
    inputs = Inputs()
    s, n = ids.shape
    for q in range(s):
        last = M.rms_norm(stored["hidden"][q].to(DEV), norm, cfg.eps)
        shared = {t: (k[q : q + 1].to(DEV), v[q : q + 1].to(DEV), torch.arange(n, device=DEV)) for t, (k, v) in stored["shared"].items()}
        tok = ids[q].to(DEV)
        places = torch.arange(0, n - DEPTH - 1, device=DEV)
        places = places[target[q, 1 : n - DEPTH].to(DEV)][:: args.stride]
        for a in range(0, places.numel(), args.places):
            t = places[a : a + args.places]
            hidden = last[t][None]
            for j in range(1, DEPTH + 1):
                _, hidden = drafter.step((embed[tok[t + j]] * scale)[None], hidden, t, shared, tap=inputs)
    del embed, stored, shared, last
    torch.cuda.empty_cache()
    print(f"[drafter] inputs: {inputs.rows} rows, {time.time() - t0:.0f} s", flush=True)

    # (matrix name in the file, weight, where its input was tapped)
    mats = [("pre", drafter.pre, ("pre", "in"))]
    for i, w in enumerate(drafter.layers):
        mats += [(f"layers.{i}.{m}", w[m], (i, INPUT_OF[m])) for m in ("q", "o", "gate", "up", "down")]
    mats += [("head", drafter.head, ("out", "in")), ("post", drafter.post, ("out", "in"))]
    bits = [int(b) for b in args.bits.split(",")]
    fits, errors, factors = {}, {}, {}
    for name, w, key in mats:
        w = w.cpu()
        if key not in factors:  # one factor on the GPU at a time, shared by the matrices with that input
            factors.clear()
            torch.cuda.empty_cache()
            factors[key] = Q.gptq_factor(C.rotated_hessian(C.rotation(w.shape[1], DEV), inputs.h.pop(key)))
        for b in bits:
            fits[(name, b)], errors[f"{name}.c{b}"] = fit_rows(w, factors[key], b)
        print(f"[drafter] {name} {tuple(w.shape)}: " + " ".join(f"c{b} {errors[f'{name}.c{b}']:.5f}" for b in bits), flush=True)
    C.save_drafter(args.name, fits, {"set": args.set, "rows": inputs.rows, "errors": errors, "seconds": round(time.time() - t0, 1)})
    print("[drafter] done", flush=True)


if __name__ == "__main__":
    main()
