# How unevenly a matrix's error sits in its rows: if a few rows carry most of what 4 bits would win back, giving 4
# bits to those rows only (a matrix split by rows) buys more per byte than upgrading whole matrices.
# For one layer: each row's output error energy at 3 and at 4 bits under the calibration inputs, and how much of the
# matrix's gain (3 -> 4 bits) its best rows hold.
import argparse
import json
import os

import torch

from sclmd import DATA, ckpt
from sclmd import codes as C
from sclmd import model as M
from sclmd.checkpoint import Checkpoint
from sclmd.stream import cap_vram
from scripts.quantize import Inputs
from scripts.reference import load_set

DEV = "cuda"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--calib", default="calib_v1")
    ap.add_argument("--seqs", type=int, default=32)
    args = ap.parse_args()
    cap_vram()
    ck = Checkpoint(ckpt("orig"))
    cfg = ck.cfg
    i = args.layer
    ids = load_set(args.calib)[0][: args.seqs].to(DEV)
    s, n = ids.shape
    positions = torch.arange(n, device=DEV)
    embed = ck.embed(DEV)
    h = torch.cat([embed[ids[a : a + 2]] * M.embed_scale(cfg, embed.dtype) for a in range(0, s, 2)])
    del embed
    inputs = Inputs()
    for j in range(i + 1):
        w = ck.layer(j, DEV)
        for a in range(0, s, 2):
            h[a : a + 2] = M.layer_forward(cfg, j, w, h[a : a + 2], positions,
                                           tap=(lambda name, x: inputs.tap(j, name, x, None)) if j == i else None)
    hs = {name: v.cpu() for name, v in inputs.h.items()}
    inputs.h.clear()
    w = {k: v.cpu() for k, v in w.items()}
    del h
    torch.cuda.empty_cache()
    report = {"layer": i, "matrices": {}}
    for group in ("attn_in", "o_in", "mlp_in", "down_in"):
        hr = C.rotated_hessian(C.rotation(hs[group].shape[0], DEV), hs.pop(group).to(DEV))
        for m in [m for m in M.LINEARS if C.GROUP[m] == group and m in w]:
            wr = C.rotation(w[m].shape[1], DEV).rotate(w[m].to(DEV).float())
            energy = {}
            for bits in (6, 8):
                codes, scales, book, _ = C.stored(args.name, i, {m: bits})[m]
                d = C.Q.vq_codes_dequant(codes.to(DEV).long(), scales.to(DEV).float()[:, None], book.to(DEV).float()) - wr
                energy[bits] = ((d @ hr) * d).sum(-1)  # each row's lost output energy
            gain = (energy[6] - energy[8]).clamp_min(0)
            top = gain.sort(descending=True).values.cumsum(0) / gain.sum()
            rows = gain.numel()
            shares = {f"{p}%": round(top[int(rows * p / 100) - 1].item(), 3) for p in (10, 25, 50)}
            signal = ((wr @ hr) * wr).sum(-1)
            report["matrices"][m] = {"rows": rows, "gain_in_best_rows": shares,
                                     "error_c6": (energy[6].sum() / signal.sum()).item(), "error_c8": (energy[8].sum() / signal.sum()).item()}
            print(f"layer {i} {m:5s} {rows:6d} rows: of the gain from 3 to 4 bits, the best 10% of rows hold {shares['10%']:.2f}, "
                  f"25% hold {shares['25%']:.2f}, 50% hold {shares['50%']:.2f}", flush=True)
        del hr
        torch.cuda.empty_cache()
    with open(os.path.join(DATA, f"row_probe_l{i}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
