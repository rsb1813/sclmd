# Fits pair codes for the layers' matrices, a layer at a time, at 3 and at 4 bits a weight.
# The calibration set runs through the original weights; each layer's matrices are fitted to the inputs they get
# there (so the layers are independent and which rate a matrix ends up with can be chosen afterwards).
# A layer that is already on disk is skipped: stop and start as you like.
import argparse
import json
import os
import time

import torch

from sclmd import DATA, ckpt
from sclmd import codes as C
from sclmd import model as M
from sclmd.checkpoint import Checkpoint
from sclmd.stream import Stream, cap_vram
from scripts.reference import load_set

DEV = "cuda"


class Inputs:
    """While one layer runs: H = X^T X of each input group, and how large the inputs get."""

    def __init__(self):
        self.h, self.peak, self.norm, self.tokens = {}, {}, {}, 0

    def tap(self, i: int, name: str, x: torch.Tensor, rows: slice):
        x = x.reshape(-1, x.shape[-1]).float()
        if name not in self.h:
            self.h[name] = torch.zeros(x.shape[-1], x.shape[-1], device=DEV, dtype=torch.float32)
        for a in range(0, x.shape[0], 1024):
            self.h[name].addmm_(x[a : a + 1024].t(), x[a : a + 1024])
        self.peak[name] = max(self.peak.get(name, 0.0), x.abs().max().item())
        self.norm[name] = max(self.norm.get(name, 0.0), x.norm(dim=-1).max().item())  # bounds the rotated input too
        if name == "attn_in":
            self.tokens += x.shape[0]


def parse_layers(spec: str, n: int) -> list:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    assert all(0 <= i < n for i in out)
    return sorted(set(out))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1", help="folder under C:\\models\\sclmd-codes")
    ap.add_argument("--ckpt", default="orig")
    ap.add_argument("--calib", default="calib_v1")
    ap.add_argument("--layers", default="0-59")
    ap.add_argument("--bits", default="6,8")
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    cap_vram()
    ck = Checkpoint(ckpt(args.ckpt))
    cfg = ck.cfg
    bits = [int(b) for b in args.bits.split(",")]
    todo = [i for i in parse_layers(args.layers, cfg.n_layers) if C.layer_info(args.name, i) is None]
    print(f"[quantize] {args.name}: layers to fit {todo}", flush=True)
    if not todo:
        return
    ids = load_set(args.calib)[0].to(DEV)
    s, n = ids.shape
    positions = torch.arange(n, device=DEV)
    embed = ck.embed(DEV)
    h = torch.cat([embed[ids[a : a + args.batch]] * M.embed_scale(cfg, embed.dtype) for a in range(0, s, args.batch)])
    del embed
    t_start = time.time()
    for i in range(todo[-1] + 1):
        t0 = time.time()
        w = ck.layer(i, DEV)
        inputs = Inputs() if i in todo else None
        for a in range(0, s, args.batch):
            rows = slice(a, min(a + args.batch, s))
            h[rows] = M.layer_forward(cfg, i, w, h[rows], positions,
                                      tap=(lambda name, x, rows=rows: inputs.tap(i, name, x, rows)) if inputs else None)
        if inputs is None:
            del w
            continue
        t_run = time.time() - t0
        hs = {name: v.cpu() for name, v in inputs.h.items()}  # one Hessian on the GPU at a time from here on
        inputs.h.clear()
        torch.cuda.empty_cache()
        fits, errors = {}, {}
        for group in ("attn_in", "o_in", "mlp_in", "down_in"):
            mats = [m for m in M.LINEARS if C.GROUP[m] == group and m in w]
            hr = C.rotated_hessian(C.rotation(hs[group].shape[0], DEV), hs.pop(group).to(DEV))
            for mat in mats:
                wm = w[mat].float()
                for b in bits:
                    f = C.fit(wm, hr, b)
                    errors[f"{mat}.c{b}"] = f.pop("error")
                    fits[(mat, b)] = {k: v.cpu() for k, v in f.items()}
                del wm
            del hr
            torch.cuda.empty_cache()
        info = {"layer": i, "type": cfg.layer_types[i], "calib": args.calib, "tokens": inputs.tokens, "errors": errors,
                "input_peak": inputs.peak, "input_norm": inputs.norm, "shapes": {m: list(w[m].shape) for m in M.LINEARS if m in w},
                "seconds": round(time.time() - t0, 1)}
        C.save_layer(args.name, i, fits, info)
        del w, fits
        torch.cuda.empty_cache()
        done = [j for j in todo if j <= i]
        print(f"[quantize] layer {i:2d} ({cfg.layer_types[i][:7]}) run {t_run:.0f} s, fit {time.time() - t0 - t_run:.0f} s, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB, {len(done)}/{len(todo)} layers, total {(time.time() - t_start) / 60:.0f} min | "
              + " ".join(f"{k} {v:.4f}" for k, v in errors.items()), flush=True)
    print("[quantize] done", flush=True)


if __name__ == "__main__":
    main()
