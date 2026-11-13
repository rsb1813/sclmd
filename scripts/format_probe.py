# One layer's matrices under each candidate storage format: the relative error of the matrix's output on held-out
# inputs. Inputs come from the checkpoint run over a token set up to that layer; even sequences fit, odd ones test.
import argparse
import json
import os
import time

import torch

from sclmd import DATA, ckpt
from sclmd import model as M
from sclmd import quant as Q
from sclmd.checkpoint import Checkpoint
from sclmd.stream import Stream, cap_vram
from scripts.reference import load_set

DEV = "cuda"
GROUP = {"q": "attn_in", "k": "attn_in", "v": "attn_in", "o": "o_in", "gate": "mlp_in", "up": "mlp_in", "down": "down_in"}
TEST_ROWS = 4096


class Inputs:
    """Per input group of the layer: H = X^T X over the fitting sequences, and a sample of held-out inputs."""

    def __init__(self, layer: int):
        self.layer, self.h, self.test, self.count = layer, {}, {}, {}

    def __call__(self, i: int, name: str, x: torch.Tensor, rows: slice):
        if i != self.layer:
            return
        for r in range(rows.start, rows.stop):
            xr = x[r - rows.start].float()
            if r % 2 == 0:
                if name not in self.h:
                    self.h[name] = torch.zeros(xr.shape[-1], xr.shape[-1], device=DEV, dtype=torch.float32)
                for a in range(0, xr.shape[0], 512):
                    self.h[name].addmm_(xr[a : a + 512].t(), xr[a : a + 512])
                self.count[name] = self.count.get(name, 0) + xr.shape[0]
            else:
                pick = torch.randperm(xr.shape[0], device=DEV)[:256]
                self.test.setdefault(name, []).append(xr[pick].cpu())


def rotated(rot, h):
    """R^T H R: the Hessian in the basis the weight is quantized in."""
    return rot.rotate(rot.rotate(h).t().contiguous())


def pair_codebook(w: torch.Tensor, bits: int) -> torch.Tensor:
    """k-means over the row-normalized weight's adjacent pairs."""
    torch.manual_seed(0)
    wn, _ = Q.row_normalize(w)
    pairs = wn.reshape(-1, 2)
    sample = pairs[torch.randperm(pairs.shape[0], device=w.device)[: 1 << 21]]
    return Q.kmeans(sample, 1 << bits, iters=25)


def variants(w: torch.Tensor, h: torch.Tensor, which: list):
    """Yield (name, bits per weight, the weight as stored)."""
    n, k = w.shape
    rot = Q.BlockHadamard(k, seed=17 + k, device=DEV)
    wr, hr = None, None
    for name in which:
        if name == "q4_0":
            yield name, 4.5, Q.q4_0(w)
            continue
        kind, bits = name.split("_")[0], int(name.split("c")[-1])  # e.g. gptq_v2c6, gptqh_v2c8, near_v2c6
        use_rot = kind.endswith("h")
        if use_rot and wr is None:
            wr, hr = rot.rotate(w), rotated(rot, h)
        ww, hh = (wr, hr) if use_rot else (w.float(), h)
        cb = pair_codebook(ww, bits)
        if kind.startswith("near"):  # nearest codeword, no error feedback
            wn, s = Q.row_normalize(ww)
            wq = Q.vq_dequant(wn, cb) * s
        else:
            codes, scales = Q.gptq_vq(ww, hh, cb, k, refit=True)
            wq = Q.vq_codes_dequant(codes, scales, cb)
        yield name, bits / 2, rot.unrotate(wq) if use_rot else wq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--set", default="eval_v1")
    ap.add_argument("--ckpt", default="orig")
    ap.add_argument("--mats", default="gate,down,q,o")
    ap.add_argument("--formats", default="q4_0,near_v2c8,gptq_v2c8,gptqh_v2c8,gptq_v2c7,gptqh_v2c7,gptq_v2c6,gptqh_v2c6")
    args = ap.parse_args()
    cap_vram()
    ids = load_set(args.set)[0]
    ck = Checkpoint(ckpt(args.ckpt))
    inputs = Inputs(args.layer)
    with torch.no_grad():
        Stream(ck, batch=2).run(ids, n_layers=args.layer + 1, tap=inputs)
        torch.cuda.empty_cache()
        w_all = ck.layer(args.layer)
        report = {"layer": args.layer, "ckpt": args.ckpt, "set": args.set, "fit_tokens": inputs.count, "matrices": {}}
        for mat in args.mats.split(","):
            if mat not in w_all:
                continue
            group = GROUP[mat]
            w, h = w_all[mat].to(DEV).float(), inputs.h[group]
            x = torch.cat(inputs.test[group])[:TEST_ROWS].to(DEV)
            ref = x @ w.t()
            denom = ref.pow(2).sum().item()
            rows = {}
            for name, bpw, wq in variants(w, h, args.formats.split(",")):
                t = time.time()
                err = (x @ wq.t() - ref).pow(2).sum().item() / denom
                werr = (wq - w).pow(2).sum().item() / w.pow(2).sum().item()
                rows[name] = {"bpw": bpw, "output_error": err, "weight_error": werr}
                print(f"layer {args.layer} {mat:5s} {tuple(w.shape)} {name:12s} {bpw:.1f} bpw: output error {err:.5f}, weight error {werr:.5f}", flush=True)
                del wq
                torch.cuda.empty_cache()
            report["matrices"][mat] = rows
            del w, h, x, ref
            torch.cuda.empty_cache()
    with open(os.path.join(DATA, f"format_probe_{args.ckpt}_l{args.layer}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
