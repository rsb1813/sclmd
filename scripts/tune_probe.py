# How much of a quantized layer's error comes back when what is cheap to change is trained with the codes held fixed:
# each matrix's per-row scales and its codebook of pairs, and the layer's norm weights and scalar. The layer is
# trained to give the original layer's output on the original layer's input (calibration sequences), and scored on
# sequences it was not trained on. One layer, as a probe: is a pass over all layers worth making?
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from sclmd import DATA, ckpt
from sclmd import codes as C
from sclmd import model as M
from sclmd.checkpoint import Checkpoint
from sclmd.stream import Stream, cap_vram
from scripts.reference import load_set

DEV = "cuda"
NORMS = ("in_ln", "post_attn_ln", "pre_ff_ln", "post_ff_ln", "q_norm", "k_norm", "scalar")


class Coded(torch.nn.Module):
    """A matrix as pair codes with its scales and codebook as parameters; the product is taken in the rotated basis."""

    def __init__(self, codes, scales, book):
        super().__init__()
        self.codes = codes.long().to(DEV)  # [rows, cols / 2]
        self.rot = C.rotation(codes.shape[1] * 2, DEV)
        self.scale = torch.nn.Parameter(scales.float().to(DEV).clone())
        self.book = torch.nn.Parameter(book.float().to(DEV).clone())

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        w = (self.book[self.codes].flatten(-2) * self.scale[:, None])  # [rows, cols] in the rotated basis
        return (self.rot.rotate(flat) @ w.t()).reshape(*x.shape[:-1], -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--rate", type=int, default=6)
    ap.add_argument("--calib", default="calib_v1")
    ap.add_argument("--train", type=int, default=24, help="sequences trained on")
    ap.add_argument("--test", type=int, default=8, help="sequences scored on (after the training ones)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--what", default="scale,book,norms", help="what is trained")
    args = ap.parse_args()
    cap_vram()
    ck = Checkpoint(ckpt("orig"))
    cfg = ck.cfg
    i = args.layer
    ids = load_set(args.calib)[0][: args.train + args.test]
    with torch.no_grad():
        h = Stream(ck).run(ids, n_layers=i)["hidden"].float()  # the layer's input over every sequence
    n = ids.shape[1]
    positions = torch.arange(n, device=DEV)
    w = {k: v.float() for k, v in ck.layer(i, DEV).items()}
    with torch.no_grad():
        want = torch.cat([M.layer_forward(cfg, i, w, h[q : q + 1], positions) for q in range(h.shape[0])])
    mats = {m: Coded(*C.stored(args.name, i, {m: args.rate})[m][:3]) for m in M.LINEARS if m in w}
    small = {k: torch.nn.Parameter(w[k].clone()) for k in NORMS}
    for m in M.LINEARS:
        w.pop(m, None)
    torch.cuda.empty_cache()
    wq = dict(small) | mats

    def run(q):
        return M.layer_forward(cfg, i, wq, h[q : q + 1], positions, linear=lambda x, m: m(x))

    def score(rows):
        """The layer's error over the sequences: the share of the energy of what the layer adds that is lost."""
        with torch.no_grad():
            lost = sum((run(q) - want[q : q + 1]).pow(2).sum().item() for q in rows)
            return lost / sum((want[q : q + 1] - h[q : q + 1]).pow(2).sum().item() for q in rows)

    train, test = list(range(args.train)), list(range(args.train, args.train + args.test))
    before = (score(train[:8]), score(test))
    what = args.what.split(",")
    params = ([p for m in mats.values() for p in ([m.scale] if "scale" in what else []) + ([m.book] if "book" in what else [])]
              + (list(small.values()) if "norms" in what else []))
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    t = time.time()
    g = torch.Generator().manual_seed(0)
    for step in range(args.steps):
        q = train[int(torch.randint(len(train), (1,), generator=g))]
        loss = F.mse_loss(run(q), want[q : q + 1]) / (want[q : q + 1] - h[q : q + 1]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:4d}: loss {loss.item():.5f}, {time.time() - t:.0f} s", flush=True)
    after = (score(train[:8]), score(test))
    report = {"layer": i, "rate": args.rate, "what": what, "steps": args.steps, "lr": args.lr, "train_before": before[0], "test_before": before[1],
              "train_after": after[0], "test_after": after[1], "seconds": round(time.time() - t, 1),
              "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print(f"layer {i} at c{args.rate}, training {'+'.join(what)}: error on held-out sequences {before[1]:.5f} -> {after[1]:.5f} "
          f"({after[1] / before[1]:.2f}x), on trained ones {before[0]:.5f} -> {after[0]:.5f}; {report['seconds']} s, peak {report['peak_vram_gib']} GiB", flush=True)
    with open(os.path.join(DATA, f"tune_probe_l{i}_c{args.rate}_{'-'.join(what)}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
