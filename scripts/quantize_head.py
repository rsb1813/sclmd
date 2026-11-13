# Fits pair codes for the output head (the embedding table as a matrix, 262,144 x 5,376), at 3 and at 4 bits a weight.
# Its inputs are the original model's last outputs over the calibration set, at the positions where the model
# predicts (the next token is one of its own): elsewhere the head's output is never used.
# The rows are fitted a part at a time with one codebook and one Hessian factor for all of them.
import argparse
import time

import torch

from sclmd import ckpt
from sclmd import codes as C
from sclmd import model as M
from sclmd import quant as Q
from sclmd.checkpoint import Checkpoint
from sclmd.stream import Stream, cap_vram
from scripts.reference import load_set

DEV = "cuda"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1", help="folder under C:\\models\\sclmd-codes")
    ap.add_argument("--calib", default="calib_v1")
    ap.add_argument("--bits", default="6,8")
    ap.add_argument("--rows", type=int, default=16384, help="rows fitted at once")
    args = ap.parse_args()
    cap_vram()
    ck = Checkpoint(ckpt("orig"))
    cfg = ck.cfg
    ids, _, _, target = load_set(args.calib)
    t0 = time.time()
    hidden = Stream(ck).run(ids, log=lambda s: print(s, flush=True) if s.split("/")[0].endswith("0") else None)["hidden"]
    use = torch.zeros_like(target)
    use[:, :-1] = target[:, 1:]  # a position counts when the token after it is the model's own
    x = M.rms_norm(hidden[use.to(DEV)].float(), ck.norm(DEV).float(), cfg.eps)
    del hidden
    torch.cuda.empty_cache()
    h = torch.zeros(cfg.hidden, cfg.hidden, device=DEV)
    for a in range(0, x.shape[0], 4096):
        h.addmm_(x[a : a + 4096].t(), x[a : a + 4096])
    print(f"[head] inputs: {x.shape[0]} positions, {time.time() - t0:.0f} s", flush=True)
    del x
    rot = C.rotation(cfg.hidden, DEV)
    factor = Q.gptq_factor(C.rotated_hessian(rot, h))
    w = ck.embed("cpu")
    vocab = w.shape[0]
    sample = torch.randperm(vocab, generator=torch.Generator().manual_seed(0))[: args.rows]
    fits, errors = {}, {}
    for bits in [int(b) for b in args.bits.split(",")]:
        t1 = time.time()
        book = C.pair_codebook(rot.rotate(w[sample].to(DEV).float()), bits)
        codes, scales, lost, energy = [], [], 0.0, 0.0
        for a in range(0, vocab, args.rows):
            f = C.fit(w[a : a + args.rows].to(DEV).float(), factor, bits, book=book)
            codes.append(f["codes"].cpu())
            scales.append(f["scales"].cpu())
            lost += f["error"] * f["energy"]
            energy += f["energy"]
        fits[bits] = {"codes": torch.cat(codes), "scales": torch.cat(scales), "book": book.float().cpu()}
        errors[f"c{bits}"] = lost / energy
        print(f"[head] c{bits}: error {lost / energy:.5f}, {time.time() - t1:.0f} s", flush=True)
    C.save_head(args.name, fits, {"calib": args.calib, "positions": int(use.sum()), "errors": errors, "shape": list(w.shape),
                                  "seconds": round(time.time() - t0, 1)})
    print("[head] done", flush=True)


if __name__ == "__main__":
    main()
