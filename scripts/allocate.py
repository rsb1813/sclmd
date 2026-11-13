# Writes the plan for a VRAM budget (which matrices get 4 bits instead of 3) from the measured shares (sclmd/plan.py),
# and prints the shares. The plan's real KL is then measured (scripts/eval_codes.py --plan), not trusted to the estimate.
import argparse
import os

from sclmd import DATA, ckpt
from sclmd import model as M
from sclmd import plan as P


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--gib", type=float, required=True, help="VRAM for the layers' codes (3 bits everywhere is 10.23)")
    ap.add_argument("--against", default="orig_fp32")
    ap.add_argument("--out", help="plan name (default: fit_<gib>)")
    ap.add_argument("--head", type=int, default=8)
    args = ap.parse_args()
    cfg = M.Config.load(ckpt("orig"))
    share, base = P.shares(cfg, args.name, args.against)
    by_type = {m: sum(s for (i, mm), s in share.items() if mm == m) for m in M.LINEARS}
    by_block = {b: sum(s for (i, m), s in share.items() if i // P.BLOCK * P.BLOCK == b) for b in range(0, cfg.n_layers, P.BLOCK)}
    print("share by type:  " + "  ".join(f"{m} {s:.2f}" for m, s in by_type.items()))
    print("share by block: " + "  ".join(f"{b}-{b + P.BLOCK - 1} {s:.2f}" for b, s in by_block.items()))
    plan, predicted = P.fit(cfg, args.name, int(args.gib * 2**30), args.against)
    used, bpw = P.code_bytes(cfg, plan) / 2**30, P.bits_per_weight(cfg, plan)
    name = args.out or f"fit_{args.gib:g}"
    path = P.write(os.path.join(DATA, "plans", name + ".json"), plan, head=args.head,
                   note=f"{used:.2f} GiB of {args.gib:g}, {bpw:.3f} bits a weight, estimated KL {predicted:.4f} against {args.against} "
                        f"(3 bits everywhere: {base:.4f})")
    at4 = {m: [i for i in plan if plan[i].get(m) == 8] for m in M.LINEARS}
    print(f"{path}: {used:.2f} GiB, {bpw:.3f} bpw, estimated KL {predicted:.4f} (3 bits everywhere: {base:.4f})")
    for m, layers in at4.items():
        print(f"  {m:5s} at 4 bits in {len(layers):2d} layers" + (f": {layers[0]}..{layers[-1]}" if layers else ""))


if __name__ == "__main__":
    main()
