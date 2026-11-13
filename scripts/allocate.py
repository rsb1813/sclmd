# Chooses which matrices get 4 bits instead of 3 under a VRAM budget, from measured gains.
# Measured (scripts/eval_codes.py on the plans of scripts/plans.py): the whole model's KL with everything at 3 bits,
# with one matrix type at 4 in every layer, and with one block of ten layers at 4. A matrix's gain is taken as its
# type's gain shared over the layers in proportion to their block's gain; the matrices with the most gain per byte
# are upgraded until the budget is used. The plan's real KL is then measured, not trusted to this estimate.
import argparse
import json
import os

from sclmd import DATA, ckpt
from sclmd import model as M
from scripts.plans import sizes, uniform, write


def kl(name: str, tag: str, against: str) -> float:
    with open(os.path.join(DATA, f"codes_{name}_{tag}.json"), encoding="utf-8") as f:
        return json.load(f)["against"][against]["all"]["kl"]


def gains(cfg: M.Config, name: str, against: str) -> tuple[dict, dict, float]:
    """(matrix type -> gain of upgrading it everywhere, block start -> gain of upgrading the block, KL at 3 bits)."""
    base = kl(name, f"l0-{cfg.n_layers - 1}_c6", against)
    by_type = {m: base - kl(name, f"c6_{m}8", against) for m in M.LINEARS}
    by_block = {b: base - kl(name, f"c6_l{b}-{b + 9}_8", against) for b in range(0, cfg.n_layers, 10)}
    return by_type, by_block, base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--gib", type=float, required=True, help="VRAM for the layers' codes")
    ap.add_argument("--against", default="orig_fp32")
    ap.add_argument("--out", help="plan name (default: fit_<gib>)")
    ap.add_argument("--head", type=int, default=8)
    args = ap.parse_args()
    cfg = M.Config.load(ckpt("orig"))
    by_type, by_block, base = gains(cfg, args.name, args.against)
    mean_block = sum(by_block.values()) / len(by_block)
    layers_with = {m: sum(1 for j in range(cfg.n_layers) if m in sizes(cfg, j)) for m in M.LINEARS}  # full-attention layers have no v
    items = []
    for i in range(cfg.n_layers):
        share = max(by_block[i // 10 * 10], 0.0) / mean_block
        for m, size in sizes(cfg, i).items():
            gain = max(by_type[m], 0.0) / layers_with[m] * share
            items.append((gain / size, gain, size, i, m))
    plan = uniform(cfg, 6)
    used = sum(s for i in range(cfg.n_layers) for s in sizes(cfg, i).values()) * 3 / 8
    budget = args.gib * 2**30
    assert used <= budget, f"everything at 3 bits is {used / 2**30:.2f} GiB"
    predicted = base
    for density, gain, size, i, m in sorted(items, reverse=True):
        if used + size / 8 <= budget:
            plan[i][m] = 8
            used += size / 8
            predicted -= gain
    total = sum(s for i in range(cfg.n_layers) for s in sizes(cfg, i).values())
    name = args.out or f"fit_{args.gib:g}"
    path = write(name, plan, head=args.head, note=f"{used / 2**30:.2f} GiB of {args.gib:g}, {used * 8 / total:.3f} bits a weight, "
                 f"estimated KL {predicted:.4f} against {args.against} (3 bits everywhere: {base:.4f})")
    print(f"gain by type: " + " ".join(f"{m} {g:.4f}" for m, g in by_type.items()))
    print(f"gain by block: " + " ".join(f"{b} {g:.4f}" for b, g in by_block.items()))
    upgraded = {m: sum(1 for i in plan if plan[i].get(m) == 8) for m in M.LINEARS}
    print(f"{path}: {used / 2**30:.2f} GiB, {used * 8 / total:.3f} bpw, estimated KL {predicted:.4f}; layers at 4 bits by type: {upgraded}")


if __name__ == "__main__":
    main()
