# The model with some layers' matrices (and the output head) replaced by their fitted pair codes, scored against the
# original. Computed in float32 (as the engine's sums are), and scored against both the original in float32 (what the
# codes alone cost) and the original in bf16 (the reference; the float32 original itself is KL 0.018 away from it).
import argparse
import json
import os
import time

import torch

from sclmd import DATA, ckpt
from sclmd import codes as C
from sclmd import model as M
from sclmd.checkpoint import Checkpoint
from sclmd.stream import Stream, cap_vram, compare
from scripts.quantize import parse_layers
from scripts.reference import by_category, load_set


def plan_of(args, cfg) -> tuple[dict, int | None]:
    """(layer -> {matrix: bits}, the head's bits or None for the original head)."""
    if args.plan:
        with open(args.plan, encoding="utf-8") as f:
            p = json.load(f)
        return {int(k): v for k, v in p["layers"].items()}, p.get("head")
    if not args.layers:
        return {}, args.head
    mats = args.mats.split(",") if args.mats else list(M.LINEARS)
    return {i: {m: args.rate for m in mats} for i in parse_layers(args.layers, cfg.n_layers)}, args.head


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--set", default="eval_v1")
    ap.add_argument("--layers", help="e.g. 30 or 0-59 or 2,17")
    ap.add_argument("--rate", type=int, default=6, help="bits a pair for --layers (6 or 8)")
    ap.add_argument("--mats", help="only these matrices of --layers (default: all)")
    ap.add_argument("--head", type=int, help="bits a pair for the output head (default: the original head)")
    ap.add_argument("--plan", help="a JSON file {layers: {layer: {matrix: bits}}, head: bits} instead of --layers and --head")
    ap.add_argument("--tag", help="name of the report (default: from the arguments)")
    ap.add_argument("--keep", action="store_true", help="keep the last layer's outputs (data/<set>_codes_<name>_<tag>.pt): what the engine is checked against")
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    cap_vram()
    ck = Checkpoint(ckpt("orig"))
    plan, head = plan_of(args, ck.cfg)
    tag = args.tag or (os.path.splitext(os.path.basename(args.plan))[0] if args.plan else
                       "_".join(([f"l{args.layers}_c{args.rate}" + (f"_{args.mats}" if args.mats else "")] if args.layers else [])
                                + ([f"head{head}"] if head else [])))
    ids, category, names, target = load_set(args.set)

    def weights(i, w):
        if i not in plan:
            return w
        fitted = C.load_layer(args.name, i, {m: b for m, b in plan[i].items() if m in w})
        return {k: fitted.get(k, v) for k, v in w.items()}

    t = time.time()
    if plan:
        hidden = Stream(ck, batch=args.batch).run(ids, weights=weights, dtype=torch.float32)["hidden"].cpu()
    else:  # only the head differs: the original's own outputs
        hidden = torch.load(os.path.join(DATA, f"{args.set}_orig_fp32.pt"), weights_only=True)["hidden"]
    seconds = time.time() - t
    if args.keep:
        torch.save({"hidden": hidden}, os.path.join(DATA, f"{args.set}_codes_{args.name}_{tag}.pt"))
    torch.cuda.empty_cache()
    norm = ck.norm("cuda")
    report = {"set": args.set, "codes": args.name, "tag": tag, "seconds": round(seconds, 1),
              "bits_per_weight": bits_per_weight(ck.cfg, plan), "head_bits_per_weight": head / 2 if head else None, "against": {}}
    for ref_name in ("orig_fp32", "orig"):
        ref = torch.load(os.path.join(DATA, f"{args.set}_{ref_name}.pt"), weights_only=True)["hidden"]
        # one float32 matrix of the head's size is 5.6 GB: the reference's and the path's are on the GPU together only
        # when they differ
        embed = ck.embed("cuda")
        ref_embed, ref_norm = (embed.float(), norm.float()) if ref_name == "orig_fp32" else (embed, norm)
        del embed
        test_embed = C.load_head(args.name, head) if head else ref_embed.float()
        result = compare(ck.cfg, ref, hidden, ref_norm, ref_embed, ids, target, test_norm=norm.float(), test_embed=test_embed)
        report["against"][ref_name] = by_category(result, category, names)
        del ref, ref_embed, test_embed
        torch.cuda.empty_cache()
    where = "" if args.set == "eval_v1" else args.set + "_"  # the evaluation set's reports carry no set name
    with open(os.path.join(DATA, f"{where}codes_{args.name}_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    a, b = report["against"]["orig_fp32"], report["against"]["orig"]
    print(f"{tag}: KL vs fp32 original {a['all']['kl']:.4f} (top1 {a['all']['top1']:.4f}), vs bf16 original {b['all']['kl']:.4f} (top1 {b['all']['top1']:.4f}), "
          f"ppl {a['all']['ppl']:.3f} | " + " ".join(f"{c} {v['kl']:.4f}" for c, v in a.items() if c != "all"), flush=True)


def bits_per_weight(cfg, plan: dict) -> float:
    """Average over the layers' matrices in the plan (bits a pair / 2)."""
    total = bits = 0
    for i, mats in plan.items():
        t = cfg.layer_types[i]
        qo, kv = cfg.heads * cfg.head_dim[t] * cfg.hidden, cfg.kv_heads[t] * cfg.head_dim[t] * cfg.hidden
        size = {"q": qo, "o": qo, "k": kv, "v": kv, "gate": cfg.mlp * cfg.hidden, "up": cfg.mlp * cfg.hidden, "down": cfg.mlp * cfg.hidden}
        for m, b in mats.items():
            if m == "v" and t == M.FULL:
                continue
            total += size[m]
            bits += size[m] * b / 2
    return bits / total if total else 0.0


if __name__ == "__main__":
    main()
