# The real model on the engine: how fast a pass is, and how far its outputs are from the original's and from the same
# codes computed in PyTorch (scripts/eval_codes.py --keep with the same plan: that difference is the engine's own).
import argparse
import json
import os
import time

import torch

from sclmd import DATA, ckpt
from sclmd import codes as C
from sclmd import engine as E
from sclmd import model as M
from sclmd.checkpoint import Checkpoint
from sclmd.stream import cap_vram, compare
from scripts.eval_codes import bits_per_weight
from scripts.reference import by_category, load_set

DEV = "cuda"


def plan_of(args, cfg) -> tuple[dict, int, str]:
    if args.plan:
        with open(args.plan, encoding="utf-8") as f:
            p = json.load(f)
        return {int(k): v for k, v in p["layers"].items()}, p.get("head", args.head), os.path.splitext(os.path.basename(args.plan))[0]
    return {i: {m: args.rate for m in M.LINEARS} for i in range(cfg.n_layers)}, args.head, f"l0-{cfg.n_layers - 1}_c{args.rate}_head{args.head}"


def timed(eng: E.Engine, ids: list, tokens: int, passes: int) -> float:
    """Milliseconds a pass of `tokens` tokens at the engine's present length (the positions are forgotten again)."""
    start = eng.length
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(passes):
        eng.feed(ids[:tokens])
        eng.crop(start)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / passes * 1000


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--keyless", action="store_true", help="the full-attention layers keep their values only (Engine: keyless)")
    ap.add_argument("--set", default="eval_v1")
    ap.add_argument("--rate", type=int, default=6, help="bits a pair for every matrix of the layers (6 or 8)")
    ap.add_argument("--head", type=int, default=8)
    ap.add_argument("--plan", help="a JSON file {layers: {layer: {matrix: bits}}, head: bits} instead of --rate")
    ap.add_argument("--capacity", type=int, default=8192)
    ap.add_argument("--seqs", type=int, default=1, help="sequences of each category to score (0: only time it)")
    ap.add_argument("--same", help="tag of an eval_codes --keep run with this plan, to compare with")
    E.add_int8_option(ap)
    args = ap.parse_args()
    int8 = E.use_int8_option(args)
    cap_vram()
    ck = Checkpoint(ckpt("orig"))
    cfg = ck.cfg
    plan, head, tag = plan_of(args, cfg)
    tag += "_keyless" if args.keyless else ""
    tag += f"_int8_{int8:g}" if int8 else ""
    t = time.time()
    eng = E.Engine.load(args.name, plan, head, args.capacity, keyless=args.keyless)
    torch.cuda.synchronize()
    report = {"codes": args.name, "tag": tag, "bits_per_weight": bits_per_weight(cfg, plan), "head_bits_per_weight": head / 2,
              "capacity": args.capacity, "load_seconds": round(time.time() - t, 1), "engine_gib": round(eng.bytes() / 2**30, 2),
              "vram_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}
    print(f"[engine] {tag}: loaded in {report['load_seconds']} s, {report['engine_gib']} GiB", flush=True)
    ids, category, names, target = load_set(args.set)

    # the outputs over whole sequences, eight tokens a pass
    rows = [i for c in range(len(names)) for i in [j for j, cj in enumerate(category) if cj == c][: args.seqs]]
    n = ids.shape[1]
    hidden = torch.empty(len(rows), n, cfg.hidden)
    sample = []  # (sequence, positions, the engine's logits there): the head's kernel against PyTorch
    t = time.time()
    for r, s in enumerate(rows):
        eng.crop(0)
        seq = ids[s].tolist()
        for a in range(0, n, eng.rows):
            logits, _ = eng.feed(seq[a : a + eng.rows])
            hidden[r, a : a + eng.rows] = eng.h[: eng.rows].cpu()
            if a == n - eng.rows:
                sample.append((r, a, logits.cpu()))
    torch.cuda.synchronize()
    if rows:
        report["scored_tokens"], report["scored_seconds"] = len(rows) * n, round(time.time() - t, 1)
        print(f"[engine] {len(rows) * n} tokens in {time.time() - t:.0f} s ({len(rows) * n / (time.time() - t):.0f} tok/s, eight a pass, moving outputs to RAM)", flush=True)

    # speed: one token or eight a pass, at several context lengths
    report["pass_ms"] = {}
    filler = ids.reshape(-1).tolist()
    eng.crop(0)
    for at in (16, 1024, 2048, 4096, args.capacity - 64):
        if at > args.capacity - 16:
            continue
        while eng.length < at:
            eng.feed(filler[eng.length : min(at, eng.length + 64)])
        timed(eng, filler, 8, 3)
        one, eight = timed(eng, filler, 1, 30), timed(eng, filler, 8, 30)
        report["pass_ms"][at] = {"one": round(one, 2), "eight": round(eight, 2)}
        print(f"[engine] at {at:5d}: a pass of one token {one:.1f} ms ({1000 / one:.1f} tok/s without drafts), of eight {eight:.1f} ms", flush=True)

    if rows:
        del eng
        torch.cuda.empty_cache()
        norm = ck.norm(DEV).float()
        test_embed = C.load_head(args.name, head)
        # the head's kernel: the engine's logits against PyTorch's from the engine's own last outputs
        worst = 0.0
        for r, a, logits in sample:
            want = M.logits_of(cfg, hidden[r, a : a + logits.shape[0]].to(DEV), norm, test_embed)
            worst = max(worst, ((logits.to(DEV) - want).abs().amax(-1) / want.abs().amax(-1)).max().item())
        report["head_kernel_worst"] = worst
        print(f"[engine] head kernel against PyTorch: worst difference {worst:.1e} of the largest logit", flush=True)
        report["against"] = {}
        refs = [("orig_fp32", f"{args.set}_orig_fp32.pt"), ("orig", f"{args.set}_orig.pt")]
        if args.same:
            refs.insert(0, ("same_codes_pytorch", f"{args.set}_codes_{args.name}_{args.same}.pt"))
        sub_cat = [category[s] for s in rows]
        for ref_name, file in refs:
            ref = torch.load(os.path.join(DATA, file), weights_only=True)["hidden"][rows]
            if ref_name == "same_codes_pytorch":
                ref_embed, ref_norm = test_embed, norm
            else:
                embed = ck.embed(DEV)
                ref_embed, ref_norm = (embed.float(), norm) if ref_name == "orig_fp32" else (embed, ck.norm(DEV))
                del embed
            each = []
            result = compare(cfg, ref, hidden, ref_norm, ref_embed, ids[rows], target[rows], test_norm=norm, test_embed=test_embed, per_token=each)
            report["against"][ref_name] = by_category(result, sub_cat, names)
            a = report["against"][ref_name]
            each = torch.cat(each).float()
            a["all"] |= {f"kl_{q}": float(v) for q, v in zip(("p50", "p90", "p99", "p999"), torch.quantile(each, torch.tensor([0.5, 0.9, 0.99, 0.999])))}
            a["all"]["kl_max"] = float(each.max())
            print(f"[engine] against {ref_name}: per-token KL p90 {a['all']['kl_p90']:.4f} p99 {a['all']['kl_p99']:.4f} p99.9 {a['all']['kl_p999']:.4f} "
                  f"max {a['all']['kl_max']:.3f}", flush=True)
            print(f"[engine] against {ref_name}: KL {a['all']['kl']:.4f}, top1 {a['all']['top1']:.4f}, ppl {a['all']['ppl']:.3f} (reference {a['all']['ppl_ref']:.3f}) | "
                  + " ".join(f"{c} {v['kl']:.4f}" for c, v in a.items() if c != "all"), flush=True)
            del ref, ref_embed
            torch.cuda.empty_cache()
    with open(os.path.join(DATA, f"engine_{args.name}_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
