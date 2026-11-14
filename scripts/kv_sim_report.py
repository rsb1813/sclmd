# The K3 simulation's numbers in one place (no GPU): the runs of scripts/eval_codes.py --kv in --dir, their KL against
# the original and against each other, the paired differences to the int8 run with a bootstrap interval, the noise
# energies, and the brief's decision. Writes <dir>/sim.json.
#   python -m scripts.kv_sim_report --dir ../logs/exp/K3-kv4-sliding --codes v1 --tag c6
import argparse
import json
import os

import torch

MODES = ("none", "int8", "k8v4", "k4v4", "k3v3", "k4v4t0", "k8v4t0")


def paired(a: torch.Tensor, b: torch.Tensor, block: int = 64, draws: int = 2000, seed: int = 0) -> dict:
    """Mean of b - a over the same positions, with a 95% interval from a bootstrap over blocks of `block` neighbouring
    positions (neighbours are not independent)."""
    d = (b.double() - a.double())
    n = d.numel() // block * block
    blocks = d[:n].view(-1, block).mean(1)
    g = torch.Generator().manual_seed(seed)
    means = blocks[torch.randint(0, blocks.numel(), (draws, blocks.numel()), generator=g)].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {"mean": d.mean().item(), "lo95": lo, "hi95": hi}


def one_set(folder: str, set_name: str, codes: str, tag: str, suffix: str) -> dict | None:
    where = "" if set_name == "eval_v1" else set_name + "_"
    runs, tokens = {}, {}
    for m in MODES:
        name = f"{tag}_kv{m}{suffix}"
        path = os.path.join(folder, f"{where}codes_{codes}_{name}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            runs[m] = json.load(f)
        tp = os.path.join(folder, f"{set_name}_codes_{codes}_{name}_tokens.pt")
        tokens[m] = torch.load(tp, weights_only=True) if os.path.exists(tp) else {}
    if not runs:
        return None
    out = {"runs": {}, "vs_int8": {}}
    for m, r in runs.items():
        a = r["against"]
        cats = lambda d: {c: v["kl"] for c, v in d.items() if c != "all" and isinstance(v, dict) and "kl" in v}  # noqa: E731
        row = {"seconds": r["seconds"]}
        if "orig_fp32" in a:
            row |= {"kl_vs_orig_fp32": a["orig_fp32"]["all"]["kl"], "top1_vs_orig_fp32": a["orig_fp32"]["all"]["top1"],
                    "kl_vs_orig_bf16": a["orig"]["all"]["kl"], "by_category_vs_orig_fp32": cats(a["orig_fp32"]),
                    "per_token_vs_orig_fp32": r.get("per_token", {}).get("orig_fp32")}
        for ref in ("none", "int8"):
            key = f"run:{tag}_kv{ref}{suffix}"
            if key in a:
                row[f"direct_vs_{ref}"] = {"kl": a[key]["all"]["kl"], "top1": a[key]["all"]["top1"], "per_token": r["per_token"].get(key),
                                           "by_category": cats(a[key]), "top1_by_category": {c: v["top1"] for c, v in a[key].items()
                                                                                             if c != "all" and isinstance(v, dict) and "top1" in v},
                                           "sequences_all_same": a[key].get("sequences_all_same")}
        kv = r.get("kv") or {}
        if kv.get("noise"):
            row["noise_energy"] = {k: v["mean_by_bits"] for k, v in kv["noise"].items()}
            row["bytes_a_position"] = kv.get("bytes_a_position")
        out["runs"][m] = row
    if "int8" in tokens and "orig_fp32" in tokens["int8"]:
        base = tokens["int8"]["orig_fp32"]
        for m, t in tokens.items():
            if m != "int8" and "orig_fp32" in t:
                out["vs_int8"][m] = paired(base, t["orig_fp32"])
    return out


def capacity(sliding_today: int = 311, full_today: int = 612, typical: int = 2048, rows: int = 32) -> dict:
    """[computed] Conversations the default server's pool holds at once by format: the same bytes as today's static
    split (311 sliding + 612 full tiles, 2.30 GiB), re-split by serve.py's pool_for rule with each format's tile size."""
    tile, heads, window, spare = 16, 16, 1024, 16
    full_tile = 10 * tile * 4 * (512 + 4) * 2  # the ten full layers, keys stored
    layer = {8: tile * heads * (256 + 4), 4: tile * heads * (128 + 4)}  # one sliding layer's keys or values a tile

    def sliding_tile(kbits, vbits):  # 49 layers in the format, layer 58 at int8
        return 49 * (layer[kbits] + layer[vbits]) + 2 * layer[8]

    budget = sliding_today * sliding_tile(8, 8) + full_today * full_tile
    ring = -(-(window + max(spare, rows)) // tile)
    per = typical / (window + spare)
    out = {"budget_gib": budget / 2**30}
    for name, (kb, vb) in {"int8": (8, 8), "k8v4": (8, 4), "k4v4": (4, 4)}.items():
        st = sliding_tile(kb, vb)
        n = int(budget // (st + per * full_tile))
        pool = {"sliding": n, "full": int(n * per)}
        fits = {}
        for tokens in (200, 500, 1000, 2000, 8000):
            s, f = min(-(-tokens // tile), ring), -(-tokens // tile)
            fits[tokens] = min(pool["sliding"] // s, pool["full"] // f)
        out[name] = {"sliding_tile_bytes": st, "pool_tiles": pool, "conversations_at": fits,
                     "gib_a_conversation_at": {t: (min(-(-t // tile), ring) * st + -(-t // tile) * full_tile) / 2**30 for t in fits}}
    return out


def decide(own: dict) -> dict:
    """The brief's rule on own_v2: GO for kv if k4v4 - int8 <= 0.010 mean KL (against the original) and its p99
    per-token KL <= 1.5x int8's; if k4v4 - int8 > 0.020, judge k8v4 by the same bar; STOP the line if both fail."""
    runs = own["runs"]

    def bar(m):
        if m not in runs or "int8" not in runs:
            return None
        d = runs[m]["kl_vs_orig_fp32"] - runs["int8"]["kl_vs_orig_fp32"]
        p99, p99_int8 = runs[m]["per_token_vs_orig_fp32"]["p99"], runs["int8"]["per_token_vs_orig_fp32"]["p99"]
        out = {"mode": m, "delta_mean_kl": d, "interval95": own["vs_int8"].get(m), "p99": p99, "p99_int8": p99_int8,
               "p99_ratio": p99 / p99_int8, "mean_ok": d <= 0.010, "p99_ok": p99 <= 1.5 * p99_int8, "go": d <= 0.010 and p99 <= 1.5 * p99_int8}
        # the other reading of the p99 bar: against the same codes without rounding (the none run), where int8's own
        # p99 is small; the brief's words (KL against the original) decide, this is reported beside it
        if "direct_vs_none" in runs[m] and "direct_vs_none" in runs["int8"]:
            a, b = runs[m]["direct_vs_none"]["per_token"]["p99"], runs["int8"]["direct_vs_none"]["per_token"]["p99"]
            out["same_codes_reading"] = {"p99": a, "p99_int8": b, "ratio": a / b, "ok": a <= 1.5 * b}
        return out

    kv, v = bar("k4v4"), bar("k8v4")
    if kv is None:
        return {"verdict": "incomplete"}
    if kv["go"]:
        verdict = "GO: --kv4 kv (k4v4)"
    elif v is not None and v["go"]:
        verdict = "GO: --kv4 v only (k8v4)" + ("" if kv["delta_mean_kl"] > 0.020 else " (k4v4 between the brief's 0.010 and 0.020: the brief leaves this open)")
    else:
        verdict = "STOP"
    # outside the brief's format: the sequence's first tile (positions 0-15) kept at int8, judged by the same bar
    variants = {m: bar(m) for m in ("k4v4t0", "k8v4t0") if m in runs}
    return {"verdict": verdict, "k4v4": kv, "k8v4": v, "first_tile_int8_variants": variants}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--codes", default="v1")
    ap.add_argument("--tag", default="c6")
    ap.add_argument("--sets", default="own_v2,eval_v1:_s1,task_tf_short,task_tf_gsm8k",
                    help="sets to gather, each with the suffix its runs' tags carry (eval_v1:_s1: run with --seqs 1)")
    args = ap.parse_args()
    report = {}
    for spec in args.sets.split(","):
        set_name, _, suffix = spec.partition(":")
        r = one_set(args.dir, set_name, args.codes, args.tag, suffix)
        if r:
            report[set_name + suffix] = r
    if "own_v2" in report:
        report["decision"] = decide(report["own_v2"])
    report["capacity_computed"] = capacity()
    with open(os.path.join(args.dir, "sim.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for s, r in report.items():
        if s in ("decision", "capacity_computed"):
            continue
        print(f"== {s}")
        for m, row in r["runs"].items():
            pt = row.get("per_token_vs_orig_fp32") or {}
            line = (f"  {m:5s} KL vs orig {row['kl_vs_orig_fp32']:.5f} top1 {row['top1_vs_orig_fp32']:.4f} p99 {pt.get('p99', float('nan')):.4f}"
                    if "kl_vs_orig_fp32" in row else f"  {m:5s}")
            for ref in ("none", "int8"):
                if f"direct_vs_{ref}" in row:
                    dd = row[f"direct_vs_{ref}"]
                    line += f" | vs {ref}: KL {dd['kl']:.5f} top1 {dd['top1']:.4f} p99 {dd['per_token']['p99']:.4f}"
            if m in r["vs_int8"]:
                p = r["vs_int8"][m]
                line += f" | minus int8 {p['mean']:+.5f} [{p['lo95']:+.5f}, {p['hi95']:+.5f}]"
            print(line)
            if "noise_energy" in row:
                print(f"        noise energy {row['noise_energy']}, {row['bytes_a_position']:,} B a position")
    cap = report["capacity_computed"]
    for name in ("int8", "k8v4", "k4v4"):
        c = cap[name]
        print(f"capacity [computed] {name}: pool {c['pool_tiles']}, conversations at 200/500/1000/2000/8000 tokens "
              + "/".join(str(v) for v in c["conversations_at"].values()) + ", GiB each " + "/".join(f"{v:.3f}" for v in c["gib_a_conversation_at"].values()))
    if "decision" in report:
        print("decision:", json.dumps(report["decision"], indent=1))


if __name__ == "__main__":
    main()
