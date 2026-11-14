# One table of speed against loss for the recipes in data/recipes.json (SAFE / BALANCED / AGGRESSIVE, today's exact
# behaviour first), from what is stored (no GPU):
#   task_eval v1 and v2: each recipe's run compared PAIRED with its baseline AND with today's chat (chrF difference and
#     its 95% interval by sentence, exact items lost / won, numbers kept), scripts/task_eval.py --compare's arithmetic
#   task_load_bench: the time to the first token and of a request at 50 / 500 / 2,000-token prompts, one client and 16
#   measured numbers the recipe file carries by hand, each with where it came from: KL on own_v2, speed_check tok/s
# A cell that has not been measured says so ("-"); nothing here is an estimate.
#   python -m scripts.recipe_table                 (markdown on the screen)
#   python -m scripts.recipe_table --out FILE.md
import argparse
import json
import os

from scripts.task_eval import FOLDER, ROOT, chrf_counts, files, load, mcnemar, numbers_kept, paired_bootstrap, right


def compare(a_tag: str, b_tag: str, items_set: str = "v1") -> dict | None:
    """B against A on the items both answered: chrF difference and interval per translation kind, exact items lost / won."""
    items_path, pa, _ = files(items_set, a_tag)
    pb = files(items_set, b_tag)[1]
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None
    with open(items_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    a, b = load(a_tag, items_set), load(b_tag, items_set)
    both = [it for it in items if it["id"] in a and it["id"] in b]
    out, lost, won, n_exact, nl, nw = {}, 0, 0, 0, 0, 0
    for kind in dict.fromkeys(it["kind"] for it in both):
        rows = [it for it in both if it["kind"] == kind]
        if kind.startswith("tr_"):
            ca = [chrf_counts(a[it["id"]]["answer"].strip(), it["gold"]) for it in rows]
            cb = [chrf_counts(b[it["id"]]["answer"].strip(), it["gold"]) for it in rows]
            out[kind] = paired_bootstrap(ca, cb)
            for it in rows:
                x, y = numbers_kept(it, a[it["id"]]["answer"]), numbers_kept(it, b[it["id"]]["answer"])
                if x is not None:
                    nl += x > y
                    nw += x < y
        else:
            ra = [right(it, a[it["id"]]["answer"]) for it in rows]
            rb = [right(it, b[it["id"]]["answer"]) for it in rows]
            lost += sum(x > y for x, y in zip(ra, rb))
            won += sum(x < y for x, y in zip(ra, rb))
            n_exact += len(rows)
            out[kind] = {"right_b": sum(rb) / len(rb)}
    out["exact"] = {"n": n_exact, "lost": lost, "won": won, "p": mcnemar(lost, won)}
    out["numbers"] = {"lost": nl, "won": nw}
    out["items"] = len(both)
    return out


def chrf_cell(c: dict | None, kind: str) -> str:
    if not c or kind not in c:
        return "-"
    x = c[kind]
    return f"{x['diff']:+.2f} [{x['diff_ci'][0]:+.2f}, {x['diff_ci'][1]:+.2f}]"


def exact_cell(c: dict | None) -> str:
    if not c:
        return "-"
    e = c["exact"]
    return f"-{e['lost']} / +{e['won']} of {e['n']}"


def load_cell(tag: str | None, size: int, clients: int, what: str = "latency_p50") -> str:
    if not tag:
        return "-"
    path = os.path.join(ROOT, "data", "task_load_bench", f"{tag}.json")
    if not os.path.exists(path):
        return "-"
    with open(path, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    for c in cells:
        if c["size"] == size and c["clients"] == clients:
            return f"{c[what]:.2f} s"
    return "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipes", default=os.path.join(ROOT, "data", "recipes.json"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    with open(args.recipes, encoding="utf-8") as f:
        spec = json.load(f)
    today = spec["today"]
    lines = ["| Recipe | Options | en->ko chrF vs base | ko->en chrF vs base | en->ko vs today | ko->en vs today | exact v1 vs today "
             "| exact v2 vs today | numbers lost/won vs today | KL own_v2 | chat tok/s | request 50 / 500 / 2,000 tok, 1 client | first token 2,000 tok, 16 clients |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in spec["recipes"]:
        base = compare(r["baseline"], r["task"]) if r.get("task") and r.get("baseline") and r["baseline"] != r["task"] else None
        vs_today = compare(today, r["task"]) if r.get("task") and r["task"] != today else None
        v2 = compare(spec.get("today_v2", today), r["task_v2"], "v2") if r.get("task_v2") and r["task_v2"] != spec.get("today_v2") else None
        nums = f"-{vs_today['numbers']['lost']} / +{vs_today['numbers']['won']}" if vs_today else "-"
        lat = " / ".join(load_cell(r.get("load"), s, 1) for s in (50, 500, 2000))
        lines.append(f"| {r['name']} | {r['options']} | {chrf_cell(base, 'tr_en_ko')} | {chrf_cell(base, 'tr_ko_en')} | "
                     f"{chrf_cell(vs_today, 'tr_en_ko')} | {chrf_cell(vs_today, 'tr_ko_en')} | {exact_cell(vs_today)} | {exact_cell(v2)} | {nums} | "
                     f"{r.get('kl', '-')} | {r.get('chat_tok_s', '-')} | {lat} | {load_cell(r.get('load'), 2000, 16, 'ttft_p50')} |")
    notes = [f"- {r['name']}: {r['note']}" for r in spec["recipes"] if r.get("note")]
    text = "\n".join(lines + [""] + notes) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


if __name__ == "__main__":
    main()
