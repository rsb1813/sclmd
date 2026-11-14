# The task check: what a setting of the engine does to the work the model is meant for here, translation between English
# and Korean and simple processing (scripts/task_eval_prepare.py makes the items). Every item is answered once, greedily
# (the same reply every time), without thinking, through the chat's path (scripts/bench.py: Replier), and scored:
#   tr_en_ko, tr_ko_en  chrF against the human translation (character n-grams up to 6, recall weighted twice: 0 to 100)
#   extract, lists      the reply's JSON is exactly the expected one
#   copy                the sentence in upper case, exactly; and how alike the reply is when it is not
#   gsm8k               the final number
# and the speed is shown beside it (tokens a second while writing, prompts included in the time).
#   python -m scripts.task_eval --tag chat                       (the chat's plan)
#   python -m scripts.task_eval --plan data/plans/c6.json --tag c6
#   python -m scripts.task_eval --compare chat c6                (no GPU: two stored runs side by side, item by item)
import argparse
import difflib
import json
import os
import re
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.join(ROOT, "data", "task_eval")


def chrf(pairs: list, order: int = 6, beta: float = 2.0) -> float:
    """chrF of (reply, reference) pairs as one corpus: for n = 1..order the character n-grams (spaces left out) that
    match, over those of the replies (precision) and of the references (recall); the two are averaged over n and
    joined with recall weighted beta times. 0 to 100. (Its own implementation: close to sacreBLEU's chrF, not it.)"""
    match, hyp, ref = [0] * order, [0] * order, [0] * order
    for h, r in pairs:
        h, r = "".join(h.split()), "".join(r.split())
        for n in range(1, order + 1):
            hc, rc = Counter(h[i : i + n] for i in range(len(h) - n + 1)), Counter(r[i : i + n] for i in range(len(r) - n + 1))
            match[n - 1] += sum((hc & rc).values())
            hyp[n - 1] += sum(hc.values())
            ref[n - 1] += sum(rc.values())
    p = [m / h for m, h in zip(match, hyp) if h]
    r = [m / x for m, x in zip(match, ref) if x]
    if not p or not r:
        return 0.0
    p, r = sum(p) / len(p), sum(r) / len(r)
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r) if p + r else 0.0


def first_json(text: str, opener: str):
    """The first JSON value in the reply that starts with `opener` ({ or [), or None."""
    closer = "}" if opener == "{" else "]"
    for a in (i for i, c in enumerate(text) if c == opener):
        for b in (i for i, c in enumerate(text) if c == closer and i > a):
            try:
                return json.loads(text[a : b + 1])
            except ValueError:
                continue
    return None


def right(item: dict, answer: str) -> float | None:
    """1 or 0 for the kinds with one right answer; None for translations (scored together)."""
    kind, gold = item["kind"], item["gold"]
    if kind == "extract":
        return float(first_json(answer, "{") == gold)
    if kind == "lists":
        return float(first_json(answer, "[") == gold)
    if kind == "copy":
        return float(answer.strip().strip('"') == gold)
    if kind == "gsm8k":
        found = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", answer)
        return float(bool(found) and found[-1].replace(",", "").rstrip(".").removesuffix(".00") == gold)
    return None


def summary(items: list, replies: dict) -> dict:
    out = {}
    for kind in dict.fromkeys(it["kind"] for it in items):
        rows = [(it, replies[it["id"]]) for it in items if it["kind"] == kind and it["id"] in replies]
        if not rows:
            continue
        if kind.startswith("tr_"):
            out[kind] = {"n": len(rows), "chrf": round(chrf([(r["answer"].strip(), it["gold"]) for it, r in rows]), 2)}
        else:
            out[kind] = {"n": len(rows), "right": round(sum(right(it, r["answer"]) for it, r in rows) / len(rows), 4)}
            if kind == "copy":
                out[kind]["alike"] = round(sum(difflib.SequenceMatcher(None, r["answer"].strip(), it["gold"]).ratio() for it, r in rows) / len(rows), 4)
    done = [r for r in replies.values() if "tokens" in r]
    out["all"] = {"items": len(done), "tokens": sum(r["tokens"] for r in done), "seconds": round(sum(r["seconds"] for r in done), 1),
                  "tok_s": round(sum(r["tokens"] for r in done) / max(sum(r["seconds"] for r in done), 1e-9), 1),
                  "per_pass": round(sum(r["per_pass"] * r["tokens"] for r in done) / max(sum(r["tokens"] for r in done), 1), 2),
                  "not_ended": sum(not r["ended"] for r in done), "errors": sum("error" in r for r in replies.values())}
    return out


def show(tag: str, s: dict) -> None:
    parts = [f"{k} chrF {v['chrf']}" if "chrf" in v else f"{k} {100 * v['right']:.1f}%" + (f" (alike {100 * v['alike']:.1f}%)" if "alike" in v else "")
             for k, v in s.items() if k != "all"]
    a = s["all"]
    print(f"[task] {tag}: " + " | ".join(parts) + f" | {a['tok_s']} tok/s, {a['per_pass']} tokens a pass, {a['items']} items"
          + (f", {a['not_ended']} not ended" if a["not_ended"] else "") + (f", {a['errors']} errors" if a["errors"] else ""), flush=True)


def load(tag: str) -> dict:
    with open(os.path.join(FOLDER, f"replies_{tag}.jsonl"), encoding="utf-8") as f:
        return {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="chat", help="the run's name (its replies and summary are stored under it)")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--plan", default=os.path.join(ROOT, "data", "plans", "fit_own_v1_11.9.json"))
    ap.add_argument("--capacity", type=int, default=4096)
    ap.add_argument("--reserve", type=float, default=0.4)
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--every", type=int, default=1, help="answer every Nth item only (a quick look)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="two stored runs side by side (no GPU)")
    args = ap.parse_args()
    with open(os.path.join(FOLDER, "items.jsonl"), encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    if args.compare:
        a, b = (load(t) for t in args.compare)
        both = [it for it in items if it["id"] in a and it["id"] in b]
        for tag, rep in zip(args.compare, (a, b)):
            show(tag, summary(both, rep))
        same = sum(a[it["id"]]["answer"] == b[it["id"]]["answer"] for it in both)
        print(f"[task] the same reply to {same} of {len(both)} items")
        for kind in dict.fromkeys(it["kind"] for it in both):
            rows = [it for it in both if it["kind"] == kind]
            if kind.startswith("tr_"):
                each = [(chrf([(a[it["id"]]["answer"], it["gold"])]), chrf([(b[it["id"]]["answer"], it["gold"])])) for it in rows]
                worse = sum(y < x - 10 for x, y in each)
                print(f"[task] {kind}: {args.compare[1]} is more than 10 chrF under {args.compare[0]} on {worse} of {len(rows)} sentences, "
                      f"over by more than 10 on {sum(y > x + 10 for x, y in each)}")
            else:
                lost = [it["id"] for it in rows if right(it, a[it["id"]]["answer"]) > right(it, b[it["id"]]["answer"])]
                won = [it["id"] for it in rows if right(it, a[it["id"]]["answer"]) < right(it, b[it["id"]]["answer"])]
                print(f"[task] {kind}: right in {args.compare[0]} and wrong in {args.compare[1]}: {len(lost)} {lost[:6]}; the other way: {len(won)}")
        return

    import torch

    from sclmd.stream import cap_vram
    from scripts.bench import Replier

    items = items[:: args.every]
    path = os.path.join(FOLDER, f"replies_{args.tag}.jsonl")
    replies = load(args.tag) if os.path.exists(path) else {}
    todo = [it for it in items if it["id"] not in replies]
    print(f"[task] {args.tag}: {len(todo)} of {len(items)} items to answer", flush=True)
    if todo:
        cap_vram(margin=int(args.reserve * 2**30))
        rep = Replier(args.name, args.plan, args.capacity)
        print(f"[task] loaded {os.path.basename(args.plan)}, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
        t0 = time.time()
        for n, item in enumerate(todo):
            rec = rep.reply(item, 0, False, {"temperature": 0.0, "top_p": 1.0, "top_k": 1}, args.max_new)
            replies[item["id"]] = rec
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (n + 1) % 50 == 0:
                print(f"[task] {n + 1}/{len(todo)} after {time.time() - t0:.0f} s", flush=True)
    s = summary(items, replies)
    with open(os.path.join(FOLDER, f"summary_{args.tag}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(s, f, indent=1)
    show(args.tag, s)


if __name__ == "__main__":
    main()
