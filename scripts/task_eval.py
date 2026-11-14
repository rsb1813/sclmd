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


def chrf_counts(h: str, r: str, order: int = 6) -> tuple:
    """One sentence's character n-gram counts (spaces left out): (matching, of the reply, of the reference) for n = 1..order."""
    h, r = "".join(h.split()), "".join(r.split())
    match, hyp, ref = [0] * order, [0] * order, [0] * order
    for n in range(1, order + 1):
        hc, rc = Counter(h[i : i + n] for i in range(len(h) - n + 1)), Counter(r[i : i + n] for i in range(len(r) - n + 1))
        match[n - 1] = sum((hc & rc).values())
        hyp[n - 1] = sum(hc.values())
        ref[n - 1] = sum(rc.values())
    return match, hyp, ref


def chrf(pairs: list, order: int = 6, beta: float = 2.0) -> float:
    """chrF of (reply, reference) pairs as one corpus: for n = 1..order the character n-grams (spaces left out) that
    match, over those of the replies (precision) and of the references (recall); the two are averaged over n and
    joined with recall weighted beta times. 0 to 100. (Its own implementation: close to sacreBLEU's chrF, not it.)"""
    return chrf_score([chrf_counts(h, r, order) for h, r in pairs], beta=beta)


def chrf_score(counts: list, pick=None, beta: float = 2.0) -> float:
    """Corpus chrF from per-sentence counts (chrf_counts), over the sentences `pick` names (indices, repeats allowed)."""
    order = len(counts[0][0]) if counts else 6
    match, hyp, ref = [0] * order, [0] * order, [0] * order
    for i in range(len(counts)) if pick is None else pick:
        m, h, r = counts[i]
        for n in range(order):
            match[n] += m[n]
            hyp[n] += h[n]
            ref[n] += r[n]
    p = [m / h for m, h in zip(match, hyp) if h]
    r = [m / x for m, x in zip(match, ref) if x]
    if not p or not r:
        return 0.0
    p, r = sum(p) / len(p), sum(r) / len(r)
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r) if p + r else 0.0


def paired_bootstrap(ca: list, cb: list, resamples: int = 1000, seed: int = 0) -> dict:
    """The corpus chrF difference B - A and its 95% interval by a paired bootstrap: the same sentences (drawn with
    replacement) are scored for both runs, `resamples` times. Also each run's own 95% interval."""
    import random

    rng = random.Random(seed)
    n = len(ca)
    diffs, sa, sb = [], [], []
    for _ in range(resamples):
        pick = [rng.randrange(n) for _ in range(n)]
        a, b = chrf_score(ca, pick), chrf_score(cb, pick)
        diffs.append(b - a)
        sa.append(a)
        sb.append(b)

    def ci(v):
        v = sorted(v)
        return round(v[int(0.025 * len(v))], 2), round(v[int(0.975 * len(v)) - 1], 2)

    return {"diff": round(chrf_score(cb) - chrf_score(ca), 2), "diff_ci": ci(diffs), "a_ci": ci(sa), "b_ci": ci(sb),
            "diff_sd": round((sum((d - sum(diffs) / len(diffs)) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5, 2)}


def mcnemar(lost: int, won: int) -> float:
    """Two-sided exact McNemar p: of the lost + won items that changed, how unlikely a split this uneven is at 50/50."""
    from math import comb

    n, k = lost + won, min(lost, won)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


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
    if kind.startswith("numbers_"):  # v2: every number of the source sentence (all have two or more digits) is in the reply
        return float(not (numbers(gold) - numbers(answer)))
    if kind == "gsm8k":
        found = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", answer)
        return float(bool(found) and found[-1].replace(",", "").rstrip(".").removesuffix(".00") == gold)
    return None


def numbers(text: str) -> Counter:
    """The numbers written in digits, as a multiset, written one way: thousands commas dropped (1,000 -> 1000), leading
    zeros dropped (07 -> 7), zero decimals dropped (12.00 -> 12)."""
    out = Counter()
    for m in re.findall(r"\d[\d,.]*", text):
        m = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", m).rstrip(".,")
        m = re.sub(r"\.0+$", "", m)
        out[m.lstrip("0") or "0"] += 1
    return out


def numbers_kept(item: dict, answer: str) -> float | None:
    """For a translation whose SOURCE sentence has numbers of two or more digits: 1 if every one of them is in the reply
    (as often), else 0. The source, not the human reference: FLORES's references convert units and spell numbers out
    (29 3/4 inches -> 75.6 cm). One-digit numbers are left out: they are spelled out and turned into month names
    (9월 -> September) by right translations. A paired measure: it is not 100% for the original either."""
    want = Counter({k: v for k, v in numbers(item["prompt"].split("\n\n", 1)[-1]).items() if len(k.replace(".", "")) >= 2})
    return float(not (want - numbers(answer))) if want else None


def summary(items: list, replies: dict) -> dict:
    out = {}
    for kind in dict.fromkeys(it["kind"] for it in items):
        rows = [(it, replies[it["id"]]) for it in items if it["kind"] == kind and it["id"] in replies]
        if not rows:
            continue
        if kind.startswith("tr_"):
            out[kind] = {"n": len(rows), "chrf": round(chrf([(r["answer"].strip(), it["gold"]) for it, r in rows]), 2)}
            kept = [x for x in (numbers_kept(it, r["answer"]) for it, r in rows) if x is not None]
            if kept:
                out[kind] |= {"numbers": round(sum(kept) / len(kept), 4), "numbers_n": len(kept)}
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
    parts = [f"{k} chrF {v['chrf']}" + (f" (numbers kept {100 * v['numbers']:.1f}% of {v['numbers_n']})" if "numbers" in v else "")
             if "chrf" in v else f"{k} {100 * v['right']:.1f}%" + (f" (alike {100 * v['alike']:.1f}%)" if "alike" in v else "")
             for k, v in s.items() if k != "all"]
    a = s["all"]
    print(f"[task] {tag}: " + " | ".join(parts) + f" | {a['tok_s']} tok/s, {a['per_pass']} tokens a pass, {a['items']} items"
          + (f", {a['not_ended']} not ended" if a["not_ended"] else "") + (f", {a['errors']} errors" if a["errors"] else ""), flush=True)


def files(items_set: str, tag: str) -> tuple:
    """(items, replies, summary) paths: v1 keeps today's names; another set gets its own (items_v2.jsonl, replies_v2_<tag>.jsonl)."""
    pre = "" if items_set == "v1" else f"{items_set}_"
    return (os.path.join(FOLDER, "items.jsonl" if items_set == "v1" else f"items_{items_set}.jsonl"),
            os.path.join(FOLDER, f"replies_{pre}{tag}.jsonl"), os.path.join(FOLDER, f"summary_{pre}{tag}.json"))


def load(tag: str, items_set: str = "v1") -> dict:
    """A run's stored replies; a last line cut short by a stopped run is left out (that item is answered again)."""
    out = {}
    with open(files(items_set, tag)[1], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                out[r["id"]] = r
    return out


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
    ap.add_argument("--resamples", type=int, default=1000, help="--compare: bootstrap resamples for the chrF interval")
    ap.add_argument("--out", default="", help="--compare: also write the comparison as JSON here")
    ap.add_argument("--set", dest="items_set", default="v1", help="the item set: v1 (items.jsonl) or v2 (items_v2.jsonl: larger exact kinds)")
    ap.add_argument("--rows", type=int, default=8, help="rows of the engine's pass (Engine.load's rows; 8 as the chat)")
    ap.add_argument("--budget", type=float, default=0, help="stop answering after this many seconds (0: no limit); run again to go on")
    ap.add_argument("--int8", type=float, nargs="?", const=5.0, default=0.0, metavar="SIGMA",
                    help="integer products (sclmd/engine.py add_int8_option; lossy option, off by default)")
    ap.add_argument("--kv4", action="store_true", help="4-bit keys and values of the sliding layers (sclmd/engine.py add_kv4_option; lossy option, off by default)")
    args = ap.parse_args()
    if args.int8:
        os.environ["SCLMD_INT8"] = repr(float(args.int8))
    if args.kv4:
        os.environ["SCLMD_KV4"] = "1"
    items_path, path, summary_path = files(args.items_set, args.tag)
    with open(items_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    if args.compare:
        a, b = (load(t, args.items_set) for t in args.compare)
        both = [it for it in items if it["id"] in a and it["id"] in b]
        for tag, rep in zip(args.compare, (a, b)):
            show(tag, summary(both, rep))
        same = sum(a[it["id"]]["answer"] == b[it["id"]]["answer"] for it in both)
        print(f"[task] the same reply to {same} of {len(both)} items")
        report = {}
        for kind in dict.fromkeys(it["kind"] for it in both):
            rows = [it for it in both if it["kind"] == kind]
            same_k = sum(a[it["id"]]["answer"] == b[it["id"]]["answer"] for it in rows)
            if kind.startswith("tr_"):
                each = [(chrf([(a[it["id"]]["answer"], it["gold"])]), chrf([(b[it["id"]]["answer"], it["gold"])])) for it in rows]
                worse = sum(y < x - 10 for x, y in each)
                print(f"[task] {kind}: {args.compare[1]} is more than 10 chrF under {args.compare[0]} on {worse} of {len(rows)} sentences, "
                      f"over by more than 10 on {sum(y > x + 10 for x, y in each)}; the same reply to {same_k}")
                ca = [chrf_counts(a[it["id"]]["answer"].strip(), it["gold"]) for it in rows]
                cb = [chrf_counts(b[it["id"]]["answer"].strip(), it["gold"]) for it in rows]
                bs = paired_bootstrap(ca, cb, args.resamples)
                report[kind] = {"n": len(rows), "same": same_k, **bs}
                print(f"[task] {kind}: chrF {args.compare[1]} - {args.compare[0]} = {bs['diff']:+.2f}, 95% interval "
                      f"[{bs['diff_ci'][0]:+.2f}, {bs['diff_ci'][1]:+.2f}] (paired bootstrap by sentence, {args.resamples} resamples; "
                      f"sd {bs['diff_sd']}); each alone: {args.compare[0]} {list(bs['a_ci'])}, {args.compare[1]} {list(bs['b_ci'])}")
                na = {it["id"]: numbers_kept(it, a[it["id"]]["answer"]) for it in rows}
                nb = {it["id"]: numbers_kept(it, b[it["id"]]["answer"]) for it in rows}
                has = [i for i in na if na[i] is not None]
                if has:
                    lost = [i for i in has if na[i] > nb[i]]
                    won = [i for i in has if na[i] < nb[i]]
                    report[kind]["numbers"] = {"n": len(has), "a": round(sum(na[i] for i in has) / len(has), 4),
                                               "b": round(sum(nb[i] for i in has) / len(has), 4), "lost": len(lost), "won": len(won),
                                               "mcnemar_p": round(mcnemar(len(lost), len(won)), 4), "lost_ids": lost}
                    print(f"[task] {kind}: numbers kept (source has numbers of 2+ digits: {len(has)} sentences): {args.compare[0]} "
                          f"{100 * report[kind]['numbers']['a']:.1f}%, {args.compare[1]} {100 * report[kind]['numbers']['b']:.1f}%; "
                          f"lost {len(lost)} {lost[:4]}, won {len(won)}, McNemar p = {mcnemar(len(lost), len(won)):.3g}")
            else:
                ra = {it["id"]: right(it, a[it["id"]]["answer"]) for it in rows}
                rb = {it["id"]: right(it, b[it["id"]]["answer"]) for it in rows}
                lost = [i for i in ra if ra[i] > rb[i]]
                won = [i for i in ra if ra[i] < rb[i]]
                p = mcnemar(len(lost), len(won))
                report[kind] = {"n": len(rows), "same": same_k, "both_right": int(sum(ra[i] and rb[i] for i in ra)),
                                "lost": len(lost), "won": len(won), "both_wrong": sum(not ra[i] and not rb[i] for i in ra),
                                "mcnemar_p": round(p, 4), "lost_ids": lost}
                print(f"[task] {kind}: right in {args.compare[0]} and wrong in {args.compare[1]}: {len(lost)} {lost[:6]}; the other way: {len(won)}; "
                      f"McNemar both right {report[kind]['both_right']} / lost {len(lost)} / won {len(won)} / both wrong "
                      f"{report[kind]['both_wrong']}, p = {p:.3g}; the same reply to {same_k}")
        exact = [k for k in report if "lost" in report[k]]
        lo, wo = sum(report[k]["lost"] for k in exact), sum(report[k]["won"] for k in exact)
        report["exact_all"] = {"n": sum(report[k]["n"] for k in exact), "lost": lo, "won": wo, "mcnemar_p": round(mcnemar(lo, wo), 4)}
        print(f"[task] exact kinds together: {report['exact_all']['n']} items, lost {lo}, won {wo}, McNemar p = {mcnemar(lo, wo):.3g}")
        if args.out:
            with open(args.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"a": args.compare[0], "b": args.compare[1], "same": same, "items": len(both), **report}, f, indent=1, ensure_ascii=False)
        return

    import torch

    from sclmd.stream import cap_vram
    from scripts.bench import Replier

    items = items[:: args.every]
    replies = load(args.tag, args.items_set) if os.path.exists(path) else {}
    if os.path.exists(path):
        with open(path, "rb") as f:  # a line cut short by a stopped run: end it, so the next reply starts on its own line
            size = f.seek(0, 2)
            cut = False
            if size:
                f.seek(-1, 2)
                cut = f.read(1) != b"\n"
        if cut:
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write("\n")
    todo = [it for it in items if it["id"] not in replies]
    print(f"[task] {args.tag}: {len(todo)} of {len(items)} items to answer", flush=True)
    if todo:
        cap_vram(margin=int(args.reserve * 2**30))
        rep = Replier(args.name, args.plan, args.capacity, rows=args.rows)
        print(f"[task] loaded {os.path.basename(args.plan)} ({args.rows} rows), VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
        t0 = time.time()
        for n, item in enumerate(todo):
            if args.budget and time.time() - t0 > args.budget:
                print(f"[task] {args.tag}: the budget of {args.budget:.0f} s is spent, {len(todo) - n} items left: run again to go on", flush=True)
                break
            rec = rep.reply(item, 0, False, {"temperature": 0.0, "top_p": 1.0, "top_k": 1}, args.max_new)
            replies[item["id"]] = rec
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (n + 1) % 50 == 0:
                print(f"[task] {n + 1}/{len(todo)} after {time.time() - t0:.0f} s", flush=True)
    s = summary(items, replies)
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(s, f, indent=1)
    show(args.tag, s)


if __name__ == "__main__":
    main()
