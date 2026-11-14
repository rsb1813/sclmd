# What a thinking budget would cost (a server that closes a thought after a budget of tokens, so that a thinking request
# that reaches max_tokens still answers: R1 part B, not built, logs/exp/R1-api-contract): stored benchmark replies
# (scripts/bench.py) whose reply ran past a cap are cut at a budget of thinking tokens, the thought is closed as such a
# budget would close it (--close nl: '\n' then <channel|>, as the template writes a thought; bare: <channel|> alone, as
# the model itself closes 249 of the 279 thoughts of MMLU-Pro's stored replies), and the answer is written greedily
# after that. A thought shorter than the budget closes as it was stored. Scored as scripts/bench_score.py scores the task
# (at 512 answer tokens and at --answer); the score against the budget is the curve.
#
#   python -m scripts.think_budget_bench --cuts 3584,2048,1024 --seconds 300     (GPU: one call of scripts/gpu.sh)
#   python -m scripts.think_budget_bench --score                                (no GPU)
#
# A run can be stopped and started again: one line per (cut, close, item) in --out; those already there are skipped,
# and a run takes no new item after --seconds (the model's loading included), so that it fits one GPU job.
import argparse
import json
import os
import sys
import time

from sclmd import ROOT

BENCH = os.environ.get("SCLMD_BENCH", os.path.join(ROOT, "data", "bench"))  # in a worktree: SCLMD_BENCH=<main copy>/data/bench
OUT = os.path.join(ROOT, "logs", "exp", "R1-api-contract", "budget_mmlu_pro.jsonl")


def read(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def long_replies(task: str, run: int, over: int) -> list:
    """The task's stored replies (an item's last one) longer than `over` tokens, thinking included."""
    last = {r["id"]: r for r in read(os.path.join(BENCH, "replies", f"{task}.run{run}.jsonl"))}
    return [r for r in last.values() if "error" not in r and r["tokens"] > over]


def answer_of(tok, out: list, stop: tuple, opening: int, closing: int, cap: int) -> tuple[str, bool]:
    """The answer's text from the first `cap` tokens after the closed thought (a thought opened again in the answer is
    left out, as bench.py leaves thinking out) -> (text, ended by a stop token within the cap)."""
    kind, ids = "answer", []
    for token in out[:cap]:
        if token == opening or token == closing:
            kind = "think" if token == opening else "answer"
        elif token not in stop and kind == "answer":
            ids.append(token)
    ended = any(t in stop for t in out[:cap])
    return tok.decode(ids, skip_special_tokens=True), ended


def score(path: str, task: str, run: int, over: int, caps: list) -> dict:
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from transformers import AutoTokenizer

    import bench_score as BS
    from sclmd import ckpt

    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
    opening, closing = tok.convert_tokens_to_ids("<|channel>"), tok.convert_tokens_to_ids("<channel|>")
    gold = {it["id"]: it for it in read(os.path.join(BENCH, "items", f"{task}.jsonl"))}
    last = {r["id"]: r for r in read(os.path.join(BENCH, "replies", f"{task}.run{run}.jsonl"))}
    right = lambda r, text: BS.letter(text, gold[r["id"]]["options"]) == gold[r["id"]]["gold"]  # noqa: E731
    short = [r for r in last.values() if "error" not in r and r["tokens"] <= over]
    long_ = [r for r in last.values() if "error" not in r and r["tokens"] > over]
    right_short, right_full = sum(right(r, r["answer"]) for r in short), sum(right(r, r["answer"]) for r in long_)
    recs = {}
    for r in read(path) if os.path.exists(path) else []:
        recs[(r["cut"], r.get("close", "nl"), r["id"])] = r
    report = {"task": task, "items": len(last), "long": len(long_), "right_short": right_short, "right_long_unlimited": right_full,
              "score_unlimited": round(100 * (right_short + right_full) / len(last), 1),
              "score_cut_at_cap_today": round(100 * right_short / len(last), 1), "cuts": {}}
    for cut, close in sorted({(c, k) for c, k, _ in recs}):
        rows = [recs[(cut, close, r["id"])] for r in long_ if (cut, close, r["id"]) in recs]
        entry = {"done": len(rows), "forced": sum(x["forced"] for x in rows)}
        for cap in caps:
            k = ended = 0
            for x in rows:
                text, end = answer_of(tok, x["out"], stop, opening, closing, cap)
                k += right(x, text)
                ended += end
            lo, hi = BS.wilson(k, len(rows))
            entry[f"answer_{cap}"] = {"right": k, "of": len(rows), "pct": round(100 * k / max(len(rows), 1), 1), "low": round(100 * lo, 1),
                                      "high": round(100 * hi, 1), "answer_ended": ended,
                                      "task_score": round(100 * (right_short + k) / len(last), 1) if len(rows) == len(long_) else None}
        entry["seconds"] = round(sum(x["seconds"] for x in rows), 1)
        report["cuts"][f"{cut} {close}"] = entry
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="mmlu_pro")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--over", type=int, default=4096, help="the replies longer than this (the server's default --max-tokens)")
    ap.add_argument("--cuts", default="3584,2048,1024", help="thinking budgets in tokens, with commas, in the order to run")
    ap.add_argument("--close", choices=("nl", "bare"), default="nl", help="how a cut thought is closed: a newline then <channel|>, or <channel|> alone")
    ap.add_argument("--answer", type=int, default=1024, help="answer tokens written after the close (scored at 512 and at this)")
    ap.add_argument("--seconds", type=float, default=300, help="no new item after this many seconds since the start")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--plan", default=os.path.join(ROOT, "data", "plans", "fit_own_v1_11.9.json"), help="the plan that wrote the stored replies")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--capacity", type=int, default=8192)
    ap.add_argument("--reserve", type=float, default=0.4)
    ap.add_argument("--score", action="store_true", help="score what is in --out (no GPU)")
    args = ap.parse_args()
    if args.score:
        print(json.dumps(score(args.out, args.task, args.run, args.over, sorted({512, args.answer})), indent=1))
        return
    t0 = time.time()
    cuts = [int(c) for c in args.cuts.split(",")]
    items = {it["id"]: it for it in read(os.path.join(BENCH, "items", f"{args.task}.jsonl"))}
    replies = long_replies(args.task, args.run, args.over)
    done = {(r["cut"], r.get("close", "nl"), r["id"]) for r in read(args.out)} if os.path.exists(args.out) else set()
    todo = [(c, r) for c in cuts for r in replies if (c, args.close, r["id"]) not in done]
    print(f"[budget] {args.task}: {len(replies)} replies over {args.over} tokens; {len(todo)} of {len(cuts) * len(replies)} "
          f"(cut, item) to do, closed '{args.close}'", flush=True)
    if not todo:
        return
    import torch
    from transformers import AutoTokenizer

    from sclmd import ckpt
    from sclmd import engine as E
    from sclmd import plan as P
    from sclmd.generate import Picker, extend
    from sclmd.stream import cap_vram
    from scripts.chat import DEPTH

    cap_vram(margin=int(args.reserve * 2**30))
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    plan, head = P.read(args.plan)
    eng = E.Engine.load(args.name, plan, head or 8, args.capacity, rows=args.rows)
    drafter = E.EngineDrafter.load(eng, ckpt("drafter"), args.name, depth=DEPTH)
    stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
    opening, closing = tok.convert_tokens_to_ids("<|channel>"), tok.convert_tokens_to_ids("<channel|>")
    nl = tok.encode("\n", add_special_tokens=False)
    assert len(nl) == 1
    print(f"[budget] loaded {os.path.basename(args.plan)} in {time.time() - t0:.0f} s, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for n, (cut, r) in enumerate(todo):
        if time.time() - t0 > args.seconds:
            print(f"[budget] {args.seconds:.0f} s are up: {len(todo) - n} left for the next run", flush=True)
            break
        it = items[r["id"]]
        messages = ([{"role": "system", "content": it["system"]}] if it.get("system") else []) + [{"role": "user", "content": it["prompt"]}]
        got = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True)
        prompt = list(got["input_ids"] if hasattr(got, "keys") else got)
        think = tok.encode(r["think"], add_special_tokens=False)  # "thought\n..." as bench.py stored it
        forced = len(think) > cut
        close = (nl + [closing] if args.close == "nl" else [closing]) if forced else [closing]  # unforced: as it was stored
        full = prompt + [opening] + think[:cut] + close
        t = time.perf_counter()
        with torch.no_grad():
            eng.crop(0)
            logits, hidden = eng.feed(full, keep="last")
            stats = {}
            out = extend(eng, len(full), logits[-1], hidden[-1], min(args.answer, eng.capacity - len(full) - eng.rows), Picker(0.0),
                         drafter.guess, stop, stats)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        text, ended = answer_of(tok, out, stop, opening, closing, len(out))
        rec = {"id": r["id"], "cut": cut, "close": args.close, "forced": forced, "think_tokens": min(len(think), cut),
               "prompt_tokens": len(full), "out": out, "tokens": len(out), "ended": ended, "seconds": round(seconds, 2), "answer": text}
        with open(args.out, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[budget] {n + 1}/{len(todo)} cut {cut} {r['id']}: {len(full)} in, {len(out)} out{' (ended)' if ended else ''}, "
              f"{seconds:.1f} s; ...{text[-50:]!r}", flush=True)


if __name__ == "__main__":
    main()
