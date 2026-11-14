# R2's kill test, second half: does the server's model (every matrix at 3 bits: data/plans/c6.json) write Gemma 4's
# tool calls well enough to serve them? The 50 items of data/tool_items_v1.json (2 or 3 tools, one question each,
# BFCL-simple style, 16 in Korean): the prompt as the server writes it (the tools declared through the chat template),
# a greedy reply on the engine, its raw token ids stored; scored without the GPU by reading the calls on token ids
# (sclmd/tools.py) against the gold: the right function with right arguments, else no call, a block that does not
# parse, the wrong function or wrong arguments. Also counted: text beside a call, a call without <tool_call|>, more
# than one call, and how each reply ended.
#
#   python -m scripts.tool_call_bench --think off --seconds 300     (GPU: one call of scripts/gpu.sh; then --think on)
#   python -m scripts.tool_call_bench --score                       (no GPU)
#
# A run can be stopped and started again: one line per (think, item) in --out; those already there are skipped, and a
# run takes no new item after --seconds (the model's loading included).
import argparse
import json
import os
import re
import time

from sclmd import ROOT, ckpt
from sclmd import tools as T

ITEMS = os.path.join(ROOT, "data", "tool_items_v1.json")
OUT = os.path.join(ROOT, "logs", "exp", "R2-tool-calls", "tool_calls.jsonl")


def read(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def items_of(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for it in d["items"]:
        tools = [{"type": "function", "function": {"name": name, **d["tools"][name]}} for name in it["tools"]]
        out.append({**it, "tool_defs": tools})
    return out


def prompt_of(tok, it: dict, think: bool) -> list:
    tools = T.tools_of({"tools": it["tool_defs"]})[0]
    got = tok.apply_chat_template([{"role": "user", "content": it["question"]}], tools=tools, tokenize=True, add_generation_prompt=True,
                                  enable_thinking=think)
    return list(got["input_ids"] if hasattr(got, "keys") else got)


def norm(v):
    """A value as BFCL's checker compares it: strings lowercased, without spaces and , . / - _ * ^ ' ; numbers as floats."""
    if isinstance(v, str):
        return re.sub(r"[ ,./\-_*^'\"]", "", v.lower())
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    return v


def judge(call: dict | None, gold: dict) -> str:
    """'right', or what is wrong with the call."""
    if call is None:
        return "no call"
    if call["name"] != gold["name"]:
        return "wrong function"
    args = call["args"]
    for key, ok in gold["args"].items():
        if key not in args:
            if key not in gold["optional"]:
                return "wrong arguments"
            continue
        if norm(args[key]) not in [norm(x) for x in ok]:
            return "wrong arguments"
    if any(key not in gold["args"] for key in args):
        return "wrong arguments"
    return "right"


def score(path: str, items: list, tok) -> dict:
    stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
    opening, closing, ids = tok.convert_tokens_to_ids("<|channel>"), tok.convert_tokens_to_ids("<channel|>"), T.Ids(tok)
    gold = {it["id"]: it for it in items}
    report, details = {}, []
    by = {}
    for r in read(path) if os.path.exists(path) else []:
        by[(r["think"], r["id"])] = r
    for think in ("off", "on"):
        rows = [by[(think, it["id"])] for it in items if (think, it["id"]) in by]
        if not rows:
            continue
        counts, extra = {}, {"text_beside_call": 0, "unclosed": 0, "more_than_one_call": 0, "malformed_blocks": 0, "lenient_only": 0,
                             "ended": {}, "cut_at_length": 0}
        by_lang = {}
        for r in rows:
            it = gold[r["id"]]
            text, calls, reader = T.read_reply(tok, r["out"], stop, opening, closing, ids, False)
            first = {"name": calls[0]["function"]["name"], "args": calls[0]["args"]} if calls else None
            verdict = judge(first, it["gold"])
            if verdict == "no call" and reader.malformed:
                verdict = "malformed"
            strict = None
            if calls:  # would the strict reading (no bare words for strings) have read it?
                blocks, inside = [], None
                for t in r["out"]:
                    if t == ids.call:
                        inside = []
                    elif t == ids.end and inside is not None:
                        blocks.append(inside)
                        inside = None
                    elif inside is not None:
                        inside.append(t)
                try:
                    strict = [T.parse_call(tok, b, ids.quote, lenient=False) for b in blocks[:1]]
                except T.Malformed:
                    extra["lenient_only"] += 1
            counts[verdict] = counts.get(verdict, 0) + 1
            lang = by_lang.setdefault(it["lang"], {"right": 0, "of": 0})
            lang["of"] += 1
            lang["right"] += verdict == "right"
            extra["text_beside_call"] += bool(calls) and bool(text.strip())
            extra["unclosed"] += reader.unclosed
            extra["more_than_one_call"] += len(calls) > 1
            extra["malformed_blocks"] += len(reader.malformed)
            end = tok.convert_ids_to_tokens(r["out"][-1]) if r["out"] and r["out"][-1] in stop else "length"
            extra["ended"][end] = extra["ended"].get(end, 0) + 1
            details.append({"think": think, "id": r["id"], "verdict": verdict, "calls": [{"name": c["function"]["name"], "args": c["args"]} for c in calls],
                            "text": text.strip()[:200], "malformed": reader.malformed[:1], "tokens": len(r["out"]), "ended": end,
                            "gold": it["gold"]})
        right = counts.get("right", 0)
        report[f"think {think}"] = {"items": len(rows), "right": right, "pct": round(100 * right / len(rows), 1), "verdicts": counts,
                                    "by_lang": by_lang, **extra, "seconds": round(sum(r["seconds"] for r in rows), 1),
                                    "mean_reply_tokens": round(sum(len(r["out"]) for r in rows) / len(rows), 1)}
    return {"report": report, "details": details}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", default=ITEMS)
    ap.add_argument("--think", choices=("off", "on"), default="off")
    ap.add_argument("--max-new", type=int, default=0, help="reply tokens at most (0: 384 thinking off, 3072 on)")
    ap.add_argument("--seconds", type=float, default=300, help="no new item after this many seconds since the start")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--plan", default=os.path.join(ROOT, "data", "plans", "c6.json"), help="the server's default weights: every matrix at 3 bits")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--capacity", type=int, default=8192)
    ap.add_argument("--reserve", type=float, default=0.4)
    ap.add_argument("--score", action="store_true", help="score what is in --out (no GPU)")
    ap.add_argument("--details", help="with --score: every item's reading to this JSON file")
    args = ap.parse_args()
    import transformers
    from transformers import AutoTokenizer

    transformers.logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    items = items_of(args.items)
    if args.score:
        got = score(args.out, items, tok)
        print(json.dumps(got["report"], ensure_ascii=False, indent=1))
        if args.details:
            with open(args.details, "w", encoding="utf-8") as f:
                json.dump(got["details"], f, ensure_ascii=False, indent=1)
        return
    t0 = time.time()
    think = args.think == "on"
    max_new = args.max_new or (3072 if think else 384)
    done = {(r["think"], r["id"]) for r in read(args.out)} if os.path.exists(args.out) else set()
    todo = [it for it in items if (args.think, it["id"]) not in done]
    print(f"[tools] {len(items)} items, thinking {args.think}: {len(todo)} to do", flush=True)
    if not todo:
        return
    import torch

    from sclmd import engine as E
    from sclmd import plan as P
    from sclmd.generate import Picker, extend
    from sclmd.stream import cap_vram
    from scripts.chat import DEPTH

    cap_vram(margin=int(args.reserve * 2**30))
    plan, head = P.read(args.plan)
    eng = E.Engine.load(args.name, plan, head or 8, args.capacity, rows=args.rows)
    drafter = E.EngineDrafter.load(eng, ckpt("drafter"), args.name, depth=DEPTH)
    stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
    print(f"[tools] loaded {os.path.basename(args.plan)} in {time.time() - t0:.0f} s, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for n, it in enumerate(todo):
        if time.time() - t0 > args.seconds:
            print(f"[tools] {args.seconds:.0f} s are up: {len(todo) - n} left for the next run", flush=True)
            break
        prompt = prompt_of(tok, it, think)
        t = time.perf_counter()
        with torch.no_grad():
            eng.crop(0)
            logits, hidden = eng.feed(prompt, keep="last")
            out = extend(eng, len(prompt), logits[-1], hidden[-1], min(max_new, eng.capacity - len(prompt) - eng.rows), Picker(0.0),
                         drafter.guess, stop, {})
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        rec = {"id": it["id"], "think": args.think, "prompt_tokens": len(prompt), "out": out, "seconds": round(seconds, 2)}
        with open(args.out, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[tools] {n + 1}/{len(todo)} {it['id']}: {len(prompt)} in, {len(out)} out, {seconds:.1f} s; "
              f"{tok.decode(out[-60:], skip_special_tokens=False)!r}", flush=True)


if __name__ == "__main__":
    main()
