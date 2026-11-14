# The owner's load on a running server (scripts/serve.py): short greedy replies to task-like prompts of about 50, 500
# and 2,000 tokens, from 1, 4 and 16 clients at once. For every cell: the time to the first token (p50 / p90), the
# time of a whole request (p50 / p90), the prompt tokens and reply tokens a second of all requests together.
#   ~50 tokens:   task_eval's own items (data/task_eval/items.jsonl, translations both ways, records, lists, copying)
#   ~500, ~2,000: a text of FLORES sentences (the items' own sentences, every text its own random draw, so the server
#                 can keep only the instruction's few tokens of it) and a short task on it: translate its first sentence
#                 (4 in 5), or say how many of its sentences have numbers (those start with a numbered line of their own)
# Every reply is greedy (temperature 0) and capped (--max-tokens, 40 as the owner's short replies). The replies of the
# 50-token cell are written in task_eval's format (data/task_eval/replies_<tag>.jsonl), so the same run is scored:
#   python -m scripts.task_eval --compare <baseline> <tag>    (with --max-tokens 600 they are task_eval's own replies)
# --shared N puts the same system prompt of about N tokens in front of every request (what pinned prefixes would save).
# Stands alone (no project imports):  python scripts/task_load_bench.py --tag load_today --sizes 50,500,2000 --clients 1,4,16
import argparse
import http.client
import json
import os
import random
import threading
import time
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = os.path.join(ROOT, "data", "task_eval")
ASK_EN = "Translate the first sentence of the following text into Korean. Write only the translation.\n\n{doc}"
ASK_KO = "다음 글의 첫 문장을 영어로 번역해 줘. 번역문만 써.\n\n{doc}"
COUNT = "How many sentences of the following text contain a number written in digits? Answer with the number only.\n\n{doc}"


def chat(url: str, messages: list, max_tokens: int, out: dict) -> None:
    """One streamed greedy chat completion: text, time to the first piece, the server's counts and timings."""
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=900)
    body = {"model": "any", "messages": messages, "stream": True, "max_tokens": max_tokens, "temperature": 0.0, "top_k": 1,
            "top_p": 1.0, "seed": 0, "thinking": False, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    conn.request("POST", u.path.rstrip("/") + "/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    out.update(status=resp.status, content="", first=None, usage=None, finish=None, timings=None)
    if resp.status != 200:
        out["error"] = resp.read().decode("utf-8", "replace")[:300]
        out["seconds"] = time.perf_counter() - t0
        return
    for raw in resp:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        data = json.loads(line[6:])
        if data.get("usage"):
            out["usage"] = data["usage"]
        if data.get("timings"):
            out["timings"] = data["timings"]
        for choice in data.get("choices", []):
            piece = choice.get("delta", {}).get("content")
            if piece and out["first"] is None:
                out["first"] = time.perf_counter() - t0
            out["content"] += piece or ""
            if choice.get("finish_reason"):
                out["finish"] = choice["finish_reason"]
    out["seconds"] = time.perf_counter() - t0
    conn.close()


def sentences(items: list) -> list:
    """The FLORES sentences of the task items, both languages, with a rough token count (about 3.5 characters a token
    in English, 1.6 in Korean: from task_eval's prompts)."""
    out = []
    for it in items:
        if it["kind"].startswith("tr_"):
            for s in (it["prompt"].split("\n\n", 1)[-1], it["gold"]):
                korean = sum("가" <= c <= "힣" for c in s) > len(s) / 4
                out.append((s, len(s) / (1.6 if korean else 3.5), korean))
    return out


def requests_for(size: int, count: int, items: list, sents: list, rng: random.Random, start: int) -> list:
    """`count` requests of about `size` prompt tokens: (id, item or None, user text)."""
    if size <= 100:
        short = [it for it in items if it["kind"] != "gsm8k"]  # gsm8k's replies are long by design
        short = [it for _, it in sorted(enumerate(short), key=lambda p: (p[0] * 7919) % len(short))]  # the kinds mixed
        return [(it["id"], it, it["prompt"]) for it in (short[(start + i) % len(short)] for i in range(count))]
    out = []
    for i in range(count):
        korean = rng.random() < 0.5
        pool = [s for s in sents if s[2] == korean]
        lines, total = [f"Text {start + i} ({rng.randrange(10**8):08d})."], 0.0
        while total < size - 40:
            s = rng.choice(pool)
            lines.append(s[0])
            total += s[1]
        kind = rng.random()
        ask = COUNT if kind < 0.2 else (ASK_KO if korean else ASK_EN)
        out.append((f"load/{size}/{start + i}", None, ask.format(doc=" ".join(lines[1:]) if ask is not COUNT else "\n".join(lines))))
    return out


def pct(v: list, q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def cell(url: str, reqs: list, clients: int, max_tokens: int, system: str) -> tuple:
    """`clients` threads take the requests in turn; -> (one record a request, wall seconds)."""
    todo, lock, done = list(reqs), threading.Lock(), []

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                rid, item, text = todo.pop(0)
            out = {}
            messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
            chat(url, messages, max_tokens, out)
            out.update(id=rid, item=item)
            with lock:
                done.append(out)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return done, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--tag", required=True, help="the run's name: data/task_load_bench/<tag>.json, data/task_eval/replies_<tag>.jsonl")
    ap.add_argument("--sizes", default="50,500,2000")
    ap.add_argument("--clients", default="1,4,16")
    ap.add_argument("--requests", type=int, default=0, help="requests a cell (0: max(8, 2 x clients) at 50 tokens, max(4, clients) longer)")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--shared", type=int, default=0, help="a system prompt of about this many tokens in front of every request")
    ap.add_argument("--warm", type=int, default=2, help="requests sent first and not counted (the server's first passes)")
    args = ap.parse_args()
    with open(os.path.join(TASKS, "items.jsonl"), encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    sents = sentences(items)
    rng = random.Random(11)
    system = ""
    if args.shared:
        pick, total = [], 0.0
        while total < args.shared:
            s = rng.choice(sents)
            pick.append(s[0])
            total += s[1]
        system = "You translate between English and Korean and do small text tasks. Reference sentences:\n" + "\n".join(pick)
    sizes = [int(x) for x in args.sizes.split(",")]
    clients = [int(x) for x in args.clients.split(",")]
    if args.warm:
        cell(args.url, requests_for(50, args.warm, items, sents, rng, 10**6), 1, args.max_tokens, system)
    results, replies, start = [], {}, 0
    for size in sizes:
        for n in clients:
            count = args.requests or (max(8, 2 * n) if size <= 100 else max(4, n))
            reqs = requests_for(size, count, items, sents, rng, start)
            start += count
            done, wall = cell(args.url, reqs, n, args.max_tokens, system)
            ok = [r for r in done if r.get("status") == 200 and r.get("usage")]
            ptok = sum(r["usage"]["prompt_tokens"] for r in ok)
            cached = sum((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0) for r in ok)
            ctok = sum(r["usage"]["completion_tokens"] for r in ok)
            ttft = [r["first"] for r in ok if r["first"] is not None]
            lat = [r["seconds"] for r in ok]
            row = {"size": size, "clients": n, "requests": len(reqs), "ok": len(ok), "wall_s": round(wall, 2),
                   "prompt_tokens_mean": round(ptok / max(len(ok), 1)), "cached_tokens": cached,
                   "ttft_p50": round(pct(ttft, 0.5), 3), "ttft_p90": round(pct(ttft, 0.9), 3),
                   "latency_p50": round(pct(lat, 0.5), 3), "latency_p90": round(pct(lat, 0.9), 3),
                   "reply_tokens_mean": round(ctok / max(len(ok), 1), 1), "reply_tok_s": round(ctok / wall, 1),
                   "all_tok_s": round((ptok - cached + ctok) / wall, 1), "requests_s": round(len(ok) / wall, 2)}
            results.append(row)
            print(f"[load] {size:>5} tokens x {n:>2} at once: {row['ok']}/{row['requests']} ok, prompt {row['prompt_tokens_mean']} "
                  f"({cached} kept), first token p50 {row['ttft_p50']:.2f} s / p90 {row['ttft_p90']:.2f} s, request p50 "
                  f"{row['latency_p50']:.2f} s / p90 {row['latency_p90']:.2f} s, replies {row['reply_tok_s']} tok/s, "
                  f"{row['requests_s']} requests/s", flush=True)
            for r in done:
                if r.get("status") != 200:
                    print(f"[load]   failed {r['id']}: {r.get('error')}", flush=True)
                elif r["item"] is not None:
                    replies[r["id"]] = {"id": r["id"], "answer": r["content"], "think": "", "prompt_tokens": r["usage"]["prompt_tokens"],
                                        "tokens": r["usage"]["completion_tokens"], "ended": r["finish"] == "stop", "looped": False,
                                        "seconds": round(r["seconds"], 2), "per_pass": 0,
                                        "first_s": round(r["first"] or 0, 3), "clients": n}
    os.makedirs(os.path.join(ROOT, "data", "task_load_bench"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "task_load_bench", f"{args.tag}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"url": args.url, "max_tokens": args.max_tokens, "shared": args.shared, "cells": results}, f, indent=1)
    with open(os.path.join(TASKS, f"replies_{args.tag}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in replies.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[load] {args.tag}: {len(results)} cells, {len(replies)} task replies written", flush=True)


if __name__ == "__main__":
    main()
