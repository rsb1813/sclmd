# Load on a running server (scripts/serve.py, or any OpenAI-compatible one): N clients ask at the same moment, each a
# conversation of its own, and then each asks a follow-up. For every N: the tokens a second of all replies together and
# of each, the time to a reply's first token, and how much of the follow-up's prompt the server had kept.
# With --check every client's two requests are then asked once more, alone, and every reply that was written under load
# is compared with that: a client's prompt and seed are the same in every round and the server's choices depend only
# on the seed and the position, so the texts must be the same, whatever else the server was doing.
# With --long 2000,6000 the load is long prompts instead: a document of about that many tokens and a question, from one
# client and from four at once; what is shown is the time to the first token and the prompt tokens a second it makes
# (every document starts with a line of its own, so the server has kept nothing of it).
# Stands alone (no project imports):  python scripts/serve_bench.py --clients 1,2,4,8
import argparse
import http.client
import json
import os
import threading
import time
from urllib.parse import urlparse

PROMPTS = [
    ("Explain how a hash table handles collisions, with a short example in Python.", "Now show the same idea with open addressing."),
    ("한국의 사계절이 농사에 어떤 영향을 주는지 설명해줘.", "그중에서 벼농사에 대해 더 자세히 알려줘."),
    ("A train leaves at 9:40 and travels 210 km at 84 km/h, then waits 25 minutes and travels 96 km at 64 km/h. When does it arrive?",
     "And if the second leg were at 80 km/h?"),
    ("Write a Go function that reverses a singly linked list, and explain each line.", "Write a test for it."),
    ("바닷물이 짠 이유를 초등학생도 이해할 수 있게 설명해줘.", "그럼 강물은 왜 안 짜?"),
    ("Describe three common causes of memory leaks in long-running Python services and how to find them.", "Which tool would you start with?"),
    ("List the planets of the solar system with one interesting fact each.", "Which of them have rings?"),
    ("이진 탐색 트리와 해시 테이블의 차이를 표로 정리하고 언제 무엇을 쓰는지 알려줘.", "정렬된 순서로 꺼내야 하면 어느 쪽이 좋아?"),
    ("What is the difference between a process and a thread? Give an example where each is the better choice.", "How does the GIL change that in Python?"),
    ("Summarize how vaccines train the immune system, in about ten sentences.", "Why do some need boosters?"),
    ("피보나치 수열을 구하는 방법 세 가지를 파이썬으로 보여주고 속도를 비교해줘.", "행렬 거듭제곱 방법도 보여줘."),
    ("Explain what a database index is and when adding one makes a query slower.", "Give an example with a composite index."),
    ("Write a short story about a lighthouse keeper who finds a message in a bottle.", "Continue it for one more paragraph."),
    ("조선 시대의 과거 제도가 어떻게 운영되었는지 설명해줘.", "문과와 무과의 차이는 뭐였어?"),
    ("How does public-key cryptography let two strangers agree on a secret? Explain without formulas.", "What breaks if the numbers are too small?"),
    ("Give me a week of dinner ideas that use rice, eggs and seasonal vegetables.", "Make Wednesday's vegetarian."),
]


def chat(url: str, messages: list, max_tokens: int, think: bool, seed: int, out: dict) -> None:
    """One streamed chat completion: the reply's text and thinking, when its first token came, and the server's counts."""
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=600)
    body = {"model": "any", "messages": messages, "stream": True, "max_tokens": max_tokens, "seed": seed,
            "thinking": think, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    conn.request("POST", u.path.rstrip("/") + "/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    out.update(status=resp.status, content="", reasoning="", first=None, usage=None, finish=None)
    if resp.status != 200:
        out["error"] = resp.read().decode("utf-8", "replace")[:300]
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
            delta = choice.get("delta", {})
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece and out["first"] is None:
                out["first"] = time.perf_counter() - t0
            out["content"] += delta.get("content") or ""
            out["reasoning"] += delta.get("reasoning_content") or ""
            if choice.get("finish_reason"):
                out["finish"] = choice["finish_reason"]
    out["seconds"] = time.perf_counter() - t0
    conn.close()


def check(url: str, seen: dict, max_tokens: int, think: bool) -> dict:
    """Asks every client's two requests alone and compares the replies written under load with them.
    seen: client -> [(clients at once, first reply, second reply), ...]"""
    same = total = 0
    wrong = []
    for i in sorted(seen):
        alone = []
        messages = [{"role": "user", "content": PROMPTS[i][0]}]
        for turn in range(2):
            out = {}
            chat(url, messages, max_tokens, think, 1000 + i, out)
            if out.get("status") != 200:
                print(f"check: client {i} alone failed: {out.get('error')}", flush=True)
                return {"failed": True}
            alone.append((out["content"], out["reasoning"]))
            messages = messages + [{"role": "assistant", "content": out["content"]}, {"role": "user", "content": PROMPTS[i][1]}]
        for n, first, second in seen[i]:
            for turn, got in enumerate((first, second)):
                if turn == 1 and first != alone[0]:
                    continue  # its second request was another one: the first reply already differed (and is counted)
                total += 1
                if got == alone[turn]:
                    same += 1
                else:
                    a, b = got[0] + got[1], alone[turn][0] + alone[turn][1]
                    at = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                    wrong.append({"client": i, "turn": turn + 1, "at_once": n, "differs_at_char": at, "under_load": a[at : at + 40], "alone": b[at : at + 40]})
    print(f"check: {same} of {total} replies written under load are the text the same request gets alone", flush=True)
    for w in wrong[:8]:
        print(f"  client {w['client']} turn {w['turn']} with {w['at_once']} at once differs from character {w['differs_at_char']}: "
              f"{w['under_load']!r} / alone {w['alone']!r}", flush=True)
    return {"replies": total, "same_as_alone": same, "different": wrong}


def together(url: str, asks: list, max_tokens: int, think: bool) -> tuple[list, float]:
    """The conversations in `asks` (each a list of messages) asked at the same moment -> (each one's result, seconds)."""
    outs = [{} for _ in asks]
    threads = [threading.Thread(target=chat, args=(url, messages, max_tokens, think, 1000 + i, outs[i])) for i, messages in enumerate(asks)]
    t = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return outs, time.perf_counter() - t


def line(label: str, outs: list, seconds: float) -> dict:
    bad = [o for o in outs if o.get("status") != 200 or not o.get("usage")]
    if bad:
        print(f"{label}: {len(bad)} of {len(outs)} failed: {bad[0].get('error') or bad[0]}", flush=True)
        return {"failed": len(bad)}
    tokens = sum(o["usage"]["completion_tokens"] for o in outs)
    firsts = sorted(o["first"] for o in outs if o["first"] is not None)
    kept = sum(o["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) for o in outs)
    prompt = sum(o["usage"]["prompt_tokens"] for o in outs)
    stats = {"clients": len(outs), "tokens": tokens, "seconds": round(seconds, 2), "tok_s": round(tokens / seconds, 1),
             "tok_s_each": round(tokens / seconds / len(outs), 1), "first_token_s": round(sum(firsts) / max(len(firsts), 1), 2),
             "first_token_worst_s": round(firsts[-1], 2) if firsts else None, "prompt_tokens": prompt, "prompt_kept": kept}
    print(f"{label}: {stats['tok_s']:6.1f} tok/s in all, {stats['tok_s_each']:5.1f} each; first token after {stats['first_token_s']:.2f} s "
          f"(worst {stats['first_token_worst_s']:.2f}); prompts {prompt} tokens, {kept} of them kept by the server", flush=True)
    return stats


def long_prompts(url: str, doc: str, lengths: list, at_once: list) -> list:
    """Documents of about `lengths` tokens with a question, from each number of clients in `at_once` at the same moment."""
    probe = {}
    chat(url, [{"role": "user", "content": f"[{time.time():.3f}]\n" + doc[:2000]}], 1, False, 1, probe)
    if probe.get("status") != 200 or not probe.get("usage"):
        print(f"long prompts: the server did not answer: {probe.get('error') or probe}", flush=True)
        return []
    per_token = 2000 / probe["usage"]["prompt_tokens"]  # characters a token of this document
    report = []
    for tokens in lengths:
        for n in at_once:
            chars, tag = int(tokens * per_token), f"{time.time():.3f}"
            asks = [[{"role": "user", "content": f"[{tag}-{i}]\n" + doc[i * 997 : i * 997 + chars] + "\n\n위 글의 요지를 한 문장으로 말해 줘."}] for i in range(n)]
            outs, seconds = together(url, asks, 16, False)
            bad = [o for o in outs if o.get("status") != 200 or not o.get("usage") or o["first"] is None]
            if bad:
                print(f"{n} at once, about {tokens} tokens each: {len(bad)} of {n} failed: {bad[0].get('error') or bad[0].get('status')}", flush=True)
                continue
            prompt, firsts = sum(o["usage"]["prompt_tokens"] for o in outs), sorted(o["first"] for o in outs)
            kept = sum(o["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) for o in outs)
            stats = {"clients": n, "prompt_tokens": prompt, "prompt_kept": kept, "first_token_s": round(sum(firsts) / n, 2),
                     "first_token_worst_s": round(firsts[-1], 2), "prompt_tok_s": round((prompt - kept) / firsts[-1], 1)}
            print(f"{n} at once, {prompt // n:5d} prompt tokens each: first token after {stats['first_token_s']:.2f} s (worst {stats['first_token_worst_s']:.2f}): "
                  f"{stats['prompt_tok_s']:7.1f} prompt tokens a second in all" + (f" ({kept} tokens were kept by the server)" if kept else ""), flush=True)
            report.append(stats)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--clients", default="1,2,4,8", help="how many ask at once (up to 16)")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between a reply and the follow-up (a user reads and types)")
    ap.add_argument("--check", action="store_true", help="afterwards ask every request alone and compare the replies written under load with that")
    ap.add_argument("--out")
    ap.add_argument("--long", default="", help="prompt lengths in tokens, with commas: the load is long prompts (see above)")
    ap.add_argument("--long-clients", default="1,4", help="how many send a long prompt at once")
    ap.add_argument("--doc", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "speed_check", "doc.txt"))
    args = ap.parse_args()
    if args.long:
        with open(args.doc, encoding="utf-8") as f:
            doc = f.read()
        report = long_prompts(args.url, doc, [int(x) for x in args.long.split(",")], [int(x) for x in args.long_clients.split(",")])
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        return
    report, seen = [], {}
    for n in [int(c) for c in args.clients.split(",")]:
        asks = [[{"role": "user", "content": PROMPTS[i][0]}] for i in range(n)]
        outs, seconds = together(args.url, asks, args.max_tokens, args.think)
        first = line(f"{n:2d} at once, first turn ", outs, seconds)
        if "failed" in first:
            break
        asks = [a + [{"role": "assistant", "content": o["content"]}, {"role": "user", "content": PROMPTS[i][1]}] for i, (a, o) in enumerate(zip(asks, outs))]
        time.sleep(args.pause)
        firsts = outs
        outs, seconds = together(args.url, asks, args.max_tokens, args.think)
        report.append({"first_turn": first, "second_turn": line(f"{n:2d} at once, second turn", outs, seconds)})
        for i, (a, b) in enumerate(zip(firsts, outs)):
            seen.setdefault(i, []).append((n, (a["content"], a["reasoning"]), (b.get("content", ""), b.get("reasoning", ""))))
    if args.check and seen:
        report.append({"check": check(args.url, seen, args.max_tokens, args.think)})
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
