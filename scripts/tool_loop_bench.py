# An agent loop through a running server (scripts/serve.py with tool calls, R2): about 2,000 tokens of system prompt
# and tools, then the model calls read_page for pages 1 to 8 one at a time (parallel_tool_calls: false), each result
# about 300 tokens, then it answers. For every step: the prompt's tokens, how many of them the server had kept
# (cached_tokens), the time to the first token of the reply (streamed), and the call. --reread: each later step is also
# sent once more with a system prompt that differs in its first character, so that nothing is kept, for the time the
# step would take if the whole conversation were read again. --clients N: N loops at once (each with pages of its own);
# --check: each client's loop is then run again alone, and every step's reply must be the same (greedy).
#
#   python scripts/tool_loop_bench.py --url http://127.0.0.1:8022/v1 --steps 8 --clients 1,4 --check --reread
#
# Stands alone (no project imports).
import argparse
import http.client
import json
import threading
import time
from urllib.parse import urlparse

TOPICS = ["solar power", "river ecology", "medieval trade", "battery chemistry", "coral reefs", "urban transit", "jazz history",
          "glaciers", "seed banks", "printing presses", "volcanoes", "honey bees", "lighthouses", "paper making", "tea farming", "comets"]
POLICY = ["Answer in the language the user writes in.", "Keep answers short unless asked for detail.", "Never invent numbers; say when a page does not give one.",
          "Quote page numbers when you use what a page says.", "Use one tool call at a time and wait for its result.",
          "When a tool fails, say so and stop.", "Do not read a page twice.", "Prefer plain words to jargon.",
          "Keep the user's names and terms as they wrote them.", "If the user asks for a list, give a numbered list."]


def system_prompt(k: int, mark: str = "") -> str:
    lines = [f"{mark}You are a careful research assistant (session {k}). Rules:"]
    for i in range(60):
        lines.append(f"{i + 1}. {POLICY[i % len(POLICY)]} (rule set {i // len(POLICY) + 1}, item {i % len(POLICY) + 1})")
    return "\n".join(lines)


def tools() -> list:
    def fn(name, desc, props, req):
        return {"type": "function", "function": {"name": name, "description": desc,
                                                 "parameters": {"type": "object", "properties": props, "required": req}}}
    s = lambda d: {"type": "string", "description": d}  # noqa: E731
    i = lambda d: {"type": "integer", "description": d}  # noqa: E731
    return [fn("read_page", "Reads one page of the report the user has open. Pages are numbered from 1. Returns the page's text, "
                            "which can be long; read pages in order and only once.", {"page": i("the page number, from 1")}, ["page"]),
            fn("search_report", "Searches the whole report for a phrase and returns the page numbers where it appears, best first.",
               {"phrase": s("the words to look for"), "limit": i("how many pages at most")}, ["phrase"]),
            fn("get_metadata", "Gets the report's title, authors, date and number of pages.", {}, []),
            fn("save_note", "Saves a short note about the report for later.", {"text": s("the note"), "page": i("the page it is about")}, ["text"]),
            fn("translate_text", "Translates a text into another language.", {"text": s("the text"), "target_language": s("ISO 639-1 code")}, ["text", "target_language"]),
            fn("send_summary", "Sends the final summary to the user's email.", {"to": s("email address"), "summary": s("the summary text")}, ["to", "summary"]),
            fn("lookup_term", "Looks up a technical term in the glossary.", {"term": s("the term")}, ["term"]),
            fn("count_words", "Counts the words of a text.", {"text": s("the text")}, ["text"])]


def page(k: int, n: int) -> str:
    topic = TOPICS[k % len(TOPICS)]
    out = [f"Page {n} of the report on {topic}."]
    for j in range(10):
        out.append(f"Paragraph {j + 1}: on page {n}, the study of {topic} notes finding {k * 100 + n * 10 + j} with care and reports it plainly.")
    return " ".join(out)


def post(url: str, body: dict) -> dict:
    """A streamed request -> its reply: content, calls, finish, usage, the time to the first token and the total time."""
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    t0 = time.perf_counter()
    conn.request("POST", u.path.rstrip("/") + "/chat/completions", json.dumps({**body, "stream": True, "stream_options": {"include_usage": True}}),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        data = resp.read().decode()
        conn.close()
        return {"status": resp.status, "error": data[:300]}
    first, content, calls, finish, usage, timings = None, "", {}, None, None, {}
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        if line == b"data: [DONE]":
            break
        chunk = json.loads(line[6:])
        usage = chunk.get("usage") or usage
        timings = chunk.get("timings") or timings
        for c in chunk.get("choices", []):
            d = c["delta"]
            if first is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                first = time.perf_counter() - t0
            content += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                calls[tc["index"]] = tc
            finish = c.get("finish_reason") or finish
    conn.close()
    return {"status": 200, "content": content, "calls": [calls[k] for k in sorted(calls)], "finish": finish, "usage": usage,
            "ttft": first, "total": time.perf_counter() - t0, "server_first": (timings.get("first_token_ms") or 0) / 1000}


def loop(url: str, k: int, steps: int, think: bool, reread: bool, shared: bool = False) -> list:
    """One client's agent loop -> a record a step. shared: every client's system prompt is the same (with --prefix-gib
    the server keeps it, and the tools, for the others)."""
    sys_ = system_prompt(0 if shared else k)
    messages = [{"role": "system", "content": sys_},
                {"role": "user", "content": f"Read pages 1 to {steps} of the report one at a time with read_page, in order, then tell me in two "
                                            f"sentences what the report on {TOPICS[k % len(TOPICS)]} says."}]
    recs, read = [], 0
    for step in range(steps + 2):
        body = {"messages": messages, "tools": tools(), "parallel_tool_calls": False, "temperature": 0, "max_tokens": 600 if think else 200,
                "thinking": think}
        got = post(url, body)
        if got["status"] != 200:
            recs.append({"step": step, "error": got})
            break
        u = got["usage"] or {}
        rec = {"step": step, "prompt": u.get("prompt_tokens"), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "completion": u.get("completion_tokens"), "ttft": round(got["ttft"] or got["total"], 3), "total": round(got["total"], 3),
               "first_token": round(got["server_first"], 3),
               "finish": got["finish"], "calls": [(c["function"]["name"], c["function"]["arguments"]) for c in got["calls"]],
               "content": got["content"][:300]}
        if reread and step > 0:  # the same step with nothing kept: a system prompt that differs in its first character
            again = json.loads(json.dumps(body))
            again["messages"][0]["content"] = f"#{step} " + sys_  # a mark of its own: no re-read is kept for the next
            g2 = post(url, again)
            rec["reread_ttft"] = round(g2["ttft"] or g2["total"], 3) if g2["status"] == 200 else None
            rec["reread_first_token"] = round(g2["server_first"], 3) if g2["status"] == 200 else None
            rec["reread_cached"] = ((g2.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        recs.append(rec)
        answer = {"role": "assistant", "content": got["content"] or None}
        if got["calls"]:
            answer["tool_calls"] = [{"id": c["id"], "type": "function", "function": c["function"]} for c in got["calls"]]
        messages = messages + [answer]
        if got["calls"]:
            for c in got["calls"]:
                try:
                    n = int(json.loads(c["function"]["arguments"]).get("page", read + 1))
                except (ValueError, AttributeError, TypeError):
                    n = read + 1
                read = max(read, n)
                result = page(k, n) if c["function"]["name"] == "read_page" else "done"
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        elif read < steps:
            messages.append({"role": "user", "content": "Go on: read the next page."})
        else:
            break
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8022/v1")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--clients", default="1")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--reread", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--shared-system", action="store_true", help="the same system prompt (and tools) for every client")
    ap.add_argument("--out")
    args = ap.parse_args()
    report = {}
    for n in [int(x) for x in args.clients.split(",")]:
        res, threads = [None] * n, []
        t0 = time.perf_counter()
        for k in range(n):
            threads.append(threading.Thread(target=lambda k=k: res.__setitem__(k, loop(args.url, k, args.steps, args.think, args.reread and n == 1,
                                                                                   args.shared_system))))
            threads[-1].start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        later = [r for recs in res for r in recs if r["step"] > 0 and "error" not in r and r["prompt"]]
        share = [r["cached"] / r["prompt"] for r in later]
        firsts = [recs[0] for recs in res if recs and "error" not in recs[0]]
        entry = {"clients": n, "wall_s": round(wall, 1), "first_steps_cached": [r["cached"] for r in firsts],
                 "first_steps_first_token": [r["first_token"] for r in firsts], "steps": sum(len(r) for r in res), "errors": [r for recs in res for r in recs if "error" in r],
                 "later_steps": len(later), "cached_share_min": round(min(share), 3) if share else None,
                 "cached_share_mean": round(sum(share) / len(share), 3) if share else None,
                 "later_ttft_mean": round(sum(r["ttft"] for r in later) / len(later), 3) if later else None,
                 "later_ttft_max": round(max(r["ttft"] for r in later), 3) if later else None,
                 "later_first_token_mean": round(sum(r["first_token"] for r in later) / len(later), 3) if later else None,
                 "later_first_token_max": round(max(r["first_token"] for r in later), 3) if later else None,
                 "calls": sum(len(r["calls"]) for recs in res for r in recs), "loops": res}
        if args.reread and n == 1:
            rr = [r for r in later if r.get("reread_ttft") is not None]
            entry["reread_ttft_mean"] = round(sum(r["reread_ttft"] for r in rr) / len(rr), 3) if rr else None
            entry["reread_first_token_mean"] = round(sum(r["reread_first_token"] for r in rr) / len(rr), 3) if rr else None
            entry["reread_ttft_sum"] = round(sum(r["reread_ttft"] for r in rr), 2) if rr else None
            entry["kept_ttft_sum"] = round(sum(r["ttft"] for r in rr), 2) if rr else None
        if args.check:
            same = total = 0
            for k in range(n):
                alone = loop(args.url, k, args.steps, args.think, False, args.shared_system)
                for a, b in zip(res[k], alone):
                    total += 1
                    same += (a.get("calls"), a.get("content"), a.get("finish")) == (b.get("calls"), b.get("content"), b.get("finish"))
                total += abs(len(res[k]) - len(alone))
            entry["check"] = f"{same} of {total} steps written under load are the reply of the same step alone"
        report[n] = entry
        print(f"{n} client(s): wall {entry['wall_s']} s, {entry['steps']} steps ({entry['calls']} calls, {len(entry['errors'])} errors); later steps: "
              f"cached {entry['cached_share_min']} min / {entry['cached_share_mean']} mean of the prompt, first token {entry['later_ttft_mean']} s mean, "
              f"{entry['later_ttft_max']} s worst (the server's first token {entry['later_first_token_mean']} s mean, {entry['later_first_token_max']} s worst)" + (f"; re-read: {entry.get('reread_ttft_mean')} s mean ({entry.get('reread_ttft_sum')} s in all "
                                                      f"against {entry.get('kept_ttft_sum')} s kept)" if "reread_ttft_mean" in entry else "")
              + (f"; {entry['check']}" if "check" in entry else "")
              + f"; first steps: {entry['first_steps_cached']} kept, first token {entry['first_steps_first_token']} s", flush=True)
        for recs in res[:1]:
            for r in recs:
                print(f"   step {r['step']}: prompt {r.get('prompt')} ({r.get('cached')} kept), first chunk {r.get('ttft')} s, server's first token {r.get('first_token')} s"
                      + (f" (re-read {r.get('reread_ttft')} s, first token {r.get('reread_first_token')} s)" if r.get("reread_ttft") is not None else "")
                      + f", {r.get('finish')}, {r.get('calls') or r.get('content', '')[:80]!r}", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
