# The server with and without kept prefixes (--prefix-gib): a fixed system prompt of about 1,000 tokens (a translator's
# rules, a glossary and worked examples) before each of 60 task items, asked by 1 and by 8 clients at once; the time
# to each reply's first token and the whole run's. Starts the server itself, asks, stops it (one GPU job a mode), and
# writes every reply so that two modes can be compared:  --compare on.json off.json (no GPU).
#
#   python -m scripts.prefix_bench --mode on --out data/prefix_bench_on.json
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

from scripts.serve_bench import chat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def items_and_system(n: int, want: int):
    from transformers import AutoTokenizer

    from sclmd import ckpt
    from scripts.prefix_check import ITEMS, system

    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    with open(ITEMS, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    shots = [it for it in items if it["kind"].startswith("tr_")][-40:]
    kinds = list(dict.fromkeys(it["kind"] for it in items))
    asked, rest = [], items[:-40]
    for k in range(n):
        asked.append(next(it for it in rest if it["kind"] == kinds[k % len(kinds)] and it not in asked))
    return asked, system(shots, want, tok)


def run(url: str, system: str, items: list, clients: int, max_tokens: int) -> dict:
    """Each client asks its share of the items one after the other."""
    outs = [dict() for _ in items]

    def client(c):
        for i in range(c, len(items), clients):
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": items[i]["prompt"]}]
            t0 = time.perf_counter()
            chat(url, msgs, max_tokens, False, 0, outs[i])
            outs[i]["sent"] = t0

    t0 = time.perf_counter()
    threads = [threading.Thread(target=client, args=(c,)) for c in range(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0
    ttft = sorted(1000 * o["first"] for o in outs if o.get("first") is not None)
    pct = lambda p: round(ttft[min(len(ttft) - 1, int(p * len(ttft)))], 1)  # noqa: E731
    cached = sum((o.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) for o in outs)
    return {"clients": clients, "total_s": round(total, 2), "ttft_p50_ms": pct(0.5), "ttft_p90_ms": pct(0.9), "cached_tokens": cached,
            "prompt_tokens": sum((o.get("usage") or {}).get("prompt_tokens", 0) for o in outs),
            "replies": {it["id"]: o.get("content") for it, o in zip(items, outs)},
            # in the order asked: the client's time to the first token, the server's (from taking the request in), what was kept
            "each": [{"id": it["id"], "ttft_ms": round(1000 * o["first"], 1) if o.get("first") is not None else None,
                      "server_first_ms": (o.get("timings") or {}).get("first_token_ms"),
                      "cached": (o.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens")} for it, o in zip(items, outs)],
            "errors": [o.get("error") for o in outs if o.get("status") != 200]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("on", "off"), default="on")
    ap.add_argument("--items", type=int, default=60)
    ap.add_argument("--system", type=int, default=1000, help="tokens of the system prompt, about")
    ap.add_argument("--clients", default="1,8")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--ram-gib", help="the server's --ram-gib (default: its own)")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        a, b = (json.load(open(p, encoding="utf-8")) for p in args.compare)
        for ra, rb in zip(a["runs"], b["runs"]):
            same = sum(ra["replies"][k] == rb["replies"].get(k) for k in ra["replies"])
            print(f"[prefix] {ra['clients']} clients: the same reply to {same} of {len(ra['replies'])}; "
                  f"ttft p50 {ra['ttft_p50_ms']} / {rb['ttft_p50_ms']} ms, p90 {ra['ttft_p90_ms']} / {rb['ttft_p90_ms']} ms, "
                  f"total {ra['total_s']} / {rb['total_s']} s")
        ab = [r["replies"] for r in a["runs"]]
        print(f"[prefix] {args.compare[0]}: 1 and 8 clients the same: {sum(ab[0][k] == ab[-1][k] for k in ab[0])} of {len(ab[0])}")
        return
    items, system = items_and_system(args.items, args.system)
    cmd = [sys.executable, "-m", "scripts.serve", "--port", str(args.port), "--context", "2048", "--sequences", "16"]
    if args.ram_gib is not None:
        cmd += ["--ram-gib", args.ram_gib]
    if args.mode == "on":
        cmd += ["--prefix-gib", "2"]
    srv = subprocess.Popen(cmd, cwd=ROOT)
    url = f"http://127.0.0.1:{args.port}/v1"
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(url.rsplit("/v1", 1)[0] + "/health", timeout=2)
                break
            except OSError:
                if srv.poll() is not None:
                    raise SystemExit("the server stopped")
                time.sleep(1)
        about = json.loads(urllib.request.urlopen(url.rsplit("/v1", 1)[0] + "/", timeout=5).read())
        runs = []
        for c in [int(x) for x in args.clients.split(",")]:
            r = run(url, system, items, c, args.max_tokens)
            runs.append(r)
            print(f"[prefix] {args.mode}, {c} clients: total {r['total_s']} s, ttft p50 {r['ttft_p50_ms']} ms, p90 {r['ttft_p90_ms']} ms, "
                  f"cached {r['cached_tokens']} of {r['prompt_tokens']} prompt tokens, errors {len(r['errors'])}", flush=True)
        after = json.loads(urllib.request.urlopen(url.rsplit("/v1", 1)[0] + "/", timeout=5).read())
        out = {"mode": args.mode, "bits": about.get("bits_per_weight"), "prefixes": after.get("prefixes"), "runs": runs}
        print(f"[prefix] {args.mode}: {about.get('bits_per_weight')} bits a weight; prefixes {after.get('prefixes')}", flush=True)
        if args.out:
            with open(args.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(out, f, indent=1, ensure_ascii=False)
    finally:
        srv.terminate()
        srv.wait(30)


if __name__ == "__main__":
    main()
