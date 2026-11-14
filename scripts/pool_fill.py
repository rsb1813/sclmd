# How many conversations of a length a running server keeps on the GPU at once, by filling it: for each length,
# longest first, prompts of about that many tokens (each a different part of a document, so nothing is shared) are sent
# one after another with a one-token reply, and /health says how many conversations the GPU keeps afterwards. A
# conversation is put to RAM only when the pool (or the sequences) runs out, oldest first: after more requests than
# fit, the GPU holds as many of this length as fit together (the longer ones of the length before went first).
#   python scripts/pool_fill.py --url http://127.0.0.1:8001/v1 --lengths 8000,2000,1000,500,200 --count 6,10,14,22,44
import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def call(url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--lengths", default="8000,2000,1000,500,200", help="tokens a conversation, longest first")
    ap.add_argument("--count", default="6,10,14,22,44", help="prompts sent of each length (more than fit)")
    ap.add_argument("--doc", default=os.path.join(ROOT, "data", "speed_check", "doc.txt"))
    ap.add_argument("--out")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from sclmd import ckpt

    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    with open(args.doc, encoding="utf-8") as f:
        doc = tok.encode(f.read(), add_special_tokens=False)
    base = args.url.rstrip("/").rsplit("/v1", 1)[0]
    report, at = [], 0
    for n, count in zip((int(x) for x in args.lengths.split(",")), (int(x) for x in args.count.split(","))):
        sizes = []
        t = time.time()
        for k in range(count):
            body = n - 24  # the chat template's tokens and the question
            if at + body > len(doc):
                at = 0
            text = f"[{n}/{k}] " + tok.decode(doc[at : at + body]) + "\n\nAnswer with one word: what is this about?"
            at += body
            r = call(args.url.rstrip("/") + "/chat/completions",
                     {"messages": [{"role": "user", "content": text}], "max_tokens": 1, "temperature": 0.0})
            sizes.append(r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"])
        time.sleep(1.0)  # the last reply's later form is read in meanwhile
        health = call(base + "/health")
        about = call(base + "/v1")
        row = {"tokens": n, "sent": count, "conversation_tokens": [min(sizes), max(sizes)], "kept_on_gpu": health["kept"],
               "in_ram": about.get("conversations_in_ram"), "keys_and_values": about.get("keys_and_values"), "seconds": round(time.time() - t, 1)}
        report.append(row)
        print(f"[fill] {n:5d} tokens ({row['conversation_tokens'][0]}-{row['conversation_tokens'][1]} a conversation): {count} sent, "
              f"{row['kept_on_gpu']} kept on the GPU, {row['in_ram']} in RAM ({row['seconds']} s)", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
