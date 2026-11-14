# Token sets for teacher-forced scoring (scripts/eval_codes.py --set task_tf_short / task_tf_gsm8k): each task_eval
# item's chat prompt (thinking off, as task_eval asks it) followed by a stored reply, the reply's tokens the targets.
# Scoring two paths on them position by position says how often a change (e.g. --kv) would flip a greedy choice in the
# work this server does: short translations, copying, extraction, lists, and longer arithmetic.
#   python -m scripts.kv_task_set [--replies replies_c6.jsonl]
import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer

from sclmd import DATA, ckpt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", default=os.path.join(DATA, "task_eval", "items.jsonl"))
    ap.add_argument("--replies", default=os.path.join(DATA, "task_eval", "replies_c6.jsonl"))
    ap.add_argument("--prefix", default="task_tf")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    end = tok.convert_tokens_to_ids("<turn|>")
    with open(args.items, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    replies = {}
    with open(args.replies, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                replies[r["id"]] = r
    parts = {"short": [], "gsm8k": []}
    for it in items:
        r = replies.get(it["id"])
        if r is None or r.get("error") or r.get("think"):
            continue
        messages = ([{"role": "system", "content": it["system"]}] if it.get("system") else []) + [{"role": "user", "content": it["prompt"]}]
        out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        prompt = list(out["input_ids"] if hasattr(out, "keys") else out)
        reply = tok.encode(r["answer"], add_special_tokens=False) + ([end] if r.get("ended") else [])
        parts["gsm8k" if it["kind"] == "gsm8k" else "short"].append((it["kind"], prompt, reply))
    for part, rows in parts.items():
        names = list(dict.fromkeys(k for k, _, _ in rows))
        n = max(len(p) + len(a) for _, p, a in rows)
        n = -(-n // 16) * 16
        tokens = np.zeros((len(rows), n), dtype=np.int32)  # padded at the end with 0 (pad): causal, so nothing earlier sees it
        target = np.zeros((len(rows), n), dtype=bool)
        category = np.array([names.index(k) for k, _, _ in rows], dtype=np.int8)
        for q, (_, p, a) in enumerate(rows):
            tokens[q, : len(p) + len(a)] = p + a
            target[q, len(p) : len(p) + len(a)] = True
        folder = os.path.join(DATA, f"{args.prefix}_{part}")
        os.makedirs(folder, exist_ok=True)
        np.savez(os.path.join(folder, "tokens.npz"), tokens=tokens, category=category, names=np.array(names), target=target)
        print(f"{folder}: {len(rows)} sequences of {n} positions, {int(target.sum())} reply tokens, {int((tokens > 0).sum())} tokens in all; "
              + ", ".join(f"{k} {int((category == i).sum())}" for i, k in enumerate(names)))


if __name__ == "__main__":
    main()
