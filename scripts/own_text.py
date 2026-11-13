# Builds a token set out of text the model writes itself on the engine: prompts go in, replies (with their thinking)
# come out, and prompt + reply become a sequence whose targets are the reply's tokens.
# The other sets are text by other models and people, on which this model's distributions are sharp and move easily.
# What a plan loses on the model's own text is scored with the usual tools on this set:
#   python -m scripts.reference --set own_v1                 (the original over it: the reference)
#   python -m scripts.reference --set own_v1 --quant fp32    (the original in float32)
#   python -m scripts.eval_codes --set own_v1 --plan ...     (any plan against them)
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from sclmd import DATA, ckpt
from sclmd import engine as E
from sclmd import model as M
from sclmd import plan as P
from sclmd.generate import Picker, extend
from sclmd.stream import cap_vram

# (category, thinking, prompt)
PROMPTS = [
    ("think_en", True, "A farmer has a rectangular field. If he increases the length by 20% and decreases the width by 10%, the area becomes "
                       "2,160 square meters. If instead he decreases the length by 10% and increases the width by 20%, what is the area? "
                       "Then explain which change is better if fencing costs 12 dollars a meter and the original length was 50 meters."),
    ("think_en", True, "Design a rate limiter for an API gateway that serves 50,000 requests a second across 12 nodes. Compare token bucket, "
                       "sliding window log and sliding window counter, pick one, and write the core of it in Python."),
    ("think_ko", True, "서울에서 부산까지 KTX로 2시간 30분, 버스로 4시간 20분, 비행기로는 공항 이동과 대기를 합쳐 3시간이 걸린다고 하자. "
                       "요금은 각각 59,800원, 34,000원, 82,000원이다. 시간의 가치를 시간당 2만 원으로 볼 때 어느 쪽이 가장 싼지 계산하고, "
                       "시간의 가치가 얼마 이상이면 KTX가 버스보다 유리해지는지도 구해줘."),
    ("think_ko", True, "파이썬으로 LRU 캐시를 직접 구현하려고 해. OrderedDict를 쓰는 방법과 이중 연결 리스트와 딕셔너리를 쓰는 방법을 비교하고, "
                       "스레드 안전하게 만들려면 무엇을 바꿔야 하는지 코드와 함께 설명해줘."),
    ("chat_en", False, "Explain how a transformer language model generates text, from tokenization to sampling, to someone who knows "
                       "linear algebra but has never studied machine learning. Be thorough."),
    ("chat_ko", False, "조선 후기의 실학이 무엇이었는지, 대표적인 학자 세 명과 그들의 주장, 그리고 그것이 당시 사회에 미친 영향을 자세히 설명해줘."),
    ("code_en", False, "Write a complete, well-commented Rust program that reads a CSV file of (timestamp, sensor id, value) rows, computes "
                       "a rolling 5-minute average per sensor, and prints anomalies more than 3 standard deviations from the mean."),
    ("code_ko", False, "C++로 스레드 풀을 구현해줘. 작업 큐, 조건 변수, 종료 처리까지 포함하고 각 부분이 왜 필요한지 주석으로 설명해줘."),
]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="own_v1", help="the set's name under data/")
    ap.add_argument("--codes", default="v1")
    ap.add_argument("--plan", default=os.path.join(DATA, "plans", "fit_11.1.json"), help="the plan that writes the text")
    ap.add_argument("--head", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=1.0, help="the model card's sampling (top-k 64, top-p 0.95)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cap_vram()
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    cfg = M.Config.load(ckpt("orig"))
    plan, head = P.read(args.plan)
    eng = E.Engine.load(args.codes, plan, head or args.head, 8192)
    drafter = E.EngineDrafter.load(eng, ckpt("drafter"), args.codes)
    stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
    names = list(dict.fromkeys(c for c, _, _ in PROMPTS))
    seqs, manifest = [], {"plan": os.path.basename(args.plan), "temperature": args.temperature, "seed": args.seed, "sequences": []}
    for i, (cat, think, text) in enumerate(PROMPTS):
        out = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, enable_thinking=think)
        prompt = list(out["input_ids"] if hasattr(out, "keys") else out)
        eng.crop(0)
        logits, hidden = eng.feed(prompt, keep="last")
        stats = {}
        t = time.perf_counter()
        reply = extend(eng, len(prompt), logits[-1], hidden[-1], args.max_new, Picker(args.temperature, seed=args.seed + i), drafter.guess, stop, stats)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        seqs.append((names.index(cat), prompt, reply))
        manifest["sequences"].append({"category": cat, "thinking": think, "prompt_tokens": len(prompt), "reply_tokens": len(reply),
                                      "ended": reply[-1] in stop, "tok_s": round(len(reply) / dt, 1),
                                      "tokens_per_pass": round(stats["tokens"] / max(stats["passes"], 1), 2)})
        print(f"[own] {cat}: prompt {len(prompt)}, reply {len(reply)} tokens ({'ended' if reply[-1] in stop else 'cut'}), "
              f"{len(reply) / dt:.1f} tok/s, {stats['tokens'] / max(stats['passes'], 1):.2f} a pass", flush=True)
    n = -(-max(len(p) + len(r) for _, p, r in seqs) // 64) * 64
    tokens = np.zeros((len(seqs), n), dtype=np.int32)  # padded with token 0 after the text: nothing there is a target
    target = np.zeros((len(seqs), n), dtype=bool)
    for q, (_, p, r) in enumerate(seqs):
        tokens[q, : len(p) + len(r)] = p + r
        target[q, len(p) : len(p) + len(r)] = True
    out_dir = os.path.join(DATA, args.name)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "tokens.npz"), tokens=tokens, target=target, category=np.array([c for c, _, _ in seqs], dtype=np.int8),
             names=np.array(names))
    with open(os.path.join(out_dir, "text.txt"), "w", encoding="utf-8") as f:
        for q, (c, p, r) in enumerate(seqs):
            f.write(f"===== {q} {names[c]} =====\n{tok.decode(p + r, skip_special_tokens=False)}\n\n")
    manifest.update(seq_len=n, total_tokens=int(tokens.size), targets=int(target.sum()))
    with open(os.path.join(DATA, args.name + ".json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[own] {args.name}: {len(seqs)} sequences of {n} tokens, {int(target.sum())} targets", flush=True)


if __name__ == "__main__":
    main()
