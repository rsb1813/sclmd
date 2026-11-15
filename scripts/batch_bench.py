# How fast the real model writes several texts at once (sclmd/batch.py): the same prompts written 1, 2, 4, 8, ... at a
# time, with the tokens a second of all texts together, of each text, and what a pass carried. Each text is also
# checked against the one written alone: it has to be the same tokens.
import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from sclmd import DATA, ckpt
from sclmd import engine as E
from sclmd import model as M
from sclmd import plan as P
from sclmd.batch import Batcher, Job
from sclmd.generate import Picker
from sclmd.stream import VRAM_MARGIN, cap_vram

DEPTH = 6
PROMPTS = [
    "Explain how a hash table handles collisions, with a short example in Python.",
    "한국의 사계절이 농사에 어떤 영향을 주는지 설명해줘.",
    "A train leaves at 9:40 and travels 210 km at 84 km/h, then waits 25 minutes and travels 96 km at 64 km/h. When does it arrive? Show the steps.",
    "Write a Go function that reverses a singly linked list, and explain each line.",
    "바닷물이 짠 이유를 초등학생도 이해할 수 있게 설명해줘.",
    "Describe three common causes of memory leaks in long-running Python services and how to find them.",
    "List the planets of the solar system with one interesting fact each.",
    "이진 탐색 트리와 해시 테이블의 차이를 표로 정리하고 언제 무엇을 쓰는지 알려줘.",
    "What is the difference between a process and a thread? Give an example where each is the better choice.",
    "Summarize how vaccines train the immune system, in about ten sentences.",
    "피보나치 수열을 구하는 방법 세 가지를 파이썬으로 보여주고 속도를 비교해줘.",
    "Explain what a database index is and when adding one makes a query slower.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "조선 시대의 과거 제도가 어떻게 운영되었는지 설명해줘.",
    "How does public-key cryptography let two strangers agree on a secret? Explain without formulas.",
    "Give me a week of dinner ideas that use rice, eggs and seasonal vegetables.",
]


def pool_for(cfg: M.Config, budget: int, full_per_sliding: float = 1.0, kv4: bool = False) -> dict:
    """Tiles of keys and values for `budget` bytes: `full_per_sliding` tiles of the full-attention layers for each tile
    of the sliding ones (texts shorter than the sliding window need one of each for every 16 tokens)."""
    unit = E.tile_bytes(cfg, M.SLIDING, kv4=kv4) + full_per_sliding * E.tile_bytes(cfg, M.FULL)
    n = int(budget // unit)
    return {M.SLIDING: n, M.FULL: int(n * full_per_sliding)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--plan", default="c6", help="c6 (3 bits everywhere: the most room for keys and values) or a plan file")
    ap.add_argument("--head", type=int, default=8)
    ap.add_argument("--counts", default="1,2,4,8,12,16", help="texts at a time")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--capacity", type=int, default=2048)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-drafter", action="store_true")
    ap.add_argument("--rows", type=int, default=32, choices=(8, 16, 32), help="the most rows of a pass")
    ap.add_argument("--wide-rows", type=int, default=0, choices=(0, 64, 128, 256), help="passes of up to this many rows while prompts "
                    "are read (scripts/serve.py --wide-rows; 0: off)")
    ap.add_argument("--out", default=os.path.join(DATA, "batch_bench.json"))
    E.add_int8_option(ap)
    E.add_kv4_option(ap)
    args = ap.parse_args()
    E.use_int8_option(args)
    kv4 = E.use_kv4_option(args)
    counts = [int(c) for c in args.counts.split(",")]
    cap_vram()
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    cfg = M.Config.load(ckpt("orig"))
    if args.plan == "c6":
        plan, head = P.uniform(cfg, 6), args.head
    else:
        plan, head = P.read(args.plan)
        head = head or args.head
    # what the VRAM that is free now leaves for keys and values, after the codes, the head, the drafter and room to load in
    free = torch.cuda.mem_get_info()[0] - VRAM_MARGIN
    rows = args.wide_rows or args.rows  # the engine's: the widest pass
    widths = tuple(w for w in E.ROWS if w <= args.rows) + tuple(w for w in E.ROWS if E.ATT_ROWS < w <= args.wide_rows)
    rest = E.fixed_bytes(cfg, head, args.capacity, pool={M.SLIDING: 0, M.FULL: 0}, rows=rows, kv4=kv4) + (0 if args.no_drafter else E.DRAFTER_BYTES)
    budget = free - P.code_bytes(cfg, plan) - rest - (320 << 20)
    pool = pool_for(cfg, budget, kv4=kv4)
    eng = E.Engine.load(args.name, plan, head, args.capacity, sequences=max(counts), pool=pool, rows=rows)
    drafter = None if args.no_drafter else E.EngineDrafter.load(eng, ckpt("drafter"), args.name, depth=DEPTH)
    torch.cuda.synchronize()
    print(f"[batch] {P.bits_per_weight(cfg, plan):.2f} bits a weight, keys and values {budget / 2**30:.2f} GiB = {pool[M.SLIDING]} tiles "
          f"({pool[M.SLIDING] * 16} tokens of texts under {cfg.window}), VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"))
    prompts = []
    for text in PROMPTS[: max(counts)]:
        out = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, enable_thinking=args.think)
        prompts.append(list(out["input_ids"] if hasattr(out, "keys") else out))

    def run(which: list) -> tuple[list, dict]:
        for seq in range(eng.sequences):
            eng.release(seq)
        batch = Batcher(eng, drafter, DEPTH, widths)
        jobs = [Job(seq, prompts[i], args.max_new, Picker(args.temperature, seed=1000 + i), stop) for seq, i in enumerate(which)]
        for j in jobs:
            batch.add(j)
        torch.cuda.synchronize()
        t = time.perf_counter()
        first, over = {}, set()
        while batch.step():
            for j in jobs:
                if j.out and j not in first:
                    first[j] = time.perf_counter() - t
                if j.done is not None and j not in over:  # a text that is over gives its keys and values back (a server
                    eng.release(j.seq)                     # moves the conversation to RAM): the ones that wait can go on
                    over.add(j)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        tokens = sum(len(j.out) for j in jobs)
        return [j.out for j in jobs], {"texts": len(jobs), "tokens": tokens, "seconds": round(seconds, 2), "tok_s": round(tokens / seconds, 1),
                                       "tok_s_each": round(tokens / seconds / len(jobs), 1), "passes": batch.passes,
                                       "ms_a_pass": round(1000 * seconds / batch.passes, 1), "rows_a_pass": round(batch.rows / batch.passes, 2),
                                       "guesses_kept": round(batch.kept_rows / max(batch.guess_rows, 1), 2), "widths": dict(batch.widths),
                                       "first_token_s": round(sum(first.values()) / len(first), 2), "ended": [j.done for j in jobs]}

    run([0])  # the first run pays for CUDA's and PyTorch's start-up
    solo = {}
    for i in range(max(counts)):
        texts, stats = run([i])
        solo[i] = texts[0]
        print(f"[batch] alone, prompt {i}: {stats['tokens']} tokens, {stats['tok_s']} tok/s, {stats['ms_a_pass']} ms a pass", flush=True)
    report = {"plan": os.path.basename(args.plan), "bits": round(P.bits_per_weight(cfg, plan), 3), "max_new": args.max_new, "think": args.think,
              "temperature": args.temperature, "drafter": not args.no_drafter, "rows": args.rows, "wide_rows": args.wide_rows, "kv_gib": round(budget / 2**30, 2), "runs": []}
    for count in counts:
        try:
            texts, stats = run(list(range(count)))
        except E.KVFull as e:
            print(f"[batch] {count} at a time: {e}", flush=True)
            break
        stats["same_as_alone"] = all(texts[i] == solo[i][: len(texts[i])] for i in range(count))  # a text that ran out of memory: as far as it got
        report["runs"].append(stats)
        print(f"[batch] {count:2d} at a time: {stats['tok_s']:6.1f} tok/s in all, {stats['tok_s_each']:5.1f} each, {stats['ms_a_pass']} ms a pass, "
              f"{stats['rows_a_pass']} rows a pass (passes by width {stats['widths']}), guesses kept {stats['guesses_kept']}, first token after {stats['first_token_s']} s, "
              f"same tokens as alone: {stats['same_as_alone']}, ended {sorted(set(stats['ended']))}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
