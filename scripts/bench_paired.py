# The original and the engine on the same multiple-choice questions, answer by answer: what the codes cost in answers.
# No thinking and no text: the prompt ends with "The answer is (" and the answer is the letter with the largest logit
# there, so the original (which this machine cannot write text with) can be asked too, one layer at a time.
#
#   --side ours   the engine with a plan (prompts read 32 tokens a pass)
#   --side orig   the original checkpoint, layer by layer over batches of padded prompts (--fp32: in float32, on the
#                 first questions only: how many answers the arithmetic's rounding alone turns)
#   --side score  the two against each other
# Answers are appended as they come: a run that is stopped goes on where it was. While the pause file exists a run ends.
import argparse
import json
import math
import os
import sys
import time

import torch
from transformers import AutoTokenizer

from sclmd import ROOT, ckpt
from sclmd import engine as E
from sclmd import model as M
from sclmd import plan as P
from sclmd.checkpoint import Checkpoint
from sclmd.stream import VRAM_MARGIN, cap_vram

BENCH = os.environ.get("SCLMD_BENCH", os.path.join(ROOT, "data", "bench"))
PAUSE = os.environ.get("SCLMD_PAUSE", os.path.join(ROOT, "logs", "pause"))
PAUSED = 75
LETTERS = "ABCDEFGHIJ"
ASK = ("Answer the following multiple choice question{about}. Reply with the letter of the correct option only.\n\n")
PREFIX = "The answer is ("


def prompts(tok, task: str, limit: int) -> list:
    """(id, prompt tokens, options, the right letter) of the task's first `limit` items: the question without the
    request to think, the model's turn opened, and the answer begun."""
    with open(os.path.join(BENCH, "items", f"{task}.jsonl"), encoding="utf-8") as f:
        items = [json.loads(line) for line in f][:limit or None]
    prefix = tok.encode(PREFIX, add_special_tokens=False)
    out = []
    for it in items:
        question = it["prompt"].split("\n\n", 1)[1]  # after bench_prepare's first paragraph (the instructions)
        about = it["prompt"].split(". The last line", 1)[0].split("multiple choice question", 1)[1]
        ids = tok.apply_chat_template([{"role": "user", "content": ASK.format(about=about) + question}], tokenize=True,
                                      add_generation_prompt=True, enable_thinking=False)
        out.append((it["id"], list(ids["input_ids"] if hasattr(ids, "keys") else ids) + prefix, it["options"], it["gold"]))
    return out


def done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def record(f, item_id: str, logits: torch.Tensor, options: int, letters: list, whole: int):
    """One answer: the logits of the option letters, the letter chosen, and whether the largest logit of all was a letter."""
    lg = logits[letters[:options]].float().tolist()
    f.write(json.dumps({"id": item_id, "pick": LETTERS[max(range(options), key=lg.__getitem__)], "logits": [round(x, 4) for x in lg],
                        "letter_first": whole in letters}) + "\n")
    f.flush()


@torch.no_grad()
def ours(args, tok, todo, letters, path):
    cap_vram(margin=int(args.reserve * 2**30))
    plan, head = P.read(args.plan)
    eng = E.Engine.load(args.name, plan, head or 8, 2048, rows=32)
    print(f"[paired] the engine with {os.path.basename(args.plan)}, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    t, tokens = time.time(), 0
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for n, (item_id, ids, options, _) in enumerate(todo):
            if os.path.exists(PAUSE):
                sys.exit(PAUSED)
            if len(ids) > eng.capacity:
                continue
            eng.crop(0)
            logits = eng.feed(ids, keep="last")[0][-1]
            record(f, item_id, logits, options, letters, int(logits.argmax()))
            tokens += len(ids)
            if n % 200 == 199:
                print(f"[paired] {n + 1}/{len(todo)}, {tokens / (time.time() - t):.0f} tok/s", flush=True)


@torch.no_grad()
def orig(args, tok, todo, letters, path):
    cap_vram()
    ck = Checkpoint(ckpt("orig"))
    cfg = ck.cfg
    dtype = torch.float32 if args.fp32 else None
    if args.fp32:  # everything is twice the size (the embedding alone 5.6 GB): a quarter of the tokens at once
        args.chunk, args.batch = args.chunk // 4, args.batch // 4
    todo = sorted(todo, key=lambda x: len(x[1]))  # prompts of like length together: little padding
    chunks, cur = [], []
    for item in todo:
        if cur and (len(cur) + 1) * len(item[1]) > args.chunk:
            chunks.append(cur)
            cur = []
        cur.append(item)
    if cur:
        chunks.append(cur)
    norm, embed = ck.norm("cuda"), ck.embed("cuda")
    if dtype is not None:
        norm, embed = norm.to(dtype), embed.to(dtype)
    t, tokens = time.time(), 0
    for c, chunk in enumerate(chunks):
        if os.path.exists(PAUSE):
            sys.exit(PAUSED)
        n = len(chunk[-1][1])
        ids = torch.zeros(len(chunk), n, dtype=torch.long, device="cuda")
        for r, (_, seq, _, _) in enumerate(chunk):
            ids[r, : len(seq)] = torch.tensor(seq, device="cuda")  # padded on the right: a position never reads what comes after it
        h = embed[ids] * M.embed_scale(cfg, embed.dtype)
        positions = torch.arange(n, device="cuda")
        batch = max(1, args.batch // n)
        for i in range(cfg.n_layers):
            w = ck.layer(i, "cuda")
            if dtype is not None:
                w = {k: v.to(dtype) for k, v in w.items()}
            for a in range(0, len(chunk), batch):
                h[a : a + batch] = M.layer_forward(cfg, i, w, h[a : a + batch], positions)
            del w
        last = torch.stack([h[r, len(seq) - 1] for r, (_, seq, _, _) in enumerate(chunk)])
        logits = M.logits_of(cfg, last, norm, embed).float()
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            for r, (item_id, _, options, _) in enumerate(chunk):
                record(f, item_id, logits[r], options, letters, int(logits[r].argmax()))
        tokens += sum(len(seq) for _, seq, _, _ in chunk)
        del h, logits
        torch.cuda.empty_cache()
        print(f"[paired] chunk {c + 1}/{len(chunks)}: {len(chunk)} prompts of up to {n} tokens, {tokens / (time.time() - t):.0f} tok/s", flush=True)


def score(args, items):
    gold = {i: g for i, _, _, g in items}
    folder = os.path.join(BENCH, "paired")

    def read(name):
        p = os.path.join(folder, f"{args.task}.{name}.jsonl")
        if not os.path.exists(p):
            return {}
        with open(p, encoding="utf-8") as f:
            return {r["id"]: r for r in map(json.loads, f)}

    def against(a: dict, b: dict, la: str, lb: str) -> dict:
        both = [i for i in a if i in b]
        n = len(both)
        if not n:
            return {}
        ra, rb = sum(a[i]["pick"] == gold[i] for i in both), sum(b[i]["pick"] == gold[i] for i in both)
        lost = sum(a[i]["pick"] == gold[i] and b[i]["pick"] != gold[i] for i in both)
        won = sum(a[i]["pick"] != gold[i] and b[i]["pick"] == gold[i] for i in both)
        other = sum(a[i]["pick"] != b[i]["pick"] for i in both)
        half = 1.96 * math.sqrt(lost + won) / n  # of the difference: the answers that turned are what varies
        out = {"questions": n, la: round(100 * ra / n, 2), lb: round(100 * rb / n, 2), "difference": round(100 * (rb - ra) / n, 2),
               "interval": round(100 * half, 2), "right_to_wrong": lost, "wrong_to_right": won, "answers_that_differ": other,
               "largest_logit_is_a_letter": round(sum(a[i]["letter_first"] for i in both) / n, 3)}
        print(f"{la} {out[la]:.2f}%, {lb} {out[lb]:.2f}% on {n} questions: {out['difference']:+.2f} points (+-{out['interval']:.2f}); "
              f"{other} answers differ, {lost} right ones turned wrong, {won} wrong ones right", flush=True)
        return out

    o, e, f32 = read("orig"), read(args.tag), read("orig_fp32")
    report = {"task": args.task, "plan": args.tag, "prompt": "no thinking, the answer is the letter after 'The answer is ('",
              "orig_vs_ours": against(o, e, "original", "ours"), "orig_vs_orig_fp32": against(o, f32, "original", "original_fp32")}
    with open(os.path.join(os.path.dirname(BENCH), f"bench_paired_{args.task}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", required=True, choices=("ours", "orig", "score"))
    ap.add_argument("--task", default="mmlu_pro")
    ap.add_argument("--limit", type=int, default=2800, help="the first N items of the task's fixed order (0: all)")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--plan", default=os.path.join(ROOT, "data", "plans", "fit_own_v1_11.9.json"))
    ap.add_argument("--fp32", action="store_true", help="the original in float32 (with --limit, a few hundred: the rounding's own share)")
    ap.add_argument("--chunk", type=int, default=65536, help="orig: padded tokens held at once (10.8 KB of VRAM each)")
    ap.add_argument("--batch", type=int, default=8192, help="orig: padded tokens a layer takes at once")
    ap.add_argument("--reserve", type=float, default=VRAM_MARGIN / 2**30)
    args = ap.parse_args()
    args.tag = os.path.splitext(os.path.basename(args.plan))[0]
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    items = prompts(tok, args.task, args.limit)
    if args.side == "score":
        return score(args, items)
    letters = [tok.encode(PREFIX + x, add_special_tokens=False)[-1] for x in LETTERS]
    os.makedirs(os.path.join(BENCH, "paired"), exist_ok=True)
    name = args.tag if args.side == "ours" else "orig_fp32" if args.fp32 else "orig"
    path = os.path.join(BENCH, "paired", f"{args.task}.{name}.jsonl")
    seen = done_ids(path)
    todo = [it for it in items if it[0] not in seen]
    print(f"[paired] {args.task} {name}: {len(todo)} of {len(items)} questions to answer, {sum(len(x[1]) for x in todo)} tokens", flush=True)
    if not todo:
        return
    if os.path.exists(PAUSE):
        sys.exit(PAUSED)
    (ours if args.side == "ours" else orig)(args, tok, todo, letters, path)


if __name__ == "__main__":
    main()
