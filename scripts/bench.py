# A benchmark's replies from the real model on the engine: each item's prompt (scripts/bench_prepare.py) goes through
# the chat template, the reply is generated as the chat generates it (thinking on, the model card's sampling, the
# drafter), and its thinking and answer are stored as text. Scoring reads the stored replies (scripts/bench_score.py).
#
# A run can be stopped and started again: replies are appended one a line, and items already answered are skipped.
# While the pause file exists the run ends (exit 75) and frees the GPU; scripts/bench_queue.sh starts it again.
import argparse
import hashlib
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer

from sclmd import ROOT, ckpt
from sclmd import engine as E
from sclmd import plan as P
from sclmd.generate import Picker, extend
from sclmd.stream import VRAM_MARGIN, cap_vram
from scripts.chat import DEPTH, looping

PAUSED = 75
BENCH = os.environ.get("SCLMD_BENCH", os.path.join(ROOT, "data", "bench"))
PAUSE = os.environ.get("SCLMD_PAUSE", os.path.join(ROOT, "logs", "pause"))

# When a reply counts as a loop here: one piece of at most 32 tokens said 40 times over at least 1,024 tokens. The chat's
# own guard (6 times over 96 tokens) is a convenience that the user can overrule; here it would cost answers: it ended
# a reply that was adding two binary numbers column by column, seven like lines in a row.
LOOP = (40, 1024)

def seed_of(task: str, item: str, run: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{task}/{item}/{run}".encode()).digest()[:4], "little") & 0x3FFFFFFF


class Replier:
    def __init__(self, name: str, plan_path: str, capacity: int, drafter: bool = True):
        self.tok = tok = AutoTokenizer.from_pretrained(ckpt("orig"))
        plan, head = P.read(plan_path)
        self.eng = E.Engine.load(name, plan, head or 8, capacity)
        self.drafter = E.EngineDrafter.load(self.eng, ckpt("drafter"), name, depth=DEPTH) if drafter else None
        self.stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
        self.open, self.close = tok.convert_tokens_to_ids("<|channel>"), tok.convert_tokens_to_ids("<channel|>")

    @torch.no_grad()
    def reply(self, item: dict, seed: int, think: bool, sampling: dict, max_new: int | None) -> dict:
        eng, tok = self.eng, self.tok
        messages = ([{"role": "system", "content": item["system"]}] if item.get("system") else []) + [{"role": "user", "content": item["prompt"]}]
        out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=think)
        prompt = list(out["input_ids"] if hasattr(out, "keys") else out)
        room = eng.capacity - len(prompt) - eng.rows
        if room < 64:
            return {"id": item["id"], "error": "the prompt does not fit the context", "prompt_tokens": len(prompt)}
        t = time.perf_counter()
        eng.crop(0)
        logits, hidden = eng.feed(prompt, keep="last")
        said, looped, stats = [], [False], {}

        def emit(token: int) -> None:
            said.append(token)
            if len(said) % 16 == 0 and looping(said, *LOOP):
                looped[0] = True

        pick = Picker(sampling["temperature"], sampling["top_k"], sampling["top_p"], seed=seed)
        extend(eng, len(prompt), logits[-1], hidden[-1], min(max_new or room, room), pick, self.drafter.guess if self.drafter else None,
               self.stop, stats, emit, lambda: looped[0])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        # the thinking is what stands between the channel's opening and closing tokens; the answer is everything else
        kinds = {"think": [], "answer": []}
        kind = "answer"
        for token in said:
            if token == self.open or token == self.close:
                kind = "think" if token == self.open else "answer"
            elif token not in self.stop:
                kinds[kind].append(token)
        ended = bool(said) and said[-1] in self.stop
        return {"id": item["id"], "answer": tok.decode(kinds["answer"], skip_special_tokens=True),
                "think": tok.decode(kinds["think"], skip_special_tokens=True), "prompt_tokens": len(prompt), "tokens": len(said),
                "think_tokens": len(kinds["think"]), "ended": ended, "looped": looped[0], "loop_guard": LOOP[1], "open_think": kind == "think",
                "seconds": round(seconds, 2), "per_pass": round(stats["tokens"] / max(stats["passes"], 1), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--limit", type=int, default=0, help="the first N items of the task's fixed order (0: all)")
    ap.add_argument("--run", type=int, default=0, help="which run (its own seed and its own file)")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--plan", default=os.path.join(ROOT, "data", "plans", "fit_own_v1_11.2.json"), help="the plan the chat loads on this machine")
    ap.add_argument("--capacity", type=int, default=8192)
    ap.add_argument("--max-new", type=int, default=0, help="tokens a reply may take (0: what the context leaves)")
    ap.add_argument("--nothink", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--reserve", type=float, default=VRAM_MARGIN / 2**30, help="GiB of the free VRAM left to other programs")
    ap.add_argument("--tag", default="", help="added to the file name (a run under other settings)")
    args = ap.parse_args()
    with open(os.path.join(BENCH, "items", f"{args.task}.jsonl"), encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    items = items[: args.limit] if args.limit else items
    os.makedirs(os.path.join(BENCH, "replies"), exist_ok=True)
    path = os.path.join(BENCH, "replies", f"{args.task}{args.tag}.run{args.run}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in (json.loads(line) for line in f if line.strip()):
                # a reply that the chat's loop guard ended is answered again (the same seed: the same reply, not cut);
                # the scorer takes an item's last reply
                done.discard(r["id"]) if r.get("looped") and r.get("loop_guard", 0) < LOOP[1] else done.add(r["id"])
    todo = [it for it in items if it["id"] not in done]
    print(f"[bench] {args.task} run {args.run}: {len(todo)} of {len(items)} items to answer", flush=True)
    if not todo:
        return
    if os.path.exists(PAUSE):
        sys.exit(PAUSED)
    cap_vram(margin=int(args.reserve * 2**30))
    rep = Replier(args.name, args.plan, args.capacity)
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}
    print(f"[bench] loaded {os.path.basename(args.plan)}, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    t0, tokens = time.time(), 0
    for n, item in enumerate(todo):
        if os.path.exists(PAUSE):
            print("[bench] paused", flush=True)
            sys.exit(PAUSED)
        try:
            rec = rep.reply(item, seed_of(args.task, item["id"], args.run), not args.nothink, sampling, args.max_new or None)
        except Exception as e:  # noqa: BLE001  (one item must not end a night's run)
            rec = {"id": item["id"], "error": f"{type(e).__name__}: {e}"}
            rep.eng.crop(0)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tokens += rec.get("tokens", 0)
        print(f"[bench] {args.task} {n + 1}/{len(todo)} {item['id']}: " + (rec["error"] if "error" in rec else
              f"{rec['tokens']} tokens ({rec['think_tokens']} thinking), {rec['tokens'] / max(rec['seconds'], 1e-9):.0f} tok/s"
              + ("" if rec["ended"] else ", looped" if rec["looped"] else ", cut")) + f" | {tokens / max(time.time() - t0, 1e-9):.0f} tok/s overall", flush=True)


if __name__ == "__main__":
    main()
