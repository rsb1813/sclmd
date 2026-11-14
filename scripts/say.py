# One prompt through the real model on the engine: the reply, how fast it came, and (--check) whether the text with
# the drafter is the text without it.
import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from sclmd import ckpt
from sclmd import engine as E
from sclmd import model as M
from sclmd.generate import Picker, generate
from sclmd.stream import cap_vram

DEPTH = 7  # as the chat (scripts/chat.py)
END_OF_TURN = 106


def load_plan(path: str | None, rate: int, head: int, cfg: M.Config) -> tuple[dict, int]:
    if path:
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
        return {int(k): v for k, v in p["layers"].items()}, p.get("head") or head
    return {i: {m: rate for m in M.LINEARS} for i in range(cfg.n_layers)}, head


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--plan")
    ap.add_argument("--rate", type=int, default=6)
    ap.add_argument("--head", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0, help="0: the first choice; the model card samples at 1.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-drafter", action="store_true")
    ap.add_argument("--check", action="store_true", help="also generate without the drafter and compare the tokens")
    ap.add_argument("--capacity", type=int, default=8192)
    args = ap.parse_args()
    cap_vram()
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    out = tok.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=args.think)
    prompt = list(out["input_ids"] if hasattr(out, "keys") else out)
    cfg = M.Config.load(ckpt("orig"))
    plan, head = load_plan(args.plan, args.rate, args.head, cfg)
    t = time.time()
    eng = E.Engine.load(args.name, plan, head, args.capacity)
    drafter = None if args.no_drafter else E.EngineDrafter.load(eng, ckpt("drafter"), args.name, depth=DEPTH)
    torch.cuda.synchronize()
    print(f"[say] loaded in {time.time() - t:.0f} s, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB, prompt {len(prompt)} tokens", flush=True)
    pick = Picker(args.temperature, seed=args.seed)
    stop = (tok.eos_token_id, END_OF_TURN)

    def run(guess):
        eng.crop(0)
        stats = {}
        torch.cuda.synchronize()
        t = time.perf_counter()
        ids = generate(eng, prompt, args.max_new, pick, guess, stop=stop, stats=stats)
        torch.cuda.synchronize()
        return ids, time.perf_counter() - t, stats

    run(None if drafter is None else drafter.guess)  # the first run pays for CUDA's and PyTorch's start-up
    ids, seconds, stats = run(None if drafter is None else drafter.guess)
    print(tok.decode(ids, skip_special_tokens=False))
    print(f"[say] {len(ids)} tokens in {seconds:.2f} s including the prompt's {len(prompt)}: {len(ids) / seconds:.1f} tok/s, "
          f"{stats['tokens'] / max(stats['passes'], 1):.2f} tokens a pass" + ("" if drafter is None else " with the drafter"), flush=True)
    if args.check and drafter is not None:
        plain, seconds2, _ = run(None)
        print(f"[say] without the drafter: {len(plain) / seconds2:.1f} tok/s; the same tokens: {plain == ids}", flush=True)


if __name__ == "__main__":
    main()
