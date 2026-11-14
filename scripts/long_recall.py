# A long prompt on the real model: two facts are put at the start and in the middle of a document of several thousand
# tokens, and asked for at the end. Shows that the keys and values far back are read right, how fast a prompt goes in,
# and (snapshot / restore) that the text with the drafter is the text without it at that length.
import argparse
import os
import time

import torch
from transformers import AutoTokenizer

from sclmd import DATA, ROOT, ckpt
from sclmd import engine as E
from sclmd import model as M
from sclmd import plan as P
from sclmd.generate import Picker, extend
from sclmd.stream import VRAM_MARGIN, cap_vram


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=7000, help="tokens of the document")
    ap.add_argument("--text", default=os.path.join(ROOT, "..", "design.md"), help="the document (default: the parent project's design notes); "
                    "several with commas are read one after the other")
    ap.add_argument("--rows", type=int, default=8, help="rows of a pass: 16 or 32 read the prompt faster")
    ap.add_argument("--new", type=int, default=120, help="tokens to write at most")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--keyless", action="store_true", help="the full-attention layers keep their values only (Engine: keyless)")
    E.add_kv4_option(ap)
    ap.add_argument("--plan", default=os.path.join(DATA, "plans", "fit_own_v1_11.1.json"))
    ap.add_argument("--capacity", type=int, default=8192)
    ap.add_argument("--reserve", type=float, default=VRAM_MARGIN / 2**30, help="GiB of VRAM left free (the benchmarks' 16,384 context with the chat's plan: 0.4)")
    args = ap.parse_args()
    E.use_kv4_option(args)
    cap_vram(margin=int(args.reserve * 2**30))
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    body = []
    for path in args.text.split(","):
        with open(path, encoding="utf-8") as f:
            body += tok.encode(f.read(), add_special_tokens=False)
    assert len(body) >= args.tokens, f"the document is only {len(body)} tokens"
    half = tok.decode(body[: args.tokens // 2]), tok.decode(body[args.tokens // 2 : args.tokens])
    text = ("Remember these two facts for later. The vault code is 48213 and its keeper is called Marisol.\n\n" + half[0]
            + "\n\n(A note in the middle: the meeting is on the 17th of March in Busan.)\n\n" + half[1]
            + "\n\nNow answer in one sentence each: (1) What is the vault code and who is its keeper? (2) When and where is the meeting?")
    out = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
    prompt = list(out["input_ids"] if hasattr(out, "keys") else out)
    plan, head = P.read(args.plan)
    eng = E.Engine.load(args.name, plan, head or 8, args.capacity, rows=args.rows, keyless=args.keyless)
    print(f"the engine holds {eng.bytes() / 2**30:.2f} GiB (context {args.capacity}, {args.rows} rows a pass)", flush=True)
    drafter = E.EngineDrafter.load(eng, ckpt("drafter"), args.name)
    torch.cuda.synchronize()
    t = time.perf_counter()
    logits, hidden = eng.feed(prompt, keep="last")
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t
    print(f"prompt {len(prompt)} tokens in {seconds:.1f} s ({len(prompt) / seconds:.0f} tok/s)", flush=True)
    eng.snapshot()
    texts = []
    for name, guess in (("with the drafter", drafter.guess), ("without", None)):
        eng.restore()  # back to the end of the prompt, however far the last run went
        stats = {}
        torch.cuda.synchronize()
        t = time.perf_counter()
        ids = extend(eng, len(prompt), logits[-1], hidden[-1], args.new, Picker(), guess, (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>")), stats)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        texts.append(ids)
        print(f"{name}: {len(ids)} tokens, {len(ids) / seconds:.1f} tok/s, {stats['tokens'] / max(stats['passes'], 1):.2f} a pass: "
              f"{tok.decode(ids, skip_special_tokens=True)!r}", flush=True)
    print(f"the same tokens: {texts[0] == texts[1]}", flush=True)


if __name__ == "__main__":
    main()
