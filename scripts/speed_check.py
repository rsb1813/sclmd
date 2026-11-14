# The speed of a reply, and that its text has not changed: fixed prompts and seeds, written on the engine as the chat
# loads it (the plan's weights, 16 rows a pass, the drafter). A reply's tokens depend on the prompt and the seed only
# (not on the drafter, its depth, or how fast anything is), so a change that is meant to cost nothing must write the
# same tokens:
#   python -m scripts.speed_check --save base        today's tokens and speed
#   python -m scripts.speed_check --against base     after a change: the same tokens? and how fast
#   python -m scripts.speed_check --profile          where a cycle's time goes (waits for the GPU after each stage)
#   python -m scripts.speed_check --chat             through the chat's own loop (its text handling counted)
import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict

import torch
from transformers import AutoTokenizer

from sclmd import ROOT, ckpt
from sclmd import engine as E
from sclmd import plan as P
from sclmd.generate import Picker, extend
from sclmd.stream import cap_vram

FOLDER = os.path.join(ROOT, "data", "speed_check")
MATH = ("Let S be the set of all ordered pairs (A, B) of subsets of {1, 2, ..., 10} such that the number of elements of the "
        "intersection of A and B is even and the union of A and B contains no three consecutive integers. "
        "Find the remainder when the number of elements of S is divided by 1000.")
KOREAN = "TCP 혼잡 제어가 왜 필요한지, 그리고 Tahoe, Reno, CUBIC, BBR이 각각 무엇을 바꿨는지 예를 들어 가며 자세히 설명해 줘."
CODE = ("Write a Python class for a persistent segment tree over an array of integers that supports a point update returning "
        "a new version and a range-sum query on any version. Explain the idea first, then give the code and a few tests.")
ASK_DOC = "\n\n위 문서에서 3bit와 4bit를 나누는 방법이 어떻게 정해졌는지, 그 근거가 된 측정과 함께 순서대로 설명해 줘."
# name, thinking, the most new tokens, the user's message or (tokens of the frozen document before the question)
PROMPTS = (("math", True, 9000, MATH), ("korean", False, 1500, KOREAN), ("code", True, 3000, CODE),
           ("doc6k", True, 2000, 6000), ("doc12k", False, 1500, 11500))
BUCKETS = (2048, 4096, 8192, 12288, 1 << 30)


def bucket(position: int) -> str:
    lo = 0
    for hi in BUCKETS:
        if position < hi:
            return f"{lo // 1024}k-{hi // 1024}k" if hi < (1 << 30) else f"{lo // 1024}k-"
        lo = hi


def seed_of(i: int) -> int:
    return random.Random(1000 + i).getrandbits(30)  # the chat draws its seed so: random.seed(1000 + i) gives the same


class Timed(Picker):
    """A picker that notes how long choosing takes, by where in the context the reply is."""

    def __init__(self, shares, where, *a, **k):
        super().__init__(*a, **k)
        self.shares, self.where = shares, where

    def many(self, logits, position):
        t = time.perf_counter()
        out = super().many(logits, position)
        self.shares[self.where()]["pick"] += time.perf_counter() - t
        return out


class Target:
    """The engine as generation uses it, with a wait for the GPU after a pass (the time is then the pass's own)."""

    def __init__(self, eng, shares, where):
        self.eng, self.shares, self.where = eng, shares, where

    def feed(self, ids):
        b = self.where()
        t = time.perf_counter()
        out = self.eng.feed(ids)
        torch.cuda.synchronize()
        self.shares[b]["pass"] += time.perf_counter() - t
        self.shares[b]["cycles"] += 1
        self.shares[b]["rows"] += len(ids)
        return out

    def crop(self, length):
        self.eng.crop(length)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--plan", default=os.path.join(ROOT, "data", "plans", "fit_own_v1_11.9.json"))
    ap.add_argument("--capacity", type=int, default=16384)
    ap.add_argument("--rows", type=int, default=16, help="rows of a pass (the chat's engine has 16)")
    ap.add_argument("--reserve", type=float, default=0.4)
    ap.add_argument("--depth", type=int, default=0, help="tokens the drafter guesses ahead (0: the chat's)")
    ap.add_argument("--no-drafter", action="store_true")
    ap.add_argument("--only", default="", help="names of the prompts to run, with commas")
    ap.add_argument("--scale", type=float, default=1.0, help="a share of each prompt's new tokens (the tokens are then a beginning of the full run's)")
    ap.add_argument("--save", default="", help="store the tokens under this name")
    ap.add_argument("--against", default="", help="compare the tokens with those stored under this name")
    ap.add_argument("--profile", action="store_true", help="wait for the GPU after each stage and show the shares by position")
    ap.add_argument("--chat", action="store_true", help="generate through the chat's Session (its emit and loop check counted)")
    args = ap.parse_args()
    from scripts import chat as CH

    cap_vram(margin=int(args.reserve * 2**30))
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    plan, head = P.read(args.plan)
    eng = E.Engine.load(args.name, plan, head or 8, args.capacity, rows=args.rows)
    depth = args.depth or CH.DEPTH
    drafter = None if args.no_drafter else E.EngineDrafter.load(eng, ckpt("drafter"), args.name, depth=depth)
    stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
    with open(os.path.join(FOLDER, "doc.txt"), encoding="utf-8") as f:
        doc = tok.encode(f.read(), add_special_tokens=False)
    print(f"[speed] {os.path.basename(args.plan)}, context {args.capacity}, {args.rows} rows, "
          + ("no drafter" if drafter is None else f"drafter depth {depth}") + f", VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    stored = {}
    if args.against:
        with open(os.path.join(FOLDER, f"{args.against}.json"), encoding="utf-8") as f:
            stored = json.load(f)
    shares = defaultdict(lambda: defaultdict(float))
    results, all_tokens, all_seconds, same = {}, 0, 0.0, True
    for i, (name, think, max_new, user) in enumerate(PROMPTS):
        if args.only and name not in args.only.split(","):
            continue
        if isinstance(user, int):
            user = tok.decode(doc[:user]) + ASK_DOC
        max_new = max(int(max_new * args.scale), 16)
        if args.chat:
            session = CH.Session(tok, eng, drafter)
            session.think = think
            said = []
            opts = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "num_predict": max_new}
            eng.crop(0)
            random.seed(1000 + i)
            real = CH.looping
            CH.looping = lambda tokens, *a, **k: said.append(tokens[-1]) or real(tokens, *a, **k)  # the tokens as emit sees them
            shown = {"think": "", "answer": ""}
            try:
                st = session.ask(user, opts, lambda kind, text: shown.__setitem__(kind, shown[kind] + text), lambda: False)
            finally:
                CH.looping = real
            torch.cuda.synchronize()
            # what the chat handed out piece by piece is the text of the tokens
            kinds, kind = {"think": [], "answer": []}, "answer"
            for token in said:
                if token == session.open or token == session.close:
                    kind = "think" if token == session.open else "answer"
                elif token not in session.stop:
                    kinds[kind].append(token)
            for kind, ids in kinds.items():
                whole = tok.decode(ids, skip_special_tokens=False)
                whole = whole[len("thought\n"):] if kind == "think" and whole.startswith("thought\n") else whole
                assert shown[kind] == whole or (whole.startswith(shown[kind]) and not whole[len(shown[kind]):].strip("\ufffd")), (
                    f"{name}: the {kind} text handed out differs from the tokens' text at "
                    f"{next((k for k, (x, y) in enumerate(zip(shown[kind], whole)) if x != y), min(len(shown[kind]), len(whole)))}")
            seconds, passes = st["seconds"], round(st["tokens"] / max(st["per_pass"], 1e-9))
            n_prompt = st["prompt_tokens"]
        else:
            messages = [{"role": "user", "content": user}]
            out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=think)
            prompt = list(out["input_ids"] if hasattr(out, "keys") else out)
            n_prompt = len(prompt)
            eng.crop(0)
            logits, hidden = eng.feed(prompt, keep="last")
            torch.cuda.synchronize()
            where = lambda: bucket(eng.length)  # noqa: E731
            stats = {}
            guess = drafter.guess if drafter else None
            if args.profile:
                pick, target = Timed(shares, where, 1.0, 64, 0.95, seed=seed_of(i)), Target(eng, shares, where)
                if guess:
                    inner = guess

                    def guess(token, hid, length, inner=inner):
                        b = where()
                        t = time.perf_counter()
                        out = inner(token, hid, length)
                        torch.cuda.synchronize()
                        shares[b]["guess"] += time.perf_counter() - t
                        return out
            else:
                pick, target = Picker(1.0, 64, 0.95, seed=seed_of(i)), eng
            marks = []  # (position, time) at each token, for the speed by position

            def emit(token):
                marks.append((eng.length, time.perf_counter()))

            t = time.perf_counter()
            said = extend(target, len(prompt), logits[-1], hidden[-1], max_new, pick, guess, stop, stats, emit)
            torch.cuda.synchronize()
            seconds, passes = time.perf_counter() - t, stats["passes"]
            for (p0, t0), (p1, t1) in zip([(len(prompt), t)] + marks, marks):
                shares[bucket(p0)]["wall"] += t1 - t0
                shares[bucket(p0)]["tokens"] += 1
        digest = hashlib.sha1(json.dumps(said).encode()).hexdigest()[:12]
        results[name] = said
        line = (f"[speed] {name:7s} prompt {n_prompt:5d}, {len(said):5d} tokens in {passes:4d} passes ({len(said) / max(passes, 1):.2f} a pass), "
                f"{seconds:6.1f} s = {len(said) / seconds:6.1f} tok/s, ended {'by itself' if said[-1] in stop else 'at the limit'}, {digest}")
        if name in stored:
            n = min(len(said), len(stored[name]))
            ok = said[:n] == stored[name][:n] and (len(said) == len(stored[name]) or args.scale != 1.0 or args.chat)
            same = same and ok
            line += f" | {'the same tokens' if ok else 'OTHER TOKENS'} as {args.against}" + ("" if len(said) == len(stored[name]) else f" (first {n} of {len(stored[name])})")
            if not ok:
                first = next((k for k in range(n) if said[k] != stored[name][k]), n)
                line += f"; they part at token {first}"
        print(line, flush=True)
        all_tokens += len(said)
        all_seconds += seconds
    print(f"[speed] all: {all_tokens} tokens in {all_seconds:.1f} s = {all_tokens / max(all_seconds, 1e-9):.1f} tok/s"
          + (f" | {'ALL THE SAME TOKENS' if same else 'TOKENS DIFFER'} as {args.against}" if args.against else ""), flush=True)
    if not args.chat:
        print("[speed] by position in the context" + (" (with a wait for the GPU after each stage)" if args.profile else "") + ":")
        for b in sorted(shares, key=lambda s: int(s.split("k")[0])):
            s = shares[b]
            if not s["tokens"]:
                continue
            line = f"[speed]   {b:8s} {int(s['tokens']):6d} tokens, {s['tokens'] / max(s['wall'], 1e-9):6.1f} tok/s"
            if args.profile and s["cycles"]:
                c = s["cycles"]
                rest = s["wall"] - s["pass"] - s["guess"] - s["pick"]
                line += (f" | a cycle {1000 * s['wall'] / c:5.1f} ms = guesses {1000 * s['guess'] / c:4.1f} + pass {1000 * s['pass'] / c:5.1f} "
                         f"({s['rows'] / c:.1f} rows) + choosing {1000 * s['pick'] / c:4.1f} + the rest {1000 * rest / c:4.1f}; {s['tokens'] / c:.2f} tokens a cycle")
            print(line, flush=True)
    if args.save:
        os.makedirs(FOLDER, exist_ok=True)
        with open(os.path.join(FOLDER, f"{args.save}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(results, f)
        print(f"[speed] stored as {args.save}", flush=True)
    if args.against and not same:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
