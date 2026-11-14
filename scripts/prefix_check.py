# Prefix states (B4's kill test): a system prompt many requests share is read once, its keys and values are kept
# (Engine.save to pinned RAM, or on a sequence of its own and copied with Engine.fork), and each request reads only
# what follows it. Checks that the reply tokens are the same as after reading the whole prompt, and times the copies.
#
#   python -m scripts.prefix_check [--items 20] [--disk 6000]
import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from sclmd import ckpt
from sclmd import engine as E
from sclmd import model as M
from sclmd import plan as P
from sclmd.batch import Batcher, Job
from sclmd.generate import Picker
from sclmd.stream import VRAM_MARGIN, cap_vram

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = os.path.join(ROOT, "data", "task_eval", "items.jsonl")
GUIDE = ("You are a careful translator and text processor for a Korean-English team. Follow these rules.\n"
         "1. Translate meaning, not word order. Keep names, numbers, units and code exactly as written.\n"
         "2. Korean output uses the polite written style (합니다체) unless the source is casual speech.\n"
         "3. English output uses plain, natural American English; do not add explanations.\n"
         "4. Keep line breaks and list markers of the source. Never add a title or a note.\n"
         "5. When asked to extract or list, answer with exactly what is asked, in the format asked.\n"
         "Glossary (use these renderings): server = 서버, request = 요청, reply = 응답, prompt = 프롬프트, "
         "translation = 번역, schedule = 일정, invoice = 청구서, deadline = 마감일, release = 출시, "
         "user = 사용자, account = 계정, settings = 설정, error = 오류, update = 업데이트.\n")


def system(items: list, want: int, tok) -> str:
    """The guide and as many worked examples (other items with their gold answers) as make about `want` tokens."""
    text, n = GUIDE + "\nExamples:\n", 0
    for it in items:
        if len(tok.encode(text, add_special_tokens=False)) >= want:
            break
        text += f"\nInput: {it['prompt']}\nOutput: {it['gold']}\n"
        n += 1
    return text


def ids(tok, messages):
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return list(out["input_ids"] if hasattr(out, "keys") else out)


def common(a: list, b: list) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def sync_ms(t0: float) -> float:
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--plan", default="c6")
    ap.add_argument("--head", type=int, default=8)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--capacity", type=int, default=6144 + 256)
    ap.add_argument("--items", type=int, default=20)
    ap.add_argument("--sizes", default="300,1000")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--disk", type=int, default=6000, help="tokens of the disk round trip (0: none)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "prefix_check.json"))
    args = ap.parse_args()
    cap_vram()
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    cfg = M.Config.load(ckpt("orig"))
    plan, head = (P.uniform(cfg, 6), args.head) if args.plan == "c6" else P.read(args.plan)
    head = head or args.head
    eng = E.Engine.load(args.name, plan, head, args.capacity, sequences=3, rows=args.rows)
    drafter = E.EngineDrafter.load(eng, ckpt("drafter"), args.name, depth=6)
    stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"))
    with open(ITEMS, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    kinds = list(dict.fromkeys(it["kind"] for it in items))
    shots = [it for it in items if it["kind"].startswith("tr_")][-40:]  # examples come from the end, the asked items from the front
    asked = []
    for k in range(args.items):  # round robin over the kinds
        pool = [it for it in items[:-40] if it["kind"] == kinds[k % len(kinds)] and it not in asked]
        asked.append(pool[0])
    pick = lambda: Picker(0.0, 1, 1.0, seed=0)  # noqa: E731

    def reply(seq: int, prompt: list, base: list) -> tuple[list, float, int]:
        b = Batcher(eng, drafter, 6)
        job = Job(seq, prompt, args.max_new, pick(), stop, base=base)
        b.add(job)
        torch.cuda.synchronize()
        t0, first = time.perf_counter(), None
        while b.step():
            if first is None and job.out:
                first = time.perf_counter() - t0
        return job.out, 1000 * first, job.passes

    report = {"sizes": {}}
    for size in [int(s) for s in args.sizes.split(",")]:
        sys_text = system(shots, size, tok)
        msgs = lambda it: [{"role": "system", "content": sys_text}, {"role": "user", "content": it["prompt"]}]  # noqa: E731
        fulls = [ids(tok, msgs(it)) for it in asked]
        n = min(common(fulls[0], f) for f in fulls[1:])  # the shared prefix, on token ids
        prefix = fulls[0][:n]
        # the prefix read once, on sequence 2
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eng.feed(prefix, keep="last", seq=2)
        read_ms = sync_ms(t0)
        t0 = time.perf_counter()
        state = eng.save(2)
        save_ms = sync_ms(t0)
        rec = {"prefix_tokens": n, "read_ms": round(read_ms, 1), "save_ms": round(save_ms, 2), "state_mb": round(state["bytes"] / 2**20, 1),
               "items": []}
        same = 0
        for it, full in zip(asked, fulls):
            a, ttft_a, _ = reply(0, full, [])
            eng.release(0)
            t0 = time.perf_counter()
            eng.unpark(1, state)
            unpark_ms = sync_ms(t0)
            b, ttft_b, _ = reply(1, full[n:], prefix)
            eng.release(1)
            t0 = time.perf_counter()
            eng.fork(2, 1)
            fork_ms = sync_ms(t0)
            c, ttft_c, _ = reply(1, full[n:], prefix)
            eng.release(1)
            ok = a == b == c
            same += ok
            rec["items"].append({"id": it["id"], "tokens": len(full), "suffix": len(full) - n, "same": ok, "reply": len(a),
                                 "ttft_full_ms": round(ttft_a, 1), "ttft_ram_ms": round(ttft_b, 1), "ttft_vram_ms": round(ttft_c, 1),
                                 "unpark_ms": round(unpark_ms, 2), "fork_ms": round(fork_ms, 2)})
            if not ok:
                print(f"[prefix] {it['id']}: DIFFERENT\n  full {a}\n  ram  {b}\n  vram {c}", flush=True)
        eng.release(2)
        its = rec["items"]
        med = lambda key: sorted(x[key] for x in its)[len(its) // 2]  # noqa: E731
        rec |= {"same": f"{same}/{len(its)}", "median": {k: med(k) for k in ("suffix", "ttft_full_ms", "ttft_ram_ms", "ttft_vram_ms", "unpark_ms", "fork_ms")}}
        report["sizes"][size] = rec
        print(f"[prefix] {n}-token prefix ({rec['state_mb']} MB, read {read_ms:.0f} ms, saved in {save_ms:.1f} ms): the same replies for "
              f"{same} of {len(its)}; medians {rec['median']}", flush=True)
    if args.disk:
        # the disk tier: a long prefix read, saved, written to disk, read back and put on a sequence
        text = " ".join(it["prompt"] for it in items)
        long = tok.encode(text, add_special_tokens=False)
        while len(long) < args.disk:
            long += long
        long = [tok.bos_token_id] + long[: args.disk - 1]
        eng.feed(long, keep="last", seq=2)
        state = eng.save(2)
        path = os.path.join(ROOT, "data", "prefix_check.pt")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.save(state, path)
        save_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        back = torch.load(path, weights_only=True)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        eng.unpark(1, back)
        unpark_s = sync_ms(t0) / 1000
        t0 = time.perf_counter()
        eng.release(1)
        for t, part in back["kinds"].items():  # pinned: the copy to the GPU at full speed
            part["data"] = [d.pin_memory() for d in part["data"]]
        pin_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        eng.unpark(1, back)
        unpark_pinned_s = sync_ms(t0) / 1000
        tail = tok.encode("\n\nSummarize the text above in one sentence.", add_special_tokens=False)
        b_out, _, _ = reply(1, tail, long)
        eng.release(1)
        a_out, _, _ = reply(0, long + tail, [])
        eng.release(0)
        eng.release(2)
        os.remove(path)
        report["disk"] = {"tokens": args.disk, "mb": round(state["bytes"] / 2**20, 1), "torch_save_s": round(save_s, 3), "torch_load_s": round(load_s, 3),
                          "unpark_unpinned_s": round(unpark_s, 3), "pin_s": round(pin_s, 3), "unpark_pinned_s": round(unpark_pinned_s, 3),
                          "round_trip_s": round(save_s + load_s + unpark_s, 3), "same_reply": a_out == b_out}
        print(f"[prefix] disk: {report['disk']}", flush=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
