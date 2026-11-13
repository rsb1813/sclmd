# How many of the drafter's guesses a verification would accept, measured without running the main model again:
# the main model's stored outputs over a token set are fed to the drafter at every place, and each guess is compared
# with the main model's own distribution there. Guesses further ahead are conditioned on the text's real tokens, which
# is the case in which they matter (a later guess is only looked at when the earlier ones were accepted).
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from sclmd import DATA, ckpt
from sclmd import model as M
from sclmd.checkpoint import Checkpoint
from sclmd.drafter import Drafter
from sclmd.stream import cap_vram
from scripts.reference import load_set

DEPTH = 6  # generation_config.json of the drafter: num_assistant_tokens


def sampler(logits: torch.Tensor, top_k: int = 64, top_p: float = 0.95) -> torch.Tensor:
    """The distribution the model card's sampling settings draw from (temperature 1, top-k 64, top-p 0.95)."""
    vals, idx = logits.topk(top_k, dim=-1)
    probs = F.softmax(vals, dim=-1)
    keep = probs.cumsum(-1) - probs < top_p  # the smallest set reaching top_p
    probs = probs * keep
    probs = probs / probs.sum(-1, keepdim=True)
    return torch.zeros_like(logits).scatter_(-1, idx, probs)


def expected_tokens(accept: torch.Tensor) -> torch.Tensor:
    """accept [places, depth]: chance each guess is accepted given the earlier ones were -> tokens gained per
    verification at each place (the accepted guesses plus the one token the main model always adds)."""
    return 1 + accept.cumprod(dim=1).sum(dim=1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="eval_v1")
    ap.add_argument("--ref", default="orig", help="whose stored outputs to draft from (data/<set>_<ref>.pt)")
    ap.add_argument("--drafter", default="drafter")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--places", type=int, default=128, help="places drafted together")
    ap.add_argument("--codes", help="the drafter's matrices as stored pair codes of this name (scripts/quantize_drafter.py)")
    ap.add_argument("--rate", type=int, default=8, help="bits a pair for --codes")
    args = ap.parse_args()
    cap_vram()
    dev = "cuda"
    ids, category, names, target = load_set(args.set)
    stored = torch.load(os.path.join(DATA, f"{args.set}_{args.ref}.pt"), weights_only=True)
    ck = Checkpoint(ckpt(args.ref))
    cfg, embed, norm = ck.cfg, ck.embed(dev), ck.norm(dev)
    scale = M.embed_scale(cfg, embed.dtype)
    drafter = Drafter.load_codes(ckpt(args.drafter), args.codes, args.rate, dev) if args.codes else Drafter.load(ckpt(args.drafter), dev)
    s, n = ids.shape
    keys = ("greedy", "sampled", "sampled_raw", "compare")
    per_seq = {k: [] for k in keys}
    depth_sum = {k: torch.zeros(DEPTH, dtype=torch.float64) for k in keys}
    count = 0
    for q in range(s):
        last = M.rms_norm(stored["hidden"][q].to(dev), norm, cfg.eps)  # [n, hidden]: what the drafter is handed
        shared = {t: (k[q : q + 1].to(dev), v[q : q + 1].to(dev), torch.arange(n, device=dev)) for t, (k, v) in stored["shared"].items()}
        tok = ids[q].to(dev)
        # t: the last position the main model has run. Only places whose next token is the model's own count
        places = torch.arange(0, n - DEPTH - 1, device=dev)
        places = places[target[q, 1 : n - DEPTH].to(dev)][:: args.stride]
        if places.numel() == 0:
            for k in keys:
                per_seq[k].append(float("nan"))
            continue
        acc = {k: [] for k in keys}
        for a in range(0, places.numel(), args.places):
            t = places[a : a + args.places]
            hidden = last[t][None]
            rows = {k: [] for k in keys}
            for j in range(1, DEPTH + 1):
                logits_q, hidden = drafter.step((embed[tok[t + j]] * scale)[None], hidden, t, shared)
                logits_q = logits_q[0].float()
                logits_p = M.logits_of(cfg, stored["hidden"][q].to(dev)[t + j], norm, embed).float()  # the main model after the same tokens
                guess = logits_q.argmax(-1)
                p_raw, q_raw = F.softmax(logits_p, -1), F.softmax(logits_q, -1)
                p_s, q_s = sampler(logits_p), sampler(logits_q)
                rows["greedy"].append((guess == logits_p.argmax(-1)).double())  # greedy decoding: the guess is the main model's choice
                rows["sampled"].append(torch.minimum(p_s, q_s).sum(-1).double())  # speculative sampling, both under the card's sampler
                rows["sampled_raw"].append(torch.minimum(p_raw, q_raw).sum(-1).double())  # the same at temperature 1 with no cut
                rows["compare"].append(p_s.gather(1, guess[:, None])[:, 0].double())  # draw from the main model, keep the guess if equal
            for k in keys:
                acc[k].append(torch.stack(rows[k], dim=1).cpu())
        for k in keys:
            m = torch.cat(acc[k])
            per_seq[k].append(expected_tokens(m).mean().item())
            depth_sum[k] += m.sum(0)
        count += places.numel()
        print(f"sequence {q + 1}/{s} ({names[category[q]]}): " + ", ".join(f"{k} {per_seq[k][-1]:.2f}" for k in keys), flush=True)
    report = {"set": args.set, "ref": args.ref, "drafter": args.drafter, "depth": DEPTH, "places": count,
              "tokens_per_verification": {}, "accept_by_depth": {k: (depth_sum[k] / count).tolist() for k in keys}}
    for ci, name in enumerate(names):
        rows = [i for i, c in enumerate(category) if c == ci]
        report["tokens_per_verification"][name] = {k: float(np.nanmean([per_seq[k][i] for i in rows])) for k in keys}
    report["tokens_per_verification"]["all"] = {k: float(np.nanmean(per_seq[k])) for k in keys}
    tag = args.drafter + (f"_{args.codes}_c{args.rate}" if args.codes else "")
    with open(os.path.join(DATA, f"draft_replay_{args.set}_{args.ref}_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
