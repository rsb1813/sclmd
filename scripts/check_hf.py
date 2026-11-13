# The reference decoder against transformers on the real weights: the first layers of the checkpoint, one long sequence.
# (The small-model test covers the logic; this covers the real config, the weight names and bf16 arithmetic.)
import argparse
import json
import os

import torch

from sclmd import ckpt
from sclmd import model as M
from sclmd.checkpoint import Checkpoint
from sclmd.stream import Stream, cap_vram
from scripts.reference import load_set


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="orig")
    ap.add_argument("--layers", type=int, default=6, help="5 sliding layers and a full-attention layer")
    ap.add_argument("--first", type=int, default=0, help="the first of the layers compared; a multiple of 6")
    ap.add_argument("--tokens", type=int, default=1536, help="past the sliding window of 1,024")
    ap.add_argument("--set", default="eval_v1")
    args = ap.parse_args()
    from transformers import Gemma4TextConfig, Gemma4TextModel
    from transformers.initialization import no_init_weights

    cap_vram()
    ck = Checkpoint(ckpt(args.ckpt))
    ids = load_set(args.set)[0][:1, : args.tokens]
    first, last = args.first, args.first + args.layers
    stream = Stream(ck, batch=1)
    ours = stream.run(ids, n_layers=last)["hidden"]
    start = stream.run(ids, n_layers=first)["hidden"]  # what both are given: our own output of the layers before

    with open(os.path.join(ck.folder, "config.json"), encoding="utf-8") as f:
        text = json.load(f)["text_config"]
    text.update(num_hidden_layers=args.layers, layer_types=text["layer_types"][first:last], vocab_size=8, dtype="bfloat16")
    with no_init_weights():
        hf = Gemma4TextModel(Gemma4TextConfig(**text)).to(torch.bfloat16)
    there = lambda name: ck.prefix + f"layers.{int(name.split('.')[1]) + first}." + name.split(".", 2)[2]  # noqa: E731
    state = {name: ck.get(there(name)) for name in hf.state_dict() if name.startswith("layers.")}
    missing = hf.load_state_dict(state, strict=False)
    assert set(missing.missing_keys) <= {"embed_tokens.weight", "norm.weight"} and not missing.unexpected_keys, missing
    hf = hf.cuda().eval()
    with torch.no_grad():
        x = start
        # transformers ends the stack with its final norm; with the norm's weight at one that is a plain RMS norm
        hf.norm.weight.fill_(1.0)
        want = hf(inputs_embeds=x).last_hidden_state
        got = M.rms_norm(ours, None, ck.cfg.eps)
    cos = torch.nn.functional.cosine_similarity(got.float(), want.float(), dim=-1)
    rel = ((got.float() - want.float()).norm(dim=-1) / want.float().norm(dim=-1))
    print(json.dumps({"first": first, "layers": args.layers, "tokens": args.tokens, "cosine_min": cos.min().item(), "cosine_mean": cos.mean().item(),
                      "relative_error_mean": rel.mean().item(), "relative_error_max": rel.max().item()}, indent=2))


if __name__ == "__main__":
    main()
