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
    ap.add_argument("--layers", type=int, default=6, help="5 sliding layers and the first full-attention layer")
    ap.add_argument("--tokens", type=int, default=1536, help="past the sliding window of 1,024")
    ap.add_argument("--set", default="eval_v1")
    args = ap.parse_args()
    from transformers import Gemma4TextConfig, Gemma4TextModel
    from transformers.modeling_utils import no_init_weights

    cap_vram()
    ck = Checkpoint(ckpt(args.ckpt))
    ids = load_set(args.set)[0][:1, : args.tokens]
    ours = Stream(ck, batch=1).run(ids, n_layers=args.layers)["hidden"]

    with open(os.path.join(ck.folder, "config.json"), encoding="utf-8") as f:
        text = json.load(f)["text_config"]
    text.update(num_hidden_layers=args.layers, layer_types=text["layer_types"][: args.layers], vocab_size=8, dtype="bfloat16")
    with no_init_weights():
        hf = Gemma4TextModel(Gemma4TextConfig(**text)).to(torch.bfloat16)
    state = {name: ck.get(ck.prefix + name) for name in hf.state_dict() if name.startswith("layers.")}
    missing = hf.load_state_dict(state, strict=False)
    assert set(missing.missing_keys) <= {"embed_tokens.weight", "norm.weight"} and not missing.unexpected_keys, missing
    hf = hf.cuda().eval()
    embed = ck.embed("cuda")
    with torch.no_grad():
        x = embed[ids.cuda()] * M.embed_scale(ck.cfg, embed.dtype)
        # transformers ends the stack with its final norm; with the norm's weight at one that is a plain RMS norm
        hf.norm.weight.fill_(1.0)
        want = hf(inputs_embeds=x).last_hidden_state
        got = M.rms_norm(ours, None, ck.cfg.eps)
    cos = torch.nn.functional.cosine_similarity(got.float(), want.float(), dim=-1)
    rel = ((got.float() - want.float()).norm(dim=-1) / want.float().norm(dim=-1))
    print(json.dumps({"layers": args.layers, "tokens": args.tokens, "cosine_min": cos.min().item(), "cosine_mean": cos.mean().item(),
                      "relative_error_mean": rel.mean().item(), "relative_error_max": rel.max().item()}, indent=2))


if __name__ == "__main__":
    main()
