# The drafter (Gemma4Assistant): a 4-layer model that guesses the next few tokens from the main model's last output.
# It has no keys of its own: every layer attends to the main model's keys and values of its last sliding and last
# full-attention layer. Its input is the main model's embedding of the last token joined with a hidden state: the main
# model's last output for the first guess, its own projected output after that.
import torch
import torch.nn.functional as F

from . import model as M
from .checkpoint import Checkpoint


class Drafter:
    def __init__(self, cfg: M.Config, layers: list, pre: torch.Tensor, post: torch.Tensor, norm: torch.Tensor, head: torch.Tensor):
        self.cfg, self.layers, self.pre, self.post, self.norm, self.head = cfg, layers, pre, post, norm, head

    @classmethod
    def load(cls, folder: str, device: str = "cuda") -> "Drafter":
        ck = Checkpoint(folder)
        layers = [ck.layer(i, device) for i in range(ck.cfg.n_layers)]
        return cls(ck.cfg, layers, ck.get("pre_projection.weight", device), ck.get("post_projection.weight", device),
                   ck.norm(device), ck.embed(device))

    @classmethod
    def from_state(cls, cfg: M.Config, state: dict) -> "Drafter":
        layers = [M.layer_weights(state, "model.", i) for i in range(cfg.n_layers)]
        return cls(cfg, layers, state["pre_projection.weight"], state["post_projection.weight"], state["model.norm.weight"],
                   state["model.embed_tokens.weight"])

    @torch.no_grad()
    def step(self, token_embed: torch.Tensor, hidden: torch.Tensor, last: torch.Tensor, shared: dict, linear=F.linear,
             rounded=None) -> tuple[torch.Tensor, torch.Tensor]:
        """One guess for each of n places in a sequence.
        token_embed [batch, n, backbone]: the main model's (scaled) embedding of the token just before the guess.
        hidden [batch, n, backbone]: the main model's last output (after its final norm) at the position before that
            token, or this function's second result when guessing further ahead.
        last [n]: the last position whose keys each place may read; the place itself sits at last + 1.
        shared: layer type -> (keys, values, their positions).
        linear, rounded: as in model.layer_forward (a test of the engine's drafter passes the engine's arithmetic).
        Returns (logits [batch, n, vocab], the hidden state for the guess after this one)."""
        cfg = self.cfg
        h = linear(torch.cat((token_embed, hidden), dim=-1), self.pre)
        for i, w in enumerate(self.layers):
            # transformers lets the drafter read `window + 1` keys in a sliding layer
            h = M.layer_forward(cfg, i, w, h, last + 1, shared=shared, see=(last, cfg.window + 1), linear=linear, rounded=rounded)
        h = M.rms_norm(h, self.norm, cfg.eps)
        return linear(h, self.head), linear(h, self.post)
