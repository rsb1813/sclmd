# Generation with drafts: a drafter guesses the next few tokens, the main model runs them all in one pass and keeps the
# guesses that are what it would have produced itself. The text is the main model's own: a guess is kept only when
# it equals the token the main model picks at that position, and the pick does not depend on the guesses.
import torch
import torch.nn.functional as F

from . import model as M


class Picker:
    """Chooses the token at an absolute position from the main model's logits there. Sampling draws its randomness from
    (seed, position), so the choice at a position is the same however the tokens before it were arrived at."""

    def __init__(self, temperature: float = 0.0, top_k: int = 64, top_p: float = 0.95, seed: int = 0):
        self.temperature, self.top_k, self.top_p, self.seed = temperature, top_k, top_p, seed

    def __call__(self, logits: torch.Tensor, position: int) -> int:
        if self.temperature <= 0:
            return int(logits.argmax())
        vals, idx = (logits.float() / self.temperature).topk(self.top_k)
        probs = F.softmax(vals, dim=-1)
        probs = probs * (probs.cumsum(-1) - probs < self.top_p)
        g = torch.Generator().manual_seed((self.seed * 1000003 + position) & 0x7FFFFFFF)
        u = torch.rand((), generator=g).item() * probs.sum().item()
        return int(idx[min(int((probs.cumsum(-1) < u).sum()), self.top_k - 1)])


class RefTarget:
    """The main model as generation needs it, on the reference decoder with the whole model in memory (small models).
    The engine offers the same four things."""

    def __init__(self, cfg: M.Config, state: dict, prefix: str = "model."):
        self.cfg, self.state, self.prefix = cfg, state, prefix
        self.cache, self.shared = M.Cache(), {}

    def feed(self, ids: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Run tokens at the next positions -> (logits [n, vocab], last outputs after the final norm [n, hidden])."""
        logits, hidden = M.forward(self.cfg, self.state, self.prefix, torch.tensor([ids]), self.cache, self.shared)
        return logits[0], hidden[0]

    def crop(self, length: int):
        """Forget every position from `length` on."""
        self.cache.crop(length)

    def embed(self, ids: list) -> torch.Tensor:
        w = self.state[f"{self.prefix}embed_tokens.weight"]
        return w[torch.tensor(ids)] * M.embed_scale(self.cfg, w.dtype)

    def keys(self, length: int) -> dict:
        """What the drafter attends to, cut to the first `length` positions."""
        return {t: (k[:, :, :length], v[:, :, :length], pos[:length]) for t, (k, v, pos) in self.shared.items()}


def drafter_guess(drafter, target, depth: int):
    """guess(token, hidden, length) -> up to `depth` tokens after `token`, which sits at position `length` (the main model
    has run the positions before it; `hidden` is its last output at length - 1)."""

    dtype = drafter.pre.dtype  # the main model hands over its own types (the engine: float32 outputs, fp16 keys)

    def guess(token: int, hidden: torch.Tensor, length: int) -> list:
        keys = {t: (k.to(dtype), v.to(dtype), pos) for t, (k, v, pos) in target.keys(length).items()}
        out, h = [], hidden[None, None].to(dtype)
        last = torch.tensor([length - 1], device=hidden.device)
        for _ in range(depth):
            logits, h = drafter.step(target.embed([token])[None].to(dtype), h, last, keys)
            token = int(logits[0, 0].argmax())
            out.append(token)
        return out

    return guess


@torch.no_grad()
def generate(target, prompt: list, max_new: int, pick: Picker, guess=None, stop: tuple = (), stats: dict | None = None) -> list:
    """Tokens after `prompt`. With `guess`, each pass of the main model runs the pending token and the guesses after it."""
    logits, hidden = target.feed(prompt)
    n = len(prompt)  # positions the main model has run and kept
    pending = pick(logits[-1], n)  # the token at position n, not yet run
    last_hidden = hidden[-1]
    out = []
    passes = 0
    while len(out) < max_new:
        out.append(pending)
        if pending in stop or len(out) >= max_new:
            break
        drafts = list(guess(pending, last_hidden, n))[: max_new - len(out)] if guess else []
        logits, hidden = target.feed([pending] + drafts)  # logits[j] is for position n + j + 1
        passes += 1
        kept = 0
        for j, d in enumerate(drafts):
            want = pick(logits[j], n + j + 1)
            if want != d or d in stop:
                break
            out.append(d)
            kept += 1
        if kept < len(drafts):
            target.crop(n + 1 + kept)
        pending = pick(logits[kept], n + kept + 1)
        last_hidden = hidden[kept]
        n += 1 + kept  # a guessed stop token is not kept above: it comes back as `pending` and ends the loop
    if stats is not None:
        stats.update(passes=passes, tokens=len(out))
    return out
