# Whole sequences through a checkpoint one layer at a time: a 62 GB model on a 16 GB card, about 1 GB of weights at once.
import time

import torch
import torch.nn.functional as F

from . import model as M
from .checkpoint import Checkpoint

VRAM_MARGIN = 768 << 20  # left free for the driver and other processes


def cap_vram(device: str = "cuda"):
    """Windows pages over-subscribed VRAM out to system RAM instead of failing, and everything slows to a crawl.
    Cap PyTorch below the VRAM that is free now, so it fails instead."""
    if not str(device).startswith("cuda"):
        return
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    free, total = torch.cuda.mem_get_info(index)
    torch.cuda.set_per_process_memory_fraction((free - VRAM_MARGIN) / total, index)


class Stream:
    def __init__(self, ckpt: Checkpoint, device: str = "cuda", batch: int = 2):
        self.ckpt, self.cfg, self.device, self.batch = ckpt, ckpt.cfg, device, batch

    @torch.no_grad()
    def run(self, ids: torch.Tensor, n_layers: int | None = None, weights=None, tap=None, log=None) -> dict:
        """ids [sequences, n], every sequence from position 0.
        weights(i, w) -> w: replaces a layer's weights (quantized variants).
        tap(i, name, tensor, rows): the input of each group of matrices of layer i, for the sequences `rows`.
        Returns "hidden" (the last layer's output, before the final norm) and "shared" (the keys and values the drafter
        attends to: layer type -> (keys, values))."""
        cfg, dev = self.cfg, self.device
        n_layers = cfg.n_layers if n_layers is None else n_layers
        ids = ids.to(dev)
        s, n = ids.shape
        positions = torch.arange(n, device=dev)
        embed = self.ckpt.embed(dev)
        h = torch.cat([embed[ids[a : a + self.batch]] * M.embed_scale(cfg, embed.dtype) for a in range(0, s, self.batch)])
        del embed
        shared_all = {}
        t0 = time.time()
        for i in range(n_layers):
            w = self.ckpt.layer(i, dev)
            if weights is not None:
                w = weights(i, w)
            for a in range(0, s, self.batch):
                rows = slice(a, min(a + self.batch, s))
                shared = {}
                h[rows] = M.layer_forward(cfg, i, w, h[rows], positions, shared=shared,
                                          tap=(lambda name, x, i=i, rows=rows: tap(i, name, x, rows)) if tap else None)
                for t, (k, v, _) in shared.items():
                    shared_all.setdefault(t, []).append((k.cpu(), v.cpu()))
            del w
            if log:
                log(f"layer {i + 1}/{n_layers}, {time.time() - t0:.0f} s")
        out = {"hidden": h, "shared": {t: (torch.cat([k for k, _ in kv]), torch.cat([v for _, v in kv])) for t, kv in shared_all.items()}}
        return out


@torch.no_grad()
def logits_chunks(cfg: M.Config, hidden: torch.Tensor, norm: torch.Tensor, embed: torch.Tensor, chunk: int = 256):
    """Yield (start, logits [chunk, vocab] in float32) over hidden [positions, hidden size]."""
    for a in range(0, hidden.shape[0], chunk):
        yield a, M.logits_of(cfg, hidden[a : a + chunk].to(embed.device), norm, embed).float()


@torch.no_grad()
def compare(cfg: M.Config, ref_hidden: torch.Tensor, hidden: torch.Tensor, norm: torch.Tensor, embed: torch.Tensor, ids: torch.Tensor,
            test_norm: torch.Tensor | None = None, test_embed: torch.Tensor | None = None) -> dict:
    """A path's next-token distributions against the reference's, per sequence, from the two last-layer outputs
    [sequences, n, hidden size]: KL(reference || path), agreement of the first choice, and both perplexities on `ids`."""
    s, n, d = hidden.shape
    test_norm = norm if test_norm is None else test_norm
    test_embed = embed if test_embed is None else test_embed
    kl, top1, nll_ref, nll = (torch.zeros(s, dtype=torch.float64) for _ in range(4))
    for q in range(s):
        target = ids[q, 1:].to(embed.device)
        for (a, lr), (_, lt) in zip(logits_chunks(cfg, ref_hidden[q], norm, embed), logits_chunks(cfg, hidden[q], test_norm, test_embed)):
            pr, pt = F.log_softmax(lr, dim=-1), F.log_softmax(lt, dim=-1)
            kl[q] += (pr.exp() * (pr - pt)).sum(-1).double().sum().cpu()
            top1[q] += (pr.argmax(-1) == pt.argmax(-1)).double().sum().cpu()
            tg = target[a : a + pr.shape[0]]
            m = tg.shape[0]  # the last position has no next token
            nll_ref[q] -= pr[:m].gather(1, tg[:, None]).double().sum().cpu()
            nll[q] -= pt[:m].gather(1, tg[:, None]).double().sum().cpu()
    return {"kl": (kl / n).tolist(), "top1": (top1 / n).tolist(), "ppl_ref": (nll_ref / (n - 1)).exp().tolist(), "ppl": (nll / (n - 1)).exp().tolist()}
