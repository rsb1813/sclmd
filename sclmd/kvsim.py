# The rounding of the keys and values the engine keeps, simulated in PyTorch (scripts/eval_codes.py --kv, through
# Stream.run's `kept` hook): int8 as the engine keeps them today, and the K3 brief's 4-bit (or 3-bit) format for the
# sliding layers, in a per-head Hadamard basis. No kernel reads this: it says what such a format would cost before
# one is written.
#
# The 4-bit format of one vector x (a position's key or value in one key head, 256 values):
#   y = H (s * x)        s: fixed random signs of (layer, head); H: the 256-point Walsh-Hadamard matrix / 16 (orthonormal)
#   r = RMS of y         kept in fp32 (today's scale slot), as r * top / 127
#   code = the nearest of the Gaussian Lloyd-Max levels, put on the int8 grid (16 levels: +-6, 18, 31, 44, 58, 75, 96, 127)
#   x' = s * (H y')      y' = code * r * top / 127
# Because the sliding layers keep their keys after the rotary turn, q . x' = (H s q) . y': a kernel can score the
# rotated query against the rotated keys, and un-rotate the weighted sum of values once per (row, head).
import math

import torch

from . import model as M

MODES = ("int8", "k8v4", "k4v4", "k3v3", "k4v4t0", "k8v4t0")  # t0: a sequence's first tile (positions 0-15) kept at int8


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi) if math.isfinite(x) else 0.0


def _cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2))) if math.isfinite(x) else (1.0 if x > 0 else 0.0)


def lloyd_max(levels: int, iters: int = 5000) -> tuple[list[float], float]:
    """The `levels` Lloyd-Max levels of a unit Gaussian (ascending) and their mean squared error, by Lloyd's iteration
    with the exact centroids of the normal between two thresholds."""
    c = [(-1 + (2 * j + 1) / levels) * 2.5 for j in range(levels)]
    for _ in range(iters):
        t = [-math.inf] + [(c[j] + c[j + 1]) / 2 for j in range(levels - 1)] + [math.inf]
        c = [(_phi(t[j]) - _phi(t[j + 1])) / (_cdf(t[j + 1]) - _cdf(t[j])) for j in range(levels)]
    t = [-math.inf] + [(c[j] + c[j + 1]) / 2 for j in range(levels - 1)] + [math.inf]
    mse = 0.0
    for j in range(levels):
        a, b = t[j], t[j + 1]
        mass = _cdf(b) - _cdf(a)
        first = _phi(a) - _phi(b)
        second = mass + (a * _phi(a) if math.isfinite(a) else 0.0) - (b * _phi(b) if math.isfinite(b) else 0.0)
        mse += second - 2 * c[j] * first + c[j] ** 2 * mass
    return c, mse


def grid_levels(bits: int) -> tuple[torch.Tensor, float]:
    """(the levels as int8 codes, ascending; the top level in units of the RMS): the Gaussian Lloyd-Max levels of
    2**bits cells, the top one at 127."""
    c, _ = lloyd_max(2**bits)
    top = c[-1]
    return torch.tensor([float(round(x * 127 / top)) for x in c]), top


def hadamard(n: int) -> torch.Tensor:
    """The n-point Walsh-Hadamard matrix (Sylvester's order) divided by sqrt(n): symmetric and orthonormal."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / math.sqrt(n)).float()


def quantiles(kl: torch.Tensor) -> dict:
    """Mean and quantiles of per-position KL values."""
    kl = kl.double()
    q = torch.quantile(kl, torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float64))
    return {"positions": kl.numel(), "mean": kl.mean().item(), "p50": q[0].item(), "p90": q[1].item(), "p99": q[2].item(),
            "p999": q[3].item(), "max": kl.max().item()}


def int8_round(x: torch.Tensor) -> torch.Tensor:
    """The engine's rounding (heads_k): each vector (last dim) divided by its largest size over 127, rounded to the
    nearest integer (ties to even, as rintf), times that scale."""
    most = x.abs().amax(-1, keepdim=True)
    scale = torch.where(most > 0, most / 127.0, torch.ones_like(most))
    return torch.round(x / scale) * scale


class Kept:
    """kept(i, kind, x) for Stream.run: layer i's new keys (kind "k", after the rotary turn) or values ("v", RMS-normed),
    x [batch, key heads, positions, head size], rounded as the mode keeps them.
    int8: every layer as the engine keeps it today. k4v4 / k3v3: 4 or 3 bits in the Hadamard basis for the keys and
    values of the sliding layers other than cfg.last_of(SLIDING) (the layer the drafter reads); int8 elsewhere.
    k8v4: the same with the keys left at int8.
    noise: (layer, kind) -> [squared error, squared size] summed over what was rounded (relative noise energy)."""

    def __init__(self, cfg: M.Config, mode: str, seed: int = 0):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}")
        self.cfg, self.mode, self.seed = cfg, mode, seed
        low = 3 if mode == "k3v3" else 4
        # positions kept at int8 whatever the layer's format (the sequence's first tile: where attention's sinks are);
        # the hook sees whole sequences from position 0, so a position is its index along dim 2
        self.first = 16 if mode.endswith("t0") else 0
        drafter = cfg.last_of(M.SLIDING)
        self.bits = {}
        for i, t in enumerate(cfg.layer_types):
            if mode == "int8" or t != M.SLIDING or i == drafter:
                self.bits[i] = {"k": 8, "v": 8}
            else:
                self.bits[i] = {"k": 8 if mode.startswith("k8") else low, "v": low}
        hd = cfg.head_dim[M.SLIDING]
        g = torch.Generator().manual_seed(seed)
        # one sign vector per (layer, key head), drawn once: the same for every batch of sequences
        self.signs = (torch.randint(0, 2, (cfg.n_layers, cfg.kv_heads[M.SLIDING], hd), generator=g) * 2 - 1).float()
        self.h = hadamard(hd)
        self.levels = {b: grid_levels(b) for b in (3, 4)}
        self._dev = {}
        self.noise = {}

    def _on(self, dev):
        if dev not in self._dev:
            self._dev[dev] = (self.signs.to(dev), self.h.to(dev),
                              {b: (lv.to(dev), ((lv[1:] + lv[:-1]) / 2).to(dev), top) for b, (lv, top) in self.levels.items()})
        return self._dev[dev]

    def low_bits(self, i: int, x: torch.Tensor, bits: int) -> torch.Tensor:
        signs, h, levels = self._on(x.device)
        lv, cut, top = levels[bits]
        s = signs[i][None, :, None, :]  # [1, key heads, 1, head size]
        y = (x.float() * s) @ h
        scale = y.pow(2).mean(-1, keepdim=True).sqrt() * (top / 127)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        code = lv[torch.bucketize(y / scale, cut)]  # the nearest level: the cuts are the midpoints
        return ((code * scale) @ h * s).to(x.dtype)

    def __call__(self, i: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        return self.at(i, kind, x, 0)

    def at(self, i: int, kind: str, x: torch.Tensor, start: int) -> torch.Tensor:
        """__call__ for positions start .. start + n - 1 (x's dim 2): the first tile is the positions below 16, whatever
        part of the sequence x holds (an engine's test feeds a sequence in pieces)."""
        bits = self.bits[i][kind]
        out = int8_round(x) if bits == 8 else self.low_bits(i, x, bits)
        if bits < 8 and self.first > start:
            out[..., : self.first - start, :] = int8_round(x[..., : self.first - start, :])
        acc = self.noise.setdefault((i, kind), torch.zeros(2, dtype=torch.float64, device=x.device))
        acc[0] += (out.double() - x.double()).pow(2).sum()
        acc[1] += x.double().pow(2).sum()
        return out

    def bytes_a_position(self) -> int:
        """Bytes the kept keys and values of one position take in all layers (a scale of 4 bytes a vector)."""
        total = 0
        for i, t in enumerate(self.cfg.layer_types):
            hd, kvh = self.cfg.head_dim[t], self.cfg.kv_heads[t]
            for kind in ("k", "v"):
                total += kvh * (hd * self.bits[i][kind] // 8 + 4)
        return total

    def noise_report(self) -> dict:
        """Relative noise energy (squared error over squared size) per kind: by layer, and the mean over layers by bits."""
        out = {}
        for kind in ("k", "v"):
            layers = {i: (e / s).item() for (i, k), (e, s) in sorted(self.noise.items()) if k == kind and s > 0}
            by_bits = {}
            for i, r in layers.items():
                by_bits.setdefault(self.bits[i][kind], []).append(r)
            out[kind] = {"by_layer": layers, "mean_by_bits": {b: sum(v) / len(v) for b, v in by_bits.items()}}
        return out
