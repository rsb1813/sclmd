# The Gemma 4 text decoder in plain PyTorch, one layer at a time: the reference every other path is compared with.
# Follows transformers' Gemma4TextModel (tests/test_model_small.py compares the two on a small random config).
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

SLIDING, FULL = "sliding_attention", "full_attention"
# A layer's weights by short name -> the name inside the checkpoint's layer.
LAYER_KEYS = {
    "in_ln": "input_layernorm.weight",
    "q": "self_attn.q_proj.weight",
    "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight",  # absent in full-attention layers: the value is the key projection
    "o": "self_attn.o_proj.weight",
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
    "post_attn_ln": "post_attention_layernorm.weight",
    "pre_ff_ln": "pre_feedforward_layernorm.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
    "post_ff_ln": "post_feedforward_layernorm.weight",
    "scalar": "layer_scalar",
}
LINEARS = ("q", "k", "v", "o", "gate", "up", "down")  # the matrices that are quantized


@dataclass
class Config:
    n_layers: int
    hidden: int
    mlp: int
    heads: int
    layer_types: list
    head_dim: dict  # layer type -> size of one head
    kv_heads: dict  # layer type -> number of key/value heads
    window: int
    vocab: int
    eps: float
    softcap: float | None
    rope: dict  # layer type -> (theta, number of rotated frequency pairs)
    kv_shared: int = 0  # the last `kv_shared` layers have no keys of their own (the drafter: all of them)

    @classmethod
    def from_text_config(cls, c: dict) -> "Config":
        k_eq_v = c.get("attention_k_eq_v", False)
        # config.json names the full-attention sizes directly; a config written back by transformers keeps them per layer
        over = next(iter(c.get("per_layer_config", {}).values()), None)
        if over is None:
            over = {"head_dim": c.get("global_head_dim", 512), "num_key_value_heads": c.get("num_global_key_value_heads")}
        head = {SLIDING: c["head_dim"], FULL: over.get("head_dim", c["head_dim"])}  # (written back without it: the same as the sliding one's)
        kv = {SLIDING: c["num_key_value_heads"], FULL: over.get("num_key_value_heads") or c["num_key_value_heads"]}
        if not k_eq_v:
            raise ValueError("only attention_k_eq_v checkpoints are supported")
        for unsupported in ("enable_moe_block", "hidden_size_per_layer_input", "use_double_wide_mlp"):
            if c.get(unsupported):
                raise ValueError(f"{unsupported} is not supported")
        rope = {}
        for t in (SLIDING, FULL):
            p = c["rope_parameters"][t]
            rotated = head[t] // 2 if p["rope_type"] == "default" else int(p.get("partial_rotary_factor", 1.0) * head[t] // 2)
            rope[t] = (p["rope_theta"], rotated)
        return cls(
            n_layers=c["num_hidden_layers"], hidden=c["hidden_size"], mlp=c["intermediate_size"], heads=c["num_attention_heads"],
            layer_types=list(c["layer_types"]), head_dim=head, kv_heads=kv, window=c["sliding_window"], vocab=c["vocab_size"],
            eps=c["rms_norm_eps"], softcap=c.get("final_logit_softcapping"), rope=rope, kv_shared=c.get("num_kv_shared_layers", 0),
        )

    @classmethod
    def load(cls, folder: str) -> "Config":
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
            c = json.load(f)
        return cls.from_text_config(c.get("text_config", c))

    def last_of(self, layer_type: str) -> int:
        """The last layer of a type with keys of its own: the layer whose keys and values the drafter attends to."""
        own = self.layer_types[: self.n_layers - self.kv_shared]
        return len(own) - 1 - own[::-1].index(layer_type)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    y = x.float()
    y = y * torch.pow(y.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        y = y * weight.float()
    return y.type_as(x)


def rope_tables(cfg: Config, layer_type: str, positions: torch.Tensor, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of shape [positions, head_dim]; pairs past the rotated ones get angle 0."""
    theta, rotated = cfg.rope[layer_type]
    dim = cfg.head_dim[layer_type]
    inv = torch.zeros(dim // 2, dtype=torch.float32, device=positions.device)
    inv[:rotated] = 1.0 / (theta ** (torch.arange(0, 2 * rotated, 2, device=positions.device).float() / dim))
    ang = positions.float()[:, None] * inv[None, :]
    ang = torch.cat((ang, ang), dim=-1)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [batch, positions, heads, head_dim]."""
    half = x.shape[-1] // 2
    turned = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos[None, :, None, :] + turned * sin[None, :, None, :]


def embed_scale(cfg: Config, dtype) -> torch.Tensor:
    return torch.tensor(cfg.hidden**0.5).to(dtype)  # rounded in the weights' type, as upstream does


class Cache:
    """Keys and values of the positions seen so far, per layer: [batch, kv_heads, positions, head_dim]."""

    def __init__(self):
        self.k, self.v = {}, {}
        self.length = 0

    def extend(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        if layer in self.k:
            k = torch.cat((self.k[layer], k), dim=2)
            v = torch.cat((self.v[layer], v), dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v

    def crop(self, length: int):
        for i in self.k:
            self.k[i], self.v[i] = self.k[i][:, :, :length], self.v[i][:, :, :length]
        self.length = length


def attend(cfg: Config, layer_type: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pos: torch.Tensor, kv_pos: torch.Tensor,
           window: int | None = None, rounded=None) -> torch.Tensor:
    """q: [batch, heads, n, head_dim], k and v: [batch, kv_heads, m, head_dim] -> [batch, n, heads * head_dim].
    A query sees keys at q_pos and before; in a sliding layer only the last `window` of them.
    rounded: applied to the attention weights (see layer_forward)."""
    window = cfg.window if window is None else window
    rep = cfg.heads // k.shape[1]
    b, _, n, d = q.shape
    qg = q.reshape(b, k.shape[1], rep, n, d)
    scores = torch.matmul(qg, k[:, :, None].transpose(-1, -2))  # no 1/sqrt(d): the queries and keys are normalized
    dist = q_pos[:, None] - kv_pos[None, :]
    seen = dist >= 0
    if layer_type == SLIDING:
        seen &= dist < window
    scores = scores.float().masked_fill(~seen, float("-inf"))
    w = F.softmax(scores, dim=-1).to(q.dtype)
    if rounded is not None:
        w = rounded(w)
    out = torch.matmul(w, v[:, :, None])  # [b, kv_heads, rep, n, d]
    return out.reshape(b, cfg.heads, n, d).transpose(1, 2).reshape(b, n, cfg.heads * d)


def layer_forward(cfg: Config, i: int, w: dict, h: torch.Tensor, positions: torch.Tensor, cache: Cache | None = None,
                  shared: dict | None = None, tap=None, linear=F.linear, see: tuple | None = None, rounded=None, kept=None,
                  made: bool = False) -> torch.Tensor:
    """One decoder layer over h [batch, n, hidden] at `positions` [n].
    see: (positions, window) the mask uses instead of the queries' own (the drafter: it sits one position past the
         last key it may read).
    cache: earlier positions' keys and values (decode); None means h holds the sequence from its start.
    shared: layer type -> (keys, values, their positions). Layers without keys of their own read it; the last layer of
            each type with keys writes it.
    tap(name, tensor): called with the input of each group of matrices ("attn_in", "o_in", "mlp_in", "down_in").
    linear(x, weight): the matrix product; quantized paths pass their own.
    kept(tensor): applied to the new keys and values instead of `rounded` (the engine keeps them as int8 with a scale).
    made: as an engine with keyless=True: of a layer whose value is its key's projection only the value is kept, and
        the key is made of the kept value.
    rounded(tensor): applied to what attention multiplies: the queries, keys, values and weights (the engine keeps
        them as fp16; a test of it passes that rounding)."""
    t = cfg.layer_types[i]
    b, n, _ = h.shape
    hd, kvh = cfg.head_dim[t], cfg.kv_heads[t]
    cos, sin = rope_tables(cfg, t, positions, h.dtype)

    x = rms_norm(h, w["in_ln"], cfg.eps)
    if tap:
        tap("attn_in", x)
    q = rotate(rms_norm(linear(x, w["q"]).view(b, n, cfg.heads, hd), w["q_norm"], cfg.eps), cos, sin).transpose(1, 2)
    if rounded is not None:
        q = rounded(q)
    if i >= cfg.n_layers - cfg.kv_shared:
        k, v, kv_pos = shared[t]
    else:
        k = linear(x, w["k"]).view(b, n, kvh, hd)
        v = linear(x, w["v"]).view(b, n, kvh, hd) if "v" in w else k
        if kept is not None and made and "v" not in w:
            # a keyless engine keeps such a layer's value only, and makes the key of what it kept: the value times k_norm, turned
            v = kept(rms_norm(v, None, cfg.eps))
            k = rotate((v.float() * w["k_norm"].float()).type_as(v), cos, sin).transpose(1, 2)
            v = v.transpose(1, 2)
        else:
            k = rotate(rms_norm(k, w["k_norm"], cfg.eps), cos, sin).transpose(1, 2)
            v = rms_norm(v, None, cfg.eps).transpose(1, 2)
            if kept is not None or rounded is not None:
                k, v = (kept or rounded)(k), (kept or rounded)(v)
        kv_pos = positions
        if cache is not None:
            k, v = cache.extend(i, k, v)
            kv_pos = torch.arange(k.shape[2], device=h.device)
        if shared is not None and i == cfg.last_of(t):
            shared[t] = (k, v, kv_pos)
    a = attend(cfg, t, q, k, v, positions, kv_pos, rounded=rounded) if see is None else attend(cfg, t, q, k, v, see[0], kv_pos, see[1], rounded)
    if tap:
        tap("o_in", a)
    h = h + rms_norm(linear(a, w["o"]), w["post_attn_ln"], cfg.eps)

    x = rms_norm(h, w["pre_ff_ln"], cfg.eps)
    if tap:
        tap("mlp_in", x)
    m = F.gelu(linear(x, w["gate"]), approximate="tanh") * linear(x, w["up"])
    if tap:
        tap("down_in", m)
    h = h + rms_norm(linear(m, w["down"]), w["post_ff_ln"], cfg.eps)
    return h * w["scalar"].to(h.dtype)


def logits_of(cfg: Config, h: torch.Tensor, norm_weight: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
    """Final norm, the tied output head and the soft cap."""
    out = F.linear(rms_norm(h, norm_weight, cfg.eps), embed)
    if cfg.softcap is not None:
        out = torch.tanh(out / cfg.softcap) * cfg.softcap
    return out


def layer_weights(state: dict, prefix: str, i: int) -> dict:
    """A layer's weights by short name out of a state dict whose layers sit under `prefix`."""
    return {short: state[f"{prefix}layers.{i}.{name}"] for short, name in LAYER_KEYS.items() if f"{prefix}layers.{i}.{name}" in state}


def forward(cfg: Config, state: dict, prefix: str, ids: torch.Tensor, cache: Cache | None = None, shared: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The whole model held in memory (small models and tests): ids [batch, n] -> (logits, last layer's output after the final norm)."""
    embed = state[f"{prefix}embed_tokens.weight"]
    start = cache.length if cache is not None else 0
    positions = torch.arange(start, start + ids.shape[1], device=ids.device)
    h = embed[ids] * embed_scale(cfg, embed.dtype)
    for i in range(cfg.n_layers):
        h = layer_forward(cfg, i, layer_weights(state, prefix, i), h, positions, cache, shared)
    if cache is not None:
        cache.length = start + ids.shape[1]
    norm = state[f"{prefix}norm.weight"]
    return logits_of(cfg, h, norm, embed), rms_norm(h, norm, cfg.eps)
