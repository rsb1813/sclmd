# The K3 simulation (sclmd/kvsim.py and Stream.run's `kept` hook), on the CPU: the levels, the rotation, which layers
# a mode rounds, and that the hook sees a layer's keys and then its values.
import torch

from sclmd import kvsim as KV
from sclmd import model as M
from sclmd.stream import Stream, _kinds

LM16 = [0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0691, 2.7326]  # Max (1960), the positive half
LM8 = [0.2451, 0.7560, 1.3439, 2.1520]


def tiny_cfg(n_layers=4, hd=16):
    types = [M.SLIDING, M.SLIDING, M.FULL, M.SLIDING][:n_layers]
    return M.Config(n_layers=n_layers, hidden=32, mlp=48, heads=4, layer_types=types, head_dim={M.SLIDING: hd, M.FULL: 2 * hd},
                    kv_heads={M.SLIDING: 2, M.FULL: 1}, window=5, vocab=50, eps=1e-6, softcap=30.0,
                    rope={M.SLIDING: (10000.0, hd // 2), M.FULL: (1e6, hd // 4)})


def tiny_weights(cfg, i, g):
    t = cfg.layer_types[i]
    hd, kvh, d = cfg.head_dim[t], cfg.kv_heads[t], cfg.hidden
    w = {"in_ln": 0.5 + torch.rand(d, generator=g), "q": torch.randn(cfg.heads * hd, d, generator=g) * 0.2,
         "k": torch.randn(kvh * hd, d, generator=g) * 0.2, "o": torch.randn(d, cfg.heads * hd, generator=g) * 0.2,
         "q_norm": 0.5 + torch.rand(hd, generator=g), "k_norm": 0.5 + torch.rand(hd, generator=g),
         "post_attn_ln": 0.5 + torch.rand(d, generator=g), "pre_ff_ln": 0.5 + torch.rand(d, generator=g),
         "gate": torch.randn(cfg.mlp, d, generator=g) * 0.2, "up": torch.randn(cfg.mlp, d, generator=g) * 0.2,
         "down": torch.randn(d, cfg.mlp, generator=g) * 0.2, "post_ff_ln": 0.5 + torch.rand(d, generator=g),
         "scalar": torch.tensor(0.8)}
    if t == M.SLIDING:
        w["v"] = torch.randn(kvh * hd, d, generator=g) * 0.2
    return w


class FakeCkpt:
    """What Stream reads of a checkpoint, for a tiny random model on the CPU."""

    def __init__(self, cfg, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.cfg = cfg
        self.layers = [tiny_weights(cfg, i, g) for i in range(cfg.n_layers)]
        self.emb = torch.randn(cfg.vocab, cfg.hidden, generator=g)

    def embed(self, dev):
        return self.emb.to(dev)

    def layer(self, i, dev):
        return {k: v.to(dev) for k, v in self.layers[i].items()}


def test_lloyd_max_levels_match_the_tables():
    c16, mse16 = KV.lloyd_max(16)
    c8, mse8 = KV.lloyd_max(8)
    assert max(abs(a - b) for a, b in zip(c16[8:], LM16)) < 2e-4 and abs(mse16 - 0.009497) < 2e-5
    assert max(abs(a - b) for a, b in zip(c8[4:], LM8)) < 2e-4 and abs(mse8 - 0.03454) < 2e-5
    assert all(abs(a + b) < 1e-9 for a, b in zip(c16, reversed(c16)))  # symmetric


def test_grid_levels_are_the_briefs_int8_codes():
    lv, top = KV.grid_levels(4)
    assert lv.tolist() == [-127, -96, -75, -58, -44, -31, -18, -6, 6, 18, 31, 44, 58, 75, 96, 127]
    assert abs(top - 2.7326) < 2e-4
    lv3, _ = KV.grid_levels(3)
    assert len(lv3) == 8 and lv3[-1] == 127


def test_hadamard_is_orthonormal_and_symmetric():
    h = KV.hadamard(256).double()
    assert torch.allclose(h @ h.T, torch.eye(256, dtype=torch.float64), atol=1e-6)
    assert torch.equal(h, h.T)


def test_int8_round_is_the_engines():
    x = torch.randn(3, 2, 5, 16)
    x[0, 0, 0] = 0  # a zero vector keeps scale 1
    got = KV.int8_round(x)
    most = x.abs().amax(-1, keepdim=True)
    scale = torch.where(most > 0, most / 127, torch.ones_like(most))
    codes = torch.round(x / scale)
    assert codes.abs().max() <= 127 and torch.equal(got, codes * scale)
    assert torch.equal(got[0, 0, 0], torch.zeros(16))


def test_low_bits_rounds_in_the_rotated_basis_and_keeps_dot_products_exact():
    cfg = tiny_cfg(hd=16)
    kept = KV.Kept(cfg, "k4v4")
    x = torch.randn(2, 2, 7, 16)
    out = kept(0, "k", x)
    s, h = kept.signs[0][None, :, None, :], kept.h
    y = (out * s) @ h  # the rotated kept vector: levels times one scale per vector
    rms = ((x * s) @ h).pow(2).mean(-1, keepdim=True).sqrt()
    units = y / (rms * kept.levels[4][1] / 127)
    assert torch.allclose(units, units.round(), atol=1e-3)
    assert set(units.round().abs().unique().tolist()) <= {6.0, 18.0, 31.0, 44.0, 58.0, 75.0, 96.0, 127.0}
    # a query scored against the rotated kept key gives the same as against the un-rotated one (what a kernel uses)
    q = torch.randn(2, 2, 7, 16)
    assert torch.allclose((q * out).sum(-1), (((q * s) @ h) * y).sum(-1), atol=1e-4)
    # the noise is near the Gaussian Lloyd-Max error (0.0095 of the energy) for Gaussian vectors
    big = torch.randn(1, 2, 4000, 16)
    err = (kept(0, "v", big) - big).pow(2).sum() / big.pow(2).sum()
    assert 0.006 < err.item() < 0.02


def test_which_layers_a_mode_rounds():
    cfg = tiny_cfg(n_layers=4)
    last = cfg.last_of(M.SLIDING)
    assert last == 3
    for mode, k, v in (("int8", 8, 8), ("k8v4", 8, 4), ("k4v4", 4, 4), ("k3v3", 3, 3), ("k4v4t0", 4, 4), ("k8v4t0", 8, 4)):
        kept = KV.Kept(cfg, mode)
        assert kept.bits[0] == {"k": k, "v": v} and kept.bits[1] == {"k": k, "v": v}
        assert kept.bits[2] == {"k": 8, "v": 8}  # a full layer
        assert kept.bits[last] == {"k": 8, "v": 8}  # the drafter's layer


def test_first_tile_modes_keep_positions_0_to_15_at_int8():
    cfg = tiny_cfg()
    x = torch.randn(2, 2, 40, 16)
    out = KV.Kept(cfg, "k4v4t0")(0, "k", x)
    assert torch.equal(out[:, :, :16], KV.int8_round(x[:, :, :16]))
    assert torch.allclose(out[:, :, 16:], KV.Kept(cfg, "k4v4")(0, "k", x)[:, :, 16:])


def test_bytes_a_position_of_the_real_model():
    layer_types = ([M.SLIDING] * 5 + [M.FULL]) * 10
    cfg = M.Config(n_layers=60, hidden=5376, mlp=21504, heads=32, layer_types=layer_types, head_dim={M.SLIDING: 256, M.FULL: 512},
                   kv_heads={M.SLIDING: 16, M.FULL: 4}, window=1024, vocab=262144, eps=1e-6, softcap=30.0,
                   rope={M.SLIDING: (10000.0, 128), M.FULL: (1e6, 64)})
    assert cfg.last_of(M.SLIDING) == 58
    full = 10 * 4 * (512 + 4) * 2  # the full layers, keys stored: 41,280
    assert KV.Kept(cfg, "int8").bytes_a_position() == 416_000 + full
    assert KV.Kept(cfg, "k4v4").bytes_a_position() == 215_296 + full
    assert KV.Kept(cfg, "k8v4").bytes_a_position() == 49 * 16 * (256 + 4 + 128 + 4) + 8_320 + full


def test_layer_forward_hands_the_hook_keys_then_values():
    cfg = tiny_cfg()
    g = torch.Generator().manual_seed(1)
    w = tiny_weights(cfg, 0, g)
    h = torch.randn(2, 6, cfg.hidden, generator=g)
    pos = torch.arange(6)
    seen = []
    M.layer_forward(cfg, 0, w, h, pos, kept=_kinds(lambda i, kind, x: seen.append((i, kind, x.clone())) or x, 0))
    assert [(i, k) for i, k, _ in seen] == [(0, "k"), (0, "v")]
    hd, kvh = cfg.head_dim[M.SLIDING], cfg.kv_heads[M.SLIDING]
    x = M.rms_norm(h, w["in_ln"], cfg.eps)
    cos, sin = M.rope_tables(cfg, M.SLIDING, pos, h.dtype)
    key = M.rotate(M.rms_norm((x @ w["k"].T).view(2, 6, kvh, hd), w["k_norm"], cfg.eps), cos, sin).transpose(1, 2)
    value = M.rms_norm((x @ w["v"].T).view(2, 6, kvh, hd), None, cfg.eps).transpose(1, 2)
    assert torch.allclose(seen[0][2], key) and torch.allclose(seen[1][2], value)


def test_stream_without_the_hook_is_unchanged_and_the_hook_sees_every_layer():
    cfg = tiny_cfg()
    ck = FakeCkpt(cfg)
    ids = torch.randint(0, cfg.vocab, (3, 9), generator=torch.Generator().manual_seed(2))
    base = Stream(ck, device="cpu", batch=2).run(ids)["hidden"]
    calls = []
    same = Stream(ck, device="cpu", batch=2).run(ids, kept=lambda i, kind, x: calls.append((i, kind)) or x)["hidden"]
    assert torch.equal(base, same)  # a hook that keeps everything as it is changes no bit
    assert calls == [(i, kind) for i in range(cfg.n_layers) for _ in range(2) for kind in ("k", "v")]  # 2 batches of rows a layer
    rounded = Stream(ck, device="cpu", batch=2).run(ids, kept=KV.Kept(cfg, "k4v4"))["hidden"]
    assert not torch.equal(base, rounded)
