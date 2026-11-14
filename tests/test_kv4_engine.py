# The engine with kv4 (brief K3: the sliding layers' keys and values at 4 bits, but the drafter's layer) on small
# random Gemma 4 models: the format against the simulation's (sclmd/kvsim.py), the kernels against PyTorch on the
# engine's own buffers, and the bit tests of tests/test_engine.py with kv4 on (a row's bits do not depend on its pass,
# its neighbours, its tiles or a trip to RAM). Two models: the stock small one (heads of 32: the narrow kernels) and one
# with heads of 256 and 16 queries a key head (the real model's head; the decode-once scores and the wide sums).
import copy

import pytest
import torch

import test_engine as TE
from sclmd import engine as E
from sclmd import kvsim as KV
from sclmd import model as M
from conftest import randomize
from test_engine import BOOST, ETEXT, WHOLE_KEPT, Small, Stored, close, fp16, ids_of, int8

pytestmark = TE.pytestmark
DEV = TE.DEV

# the bit tests of test_engine, run here with the kv4 engines of the fixture below
from test_engine import (test_a_parked_sequence_comes_back_on_another_number_and_other_tiles,  # noqa: E402, F401
                         test_a_position_is_the_same_bits_alone_or_among_eight, test_a_position_is_the_same_bits_in_passes_of_any_width,
                         test_a_sequence_does_not_depend_on_what_the_others_hold, test_cropped_positions_are_forgotten,
                         test_going_back_further_than_the_ring_keeps_is_refused, test_restore_comes_back_from_further_than_the_ring_keeps,
                         test_sequences_side_by_side_are_the_same_bits_as_each_alone, test_snapshot_and_restore_of_one_sequence_among_others,
                         test_text_is_the_same_with_any_mix_of_right_and_wrong_guesses, test_the_drafter_guesses_for_several_sequences_at_once,
                         test_wide_passes_over_several_sequences_and_the_drafter, test_with_the_drafter_the_text_is_unchanged)

WIDE = dict(ETEXT, head_dim=256, global_head_dim=512, num_attention_heads=16, num_key_value_heads=1, num_global_key_value_heads=1)


class Reference4(TE.Reference):
    """The reference decoder keeping keys and values as an engine with kv4 does: sclmd/kvsim.py's k4v4t0 (by position:
    the first tile at int8), int8 in the other layers."""

    def feed(self, ids):
        cfg, start = self.cfg, self.cache.length
        fmt = KV.Kept(cfg, "k4v4t0")
        pos = torch.arange(start, start + len(ids), device=DEV)
        h = self.embed(ids)[None]
        for i, w in enumerate(self.layers):
            kinds = iter(("k", "v"))
            kept = (lambda x, i=i, kinds=kinds: fmt.at(i, next(kinds), x.float(), start)) if i in E.kv4_layers(cfg) else int8
            h = M.layer_forward(cfg, i, w, h, pos, self.cache, self.shared, linear=lambda x, m: m(x), rounded=fp16, kept=kept, made=self.made)
        self.cache.length = start + len(ids)
        hn = M.rms_norm(h[0], self.norm, cfg.eps)
        logits = self.head(hn)
        return (torch.tanh(logits / cfg.softcap) * cfg.softcap if cfg.softcap else logits), hn


class Small4(Small):
    def __init__(self, text=ETEXT):
        saved = TE.ETEXT
        TE.ETEXT = text  # Small builds the model of TE.ETEXT
        try:
            super().__init__()
        finally:
            TE.ETEXT = saved
        self.text = text

    def engine(self, capacity: int = 96, **more) -> E.Engine:
        return E.Engine(self.cfg, self.eng_layers, self.head.stored(), self.norm, self.embed, capacity=capacity, keyless=self.keyless, kv4=True, **more)

    def reference(self):
        return Reference4(self.cfg, self.ref_layers, self.head, self.norm, self.embed, self.keyless)


_MODELS = {}


def model(name: str) -> Small4:
    if name not in _MODELS:
        _MODELS[name] = Small4({"small": ETEXT, "wide": WIDE, "w16": dict(ETEXT, sliding_window=16)}[name])
    return _MODELS[name]


# every test runs on both models, keys kept in the full layers and keyless
@pytest.fixture(scope="module", params=[("small", False), ("small", True), ("wide", False), ("wide", True)],
                ids=["small-keys", "small-keyless", "wide-keys", "wide-keyless"])
def small_e(request):
    one = copy.copy(model(request.param[0]))
    one.keyless = request.param[1]
    return one


_DRAFTERS = {}


@pytest.fixture(scope="module")
def drafter_e(small_e):
    """test_engine's drafter_e for the model of small_e (a drafter's heads are its model's)."""
    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4TextConfig

    base = small_e.text
    key = base["head_dim"]
    if key not in _DRAFTERS:
        text = dict(base, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_kv_shared_layers=4, final_logit_softcapping=None,
                    layer_types=["sliding_attention"] * 3 + ["full_attention"], vocab_size_per_layer_input=0)
        config = Gemma4AssistantConfig(text_config=Gemma4TextConfig(**text), backbone_hidden_size=base["hidden_size"])
        hf = randomize(Gemma4AssistantForCausalLM(config).eval(), 6)
        hf.lm_head.weight = hf.model.embed_tokens.weight
        state = {k: v.to(DEV) for k, v in hf.state_dict().items()}
        cfg = M.Config.from_text_config(config.text_config.to_dict())
        layers = []
        for i in range(cfg.n_layers):
            w = M.layer_weights(state, "model.", i)
            layers.append({k: (Stored(v, 6 if (i + j) % 2 else 8) if k in M.LINEARS else v) for j, (k, v) in enumerate(w.items())})
        _DRAFTERS[key] = (cfg, layers, Stored(state["pre_projection.weight"], 8), Stored(state["post_projection.weight"], 6), state["model.norm.weight"],
                          Stored(state["model.embed_tokens.weight"], 8))
    return _DRAFTERS[key]


def first_only(small_e, i: int) -> E.Engine:
    """An engine of layer i (a sliding layer) and a second sliding layer, which runs its first only: layer i is then a
    4-bit layer (the last sliding layer is the drafter's, int8), and the work memory holds its attention after a pass."""
    base = small_e.cfg
    cfg = M.Config(**{**vars(base), "n_layers": 2, "layer_types": [M.SLIDING, M.SLIDING]})
    eng = E.Engine(cfg, [small_e.eng_layers[i]] * 2, small_e.head.stored(), small_e.norm, small_e.embed, capacity=96, keyless=small_e.keyless,
                   kv4=True)
    assert eng.kv4_set == {0}
    eng.model.n_layers = 1
    return eng


def kvsim_codes(fmt: KV.Kept, i: int, x: torch.Tensor) -> tuple:
    """What kvsim's 4-bit rounding makes of x [positions, heads, head_dim]: (codes 0..15, scales [positions, heads], the
    distance of y / scale to the nearest midpoint)."""
    signs, h, levels = fmt._on(x.device)
    lv, cut, top = levels[4]
    y = (x.float() * signs[i][None]) @ h
    scale = y.pow(2).mean(-1, keepdim=True).sqrt() * (top / 127)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    u = y / scale
    return torch.bucketize(u, cut), scale[..., 0], (u[..., None] - cut).abs().amin(-1)


def test_the_format_is_the_simulations_and_its_constants_are_the_kernels():
    cfg = model("small").cfg
    fmt = KV.Kept(cfg, "k4v4t0")
    assert E.kv4_layers(cfg) == [i for i in range(cfg.n_layers) if fmt.bits[i]["k"] == 4] == [0, 1, 3]  # not layer 4, the drafter's
    eng = model("small").engine()  # its constructor checks the DLL's levels, midpoints and unit against kvsim
    assert eng.kv4 and eng.extra == {M.SLIDING: 1, M.FULL: 0}
    for i in eng.kv4_set:
        assert torch.equal(eng.kv4_signs[i].cpu(), fmt.signs[i].float())
    # bytes: 4-bit layers hold half a byte a value and a scale a vector; the drafter's and the full layers as ever
    hd, kvh = cfg.head_dim[M.SLIDING], cfg.kv_heads[M.SLIDING]
    assert E.tile_bytes(cfg, M.SLIDING, kv4=True) == 3 * 16 * kvh * (hd // 2 + 4) * 2 + 16 * kvh * (hd + 4) * 2
    assert E.tile_bytes(cfg, M.FULL, kv4=True) == E.tile_bytes(cfg, M.FULL)
    real = M.Config(n_layers=60, hidden=5376, mlp=21504, heads=32, layer_types=([M.SLIDING] * 5 + [M.FULL]) * 10,
                    head_dim={M.SLIDING: 256, M.FULL: 512}, kv_heads={M.SLIDING: 16, M.FULL: 4}, window=1024, vocab=262144, eps=1e-6,
                    softcap=30.0, rope={M.SLIDING: (10000.0, 128), M.FULL: (1e6, 64)})
    assert E.tile_bytes(real, M.SLIDING) == 6_656_000 and E.tile_bytes(real, M.SLIDING, kv4=True) == 16 * 215_296  # kvsim's bytes a position
    assert E.extra_tiles(real, M.SLIDING, True) == 1 and E.extra_tiles(real, M.FULL, True) == 0 and E.extra_tiles(real, M.SLIDING, False) == 0


def test_kept_keys_and_values_are_kvsims_rounding_of_the_layers_own(small_e):
    # one 4-bit layer, its positions written in passes of several widths, past the first tile and round the ring (32)
    for i in (0, 3):
        eng = first_only(small_e, i)
        cfg, fmt = eng.cfg, KV.Kept(eng.cfg, "k4v4t0")
        g = torch.Generator().manual_seed(10 + i)
        pos, near, checked = 0, 0, 0
        for n in (1, 1, 8, 3, 8, 8, 8, 5):  # 42 positions
            h = (torch.randn(n, cfg.hidden, generator=g) * 3).to(DEV)
            eng.h[:n] = h
            eng.advance([(0, n)])
            positions = torch.arange(pos, pos + n, device=DEV)
            seen = []  # the layer's keys and values before they are kept, as the reference computes them
            M.layer_forward(cfg, 0, small_e.ref_layers[i], h[None], positions, linear=lambda x, m: m(x), rounded=fp16,
                            kept=lambda x: seen.append(x[0].transpose(0, 1).float()) or x)
            got_k, got_v = eng.kv4_float(0, positions)
            codes = eng.kv4_codes(0, positions)
            for kind, x, got, code, (_, scales) in zip("kv", seen, (got_k, got_v), codes, ((0, eng.kv[0][2]), (0, eng.kv[0][3]))):
                want = fmt.at(0, kind, x.transpose(0, 1)[None], pos)[0].transpose(0, 1)  # kvsim: [positions, heads, head_dim]
                first = positions < 16
                if first.any():  # the first tile: int8 of the vector itself
                    assert close(got[first], int8(x[first]), TE.KEPT) and close(got[first], want[first], TE.KEPT), (i, pos, kind)
                rest = ~first
                if rest.any():
                    c, s, gap = kvsim_codes(fmt, 0, x[rest])
                    slot = eng.slots(M.SLIDING, positions[rest])
                    assert torch.allclose(scales[slot].float(), s, rtol=1e-5), (i, pos, kind)
                    differ = code[rest] != c
                    near += int(differ.sum())
                    checked += differ.numel()
                    assert (gap[differ] < 1e-3).all(), (i, pos, kind)  # a code differs only where the value sits on a midpoint
                    same = ~differ.any(-1)  # the vectors whose codes are all kvsim's: the same numbers back
                    assert close(got[rest][same], want[rest][same], 1e-4), (i, pos, kind)
            pos += n
        assert near <= checked // 1000, (near, checked)


def test_attention_kernels_against_pytorch_on_the_engines_own_buffers(small_e):
    # the 4-bit layer's queries (turned and as they are), scores, weights and output after each pass, from its own codes
    base = small_e.cfg
    g = torch.Generator().manual_seed(4)
    eng = first_only(small_e, 0)
    cfg = eng.cfg
    hd, kvh, heads = cfg.head_dim[M.SLIDING], cfg.kv_heads[M.SLIDING], cfg.heads
    rep, pos = heads // kvh, 0
    signs, h = eng.kv4_signs[0].double(), eng.kv4_h.double()
    for n in (1, 1, 8, 3, 8, 8, 8, 5, 1, 2):
        eng.h[:n] = (torch.randn(n, base.hidden, generator=g) * 3).to(DEV)
        eng.advance([(0, n)])
        lo = max(pos - cfg.window + 1, 0)
        first_tile = lo & ~15
        stride, nq = ((pos + n - 1 >> 4) + 1) * 16 - first_tile, rep * n
        q = eng.work["qhat"].view(-1)[: n * heads * hd].view(n, heads, hd).double()  # turned
        plain = eng.work["qhat"].view(-1)[n * heads * hd : 2 * n * heads * hd].view(n, heads, hd).double()
        scores = eng.work["scores"][: kvh * nq * stride].view(kvh, nq, stride)
        probs = eng.work["probs"][: kvh * nq * stride].view(kvh, nq, stride)
        attn = eng.work["attn"].view(-1)[: n * heads * hd].view(n, heads, hd).double()
        positions = torch.arange(first_tile, first_tile + stride, device=DEV)
        kc, vc = eng.kv4_codes(0, positions)
        slot = eng.slots(M.SLIDING, positions)
        ks, vs = eng.kv[0][2][slot].double(), eng.kv[0][3][slot].double()
        levels = eng.kv4_levels.double()
        keys4, values4 = levels[kc] * ks[..., None], levels[vc]  # the turned keys; the values' levels (their scales are in the weights)
        k_first, v_first = (x.double() for x in eng.kv4_float(0, positions))
        first = positions < 16
        v_first = torch.where(first[:, None, None], v_first / scale_first(eng, positions).clamp(min=1e-30)[..., None], 0)  # the first tile's int8
        for kh in range(kvh):
            for query in range(nq):
                token, head = query // rep, kh * rep + query % rep
                a = max(pos + token - cfg.window + 1, 0) - first_tile
                b = pos + token - first_tile + 1
                where = (pos, n, kh, query)
                want = torch.where(first, k_first[:, kh] @ plain[token, head], keys4[:, kh] @ q[token, head])
                assert close(scores[kh, query, a:b].double(), want[a:b], 1e-5), where
                sc = torch.where(first, scale_first(eng, positions)[:, kh], vs[:, kh])  # the values' scales, which the weights carry
                assert close(probs[kh, query, a:b].float(), torch.softmax(scores[kh, query, a:b], -1) * sc[a:b].float() * BOOST, 1e-3), where
                assert not probs[kh, query, :a].any() and not probs[kh, query, b:].any(), where
                p = probs[kh, query].double()
                turned = torch.where(first[:, None], 0, values4[:, kh]).T @ p  # the 4-bit tiles' sum, turned
                want = ((turned @ h) * signs[kh] + v_first[:, kh].T @ p) / BOOST
                assert close(attn[token, head], want, 1e-5), where
        pos += n


def scale_first(eng: E.Engine, positions: torch.Tensor) -> torch.Tensor:
    """The values' scales where the first tile keeps them ([positions, heads]; 0 past it)."""
    own = eng.tiles[M.SLIDING][0]
    p = positions.long()
    tile = torch.where(p < 8, own[0], own[eng.ring[M.SLIDING]])
    return torch.where(p[:, None] < 16, eng.kv[0][3][(tile * 16 + (p & 7)).clamp(max=eng.kv[0][3].shape[0] - 1)].double(), 0)


def test_passes_match_the_kv4_reference_closer_than_the_int8_one(small_e):
    eng, ref, ref8 = small_e.engine(), small_e.reference(), Small.reference(small_e)
    ids = ids_of(60, 1)  # round a sliding layer's ring almost twice
    want_logits, want_hidden = ref.feed(ids)
    _, int8_hidden = ref8.feed(ids)
    got = torch.cat([eng.feed(part)[1] for part in (ids[:13], ids[13:14], ids[14:15], ids[15:18], ids[18:26], ids[26:27], ids[27:60])])
    err = lambda a, b: float(((a - b).abs().amax(-1) / b.abs().amax(-1)).mean())  # noqa: E731
    # the engine is nearer the kv4 reference than the two references are to each other (4-bit roundings that a last-bit
    # difference tips the other way are a tenth of a vector's size, not int8's hundredth, and a random model's sharp
    # softmax spreads them: the engine is not as near its reference as with int8)
    e4, e8, e48 = err(got, want_hidden), err(got, int8_hidden), err(want_hidden, int8_hidden)
    print(f'[kv4] mean difference: to the kv4 reference {e4:.4f}, to the int8 one {e8:.4f}; the two references {e48:.4f}')
    assert e4 < e48 / 2 and e4 < e8 / 2, (e4, e8, e48)


def test_the_first_tile_takes_a_second_tile_that_is_given_back_with_it(small_e):
    cfg = small_e.cfg
    ring = {t: E.ring_tiles(cfg, t, 96) for t in (M.SLIDING, M.FULL)}
    eng = small_e.engine(sequences=3, pool={M.SLIDING: ring[M.SLIDING] + 3, M.FULL: ring[M.FULL] + 1})
    assert eng.lacks(0, 0, 5) == {M.SLIDING: 2, M.FULL: 1} and eng.lacks(0, 16, 5) == {M.SLIDING: 1, M.FULL: 1}
    eng.feed(ids_of(10, 29), seq=1)  # one ring tile and the first tile's second
    assert eng.free_tiles()[M.SLIDING] == ring[M.SLIDING] + 1 and int(eng.firsts[1]) == eng.tiles[M.SLIDING][1][ring[M.SLIDING]] >= 0
    a = ids_of(70, 27)
    got = eng.feed(a[:30], seq=0)[0]
    assert eng.free_tiles()[M.SLIDING] == 0
    with pytest.raises(E.KVFull):
        eng.feed(ids_of(5, 28), seq=2)
    eng.release(1)
    assert eng.free_tiles()[M.SLIDING] == 2
    got = torch.cat([got, eng.feed(a[30:], seq=0)[0]])  # round the ring: the ring tile 0 goes 4-bit, the second tile stays
    assert torch.equal(got, TE.alone(small_e, a, []))
    assert sum(x >= 0 for x in eng.tiles[M.SLIDING][0]) == ring[M.SLIDING] + 1
    eng.crop(3, seq=0)
    eng.trim(0)  # the ring has gone round: the sliding tiles stay
    assert sum(x >= 0 for x in eng.tiles[M.SLIDING][0]) == ring[M.SLIDING] + 1
    eng.release(0)
    assert eng.free_tiles() == {M.SLIDING: ring[M.SLIDING] + 3, M.FULL: ring[M.FULL] + 1}
    b = ids_of(20, 30)
    want = eng.feed(b, seq=2)[0]
    eng.crop(3, seq=2)
    eng.trim(2)  # 3 positions: the ring tile 0 and the second tile stay
    assert eng.free_tiles()[M.SLIDING] == ring[M.SLIDING] + 1 and torch.equal(eng.feed(b[3:], seq=2)[0], want[3:])
    eng.crop(0, seq=2)
    eng.trim(2)
    assert eng.free_tiles()[M.SLIDING] == ring[M.SLIDING] + 3


def test_a_saved_or_forked_prefix_goes_on_as_if_read_and_its_bytes_are_the_formats(small_e):
    a, more = ids_of(25, 40), ids_of(12, 41)
    want = TE.alone(small_e, a + more, [])[25:]
    eng = small_e.engine(sequences=3)
    eng.feed(a, seq=0)
    state = eng.save(0)
    eng.fork(0, 1)
    assert torch.equal(eng.feed(more, seq=1)[0], want)
    eng.unpark(2, state)
    assert torch.equal(eng.feed(more, seq=2)[0], want) and torch.equal(eng.feed(more, seq=0)[0], want)
    plain = E.Engine(small_e.cfg, small_e.eng_layers, small_e.head.stored(), small_e.norm, small_e.embed, capacity=96, keyless=small_e.keyless)
    plain.feed(a)
    cfg = small_e.cfg
    held = {t: len(part["js"]) for t, part in state["kinds"].items()}
    assert state["bytes"] == sum(E.tile_bytes(cfg, t, small_e.keyless, True) * n for t, n in held.items())
    assert held[M.SLIDING] == plain.save(0)["kinds"][M.SLIDING]["js"].__len__() + 1  # the first tile's second
    # what a parked or saved state holds: the extra tile's entry comes back as the first tile's second
    assert state["kinds"][M.SLIDING]["js"][-1] == eng.ring[M.SLIDING]
    assert int(eng.firsts[2]) == eng.tiles[M.SLIDING][2][eng.ring[M.SLIDING]] != eng.tiles[M.SLIDING][0][eng.ring[M.SLIDING]]


def test_the_first_tile_outlives_the_ring_going_round_within_a_pass_and_a_crop_back():
    # a window of 16 and passes of 16 rows: the ring is two tiles (32 positions), as tight as the real model's (window +
    # rows). A pass over positions 22..37 writes 32..37 over the ring tile 0, whose bytes hold the first tile's int8
    # positions 0..7, while its first row still reads positions 7..22; and after 34 positions a turn may be cropped back
    # to 22, whose row reads 7.. again. Both as if every position had been fed by itself.
    m = model("w16")
    ids = ids_of(40, 50)
    one = m.engine(rows=16)
    want = torch.cat([one.feed([t])[0] for t in ids])
    eng = m.engine(rows=16)
    assert eng.ring[M.SLIDING] == 2
    got = torch.cat([eng.feed(ids[:22])[0], eng.feed(ids[22:38])[0], eng.feed(ids[38:])[0]])
    assert torch.equal(got, want)
    eng = m.engine(rows=16)
    eng.feed(ids[:34])
    eng.crop(22)
    assert torch.equal(eng.feed(ids[22:26])[0], want[22:26])


def test_off_is_the_engine_as_before(small_e):
    plain = E.Engine(small_e.cfg, small_e.eng_layers, small_e.head.stored(), small_e.norm, small_e.embed, capacity=96, keyless=small_e.keyless,
                     kv4=False)
    assert not plain.kv4 and plain.firsts is None and plain.extra == {M.SLIDING: 0, M.FULL: 0}
    assert all(len(group) in (2, 4) for group in plain.kv_all.values())
    assert all(cl.kv4 == 0 for cl in plain.c_layers)
