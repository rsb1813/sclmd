# The engine (engine/sclmd.cu through sclmd/engine.py) against the reference decoder on a small random Gemma 4.
# The reference gets the same stored matrices and the engine's two roundings (a matrix's input and the kept keys and
# values, attention's queries and weights are fp16), so what is left between the two is the order of float32 sums.
import os

import pytest
import torch

from conftest import TEXT, randomize
from sclmd import codes as C
from sclmd import engine as E
from sclmd import model as M
from sclmd import quant as Q
from sclmd.drafter import Drafter
from sclmd.generate import Picker, drafter_guess, generate
from test_generate import PROMPT, oracle

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.exists(E.DLL),
                                reason="needs a GPU and engine\\build\\sclmd.dll (engine\\build.bat)")
DEV = "cuda"
# what the kernels ask of the sizes: a matrix's rows a multiple of 16, its columns of 64, a head of 32
ETEXT = dict(TEXT, vocab_size=320, hidden_size=192, intermediate_size=384, head_dim=32, global_head_dim=64)


class Stored:
    """A matrix as the engine holds it (pair codes in the rotated basis, nearest codeword: no fitting here), and its
    product with an input computed in PyTorch the way the kernels compute it."""

    def __init__(self, w: torch.Tensor, bits: int):
        rows, cols = w.shape
        self.rot = C.rotation(cols, DEV)
        wr = self.rot.rotate(w.to(DEV))
        wn, s = Q.row_normalize(wr)
        self.book = C.pair_codebook(wr.cpu(), bits).to(DEV)  # on the CPU: the GPU's sums come out in another order every run
        self.codes = Q.vq_assign(wn.reshape(-1, 2), self.book).view(rows, cols // 2).to(torch.uint8)
        self.scales, self.bits = s[:, 0].contiguous(), bits
        self.unit = self.book.half().float()[self.codes.long()].flatten(-2)  # [rows, cols]: the codebook is kept as fp16

    def stored(self) -> tuple:
        return self.codes, self.scales, self.book, self.bits

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1]).float()
        length = flat.norm(dim=-1, keepdim=True)
        fps = torch.where(length > 16384, torch.exp2(torch.ceil(torch.log2(length / 16384))), torch.ones_like(length))
        x16 = (self.rot.rotate(flat) / fps).half().float()
        return ((x16 @ self.unit.t()) * (self.scales * fps)).reshape(*x.shape[:-1], -1)


def fp16(t: torch.Tensor) -> torch.Tensor:
    """What the engine keeps as fp16: the queries, keys and values of attention, and its weights."""
    return t.half().float()


class Reference:
    """The reference decoder over stored matrices. The same four things generation asks of the engine."""

    def __init__(self, cfg, layers, head, norm, embed):
        self.cfg, self.layers, self.head, self.norm, self.table = cfg, layers, head, norm, embed
        self.cache, self.shared = M.Cache(), {}

    def embed(self, ids):
        return self.table[torch.tensor(ids, device=DEV)] * M.embed_scale(self.cfg, self.table.dtype)

    def feed(self, ids):
        cfg, start = self.cfg, self.cache.length
        pos = torch.arange(start, start + len(ids), device=DEV)
        h = self.embed(ids)[None]
        for i, w in enumerate(self.layers):
            h = M.layer_forward(cfg, i, w, h, pos, self.cache, self.shared, linear=lambda x, m: m(x), rounded=fp16)
        self.cache.length = start + len(ids)
        hn = M.rms_norm(h[0], self.norm, cfg.eps)
        logits = self.head(hn)
        return (torch.tanh(logits / cfg.softcap) * cfg.softcap if cfg.softcap else logits), hn

    def crop(self, length):
        self.cache.crop(length)


class Small:
    def __init__(self):
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig

        config = Gemma4TextConfig(**ETEXT)
        state = {k: v.to(DEV) for k, v in randomize(Gemma4ForCausalLM(config).eval(), 5).state_dict().items()}
        self.cfg = cfg = M.Config.from_text_config(config.to_dict())
        self.ref_layers, self.eng_layers = [], []
        for i in range(cfg.n_layers):
            w = M.layer_weights(state, "model.", i)
            mats = {name: Stored(w[name], 6 if (i + j) % 2 else 8) for j, name in enumerate(M.LINEARS) if name in w}
            rest = {k: v for k, v in w.items() if k not in mats}
            self.ref_layers.append(rest | mats)
            self.eng_layers.append(rest | {"mats": {name: m.stored() for name, m in mats.items()}})
        self.embed, self.norm = state["model.embed_tokens.weight"], state["model.norm.weight"]
        self.head = Stored(self.embed, 6)

    def engine(self, capacity: int = 96, **more) -> E.Engine:
        return E.Engine(self.cfg, self.eng_layers, self.head.stored(), self.norm, self.embed, capacity=capacity, **more)

    def reference(self) -> Reference:
        return Reference(self.cfg, self.ref_layers, self.head, self.norm, self.embed)


@pytest.fixture(scope="module")
def small_e():
    return Small()


def ids_of(n: int, seed: int) -> list:
    return torch.randint(3, ETEXT["vocab_size"], (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def close(got: torch.Tensor, want: torch.Tensor, tol: float) -> bool:
    """Every row within `tol` of the row's largest value."""
    return bool(((got - want).abs().amax(-1) <= tol * want.abs().amax(-1)).all())


def same_choice(got: torch.Tensor, want: torch.Tensor, tol: float) -> bool:
    """Every row's first choice in `got` is, in `want`, the first choice or within `tol` of it (random models have
    near-ties, and two computations that agree to `tol` may break one either way)."""
    picked = want.gather(-1, got.argmax(-1, keepdim=True))[..., 0]
    return bool((picked >= want.amax(-1) - tol * want.abs().amax(-1)).all())


def test_one_matrix_times_up_to_eight_inputs():
    # an engine without layers, as wide as the real model: its work memory holds any of the real matrices' inputs
    cfg = M.Config(n_layers=0, hidden=5376, mlp=21504, heads=4, layer_types=[], head_dim={M.SLIDING: 32, M.FULL: 64},
                   kv_heads={M.SLIDING: 2, M.FULL: 1}, window=5, vocab=32, eps=1e-6, softcap=None, rope={})
    g = torch.Generator().manual_seed(0)
    table = torch.randn(32, 5376, generator=g)
    eng = E.Engine(cfg, [], Stored(table, 8).stored(), torch.ones(5376), table, capacity=16)
    for rows, cols, bits in ((64, 192, 6), (64, 192, 8), (32, 2048, 8), (48, 5376, 6), (32, 5376, 8), (16, 8192, 6), (32, 21504, 6), (16, 21504, 8)):
        m = Stored(torch.randn(rows, cols, generator=g), bits)
        mat = eng._mat(*m.stored())
        for tokens in (1, 3, 8):
            x = torch.randn(tokens, cols, generator=g).to(DEV)
            x[-1] *= 3000  # too long for fp16 as it is: the kernel scales it down by a power of two and back
            assert close(eng.linear(mat, x), m(x), 2e-5), (rows, cols, bits, tokens)


# How close the two can be. Sums in another order differ in the last bit, and now and then that turns a value's
# rounding to fp16 the other way: one such flip moves a matrix's output by about 1e-4 of its size, and a flipped
# attention weight near one moves attention's output by 5e-4. Within one layer that is all there is; over the layers
# the flips add up and attention's sharp softmax multiplies them. (The kernels themselves are checked without
# flips: test_one_matrix..., test_attention_kernels...)
ONE_LAYER, WHOLE_MODEL = 5e-3, 2e-2


def test_attention_kernels_against_pytorch_on_the_engines_own_buffers(small_e):
    # after a pass of a one-layer engine its work memory holds that layer's queries, scores, weights and attention output
    base = small_e.cfg
    g = torch.Generator().manual_seed(4)
    for i in (1, 2):  # a sliding layer and a full-attention one
        t = base.layer_types[i]
        cfg = M.Config(**{**vars(base), "n_layers": 1, "layer_types": [t]})
        eng = E.Engine(cfg, [small_e.eng_layers[i]], small_e.head.stored(), small_e.norm, small_e.embed, capacity=96)
        hd, kvh, heads, slots = cfg.head_dim[t], cfg.kv_heads[t], cfg.heads, eng.kv[0][0].shape[0]
        rep, pos = heads // kvh, 0
        for n in (1, 1, 8, 3, 8, 8, 8, 5):
            eng.h[:n] = (torch.randn(n, base.hidden, generator=g) * 3).to(DEV)
            eng.advance([(0, n)])
            lo = 0 if t == M.FULL else max(pos - cfg.window + 1, 0)
            first_tile = lo & ~15
            stride, nq = ((pos + n - 1 >> 4) + 1) * 16 - first_tile, rep * n
            q = eng.work["qhat"].view(-1)[: n * heads * hd].view(n, heads, hd).double()
            scores = eng.work["scores"][: kvh * nq * stride].view(kvh, nq, stride)
            probs = eng.work["probs"][: kvh * nq * stride].view(kvh, nq, stride)
            attn = eng.work["attn"].view(-1)[: n * heads * hd].view(n, heads, hd).double()
            slot = torch.arange(first_tile, first_tile + stride, device=DEV) % slots
            keys, values = eng.kv[0][0][slot].double(), eng.kv[0][1][slot].double()  # [positions, key heads, head_dim]
            for kh in range(kvh):
                for query in range(nq):
                    token, head = query // rep, kh * rep + query % rep
                    a = (max(pos + token - cfg.window + 1, 0) if t == M.SLIDING else 0) - first_tile
                    b = pos + token - first_tile + 1  # the positions this query reads, as indices
                    where = (i, pos, n, kh, query)
                    assert close(scores[kh, query, a:b].double(), keys[a:b, kh] @ q[token, head], 1e-5), where
                    assert close(probs[kh, query, a:b].float(), torch.softmax(scores[kh, query, a:b], -1), 1e-3), where  # fp16: 5e-4
                    assert not probs[kh, query, :a].any() and not probs[kh, query, b:].any(), where
                    assert close(attn[token, head], probs[kh, query].double() @ values[:, kh], 1e-5), where
            pos += n


def test_each_layer_alone_matches_the_reference_on_the_same_inputs(small_e):
    base = small_e.cfg
    g = torch.Generator().manual_seed(3)
    for i in range(base.n_layers):
        t = base.layer_types[i]
        cfg = M.Config(**{**vars(base), "n_layers": 1, "layer_types": [t]})
        eng = E.Engine(cfg, [small_e.eng_layers[i]], small_e.head.stored(), small_e.norm, small_e.embed, capacity=96)
        cache, taps, pos, width = M.Cache(), {}, 0, cfg.heads * cfg.head_dim[t]
        for n in (1, 1, 8, 3, 8, 8, 8, 5):  # 42 positions: past a sliding layer's ring of 32
            h = (torch.randn(n, base.hidden, generator=g) * 3).to(DEV)
            eng.h[:n] = h
            eng.advance([(0, n)])
            positions = torch.arange(pos, pos + n, device=DEV)
            want = M.layer_forward(cfg, 0, small_e.ref_layers[i], h[None], positions, cache, linear=lambda x, m: m(x), rounded=fp16,
                                   tap=lambda name, x: taps.__setitem__(name, x[0]))[0]
            where = (i, pos, n)
            slot = positions % eng.kv[0][0].shape[0]
            assert close(eng.kv[0][0][slot].float().transpose(0, 1), cache.k[0][0, :, pos : pos + n], ONE_LAYER), where
            assert close(eng.kv[0][1][slot].float().transpose(0, 1), cache.v[0][0, :, pos : pos + n], ONE_LAYER), where
            assert close(eng.work["attn"].view(-1)[: n * width].view(n, width), taps["o_in"], ONE_LAYER), where
            assert close(eng.h[:n], want, ONE_LAYER), where
            pos += n


def test_passes_match_the_reference_past_the_window_and_the_ring(small_e):
    eng, ref = small_e.engine(), small_e.reference()
    ids = ids_of(60, 1)  # a sliding layer keeps window + SPARE positions in whole tiles, 32: 60 go round the ring almost twice
    want_logits, want_hidden = ref.feed(ids)
    got_logits, got_hidden = [], []
    for part in (ids[:13], ids[13:14], ids[14:15], ids[15:18], ids[18:26], ids[26:27], ids[27:60]):
        a, b = eng.feed(part)
        got_logits.append(a)
        got_hidden.append(b)
    assert close(torch.cat(got_hidden), want_hidden, WHOLE_MODEL)
    assert close(torch.cat(got_logits), want_logits, WHOLE_MODEL)
    assert same_choice(torch.cat(got_logits), want_logits, 2 * WHOLE_MODEL)


def test_a_position_is_the_same_bits_alone_or_among_eight(small_e):
    ids = ids_of(40, 2)
    one, eight = small_e.engine(), small_e.engine()
    a = torch.cat([one.feed([t])[0] for t in ids])
    b = eight.feed(ids)[0]
    assert torch.equal(a, b)


def test_cropped_positions_are_forgotten(small_e):
    ids = ids_of(30, 3)
    whole = small_e.engine().feed(ids)[0]
    eng = small_e.engine()
    eng.feed(ids[:17])
    eng.feed(ids_of(6, 4))  # rejected drafts
    eng.crop(17)
    assert torch.equal(eng.feed(ids[17:])[0], whole[17:])


def test_keys_for_the_drafter_match_the_reference(small_e):
    eng, ref = small_e.engine(), small_e.reference()
    ids = ids_of(33, 5)
    eng.feed(ids)
    ref.feed(ids)
    cfg = small_e.cfg
    for length in (33, 30, 25):  # as after a pass whose guesses were all kept, partly kept, all rejected
        got = eng.keys(length)
        for t in (M.SLIDING, M.FULL):
            k, v, pos = ref.shared[t]
            first = 0 if t == M.FULL else max(0, length - (cfg.window + 1))
            assert got[t][2].tolist() == list(range(first, length))
            assert close(got[t][0], k[:, :, first:length], WHOLE_MODEL) and close(got[t][1], v[:, :, first:length], WHOLE_MODEL)
    with pytest.raises(AssertionError):
        eng.keys(3)  # a sliding layer no longer has positions that far back: the ring has gone round


def test_going_back_further_than_the_ring_keeps_is_refused(small_e):
    eng = small_e.engine()
    eng.feed(ids_of(40, 6))
    eng.crop(40 - eng.kept() - 2)
    with pytest.raises(AssertionError):
        eng.feed([5])
    eng.crop(0)  # from the start is always fine: every position is written again before it is read
    eng.feed(ids_of(9, 7))


def test_text_is_the_same_with_any_mix_of_right_and_wrong_guesses(small_e):
    truth = generate(small_e.engine(), PROMPT, 40, Picker())
    for right in ([0], [4], [1, 0, 3, 4, 2]):
        guess, calls = oracle(truth, len(PROMPT), right)
        assert generate(small_e.engine(), PROMPT, 40, Picker(), guess) == truth
    pick = Picker(temperature=1.0, top_k=64, top_p=0.95, seed=7)
    sampled = generate(small_e.engine(), PROMPT, 30, pick)
    guess, _ = oracle(sampled, len(PROMPT), [2, 0, 4, 1])
    assert generate(small_e.engine(), PROMPT, 30, pick, guess) == sampled


@pytest.fixture(scope="module")
def drafter_e():
    """A small random drafter for the `small_e` model, its matrices as stored pair codes: (config, its small weights and
    Stored matrices by layer, pre, post, norm, head)."""
    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4TextConfig

    text = dict(ETEXT, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_kv_shared_layers=4, final_logit_softcapping=None,
                layer_types=["sliding_attention"] * 3 + ["full_attention"], vocab_size_per_layer_input=0)
    config = Gemma4AssistantConfig(text_config=Gemma4TextConfig(**text), backbone_hidden_size=ETEXT["hidden_size"])
    hf = randomize(Gemma4AssistantForCausalLM(config).eval(), 6)
    hf.lm_head.weight = hf.model.embed_tokens.weight
    state = {k: v.to(DEV) for k, v in hf.state_dict().items()}
    cfg = M.Config.from_text_config(config.text_config.to_dict())
    layers = []
    for i in range(cfg.n_layers):
        w = M.layer_weights(state, "model.", i)
        layers.append({k: (Stored(v, 6 if (i + j) % 2 else 8) if k in M.LINEARS else v) for j, (k, v) in enumerate(w.items())})
    return (cfg, layers, Stored(state["pre_projection.weight"], 8), Stored(state["post_projection.weight"], 6), state["model.norm.weight"],
            Stored(state["model.embed_tokens.weight"], 8))


def engine_drafter(eng: E.Engine, drafter_e) -> E.EngineDrafter:
    cfg, layers, pre, post, norm, head = drafter_e
    lays = [{k: v for k, v in w.items() if k not in M.LINEARS} | {"mats": {k: v.stored() for k, v in w.items() if k in M.LINEARS}} for w in layers]
    return E.EngineDrafter(eng, cfg, lays, pre.stored(), post.stored(), norm, head.stored())


def torch_drafter(drafter_e) -> Drafter:
    """The same drafter in PyTorch, its matrices as the codes give them."""
    cfg, layers, pre, post, norm, head = drafter_e
    weight = lambda m: C.weight(m.codes, m.scales, m.book.half().float())  # noqa: E731
    lays = [{k: (weight(v) if k in M.LINEARS else v) for k, v in w.items()} for w in layers]
    return Drafter(cfg, lays, weight(pre), weight(post), norm, weight(head))


def test_the_drafter_on_the_engine_matches_the_reference_drafter(small_e, drafter_e):
    cfg, layers, pre, post, norm, head = drafter_e
    ref = Drafter(cfg, layers, pre, post, norm, head)  # its matrices are Stored: the product as the kernels compute it
    eng = small_e.engine()
    nat = engine_drafter(eng, drafter_e)
    ids = ids_of(44, 12)
    hidden = torch.cat([eng.feed(ids[a : a + 11])[1] for a in range(0, 44, 11)])
    g = torch.Generator().manual_seed(13)
    for length in (44, 41, 36):  # as after a pass whose guesses were all kept, partly kept, all rejected
        keys = {t: (k.float(), v.float(), pos) for t, (k, v, pos) in eng.keys(length).items()}
        embed, state = eng.embed([ids[length - 1]])[0], hidden[length - 1]
        for step in range(3):
            want_logits, want_next = ref.step(embed[None, None], state[None, None], torch.tensor([length - 1], device=DEV), keys,
                                              linear=lambda x, m: m(x), rounded=fp16)
            logits, state = nat.step(embed, state, length)
            assert close(logits[None], want_logits[0], WHOLE_MODEL) and close(state[None], want_next[0], WHOLE_MODEL), (length, step)
            assert same_choice(logits[None], want_logits[0], 2 * WHOLE_MODEL)
            state = state.clone()
            embed = eng.embed([int(torch.randint(3, 300, (1,), generator=g))])[0]
    # whole guesses: six tokens, and the first is the PyTorch drafter's first or tied with it
    plain = torch_drafter(drafter_e)
    for length in (44, 40):
        got = nat.guess(ids[length - 1], hidden[length - 1], length)
        keys = {t: (k.float(), v.float(), pos) for t, (k, v, pos) in eng.keys(length).items()}
        want_logits, _ = plain.step(eng.embed([ids[length - 1]])[None], hidden[length - 1][None, None], torch.tensor([length - 1], device=DEV), keys)
        assert len(got) == 6 and want_logits[0, 0, got[0]] >= want_logits.max() - 2 * WHOLE_MODEL * want_logits.abs().max()


def test_with_the_drafter_the_text_is_unchanged(small_e, drafter_e):
    truth = generate(small_e.engine(), PROMPT, 40, Picker())
    eng = small_e.engine()
    assert generate(eng, PROMPT, 40, Picker(), engine_drafter(eng, drafter_e).guess) == truth
    eng = small_e.engine()
    assert generate(eng, PROMPT, 40, Picker(), drafter_guess(torch_drafter(drafter_e), eng, 6)) == truth
    pick = Picker(temperature=1.0, top_k=64, top_p=0.95, seed=3)
    sampled = generate(small_e.engine(), PROMPT, 40, pick)
    eng = small_e.engine()
    assert generate(eng, PROMPT, 40, pick, engine_drafter(eng, drafter_e).guess) == sampled


def test_packing_a_part_of_the_rows_at_a_time_is_the_same():
    codes = torch.randint(0, 64, (48, 96), generator=torch.Generator().manual_seed(8)).to(torch.uint8)
    for bits in (6, 8):
        assert torch.equal(E.pack(codes, bits, strips_at_once=1), E.pack(codes, bits))


def test_restore_comes_back_from_further_than_the_ring_keeps(small_e):
    a, b, c = ids_of(30, 9), ids_of(50, 10), ids_of(12, 11)  # 50 tokens: more than once round a sliding layer's ring
    straight = small_e.engine()
    straight.feed(a)
    want = straight.feed(c)[0]
    eng = small_e.engine()
    eng.feed(a)
    eng.snapshot()
    assert eng.feed(b, keep="last")[0].shape[0] == 50 % eng.rows
    eng.restore()
    assert eng.length == 30 and torch.equal(eng.feed(c)[0], want)


# ---- several sequences in one engine

def alone(small_e, ids: list, parts: list) -> torch.Tensor:
    """The logits of one sequence fed by itself, in the given parts."""
    eng = small_e.engine()
    return torch.cat([eng.feed(ids[a:b])[0] for a, b in zip([0] + parts, parts + [len(ids)])])


def test_sequences_side_by_side_are_the_same_bits_as_each_alone(small_e):
    a, b, c = ids_of(60, 21), ids_of(45, 22), ids_of(9, 23)  # a and b go round the sliding ring; c stays short
    eng = small_e.engine(sequences=3)
    got = {0: [], 1: [], 2: []}
    # passes that mix the three in every way: (sequence, tokens) by row, eight rows at most
    plan = [[(0, 3), (1, 5)], [(1, 1), (0, 1), (2, 6)], [(2, 3), (1, 2), (0, 3)], [(0, 8)], [(1, 4), (0, 4)], [(1, 8)], [(0, 1), (1, 7)],
            [(0, 7), (1, 1)], [(0, 8)], [(1, 8)], [(0, 8)], [(1, 8)], [(0, 8)], [(1, 1), (0, 7)], [(0, 2)]]
    at = {0: 0, 1: 0, 2: 0}
    text = {0: a, 1: b, 2: c}
    for parts in plan:
        logits, _ = eng.run([(seq, text[seq][at[seq] : at[seq] + n]) for seq, n in parts])
        row = 0
        for seq, n in parts:
            got[seq].append(logits[row : row + n].clone())
            at[seq], row = at[seq] + n, row + n
    assert at == {0: 60, 1: 45, 2: 9}
    for seq, ids in text.items():
        assert torch.equal(torch.cat(got[seq]), alone(small_e, ids, [])), seq


def test_a_sequence_does_not_depend_on_what_the_others_hold(small_e):
    a = ids_of(40, 24)
    outs = []
    for other in (ids_of(40, 25), ids_of(40, 26)):
        eng = small_e.engine(sequences=2)
        got = []
        for i in range(0, 40, 4):
            logits, _ = eng.run([(1, other[i : i + 4]), (0, a[i : i + 4])])
            got.append(logits[4:].clone())
        outs.append(torch.cat(got))
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], alone(small_e, a, []))


def test_tiles_are_given_back_and_used_again(small_e):
    cfg = small_e.cfg
    ring = {t: E.ring_tiles(cfg, t, 96) for t in (M.SLIDING, M.FULL)}
    # a pool with room for one long sequence and a little more
    eng = small_e.engine(sequences=3, pool={M.SLIDING: ring[M.SLIDING] + 1, M.FULL: ring[M.FULL] + 1})
    a, b = ids_of(70, 27), ids_of(20, 28)
    eng.feed(ids_of(10, 29), seq=1)  # takes the first tile of each kind: sequence 0 gets the ones after it
    first = eng.feed(a[:30], seq=0)[0]
    assert eng.tiles[M.FULL][0][:2] == [1, 2] and eng.free_tiles()[M.SLIDING] == 0
    with pytest.raises(E.KVFull):
        eng.feed(b, seq=2)  # nothing is left for a third
    assert eng.lengths[2] == 0 and eng.free_tiles()[M.SLIDING] == 0
    eng.release(1)
    assert eng.free_tiles() == {M.SLIDING: 1, M.FULL: ring[M.FULL] - 1}
    rest = eng.feed(a[30:], seq=0)[0]  # goes on over tiles that are not in order, and round its ring
    assert torch.equal(torch.cat([first, rest]), alone(small_e, a, []))
    eng.release(0)
    assert eng.free_tiles() == {M.SLIDING: ring[M.SLIDING] + 1, M.FULL: ring[M.FULL] + 1}
    assert torch.equal(eng.feed(b, seq=2)[0], alone(small_e, b, []))  # over tiles another sequence has written
    # after a crop the tiles past the length can be given back
    eng.crop(3, seq=2)
    eng.trim(2)
    assert eng.free_tiles() == {M.SLIDING: ring[M.SLIDING], M.FULL: ring[M.FULL]}
    assert torch.equal(eng.feed(b[3:], seq=2)[0], alone(small_e, b, [])[3:])


def test_snapshot_and_restore_of_one_sequence_among_others(small_e):
    a, b, c, other = ids_of(30, 30), ids_of(50, 31), ids_of(12, 32), ids_of(64, 33)
    want = alone(small_e, a + c, [])[30:]
    eng = small_e.engine(sequences=2)
    eng.feed(other[:5], seq=0)
    eng.feed(a[:10], seq=1)
    eng.feed(other[5:40], seq=0)  # the two sequences' tiles alternate in the pool
    eng.feed(a[10:], seq=1)
    eng.snapshot(1)
    eng.feed(b, keep="last", seq=1)
    eng.feed(other[40:], keep="last", seq=0)
    eng.restore(1)
    assert eng.lengths == [64, 30] and torch.equal(eng.feed(c, seq=1)[0], want)


def test_the_drafter_guesses_for_several_sequences_at_once(small_e, drafter_e):
    texts = [ids_of(44, 34), ids_of(23, 35), ids_of(37, 36)]
    eng = small_e.engine(sequences=3)
    nat = engine_drafter(eng, drafter_e)
    hidden = [eng.feed(ids, seq=seq)[1][-1].clone() for seq, ids in enumerate(texts)]
    asks = [(ids[-1], hidden[seq], len(ids), seq, depth) for (seq, ids), depth in zip(enumerate(texts), (6, 2, 4))]
    one = [nat.guess(token, h, length, seq)[:depth] for token, h, length, seq, depth in asks]
    assert nat.guess_many(asks) == one
    assert nat.guess_many(asks[::-1]) == one[::-1]
    # and the same as a drafter beside an engine that holds only that sequence
    for seq, ids in enumerate(texts):
        solo = small_e.engine()
        h = solo.feed(ids)[1][-1]
        assert engine_drafter(solo, drafter_e).guess(ids[-1], h, len(ids)) [: len(one[seq])] == one[seq]


def test_a_position_is_the_same_bits_in_passes_of_any_width(small_e):
    ids = ids_of(70, 37)
    narrow = small_e.engine()
    want = narrow.feed(ids)[0]
    for rows in (16, 32):
        wide = small_e.engine(rows=rows)
        assert torch.equal(wide.feed(ids)[0], want), rows  # 70 tokens: whole passes and a last one of 6
        mixed = small_e.engine(rows=rows)
        got = torch.cat([mixed.feed(ids[a:b])[0] for a, b in ((0, 1), (1, 10), (10, 27), (27, 28), (28, 53), (53, 70))])
        assert torch.equal(got, want), rows


def test_wide_passes_over_several_sequences_and_the_drafter(small_e, drafter_e):
    texts = [ids_of(20 + 3 * i, 110 + i) for i in range(12)]
    eng = small_e.engine(sequences=12, rows=32)
    got = [[] for _ in texts]
    for a in range(0, 54, 2):  # every pass: two tokens of each sequence that still has some (24 rows at first)
        parts = [(seq, ids[a : a + 2]) for seq, ids in enumerate(texts) if a < len(ids)]
        logits, _ = eng.run(parts)
        row = 0
        for seq, part in parts:
            got[seq].append(logits[row : row + len(part)].clone())
            row += len(part)
    for seq, ids in enumerate(texts):
        assert torch.equal(torch.cat(got[seq]), alone(small_e, ids, [])), seq
    # the drafter guesses for all twelve at once, each as it does alone
    nat = engine_drafter(eng, drafter_e)
    hidden = {}
    for seq, ids in enumerate(texts):
        solo = small_e.engine()
        hidden[seq] = solo.feed(ids)[1][-1].clone()
    asks = [(ids[-1], hidden[seq], len(ids), seq, 3 + seq % 4) for seq, ids in enumerate(texts)]
    assert nat.guess_many(asks) == [nat.guess(token, h, length, seq)[:depth] for token, h, length, seq, depth in asks]


def test_a_parked_sequence_comes_back_on_another_number_and_other_tiles(small_e):
    a, b, more = ids_of(45, 120), ids_of(20, 121), ids_of(10, 122)  # a has gone round the sliding ring
    want = alone(small_e, a + more, [])[45:]
    eng = small_e.engine(sequences=3)
    eng.feed(a[:20], seq=0)
    eng.feed(b, seq=1)
    eng.feed(a[20:], seq=0)
    before = eng.free_tiles()
    state = eng.park(0)
    assert eng.lengths[0] == 0 and all(eng.free_tiles()[t] > before[t] for t in before) and state["bytes"] > 0
    eng.feed(ids_of(30, 123), seq=0)  # the number and the tiles are used by another text meanwhile
    eng.unpark(2, state)
    assert eng.lengths[2] == 45 and torch.equal(eng.feed(more, seq=2)[0], want)
    full = small_e.engine(sequences=2, pool={t: 2 for t in (M.SLIDING, M.FULL)})
    with pytest.raises(E.KVFull):
        full.unpark(0, state)
    assert full.free_tiles() == {M.SLIDING: 2, M.FULL: 2}
