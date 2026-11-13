# The engine (engine/sclmd.cu through sclmd/engine.py) against the reference decoder on a small random Gemma 4.
# The reference gets the same stored matrices and the engine's two roundings (a matrix's input and the kept keys and
# values are fp16), so what is left between the two is the order of float32 sums.
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
        self.book = C.pair_codebook(wr, bits)
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


class RoundedCache(M.Cache):
    def extend(self, layer, k, v):
        return super().extend(layer, k.half().float(), v.half().float())


class Reference:
    """The reference decoder over stored matrices. The same four things generation asks of the engine."""

    def __init__(self, cfg, layers, head, norm, embed):
        self.cfg, self.layers, self.head, self.norm, self.table = cfg, layers, head, norm, embed
        self.cache, self.shared = RoundedCache(), {}

    def embed(self, ids):
        return self.table[torch.tensor(ids, device=DEV)] * M.embed_scale(self.cfg, self.table.dtype)

    def feed(self, ids):
        cfg, start = self.cfg, self.cache.length
        pos = torch.arange(start, start + len(ids), device=DEV)
        h = self.embed(ids)[None]
        for i, w in enumerate(self.layers):
            h = M.layer_forward(cfg, i, w, h, pos, self.cache, self.shared, linear=lambda x, m: m(x))
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

    def engine(self, capacity: int = 96) -> E.Engine:
        return E.Engine(self.cfg, self.eng_layers, self.head.stored(), self.norm, self.embed, capacity=capacity)

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
# rounding to fp16 the other way: one such flip moves a matrix's output by about 1e-4 of its size. Within one layer
# that is all there is; over the layers the flips add up and attention's sharp softmax multiplies them.
ONE_LAYER, WHOLE_MODEL = 1e-3, 1e-2


def test_each_layer_alone_matches_the_reference_on_the_same_inputs(small_e):
    base = small_e.cfg
    g = torch.Generator().manual_seed(3)
    for i in range(base.n_layers):
        t = base.layer_types[i]
        cfg = M.Config(**{**vars(base), "n_layers": 1, "layer_types": [t]})
        eng = E.Engine(cfg, [small_e.eng_layers[i]], small_e.head.stored(), small_e.norm, small_e.embed, capacity=96)
        cache, taps, pos, width = RoundedCache(), {}, 0, cfg.heads * cfg.head_dim[t]
        for n in (1, 1, 8, 3, 8, 8, 8, 5):  # 42 positions: twice round a sliding layer's ring of 21
            h = (torch.randn(n, base.hidden, generator=g) * 3).to(DEV)
            eng.h[:n] = h
            assert eng.lib.sclmd_pass(E.C.byref(eng.model), n, pos) == 0
            positions = torch.arange(pos, pos + n, device=DEV)
            want = M.layer_forward(cfg, 0, small_e.ref_layers[i], h[None], positions, cache, linear=lambda x, m: m(x),
                                   tap=lambda name, x: taps.__setitem__(name, x[0]))[0]
            where = (i, pos, n)
            slot = positions % eng.kv[0][0].shape[0]
            assert close(eng.kv[0][0][slot].float().transpose(0, 1), cache.k[0][0, :, pos : pos + n], ONE_LAYER), where
            assert close(eng.kv[0][1][slot].float().transpose(0, 1), cache.v[0][0, :, pos : pos + n], ONE_LAYER), where
            assert close(eng.work["attn"].view(-1)[: n * width].view(n, width), taps["o_in"], ONE_LAYER), where
            assert close(eng.work["x"].view(-1)[: n * cfg.mlp].view(n, cfg.mlp), taps["down_in"], ONE_LAYER), where
            assert close(eng.h[:n], want, ONE_LAYER), where
            pos += n


def test_passes_match_the_reference_past_the_window_and_the_ring(small_e):
    eng, ref = small_e.engine(), small_e.reference()
    ids = ids_of(60, 1)  # a sliding layer keeps window + SPARE = 21 positions: 60 go round the ring almost three times
    want_logits, want_hidden = ref.feed(ids)
    got_logits, got_hidden = [], []
    for part in (ids[:13], ids[13:14], ids[14:15], ids[15:18], ids[18:26], ids[26:27], ids[27:60]):
        a, b = eng.feed(part)
        got_logits.append(a)
        got_hidden.append(b)
    assert close(torch.cat(got_hidden), want_hidden, WHOLE_MODEL)
    assert close(torch.cat(got_logits), want_logits, WHOLE_MODEL)
    assert (torch.cat(got_logits).argmax(-1) == want_logits.argmax(-1)).all()


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
    eng.crop(40 - E.SPARE - 2)
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


def test_with_the_drafter_the_text_is_unchanged(small_e):
    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4TextConfig

    text = dict(ETEXT, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_kv_shared_layers=4, final_logit_softcapping=None,
                layer_types=["sliding_attention"] * 3 + ["full_attention"], vocab_size_per_layer_input=0)
    config = Gemma4AssistantConfig(text_config=Gemma4TextConfig(**text), backbone_hidden_size=ETEXT["hidden_size"])
    hf = randomize(Gemma4AssistantForCausalLM(config).eval(), 6)
    hf.lm_head.weight = hf.model.embed_tokens.weight
    drafter = Drafter.from_state(M.Config.from_text_config(config.text_config.to_dict()), {k: v.to(DEV) for k, v in hf.state_dict().items()})
    truth = generate(small_e.engine(), PROMPT, 40, Picker())
    eng, stats = small_e.engine(), {}
    assert generate(eng, PROMPT, 40, Picker(), drafter_guess(drafter, eng, 6), stats=stats) == truth
    # and the drafter on the engine's keys guesses what it guesses on the reference's
    ref = small_e.reference()
    ref.keys = lambda length: {t: (k[:, :, :length], v[:, :, :length], pos[:length]) for t, (k, v, pos) in ref.shared.items()}
    eng = small_e.engine()
    (_, he), (_, hr) = eng.feed(PROMPT), ref.feed(PROMPT)
    assert drafter_guess(drafter, eng, 6)(truth[0], he[-1], len(PROMPT)) == drafter_guess(drafter, ref, 6)(truth[0], hr[-1], len(PROMPT))


def test_packing_a_part_of_the_rows_at_a_time_is_the_same():
    codes = torch.randint(0, 64, (48, 96), generator=torch.Generator().manual_seed(8)).to(torch.uint8)
    for bits in (6, 8):
        assert torch.equal(E.pack(codes, bits, strips_at_once=1), E.pack(codes, bits))
