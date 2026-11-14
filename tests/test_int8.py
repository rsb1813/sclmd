# The integer products (Engine(int8=sigma), brief B1, an option off by default): a matrix's input rounded to int8 per
# token with the clip at sigma times its rms, the codebook rounded to int8, int32 sums on the s8 tensor-core
# instruction. Checked against PyTorch with the same roundings, and for the same bits at any width.
import os

import pytest
import torch

from sclmd import engine as E
from sclmd import model as M
from test_engine import DEV, Stored, ids_of, small_e  # noqa: F401  (small_e: the fixture)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.exists(E.DLL),
                                reason="needs a GPU and engine\\build\\sclmd.dll (engine\\build.bat)")
SIGMA = E.INT8_SIGMA


def int8_reference(m: Stored, x: torch.Tensor, sigma: float) -> torch.Tensor:
    """What the integer kernel computes: rotated input / (sigma * rms / 127) rounded and clipped, times the int8
    codebook, the exact sums times scale[row] * book_scale * the input's step (in float64)."""
    flat = x.reshape(-1, x.shape[-1]).float()
    n = flat.shape[-1]
    step = sigma * flat.norm(dim=-1, keepdim=True) / n**0.5 / 127.0
    q = torch.clamp(torch.round(m.rot.rotate(flat) / step), -127, 127).double()
    _, s = E.lut8(m.book)
    book = torch.clamp(torch.round(m.book.float() / s), -127, 127).double()
    unit = book[m.codes.long()].flatten(-2)  # [rows, cols]
    return ((q @ unit.t()) * (m.scales.double() * s) * step.double()).float()


def test_one_matrix_on_the_integer_instruction():
    cfg = M.Config(n_layers=0, hidden=5376, mlp=21504, heads=4, layer_types=[], head_dim={M.SLIDING: 32, M.FULL: 64},
                   kv_heads={M.SLIDING: 2, M.FULL: 1}, window=5, vocab=32, eps=1e-6, softcap=None, rope={})
    g = torch.Generator().manual_seed(0)
    table = torch.randn(32, 5376, generator=g)
    eng = E.Engine(cfg, [], Stored(table, 8).stored(), torch.ones(5376), table, capacity=16, rows=32, int8=SIGMA)
    for rows, cols, bits in ((64, 192, 6), (64, 192, 8), (32, 2048, 8), (48, 5376, 6), (32, 5376, 8), (16, 21504, 6), (16, 21504, 8)):
        m = Stored(torch.randn(rows, cols, generator=g), bits)
        mat = eng._mat(*m.stored())
        for tokens in (1, 3, 8, 13, 32):
            x = torch.randn(tokens, cols, generator=g).to(DEV)
            x[-1] *= 3000
            x[0, 5] = 40.0  # a hot value: clipped
            got, want = eng.linear(mat, x), int8_reference(m, x, SIGMA)
            # the rotation's float32 sums may put a value's rounding the other way: a step of 1/127 of sigma rms on
            # one input; everything else is exact
            err = ((got - want).abs().amax(-1) / want.abs().amax(-1)).max()
            assert err < 2e-3, (rows, cols, bits, tokens, float(err))


def test_the_integer_option_is_off_by_default_and_changes_the_bits_when_on(small_e):
    off, on = small_e.engine(), small_e.engine(int8=SIGMA)
    assert off.int8 == 0 or os.environ.get("SCLMD_INT8")
    ids = ids_of(20, 5)
    a, b = off.feed(ids)[0], on.feed(ids)[0]
    assert not torch.equal(a, b)
    # the loss is small next to the logits
    assert ((a - b).abs().amax(-1) / a.abs().amax(-1)).mean() < 0.1


def test_integer_passes_are_the_same_bits_alone_or_at_any_width(small_e):
    ids = ids_of(70, 37)
    narrow = small_e.engine(int8=SIGMA)
    want = narrow.feed(ids)[0]
    one = small_e.engine(int8=SIGMA)
    assert torch.equal(torch.cat([one.feed([t])[0] for t in ids[:20]]), want[:20])
    for rows in (16, 32):
        wide = small_e.engine(rows=rows, int8=SIGMA)
        assert torch.equal(wide.feed(ids)[0], want), rows
        mixed = small_e.engine(rows=rows, int8=SIGMA)
        got = torch.cat([mixed.feed(ids[a:b])[0] for a, b in ((0, 1), (1, 10), (10, 27), (27, 28), (28, 53), (53, 70))])
        assert torch.equal(got, want), rows
