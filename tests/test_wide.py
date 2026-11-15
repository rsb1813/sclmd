# Passes of 64, 128 and 256 rows (brief B2): a row has the same bits in them as in a pass of 8, with fp16 products
# (gemm_k in 32-row launches) and with the integer option (gemm8_wide_k); attention takes as many rows at a time as
# its memory holds (32 at least), segments cut at those edges; a wide pass has the logits of its last 32 rows at most.
import copy
import os

import pytest
import torch

from sclmd import engine as E
from test_engine import Small, drafter_e, engine_drafter, ids_of, small_e  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.exists(E.DLL),
                                reason="needs a GPU and engine\\build\\sclmd.dll (engine\\build.bat)")
CAP = 336  # positions: a pass of 256 rows and more


def cuts(rows: int, n: int) -> list:
    """Passes of uneven sizes up to `rows` over n tokens: one row, a full pass, one past a group, ..."""
    sizes, out, a = [1, rows, rows - 7, 33, 3, rows, 40], [], 0
    for s in sizes * 4:
        if a >= n:
            break
        out.append((a, min(n, a + s)))
        a = min(n, a + s)
    return out


@pytest.mark.parametrize("int8", [0.0, E.INT8_SIGMA], ids=["fp16", "int8"])
def test_a_row_is_the_same_bits_in_passes_of_64_to_256_rows(small_e, int8):
    ids = ids_of(300, 41)
    narrow = small_e.engine(capacity=CAP, int8=int8)
    want, want_hn = narrow.feed(ids)
    for rows in (64, 128, 256):
        wide = small_e.engine(capacity=CAP, rows=rows, int8=int8)
        for a, b in cuts(rows, len(ids)):
            logits, hn = wide.run([(0, ids[a:b])])
            assert torch.equal(hn, want_hn[a:b]), (rows, a, b)
            first = E.logits_first(b - a, 0)
            assert logits.shape[0] == b - a - first and torch.equal(logits, want[a + first : b]), (rows, a, b)
        # what the keys and values hold: a prompt read in wide passes, then one row
        lg, _ = wide.run([(0, [5])])
        ref = small_e.engine(capacity=CAP, int8=int8)
        ref.feed(ids)
        assert torch.equal(lg, ref.run([(0, [5])])[0]), rows


@pytest.mark.parametrize("int8", [0.0, E.INT8_SIGMA], ids=["fp16", "int8"])
def test_a_wide_pass_of_several_sequences_cut_at_attentions_edges(small_e, int8):
    # 128 rows: segments of 30, 45, 3 and 50 rows, across the 32-row edges of attention
    texts = [ids_of(n, 60 + i) for i, n in enumerate((90, 135, 9, 150))]
    eng = small_e.engine(capacity=CAP, sequences=4, rows=128, int8=int8)
    got = [[] for _ in texts]
    for k in range(3):
        parts = [(seq, ids[k * len(ids) // 3 : (k + 1) * len(ids) // 3]) for seq, ids in enumerate(texts)]
        _, hn = eng.run(parts)
        row = 0
        for seq, part in parts:
            got[seq].append(hn[row : row + len(part)].clone())
            row += len(part)
    for seq, ids in enumerate(texts):
        alone = small_e.engine(capacity=CAP, int8=int8)
        assert torch.equal(torch.cat(got[seq]), alone.feed(ids)[1]), seq


def test_the_drafter_on_a_wide_engine(small_e, drafter_e):
    ids = ids_of(100, 7)
    wide = small_e.engine(capacity=CAP, rows=64)
    _, hn = wide.feed(ids, keep="last")
    narrow = small_e.engine(capacity=CAP)
    want_hn = narrow.feed(ids)[1]
    assert torch.equal(hn[-1], want_hn[-1])
    nat = engine_drafter(wide, drafter_e)
    assert nat.rows == E.ATT_ROWS
    assert nat.guess(ids[-1], hn[-1], len(ids)) == engine_drafter(narrow, drafter_e).guess(ids[-1], want_hn[-1], len(ids))


# ---- Attention's chunks of 33 to 256 rows in the kernels the real model takes (brief B2 on top of B6)
# The model of test_engine has heads of 32 and 64 values; the real model's are 256 (sliding) and 512 (full), and only
# those go through scores_once_k (from three query groups a key head: 9 rows of a sliding layer, 5 of a full one;
# keyless too, its 64 turned pairs being whole 32s). attend_wide_k takes four sixteens a warp from 32 groups a key
# head: a chunk of 128 rows or more of a sliding layer, 64 of a full one. A pass of one row takes neither (one group:
# scores32_k, attend32_k), so it is the reference: the bits of the kernels before B6.
BIG_CAP = 2048  # the scores' memory holds a whole chunk of 256 rows at the first positions (4 heads * 32 * (2,048 + 32) values)
_BIG = []


@pytest.fixture(scope="module", params=[False, True], ids=["keys", "keyless"])
def big_e(request):
    if not _BIG:
        _BIG.append(Small(head_dim=256, global_head_dim=512))
    one = copy.copy(_BIG[0])
    one.keyless = request.param
    return one


def chunks(eng: E.Engine, parts: list, full: bool) -> list:
    """The rows attention takes at a time in a pass of parts [(first position, rows)] in row order, in a layer of
    each kind (engine/sclmd.cu: pass(), att_for, rows_of): all of them, or as many as the scores' memory holds, in
    steps of ATT_ROWS rows."""
    cfg, room = eng.cfg, eng.model.scores_room
    segs, tokens = [], 0
    for pos, n in parts:
        segs.append((tokens, n, pos))
        tokens += n
    if tokens <= E.ATT_ROWS:
        return [tokens]
    out, c0 = [], 0
    while c0 < tokens:
        cn = tokens - c0
        while True:
            need = 0
            for row0, n, pos in segs:
                lo, end = max(row0, c0), min(row0 + n, c0 + cn)
                if lo < end:
                    first, last = pos + lo - row0, pos + end - 1 - row0
                    base = (0 if full else max(first - (cfg.window - 1), 0)) & ~15
                    need += cfg.heads * (end - lo) * (((last >> 4) + 1) * 16 - base)
            if cn <= E.ATT_ROWS or need <= room:
                break
            cn = max(cn - E.ATT_ROWS, E.ATT_ROWS)
        out.append(cn)
        c0 += cn
    return out


def one_row_a_pass(big_e, ids: list, int8: float, seq_room: int = BIG_CAP) -> tuple:
    """Logits and outputs of every row, each read in a pass of its own, and the engine after."""
    ref = big_e.engine(capacity=seq_room, int8=int8)
    lg, hn = zip(*(tuple(x.clone() for x in ref.run([(0, [t])])) for t in ids))
    return torch.cat(lg), torch.cat(hn), ref


@pytest.mark.parametrize("int8", [0.0, E.INT8_SIGMA], ids=["fp16", "int8"])
def test_chunks_of_33_to_256_rows_in_the_real_models_kernels(big_e, int8):
    # passes of 256 (whole in attention), 255, 129, 128, 65, 64 and 33 rows (gemm8_wide_k's 8 / 4 / 2 sets of groups
    # at their edges), at positions where the full-attention layers cut them into chunks that are not whole 8s
    sizes = [256, 1, 255, 129, 128, 65, 64, 33, 7]
    ids = ids_of(sum(sizes), 17)
    want, want_hn, ref = one_row_a_pass(big_e, ids, int8)
    wide = big_e.engine(capacity=BIG_CAP, rows=256, int8=int8)
    seen, a = {True: [], False: []}, 0
    for k, n in enumerate(sizes):
        for full in seen:
            seen[full].append(chunks(wide, [(a, n)], full))
        # the logits from a row in the middle of the pass on (of the last 32 at most)
        start = (0, n // 2, n - 1, n)[k % 4] if n > E.ATT_ROWS else 0
        logits, hn = wide.run([(0, ids[a : a + n])], start)
        assert torch.equal(hn, want_hn[a : a + n]), (n, a)
        first = E.logits_first(n, start)
        assert logits.shape[0] == n - first and torch.equal(logits, want[a + first : a + n]), (n, a, start)
        a += n
    # what the keys and values (and a sliding layer's ring) hold after: the next row
    assert torch.equal(wide.run([(0, [9])])[0], ref.run([(0, [9])])[0])
    # the test is not empty: whole chunks of 256 rows, passes cut in two or more, chunks that are not whole 8s
    every = [c for full in seen for p in seen[full] for c in p]
    assert 256 in every and any(len(p) > 1 for p in seen[True]) and any(32 < c < 256 and c % 8 for c in every), seen


@pytest.mark.parametrize("int8", [0.0, E.INT8_SIGMA], ids=["fp16", "int8"])
def test_a_wide_pass_of_more_than_32_sequences(big_e, int8):
    # a segment a sequence: up to 64 in one pass (MAX_SEGS), cut at attention's chunk edges
    texts = [ids_of(6, 100 + s) for s in range(48)]
    eng = big_e.engine(capacity=64, sequences=48, rows=128, int8=int8)
    # less room than its memory has (which would hold every pass here whole): attention in chunks of 64 and 32 rows,
    # of 56 and 64 in the pass of three rows a sequence (cut in a sequence's rows)
    eng.model.scores_room = 5000
    got, cut = [[] for _ in texts], []
    for k, width in enumerate((2, 1, 3)):  # 96, 48 and then 120 rows (40 sequences)
        live = range(48) if k < 2 else range(3, 43)
        done = [sum(len(x) for x in got[s]) for s in live]
        parts = [(s, texts[s][d : d + width]) for s, d in zip(live, done)]
        cut += [chunks(eng, [(eng.lengths[s], len(p)) for s, p in parts], full) for full in (True, False)]
        _, hn = eng.run(parts)
        row = 0
        for s, part in parts:
            got[s].append(hn[row : row + len(part)].clone())
            row += len(part)
    for s, ids in enumerate(texts):
        n = sum(len(x) for x in got[s])
        assert torch.equal(torch.cat(got[s]), one_row_a_pass(big_e, ids[:n], int8, 64)[1]), s
    assert cut[0] == cut[1] == [64, 32] and cut[4] == cut[5] == [56, 64], cut  # the test is not empty
    many = big_e.engine(capacity=16, sequences=E.MAX_SEGS + 1, rows=128, int8=int8)
    with pytest.raises(AssertionError):
        many.run([(s, [5]) for s in range(E.MAX_SEGS + 1)])
