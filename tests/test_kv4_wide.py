# Passes of 64, 128 and 256 rows (brief B2) on engines with kv4 (brief K3: the sliding layers' keys and values at 4
# bits): B2's own tests (tests/test_wide.py, tests/test_batch_wide.py) run on kv4 engines, and the ring going round
# within a wide pass. kv4_write keeps every row's queries, keys and values first; kv4_read then takes attention's
# chunks (as many rows as the scores' memory holds), so a chunk that reads the sequence's first tile (int8, its
# positions 0..7 in the bytes of the ring tile 0) runs after the pass has written positions 16 ring + p over that tile:
# it must still find every position it reads (a wide engine's ring holds window + rows positions).
import copy

import pytest
import torch

import test_batch_wide as TB
import test_kv4_engine as T4
import test_wide as TW
from sclmd import engine as E
from sclmd import model as M
from test_engine import Small, drafter_e, ids_of, small_e  # noqa: F401  (fixtures)
from test_wide import big_e  # noqa: F401  (a fixture: heads of 256 and 512, the real model's kernels)

pytestmark = TW.pytestmark
CAP = TW.CAP


def kv4(one):
    """A copy of a test model whose engines all keep their sliding layers' keys and values at 4 bits."""
    two = copy.copy(one)
    two.engine = lambda capacity=96, **more: Small.engine(two, capacity, kv4=True, **more)
    return two


# ---- B2's tests, every engine with kv4

@pytest.mark.parametrize("int8", [0.0, E.INT8_SIGMA], ids=["fp16", "int8"])
def test_kv4_a_row_is_the_same_bits_in_passes_of_64_to_256_rows(small_e, int8):
    TW.test_a_row_is_the_same_bits_in_passes_of_64_to_256_rows(kv4(small_e), int8)


@pytest.mark.parametrize("int8", [0.0, E.INT8_SIGMA], ids=["fp16", "int8"])
def test_kv4_a_wide_pass_of_several_sequences_cut_at_attentions_edges(small_e, int8):
    TW.test_a_wide_pass_of_several_sequences_cut_at_attentions_edges(kv4(small_e), int8)


def test_kv4_the_drafter_on_a_wide_engine(small_e, drafter_e):
    TW.test_the_drafter_on_a_wide_engine(kv4(small_e), drafter_e)


@pytest.mark.parametrize("int8", [0.0, E.INT8_SIGMA], ids=["fp16", "int8"])
def test_kv4_chunks_of_33_to_256_rows_in_the_real_models_kernels(big_e, int8):
    # heads of 256: kv4_scores_once_k and kv4_attend_k with two and four sixteens a warp; the reference passes of one
    # row take kv4_scores_k and one sixteen
    TW.test_chunks_of_33_to_256_rows_in_the_real_models_kernels(kv4(big_e), int8)


@pytest.mark.parametrize("int8", [0.0, E.INT8_SIGMA], ids=["fp16", "int8"])
def test_kv4_a_wide_pass_of_more_than_32_sequences(big_e, int8):
    TW.test_a_wide_pass_of_more_than_32_sequences(kv4(big_e), int8)


@pytest.mark.parametrize("int8", [0.0, E.INT8_SIGMA], ids=["fp16", "int8"])
def test_kv4_texts_written_beside_wide_passes_are_the_texts_written_alone(small_e, drafter_e, int8):
    TB.test_texts_written_beside_wide_passes_are_the_texts_written_alone(kv4(small_e), drafter_e, int8)


def test_kv4_wide_passes_when_keys_and_values_run_short(small_e, drafter_e):
    TB.test_wide_passes_when_keys_and_values_run_short(kv4(small_e), drafter_e)


def test_kv4_a_server_with_wide_passes_answers_as_a_fresh_engine(small_e, drafter_e):
    TB.test_a_server_with_wide_passes_answers_as_a_fresh_engine(kv4(small_e), drafter_e)


# ---- the ring going round within a wide pass

_MODELS = {}


def model16(name: str) -> T4.Small4:
    """A window of 16 (a wide engine's ring is then window + rows positions exactly, as the real model's): the stock
    small model's heads, or heads of 256 with 16 queries a key head (the decode-once scores, four sixteens a warp)."""
    if name == "w16":
        return T4.model("w16")  # (test_kv4_engine.py's)
    if name not in _MODELS:
        _MODELS[name] = T4.Small4(dict(T4.WIDE, sliding_window=16))
    return _MODELS[name]


@pytest.mark.parametrize("int8", [0.0, E.INT8_SIGMA], ids=["fp16", "int8"])
@pytest.mark.parametrize("name", ["w16", "w16wide"])
def test_the_first_tile_outlives_the_ring_going_round_within_a_wide_pass(name, int8):
    m = model16(name)
    window = m.cfg.window
    ids = ids_of(330, 52)
    one = m.engine(capacity=CAP, int8=int8)  # the reference: every position in a pass of its own
    got = [tuple(x.clone() for x in one.run([(0, [t])])) for t in ids]
    want, want_hn = torch.cat([g[0] for g in got]), torch.cat([g[1] for g in got])
    cases = []

    def same(eng, parts):
        lg, hn = eng.run([(seq, ids[a:b]) for seq, a, b in parts])
        total, row = sum(b - a for _, a, b in parts), 0
        first = E.logits_first(total, 0) if total > E.ATT_ROWS else 0
        for seq, a, b in parts:
            assert torch.equal(hn[row : row + b - a], want_hn[a:b]), (name, eng.rows, seq, a, b)
            lo, hi = max(row, first), row + b - a  # its rows that have logits (a wide pass: its last 32 at most)
            if lo < hi:
                assert torch.equal(lg[lo - first : hi - first], want[a + lo - row : b]), (name, eng.rows, seq, a, b)
            row += b - a

    for rows in (64, 128, 256):
        # (capacity, scores' room): attention in one chunk (2,304: room for 256 rows at once), in the chunks the memory
        # of a 336-position engine holds, and in chunks of 32 rows (the room cut to nothing)
        for capacity, room in ((2304, None), (CAP, None), (CAP, 1)):
            eng = m.engine(capacity=capacity, rows=rows, int8=int8, sequences=2)
            if room:
                eng.model.scores_room = room
            wrap = eng.ring[M.SLIDING] * E.TILE
            assert wrap == window + rows  # as tight as the real model's (1,024 + rows)
            # a pass over 22 .. 21 + rows writes wrap .. wrap + 5 over the ring tile 0 (the first tile's int8 positions
            # 0..2, scales 0..5) while its first row reads from 7 on, in the chunk that reads the first tile
            start = 22
            assert start - window + 1 <= 15 and start + rows > wrap
            cases.append((rows, capacity, room, TW.chunks(eng, [(start, rows)], False)))
            eng.feed(ids[:start])
            same(eng, [(0, start, start + rows)])
            same(eng, [(0, start + rows, start + rows + 1)])  # the next row: what the ring holds after
            # a crop back to a turn start whose first row reads the first tile again (allowed: _kept), and the same
            # wide pass across the wrap once more
            back = start + rows + 1 - wrap + window - 1  # (= 22: reads from 7, the first position the wrap left)
            assert eng._kept(back - window + 1) and not eng._kept(back - window)
            eng.crop(back)
            same(eng, [(0, back, back + rows)])
            # a pass shared by two sequences: one being written (4 rows), the other's rows across its wrap while its
            # first row reads its first tile from 9 on
            eng.release(0)
            eng.feed(ids[:40], seq=0)
            eng.feed(ids[:24], seq=1)
            same(eng, [(0, 40, 44), (1, 24, 20 + rows)])
    # the test is not empty: the sliding layers' attention at once, in chunks of 64 and of 32 rows
    every = {c for *_, cut in cases for c in cut}
    assert {256, 64, 32} <= every and any(len(cut) > 1 for *_, cut in cases), cases


# ---- what scripts/prefill_check.py --core --kv4 sums of the sliding layers: their keys and values by position

def test_prefill_checks_sliding_sums_by_position_are_the_same_at_every_width():
    from scripts.bits_check import bits
    from scripts.prefill_check import by_position

    m = model16("w16")
    window, ids = m.cfg.window, ids_of(300, 61)
    for n in (10, 20, 40, 300):  # the first tile in the window, partly, not (its second tile always), the ring gone round
        seen = []
        for rows in (8, 32, 64, 256):
            eng = m.engine(capacity=CAP, rows=rows)
            eng.feed(ids[:n], keep="last")
            got = by_position(eng, n)
            seen.append([x for t in got for x in bits(t)])
            # what it gathers is what the engine keeps (kv4_float: the first tile's int8 times its scales)
            pos = torch.arange(max(0, n - window), n, device=T4.DEV)
            later = pos[pos >= E.TILE]
            first = [p for p in range(min(n, E.TILE)) if p >= 8 or p >= n - window]
            for j, i in enumerate(sorted(eng.kv4_set)):
                assert torch.equal(got[6][j], eng.kv[i][2][eng.slots(M.SLIDING, later).long()]), (n, rows)  # 4-bit keys' scales
                if first:
                    want_k, want_v = eng.kv4_float(i, torch.tensor(first, device=T4.DEV))
                    assert torch.equal(got[8][j].view(torch.int8).float() * got[10][j][..., None], want_k), (n, rows)
                    assert torch.equal(got[9][j].view(torch.int8).float() * got[11][j][..., None], want_v), (n, rows)
            assert len(got) == (12 if first else 8)
        assert all(s == seen[0] for s in seen) and any(seen[0]), n


# ---- the memory a wide kv4 engine works in

_NARROW_FULL = []


def narrow_full() -> Small:
    """A model whose full-attention head (32) is not twice its sliding one (32): kv4's twice-the-queries room is then
    more than the widest head (work_sizes(cfg, kv4=True)), as in no other test model and not in the real one."""
    if not _NARROW_FULL:
        _NARROW_FULL.append(Small(global_head_dim=32))
    return _NARROW_FULL[0]


@pytest.mark.parametrize("int8", [0.0, E.INT8_SIGMA], ids=["fp16", "int8"])
def test_kv4_wide_passes_where_the_queries_twice_are_wider_than_any_head(int8):
    m = kv4(narrow_full())
    cfg = m.cfg
    assert E.work_sizes(cfg, True)[0] == 2 * cfg.heads * cfg.head_dim[M.SLIDING] > E.work_sizes(cfg, False)[0]
    TW.test_a_row_is_the_same_bits_in_passes_of_64_to_256_rows(m, int8)


def test_a_wide_kv4_engines_memory_is_what_fixed_bytes_counts(small_e):
    # its queries and outputs twice for every row (turned and as they are), and fixed_bytes' rows past 32 the same
    # bytes as the engine holds more than a 32-row one
    for one in (small_e, model16("w16wide"), narrow_full()):
        m = kv4(one)
        cfg = m.cfg
        narrow = m.engine(capacity=CAP, rows=32)
        for rows in (64, 128, 256):
            eng = m.engine(capacity=CAP, rows=rows)
            twice = 2 * rows * cfg.heads * cfg.head_dim[M.SLIDING]
            assert eng.work["qhat"].numel() >= twice and eng.work["attn"].numel() >= twice
            grew = eng.bytes() - narrow.bytes()
            counted = E.fixed_bytes(cfg, 8, CAP, rows=rows, keyless=one.keyless, kv4=True) - E.fixed_bytes(cfg, 8, CAP, rows=32, keyless=one.keyless, kv4=True)
            tables = 4 * (eng.ring[M.SLIDING] - narrow.ring[M.SLIDING])  # (the sliding tiles' table, an int32 a tile of the ring: not counted)
            assert grew == counted + tables, (rows, grew, counted, tables)
