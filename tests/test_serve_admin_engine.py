# The audit of the keys and values' books (scripts/serve_admin.py: audit) and the timed engine methods on the small
# random model (GPU): after every kind of traffic the server sees (conversations kept and read ahead, more of them than
# sequences, keys and values running out, RAM, cancellations, a failed engine thread, shared prefixes) the books hold at
# idle, and the timing wrappers change no token. A spoilt book is caught.
import pytest
import torch

from scripts import serve_admin as A
from scripts.serve import Request, Server
from sclmd import engine as E
from sclmd import model as M
from test_engine import engine_drafter, ids_of
from test_engine import drafter_e, small_e  # noqa: F401  (fixtures)
from test_serve import STOP, ask, finish, finish_one, fresh, hot

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not __import__("os").path.exists(E.DLL), reason="needs a GPU and the engine")


def timed(server: Server) -> A.Admin:
    adm = A.Admin(None, server)
    adm.time_engine()
    return adm


def clean(server: Server) -> dict:
    result = A.audit_at_idle(server)
    assert result["idle"], result
    assert result["ok"], result["problems"]
    return result


def test_the_books_hold_after_kept_conversations_read_ahead_and_restarted_turns(small_e, drafter_e):  # noqa: F811
    eng = small_e.engine(sequences=2)
    server = Server(None, eng, engine_drafter(eng, drafter_e), STOP)
    adm = timed(server)
    written = ids_of(9, 95)
    server.canon = lambda req, tokens: req.full[: req.start] + written
    full1 = ids_of(15, 96) + [5, 6]
    start1 = len(full1) - 1
    (reply1, _), = finish(server, ask(server, full1, start1, 44, hot(1)))  # past the sliding ring: a snapshot
    assert reply1 == fresh(small_e, full1, 44, hot(1))
    clean(server)
    full2 = full1[:start1] + written + ids_of(7, 97) + [5, 6]
    (reply2, _), = finish(server, ask(server, full2, len(full2) - 1, 15, hot(2)))
    assert reply2 == fresh(small_e, full2, 15, hot(2))
    clean(server)
    full3 = full2[: len(full2) - 1] + ids_of(4, 98) + ids_of(6, 99) + [5, 6]  # the reply sent back otherwise: back to the turn start
    (reply3, _), = finish(server, ask(server, full3, len(full3) - 1, 15, hot(3)))
    assert reply3 == fresh(small_e, full3, 15, hot(3))
    clean(server)
    assert sum(adm.pass_n.values()) == server.batch.passes and adm.copy_n.get("snapshot", 0) >= 1


def test_the_books_hold_when_conversations_outnumber_sequences_and_memory(small_e):  # noqa: F811
    cfg = small_e.cfg
    ring = {t: E.ring_tiles(cfg, t, 96) for t in (M.SLIDING, M.FULL)}
    for ram in (1 << 30, 0):
        eng = small_e.engine(sequences=2, pool={M.SLIDING: ring[M.SLIDING] + 2, M.FULL: ring[M.FULL] + 2})
        server = Server(None, eng, None, STOP, ram=ram)
        adm = timed(server)
        firsts = [ids_of(22 + 3 * i, 130 + i) + [5, 6] for i in range(4)]
        for full in firsts:
            (reply, _), = finish(server, ask(server, full, len(full) - 1, 40, hot()))
            assert reply == fresh(small_e, full, 40, hot())
            clean(server)
        for i in (0, 2, 1, 3):
            full2 = firsts[i][: len(firsts[i]) - 1] + ids_of(6, 140 + i) + [5, 6]
            (reply2, _), = finish(server, ask(server, full2, len(full2) - 1, 9, hot(i)))
            assert reply2 == fresh(small_e, full2, 9, hot(i))
            counts = clean(server)["counts"]
        if ram:
            assert adm.copy_n.get("park", 0) >= 2 and adm.copy_n.get("unpark", 0) >= 1 and counts["parked"] >= 1
        reqs = [ask(server, ids_of(8 + i, 70 + i), 7 + i, 8, hot()) for i in range(5)]  # five at once on two sequences
        finish(server, *reqs)
        clean(server)


def test_the_books_hold_after_a_cancel_a_failure_and_a_conversation_that_left_while_read_ahead(small_e):  # noqa: F811
    cfg = small_e.cfg
    ring = {t: E.ring_tiles(cfg, t, 96) for t in (M.SLIDING, M.FULL)}
    eng = small_e.engine(sequences=2, pool={M.SLIDING: ring[M.SLIDING] + 1, M.FULL: ring[M.FULL] + 1})
    server = Server(None, eng, None, STOP, ram=1 << 30)
    timed(server)
    written = ids_of(40, 160)
    first = ids_of(12, 161) + [5, 6]
    server.canon = lambda req, tokens: req.full[: req.start] + written if req.full[0] == first[0] else None
    finish_one(server, ask(server, first, len(first) - 1, 6, hot()))
    assert A.audit_at_idle(server)["idle"] is False  # its reading ahead is under way: no audit now
    second = ids_of(12, 162) + [5, 6]
    finish(server, ask(server, second, len(second) - 1, 20, hot(6)))  # the first leaves the GPU in the middle of a pass
    clean(server)
    req = ask(server, ids_of(40, 90), 39, 30, hot())
    for _ in range(3):
        server.turn(wait=False)
    req.cancelled = True
    finish(server, req)
    clean(server)
    adm = A.Admin(None, server)
    adm.time_engine()
    adm.faults["engine"] = 1  # the engine's thread fails (as POST /debug/fault does): every request is told, the engine resets
    doomed = ask(server, ids_of(30, 170), 29, 20, hot())
    with pytest.raises(A.InjectedFault):
        server.turn(wait=False)
    server.reset()
    assert doomed.events.empty() or doomed.events.get()[0] == "token"
    clean(server)
    (tokens, _), = finish(server, ask(server, ids_of(12, 171), 11, 8, hot()))
    assert tokens == fresh(small_e, ids_of(12, 171), 8, hot())
    clean(server)


def test_the_books_hold_with_shared_prefixes(small_e, drafter_e):  # noqa: F811
    eng = small_e.engine(sequences=4)
    server = Server(None, eng, engine_drafter(eng, drafter_e), STOP, prefix_ram=1 << 30)
    adm = timed(server)
    system = ids_of(50, 230)
    fulls = [system + ids_of(5 + i, 240 + i) + [5, 6] for i in range(5)]
    reqs = [Request(f, len(f) - 1, 12, hot(i), False, cut=50) for i, f in enumerate(fulls)]
    for r in reqs[:2]:
        server.submit(r)
    got = finish(server, *reqs[:2])
    clean(server)
    for r in reqs[2:]:
        server.submit(r)
        got += finish(server, r)
        clean(server)
    for (tokens, _), f, i in zip(got, fulls, range(5)):
        assert tokens == fresh(small_e, f, 12, hot(i))
    assert server.prefixes.hits == 3 and adm.copy_n.get("save", 0) == 1
    numbers = adm.numbers()
    assert numbers["prefixes"]["bytes"] > 0 and "cuda" in numbers and numbers["cuda"]["reserved_bytes"] > 0


def test_the_books_and_the_numbers_hold_with_kv4():
    """--kv4 (K3): a sliding tile more a sequence for its first tile's int8, and other tile bytes. The audit, the timed
    copies and /metrics' bytes on such an engine, with conversations parked to RAM and back."""
    from test_kv4_engine import model as kv4_model
    from test_serve_admin import parse_metrics

    small = kv4_model("small")
    cfg = small.cfg
    one = {t: E.ring_tiles(cfg, t, 96) + E.extra_tiles(cfg, t, True) for t in (M.SLIDING, M.FULL)}
    eng = small.engine(sequences=2, pool={t: one[t] + 2 for t in one})  # about one long conversation's worth
    assert eng.kv4 and eng.extra[M.SLIDING] == 1
    server = Server(None, eng, None, STOP, ram=1 << 30)
    adm = timed(server)
    firsts = [ids_of(22 + 3 * i, 330 + i) + [5, 6] for i in range(4)]
    for full in firsts:
        (reply, _), = finish(server, ask(server, full, len(full) - 1, 40, hot()))
        assert reply == fresh(small, full, 40, hot())
        clean(server)
    for i in (0, 2, 1, 3):
        full2 = firsts[i][: len(firsts[i]) - 1] + ids_of(6, 340 + i) + [5, 6]
        (reply2, _), = finish(server, ask(server, full2, len(full2) - 1, 9, hot(i)))
        assert reply2 == fresh(small, full2, 9, hot(i))
        clean(server)
    assert adm.copy_n.get("park", 0) >= 2 and adm.copy_n.get("unpark", 0) >= 1
    n = adm.numbers()
    assert n["kv_bytes"] == sum(E.tile_bytes(cfg, t, eng.lean, True) * eng.pool[t] for t in eng.ring)
    assert n["kv_free_bytes"] == sum(E.tile_bytes(cfg, t, eng.lean, True) * eng.free_tiles()[t] for t in eng.ring)
    fams = parse_metrics(A.Admin(type("App", (), {"fingerprint": "x"})(), server).metrics())
    assert {s[1]["kind"]: s[2] for s in fams["sclmd_kv_tiles"]["samples"]} == {t: eng.pool[t] for t in eng.ring}


def test_a_spoilt_book_is_caught(small_e):  # noqa: F811
    eng = small_e.engine(sequences=2)
    server = Server(None, eng, None, STOP)
    finish(server, ask(server, ids_of(20, 250), 19, 8, hot()))
    clean(server)
    seq = next(iter(server.kept))
    tile = next(x for x in eng.tiles[M.FULL][seq] if x >= 0)
    eng.free[M.FULL].append(tile)  # a tile held and free at once
    result = A.audit(server)
    assert not result["ok"] and any("twice" in p for p in result["problems"])
    eng.free[M.FULL].pop()
    eng.lengths[seq] += 64  # a kept conversation that holds more than it has tiles for
    result = A.audit(server)
    assert not result["ok"] and any("missing" in p or "holds" in p for p in result["problems"])
    eng.lengths[seq] -= 64
    server.unused.append(seq)  # a sequence both kept and unused
    assert any("both unused and kept" in p for p in A.audit(server)["problems"])
