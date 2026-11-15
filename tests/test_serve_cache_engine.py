# The prompt caches (scripts/serve_cache.py) on the small random model, through the Server without HTTP: a request
# is its tokens, req.points its marks, req.scope its user. States saved at the marks as the prompt is read give the
# replies a fresh engine gives; a scope never finds another's kept or parked conversation, prefix state or entry;
# what is past its time goes; parked conversations are page-locked in the one budget; an hour's entry comes back from
# the disk after RAM let it go.
import os
import time

import pytest
import torch

from scripts import serve_admin as A
from scripts import serve_cache as K
from scripts.serve import Request, Server
from sclmd import engine as E
from sclmd import model as M
from test_engine import engine_drafter, ids_of
from test_engine import drafter_e, small_e  # noqa: F401  (fixtures)
from test_serve import STOP, finish, fresh, hot

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.exists(E.DLL), reason="needs a GPU and the engine")


def cached(eng, drafter=None, ram: int = 1 << 30, folder: str = "", disk: int = 0, turns: bool = False) -> Server:
    server = Server(None, eng, drafter, STOP)
    K.attach(server, K.Cache(ram, folder, disk, turns))
    return server


def made(full, max_new, pick, scope=None, points=(), cut=0) -> Request:
    req = Request(full, len(full) - 1, max_new, pick, False, cut=cut)
    req.scope, req.points = scope, list(points)
    return req


def ask(server, full, max_new, pick, scope=None, points=(), cut=0) -> Request:
    req = made(full, max_new, pick, scope, points, cut)
    server.submit(req)
    return req


def details(result) -> dict:
    return result["usage"]["prompt_tokens_details"]


def test_marks_are_saved_as_the_prompt_is_read_and_give_the_same_replies(small_e, drafter_e):  # noqa: F811
    eng = small_e.engine(sequences=4)
    server = cached(eng, engine_drafter(eng, drafter_e))
    doc = ids_of(60, 300)  # longer than the sliding layers' ring (32 positions)
    fulls = [doc + ids_of(5 + i, 310 + i) + [5, 6] for i in range(4)]
    marks = [(24, 900, "mark"), (60, 3600, "mark")]
    (t1, r1), = finish(server, ask(server, fulls[0], 12, hot(1), "key:a", marks))
    assert t1 == fresh(small_e, fulls[0], 12, hot(1))
    assert details(r1) == {"cached_tokens": 0, "cache_write_tokens": 60}
    assert sorted((e.n, e.ttl, e.scope) for e in server.cache.store.held.values()) == [(24, 900, "key:a"), (60, 3600, "key:a")]
    # another question on the same document: the state at its end, the same reply as reading it all
    (t2, r2), = finish(server, ask(server, fulls[1], 12, hot(2), "key:a", marks))
    assert details(r2) == {"cached_tokens": 60, "cache_write_tokens": 0} and t2 == fresh(small_e, fulls[1], 12, hot(2))
    # without marks too (an entry is found by the tokens, wherever the request marks)
    (t3, r3), = finish(server, ask(server, fulls[2], 12, hot(3), "key:a"))
    assert details(r3)["cached_tokens"] == 60 and t3 == fresh(small_e, fulls[2], 12, hot(3))
    # another user: nothing of it, and the same reply
    (t4, r4), = finish(server, ask(server, fulls[3], 12, hot(4), "key:b", marks[:1]))
    assert details(r4) == {"cached_tokens": 0, "cache_write_tokens": 24} and t4 == fresh(small_e, fulls[3], 12, hot(4))
    assert server.cache.store.hits == 2 and A.audit(server)["ok"]


def test_several_pauses_in_one_prompt_and_the_prefix_state(small_e):  # noqa: F811
    eng = small_e.engine(sequences=3)
    server = cached(eng)
    system = ids_of(20, 320)
    full = system + ids_of(50, 321) + [5, 6]
    (t1, r1), = finish(server, ask(server, full, 10, hot(), "s", [(40, 900, "mark"), (41, 900, "mark"), (66, 900, "mark")], cut=20))
    assert t1 == fresh(small_e, full, 10, hot())
    assert sorted(e.n for e in server.cache.store.held.values()) == [20, 40, 41, 66] and details(r1)["cache_write_tokens"] == 66
    for n in (20, 40, 41, 66):  # each state reads on to the same reply
        other = full[:n] + [t for t in ids_of(8, 330 + n) if t != full[n]][:4] + [5, 6]  # (not full's next token)
        (t, r), = finish(server, ask(server, other, 8, hot(n), "s"))
        assert details(r)["cached_tokens"] == n and t == fresh(small_e, other, 8, hot(n)), n
    assert A.audit(server)["ok"]


def test_a_user_never_goes_on_from_anothers_kept_or_parked_conversation(small_e):  # noqa: F811
    cfg = small_e.cfg
    ring = {t: E.ring_tiles(cfg, t, 96) for t in (M.SLIDING, M.FULL)}
    eng = small_e.engine(sequences=2, pool={M.SLIDING: ring[M.SLIDING] + 2, M.FULL: ring[M.FULL] + 2})
    server = cached(eng)
    first = ids_of(30, 340) + [5, 6]
    finish(server, ask(server, first, 10, hot(), "key:a"))
    nxt = first[:-1] + ids_of(6, 341) + [5, 6]  # its next turn
    (tb, rb), = finish(server, ask(server, nxt, 8, hot(1), "key:b"))
    assert details(rb)["cached_tokens"] == 0 and tb == fresh(small_e, nxt, 8, hot(1))
    (ta, ra), = finish(server, ask(server, nxt, 8, hot(1), "key:a"))
    assert details(ra)["cached_tokens"] == len(first) - 1 and ta == fresh(small_e, nxt, 8, hot(1))
    # parked: three more conversations push key:a's off the GPU
    for i in range(3):
        f = ids_of(30, 350 + i) + [5, 6]
        finish(server, ask(server, f, 10, hot(), "key:c"))
    assert any(p.scope == "key:a" for p in server.parked)
    assert all(t._locked.ok and t.is_pinned() for p in server.parked for k in p.state["kinds"].values() for t in k["data"] if t.numel())
    again = nxt[:-1] + ids_of(5, 360) + [5, 6]
    (tb, rb), = finish(server, ask(server, again, 8, hot(2), "key:d"))  # (key:b's own conversation was nxt: key:b would go on from it)
    assert details(rb)["cached_tokens"] == 0 and tb == fresh(small_e, again, 8, hot(2))
    (ta, ra), = finish(server, ask(server, again, 8, hot(2), "key:a"))
    assert details(ra)["cached_tokens"] >= len(nxt) - 1 and ta == fresh(small_e, again, 8, hot(2))
    assert A.audit(server)["ok"]


def test_kept_parked_and_saved_states_go_after_their_time(small_e):  # noqa: F811
    cfg = small_e.cfg
    ring = {t: E.ring_tiles(cfg, t, 96) for t in (M.SLIDING, M.FULL)}
    eng = small_e.engine(sequences=2, pool={M.SLIDING: ring[M.SLIDING] + 2, M.FULL: ring[M.FULL] + 2})
    server = cached(eng)
    for i in range(3):
        f = ids_of(30, 370 + i) + [5, 6]
        finish(server, ask(server, f, 6, hot(), "s", [(24, 3600 if i == 0 else 900, "mark")]))
    assert server.kept and server.parked and len(server.cache.store.held) == 3
    now = time.time()
    clock = lambda: now + 15 * 60 + 5  # noqa: E731
    server.cache.clock = server.cache.store.clock = clock
    server.cache.tick(server)  # (what the engine's thread does at the start of a turn)
    assert not server.kept and not server.parked and [e.ttl for e in server.cache.store.held.values()] == [3600]
    assert all(eng.lengths[s] == 0 for s in range(eng.sequences)) and A.audit(server)["ok"]


def test_one_budget_for_parked_conversations_and_entries(small_e):  # noqa: F811
    cfg = small_e.cfg
    ring = {t: E.ring_tiles(cfg, t, 96) for t in (M.SLIDING, M.FULL)}
    eng = small_e.engine(sequences=2, pool={M.SLIDING: ring[M.SLIDING] + 2, M.FULL: ring[M.FULL] + 2})
    one = sum(E.tile_bytes(cfg, t, eng.lean, eng.kv4) * ring[t] for t in ring)  # about a conversation of 32+ tokens
    server = cached(eng, ram=3 * one)
    for i in range(5):
        f = ids_of(34, 380 + i) + [5, 6]
        finish(server, ask(server, f, 6, hot(), "s", [(33, 900, "mark")]))
        held = server.cache.store.bytes() + sum(p.state["bytes"] for p in server.parked)
        assert held <= 3 * one, (i, held)
    assert server.cache.store.evicted > 0 and A.audit(server)["ok"]


def test_cache_turns_keeps_a_boundary_it_reads_and_not_a_kept_conversations_end(small_e):  # noqa: F811
    eng = small_e.engine(sequences=3)
    server = cached(eng, turns=True)
    server.canon = lambda req, tokens: req.full[: req.start] + [t for t in tokens if t not in STOP] + [9, 9]  # a later turn's form
    doc = ids_of(40, 390)  # the document as a message of its own, then the question
    first = doc + ids_of(6, 391) + [5, 6]
    (t1, r1), = finish(server, ask(server, first, 8, hot(), "s", [(40, 900, "turn")]))
    assert t1 == fresh(small_e, first, 8, hot()) and details(r1)["cache_write_tokens"] == 40
    other = doc + ids_of(7, 392) + [5, 6]  # another question on it
    (t2, r2), = finish(server, ask(server, other, 8, hot(1), "s", [(40, 900, "turn")]))
    assert details(r2)["cached_tokens"] == 40 and t2 == fresh(small_e, other, 8, hot(1))
    later = other[:-1] + [t for t in t2 if t not in STOP] + [9, 9]  # (read in ahead on its sequence)
    turn2 = later + ids_of(5, 393) + [5, 6]  # its next turn: it goes on from the kept conversation, which is not copied
    (t3, r3), = finish(server, ask(server, turn2, 8, hot(2), "s", [(len(later), 900, "turn")]))
    assert details(r3) == {"cached_tokens": len(later), "cache_write_tokens": 0} and t3 == fresh(small_e, turn2, 8, hot(2))
    assert server.cache.store.get("s", later) is None
    assert A.audit(server)["ok"]


def test_an_hours_entry_comes_back_from_disk_after_ram_let_it_go(small_e, tmp_path):  # noqa: F811
    eng = small_e.engine(sequences=3)
    folder = str(tmp_path / "cache")
    server = cached(eng, folder=folder, disk=1 << 30)
    doc = ids_of(60, 400)
    f1, f2 = doc + ids_of(5, 401) + [5, 6], doc + ids_of(7, 402) + [5, 6]
    finish(server, ask(server, f1, 8, hot(), "key:a", [(60, 3600, "mark")]))
    disk = server.cache.disk
    t = time.time()
    while disk.written < 1:
        assert time.time() - t < 20
        time.sleep(0.02)
    assert len(os.listdir(folder)) == 1
    with server.cache.store.lock:  # RAM lets it go
        for e in list(server.cache.store.held.values()):
            server.cache.store._drop(e)
    for k in list(server.kept.values()):
        server._forget(k)
    other = made(f2, 8, hot(1), "key:b")
    server.cache.prefetch(other)  # (as do_POST does before the request goes to the engine)
    assert server.cache.arrivals.empty()  # another user: nothing read
    server.submit(other)
    (tb, rb), = finish(server, other)
    assert details(rb)["cached_tokens"] == 0 and tb == fresh(small_e, f2, 8, hot(1))
    mine = made(f2, 8, hot(1), "key:a")
    server.cache.prefetch(mine)
    assert server.cache.from_disk == 1 and disk.read_n == 1
    server.submit(mine)
    (ta, ra), = finish(server, mine)
    assert details(ra)["cached_tokens"] == 60 and ta == tb
    disk.close()
    assert os.listdir(folder) == []


def test_a_point_the_sequence_has_written_past_sets_no_pause(small_e):  # noqa: F811
    eng = small_e.engine(sequences=2)
    server = cached(eng)
    first = ids_of(30, 410) + [5, 6]
    finish(server, ask(server, first, 20, hot()))  # a reply of 20: the sliding ring goes past the turn start
    nxt = first[:-1] + ids_of(10, 411) + [5, 6]
    req = ask(server, nxt, 8, hot(1), None, [(len(first) + 3, 900, "mark")])
    server.turn(wait=False)
    assert req.job is not None and req.reused == len(first) - 1 and eng.highs[req.job.seq] > len(first) + 3
    assert req.job.pause is None and server.cache.skipped == 1  # it could not be saved: no pause for it
    (t, r), = finish(server, req)
    assert t == fresh(small_e, nxt, 8, hot(1)) and details(r)["cache_write_tokens"] == 0 and not server.cache.store.held


def test_cache_turns_sets_no_pause_in_a_conversation_it_goes_on_from(small_e):  # noqa: F811
    from test_serve import finish_one

    eng = small_e.engine(sequences=3)
    server = cached(eng, turns=True)
    server.canon = lambda req, tokens: req.full[: req.start] + [t for t in tokens if t not in STOP] + [9, 9]
    first = ids_of(30, 420) + [5, 6]
    (t1, _), = finish_one(server, ask(server, first, 8, hot(), "s"))
    later = first[:-1] + [t for t in t1 if t not in STOP] + [9, 9]
    turn2 = later + ids_of(5, 421) + [5, 6]
    # it comes before the reading ahead is done: it goes on from what is read, and the boundary is further on
    req = ask(server, turn2, 8, hot(1), "s", [(len(later), 900, "turn")])
    server.turn(wait=False)
    assert req.went_on and req.reused < len(later) and req.job.pause is None
    (t2, r2), = finish(server, req)
    assert t2 == fresh(small_e, turn2, 8, hot(1)) and details(r2)["cache_write_tokens"] == 0 and not server.cache.store.held


def test_states_that_cannot_be_page_locked_are_not_kept(small_e, monkeypatch):  # noqa: F811
    cfg = small_e.cfg
    ring = {t: E.ring_tiles(cfg, t, 96) for t in (M.SLIDING, M.FULL)}
    eng = small_e.engine(sequences=2, pool={M.SLIDING: ring[M.SLIDING] + 2, M.FULL: ring[M.FULL] + 2})
    server = cached(eng)
    monkeypatch.setattr(K, "locked", lambda state: False)  # (as when cudaHostRegister refuses)
    doc = ids_of(40, 430)
    full = doc + ids_of(5, 431) + [5, 6]
    (t, r), = finish(server, ask(server, full, 6, hot(), "s", [(40, 900, "mark")]))
    assert t == fresh(small_e, full, 6, hot()) and details(r)["cache_write_tokens"] == 0
    assert not server.cache.store.held and server.cache.unlocked == 1
    for i in range(3):  # conversations that leave the GPU are forgotten, not parked in pageable RAM
        f = ids_of(30, 440 + i) + [5, 6]
        finish(server, ask(server, f, 6, hot(), "s"))
    assert not server.parked and A.audit(server)["ok"]
