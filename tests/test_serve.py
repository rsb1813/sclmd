# The server's conversations (scripts/serve.py: Server) on the small random model, without HTTP or a tokenizer: a
# request is its tokens. A conversation's next turn goes on from what was kept, and its reply is the one a fresh
# engine gives for the whole conversation.
import pytest
import torch

from scripts.serve import Request, Server
from sclmd import engine as E
from sclmd import model as M
from sclmd.generate import Picker, generate
from test_engine import engine_drafter, ids_of
from test_engine import drafter_e, small_e  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not __import__("os").path.exists(E.DLL), reason="needs a GPU and the engine")
STOP = (7,)


def ask(server: Server, full: list, start: int, max_new: int, pick: Picker) -> Request:
    req = Request(full, start, max_new, pick, False)
    server.submit(req)
    return req


def finish(server: Server, *reqs) -> list:
    """Turns until the requests are done -> each one's (tokens, result)."""
    while server.turn(wait=False):
        pass
    out = []
    for req in reqs:
        tokens, result = [], None
        while not req.events.empty():
            event = req.events.get()
            if event[0] == "token":
                tokens.append(event[1])
            else:
                result = event[1]
        out.append((tokens, result))
    return out


def fresh(small, full: list, max_new: int, pick: Picker) -> list:
    return generate(small.engine(), full, max_new, pick, stop=STOP)


def test_a_conversations_next_turn_goes_on_from_what_was_kept(small_e, drafter_e):  # noqa: F811
    for reply_room in (12, 44):  # 44 tokens of reply: past what the sliding ring keeps of the turn start
        eng = small_e.engine(sequences=3)
        server = Server(None, eng, engine_drafter(eng, drafter_e), STOP)
        system, user1, user2 = ids_of(6, 60), ids_of(9, 61), ids_of(7, 62)
        gen = [5, 6]  # what a template adds for the model's turn: the turn start comes after the first of the two
        full1, start1 = system + user1 + gen, len(system + user1) + 1
        (reply1, r1), = finish(server, ask(server, full1, start1, reply_room, Picker(seed=1)))
        assert reply1 == fresh(small_e, full1, reply_room, Picker(seed=1)) and r1["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        # the next turn: the template writes the earlier reply in another form (here: other tokens), then the new message
        full2 = full1[:start1] + ids_of(5, 63) + user2 + gen
        other = ask(server, ids_of(20, 64), 19, 10, Picker())  # another conversation at the same time
        (reply2, r2), _ = finish(server, ask(server, full2, len(full2) - 1, 15, Picker(seed=2)), other)
        assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] == start1
        assert reply2 == fresh(small_e, full2, 15, Picker(seed=2))
        assert len(server.kept) == 2 and not server.active


def test_more_conversations_than_sequences_take_the_oldest_kept_ones_place(small_e):  # noqa: F811
    eng = small_e.engine(sequences=2)
    server = Server(None, eng, None, STOP)
    prompts = [ids_of(8 + i, 70 + i) for i in range(5)]
    reqs = [ask(server, p, len(p) - 1, 8, Picker()) for p in prompts]  # five at once on two sequences: three wait
    for (tokens, result), p in zip(finish(server, *reqs), prompts):
        assert tokens == fresh(small_e, p, 8, Picker()) and result["finish_reason"] in ("stop", "length")
    assert len(server.kept) == 2 and server.totals["requests"] == 5


def test_kept_conversations_give_way_when_keys_and_values_run_out(small_e):  # noqa: F811
    cfg = small_e.cfg
    ring = {t: E.ring_tiles(cfg, t, 96) for t in (M.SLIDING, M.FULL)}
    eng = small_e.engine(sequences=4, pool={M.SLIDING: ring[M.SLIDING] + 1, M.FULL: ring[M.FULL] + 1})  # one long conversation's worth
    server = Server(None, eng, None, STOP)
    a, b = ids_of(30, 80), ids_of(34, 81)
    (first, _), = finish(server, ask(server, a, 29, 20, Picker()))
    assert first == fresh(small_e, a, 20, Picker()) and len(server.kept) == 1
    (second, result), = finish(server, ask(server, b, 33, 20, Picker()))  # needs the tiles the kept one holds
    assert second == fresh(small_e, b, 20, Picker()) and result["ended"] in ("stop", "length")
    assert len(server.kept) == 1


def test_a_cancelled_request_frees_its_sequence(small_e):  # noqa: F811
    eng = small_e.engine(sequences=2)
    server = Server(None, eng, None, STOP)
    req = ask(server, ids_of(40, 90), 39, 30, Picker())
    for _ in range(3):
        server.turn(wait=False)
    req.cancelled = True
    finish(server, req)
    assert not server.active and not server.kept and sorted(server.unused) == [0, 1] and eng.free_tiles() == eng.pool


def test_a_reply_is_read_in_ahead_as_a_later_turn_writes_it(small_e, drafter_e):  # noqa: F811
    for reply_room in (12, 44):
        eng = small_e.engine(sequences=2)
        server = Server(None, eng, engine_drafter(eng, drafter_e), STOP)
        written = ids_of(9, 95)  # the earlier reply in a later turn's form
        server.canon = lambda req, tokens: req.full[: req.start] + written
        full1 = ids_of(15, 96) + [5, 6]
        start1 = len(full1) - 1
        (reply1, _), = finish(server, ask(server, full1, start1, reply_room, Picker(seed=1)))
        kept = server.kept[0]
        assert kept.job.done == "read" and kept.now(eng) == full1[:start1] + written  # read in while nothing else ran
        full2 = full1[:start1] + written + ids_of(7, 97) + [5, 6]
        (reply2, r2), = finish(server, ask(server, full2, len(full2) - 1, 15, Picker(seed=2)))
        assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] == start1 + len(written)
        assert reply2 == fresh(small_e, full2, 15, Picker(seed=2))
        # a client that sends the reply back in another form: the sequence goes back to the turn's start
        full3 = full2[: len(full2) - 1] + ids_of(4, 98) + ids_of(6, 99) + [5, 6]
        (reply3, r3), = finish(server, ask(server, full3, len(full3) - 1, 15, Picker(seed=3)))
        assert r3["usage"]["prompt_tokens_details"]["cached_tokens"] == len(full2) - 1
        assert reply3 == fresh(small_e, full3, 15, Picker(seed=3))


def test_the_next_turn_may_come_before_the_reading_ahead_is_done(small_e):  # noqa: F811
    eng = small_e.engine(sequences=2)
    server = Server(None, eng, None, STOP)
    written = ids_of(30, 100)
    server.canon = lambda req, tokens: req.full[: req.start] + written
    full1 = ids_of(10, 101) + [5, 6]
    req = ask(server, full1, len(full1) - 1, 6, Picker())
    while req.job is None or req.job.done is None:
        server.turn(wait=False)
    server.turn(wait=False)  # one pass of the reading ahead: 8 of its 30 tokens
    assert 0 < server.kept[0].job.read < 30
    full2 = full1[: len(full1) - 1] + written + ids_of(5, 102) + [5, 6]
    (reply2, r2), = finish(server, ask(server, full2, len(full2) - 1, 12, Picker(seed=4)))
    assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] == len(full1) - 1 + 8
    assert reply2 == fresh(small_e, full2, 12, Picker(seed=4))


def test_conversations_wait_in_ram_when_the_gpu_is_short_and_come_back(small_e):  # noqa: F811
    cfg = small_e.cfg
    ring = {t: E.ring_tiles(cfg, t, 96) for t in (M.SLIDING, M.FULL)}
    # two sequences and keys and values for about one long conversation: the others have to leave the GPU
    eng = small_e.engine(sequences=2, pool={M.SLIDING: ring[M.SLIDING] + 2, M.FULL: ring[M.FULL] + 2})
    server = Server(None, eng, None, STOP, ram=1 << 30)
    firsts = [ids_of(22 + 3 * i, 130 + i) + [5, 6] for i in range(4)]
    replies = []
    for full in firsts:  # four conversations, one after the other
        (reply, _), = finish(server, ask(server, full, len(full) - 1, 10, Picker()))
        assert reply == fresh(small_e, full, 10, Picker())
        replies.append(reply)
    assert len(server.kept) + len(server.parked) == 4 and len(server.parked) >= 2
    for i in (0, 2, 1, 3):  # each one's next turn: all of it but the new message is found again
        full2 = firsts[i][: len(firsts[i]) - 1] + ids_of(6, 140 + i) + [5, 6]
        (reply2, r2), = finish(server, ask(server, full2, len(full2) - 1, 9, Picker(seed=i)))
        assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] == len(firsts[i]) - 1, i
        assert reply2 == fresh(small_e, full2, 9, Picker(seed=i)), i
    # with no RAM allowed they are forgotten instead, and the next turn starts anew
    eng = small_e.engine(sequences=2, pool={M.SLIDING: ring[M.SLIDING] + 2, M.FULL: ring[M.FULL] + 2})
    server = Server(None, eng, None, STOP)
    for full in firsts:
        finish(server, ask(server, full, len(full) - 1, 10, Picker()))
    full2 = firsts[0][: len(firsts[0]) - 1] + ids_of(6, 140) + [5, 6]
    (reply2, r2), = finish(server, ask(server, full2, len(full2) - 1, 9, Picker()))
    assert not server.parked and r2["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and reply2 == fresh(small_e, full2, 9, Picker())
