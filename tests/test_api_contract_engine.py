# The server's conversations (scripts/serve.py: Server) when a client's stop string ends a reply, on the small random
# model (GPU): the conversation is kept with the answer as cut, and its next turn goes on from it.
import pytest
import torch

from scripts.serve import Request, Server
from sclmd import engine as E
from sclmd.generate import Picker, generate
from test_engine import engine_drafter, ids_of
from test_engine import drafter_e, small_e  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not __import__("os").path.exists(E.DLL), reason="needs a GPU and the engine")
STOP = (7,)


def hot(seed: int = 0) -> Picker:
    return Picker(temperature=5.0, top_k=16, top_p=1.0, seed=seed)  # as tests/test_serve.py: the text depends on what came before


def fresh(small, full: list, max_new: int, pick: Picker) -> list:
    return generate(small.engine(), full, max_new, pick, stop=STOP)


def events(req) -> tuple:
    tokens, result = [], None
    while not req.events.empty():
        event = req.events.get()
        if event[0] == "token":
            tokens.append(event[1])
        else:
            result = event[1]
    return tokens, result


def finish(server: Server):
    while server.turn(wait=False):
        pass


def cut_by_a_stop_string(server: Server, full: list, start: int, max_new: int, pick: Picker, after: int, start_over: bool = False) -> Request:
    """A request whose answer a stop string cuts after `after` tokens, as the HTTP thread does it: the answer as cut
    first, then cancelled. start_over: the job had to give its memory up just before (Batcher._start_over)."""
    req = Request(full, start, max_new, pick, False)
    server.submit(req)
    while req.job is None or (len(req.job.out) < after and req.job.done is None):
        server.turn(wait=False)
    assert req.job.done is None, "the reply ended by itself before the stop string (another seed)"
    if start_over:
        server.batch._start_over(req.job)
    req.stop_text = "the answer as cut"
    req.cancelled = True
    return req


def test_a_conversation_that_a_stop_string_ended_is_kept_and_goes_on(small_e, drafter_e):  # noqa: F811
    for reply_room in (12, 44):  # 44: past what the sliding ring keeps of the turn start
        eng = small_e.engine(sequences=3)
        server = Server(None, eng, engine_drafter(eng, drafter_e), STOP)
        written = ids_of(6, 200)  # the cut answer as a later turn writes it
        cuts = []

        def canon(req, tokens):
            cuts.append(req.stop_text)
            return req.full[: req.start] + written

        server.canon = canon
        full1 = ids_of(30, 201) + [5, 6]
        start1 = len(full1) - 1
        after = reply_room - 4  # 40: the turn start has gone to RAM by then
        seed = next(s for s in range(1, 50) if STOP[0] not in fresh(small_e, full1, after, hot(s)))  # a reply that does not end by itself first
        req = cut_by_a_stop_string(server, full1, start1, reply_room, hot(seed), after=after)
        finish(server)
        tokens, result = events(req)
        assert result["finish_reason"] == "stop" and result["ended"] == "cancelled"
        assert cuts == ["the answer as cut"] and len(server.kept) == 1  # before: released, nothing kept
        kept = next(iter(server.kept.values()))
        assert kept.later == full1[:start1] + written and kept.job.done == "read"
        full2 = full1[:start1] + written + ids_of(7, 202) + [5, 6]
        req2 = Request(full2, len(full2) - 1, 15, hot(2), False)
        server.submit(req2)
        finish(server)
        reply2, r2 = events(req2)
        assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] == start1 + len(written)  # all but the new message
        assert reply2 == fresh(small_e, full2, 15, hot(2))


def test_a_stop_string_while_the_job_reads_everything_again_keeps_nothing(small_e):  # noqa: F811
    eng = small_e.engine(sequences=2)
    server = Server(None, eng, None, STOP)
    full1 = ids_of(30, 210) + [5, 6]
    req = cut_by_a_stop_string(server, full1, len(full1) - 1, 30, hot(3), after=3, start_over=True)
    finish(server)
    assert not server.kept and sorted(server.unused) == [0, 1] and eng.free_tiles() == eng.pool
    full2 = full1[: len(full1) - 1] + ids_of(5, 211) + ids_of(4, 212) + [5, 6]  # its next turn: read from the start, right
    req2 = Request(full2, len(full2) - 1, 12, hot(4), False)
    server.submit(req2)
    finish(server)
    reply2, r2 = events(req2)
    assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and reply2 == fresh(small_e, full2, 12, hot(4))


def test_a_client_that_left_keeps_nothing_as_before(small_e):  # noqa: F811
    eng = small_e.engine(sequences=2)
    server = Server(None, eng, None, STOP)
    req = Request(ids_of(30, 220) + [5, 6], 30, 30, hot(), False)
    server.submit(req)
    while req.job is None or (len(req.job.out) < 3 and req.job.done is None):
        server.turn(wait=False)
    assert req.job.done is None
    req.cancelled = True  # no stop_text: the client left
    finish(server)
    assert not server.kept and eng.free_tiles() == eng.pool
