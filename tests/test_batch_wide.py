# The batcher's passes of 64, 128 and 256 rows (sclmd/batch.py on an engine of that many rows; scripts/serve.py
# --wide-rows): several prompts are read in one pass, and the rows whose logits are looked at (the texts being
# written, their guesses, a prompt's last row) are where a wide pass has logits, its last 32 rows at most. Whatever
# the mix, every text is the one generate() gives alone.
import os
import random
from types import SimpleNamespace

import pytest
import torch

from sclmd import batch as B
from sclmd import engine as E
from sclmd.batch import Batcher, Job
from sclmd.generate import Picker, generate

GPU = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.exists(E.DLL), reason="needs a GPU and the engine")
STOP = (7,)
CAP = 336  # positions: prompts of up to 150 tokens and their replies

if torch.cuda.is_available() and os.path.exists(E.DLL):
    from test_engine import engine_drafter, ids_of
    from test_engine import drafter_e, small_e  # noqa: F401  (fixtures)


def hot(seed: int) -> Picker:
    return Picker(temperature=5.0, top_k=16, top_p=1.0, seed=seed)  # a wrong context shows in the text (tests/test_serve.py)


def laid_out(rows: dict) -> tuple:
    """(rows of the pass, the first row whose logits are looked at) as Batcher.step lays a pass out: the quiet rows,
    then the prompts that end in it (the one with the most rows first, of which only the last row's), the writers."""
    ends = sorted((j for j in rows if Batcher._ends(j, rows[j])), key=lambda j: -rows[j])
    quiet = sum(n for j, n in rows.items() if j.pending is None and j not in ends)
    return sum(rows.values()), quiet + (rows[ends[0]] - 1 if ends else 0)


def test_wide_passes_are_fitted_to_the_rows_they_have_logits_of():
    # no GPU: _fit alone, on random mixes of writers (with guesses), prompts that end in the pass, prompts that do
    # not, and reading ahead, in passes of 33 to 256 rows
    batch = Batcher(SimpleNamespace(rows=256, int8=0.0))
    rng = random.Random(0)
    cut = deferred = 0
    for case in range(3000):
        rows = {}
        for seq in range(rng.randrange(1, 40)):
            kind = rng.random()
            if kind < 0.4:  # a writer: its token and up to 6 guesses
                j = Job(seq, [5], 50, Picker())
                j.todo, j.pending = [], 9
                rows[j] = 1 + rng.randrange(7)
            elif kind < 0.6:  # a prompt that ends in the pass
                j = Job(seq, [5] * rng.randrange(1, 120), 10, Picker())
                rows[j] = len(j.todo)
            elif kind < 0.9:  # a prompt that does not
                j = Job(seq, [5] * rng.randrange(2, 300), 10, Picker())
                rows[j] = rng.randrange(1, len(j.todo))
            else:  # reading ahead (no logits ever)
                j = Job(seq, [5] * rng.randrange(1, 100), 0, None, only_read=True)
                rows[j] = rng.randrange(1, len(j.todo) + 1)
        while sum(rows.values()) > 256:
            rows.pop(next(iter(rows)))
        if sum(rows.values()) <= E.ATT_ROWS:
            continue
        got = batch._fit(rows)
        total, first = laid_out(got)
        assert total <= E.ATT_ROWS or E.logits_first(total, first) <= first, (case, total, first)  # the pass has its logits
        for j, n in got.items():  # rows are only taken away (what does not end stays so); a prompt's end may be put off
            assert 0 < n <= rows[j]
            if not Batcher._ends(j, rows[j]):
                assert not Batcher._ends(j, n)
            deferred += Batcher._ends(j, rows[j]) and not Batcher._ends(j, n)
        assert sum(n for j, n in got.items() if j.pending is not None) <= E.ATT_ROWS
        cut += sum(got.values()) < sum(rows.values())
    assert cut > 100 and deferred > 100  # the test is not empty


def test_a_wide_pass_holds_64_sequences_and_32_writers_at_most():
    # no GPU: more sequences than a pass has segments (E.MAX_SEGS), and more writers than its logits' rows
    seqs = 120
    eng = SimpleNamespace(rows=256, int8=0.0, capacity=4096, lengths=[0] * seqs)
    batch = Batcher(eng)
    jobs = [Job(s, [5] * (3 + s % 5), 10, Picker()) for s in range(seqs)]
    for j in jobs[:40]:  # forty texts being written
        j.todo, j.pending = [], 9
    for width in (64, 128, 256):
        rows = batch._fill(jobs, width)
        assert len(rows) <= E.MAX_SEGS and sum(rows.values()) <= width, (width, len(rows))
        assert sum(j.pending is not None for j in rows) <= E.ATT_ROWS
        total, first = laid_out(rows)
        assert total <= E.ATT_ROWS or E.logits_first(total, first) <= first
    assert batch._unread(jobs, batch._fill(jobs, 32))  # 32 rows leave prompts unread: the wide widths are looked at
    assert len(batch._share(jobs)) <= E.MAX_SEGS


def test_the_cost_of_a_wide_pass_goes_on_from_32_rows():
    for int8 in (False, True):
        costs = [B.cost(n, int8) for n in range(1, 257)]
        assert costs[31] == B.COST[-1][1] and costs[32] > costs[31]
        assert all(b >= a for a, b in zip(costs, costs[1:]))
        assert all(B.cost(n, int8) / n < B.cost(32, int8) / 32 for n in (64, 128, 256))  # a row costs less in a wider pass
    assert Batcher(SimpleNamespace(rows=32, int8=0.0)).choices == (8, 16, 32)  # the default: as before
    assert Batcher(SimpleNamespace(rows=16, int8=0.0)).choices == (8, 16)
    assert Batcher(SimpleNamespace(rows=128, int8=0.0)).choices == (8, 16, 32, 64, 128)
    assert Batcher(SimpleNamespace(rows=256, int8=0.0), widths=(8, 16, 32, 256)).choices == (8, 16, 32, 256)


class Tokens:
    """No GPU, no model: a chat template that writes a message's content as its characters and the reply's turn as
    two tokens (scripts/serve.py App.request)."""

    def encode(self, text, add_special_tokens=False):
        return [90, 91]

    def convert_tokens_to_ids(self, token):
        return 1

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, enable_thinking=False):
        return [ord(c) for m in messages for c in m["content"]] + ([90, 91] if add_generation_prompt else [])


def test_a_wide_servers_requests_keep_the_margin_of_its_narrow_passes():
    # no GPU: a --rows 16 --wide-rows 64 server keeps 16 rows below the context, as a --rows 16 one does, not 32
    from scripts.serve import App, Refused, Server, narrow

    for rows, widths, margin in ((16, (8, 16), 16), (64, (8, 16, 64), 16), (128, None, 32), (32, None, 32), (8, None, 8)):
        eng = SimpleNamespace(rows=rows, int8=0.0, sequences=2, capacity=200, lengths=[0, 0])
        server = Server(None, eng, None, STOP, widths=widths)
        assert narrow(eng) == margin, (rows, widths)
        app = App(server, Tokens(), False, 100, {"bits_per_weight": 3.0})
        ok = app.request({"messages": [{"role": "user", "content": "x" * (200 - margin - 16 - 2)}]})  # 16 tokens of room
        assert ok.max_new == 16, (rows, widths)
        with pytest.raises(Refused):
            app.request({"messages": [{"role": "user", "content": "x" * (200 - margin - 16 - 1)}]})
    assert narrow(SimpleNamespace(rows=256)) == E.ATT_ROWS  # an engine no Server has (stand-ins): its rows, 32 at most


def _passes(eng: E.Engine) -> list:
    """Every pass the engine runs from now on: (rows, the first row whose logits are asked for, each sequence's rows)."""
    seen, run = [], eng.run

    def watched(parts, logits_from=0):
        seen.append((sum(len(ids) for _, ids in parts), logits_from, [len(ids) for _, ids in parts]))
        return run(parts, logits_from)

    eng.run = watched
    return seen


@GPU
@pytest.mark.parametrize("int8", [0.0, E.INT8_SIGMA], ids=["fp16", "int8"])
def test_texts_written_beside_wide_passes_are_the_texts_written_alone(small_e, drafter_e, int8):  # noqa: F811
    # prompts of 3 to 150 tokens, the later ones arriving while the first are written; greedy and hot samplers
    sizes = (40, 5, 150, 70, 9, 100, 33, 64, 3, 50, 120, 20, 90, 31)
    asks = [(ids_of(n, 300 + i), 12 + 4 * (i % 5), hot(i) if i % 3 else Picker()) for i, n in enumerate(sizes)]
    want = [generate(small_e.engine(capacity=CAP, int8=int8), p, m, pick, stop=STOP) for p, m, pick in asks]
    for rows, with_drafter in ((64, True), (256, False), (256, True)):
        eng = small_e.engine(capacity=CAP, sequences=len(asks), rows=rows, int8=int8)
        batch = Batcher(eng, engine_drafter(eng, drafter_e) if with_drafter else None)
        passes = _passes(eng)
        jobs = [Job(seq, p, m, pick, STOP) for seq, (p, m, pick) in enumerate(asks)]
        paused = []  # a prompt that stops for a moment (scripts/serve.py keeps a shared prefix there)
        jobs[2].pause, jobs[2].at_pause = 37, lambda j: paused.append(eng.lengths[j.seq])
        for j in jobs[:8]:
            batch.add(j)
        for _ in range(3):
            batch.step()
        for j in jobs[8:]:
            batch.add(j)
        batch.run()
        assert [j.out for j in jobs] == want, (rows, with_drafter)
        assert all(j.done in ("stop", "length") for j in jobs) and paused == [37]
        wide = [p for p in passes if p[0] > E.ATT_ROWS]
        # the test is not empty: wide passes, counted by width, some beside texts being written (logits from a row
        # inside the pass), some reading several prompts
        assert wide and sum(n for w, n in batch.widths.items() if w > E.ATT_ROWS) == len(wide), batch.widths
        assert any(0 < p[1] < p[0] for p in wide) and any(sum(n > 8 for n in p[2]) > 1 for p in wide), wide
        if with_drafter:
            assert batch.guess_rows > 0


@GPU
def test_wide_passes_when_keys_and_values_run_short(small_e, drafter_e):  # noqa: F811
    # room for about two of the texts at a time: texts wait for memory, give theirs up and read everything again
    # (their reply so far included) in wide passes; a wide pass is fitted again when some of its jobs have to wait
    from sclmd import model as M

    texts = [ids_of(n, 500 + i) for i, n in enumerate((90, 60, 110, 45, 75))]
    picks = [hot(30 + i) for i in range(len(texts))]
    want = [generate(small_e.engine(capacity=CAP), t, 30, pick, stop=()) for t, pick in zip(texts, picks)]
    for with_drafter in (False, True):
        eng = small_e.engine(capacity=CAP, sequences=len(texts), rows=128, pool={M.SLIDING: 17, M.FULL: 17})
        batch = Batcher(eng, engine_drafter(eng, drafter_e) if with_drafter else None)
        passes = _passes(eng)
        seen = [[] for _ in texts]
        jobs = [Job(seq, t, 30, pick, (), emit=seen[seq].append) for seq, (t, pick) in enumerate(zip(texts, picks))]
        freed, waited = [], 0

        def make_room() -> bool:  # what the server does: a text that is over gives its memory up
            for j in jobs:
                if j.done is not None and j.seq not in freed:
                    eng.release(j.seq)
                    freed.append(j.seq)
                    return True
            return False

        batch.make_room = make_room
        for j in jobs:
            batch.add(j)
        while batch.step():
            waited += batch.starved > 0
        assert [j.out for j in jobs] == want and seen == want, with_drafter  # and nothing was said twice
        assert waited > 0 and batch.restarts > 0 and any(p[0] > E.ATT_ROWS for p in passes), (waited, batch.restarts)


@GPU
def test_a_server_with_wide_passes_answers_as_a_fresh_engine(small_e, drafter_e):  # noqa: F811
    from scripts.serve import Request, Server

    eng = small_e.engine(capacity=CAP, sequences=6, rows=128)
    server = Server(None, eng, engine_drafter(eng, drafter_e), STOP, widths=(8, 16, 32, 64, 128))
    passes = _passes(eng)
    prompts = [ids_of(n, 400 + i) for i, n in enumerate((60, 90, 12, 140, 45))]
    reqs = [Request(p, len(p) - 1, 16, hot(20 + i), False) for i, p in enumerate(prompts)]
    for req in reqs:
        server.submit(req)
    while server.turn(wait=False):
        pass
    for i, (req, p) in enumerate(zip(reqs, prompts)):
        tokens = []
        while not req.events.empty():
            event = req.events.get()
            if event[0] == "token":
                tokens.append(event[1])
        assert tokens == generate(small_e.engine(capacity=CAP), p, 16, hot(20 + i), stop=STOP), i
    assert any(p[0] > E.ATT_ROWS for p in passes) and server.about()["passes_by_width"][128] > 0
