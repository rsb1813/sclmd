# Several texts at once (sclmd/batch.py) on the small random model: whatever the mix of sequences in the passes, each
# text is the one generate() gives alone.
import pytest
import torch

from sclmd import engine as E
from sclmd import model as M
from sclmd.batch import Batcher, Job
from sclmd.generate import Picker, generate
from test_engine import Small, engine_drafter, ids_of
from test_engine import drafter_e, small_e  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not __import__("os").path.exists(E.DLL), reason="needs a GPU and the engine")

STOP = (7,)


def pickers():
    return [Picker(), Picker(temperature=5.0, top_k=16, top_p=1.0, seed=3), Picker(temperature=3.0, top_k=20, top_p=0.98, seed=11)]  # hot: a wrong context shows in the text (tests/test_serve.py: hot)


def alone(small: Small, prompt: list, max_new: int, pick: Picker) -> list:
    return generate(small.engine(), prompt, max_new, pick, stop=STOP)


def test_texts_written_side_by_side_are_the_texts_written_alone(small_e, drafter_e):  # noqa: F811
    asks = [(ids_of(5 + 4 * i, 40 + i), 20 + 5 * (i % 4), pickers()[i % 3]) for i in range(11)]  # more texts than a pass has rows
    want = [alone(small_e, prompt, max_new, pick) for prompt, max_new, pick in asks]
    for with_drafter in (False, True):
        for rows in (8, 32):
            eng = small_e.engine(sequences=len(asks), rows=rows)
            batch = Batcher(eng, engine_drafter(eng, drafter_e) if with_drafter else None)
            jobs = [Job(seq, prompt, max_new, pick, STOP) for seq, (prompt, max_new, pick) in enumerate(asks)]
            for j in jobs[:6]:
                batch.add(j)
            for _ in range(5):  # the rest arrive while the first are under way
                batch.step()
            for j in jobs[6:]:
                batch.add(j)
            batch.run()
            assert [j.out for j in jobs] == want, (with_drafter, rows)
            assert all(j.done in ("stop", "length") for j in jobs)
            if with_drafter:
                assert batch.guess_rows > 0
            if rows == 32:
                assert batch.widths[16] + batch.widths[32] > 0  # eleven texts: passes wider than eight rows were worth it


def test_a_text_alone_gets_the_drafters_guesses_and_emits_in_order(small_e, drafter_e):  # noqa: F811
    prompt, pick = ids_of(9, 50), Picker()
    want = alone(small_e, prompt, 40, pick)
    eng = small_e.engine()
    seen = []
    job = Job(0, prompt, 40, pick, STOP, emit=seen.append)
    batch = Batcher(eng, engine_drafter(eng, drafter_e))
    batch.add(job)
    batch.run()
    assert job.out == want and seen == want
    assert batch.rows > batch.passes  # passes carried guesses beside the pending token


def test_cancelled_and_out_of_memory_texts_end_and_the_others_go_on(small_e):  # noqa: F811
    cfg = small_e.cfg
    ring = {t: E.ring_tiles(cfg, t, 96) for t in (M.SLIDING, M.FULL)}
    a, b, c = ids_of(40, 51), ids_of(40, 52), ids_of(10, 53)
    want = alone(small_e, a, 30, Picker())
    # room for one long text and one tile more: the second long one cannot go on beside the first, and nothing frees
    # memory for it when the first is done either (its sequence keeps what it holds): it ends for lack of memory
    eng = small_e.engine(sequences=3, pool={M.SLIDING: ring[M.SLIDING] + 1, M.FULL: ring[M.FULL] + 1})
    batch = Batcher(eng)
    first, second, third = Job(0, a, 30, Picker(), STOP), Job(1, b, 30, Picker(), STOP), Job(2, c, 30, Picker(), STOP)
    batch.add(first)
    batch.add(second)
    for _ in range(4):
        batch.step()
    batch.add(third)
    third.cancel = True
    batch.run()
    assert first.out == want and first.done in ("stop", "length")
    assert second.done == "full" and third.done == "cancelled"


def test_texts_wait_for_keys_and_values_and_go_on_when_some_are_freed(small_e):  # noqa: F811
    cfg = small_e.cfg
    ring = {t: E.ring_tiles(cfg, t, 96) for t in (M.SLIDING, M.FULL)}
    texts = [ids_of(40, 57), ids_of(40, 58), ids_of(36, 59)]
    want = [alone(small_e, t, 24, Picker()) for t in texts]
    eng = small_e.engine(sequences=3, pool={M.SLIDING: ring[M.SLIDING] + 1, M.FULL: ring[M.FULL] + 1})  # one long text at a time
    batch = Batcher(eng)
    jobs = [Job(seq, t, 24, Picker(), STOP) for seq, t in enumerate(texts)]
    freed = []

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
    waited = 0
    while batch.step():
        waited += batch.starved > 0
    assert [j.out for j in jobs] == want and all(j.done in ("stop", "length") for j in jobs)
    assert waited > 0 and freed == [0, 1]


def test_a_text_that_starts_over_for_lack_of_memory_goes_on_with_the_same_tokens(small_e, drafter_e):  # noqa: F811
    cfg = small_e.cfg
    ring = {t: E.ring_tiles(cfg, t, 96) for t in (M.SLIDING, M.FULL)}
    texts = [ids_of(10, 150), ids_of(11, 151), ids_of(9, 152)]  # short prompts: the three are writing when memory runs out
    picks = [Picker(), Picker(temperature=5.0, top_k=16, top_p=1.0, seed=5), Picker(temperature=5.0, top_k=16, top_p=1.0, seed=6)]
    want = [generate(small_e.engine(), t, 30, pick, stop=()) for t, pick in zip(texts, picks)]
    for with_drafter in (False, True):
        eng = small_e.engine(sequences=3, pool={M.SLIDING: ring[M.SLIDING] + 1, M.FULL: ring[M.FULL] + 1})
        batch = Batcher(eng, engine_drafter(eng, drafter_e) if with_drafter else None)
        seen = [[] for _ in texts]
        jobs = [Job(seq, t, 30, pick, (), emit=seen[seq].append) for seq, (t, pick) in enumerate(zip(texts, picks))]
        freed = []

        def make_room() -> bool:
            for j in jobs:
                if j.done is not None and j.seq not in freed:
                    eng.release(j.seq)
                    freed.append(j.seq)
                    return True
            return False

        batch.make_room = make_room
        for j in jobs:
            batch.add(j)
        batch.run()
        assert batch.restarts > 0  # each held a tile and needed another: the last to come gave its up and began again
        assert [j.out for j in jobs] == want and seen == want  # and nothing was said twice


def test_a_turn_start_can_be_come_back_to_after_a_long_reply(small_e):  # noqa: F811
    prompt, more = ids_of(30, 54), ids_of(6, 55)
    solo = small_e.engine()
    solo.feed(prompt)
    want = solo.feed(more)[0]
    eng = small_e.engine(sequences=2)
    job = Job(1, prompt, 50, Picker(), (), start=30)  # 50 tokens: past what the sliding ring keeps of the turn start
    other = Job(0, ids_of(12, 56), 20, Picker(), ())
    batch = Batcher(eng)
    batch.add(other)
    batch.add(job)
    batch.run()
    assert job.snapped and eng.lengths[1] == 30 + 49  # the last token of a reply is not run
    eng.restore(1)
    eng.crop(30, 1)
    assert torch.equal(eng.feed(more, seq=1)[0], want)
