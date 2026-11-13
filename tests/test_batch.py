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
    return [Picker(), Picker(temperature=1.0, top_k=64, top_p=0.95, seed=3), Picker(temperature=0.7, top_k=20, top_p=0.9, seed=11)]


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
    # room for one long text and one tile more: the second long one runs out, the first is not disturbed
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
