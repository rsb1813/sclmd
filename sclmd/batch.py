# Several texts at once on one engine: every pass of the main model carries rows of several sequences.
#
# A pass has eight rows. A text that is being written needs one (the token it settled last); a prompt that is still
# being read takes as many as it is given; rows left over carry the drafter's guesses, which the main model verifies
# as in generate.py. Whatever the mix, a text's tokens are the ones it would get alone: the engine computes a row the
# same beside any others, and a choice depends only on (seed, position).
import torch

from . import engine as E
from .generate import Picker


class Job:
    """One text on a sequence of the engine: the prompt's tokens that are still to be read, then the reply.

    start: the position the sequence should be able to come back to after the reply (a conversation's turn start:
    scripts/serve.py); the batcher copies the sliding layers' keys to RAM before the reply would write over what
    that position needs."""

    def __init__(self, seq: int, prompt: list, max_new: int, pick: Picker, stop: tuple = (), emit=None, start: int | None = None):
        self.seq, self.todo, self.max_new, self.pick, self.stop, self.emit, self.start = seq, list(prompt), max_new, pick, tuple(stop), emit, start
        self.out = []  # the reply so far
        self.pending = None  # its last token: settled, not yet run
        self.hidden = None  # the main model's output at the position before `pending`
        self.done = None  # None while it runs; then "stop", "length", "cancelled" or "full" (no memory left for it)
        self.cancel = False  # set from outside: the text ends at the next pass
        self.snapped = False
        self.read = self.passes = self.rows = 0  # prompt tokens read here, passes it had rows in, rows it was given

    def _settle(self, token: int):
        self.pending = token
        self.out.append(token)
        if self.emit:
            self.emit(token)
        if token in self.stop:
            self.done = "stop"
        elif len(self.out) >= self.max_new:
            self.done = "length"


class Batcher:
    def __init__(self, eng: E.Engine, drafter: E.EngineDrafter | None = None, depth: int = 6):
        self.eng, self.drafter, self.depth = eng, drafter, depth
        self.jobs = []
        self.turn = 0
        self.passes = self.rows = self.guess_rows = self.kept_rows = 0

    def add(self, job: Job):
        assert job.todo or job.pending is not None, "a job starts with a prompt"
        self.jobs.append(job)

    def _share(self, live: list) -> dict:
        """Rows of the next pass by job. Every text being written gets one (when there are more than eight, in turn);
        the prompts being read share what is left; then guesses, one more for each writer in turn."""
        eng = self.eng
        at = self.turn % len(live)
        order = live[at:] + live[:at]
        writers, readers = [j for j in order if j.pending is not None], [j for j in order if j.pending is None]
        rows, free = {}, E.MAX_TOKENS
        for j in writers[:free]:
            rows[j] = 1
            free -= 1
        room = {j: eng.capacity - eng.lengths[j.seq] for j in live}
        while free and any(rows.get(j, 0) < min(len(j.todo), room[j]) for j in readers):
            for j in readers:
                if free and rows.get(j, 0) < min(len(j.todo), room[j]):
                    rows[j] = rows.get(j, 0) + 1
                    free -= 1
        if self.drafter is not None:
            able = lambda j: min(self.depth, j.max_new - len(j.out), room[j] - 1)  # noqa: E731
            while free and any(rows[j] - 1 < able(j) for j in writers if j in rows):
                for j in writers:
                    if free and j in rows and rows[j] - 1 < able(j):
                        rows[j] += 1
                        free -= 1
        return rows

    @torch.no_grad()
    def step(self) -> bool:
        """One pass. False: no job had anything to run."""
        eng = self.eng
        for j in self.jobs:
            if j.done is None and j.cancel:
                j.done = "cancelled"
            if j.done is None and eng.lengths[j.seq] >= eng.capacity:
                j.done = "length"  # the sequence is as long as it can be
        self.jobs = [j for j in self.jobs if j.done is None]
        live = list(self.jobs)
        if not live:
            return False
        rows = self._share(live)
        self.turn += 1
        # the tiles the pass needs: when the pool is short, the jobs that came last wait
        while True:
            use = [j for j in live if rows.get(j)]
            try:
                eng._provide([(j.seq, eng.lengths[j.seq], rows[j]) for j in use])
                break
            except E.KVFull:
                if len(use) == 1:  # nothing else holds what it lacks back: it cannot go on
                    use[0].done = "full"
                    self.jobs.remove(use[0])
                    return True
                rows.pop(use[-1])
        # a turn start that the reply is about to write over is copied to RAM first
        for j in use:
            if j.start is not None and not j.snapped:
                limit = eng.kept() + max(j.start - eng.cfg.window + 1, 0)
                if max(eng.highs[j.seq], eng.lengths[j.seq] + rows[j]) > limit:
                    eng.snapshot(j.seq)
                    j.snapped = True
        guesses = {}
        asks = [(j, rows[j] - 1) for j in use if j.pending is not None and rows[j] > 1]
        if asks:
            got = self.drafter.guess_many([(j.pending, j.hidden, eng.lengths[j.seq], j.seq, k) for j, k in asks])
            guesses = {j: g for (j, _), g in zip(asks, got)}
        parts = []
        for j in use:
            if j.pending is None:
                parts.append((j.seq, j.todo[: rows[j]]))
            else:
                parts.append((j.seq, [j.pending] + guesses.get(j, [])))
        logits, hidden = eng.run(parts)
        self.passes += 1
        row = 0
        for j, (_, ids) in zip(use, parts):
            n, lg, hn = len(ids), logits[row : row + len(ids)], hidden[row : row + len(ids)]
            row += n
            j.passes, j.rows = j.passes + 1, j.rows + n
            self.rows += n
            at = eng.lengths[j.seq]  # after the pass: the position after the rows
            if j.pending is None:  # a prompt: when its last token is in, the reply's first token is chosen
                j.todo, j.read = j.todo[n:], j.read + n
                if not j.todo:
                    j.hidden = hn[-1].clone()
                    j._settle(j.pick(lg[-1], at))
                continue
            drafts = ids[1:]
            wants = j.pick.many(lg, at - n + 1)  # the main model's own token at each position after the pending one
            kept = 0
            for d, want in zip(drafts, wants):
                if want != d or d in j.stop:
                    break
                j.out.append(d)
                if j.emit:
                    j.emit(d)
                kept += 1
            self.guess_rows, self.kept_rows = self.guess_rows + len(drafts), self.kept_rows + kept
            if kept < len(drafts):
                eng.crop(at - n + 1 + kept, j.seq)
            if len(j.out) >= j.max_new:
                j.done = "length"
            else:
                j.hidden = hn[kept].clone()
                j._settle(wants[kept])  # a guessed stop token is not kept above: it comes back here and ends the text
        self.jobs = [j for j in self.jobs if j.done is None]
        return True

    def run(self):
        """Until every job is done."""
        while self.step():
            pass
