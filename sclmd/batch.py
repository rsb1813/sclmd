# Several texts at once on one engine: every pass of the main model carries rows of several sequences.
#
# A pass has 8, 16 or 32 rows. A text that is being written needs one (the token it settled last); a prompt that is
# still being read takes as many as it is given; rows left over carry the drafter's guesses, which the main model
# verifies as in generate.py. A wider pass costs more (COST), so the width is the one whose rows are expected to
# yield the most tokens for the time. Whatever the mix, a text's tokens are the ones it would get alone: the engine
# computes a row the same beside any others, and a choice depends only on (seed, position).
import torch

from . import engine as E
from .generate import Picker


# What a pass of up to 16, 24 or 32 rows costs next to one of up to 8 (scripts/pass_profile.py --rows 32 on the real
# model at 3 bits a weight: 21.7 ms for 8 rows, 27.0 for 16, 38.3 for 24, 42.4 for 32)
COST = ((8, 1.0), (16, 1.25), (24, 1.77), (32, 1.95))


def cost(rows: int) -> float:
    return next(c for most, c in COST if rows <= most)

# How often the guess at depth 1, 2, ... is kept: where the batcher starts from; it goes by what it sees after that
KEEP = (0.75, 0.56, 0.42, 0.32, 0.25, 0.20)


class Job:
    """One text on a sequence of the engine: the prompt's tokens that are still to be read, then the reply.

    start: the position the sequence should be able to come back to after the reply (a conversation's turn start:
    scripts/serve.py); the batcher copies the sliding layers' keys to RAM before the reply would write over what
    that position needs."""

    def __init__(self, seq: int, prompt: list, max_new: int, pick: Picker | None, stop: tuple = (), emit=None, start: int | None = None,
                 only_read: bool = False):
        self.seq, self.todo, self.max_new, self.pick, self.stop, self.emit, self.start = seq, list(prompt), max_new, pick, tuple(stop), emit, start
        self.only_read = only_read  # no reply: the job is over ("read") when its tokens are in, and takes only rows nothing else wants
        self.out = []  # the reply so far
        self.pending = None  # its last token: settled, not yet run
        self.hidden = None  # the main model's output at the position before `pending`
        self.done = None  # None while it runs; then "stop", "length", "read", "cancelled" or "full" (no memory left for it)
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
        self.make_room = None  # called when the pool of keys and values is empty: frees some and says so, or says no
        self.passes = self.rows = self.guess_rows = self.kept_rows = 0
        self.widths = {8: 0, 16: 0, 32: 0}  # passes by width
        self.sent = [20.0] * len(KEEP)  # guesses sent at each depth, and those kept (starting as if KEEP had been seen 20 times)
        self.held = [20.0 * k for k in KEEP]

    def add(self, job: Job):
        assert job.todo or job.pending is not None, "a job starts with a prompt"
        self.jobs.append(job)

    def _share(self, live: list) -> dict:
        """Rows of the next pass by job, at the width that is expected to yield the most tokens for its time."""
        at = self.turn % len(live)
        order = live[at:] + live[:at]
        keep = [h / s for h, s in zip(self.held, self.sent)]
        best = None
        for width in (8, 16, 32):
            if width > self.eng.rows:
                break
            rows = self._fill(order, width)
            total = sum(rows.values())
            gain = sum(n if j.pending is None else 1 + sum(keep[: n - 1]) for j, n in rows.items())
            rate = gain / cost(total)
            if best is None or rate > 1.03 * best[0]:  # a wider pass has to be clearly worth it
                best = (rate, rows)
        return best[1]

    def _fill(self, order: list, width: int) -> dict:
        """Rows of a pass of `width` rows by job. Every text being written gets one (when there are more than fit, in
        turn); the prompts being read share what is left; then guesses, one more for each writer in turn."""
        eng, live = self.eng, order
        writers = [j for j in order if j.pending is not None]
        rows, free = {}, width
        for j in writers[:free]:
            rows[j] = 1
            free -= 1
        room = {j: eng.capacity - eng.lengths[j.seq] for j in live}
        # prompts someone is waiting on, then the reading that is done ahead of need
        for readers in ([j for j in order if j.pending is None and not j.only_read], [j for j in order if j.only_read]):
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
                if self.make_room is not None and self.make_room():
                    continue
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
        # the choices of all texts come off the GPU together: every row's first choice, and for the texts that sample
        # every row's largest logits (one wait for the pass, not one a text)
        top = max((j.pick.top_k for j in use if j.pick is not None and j.pick.temperature > 0), default=0)
        vals, idx = (t.cpu() for t in logits.float().topk(top, dim=-1)) if top else (None, None)
        first = logits.argmax(-1).cpu() if any(j.pick is not None and j.pick.temperature <= 0 for j in use) else None

        def choose(j: Job, a: int, b: int, position: int) -> list:
            """Job j's tokens at `position`.. from rows a..b-1 of the pass."""
            return first[a:b].tolist() if j.pick.temperature <= 0 else j.pick.draw(vals[a:b], idx[a:b], position)

        self.passes += 1
        total = sum(len(ids) for _, ids in parts)
        self.widths[8 if total <= 8 else 16 if total <= 16 else 32] += 1
        row = 0
        for j, (_, ids) in zip(use, parts):
            n, hn = len(ids), hidden[row : row + len(ids)]
            row += n
            j.passes, j.rows = j.passes + 1, j.rows + n
            self.rows += n
            at = eng.lengths[j.seq]  # after the pass: the position after the rows
            if j.pending is None:  # a prompt: when its last token is in, the reply's first token is chosen
                j.todo, j.read = j.todo[n:], j.read + n
                if not j.todo and j.only_read:
                    j.done = "read"
                elif not j.todo:
                    j.hidden = hn[-1].clone()
                    j._settle(choose(j, row - 1, row, at)[0])
                continue
            drafts = ids[1:]
            wants = choose(j, row - n, row, at - n + 1)  # the main model's own token at each position after the pending one
            kept = 0
            for d, want in zip(drafts, wants):
                if want != d or d in j.stop:
                    break
                j.out.append(d)
                if j.emit:
                    j.emit(d)
                kept += 1
            self.guess_rows, self.kept_rows = self.guess_rows + len(drafts), self.kept_rows + kept
            for k in range(len(drafts)):
                self.sent[k] += 1
                self.held[k] += k < kept
            if kept < len(drafts):
                eng.crop(at - n + 1 + kept, j.seq)
            if len(j.out) >= j.max_new:
                j.done = "length"
            else:
                j.hidden = hn[kept].clone()
                j._settle(wants[kept])  # a guessed stop token is not kept above: it comes back here and ends the text
        self.jobs = [j for j in self.jobs if j.done is None]
        return True

    def drop(self, job: Job):
        """Takes a job out between passes (its sequence keeps what it has read)."""
        if job.done is None:
            job.done = "cancelled"
        if job in self.jobs:
            self.jobs.remove(job)

    def run(self):
        """Until every job is done."""
        while self.step():
            pass
