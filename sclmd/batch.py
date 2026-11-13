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
# model at 3 bits a weight: 21.7 ms for 8 rows, 26.6 for 16, 35.6 for 24, 37.8 for 32 at 64 positions; 24.4, 30.8,
# 41.6, 45.2 at 2,048)
COST = ((8, 1.0), (16, 1.25), (24, 1.67), (32, 1.80))


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
                 only_read: bool = False, base: list | None = None):
        self.seq, self.todo, self.max_new, self.pick, self.stop, self.emit, self.start = seq, list(prompt), max_new, pick, tuple(stop), emit, start
        self.base = list(base or []) + list(prompt)  # everything before the reply, from the sequence's first position (base: what it held already)
        self.resume = None  # after starting over: the token to go on with once everything is read again
        self.held = None  # after starting over: it waits until a job has ended (the count of ended jobs it was held at)
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
        self.starved = 0  # jobs that had to wait for keys and values in the last pass
        self.ended = self.restarts = 0  # jobs that have ended; times a job had to start over for lack of memory
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

        def read(readers: list, most: int) -> int:
            """Up to `most` rows shared among the readers; how many were given."""
            given = 0
            while given < most and any(rows.get(j, 0) < min(len(j.todo), room[j]) for j in readers):
                for j in readers:
                    if given < most and rows.get(j, 0) < min(len(j.todo), room[j]):
                        rows[j] = rows.get(j, 0) + 1
                        given += 1
            return given

        # prompts someone is waiting on; then the reading done ahead of need, a quarter of the pass at most while
        # texts are being written (their guesses come before the rest of it)
        ahead = [j for j in order if j.only_read]
        free -= read([j for j in order if j.pending is None and not j.only_read], free)
        free -= read(ahead, min(free, width // 4) if writers else free)
        if self.drafter is not None:
            able = lambda j: min(self.depth, j.max_new - len(j.out), room[j] - 1)  # noqa: E731
            while free and any(rows[j] - 1 < able(j) for j in writers if j in rows):
                for j in writers:
                    if free and j in rows and rows[j] - 1 < able(j):
                        rows[j] += 1
                        free -= 1
        read(ahead, free)
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
        self.ended += sum(j.done is not None for j in self.jobs)
        self.jobs = [j for j in self.jobs if j.done is None]
        if not self.jobs:
            return False
        live = [j for j in self.jobs if j.held is None or self.ended > j.held]  # in the order they came
        if not live:  # only jobs that wait for another to end, and none will: they try again
            for j in self.jobs:
                j.held = None
            live = list(self.jobs)
        rows = self._share(live)
        self.turn += 1
        # The tiles the pass needs. When the pool is short (and make_room frees nothing more), the jobs get theirs in
        # the order they came and the others wait for a pass in which there are some. When not one can go on, the
        # last to come that holds any gives them up and starts over once a job has ended (_start_over); a job that
        # is the only one holding any cannot be helped and ends ("full").
        while True:
            # make_room may have dropped a job (the reading ahead of a conversation it moved off the GPU): its sequence
            # is no longer its own, and it gets nothing in this pass
            live = [j for j in live if j.done is None]
            use, free = [], eng.free_tiles()
            for j in live:  # in the order they came
                if not rows.get(j):
                    continue
                need = eng.lacks(j.seq, eng.lengths[j.seq], rows[j])
                if any(need[t] > free[t] for t in need) and j.pending is not None and rows[j] > 1:
                    rows[j] = 1  # without its guesses it may need none
                    need = eng.lacks(j.seq, eng.lengths[j.seq], 1)
                if all(need[t] <= free[t] for t in need):
                    free = {t: free[t] - need[t] for t in need}
                    use.append(j)
            self.starved = sum(1 for j in live if rows.get(j)) - len(use)
            if not self.starved or self.make_room is None or not self.make_room():
                break
        if not live:  # every job that had rows was dropped
            return True
        if not use:
            holders = [j for j in self.jobs if any(x >= 0 for t in eng.ring for x in eng.tiles[t][j.seq])]
            if len(holders) > 1:
                self._start_over(holders[-1])
            else:
                victim = holders[-1] if holders else live[-1]
                victim.done = "full"
                self.jobs.remove(victim)
                self.ended += 1
            return True
        eng._provide([(j.seq, eng.lengths[j.seq], rows[j]) for j in use])
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
                elif not j.todo and j.resume is not None:  # started over, and back where it was
                    j.hidden, j.pending, j.resume = hn[-1].clone(), j.resume, None
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

    def _start_over(self, job: Job):
        """The job gives its sequence's memory up. Later it reads everything again, its own reply so far included,
        and goes on with the token it had settled last: the same text as if it had never stopped."""
        self.eng.release(job.seq)
        job.todo = job.base + job.out[:-1] if job.pending is not None or job.resume is not None else list(job.base)
        if job.out and not job.only_read:
            job.resume = job.out[-1]
        job.pending, job.hidden, job.snapped, job.held = None, None, False, self.ended
        self.restarts += 1

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
