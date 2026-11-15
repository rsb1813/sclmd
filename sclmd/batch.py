# Several texts at once on one engine: every pass of the main model carries rows of several sequences.
#
# A pass has 8, 16 or 32 rows. A text that is being written needs one (the token it settled last); a prompt that is
# still being read takes as many as it is given; rows left over carry the drafter's guesses, which the main model
# verifies as in generate.py. A wider pass costs more (COST), so the width is the one whose rows are expected to
# yield the most tokens for the time. Whatever the mix, a text's tokens are the ones it would get alone: the engine
# computes a row the same beside any others, and a choice depends only on (seed, position).
#
# On an engine of 64, 128 or 256 rows (brief B2; scripts/serve.py --wide-rows) a pass may also have that many while
# prompts someone waits on are left that 32 rows do not take: such a pass has the logits of its last 32 rows at most
# (E.logits_first), so the rows whose logits are looked at (the texts being written and their guesses, a prompt's last
# row) are put at its end (_fit).
#
# With an Order (sclmd/order.py: scripts/serve.py --priorities; Batcher.order) the jobs get the tiles, the prompts'
# rows and the guesses of a pass by priority instead of as they came and in turn; the texts are the same.
import torch

from . import engine as E
from .generate import Picker


# What a pass of up to 16, 24 or 32 rows costs next to one of up to 8 (scripts/pass_profile.py --rows 32 on the real
# model at 3 bits a weight: 20.2 ms for 8 rows, 22.6 for 16, 30.0 for 24, 31.3 for 32 at 64 positions; 22.0, 25.8,
# 34.7, 37.4 at 2,048; before the matrix kernel's blocks took four strips: 21.7, 26.6, 35.6, 37.8 and 1.25, 1.67, 1.80)
COST = ((8, 1.0), (16, 1.17), (24, 1.58), (32, 1.70))

# What a pass of up to 64, 128 or 256 rows costs next to one of 32, by the engine's products (True: --int8, whose
# kernel takes two, four or eight sets of groups, so a pass costs about what the widest of its set does: 48 rows 1.52,
# 96 rows 2.21, 192 rows 3.84; fp16 products run 32 rows a launch and cost about their rows: 48 rows 1.53, 96 2.70,
# 192 5.29). scripts/pass_profile.py --rows 256 [--int8] on the real model at 3 bits a weight, 2026-11-15, the mean
# of the ratios at 64 and 2,048 positions (32, 64, 128, 256 rows: int8 25.4, 39.6, 57.7, 102.1 ms at 64 positions and
# 27.7, 43.8, 65.0, 116.4 at 2,048; fp16 31.3, 57.4, 111.8, 220.6 and 33.6, 61.8, 119.0, 234.0;
# logs/exp/B2-server/profile_int8.log, profile_fp16.log)
WIDE_COST = {True: ((64, 1.57), (128, 2.31), (256, 4.11)), False: ((64, 1.84), (128, 3.56), (256, 7.01))}


def cost(rows: int, int8: bool = False) -> float:
    if rows <= COST[-1][0]:
        return next(c for most, c in COST if rows <= most)
    return COST[-1][1] * next(c for most, c in WIDE_COST[bool(int8)] if rows <= most)

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
        # the order of service (sclmd/order.py, --priorities): when its request came (None: when the batcher first saw
        # it), its priority (lower first) and class; its level and place in this pass
        self.t_arrival, self.priority, self.klass, self.level, self.rank = None, 0, 0, 0, 0
        # a position at which the reading stops for a moment: when the sequence holds exactly that many positions,
        # at_pause(job) is called between passes (scripts/serve.py keeps the shared prefix there), then it reads on
        self.pause, self.at_pause = None, None

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
    """widths: those a pass may have (default: 8, 16 and 32 up to the engine's rows, and an engine's wider ones)."""

    def __init__(self, eng: E.Engine, drafter: E.EngineDrafter | None = None, depth: int = 6, widths: tuple | None = None):
        self.eng, self.drafter, self.depth = eng, drafter, depth
        self.choices = tuple(sorted(widths)) if widths else tuple(w for w in E.ROWS if w <= eng.rows)
        assert all(w in E.ROWS and w <= eng.rows for w in self.choices) and self.choices[0] <= E.ATT_ROWS, self.choices
        self.int8 = bool(getattr(eng, "int8", 0))  # the products a wide pass's cost is measured for (WIDE_COST)
        self.jobs = []
        self.turn = 0
        self.make_room = None  # called when the pool of keys and values is empty: frees some and says so, or says no
        self.order = None  # sclmd/order.py's Order (--priorities): the tiles, rows and guesses of a pass by priority (None: as below)
        self.passes = self.rows = self.guess_rows = self.kept_rows = 0
        self.starved = 0  # jobs that had to wait for keys and values in the last pass
        self.ended = self.restarts = 0  # jobs that have ended; times a job had to start over for lack of memory
        self.widths = dict.fromkeys(sorted({8, 16, 32} | set(self.choices)), 0)  # passes by width
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
        best, narrow = None, {}
        for width in self.choices:
            if width > E.ATT_ROWS and not self._unread(order, narrow):
                break  # a wider pass is for prompts someone waits on that the narrower ones leave unread
            rows = self._fill(order, width)
            if width <= E.ATT_ROWS:
                narrow = rows
            total = sum(rows.values())
            gain = sum(n if j.pending is None else 1 + sum(keep[: n - 1]) for j, n in rows.items())
            rate = gain / cost(total, self.int8)
            if best is None or rate > 1.03 * best[0]:  # a wider pass has to be clearly worth it
                best = (rate, rows)
        return best[1]

    def _room(self, j: Job) -> int:
        """Positions the job's sequence can take in the next pass: up to its capacity, or to its pause."""
        eng = self.eng
        room = eng.capacity - eng.lengths[j.seq]
        if j.pause is not None and eng.lengths[j.seq] < j.pause:  # a prompt read up to its pause first
            room = min(room, j.pause - eng.lengths[j.seq])
        return room

    def _unread(self, order: list, rows: dict) -> bool:
        """Whether a prompt someone waits on has tokens it could read that `rows` does not give it."""
        return any(j.pending is None and not j.only_read and rows.get(j, 0) < min(len(j.todo), self._room(j)) for j in order)

    @staticmethod
    def _ends(j: Job, n: int) -> bool:
        """Whether n rows read the last of the job's prompt, whose logits choose the reply's first token (step(): not quiet)."""
        return j.pending is None and not j.only_read and j.resume is None and n >= len(j.todo)

    def _fit(self, rows: dict) -> dict:
        """The rows of a pass of more than ATT_ROWS rows by job, such that every row whose logits are looked at is one
        that the pass has the logits of (its last ATT_ROWS at most, from a group of eight: E.logits_first). step() puts
        them at the end: the prompts that end in the pass, the one with the most rows first (only its last row needs
        them), then the texts being written. While they do not fit, in this order: up to 7 rows fewer of the prompts
        that do not end (a pass of a whole number of eights has the logits of its last 32), a guess fewer, a prompt
        that ends reads all but its last token (which the next pass reads), a text being written waits a pass."""
        rows = dict(rows)
        while True:
            total = sum(rows.values())
            if total <= E.ATT_ROWS:
                return rows
            ends = sorted((j for j in rows if self._ends(j, rows[j])), key=lambda j: -rows[j])
            writers = [j for j in rows if j.pending is not None]
            tail = (1 + sum(rows[j] for j in ends[1:]) if ends else 0) + sum(rows[j] for j in writers)
            if tail <= total - E.logits_first(total, 0):
                return rows
            quiet = sorted((j for j in rows if j.pending is None and j not in ends), key=lambda j: -rows[j])
            over = total % 8
            if tail <= E.ATT_ROWS and 0 < over <= sum(rows[j] for j in quiet):
                for j in quiet:
                    cut = min(over, rows[j])
                    rows[j] -= cut
                    over -= cut
                    if not rows[j]:
                        del rows[j]
                    if not over:
                        break
                continue
            guessing = [j for j in writers if rows[j] > 1]
            if guessing:
                rows[max(reversed(guessing), key=lambda j: rows[j])] -= 1
            elif len(ends) > 1:
                rows[ends[-1]] -= 1
                if not rows[ends[-1]]:
                    del rows[ends[-1]]
            else:
                del rows[writers[-1]]

    def _fill(self, order: list, width: int) -> dict:
        """Rows of a pass of `width` rows by job. Every text being written gets one (when there are more than fit, in
        turn); the prompts being read share what is left; then guesses, one more for each writer in turn. A pass of
        more than ATT_ROWS rows takes ATT_ROWS writers at most and no reading ahead (which stays in the narrower
        passes), and its rows are fitted to its logits (_fit)."""
        eng, live = self.eng, order
        wide = width > E.ATT_ROWS
        writers = [j for j in order if j.pending is not None]
        if self.order is not None:  # by level (sclmd/order.py): the best writers' rows first
            writers = self.order.writers(writers)
        rows, free = {}, width
        for j in writers[: min(free, E.ATT_ROWS)]:
            rows[j] = 1
            free -= 1
        room = {j: self._room(j) for j in live}

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
        ahead = [] if wide else [j for j in order if j.only_read]
        waiting = [j for j in order if j.pending is None and not j.only_read]
        if self.order is not None:
            waiting = self.order.readers(waiting)
        if wide:  # a sequence a segment, MAX_SEGS of them in a pass (a narrow pass has fewer rows than that)
            waiting = waiting[: E.MAX_SEGS - len(rows)]
        able = (lambda j: min(self.depth, j.max_new - len(j.out), room[j] - 1)) if self.drafter is not None else None  # noqa: E731
        guessed = 0  # (sclmd/order.py) guesses the best class's writers took before the prompts of a worse class
        if self.order is None:
            free -= read(waiting, free)
        else:  # level by level; the best class's writers take their whole guess depth before a worse class's prompts
            given, guessed = self.order.prompts(waiting, writers, rows, read, free, E.ATT_ROWS - sum(j in rows for j in writers) if wide else None, able)
            free -= given + guessed
        free -= read(ahead, min(free, width // 4) if writers else free)
        if self.drafter is not None:
            # (a wide pass: the guesses that fit beside the writers in the rows it has the logits of)
            spare = min(free, E.ATT_ROWS - sum(j in rows for j in writers) - guessed) if wide else free
            while spare and any(rows[j] - 1 < able(j) for j in writers if j in rows):
                for j in writers:
                    if spare and j in rows and rows[j] - 1 < able(j):
                        rows[j] += 1
                        free, spare = free - 1, spare - 1
        read(ahead, free)
        return self._fit(rows) if wide else rows

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
        if self.order is not None:  # (sclmd/order.py) every job's level; the tiles go to the jobs in its order
            live = self.order.before(self, live)
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
            if len(holders) > 1:  # (sclmd/order.py: the last of the worst level)
                self._start_over(holders[-1] if self.order is None else self.order.last(holders))
            else:
                victim = holders[-1] if holders else live[-1]
                victim.done = "full"
                self.jobs.remove(victim)
                self.ended += 1
            return True
        wide = sum(rows[j] for j in use) > E.ATT_ROWS
        if wide:  # some jobs may have waited for keys and values: the rest fitted to the pass's logits again
            rows = self._fit({j: rows[j] for j in use})
            use = [j for j in use if j in rows]
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
        # The rows whose logits nobody looks at go first: a prompt being read needs none until its last token is in
        # (and none at all when it is read ahead, or read again after starting over). The pass computes the logits
        # from the first row that needs them, and only those rows' choices are fetched.
        quiet = lambda j: j.pending is None and (len(j.todo) > rows[j] or j.only_read or j.resume is not None)  # noqa: E731
        ends = []
        if wide:  # (_fit) the prompts that end, the one with the most rows first, then the texts being written
            ends = sorted((j for j in use if not quiet(j) and j.pending is None), key=lambda j: -rows[j])
            use = [j for j in use if quiet(j)] + ends + [j for j in use if j.pending is not None]
        else:
            use = [j for j in use if quiet(j)] + [j for j in use if not quiet(j)]
        parts = []
        for j in use:
            if j.pending is None:
                parts.append((j.seq, j.todo[: rows[j]]))
            else:
                parts.append((j.seq, [j.pending] + guesses.get(j, [])))
        skip = sum(len(ids) for j, (_, ids) in zip(use, parts) if quiet(j))  # the rows without logits
        if ends:
            skip += rows[ends[0]] - 1  # of the first prompt that ends, only the last row's are looked at
        logits, hidden = eng.run(parts, skip)
        total = sum(len(ids) for _, ids in parts)
        # the pass row whose logits are logits[0]: a pass of more than ATT_ROWS rows has those of its last rows only
        base = E.logits_first(total, skip) if total > E.ATT_ROWS else 0
        assert base <= skip, (total, skip, base)
        # the choices of all texts come off the GPU together: every row's first choice, and for the texts that sample
        # every row's largest logits (one wait for the pass, not one a text)
        loud = [j for j in use if not quiet(j)]
        top = max((j.pick.top_k for j in loud if j.pick is not None and j.pick.temperature > 0), default=0)
        vals, idx = (t.cpu() for t in logits[skip - base :].float().topk(top, dim=-1)) if top else (None, None)
        first = logits[skip - base :].argmax(-1).cpu() if any(j.pick is not None and j.pick.temperature <= 0 for j in loud) else None

        def choose(j: Job, a: int, b: int, position: int) -> list:
            """Job j's tokens at `position`.. from rows a..b-1 of the pass."""
            a, b = a - skip, b - skip
            return first[a:b].tolist() if j.pick.temperature <= 0 else j.pick.draw(vals[a:b], idx[a:b], position)

        self.passes += 1
        self.widths[8 if total <= 8 else 16 if total <= 16 else next(w for w in self.widths if w >= total)] += 1
        row = 0
        for j, (_, ids) in zip(use, parts):
            n, hn = len(ids), hidden[row : row + len(ids)]
            row += n
            j.passes, j.rows = j.passes + 1, j.rows + n
            self.rows += n
            at = eng.lengths[j.seq]  # after the pass: the position after the rows
            if j.pending is None:  # a prompt: when its last token is in, the reply's first token is chosen
                j.todo, j.read = j.todo[n:], j.read + n
                if j.pause is not None and at >= j.pause:
                    if at == j.pause and j.todo and j.at_pause is not None:
                        j.at_pause(j)
                    j.pause = None
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
