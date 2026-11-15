# The order of service by priority (design.md 6.16 "우선순위", 6.18; scripts/serve.py --priorities). Off by default:
# Server.order and Batcher.order are None and every code path is the one before. None of it changes a text's tokens
# (a row is computed the same beside any others, and a choice depends only on (seed, position)): only who gets the
# rows and the tiles of a pass, which request is placed on a sequence and which kept conversation leaves the GPU first.
#
# A request's class is LOCAL (this PC), IN_QUOTA or OVER_QUOTA (= IN_QUOTA + gap; scripts/serve_quota.py says which:
# the one it was taken in with); its priority, a number (lower first), is its class and a step more for each earlier
# request of its tenant that is still under way or waiting (its second, third ... concurrent request; no cap). The
# steps are worked out again every turn: when a tenant's first request ends, its second moves up at once.
# Aging: a request's level is its priority less one step for every `age` seconds since it came, but never as good as
# a request of this PC (a request that is not this PC's stops at IN_QUOTA). So an over-quota request goes before an
# in-quota one that has waited gap x age seconds less than it (2 minutes by default), and only this PC's own requests
# can hold one back for good.
# What it orders (the S1 brief's --priorities part; its --prompt-order shortest, --burst-hold, the request's own
# priority and service_tier fields and its reserved sequences are left out):
# - the waiting list (Server.turn): by (level, arrival); any request that can be placed is, not only the head (a head
#   that has waited GUARD seconds is not passed by a longer request); a request of a better priority than every one
#   under way is placed even while texts wait for keys and values (it gets their tiles first);
# - a pass (Batcher): the tiles go to the jobs by level; the prompts' rows by level (the jobs of a level share theirs
#   as before, one row each in turn; a wide pass's segments go to the best first); the writers of the best class take
#   their whole guess depth before a prompt or a writer of a worse class gets rows (within a class the writers share
#   guesses in turn, as before: a tenant's own concurrent requests do not take the depth from each other); when no job
#   can go on, the last of the worst level gives its memory up and starts over;
# - make room (Server._make_room): the kept conversation of the worst class leaves the GPU first, then the one used
#   longest ago.
#
# Tiers and the concurrency ladder (design.md 6.20, 6.21; scripts/serve_tiers.py: --tiers, --concurrency). Off (the
# defaults: ladder None, tiers the one of N2) the numbers above are what they were. With a ladder, a tenant's requests
# under way or waiting are not a step each but a band each: the first BANDS[0] of them +0 ("priority"), the next
# +1 ("normal"), the next +2 ("low"), so its 13th is no worse than its 5th (the server refuses a 13th: 429). With
# tiers, the class is the tier's base plus the over-quota gap: this PC 0; then the key's, the verified address's and
# the unverified address's bases, each `tier_steps` above the one before. The first base is past every band of this
# PC (3 with a ladder), so this PC's requests stay ahead of any key's; tier_steps is the gap plus the bands plus one
# by default, so that an over-quota key in its low band is still ahead of an in-quota verified address: tier first,
# then quota, then band. Aging stops at the first base for everyone but this PC (as N2: nobody waits for ever).
import collections
import time

LOCAL = 0  # this PC: a loopback request through no proxy (scripts/serve_net.py)
IN_QUOTA = 1
GAP = 12  # steps from in quota to over quota (--over-quota-steps)
AGE = 10.0  # seconds of waiting that make a request one step better (--priority-age)
GUARD = 5.0  # seconds after which the waiting list's head is not passed by a request longer than it
SCAN = 4  # requests past a refused one that the waiting list tries a turn
TIERS = ("key", "ip_verified", "ip_unverified")  # best first (scripts/serve_tiers.py); None (no --tiers) is the first
BANDS = (1, 3, 8)  # a tenant's requests at once: so many "priority", "normal", "low" (--concurrency); the next is refused
BAND_STEPS = (0, 1, 2)  # what each band adds to the class


def best_of(klass: int) -> int:
    """The best level aging may give a request of this class: only this PC's are LOCAL (without tiers or a ladder;
    Order.best_of in general)."""
    return LOCAL if klass == LOCAL else IN_QUOTA


class Order:
    """klass(req) -> the request's class now (LOCAL, IN_QUOTA or over); without it: this PC or in quota. ladder: the
    concurrency bands' sizes (None: a step for each earlier request of the tenant, as N2). tier_steps: the steps from
    one tier's base to the next (0: gap + the bands + 1)."""

    def __init__(self, age: float = AGE, gap: int = GAP, klass=None, clock=time.perf_counter, ladder=None, tier_steps: int = 0):
        assert age > 0 and gap >= 0 and tier_steps >= 0, (age, gap, tier_steps)
        self.age, self.gap, self.clock = float(age), int(gap), clock
        self.ladder = tuple(int(x) for x in ladder) if ladder else None
        span = BAND_STEPS[len(self.ladder) - 1] if self.ladder else 0
        self.first = IN_QUOTA + span  # the best class of a request that is not this PC's: past every band of this PC's
        self.tier_steps = int(tier_steps) or self.gap + span + 1
        self.over = self.first + self.gap
        self.klass = klass or (lambda req: LOCAL if getattr(req, "local", False) else self.first)
        self.now = 0.0
        self.classes = {}  # sequence -> the class of the request last placed on it (make room: its kept conversation's)
        self.n = collections.Counter()

    def class_of(self, tier, over: bool) -> int:
        """A request's class that is not this PC's: its tier's base (None, no --tiers: the first), and the gap when it
        came over its hour's quota."""
        base = self.first + (TIERS.index(tier) * self.tier_steps if tier in TIERS else 0)
        return base + (self.gap if over else 0)

    def band(self, earlier: int) -> int:
        """The steps a tenant's request gets for the `earlier` ones of its own under way or waiting before it."""
        if self.ladder is None:
            return earlier
        top = 0
        for i, size in enumerate(self.ladder):
            top += size
            if earlier < top:
                return BAND_STEPS[i]
        return BAND_STEPS[len(self.ladder) - 1]

    def best_of(self, klass: int) -> int:
        return LOCAL if klass == LOCAL else self.first

    def level(self, priority: int, since: float, klass: int) -> int:
        return max(priority - int((self.now - since) // self.age), self.best_of(klass))

    # ---- the waiting list (scripts/serve.py: Server.turn)

    def place(self, server) -> None:
        """Every turn: the priorities of the requests under way and waiting (their class, and their tenant's earlier ones
        still there), then waiting requests onto sequences."""
        now = self.now = self.clock()
        seen = collections.Counter()
        for r in sorted(server.active + server.waiting, key=lambda r: r.t_in):
            r.klass = k = self.klass(r)
            tenant = getattr(r, "tenant", None)
            r.priority = k + (self.band(seen[tenant]) if tenant is not None else 0)  # its tenant's second ... concurrent request: a step (band) each
            if tenant is not None:
                seen[tenant] += 1
            r.level = self.level(r.priority, r.t_in, k)
            if r.job is not None:  # the batcher's jobs follow (a job of a request: its arrival is the request's)
                r.job.priority, r.job.klass, r.job.t_arrival = r.priority, k, r.t_in
        if not server.waiting:
            return
        line = sorted(server.waiting, key=lambda r: (r.level, r.t_in))
        head, refused = line[0], 0
        for req in line:
            best = min((r.priority for r in server.active), default=None)
            req.outranks = best is not None and req.priority < best
            if req is not head and head in server.waiting and now - head.t_in > GUARD and len(req.full) > len(head.full):
                continue  # the head has waited long: nobody longer than it goes before it
            if not req.outranks and refused and self.blocked(server):
                break  # what refused the last one refuses this one too
            if self._try(server, req):
                server.waiting.remove(req)
                self.classes[req.job.seq] = req.klass
                req.job.priority, req.job.klass, req.job.t_arrival = req.priority, req.klass, req.t_in
                self.n["placed"] += 1
                self.n["placed_past_the_head"] += req is not head and head in server.waiting
                continue
            refused += 1
            if refused >= SCAN:
                break

    def _try(self, server, req) -> bool:
        """server._place(req). It keeps every request out while texts wait for keys and values; one of a better priority
        than every request under way is placed all the same (the pass gives it tiles first: before())."""
        batch = server.batch
        if req.outranks and batch.starved and server.active:
            starved, batch.starved = batch.starved, 0
            try:
                placed = server._place(req)
            finally:
                batch.starved = starved
            self.n["placed_while_starved"] += placed
            return placed
        return server._place(req)

    @staticmethod
    def blocked(server) -> bool:
        """Whether _place refuses any request now (scripts/serve.py): texts wait for keys and values, or no sequence is
        free (unused, or a kept conversation's)."""
        if server.batch.starved and server.active:
            return True
        return not server.unused and not server.kept

    # ---- make room (Server._make_room)

    def to_park(self, server):
        """The kept conversation that leaves the GPU first: of the worst class, then the one used longest ago."""
        return min(server.kept.values(), key=lambda k: (-self.classes.get(k.seq, self.first), k.used))

    # ---- the batcher (sclmd/batch.py)

    def before(self, batch, live: list) -> list:
        """Once a pass, before its rows are shared: every job's level (a job no request placed, the reading ahead of a
        reply, takes the class of the request last placed on its sequence, from now); the jobs in the order they get
        tiles: by level, then as they came."""
        self.now = self.clock()
        for i, j in enumerate(batch.jobs):
            if j.t_arrival is None:
                j.t_arrival = self.now
                j.priority = j.klass = self.classes.get(j.seq, self.first)
            j.rank = i  # as they came (Batcher.jobs is in that order)
            j.level = self.level(j.priority, j.t_arrival, j.klass)
        return sorted(live, key=lambda j: (j.level, j.rank))

    @staticmethod
    def writers(writers: list) -> list:
        """The writers in the order they get rows and guesses: by level (stable: within a level, the pass's turn)."""
        return sorted(writers, key=lambda j: j.level)

    @staticmethod
    def readers(readers: list) -> list:
        """The prompts someone waits on, by level (stable): a wide pass's segments go to the best first."""
        return sorted(readers, key=lambda j: j.level)

    def prompts(self, readers: list, writers: list, rows: dict, read, most: int, cap: int | None, able) -> tuple:
        """Up to `most` rows for the prompts someone waits on (read: the batcher's own, one row to each in turn), level
        by level; the writers of the best class take their whole guess depth (able: the depth a writer may have; None:
        no drafter) before a prompt or a writer of a worse class gets rows. cap: the guesses a pass may have at most (a
        wide pass: beside the writers in the rows it has the logits of). -> (prompt rows, guess rows) given."""
        ready = [j for j in writers if j in rows]
        top = min((j.klass for j in ready), default=None)
        first = able is not None and top is not None and (any(j.klass > top for j in readers) or any(j.klass > top for j in ready))
        best = [j for j in ready if j.klass == top] if first else []
        given = guessed = 0
        for p in sorted({j.level for j in readers}):
            group = [j for j in readers if j.level == p]
            if first and any(j.klass > top for j in group):
                guessed, first = self._guesses(best, rows, able, min(most - given, cap if cap is not None else most)), False
            given += read(group, most - given - guessed)
        if first:
            guessed = self._guesses(best, rows, able, min(most - given, cap if cap is not None else most))
        self.n["best_guess_rows"] += guessed
        return given, guessed

    @staticmethod
    def _guesses(writers: list, rows: dict, able, budget: int) -> int:
        """Guesses for these writers up to their depth, one more each in turn (as the batcher deals them); rows given."""
        given = 0
        while given < budget and any(rows[j] - 1 < able(j) for j in writers):
            for j in writers:
                if given < budget and rows[j] - 1 < able(j):
                    rows[j] += 1
                    given += 1
        return given

    @staticmethod
    def last(holders: list):
        """The job that gives its memory up when no job can go on: the last to come of the worst level."""
        return max(reversed(holders), key=lambda j: j.level)

    def about(self) -> dict:
        out = {"age_s": self.age, "over_quota_steps": self.gap}
        if self.ladder is not None:  # (only then: GET / of a server without it stays as it was)
            out.update(bands=list(self.ladder), tier_steps=self.tier_steps)
        return {**out, **dict(self.n)}
