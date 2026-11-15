# The order of service by priority (sclmd/order.py: scripts/serve.py --priorities). On a stand-in engine on the CPU
# (Fake: the engine's own bookkeeping of tiles, park, unpark and snapshots, with logits that are a hash of what each
# position reads through its tiles, so a tile that holds the wrong thing changes the text; brief K2's, brought to
# master's Engine) and, given a GPU, on the small random model at wide rows. Texts are sampled at temperature 5 and
# compared with the same request alone: no order may change a text, only when its tokens come.
import collections
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from scripts.serve import Request, Server  # noqa: E402
from sclmd import engine as E  # noqa: E402
from sclmd import model as M  # noqa: E402
from sclmd import order as O  # noqa: E402
from sclmd.batch import Batcher, Job  # noqa: E402
from sclmd.generate import Picker  # noqa: E402

STOP = (10**6,)  # no stop token: every reply is as long as it may be
S, F = M.SLIDING, M.FULL
VOCAB = 40
GPU = torch.cuda.is_available() and os.path.exists(E.DLL)
OVER = O.IN_QUOTA + O.GAP


class Fake(E.Engine):
    """The engine's bookkeeping on the CPU (window 5; a sliding ring of window + rows). What a position keeps is a hash:
    in the sliding tiles of its token and what its window kept before it, in the full ones of its token and that. A
    row's logits are a hash of what it reads: its window in the sliding tiles and every position before it in the full
    ones, each through the sequence's tile table. A pass of more than ATT_ROWS rows gives the logits of its last rows
    only, as the engine does (E.logits_first)."""

    def __init__(self, capacity: int = 96, sequences: int = 4, pool: dict | None = None, rows: int = 8, window: int = 5):
        cfg = SimpleNamespace(window=window, layer_types=[S, F], kv_heads={S: 1, F: 1}, head_dim={S: 4, F: 4}, vocab=VOCAB)
        self.cfg, self.device, self.capacity, self.sequences, self.rows = cfg, "cpu", capacity, sequences, rows
        self.lengths, self.highs, self._snaps = [0] * sequences, [0] * sequences, {}
        self.ring = {t: E.ring_tiles(cfg, t, capacity, rows) for t in (S, F)}
        self.extra, self.width = {S: 0, F: 0}, dict(self.ring)
        self.pool = {t: sequences * self.ring[t] for t in (S, F)} if pool is None else dict(pool)
        self.free = {t: list(range(self.pool[t] - 1, -1, -1)) for t in (S, F)}
        self.tiles = {t: [[-1] * self.width[t] for _ in range(sequences)] for t in (S, F)}
        self.tables = {t: torch.zeros(sequences, self.ring[t], dtype=torch.int32) for t in (S, F)}
        self.firsts = [-1] * sequences
        self.lean, self.keyless, self.kv4, self.int8 = False, {S: False, F: False}, False, 0.0
        self.kv_all = {t: (torch.full((1, self.pool[t] * E.TILE), -1, dtype=torch.int64),) for t in (S, F)}

    def _at(self, kind: str, seq: int, pos: int) -> int:
        tile = self.tiles[kind][seq][(pos >> 4) % self.ring[kind]]
        assert tile >= 0, f"position {pos} of sequence {seq} has no {kind} tile"
        return tile * E.TILE + (pos & 15)

    def run(self, parts: list, logits_from: int = 0):
        total = sum(len(ids) for _, ids in parts)
        assert 0 < total <= self.rows and len({s for s, _ in parts}) == len(parts) and len(parts) <= E.MAX_SEGS
        for seq, ids in parts:
            assert self.lengths[seq] + len(ids) <= self.capacity
            assert self._kept(self.lengths[seq] - self.cfg.window + 1, seq), "cropped further back than the sliding layers keep"
        self._provide([(seq, self.lengths[seq], len(ids)) for seq, ids in parts])
        sliding, full = self.kv_all[S][0][0], self.kv_all[F][0][0]
        rows = []
        for seq, ids in parts:
            for k, tok in enumerate(ids):
                q = self.lengths[seq] + k
                before = tuple(int(sliding[self._at(S, seq, p)]) for p in range(max(0, q - self.cfg.window + 1), q))
                sliding[self._at(S, seq, q)] = hash((tok, before)) & 0xFFFFFFFFFF
                full[self._at(F, seq, q)] = hash((tok, int(sliding[self._at(S, seq, q)]))) & 0xFFFFFFFFFF
                near = before + (int(sliding[self._at(S, seq, q)]),)
                far = tuple(int(full[self._at(F, seq, p)]) for p in range(q + 1))
                g = torch.Generator().manual_seed(hash((near, far)) & 0x7FFFFFFF)
                rows.append(torch.randn(VOCAB, generator=g) * 3)
        for seq, ids in parts:
            self.lengths[seq] += len(ids)
            self.highs[seq] = max(self.highs[seq], self.lengths[seq])
        logits = torch.stack(rows)
        if total > E.ATT_ROWS:
            logits = logits[E.logits_first(total, logits_from) :]
        return logits, torch.zeros(total, 4)

    def snapshot(self, seq: int = 0):
        """As Engine.snapshot, in pageable memory."""
        own = [x for x in self.tiles[S][seq] if x >= 0]
        snap = self._snaps.get(seq)
        if snap is None or not snap["data"]:
            snap = self._snaps[seq] = {"data": [torch.empty(t.shape[0], self.ring[S], E.TILE, dtype=t.dtype) for t in self.kv_all[S]]}
        self._copy_tiles(S, own, snap["data"], out=True)
        snap["tiles"], snap["at"] = own, (self.lengths[seq], self.highs[seq])


class Guesses:
    """A drafter that guesses at random (all the batcher asks of one: guess_many): some guesses are kept, most not."""

    def guess_many(self, asks: list) -> list:
        return [[3 + hash((seq, length, i, pending)) % (VOCAB - 3) for i in range(k)] for pending, _h, length, seq, k in asks]


class Tick:
    """A clock that goes on `step` seconds each time it is read (virtual time for aging)."""

    def __init__(self, step: float = 0.01):
        self.t, self.step = 1000.0, step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def ids(n: int, seed: int) -> list:
    return torch.randint(3, VOCAB, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def hot(seed: int = 0) -> Picker:
    return Picker(temperature=5.0, top_k=16, top_p=1.0, seed=seed)


def classes(table: dict):
    """An Order's klass: by tenant (this PC: LOCAL), in quota when not in the table."""
    return lambda req: O.LOCAL if getattr(req, "local", False) else table.get(getattr(req, "tenant", None), O.IN_QUOTA)


def server(sequences: int = 4, pool: dict | None = None, rows: int = 8, order: O.Order | None = None, drafter=None, capacity: int = 96,
           ram: int = 1 << 30, eng=None) -> Server:
    srv = Server(None, eng or Fake(capacity, sequences, pool, rows), drafter, STOP, ram=ram)
    if order is not None:
        srv.order = srv.batch.order = order
    return srv


def ask(srv: Server, full: list, max_new: int, seed: int, tenant: str | None = None, local: bool = False) -> Request:
    req = Request(full, len(full) - 1, max_new, hot(seed), False)
    req.tenant, req.local = tenant, local
    if srv.order is not None:
        req.t_in = srv.order.clock()
    srv.submit(req)
    return req


def events(req: Request) -> tuple:
    tokens, result = [], None
    while not req.events.empty():
        event = req.events.get()
        if event[0] == "token":
            tokens.append(event[1])
        else:
            result = event[1]
    return tokens, result


def finish(srv: Server, *reqs) -> list:
    while srv.turn(wait=False):
        pass
    return [events(r) for r in reqs]


def alone(full: list, max_new: int, seed: int, capacity: int = 96) -> list:
    """The request's text on a fresh server with room for everything and no order."""
    srv = server(2, capacity=capacity)
    return finish(srv, ask(srv, full, max_new, seed))[0][0]


def overflow() -> dict:
    """Keys and values for three conversations of 50 positions in the sliding layers, but not in the full ones (9 tiles
    of 16 positions): when the three are at position 48, none can go on."""
    return {S: 8, F: 9}


MIX = [("a", False, 30), ("b", False, 7), (None, True, 18), ("c", False, 3), ("a", False, 25), (None, True, 11)]  # (tenant, this PC, prompt)


@pytest.mark.parametrize("rows", [8, 64])
@pytest.mark.parametrize("short", [False, True], ids=["room", "short"])
def test_no_order_changes_a_text(rows, short):
    prompts = [ids(n, 600 + i) for i, (_, _, n) in enumerate(MIX)]
    want = [alone(p, 20, i) for i, p in enumerate(prompts)]
    for age in (10.0, 0.05):  # (with fast aging the levels cross while the texts are written)
        order = O.Order(age, klass=classes({"a": OVER, "c": OVER}), clock=Tick())
        srv = server(4 if short else 8, overflow() if short else None, rows=rows, order=order, drafter=Guesses())
        reqs = [ask(srv, p, 20, i, tenant, local) for i, (p, (tenant, local, _)) in enumerate(zip(prompts, MIX))]
        got = finish(srv, *reqs)
        assert [g[0] for g in got] == want, (rows, short, age)
        assert not srv.active and not srv.waiting and sorted(srv.unused + list(srv.kept)) == list(range(srv.eng.sequences))
        assert order.n["placed"] == len(MIX)
        if short:
            assert srv.batch.restarts or srv.parked  # the pool did run short: texts started over, or kept ones left the GPU


def test_the_waiting_list_goes_by_priority_and_places_past_its_head():
    order = O.Order(klass=classes({"bg": OVER}), clock=Tick())
    srv = server(1, order=order, ram=0)
    first = ask(srv, ids(6, 760), 4, 0, "x")
    srv.turn(wait=False)
    assert first.job is not None
    later = [ask(srv, ids(6, 761), 4, 1, "bg"), ask(srv, ids(6, 762), 4, 2, "y"), ask(srv, ids(6, 763), 4, 3, None, local=True)]
    placed = []
    while srv.turn(wait=False):
        for r in later:
            if r.job is not None and r not in placed:
                placed.append(r)
    assert placed == [later[2], later[1], later[0]]  # this PC, then in quota, then over quota: not as they came
    assert all(events(r)[1]["ended"] == "length" for r in later)
    # without an order: as before, the head first while it can be placed
    srv = server(1, ram=0)
    assert srv.batch.order is None and srv.order is None
    reqs = [ask(srv, ids(6, 770 + i), 4, i, t, local=t is None) for i, t in enumerate(("x", "bg", "y", None))]
    seen = []
    while srv.turn(wait=False):
        for r in reqs:
            if r.job is not None and r not in seen:
                seen.append(r)
    assert seen == reqs


def test_a_tenants_later_requests_are_a_step_lower_each_and_move_up_when_one_ends():
    order = O.Order(klass=classes({}), clock=Tick())
    srv = server(8, order=order)
    mine = [ask(srv, ids(5, 780 + i), 30, i, "t") for i in range(3)]
    other = ask(srv, ids(5, 789), 30, 9, "u")
    srv.turn(wait=False)
    assert [r.priority for r in mine] == [1, 2, 3] and other.priority == 1
    assert [r.job.priority for r in mine] == [1, 2, 3]  # the batcher's jobs follow
    mine[0].cancelled = True
    srv.turn(wait=False)
    srv.turn(wait=False)
    assert [r.priority for r in mine[1:]] == [1, 2]
    finish(srv, *mine, other)


def test_a_better_request_is_placed_while_texts_wait_for_keys_and_values():
    prompts = [ids(10, 780 + i) for i in range(3)]
    want = [alone(p, 40, i) for i, p in enumerate(prompts)]
    chat_prompt = ids(4, 789)
    for on in (True, False):
        order = O.Order(klass=classes({"bg": OVER}), clock=Tick()) if on else None
        srv = server(4, overflow(), order=order)
        bg = [ask(srv, p, 40, i, "bg") for i, p in enumerate(prompts)]
        for _ in range(300):
            if srv.batch.starved:
                break
            srv.turn(wait=False)
        assert srv.batch.starved
        chat = ask(srv, chat_prompt, 4, 9, None, local=True)
        srv.turn(wait=False)
        assert (chat.job is not None) == on  # without an order, no request is placed while a pass starved
        got = finish(srv, *bg, chat)
        assert [g[0] for g in got] == want + [alone(chat_prompt, 4, 9)]


def test_prompt_rows_go_by_level_and_the_best_writers_guess_first():
    eng = Fake(256, 4, rows=8)
    batch = Batcher(eng, Guesses(), depth=6)
    order = batch.order = O.Order(clock=Tick())
    jobs = [Job(i, ids(20, 800 + i), 30, hot(i), STOP) for i in range(3)]
    for j, p in zip(jobs, (OVER, O.LOCAL, O.IN_QUOTA)):
        j.priority, j.klass, j.t_arrival = p, p, order.clock()
        batch.add(j)
    live = order.before(batch, list(batch.jobs))
    rows = batch._fill(live, 8)
    assert rows == {jobs[1]: 8}  # this PC's prompt takes the pass
    batch.order = None
    assert sorted(batch._fill(list(batch.jobs), 8).values()) == [2, 3, 3]  # without an order: one row each in turn
    batch.order = order
    while jobs[1].pending is None:  # this PC's prompt is read: now it writes beside the other two prompts
        batch.step()
    rows = batch._fill(order.before(batch, list(batch.jobs)), 8)
    assert rows[jobs[1]] == 1 + 6 and sum(rows.values()) == 8  # its whole guess depth before the worse prompts get a row
    assert rows.get(jobs[2], 0) == 1 and jobs[0] not in rows
    batch.order = None
    plain = batch._fill(list(batch.jobs), 8)
    assert plain[jobs[1]] == 1  # without an order the prompts take the rows and the writer gets no guess


def test_a_wide_pass_gives_its_rows_by_level():
    eng = Fake(400, 3, rows=64)
    batch = Batcher(eng, Guesses(), depth=6)
    order = batch.order = O.Order(clock=Tick())
    jobs = [Job(0, ids(200, 810), 8, hot(0), STOP), Job(1, ids(50, 811), 8, hot(1), STOP)]
    for j, p in zip(jobs, (OVER, O.LOCAL)):
        j.priority, j.klass, j.t_arrival = p, p, order.clock()
        batch.add(j)
    rows = batch._share(order.before(batch, list(batch.jobs)))
    assert sum(rows.values()) > E.ATT_ROWS and rows[jobs[1]] == 50  # this PC's prompt whole in the wide pass, the rest to the other
    first = {}
    while batch.step():
        for j in jobs:
            if j.out and j not in first:
                first[j] = batch.passes
    assert first[jobs[1]] == 1 and first[jobs[1]] < first[jobs[0]]
    want = [alone(j.base, 8, i, capacity=400) for i, j in enumerate(jobs)]
    assert [j.out for j in jobs] == want


def test_the_worst_level_starts_over_when_no_text_can_go_on():
    prompts = [ids(10, 820 + i) for i in range(3)]
    for order_on in (False, True):
        eng = Fake(96, 4, overflow())
        batch = Batcher(eng, None)
        if order_on:
            batch.order = O.Order(clock=Tick())
        jobs = [Job(i, p, 40, hot(i), STOP) for i, p in enumerate(prompts)]
        for j, p in zip(jobs, (O.IN_QUOTA, OVER, O.LOCAL)):  # the worst came second, this PC's last
            j.priority, j.klass, j.t_arrival = p, p, 0.0 if batch.order is None else batch.order.clock()
            batch.add(j)
        over, freed = [], []
        start = batch._start_over
        batch._start_over = lambda j: over.append(j) or start(j)

        def make_room() -> bool:  # what the server does: a text that is over gives its memory up
            for j in jobs:
                if j.done is not None and j.seq not in freed:
                    eng.release(j.seq)
                    freed.append(j.seq)
                    return True
            return False

        batch.make_room = make_room
        batch.run()
        assert over and set(over) == ({jobs[1]} if order_on else {jobs[2]}), order_on  # without an order: the last to come
        assert [j.out for j in jobs] == [alone(p, 40, i) for i, p in enumerate(prompts)]


def test_make_room_takes_the_worst_class_off_the_gpu_first():
    order = O.Order(klass=classes({"bg": OVER}), clock=Tick())
    srv = server(3, order=order)
    bg = ask(srv, ids(8, 830), 4, 0, "bg")
    finish(srv, bg)
    mine = ask(srv, ids(8, 831), 4, 1, None, local=True)
    finish(srv, mine)
    other = ask(srv, ids(8, 832), 4, 2, "x")
    finish(srv, other)
    assert len(srv.kept) == 3 and not srv.unused
    for r, used in ((mine, 1.0), (other, 2.0), (bg, 3.0)):  # the over-quota one was used last: the order of before takes this PC's
        srv.kept[r.job.seq].used = used
    kept = {k.seq: k for k in srv.kept.values()}
    gone = order.to_park(srv)
    assert gone.seq == bg.job.seq  # over quota first, though the others were used longer ago
    srv._make_room()
    assert bg.job.seq not in srv.kept and len(srv.parked) == 1 and mine.job.seq in srv.kept and set(kept) - set(srv.kept) == {bg.job.seq}
    srv2 = server(3)  # without an order: the one used longest ago (the same call, as before)
    for i in range(3):
        finish(srv2, ask(srv2, ids(8, 840 + i), 4, i))
    oldest = min(srv2.kept.values(), key=lambda k: k.used)
    srv2._make_room()
    assert oldest.seq not in srv2.kept


def reads_until_first(order: O.Order, stream_class: int, steps: int = 600) -> int | None:
    """An over-quota prompt of 20 tokens and, every pass, a new prompt of `stream_class` that wants more rows than a
    pass has: the pass at which the over-quota one is first given a row (None: never in `steps` passes)."""
    eng = Fake(256, 12, rows=8)
    batch = Batcher(eng, None)
    batch.order = order
    bg = Job(0, ids(20, 850), 1, hot(0), STOP)
    bg.priority, bg.klass, bg.t_arrival = OVER, OVER, order.clock()
    batch.add(bg)
    for step in range(steps):
        used = {j.seq for j in batch.jobs}
        free = [s for s in range(1, 12) if s not in used]
        if free:
            eng.release(free[0])
            j = Job(free[0], ids(12, 860 + step), 1, hot(step), STOP)
            j.priority, j.klass, j.t_arrival = stream_class, stream_class, order.clock()
            batch.add(j)
        batch.step()
        if bg.read:
            return step
    return None


def test_aging_lets_an_over_quota_prompt_through_a_stream_of_in_quota_ones_but_never_past_this_pc():
    clock = Tick(0.1)  # (read twice a pass: 0.2 s a pass)
    slow = reads_until_first(O.Order(10.0, clock=clock), O.IN_QUOTA, 300)
    assert slow is None  # 60 s of passes: not yet (the gap is 12 steps of 10 s)
    passes = reads_until_first(O.Order(1.0, clock=Tick(0.1)), O.IN_QUOTA)
    assert passes is not None and 40 <= passes <= 90  # about 12 steps of 1 s, at 0.2 s a pass
    assert reads_until_first(O.Order(1.0, clock=Tick(0.1)), O.LOCAL) is None  # this PC's own requests always go first


def test_the_level_of_a_request_is_its_priority_less_its_waiting_and_stops_at_its_best():
    order = O.Order(10.0, clock=lambda: 0.0)
    order.now = 125.0
    assert order.level(OVER, 0.0, O.IN_QUOTA) == OVER - 12 == O.IN_QUOTA
    assert order.level(OVER, 0.0, O.IN_QUOTA) == order.level(OVER, -1000.0, O.IN_QUOTA)  # never better than in quota
    assert order.level(O.IN_QUOTA + 3, 100.0, O.IN_QUOTA) == O.IN_QUOTA + 1
    assert order.level(2, 0.0, O.LOCAL) == O.LOCAL  # this PC's second request ages back to the first's
    assert O.best_of(O.LOCAL) == O.LOCAL and O.best_of(OVER) == O.IN_QUOTA


def test_without_an_order_the_batcher_and_server_are_as_before():
    """The default: no Order, no hook is called (the paths of before), and Job's new fields are not read."""
    srv = server(4)
    assert srv.order is None and srv.batch.order is None
    reqs = [ask(srv, ids(n, 870 + n), 10, n) for n in (5, 9, 14)]
    for r in reqs:
        r.priority = -5  # (ignored)
    got = finish(srv, *reqs)
    assert [g[0] for g in got] == [alone(r.full, 10, n) for r, n in zip(reqs, (5, 9, 14))]
    assert "order" not in srv.about() and all(r.job.priority == 0 and r.job.t_arrival is None for r in reqs)


@pytest.mark.skipif(not GPU, reason="needs a GPU and the engine")
@pytest.mark.parametrize("rows", [32, 128])
def test_the_small_model_writes_the_same_texts_by_priority(small_e, drafter_e, rows):  # noqa: F811
    from test_engine import engine_drafter, ids_of
    from sclmd.generate import generate

    cap = 336
    sizes = (60, 90, 12, 140, 45, 7, 100)
    tenants = [("bg", False), ("bg", False), (None, True), ("x", False), ("bg", False), (None, True), ("x", False)]
    prompts = [ids_of(n, 900 + i) for i, n in enumerate(sizes)]
    want = [generate(small_e.engine(capacity=cap), p, 16, hot(40 + i), stop=STOP) for i, p in enumerate(prompts)]
    for pool in (None, {S: 30, F: 30}):  # room, and keys and values short (texts wait, start over, kept ones leave)
        eng = small_e.engine(capacity=cap, sequences=5, rows=rows, pool=pool)
        order = O.Order(0.05, klass=classes({"bg": OVER}), clock=Tick())
        srv = server(eng=eng, order=order, drafter=engine_drafter(eng, drafter_e))
        reqs = [ask(srv, p, 16, 40 + i, t, local) for i, (p, (t, local)) in enumerate(zip(prompts, tenants))]
        got = finish(srv, *reqs)
        assert [g[0] for g in got] == want, (rows, pool)
        assert order.n["placed"] == len(reqs)
        if rows > E.ATT_ROWS:
            assert any(w > E.ATT_ROWS and n for w, n in srv.batch.widths.items()), srv.batch.widths


if GPU:
    from test_engine import drafter_e, small_e  # noqa: F401, E402  (fixtures)


def test_a_tenants_own_writers_share_the_guesses_in_turn():
    """Writers of one class (a tenant's concurrent requests: priorities a step apart) get guesses one more each in
    turn, as without an order; only a better class takes its whole depth first."""
    eng = Fake(256, 4, rows=8)
    batch = Batcher(eng, Guesses(), depth=6)
    order = batch.order = O.Order(clock=Tick())
    jobs = [Job(i, ids(3, 880 + i), 60, hot(i), STOP) for i in range(4)]
    for i, j in enumerate(jobs):
        j.priority, j.klass, j.t_arrival = O.IN_QUOTA + i, O.IN_QUOTA, order.clock()
        batch.add(j)
    while any(j.pending is None for j in jobs):
        batch.step()
    rows = batch._fill(order.before(batch, list(batch.jobs)), 8)
    assert sorted(rows.values()) == [2, 2, 2, 2]  # 8 rows: four writers, a guess each
    jobs[2].klass = jobs[2].priority = O.LOCAL  # one of this PC's among them: its whole depth first
    rows = batch._fill(order.before(batch, list(batch.jobs)), 8)
    assert rows[jobs[2]] == 1 + 4 and all(rows[j] == 1 for j in jobs if j is not jobs[2])


def test_the_tiles_go_to_the_better_level_first():
    """Keys and values for one prompt's first tile: this PC's prompt, though it came second, gets them; the other waits."""
    for on in (True, False):
        eng = Fake(96, 3, {S: 1, F: 1}, rows=16)
        batch = Batcher(eng, None)
        order = O.Order(clock=Tick()) if on else None
        batch.order = order
        first, second = Job(0, ids(10, 890), 4, hot(0), STOP), Job(1, ids(10, 891), 4, hot(1), STOP)
        for j, k in ((first, OVER), (second, O.LOCAL)):
            j.priority, j.klass, j.t_arrival = k, k, 0.0 if order is None else order.clock()
            batch.add(j)
        live = list(batch.jobs) if order is None else order.before(batch, list(batch.jobs))
        rows = batch._fill(live, 16)
        assert rows == ({second: 10, first: 6} if on else {first: 8, second: 8})  # (the prompts' rows by level too)
        batch.step()
        assert batch.starved == 1
        assert (second.read, first.read) == ((10, 0) if on else (0, 8)), on
