# An OpenAI-compatible server for Gemma 4 31B on the engine, for several users at once: every pass of the main model
# carries tokens of all the requests that are under way (sclmd/batch.py), and a conversation's keys and values stay
# on the GPU after its reply, so its next turn reads only what is new.
#
#   POST /v1/chat/completions   messages, stream, max_tokens, temperature, top_p, top_k, seed, stop,
#                               reasoning_effort or thinking (the reply's thinking comes as reasoning_content)
#   GET  /v1/models, /health, / (what the server is doing)
#
# One thread owns the engine; the HTTP threads only hand it requests and turn the tokens it settles into text.
# Fields it does not serve, the body's size and the connections: scripts/serve_contract.py. API keys, back-pressure,
# GET /metrics, the request log and the status file: scripts/serve_admin.py. Who a request is, the prompt's and the
# reply's limits and a stream's pings (the internet server): scripts/serve_net.py. Quotas and the order of service by
# priority: scripts/serve_quota.py, sclmd/order.py. Prompt caches per user, cache_control and the 1 h disk cache
# (--prompt-cache): scripts/serve_cache.py. ZDR: what it logs of a request is metadata (scripts/serve_contract.py:
# why, trace).
import argparse
import json
import os
import queue
import select
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler

import torch

from sclmd import ckpt
from sclmd import engine as E
from sclmd import model as M
from sclmd import plan as P
from sclmd import tools as T
from sclmd.batch import Batcher, Job
from sclmd.generate import Picker
from sclmd.text import Pieces
from sclmd.stream import VRAM_MARGIN, cap_vram
from scripts import serve_admin as A
from scripts import serve_cache as K
from scripts import serve_contract as C
from scripts import serve_net as N
from scripts import serve_ops

DEPTH = 6  # the drafter's generation_config.json: num_assistant_tokens
SHARES = "own_v1"  # the measurements a plan is fitted from (scripts/chat.py)
LOAD_ROOM = 320 << 20
MODEL = "gemma-4-31b-it"


Refused = C.Refused  # a request that cannot be served: (HTTP status, message[, field])


def narrow(eng) -> int:
    """The rows of the passes that write (the margin a conversation keeps below the context): the widest of the
    batcher's widths up to 32, which the Server keeps on its engine (--rows; --wide-rows: the wider passes only read
    prompts, so a request fits as on a server without them, a --rows 16 one's too); else an engine's rows, 32 at most."""
    return getattr(eng, "narrow_rows", None) or min(eng.rows, E.ATT_ROWS)


class Request:
    """A chat completion under way: what the HTTP thread and the engine's thread know of it."""

    def __init__(self, full: list, start: int, max_new: int, pick: Picker, think: bool, messages: list | None = None, cut: int = 0):
        self.full, self.start, self.max_new, self.pick, self.think, self.messages = full, start, max_new, pick, think, messages
        self.cut = cut  # the length of its shared part (the system turn, or the messages the client marked): kept for others (Prefixes)
        self.events = queue.Queue()  # ("token", id), then ("done", {...}) or ("error", status, message)
        self.cancelled = False
        self.job = None
        self.reused = 0
        self.t_in, self.t_first = time.perf_counter(), None
        self.stops = ()  # the client's stop strings
        self.stop_text = None  # the answer as a stop string cut it: set by the HTTP thread before `cancelled`
        self.ignored = []  # fields the request gave that are not served (serve_contract.check_fields)
        self.tools = None  # the tools declared to the model (sclmd/tools.py); None: none, the reply is read as before
        self.opened = False  # the reply begins inside a thought the prompt opened (after a tool's result, thinking on)
        self.stop_ids = None  # the tokens that end its reply, where they are not the server's (parallel_tool_calls: false)
        self.stop_calls = []  # the calls the client had got when a stop string cut the answer: set before stop_text
        # who sent it (scripts/serve_net.py: tag), the seams of the later stages: the tenant's id (None: tenancy off),
        # its priority (0: everyone alike), the scope of its prompt caches (None: shared, as before)
        self.tenant, self.priority, self.scope = None, 0, None
        # --prompt-cache (scripts/serve_cache.py): the positions whose states it asks to be cached [(n, seconds, kind)],
        # and the tokens a state saved for it covers past what it read from a cache (cache_write_tokens); went_on: it goes
        # on from a conversation of its own, kept or parked (Server._place)
        self.points, self.wrote, self.went_on = [], 0, False


class Kept:
    """A conversation whose reply is over, kept on its sequence. base: the conversation up to where the model's last
    turn began, which the sequence can always go back to (snapped: through the copy in RAM, Engine.restore).
    later: the conversation with that reply as a later turn writes it, which the sequence is reading in or has read
    (job), so that the next turn finds it there."""

    def __init__(self, seq: int, base: list, snapped: bool, scope=None):
        self.seq, self.base, self.snapped, self.used = seq, base, snapped, time.time()
        self.later, self.job = None, None
        self.scope = scope  # whose it is (req.scope): only a request of the same scope goes on from it

    def now(self, eng: E.Engine) -> list:
        """What the sequence holds."""
        return self.base if self.later is None else self.later[: eng.lengths[self.seq]]


class Parked:
    """A kept conversation that has left the GPU: its tokens and the engine's state of it in RAM (Engine.park)."""

    def __init__(self, tokens: list, state: dict, used: float, scope=None):
        self.tokens, self.state, self.used, self.scope = tokens, state, used, scope


class Prefixes:
    """Shared prompt prefixes (system turns, instructions, few-shot examples many requests begin with): the engine's
    state of each one in page-locked RAM (Engine.save), put on a request's sequence (Engine.unpark) so that it reads
    only what comes after. Up to `ram` bytes; the one used longest ago goes first. The same tokens as reading it all:
    the engine computes a row the same whatever pass it is in."""

    def __init__(self, ram: int, least: int = 16):
        self.ram, self.least = ram, least
        self.held = {}  # tuple of tokens -> [state, last used]
        self.too_big = set()  # prefixes larger than the whole budget: never saved
        self.hits = self.stored = 0
        self.waits = 0  # times a request that begins with one waited for keys and values to put it on

    def find(self, full: list, longer_than: int, scope=None):
        """The longest prefix held that `full` begins with and is longer than `longer_than` (and shorter than full) -> (tokens, state) or None.
        (scope: none here; scripts/serve_cache.py's Store takes this place with --prompt-cache.)"""
        best = None
        for key, entry in self.held.items():
            if longer_than < len(key) < len(full) and (best is None or len(key) > len(best)) and tuple(full[: len(key)]) == key:
                best = key
        if best is None:
            return None
        self.held[best][1] = time.time()
        return best, self.held[best][0]

    def wants(self, tokens: list) -> bool:
        return self.ram > 0 and len(tokens) >= self.least and tuple(tokens) not in self.held and tuple(tokens) not in self.too_big

    def put(self, tokens: list, state: dict) -> None:
        if state["bytes"] > self.ram:
            return
        self.held[tuple(tokens)] = [state, time.time()]
        self.stored += 1
        while self.bytes() > self.ram:
            del self.held[min(self.held, key=lambda k: self.held[k][1])]

    def bytes(self) -> int:
        return sum(e[0]["bytes"] for e in self.held.values())


class Server:
    """The engine's side: requests come in through `inbox`, each becomes a job on a sequence, and run() makes passes
    while there is anything to do. Conversations whose reply is over stay on their sequences (kept); when sequences or
    keys and values run short, the one used longest ago moves to RAM (parked, up to `ram` bytes), and comes back
    when its next turn arrives."""

    def __init__(self, tok, eng: E.Engine, drafter, stop: tuple, ram: int = 0, prefix_ram: int = 0, widths: tuple | None = None):
        self.tok, self.eng, self.stop, self.ram = tok, eng, stop, ram
        self.prefixes = Prefixes(prefix_ram)
        self.parked = []
        self.batch = Batcher(eng, drafter, DEPTH, widths)  # widths: of a pass (Batcher; --rows and --wide-rows)
        eng.narrow_rows = max(w for w in self.batch.choices if w <= E.ATT_ROWS)  # narrow(): from the widths, not the engine's rows
        self.batch.make_room = self._make_room
        self.canon = None  # (request, the reply's tokens) -> the conversation's tokens with the reply as a later turn writes it
        self.inbox, self.waiting, self.active = queue.Queue(), [], []
        self.kept = {}  # sequence -> Kept
        self.unused = list(range(eng.sequences - 1, -1, -1))
        self.totals = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
        self.t0, self.busy_s = time.time(), 0.0
        self.ops = None  # --crash-only: the lifecycle (scripts/serve_ops.py: install); None: as before
        self.idle_wait = None  # seconds an idle engine thread waits for a request before it turns anyway (None: until one comes)
        self.order = None  # --priorities (sclmd/order.py: scripts/serve_quota.py installs it): the waiting list, the passes and make room by priority
        self.cache = None  # --prompt-cache: scripts/serve_cache.py's Cache (install); None: as before

    def submit(self, req: Request) -> None:
        self.inbox.put(req)

    def _make_room(self) -> bool:
        """The pool of keys and values is empty, or every sequence is taken: the kept conversation that was used
        longest ago leaves the GPU (to RAM while there is room for it there). False: none is kept."""
        if not self.kept:
            return False
        old = min(self.kept.values(), key=lambda k: k.used) if self.order is None else self.order.to_park(self)  # (--priorities: the worst class first)
        if self.ram <= 0:
            self._forget(old)
            return True
        if old.job is not None:
            self.batch.drop(old.job)
        del self.kept[old.seq]
        self.unused.append(old.seq)
        if old.later is None and not self._to_start(old):  # nothing of it can be gone on from
            self.eng.release(old.seq)
            return True
        if self.cache is not None:  # --prompt-cache: page-locked, in the one budget of parked conversations and cache entries
            tokens = old.now(self.eng)
            state = self.eng.park(old.seq, alloc=K.alloc)
            if K.locked(state):  # (RAM that could not be page-locked keeps nothing: the conversation is forgotten)
                self.parked.append(Parked(tokens, state, old.used, old.scope))
                self.cache.fit(self)
            return True
        self.parked.append(Parked(old.now(self.eng), self.eng.park(old.seq), old.used))
        while sum(p.state["bytes"] for p in self.parked) > self.ram:  # RAM is full: the oldest parked ones are forgotten
            self.parked.remove(min(self.parked, key=lambda p: p.used))
        return True

    def _to_start(self, kept: Kept) -> bool:
        """Takes a kept conversation's sequence back to where its last turn began (its base): past it the sequence
        holds the reply as it was written, which a later turn does not begin with. False: the sliding layers no
        longer hold what that position needs (should not happen: the batcher copies a turn start to RAM before a
        reply writes over it)."""
        eng, seq, n = self.eng, kept.seq, len(kept.base)
        if eng.lengths[seq] == n:
            return True
        if kept.snapped or (kept.job is not None and kept.job.snapped):
            eng.restore(seq)
        if not eng._kept(n - eng.cfg.window + 1, seq):
            return False
        eng.crop(n, seq)
        eng.trim(seq)
        return True

    def _forget(self, kept: Kept) -> None:
        if kept.job is not None:
            self.batch.drop(kept.job)
        del self.kept[kept.seq]
        self.eng.release(kept.seq)
        self.unused.append(kept.seq)

    def _place(self, req: Request) -> bool:
        """A sequence for the request: the kept conversation it continues, else an unused one, else the one kept
        longest. False: every sequence is writing, or texts under way are waiting for keys and values (the request waits)."""
        eng, best, most = self.eng, None, 0
        if self.batch.starved and self.active:
            return False
        for k in self.kept.values():
            if k.scope != req.scope:  # another user's (--prompt-cache): never gone on from
                continue
            for held in (k.now(eng), k.base):  # as far as the sequence has got, else back at its last turn's start
                n = len(held)
                if most < n < len(req.full) and req.full[:n] == held:
                    best, most = k, n
        back = max((p for p in self.parked if p.scope == req.scope and most < len(p.tokens) < len(req.full) and req.full[: len(p.tokens)] == p.tokens),
                   key=lambda p: len(p.tokens), default=None)
        shared = self.prefixes.find(req.full, len(back.tokens) if back is not None else most, req.scope) if self.prefixes.held else None
        if shared is not None:  # a prefix kept for everyone: a copy of it goes onto a sequence (the copy in RAM stays)
            back = Parked(list(shared[0]), shared[1], time.time())
        reused = 0
        if back is not None:  # the conversation is in RAM: back onto a sequence, making room as needed
            if shared is None:
                self.parked.remove(back)
            if not self.unused and not self._make_room():
                if shared is None:
                    self.parked.append(back)
                return False
            seq = self.unused.pop()
            while True:
                try:
                    eng.unpark(seq, back.state)
                    reused = len(back.tokens)
                    break
                except E.KVFull:
                    if self._make_room():
                        continue
                    if shared is not None and self.active:  # a kept prefix stays kept: the request waits for the texts under way
                        self.unused.append(seq)
                        self.prefixes.waits += 1
                        return False
                    break  # the texts under way hold too much for it now: it starts anew
        elif best is not None:
            seq = best.seq
            if best.job is not None:
                self.batch.drop(best.job)
            del self.kept[seq]
            reused = most
            if eng.lengths[seq] != most and not self._to_start(best):  # back to the last turn's start
                eng.release(seq)
                reused = 0
        elif self.unused:
            seq = self.unused.pop()
        elif self._make_room():
            seq = self.unused.pop()
        else:
            return False
        # the sequence holds exactly what is counted as kept: nothing of another conversation
        assert eng.lengths[seq] == reused, f"sequence {seq} holds {eng.lengths[seq]} positions where {reused} were expected"
        req.reused = reused
        req.went_on = bool(reused) and shared is None and (back is not None or best is not None)  # from its own kept or parked conversation
        stop = self.stop if req.stop_ids is None else req.stop_ids
        req.job = Job(seq, req.full[reused:], req.max_new, req.pick, stop, emit=lambda t, q=req.events: q.put(("token", t)),
                      start=req.start, base=req.full[:reused])
        if shared is not None and reused:
            self.prefixes.hits += 1
        if self.cache is not None:  # --prompt-cache: its prefix state and the states it asks for, saved as it is read
            self.cache.arm(self, req)
        elif reused < req.cut < len(req.full) and self.prefixes.wants(req.full[: req.cut]):  # its shared part is kept when read
            req.job.pause, req.job.at_pause = req.cut, lambda job, cut=req.cut: self._keep_prefix(job, cut)
        self.batch.add(req.job)
        self.active.append(req)
        return True

    def _keep_prefix(self, job: Job, cut: int) -> None:
        """The job's sequence holds exactly its shared prefix: a copy of it goes to RAM for the requests that begin with it."""
        tokens, eng = job.base[:cut], self.eng
        if self.prefixes.wants(tokens) and eng.highs[job.seq] == cut:
            size = sum(E.tile_bytes(eng.cfg, t, eng.lean, eng.kv4) * sum(x >= 0 for x in eng.tiles[t][job.seq]) for t in eng.ring)
            if size > self.prefixes.ram:
                self.prefixes.too_big.add(tuple(tokens))
                return
            self.prefixes.put(tokens, eng.save(job.seq))

    def _finish(self, req: Request) -> None:
        eng, job = self.eng, req.job
        seq = job.seq
        # a reply that a client's stop string ended is kept as one that ended by itself (with the answer as cut), unless
        # it was cut while it read everything again after starting over (its sequence does not hold the turn start then)
        stopped = job.done == "cancelled" and req.stop_text is not None and job.pending is not None and eng.lengths[seq] >= req.start
        if (job.done in ("stop", "length") or stopped) and job.out:
            kept = self.kept[seq] = Kept(seq, req.full[: req.start], job.snapped, req.scope)
            try:
                later = self.canon(req, job.out) if self.canon is not None else None
            except Exception as e:  # noqa: BLE001  (a template that cannot write the reply: nothing is read ahead)
                print(f"[serve] seq {seq}: the reply's later form could not be made ({C.why(e)})", flush=True)
                later = None
            if later is not None and later[: req.start] == kept.base and req.start < len(later) <= eng.capacity - 2 * narrow(eng):
                # the reply as a later turn writes it is read in from the turn's start, with rows nothing else needs
                if job.snapped:
                    eng.restore(seq)
                if eng._kept(req.start - eng.cfg.window + 1, seq):
                    eng.crop(req.start, seq)
                    eng.trim(seq)
                    kept.later = later
                    kept.job = Job(seq, later[req.start :], 0, None, start=req.start, only_read=True, base=later[: req.start])
                    kept.job.snapped = job.snapped
                    self.batch.add(kept.job)
        else:  # cancelled or out of memory: nothing of it is kept
            eng.release(seq)
            self.unused.append(seq)
        now = time.perf_counter()
        usage = {"prompt_tokens": len(req.full), "completion_tokens": len(job.out), "total_tokens": len(req.full) + len(job.out),
                 "prompt_tokens_details": {"cached_tokens": req.reused}}
        K.usage(self, req, usage["prompt_tokens_details"])  # --prompt-cache: cache_write_tokens
        self.totals["requests"] += 1
        self.totals["prompt_tokens"] += len(req.full)
        self.totals["cached_tokens"] += req.reused
        self.totals["completion_tokens"] += len(job.out)
        first = req.t_first or now
        timings = {"total_ms": round(1000 * (now - req.t_in)), "first_token_ms": round(1000 * (first - req.t_in)),
                   "tokens_per_second": round(len(job.out) / max(now - first, 1e-9), 1), "prompt_read": job.read,
                   "rows_per_pass": round(job.rows / max(job.passes, 1), 2)}
        reason = {"stop": "stop", "length": "length", "full": "length", "cancelled": "stop"}[job.done]
        req.events.put(("done", {"finish_reason": reason, "usage": usage, "timings": timings, "ended": job.done}))
        print(f"[serve] seq {seq}: prompt {len(req.full)} ({req.reused} kept), reply {len(job.out)} tokens, first after "
              f"{timings['first_token_ms'] / 1000:.2f} s, {timings['tokens_per_second']} tok/s, {job.done}; "
              f"{len(self.active)} under way, {len(self.kept)} kept, {len(self.parked)} in RAM", flush=True)

    @torch.no_grad()
    def turn(self, wait: bool = True) -> bool:
        """Takes in the requests that have come and makes one pass. With nothing under way it waits for a request
        (wait) or returns False."""
        try:
            idle = not self.active and not self.waiting and not self.batch.jobs
            if idle and not wait and self.inbox.empty():
                return False
            self.waiting.append(self.inbox.get(timeout=self.idle_wait if idle else 0))
            while True:
                self.waiting.append(self.inbox.get_nowait())
        except queue.Empty:
            pass
        self.waiting = [r for r in self.waiting if not r.cancelled]
        if self.cache is not None:  # --prompt-cache: states read from disk come in; what is past its time goes
            self.cache.tick(self)
        if self.order is None:
            while self.waiting and self._place(self.waiting[0]):
                self.waiting.pop(0)
        else:  # --priorities: by priority, any request that can be placed (sclmd/order.py)
            self.order.place(self)
        for req in self.active:
            if req.cancelled:
                req.job.cancel = True
        t = time.perf_counter()
        self.batch.step()
        self.busy_s += time.perf_counter() - t
        for req in self.active:
            if req.t_first is None and req.job.out:
                req.t_first = time.perf_counter()
        for req in [r for r in self.active if r.job.done is not None]:
            self.active.remove(req)
            try:
                self._finish(req)
            except Exception:  # it is in no list any more: told here, the others by run()
                req.events.put(("error", 500, "the server failed while ending the reply"))
                raise
        return True

    def run(self) -> None:
        """Turns for ever. If one fails, every request that is under way or waiting is told (the HTTP threads answer
        with an error instead of waiting for tokens that will not come), and the engine starts from nothing. The log
        gets the failure's frames and type, not its message (ZDR)."""
        while True:
            try:
                self.turn()
                if self.ops is not None:
                    self.ops.beat()
            except Exception as e:  # noqa: BLE001
                C.trace(e)
                told = self.ops.failed(e) if self.ops is not None else None  # --crash-only: (503, why) when the engine is dead
                print(f"[serve] the engine's thread failed: {C.why(e)}; {len(self.active) + len(self.waiting)} requests are told", flush=True)
                for req in self.active + self.waiting:
                    req.events.put(("error",) + (told or (500, f"the server failed while answering: {type(e).__name__}")))
                self.reset()
                if self.ops is not None:
                    self.ops.after_failure()  # --crash-only: a dead engine ends the process here (exit 75)

    def reset(self) -> None:
        """Forgets every conversation and job: the engine as it was when it was loaded."""
        self.active, self.waiting, self.kept, self.parked = [], [], {}, []
        self.batch.jobs, self.batch.starved = [], 0
        for seq in range(self.eng.sequences):
            self.eng.release(seq)
        self.unused = list(range(self.eng.sequences - 1, -1, -1))

    def about(self) -> dict:
        eng = self.eng
        tile = {t: E.tile_bytes(eng.cfg, t, eng.lean, eng.kv4) for t in eng.ring}
        free = eng.free_tiles()
        order = {"order": self.order.about()} if self.order is not None else {}  # --priorities
        return {"requests": {"under_way": len(self.active), "waiting": len(self.waiting) + self.inbox.qsize(), **self.totals}, **order,
                "conversations_kept": len(self.kept), "conversations_in_ram": len(self.parked),
                "ram_gib": round(sum(p.state["bytes"] for p in self.parked) / 2**30, 2), "sequences": eng.sequences,
                "prefixes": {"held": len(self.prefixes.held), "gib": round(self.prefixes.bytes() / 2**30, 3), "stored": self.prefixes.stored,
                             "hits": self.prefixes.hits, "waits": self.prefixes.waits},
                "keys_and_values": {"gib": round(sum(tile[t] * eng.pool[t] for t in tile) / 2**30, 2),
                                    "free_gib": round(sum(tile[t] * free[t] for t in tile) / 2**30, 2)},
                "started_over_for_memory": self.batch.restarts, "passes": self.batch.passes, "passes_by_width": dict(self.batch.widths), "rows_per_pass": round(self.batch.rows / max(self.batch.passes, 1), 2),
                "guesses_kept": round(self.batch.kept_rows / max(self.batch.guess_rows, 1), 2),
                "tokens_per_second_while_busy": round(self.totals["completion_tokens"] / max(self.busy_s, 1e-9), 1),
                **({"prompt_cache": self.cache.about()} if self.cache is not None else {})}


class App:
    def __init__(self, server: Server, tok, think: bool, max_tokens: int, facts: dict):
        self.server, self.tok, self.think, self.max_tokens, self.facts = server, tok, think, max_tokens, facts
        self.created, self.started = int(time.time()), time.time()
        self.turn = tok.encode("<|turn>model\n", add_special_tokens=False)
        self.opening, self.closing = tok.convert_tokens_to_ids("<|channel>"), tok.convert_tokens_to_ids("<channel|>")
        self.fingerprint = f"sclmd-{facts['bits_per_weight']}bit"
        self.lock = threading.Lock()  # the tokenizer's template is not known to be safe from several threads
        server.canon = self.canonical
        self.fields, self.max_body = "default", C.MAX_BODY  # --lenient-fields / --strict-fields, --max-body-mb
        self.look_s = C.LOOK_S  # --look-gone
        self.max_prompt, self.max_output, self.ping_s = 0, 0, 0.0  # --max-prompt, --max-output, --stream-ping (scripts/serve_net.py; 0: off)
        self._marks = None  # the tool format's tokens (sclmd/tools.py), when a request first needs them

    def marks(self) -> dict:
        if self._marks is None:
            enc = lambda s: self.tok.encode(s, add_special_tokens=False)  # noqa: E731
            self._marks = {"ids": T.Ids(self.tok), "thought": enc("<|channel>thought\n"), "closed": enc("<turn|>\n"),
                           "empty": enc("<|channel>thought\n<channel|>")}
        return self._marks

    def _writable(self, tool: dict) -> bool:
        """Whether the chat template can write this tool's declaration (under self.lock)."""
        try:
            self.tok.apply_chat_template([{"role": "user", "content": "x"}], tools=[tool], tokenize=False, add_generation_prompt=True)
            return True
        except Exception:  # noqa: BLE001
            return False

    def pieces(self, req: Request):
        """What reads the request's reply as it comes: thinking and answer apart (Pieces), and the calls too when it
        declared tools (tools.Reader)."""
        if req.tools is not None:
            return T.Reader(self.tok, self.server.stop, self.opening, self.closing, self.marks()["ids"], req.opened)
        pieces = Pieces(self.tok, self.server.stop, self.opening, self.closing)
        if req.opened:
            pieces.kind, pieces.name = "think", False
        return pieces

    def canonical(self, req: Request, tokens: list):
        """The conversation with the reply just given as a later turn writes it: the answer as the client got it
        (without the thinking, and cut where a stop string of the client's cut it), through the template. A reply with
        tool calls: the calls (tools.later_message)."""
        calls = []
        if req.stop_text is not None:
            text, calls = req.stop_text, req.stop_calls
        else:
            pieces, text = self.pieces(req), ""
            for token in tokens:
                piece = pieces.add(token)
                if piece is not None and piece[0] == "answer":
                    text += piece[1]
            if req.tools is not None:
                piece = pieces.end()
                text = text + (piece[1] if piece is not None else "")
                calls = T.kept_before(pieces.calls, text, req.stops)  # those before a stop string, as the client got them
            text = C.cut_at_stops(text, req.stops)  # the reply may have ended before the HTTP thread saw the stop string
        message = T.later_message(text, calls) if calls else {"role": "assistant", "content": text}
        kw = {"tools": req.tools} if req.tools else {}
        with self.lock:
            out = self.tok.apply_chat_template(req.messages + [message], tokenize=True, add_generation_prompt=False, enable_thinking=req.think, **kw)
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    def card(self) -> dict:
        eng = self.server.eng
        limits = {"max_prompt_tokens": self.max_prompt} if getattr(self, "max_prompt", 0) else {}  # --max-prompt
        return {"id": MODEL, "object": "model", "created": self.created, "owned_by": "sclmd", "root": MODEL, "parent": None,
                "max_model_len": eng.capacity, "context_length": eng.capacity, **limits,
                "max_output_tokens": getattr(self, "max_output", 0) or self.max_tokens,
                "quantization": f"{self.facts['bits_per_weight']} bits a weight",
                "capabilities": {"streaming": True, "reasoning": True, "tools": True, "prompt_cache": True, "vision": False},
                "defaults": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "thinking": self.think, "max_tokens": self.max_tokens}}

    def about(self, totals: bool = True) -> dict:
        """GET /. totals False (--metrics-local, a request from elsewhere than this PC): without the server's facts and
        counts (requests, tokens, kept conversations, prefixes held and hit), which tell one user about the others."""
        out = {"server": "sclmd serve", "model": MODEL, "system_fingerprint": self.fingerprint, "uptime_s": round(time.time() - self.started),
               "endpoints": ["GET /", "GET /health", "GET /v1/models", "GET /v1/models/{id}", "POST /v1/chat/completions"],
               "context_length": self.server.eng.capacity, "max_output_tokens": self.max_tokens}
        return {**out, **self.facts, **self.server.about()} if totals else out

    def request(self, body: dict) -> Request:
        ignored = C.check_fields(body, self.fields)  # fields that are not served: refused, or named in the answer
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            raise Refused(400, "messages: a list of {role, content} is needed")
        strict = self.fields == "strict"
        try:  # {role, content}; tool calls and their results (sclmd/tools.py); the tools to declare
            messages, left = T.messages_of(body["messages"], strict)
            tools, more, at = T.tools_of(body, strict)
        except T.BadRequest as e:
            raise Refused(400, str(e), e.param) from None
        ignored = sorted(set(ignored) | set(more) | set(left))
        if messages[-1]["role"] == "assistant":
            raise Refused(400, "the last message is the one to answer: a user's")
        if N.sure_too_long(messages, getattr(self, "max_prompt", 0)):  # --max-prompt: sure from the text alone, before the lock
            raise Refused(*N.context_exceeded(self.max_prompt, 0, sure=True))
        think = self.think
        if "thinking" in body:
            t = body["thinking"]
            think = t.get("type") == "enabled" if isinstance(t, dict) else bool(t)
        elif body.get("reasoning_effort") is not None:
            think = body["reasoning_effort"] not in ("none", "minimal")
        # the shared part kept for other requests (--prefix-gib): the messages the client marks (sclmd_prefix_messages:
        # how many of the first), else the leading system messages. Read only when the store is on; a mark that is not a
        # number counts as no mark.
        shared = 0
        prefixes = getattr(self.server, "prefixes", None)  # stand-ins for the engine (serve_canned, tests) have none
        if prefixes is not None and prefixes.ram > 0:
            system = next((i for i, m in enumerate(messages) if m["role"] != "system"), len(messages))
            try:
                shared = int(body["sclmd_prefix_messages"]) if body.get("sclmd_prefix_messages") is not None else system
            except (TypeError, ValueError):
                shared = system
            shared = max(0, min(shared, len(messages) - 1))
        def written(tools, messages):
            """The prompt without and with the model's turn opened, and the shared part, through the template."""
            kw = {"tools": tools} if tools else {}  # declared at the top of the system turn: in the shared part (--prefix-gib)
            # no system message (and no mark of the client's): the tools' system turn alone
            sharing = prefixes is not None and prefixes.ram > 0 and (shared > 0 or (bool(tools) and body.get("sclmd_prefix_messages") is None))
            ids = lambda prompt, ms=messages: self.tok.apply_chat_template(ms, tokenize=True, add_generation_prompt=prompt, enable_thinking=think, **kw)  # noqa: E731
            return ids(False), ids(True), (ids(False, messages[:shared]) if sharing and shared else []), sharing

        def without_calls(messages):
            """The conversation with its calls left out, as before tool calls were served (their results are then
            written nowhere) -> (it, the names of what was left out)."""
            out, names = [], []
            for i, m in enumerate(messages):
                if m.get("tool_calls"):
                    out.append({"role": "assistant", "content": m["content"]})
                    names.append(f"messages[{i}].tool_calls")
                else:
                    names += [f"messages[{i}]"] if m["role"] == "tool" else []
                    out.append(m)
            return out, names

        def prompt(tools, messages, ignored):
            """-> the prompt's parts through the template, the tools and the messages it was written with, and the names
            left out. What the template cannot write is left out and named, as before tool calls were served (strict:
            400): a tool's declaration (each one tried alone), else the conversation's calls."""
            try:
                return written(tools, messages), tools, messages, ignored
            except Exception as e:  # noqa: BLE001  (tools or tool messages the template cannot write)
                if not (tools or T.uses_tools(messages)):
                    raise
                why = e
            bad = [k for t, k in zip(tools or [], at) if not self._writable(t)]
            if bad and strict:
                raise Refused(400, f"tools[{bad[0]}]: the chat template cannot write this tool's declaration (--strict-fields)", "tools")
            if bad:
                tools = [t for t, k in zip(tools, at) if k not in bad] or None
                ignored = sorted(set(ignored) | {f"tools[{k}]" for k in bad})
                try:
                    return written(tools, messages), tools, messages, ignored
                except Exception as e:  # noqa: BLE001
                    why = e
            if strict or not T.uses_tools(messages):
                raise Refused(400, f"the chat template could not write these tools or tool messages: {type(why).__name__}: {why}", "messages")
            messages, names = without_calls(messages)
            try:
                return written(tools, messages), tools, messages, sorted(set(ignored) | set(names))
            except Exception as e:  # noqa: BLE001
                raise Refused(400, f"the chat template could not write these tools or tool messages: {type(e).__name__}: {e}", "messages") from None

        with self.lock:
            (before, full, part, sharing), tools, messages, ignored = prompt(tools, messages, ignored)
            eng = self.server.eng
            if tools and not strict and eng.capacity - len(T.ids_of(full)) - narrow(eng) < 16:
                # the tools' declarations leave no room in the context: left out and named, as before they were served
                (before, full, part, sharing), tools, messages, ignored = prompt(None, messages, sorted(set(ignored) | {"tools"}))
        before, full, part = (T.ids_of(x) for x in (before, full, part))
        start, opened = len(before) + len(self.turn), False
        if full[: len(before)] != before or full[len(before) : start] != self.turn:
            # after a tool's result the template writes no turn: the reply goes on in the model's turn
            mk = self.marks() if messages[-1]["role"] == "tool" else None
            got = T.continuation(before, full, self.turn, mk["thought"], mk["closed"], mk["empty"]) if mk else None
            if got is None:
                raise Refused(500, "the chat template did not write the turn as expected")
            start, full, opened = got
        cut = 0  # on token ids: where the template writes the part otherwise than in the whole, the part ends before that
        while cut < min(len(part), len(full)) and part[cut] == full[cut]:
            cut += 1
        if sharing and not shared:  # the tools' system turn alone (the template writes nothing for no message): up to its end
            closed = self.marks()["closed"]
            cut = next((i + len(closed) for i in range(len(full)) if full[i : i + len(closed)] == closed), 0)
        room = eng.capacity - len(full) - narrow(eng)
        most = getattr(self, "max_prompt", 0)  # --max-prompt (scripts/serve_net.py): OpenAI's 400
        if most and len(full) > most:
            raise Refused(*N.context_exceeded(most, len(full)))
        if room < 16:
            if most:  # in OpenAI's shape too
                raise Refused(*N.context_exceeded(eng.capacity, len(full), "context"))
            raise Refused(400, f"the conversation is {len(full)} tokens: the context holds {eng.capacity}")
        asked = body.get("max_completion_tokens", body.get("max_tokens"))
        wanted = int(asked) if asked else self.max_tokens
        cap = getattr(self, "max_output", 0)  # --max-output: more is cut to it and named
        if cap and wanted > cap:
            wanted = cap
            ignored = sorted(set(ignored) | {"max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"})
        max_new = min(wanted, room)
        if max_new < 1:
            raise Refused(400, "max_tokens is at least 1")
        points = []
        if getattr(self.server, "cache", None) is not None:  # --prompt-cache: cache_control's marks (scripts/serve_cache.py)
            with self.lock:
                points, more = K.points(self, body, messages, tools, think, full, strict, self.server.cache.turns)
            ignored = sorted(set(ignored) | set(more))
        temperature = float(body.get("temperature", 1.0) if body.get("temperature") is not None else 1.0)
        top_p = float(body.get("top_p", 0.95) if body.get("top_p") is not None else 0.95)
        top_k = int(body.get("top_k", 64) or 64)
        seed = body.get("seed")
        pick = Picker(temperature, max(1, min(top_k, 1024)), top_p, seed=int(seed) & 0x3FFFFFFF if seed is not None else uuid.uuid4().int & 0x3FFFFFFF)
        req = Request(full, start, max_new, pick, think, messages, cut)
        req.ignored = ignored
        req.points = points
        req.tools, req.opened = tools, opened
        if tools and body.get("parallel_tool_calls") is False:  # one call: the reply ends with it
            req.stop_ids = tuple(self.server.stop) + (self.marks()["ids"].end,)
        return req


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "sclmd"
    timeout = C.IDLE_TIMEOUT  # a connection's thread ends when it has been idle this long (--idle-timeout; None: for ever)

    def log_message(self, *args) -> None:  # the engine's thread prints a line per request
        pass

    def send_response(self, code, message=None) -> None:
        A.answered(self, code)  # every answer's status: /metrics and the request log (scripts/serve_admin.py)
        super().send_response(code, message)
        A.status_headers(self)  # --quotas: the tenant's x-ratelimit-* headers (scripts/serve_quota.py)

    def _json(self, status: int, obj: dict, headers: dict | None = None) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        if self.close_connection:  # a body that was not read: the connection cannot carry another request
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, param: str | None = None, code: str | None = None) -> None:
        kind = "invalid_request_error" if status < 500 else "server_error"
        A.message(self, param, code)  # the request log keeps the field and the code, not the message (ZDR)
        self._json(status, {"error": {"message": message, "type": kind, "param": param, "code": code}})

    def _gone(self) -> bool:
        """Whether the client has closed the connection."""
        try:
            if not select.select([self.connection], [], [], 0)[0]:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def do_GET(self) -> None:
        app = self.server.app
        path = self.path.split("?")[0].rstrip("/")
        if A.get(self):  # an API key asked and not given (401), /metrics, /debug/audit (scripts/serve_admin.py)
            return
        if path == "/v1/models":
            self._json(200, {"object": "list", "data": [app.card()]})
        elif path.startswith("/v1/models/"):
            if path[len("/v1/models/"):] == MODEL:
                self._json(200, app.card())
            else:
                self._error(404, f"the model served is {MODEL}")
        elif path == "/health":
            about = app.server.about()
            self._json(200, {"status": "ok", "model": MODEL, "under_way": about["requests"]["under_way"],
                             "waiting": about["requests"]["waiting"], "kept": about["conversations_kept"]})
        elif path in ("", "/v1"):
            self._json(200, app.about(totals=not N.metrics_refused(self)))  # --metrics-local: the counts from this PC only
        else:
            self._error(404, "unknown path: GET / lists the endpoints")

    def handle_expect_100(self) -> bool:
        """A client that waits to be told to send its body (Expect: 100-continue) hears at once when it would be refused."""
        refusal = C.body_refusal(self.headers, getattr(self.server.app, "max_body", C.MAX_BODY)) if self.command == "POST" else None
        if refusal is None:
            return super().handle_expect_100()
        self.close_connection = True
        self._error(*refusal[0].args)
        C.linger(self.connection, self.rfile, refusal[1], seconds=2.0, idle=0.5)  # it was not told to send, but may
        return False

    def do_POST(self) -> None:
        try:
            self._post()
        finally:
            A.posted(self)  # --concurrency: the request's place is given back, however it ended (scripts/serve_tiers.py)

    def _post(self) -> None:
        app = self.server.app
        if A.post(self):  # an API key asked and not given (401), /debug/fault (scripts/serve_admin.py)
            return
        if self.path.split("?")[0].rstrip("/") != "/v1/chat/completions":
            return self._error(404, "unknown path: GET / lists the endpoints")
        fields = getattr(app, "fields", "default")
        self.unread = 0
        try:
            body = json.loads(C.read_body(self, getattr(app, "max_body", C.MAX_BODY)))
            if not A.take_place(self):  # --concurrency: a place once the body has come (scripts/serve_tiers.py)
                return
            stops, odd = C.stops_of(body, fields)
            req = app.request(body)
        except Refused as e:
            self._error(*e.args)
            if self.unread != 0:  # a body that was not read: it is thrown away after the answer, or the client gets a reset
                C.linger(self.connection, self.rfile, self.unread)
            return
        except (ValueError, TypeError, AttributeError, OverflowError, RecursionError) as e:
            return self._error(400, f"the body is not a chat completion request: {str(e)[:300]}")
        N.tag(self, req)  # who sent it: req.tenant, req.priority, req.scope (scripts/serve_net.py)
        K.before(app, req)  # --prompt-cache: a state of the disk cache its prompt begins with, read into RAM first
        req.stops = stops
        if odd:  # a stop that is not text: what is text of it is served
            req.ignored = sorted(set(req.ignored) | {"stop"})
        head = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": MODEL, "system_fingerprint": app.fingerprint}
        named = {"x-sclmd-ignored": C.names_of(req.ignored)} if req.ignored else {}  # fields given that are not served
        if req.ignored:
            print(f"[serve] {head['id']}: {C.names_of(req.ignored)} not served: ignored", flush=True)
        stream = bool(body.get("stream"))
        tooled = req.tools is not None  # its reply's calls are read (sclmd/tools.py)
        pieces = app.pieces(req) if tooled or req.opened else Pieces(app.tok, app.server.stop, app.opening, app.closing)
        reasoning = C.Reasoning(app.opening, app.closing)
        reasoning.inside = req.opened
        calls = []  # the reply's tool calls, as the client gets them
        text = {"think": "", "answer": ""}
        sent = 0  # of the answer: what has gone out (a stop string may still be forming at its end)
        hold = max((len(s) for s in stops), default=1) - 1
        stopped = False
        event, looked = None, time.monotonic()  # the last event taken; when the connection was last looked at
        finish = None  # what the client is told the reply ended with (the request log)
        ping = getattr(app, "ping_s", 0.0) if stream else 0.0  # --stream-ping: ": ping" until the first token
        if not A.submit(self, req, head["id"], stream):  # app.server.submit(req), or 429 past --max-waiting
            return

        def chunk(delta: dict, finish: str | None = None, **more) -> None:
            data = {**head, "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}], **more}
            self.wfile.write(b"data: " + json.dumps(data, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

        try:
            if not A.started(self, req):  # --queue-timeout: 503 when no sequence came in time (nothing was written before)
                return
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("x-request-id", head["id"])
                for name, value in named.items():
                    self.send_header(name, value)
                self.end_headers()
                self.close_connection = True
                chunk({"role": "assistant", "content": ""})
            wrote = time.monotonic()  # the stream's last bytes: a ping S seconds after them, until anything comes
            while True:
                try:
                    event = req.events.get(timeout=min(1.0, ping) if ping else 1.0)
                except queue.Empty:
                    if self._gone():
                        raise ConnectionError
                    if ping and time.monotonic() - wrote >= ping:  # an SSE comment: clients skip it, proxies see the stream alive
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        wrote = time.monotonic()
                    continue
                ping = 0.0  # a token (or the end) has come: no more pings
                if event[0] == "error":
                    if stream:
                        self.wfile.write(b"data: " + json.dumps({"error": {"message": event[2], "type": "server_error"}}).encode() + b"\n\n")
                        break
                    return self._error(event[1], event[2])
                if event[0] == "done":
                    result = event[1]
                    break
                reasoning.add(event[1])
                if stopped:
                    continue
                # the tokens that have come by now are taken together (waiting 30 ms for more made no difference
                # to the speed of eight clients: 293 tok/s with it, 300 without)
                tokens, last = [event[1]], None
                try:
                    while last is None:
                        more = req.events.get_nowait()
                        if more[0] == "token":
                            tokens.append(more[1])
                            reasoning.add(more[1])
                        else:
                            last = more
                except queue.Empty:
                    pass
                look = getattr(app, "look_s", C.LOOK_S)  # --look-gone (off by default: a half-closed client reads as gone)
                if look and not stream and time.monotonic() - looked > look:  # tokens keep coming: is anyone still waiting for them?
                    looked = time.monotonic()
                    if self._gone():
                        raise ConnectionError
                new = {"think": "", "answer": ""}
                for token in tokens:
                    piece = pieces.add(token)
                    if piece is not None:
                        new[piece[0]] += piece[1]
                        text[piece[0]] += piece[1]
                if new["think"] and stream:
                    chunk({"reasoning_content": new["think"]})
                done_calls = pieces.take() if tooled else []
                calls += done_calls
                if new["answer"]:
                    cut = min((text["answer"].find(s) for s in stops if s in text["answer"]), default=-1)
                    if cut >= 0:  # a stop string: the answer ends before it, and the engine is told to stop
                        if tooled:  # and so do its calls: those after it are cut with it, however the tokens came
                            calls = T.kept_before(calls, text["answer"], stops)
                            done_calls = [c for c in done_calls if c in calls]
                        text["answer"], stopped = text["answer"][:cut], True
                        req.stop_calls = list(calls)
                        req.stop_text = text["answer"]  # first: the engine's thread keeps the conversation with it
                        req.cancelled = True
                    upto = len(text["answer"]) if stopped else len(text["answer"]) - hold
                    if stream and upto > sent:
                        chunk({"content": text["answer"][sent:upto]})
                        sent = upto
                for call in done_calls:  # a call goes out whole once its block has closed
                    if stream:
                        chunk({"tool_calls": [{"index": calls.index(call), **T.public(call)}]})
                if last is not None:
                    event = last
                    if event[0] == "error":
                        if stream:
                            self.wfile.write(b"data: " + json.dumps({"error": {"message": event[2], "type": "server_error"}}).encode() + b"\n\n")
                            break
                        return self._error(event[1], event[2])
                    result = event[1]
                    break
            if event[0] == "done":
                if tooled and not stopped:  # a block the reply's end left open: read as it is
                    piece = pieces.end()
                    whole = text["answer"] + (piece[1] if piece is not None else "")
                    ended = pieces.take()
                    text["answer"] = C.cut_at_stops(whole, stops)
                    if len(text["answer"]) < len(whole):  # a stop string in that block: the answer ends before it, as the kept conversation does
                        stopped = True
                        kept = T.kept_before(calls + ended, whole, stops)
                        ended = [c for c in ended if c in kept]
                    for call in ended:
                        calls.append(call)
                        if stream:
                            chunk({"tool_calls": [{"index": len(calls) - 1, **T.public(call)}]})
                if tooled and pieces.malformed:  # (counted, not quoted: the block and why it did not parse are the model's text: ZDR)
                    print(f"[serve] {head['id']}: {len(pieces.malformed)} tool call block(s) did not parse: left in the content", flush=True)
                undeclared = [c["function"]["name"] for c in calls if c["function"]["name"] not in {t["function"]["name"] for t in req.tools or []}]
                if undeclared:  # returned as the model wrote them: the client decides (the names are the model's text: counted)
                    print(f"[serve] {head['id']}: {len(undeclared)} call(s) of a function not declared: returned as the model wrote them", flush=True)
                if stream and len(text["answer"]) > sent:
                    chunk({"content": text["answer"][sent:]})
                finish = "stop" if stopped else result["finish_reason"]
                if calls and finish == "stop":
                    finish = "tool_calls"
                usage = C.with_reasoning(result["usage"], reasoning.count)
                if not stream:
                    message = {"role": "assistant", "content": text["answer"]}
                    if calls:
                        message["content"] = text["answer"] if text["answer"].strip() else None
                        message["tool_calls"] = [T.public(c) for c in calls]
                    if text["think"]:
                        message["reasoning_content"] = text["think"]
                    choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": finish}
                    return self._json(200, {**head, "object": "chat.completion", "choices": [choice], "usage": usage,
                                            "timings": result["timings"]},
                                      {"x-request-id": head["id"], "openai-processing-ms": str(result["timings"]["total_ms"]), **named})
                chunk({}, finish, timings=result["timings"])
                if C.include_usage(body):
                    data = {**head, "object": "chat.completion.chunk", "choices": [], "usage": usage}
                    self.wfile.write(b"data: " + json.dumps(data).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (ConnectionError, OSError):  # the client left: its text ends at the next pass
            req.cancelled = True
            self.close_connection = True
        except Exception as e:  # noqa: BLE001  (a fault here: the client is told what it can be, the engine stops the reply)
            C.trace(e)  # the frames and the type, not the message (ZDR)
            print(f"[serve] {head['id']}: the reply could not be sent: {C.why(e)}", flush=True)
            A.failed(self, e)  # logged as the server's failure, not as the client leaving
            self.close_connection = True
            try:
                if stream:
                    self.wfile.write(b"data: " + json.dumps({"error": {"message": f"the server failed while sending the reply: {type(e).__name__}",
                                                                       "type": "server_error"}}).encode() + b"\n\ndata: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    self._error(500, f"the server failed while sending the reply: {type(e).__name__}")
            except OSError:
                pass
        finally:
            if event is None or event[0] not in ("done", "error"):  # the reply is not over, and nobody will read it
                req.cancelled = True
            A.ended(self, req, event, finish)  # the request log, the histograms


def conversations_at(eng: E.Engine, lengths) -> dict:
    """How many conversations of each length the engine's pool holds at once, by its tiles (the sliding layers' ring
    and extra tiles, a full-attention tile every 16 tokens)."""
    out = {}
    for n in lengths:
        need = {t: (min(-(-n // E.TILE), eng.ring[t]) + eng.extra[t] if t == M.SLIDING else -(-n // E.TILE)) for t in eng.ring}
        out[n] = min(eng.pool[t] // need[t] for t in eng.ring)
    return out


def pool_for(cfg: M.Config, budget: int, full_per_sliding: float, keyless: bool = False, kv4: bool = False) -> dict:
    """Tiles of keys and values for `budget` bytes: `full_per_sliding` tiles of the full-attention layers for each of
    the sliding ones (a conversation needs one of each for every 16 tokens up to the sliding window, 1,024, and only
    full-attention ones after that; with kv4 a sliding tile more for its first tile's int8)."""
    unit = E.tile_bytes(cfg, M.SLIDING, kv4=kv4) + full_per_sliding * E.tile_bytes(cfg, M.FULL, keyless)
    n = int(budget // unit)
    return {M.SLIDING: n, M.FULL: int(n * full_per_sliding)}


def parser() -> argparse.ArgumentParser:
    """The server's options (serve.toml takes the same names: scripts/serve_admin.py parse_args)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on (this machine only by default; 0.0.0.0 lets others in: "
                    "set SCLMD_API_KEY, else there is no password)")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--name", default="v1", help="the codes' folder under C:\\models\\sclmd-codes")
    ap.add_argument("--kv-gib", type=float, default=0.0, help="VRAM for the conversations' keys and values (0: all that 3 bits a weight "
                    "leaves, about 2.4 GiB). A conversation takes 0.46 MB a token up to 1,024 tokens and 41 KB a token after (21 KB with --keyless); what is "
                    "not set aside here goes to 4-bit matrices (quality)")
    ap.add_argument("--context", type=int, default=8192, help="tokens one conversation can hold (prompt and reply)")
    ap.add_argument("--sequences", type=int, default=32, help="conversations held at once, those kept for their next turn included")
    ap.add_argument("--ram-gib", type=float, default=24.0, help="RAM for conversations that wait for their next turn after the GPU "
                    "has run short (0: they are forgotten instead and read again from the start)")
    ap.add_argument("--prefix-gib", type=float, default=0.0, help="page-locked RAM for shared prompt prefixes (system turns, "
                    "instructions, few-shot examples): read once, then copied onto each request that begins with one (0: off). "
                    "A prefix takes 0.42 MB a token up to 1,024 tokens; the same reply tokens as reading it")
    ap.add_argument("--typical", type=int, default=2048, help="tokens of a typical conversation: how the keys and values are split "
                    "between the two kinds of layer")
    ap.add_argument("--max-tokens", type=int, default=4096, help="the most tokens of a reply when the request does not say (thinking included)")
    ap.add_argument("--think", action="store_true", help="think unless the request says otherwise")
    ap.add_argument("--head", type=int, default=8)
    ap.add_argument("--no-drafter", action="store_true")
    ap.add_argument("--rows", type=int, default=32, choices=(8, 16, 32), help="the most rows of a pass: tokens of all requests together")
    ap.add_argument("--wide-rows", type=int, default=0, choices=(0, 64, 128, 256), help="passes of up to this many rows while prompts "
                    "are being read (0: off): several requests' prompts in one pass, the rows that write at its end; the same "
                    "replies. They pay with --int8 (256 rows: 1.3, 1.6 and 1.8 times the requests a second of 16 clients with "
                    "prompts of 50, 500 and 2,000 tokens; fp16 products: 1.00-1.15 times; with --kv4 too; design.md 6.13). The "
                    "engine's work memory grows (256 rows: 81 MiB, 128 rows: 35, computed), and a conversation of more than 1,056 "
                    "tokens keeps that many positions past the sliding window for 32 (256 rows: up to 89 MiB more, 46 with --kv4; "
                    "128 rows: 38): fewer conversations fit")
    ap.add_argument("--reserve", type=float, default=VRAM_MARGIN / 2**30, help="GiB of the VRAM that is free now to leave to other programs")
    ap.add_argument("--keyless", action="store_true", help="long conversations: the ten full-attention layers keep their values only and "
                    "the keys are made of them when read: 21 KB a token past 1,024 tokens for 41 (a conversation of 40,960 tokens 1.2 GiB "
                    "for 2.0); other bits than without it, of the same precision")
    E.add_int8_option(ap)
    E.add_kv4_option(ap)
    C.add_options(ap)
    A.add_options(ap)
    K.add_options(ap)
    serve_ops.add_options(ap)
    return ap


def main() -> None:
    C.quiet_hooks()  # ZDR: an exception that escapes a thread prints its frames and type, not its message
    ap = parser()
    cli = A.parse_args(ap)  # serve.toml's options (--config) under the command line's
    keys = A.start(cli)  # the API keys; a host other than loopback without one: a warning (--require-key: no start)
    K.check(cli)
    E.use_int8_option(cli)  # the whole engine: every request's rows on the integer products, so a reply's bits do not depend on its neighbours
    kv4 = E.use_kv4_option(cli)
    import sys

    import transformers
    from transformers import AutoTokenizer

    sys.setswitchinterval(0.0005)  # the engine's thread does not wait 5 ms for an HTTP thread that holds the interpreter

    transformers.logging.set_verbosity_error()
    reserve = int(cli.reserve * 2**30)
    early = serve_ops.bind_early(cli)  # --crash-only: the port answers 503 "starting" while the engine loads
    print("[serve] loading (about 15 s) ...", flush=True)
    cap_vram(margin=reserve)
    tok = serve_ops.Later(lambda: AutoTokenizer.from_pretrained(ckpt("orig")), cli.image != "off")  # --image: while the image is read
    cfg = M.Config.load(ckpt("orig"))
    context = -(-cli.context // E.TILE) * E.TILE
    # the VRAM that is free now, less what is left to other programs, the head, the work memory, the drafter and room
    # to load in: that much is shared between the layers' codes and the conversations' keys and values
    free = torch.cuda.mem_get_info()[0] - reserve
    rows = cli.wide_rows or cli.rows  # the engine's: the widest pass
    widths = tuple(w for w in E.ROWS if w <= cli.rows) + tuple(w for w in E.ROWS if E.ATT_ROWS < w <= cli.wide_rows)
    rest = E.fixed_bytes(cfg, cli.head, context, pool={M.SLIDING: 0, M.FULL: 0}, rows=rows, keyless=cli.keyless, kv4=kv4) + (0 if cli.no_drafter else E.DRAFTER_BYTES)
    both = free - rest - LOAD_ROOM
    least = P.code_bytes(cfg, P.uniform(cfg, 6))
    if both - least < (256 << 20):
        raise SystemExit(f"not enough VRAM: {(free + reserve) / 2**30:.1f} GiB are free, and 3 bits a weight with a little room for "
                         f"conversations needs {(least + rest + LOAD_ROOM + reserve) / 2**30 + 0.25:.1f} GiB")
    kv = min(int(cli.kv_gib * 2**30), both - least) if cli.kv_gib > 0 else both - least
    plan, estimate = P.fit(cfg, cli.name, both - kv, of=SHARES) if both - kv > least else (P.uniform(cfg, 6), None)
    typical = max(cli.typical, cfg.window)
    pool = pool_for(cfg, kv, typical / (cfg.window + E.SPARE + (E.TILE if kv4 else 0)), cli.keyless, kv4)
    with torch.no_grad():
        eng = E.Engine.load(cli.name, plan, cli.head, context, sequences=cli.sequences, pool=pool, rows=rows, keyless=cli.keyless, kv4=kv4,
                            image=serve_ops.image_mode(cli, estimate is not None))
        drafter = None if cli.no_drafter else E.EngineDrafter.load(eng, ckpt("drafter"), cli.name, depth=DEPTH)
    tok = tok.get()
    stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
    server = Server(tok, eng, drafter, stop, ram=int(cli.ram_gib * 2**30), prefix_ram=int(cli.prefix_gib * 2**30), widths=widths)
    ops = serve_ops.install(server, cli)  # --crash-only (None without it)
    bits = round(P.bits_per_weight(cfg, plan), 2)
    one = (E.tile_bytes(cfg, M.SLIDING, kv4=kv4) * (E.ring_tiles(cfg, M.SLIDING, context, rows) + E.extra_tiles(cfg, M.SLIDING, kv4))
           + E.tile_bytes(cfg, M.FULL, cli.keyless) * typical // E.TILE)
    facts = {"bits_per_weight": bits, "kl_estimate": None if estimate is None else round(estimate, 3),
             "vram_gib": round(torch.cuda.memory_allocated() / 2**30, 2), "drafter": drafter is not None}
    if cli.wide_rows:  # (only then: the facts of a server without it stay as they were)
        facts["wide_rows"] = cli.wide_rows
    app = App(server, tok, cli.think, cli.max_tokens, facts)
    C.apply(cli, app, Handler)
    engine_thread = threading.Thread(target=server.run, daemon=True)
    engine_thread.start()
    httpd = C.BoundedHTTPServer((cli.host, cli.port), Handler, cli.max_connections) if early is None else early
    httpd.app = app
    A.install(app, server, cli, httpd, keys)  # /metrics, keys, back-pressure, the request log and status file (scripts/serve_admin.py)
    K.install(app, server, cli)  # --prompt-cache (scripts/serve_cache.py)
    if early is not None:  # --crash-only: the port already serves its placeholder; the real handler only once keys and limits are set
        serve_ops.serving(early, Handler, app)
    serve_ops.started(app, ops, engine_thread)  # --crash-only's watchdog; --image auto: the image written while it serves
    print(f"[serve] {MODEL} at http://{cli.host}:{cli.port}/v1 · {bits} bits a weight"
          + (f" (KL about {estimate:.3f} on its own text)" if estimate else " (KL about 0.10 on its own text)")
          + f" · VRAM {facts['vram_gib']} GiB\n[serve] keys and values {kv / 2**30:.2f} GiB: about {kv // one} conversations of {typical} tokens "
          f"at once, or {kv // (E.tile_bytes(cfg, M.SLIDING, kv4=kv4) + E.tile_bytes(cfg, M.FULL, cli.keyless)) * E.TILE} tokens of shorter ones; "
          f"a conversation holds up to {context} tokens; thinking {'on' if cli.think else 'off'} unless asked"
          + (f"\n[serve] passes of up to {cli.wide_rows} rows while prompts are read (widths {', '.join(map(str, widths))}; "
             f"the engine's work memory and sliding rings for {rows} rows)" if cli.wide_rows else ""), flush=True)
    print(f"[serve] the pool's tiles {dict(eng.pool)}" + (" (keys and values of the sliding layers at 4 bits: --kv4)" if kv4 else "")
          + ": conversations at once of " + ", ".join(f"{n:,} tokens {c}" for n, c in conversations_at(eng, (200, 500, 1000, 2000, 8000)).items())
          + f" (and at most {eng.sequences} sequences)", flush=True)
    if ops is not None:  # --crash-only: the port serves in a thread of its own; Ctrl+C or Ctrl+Break stops it gracefully
        return serve_ops.until_stopped(ops, cli.drain_s)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
