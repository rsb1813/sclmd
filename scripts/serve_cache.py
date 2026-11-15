# The prompt caches of the internet server (N3; design.md 6.16 and 6.19). All of it is off unless --prompt-cache:
# without it the server keeps conversations and shared prefixes as before (scripts/serve.py: Server._place, Prefixes)
# and a request's cache_control is read nowhere, as before.
#
# With --prompt-cache:
# - Every reuse is the request's own scope's (req.scope, scripts/serve_net.py: the tenant; None when tenancy is off,
#   and then shared as before): a kept conversation, a parked one, a prefix state or a cache entry of another scope is
#   never found. A lookup only hashes the request's own tokens with its own scope, so what other scopes hold does not
#   change how long it takes.
# - Implicit, 15 minutes since last use: the conversations kept on the GPU and parked in RAM, the prefix states (the
#   system turn, sclmd_prefix_messages; --prefix-gib's), and with --cache-turns the state at the end of the messages
#   before the last user message when that is at least 1,024 tokens (a new question on the same document).
# - Explicit: cache_control {"type": "ephemeral", "ttl": "15m" | "1h"} on a content part (or on a message): the state
#   at the end of that part in the rendered prompt (the common token prefix of the prompt cut after the part and the
#   whole one). Up to 4 a request; a prefix under 1,024 tokens is not cached; both are named in x-sclmd-ignored.
#   "15m" (and "5m", and no ttl): in RAM; "1h": in RAM and in the disk cache (--cache-dir), each an hour since last use.
# - One page-locked RAM budget (--cache-ram-gib) for parked conversations, prefix states and cache entries; the one
#   used longest ago goes first. Page-locked: a pageable tensor of exactly that size registered with CUDA
#   (cudaHostRegister); torch's pinned tensors are rounded up to a power of two and stay pinned when freed (measured).
# - The disk cache: the states' key and value tensors (and their layout: positions and shapes), never token ids or
#   text; a file's name is an HMAC of (scope, tokens), its contents AES-GCM in 16 MiB pieces; both keys are made at
#   start and live in this process only, so the folder's files of a past run cannot be read and are deleted at start.
#   A file goes at its time, or least recently used first past --cache-disk-gib. Read into RAM on the request's HTTP
#   thread, before the request goes to the engine (the engine's thread never waits for the disk); written by a thread
#   of its own.
# - Usage: prompt_tokens_details.cached_tokens (from any cache, as before) and cache_write_tokens (the tokens a saved
#   state covers past what was read from a cache).
# Logs carry counts, sizes and times only (ZDR).
import array
import hashlib
import hmac
import json
import math
import os
import queue
import re
import threading
import time

IMPLICIT_S = 15 * 60
TTLS = {None: IMPLICIT_S, "5m": IMPLICIT_S, "15m": IMPLICIT_S, "1h": 60 * 60}  # "5m" (Anthropic's default) is kept as 15m
LONG_S = TTLS["1h"]
LEAST = 1024  # a breakpoint's prefix is cached from this many tokens
TURN_NEAR = 64  # --cache-turns saves no state within this many tokens past one the request read from or its user holds
MOST = 4  # breakpoints a request
PIECE = 16 << 20  # AES-GCM's pieces
MAGIC = b"SCLMDKV1"
FILE = re.compile(r"^[0-9a-f]{64}\.kv$")
TEMP = re.compile(r"^[0-9a-f]{64}\.kv\.tmp$")
SWEEP_S = 1.0  # the engine's thread looks for what is past its time at most this often
IDLE_S = 5.0  # an idle engine thread turns this often (Server.idle_wait), to let go of what is past its time
JANITOR_S = 5.0  # the disk thread deletes files past their time this often
GIB = 1 << 30


def _refused(*args):
    from scripts import serve_contract as C

    return C.Refused(*args)


# page-locked RAM

class Locked:
    """n bytes of page-locked RAM: a pageable tensor registered with CUDA, unregistered when the last tensor made of it
    is freed. Without CUDA (or when registering fails) it stays pageable (ok False; the first time says so), and the
    caches keep nothing in it (locked())."""

    told = False

    def __init__(self, n: int):
        import torch

        self.buf = torch.empty(max(int(n), 1), dtype=torch.uint8)
        self.ok = False
        try:
            if torch.cuda.is_available():
                self.ok = int(torch.cuda.cudart().cudaHostRegister(self.buf.data_ptr(), self.buf.numel(), 0)) == 0
        except Exception:  # noqa: BLE001
            self.ok = False
        if not self.ok and not Locked.told:
            Locked.told = True
            print("[serve] cache: RAM could not be page-locked (cudaHostRegister): such a state is not kept", flush=True)

    def __del__(self):
        if self.ok:
            try:
                import torch

                torch.cuda.cudart().cudaHostUnregister(self.buf.data_ptr())
            except Exception:  # noqa: BLE001
                pass


def alloc(specs: list) -> list:
    """Tensors of the given [(shape, dtype)] in one page-locked block (Engine.park and Engine.save, a kind of layer at a
    time). Each tensor keeps its block alive."""
    sizes = [math.prod(shape) * dtype.itemsize for shape, dtype in specs]
    offsets, at = [], 0
    for s in sizes:
        offsets.append(at)
        at += -(-s // 64) * 64
    block = Locked(at)
    out = []
    for (shape, dtype), o, s in zip(specs, offsets, sizes):
        t = block.buf[o : o + s].view(dtype).view(tuple(shape))
        t._locked = block
        out.append(t)
    return out


def locked(state: dict) -> bool:
    """Whether every tensor of a state is in page-locked RAM made by alloc (6.16: the RAM caches do not go to the page
    file; a state that is not is not kept)."""
    return all(getattr(getattr(d, "_locked", None), "ok", False) for part in state["kinds"].values() for d in part["data"])


def nbytes(state: dict) -> int:
    return state["bytes"]


# names: HMAC of (scope, tokens)

class Names:
    """The HMAC that names a state by (scope, tokens), for the RAM store and the disk cache. Its key is made here and
    lives in this process only."""

    def __init__(self):
        self.key = os.urandom(32)

    def _base(self, scope):
        s = b"\xff\xff\xff\xff" if scope is None else len(str(scope).encode()).to_bytes(4, "little") + str(scope).encode()
        return hmac.new(self.key, b"sclmd-kv1\0" + s, hashlib.sha256)

    def of(self, scope, tokens, lengths) -> dict:
        """{n: the name of (scope, tokens[:n])} for each n in lengths (one pass over the tokens)."""
        ns = sorted({int(n) for n in lengths if 0 < n <= len(tokens)})
        if not ns:
            return {}
        a = array.array("i", tokens[: ns[-1]])
        assert a.itemsize == 4
        buf = memoryview(a.tobytes())
        h, at, out = self._base(scope), 0, {}
        for n in ns:
            h.update(buf[4 * at : 4 * n])
            at = n
            out[n] = h.copy().hexdigest()
        return out

    def one(self, scope, tokens) -> str:
        return self.of(scope, tokens, [len(tokens)])[len(tokens)]


# the RAM store

class Entry:
    """A state at a token prefix in page-locked RAM: put on a new sequence as often as asked (Engine.unpark), the copy
    stays. kind: "prefix" (the shared part), "turn" (--cache-turns), "mark" (cache_control). It keeps the prefix's
    length and its name (the HMAC of the scope and the tokens), never the tokens: those are a prompt's content, and
    only the keys and values are in page-locked memory (ZDR, 6.16). A request's own tokens are matched by the name."""

    __slots__ = ("digest", "scope", "n", "state", "used", "ttl", "kind")

    def __init__(self, digest, scope, n, state, ttl, kind, used):
        self.digest, self.scope, self.n, self.state, self.ttl, self.kind, self.used = digest, scope, int(n), state, ttl, kind, used


class Store:
    """Prefix states and cache entries by (scope, tokens), with scripts/serve.py's Prefixes interface (find, wants, put,
    bytes, held, too_big, hits, stored, waits, ram); their bytes and the parked conversations' share `ram` (fit)."""

    def __init__(self, ram: int, names: Names, clock=time.time, least: int = 16):
        self.ram, self.names, self.clock, self.least = int(ram), names, clock, least
        self.held = {}  # name -> Entry
        self.scopes = {}  # scope -> {n: entries of n tokens}
        self.too_big = set()  # names of prefixes larger than the whole budget: never saved
        self.hits = self.stored = self.waits = 0
        self.expired = self.evicted = 0
        self.on_hit = None  # called with an entry found (Cache: an hour's entry's file counts as used)
        self.lock = threading.RLock()

    def _add(self, e: Entry) -> None:
        old = self.held.get(e.digest)
        if old is not None:
            self._drop(old)
        self.held[e.digest] = e
        lengths = self.scopes.setdefault(e.scope, {})
        lengths[e.n] = lengths.get(e.n, 0) + 1

    def _drop(self, e: Entry) -> None:
        if self.held.get(e.digest) is e:
            del self.held[e.digest]
            lengths = self.scopes.get(e.scope, {})
            lengths[e.n] = lengths.get(e.n, 1) - 1
            if lengths[e.n] <= 0:
                del lengths[e.n]
            if not lengths:
                self.scopes.pop(e.scope, None)

    def longest(self, scope, full: list, longer_than: int = 0):
        """The longest entry of the scope that `full` begins with, longer than longer_than and shorter than full."""
        with self.lock:
            ns = [n for n in self.scopes.get(scope, {}) if longer_than < n < len(full)]
            if not ns:
                return None
            names = self.names.of(scope, full, ns)  # one pass over the request's own tokens, a name at each length held
            for n in sorted(ns, reverse=True):
                e = self.held.get(names[n])
                if e is not None and e.scope == scope and e.n == n:
                    return e
            return None

    def find(self, full: list, longer_than: int, scope=None):
        """Prefixes.find, in the scope: -> (tokens, state) or None; the entry counts as used now. The tokens are the
        request's own first n (the name says they are the entry's)."""
        with self.lock:
            e = self.longest(scope, full, longer_than)
            if e is None:
                return None
            e.used = self.clock()
            if self.on_hit is not None:
                self.on_hit(e)
            return list(full[: e.n]), e.state

    def get(self, scope, tokens):
        with self.lock:
            e = self.held.get(self.names.one(scope, tokens))
            return e if e is not None and e.scope == scope and e.n == len(tokens) else None

    def wants(self, tokens: list, scope=None) -> bool:
        if self.ram <= 0 or len(tokens) < self.least:
            return False
        with self.lock:
            name = self.names.one(scope, tokens)
            return name not in self.held and name not in self.too_big

    def put(self, tokens: list, state: dict, scope=None, ttl: float = IMPLICIT_S, kind: str = "prefix", parked=None):
        """-> the Entry, or None when the state is larger than the whole budget. Then the budget is kept (fit)."""
        name = self.names.one(scope, tokens)
        with self.lock:
            if nbytes(state) > self.ram:
                self.too_big.add(name)
                return None
            e = Entry(name, scope, len(tokens), state, ttl, kind, self.clock())
            self._add(e)
            self.stored += 1
            self.fit(parked or [])
            return e if self.held.get(name) is e else None

    def bytes(self) -> int:
        with self.lock:
            return sum(nbytes(e.state) for e in self.held.values())

    def fit(self, parked: list) -> int:
        """While the parked conversations and the entries hold more than `ram`, the one used longest ago goes (a parked
        one is forgotten: read again from its start next time) -> how many went."""
        gone = 0
        with self.lock:
            total = self.bytes() + sum(nbytes(p.state) for p in parked)
            while total > self.ram and (parked or self.held):
                old_p = min(parked, key=lambda p: p.used, default=None)
                old_e = min(self.held.values(), key=lambda e: e.used, default=None)
                if old_e is None or (old_p is not None and old_p.used <= old_e.used):
                    parked.remove(old_p)
                    total -= nbytes(old_p.state)
                else:
                    self._drop(old_e)
                    total -= nbytes(old_e.state)
                self.evicted += 1
                gone += 1
        return gone

    def sweep(self, now: float) -> int:
        """The entries past their time go -> how many."""
        gone = 0
        with self.lock:
            for e in [e for e in self.held.values() if now - e.used > e.ttl]:
                self._drop(e)
                gone += 1
        self.expired += gone
        return gone


# the disk cache

def _dtype(name: str):
    import torch

    return getattr(torch, name.split(".")[-1])


def _bytes_of(t):
    """A contiguous tensor's bytes, without a copy."""
    import torch

    return memoryview(t.reshape(-1).view(torch.uint8).numpy())


class Disk:
    """The 1 h cache on disk: (name -> its scope, length, bytes, last use, time to live) in this process's memory; in the
    folder only the files <name>.kv. A file: MAGIC, an 8-byte nonce prefix, the count of pieces, then each piece's length
    and its AES-GCM text; piece 0 is the layout (length, high, the tiles' places and the tensors' shapes), the others
    the tensors' bytes. A piece's nonce is the prefix and its number; its associated data the name, its number and the
    count (a piece cannot move to another file or place)."""

    def __init__(self, folder: str, budget: int, names: Names, clock=time.time):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        self.folder, self.budget, self.names, self.clock = os.path.abspath(folder), int(budget), names, clock
        self.aes = AESGCM(AESGCM.generate_key(bit_length=256))  # in this process only
        self.index = {}  # name -> [scope, n, bytes, last used, ttl]
        self.scopes = {}  # scope -> {n: files}
        self.lock = threading.Lock()
        self.reading = set()  # names being read (not deleted under the reader: Windows cannot delete an open file)
        self.doomed = set()  # names to delete once nobody reads them
        self.jobs = queue.Queue(maxsize=2)  # states waiting to be written (each holds its RAM until written)
        self.written = self.read_n = self.deleted = self.skipped = self.failed = 0
        self.write_s = self.read_s = 0.0
        self.written_bytes = self.read_bytes = 0
        os.makedirs(self.folder, exist_ok=True)
        self.removed, self.others = self.empty()
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name="sclmd-cache-disk", daemon=True)
        self.thread.start()

    def path(self, name: str) -> str:
        return os.path.join(self.folder, name + ".kv")

    def empty(self) -> tuple:
        """Deletes this cache's files in the folder (only names of 64 hex digits and .kv that begin with MAGIC, and
        their .kv.tmp; nothing else, and no folder within) -> (deleted, other entries left)."""
        removed = others = 0
        for entry in os.scandir(self.folder):
            ours = entry.is_file(follow_symlinks=False) and (TEMP.match(entry.name) or FILE.match(entry.name))
            if ours and FILE.match(entry.name):
                try:
                    with open(entry.path, "rb") as f:
                        ours = f.read(len(MAGIC)) == MAGIC
                except OSError:
                    ours = False
            if not ours:
                others += 1
                continue
            try:
                os.remove(entry.path)
                removed += 1
            except OSError:
                others += 1
        return removed, others

    def bytes(self) -> int:
        with self.lock:
            return sum(v[2] for v in self.index.values())

    def _add(self, name, scope, n, size, ttl) -> None:
        self.index[name] = [scope, n, size, self.clock(), ttl]
        lengths = self.scopes.setdefault(scope, {})
        lengths[n] = lengths.get(n, 0) + 1

    def _forget(self, name) -> None:
        v = self.index.pop(name, None)
        if v is None:
            return
        lengths = self.scopes.get(v[0], {})
        lengths[v[1]] = lengths.get(v[1], 1) - 1
        if lengths[v[1]] <= 0:
            del lengths[v[1]]
        if not lengths:
            self.scopes.pop(v[0], None)

    def _delete(self, name) -> None:
        """(under the lock) Out of the index; the file goes now, or once its reader is done."""
        self._forget(name)
        if name in self.reading:
            self.doomed.add(name)
            return
        try:
            os.remove(self.path(name))
            self.deleted += 1
        except FileNotFoundError:
            pass
        except OSError:
            self.doomed.add(name)

    def has(self, name) -> bool:
        with self.lock:
            return name in self.index

    def longest(self, scope, full: list, longer_than: int = 0):
        """The longest file of the scope that `full` begins with -> (name, n) or None."""
        with self.lock:
            ns = [n for n in self.scopes.get(scope, {}) if longer_than < n < len(full)]
        if not ns:
            return None
        names = self.names.of(scope, full, ns)
        with self.lock:
            for n in sorted(ns, reverse=True):
                v = self.index.get(names[n])
                if v is not None and v[0] == scope and v[1] == n:
                    return names[n], n
        return None

    def touch(self, name, ttl=None) -> None:
        with self.lock:
            v = self.index.get(name)
            if v is not None:
                v[3] = self.clock()
                if ttl is not None:
                    v[4] = max(v[4], ttl)

    def put(self, name, scope, n, state, ttl) -> bool:
        """Queues the state to be written (False: the writer is behind, it is not written)."""
        if self.budget <= 0 or nbytes(state) > self.budget:
            return False
        if self.has(name):
            self.touch(name, ttl)
            return True
        try:
            self.jobs.put_nowait((name, scope, n, state, ttl))
            return True
        except queue.Full:
            self.skipped += 1
            return False

    def _aad(self, name, i, count) -> bytes:
        return MAGIC + bytes.fromhex(name) + i.to_bytes(4, "little") + count.to_bytes(4, "little")

    @staticmethod
    def layout(state: dict) -> dict:
        return {"length": state["length"], "high": state["high"],
                "kinds": {t: {"js": list(part["js"]), "specs": [[list(d.shape), str(d.dtype)] for d in part["data"]]}
                          for t, part in state["kinds"].items()}}

    def write(self, name, scope, n, state, ttl) -> None:
        """(the disk thread) One state to its file."""
        if self.has(name):
            return
        t0 = time.perf_counter()
        views = [_bytes_of(d) for part in state["kinds"].values() for d in part["data"] if d.numel()]
        count = 1 + sum(-(-len(v) // PIECE) for v in views)
        prefix = os.urandom(8)
        path = self.path(name)
        tmp = path + ".tmp"
        size = 0
        with open(tmp, "wb") as f:
            f.write(MAGIC + prefix + count.to_bytes(4, "little"))

            def piece(i, data):
                c = self.aes.encrypt(prefix + i.to_bytes(4, "little"), data, self._aad(name, i, count))
                f.write(len(c).to_bytes(4, "little") + c)
                return len(c) + 4

            size += piece(0, json.dumps(self.layout(state)).encode())
            i = 1
            for v in views:
                for o in range(0, len(v), PIECE):
                    size += piece(i, v[o : o + PIECE])
                    i += 1
        os.replace(tmp, path)
        with self.lock:
            self._add(name, scope, n, size + 20, ttl)
            self.written += 1
            self.written_bytes += size + 20
            self.write_s += time.perf_counter() - t0
            over = self.bytes_locked() - self.budget
            while over > 0 and self.index:  # past the budget: the file used longest ago goes
                old = min(self.index, key=lambda k: self.index[k][3])
                over -= self.index[old][2]
                self._delete(old)
        print(f"[serve] cache: a state of {n:,} tokens, {(size + 20) / GIB:.2f} GiB, written to disk in {time.perf_counter() - t0:.2f} s", flush=True)

    def bytes_locked(self) -> int:
        return sum(v[2] for v in self.index.values())

    def read(self, name, make=alloc):
        """The state of a file into page-locked RAM -> state, or None (gone, or it does not decrypt: then deleted)."""
        from cryptography.exceptions import InvalidTag

        with self.lock:
            if name not in self.index:
                return None
            self.reading.add(name)
        t0 = time.perf_counter()
        try:
            with open(self.path(name), "rb") as f:
                if f.read(len(MAGIC)) != MAGIC:
                    raise ValueError("not a cache file")
                prefix, count = f.read(8), int.from_bytes(f.read(4), "little")

                def piece(i):
                    n = int.from_bytes(f.read(4), "little")
                    return self.aes.decrypt(prefix + i.to_bytes(4, "little"), f.read(n), self._aad(name, i, count))

                lay = json.loads(piece(0))
                state = {"length": lay["length"], "high": lay["high"], "kinds": {}, "bytes": 0}
                i = 1
                for t, k in lay["kinds"].items():
                    data = make([(tuple(s), _dtype(d)) for s, d in k["specs"]])
                    for d in data:
                        if not d.numel():
                            continue
                        v = _bytes_of(d)
                        for o in range(0, len(v), PIECE):
                            plain = piece(i)
                            v[o : o + len(plain)] = plain
                            i += 1
                    state["kinds"][t] = {"js": k["js"], "data": data}
                    state["bytes"] += sum(d.numel() * d.element_size() for d in data)
                if i != count:
                    raise ValueError("pieces missing")
        except (OSError, ValueError, KeyError, InvalidTag) as e:
            with self.lock:
                self.reading.discard(name)
                self.failed += 1
                self._delete(name)
            print(f"[serve] cache: a file could not be read ({type(e).__name__}): deleted", flush=True)
            return None
        with self.lock:
            self.reading.discard(name)
            v = self.index.get(name)
            if v is not None:
                v[3] = self.clock()
            if name in self.doomed:
                self._delete(name)
            self.read_n += 1
            self.read_bytes += state["bytes"]
            self.read_s += time.perf_counter() - t0
        return state

    def sweep(self) -> int:
        """Files past their time go (and those that could not be deleted before) -> how many."""
        now, gone = self.clock(), 0
        with self.lock:
            for name in [k for k, v in self.index.items() if now - v[3] > v[4]]:
                self._delete(name)
                gone += 1
            for name in list(self.doomed):
                if name in self.reading:
                    continue
                try:
                    os.remove(self.path(name))
                    self.deleted += 1
                    self.doomed.discard(name)
                except FileNotFoundError:
                    self.doomed.discard(name)
                except OSError:
                    pass
        return gone

    def _run(self) -> None:
        while not self.stopping:
            try:
                job = self.jobs.get(timeout=JANITOR_S)
            except queue.Empty:
                job = None
            if job is not None:
                try:
                    self.write(*job)
                except Exception as e:  # noqa: BLE001  (a full disk, ...: the entry stays in RAM only)
                    self.failed += 1
                    try:
                        os.remove(self.path(job[0]) + ".tmp")
                    except OSError:
                        pass
                    print(f"[serve] cache: a state could not be written to disk ({type(e).__name__})", flush=True)
                job = None
            try:
                self.sweep()
            except Exception as e:  # noqa: BLE001
                print(f"[serve] cache: the disk cache's sweep failed ({type(e).__name__})", flush=True)

    def close(self) -> None:
        """At exit: this run's files go (they could not be read by another run anyway)."""
        self.stopping = True
        with self.lock:
            for name in list(self.index):
                self._delete(name)


# the request's breakpoints (App.request, under its lock)

def _common(a: list, b: list) -> int:
    n, most = 0, min(len(a), len(b))
    step = 1024
    while n + step <= most and a[n : n + step] == b[n : n + step]:
        n += step
    while n < most and a[n] == b[n]:
        n += 1
    return n


def _ttl_of(cc):
    """A cache_control's time to live in seconds, or None when it is not one served."""
    if not isinstance(cc, dict) or cc.get("type") != "ephemeral":
        return None
    ttl = cc.get("ttl")
    return TTLS.get(ttl) if (ttl is None or isinstance(ttl, str)) else None


def points(app, body: dict, messages: list, tools, think: bool, full: list, strict: bool, turns: bool) -> tuple:
    """The positions in `full` whose states the request asks to be cached -> ([(n, seconds, kind)], names not served).
    A mark's position: the common token prefix of the prompt cut after its part (or message) and the whole prompt.
    messages: what the template wrote (one for each of the request's messages)."""
    raw = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(raw, list) or len(raw) != len(messages):
        return [], []
    kw = {"tools": tools} if tools else {}

    def ids(ms):
        out = app.tok.apply_chat_template(ms, tokenize=True, add_generation_prompt=False, enable_thinking=think, **kw)
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    marks, names = [], []
    for i, m in enumerate(raw):
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, list):
            for j, p in enumerate(content):
                if isinstance(p, dict) and "cache_control" in p:
                    marks.append((i, j, p["cache_control"], f"messages[{i}].content[{j}].cache_control"))
        if "cache_control" in m:
            marks.append((i, None, m["cache_control"], f"messages[{i}].cache_control"))
    out, served = {}, 0
    for i, j, cc, name in marks:
        ttl = _ttl_of(cc)
        if ttl is None:
            if strict:
                raise _refused(400, f"{name}: {{\"type\": \"ephemeral\", \"ttl\": \"15m\" | \"1h\"}} is served (--strict-fields)", name)
            names.append(name)
            continue
        if served >= MOST:
            if strict:
                raise _refused(400, f"{name}: at most {MOST} cache_control breakpoints a request (--strict-fields)", name)
            names.append(name)
            continue
        extra = sorted(str(k) for k in cc if k not in ("type", "ttl"))  # served without them, and they are named (R1)
        if extra and strict:
            raise _refused(400, f"{name}.{extra[0]}: not served (--strict-fields)", f"{name}.{extra[0]}")
        names += [f"{name}.{k}" for k in extra]
        served += 1
        cut = dict(messages[i])
        content = raw[i].get("content")
        if j is not None and isinstance(content, list):
            texts = [p.get("text", "") if isinstance(p, dict) and isinstance(p.get("text", ""), str) else "" for p in content]
            if cut.get("content") == "".join(texts):  # the parts are the message's text, joined (sclmd/tools.py)
                cut["content"] = "".join(texts[: j + 1])
        try:
            n = _common(ids(messages[:i] + [cut]), full)
        except Exception:  # noqa: BLE001  (the template cannot write it cut there: not cached)
            n = 0
        if n < LEAST or n >= len(full):
            names.append(name)
            continue
        out[n] = max(out.get(n, 0), ttl)
    result = [(n, s, "mark") for n, s in sorted(out.items())]
    if turns:
        last = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=0)
        if last > 0:
            # cheap first: a token is at least a byte of the text, and the template adds a few a message
            size = sum(len(str(m.get("content") or "").encode()) + len(json.dumps(m.get("tool_calls") or [])) for m in messages[:last])
            size += len(json.dumps(tools or [])) + 16 * last
            if size >= LEAST:
                try:
                    n = _common(ids(messages[:last]), full)
                except Exception:  # noqa: BLE001
                    n = 0
                if LEAST <= n < len(full) and n not in out:
                    result.append((n, IMPLICIT_S, "turn"))
    return sorted(result), sorted(set(names))


# the engine's side (scripts/serve.py's Server calls in)

class Cache:
    def __init__(self, ram: int, folder: str = "", disk: int = 0, turns: bool = False, clock=time.time):
        self.names = Names()
        self.store = Store(ram, self.names, clock)
        self.disk = Disk(folder, disk, self.names, clock) if folder else None
        self.turns, self.clock = turns, clock
        self.arrivals = queue.Queue()  # states read from disk by HTTP threads, for the engine's thread to take in
        self.loading = threading.Lock()  # one read from disk at a time
        self.coming = set()  # names read from disk and not yet taken in
        self.swept = 0.0
        self.saves = self.save_bytes = self.skipped = self.unlocked = self.from_disk = self.expired_kept = self.expired_parked = 0
        self.near = 0  # --cache-turns' points not saved: a few tokens past a state already there (TURN_NEAR)
        if self.disk is not None:  # a hit in RAM on an hour's entry is a use of its file too
            self.store.on_hit = lambda e: self.disk.touch(e.digest) if e.ttl >= LONG_S else None
        self.save_s = 0.0

    # the HTTP thread

    def prefetch(self, req) -> None:
        """Before the request goes to the engine: the longest file of its scope that its prompt begins with, when no
        state in RAM is as long, is read into page-locked RAM and handed to the engine's thread (arrivals)."""
        if self.disk is None or not req.full:
            return
        scope, full = req.scope, req.full
        e = self.store.longest(scope, full)
        hit = self.disk.longest(scope, full, e.n if e is not None else 0)
        if hit is None:
            return
        name, n = hit
        with self.loading:
            with self.store.lock:
                if name in self.store.held or name in self.coming:
                    return
            t0 = time.perf_counter()
            state = self.disk.read(name)
            if state is None or not locked(state):  # (not page-locked: not kept)
                return
            with self.store.lock:
                self.coming.add(name)
            self.arrivals.put(Entry(name, scope, n, state, LONG_S, "mark", self.clock()))
            self.from_disk += 1
            print(f"[serve] cache: a state of {n:,} tokens, {state['bytes'] / GIB:.2f} GiB, read from disk in {time.perf_counter() - t0:.2f} s", flush=True)

    # the engine's thread

    def tick(self, server) -> None:
        """At the start of a turn: the states read from disk come in; at most once a second, what is past its time goes
        (kept and parked conversations: 15 minutes since their last use; entries: their time to live)."""
        while True:
            try:
                e = self.arrivals.get_nowait()
            except queue.Empty:
                break
            with self.store.lock:
                self.coming.discard(e.digest)
                e.used = self.clock()
                self.store._add(e)
                self.store.fit(server.parked)
        now = self.clock()
        if now - self.swept < SWEEP_S:
            return
        self.swept = now
        for k in [k for k in server.kept.values() if now - k.used > IMPLICIT_S]:
            server._forget(k)
            self.expired_kept += 1
        old = [p for p in server.parked if now - p.used > IMPLICIT_S]
        for p in old:
            server.parked.remove(p)
        self.expired_parked += len(old)
        self.store.sweep(now)

    def fit(self, server) -> None:
        self.store.fit(server.parked)

    def arm(self, server, req) -> None:
        """The request has its sequence and job: the states it asks for (its prefix state, cache_control's marks, the
        boundary before its last user message) that are not held yet are saved as its prompt is read (the job pauses
        at each); those held already count as used (and as wanted for longer)."""
        store, full, scope, job = self.store, req.full, req.scope, req.job
        want = {}
        if req.cut and req.cut < len(full):
            want[req.cut] = (IMPLICIT_S, "prefix")
        for n, ttl, kind in getattr(req, "points", None) or ():
            if n not in want or ttl > want[n][0]:
                want[n] = (ttl, kind)
        if not want or store.ram <= 0:
            return
        names = self.names.of(scope, full, list(want))
        later = []
        for n in sorted(want):
            ttl, kind = want[n]
            if n < store.least or n >= len(full):
                continue
            e = store.held.get(names[n])
            if e is not None and e.scope == scope:
                with store.lock:
                    e.used, e.ttl = self.clock(), max(e.ttl, ttl)
                if ttl >= LONG_S and self.disk is not None:
                    self.disk.put(e.digest, scope, n, e.state, ttl)
                continue
            if self.disk is not None and self.disk.has(names[n]):
                self.disk.touch(names[n], ttl)
            if names[n] in store.too_big or n < req.reused:
                continue
            if kind == "turn" and getattr(req, "went_on", False):
                # --cache-turns does not copy a conversation that the request goes on from (kept or parked): its end is
                # this boundary, give or take the few tokens a reply's text is written again with; a copy at every turn
                # would be 0.3-0.9 GiB a turn
                continue
            if kind == "turn" and ((req.reused > 0 and n - req.reused <= TURN_NEAR)
                                   or store.longest(scope, full[:n], max(0, n - TURN_NEAR - 1)) is not None):
                # nor a state a few tokens past one this user holds already (a marked part's end and its message's
                # boundary, 2 tokens apart, measured): 0.3-0.9 GiB of page-locked RAM for a few tokens of reading
                self.near += 1
                continue
            if server.eng.highs[req.job.seq] > n:  # the sequence has written past it (a reply's ring): it cannot be saved
                self.skipped += 1
                continue
            if n == req.reused:  # the sequence holds exactly this prefix now (a state's end)
                self.save(server, req, n, ttl, kind, names[n])
                continue
            later.append((n, ttl, kind, names[n]))
        if not later:
            return
        job.pause = later[0][0]

        def at_pause(job, later=later):
            here = job.pause
            for n, ttl, kind, name in later:
                if n == here:
                    self.save(server, req, n, ttl, kind, name)
            job.pause = next((n for n, _, _, _ in later if n > here), None)

        job.at_pause = at_pause

    def save(self, server, req, n, ttl, kind, name) -> bool:
        """The job's sequence holds exactly the first n tokens: a copy of its state into page-locked RAM (and to disk
        for an hour's)."""
        from sclmd import engine as E

        eng, seq, store = server.eng, req.job.seq, self.store
        if eng.highs[seq] != n or eng.lengths[seq] != n:
            self.skipped += 1  # positions past it were written (a kept conversation's ring went round): not saved
            return False
        size = sum(E.tile_bytes(eng.cfg, t, eng.lean, eng.kv4) * sum(x >= 0 for x in eng.tiles[t][seq]) for t in eng.ring)
        if size > store.ram:
            with store.lock:
                store.too_big.add(name)
            return False
        t0 = time.perf_counter()
        state = eng.save(seq, alloc=alloc)
        took = time.perf_counter() - t0
        if not locked(state):  # not page-locked: not kept
            self.unlocked += 1
            return False
        e = store.put(req.full[:n], state, req.scope, ttl, kind, server.parked)
        self.save_s += took
        if e is None:
            return False
        if kind != "prefix":  # (a cache_control mark or --cache-turns: 1,024 tokens and more)
            print(f"[serve] cache: a state of {n:,} tokens ({kind}, {ttl // 60:.0f} min), {state['bytes'] / GIB:.2f} GiB, saved in RAM in "
                  f"{took * 1000:.0f} ms", flush=True)
        self.saves += 1
        self.save_bytes += state["bytes"]
        req.wrote = max(getattr(req, "wrote", 0), n - req.reused)
        if ttl >= LONG_S and self.disk is not None:
            self.disk.put(name, req.scope, n, state, ttl)
        return True

    def about(self) -> dict:
        s = self.store
        out = {"entries": len(s.held), "gib": round(s.bytes() / GIB, 3), "budget_gib": round(s.ram / GIB, 2), "saved": self.saves,
               "saved_gib": round(self.save_bytes / GIB, 2), "save_s": round(self.save_s, 3), "not_saved": self.skipped, "not_locked": self.unlocked, "near": self.near, "hits": s.hits,
               "evicted": s.evicted, "expired": s.expired, "expired_kept": self.expired_kept, "expired_parked": self.expired_parked}
        d = self.disk
        if d is not None:
            out["disk"] = {"files": len(d.index), "gib": round(d.bytes() / GIB, 3), "budget_gib": round(d.budget / GIB, 2), "written": d.written,
                           "written_s": round(d.write_s, 2), "read": d.read_n, "read_s": round(d.read_s, 2), "deleted": d.deleted,
                           "not_written": d.skipped, "failed": d.failed, "read_into_ram": self.from_disk}
        return out


# start-up

def add_options(ap) -> None:
    g = ap.add_argument_group("prompt caches (scripts/serve_cache.py; all off by default; design.md 6.19)")
    g.add_argument("--prompt-cache", action="store_true", help="prompt caches per user (the tenant: an API key, an address; shared "
                   "when there is no tenancy): a request never reuses another user's kept conversation, prefix state or cache entry. "
                   "Kept and parked conversations and prefix states go 15 minutes after their last use; cache_control {\"type\": "
                   "\"ephemeral\", \"ttl\": \"15m\" | \"1h\"} on a content part or a message keeps the state at its end (4 a request, "
                   "1,024 tokens at least; else named in x-sclmd-ignored); usage gets prompt_tokens_details.cache_write_tokens. "
                   "One page-locked RAM budget (--cache-ram-gib) takes the place of --ram-gib and --prefix-gib. Without it: as before")
    g.add_argument("--cache-ram-gib", type=float, default=32.0, help="with --prompt-cache: page-locked RAM for parked conversations, "
                   "prefix states and cache entries together; the one used longest ago goes first")
    g.add_argument("--cache-dir", default="", help="with --prompt-cache: the folder of the 1 h disk cache (\"\": none, an hour's entries "
                   "stay in RAM only). Its cache files (and only those) are deleted at start and at exit: give a folder of its own, "
                   "e.g. C:/models/sclmd-cache")
    g.add_argument("--cache-disk-gib", type=float, default=64.0, help="with --cache-dir: the disk cache's size; the file used longest ago goes first")
    g.add_argument("--cache-turns", action="store_true", help="with --prompt-cache: also keep the state at the end of the messages before "
                   "the last user message when that is at least 1,024 tokens, for 15 minutes (a new question on the same document, a "
                   "last message edited): a copy in RAM when the prompt is read through that point, none when the request goes on "
                   "from its own kept or parked conversation")


def check(cli) -> None:
    """Checks made before the model loads."""
    on = getattr(cli, "prompt_cache", False)
    if not on and (getattr(cli, "cache_dir", "") or getattr(cli, "cache_turns", False)):
        raise SystemExit("--cache-dir and --cache-turns are parts of --prompt-cache")
    if getattr(cli, "cache_ram_gib", 0) < 0 or getattr(cli, "cache_disk_gib", 0) < 0:
        raise SystemExit("--cache-ram-gib and --cache-disk-gib are 0 or more")
    folder = getattr(cli, "cache_dir", "")
    if on and folder:
        if os.path.exists(folder) and not os.path.isdir(folder):
            raise SystemExit(f"--cache-dir {folder}: not a folder")
        here = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        if os.path.abspath(folder) in (here, os.path.abspath(os.path.splitdrive(here)[0] + os.sep)):
            raise SystemExit(f"--cache-dir {folder}: give the cache a folder of its own")


def attach(server, cache: Cache) -> Cache:
    """The Cache onto the Server: its store in the place of the prefixes, its budget the parked conversations' too,
    and an idle engine thread that turns every few seconds (to let go of what is past its time)."""
    server.cache = cache
    server.prefixes = cache.store
    server.ram = cache.store.ram
    server.idle_wait = min(server.idle_wait or IDLE_S, IDLE_S)
    return cache


def install(app, server, cli):
    """--prompt-cache onto a running server: server.cache, its store in the place of the prefixes, the one RAM budget,
    the users' scopes (scripts/serve_net.py: Users.per_tenant). -> the Cache, or None without the option."""
    if not getattr(cli, "prompt_cache", False):
        return None
    folder = cli.cache_dir or ""
    cache = attach(server, Cache(int(cli.cache_ram_gib * GIB), folder, int(cli.cache_disk_gib * GIB) if folder else 0, bool(cli.cache_turns)))
    adm = getattr(app, "admin", None)
    users = getattr(adm, "users", None)
    if users is not None:
        users.per_tenant = True
    if cache.disk is not None:
        import atexit

        atexit.register(cache.disk.close)
    apart = users is not None and users.on
    print(f"[serve] prompt caches: {'per user' if apart else 'shared (no API key and no --anonymous: one user)'}; kept conversations, "
          f"prefix states{' and the boundary before the last user message (--cache-turns)' if cache.turns else ''} for 15 min since last use; "
          f"cache_control 15m in RAM, 1h {'in RAM and on disk' if cache.disk is not None else 'in RAM (no --cache-dir)'}; page-locked RAM "
          f"{cli.cache_ram_gib:g} GiB for parked conversations, prefix states and entries (in the place of --ram-gib and --prefix-gib)"
          + (f"; disk {cli.cache_disk_gib:g} GiB at {cache.disk.folder} (cache files of a past run deleted: {cache.disk.removed}"
             + (f"; other entries left as they are: {cache.disk.others}" if cache.disk.others else "") + ")" if cache.disk is not None else ""),
          flush=True)
    return cache


def before(app, req) -> None:
    """do_POST, after the request is tagged (req.scope) and before it goes to the engine: an entry of the disk cache the
    prompt begins with is read into RAM (--prompt-cache with --cache-dir)."""
    cache = getattr(getattr(app, "server", None), "cache", None)
    if cache is None:
        return
    try:
        cache.prefetch(req)
    except Exception as e:  # noqa: BLE001  (the request is then served without it)
        print(f"[serve] cache: reading from disk failed ({type(e).__name__})", flush=True)


def usage(server, req, details: dict) -> None:
    """prompt_tokens_details gets cache_write_tokens with --prompt-cache."""
    if getattr(server, "cache", None) is not None:
        details["cache_write_tokens"] = int(getattr(req, "wrote", 0))
