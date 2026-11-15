# Operating the server (scripts/serve.py calls in through a few hooks; R4). Everything here is off, or answers only
# what nobody asked before, unless an option turns it on; the reply tokens are the same either way.
# - API keys: SCLMD_API_KEY (a key, or name:key pairs with commas) or --api-key-file (a key a line, "name: key" or
#   "key"; keep the file outside the repository, which is pushed to a mirror). A request gives its key as
#   "Authorization: Bearer <key>" or "x-api-key: <key>"; it is compared in constant time with every key. Without a
#   key: 401. GET /health stays open. A --host other than loopback without a key: a loud warning (--require-key: the
#   server does not start).
# - back-pressure (off by default): --max-waiting N answers 429 (Retry-After, retry-after-ms) to a request when N
#   requests already wait for a sequence beyond those the engine has free now (unused, and kept conversations', which
#   give their place); --queue-timeout S answers 503 to one that has waited S seconds for one. With long prompts the
#   keys and values, not the sequences, run short: the queue can then be up to N + the free sequences long.
# - GET /metrics (on: a new endpoint; with a key it needs the key): Prometheus text (counts, histograms of the time to
#   the first token and of the request, pass time by width, copy time of park / unpark / snapshot, pinned RAM, process
#   memory and threads, the queue). Every label takes a value from a fixed set (the method, the path, the status),
#   so no request can add series.
# - --request-log PATH: a JSON line a request (rotated at --request-log-mb); --status-file PATH: the server's state,
#   rewritten every 5 s.
# - serve.toml (--config): every option, the command line over it; keys are never read from it.
# - --debug (a loopback host only): POST /debug/fault arms an injected fault, GET /debug/audit checks the books of the
#   keys and values at idle (scripts/soak.py uses both).
# Quotas and priorities (N2): scripts/serve_quota.py, through the hooks here (submit, ended, get, post, status_headers).
import argparse
import hmac
import ipaddress
import json
import math
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime

from scripts import serve_contract as C
from scripts import serve_net as N
from scripts import serve_quota as Q

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "serve.toml")  # read when it is there (--config another; --config "" none)
ENV_KEY = "SCLMD_API_KEY"
RATE = 500.0  # tokens a second the queue is drained at until the server has measured its own
# The longest Retry-After sent. The openai Python client 3.51.0 (measured: logs/exp/R4-land/retry_after.py) waits
# what retry-after-ms (else Retry-After) says when it is above 0 and at most 120 s; past 120 s it does not retry at
# all (the caller gets the 429 at once), and 0 or none gives its own 0.5-8 s backoff. 60 stays well inside.
RETRY_MOST = 60.0
TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120)
LATENCY_BUCKETS = (0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600)
SECRET = {"api_key", "api_keys", "key", "keys", "sclmd_api_key"}  # never from serve.toml
SERVE_BAT = {"supervise"}  # serve.toml's keys for serve.bat (scripts/supervise.py: wanted), which the server passes over
PATHS = ("/v1/chat/completions", "/v1/models", "/health", "/metrics", "/debug", "/")
# the label values of /metrics: fixed sets, whatever a request says (a made-up method is "other")
METHODS = ("GET", "POST", "HEAD", "PUT", "DELETE", "OPTIONS", "PATCH")
CODES = (200, 400, 401, 404, 405, 408, 409, 411, 413, 414, 415, 422, 429, 431, 500, 501, 502, 503, 504, 505)
FINISHES = ("stop", "length", "tool_calls", "error", "server_error", "client_left", "queue_timeout")
FAULT_BODY = 1 << 16  # bytes of a POST /debug/fault body at most


class InjectedFault(RuntimeError):
    """A fault that was asked for (POST /debug/fault, --debug): the soak tells these apart from real ones by the name."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def method_label(command) -> str:
    return command if command in METHODS else "other"


def code_label(code) -> str:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "other"
    return str(code) if code in CODES else f"{code // 100}xx" if 100 <= code <= 599 else "other"


def finish_label(finish) -> str:
    return finish if finish in FINISHES else "other"


# API keys

_NAMED = re.compile(r"^([A-Za-z0-9_.-]{1,40})\s*:\s*(\S.*)$")
_LIMIT = re.compile(r"^(tph|rph|tpd|rpd)=(\d+)$")  # a key's own quota after it (scripts/serve_quota.py)


def parse_keys(text: str, source: str, limits: dict | None = None) -> list:
    """name:key or key entries, by commas or lines ('#' starts a comment line) -> [(name, key bytes)]. With `limits`
    (--quotas), words after the key that are all tph=N, rph=N, tpd=N or rpd=N are the key's own quota (into `limits`,
    by name); an entry with any other words after the key is the key as a whole. Without it (no --quotas) every entry
    is the key as a whole, as before: the string that authenticates does not change without the option."""
    keys = []
    for part in re.split(r"[,\r\n]", text or ""):
        part = part.strip()
        if not part or part.startswith("#"):
            continue
        m = _NAMED.match(part)
        name, key = (m.group(1), m.group(2).strip()) if m else (f"{source}{len(keys) + 1}", part)
        words = key.split()
        if limits is not None and len(words) > 1 and all(_LIMIT.match(w) for w in words[1:]):
            key = words[0]
            limits[name] = {_LIMIT.match(w).group(1): int(_LIMIT.match(w).group(2)) for w in words[1:]}
        keys.append((name, key.encode()))
    return keys


def load_keys(cli) -> list:
    """The API keys; with --quotas a key's own quota goes to cli.key_limits (name -> {tph, rph, tpd, rpd})."""
    limits = {} if getattr(cli, "quotas", False) else None
    keys = parse_keys(os.environ.get(ENV_KEY, ""), "key", limits)
    if getattr(cli, "api_key_file", ""):
        try:
            with open(cli.api_key_file, encoding="utf-8") as f:
                keys += parse_keys(f.read(), "file", limits)
        except OSError as e:
            raise SystemExit(f"--api-key-file {cli.api_key_file}: {e}")
    names = [n for n, _ in keys]
    if len(set(names)) != len(names):
        raise SystemExit("two API keys have the same name")
    cli.key_limits = limits or {}
    return keys


presented = N.presented  # the key a request gives: Authorization: Bearer <key>, else x-api-key


def key_name(keys: list, got: bytes | None) -> str | None:
    """The name of the key `got` is, compared in constant time with every key; None: none."""
    if got is None:
        return None
    found = None
    for name, key in keys:
        if hmac.compare_digest(got, key) and found is None:
            found = name
    return found


# numbers for /metrics

class Histogram:
    def __init__(self, buckets: tuple):
        self.buckets, self.counts, self.sum, self.count = tuple(buckets), [0] * len(buckets), 0.0, 0
        self.lock = threading.Lock()

    def observe(self, v: float) -> None:
        with self.lock:
            for i, b in enumerate(self.buckets):
                if v <= b:
                    self.counts[i] += 1
            self.sum += v
            self.count += 1

    def samples(self, name: str) -> list:
        with self.lock:
            out = [(f"{name}_bucket", {"le": _num(b)}, c) for b, c in zip(self.buckets, self.counts)]
            return out + [(f"{name}_bucket", {"le": "+Inf"}, self.count), (f"{name}_sum", {}, self.sum), (f"{name}_count", {}, self.count)]


def _num(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    v = float(v)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    return repr(v) if v != int(v) or abs(v) >= 1e15 else str(int(v))


def _labels(labels: dict) -> str:
    if not labels:
        return ""
    esc = lambda s: str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")  # noqa: E731
    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in labels.items()) + "}"


def _safe(fn, default=None, tries: int = 3):
    """fn() read from an HTTP thread while the engine's thread changes what it reads: a RuntimeError (a dict that
    changed size) is tried again; anything else, or failing every time, gives `default`."""
    for _ in range(tries):
        try:
            return fn()
        except RuntimeError:
            continue
        except Exception:  # noqa: BLE001
            return default
    return default


def process_numbers() -> dict:
    """The server process's resident and private bytes and its OS threads (no psutil: ctypes on Windows, /proc else)."""
    out = {}
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                               "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]

        class THREADENTRY32(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ThreadID", wintypes.DWORD),
                        ("th32OwnerProcessID", wintypes.DWORD), ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG), ("dwFlags", wintypes.DWORD)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            out["resident_bytes"], out["private_bytes"] = pmc.WorkingSetSize, pmc.PrivateUsage
        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        k32.Thread32First.argtypes = k32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        snap = k32.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD
        if snap and snap != wintypes.HANDLE(-1).value:
            te, pid, n = THREADENTRY32(), os.getpid(), 0
            te.dwSize = ctypes.sizeof(THREADENTRY32)
            ok = k32.Thread32First(snap, ctypes.byref(te))
            while ok:
                n += te.th32OwnerProcessID == pid
                ok = k32.Thread32Next(snap, ctypes.byref(te))
            k32.CloseHandle(snap)
            out["os_threads"] = n
    else:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        out["resident_bytes"] = int(line.split()[1]) * 1024
                    elif line.startswith("Threads:"):
                        out["os_threads"] = int(line.split()[1])
        except OSError:
            pass
    return out


# the request log and the status file

class RequestLog:
    """JSON lines, written by a thread of its own (the HTTP threads only queue a record); the file is rotated when it
    would pass `most` bytes (path.1 ... path.<keep>)."""

    def __init__(self, path: str, most: int = 64 << 20, keep: int = 3):
        self.path, self.most, self.keep = path, most, keep
        self.q, self.written, self.failed = queue.SimpleQueue(), 0, 0
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        threading.Thread(target=self._run, daemon=True, name="request-log").start()

    def write(self, record: dict) -> None:
        self.q.put(record)

    def _rotate(self) -> None:
        for i in range(self.keep - 1, 0, -1):
            if os.path.exists(f"{self.path}.{i}"):
                os.replace(f"{self.path}.{i}", f"{self.path}.{i + 1}")
        os.replace(self.path, f"{self.path}.1")

    def _run(self) -> None:
        f = None
        while True:
            record = self.q.get()
            try:
                line = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
                if f is None:
                    f = open(self.path, "ab")
                if self.most and os.fstat(f.fileno()).st_size + len(line) > self.most and os.fstat(f.fileno()).st_size > 0:
                    f.close()
                    f = None
                    try:
                        self._rotate()
                    except OSError as e:  # another program holds it open (Windows): go on in the same file
                        print(f"[serve] the request log could not be rotated: {C.why(e)}", flush=True)
                    f = open(self.path, "ab")
                f.write(line)
                f.flush()
                self.written += 1
            except Exception as e:  # noqa: BLE001  (the log never stops the server)
                self.failed += 1
                if self.failed in (1, 10, 100):
                    print(f"[serve] the request log failed ({self.failed}): {C.why(e)}", flush=True)
                try:
                    if f is not None:
                        f.close()
                except OSError:
                    pass
                f = None


# the server's side

class Entry:
    """A request the server has taken in, from submit until its HTTP thread is done with it."""

    __slots__ = ("req", "rid", "key", "stream", "client", "finish", "error")

    def __init__(self, req, rid, key, stream, client):
        self.req, self.rid, self.key, self.stream, self.client, self.finish, self.error = req, rid, key, stream, client, None, None


class Admin:
    def __init__(self, app, server, httpd=None, keys=(), max_waiting: int = 0, queue_timeout: float = 0.0, debug: bool = False,
                 request_log: str = "", request_log_mb: float = 64.0, status_file: str = "", status_every: float = 5.0, port: int = 0,
                 host: str = "127.0.0.1", users=None):
        self.app, self.server, self.httpd = app, server, httpd
        self.keys, self.max_waiting, self.queue_timeout, self.debug = list(keys), max_waiting, queue_timeout, debug
        self.users = users if users is not None else N.Users(self.keys, name_of=key_name)  # who a request is (scripts/serve_net.py)
        self.quota = self.order = None  # --quotas, --priorities (scripts/serve_quota.py: install)
        self.host, self.port = host, port
        self.lock = threading.Lock()
        self.inflight = {}  # id(request) -> Entry
        self.t0 = time.time()
        self.responses = {}  # (method, path, code) labels, each from a fixed set -> answers
        self.refused = {"unauthorized": 0, "queue_full": 0, "queue_timeout": 0}
        self.finishes = {}  # how the requests that reached the engine ended (FINISHES) -> count
        self.ttft, self.latency = Histogram(TTFT_BUCKETS), Histogram(LATENCY_BUCKETS)
        self.pass_s, self.pass_n = {}, {}  # by width: seconds of the engine's steps that made a pass of it, and their count
        self.copy_s, self.copy_n = {}, {}  # by kind: park, unpark, snapshot, restore, save
        self.faults = {"handler": 0, "engine": 0}  # armed by POST /debug/fault
        self.injected = {"handler": 0, "engine": 0}
        self.last_request = None
        self.log = RequestLog(request_log, int(request_log_mb * 2**20)) if request_log else None
        self.tile = {}
        eng = getattr(server, "eng", None)
        if hasattr(eng, "ring"):
            from sclmd import engine as E

            self.tile = {t: E.tile_bytes(eng.cfg, t, eng.lean, getattr(eng, "kv4", False)) for t in eng.ring}
        self._proc, self._proc_t = {}, 0.0
        self.status_file, self.status_every = status_file, status_every
        if status_file:
            os.makedirs(os.path.dirname(os.path.abspath(status_file)), exist_ok=True)
            threading.Thread(target=self._status_loop, daemon=True, name="status-file").start()

    # the engine's methods, timed (installed by `install`; the same calls, the same results)

    def time_engine(self) -> None:
        s = self.server
        batch, eng = getattr(s, "batch", None), getattr(s, "eng", None)
        if batch is not None and hasattr(batch, "step"):
            step = batch.step

            def timed_step(*a, **k):
                if self.faults["engine"] > 0:
                    self.faults["engine"] -= 1
                    self.injected["engine"] += 1
                    raise InjectedFault("an engine fault that was asked for (POST /debug/fault)")
                widths = getattr(batch, "widths", None) or {}
                before = dict(widths)
                t = time.perf_counter()
                try:
                    return step(*a, **k)
                finally:
                    dt = time.perf_counter() - t
                    for w, n in widths.items():
                        if n != before.get(w, 0):
                            self.pass_s[w] = self.pass_s.get(w, 0.0) + dt
                            self.pass_n[w] = self.pass_n.get(w, 0) + 1
                            break

            batch.step = timed_step
        for name in ("park", "unpark", "snapshot", "restore", "save"):
            fn = getattr(eng, name, None)
            if fn is None:
                continue

            def timed(*a, _fn=fn, _name=name, **k):
                t = time.perf_counter()
                try:
                    return _fn(*a, **k)
                finally:
                    self.copy_s[_name] = self.copy_s.get(_name, 0.0) + time.perf_counter() - t
                    self.copy_n[_name] = self.copy_n.get(_name, 0) + 1

            setattr(eng, name, timed)

    # admission

    def _waiting(self) -> list:
        """The requests taken in that have no sequence yet (the real server's _place gives one a job)."""
        return [e for e in self.inflight.values() if getattr(e.req, "job", None) is None and not e.req.cancelled]

    def _free(self) -> int:
        """Sequences the engine's next turn gives a request without its waiting: the unused ones and the kept
        conversations' (the one kept longest gives its place). A burst at an idle server is taken in up to these
        before --max-waiting counts (measured: 64 at once, --max-waiting 8, 23 sequences free: 9 taken in, 55 refused,
        when every request not yet placed counted). A stand-in has none."""
        s = self.server
        return len(getattr(s, "unused", None) or ()) + len(getattr(s, "kept", None) or ())

    def retry_after(self, waiting: list | None = None) -> float:
        """Seconds until the waiting list has drained, roughly: its tokens (prompts, and replies as long as the mean so
        far) over the tokens a second the engine has done while busy; 1 to RETRY_MOST s."""
        s = self.server
        waiting = self._waiting() if waiting is None else waiting
        totals, busy = getattr(s, "totals", None), getattr(s, "busy_s", 0.0)
        rate = RATE
        if totals and busy > 5:
            rate = max((totals["prompt_tokens"] - totals["cached_tokens"] + totals["completion_tokens"]) / busy, 1.0)
        mean = totals["completion_tokens"] / totals["requests"] if totals and totals.get("requests") else None
        queued = sum(len(e.req.full) + min(e.req.max_new, mean if mean is not None else e.req.max_new) for e in waiting)
        return min(max(queued / rate, 1.0), RETRY_MOST)

    def submit(self, handler, req, rid: str, stream: bool) -> bool:
        """Takes the request in (and hands it to the engine), or answers 429 when --max-waiting requests wait already
        beyond the sequences that are free."""
        entry = Entry(req, rid, getattr(handler, "_r4_key", None), stream, handler.client_address[0] if handler.client_address else None)
        with self.lock:  # the count and the hand-over together: requests that come at once cannot all pass a count of N - 1
            daily = Q.admit(self, req)  # --quotas: a daily limit (429), before anything is counted
            waiting = self._waiting() if self.max_waiting > 0 and daily is None else None
            if waiting is not None and self.order is not None:  # --priorities: only those it would wait behind count
                waiting = [e for e in waiting if getattr(e.req, "klass", e.req.priority) <= req.priority]
            over = len(waiting) - self._free() if waiting is not None else 0
            if waiting is not None and over >= self.max_waiting:
                after = self.retry_after(waiting)
                self.refused["queue_full"] += 1
                refused = True
            elif daily is not None:
                refused = True
            else:
                self.inflight[id(req)] = entry
                if self.quota is not None:
                    self.quota.taken(req)
                self.server.submit(req)
                refused = False
        if daily is not None:  # (scripts/serve_quota.py) OpenAI's 429 with Retry-After; the log keeps which limit
            handler._r4_logged = True
            self._write({"id": rid, "key": entry.key, "client": entry.client, "status": 429, "finish": "daily_limit", "limit": daily.kind,
                         "tenant": req.tenant})
            handler._json(429, daily.body(), daily.headers())
            return False
        if refused:
            handler._r4_logged = True
            self._write({"id": rid, "key": entry.key, "client": entry.client, "status": 429, "finish": "queue_full", "waiting": over})
            handler._json(429, {"error": {"message": f"{over} requests wait for a sequence already (--max-waiting {self.max_waiting}): "
                                                     f"try again in {math.ceil(after)} s", "type": "requests", "param": None, "code": "rate_limit_exceeded"}},
                          {"Retry-After": str(math.ceil(after)), "retry-after-ms": str(int(after * 1000)), "x-should-retry": "true"})
            return False
        handler._r4 = entry
        self.last_request = time.time()
        return True

    def started(self, handler, req) -> bool:
        """In the reply's try: an injected handler fault, then (--queue-timeout) the wait for a sequence before anything
        is written. False: a 503 has been sent. (A handler fault is put only into a request that is not streamed: a
        stream's fault path writes events, which needs its headers out first.)"""
        entry = getattr(handler, "_r4", None)
        if self.faults["handler"] > 0 and entry is not None and not entry.stream:
            with self.lock:
                fire = self.faults["handler"] > 0
                if fire:
                    self.faults["handler"] -= 1
                    self.injected["handler"] += 1
            if fire:
                raise InjectedFault("a handler fault that was asked for (POST /debug/fault)")
        if self.queue_timeout <= 0:
            return True
        try:
            return self._wait_placed(handler, req)
        except ConnectionError:
            raise
        except Exception as e:  # noqa: BLE001  (a fault of this wait: the request goes on as without the option)
            print(f"[serve] --queue-timeout's wait failed: {C.why(e)}", flush=True)
            return True

    def _wait_placed(self, handler, req) -> bool:
        q, deadline = req.events, req.t_in + self.queue_timeout
        while getattr(req, "job", None) is None:
            with q.not_empty:  # wakes when the first event comes (a token, or the end), else every half second
                if len(q.queue):
                    return True
                q.not_empty.wait(min(0.5, max(deadline - time.perf_counter(), 0.01)))
                if len(q.queue):
                    return True
            if getattr(req, "job", None) is not None:
                return True
            if handler._gone():
                raise ConnectionError("the client left while its request waited")
            if time.perf_counter() >= deadline:
                req.cancelled = True  # the engine's thread drops it from the waiting list
                with self.lock:
                    after = self.retry_after()
                    self.refused["queue_timeout"] += 1
                entry = getattr(handler, "_r4", None)
                if entry is not None:
                    entry.finish = "queue_timeout"
                handler._json(503, {"error": {"message": f"the request waited {self.queue_timeout:g} s for a place (--queue-timeout): try again in "
                                                         f"{math.ceil(after)} s", "type": "server_error", "param": None, "code": "queue_timeout"}},
                              {"Retry-After": str(math.ceil(after)), "retry-after-ms": str(int(after * 1000))})
                return False
        return True

    def failed(self, handler, error: BaseException) -> None:
        """The reply's loop failed on the server's side (not the client leaving): its finish is server_error. The log
        gets the error's type, not its message (ZDR)."""
        entry = getattr(handler, "_r4", None)
        if entry is not None:
            entry.finish, entry.error = "server_error", C.why(error)

    def ended(self, handler, req, event, finish=None) -> None:
        """The HTTP thread is done with the request: its line in the log, the histograms. finish: what the client was
        told the reply ended with (tool_calls, stop ...), when it was told."""
        entry = getattr(handler, "_r4", None)
        with self.lock:
            self.inflight.pop(id(req), None)
        if entry is None:
            return
        handler._r4 = None  # the connection's next request starts afresh
        if self.quota is not None:  # --quotas: what it cost (nothing if it never reached the engine)
            self.quota.settle(req, event)
        job = getattr(req, "job", None)
        record = {"id": entry.rid, "key": entry.key, "client": entry.client, "status": getattr(handler, "_r4_status", None), "stream": entry.stream,
                  "prompt": len(req.full), "kept": getattr(req, "reused", 0), "reply": len(getattr(job, "out", ()))}
        if entry.finish is not None:  # the server failed while sending (server_error), or the wait for a place ended (queue_timeout)
            record["finish"] = entry.finish
            if entry.error:
                record["error"] = entry.error
        elif event is not None and event[0] == "done":
            result, timings = event[1], event[1].get("timings", {})
            record.update(reply=result["usage"]["completion_tokens"], ttft_ms=timings.get("first_token_ms"), total_ms=timings.get("total_ms"),
                          tok_s=timings.get("tokens_per_second"), ended=result.get("ended"),
                          finish=finish or ("stop" if req.stop_text is not None else result.get("finish_reason")))
            if record["reply"] and timings.get("first_token_ms") is not None:
                self.ttft.observe(timings["first_token_ms"] / 1000)
            if timings.get("total_ms") is not None:
                self.latency.observe(timings["total_ms"] / 1000)
        elif event is not None and event[0] == "error":
            record.update(finish="error", error=event[2], status=getattr(handler, "_r4_status", None) or event[1])
        else:  # the client left (the connection closed or failed under the reply)
            record["finish"] = "client_left"
        if req.ignored:
            record["ignored"] = req.ignored
        if getattr(req, "tenant", None) is not None:  # tenancy on (scripts/serve_net.py): who it was
            record["tenant"] = req.tenant
        if self.order is not None:  # --priorities: its priority when it was last ordered (sclmd/order.py)
            record["priority"] = getattr(req, "priority", 0)
        if getattr(job, "seq", None) is not None:
            record["seq"] = job.seq
        s = self.server
        eng = getattr(s, "eng", None)
        if self.tile and hasattr(eng, "free_tiles"):
            free = eng.free_tiles()
            record["kv_free_gib"] = round(sum(self.tile[t] * free[t] for t in self.tile) / 2**30, 3)
        if hasattr(s, "kept"):
            record.update(kept_conversations=len(s.kept), in_ram=len(s.parked), under_way=len(s.active))
        label = finish_label(record["finish"])
        with self.lock:
            self.finishes[label] = self.finishes.get(label, 0) + 1
        self._write(record)

    def _write(self, record: dict) -> None:
        if self.log is not None:
            self.log.write({"t": now_iso(), **record})

    # numbers

    def numbers(self) -> dict:
        """Everything /metrics and the status file show, read from the engine's state as it is now."""
        s, out = self.server, {}
        eng, batch = getattr(s, "eng", None), getattr(s, "batch", None)
        out["uptime_s"] = time.time() - self.t0
        out["under_way"] = len(getattr(s, "active", ()))
        out["waiting"] = len(getattr(s, "waiting", ())) + (s.inbox.qsize() if hasattr(s, "inbox") else 0)
        with self.lock:
            out["admission_waiting"] = len(self._waiting())
            out["inflight"] = len(self.inflight)
        totals = getattr(s, "totals", None)
        if totals:
            out["totals"] = dict(totals)
            out["busy_s"] = getattr(s, "busy_s", 0.0)
        if hasattr(s, "kept"):
            out["kept"], out["in_ram"] = len(s.kept), len(s.parked)
            out["ram_bytes"] = _safe(lambda: sum(p.state["bytes"] for p in list(s.parked)), 0)
            out["sequences"], out["sequences_free"] = eng.sequences, len(s.unused)
        if self.tile and eng is not None:
            free = eng.free_tiles()
            out["tiles"] = {t: (eng.pool[t], free[t]) for t in self.tile}
            out["kv_bytes"] = sum(self.tile[t] * eng.pool[t] for t in self.tile)
            out["kv_free_bytes"] = sum(self.tile[t] * free[t] for t in self.tile)
            out["snapshot_bytes"] = _safe(lambda: sum(d.numel() * d.element_size() for v in list(eng._snaps.values()) for d in v.get("data", ())), 0)
        prefixes = getattr(s, "prefixes", None)
        if prefixes is not None and hasattr(prefixes, "held"):
            out["prefixes"] = {"held": len(prefixes.held), "bytes": _safe(prefixes.bytes, 0), "hits": prefixes.hits, "stored": prefixes.stored,
                               "waits": prefixes.waits}
        if batch is not None and hasattr(batch, "widths"):
            out["passes"] = dict(batch.widths)
            out["batch"] = {"passes": batch.passes, "rows": batch.rows, "guess_rows": batch.guess_rows, "kept_rows": batch.kept_rows,
                            "restarts": batch.restarts}
        out["pass_s"], out["pass_n"] = dict(self.pass_s), dict(self.pass_n)
        out["copy_s"], out["copy_n"] = dict(self.copy_s), dict(self.copy_n)
        with self.lock:
            out["responses"] = dict(self.responses)
            out["refused"] = dict(self.refused)
            out["finishes"] = dict(self.finishes)
        out["injected"] = dict(self.injected)
        if time.time() - self._proc_t > 2.0:  # the OS's numbers are read at most every 2 s
            self._proc, self._proc_t = _safe(process_numbers, {}), time.time()
        out["process"] = dict(self._proc, python_threads=threading.active_count())
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            out["cuda"] = {"reserved_bytes": torch.cuda.memory_reserved(), "allocated_bytes": torch.cuda.memory_allocated()}
            host = _safe(torch.cuda.host_memory_stats, {})
            out["cuda"]["host_pinned_bytes"] = host.get("allocated_bytes.current")
            out["cuda"]["host_pinned_active_bytes"] = host.get("active_bytes.current")
        if self.httpd is not None and hasattr(self.httpd, "live"):
            out["connections"] = {"live": self.httpd.live, "turned_away": self.httpd.turned_away}
        if self.log is not None:
            out["request_log"] = {"written": self.log.written, "failed": self.log.failed}
        return out

    def metrics(self) -> str:
        """The numbers as Prometheus text (exposition format 0.0.4)."""
        n = self.numbers()
        lines = []

        def family(name: str, kind: str, text: str, samples: list) -> None:
            if not samples:
                return
            lines.append(f"# HELP {name} {text}")
            lines.append(f"# TYPE {name} {kind}")
            for sample in samples:
                sname, labels, value = sample if len(sample) == 3 else (name, *sample)
                if value is not None:
                    lines.append(f"{sname}{_labels(labels)} {_num(value)}")

        app = self.app
        family("sclmd_info", "gauge", "the model served and its weights", [({"model": "gemma-4-31b-it", "fingerprint": getattr(app, "fingerprint", "")}, 1)])
        family("sclmd_uptime_seconds", "gauge", "seconds since the server started", [({}, n["uptime_s"])])
        family("sclmd_requests_under_way", "gauge", "requests on a sequence now", [({}, n["under_way"])])
        family("sclmd_requests_waiting", "gauge", "requests waiting for a sequence (the engine's list and its inbox)", [({}, n["waiting"])])
        family("sclmd_admission_waiting", "gauge", "requests taken in without a sequence yet (what --max-waiting bounds)", [({}, n["admission_waiting"])])
        if "totals" in n:
            t = n["totals"]
            family("sclmd_requests_finished_total", "counter", "requests the engine has ended", [({}, t["requests"])])
            family("sclmd_prompt_tokens_total", "counter", "prompt tokens of the requests ended", [({}, t["prompt_tokens"])])
            family("sclmd_cached_tokens_total", "counter", "prompt tokens that were kept from before (not read again)", [({}, t["cached_tokens"])])
            family("sclmd_completion_tokens_total", "counter", "reply tokens written", [({}, t["completion_tokens"])])
            family("sclmd_busy_seconds_total", "counter", "seconds the engine's thread spent in passes", [({}, n["busy_s"])])
        family("sclmd_http_responses_total", "counter", "answers by method, path and status (each from a fixed set; a made-up method is other)",
               [({"method": m, "path": p, "code": c}, v) for (m, p, c), v in sorted(n["responses"].items())])
        family("sclmd_refused_total", "counter", "requests refused by the operator's options", [({"reason": k}, v) for k, v in sorted(n["refused"].items())])
        family("sclmd_finished_total", "counter", "how the requests that reached the engine ended, as their HTTP thread saw it "
               "(server_error: the server failed while sending; client_left: the connection closed under the reply)",
               [({"finish": k}, v) for k, v in sorted(n["finishes"].items())])
        if "kept" in n:
            family("sclmd_conversations_kept", "gauge", "conversations kept on a sequence for their next turn", [({}, n["kept"])])
            family("sclmd_conversations_in_ram", "gauge", "conversations parked in RAM", [({}, n["in_ram"])])
            family("sclmd_parked_bytes", "gauge", "RAM held by parked conversations (--ram-gib bounds it)", [({}, n["ram_bytes"])])
            family("sclmd_sequences", "gauge", "sequences of the engine", [({}, n["sequences"])])
            family("sclmd_sequences_free", "gauge", "sequences neither writing nor kept", [({}, n["sequences_free"])])
        if "tiles" in n:
            family("sclmd_kv_tiles", "gauge", "tiles of keys and values in the pool, by kind of layer", [({"kind": k}, v[0]) for k, v in n["tiles"].items()])
            family("sclmd_kv_free_tiles", "gauge", "free tiles of keys and values, by kind of layer", [({"kind": k}, v[1]) for k, v in n["tiles"].items()])
            family("sclmd_kv_bytes", "gauge", "VRAM of the keys and values pool", [({}, n["kv_bytes"])])
            family("sclmd_kv_free_bytes", "gauge", "VRAM of the free tiles", [({}, n["kv_free_bytes"])])
            family("sclmd_pinned_bytes", "gauge", "page-locked RAM the server holds, by what for",
                   [({"kind": "snapshots"}, n["snapshot_bytes"])] + ([({"kind": "prefixes"}, n["prefixes"]["bytes"])] if "prefixes" in n else []))
        if "prefixes" in n:
            p = n["prefixes"]
            family("sclmd_prefixes_held", "gauge", "shared prefixes kept in RAM (--prefix-gib)", [({}, p["held"])])
            family("sclmd_prefix_hits_total", "counter", "requests a kept prefix was put on", [({}, p["hits"])])
            family("sclmd_prefix_waits_total", "counter", "times a request waited for room to put a kept prefix on", [({}, p["waits"])])
        if "passes" in n:
            b = n["batch"]
            family("sclmd_passes_total", "counter", "passes of the main model, by width (rows)", [({"width": str(w)}, v) for w, v in sorted(n["passes"].items())])
            family("sclmd_pass_seconds_total", "counter", "seconds of the engine's steps that made a pass, by its width",
                   [({"width": str(w)}, n["pass_s"].get(w, 0.0)) for w in sorted(n["passes"])])
            family("sclmd_rows_total", "counter", "rows of all passes", [({}, b["rows"])])
            family("sclmd_guess_rows_total", "counter", "rows that carried the drafter's guesses", [({}, b["guess_rows"])])
            family("sclmd_guesses_kept_total", "counter", "guesses the main model kept", [({}, b["kept_rows"])])
            family("sclmd_started_over_total", "counter", "texts that gave their memory up and read everything again", [({}, b["restarts"])])
        kinds = sorted(set(n["copy_n"]))
        family("sclmd_copies_total", "counter", "copies of keys and values between VRAM and RAM, by kind", [({"op": k}, n["copy_n"][k]) for k in kinds])
        family("sclmd_copy_seconds_total", "counter", "seconds of those copies, by kind", [({"op": k}, n["copy_s"][k]) for k in kinds])
        family("sclmd_ttft_seconds", "histogram", "time to the first reply token, from the request's arrival", self.ttft.samples("sclmd_ttft_seconds"))
        family("sclmd_request_seconds", "histogram", "time of a whole request, from its arrival to its end", self.latency.samples("sclmd_request_seconds"))
        p = n["process"]
        family("sclmd_process_resident_bytes", "gauge", "the process's resident memory (working set)", [({}, p.get("resident_bytes"))])
        family("sclmd_process_private_bytes", "gauge", "the process's private memory (commit)", [({}, p.get("private_bytes"))])
        family("sclmd_process_threads", "gauge", "threads of the process", [({"kind": "python"}, p.get("python_threads")), ({"kind": "os"}, p.get("os_threads"))])
        if "cuda" in n:
            c = n["cuda"]
            family("sclmd_cuda_reserved_bytes", "gauge", "VRAM held by torch's caching allocator", [({}, c["reserved_bytes"])])
            family("sclmd_cuda_allocated_bytes", "gauge", "VRAM in use by tensors", [({}, c["allocated_bytes"])])
            family("sclmd_host_pinned_bytes", "gauge", "page-locked RAM held by torch's host allocator (in use and cached)",
                   [({"state": "held"}, c["host_pinned_bytes"]), ({"state": "active"}, c["host_pinned_active_bytes"])])
        if "connections" in n:
            family("sclmd_connections", "gauge", "connections served now (a thread each)", [({}, n["connections"]["live"])])
            family("sclmd_connections_turned_away_total", "counter", "connections answered 503 past --max-connections", [({}, n["connections"]["turned_away"])])
        family("sclmd_injected_faults_total", "counter", "faults injected through /debug/fault", [({"kind": k}, v) for k, v in sorted(n["injected"].items())])
        if self.quota is not None:  # --quotas (scripts/serve_quota.py): counts only, never a tenant
            self.quota.families(family)
        if self.order is not None:  # --priorities (sclmd/order.py)
            o = self.order.n
            family("sclmd_order_total", "counter", "the order of service: requests placed, placed past the waiting list's head, placed while "
                   "texts waited for keys and values; guess rows the best writers took before worse prompts",
                   [({"what": k}, o[k]) for k in ("placed", "placed_past_the_head", "placed_while_starved", "best_guess_rows")])
        return "\n".join(lines) + "\n"

    # the status file

    def status(self) -> dict:
        n = self.numbers()
        out = {"t": now_iso(), "pid": os.getpid(), "url": f"http://{self.host}:{self.port}/v1", "uptime_s": round(n["uptime_s"]),
               "under_way": n["under_way"], "waiting": n["waiting"],
               "last_request": datetime.fromtimestamp(self.last_request).astimezone().isoformat(timespec="seconds") if self.last_request else None,
               "refused": n["refused"], "finishes": n["finishes"], "injected": n["injected"],
               "errors_5xx": sum(v for (_, _, c), v in n["responses"].items() if c.startswith("5"))}
        if "totals" in n:
            out["totals"] = n["totals"]
            out["tokens_per_second_while_busy"] = round(n["totals"]["completion_tokens"] / max(n["busy_s"], 1e-9), 1)
        for k in ("kept", "in_ram", "sequences_free"):
            if k in n:
                out[k] = n[k]
        if "kv_free_bytes" in n:
            out["kv_free_gib"], out["kv_gib"] = round(n["kv_free_bytes"] / 2**30, 2), round(n["kv_bytes"] / 2**30, 2)
        out["process"] = n["process"]
        if "cuda" in n:
            out["vram_reserved_gib"] = round(n["cuda"]["reserved_bytes"] / 2**30, 2)
        return out

    def _status_loop(self) -> None:
        tmp = self.status_file + ".tmp"
        while True:
            try:
                with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(self.status(), f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.status_file)
            except Exception as e:  # noqa: BLE001  (a reader holding the file open, say: next time)
                self.status_errors = getattr(self, "status_errors", 0) + 1
                if self.status_errors in (1, 10, 100):
                    print(f"[serve] the status file could not be written ({self.status_errors}): {C.why(e)}", flush=True)
            time.sleep(self.status_every)


# the hooks in scripts/serve.py's Handler: each does nothing when the app has no Admin (the tests' stand-ins)

def _admin(handler):
    return getattr(getattr(handler.server, "app", None), "admin", None)


def _path(raw: str) -> str:
    p = raw.split("?")[0].rstrip("/") or "/"
    return next((x for x in PATHS if p == x or (x != "/" and p.startswith(x + "/"))), "other")


def _begin(handler) -> None:
    """A request's own state on its handler (a connection's handler serves its requests one after another)."""
    handler._r4, handler._r4_status, handler._r4_logged, handler._r4_key, handler._r4_message = None, None, False, None, None
    handler._r4_line = getattr(handler, "raw_requestline", None)
    handler._n1_who = None


def _fresh(handler) -> None:
    """The state of the request being answered: one answered before do_GET or do_POST ran (a bad request line or
    header, a method not served) must not see the state of the connection's previous request."""
    if getattr(handler, "_r4_line", None) is not getattr(handler, "raw_requestline", None):
        _begin(handler)


UNKNOWN_KEY = ("this API key is not known here: send a key this server gave you, or none to be served as your address "
               "(the openai library: default_headers={\"Authorization\": \"\"})")


def _allowed(adm, handler, path: str) -> bool:
    """Whether the request may go on; else a 401 has been sent. Who it is (scripts/serve_net.py) goes on the handler
    first. Without keys (and without --anonymous) everything may."""
    w = handler._n1_who = adm.users.who(handler)
    if path == "/health" or w.refusal is None:
        handler._r4_key = w.key
        return True
    with adm.lock:
        adm.refused["unauthorized"] += 1
    handler._r4_logged = True
    adm._write({"status": 401, "finish": "unauthorized", "client": handler.client_address[0], "path": _path(handler.path)})
    unread = 0
    if handler.command == "POST":
        # the body is not read into memory: the 401 goes out, then (as for R1's 413) what the client sends is read and
        # thrown away, its announced length (or until it stops: no length), for at most 30 s and 2 s without a byte, and
        # the connection is closed. A connection closed with data unread is reset on Windows, and a client that sends
        # its whole body before it reads (http.client, the openai library) would lose the 401.
        length = handler.headers.get("Content-Length")
        try:
            unread = int(length) if length is not None else None
        except ValueError:
            unread = None
        unread = None if unread is not None and unread < 0 else unread
    if unread != 0:
        handler.close_connection = True
    text = UNKNOWN_KEY if adm.users.anonymous and w.refusal == "unknown_key" else "an API key is needed: Authorization: Bearer <key> (or x-api-key)"
    handler._json(401, {"error": {"message": text, "type": "invalid_request_error",
                                  "param": None, "code": "invalid_api_key"}}, {"WWW-Authenticate": 'Bearer realm="sclmd"'})
    if unread != 0:
        C.linger(handler.connection, handler.rfile, unread)
    return False


def _local(handler) -> bool:
    """This PC: a loopback connection that carries no X-Forwarded-For or Forwarded (one that does came through a proxy
    or tunnel on this PC, serve_net.origin)."""
    if not handler.client_address:
        return False
    return N.origin(handler.client_address[0], handler.headers, ())[1]


def get(handler) -> bool:
    """GET: the key, then /metrics and /debug/audit. True: answered here."""
    _begin(handler)
    adm = _admin(handler)
    if adm is None:
        return False
    path = handler.path.split("?")[0].rstrip("/")
    if not _allowed(adm, handler, path):
        return True
    if path == "/metrics" and N.metrics_refused(handler):  # --metrics-local: an unknown path from anywhere but this PC
        return False
    if path == "/metrics":
        data = adm.metrics().encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
        return True
    if path == "/v1/usage" and adm.quota is not None:  # --quotas: the caller's own usage and limits (scripts/serve_quota.py)
        handler._json(200, adm.quota.usage_of(N.who_of(handler).tenant))
        return True
    if path == "/debug/audit" and adm.debug and _local(handler):
        s = adm.server
        if not hasattr(s, "batch"):
            handler._json(404, {"error": {"message": "no engine to audit (a stand-in server)", "type": "invalid_request_error", "param": None,
                                          "code": "no_engine"}})
            return True
        result = audit_at_idle(s)
        handler._json(200 if result["idle"] else 409, result)
        return True
    return False


def _fault_body(handler) -> dict:
    """POST /debug/fault's body: {"kind": "handler" | "engine", "count": 1..100}, or {"kind": "clear"} (every armed
    fault disarmed). The kind must be given."""
    length = handler.headers.get("Content-Length")
    if length is None or not length.isdigit() or int(length) > FAULT_BODY:
        raise ValueError(f"a JSON body of at most {FAULT_BODY} bytes, with its Content-Length")
    body = json.loads(handler.rfile.read(int(length)) or b"null")
    if not isinstance(body, dict) or "kind" not in body:
        raise ValueError('{"kind": "handler" | "engine" | "clear", "count": 1 to 100}')
    return body


def post(handler) -> bool:
    """POST: the key, then /debug/fault. True: answered here."""
    _begin(handler)
    adm = _admin(handler)
    if adm is None:
        return False
    path = handler.path.split("?")[0].rstrip("/")
    if not _allowed(adm, handler, path):
        return True
    if path == "/v1/chat/completions" and adm.quota is not None and Q.refused_early(adm, handler):  # --quotas: already at a daily limit
        return True
    if path == "/debug/fault" and adm.debug and _local(handler):
        try:
            body = _fault_body(handler)
            kind = body["kind"]
            if kind == "clear":
                with adm.lock:
                    adm.faults = dict.fromkeys(adm.faults, 0)
                handler._json(200, {"armed": dict(adm.faults)})
                return True
            count = int(body.get("count", 1))
            if kind not in adm.faults or not 0 < count <= 100:
                raise ValueError(f"kind: {', '.join(adm.faults)} or clear; count 1 to 100")
        except (ValueError, TypeError, AttributeError) as e:
            handler.close_connection = True
            handler._json(400, {"error": {"message": f"/debug/fault: {e}", "type": "invalid_request_error", "param": None, "code": None}})
            return True
        with adm.lock:
            adm.faults[kind] += count
        print(f"[serve] /debug/fault: {count} {kind} fault(s) armed", flush=True)
        handler._json(200, {"armed": dict(adm.faults)})
        return True
    return False


def submit(handler, req, rid: str, stream: bool) -> bool:
    """In place of app.server.submit(req): False when a 429 has been sent (--max-waiting)."""
    adm = _admin(handler)
    if adm is None:
        handler.server.app.server.submit(req)
        return True
    return adm.submit(handler, req, rid, stream)


def started(handler, req) -> bool:
    adm = _admin(handler)
    return True if adm is None else adm.started(handler, req)


def failed(handler, error: BaseException) -> None:
    adm = _admin(handler)
    if adm is not None:
        adm.failed(handler, error)


def ended(handler, req, event, finish=None) -> None:
    adm = _admin(handler)
    if adm is None:
        return
    try:
        adm.ended(handler, req, event, finish)
    except Exception as e:  # noqa: BLE001  (bookkeeping never fails a request)
        print(f"[serve] the request's bookkeeping failed: {C.why(e)}", flush=True)


def answered(handler, code: int) -> None:
    """Every answer's status (Handler.send_response): counted; a refusal that is not a request's (400, 404, 411, 413
    before it was taken in) gets its own line in the request log: the field it is about and the error's code, never
    its message (ZDR: a message may quote the request)."""
    adm = _admin(handler)
    if adm is None:
        return
    _fresh(handler)
    handler._r4_status = code
    command = getattr(handler, "command", None)  # a bad request line has none
    path = _path(getattr(handler, "path", "") or "")
    key = (method_label(command), path, code_label(code))
    with adm.lock:
        adm.responses[key] = adm.responses.get(key, 0) + 1
    if code >= 400 and getattr(handler, "_r4", None) is None and not getattr(handler, "_r4_logged", False) and command == "POST":
        handler._r4_logged = True
        record = {"status": code, "finish": "refused", "client": handler.client_address[0], "key": getattr(handler, "_r4_key", None),
                  "path": path}
        param, kind = getattr(handler, "_r4_message", None) or (None, None)
        record.update({k: v for k, v in (("param", param), ("code", kind)) if v is not None})
        tenant = N.who_of(handler).tenant
        if tenant is not None:
            record["tenant"] = tenant
        adm._write(record)


def status_headers(handler) -> None:
    """Right after every answer's status line (Handler.send_response): with --quotas, the tenant's x-ratelimit-*."""
    adm = _admin(handler)
    if adm is not None and adm.quota is not None:
        Q.status_headers(adm, handler)


def message(handler, param=None, code=None) -> None:
    """What the request log keeps of an error answer (Handler._error): the field it is about and its code, not its
    message (ZDR). It may come before do_POST ran (a refused Expect: 100-continue)."""
    _fresh(handler)
    handler._r4_message = (param, code)


# the books of the keys and values (the soak's audit)

def audit(server) -> dict:
    """What must hold of the engine's pool and the server's conversations whenever no pass runs: every tile is free
    or held by one sequence, never both or twice; a sequence is kept or unused (at idle nothing writes); an unused one
    holds nothing; a kept one holds what its record says (read ahead to the reply's later form, or past its turn
    start); a parked one's length is its tokens'; snapshots belong to kept sequences. -> {"ok", "problems", "counts"}"""
    from sclmd import model as M

    eng, problems = server.eng, []
    counts = {"kept": len(server.kept), "parked": len(server.parked), "unused": len(server.unused), "active": len(server.active)}
    for t in eng.ring:
        held = [x for seq in range(eng.sequences) for x in eng.tiles[t][seq] if x >= 0]
        free = list(eng.free[t])
        counts[f"tiles_{t}"] = {"held": len(held), "free": len(free), "pool": eng.pool[t]}
        if len(held) + len(free) != eng.pool[t]:
            problems.append(f"{t}: {len(held)} tiles held + {len(free)} free != the pool's {eng.pool[t]}")
        both = held + free
        if len(set(both)) != len(both):
            twice = sorted({x for x in both if both.count(x) > 1})[:8]
            problems.append(f"{t}: tiles counted twice (held or free): {twice}")
        if any(not 0 <= x < eng.pool[t] for x in both):
            problems.append(f"{t}: tile numbers out of the pool")
    unused, kept = list(server.unused), set(server.kept)
    if len(set(unused)) != len(unused):
        problems.append(f"a sequence is unused twice: {sorted(unused)}")
    if set(unused) & kept:
        problems.append(f"sequences both unused and kept: {sorted(set(unused) & kept)}")
    busy = {r.job.seq for r in server.active if r.job is not None}
    missing = set(range(eng.sequences)) - set(unused) - kept - busy
    if missing:
        problems.append(f"sequences neither unused, kept nor writing: {sorted(missing)}")
    full = M.FULL if M.FULL in eng.ring else None
    for seq in unused:
        if eng.lengths[seq] or eng.highs[seq] or any(x >= 0 for t in eng.ring for x in eng.tiles[t][seq]):
            problems.append(f"unused sequence {seq} holds {eng.lengths[seq]} positions or tiles")
        if seq in eng._snaps:
            problems.append(f"unused sequence {seq} has a snapshot in RAM")
    for seq, k in server.kept.items():
        n = eng.lengths[seq]
        if k.seq != seq:
            problems.append(f"kept conversation on sequence {seq} says {k.seq}")
        if eng.highs[seq] < n:
            problems.append(f"sequence {seq}: written up to {eng.highs[seq]} but holds {n}")
        if k.later is not None:
            if not len(k.base) <= n <= len(k.later):
                problems.append(f"kept sequence {seq} holds {n} positions: its record says {len(k.base)} to {len(k.later)}")
            elif n != len(k.later) and (k.job is None or k.job.done == "read"):
                problems.append(f"kept sequence {seq} holds {n} of its later form's {len(k.later)} with its reading ahead over")
        elif n < len(k.base):
            problems.append(f"kept sequence {seq} holds {n} positions, less than its turn start {len(k.base)}")
        if full is not None:
            need = -(-n // 16)
            if any(eng.tiles[full][seq][j] < 0 for j in range(min(need, len(eng.tiles[full][seq])))):
                problems.append(f"kept sequence {seq}: a full-attention tile of its first {n} positions is missing")
    for seq in eng._snaps:
        if seq not in kept and seq not in busy:
            problems.append(f"a snapshot of sequence {seq}, which is neither kept nor writing")
    for p in server.parked:
        if p.state["length"] != len(p.tokens):
            problems.append(f"a parked conversation of {len(p.tokens)} tokens holds {p.state['length']} positions")
    ram = sum(p.state["bytes"] for p in server.parked)
    counts["parked_bytes"] = ram
    if server.ram and ram > server.ram:
        problems.append(f"parked conversations hold {ram} bytes, past --ram-gib's {server.ram}")
    pre = getattr(server, "prefixes", None)
    if pre is not None and pre.ram and pre.bytes() > pre.ram:
        problems.append(f"kept prefixes hold {pre.bytes()} bytes, past --prefix-gib's {pre.ram}")
    return {"ok": not problems, "problems": problems, "counts": counts}


def audit_at_idle(server) -> dict:
    """The audit, only when the engine is idle (nothing under way, waiting, coming in or reading ahead) and stays so
    while it is read; else {"idle": False}."""
    batch = server.batch

    def idle():
        return not server.active and not server.waiting and server.inbox.empty() and not [j for j in batch.jobs if j.done is None]

    if not idle():
        return {"idle": False, "under_way": len(server.active), "waiting": len(server.waiting), "jobs": len(batch.jobs)}
    passes = batch.passes
    try:
        result = audit(server)
    except Exception as e:  # noqa: BLE001  (state changed under it)
        return {"idle": False, "error": f"{type(e).__name__}: {e}"}
    if batch.passes != passes or not idle():
        return {"idle": False, "moved": True}
    return {"idle": True, **result}


# start-up

def add_options(ap) -> None:
    g = ap.add_argument_group("operating the server (scripts/serve_admin.py)")
    g.add_argument("--config", default=None, help=f"a TOML file with options (the command line wins over it); default {os.path.basename(CONFIG)} "
                   "in the repository's folder when it is there, \"\" for none. API keys are never read from it")
    g.add_argument("--api-key-file", default="", help=f"API keys, one a line (\"name: key\" or \"key\"), beside or instead of the "
                   f"environment variable {ENV_KEY} (a key, or name:key pairs with commas). With a key, every request but GET /health needs "
                   "it (Authorization: Bearer, or x-api-key). Keep the file outside the repository")
    g.add_argument("--require-key", action="store_true", help="refuse to start on a --host other than loopback without an API key "
                   "(without it: a warning)")
    g.add_argument("--max-waiting", type=int, default=0, help="answer 429 (with Retry-After) to a request when this many already wait for "
                   "a sequence beyond the sequences free now (unused or kept: a burst at an idle server is taken in up to those first; "
                   "0: no bound, as before; 64 suggested; with --priorities only those of a class no worse than its own count; "
                   "serve.public.example.toml leaves it off: 6.16 refuses only at the daily limit)")
    g.add_argument("--queue-timeout", type=float, default=0.0, help="answer 503 (with Retry-After) to a request that has waited this many "
                   "seconds for a sequence; with it, a stream's headers go out when it has one (0: wait for ever, as before; 120 suggested; "
                   "serve.public.example.toml leaves it off: it would refuse over-quota requests that --priorities keeps waiting)")
    g.add_argument("--request-log", default="", help="a JSON line a request here, e.g. logs/serve_requests.jsonl (\"\": none, as before)")
    g.add_argument("--request-log-mb", type=float, default=64.0, help="the request log is rotated (.1, .2, .3) past this size")
    g.add_argument("--status-file", default="", help="the server's state as JSON, rewritten every 5 s, e.g. logs/serve_status.json (\"\": none)")
    g.add_argument("--debug", action="store_true", help="POST /debug/fault and GET /debug/audit, for scripts/soak.py (a loopback --host only)")
    N.add_options(ap)  # who a request is and the limits (scripts/serve_net.py)
    Q.add_options(ap)  # quotas and priorities (scripts/serve_quota.py)


def _defaults_from(ap, data: dict, path: str) -> dict:
    actions = {a.dest: a for a in ap._actions if a.dest != "help"}
    out = {}
    for k, v in data.items():
        dest = k.replace("-", "_")
        if dest in SERVE_BAT:  # serve.bat's own (supervise = true: through scripts/supervise.py), not the server's
            if not isinstance(v, bool):
                raise SystemExit(f"{path}: {k} is a switch: true or false")
            continue
        if dest in SECRET:
            raise SystemExit(f"{path}: {k}: API keys are not read from a file that may be committed: set {ENV_KEY} or --api-key-file")
        if dest == "config" or dest not in actions:
            raise SystemExit(f"{path}: {k} is not an option of the server (python -m scripts.serve -h lists them)")
        a = actions[dest]
        if a.nargs == "?" and a.const is not None and isinstance(v, bool):  # int8 = true: the option given alone (its const)
            v = a.const if v else a.default
        elif a.nargs == 0:  # a switch
            if not isinstance(v, bool):
                raise SystemExit(f"{path}: {k} is a switch: true or false")
        elif isinstance(v, (dict, list, bool)):
            raise SystemExit(f"{path}: {k}: a {type(v).__name__} where a value is needed")
        elif a.type is not None:
            try:
                v = a.type(v)
            except (TypeError, ValueError) as e:
                raise SystemExit(f"{path}: {k}: {e}")
        if a.choices is not None and v not in a.choices:
            raise SystemExit(f"{path}: {k}: one of {', '.join(map(str, a.choices))}")
        out[dest] = v
    return out


def parse_args(ap, argv=None):
    """ap.parse_args, with the options of serve.toml (--config) as its defaults: the command line wins."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)  # --con must not read as --config (it was --context)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    path = known.config if known.config is not None else (CONFIG if os.path.exists(CONFIG) else "")
    data = {}
    if path:
        import tomllib

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise SystemExit(f"--config {path}: {e}")
        ap.set_defaults(**_defaults_from(ap, data, path))
    cli = ap.parse_args(argv)
    if cli.config != known.config:  # an abbreviation (--conf x) the full parser took but the file was not read from
        ap.error("write --config in full")
    cli.config = path
    if data:
        print(f"[serve] options from {path}: {', '.join(sorted(data))}", flush=True)
    return cli


def start(cli) -> list:
    """Checks made before the model loads -> the API keys. A host other than loopback without a key: a loud warning
    (--require-key: no start)."""
    keys = load_keys(cli)
    N.check(cli)
    Q.check(cli, keys)
    if cli.debug and not loopback(cli.host):
        raise SystemExit("--debug serves its fault endpoint only on a loopback --host (127.0.0.1)")
    if not loopback(cli.host) and not keys:
        if cli.require_key:
            raise SystemExit(f"--host {cli.host} lets other machines in, and no API key is set ({ENV_KEY} or --api-key-file): not started "
                             "(--require-key)")
        bar = "!" * 100
        print(f"[serve] {bar}\n[serve] WARNING: --host {cli.host} lets other machines in, and there is NO API KEY: anyone who can reach "
              f"this port can use the model.\n[serve] Set {ENV_KEY} (or --api-key-file) to ask for a key; --require-key refuses to start "
              f"like this.\n[serve] {bar}", flush=True)
    for name, key in keys:
        if len(key) < 16:
            print(f"[serve] warning: the API key {name!r} has only {len(key)} characters", flush=True)
        if getattr(cli, "quotas", False) and len(key.split()) > 1:  # (its own quota written wrongly? tph=, rph=, tpd=, rpd= take whole numbers)
            print(f"[serve] warning: the API key {name!r} has a space in it", flush=True)
    return keys


def install(app, server, cli, httpd, keys: list) -> Admin:
    """The operator's side of a running server: app.admin, the engine's methods timed, the log and the status writer."""
    users = N.users_of(cli, keys, key_name)  # who a request is (scripts/serve_net.py): off without --anonymous and keys
    adm = Admin(app, server, httpd, keys, max_waiting=cli.max_waiting, queue_timeout=cli.queue_timeout, debug=cli.debug,
                request_log=cli.request_log, request_log_mb=cli.request_log_mb, status_file=cli.status_file, port=cli.port, host=cli.host,
                users=users)
    adm.time_engine()
    app.admin = adm
    N.apply(cli, app)
    on = [f"API keys: {', '.join(n for n, _ in keys)}" if keys else None, f"--max-waiting {cli.max_waiting}" if cli.max_waiting else None,
          f"--queue-timeout {cli.queue_timeout:g} s" if cli.queue_timeout else None, f"request log {cli.request_log}" if cli.request_log else None,
          f"status file {cli.status_file}" if cli.status_file else None, "--debug (fault and audit endpoints)" if cli.debug else None]
    on = [x for x in on if x]
    print("[serve] GET /metrics" + (f"; {'; '.join(on)}" if on else ""), flush=True)
    said = N.line(cli, users)
    if said:
        print(said, flush=True)
    Q.install(adm, server, cli)  # --quotas, --priorities (scripts/serve_quota.py; nothing without them)
    return adm
