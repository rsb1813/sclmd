# The server's start-up and lifecycle (R3; scripts/serve.py calls in through a few hooks). All of it is off unless
# its option is given: without --image and --crash-only the server starts, answers /health and turns exactly as before.
#
# --image auto|read: the engine's packed codes from an engine image (sclmd/image.py): a start-up of about 6 s for 13.
#
# --crash-only (what scripts/supervise.py runs the server with):
# - the port is taken first and answers 503 "starting" while the engine loads (Starting);
# - after a failed turn the device is probed (a synchronize and a one-element op). A sticky CUDA error (an illegal
#   address, a driver reset) fails the probe, and every later pass would fail the same way (R3's kill test 3): the
#   engine is dead, the requests under way are told 503, and the process ends with EXIT_FATAL for the supervisor to
#   start it again. --restart-failures (3) failures within a minute count as dead too, whatever the probe says;
# - a heartbeat at the end of every turn of the engine's thread (idle, the thread wakes every second) and of every
#   call that makes room in the pool (parking a long conversation takes seconds); /health says 503 "stale" when it is
#   older than --watchdog-s, and the watchdog then dumps every thread's stack and ends the process;
# - /health says 503 "dead", "stale" or "draining" (with Retry-After and beat_s), and new requests get 503 while the
#   server is dead or stopping;
# - Ctrl+C or Ctrl+Break lets the requests under way finish (--drain-s) and their answers go out before it ends.
#
# Every 503 of a placeholder (Starting, the supervisor's Holder) or of a dead or stopping server closes the connection
# after it (Connection: close), so that a client that keeps its connection (http.client, httpx, the openai library)
# comes back to whatever serves the port next. Its body is not read first: a Content-Length that is not a number is
# answered 400 (none, on a POST: 411) as the server answers it, and what the client sends is then read and thrown away
# (serve_contract.linger) so that it gets the answer instead of a reset. Nothing here imports torch at import time
# (scripts/supervise.py uses the placeholder).
import faulthandler
import json
import os
import sys
import threading
import time
import weakref
from collections import deque
from http.server import BaseHTTPRequestHandler

from scripts import serve_contract as C

EXIT_FATAL = 75  # the engine cannot go on: start the server again (scripts/supervise.py)
RETRY_AFTER = {"starting": 15, "dead": 20, "stale": 20, "draining": 30}
PLACEHOLDER_TIMEOUT = 10.0  # seconds a placeholder's connection may send nothing before it is closed


def add_options(ap) -> None:
    g = ap.add_argument_group("start-up and restarts (scripts/serve_ops.py; all off by default)")
    g.add_argument("--image", default="off", choices=("off", "auto", "read"), help="the engine's packed codes from an image under "
                   "data/images (sclmd/image.py): a start-up of about 6 s instead of 13, the same bytes on the GPU. auto reads it when it "
                   "matches the plan, the codes and the build, else loads as without it and writes one afterwards (about 12 GB, in the "
                   "background; only for 3 bits everywhere, not for plans fitted to the VRAM that is free); read never writes")
    g.add_argument("--crash-only", action="store_true", help="for a supervisor (scripts/supervise.py, serve.bat --supervise): the port "
                   "answers 503 while the engine loads; a dead engine (a sticky CUDA error, or --restart-failures failures in a minute) or a "
                   "hung engine thread (--watchdog-s) ends the process with exit code 75; /health says 503 dead, stale or draining; "
                   "Ctrl+C or Ctrl+Break lets the requests under way finish (--drain-s) before it ends")
    g.add_argument("--watchdog-s", type=float, default=15.0, help="with --crash-only: seconds without an ended turn of the engine's "
                   "thread that count as a hang")
    g.add_argument("--drain-s", type=float, default=30.0, help="with --crash-only: the longest a stop waits for the requests under way")
    g.add_argument("--restart-failures", type=int, default=3, help="with --crash-only: failed turns within a minute that end the process "
                   "although the device still answers (0: only a lost device does)")


def image_mode(cli, fitted: bool) -> str:
    """Engine.load's image= for the server: an image is written only for 3 bits everywhere (a plan fitted to the VRAM
    free now changes from start to start: it is only read); "background": written once the server serves."""
    if cli.image == "off":
        return "off"
    return ("read" if fitted else cli.image) + ":background"


def cuda_probe() -> None:
    """Raises when the CUDA context is spoilt: after a sticky error every call on it fails."""
    import torch

    torch.cuda.synchronize()
    x = torch.ones(1, device="cuda")
    if float((x + 1).item()) != 2.0:
        raise RuntimeError("the device computed 1 + 1 wrongly")


class Ops:
    """The lifecycle state of one server (--crash-only). exits: end the process on a fatal failure and from the
    watchdog; watchdog_s: the heartbeat's age that counts as a hang; probe: raises when the device is lost; failures:
    failed turns within window_s that count as dead although the probe passes (0: none)."""

    def __init__(self, server, exits: bool = True, watchdog_s: float = 15.0, probe=cuda_probe, failures: int = 3, window_s: float = 60.0,
                 exit=os._exit):
        self.server, self.exits, self.watchdog_s, self.probe = server, exits, watchdog_s, probe
        self.failures, self.window_s, self._exit = failures, window_s, exit
        self.recent = deque()  # times of the failed turns in the last window_s
        self.last_beat = time.monotonic()
        self.dead = None  # why the engine cannot go on
        self.draining = False
        self.thread = None  # the engine's thread, once started
        self.exiting = False
        self.events = deque(maxlen=256)  # (time, what) for the log and the tests: the last ones
        self.requests = weakref.WeakSet()  # requests handed to the engine (a stop waits for their answers to be taken)

    # the engine's thread

    def beat(self) -> None:
        self.last_beat = time.monotonic()

    def failed(self, error: Exception):
        """A turn failed (Server.run). -> (status, message) to tell the requests under way, or None: as before (500)."""
        now = time.monotonic()
        self.recent.append(now)
        while self.recent and now - self.recent[0] > self.window_s:
            self.recent.popleft()
        try:
            self.probe()
            # failures that keep coming on a device that answers: a restart is the remedy
            many = self.exits and self.failures > 0 and len(self.recent) >= self.failures
            why = f"{len(self.recent)} failures within {self.window_s:.0f} s" if many else None
        except Exception as e:  # noqa: BLE001
            why = f"the device is lost ({C.why(e)})"  # the type, and a CUDA error's own words (ZDR: no other message)
        self.beat()
        if why is None:
            self._log(f"a turn failed ({type(error).__name__}); the device answers: going on")
            return None
        self.dead = self.dead or why
        self._log(f"the engine is dead: {why}")
        if self.exits:
            return 503, f"the server's engine failed and the server is restarting: retry in {RETRY_AFTER['dead']} s"
        return None

    def after_failure(self) -> None:
        """After the requests were told and the engine was reset: a dead engine ends the process."""
        if self.dead and self.exits:
            self.end(EXIT_FATAL, f"exit {EXIT_FATAL}: {self.dead}")

    def end(self, code: int, why: str, grace_s: float = 0.5) -> None:
        """Ends the process: the HTTP threads have grace_s to write what they were told."""
        if self.exiting:
            return
        self.exiting = True
        self._log(why)
        time.sleep(grace_s)
        sys.stdout.flush()
        sys.stderr.flush()
        self._exit(code)

    # the watchdog

    def start(self, thread: threading.Thread) -> None:
        """The engine's thread is running; with exits, the watchdog starts too."""
        self.thread = thread
        self.beat()
        if self.exits and self.watchdog_s > 0:
            threading.Thread(target=self._watch, daemon=True, name="watchdog").start()

    def _watch(self) -> None:
        while not self.exiting:
            time.sleep(min(1.0, self.watchdog_s / 4))
            age = time.monotonic() - self.last_beat
            if age > self.watchdog_s or (self.thread is not None and not self.thread.is_alive()):
                why = f"no turn of the engine's thread has ended for {age:.1f} s" if age > self.watchdog_s else "the engine's thread has ended"
                self.dead = self.dead or why
                print(f"[serve] watchdog: {why}: every thread's stack follows", flush=True)
                try:
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                except (AttributeError, ValueError, OSError):  # a stderr without a file (captured)
                    faulthandler.dump_traceback(file=sys.__stderr__, all_threads=True)
                if self.thread is not None and self.thread.ident in sys._current_frames():  # and the engine's as Python sees it
                    import traceback

                    print("[serve] the engine's thread:\n" + "".join(traceback.format_stack(sys._current_frames()[self.thread.ident])), flush=True)
                self._tell_all(503, f"the server's engine hung and the server is restarting: retry in {RETRY_AFTER['dead']} s")
                self.end(EXIT_FATAL, f"exit {EXIT_FATAL}: {why}")
                return

    def _tell_all(self, status: int, message: str) -> None:
        s = self.server
        for req in list(s.active) + list(s.waiting):
            req.events.put(("error", status, message))
        while True:  # and those not yet taken in
            try:
                s.inbox.get_nowait().events.put(("error", status, message))
            except Exception:  # noqa: BLE001  (queue.Empty)
                break

    # the HTTP side

    def state(self) -> str:
        if self.dead or (self.thread is not None and not self.thread.is_alive()):
            return "dead"
        if self.draining:
            return "draining"
        if self.watchdog_s > 0 and time.monotonic() - self.last_beat > self.watchdog_s:
            return "stale"
        return "ok"

    def health(self, base: dict) -> tuple:
        """(status, body, headers) of GET /health. base: what /health says without the lifecycle (status "ok", counts)."""
        state = self.state()
        body = dict(base, status=state, beat_s=round(time.monotonic() - self.last_beat, 2))
        if state == "ok":
            return 200, body, {}
        body["reason"] = self.dead if state == "dead" else ("the server is stopping" if state == "draining" else "the engine's thread is not answering")
        return 503, body, {"Retry-After": str(RETRY_AFTER[state])}

    def refusal(self):
        """(status name, message, Retry-After) for a new request, or None: it is served. A dead or stopping server answers
        503 at once instead of taking the request in (a stale one takes it: the watchdog decides)."""
        state = self.state()
        if not self.exits or state in ("ok", "stale"):
            return None
        why = "the server is restarting" if state == "dead" else "the server is stopping"
        return state, f"{why}: retry in {RETRY_AFTER[state]} s", RETRY_AFTER[state]

    def _unsent(self) -> bool:
        """Whether a request's answer is still to be taken by its HTTP thread (one whose client left does not count)."""
        return any(not r.events.empty() and not r.cancelled for r in list(self.requests))

    def drain(self, most_s: float) -> None:
        """A graceful stop: new requests are refused; those under way or waiting end and their answers are taken by their
        HTTP threads, for at most most_s seconds; then a moment for the last bytes to go out."""
        self.draining = True
        s = self.server
        t = time.monotonic()
        busy = lambda: bool(s.active or s.waiting or s.batch.jobs and any(not j.only_read for j in s.batch.jobs) or not s.inbox.empty())  # noqa: E731
        self._log(f"stopping: {len(s.active) + len(s.waiting) + s.inbox.qsize()} requests are let finish (at most {most_s:.0f} s)")
        while (busy() or self._unsent()) and time.monotonic() - t < most_s and not self.dead:
            try:
                time.sleep(0.05)
            except KeyboardInterrupt:  # a second Ctrl+Break (the console's and the supervisor's): already stopping
                pass
        left = busy() or self._unsent()
        try:
            time.sleep(0.3)  # the HTTP threads write what they took
        except KeyboardInterrupt:
            pass
        self._log("stopped" + (" with requests still under way" if left else ""))

    def _log(self, what: str) -> None:
        self.events.append((time.time(), what))
        print(f"[serve] {what}", flush=True)


_OPS = []  # the Ops of this process (install): a Ctrl+Break marks it draining at once


def install(server, cli=None, exits: bool = True, watchdog_s: float = 15.0, probe=cuda_probe, failures: int = 3, **more):
    """--crash-only's lifecycle hooked into a Server (scripts/serve.py): the heartbeat, the failure handling, the idle
    thread waking every second, the requests tracked for a stop; and SCLMD_FAULT (tests only). cli: serve.py's options
    (None without --crash-only: nothing is installed, the server is as before) -> the Ops or None."""
    if cli is not None:
        if not cli.crash_only:
            return None
        watchdog_s, failures = cli.watchdog_s, cli.restart_failures
    ops = server.ops = Ops(server, exits, watchdog_s, probe, failures, **more)
    _OPS[:] = [ops]
    server.idle_wait = 1.0
    room = server._make_room

    def make_room():  # parking a long conversation takes seconds: each one beats (from a pass and from _place)
        ops.beat()
        try:
            return room()
        finally:
            ops.beat()

    server._make_room = make_room
    if server.batch.make_room is not None:
        server.batch.make_room = make_room
    submit = server.submit

    def tracked(req):
        ops.requests.add(req)
        return submit(req)

    server.submit = tracked
    if cli is not None:
        inject_fault(server.eng, os.environ.get("SCLMD_FAULT"))
    return ops


def started(app, ops, engine_thread: threading.Thread) -> None:
    """The engine's thread runs: the watchdog starts (--crash-only), and with --image auto and no image yet the image
    is written in the background while the server serves; GET / says whether the engine came from an image."""
    if ops is not None:
        ops.start(engine_thread)
    image = getattr(app.server.eng, "image", None)
    if image:
        app.facts["engine_image"] = {"used": image["used"], "load_s": image["load_s"]}
        if image.get("write"):
            image.pop("write")()


# ---- the HTTP side of --crash-only, and the placeholders

def answer_503(handler, state: str, reason: str, retry: int, max_body: int = 0) -> None:
    """A placeholder's (or a dead or stopping server's) answer to any request: 503 "state" with Retry-After, and the
    connection closed after it. A body is not read first: a Content-Length that is not a number is answered 400 (none,
    on a POST: 411; over max_body: 413) as the server answers it; what the client sends is then thrown away (linger)."""
    handler.close_connection = True
    path = handler.path.split("?")[0].rstrip("/")
    has_body = handler.command not in ("GET", "HEAD") or handler.headers.get("Content-Length") is not None
    refusal = C.body_refusal(handler.headers, max_body) if has_body else None
    headers = {"Connection": "close"}
    if refusal is not None:
        status, unread = refusal[0].args[0], refusal[1]
        body = {"error": {"message": refusal[0].args[1], "type": "invalid_request_error", "param": None, "code": None}}
    else:
        status, unread = 503, (int(handler.headers.get("Content-Length") or 0) if has_body else 0)
        headers["Retry-After"] = str(retry)
        body = ({"status": state, "reason": reason} if path == "/health" else
                {"error": {"message": f"the server is {state}: retry in {retry} s" if state in ("starting", "restarting", "asleep") else reason,
                           "type": "server_error", "param": None, "code": None}})
    data = json.dumps(body, ensure_ascii=False).encode()
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(data)))
        for name, value in headers.items():
            handler.send_header(name, value)
        handler.end_headers()
        handler.wfile.write(data)
        handler.wfile.flush()
    except OSError:
        return
    if unread != 0:  # None: not known, read until the client stops (bounded)
        C.linger(handler.connection, handler.rfile, unread)


def placeholder(state) -> type:
    """A handler that answers every request 503 (answer_503); state() -> (state, reason, Retry-After seconds)."""

    class Placeholder(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "sclmd"
        timeout = PLACEHOLDER_TIMEOUT

        def log_message(self, *args) -> None:
            pass

        def _answer(self) -> None:
            answer_503(self, *state())

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _answer

    return Placeholder


Starting = placeholder(lambda: ("starting", "the engine is loading", RETRY_AFTER["starting"]))


def crash_only(handler: type) -> type:
    """serve.py's Handler with --crash-only: /health from the lifecycle (Ops.health over what it would say), and a new
    chat request answered 503 (answer_503) while the server is dead or stopping. The rest is the Handler's own."""

    class CrashOnly(handler):
        _health = False

        def do_GET(self) -> None:
            self._health = self.path.split("?")[0].rstrip("/") == "/health"
            try:
                super().do_GET()
            finally:
                self._health = False

        def _json(self, status: int, obj: dict, headers: dict | None = None) -> None:
            ops = getattr(self.server.app.server, "ops", None)
            if self._health and status == 200 and ops is not None:
                status, obj, more = ops.health(obj)
                headers = {**(headers or {}), **more}
            return super()._json(status, obj, headers)

        def do_POST(self) -> None:
            app = self.server.app
            ops = getattr(app.server, "ops", None)
            refused = ops.refusal() if ops is not None else None
            if refused is not None and self.path.split("?")[0].rstrip("/") == "/v1/chat/completions":
                return answer_503(self, *refused, max_body=getattr(app, "max_body", C.MAX_BODY))
            return super().do_POST()

    CrashOnly.__name__ = CrashOnly.__qualname__ = f"CrashOnly{handler.__name__}"
    return CrashOnly


def inject_fault(eng, spec: str | None) -> None:
    """SCLMD_FAULT, for tests of the lifecycle only (read with --crash-only only). "tile:N": before the N-th pass a tile
    number far out of the pool is written into the tables (the kernels read and write an illegal address: a sticky
    CUDA error); "hang:S:N": the engine's thread is held S seconds in its N-th pass. With SCLMD_FAULT_FILE, only while
    that file is there (it is removed when the fault is set: a restarted server runs clean)."""
    if not spec:
        return
    once = os.environ.get("SCLMD_FAULT_FILE")
    if once:
        if not os.path.exists(once):
            return
        os.remove(once)
    kind, *args = spec.split(":")
    n, count = int(args[-1]), [0]
    if kind == "tile":
        provide = eng._provide

        def bad(need):
            provide(need)
            count[0] += 1
            if count[0] == n:
                kinds = list(eng.tables)
                eng.tables[kinds[0]][need[0][0], 0] = 1 << 27

        eng._provide = bad
    elif kind == "hang":
        run, s = eng.run, float(args[0])

        def slow(*a, **k):
            count[0] += 1
            if count[0] == n:
                time.sleep(s)
            return run(*a, **k)

        eng.run = slow
    else:
        raise SystemExit(f"SCLMD_FAULT={spec}: tile:N or hang:S:N")
    print(f"[serve] SCLMD_FAULT={spec}: a fault for a test of the lifecycle is set", flush=True)


def bind_early(cli):
    """--crash-only: the port taken before the engine loads, answering 503 "starting" (with --max-connections as the
    server's); Ctrl+Break then stops the server as Ctrl+C does (a supervisor's graceful stop). None without it."""
    if not cli.crash_only:
        return None
    ctrl_break_stops()
    httpd = C.BoundedHTTPServer((cli.host, cli.port), Starting, cli.max_connections)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="http").start()
    return httpd


def serving(httpd, handler: type, app):
    """The port taken early serves the loaded server from now on (connections opened before keep their placeholder
    for one answer, which closes them)."""
    httpd.app = app
    httpd.RequestHandlerClass = crash_only(handler)
    return httpd


def until_stopped(ops: Ops, drain_s: float) -> None:
    """The main thread of a server whose port serves in a thread of its own: until Ctrl+C or Ctrl+Break, then a graceful
    stop. (A Ctrl+Break's handler runs between this thread's sleeps; it marks the server draining itself.)"""
    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        ops.drain(drain_s)


class Later:
    """fn() in a thread of its own when `apart` (the tokenizer while the image is read), else at once; get(): its
    result, or its error."""

    def __init__(self, fn, apart: bool = True):
        self.out, self.error, self.thread = None, None, None

        def run():
            try:
                self.out = fn()
            except BaseException as e:  # noqa: BLE001
                self.error = e

        if apart:
            self.thread = threading.Thread(target=run, daemon=True)
            self.thread.start()
        else:
            run()

    def get(self):
        if self.thread is not None:
            self.thread.join()
        if self.error is not None:
            raise self.error
        return self.out


def ctrl_break_stops() -> None:
    """Ctrl+Break (the signal a supervisor sends to stop the server gracefully) as Ctrl+C; a server already serving is
    marked draining at once (no new request is taken in from then on)."""
    import signal

    if hasattr(signal, "SIGBREAK"):
        def stop(*_):
            for ops in _OPS:
                ops.draining = True
            raise KeyboardInterrupt
        signal.signal(signal.SIGBREAK, stop)
