# The server's lifecycle (R3) without a model or a GPU: the real Server and Handler (scripts/serve.py) over a stand-in
# engine whose passes write a token a job and can be made to fail; the placeholders (scripts/serve_ops.py) and
# scripts/supervise.py over a stand-in server. Without --crash-only nothing of it is installed: the server is as before.
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from scripts import serve_ops
from scripts.serve import Handler, Request, Server
from sclmd import model as M

STOP = (7,)
OPENING, CLOSING = 1, 2


class Letters:
    """A tokenizer whose token n is a letter."""

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(97 + i % 26) for i in ids)


class FakeEngine:
    """What the Server reads of an engine's bookkeeping, without a GPU."""

    def __init__(self, sequences: int = 2):
        self.cfg = SimpleNamespace(layer_types=[M.SLIDING, M.FULL], kv_heads={M.SLIDING: 1, M.FULL: 1}, head_dim={M.SLIDING: 32, M.FULL: 32},
                                   window=8)
        self.sequences, self.capacity, self.rows, self.lean, self.kv4 = sequences, 1024, 8, False, False
        self.ring, self.pool = {M.SLIDING: 2, M.FULL: 64}, {M.SLIDING: 4, M.FULL: 128}
        self.extra = {M.SLIDING: 0, M.FULL: 0}
        self.lengths = [0] * sequences

    def free_tiles(self) -> dict:
        return dict(self.pool)

    def release(self, seq: int) -> None:
        self.lengths[seq] = 0

    def _kept(self, first: int, seq: int = 0) -> bool:
        return True

    def crop(self, length: int, seq: int = 0) -> None:
        self.lengths[seq] = length

    def trim(self, seq: int) -> None:
        pass


class StandIn:
    """The real Server on a FakeEngine. Its passes give every job one token (10, 11, ...); fault() makes every pass
    from then on raise as a sticky CUDA error does; hang(s) makes the next pass take s seconds."""

    def __init__(self, sequences: int = 2):
        self.server = Server(None, FakeEngine(sequences), None, STOP)
        self.failing, self.hang_s, self.passes = False, 0.0, 0
        self.server.batch.step = self.step

    def step(self) -> bool:
        batch = self.server.batch
        batch.jobs = [j for j in batch.jobs if j.done is None]
        if not batch.jobs:  # as Batcher.step: nothing to run, no pass
            return False
        if self.failing:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        if self.hang_s:
            s, self.hang_s = self.hang_s, 0.0
            time.sleep(s)
        for j in batch.jobs:
            if j.done is None and j.cancel:
                j.done = "cancelled"
            elif j.done is None:
                j.todo = []
                j._settle(10 + len(j.out))
        batch.jobs = [j for j in batch.jobs if j.done is None]
        self.passes += 1
        return True

    def fault(self) -> None:
        self.failing = True


class StubApp:
    """App's part that the Handler uses, without a tokenizer: a request is four tokens and up to `max_tokens` (3) of reply."""

    tok, opening, closing, fingerprint = Letters(), OPENING, CLOSING, "sclmd-test"

    def __init__(self, server: Server):
        self.server, self.count = server, 0

    def request(self, body: dict) -> Request:
        self.count += 1  # every request its own conversation
        return Request([3 + self.count % 250] * 4, 3, int(body.get("max_tokens") or 3), None, False, body["messages"])

    def pieces(self, req):
        from sclmd.text import Pieces

        return Pieces(self.tok, self.server.stop, self.opening, self.closing)


def serve_standin(standin: StandIn, run: bool = True, handler=Handler, port: int = 0):
    """An HTTP server on a free port over the stand-in, and (run) the engine's thread."""
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    httpd.daemon_threads = True
    httpd.app = StubApp(standin.server)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    if run:
        threading.Thread(target=standin.server.run, daemon=True).start()
    return httpd


def call(port: int, method: str, path: str, body: dict | None = None, timeout: float = 20, conn=None) -> tuple:
    """(status, parsed body, headers) of one request (on `conn` when given: a connection that is kept)."""
    own = conn is None
    conn = conn or http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    if own:
        conn.close()
    try:
        parsed = json.loads(data)
    except ValueError:
        parsed = data.decode(errors="replace")
    return resp.status, parsed, dict(resp.getheaders())


def chat(port: int, timeout: float = 20, conn=None) -> tuple:
    return call(port, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}, timeout, conn)


def raw(port: int, data: bytes, wait: float = 5.0) -> tuple:
    """Sends bytes as they are and reads until the server closes or `wait` passes -> (the answer's bytes, seconds until
    the first byte, seconds until it closed or None)."""
    s = socket.create_connection(("127.0.0.1", port), timeout=wait)
    t = time.time()
    s.sendall(data)
    got, first, closed = b"", None, None
    try:
        while True:
            piece = s.recv(65536)
            if first is None:
                first = time.time() - t
            if not piece:
                closed = time.time() - t
                break
            got += piece
    except (socket.timeout, ConnectionError):
        pass
    s.close()
    return got, first, closed


def status_of(answer: bytes) -> int:
    return int(answer.split(b" ", 2)[1]) if answer.startswith(b"HTTP/") else 0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def standin():
    s = StandIn()
    httpd = serve_standin(s)
    yield s, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def lifecycle(s: StandIn, **more):
    """--crash-only's lifecycle (scripts/serve_ops.py) on the stand-in: its probe fails while the stand-in fails, and an
    exit is noted instead of ending the test."""
    exits = []

    def probe():
        if s.failing:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    ops = serve_ops.install(s.server, probe=probe, exit=exits.append, **more)
    return ops, exits


def started(s: StandIn, ops):
    """The stand-in served as serve.py serves it with --crash-only (its Handler through serve_ops.crash_only)."""
    httpd = serve_standin(s, run=False, handler=serve_ops.crash_only(Handler))
    thread = threading.Thread(target=s.server.run, daemon=True)
    thread.start()
    ops.start(thread)
    return httpd, httpd.server_address[1]


def wait_for(cond, most: float = 10.0) -> float:
    t = time.time()
    while not cond():
        assert time.time() - t < most, "it did not happen"
        time.sleep(0.02)
    return time.time() - t


# ---- without --crash-only: as before R3

def test_without_crash_only_nothing_is_installed_and_the_server_is_as_before(standin):
    """R3's kill test (2): without the lifecycle a sticky fault makes every request a 500 and /health says ok, as on
    master; install() with serve.py's options and no --crash-only installs nothing."""
    s, port = standin
    cli = SimpleNamespace(crash_only=False, watchdog_s=15.0, restart_failures=3)
    assert serve_ops.install(s.server, cli) is None and s.server.ops is None and s.server.idle_wait is None
    status, body, headers = call(port, "GET", "/health")
    assert status == 200 and body == {"status": "ok", "model": "gemma-4-31b-it", "under_way": 0, "waiting": 0, "kept": 0}
    assert "Connection" not in headers
    assert chat(port)[0] == 200
    s.fault()
    for _ in range(3):
        status, body, _ = chat(port)
        assert status == 500 and body["error"]["message"] == "the server failed while answering: RuntimeError"
        status, body, _ = call(port, "GET", "/health")
        assert status == 200 and body["status"] == "ok" and "beat_s" not in body


def test_the_options_are_off_by_default():
    import argparse

    ap = argparse.ArgumentParser()
    serve_ops.add_options(ap)
    cli = ap.parse_args([])
    assert cli.image == "off" and not cli.crash_only and serve_ops.image_mode(cli, False) == "off"
    assert serve_ops.bind_early(SimpleNamespace(crash_only=False)) is None
    calls = []
    later = serve_ops.Later(lambda: calls.append(threading.current_thread()) or 7, apart=False)
    assert calls == [threading.current_thread()] and later.get() == 7  # without --image the tokenizer loads where it did


# ---- --crash-only

def test_crash_only_health_tells_the_truth_and_ends_the_process_soon_after_a_sticky_fault():
    s = StandIn()
    ops, exits = lifecycle(s)
    httpd, port = started(s, ops)
    assert chat(port)[0] == 200
    status, body, _ = call(port, "GET", "/health")
    assert status == 200 and body["status"] == "ok" and {"model", "under_way", "waiting", "kept", "beat_s"} <= set(body)
    s.fault()
    t = time.time()
    status, body, _ = chat(port)  # the request under way is told the server restarts
    assert status == 503 and "restarting" in body["error"]["message"]
    wait_for(lambda: exits, 5.0)
    assert exits == [75] and time.time() - t < 5.0
    status, body, headers = chat(port)  # and a new one is not taken in
    assert status == 503 and int(headers["Retry-After"]) > 0 and headers["Connection"] == "close"
    status, body, headers = call(port, "GET", "/health")
    assert status == 503 and body["status"] == "dead" and "device is lost" in body["reason"] and int(headers["Retry-After"]) > 0
    httpd.shutdown()


def test_one_failure_on_a_device_that_answers_goes_on_three_in_a_minute_do_not():
    s = StandIn()
    ops, exits = lifecycle(s)
    httpd, port = started(s, ops)
    step, left = s.step, [1]

    def once():  # a failure that is not the device's: the probe passes
        if left[0]:
            left[0] -= 1
            raise ValueError("a bug in one request")
        return step()

    s.server.batch.step = once
    assert chat(port)[0] == 500
    assert chat(port)[0] == 200 and call(port, "GET", "/health")[0] == 200 and not exits
    left[0] = 2
    assert chat(port)[0] == 500
    status, body, _ = chat(port)  # the third failure within the minute
    assert status == 503
    wait_for(lambda: exits)
    assert exits == [75] and "3 failures" in ops.dead
    httpd.shutdown()


def test_restart_failures_0_restarts_only_for_a_lost_device():
    s = StandIn()
    ops, exits = lifecycle(s, failures=0)
    httpd, port = started(s, ops)
    step, left = s.step, [5]

    def flaky():
        if left[0]:
            left[0] -= 1
            raise ValueError("a bug in one request")
        return step()

    s.server.batch.step = flaky
    assert [chat(port)[0] for _ in range(5)] == [500] * 5
    assert chat(port)[0] == 200 and not exits and ops.dead is None
    httpd.shutdown()


def test_the_watchdog_ends_a_hung_engine_thread():
    s = StandIn()
    ops, exits = lifecycle(s, watchdog_s=1.0)
    httpd, port = started(s, ops)
    assert chat(port)[0] == 200
    s.hang_s = 4.0
    got = {}
    t = time.time()
    threading.Thread(target=lambda: got.setdefault("chat", chat(port)), daemon=True).start()
    wait_for(lambda: exits, 5.0)
    assert exits == [75] and "no turn" in ops.dead and time.time() - t < 3.0
    wait_for(lambda: "chat" in got)
    assert got["chat"][0] == 503  # the request under way is told
    httpd.shutdown()


def test_health_says_stale_while_the_engine_thread_hangs():
    s = StandIn()
    ops, exits = lifecycle(s, exits=False, watchdog_s=1.0)  # the state alone, without the watchdog's exit
    httpd, port = started(s, ops)
    s.hang_s = 3.0
    got = {}
    threading.Thread(target=lambda: got.setdefault("chat", chat(port)), daemon=True).start()
    wait_for(lambda: call(port, "GET", "/health")[1]["status"] == "stale", 2.5)
    status, body, headers = call(port, "GET", "/health")
    assert status == 503 and "Retry-After" in headers
    wait_for(lambda: "chat" in got, 5.0)
    assert got["chat"][0] == 200 and call(port, "GET", "/health")[0] == 200 and not exits  # it was only slow
    httpd.shutdown()


def test_an_idle_engine_thread_beats():
    s = StandIn()
    ops, _ = lifecycle(s, watchdog_s=1.5)
    httpd, port = started(s, ops)
    time.sleep(3.0)  # nothing to do: the thread wakes every second
    status, body, _ = call(port, "GET", "/health")
    assert status == 200 and body["beat_s"] < 1.5
    httpd.shutdown()


def test_a_graceful_stop_lets_the_request_under_way_finish_and_refuses_new_ones():
    s = StandIn()
    ops, exits = lifecycle(s)
    httpd, port = started(s, ops)
    step = s.step

    def slow():
        time.sleep(0.05)
        return step()

    s.server.batch.step = slow
    got = {}
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 30}
    first = threading.Thread(target=lambda: got.setdefault("first", call(port, "POST", "/v1/chat/completions", body)))
    first.start()
    wait_for(lambda: s.server.active)
    drain = threading.Thread(target=ops.drain, args=(10.0,))
    drain.start()
    wait_for(lambda: ops.draining)
    status, refused, headers = chat(port)
    assert status == 503 and "stopping" in refused["error"]["message"] and "Retry-After" in headers and headers["Connection"] == "close"
    assert call(port, "GET", "/health")[1]["status"] == "draining"
    first.join(10)
    drain.join(10)
    assert got["first"][0] == 200 and len(got["first"][1]["choices"][0]["message"]["content"]) == 30 and not drain.is_alive()
    httpd.shutdown()


def test_a_stop_waits_for_the_answers_to_be_taken():
    """The engine's lists are empty once a reply is settled; the stop waits until its HTTP thread has taken it (a client
    that left does not hold it up)."""
    s = StandIn()
    ops, _ = lifecycle(s)
    req = Request([3, 4], 1, 1, None, False)
    req.events.put(("done", {}))  # settled, not yet taken
    ops.requests.add(req)
    t = time.time()
    ops.drain(1.0)
    assert 1.0 <= time.time() - t < 2.0 and "still under way" in ops.events[-1][1]
    req.cancelled = True  # its client left
    t = time.time()
    ops.drain(5.0)
    assert time.time() - t < 1.0 and ops.events[-1][1] == "stopped"


def test_ctrl_break_marks_a_serving_server_draining_at_once():
    if not hasattr(signal, "SIGBREAK"):
        pytest.skip("Windows: Ctrl+Break")
    before = signal.getsignal(signal.SIGBREAK)
    s = StandIn()
    ops, _ = lifecycle(s)
    try:
        serve_ops.ctrl_break_stops()
        with pytest.raises(KeyboardInterrupt):
            signal.getsignal(signal.SIGBREAK)(signal.SIGBREAK, None)
        assert ops.draining and ops.refusal()[0] == "draining"
    finally:
        signal.signal(signal.SIGBREAK, before)


def test_the_lifecycles_log_is_bounded():
    ops, _ = lifecycle(StandIn())
    for k in range(1000):
        ops._log(f"line {k}")
    assert len(ops.events) == 256 and ops.events[-1][1] == "line 999"


# ---- the placeholders: 503, the connection closed after it, hostile lengths

def test_the_port_answers_503_while_the_engine_loads_and_a_kept_connection_reaches_the_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve_ops.Starting)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)  # one connection, kept as a pooled client keeps it
    status, body, headers = call(port, "GET", "/health", conn=conn)
    assert status == 503 and body["status"] == "starting" and headers["Retry-After"] == "15" and headers["Connection"] == "close"
    status, body, headers = chat(port, conn=conn)
    assert status == 503 and body["error"]["type"] == "server_error" and headers["Connection"] == "close"
    s = StandIn()  # loaded: the same port serves (serve.py swaps the handler)
    threading.Thread(target=s.server.run, daemon=True).start()
    serve_ops.serving(httpd, Handler, StubApp(s.server))
    assert chat(port, conn=conn)[0] == 200 and call(port, "GET", "/health", conn=conn)[0] == 200  # the same connection object
    httpd.shutdown()


def test_the_supervisor_holds_the_port_serve_toml_gives(tmp_path):
    """With R4's serve.toml the server may take another host and port than its command line says: the supervisor holds
    and watches that one (the command line still wins; --config "" reads no file)."""
    from scripts.supervise import served_at

    cfg = tmp_path / "serve.toml"
    cfg.write_text('port = 8044\nhost = "0.0.0.0"\n', encoding="utf-8")
    none = str(tmp_path / "absent.toml")
    assert served_at([], none) == ("127.0.0.1", 8001)
    assert served_at([], str(cfg)) == ("0.0.0.0", 8044)
    assert served_at(["--port", "8050"], str(cfg)) == ("0.0.0.0", 8050)
    assert served_at(["--port=8051", "--host", "127.0.0.1"], str(cfg)) == ("127.0.0.1", 8051)
    assert served_at(["--config", ""], str(cfg)) == ("127.0.0.1", 8001)
    other = tmp_path / "other.toml"
    other.write_text("port = 8060\n", encoding="utf-8")
    assert served_at(["--config", str(other)], none) == ("127.0.0.1", 8060)


def test_the_supervisors_placeholder_closes_kept_connections():
    from scripts.supervise import Holder

    port, lines = free_port(), []
    holder = Holder("127.0.0.1", port, lines.append)
    holder.start(5.0)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    status, body, headers = call(port, "GET", "/health", conn=conn)
    assert status == 503 and body["status"] == "restarting" and int(headers["Retry-After"]) >= 15 and headers["Connection"] == "close"
    status, body, headers = chat(port, conn=conn)
    assert status == 503 and "restarting" in body["error"]["message"]
    holder.stop()
    s = StandIn()  # the server is up again on the port
    httpd = serve_standin(s, port=port)
    assert chat(port, conn=conn)[0] == 200 and call(port, "GET", "/health", conn=conn)[0] == 200
    httpd.shutdown()
    httpd.server_close()


@pytest.mark.parametrize("where", ["starting", "dead"])
def test_hostile_lengths_are_answered_at_once_and_never_hang(where):
    if where == "starting":
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve_ops.Starting)
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        port = httpd.server_address[1]
    else:  # a dead server (--crash-only) refusing a new request
        s = StandIn()
        ops, exits = lifecycle(s)
        httpd, port = started(s, ops)
        ops.dead = "a test"
    post = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
    for length, want in ((b"abc", 400), (b"-5", 400), (b"1e3", 400)):
        answer, first, closed = raw(port, post + b"Content-Length: " + length + b"\r\n\r\n{}")
        assert status_of(answer) == want and b"Connection: close" in answer and first < 1.0 and closed is not None and closed < 3.0, (length, answer, first, closed)
    answer, first, closed = raw(port, post + b"\r\n")  # no length
    assert status_of(answer) == 411 and closed is not None and closed < 3.0
    # a length announced and not sent: answered before the body, the connection closed a moment after (linger's idle)
    answer, first, closed = raw(port, post + b"Content-Length: 1000000000\r\n\r\n")
    assert status_of(answer) == 503 and b"Retry-After" in answer and first < 1.0 and closed is not None and closed < 3.0, (answer, first, closed)
    # a whole body sent before reading: the answer, not a reset
    body = json.dumps({"messages": [{"role": "user", "content": "x" * 200_000}]}).encode()
    answer, first, closed = raw(port, post + b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    assert status_of(answer) == 503 and closed is not None
    httpd.shutdown()


# ---- scripts/supervise.py over tests/lifecycle_child.py (subprocesses; no GPU)

WINDOWS = pytest.mark.skipif(os.name != "nt", reason="Windows process groups and Ctrl+Break")


def alive(pid: int) -> bool:
    """Whether a process runs (OpenProcess and its exit code; never a signal)."""
    import ctypes

    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value == 259  # STILL_ACTIVE


def supervise(tmp_path, env: dict, *more) -> tuple:
    """-> (the supervisor's process, its port: a free one)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    port = free_port()
    cmd = [sys.executable, "-u", "-m", "scripts.supervise", "--module", "tests.lifecycle_child", "--log", str(tmp_path / "sup.log"), *more,
           "--port", str(port)]
    full = dict(os.environ, R3_CHILD_LOG=str(tmp_path / "child.log"), **env)
    proc = subprocess.Popen(cmd, cwd=root, env=full, stdout=open(tmp_path / "sup.out", "w"), stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    return proc, port


def statuses(port: int, seconds: float, until=None) -> list:
    """/health on the supervised port every 0.1 s: [(t, its status or the error's name, its body's status)]."""
    out, t = [], time.time()
    while time.time() - t < seconds:
        try:
            status, body, _ = call(port, "GET", "/health", timeout=2)
            out.append((round(time.time() - t, 2), status, body.get("status") if isinstance(body, dict) else None))
        except OSError as e:
            out.append((round(time.time() - t, 2), type(e).__name__, None))
        if until is not None and until(out):
            break
        time.sleep(0.1)
    return out


def lines(path) -> list:
    try:
        with open(path, encoding="utf-8") as f:
            return f.readlines()
    except OSError:
        return []


def stop(sup) -> int:
    os.kill(sup.pid, signal.CTRL_BREAK_EVENT)
    try:
        return sup.wait(30)
    finally:
        if sup.poll() is None:  # the test failed: nothing it started is left behind
            subprocess.run(["taskkill", "/PID", str(sup.pid), "/T", "/F"], capture_output=True)


@WINDOWS
def test_the_supervisor_starts_the_server_again_with_a_back_off_and_stops_it_gracefully(tmp_path):
    sup, port = supervise(tmp_path, {"R3_CHILD_FAILS": "2", "R3_CHILD_AFTER": "0.3"})
    try:
        seen = statuses(port, 30, until=lambda o: len(lines(tmp_path / "child.log")) >= 3 and sum(1 for x in o[-5:] if x[2] == "ok") >= 5)
    finally:
        code = stop(sup)
    runs = [line.split() for line in lines(tmp_path / "child.log")]
    log = "".join(lines(tmp_path / "sup.log"))
    assert code == 0, log
    assert len(runs) == 3, log  # two ends with 75, then one that serves
    assert log.count("(75) after") == 2 and "again in 1 s" in log and "again in 2 s" in log, log
    assert any(x[2] == "restarting" for x in seen), seen  # the port answered 503 while the server was down
    assert "gracefully" in log and "the supervisor stops" in log, log
    assert "stopping gracefully" in "".join(lines(tmp_path / "sup.out"))  # Ctrl+Break reached the server
    time.sleep(0.5)
    assert not any(alive(int(r[0])) for r in runs), runs  # nothing is left


@WINDOWS
def test_a_kept_connection_gets_the_server_again_after_a_restart(tmp_path):
    """The one keep-alive connection the earlier check found stuck on 503 "restarting": it reaches the server."""
    sup, port = supervise(tmp_path, {"R3_CHILD_FAILS": "1", "R3_CHILD_AFTER": "1.0"})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    seen = []
    try:
        t = time.time()
        while time.time() - t < 30 and not (len(lines(tmp_path / "child.log")) >= 2 and seen[-3:] == [(200, "ok")] * 3):
            try:
                status, body, _ = call(port, "GET", "/health", conn=conn)
                seen.append((status, body.get("status") if isinstance(body, dict) else None))
            except (OSError, http.client.HTTPException) as e:
                seen.append((type(e).__name__, None))
                conn.close()
            time.sleep(0.1)
    finally:
        code = stop(sup)
    assert (503, "restarting") in seen and seen[-3:] == [(200, "ok")] * 3, seen
    assert code == 0


@WINDOWS
def test_a_sleep_file_stops_the_server_and_its_removal_starts_it_again(tmp_path):
    sleep = tmp_path / "serve_sleep"
    sup, port = supervise(tmp_path, {}, "--sleep-file", str(sleep))
    try:
        statuses(port, 20, until=lambda o: o[-1][2] == "ok")
        sleep.touch()
        t = time.time()
        seen = statuses(port, 20, until=lambda o: o[-1][2] == "asleep")
        asleep_after = time.time() - t
        assert seen[-1][1] == 503
        time.sleep(1.0)
        assert call(port, "POST", "/v1/chat/completions", {"messages": []})[0] == 503  # requests are told to come back
        sleep.unlink()
        t = time.time()
        statuses(port, 20, until=lambda o: o[-1][2] == "ok")
        awake_after = time.time() - t
    finally:
        code = stop(sup)
    runs = [line.split() for line in lines(tmp_path / "child.log")]
    log = "".join(lines(tmp_path / "sup.log"))
    assert code == 0 and len(runs) == 2, log  # the first run stopped to sleep, the second after the wake
    assert "asleep while" in log and "awake after" in log and log.count("stopping the server") == 2, log
    assert "".join(lines(tmp_path / "sup.out")).count("stopping gracefully") == 2  # both ended through Ctrl+Break
    assert asleep_after < 10 and awake_after < 10, (asleep_after, awake_after)
    time.sleep(0.5)
    assert not any(alive(int(r[0])) for r in runs), runs


@WINDOWS
def test_a_server_that_stops_answering_is_stopped_and_started_again(tmp_path):
    sup, port = supervise(tmp_path, {"R3_CHILD_STUCK": "1"}, "--health-grace", "2", "--max-runs", "2")
    try:
        code = sup.wait(90)
    finally:
        if sup.poll() is None:  # the test failed: nothing it started is left behind
            subprocess.run(["taskkill", "/PID", str(sup.pid), "/T", "/F"], capture_output=True)
    runs = [line.split() for line in lines(tmp_path / "child.log")]
    log = "".join(lines(tmp_path / "sup.log"))
    assert len(runs) == 2 and log.count("and its children") == 2 and "has not said ok" in log, log
    assert code == 0
    time.sleep(0.5)
    assert not any(alive(int(r[0])) for r in runs), runs
