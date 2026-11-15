# The operator's side of the server (scripts/serve_admin.py and its hooks in scripts/serve.py) without a model or a
# GPU: the real App and Handler over a tokenizer of letters, and the test in the engine's place (it holds a request
# until it answers it, and "places" one by giving it a job).
import argparse
import http.client
import json
import math
import os
import re
import socket
import threading
import time

import pytest

from scripts import serve as S
from scripts import serve_admin as A
from scripts import serve_contract as C
from test_api_contract import Tok

DONE = {"finish_reason": "stop", "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        "timings": {"total_ms": 1500, "first_token_ms": 300, "tokens_per_second": 2.5}, "ended": "stop"}
HI = [{"role": "user", "content": "hi"}]


class Inbox:
    """In the engine's place: the requests that came, for the test to answer."""

    stop = (7,)

    def __init__(self):
        self.reqs, self.canon = [], None
        self.eng = type("Eng", (), {"capacity": 4096, "rows": 32})()
        self.prefixes = S.Prefixes(0)

    def submit(self, req) -> None:
        self.reqs.append(req)

    def next(self):
        t = time.time()
        while not self.reqs:
            assert time.time() - t < 10, "the request did not come"
            time.sleep(0.01)
        return self.reqs.pop(0)

    def about(self) -> dict:
        return {"requests": {"under_way": 0, "waiting": len(self.reqs)}, "conversations_kept": 0}


def options(*argv) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    A.add_options(ap)
    cli = ap.parse_args(list(argv))
    cli.host, cli.port = "127.0.0.1", 0
    return cli


def serve(*argv, keys=(), max_waiting: int = 0):
    """A server on a free port with the operator's options -> (httpd, app, inbox)."""
    cli = options(*argv, *(["--max-waiting", str(max_waiting)] if max_waiting else []))
    httpd = C.BoundedHTTPServer(("127.0.0.1", 0), S.Handler)
    inbox = Inbox()
    app = httpd.app = S.App(inbox, Tok(), False, 64, {"bits_per_weight": 3.0})
    A.install(app, inbox, cli, httpd, list(keys))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, app, inbox


@pytest.fixture()
def stand():
    made = []

    def make(*argv, **kw):
        made.append(serve(*argv, **kw))
        return made[-1]

    yield make
    for httpd, _, _ in made:
        httpd.shutdown()
        httpd.server_close()


def call(httpd, method: str, path: str, body=None, headers=None, raw: bytes | None = None) -> tuple:
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=20)
    data = raw if raw is not None else (json.dumps(body) if body is not None else None)
    conn.request(method, path, data, {"Content-Type": "application/json", **(headers or {})})
    r = conn.getresponse()
    out = r.read()
    conn.close()
    return r.status, {k.lower(): v for k, v in r.getheaders()}, out


def answer(req, tokens=(10, 11, 12), done=DONE) -> None:
    for t in tokens:
        req.events.put(("token", t))
    req.events.put(("done", done))


def chat_async(httpd, body: dict, headers=None) -> tuple:
    """A request in a thread of its own -> (thread, a dict that gets status, headers, body)."""
    out = {}

    def go():
        out["status"], out["headers"], out["body"] = call(httpd, "POST", "/v1/chat/completions", body, headers)

    th = threading.Thread(target=go)
    th.start()
    return th, out


# (a) API keys

def test_without_a_key_everything_is_as_before(stand):
    httpd, app, inbox = stand()
    for path in ("/", "/v1/models", "/health", "/metrics"):
        assert call(httpd, "GET", path)[0] == 200
    th, out = chat_async(httpd, {"messages": HI})
    answer(inbox.next())
    th.join(10)
    assert out["status"] == 200 and json.loads(out["body"])["choices"][0]["message"]["content"] == "klm"
    assert "retry-after" not in out["headers"]


def test_a_key_is_asked_of_every_request_but_health(stand):
    httpd, app, inbox = stand(keys=A.parse_keys("alice:sk-alice-0123456789abcdef, sk-plain-0123456789abcdef", "key"))
    for path in ("/", "/v1/models", "/metrics"):
        status, headers, body = call(httpd, "GET", path)
        assert status == 401 and json.loads(body)["error"]["code"] == "invalid_api_key" and "bearer" in headers["www-authenticate"].lower()
        assert call(httpd, "GET", path, headers={"Authorization": "Bearer wrong"})[0] == 401
        assert call(httpd, "GET", path, headers={"Authorization": "Bearer sk-alice-0123456789abcdef"})[0] == 200
        assert call(httpd, "GET", path, headers={"x-api-key": "sk-plain-0123456789abcdef"})[0] == 200
    assert call(httpd, "GET", "/health")[0] == 200
    # nothing reached the engine; the body is not read into memory: thrown away after the 401, and the connection closed
    for size in (10, 100_000, 3 << 20):
        big = json.dumps({"messages": [{"role": "user", "content": "x" * size}]})
        status, headers, body = call(httpd, "POST", "/v1/chat/completions", raw=big.encode())
        assert status == 401 and headers.get("connection") == "close" and not inbox.reqs, size
    assert call(httpd, "POST", "/v1/chat/completions", raw=b"")[0] == 401  # no body: the connection may go on
    th, out = chat_async(httpd, {"messages": HI}, {"Authorization": "bearer sk-alice-0123456789abcdef"})
    req = inbox.next()
    answer(req)
    th.join(10)
    assert out["status"] == 200
    assert app.admin.refused["unauthorized"] == 10


def test_a_keyless_post_that_announces_more_than_it_sends_does_not_hold_its_thread(stand):
    httpd, app, inbox = stand(keys=A.parse_keys("sk-only-0123456789abcdef", "key"))
    for announced in (1_000_000, 2_000_000_000):
        s = socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=10)
        s.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: {announced}\r\n\r\n".encode() + b'{"messages"')
        t = time.perf_counter()
        got = b""
        while b"\r\n\r\n" not in got:
            got += s.recv(65536)
        assert got.startswith(b"HTTP/1.1 401") and time.perf_counter() - t < 1.0  # at once, the body not waited for
        while True:  # then the server closes it: after 2 s without a byte (C.LINGER_IDLE_S), not when the length has come
            try:
                if not s.recv(65536):
                    break
            except (ConnectionResetError, ConnectionAbortedError):
                break
        assert time.perf_counter() - t < C.LINGER_IDLE_S + 2.0
        s.close()
    t = time.time()
    while httpd.live:  # every connection's thread is done
        assert time.time() - t < 5
        time.sleep(0.05)


def test_keys_come_from_the_environment_and_a_file_never_from_the_config(tmp_path, monkeypatch):
    monkeypatch.setenv(A.ENV_KEY, "a:sk-a-0123456789abcdef,sk-b-0123456789abcdef")
    f = tmp_path / "keys.txt"
    f.write_text("# the team\nbob: sk-bob-0123456789abcdef\n", encoding="utf-8")
    keys = A.load_keys(options("--api-key-file", str(f)))
    assert [n for n, _ in keys] == ["a", "key2", "bob"]
    assert A.key_name(keys, b"sk-bob-0123456789abcdef") == "bob" and A.key_name(keys, b"sk-b-0123456789abcdef") == "key2"
    assert A.key_name(keys, b"sk-bob-0123456789abcdeX") is None and A.key_name(keys, None) is None
    cfg = tmp_path / "serve.toml"
    cfg.write_text('api_key = "sk-x"\n', encoding="utf-8")
    ap = argparse.ArgumentParser()
    A.add_options(ap)
    with pytest.raises(SystemExit, match="API keys are not read"):
        A.parse_args(ap, ["--config", str(cfg)])


def test_a_host_other_than_loopback_without_a_key_warns_and_starts_unless_required(capsys, monkeypatch):
    monkeypatch.delenv(A.ENV_KEY, raising=False)
    cli = options()
    cli.host = "0.0.0.0"
    assert A.start(cli) == []
    assert "WARNING" in capsys.readouterr().out
    cli.require_key = True
    with pytest.raises(SystemExit, match="not started"):
        A.start(cli)
    cli.host, cli.require_key = "127.0.0.1", True
    assert A.start(cli) == []  # loopback needs no key, even with --require-key
    cli.host, cli.debug = "0.0.0.0", True
    monkeypatch.setenv(A.ENV_KEY, "sk-a-0123456789abcdef")
    with pytest.raises(SystemExit, match="loopback"):
        A.start(cli)


# (b) back-pressure

def flood(httpd, n: int, stream: bool = False) -> tuple:
    out, gate = [None] * n, threading.Barrier(n)

    def one(i):
        gate.wait()
        try:
            out[i] = call(httpd, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": stream})
        except Exception as e:  # noqa: BLE001
            out[i] = (None, {}, repr(e).encode())

    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for th in threads:
        th.start()
    return threads, out


@pytest.mark.parametrize("stream", [False, True])
def test_a_flood_past_max_waiting_gets_429_exactly_and_nothing_hangs(stand, stream):
    httpd, app, inbox = stand(max_waiting=20)
    threads, out = flood(httpd, 100, stream)
    t = time.time()
    while sum(o is not None for o in out) < 80 or len(inbox.reqs) < 20:
        assert time.time() - t < 20, f"{len(inbox.reqs)} came, {sum(o is not None for o in out)} answered"
        time.sleep(0.02)
    time.sleep(0.3)  # nothing more comes while the engine holds them
    assert len(inbox.reqs) == 20 and sum(o is not None and o[0] == 429 for o in out) == 80
    for status, headers, body in [o for o in out if o is not None]:
        err = json.loads(body)["error"]
        assert err["code"] == "rate_limit_exceeded" and err["type"] == "requests"
        assert 1 <= int(headers["retry-after"]) <= 60 and int(headers["retry-after-ms"]) >= 1000 and headers["x-should-retry"] == "true"
    for req in list(inbox.reqs):
        answer(req)
    inbox.reqs.clear()
    for th in threads:
        th.join(20)
    assert not any(th.is_alive() for th in threads)
    assert sorted(o[0] for o in out).count(200) == 20
    assert app.admin.refused["queue_full"] == 80 and not app.admin.inflight


def test_a_burst_is_taken_in_up_to_the_free_sequences_before_max_waiting_counts(stand):
    httpd, app, inbox = stand(max_waiting=4)
    inbox.unused, inbox.kept = [0, 1, 2], {3: object(), 4: object()}  # five sequences the engine's next turn gives at once
    threads, out = flood(httpd, 40)
    t = time.time()
    while sum(o is not None for o in out) < 31 or len(inbox.reqs) < 9:
        assert time.time() - t < 20, f"{len(inbox.reqs)} came, {sum(o is not None for o in out)} answered"
        time.sleep(0.02)
    time.sleep(0.3)
    assert len(inbox.reqs) == 9 and sum(o is not None and o[0] == 429 for o in out) == 31  # 5 free + 4 waiting
    assert "4 requests wait for a sequence" in json.loads(next(o for o in out if o is not None)[2])["error"]["message"]
    for req in list(inbox.reqs):
        answer(req)
    inbox.reqs.clear()
    for th in threads:
        th.join(20)
    assert sorted(o[0] for o in out).count(200) == 9


def test_a_request_with_a_sequence_does_not_count_as_waiting(stand):
    httpd, app, inbox = stand(max_waiting=1)
    th1, out1 = chat_async(httpd, {"messages": HI})
    r1 = inbox.next()
    status, headers, _ = call(httpd, "POST", "/v1/chat/completions", {"messages": HI})
    assert status == 429
    r1.job = object()  # the engine placed it: it is under way, not waiting
    th2, out2 = chat_async(httpd, {"messages": HI})
    r2 = inbox.next()
    answer(r1)
    answer(r2)
    th1.join(10)
    th2.join(10)
    assert out1["status"] == out2["status"] == 200


def test_retry_after_is_the_queue_over_the_measured_rate():
    server = type("S", (), {})()
    server.totals = {"requests": 10, "prompt_tokens": 9000, "cached_tokens": 1000, "completion_tokens": 2000}
    server.busy_s = 20.0  # 10,000 tokens in 20 s: 500 tok/s
    adm = A.Admin(None, server)
    waiting = [A.Entry(S.Request([0] * 1800, 1, 400, None, False), "x", None, False, None) for _ in range(5)]  # 5 x (1,800 + 200)
    assert adm.retry_after(waiting) == pytest.approx(20.0)
    assert adm.retry_after(waiting * 10) == A.RETRY_MOST and adm.retry_after([]) == 1.0


@pytest.mark.parametrize("stream", [False, True])
def test_a_request_that_waits_past_queue_timeout_gets_503_and_leaves_the_queue(stand, stream):
    httpd, app, inbox = stand("--queue-timeout", "0.6")
    t = time.perf_counter()
    status, headers, body = call(httpd, "POST", "/v1/chat/completions", {"messages": HI, "stream": stream})
    assert status == 503 and 0.5 < time.perf_counter() - t < 3  # a stream too: its headers had not gone out
    assert json.loads(body)["error"]["code"] == "queue_timeout" and int(headers["retry-after"]) >= 1
    req = inbox.next()
    assert req.cancelled  # the engine's thread drops it from its waiting list
    assert app.admin.refused["queue_timeout"] == 1


def test_a_placed_request_outlives_queue_timeout_and_streams_as_before(stand):
    httpd, app, inbox = stand("--queue-timeout", "0.5")
    th, out = chat_async(httpd, {"messages": HI, "stream": True})
    req = inbox.next()
    req.job = object()  # placed in time; its prompt takes longer than the timeout to read
    time.sleep(1.2)
    answer(req)
    th.join(10)
    assert out["status"] == 200 and out["body"].endswith(b"data: [DONE]\n\n")
    th, out = chat_async(httpd, {"messages": HI})
    req = inbox.next()
    req.events.put(("token", 10))  # its first token comes before it is seen to be placed
    time.sleep(0.8)
    answer(req, tokens=(11, 12))
    th.join(10)
    assert out["status"] == 200 and json.loads(out["body"])["choices"][0]["message"]["content"] == "klm"


# (c) /metrics

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\\n]|\\.)*"(?:,[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\\n]|\\.)*")*)\})? '
                  r'(-?[0-9.]+(?:e[+-]?[0-9]+)?|NaN|[+-]Inf)$')


def parse_metrics(text: str) -> dict:
    """A strict reader of Prometheus' text format 0.0.4 (in place of promtool): every sample belongs to the family
    declared last (HELP then TYPE, once each), counters end in _total, histograms' buckets are cumulative and end in
    +Inf = _count. -> {family: {"type", "samples": [(name, labels, value)]}}"""
    fams, current = {}, None
    assert text.endswith("\n")
    for line in text.splitlines():
        if line.startswith("# HELP "):
            name = line.split()[2]
            assert name not in fams, f"{name} declared twice"
            fams[name], current = {"type": None, "samples": []}, name
        elif line.startswith("# TYPE "):
            _, _, name, kind = line.split()
            assert name == current and fams[name]["type"] is None and kind in ("counter", "gauge", "histogram")
            fams[name]["type"] = kind
        else:
            m = LINE.match(line)
            assert m, f"not a sample: {line!r}"
            name, labels = m.group(1), dict(re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"', m.group(3) or ""))
            kind = fams[current]["type"]
            allowed = {current} if kind != "histogram" else {current + "_bucket", current + "_sum", current + "_count"}
            assert name in allowed, f"{name} under {current}"
            fams[current]["samples"].append((name, labels, float(m.group(4).replace("Inf", "inf"))))
    for name, f in fams.items():
        assert f["samples"], f"{name} has no samples"
        if f["type"] == "counter":
            assert name.endswith("_total"), name
        if f["type"] == "histogram":
            buckets = [(s[1]["le"], s[2]) for s in f["samples"] if s[0].endswith("_bucket")]
            values = [v for _, v in buckets]
            assert buckets[-1][0] == "+Inf" and values == sorted(values)
            assert [s[2] for s in f["samples"] if s[0].endswith("_count")] == [values[-1]]
    return fams


def test_metrics_are_prometheus_text_and_count_what_happened(stand):
    httpd, app, inbox = stand(max_waiting=1)
    th, out = chat_async(httpd, {"messages": HI})
    req = inbox.next()
    assert call(httpd, "POST", "/v1/chat/completions", {"messages": HI})[0] == 429
    answer(req)
    th.join(10)
    assert call(httpd, "POST", "/v1/chat/completions", {"messages": HI, "n": 3})[0] == 400
    status, headers, body = call(httpd, "GET", "/metrics")
    assert status == 200 and headers["content-type"].startswith("text/plain; version=0.0.4")
    fams = parse_metrics(body.decode())
    responses = {(s[1]["method"], s[1]["path"], s[1]["code"]): s[2] for s in fams["sclmd_http_responses_total"]["samples"]}
    assert responses[("POST", "/v1/chat/completions", "200")] == 1 and responses[("POST", "/v1/chat/completions", "429")] == 1
    assert responses[("POST", "/v1/chat/completions", "400")] == 1
    assert {s[1]["reason"]: s[2] for s in fams["sclmd_refused_total"]["samples"]}["queue_full"] == 1
    ttft = {s[1].get("le"): s[2] for s in fams["sclmd_ttft_seconds"]["samples"] if s[0].endswith("_bucket")}
    assert ttft["0.25"] == 0 and ttft["0.5"] == 1 and ttft["+Inf"] == 1  # 300 ms
    for name in ("sclmd_process_resident_bytes", "sclmd_process_threads", "sclmd_admission_waiting", "sclmd_uptime_seconds"):
        assert name in fams


def raw_request(httpd, data: bytes) -> bytes:
    """One request as bytes on a connection of its own -> what came back until the server closed it or 2 s passed."""
    s = socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=2)
    s.sendall(data)
    got = b""
    try:
        while True:
            piece = s.recv(65536)
            if not piece:
                break
            got += piece
    except OSError:
        pass
    s.close()
    return got


@pytest.mark.parametrize("keyed", [False, True])
def test_made_up_methods_and_paths_add_no_series(stand, keyed):
    httpd, app, inbox = stand(keys=A.parse_keys("sk-only-0123456789abcdef", "key") if keyed else ())
    auth = {"Authorization": "Bearer sk-only-0123456789abcdef"} if keyed else {}
    metrics = lambda: call(httpd, "GET", "/metrics", headers=auth)[2].decode()  # noqa: E731
    before = metrics()
    for i in range(300):  # answered 501 by http.server before any hook, and 404s on paths nobody serves
        assert raw_request(httpd, f"M{i}X /x HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n".encode()).startswith(b"HTTP/1.")
        call(httpd, "GET", f"/nothing/{i}", headers=auth)
    after = metrics()
    fams = parse_metrics(after)
    labels = {(s[1]["method"], s[1]["path"], s[1]["code"]) for s in fams["sclmd_http_responses_total"]["samples"]}
    assert {m for m, _, _ in labels} <= set(A.METHODS) | {"other"} and {p for _, p, _ in labels} <= set(A.PATHS) | {"other"}
    assert ("other", "other", "501") in labels and ("GET", "other", "404") in labels and len(labels) <= 6
    assert len(after) < len(before) + 2000  # a few new series, not 600
    series = lambda text: sorted((s[0], tuple(sorted(s[1].items()))) for f in parse_metrics(text).values() for s in f["samples"])  # noqa: E731
    for k in range(300, 600):  # and no more of them
        raw_request(httpd, f"M{k}X /x HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n".encode())
        call(httpd, "GET", f"/nothing/{k}", headers=auth)
    assert series(metrics()) == series(after)  # the same series: only their numbers changed


def test_labels_take_values_of_fixed_sets():
    assert A.method_label("GET") == "GET" and A.method_label("FOO") == "other" and A.method_label(None) == "other"
    assert A.code_label(200) == "200" and A.code_label(418) == "4xx" and A.code_label(999) == "other" and A.code_label("x") == "other"
    assert A.finish_label("tool_calls") == "tool_calls" and A.finish_label("whatever") == "other"


def test_a_failure_while_sending_is_the_servers_not_the_client_leaving(stand, tmp_path, monkeypatch):
    path = tmp_path / "requests.jsonl"
    httpd, app, inbox = stand("--request-log", str(path))

    def broken(*a, **k):  # the reply's loop fails on the server's side
        raise KeyError("a server bug")

    monkeypatch.setattr(C, "with_reasoning", broken)
    for stream in (False, True):
        th, out = chat_async(httpd, {"messages": HI, "stream": stream})
        answer(inbox.next())
        th.join(10)
        assert out["status"] == (200 if stream else 500)
    t = time.time()
    while app.admin.log.written < 2:
        assert time.time() - t < 5
        time.sleep(0.02)
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [x["finish"] for x in lines] == ["server_error", "server_error"] and all("KeyError" in x["error"] for x in lines)
    assert app.admin.finishes == {"server_error": 2}


def test_the_finish_the_client_was_told_is_logged(stand, tmp_path):
    path = tmp_path / "requests.jsonl"
    httpd, app, inbox = stand("--request-log", str(path))
    th, out = chat_async(httpd, {"messages": HI, "stop": ["l"]})  # the answer klm is cut by the stop string
    answer(inbox.next())
    th.join(10)
    assert json.loads(out["body"])["choices"][0]["finish_reason"] == "stop"
    done = dict(DONE, finish_reason="length")
    th, out = chat_async(httpd, {"messages": HI})
    answer(inbox.next(), done=done)
    th.join(10)
    t = time.time()
    while app.admin.log.written < 2:
        assert time.time() - t < 5
        time.sleep(0.02)
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [x["finish"] for x in lines] == ["stop", "length"]
    entry = A.Entry(S.Request([1, 2], 1, 4, None, False), "chatcmpl-x", None, False, None)  # tool calls: what the handler says
    handler = type("H", (), {"_r4": entry, "_r4_status": 200})()
    app.admin.ended(handler, entry.req, ("done", DONE), "tool_calls")
    assert app.admin.finishes["tool_calls"] == 1


def test_a_bad_request_after_a_good_one_on_the_same_connection_is_logged(stand, tmp_path):
    path = tmp_path / "requests.jsonl"
    httpd, app, inbox = stand("--request-log", str(path))
    s = socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=10)
    body = json.dumps({"messages": HI}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
    answer(inbox.next())
    got = b""
    while b"klm" not in got:
        got += s.recv(65536)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n" + b"X-Long: " + b"y" * 70000 + b"\r\n\r\n")  # a header past http.server's 65,536
    while b"431" not in got and b"400" not in got.split(b"klm", 1)[1]:
        piece = s.recv(65536)
        if not piece:
            break
        got += piece
    s.close()
    t = time.time()
    while app.admin.log.written < 2:
        assert time.time() - t < 5, path.read_text(encoding="utf-8")
        time.sleep(0.02)
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["status"] == 200 and lines[1]["finish"] == "refused" and lines[1]["status"] in (400, 431)


def test_label_values_are_escaped():
    assert A._labels({"a": 'x"y\\z\nw'}) == '{a="x\\"y\\\\z\\nw"}'
    parse_metrics('# HELP m x\n# TYPE m gauge\nm{a="x\\"y\\\\z\\nw"} 1\n')


# (d) the request log, (e) the status file and the configuration

def test_every_request_gets_a_line_in_the_log_and_the_log_is_rotated(stand, tmp_path):
    path = tmp_path / "requests.jsonl"
    httpd, app, inbox = stand("--request-log", str(path), "--request-log-mb", "0.0004")  # about 420 bytes: a line or two a file
    for i in range(6):
        th, out = chat_async(httpd, {"messages": HI, "stream": i % 2 == 1})
        answer(inbox.next())
        th.join(10)
    assert call(httpd, "POST", "/v1/chat/completions", raw=b"{not json")[0] == 400
    t = time.time()
    while app.admin.log.written < 7:
        assert time.time() - t < 5
        time.sleep(0.02)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["requests.jsonl", "requests.jsonl.1", "requests.jsonl.2", "requests.jsonl.3"]  # three old ones kept
    lines = [json.loads(x) for p in sorted(tmp_path.iterdir(), reverse=True) for x in p.read_text(encoding="utf-8").splitlines()]
    last = lines[-1]
    # the refusal's line has its status, not its message (ZDR: a message may quote the request; scripts/serve_net.py)
    assert last["status"] == 400 and last["finish"] == "refused" and "error" not in last and "not a chat" not in json.dumps(last)
    served = [x for x in lines if x.get("status") == 200]
    assert served and all(x["id"].startswith("chatcmpl-") and x["finish"] == "stop" and x["ttft_ms"] == 300 and x["reply"] == 3 for x in served)
    assert all(re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d$", x["t"]) for x in lines)


def test_a_client_that_leaves_is_logged_as_such(stand, tmp_path):
    path = tmp_path / "requests.jsonl"
    httpd, app, inbox = stand("--request-log", str(path))
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=20)
    conn.request("POST", "/v1/chat/completions", json.dumps({"messages": HI, "stream": True}), {"Content-Type": "application/json"})
    req = inbox.next()
    conn.close()
    t = time.time()
    while not req.cancelled or app.admin.log.written < 1:
        assert time.time() - t < 10
        time.sleep(0.05)
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[-1])["finish"] == "client_left"


def test_the_status_file_is_rewritten(stand, tmp_path):
    path = tmp_path / "status.json"
    cli_args = ("--status-file", str(path))
    httpd, app, inbox = stand(*cli_args)
    app.admin.status_every = 0.1
    t = time.time()
    while not path.exists():
        assert time.time() - t < 10
        time.sleep(0.05)
    got = json.loads(path.read_text(encoding="utf-8"))
    assert got["pid"] == os.getpid() and "process" in got and got["waiting"] == 0


def test_the_config_file_gives_defaults_the_command_line_overrides(tmp_path, capsys):
    cfg = tmp_path / "serve.toml"
    cfg.write_text('max-waiting = 12\nqueue_timeout = 30\nrequest-log = "logs/x.jsonl"\ndebug = true\n', encoding="utf-8")
    ap = argparse.ArgumentParser()
    A.add_options(ap)
    cli = A.parse_args(ap, ["--config", str(cfg), "--max-waiting", "3"])
    assert (cli.max_waiting, cli.queue_timeout, cli.request_log, cli.debug) == (3, 30.0, "logs/x.jsonl", True)
    assert cli.config == str(cfg) and "options from" in capsys.readouterr().out
    for bad, why in (('max-wait = 3\n', "not an option"), ('debug = "yes"\n', "switch"), ('max-waiting = "x"\n', "max-waiting")):
        cfg.write_text(bad, encoding="utf-8")
        with pytest.raises(SystemExit, match=why):
            A.parse_args(ap, ["--config", str(cfg)])
    fresh = argparse.ArgumentParser()
    A.add_options(fresh)
    assert A.parse_args(fresh, ["--config", ""]).max_waiting == 0  # no file at all: as before


def test_an_abbreviation_never_reads_as_config(tmp_path, capsys):
    """--con was --context before --config came: it must not be taken as a file name; --conf x is refused, not
    half-read (the file's options would be missing)."""
    cfg = tmp_path / "serve.toml"
    cfg.write_text("max-waiting = 12\n", encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=8192)
    A.add_options(ap)
    with pytest.raises(SystemExit):
        A.parse_args(ap, ["--config", "", "--con", "4096"])
    assert "ambiguous" in capsys.readouterr().err
    assert A.parse_args(ap, ["--config", "", "--cont", "4096"]).context == 4096
    with pytest.raises(SystemExit):
        A.parse_args(ap, ["--conf", str(cfg)])
    assert "--config in full" in capsys.readouterr().err
    assert A.parse_args(ap, ["--config", str(cfg)]).max_waiting == 12


# --debug

def test_debug_faults_only_with_debug(stand):
    httpd, app, inbox = stand()
    assert call(httpd, "POST", "/debug/fault", {"kind": "handler"})[0] == 404
    httpd, app, inbox = stand("--debug")
    status, _, body = call(httpd, "POST", "/debug/fault", {"kind": "handler", "count": 1})
    assert status == 200 and json.loads(body)["armed"]["handler"] == 1
    status, _, body = call(httpd, "POST", "/v1/chat/completions", {"messages": HI})
    req = inbox.next()
    assert status == 500 and "InjectedFault" in json.loads(body)["error"]["message"] and req.cancelled
    assert app.admin.finishes == {"server_error": 1}  # an injected fault is the server's failure too
    assert call(httpd, "POST", "/debug/fault", {"kind": "nothing"})[0] == 400
    assert call(httpd, "POST", "/debug/fault", raw=b"")[0] == 400  # the kind must be said: an empty body arms nothing
    assert call(httpd, "POST", "/debug/fault", {})[0] == 400 and app.admin.faults == {"handler": 0, "engine": 0}
    call(httpd, "POST", "/debug/fault", {"kind": "handler", "count": 3})
    call(httpd, "POST", "/debug/fault", {"kind": "engine", "count": 2})
    status, _, body = call(httpd, "POST", "/debug/fault", {"kind": "clear"})  # what the soak sends when its traffic ends
    assert status == 200 and json.loads(body)["armed"] == {"handler": 0, "engine": 0} == app.admin.faults


# the audit and the timed engine methods on a pretend engine (no GPU; tests/test_serve_admin_engine.py runs them on
# the small model)

class FakeEngine:
    """The engine's books as scripts/serve.py reads them: two kinds of tile, a ring of 4 a sequence."""

    def __init__(self, sequences=3, pool=8):
        from sclmd import model as M

        self.kinds = (M.SLIDING, M.FULL)
        self.sequences = sequences
        self.ring = {t: 4 for t in self.kinds}
        self.pool = {t: pool for t in self.kinds}
        self.free = {t: list(range(pool - 1, -1, -1)) for t in self.kinds}
        self.tiles = {t: [[-1] * 4 for _ in range(sequences)] for t in self.kinds}
        self.lengths, self.highs, self._snaps = [0] * sequences, [0] * sequences, {}

    def take(self, seq: int, n: int) -> None:
        """The sequence holds n positions (up to 64): tiles for them, from the free lists."""
        for t in self.kinds:
            for j in range(-(-n // 16)):
                self.tiles[t][seq][j] = self.free[t].pop()
        self.lengths[seq] = self.highs[seq] = n


def fake_server(eng):
    s = type("Server", (), {})()
    s.eng, s.kept, s.parked, s.active, s.waiting, s.ram = eng, {}, [], [], [], 0
    s.unused = list(range(eng.sequences - 1, -1, -1))
    s.prefixes = S.Prefixes(0)
    return s


def test_the_audit_passes_good_books_and_names_each_kind_of_bad_one():
    eng = FakeEngine()
    s = fake_server(eng)
    assert A.audit(s)["ok"]
    eng.take(2, 40)  # a kept conversation: 40 positions, read ahead to its later form of 40
    s.unused.remove(2)
    s.kept[2] = S.Kept(2, list(range(30)), False)
    s.kept[2].later = list(range(40))
    s.parked.append(S.Parked(list(range(20)), {"length": 20, "bytes": 100, "kinds": {}}, 0.0))
    good = A.audit(s)
    assert good["ok"], good["problems"]
    assert good["counts"][f"tiles_{eng.kinds[0]}"]["held"] == 3  # 40 positions: 3 tiles of 16
    bad, free = [], eng.free[eng.kinds[1]]
    free.append(eng.tiles[eng.kinds[1]][2][0])  # held and free
    bad.append(("twice", A.audit(s)))
    free.pop()
    lost = free.pop()  # a tile lost: neither held nor free
    bad.append(("!= the pool", A.audit(s)))
    free.append(lost)
    assert A.audit(s)["ok"]
    eng.lengths[2] = 35  # holds less than its later form though its reading ahead is over
    bad.append(("reading ahead over", A.audit(s)))
    eng.lengths[2] = 40
    s.unused.append(2)  # kept and unused at once
    bad.append(("both unused and kept", A.audit(s)))
    s.unused.remove(2)
    eng.lengths[0] = 5  # an unused sequence that holds positions
    bad.append(("unused sequence 0", A.audit(s)))
    eng.lengths[0] = 0
    eng._snaps[1] = {"data": []}  # a snapshot of an unused sequence
    bad.append(("snapshot", A.audit(s)))
    del eng._snaps[1]
    s.parked[0].state["length"] = 19  # a parked conversation whose state is not its tokens
    bad.append(("parked", A.audit(s)))
    s.parked[0].state["length"] = 20
    s.unused.remove(1)  # a sequence that is in no list
    bad.append(("neither unused, kept nor writing", A.audit(s)))
    for want, result in bad:
        assert not result["ok"] and any(want in p for p in result["problems"]), (want, result["problems"])


def test_the_audit_waits_for_idle():
    eng = FakeEngine()
    s = fake_server(eng)
    s.inbox = __import__("queue").Queue()
    s.batch = type("B", (), {"jobs": [], "passes": 0})()
    assert A.audit_at_idle(s)["idle"] is True
    s.batch.jobs = [type("J", (), {"done": None})()]  # a reading ahead under way
    assert A.audit_at_idle(s)["idle"] is False
    s.batch.jobs = []
    s.inbox.put(object())
    assert A.audit_at_idle(s)["idle"] is False


def test_the_timed_engine_methods_return_what_they_did_and_count_by_width():
    calls = []

    class Batch:
        widths = {8: 0, 16: 0, 32: 0}

        def step(self):
            calls.append("step")
            self.widths[16 if len(calls) % 2 else 32] += 1
            return len(calls)

    class Eng:
        def park(self, seq):
            return {"parked": seq}

    s = type("Server", (), {})()
    s.batch, s.eng = Batch(), Eng()
    adm = A.Admin(None, s)
    adm.time_engine()
    assert [s.batch.step() for _ in range(3)] == [1, 2, 3] and s.eng.park(5) == {"parked": 5}
    assert adm.pass_n == {16: 2, 32: 1} and adm.copy_n == {"park": 1}
    adm.faults["engine"] = 1
    with pytest.raises(A.InjectedFault):
        s.batch.step()
    assert s.batch.step() == 4 and adm.injected["engine"] == 1  # once


# the openai client and the 429 (when the package is there: it is not in the project's environment)

def test_the_openai_client_retries_a_429_after_retry_after(stand):
    openai = pytest.importorskip("openai")
    httpd, app, inbox = stand(max_waiting=1)
    th, out = chat_async(httpd, {"messages": HI})
    first = inbox.next()
    client = openai.OpenAI(base_url=f"http://127.0.0.1:{httpd.server_address[1]}/v1", api_key="none", max_retries=2, timeout=30)
    got = {}

    def ask():
        t = time.perf_counter()
        got["reply"] = client.chat.completions.create(model="x", messages=HI)
        got["seconds"] = time.perf_counter() - t

    asker = threading.Thread(target=ask)
    asker.start()
    time.sleep(0.5)
    answer(first)  # the queue drains: the client's retry after Retry-After (1 s) is served
    second = inbox.next()
    answer(second)
    asker.join(30)
    th.join(10)
    assert got["reply"].choices[0].message.content == "klm" and got["seconds"] >= 0.9
