# The server's contract with its clients (scripts/serve_contract.py and its hooks in scripts/serve.py) without a model
# or a GPU: the real App and Handler over a tokenizer of letters, and the test in the engine's place.
import http.client
import json
import socket
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from scripts import serve as S
from scripts import serve_contract as C

OPENING, CLOSING, STOP = 1, 2, (7,)
TURN = [3, 4]


class Tok:
    """A chat template over letters: token n is the letter chr(97 + n % 26); a message is 5, its letters, 6; the model's
    turn begins with 3, 4."""

    def encode(self, s, add_special_tokens=False):
        return list(TURN) if s == "<|turn>model\n" else [ord(c) - 97 for c in s]

    def convert_tokens_to_ids(self, s):
        return {"<|channel>": OPENING, "<channel|>": CLOSING}[s]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(97 + i % 26) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, enable_thinking=False):
        ids = [0]
        for m in messages:
            ids += [5] + [(ord(c) - 97) % 26 for c in m["content"]] + [6]
        return ids + (list(TURN) if add_generation_prompt else [])


class Inbox:
    """In the engine's place: the requests that came, for the test to answer."""

    stop = STOP

    def __init__(self):
        self.reqs, self.canon = [], None
        self.prefixes = type("Prefixes", (), {"ram": 0, "held": {}})()  # verify: main's App.request reads server.prefixes.ram (B4)
        self.eng = type("Eng", (), {"capacity": 4096, "rows": 32})()

    def submit(self, req) -> None:
        self.reqs.append(req)

    def next(self):
        t = time.time()
        while not self.reqs:
            assert time.time() - t < 10, "the request did not come"
            time.sleep(0.01)
        return self.reqs.pop(0)


DONE = {"finish_reason": "stop", "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}, "timings": {"total_ms": 5}}


@pytest.fixture()
def served():
    httpd = C.BoundedHTTPServer(("127.0.0.1", 0), S.Handler)
    httpd.app = S.App(Inbox(), Tok(), False, 64, {"bits_per_weight": 3.0})
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def send(httpd, body: dict, raw: bytes | None = None) -> http.client.HTTPConnection:
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=20)
    conn.request("POST", "/v1/chat/completions", raw if raw is not None else json.dumps(body), {"Content-Type": "application/json"})
    return conn


def answer(httpd, body: dict, tokens=(10, 11, 12), done=DONE, served=True) -> tuple:
    """Sends a request and, when it is to reach the engine (served), answers it with `tokens` -> (status, headers, the
    parsed body or a stream's events, the request or None)."""
    conn = send(httpd, body)
    req = None
    if served:
        req = httpd.app.server.next()
        for token in tokens:
            req.events.put(("token", token))
        req.events.put(("done", done))
    resp = conn.getresponse()
    data = resp.read()
    got = json.loads(data) if not body.get("stream") or resp.status != 200 else events_of(data)
    return resp.status, dict(resp.getheaders()), got, req


def events_of(body: bytes) -> list:
    out = []
    for line in body.decode().split("\n\n"):
        if line.startswith("data: "):
            out.append(line[6:] if line[6:] == "[DONE]" else json.loads(line[6:]))
    return out


HI = [{"role": "user", "content": "hi"}]
NEUTRAL = {"n": 1, "logprobs": False, "top_logprobs": 0, "logit_bias": {}, "presence_penalty": 0, "frequency_penalty": 0.0,
           "response_format": {"type": "text"}, "tools": [], "tool_choice": "auto", "stream_options": None, "modalities": ["text"]}
# values of those fields that ask for nothing the server does not do: as before R1, served without a word
QUIET = [("n", "1"), ("n", 1.0), ("n", True), ("logprobs", "false"), ("logprobs", 0), ("logprobs", None), ("top_logprobs", None),
         ("top_logprobs", "0"), ("modalities", "text"), ("modalities", []), ("response_format", "text"), ("response_format", None),
         ("presence_penalty", "0"), ("repetition_penalty", 1), ("stop", ""), ("stop", []), ("stop", [""]), ("stop", None)]


# (2) fields that are not served

def test_neutral_values_of_fields_that_are_not_served_pass_in_every_mode():
    for mode in C.MODES:
        assert C.check_fields({"messages": HI, **NEUTRAL}, mode) == []
        for name, value in QUIET:
            assert C.check_fields({"messages": HI, name: value}, mode) == [] and C.stops_of({name: value}, mode)[1] is False, (name, value)


@pytest.mark.parametrize("name, value", QUIET)
def test_a_neutral_value_is_served_without_a_header(served, name, value):
    status, headers, got, req = answer(served, {"messages": HI, name: value})
    assert status == 200 and "x-sclmd-ignored" not in headers and req.ignored == []


@pytest.mark.parametrize("name, value", [("n", 2), ("n", 0), ("n", "2"), ("logprobs", True), ("logprobs", "true"), ("top_logprobs", 3),
                                         ("response_format", {"type": "json_schema", "json_schema": {"name": "x", "schema": {}, "strict": True}})])
def test_a_field_whose_answer_would_have_another_shape_is_refused(served, name, value):
    status, headers, got, req = answer(served, {"messages": HI, name: value}, served=False)
    assert status == 400 and not served.app.server.reqs  # nothing reached the engine
    assert got["error"]["param"] == name and got["error"]["message"].startswith(name) and got["error"]["type"] == "invalid_request_error"
    served.app.fields = "lenient"  # --lenient-fields: accepted and named, as before (but named)
    status, headers, got, req = answer(served, {"messages": HI, name: value})
    assert status == 200 and headers["x-sclmd-ignored"] == name and got["choices"][0]["message"]["content"] == "klm"


@pytest.mark.parametrize("name, value", [("presence_penalty", 1.0), ("frequency_penalty", -0.5), ("logit_bias", {"5": -100}),
                                         ("tool_choice", "required"),  # tools themselves are served (R2: tests/test_tool_calls.py)
                                         ("response_format", {"type": "json_object"}), ("response_format", "json_object"),
                                         ("response_format", {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}),
                                         ("modalities", ["text", "audio"]), ("modalities", "audio"), ("audio", {"voice": "x", "format": "wav"}),
                                         ("n", "two"), ("logprobs", {}), ("top_logprobs", "many")])
def test_a_field_that_only_steers_the_sampling_is_accepted_and_named(served, name, value):
    for stream in (False, True):
        status, headers, got, req = answer(served, {"messages": HI, name: value, "stream": stream})
        assert status == 200 and headers["x-sclmd-ignored"] == name and req.ignored == [name]
    served.app.fields = "strict"  # --strict-fields: refused
    status, headers, got, req = answer(served, {"messages": HI, name: value}, served=False)
    assert status == 400 and got["error"]["param"] == name and "--strict-fields" in got["error"]["message"]


def test_a_thinking_budget_is_named_as_not_served(served):
    status, headers, got, req = answer(served, {"messages": HI, "thinking": {"type": "enabled", "budget_tokens": 2048}})
    assert status == 200 and headers["x-sclmd-ignored"] == "thinking.budget_tokens" and req.think  # thinking is on, as before
    assert C.check_fields({"thinking": {"type": "enabled"}}) == [] and C.check_fields({"thinking": True}) == []


def test_a_request_without_such_fields_has_no_header(served):
    status, headers, got, req = answer(served, {"messages": HI, **NEUTRAL})
    assert status == 200 and "x-sclmd-ignored" not in headers and req.ignored == []


# (3) stream_options that is not an object; a stop that is not text

def test_stream_options_true_ends_the_stream_as_usual(served):
    status, headers, events, req = answer(served, {"messages": HI, "stream": True, "stream_options": True})
    assert status == 200 and events[-1] == "[DONE]" and events[-2]["choices"][0]["finish_reason"] == "stop"
    assert headers["x-sclmd-ignored"] == "stream_options"
    status, headers, events, req = answer(served, {"messages": HI, "stream": True, "stream_options": {"include_usage": True}})
    assert events[-1] == "[DONE]" and events[-2]["usage"]["completion_tokens"] == 3 and events[-2]["choices"] == []


@pytest.mark.parametrize("stop, strings", [(False, []), (0, []), (5, []), (True, []), ([5], []), ({"a": 1}, ["a"]), (["x", None], ["x"]),
                                           (["x", 3, "", "y"], ["x", "y"])])
def test_a_stop_that_is_not_text_is_served_as_its_text_and_named(served, stop, strings):
    # before R1: false and 0 were no stop, a list's strings and an object's keys were taken (200); 5 and true dropped
    # the connection
    for stream in (False, True):
        status, headers, got, req = answer(served, {"messages": HI, "stop": stop, "stream": stream})
        assert status == 200 and headers["x-sclmd-ignored"] == "stop" and req.stops == strings and req.ignored == ["stop"]
    status, headers, got, req = answer(served, {"messages": HI, "stop": stop, "presence_penalty": 1}, served=True)
    assert headers["x-sclmd-ignored"] == "presence_penalty, stop"
    served.app.fields = "strict"  # --strict-fields: refused
    status, headers, got, req = answer(served, {"messages": HI, "stop": stop}, served=False)
    assert status == 400 and got["error"]["param"] == "stop" and not served.app.server.reqs


# (1) a stop string: the answer as cut is what the engine keeps the conversation with

@pytest.mark.parametrize("stream", [False, True])
def test_a_stop_string_cuts_the_answer_and_tells_the_engine_what_was_cut(served, stream):
    conn = send(served, {"messages": HI, "stream": stream, "stop": ["mn"]})
    req = served.app.server.next()
    for token in (10, 11, 12, 13, 14):  # k l m n o: the stop string "mn" is in it
        req.events.put(("token", token))
    t = time.time()
    while not req.cancelled:
        assert time.time() - t < 10
        time.sleep(0.01)
    assert req.stop_text == "kl" and req.stops == ["mn"]
    req.events.put(("token", 15))  # the engine writes a little more before it sees it
    req.events.put(("done", {**DONE, "finish_reason": "stop"}))
    resp = conn.getresponse()
    data = resp.read()
    if stream:
        events = events_of(data)
        assert "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1]) == "kl" and events[-2]["choices"][0]["finish_reason"] == "stop"
    else:
        got = json.loads(data)
        assert got["choices"][0]["message"]["content"] == "kl" and got["choices"][0]["finish_reason"] == "stop"


def test_the_later_form_of_a_reply_is_made_from_the_answer_as_cut():
    app = S.App(Inbox(), Tok(), False, 64, {"bits_per_weight": 3.0})
    req = app.request({"messages": HI, "stop": "mn"})
    req.stops = ["mn"]
    whole = app.canonical(req, [10, 11, 12, 13, 14, 7])  # the reply ended by itself before the HTTP thread cut it
    cut = Tok().apply_chat_template(HI + [{"role": "assistant", "content": "kl"}])
    assert whole == cut
    req.stop_text = "k"  # what the HTTP thread cut, when it did
    assert app.canonical(req, [10, 11, 12, 13, 14, 7]) == Tok().apply_chat_template(HI + [{"role": "assistant", "content": "k"}])
    req = app.request({"messages": HI})  # no stop strings: the whole answer, as before
    assert app.canonical(req, [10, 11, 12, 13, 14, 7]) == Tok().apply_chat_template(HI + [{"role": "assistant", "content": "klmno"}])


# (4) a fault in the HTTP thread after the request went to the engine

@pytest.mark.parametrize("stream", [False, True])
def test_a_fault_while_sending_stops_the_reply_and_tells_the_client(served, stream, monkeypatch):
    class Failing(S.Pieces):
        def add(self, token):
            if token == 12:
                raise RuntimeError("injected")
            return super().add(token)

    monkeypatch.setattr(S, "Pieces", Failing)
    conn = send(served, {"messages": HI, "stream": stream})
    req = served.app.server.next()
    for token in (10, 11, 12, 13):
        req.events.put(("token", token))
    resp = conn.getresponse()
    data = resp.read()
    if stream:
        events = events_of(data)
        assert events[-1] == "[DONE]" and events[-2]["error"]["type"] == "server_error"
    else:
        assert resp.status == 500 and "RuntimeError" in json.loads(data)["error"]["message"]
    assert req.cancelled  # the engine stops writing it


def test_a_client_that_leaves_a_reply_that_is_not_streamed_is_noticed_while_tokens_come(served):
    served.app.look_s = 1.0  # --look-gone 1 (off by default: see the next test)
    conn = send(served, {"messages": HI})
    req = served.app.server.next()
    req.events.put(("token", 10))
    time.sleep(0.2)
    conn.close()
    t = time.time()
    while not req.cancelled:  # tokens keep coming every 10 ms: the queue is never empty for a second
        req.events.put(("token", 11))
        assert time.time() - t < 10, "the server did not notice"
        time.sleep(0.01)


def test_a_client_that_closes_its_sending_side_gets_a_reply_that_is_not_streamed_by_default(served):
    """verify:R1-land: a client that sends its request, shuts its sending side (shutdown) and reads got 200 from
    master for a reply of several seconds; the look while tokens come (--look-gone) takes it as gone, so it is off."""
    body = json.dumps({"messages": HI}).encode()
    s = socket.create_connection(("127.0.0.1", served.server_address[1]))
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
              + str(len(body)).encode() + b"\r\n\r\n" + body)
    s.shutdown(socket.SHUT_WR)
    req = served.app.server.next()
    t = time.time()
    while time.time() - t < 2.5:  # tokens every 10 ms for 2.5 s: the queue is never empty for a second
        req.events.put(("token", 11))
        time.sleep(0.01)
    assert not req.cancelled
    req.events.put(("done", DONE))
    s.settimeout(10)
    got = b""
    while True:
        d = s.recv(65536)
        if not d:
            break
        got += d
    s.close()
    assert got.startswith(b"HTTP/1.1 200")


# (5) connections

def test_an_idle_connection_is_closed_after_the_idle_timeout(served, monkeypatch):
    monkeypatch.setattr(S.Handler, "timeout", 2.5)  # room for 20 connects (a listen backlog of 5) on a busy machine
    t = time.time()
    socks = [socket.create_connection(("127.0.0.1", served.server_address[1])) for _ in range(20)]
    while served.live < 20:
        assert time.time() - t < 2.3, f"{served.live} of 20 connections have a thread"
        time.sleep(0.01)
    while served.live:
        assert time.time() - t < 15, "idle connections still hold their threads"
        time.sleep(0.05)
    assert time.time() - t > 2.4  # not before the timeout
    for s in socks:
        s.settimeout(2)
        assert s.recv(10) == b""  # closed by the server
        s.close()


@pytest.mark.parametrize("stream", [False, True])
def test_the_idle_timeout_does_not_cut_a_reply_that_is_slow_to_come(served, monkeypatch, stream):
    monkeypatch.setattr(S.Handler, "timeout", 0.5)  # a first token after 1.5 s, and 1 s between the next two
    conn = send(served, {"messages": HI, "stream": stream})
    req = served.app.server.next()
    time.sleep(1.5)
    for token in (10, 11):
        req.events.put(("token", token))
        time.sleep(1.0)
    req.events.put(("done", DONE))
    resp = conn.getresponse()
    data = resp.read()
    assert resp.status == 200 and not req.cancelled
    if stream:
        events = events_of(data)
        assert events[-1] == "[DONE]" and "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1]) == "kl"
    else:
        assert json.loads(data)["choices"][0]["message"]["content"] == "kl"


def test_connections_past_the_bound_are_answered_503():
    httpd = C.BoundedHTTPServer(("127.0.0.1", 0), S.Handler, 3)
    httpd.app = S.App(Inbox(), Tok(), False, 64, {"bits_per_weight": 3.0})
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        held = [socket.create_connection(("127.0.0.1", httpd.server_address[1])) for _ in range(3)]
        time.sleep(0.2)
        extra = socket.create_connection(("127.0.0.1", httpd.server_address[1]))
        extra.settimeout(5)
        got = b""
        while chunk := extra.recv(4096):
            got += chunk
        assert got.startswith(b"HTTP/1.1 503") and b"--max-connections" in got and httpd.turned_away == 1
        extra.close()
        # clients that send their whole request at once and only then read (http.client) get the 503, not a reset
        big = json.dumps({"messages": [{"role": "user", "content": "x" * 300_000}]})
        for k in range(20):
            conn = send(httpd, {}, raw=big if k % 2 else json.dumps({"messages": HI}))
            resp = conn.getresponse()
            assert resp.status == 503 and resp.getheader("Retry-After") == "1" and b"--max-connections" in resp.read()
            conn.close()
        held[0].close()  # a slot is free again: served as usual
        time.sleep(0.3)
        status, _, got, _ = answer(httpd, {"messages": HI})
        assert status == 200 and got["choices"][0]["message"]["content"] == "klm"
        for s in held[1:]:
            s.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


# (6) the body

@pytest.mark.parametrize("length, status", [("2000000000", 413), ("-5", 400), ("lots", 400), (None, 411)])
def test_a_body_of_the_wrong_length_is_refused_without_being_read(served, length, status):
    served.app.max_body = 32 << 20  # --max-body-mb 32 (off by default)
    s = socket.create_connection(("127.0.0.1", served.server_address[1]))
    s.settimeout(5)
    head = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
    s.sendall(head + (f"Content-Length: {length}\r\n".encode() if length is not None else b"") + b"\r\n{\"messages\":")
    got = b""
    t = time.time()
    while chunk := s.recv(4096):
        got += chunk
    assert time.time() - t < 3
    assert got.startswith(f"HTTP/1.1 {status}".encode()) and b"Connection: close" in got
    s.close()


def test_by_default_a_large_body_is_served_as_before_and_read_in_pieces(served, monkeypatch):
    # no cap by default: a body of 3 MB whose bulk is in a field the server ignores is served, as before R1
    big = json.dumps({"messages": HI, "metadata": {"notes": "x" * (3 << 20)}})
    monkeypatch.setattr(C, "PIECE", 1 << 16)  # 48 pieces
    conn = send(served, {}, raw=big)
    req = served.app.server.next()
    req.events.put(("done", DONE))
    resp = conn.getresponse()
    assert resp.status == 200 and json.loads(resp.read())["choices"][0]["message"]["content"] == "" and req.messages == HI
    # a length announced and not sent: the server waits for it without taking memory for it, and the client may leave
    s = socket.create_connection(("127.0.0.1", served.server_address[1]))
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2000000000\r\n\r\n{\"messages\":")
    time.sleep(0.3)
    s.close()
    status, _, got, _ = answer(served, {"messages": HI})  # the server serves on
    assert status == 200


def test_a_body_over_the_cap_reaches_a_client_that_sends_it_all_before_reading(served):
    served.app.max_body = 1 << 20  # 1 MB; a body of 3 MB from http.client, which reads only when it has sent it all
    big = json.dumps({"messages": [{"role": "user", "content": "x" * (3 << 20)}]})
    for _ in range(10):
        conn = send(served, {}, raw=big)
        resp = conn.getresponse()
        got = json.loads(resp.read())
        assert resp.status == 413 and "--max-body-mb" in got["error"]["message"] and resp.getheader("Connection") == "close"
        conn.close()
    assert not served.app.server.reqs
    status, _, got, _ = answer(served, {"messages": HI})  # the server serves on
    assert status == 200


def test_a_client_that_waits_for_100_continue_hears_413_without_sending(served):
    served.app.max_body = 32 << 20  # --max-body-mb 32
    s = socket.create_connection(("127.0.0.1", served.server_address[1]))
    s.settimeout(5)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2000000000\r\nExpect: 100-continue\r\n\r\n")
    got = b""
    while chunk := s.recv(4096):
        got += chunk
    s.close()
    assert got.startswith(b"HTTP/1.1 413") and b"100 Continue" not in got
    s = socket.create_connection(("127.0.0.1", served.server_address[1]))  # a body that will be read: told to send it
    s.settimeout(5)
    one = json.dumps({"messages": HI}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\nExpect: 100-continue\r\n\r\n" % len(one))
    assert s.recv(4096).startswith(b"HTTP/1.1 100 Continue")
    s.sendall(one)
    req = served.app.server.next()
    req.events.put(("done", DONE))
    got = b""
    while b"}}" not in got and (chunk := s.recv(4096)):
        got += chunk
    assert got.startswith(b"HTTP/1.1 200")
    s.close()


@pytest.mark.parametrize("raw", [b'{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1e999}',
                                 b'{"messages": [{"role": "user", "content": "hi"}], "seed": 1e999}',
                                 b'{"messages": ' + b"[" * 100_000 + b"]" * 100_000 + b"}"])
def test_numbers_too_large_and_bodies_nested_too_deep_get_400(served, raw):
    conn =http.client.HTTPConnection("127.0.0.1", served.server_address[1], timeout=20)
    conn.request("POST", "/v1/chat/completions", raw, {"Content-Type": "application/json"})
    resp = conn.getresponse()  # before: the connection dropped with no answer
    assert resp.status == 400 and json.loads(resp.read())["error"]["type"] == "invalid_request_error"


def test_the_body_cap_is_an_option(served):
    served.app.max_body = 100
    status, _, got, _ = answer(served, {"messages": [{"role": "user", "content": "x" * 200}]}, served=False)
    assert status == 413 and "--max-body-mb" in got["error"]["message"]
    served.app.max_body = 0  # no cap
    status, _, got, _ = answer(served, {"messages": [{"role": "user", "content": "x" * 200}]})
    assert status == 200


# (7) reasoning tokens

def test_usage_counts_the_reasoning_tokens(served):
    for stream in (False, True):
        status, _, got, _ = answer(served, {"messages": HI, "stream": stream, "stream_options": {"include_usage": True}},
                                   tokens=(OPENING, 20, 21, 22, CLOSING, 10, 11))
        usage = got["usage"] if not stream else got[-2]["usage"]
        assert usage["completion_tokens_details"]["reasoning_tokens"] == 5
        if not stream:
            assert got["choices"][0]["message"]["content"] == "kl"


def test_the_options_and_their_defaults():
    import argparse

    ap = argparse.ArgumentParser()
    C.add_options(ap)
    cli = ap.parse_args([])
    app = S.App(Inbox(), Tok(), False, 64, {"bits_per_weight": 3.0})

    class H:
        timeout = None

    C.apply(cli, app, H)
    assert (app.fields, app.max_body, H.timeout, cli.max_connections) == ("default", 0, None, 0)  # as before R1
    C.apply(ap.parse_args(["--lenient-fields", "--idle-timeout", "120", "--max-body-mb", "32"]), app, H)
    assert (app.fields, app.max_body, H.timeout) == ("lenient", 32 << 20, 120.0)
    with pytest.raises(SystemExit):
        C.apply(ap.parse_args(["--lenient-fields", "--strict-fields"]), app, H)
