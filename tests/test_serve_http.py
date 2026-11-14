# The server's HTTP side (scripts/serve.py: Handler) without a model or a GPU: the engine's thread is this test, which
# puts a request's events itself. What a client sees when the reply ends well, when the server fails in the middle of
# it, and what the server sees when the client leaves.
import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from scripts.serve import Handler, Request

STOP = (7,)
OPENING, CLOSING = 1, 2


class Letters:
    """A tokenizer whose token n is a letter."""

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(97 + i % 26) for i in ids)


class Inbox:
    """In the engine's place: the requests that came, for the test to answer."""

    stop = STOP

    def __init__(self):
        self.reqs = []

    def submit(self, req: Request) -> None:
        self.reqs.append(req)

    def next(self) -> Request:
        t = time.time()
        while not self.reqs:
            assert time.time() - t < 10, "the request did not come"
            time.sleep(0.01)
        return self.reqs.pop(0)


class App:
    tok, opening, closing, fingerprint = Letters(), OPENING, CLOSING, "sclmd-test"

    def __init__(self):
        self.server = Inbox()

    def request(self, body: dict) -> Request:
        return Request([0] * 4, 3, 8, None, False, body["messages"])


@pytest.fixture()
def served():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    httpd.app = App()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def post(httpd, stream: bool) -> http.client.HTTPConnection:
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=20)
    conn.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": stream}),
                 {"Content-Type": "application/json"})
    return conn


def events_of(body: bytes) -> list:
    """A stream's events: the parsed JSON of every data line, "[DONE]" as it is."""
    out = []
    for line in body.decode().split("\n\n"):
        if line.startswith("data: "):
            out.append(line[6:] if line[6:] == "[DONE]" else json.loads(line[6:]))
    return out


DONE = {"finish_reason": "stop", "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}, "timings": {"total_ms": 5}}


def test_a_reply_that_ends_well(served):
    for stream in (False, True):
        conn = post(served, stream)
        req = served.app.server.next()
        for token in (10, 11, 12):
            req.events.put(("token", token))
        req.events.put(("done", DONE))
        resp = conn.getresponse()
        body = resp.read()
        assert resp.status == 200
        if not stream:
            got = json.loads(body)
            assert got["choices"][0]["message"]["content"] == "klm" and got["choices"][0]["finish_reason"] == "stop"
            assert got["usage"] == {**DONE["usage"], "completion_tokens_details": {"reasoning_tokens": 0}}
        else:
            events = events_of(body)
            assert events[-1] == "[DONE]"
            assert "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1]) == "klm"
            assert events[-2]["choices"][0]["finish_reason"] == "stop"


def test_a_reply_without_a_stream_is_an_error_when_the_server_fails_in_the_middle(served):
    conn = post(served, False)
    req = served.app.server.next()
    req.events.put(("token", 10))
    time.sleep(0.2)  # the token is taken before the error comes
    req.events.put(("error", 500, "the server failed while answering: RuntimeError"))
    resp = conn.getresponse()
    got = json.loads(resp.read())
    assert resp.status == 500
    assert got["error"]["type"] == "server_error" and "RuntimeError" in got["error"]["message"]
    assert "choices" not in got  # not a reply cut short that looks whole


@pytest.mark.parametrize("together", [False, True])
def test_a_stream_ends_with_an_error_event_when_the_server_fails_in_the_middle(served, together):
    conn = post(served, True)
    req = served.app.server.next()
    req.events.put(("token", 10))
    req.events.put(("token", 11))
    if not together:  # the error comes alone, after the tokens have gone out (else: with them, in one read of the queue)
        time.sleep(0.2)
    req.events.put(("error", 500, "the server failed while answering: RuntimeError"))
    resp = conn.getresponse()
    events = events_of(resp.read())  # the read ends: the server closes the connection
    assert resp.status == 200  # the headers had gone out before the failure
    assert events[-1] == "[DONE]" and events[-2] == {"error": {"message": "the server failed while answering: RuntimeError", "type": "server_error"}}
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-2]) == "kl"  # what was written came
    assert all(e["choices"][0]["finish_reason"] is None for e in events[:-2])  # and no chunk says the reply ended


def test_a_request_that_fails_before_its_first_token(served):
    for stream in (False, True):
        conn = post(served, stream)
        served.app.server.next().events.put(("error", 500, "the server failed while answering: RuntimeError"))
        resp = conn.getresponse()
        body = resp.read()
        if not stream:
            assert resp.status == 500 and json.loads(body)["error"]["type"] == "server_error"
        else:
            events = events_of(body)
            assert events[-1] == "[DONE]" and events[-2]["error"]["type"] == "server_error"


def test_a_client_that_leaves_is_noticed(served):
    for tokens_first in (False, True):  # while it waits for the first token, and in the middle of a stream
        conn = post(served, True)
        req = served.app.server.next()
        resp = None
        if tokens_first:
            req.events.put(("token", 10))
            resp = conn.getresponse()
            assert resp.read1(4096).startswith(b"data: ")
        if resp is not None:
            resp.close()
        conn.close()
        t = time.time()
        while not req.cancelled:
            if tokens_first:
                req.events.put(("token", 11))  # the engine writes on until it is told
            assert time.time() - t < 10, "the server did not notice"
            time.sleep(0.05)
