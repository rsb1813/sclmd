# The internet server's base (scripts/serve_net.py and its hooks; N1) without a model or a GPU: who a request is,
# the seams on the request object, the limits, the pings, /metrics from this PC only, and ZDR (what the logs keep).
import argparse
import email.message
import http.client
import http.server
import io
import json
import os
import socket
import threading
import time

import pytest

from scripts import serve as S
from scripts import serve_admin as A
from scripts import serve_contract as C
from scripts import serve_net as N
from scripts import supervise
from test_serve_admin import HI, Inbox, answer, call, chat_async, options
from test_api_contract import Tok

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KEYS = [("alice", b"sk-alice-0123456789"), ("bob", b"sk-bob-0123456789ab")]
PROXY = N.networks("192.168.0.10")


def headers(*pairs) -> email.message.Message:
    h = email.message.Message()
    for k, v in pairs:
        h[k] = v
    return h


def who(peer, *pairs, keys=KEYS, anonymous=True, proxies=PROXY):
    return N.who(peer, headers(*pairs), keys, anonymous, proxies, A.key_name)


# who a request is

def test_a_key_is_the_user_and_an_unknown_key_is_refused_never_the_address():
    assert who("203.0.113.7", ("Authorization", "Bearer sk-alice-0123456789")).tenant == "key:alice"
    assert who("203.0.113.7", ("x-api-key", "sk-bob-0123456789ab")).tenant == "key:bob"
    w = who("203.0.113.7", ("Authorization", "Bearer sk-not-ours"))
    assert w.refusal == "unknown_key" and w.tenant is None
    assert who("127.0.0.1", ("Authorization", "Bearer sk-not-ours")).refusal == "unknown_key"  # this PC too
    assert who("203.0.113.7", ("Authorization", "Bearer none"), keys=[]).refusal == "unknown_key"  # --anonymous without keys
    assert who("203.0.113.7", ("Authorization", "")).tenant == "ip:203.0.113.7"  # an empty header is no key (the openai library's way)


def test_no_key_is_the_address_ipv4_whole_ipv6_its_64():
    assert who("203.0.113.7").tenant == "ip:203.0.113.7"
    assert who("2001:db8:1:2:aaaa:bbbb:cccc:dddd").tenant == "ip:2001:db8:1:2::/64"
    assert who("2001:db8:1:2::1").tenant == who("2001:db8:1:2:ffff::9").tenant != who("2001:db8:1:3::1").tenant
    assert who("::ffff:203.0.113.7").tenant == "ip:203.0.113.7"


def test_this_pc_is_loopback_through_no_proxy():
    for peer in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
        w = who(peer)
        assert w.tenant == N.LOCAL and w.local
    # 127.0.0.1 is no trusted proxy here: the header is not read, and a loopback connection that carries it came through
    # a proxy or tunnel on this PC (cloudflared, ngrok): its loopback address, never this PC
    for pair in (("X-Forwarded-For", "198.51.100.1"), ("Forwarded", "for=198.51.100.1"), ("X-Forwarded-For", "")):
        w = who("127.0.0.1", pair)
        assert (w.tenant, w.local, w.proxied) == (("ip:127.0.0.1", False, False) if pair[1] else (N.LOCAL, True, False)), pair
    assert not who("::1", ("X-Forwarded-For", "198.51.100.1")).local
    w = who("127.0.0.1", ("X-Forwarded-For", "198.51.100.1"), proxies=N.networks("127.0.0.1"))  # through a proxy on this PC
    assert w.tenant == "ip:198.51.100.1" and not w.local and w.proxied
    assert who("127.0.0.1", proxies=N.networks("127.0.0.1")).local  # that proxy's own request, with no header: this PC


def test_x_forwarded_for_only_from_a_trusted_proxy_from_the_right():
    assert who("203.0.113.7", ("X-Forwarded-For", "198.51.100.1")).tenant == "ip:203.0.113.7"  # not a proxy: not read
    assert who("192.168.0.10", ("X-Forwarded-For", "198.51.100.1")).tenant == "ip:198.51.100.1"
    # a client's own header comes first; the proxy appends what it saw: the rightmost untrusted address is the user
    assert who("192.168.0.10", ("X-Forwarded-For", "127.0.0.1, 10.0.0.1, 198.51.100.1")).tenant == "ip:198.51.100.1"
    assert who("192.168.0.10", ("X-Forwarded-For", "198.51.100.1, 192.168.0.10")).tenant == "ip:198.51.100.1"  # trusted hops skipped
    # every header, in order (headers.get would give the first only)
    assert who("192.168.0.10", ("X-Forwarded-For", "127.0.0.1"), ("X-Forwarded-For", "198.51.100.2")).tenant == "ip:198.51.100.2"
    assert who("192.168.0.10", ("X-Forwarded-For", "[2001:db8:1:2::5]:443")).tenant == "ip:2001:db8:1:2::/64"
    assert who("192.168.0.10", ("X-Forwarded-For", "198.51.100.3:5000")).tenant == "ip:198.51.100.3"
    # an X-Forwarded-For address is never this PC, whatever it says
    w = who("192.168.0.10", ("X-Forwarded-For", "127.0.0.1"))
    assert w.tenant == "ip:127.0.0.1" and not w.local
    # what a trusted proxy wrote wrongly: its own address, not one it did not see
    assert who("192.168.0.10", ("X-Forwarded-For", "198.51.100.1, garbage")).tenant == "ip:192.168.0.10"
    assert who("192.168.0.10", ("X-Forwarded-For", "")).tenant == "ip:192.168.0.10"  # empty: the connection's
    assert who("192.168.0.10", ("X-Forwarded-For", "192.168.0.10")).tenant == "ip:192.168.0.10"  # all trusted: the leftmost
    assert who("192.168.0.10", ("X-Forwarded-For", "fe80::1%eth0")).tenant == "ip:fe80::/64"  # a zone: the address without it
    assert who("192.168.0.10", ("X-Forwarded-For", "198.51.100.3:99999")).tenant == "ip:198.51.100.3"  # a port is not read
    for junk in ("%", "[", "]:", "1.2.3.4.5", "::ffff:", "x" * 5000, "١.٢.٣.٤", "[::1"):
        assert who("192.168.0.10", ("X-Forwarded-For", junk)).tenant == "ip:192.168.0.10", junk  # never raises: the proxy's own


def test_tenancy_is_off_without_keys_and_anonymous_and_r4_is_as_before():
    w = who("203.0.113.7", ("Authorization", "Bearer anything"), keys=[], anonymous=False)
    assert w.tenant is None and w.refusal is None  # as before: no keys, every request served, any key ignored
    assert who("203.0.113.7", keys=KEYS, anonymous=False).refusal == "key_needed"  # R4: keys and no key is 401
    assert who("127.0.0.1", keys=KEYS, anonymous=False).refusal == "key_needed"


def test_trusted_proxies_are_parsed_or_refused():
    nets = N.networks("192.168.0.10, 10.0.0.0/8 ::1")
    assert [str(n) for n in nets] == ["192.168.0.10/32", "10.0.0.0/8", "::1/128"]
    assert N.networks("") == ()
    with pytest.raises(ValueError):
        N.networks("192.168.0.X")
    cli = argparse.Namespace(trusted_proxies="nope", max_prompt=0, max_output=0, stream_ping=0.0, max_tokens=4096)
    with pytest.raises(SystemExit):
        N.check(cli)
    with pytest.raises(SystemExit):  # a default reply longer than the cap
        N.check(argparse.Namespace(trusted_proxies="", max_prompt=0, max_output=1000, stream_ping=0.0, max_tokens=4096))
    N.check(argparse.Namespace(trusted_proxies="", max_prompt=32768, max_output=8192, stream_ping=15.0, max_tokens=8192))


# the stand-in server with the operator's side

def stand_in(*argv, keys=(), max_tokens: int = 64, tok=None):
    cli = options(*argv)
    httpd = C.BoundedHTTPServer(("127.0.0.1", 0), S.Handler)
    inbox = Inbox()
    app = httpd.app = S.App(inbox, tok or Tok(), False, max_tokens, {"bits_per_weight": 3.0})
    A.install(app, inbox, cli, httpd, list(keys))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, app, inbox


@pytest.fixture()
def net():
    made = []

    def make(*argv, **kw):
        made.append(stand_in(*argv, **kw))
        return made[-1]

    yield make
    for httpd, _, _ in made:
        httpd.shutdown()
        httpd.server_close()


def served(httpd, inbox, body, hdrs=None):
    th, out = chat_async(httpd, body, hdrs)
    req = inbox.next()
    answer(req)
    th.join(10)
    return req, out


def test_the_request_object_gets_tenant_priority_and_scope(net):
    httpd, app, inbox = net()  # no option: tenancy off
    req, out = served(httpd, inbox, {"messages": HI})
    assert out["status"] == 200 and (req.tenant, req.priority, req.scope) == (None, 0, None)
    httpd, app, inbox = net("--anonymous", "--trusted-proxies", "127.0.0.1", keys=KEYS)
    req, _ = served(httpd, inbox, {"messages": HI}, {"Authorization": "Bearer sk-alice-0123456789"})
    assert (req.tenant, req.priority, req.scope) == ("key:alice", 0, None)
    req, _ = served(httpd, inbox, {"messages": HI})
    assert req.tenant == "local"
    req, _ = served(httpd, inbox, {"messages": HI}, {"X-Forwarded-For": "2001:db8:5:6::1"})
    assert req.tenant == "ip:2001:db8:5:6::/64"
    app.admin.users.per_tenant = True  # the cache stage's switch: the scope is the tenant
    req, _ = served(httpd, inbox, {"messages": HI}, {"X-Forwarded-For": "198.51.100.4"})
    assert req.scope == req.tenant == "ip:198.51.100.4"


def test_keys_with_anonymous_unknown_key_401_and_without_anonymous_as_r4(net):
    httpd, _, inbox = net("--anonymous", keys=KEYS)
    status, h, body = call(httpd, "POST", "/v1/chat/completions", {"messages": HI}, {"Authorization": "Bearer sk-wrong"})
    err = json.loads(body)["error"]
    assert status == 401 and err["code"] == "invalid_api_key" and "not known" in err["message"] and not inbox.reqs
    httpd, _, inbox = net(keys=KEYS)
    for hdrs in ({}, {"Authorization": "Bearer sk-wrong"}):  # R4's answer, byte for byte the same message
        status, h, body = call(httpd, "POST", "/v1/chat/completions", {"messages": HI}, hdrs)
        assert status == 401 and json.loads(body)["error"]["message"] == "an API key is needed: Authorization: Bearer <key> (or x-api-key)"


def test_a_prompt_above_max_prompt_is_openais_400(net):
    httpd, app, inbox = net("--max-prompt", "20")
    status, _, body = call(httpd, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x" * 30}]})
    err = json.loads(body)["error"]
    assert status == 400 and err["code"] == "context_length_exceeded" and err["param"] == "messages" and err["type"] == "invalid_request_error"
    assert err["message"].startswith("This model's maximum prompt length is 20 tokens. However, your messages resulted in 35 tokens.")
    req, out = served(httpd, inbox, {"messages": [{"role": "user", "content": "x" * 15}]})  # 20 tokens: served
    assert out["status"] == 200 and len(req.full) == 20
    httpd, app, inbox = net()  # without it: the context alone bounds it, as before
    req, out = served(httpd, inbox, {"messages": [{"role": "user", "content": "x" * 300}]})
    assert out["status"] == 200


def test_a_prompt_sure_to_be_over_max_prompt_is_refused_before_the_tokenizer(net):
    """More text than --max-prompt tokens can hold (48 bytes a token at most) is refused before the template and the
    tokenizer run under the prompt lock: a 30 MB body does not hold every other request for half a minute."""
    httpd, app, inbox = net("--max-prompt", "20")
    seen = []
    real = app.tok.apply_chat_template
    app.tok.apply_chat_template = lambda *a, **k: seen.append(1) or real(*a, **k)
    big = [{"role": "system", "content": "s" * 500}, {"role": "user", "content": [{"type": "text", "text": "ü" * 300}]}]  # 1,100 bytes > 960
    status, _, body = call(httpd, "POST", "/v1/chat/completions", {"messages": big})
    err = json.loads(body)["error"]
    assert status == 400 and err["code"] == "context_length_exceeded" and err["param"] == "messages" and not seen
    assert "more than 960 bytes of text, more than 20 tokens can hold" in err["message"]
    status, _, body = call(httpd, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x" * 900}]})
    assert status == 400 and "resulted in 905 tokens" in json.loads(body)["error"]["message"] and seen  # not sure: tokenized
    assert N.text_bytes(big) == 6 + 500 + 4 + 4 + 600  # roles and part types count too: a turn's own template tokens more than cover them
    assert not N.sure_too_long(big, 0) and not N.sure_too_long(big, 24) and N.sure_too_long(big, 23)
    assert N.text_bytes({"a": ["xy", {"b": "€"}], "n": 5}) == 5


def test_max_tokens_above_max_output_is_cut_and_named(net):
    httpd, app, inbox = net("--max-output", "50", max_tokens=40)
    req, out = served(httpd, inbox, {"messages": HI, "max_tokens": 500})
    assert out["status"] == 200 and req.max_new == 50 and out["headers"]["x-sclmd-ignored"] == "max_tokens"
    req, out = served(httpd, inbox, {"messages": HI, "max_completion_tokens": 51, "temperature": 0})
    assert req.max_new == 50 and out["headers"]["x-sclmd-ignored"] == "max_completion_tokens"
    req, out = served(httpd, inbox, {"messages": HI, "max_tokens": 50})
    assert req.max_new == 50 and "x-sclmd-ignored" not in out["headers"]
    req, out = served(httpd, inbox, {"messages": HI})
    assert req.max_new == 40 and "x-sclmd-ignored" not in out["headers"]
    assert json.loads(call(httpd, "GET", "/v1/models")[2])["data"][0]["max_output_tokens"] == 50
    httpd, app, inbox = net()
    req, out = served(httpd, inbox, {"messages": HI, "max_tokens": 500})
    assert req.max_new == 500 and "x-sclmd-ignored" not in out["headers"]


def stream_events(httpd, inbox, hold_s: float, *, tokens=(10, 11, 12)) -> list:
    th, out = chat_async(httpd, {"messages": HI, "stream": True})
    req = inbox.next()
    time.sleep(hold_s)  # a long prompt being read: no token yet
    answer(req, tokens)
    th.join(10)
    return out["body"].decode().split("\n\n")


def test_a_stream_pings_until_its_first_token_and_not_after(net):
    httpd, _, inbox = net("--stream-ping", "0.3")
    events = stream_events(httpd, inbox, 1.25)
    pings = [i for i, e in enumerate(events) if e == ": ping"]
    first = next(i for i, e in enumerate(events) if e.startswith("data: ") and '"content": "k' in e)
    assert 3 <= len(pings) <= 5 and max(pings) < first and events[-2] == "data: [DONE]"
    httpd, _, inbox = net()  # off: as before
    assert not [e for e in stream_events(httpd, inbox, 1.25) if e.startswith(":")]


def test_metrics_from_this_pc_only(net):
    httpd, _, _ = net("--metrics-local", "--trusted-proxies", "127.0.0.1")
    assert call(httpd, "GET", "/metrics")[0] == 200
    status, _, body = call(httpd, "GET", "/metrics", headers={"X-Forwarded-For": "198.51.100.7"})
    assert status == 404 and b"sclmd_" not in body
    httpd, _, _ = net("--trusted-proxies", "127.0.0.1")  # without --metrics-local: as before
    assert call(httpd, "GET", "/metrics", headers={"X-Forwarded-For": "198.51.100.7"})[0] == 200


def test_metrics_from_another_machine_is_refused(net, monkeypatch):
    httpd, app, _ = net("--metrics-local")
    handler = type("H", (), {})()
    handler.server, handler.client_address, handler.headers = httpd, ("203.0.113.9", 5000), headers()
    handler._n1_who = app.admin.users.who(handler)
    assert N.metrics_refused(handler)
    handler.client_address = ("127.0.0.1", 5000)
    handler._n1_who = app.admin.users.who(handler)
    assert not N.metrics_refused(handler)


def test_a_tunnel_on_this_pc_is_not_this_pc(net, capsys):
    """A proxy or tunnel on this PC that is not in --trusted-proxies: its loopback connections carry X-Forwarded-For.
    They are the keyless user ip:127.0.0.1, not this PC: no /metrics, no counts on GET /, and the console is told."""
    httpd, app, inbox = net("--metrics-local", "--anonymous", "--trusted-proxies", "192.168.0.10", keys=KEYS)
    tunnel = {"X-Forwarded-For": "198.51.100.7"}
    status, _, body = call(httpd, "GET", "/metrics", headers=tunnel)
    assert status == 404 and b"sclmd_" not in body
    about = json.loads(call(httpd, "GET", "/", headers=tunnel)[2])
    assert about["model"] == S.MODEL and "requests" not in about and "conversations_kept" not in about
    req, _ = served(httpd, inbox, {"messages": HI}, tunnel)
    assert req.tenant == "ip:127.0.0.1"
    assert "is not in --trusted-proxies" in capsys.readouterr().out
    assert call(httpd, "GET", "/metrics")[0] == 200  # this PC itself
    assert "requests" in json.loads(call(httpd, "GET", "/")[2])
    httpd, _, _ = net()  # without --metrics-local: GET / as before, counts for everyone
    assert "requests" in json.loads(call(httpd, "GET", "/", headers=tunnel)[2])


# ZDR

CANARY = "Zdr0123456789q"


def test_the_request_log_keeps_no_message_and_the_tenant(net, tmp_path):
    path = tmp_path / "requests.jsonl"
    httpd, app, inbox = net("--request-log", str(path), "--anonymous", "--trusted-proxies", "127.0.0.1", keys=KEYS)
    for k, body in enumerate(({"messages": HI, "max_tokens": CANARY}, {"messages": HI, "temperature": CANARY}, {"messages": [{"role": CANARY, "content": "x"}]})):
        status, _, out = call(httpd, "POST", "/v1/chat/completions", body, {"X-Forwarded-For": "198.51.100.7"})
        assert status == 400 and (CANARY.encode() in out or k == 2)  # the client is told what it sent, as before
    served(httpd, inbox, {"messages": HI}, {"X-Forwarded-For": "198.51.100.8"})
    t = time.time()
    while app.admin.log.written < 4:
        assert time.time() - t < 5
        time.sleep(0.02)
    text = path.read_text(encoding="utf-8")
    lines = [json.loads(x) for x in text.splitlines()]
    assert CANARY not in text and all("error" not in x for x in lines)
    assert [x.get("tenant") for x in lines] == ["ip:198.51.100.7"] * 3 + ["ip:198.51.100.8"]
    assert lines[2].get("param") == "messages"


def test_trace_and_why_leave_the_message_out():
    try:
        try:
            int(CANARY)
        except ValueError as e:
            raise KeyError(CANARY) from e
    except KeyError as e:
        out = io.StringIO()
        C.trace(e, out)
        text = out.getvalue()
    assert CANARY not in text and "ValueError" in text and "KeyError" in text and "int(CANARY)" in text
    assert C.why(ValueError(CANARY)) == "ValueError"
    assert C.why(RuntimeError("CUDA error: an illegal memory access was encountered\nmore")) == \
        "RuntimeError: CUDA error: an illegal memory access was encountered"
    assert C.why(ValueError("CUDA error " + CANARY)) == "ValueError"  # only torch's RuntimeError keeps the driver's words
    assert C.why(RuntimeError("CUDAx " + CANARY)) == "RuntimeError"


def test_quiet_hooks_print_an_escaped_exception_without_its_message(monkeypatch, capsys):
    """serve.main's C.quiet_hooks(): an exception that escapes a thread, the main program or a __del__ prints its
    frames and type, as trace() does, not Python's default with the message."""
    import sys
    for mod, name in ((sys, "excepthook"), (threading, "excepthook"), (sys, "unraisablehook")):
        monkeypatch.setattr(mod, name, getattr(mod, name))  # put back after the test
    C.quiet_hooks()

    def fail():
        raise ValueError(CANARY)

    th = threading.Thread(target=fail, name="engine")
    th.start()
    th.join()
    try:
        fail()
    except ValueError as e:
        sys.excepthook(type(e), e, e.__traceback__)

    class Dying:
        def __del__(self):
            raise KeyError(CANARY)

    Dying()
    err = capsys.readouterr().err
    assert "Exception in thread engine:" in err and err.count("\nValueError\n") == 2 and "\nKeyError\n" in err
    assert "Exception ignored (raised where it could not go on" in err and CANARY not in err
    threading.Thread(target=lambda: sys.exit(3)).start()  # a thread that exits is no error, as with Python's own hook
    time.sleep(0.2)
    assert capsys.readouterr().err == ""


def test_a_handler_that_fails_outside_its_answers_is_logged_without_the_message(capsys):
    class Boom(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            raise ValueError(CANARY)

        def log_message(self, *a):
            pass

    httpd = C.BoundedHTTPServer(("127.0.0.1", 0), Boom)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        s = socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=5)
        s.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        s.recv(100)
        s.close()
        time.sleep(0.3)
    finally:
        httpd.shutdown()
        httpd.server_close()
    err = capsys.readouterr().err
    assert "ValueError" in err and CANARY not in err


class ToolTok(Tok):
    """The letters' template, with tools (declared nowhere) and the tool format's tokens."""

    def convert_tokens_to_ids(self, s):
        return {"<|tool_call>": 90, "<tool_call|>": 91, "<|tool_response>": 92, '<|"|>': 93}.get(s) or super().convert_tokens_to_ids(s)

    def apply_chat_template(self, messages, tools=None, **kw):
        return super().apply_chat_template(messages, **kw)


def test_malformed_and_undeclared_calls_are_counted_not_quoted(net, capsys, monkeypatch):
    httpd, app, inbox = net(tok=ToolTok())

    class Reader:  # a reply whose calls did not parse, and one of a function that was not declared
        def __init__(self, *a, **k):
            self.malformed, self.calls = [(f"no ':' after the key {CANARY!r}", f"<|tool_call>call:f{{{CANARY}}}")], []

        def add(self, token):
            return ("answer", "k")

        def take(self):
            out, self.calls = self.calls, []
            return out

        def end(self):
            self.calls = [{"id": "call_1", "type": "function", "function": {"name": CANARY, "arguments": "{}"}, "args": {}, "at": 0}]
            return None

    monkeypatch.setattr(S.T, "Reader", Reader)
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    req, out = served(httpd, inbox, {"messages": HI, "tools": tools})
    assert req.tools
    assert out["status"] == 200 and CANARY.encode() in out["body"]  # the client gets the calls
    said = capsys.readouterr().out
    assert "1 tool call block(s) did not parse" in said and "1 call(s) of a function not declared" in said and CANARY not in said


# the public configuration

def test_serve_toml_supervise_and_int8_true(tmp_path):
    cfg = tmp_path / "serve.toml"
    cfg.write_text('supervise = true\nint8 = true\nanonymous = true\ntrusted-proxies = "192.168.0.10"\nmax-prompt = 32768\n', encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", type=float, nargs="?", const=5.0, default=0.0)
    A.add_options(ap)
    cli = A.parse_args(ap, ["--config", str(cfg)])
    assert cli.int8 == 5.0 and cli.anonymous and cli.trusted_proxies == "192.168.0.10" and cli.max_prompt == 32768
    assert not hasattr(cli, "supervise")
    cfg.write_text("supervise = 1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        A.parse_args(ap, ["--config", str(cfg)])
    cfg.write_text("supervise = true\nport = 8070\n", encoding="utf-8")
    assert supervise.wanted(["--config", str(cfg)]) and supervise.served_at(["--config", str(cfg)]) == ("127.0.0.1", 8070)
    assert supervise.wanted(["--SUPERVISE", "--config", ""]) and not supervise.wanted(["--config", ""])
    cfg.write_text("supervise = false\n", encoding="utf-8")
    assert not supervise.wanted(["--config", str(cfg)])


def test_the_public_example_is_read_whole():
    path = os.path.join(ROOT, "serve.public.example.toml")
    cli = A.parse_args(S.parser(), ["--config", path])
    assert (cli.int8, cli.kv4, cli.wide_rows, cli.keyless, cli.context, cli.typical, cli.prefix_gib, cli.image) == (5.0, True, 256, True, 40992, 8192, 2.0, "auto")
    assert (cli.max_prompt, cli.max_output, cli.max_tokens, cli.stream_ping) == (32768, 8192, 8192, 15.0)
    assert (cli.max_body_mb, cli.idle_timeout, cli.max_connections) == (32.0, 120.0, 256)
    assert cli.anonymous and cli.metrics_local and cli.require_key and cli.trusted_proxies and cli.host == "0.0.0.0"
    assert supervise.wanted(["--config", path])
    N.check(cli)


def test_the_r4_default_answers_are_unchanged_without_options(net):
    httpd, app, inbox = net()
    assert app.admin.users.on is False and N.line(options(), app.admin.users) is None
    status, _, body = call(httpd, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x" * 5000}]})
    err = json.loads(body)["error"]
    assert status == 400 and err["message"].startswith("the conversation is 50") and err["message"].endswith("tokens: the context holds 4096")
    assert (err["type"], err["param"], err["code"]) == ("invalid_request_error", None, None)
