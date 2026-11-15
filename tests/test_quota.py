# Quotas (scripts/serve_quota.py: --quotas, and --priorities' seam on the HTTP side) without a model or a GPU: the
# counters on a clock of the test's, the daily limit's 429, the x-ratelimit-* headers, GET /v1/usage, the counters'
# file, the key file's own limits, and the real App and Handler over the test's stand-in engine (tests/test_serve_admin.py).
import json
import os
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from scripts import serve as S
from scripts import serve_admin as A
from scripts import serve_contract as C
from scripts import serve_quota as Q
from test_api_contract import Tok
from test_serve_admin import HI, Inbox, answer, call, chat_async, options

KEYS = [("alice", b"sk-alice-0123456789"), ("bob", b"sk-bob-0123456789ab")]
ALICE = {"Authorization": "Bearer sk-alice-0123456789"}
BOB = {"Authorization": "Bearer sk-bob-0123456789ab"}


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def req_of(tenant, prompt: int):
    return SimpleNamespace(tenant=tenant, full=[5] * prompt, job=None, reused=0)


def done(prompt: int, cached: int, out: int) -> tuple:
    return ("done", {"usage": {"prompt_tokens": prompt, "completion_tokens": out, "prompt_tokens_details": {"cached_tokens": cached}}})


def spend(q: Q.Quota, tenant: str, prompt: int, cached: int = 0, out: int = 0, placed: bool = True):
    r = req_of(tenant, prompt)
    q.taken(r)
    if placed:
        r.job = SimpleNamespace(out=[1] * out, seq=0)
        r.reused = cached
    q.settle(r, done(prompt, cached, out) if placed else None)
    return r


# the counters

def test_a_requests_tokens_are_its_uncached_prompt_its_reply_and_a_tenth_of_its_cached_prompt():
    clock = Clock()
    q = Q.Quota(clock=clock)
    spend(q, "key:alice", 1000, cached=600, out=50)
    assert q.usage_of("key:alice")["hour"]["tokens"] == 400 + 50 + 60 and q.usage_of("key:alice")["hour"]["requests"] == 1
    r = req_of("key:alice", 300)
    q.taken(r)  # counted with its whole prompt while it is under way
    assert q.usage_of("key:alice")["hour"]["tokens"] == 510 + 300
    r.job, r.reused = SimpleNamespace(out=[1, 2, 3]), 100
    q.settle(r, None)  # its client left: what it had read and written by then
    assert q.usage_of("key:alice")["hour"]["tokens"] == 510 + 200 + 3 + 10
    spend(q, "key:alice", 5000, placed=False)  # it never reached the engine (a 503, a client that left while it waited)
    assert q.usage_of("key:alice")["hour"]["requests"] == 2


def test_the_hour_and_the_day_roll():
    clock = Clock()
    q = Q.Quota(clock=clock)
    spend(q, "key:alice", 100)
    clock.t += 1800
    spend(q, "key:alice", 10)
    assert q.usage_of("key:alice")["hour"]["tokens"] == 110
    clock.t += 1800 + 61  # the first has left the hour (minute buckets: a minute counts while any of it is inside)
    u = q.usage_of("key:alice")
    assert u["hour"]["tokens"] == 10 and u["day"]["tokens"] == 110
    clock.t += Q.DAY
    assert q.usage_of("key:alice")["day"]["tokens"] == 0
    q.prune()
    assert not q.used and not q.keyless  # idle tenants are dropped


def test_over_quota_by_tokens_or_requests_and_back_in_quota():
    clock = Clock()
    q = Q.Quota(key=(1000, 3), ip=(100, 2), clock=clock)
    spend(q, "key:alice", 999)
    assert not q.over("key:alice")
    spend(q, "key:alice", 1)
    assert q.over("key:alice") and q.usage_of("key:alice")["over_quota"]
    assert 3600 <= q.usage_of("key:alice")["back_in_quota_s"] <= 3660  # (its minute counts while any of it is inside)
    spend(q, "ip:203.0.113.7", 1)
    spend(q, "ip:203.0.113.7", 1)
    assert q.over("ip:203.0.113.7") and not q.over("ip:203.0.113.8")  # requests: 2 of 2
    clock.t += 3661
    assert not q.over("key:alice") and not q.over("ip:203.0.113.7")
    assert not q.over("local") and q.limits_of("local") is None and q.limits_of(None) is None


def test_the_daily_limit_refuses_with_the_time_until_it_fits():
    clock = Clock()
    q = Q.Quota(key=(1000, 10), daily=2, clock=clock)  # the day: 2,000 tokens and 20 requests
    spend(q, "key:alice", 1500)
    clock.t += 7200
    spend(q, "key:alice", 400)
    assert q.refusal("key:alice", 100) is None
    r = q.refusal("key:alice", 101)
    assert r.kind == "tokens_per_day" and r.limit == 2000 and r.used == 1900
    assert Q.DAY - 7200 - 60 <= r.retry_s <= Q.DAY - 7200 + 60  # when the 1,500 leave the day
    assert r.body()["error"]["code"] == "rate_limit_exceeded" and r.body()["error"]["type"] == "tokens"
    assert int(r.headers()["Retry-After"]) == -(-r.retry_s // 1) and int(r.headers()["retry-after-ms"]) == int(r.retry_s * 1000)
    assert q.refusal("key:alice", 5000).retry_s == Q.DAY  # more than a whole day's: never fits
    for _ in range(19):
        spend(q, "key:bob", 1)
    assert q.refusal("key:bob", 1) is None  # the 20th of 20
    spend(q, "key:bob", 1)
    assert q.refusal("key:bob", 1).kind == "requests_per_day"
    assert q.refusal("local", 10**9) is None


def test_all_requests_without_a_key_share_a_day():
    clock = Clock()
    q = Q.Quota(ip=(10**9, 10**6), keyless_day=1000, clock=clock)
    spend(q, "ip:203.0.113.7", 600)
    spend(q, "key:alice", 5000)  # a key's tokens are not the keyless day's
    spend(q, "ip:2001:db8:1:2::/64", 300)
    assert q.refusal("ip:198.51.100.1", 100) is None
    r = q.refusal("ip:198.51.100.1", 101)
    assert r.kind == "keyless_tokens_per_day" and r.used == 900 and "without an API key" in r.body()["error"]["message"]
    assert q.refusal("key:alice", 101) is None
    assert q.usage_of("ip:198.51.100.1")["keyless_day"]["tokens"] == 900


def test_x_ratelimit_headers_are_openais_names_and_the_hours():
    clock = Clock(3600.0 * 1000 + 10)
    q = Q.Quota(key=(500, 4), clock=clock)
    assert q.headers("local") == {} and q.headers(None) == {}
    h = q.headers("key:alice")
    assert list(h) == ["x-ratelimit-limit-requests", "x-ratelimit-limit-tokens", "x-ratelimit-remaining-requests",
                       "x-ratelimit-remaining-tokens", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"]
    assert h["x-ratelimit-remaining-tokens"] == "500" and h["x-ratelimit-reset-tokens"] == "0s"
    spend(q, "key:alice", 120)
    h = q.headers("key:alice")
    assert h["x-ratelimit-remaining-tokens"] == "380" and h["x-ratelimit-remaining-requests"] == "3"
    assert h["x-ratelimit-reset-tokens"] == "1h0m50s"  # until all of it has left the hour: the minute's end + 1 h
    spend(q, "key:alice", 1000)
    assert q.headers("key:alice")["x-ratelimit-remaining-tokens"] == "0"
    assert [Q.duration(s) for s in (0, 0.2, 1, 59.5, 360, 3599, 3661)] == ["0s", "1s", "1s", "1m0s", "6m0s", "59m59s", "1h1m1s"]
    q2 = Q.Quota(key=(0, 5), clock=clock)  # no hourly token limit: no token headers
    assert "x-ratelimit-limit-tokens" not in q2.headers("key:alice") and "x-ratelimit-limit-requests" in q2.headers("key:alice")


def test_the_counters_are_written_atomically_and_read_back(tmp_path):
    path = str(tmp_path / "quota.json")
    clock = Clock()
    q = Q.Quota(path=path, clock=clock)
    spend(q, "key:alice", 1000, cached=500, out=20)
    spend(q, "ip:203.0.113.7", 30)
    q.save()
    assert os.listdir(tmp_path) == ["quota.json"]  # no temporary left
    again = Q.Quota(path=path, clock=clock)
    assert again.load() == 2
    for tenant in ("key:alice", "ip:203.0.113.7"):
        assert again.usage_of(tenant) == q.usage_of(tenant)
    assert again.refusal("ip:1.2.3.4", 1) is None and sum(again.keyless.values()) == 30
    data = json.load(open(path, encoding="utf-8"))
    assert set(data) == {"version", "bucket_s", "saved", "tenants", "keyless"}  # numbers and tenant ids only
    clock.t += Q.DAY + 120
    assert Q.Quota(path=path, clock=clock).load() == 0  # a day later: nothing left to read back
    open(path, "w").write("{not json")
    bad = Q.Quota(path=path, clock=clock)
    assert bad.load() == 0 and os.path.exists(path + ".bad") and not os.path.exists(path)


# the key file

def test_a_keys_own_limits_follow_it_and_r4s_lines_read_as_before():
    r4 = "# keys\nalice: sk-alice-0123456789\nsk-plain-0123456789\nbob : sk-bob-0123456789ab\n"
    assert A.parse_keys(r4, "file") == [("alice", b"sk-alice-0123456789"), ("file2", b"sk-plain-0123456789"), ("bob", b"sk-bob-0123456789ab")]
    limits = {}
    got = A.parse_keys("alice: sk-alice-0123456789 tph=1000000 rph=1200\nbob: sk-bob tpd=0\nsk-x tph=5\ncarol: sk with words\n", "file", limits)
    assert got == [("alice", b"sk-alice-0123456789"), ("bob", b"sk-bob"), ("file3", b"sk-x"), ("carol", b"sk with words")]
    assert limits == {"alice": {"tph": 1000000, "rph": 1200}, "bob": {"tpd": 0}, "file3": {"tph": 5}}
    assert A.parse_keys("a:k1 tph=1,b:k2", "key", {}) == [("a", b"k1"), ("b", b"k2")]  # SCLMD_API_KEY's commas
    # without --quotas (no limits to fill) the whole entry is the key, as before N2: what authenticates does not change
    assert A.parse_keys("alice: sk-alice-0123456789 tph=1000000\nsk-x tph=5", "file") == [("alice", b"sk-alice-0123456789 tph=1000000"),
                                                                                            ("file2", b"sk-x tph=5")]
    q = Q.Quota(per_key={"alice": {"tph": 1000000, "rph": 1200}, "bob": {"tpd": 0}})
    assert vars_of(q.limits_of("key:alice")) == (1000000, 1200, 24000000, 28800)
    assert vars_of(q.limits_of("key:bob")) == (500000, 600, 0, 14400)  # tpd 0: no daily token limit
    assert vars_of(q.limits_of("key:dave")) == (500000, 600, 12000000, 14400)
    assert vars_of(q.limits_of("ip:203.0.113.7")) == (100000, 120, 2400000, 2880)


def vars_of(lim) -> tuple:
    return lim.tph, lim.rph, lim.tpd, lim.rpd


def test_the_key_files_limits_reach_the_server(tmp_path, monkeypatch, capsys):
    f = tmp_path / "keys.txt"
    f.write_text("alice: sk-alice-0123456789 tph=7 rph=9\n", encoding="utf-8")
    monkeypatch.delenv("SCLMD_API_KEY", raising=False)
    cli = options("--quotas", "--api-key-file", str(f))
    keys = A.start(cli)
    assert keys == [("alice", b"sk-alice-0123456789")] and cli.key_limits == {"alice": {"tph": 7, "rph": 9}}
    assert "has a space in it" not in capsys.readouterr().out
    cli = options("--api-key-file", str(f))  # no --quotas: R4's reading, the whole line is the key
    assert A.start(cli) == [("alice", b"sk-alice-0123456789 tph=7 rph=9")] and cli.key_limits == {}
    assert "has a space in it" not in capsys.readouterr().out  # that warning is --quotas' only (a miswritten limit)
    with pytest.raises(SystemExit):
        A.start(options("--quotas"))  # nobody to count: no key and no --anonymous


# the server (the real App and Handler; the test answers in the engine's place)

def stand_in(*argv, keys=KEYS, limits=None, max_tokens: int = 64):
    cli = options(*argv)
    cli.key_limits = limits or {}
    httpd = C.BoundedHTTPServer(("127.0.0.1", 0), S.Handler)
    inbox = Inbox()
    app = httpd.app = S.App(inbox, Tok(), False, max_tokens, {"bits_per_weight": 3.0})
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


def served(httpd, inbox, body=None, hdrs=None, out_tokens=(10, 11, 12)):
    """A request the test answers as the engine would, having placed it (a job)."""
    th, out = chat_async(httpd, body or {"messages": HI}, hdrs)
    req = inbox.next()
    req.job, req.reused = SimpleNamespace(out=list(out_tokens), seq=0), 0
    answer(req, out_tokens, {"finish_reason": "stop", "ended": "stop", "timings": {"total_ms": 5, "first_token_ms": 1, "tokens_per_second": 1.0},
                             "usage": {"prompt_tokens": len(req.full), "completion_tokens": len(out_tokens), "total_tokens": len(req.full) + 3,
                                       "prompt_tokens_details": {"cached_tokens": 0}}})
    th.join(10)
    return req, out


PUBLIC = ("--quotas", "--priorities", "--anonymous", "--trusted-proxies", "127.0.0.1")


def test_every_answer_to_a_tenant_carries_its_hours_headers(net):
    httpd, app, inbox = net(*PUBLIC, limits={"bob": {"tph": 50, "rph": 2}})
    req, out = served(httpd, inbox, hdrs=ALICE)
    assert out["status"] == 200 and out["headers"]["x-ratelimit-limit-tokens"] == "500000"
    assert out["headers"]["x-ratelimit-limit-requests"] == "600" and out["headers"]["x-ratelimit-remaining-requests"] == "599"
    assert int(out["headers"]["x-ratelimit-remaining-tokens"]) == 500000 - len(req.full)  # the prompt counted on its way in
    _, out = served(httpd, inbox, hdrs={"X-Forwarded-For": "203.0.113.7"})
    assert out["headers"]["x-ratelimit-limit-tokens"] == "100000" and out["headers"]["x-ratelimit-limit-requests"] == "120"
    _, out = served(httpd, inbox)  # this PC (loopback, no proxy): exempt, no headers
    assert not any(k.startswith("x-ratelimit") for k in out["headers"])
    status, h, _ = call(httpd, "POST", "/v1/chat/completions", {"messages": "not a list"}, BOB)  # a 400 too
    assert status == 400 and h["x-ratelimit-limit-tokens"] == "50"
    status, h, _ = call(httpd, "GET", "/v1/models", headers=BOB)
    assert status == 200 and h["x-ratelimit-limit-requests"] == "2"
    status, h, _ = call(httpd, "GET", "/v1/models", headers={"Authorization": "Bearer sk-unknown"})  # 401: nobody to count
    assert status == 401 and not any(k.startswith("x-ratelimit") for k in h)
    # the cost: 200s cost what they read and wrote; the 400 and the GETs nothing
    u = app.admin.quota.usage_of("key:alice")
    assert u["hour"]["requests"] == 1 and u["hour"]["tokens"] == len(req.full) + 3
    assert app.admin.quota.usage_of("key:bob")["hour"]["requests"] == 0


def test_without_quotas_no_answer_has_the_headers_and_usage_is_an_unknown_path(net):
    for argv in ((), ("--anonymous",)):
        httpd, app, inbox = net(*argv)
        _, out = served(httpd, inbox, hdrs=ALICE)
        assert out["status"] == 200 and not any(k.startswith("x-ratelimit") for k in out["headers"])
        status, _, body = call(httpd, "GET", "/v1/usage", headers=ALICE)
        assert status == 404 and app.admin.quota is None and app.admin.order is None
        assert app.admin.users.priority_of(SimpleNamespace(), None) == 0  # N1's seam as it was


def test_usage_tells_the_caller_its_own(net):
    httpd, app, inbox = net(*PUBLIC, limits={"alice": {"tph": 10}})
    served(httpd, inbox, hdrs=ALICE)
    status, h, body = call(httpd, "GET", "/v1/usage", headers=ALICE)
    u = json.loads(body)
    assert status == 200 and u["user"] == "key:alice" and u["hour"]["requests"] == 1 and u["over_quota"] and u["back_in_quota_s"] > 3000
    assert u["hour"]["limit_tokens"] == 10 and u["day"]["limit_tokens"] == 240 and h["x-ratelimit-remaining-tokens"] == "0"
    u = json.loads(call(httpd, "GET", "/v1/usage", headers=BOB)[2])
    assert u["user"] == "key:bob" and u["hour"]["requests"] == 0 and not u["over_quota"]
    u = json.loads(call(httpd, "GET", "/v1/usage", headers={"X-Forwarded-For": "2001:db8:1:2::9"})[2])
    assert u["user"] == "ip:2001:db8:1:2::/64" and "keyless_day" in u
    assert json.loads(call(httpd, "GET", "/v1/usage")[2])["exempt"] is True


def test_the_daily_limit_is_429_with_retry_after_and_costs_nothing(net, tmp_path):
    log = tmp_path / "requests.jsonl"
    httpd, app, inbox = net(*PUBLIC, "--request-log", str(log), limits={"alice": {"tpd": 100, "rpd": 3}})
    req, out = served(httpd, inbox, {"messages": [{"role": "user", "content": "x" * 60}], "max_tokens": 20}, ALICE)  # 65 + 20 held of the day's 100
    assert out["status"] == 200
    status, h, body = call(httpd, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "y" * 60}]}, ALICE)
    err = json.loads(body)["error"]
    assert status == 429 and err["code"] == "rate_limit_exceeded" and err["type"] == "tokens" and "tokens per day" in err["message"]
    assert Q.DAY - 120 <= int(h["retry-after"]) <= Q.DAY + 60 and "x-ratelimit-limit-tokens" in h
    assert not inbox.reqs  # never reached the engine
    status, _, body = call(httpd, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "z"}]}, ALICE)
    assert status == 429 and "Requested 70" in json.loads(body)["error"]["message"]  # its reply counts at its max_tokens (64)
    short = {"messages": [{"role": "user", "content": "z"}], "max_tokens": 4}
    served(httpd, inbox, short, ALICE)  # a short one that asks for a short reply still fits
    served(httpd, inbox, short, ALICE)
    assert app.admin.quota.usage_of("key:alice")["day"]["requests"] == 3
    # at the limit: refused before the body is read (a body announced and never sent does not hold the thread)
    s = socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=10)
    t = time.time()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer sk-alice-0123456789\r\n"
              b"Content-Type: application/json\r\nContent-Length: 1000000\r\n\r\n{\"mess")
    got = b""
    while b"per day" not in got and time.time() - t < 3:
        got += s.recv(65536)
    assert got.startswith(b"HTTP/1.1 429") and b"requests per day" in got and time.time() - t < 3
    s.close()
    assert app.admin.quota.usage_of("key:alice")["day"]["requests"] == 3 and app.admin.quota.n["requests_per_day"] == 1
    time.sleep(0.3)
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    refused = [x for x in lines if x.get("finish") == "daily_limit"]
    assert len(refused) == 3 and {x["limit"] for x in refused} == {"tokens_per_day", "requests_per_day"}
    assert all("message" not in x and "error" not in x for x in refused)  # ZDR: which limit, never a message
    assert all(x.get("priority") is not None for x in lines if x.get("status") == 200)


def test_max_waiting_with_priorities_counts_only_those_a_request_would_wait_behind(net):
    httpd, app, inbox = net(*PUBLIC, "--max-waiting", "2", limits={"bob": {"tph": 1, "tpd": 0}})  # (tpd 0: no day to pass)
    served(httpd, inbox, hdrs=BOB)  # bob is over his quota now
    held = [chat_async(httpd, {"messages": HI}, BOB) for _ in range(2)]
    while len(inbox.reqs) < 2:
        time.sleep(0.01)
    status, _, _ = call(httpd, "POST", "/v1/chat/completions", {"messages": HI}, BOB)
    assert status == 429  # a third over-quota one: two of its class wait already
    th, out = chat_async(httpd, {"messages": HI}, ALICE)  # in quota: the over-quota ones are not ahead of it
    while len(inbox.reqs) < 3:
        time.sleep(0.01)
    for req in list(inbox.reqs):
        answer(req)
    inbox.reqs.clear()
    th.join(10)
    for t, _ in held:
        t.join(10)
    assert out["status"] == 200


def test_the_seam_gives_the_class_when_a_request_comes(net):
    httpd, app, inbox = net(*PUBLIC, limits={"bob": {"tph": 1, "tpd": 0}})
    req, _ = served(httpd, inbox, hdrs=ALICE)
    assert req.priority == 1 and not req.local  # in quota
    served(httpd, inbox, hdrs=BOB)
    req, _ = served(httpd, inbox, hdrs=BOB)
    assert req.priority == 1 + app.admin.order.gap  # over quota
    req, _ = served(httpd, inbox)
    assert req.priority == 0 and req.local and req.tenant == "local"


def test_metrics_count_and_name_no_tenant(net):
    httpd, app, inbox = net(*PUBLIC, limits={"alice": {"rpd": 1}})
    served(httpd, inbox, hdrs=ALICE)
    call(httpd, "POST", "/v1/chat/completions", {"messages": HI}, ALICE)
    served(httpd, inbox, hdrs={"X-Forwarded-For": "203.0.113.7"})
    text = call(httpd, "GET", "/metrics")[2].decode()
    assert 'sclmd_quota_refused_total{limit="requests_per_day"} 1' in text
    assert 'sclmd_quota_taken_total{class="in_quota"} 2' in text and 'sclmd_quota_tenants{kind="ip"} 1' in text
    assert "alice" not in text and "203.0.113.7" not in text


def test_a_requests_class_is_its_tenants_hour_before_it_and_stays():
    clock = Clock()
    q = Q.Quota(key=(100, 2), clock=clock)
    first = spend(q, "key:alice", 500)  # its own prompt passes the hour's tokens: still in quota
    assert first.over_quota is False and q.over("key:alice")
    q2 = Q.Quota(key=(10**9, 2), clock=clock)
    rs = [spend(q2, "key:bob", 1) for _ in range(3)]
    assert [r.over_quota for r in rs] == [False, False, True]  # the 2nd of 2 is in quota, the 3rd over
    assert q2.n["class_in_quota"] == 2 and q2.n["class_over_quota"] == 1


def test_replies_under_way_are_held_for_the_daily_check():
    """Requests that come at once cannot pass a day's tokens by their replies: each holds its max_tokens until it is
    settled (verify-N2-quota P1: 12 at once against tpd 100 ended the day at 161)."""
    clock = Clock()
    q = Q.Quota(key=(1000, 100), daily=0.1, clock=clock)  # the day: 100 tokens
    under_way = []
    for _ in range(12):
        r = req_of("key:alice", 5)
        r.max_new = 20
        if q.refusal("key:alice", 5, reply=20) is None:
            q.taken(r)
            under_way.append(r)
    assert len(under_way) == 4 and q.held["key:alice"] == 80  # 4 x (5 + 20) = 100
    worst = 0.0
    for r in under_way:  # each writes its whole max_tokens
        r.job = SimpleNamespace(out=[1] * 20, seq=0)
        q.settle(r, done(5, 0, 20))
        worst = max(worst, q.usage_of("key:alice")["day"]["tokens"])
    assert worst == 100 and "key:alice" not in q.held  # never past the day
    r = q.refusal("key:alice", 1, reply=0)
    assert r.kind == "tokens_per_day" and r.retry_s > Q.HOUR  # the day is full: the counters' wait, not HELD_RETRY
    q2 = Q.Quota(key=(1000, 100), daily=0.1, clock=clock)
    a = req_of("key:bob", 5)
    a.max_new = 90
    q2.taken(a)
    r = q2.refusal("key:bob", 5, reply=10)  # only the reply under way is in the way
    assert r is not None and r.retry_s == q2.HELD_RETRY and r.used == 95
    assert q2.refusal("key:bob", 5, reply=10, held=False) is None  # the check before a body is read: the counters alone
    q3 = Q.Quota(ip=(10**9, 10**6), keyless_day=100, clock=clock)  # every keyless tenant's replies, together
    b = req_of("ip:203.0.113.7", 10)
    b.max_new = 80
    q3.taken(b)
    assert q3.held_keyless == 80 and q3.refusal("ip:198.51.100.1", 5, reply=10).kind == "keyless_tokens_per_day"
    q3.settle(b, None)
    assert q3.held_keyless == 0 and q3.refusal("ip:198.51.100.1", 5, reply=10) is None


def test_prune_drops_worked_out_over_entries_of_tenants_never_counted():
    """verify-N2-quota P2: a tenant asked about (the priority seam) but never counted (refused after its body was read)
    left its _over entry for ever."""
    clock = Clock()
    q = Q.Quota(clock=clock)
    for i in range(500):
        q.over(f"ip:198.51.100.{i % 250}") if i < 250 else q.over(f"ip:2001:db8:{i}::/64")
    assert len(q._over) == 500 and not q.used
    clock.t += 2
    q.prune()
    assert not q._over


def test_the_counters_are_pruned_without_a_file_too():
    clock = Clock()
    q = Q.Quota(clock=clock)  # no --quota-file
    for i in range(50):
        spend(q, f"ip:203.0.113.{i}", 10)
    assert len(q.used) == 50
    clock.t += Q.DAY + 120
    q.tick()
    assert not q.used and not q.keyless and q.saved == 0
