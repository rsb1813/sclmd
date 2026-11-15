# Tiers, concurrency, the request log's retention and GET /v1/status (scripts/serve_tiers.py, design.md 6.21) without a
# model or a GPU: the tier of an address from a test's lookup (and from the real DB-IP Lite files when they are in
# C:/models/sclmd-ipdb), its demotion by behaviour, each tier's limits and the output weight, the concurrency ladder and
# the 13th request's 429, the order of the tiers through sclmd/order.py (on tests/test_order.py's stand-in engine), the
# 30 days, /v1/status, and X-Forwarded-For that a client writes.
import ipaddress
import json
import re
import os
import socket
import threading
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from scripts import serve_admin as A
from scripts import serve_net as N
from scripts import serve_quota as Q
from scripts import serve_tiers as T
from sclmd import order as O
from test_quota import Clock, done, req_of
from test_serve_admin import HI, answer, call, chat_async
from test_serve_net import headers as hdrs_of

KEYS = [("alice", b"sk-alice-0123456789"), ("bob", b"sk-bob-0123456789ab")]
ALICE = {"Authorization": "Bearer sk-alice-0123456789"}
HOME = "121.134.0.1"  # in the test's book: a Korean home ISP's
HOME2 = "121.134.0.2"
PHONE6 = "2001:2d8:1:2::9"  # a Korean mobile ISP's IPv6
CLOUD = "3.34.0.1"  # in Korea, a cloud's
ABROAD = "8.8.8.8"
IPDB = "C:/models/sclmd-ipdb"


class Book:
    """In the AddressBook's place: kind(ip) from the test's table (abroad when it is not there)."""

    TABLE = {HOME: "kr_consumer", HOME2: "kr_consumer", PHONE6: "kr_consumer", CLOUD: "kr_other", ABROAD: "abroad"}

    def __init__(self, folder=None, asns=None):
        self.asns = frozenset(T.KR_CONSUMER)

    def kind(self, ip) -> str:
        if ip is None or not ip.is_global:
            return "unknown"
        return self.TABLE.get(str(ip), "abroad")

    def about(self) -> dict:
        return {"country": "test", "asn": "test", "asns": len(self.asns)}


def who(peer, *pairs, keys=KEYS, proxies=("127.0.0.1",)):
    return N.who(peer, hdrs_of(*pairs), keys, True, N.networks(proxies), A.key_name)


def tiers(rules=None, quota=None, clock=None):
    q = quota or Q.Quota(clock=clock or Clock(), verified=T.VERIFIED, output=5.0)
    return T.Tiers(Book(), q, rules), q


def decide(t, w, prompt: int = 100):
    req = req_of(w.tenant, prompt)
    req.max_new = 10
    tier = t.decide(req, w)
    req.tier = tier
    return req, tier


# the tier of a request

def test_the_tier_is_the_key_this_pc_or_the_address_kind():
    t, _ = tiers()
    assert decide(t, who("127.0.0.1", ("X-Forwarded-For", ABROAD), ("Authorization", "Bearer sk-alice-0123456789")))[1] == "key"
    assert decide(t, who("127.0.0.1"))[1] == "local"
    assert decide(t, who("127.0.0.1", ("X-Forwarded-For", HOME)))[1] == "ip_verified"
    w = who("127.0.0.1", ("X-Forwarded-For", PHONE6))
    assert w.tenant == "ip:2001:2d8:1:2::/64" and decide(t, w)[1] == "ip_verified"  # IPv6: the address's kind, the /64's tenant
    for addr in (CLOUD, ABROAD, "10.0.0.7", "garbage"):  # a cloud in Korea, abroad, a private address, not an address (the proxy's own)
        assert decide(t, who("127.0.0.1", ("X-Forwarded-For", addr)))[1] == "ip_unverified", addr
    assert t.n["tier_ip_verified"] == 2 and t.n["tier_key"] == 1 and t.n["tier_local"] == 1
    assert decide(t, N.who("127.0.0.1", hdrs_of(), (), False))[1] is None  # tenancy off: no tier


def test_x_forwarded_for_a_client_writes_does_not_change_its_tier():
    t, _ = tiers()
    # through the trusted proxy: the proxy appends the address it saw; whatever the client wrote to the left is not read
    w = who("127.0.0.1", ("X-Forwarded-For", f"{HOME}, {ABROAD}"))
    assert w.tenant == f"ip:{ABROAD}" and decide(t, w)[1] == "ip_unverified"
    # straight from the internet (not a trusted proxy): the header is not read; the connection's address is the user
    w = who(ABROAD, ("X-Forwarded-For", HOME), proxies=("192.168.0.40",))
    assert w.tenant == f"ip:{ABROAD}" and decide(t, w)[1] == "ip_unverified"
    w = who(CLOUD, ("X-Forwarded-For", HOME), proxies=("192.168.0.40",))
    assert decide(t, w)[1] == "ip_unverified"
    # a proxy that is not trusted, on a Korean home address: it stands for many users (Who.unsure), never verified
    users = N.Users(KEYS, True, (), name_of=A.key_name)
    w = users.who(SimpleNamespace(client_address=(HOME, 1), headers=hdrs_of(("X-Forwarded-For", ABROAD))))
    assert w.unsure and w.tenant == f"ip:{HOME}" and decide(t, w)[1] == "ip_unverified"
    assert t.tier_of(w) == "ip_unverified"


@pytest.mark.skipif(not os.path.isdir(IPDB), reason="the DB-IP Lite files are not in " + IPDB)
def test_the_real_files_place_korean_home_and_mobile_isps():
    book = T.AddressBook(IPDB)
    kinds = {s: book.kind(ipaddress.ip_address(s)) for s in ("39.7.0.1", "121.134.0.1", "223.38.0.1", "106.101.0.1", "2001:2d8::1", "2001:e60::1",
                                                               "3.34.0.1", "110.93.150.1", "8.8.8.8", "192.168.0.40", "127.0.0.1")}
    assert [kinds[s] for s in ("39.7.0.1", "121.134.0.1", "223.38.0.1", "106.101.0.1", "2001:2d8::1", "2001:e60::1")] == ["kr_consumer"] * 6
    assert kinds["3.34.0.1"] == kinds["110.93.150.1"] == "kr_other"  # AWS Seoul, NAVER Cloud
    assert kinds["8.8.8.8"] == "abroad" and kinds["192.168.0.40"] == kinds["127.0.0.1"] == "unknown"
    assert T.AddressBook(IPDB, asns={1}).kind(ipaddress.ip_address("39.7.0.1")) == "kr_other"  # --kr-consumer-asns replaces the list


def test_a_newer_month_of_the_files_is_read(tmp_path):
    for name in ("dbip-country-lite-2026-10.mmdb", "dbip-country-lite-2026-11.mmdb", "dbip-asn-lite-2026-11.mmdb", "other.mmdb"):
        (tmp_path / name).write_bytes(b"")
    assert T.AddressBook.newest(str(tmp_path), "country").endswith("2026-11.mmdb")
    (tmp_path / "dbip-asn-lite-2026-11.mmdb").unlink()
    with pytest.raises(SystemExit):
        T.AddressBook.newest(str(tmp_path), "asn")


# behaviour

def spend(q, tenant, prompt, out=0, tier=None, req=None):
    r = req or req_of(tenant, prompt)
    r.tier = tier
    q.taken(r)
    r.job = SimpleNamespace(out=[1] * out, seq=0)
    q.settle(r, done(prompt, 0, out))
    return r


def test_a_verified_address_is_demoted_by_its_rate_hour_and_sizes():
    clock = Clock()
    rules = {"rpm": 5, "rph": 12, "tph": 10_000, "most": 2_000, "outside": 0.5, "window": 10}
    t, q = tiers(rules, clock=clock)
    w = who("127.0.0.1", ("X-Forwarded-For", HOME))
    tenant = w.tenant
    for i in range(5):
        req, tier = decide(t, w)
        assert tier == "ip_verified", i
        spend(q, tenant, 100, tier=tier, req=req)
        t.taken(req)
    assert decide(t, w)[1] == "ip_unverified" and t.n["demoted_rate"] == 1  # a 6th within the minute: past 5
    clock.t += 120  # the minute buckets have gone; the hour has 6 requests
    assert decide(t, w)[1] == "ip_verified"
    for _ in range(7):
        spend(q, tenant, 10, tier="ip_verified")
        clock.t += 61
    assert decide(t, w)[1] == "ip_unverified" and t.n["demoted_requests"] == 1  # 12 in the hour already: this would be the 13th
    clock.t += 3700
    spend(q, tenant, 2_100, tier="ip_verified")  # 2,100 tokens
    spend(q, tenant, 0, out=1_600, tier="ip_verified")  # + 1,600 x 5 = 10,100: at 10,000 already
    clock.t += 61
    assert decide(t, w)[1] == "ip_unverified" and t.n["demoted_tokens"] == 1
    # sizes: of its last 10 prompts more than half above 2,000 tokens
    clock.t += 3700
    w2 = who("127.0.0.1", ("X-Forwarded-For", HOME2))
    for p in (100, 100, 100, 3000, 3000, 3000, 3000):
        req, tier = decide(t, w2, p)
        assert tier == "ip_verified"  # (at most 3 of the 6 before it outside)
        t.taken(req)
        clock.t += 61
    assert decide(t, w2)[1] == "ip_unverified" and t.n["demoted_size"] == 1  # 4 of 7 outside: more than half
    for _ in range(2):  # two more short ones: 4 of 9 is not more than half
        r, _ = decide(t, w2, 100)
        t.taken(r)
    assert decide(t, w2)[1] == "ip_verified"


def test_the_spread_condition_and_a_growing_conversation():
    """The spread condition (off by default) on a chat that grows a turn at a time: its 90th percentile over its 10th
    passes 4 by its 6th turn (measured: 2 and 4 by the 6th, 8 by the 8th, 16 never in 40 turns), so 4 would demote an ordinary long chat; off it never does."""
    clock = Clock()
    grows = [300 + 450 * i for i in range(20)]  # a conversation of 20 turns, 450 tokens more each (a question and its reply)
    for spread, worst in ((4.0, "ip_unverified"), (0.0, "ip_verified")):
        t, _ = tiers({"spread": spread, "rpm": 0, "rph": 0, "tph": 0}, clock=clock)
        w = who("127.0.0.1", ("X-Forwarded-For", HOME))
        seen = []
        for p in grows:
            req, tier = decide(t, w, p)
            seen.append(tier)
            t.taken(req)
        assert seen[-1] == worst and seen[:5] == ["ip_verified"] * 5, (spread, seen)
    t, _ = tiers({"rpm": 0, "rph": 0, "tph": 0}, clock=clock)  # the defaults' range (0 .. 16,384) keeps it verified to 20 turns
    w = who("127.0.0.1", ("X-Forwarded-For", HOME))
    tiers_seen = []
    for p in grows:
        req, tier = decide(t, w, p)
        tiers_seen.append(tier)
        t.taken(req)
    assert set(tiers_seen) == {"ip_verified"}


def test_the_sizes_kept_are_bounded_and_only_of_addresses_that_could_be_verified():
    t, _ = tiers()
    t.history = 50
    for i in range(80):
        req, _ = decide(t, who("127.0.0.1", ("X-Forwarded-For", f"121.134.1.{i}")))
        t.taken(req)
    Book.TABLE.update({f"121.134.1.{i}": "kr_consumer" for i in range(80)})
    try:
        for i in range(80):
            req, _ = decide(t, who("127.0.0.1", ("X-Forwarded-For", f"121.134.1.{i}")))
            t.taken(req)
    finally:
        for i in range(80):
            Book.TABLE.pop(f"121.134.1.{i}", None)
    assert len(t.sizes) == 50 and "ip:121.134.1.79" in t.sizes and "ip:121.134.1.0" not in t.sizes


# limits

def test_each_tier_has_its_limits_and_a_reply_counts_five_times():
    clock = Clock()
    q = Q.Quota((1_250_000, 600, 40_000_000), (300_000, 120, 7_500_000), clock=clock, verified=(750_000, 120, 15_000_000), output=5.0)
    assert (q.limits_of("key:alice").tph, q.limits_of("key:alice").tpd) == (1_250_000, 40_000_000)
    assert (q.limits_of("ip:203.0.113.7", "ip_verified").tph, q.limits_of("ip:203.0.113.7", "ip_verified").tpd) == (750_000, 15_000_000)
    assert (q.limits_of("ip:203.0.113.7", "ip_unverified").tph, q.limits_of("ip:203.0.113.7").tpd) == (300_000, 7_500_000)
    assert q.limits_of("ip:203.0.113.7", "ip_verified").rph == q.limits_of("ip:203.0.113.7").rph == 120 and q.limits_of("key:a").rph == 600
    r = req_of("key:alice", 1000)
    r.max_new = 100
    q.taken(r)
    assert q.held["key:alice"] == 500  # the reply held at its max_tokens x 5
    r.job, r.reused = SimpleNamespace(out=[1] * 40, seq=0), 0
    q.settle(r, done(1000, 600, 40))
    u = q.usage_of("key:alice", "key")
    assert u["hour"]["tokens"] == 400 + 40 * 5 + 60 and u["tier"] == "key" and "reply x 5" in u["tokens_are"]
    left = req_of("key:alice", 10)  # its client left after 7 tokens
    q.taken(left)
    left.job = SimpleNamespace(out=[1] * 7, seq=0)
    q.settle(left, None)
    assert q.usage_of("key:alice")["hour"]["tokens"] == 660 + 10 + 35
    # the daily check counts a request's max_tokens x 5 (admit), so the same prompt is refused sooner
    q2 = Q.Quota((10**9, 10**6, 1_000), clock=clock, output=5.0)
    adm = SimpleNamespace(quota=q2)
    big = req_of("key:bob", 100)
    big.max_new = 200  # 100 + 1,000 > 1,000
    assert Q.admit(adm, big).kind == "tokens_per_day"
    big.max_new = 180  # 100 + 900
    assert Q.admit(adm, big) is None
    # a verified address's day is 15,000,000, an unverified one's 7,500,000: the same use refused only for the second
    q3 = Q.Quota((1_250_000, 600, 40_000_000), (300_000, 120, 7_500_000), clock=clock, verified=(750_000, 120, 15_000_000), output=5.0)
    spend(q3, "ip:203.0.113.9", 8_000_000, tier="ip_verified")
    assert q3.refusal("ip:203.0.113.9", 10, tier="ip_verified") is None
    assert q3.refusal("ip:203.0.113.9", 10, tier="ip_unverified").kind == "tokens_per_day"


def test_without_the_new_options_the_quota_is_n2s():
    q = Q.Quota()
    assert (q.key.tpd, q.ip.tpd, q.output, q.verified) == (12_000_000, 2_400_000, 1.0, None)
    assert q.limits_of("ip:1.2.3.4", "ip_verified") is q.ip
    assert q.tokens_are() == "the prompt that was not cached + the reply + the cached prompt x 0.1"
    assert "tier" not in q.usage_of("key:a")


# the order of the tiers (sclmd/order.py)

def test_the_classes_go_tier_then_quota_then_band():
    o = O.Order(ladder=T.BANDS)
    assert o.first == 3 and o.tier_steps == 15
    key, ver, unv = (o.class_of(t, False) for t in O.TIERS)
    assert O.LOCAL < key < ver < unv and (key, ver, unv) == (3, 18, 33)
    # this PC in its low band is ahead of any key; a key over quota in its low band ahead of a verified address in quota
    assert O.LOCAL + o.band(11) < key and o.class_of("key", True) + o.band(11) < ver
    assert o.class_of("ip_verified", True) + o.band(11) < unv and o.class_of("ip_unverified", True) == 45
    assert [o.band(i) for i in range(14)] == [0, 1, 1, 1] + [2] * 10
    assert o.level(o.class_of("ip_unverified", True), -10**6, unv) == o.first  # aging stops at the first base (never past this PC)
    n2 = O.Order()  # no ladder, no tiers: N2's numbers
    assert (n2.first, n2.over, n2.class_of(None, False), n2.class_of(None, True), n2.band(5)) == (1, 13, 1, 13, 5)


def test_the_waiting_list_and_passes_follow_the_tiers_on_the_stand_in_engine():
    from test_order import Tick, ask, finish, ids, server

    tier = {"u": "ip_unverified", "v": "ip_verified", "k": "key"}
    order = O.Order(klass=None, clock=Tick(), ladder=T.BANDS)
    order.klass = lambda req: O.LOCAL if req.local else order.class_of(tier[req.tenant], False)
    srv = server(sequences=1, order=order)  # one sequence: they are served one after another, in the order of the waiting list
    blocker = ask(srv, ids(6, 99), 4, 99, tenant="u")
    srv.turn()  # placed: the others wait behind it
    reqs = [ask(srv, ids(6, i), 3, i, tenant=t) for i, t in enumerate(("u", "v", "k"))] + [ask(srv, ids(6, 7), 3, 7, local=True, tenant="local")]
    srv.turn()
    assert [round(r.priority) for r in reqs] == [33 + 1, 18, 3, 0]  # u's second request is in its normal band
    firsts = []
    for _ in range(400):
        srv.turn()
        for r in reqs:
            if r.job is not None and r not in firsts:
                firsts.append(r)
        if len(firsts) == 4:
            break
    finish(srv, blocker, *reqs)
    assert [r.tenant for r in firsts] == ["local", "k", "v", "u"]


# the server (the real App and Handler; the test answers in the engine's place)

def stand_in(*argv, keys=KEYS, limits=None, max_tokens: int = 64, monkeypatch=None):
    from test_quota import stand_in as quota_stand_in

    return quota_stand_in(*argv, keys=keys, limits=limits, max_tokens=max_tokens)


PUBLIC = ("--quotas", "--priorities", "--anonymous", "--trusted-proxies", "127.0.0.1", "--tiers", "--ipdb", "unused", "--output-weight", "5",
          "--tph", "1250000", "--tpd", "40000000", "--ip-tph", "300000", "--ip-tpd", "7500000", "--concurrency", "1,3,8", "--public-status")


@pytest.fixture()
def net(monkeypatch):
    monkeypatch.setattr(T, "AddressBook", Book)
    made = []

    def make(*argv, **kw):
        made.append(stand_in(*argv, **kw))
        return made[-1]

    yield make
    for httpd, _, _ in made:
        httpd.shutdown()
        httpd.server_close()


def served(httpd, inbox, body=None, hdrs=None, out_tokens=(10, 11, 12)):
    th, out = chat_async(httpd, body or {"messages": HI}, hdrs)
    req = inbox.next()
    req.job, req.reused = SimpleNamespace(out=list(out_tokens), seq=0), 0
    answer(req, out_tokens, {"finish_reason": "stop", "ended": "stop", "timings": {"total_ms": 5, "first_token_ms": 1, "tokens_per_second": 1.0},
                             "usage": {"prompt_tokens": len(req.full), "completion_tokens": len(out_tokens), "total_tokens": len(req.full) + 3,
                                       "prompt_tokens_details": {"cached_tokens": 0}}})
    th.join(10)
    return req, out


def test_a_requests_tier_sets_its_headers_priority_and_log_line(net, tmp_path):
    log = tmp_path / "requests.jsonl"
    httpd, app, inbox = net(*PUBLIC, "--request-log", str(log))
    req, out = served(httpd, inbox, hdrs={"X-Forwarded-For": HOME})
    assert req.tier == "ip_verified" and out["headers"]["x-ratelimit-limit-tokens"] == "750000" and req.priority == 18
    req, out = served(httpd, inbox, hdrs={"X-Forwarded-For": f"{HOME}, {ABROAD}"})  # the client's own header to the left is not read
    assert req.tier == "ip_unverified" and out["headers"]["x-ratelimit-limit-tokens"] == "300000" and req.priority == 33
    req, out = served(httpd, inbox, hdrs=ALICE)
    assert req.tier == "key" and out["headers"]["x-ratelimit-limit-tokens"] == "1250000" and req.priority == 3
    req, out = served(httpd, inbox)
    assert req.tier == "local" and req.priority == 0 and not any(k.startswith("x-ratelimit") for k in out["headers"])
    u = json.loads(call(httpd, "GET", "/v1/usage", headers={"X-Forwarded-For": HOME})[2])
    assert u["tier"] == "ip_verified" and u["hour"]["limit_tokens"] == 750_000 and u["day"]["limit_tokens"] == 15_000_000
    assert u["hour"]["requests"] == 1 and u["hour"]["tokens"] > 3 * 5  # its prompt + 3 reply tokens x 5
    time.sleep(0.3)
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert [x.get("tier") for x in lines if x.get("status") == 200] == ["ip_verified", "ip_unverified", "key", "local"]
    text = call(httpd, "GET", "/metrics")[2].decode()
    assert 'sclmd_tier_taken_total{tier="ip_verified"} 1' in text and HOME not in text and "alice" not in text


def raw_post(port: int, headers: dict, body: bytes | None = None, length: int | None = None) -> socket.socket:
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    data = body if body is not None else json.dumps({"messages": HI}).encode()
    h = {"Host": "x", "Content-Type": "application/json", "Content-Length": str(len(data) if length is None else length), **headers}
    s.sendall(("POST /v1/chat/completions HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in h.items()) + "\r\n").encode() + data)
    return s


def read_answer(s: socket.socket, wait: float = 5.0) -> bytes:
    got, t = b"", time.time()
    s.settimeout(0.5)
    while time.time() - t < wait:
        try:
            chunk = s.recv(65536)
        except socket.timeout:
            if got:
                break
            continue
        if not chunk:
            break
        got += chunk
    return got


def wait_for(cond, seconds: float = 10.0):
    t = time.time()
    while not cond():
        assert time.time() - t < seconds, "timed out"
        time.sleep(0.01)


def test_the_thirteenth_request_at_once_is_429_before_its_body_and_twenty_at_once_take_exactly_twelve(net):
    httpd, app, inbox = net(*PUBLIC)
    slots, port = app.admin.slots, httpd.server_address[1]
    me = {"X-Forwarded-For": HOME}
    socks = [raw_post(port, me) for _ in range(20)]  # twenty at once
    wait_for(lambda: len(inbox.reqs) >= 12)
    time.sleep(0.3)
    answers = [read_answer(s, 1.0) for s in socks]
    refused = [a for a in answers if a.startswith(b"HTTP/1.1 429")]
    assert len(inbox.reqs) == 12 and len(refused) == 8 and slots.held == {"ip:121.134.0.1": 12}
    body = json.loads(refused[0].split(b"\r\n\r\n", 1)[1])
    assert body["error"]["code"] == "concurrency_limit" and b"Retry-After: 2" in refused[0]
    # before its body: a 13th that announces a body it never sends is answered at once
    s = raw_post(port, me, body=b"{\"mess", length=1_000_000)
    t = time.time()
    got = read_answer(s, 3.0)
    assert got.startswith(b"HTTP/1.1 429") and time.time() - t < 3 and b"concurrency_limit" in got
    s.close()
    # another user, and this PC, are not held back
    th, out = chat_async(httpd, {"messages": HI}, {"X-Forwarded-For": HOME2})
    wait_for(lambda: len(inbox.reqs) >= 13)
    th2, out2 = chat_async(httpd, {"messages": HI})
    wait_for(lambda: len(inbox.reqs) >= 14)
    for r in list(inbox.reqs):
        answer(r)
    inbox.reqs.clear()
    th.join(10)
    th2.join(10)
    for s in socks:
        read_answer(s, 1.0)
        s.close()
    assert out["status"] == out2["status"] == 200
    wait_for(lambda: slots.held == {})
    assert slots.refused == 9


def test_every_way_a_request_ends_gives_its_place_back(net):
    httpd, app, inbox = net(*PUBLIC, "--queue-timeout", "0.5")
    app.max_body = 1 << 20  # (--max-body-mb 1: scripts/serve_contract.py's)
    slots, port = app.admin.slots, httpd.server_address[1]
    me = {"X-Forwarded-For": HOME}
    st, _, _ = call(httpd, "POST", "/v1/chat/completions", {"messages": "not a list"}, me)  # 400
    assert st == 400 and slots.held == {}
    s = raw_post(port, me, body=b"x", length=2 << 20)  # 413
    assert read_answer(s).startswith(b"HTTP/1.1 413")
    s.close()
    wait_for(lambda: slots.held == {})
    s = raw_post(port, me, body=b"{\"messages\": [", length=500)  # the client leaves in the middle of its body
    time.sleep(0.2)
    s.close()
    wait_for(lambda: slots.held == {})
    st, _, _ = call(httpd, "POST", "/v1/chat/completions", {"messages": HI}, me)  # nobody places it: 503 after the queue's timeout
    assert st == 503
    inbox.reqs.clear()
    wait_for(lambda: slots.held == {})
    s = raw_post(port, me, body=json.dumps({"messages": HI, "stream": True}).encode())  # a stream its client leaves
    req = inbox.next()
    req.job, req.reused = SimpleNamespace(out=[10], seq=0), 0
    req.events.put(("token", 10))
    time.sleep(0.2)
    s.close()
    time.sleep(0.2)
    answer(req, (11, 12))
    wait_for(lambda: slots.held == {})
    req, out = served(httpd, inbox, hdrs=me)  # and a plain 200
    assert out["status"] == 200 and slots.held == {}


def test_v1_status_is_open_and_says_what_the_contract_says_and_no_more(net):
    httpd, app, inbox = net(*PUBLIC, "--require-key")
    st, h, body = call(httpd, "GET", "/v1/status")
    s = json.loads(body)
    assert st == 200 and set(s) == {"ok", "model", "version", "uptime_s", "queue", "zdr", "limits", "caller"}
    assert s["ok"] is True and isinstance(s["version"], str) and isinstance(s["uptime_s"], int) and s["queue"] == {"under_way": 0, "waiting": 0}
    assert set(s["zdr"]) == {"prompt_logging", "cache_ram_minutes", "cache_disk_hours", "disk_encryption", "shared_between_users"}
    assert s["zdr"]["prompt_logging"] is False
    assert s["limits"] == {"ip_unverified": {"tph": 300000, "tpd": 7500000}, "ip_verified": {"tph": 750000, "tpd": 15000000},
                           "key": {"tph": 1250000, "tpd": 40000000}, "output_weight": 5, "cached_weight": 0.1,
                           "concurrency": {"priority": 1, "normal": 3, "low": 8}}
    assert s["caller"] == {"tier": "local"}
    assert json.loads(call(httpd, "GET", "/v1/status", headers={"X-Forwarded-For": HOME})[2])["caller"] == {"tier": "ip_verified"}
    assert json.loads(call(httpd, "GET", "/v1/status", headers={"X-Forwarded-For": ABROAD})[2])["caller"] == {"tier": "ip_unverified"}
    assert json.loads(call(httpd, "GET", "/v1/status", headers=ALICE)[2])["caller"] == {"tier": "key"}
    st, _, body = call(httpd, "GET", "/v1/status", headers={"Authorization": "Bearer sk-not-ours", "X-Forwarded-For": HOME})
    assert st == 200 and json.loads(body)["caller"] == {"tier": "ip_verified"}  # open as /health is: an unknown key is judged by its address
    assert call(httpd, "GET", "/v1/models", headers={"Authorization": "Bearer sk-not-ours"})[0] == 401  # (the API is not)
    # no threshold, no user: the behaviour defaults appear nowhere in it, and no other user's address or key name
    served(httpd, inbox, hdrs={"X-Forwarded-For": HOME2})
    text = call(httpd, "GET", "/v1/status", headers={"X-Forwarded-For": ABROAD})[2].decode()
    numbers = set(re.findall(r"\d+(?:\.\d+)?", text))
    for secret in ("16384", "240", "1500000", "30", "0.5", "20"):  # the behaviour defaults
        assert secret not in numbers, secret
    for secret in ("alice", "bob", HOME2, "rpm", "verify", "spread"):
        assert secret not in text, secret


def test_without_the_options_v1_status_is_unknown_and_nothing_is_new(net):
    httpd, app, inbox = net("--quotas", "--priorities", "--anonymous")
    assert call(httpd, "GET", "/v1/status")[0] == 404
    assert app.admin.tiers is None and app.admin.slots is None and app.admin.status_api is None
    req, out = served(httpd, inbox, hdrs=ALICE)
    assert req.tier is None and req.priority == 1 and out["headers"]["x-ratelimit-limit-tokens"] == "500000"
    text = call(httpd, "GET", "/metrics")[2].decode()
    assert "sclmd_tier" not in text and "concurrency" not in text
    httpd, app, inbox = net(keys=())  # no option and no key at all
    assert call(httpd, "GET", "/v1/status")[0] == 404


def test_the_options_are_checked():
    from test_serve_admin import options

    with pytest.raises(SystemExit):
        T.check(options("--tiers"))  # no --quotas
    with pytest.raises(SystemExit):
        T.check(options("--tiers", "--quotas"))  # no --ipdb
    with pytest.raises(SystemExit):
        T.check(options("--public-status"))
    for bad in ("1,3", "a,b,c", "0,0,0", "-1,3,8"):
        with pytest.raises(SystemExit):
            T.check(options("--concurrency", bad))
    T.check(options("--concurrency", "1,3,8"))


# the request log's 30 days

def test_lines_older_than_the_days_are_removed_at_start_and_every_hour(tmp_path):
    path = tmp_path / "requests.jsonl"
    now = time.time()

    def line(days_ago: float) -> str:
        t = datetime.fromtimestamp(now - days_ago * 86400).astimezone().isoformat(timespec="milliseconds")
        return json.dumps({"t": t, "status": 200}) + "\n"

    path.write_text(line(40) + line(31) + "not json\n" + line(29) + line(1), encoding="utf-8")
    old = tmp_path / "requests.jsonl.1"
    old.write_text(line(45) + line(35), encoding="utf-8")
    os.utime(old, (now - 35 * 86400, now - 35 * 86400))  # its last write: 35 days ago
    newer = tmp_path / "requests.jsonl.2"
    newer.write_text(line(10), encoding="utf-8")
    clock = [now]
    log = A.RequestLog(str(path), days=30, clock=lambda: clock[0])
    wait_for(lambda: log.pruned_at is not None)
    log.write({"t": A.now_iso(), "status": 200})
    wait_for(lambda: log.written == 1)
    kept = path.read_text(encoding="utf-8").splitlines()
    assert len(kept) == 3 and [json.loads(x)["status"] for x in kept] == [200, 200, 200] and not old.exists() and newer.exists()
    assert log.pruned == 3 + 2
    # an hour on, a day older: what was 29 days old goes
    clock[0] = now + 1.5 * 86400
    log.pruned_at = clock[0] - A.PRUNE_S - 1
    log.write({"t": A.now_iso(), "status": 201})
    wait_for(lambda: log.written == 2)
    kept = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [x["status"] for x in kept] == [200, 200, 201] and log.pruned == 6
    assert datetime.fromisoformat(kept[0]["t"]).timestamp() > now - 2 * 86400


def test_without_days_the_log_is_kept_as_before(tmp_path):
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps({"t": "2000-01-01T00:00:00.000+09:00"}) + "\n", encoding="utf-8")
    log = A.RequestLog(str(path))
    log.write({"t": A.now_iso()})
    wait_for(lambda: log.written == 1)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2 and log.pruned_at is None


def test_the_public_example_turns_them_on_with_the_owners_numbers():
    from scripts import serve as S

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "serve.public.example.toml")
    cli = A.parse_args(S.parser(), ["--config", path])
    assert (cli.tiers, cli.ipdb, cli.output_weight, cli.concurrency, cli.public_status, cli.request_log_days) == (
        True, "C:/models/sclmd-ipdb", 5.0, "1,3,8", True, 30.0)
    assert (cli.tph, cli.tpd, cli.verified_tph, cli.verified_tpd, cli.ip_tph, cli.ip_tpd) == (1_250_000, 40_000_000, 750_000, 15_000_000, 300_000, 7_500_000)
    assert (cli.rph, cli.ip_rph, cli.verified_rph, cli.keyless_day) == (600, 120, 120, 20_000_000)
    if os.path.isdir(IPDB):
        T.check(cli)


def test_v1_status_zdr_says_what_the_cache_does(net):
    httpd, app, inbox = net(*PUBLIC)
    app.admin.users.per_tenant = True  # (scripts/serve_cache.py's install with --prompt-cache)
    inbox.cache = SimpleNamespace(disk=object())  # --cache-dir: the hour's entries on disk, AES-GCM
    z = json.loads(call(httpd, "GET", "/v1/status")[2])["zdr"]
    assert z == {"prompt_logging": False, "cache_ram_minutes": 15, "cache_disk_hours": 1, "disk_encryption": "AES-GCM", "shared_between_users": False}
    inbox.cache = SimpleNamespace(disk=None)  # no --cache-dir
    z = json.loads(call(httpd, "GET", "/v1/status")[2])["zdr"]
    assert z == {"prompt_logging": False, "cache_ram_minutes": 15, "cache_disk_hours": 0, "disk_encryption": "none", "shared_between_users": False}
    del inbox.cache  # no --prompt-cache: kept conversations are shared, for as long as they are kept
    z = json.loads(call(httpd, "GET", "/v1/status")[2])["zdr"]
    assert z == {"prompt_logging": False, "cache_ram_minutes": None, "cache_disk_hours": 0, "disk_encryption": "none", "shared_between_users": True}


# fixes at merge (verify-P1-tiers)

def test_idc_prefixes_inside_a_carrier_asn_and_v4_compatible_v6_are_never_verified():
    from scripts import serve_tiers as T
    book = object.__new__(T.AddressBook)
    book.asns = frozenset({4766})
    book.denied = (ipaddress.ip_network("211.252.122.0/24"),)
    book.where = lambda ip: ("KR", 4766)
    assert book.kind(ipaddress.ip_address("121.134.0.7")) == "kr_consumer"
    assert book.kind(ipaddress.ip_address("211.252.122.215")) == "kr_other"  # a KT Cloud VPS in AS4766
    assert book.kind(ipaddress.ip_address("::121.134.0.7")) == "unknown"  # IPv4-compatible IPv6: not a real source
    assert book.kind(ipaddress.ip_address("2001:e60::1")) == "kr_consumer"


def test_a_body_on_its_way_holds_no_place():
    """The places are taken once a body has come: headers alone (a slow sender) never hold one."""
    from scripts import serve_tiers as T
    slots = T.Slots((1, 3, 8))
    assert not slots.full("ip:203.0.113.5")
    for _ in range(12):
        assert slots.take("ip:203.0.113.5")
    assert slots.full("ip:203.0.113.5") and not slots.take("ip:203.0.113.5")
    slots.give("ip:203.0.113.5")
    assert not slots.full("ip:203.0.113.5")


def test_the_behaviour_thresholds_are_not_in_the_repository():
    """6.20: the owner's thresholds live in serve.toml only; the code's own are off."""
    from scripts import serve_tiers as T
    assert all(T.RULES[k] == 0 for k in ("rpm", "rph", "tph", "least", "most"))
