# Quotas and priorities of the internet server (design.md 6.16 "할당량" and "우선순위", 6.18; N2). scripts/serve_admin.py
# calls in through a few hooks, and sclmd/order.py orders the engine's work. Off unless --quotas or --priorities: without
# them nothing here runs, and no answer, log line or pass changes.
#
# --quotas (with tenancy: API keys or --anonymous; scripts/serve_net.py says who a request is):
# - each tenant (key:NAME, ip:ADDRESS, ip:PREFIX/64; "local", this PC, is exempt and not counted) has tokens an hour
#   (TPH) and requests an hour (RPH) over the last 60 minutes, rolling. A request's tokens: its prompt that was not
#   cached + its reply + its cached prompt x 0.1. A request that comes when its tenant's hour is at either is over
#   quota for the rest of its life: still served, at a much lower priority (with --priorities: sclmd/order.py).
# - the one refusal: the daily limit over the last 24 hours (--daily-times x TPH and RPH), and --keyless-day tokens for
#   all requests without a key together: 429 with Retry-After (until enough of the day has gone by), OpenAI's
#   rate_limit_exceeded. A tenant already at a limit is refused before its body is read; one whose prompt would pass
#   it, after the prompt is known.
# - counting: a request is counted when it is taken in, with its whole prompt, and its reply is held at its max_tokens
#   for the daily check (Quota.held) until it is settled, so that requests that come at once cannot all pass the daily
#   check, by their prompts or by their replies (the hourly class and the headers count only what was counted, not
#   what is held); when its HTTP thread is done, with what it cost (the cached part at 0.1, its
#   reply so far), or with nothing when it never reached the engine (a 503 of --queue-timeout, a client that left while
#   it waited). A 429, 401 or 400 costs nothing. Minute buckets by the wall clock: a minute counts while any of it is
#   inside the window.
# - defaults (6.16): a key 500,000 TPH and 600 RPH, an address without a key 100,000 and 120, the day 24 times that,
#   all requests without a key 20,000,000 tokens a day. A key's own, after it in --api-key-file (or SCLMD_API_KEY):
#   "alice: sk-... tph=1000000 rph=1200 tpd=0" (tph, rph, tpd, rpd; 0: no limit of that kind).
# - every answer to a tenant with limits carries OpenAI's x-ratelimit-{limit,remaining,reset}-{requests,tokens} (the
#   hour's; reset: until every use counted now has left the window), the 429 too; GET /v1/usage gives the caller its
#   own usage and limits (the hour and the day).
# - --quota-file: the counters (tenant ids and numbers: metadata only) written atomically every 10 s and at exit, and
#   read back at start. Buckets that have left the day are dropped every 10 s, with or without the file.
# --priorities: sclmd/order.py on the server's waiting list, passes and make room. A request's class is this PC, in
#   quota or over quota (--quotas), its priority a step lower for each earlier request of its tenant under way.
# Nothing here imports torch.
import atexit
import collections
import json
import math
import os
import threading
import time

from sclmd import order as O
from scripts import serve_contract as C
from scripts import serve_net as N

HOUR, DAY = 3600.0, 86400.0
BUCKET = 60.0  # seconds a bucket of the counters covers
SAVE_S = 10.0  # the counters are written this often (when they changed)
CACHED = 0.1  # what a cached prompt token counts for
KEY = (500_000, 600)  # a key's tokens and requests an hour (6.16)
IP = (100_000, 120)  # an address without a key
DAILY = 24.0  # the daily limit: this many times the hour's
KEYLESS_DAY = 20_000_000  # tokens a day of all requests without a key together


def duration(s: float) -> str:
    """Seconds as OpenAI writes a reset time: 1s, 6m0s, 1h2m3s."""
    total = max(0, math.ceil(s))
    h, rest = divmod(total, 3600)
    m, sec = divmod(rest, 60)
    return (f"{h}h" if h else "") + (f"{m}m" if h or m else "") + f"{sec}s"


class Limits:
    __slots__ = ("tph", "rph", "tpd", "rpd")

    def __init__(self, tph: int, rph: int, tpd: int, rpd: int):
        self.tph, self.rph, self.tpd, self.rpd = int(tph), int(rph), int(tpd), int(rpd)

    def __repr__(self) -> str:
        return f"Limits(tph={self.tph}, rph={self.rph}, tpd={self.tpd}, rpd={self.rpd})"


class Refusal:
    """A request refused at a daily limit: which, the limit, what was used, what it asked, seconds until it fits."""

    WHAT = {"requests_per_day": ("requests", "requests per day (RPD)"), "tokens_per_day": ("tokens", "tokens per day (TPD)"),
            "keyless_tokens_per_day": ("tokens", "tokens per day of all requests without an API key")}

    def __init__(self, kind: str, tenant: str, limit: int, used: float, asked: float, retry_s: float):
        self.kind, self.tenant, self.limit, self.used, self.asked, self.retry_s = kind, tenant, limit, used, asked, max(retry_s, 1.0)

    def body(self) -> dict:
        kind, what = self.WHAT[self.kind]
        who = "requests without an API key" if self.kind == "keyless_tokens_per_day" else self.tenant
        return {"error": {"message": f"Rate limit reached for {who} on {what}: Limit {self.limit}, Used {int(self.used)}, Requested "
                                     f"{int(math.ceil(self.asked))}. Please try again in {duration(self.retry_s)}.",
                          "type": kind, "param": None, "code": "rate_limit_exceeded"}}

    def headers(self) -> dict:
        return {"Retry-After": str(math.ceil(self.retry_s)), "retry-after-ms": str(int(self.retry_s * 1000))}


class Quota:
    """The counters, the limits and what they say. clock: the wall clock (buckets survive a restart)."""

    def __init__(self, key=KEY, ip=IP, daily: float = DAILY, keyless_day: int = KEYLESS_DAY, per_key: dict | None = None, path: str = "",
                 clock=time.time, bucket: float = BUCKET):
        self.daily, self.keyless_day, self.path, self.clock, self.bucket = float(daily), int(keyless_day), path, clock, float(bucket)
        self.key, self.ip = self._limits(*key), self._limits(*ip)
        self.per_key = {name: self._limits(d.get("tph", key[0]), d.get("rph", key[1]), d.get("tpd"), d.get("rpd")) for name, d in (per_key or {}).items()}
        self.lock = threading.Lock()
        self.used = {}  # tenant -> {bucket: [tokens, requests]}
        self.keyless = {}  # bucket -> tokens of every ip: tenant
        self._over = {}  # tenant -> (until, over the hour's quota): worked out at most once a second
        self.held = {}  # tenant -> reply tokens its requests under way may still write (their max_tokens): the daily check counts them
        self.held_keyless = 0.0  # ... of every ip: tenant together (--keyless-day)
        self.n = collections.Counter()  # refusals by kind; requests taken in by class
        self.dirty, self.saved, self.save_failed = False, 0, 0

    def _limits(self, tph, rph, tpd=None, rpd=None) -> Limits:
        return Limits(tph, rph, round(self.daily * int(tph)) if tpd is None else tpd, round(self.daily * int(rph)) if rpd is None else rpd)

    def limits_of(self, tenant) -> Limits | None:
        """None: no limits (this PC, or tenancy off)."""
        if not isinstance(tenant, str) or tenant == N.LOCAL:
            return None
        if tenant.startswith("key:"):
            return self.per_key.get(tenant[4:], self.key)
        return self.ip

    # the counters (under self.lock)

    def _first(self, now: float, span: float) -> int:
        """The oldest bucket still inside a window of `span` seconds ending now."""
        return int((now - span) // self.bucket)

    def _sum(self, b: dict, now: float, span: float) -> tuple:
        first = self._first(now, span)
        tokens = requests = 0
        for m, v in b.items():
            if m >= first:
                tokens += v[0]
                requests += v[1]
        return tokens, requests

    def _leaves(self, m: int, span: float) -> float:
        """When bucket m leaves a window of `span` seconds."""
        return (m + 1) * self.bucket + span

    def _retry(self, b: dict, now: float, span: float, fits) -> float:
        """Seconds until fits(tokens, requests) holds of what is left inside the window as buckets leave it."""
        first = self._first(now, span)
        inside = sorted((m, v) for m, v in b.items() if m >= first)
        tokens, requests = sum(v[0] for _, v in inside), sum(v[1] for _, v in inside)
        for m, v in inside:
            tokens, requests = tokens - v[0], requests - v[1]
            if fits(tokens, requests):
                return self._leaves(m, span) - now
        return span

    def _add(self, tenant: str, m: int, tokens: float, requests: int) -> None:
        b = self.used.setdefault(tenant, {})
        v = b.setdefault(m, [0.0, 0])
        v[0], v[1] = max(0.0, v[0] + tokens), max(0, v[1] + requests)
        if tenant.startswith("ip:"):
            self.keyless[m] = max(0.0, self.keyless.get(m, 0.0) + tokens)
        self._over.pop(tenant, None)
        self.dirty = True

    # what the server asks

    def over(self, tenant) -> bool:
        """Whether the tenant is at or over its hourly quota (tokens or requests) now: a request that comes now is over
        quota (what it counts itself is not in it yet: the 600th request of 600, or one whose own prompt passes the
        tokens, is still in quota)."""
        lim = self.limits_of(tenant)
        if lim is None:
            return False
        now = self.clock()
        with self.lock:
            hit = self._over.get(tenant)
            if hit is not None and hit[0] > now:
                return hit[1]
            tokens, requests = self._sum(self.used.get(tenant, {}), now, HOUR)
            over = bool((lim.tph and tokens >= lim.tph) or (lim.rph and requests >= lim.rph))
            self._over[tenant] = (now + 1.0, over)
            return over

    def refusal(self, tenant, prompt: float, now: float | None = None, reply: float = 0.0, held: bool = True) -> Refusal | None:
        """A request of `prompt` tokens that may write `reply` more (its max_tokens) is refused when the tenant's day
        would pass a daily limit with it (its requests or tokens), or when it has no key and all such requests' day would
        pass --keyless-day. held: the replies of the tenant's requests under way count at their max_tokens until they are
        settled, so that requests at once cannot pass a day's tokens by their replies (False: the day's counters alone,
        for the check before a body is read)."""
        lim = self.limits_of(tenant)
        if lim is None:
            return None
        now = self.clock() if now is None else now
        asked = prompt + reply
        with self.lock:
            b = self.used.get(tenant, {})
            tokens, requests = self._sum(b, now, DAY)
            mine = self.held.get(tenant, 0.0) if held else 0.0
            if lim.rpd and requests + 1 > lim.rpd:
                return Refusal("requests_per_day", tenant, lim.rpd, requests, 1, self._retry(b, now, DAY, lambda t, r: r + 1 <= lim.rpd))
            if lim.tpd and tokens + mine + asked > lim.tpd:
                wait = DAY if asked > lim.tpd else self._wait(b, now, mine, lambda t, r: t + mine + asked <= lim.tpd)
                return Refusal("tokens_per_day", tenant, lim.tpd, tokens + mine, asked, wait)
            if tenant.startswith("ip:") and self.keyless_day:
                all_ = sum(v for m, v in self.keyless.items() if m >= self._first(now, DAY))
                every = self.held_keyless if held else 0.0
                if all_ + every + asked > self.keyless_day:
                    kb = {m: (v, 0) for m, v in self.keyless.items()}
                    wait = DAY if asked > self.keyless_day else self._wait(kb, now, every, lambda t, r: t + every + asked <= self.keyless_day)
                    return Refusal("keyless_tokens_per_day", tenant, self.keyless_day, all_ + every, asked, wait)
        return None

    HELD_RETRY = 60.0  # seconds: Retry-After when only the replies under way are in the way (they are settled as they end)

    def _wait(self, b: dict, now: float, held: float, fits) -> float:
        """Retry-After for a day's tokens: until buckets leave the day enough; when they never would but the replies
        under way hold some of it, HELD_RETRY (those settle to what they wrote, mostly far less than their max_tokens)."""
        wait = self._retry(b, now, DAY, fits)
        return self.HELD_RETRY if held and wait >= DAY else wait

    def taken(self, req) -> None:
        """The request is taken in: its class for the rest of its life (req.over_quota: its tenant's hour, before it, was
        at the quota), then counted with its whole prompt, until settle() counts what it cost."""
        tenant = getattr(req, "tenant", None)
        req.over_quota = False
        if self.limits_of(tenant) is None:
            self.n["class_local"] += tenant == N.LOCAL
            return
        req.over_quota = self.over(tenant)
        self.n["class_over_quota" if req.over_quota else "class_in_quota"] += 1
        now = self.clock()
        m = int(now // self.bucket)
        reply = float(max(0, int(getattr(req, "max_new", 0) or 0)))  # held for the daily check until settled
        req._quota = (tenant, m, float(len(req.full)), reply)
        with self.lock:
            self._add(tenant, m, float(len(req.full)), 1)
            self._hold(tenant, reply)

    def _hold(self, tenant: str, reply: float) -> None:
        """(under self.lock) reply tokens held (+) or let go (-) for the daily check."""
        left = self.held.get(tenant, 0.0) + reply
        if left > 0.5:
            self.held[tenant] = left
        else:
            self.held.pop(tenant, None)
        if tenant.startswith("ip:"):
            self.held_keyless = max(0.0, self.held_keyless + reply)

    def settle(self, req, event) -> None:
        """Its HTTP thread is done with it: what it cost goes where its admission was counted (nothing, when it never
        reached the engine), and the reply it held for the daily check is let go."""
        q = getattr(req, "_quota", None)
        if q is None:
            return
        req._quota = None
        tenant, m, counted, reply = q
        with self.lock:
            self._hold(tenant, -reply)
        job = getattr(req, "job", None)
        if job is None and (event is None or event[0] != "done"):  # it never reached the engine (a stand-in's answer did)
            tokens, requests = -counted, -1
        else:
            if event is not None and event[0] == "done":
                u = event[1]["usage"]
                prompt, cached, out = u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0), u["completion_tokens"]
            else:  # failed, or its client left: what it had read and written by then
                prompt, cached, out = len(req.full), getattr(req, "reused", 0), len(getattr(job, "out", ()))
            tokens, requests = (prompt - cached) + out + CACHED * cached - counted, 0
        with self.lock:
            if m >= self._first(self.clock(), DAY):
                self._add(tenant, m, tokens, requests)

    def headers(self, tenant) -> dict:
        """OpenAI's x-ratelimit-* of the hour for an answer to this tenant ({}: no limits)."""
        lim = self.limits_of(tenant)
        if lim is None:
            return {}
        now = self.clock()
        with self.lock:
            b = self.used.get(tenant, {})
            tokens, requests = self._sum(b, now, HOUR)
            first = self._first(now, HOUR)
            newest = max((m for m in b if m >= first), default=None)
        reset = duration(self._leaves(newest, HOUR) - now) if newest is not None else "0s"
        out = {}
        if lim.rph:
            out["x-ratelimit-limit-requests"] = str(lim.rph)
        if lim.tph:
            out["x-ratelimit-limit-tokens"] = str(lim.tph)
        if lim.rph:
            out["x-ratelimit-remaining-requests"] = str(max(0, lim.rph - requests))
        if lim.tph:
            out["x-ratelimit-remaining-tokens"] = str(max(0, int(lim.tph - tokens)))
        if lim.rph:
            out["x-ratelimit-reset-requests"] = reset
        if lim.tph:
            out["x-ratelimit-reset-tokens"] = reset
        return out

    def usage_of(self, tenant) -> dict:
        """GET /v1/usage: the caller's own usage and limits."""
        if tenant == N.LOCAL:
            return {"object": "usage", "user": tenant, "exempt": True, "priority": "this PC: no quota, served first"}
        lim = self.limits_of(tenant)
        if lim is None:
            return {"object": "usage", "user": None, "exempt": True}
        now = self.clock()
        with self.lock:
            b = self.used.get(tenant, {})
            ht, hr = self._sum(b, now, HOUR)
            dt, dr = self._sum(b, now, DAY)
            first = self._first(now, HOUR)
            newest = max((m for m in b if m >= first), default=None)
            over = bool((lim.tph and ht >= lim.tph) or (lim.rph and hr >= lim.rph))
            back = self._retry(b, now, HOUR, lambda t, r: (not lim.tph or t < lim.tph) and (not lim.rph or r < lim.rph)) if over else 0.0
            dfirst = self._first(now, DAY)
            keyless = sum(v for m, v in self.keyless.items() if m >= dfirst) if tenant.startswith("ip:") else None

        def part(tokens, requests, lt, lr, reset):
            return {"tokens": round(tokens, 1), "requests": requests, "limit_tokens": lt or None, "limit_requests": lr or None,
                    "remaining_tokens": max(0, int(lt - tokens)) if lt else None, "remaining_requests": max(0, lr - requests) if lr else None,
                    "reset_s": reset}

        out = {"object": "usage", "user": tenant, "exempt": False,
               "tokens_are": "the prompt that was not cached + the reply + the cached prompt x 0.1",
               "hour": part(ht, hr, lim.tph, lim.rph, round(self._leaves(newest, HOUR) - now) if newest is not None else 0),
               "day": part(dt, dr, lim.tpd, lim.rpd, None),
               "over_quota": over, "back_in_quota_s": round(back),
               "priority": "over quota: served, at a much lower priority" if over else "in quota"}
        if keyless is not None:
            out["keyless_day"] = {"tokens": round(keyless, 1), "limit_tokens": self.keyless_day or None}
        return out

    # the file

    def prune(self, now: float | None = None) -> None:
        """Buckets that have left the day go, and the tenants left with none; and every worked-out 'over' whose second
        has passed (a tenant asked about but never counted, one refused after its body was read, would stay for ever)."""
        now = self.clock() if now is None else now
        with self.lock:
            first = self._first(now, DAY)
            for tenant in list(self.used):
                b = self.used[tenant]
                for m in [m for m in b if m < first]:
                    del b[m]
                if not b:
                    del self.used[tenant]
                    self._over.pop(tenant, None)
            for tenant in [t for t, (until, _) in self._over.items() if until <= now]:
                del self._over[tenant]
            for m in [m for m in self.keyless if m < first]:
                del self.keyless[m]

    def state(self) -> dict:
        with self.lock:
            return {"version": 1, "bucket_s": self.bucket, "saved": round(self.clock(), 3),
                    "tenants": {t: [[m, round(v[0], 1), v[1]] for m, v in sorted(b.items())] for t, b in self.used.items()},
                    "keyless": [[m, round(v, 1)] for m, v in sorted(self.keyless.items())]}

    def save(self) -> None:
        """The counters to the file, atomically (a temporary beside it, then os.replace)."""
        if not self.path:
            return
        self.prune()
        with self.lock:
            self.dirty = False
        data = json.dumps(self.state(), separators=(",", ":")).encode()
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self.saved += 1

    def load(self) -> int:
        """The counters from the file, if it is there -> tenants read. One that cannot be read is put aside (.bad)."""
        if not self.path or not os.path.exists(self.path):
            return 0
        try:
            with open(self.path, "rb") as f:
                data = json.loads(f.read())
            scale = float(data.get("bucket_s", self.bucket)) / self.bucket
            used, keyless = {}, {}
            for tenant, rows in data["tenants"].items():
                b = used.setdefault(str(tenant), {})
                for m, tokens, requests in rows:
                    v = b.setdefault(int(int(m) * scale), [0.0, 0])
                    v[0], v[1] = v[0] + float(tokens), v[1] + int(requests)
            for m, tokens in data.get("keyless", []):
                keyless[int(int(m) * scale)] = keyless.get(int(int(m) * scale), 0.0) + float(tokens)
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"[serve] the quota file could not be read ({type(e).__name__}): the counters start from zero; it is kept as .bad", flush=True)
            try:
                os.replace(self.path, self.path + ".bad")
            except OSError:
                pass
            return 0
        with self.lock:
            self.used, self.keyless = used, keyless
        self.prune()
        return len(self.used)

    def tick(self) -> None:
        """Every SAVE_S seconds (keep_saving): the buckets that have left the day go (with or without a file), and the
        counters are written when they changed."""
        self.prune()
        if self.path and self.dirty:
            self._save_safely()

    def keep_saving(self) -> None:
        """A thread that ticks every SAVE_S seconds; with a file, the counters are written once more at exit."""
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            atexit.register(self._save_safely)

        def loop():
            while True:
                time.sleep(SAVE_S)
                try:
                    self.tick()
                except Exception as e:  # noqa: BLE001  (the counters never stop the server)
                    print(f"[serve] the quota counters could not be pruned: {C.why(e)}", flush=True)

        threading.Thread(target=loop, daemon=True, name="quota-counters").start()

    def _save_safely(self) -> None:
        try:
            self.save()
        except Exception as e:  # noqa: BLE001  (the counters never stop the server)
            self.save_failed += 1
            if self.save_failed in (1, 10, 100):
                print(f"[serve] the quota file could not be written ({self.save_failed}): {C.why(e)}", flush=True)

    # /metrics (label values from fixed sets; never a tenant)

    def families(self, family) -> None:
        now = self.clock()
        with self.lock:
            first = self._first(now, DAY)
            live = [t for t, b in self.used.items() if any(m >= first for m in b)]
            keyless = sum(v for m, v in self.keyless.items() if m >= first)
        family("sclmd_quota_refused_total", "counter", "requests refused at a daily limit (429), by the limit",
               [({"limit": k}, self.n[k]) for k in Refusal.WHAT])
        family("sclmd_quota_taken_total", "counter", "requests taken in, by their class then (local: this PC; over_quota: past its hourly quota)",
               [({"class": c}, self.n["class_" + c]) for c in ("local", "in_quota", "over_quota")])
        family("sclmd_quota_tenants", "gauge", "tenants with use in the last 24 hours, by kind",
               [({"kind": "key"}, sum(t.startswith("key:") for t in live)), ({"kind": "ip"}, sum(t.startswith("ip:") for t in live))])
        family("sclmd_quota_keyless_day_tokens", "gauge", "tokens of all requests without a key in the last 24 hours (--keyless-day bounds it)",
               [({}, round(keyless, 1))])


# the hooks (scripts/serve_admin.py)

def admit(adm, req) -> Refusal | None:
    """Under the Admin's lock, before --max-waiting counts: a daily limit refuses the request (nothing counted)."""
    quota = adm.quota
    if quota is None:
        return None
    refusal = quota.refusal(getattr(req, "tenant", None), float(len(req.full)), reply=float(getattr(req, "max_new", 0) or 0))
    if refusal is not None:
        quota.n[refusal.kind] += 1
    return refusal


def refused_early(adm, handler) -> bool:
    """POST /v1/chat/completions of a tenant already at a daily limit: the 429 before the body is read (whose bytes are
    then read and thrown away, as for R4's 401). True: answered."""
    quota = adm.quota
    w = N.who_of(handler)
    refusal = quota.refusal(w.tenant, 1.0, held=False) if quota is not None else None  # the day's counters: replies under way are not known to stay
    if refusal is None:
        return False
    quota.n[refusal.kind] += 1
    length = handler.headers.get("Content-Length")
    try:
        unread = int(length) if length is not None else None
    except ValueError:
        unread = None
    unread = None if unread is not None and unread < 0 else unread
    if unread != 0:
        handler.close_connection = True
    handler._r4_logged = True
    adm._write({"status": 429, "finish": "daily_limit", "limit": refusal.kind, "client": handler.client_address[0], "tenant": w.tenant})
    handler._json(429, refusal.body(), refusal.headers())
    if unread != 0:
        C.linger(handler.connection, handler.rfile, unread)
    return True


def status_headers(adm, handler) -> None:
    """After an answer's status line: the tenant's x-ratelimit-* headers."""
    for name, value in adm.quota.headers(N.who_of(handler).tenant).items():
        handler.send_header(name, value)


# start-up

def add_options(ap) -> None:
    g = ap.add_argument_group("quotas and priorities (scripts/serve_quota.py, sclmd/order.py; all off by default; design.md 6.18)")
    g.add_argument("--quotas", action="store_true", help="each tenant (API key, or address without a key; this PC is exempt) gets tokens "
                   "and requests an hour over the last 60 minutes (a request's tokens: prompt not cached + reply + cached prompt x 0.1); "
                   "over them it is still served, at a lower priority (--priorities). The one refusal is the day's limit (429 with "
                   "Retry-After). Answers carry x-ratelimit-* headers; GET /v1/usage gives the caller its own. Needs API keys or "
                   "--anonymous. A key's own limits: \"name: key tph=N rph=N tpd=N rpd=N\" in --api-key-file (0: none)")
    g.add_argument("--quota-file", default="", help="the counters (tenant ids and numbers only) are kept here, written every 10 s and "
                   "at exit, read back at start (\"\": in memory only); e.g. logs/serve_quota.json")
    g.add_argument("--tph", type=int, default=KEY[0], help="tokens an hour of an API key (0: no limit)")
    g.add_argument("--rph", type=int, default=KEY[1], help="requests an hour of an API key (0: no limit)")
    g.add_argument("--ip-tph", type=int, default=IP[0], help="tokens an hour of an address without a key (0: no limit)")
    g.add_argument("--ip-rph", type=int, default=IP[1], help="requests an hour of an address without a key (0: no limit)")
    g.add_argument("--daily-times", type=float, default=DAILY, help="the daily limit, over the last 24 hours: this many times the hour's "
                   "(a key's own tpd, rpd win)")
    g.add_argument("--keyless-day", type=int, default=KEYLESS_DAY, help="tokens a day of all requests without a key together (0: no limit)")
    g.add_argument("--priorities", action="store_true", help="serve by priority (sclmd/order.py): this PC first, then requests in quota, "
                   "then over quota far below; a tenant's second and later requests under way a step lower each; a step better for every "
                   "--priority-age seconds waited (never past this PC). It orders the waiting list (any request that fits is placed), "
                   "the tiles, the prompts' rows and the drafter's guesses of a pass, and which kept conversation leaves the GPU "
                   "first. The same reply tokens")
    g.add_argument("--priority-age", type=float, default=O.AGE, help="seconds of waiting that raise a request one step")
    g.add_argument("--over-quota-steps", type=int, default=O.GAP, help="steps from in quota to over quota (times --priority-age: how much "
                   "longer an over-quota request waits than an in-quota one when both wait)")


def check(cli, keys) -> None:
    """Checks made before the model loads."""
    if getattr(cli, "quotas", False) and not (keys or getattr(cli, "anonymous", False)):
        raise SystemExit("--quotas counts by API key or address: set an API key (SCLMD_API_KEY or --api-key-file) or --anonymous")
    for name in ("tph", "rph", "ip_tph", "ip_rph", "keyless_day", "over_quota_steps"):
        if getattr(cli, name, 0) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} is 0 or more")
    if getattr(cli, "daily_times", 1.0) < 0 or getattr(cli, "priority_age", 1.0) <= 0:
        raise SystemExit("--daily-times is 0 or more and --priority-age more than 0")


def install(adm, server, cli) -> None:
    """--quotas and --priorities onto a running server (adm: its Admin)."""
    quota = None
    if getattr(cli, "quotas", False):
        quota = Quota((cli.tph, cli.rph), (cli.ip_tph, cli.ip_rph), cli.daily_times, cli.keyless_day, getattr(cli, "key_limits", None) or {},
                      cli.quota_file)
        back = quota.load()
        quota.keep_saving()
        adm.quota = quota
        own = ", ".join(f"{name} {lim.tph:,}/{lim.rph:,}" for name, lim in sorted(quota.per_key.items()))
        print(f"[serve] quotas: a key {quota.key.tph:,} tokens and {quota.key.rph:,} requests an hour, an address without a key "
              f"{quota.ip.tph:,} and {quota.ip.rph:,}; the day {cli.daily_times:g} times; all requests without a key {quota.keyless_day:,} "
              f"tokens a day" + (f"; keys of their own: {own}" if own else "")
              + (f"; counters in {cli.quota_file} ({back} tenants read back)" if cli.quota_file else "; counters in memory only"), flush=True)
    if getattr(cli, "priorities", False):
        def klass(req) -> int:  # this PC; else the class it was taken in with (Quota.taken: its tenant's hour before it)
            if getattr(req, "local", False):
                return O.LOCAL
            return order.over if getattr(req, "over_quota", False) else O.IN_QUOTA

        order = O.Order(cli.priority_age, cli.over_quota_steps, klass)
        adm.order = order

        def priority_of(req, w) -> int:  # the seam of scripts/serve_net.py: its class as it comes (Quota.taken fixes it)
            req.local = bool(w.local) and w.tenant in (None, N.LOCAL)
            if req.local:
                return O.LOCAL
            return order.over if quota is not None and quota.over(getattr(req, "tenant", None)) else O.IN_QUOTA

        adm.users.priority_of = priority_of
        if hasattr(server, "batch"):
            server.order = server.batch.order = order
            print(f"[serve] priorities: this PC, then in quota, then over quota {order.gap} steps lower; a tenant's later requests under way "
                  f"a step lower each; a step better every {order.age:g} s of waiting (never past this PC)", flush=True)
        else:
            print("[serve] --priorities: no batcher here (a stand-in engine): the order is not installed", flush=True)
    elif quota is not None:
        print("[serve] --quotas without --priorities: over quota changes nothing but the headers and /v1/usage", flush=True)
