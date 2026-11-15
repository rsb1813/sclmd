# Tiers, concurrency and the public status of the internet server (design.md 6.20 "사용자와 등급", "한도", "우선순위";
# 6.21; P1). scripts/serve_admin.py, scripts/serve_quota.py and scripts/serve_net.py call in through a few hooks. Off
# unless --tiers, --concurrency or --public-status is given: without them nothing here runs, and no answer, log line or
# pass changes.
#
# --tiers (with --quotas; --ipdb names the DB-IP Lite files): every request is in a tier, decided when it comes (its
# prompt known) and kept for its life (req.tier):
#   key            a request with a key the server has;
#   ip_verified    no key, from an address that DB-IP Lite places in Korea (country KR) in an ASN of KR_CONSUMER below
#                  (Korean home and mobile ISPs), and whose behaviour keeps to the conditions below;
#   ip_unverified  any other address: abroad, not a home or mobile ISP (IDC, hosting, cloud, enterprise, school), not in
#                  the files, not an address (ip:unknown), the address of a proxy that is not trusted, or a verified
#                  one whose behaviour broke a condition;
#   local          this PC (no quota; served first).
# The address is the one scripts/serve_net.py read (X-Forwarded-For only from --trusted-proxies, from the right), so a
# client cannot pick its tier by writing the header. IPv6: the lookup is of the address, the tenant its /64 (N1).
# Behaviour (thresholds are the options' and serve.toml's, never on a public page; 0 turns one off): a verified
# address is demoted, for that request, when its last minute bucket and the one before hold --verify-rpm requests
# already, its hour --verify-rph requests or --verify-tph tokens (this request would be past them; as N2's quota,
# what came before it), more than --verify-outside of its last
# --verify-window prompts lie outside --verify-least .. --verify-most tokens, or (--verify-spread) the 90th
# percentile of those prompts is more than that many times the 10th. A tier's limits: scripts/serve_quota.py.
# --concurrency P,N,L (6.20: 1,3,8): a tenant's first P requests at once are "priority", the next N "normal", the
# next L "low" (sclmd/order.py: +0, +1, +2 steps with --priorities); the next one is refused before its body is read:
# 429, code concurrency_limit, Retry-After 2 s. A request holds its place from the moment its headers are read until
# its HTTP thread is done (serve.py's do_POST gives it back in a finally). This PC is not limited.
# --public-status: GET /v1/status, open without a key: the model, the version, the queue's two counts, what is kept
# and for how long (ZDR), the tiers' limits and the caller's tier. Never a threshold, never another user's anything.
# Nothing here imports torch.
import array
import collections
import os
import re
import ipaddress
import subprocess
import threading
import time

from scripts import serve_net as N

TIERS = ("key", "ip_verified", "ip_unverified")  # best first (sclmd/order.py's TIERS)
VERIFIED = (750_000, 120, 15_000_000)  # a verified address's tokens and requests an hour, tokens a day (6.20; RPH: N2's address default)
BANDS = (1, 3, 8)  # 6.20: a tenant's requests at once, "priority", "normal", "low"; the next is refused
BUSY_RETRY = 2.0  # seconds: Retry-After of a concurrency_limit 429 (a place frees when one of its requests ends)
HISTORY = 65_536  # addresses whose recent prompt sizes are kept (least recently seen go first)
LEAST_SAMPLES = 5  # prompts an address must have sent before the size conditions judge it
# the behaviour conditions: all off here (0). 6.20 keeps the owner's thresholds out of the repository: serve.toml
# (git ignores it) sets them; "outside" and "window" only shape the size rule and are not secret
RULES = {"rpm": 0, "rph": 0, "tph": 0, "least": 0, "most": 0, "outside": 0.5, "spread": 0.0, "window": 20}

# Korean home and mobile ISPs: the ASNs whose addresses are verified (in Korea by DB-IP's country too). Chosen from
# (1) APNIC Labs' estimates of users per AS for KR (https://stats.labs.apnic.net/aspop?c=KR, 13/11/2026, a 60-day
# window: the networks where people are, measured by ads), and (2) DB-IP Lite ASN 2026-11's organisations: the three
# carriers' home and mobile networks, the cable companies (SOs) and their former ASNs now in SK Broadband's name, and two
# apartment and regional ISPs. Left out though some have users: clouds and CDNs (Amazon, Vultr, Cloudflare, Microsoft,
# Oracle, Tencent, Alibaba, NAVER Cloud, Kakao), IDCs and hosting (KINX, Hostcenter, BROADBANDIDC, SMILESERV, DAOU,
# Elimnet), enterprises (Samsung SDS, LG CNS, Hyundai Autoever), schools, universities and research (KISTI, SNU, the
# offices of education), government (NIRS, SKB-GOV, SKBMIL, NICNET), and ASNs with few users for their size (Sejong
# Networks 7,230 users for 794,624 addresses, Shinbiro, abcle, SK Telecom's AS18302, SKTelink, PUBNETPLUS2). The big
# carriers' ASNs also carry some of their own IDC and business lines: DB-IP Lite has no usage type, so those are
# verified too (the behaviour conditions are what bounds them). Measured with these files: design.md 6.21.
KR_CONSUMER = {
    # KT (fixed and mobile; IPv6 2001:e60::/ is AS3559 in DB-IP)
    4766: "KT", 3559: "KT",
    # LG U+ (home: LG POWERCOMM, LG DACOM, PUBNETPLUS; mobile: LGTELECOM)
    17858: "LG U+", 3786: "LG U+", 9316: "LG U+", 17853: "LG U+ mobile",
    # SK Telecom (mobile) and SK Broadband (home; with the former cable ASNs of Tbroad and others in its name)
    9644: "SKT mobile", 9318: "SK Broadband",
    17849: "SK Broadband", 17864: "SK Broadband", 9689: "SK Broadband", 23563: "SK Broadband", 17857: "SK Broadband",
    18310: "SK Broadband", 9756: "SK Broadband", 38103: "SK Broadband", 17854: "SK Broadband", 38109: "SK Broadband",
    17597: "SK Broadband", 10164: "SK Broadband", 23578: "SK Broadband", 38096: "SK Broadband", 17573: "SK Broadband",
    9971: "SK Broadband", 38095: "SK Broadband", 38097: "SK Broadband",
    # cable: LG HelloVision
    9845: "LG HelloVision", 10066: "LG HelloVision", 9770: "LG HelloVision", 17839: "LG HelloVision", 9697: "LG HelloVision",
    18313: "LG HelloVision", 38091: "LG HelloVision", 38121: "LG HelloVision", 38669: "LG HelloVision", 17577: "LG HelloVision",
    9698: "LG HelloVision", 17586: "LG HelloVision", 18318: "LG HelloVision", 17598: "LG HelloVision", 45365: "LG HelloVision",
    38120: "LG HelloVision",
    # cable: D'LIVE, HCN (kt HCN, now KT skylife), CMB
    10036: "D'LIVE",
    9762: "HCN", 9319: "HCN", 7562: "HCN", 7623: "HCN", 10175: "HCN", 9569: "HCN",
    38684: "CMB", 10054: "CMB", 38680: "CMB", 9757: "CMB", 38133: "CMB", 18033: "CMB", 38387: "CMB", 18168: "CMB",
    # cable: the independent SOs
    9694: "Seokyung", 9946: "KCTV Jeju", 45361: "JCN Ulsan", 9976: "Namincheon", 9781: "Green cable", 9943: "KangNam cable",
    38673: "KCTV Kwangju", 17608: "ABN", 45374: "CCS", 23584: "Pusan cable", 17583: "Keumgang cable", 18334: "GCN cable",
    # apartment and regional home ISPs
    9981: "Saero Network", 9457: "Dreamline",
}


def _num(x):
    """A number as the status writes it: 5 rather than 5.0."""
    return int(x) if float(x) == int(x) else float(x)


# where an address is

class AddressBook:
    """DB-IP Lite's country and ASN files (MaxMind DB format, read through maxminddb's C extension, memory-mapped; a
    lookup of both is about 3 us). kind(ip) -> "kr_consumer", "kr_other", "abroad" or "unknown" (not a global address,
    or not in the files). The files are found in a folder by name (the newest month of each); a new month's files are
    put beside the old ones and read at the next start (a mapped file cannot be replaced on Windows)."""

    def __init__(self, folder: str, asns=None, denied=()):
        import maxminddb

        self.country_path, self.asn_path = self.newest(folder, "country"), self.newest(folder, "asn")
        self.country = maxminddb.open_database(self.country_path, maxminddb.MODE_MMAP_EXT)
        self.asn = maxminddb.open_database(self.asn_path, maxminddb.MODE_MMAP_EXT)
        self.asns = frozenset(KR_CONSUMER if asns is None else asns)
        self.denied = tuple(denied)  # networks inside the consumer ASNs that are not homes (IDC, cloud, hosting): kr_other

    @staticmethod
    def newest(folder: str, kind: str) -> str:
        """The newest dbip-<kind>-lite-YYYY-MM.mmdb in the folder (raises SystemExit when there is none)."""
        pattern = re.compile(rf"^dbip-{kind}-lite-(\d{{4}}-\d{{2}})\.mmdb$")
        try:
            names = sorted(n for n in os.listdir(folder) if pattern.match(n))
        except OSError as e:
            raise SystemExit(f"--ipdb {folder}: {type(e).__name__}")
        if not names:
            raise SystemExit(f"--ipdb {folder}: no dbip-{kind}-lite-YYYY-MM.mmdb in it (DB-IP Lite, design.md 6.21: how to get it)")
        return os.path.join(folder, names[-1])

    def where(self, ip) -> tuple:
        """(country ISO code, ASN), each None when the files do not say."""
        c, a = self.country.get(ip), self.asn.get(ip)
        country = ((c or {}).get("country") or {}).get("iso_code")
        return country, (a or {}).get("autonomous_system_number")

    def kind(self, ip) -> str:
        if ip is None or not ip.is_global:
            return "unknown"
        if ip.version == 6 and int(ip) >> 32 == 0:  # ::a.b.c.d (IPv4-compatible, deprecated): not a real source address
            return "unknown"
        try:
            country, asn = self.where(ip)
        except Exception:  # noqa: BLE001  (a file that cannot be read here: unverified, never verified)
            return "unknown"
        if country is None:
            return "unknown"
        if country != "KR":
            return "abroad"
        if asn not in self.asns:
            return "kr_other"
        if any(ip.version == n.version and ip in n for n in self.denied):
            return "kr_other"
        return "kr_consumer"

    def about(self) -> dict:
        return {"country": os.path.basename(self.country_path), "asn": os.path.basename(self.asn_path), "asns": len(self.asns)}


# the tiers

class Tiers:
    """The tier of a request. book: kind(ip) (AddressBook, or a test's); quota: scripts/serve_quota.py's Quota (its
    counters give the rate, the hour's requests and tokens); rules: RULES' names (thresholds; 0: off)."""

    def __init__(self, book, quota, rules: dict | None = None, history: int = HISTORY):
        self.book, self.quota = book, quota
        self.rules = {**RULES, **(rules or {})}
        self.window = max(1, int(self.rules["window"]))
        self.history = history
        self.sizes = collections.OrderedDict()  # tenant -> [array of its last prompts' sizes, how many it has sent]
        self.lock = threading.Lock()
        self.n = collections.Counter()

    def demoted(self, tenant: str):
        """Why a verified address's behaviour demotes it now (None: it keeps to every condition)."""
        r, q = self.rules, self.quota
        if q is not None:
            if r["rpm"] and q.used_of(tenant, 60.0)[1] >= r["rpm"]:  # (this one would be past it)
                return "rate"
            tokens, requests = q.used_of(tenant)
            if r["rph"] and requests >= r["rph"]:
                return "requests"
            if r["tph"] and tokens >= r["tph"]:
                return "tokens"
        if r["most"] or r["least"] or r["spread"]:
            with self.lock:
                got = self.sizes.get(tenant)
                sizes = list(got[0][: min(got[1], self.window)]) if got is not None else []
            if len(sizes) >= LEAST_SAMPLES:
                if r["most"] or r["least"]:
                    out = sum(1 for p in sizes if (r["most"] and p > r["most"]) or p < r["least"])
                    if out > r["outside"] * len(sizes):
                        return "size"
                if r["spread"]:
                    s = sorted(sizes)
                    low, high = s[int(0.1 * (len(s) - 1))], s[int(0.9 * (len(s) - 1))]
                    if high > r["spread"] * max(low, 1):
                        return "spread"
        return None

    def tier_of(self, w):
        """The tier of a Who now (counts nothing). Without a key the address's: this PC "local". A Who refused for its
        key (an unknown key) is judged by its address (GET /v1/status alone answers it)."""
        if w.key is not None:
            return "key"
        if w.local:
            return "local"
        tenant = w.tenant if w.tenant is not None else (N.tenant_of(w.address) if w.address is not None else None)
        if tenant is None or getattr(w, "unsure", False) or self.book.kind(w.address) != "kr_consumer":
            return "ip_unverified"
        return "ip_verified" if self.demoted(tenant) is None else "ip_unverified"

    def decide(self, req, w):
        """Users.tier_of's seam (scripts/serve_net.py: tag): the request's tier, decided once as it comes, kept on the
        request (req.tier) and its Who (the answer's x-ratelimit-* and this request's limits)."""
        if w.tenant is None:  # tenancy off: no tier
            return None
        if w.key is not None:
            tier = "key"
        elif w.local:
            tier = "local"
        elif getattr(w, "unsure", False) or self.book.kind(w.address) != "kr_consumer":
            tier = "ip_unverified"
        else:
            req._p1_sized = w.tenant  # its prompt's size is kept when it is taken in (taken)
            why = self.demoted(w.tenant)
            if why is not None:
                self.n["demoted_" + why] += 1
            tier = "ip_verified" if why is None else "ip_unverified"
        self.n["tier_" + tier] += 1
        w.tier = tier
        return tier

    def taken(self, req) -> None:
        """The request was taken in (Admin.submit): its prompt's size goes into its address's last --verify-window
        (only an address that could be verified keeps one; least recently seen ones go past HISTORY)."""
        tenant = getattr(req, "_p1_sized", None)
        if tenant is None:
            return
        with self.lock:
            got = self.sizes.get(tenant)
            if got is None:
                got = self.sizes[tenant] = [array.array("I", [0] * self.window), 0]
                while len(self.sizes) > self.history:
                    self.sizes.popitem(last=False)
            else:
                self.sizes.move_to_end(tenant)
            got[0][got[1] % self.window] = min(len(req.full), 0xFFFFFFFF)
            got[1] += 1

    def families(self, family) -> None:
        """/metrics: counts only (label values from fixed sets)."""
        family("sclmd_tier_taken_total", "counter", "requests by the tier they were given as they came",
               [({"tier": t}, self.n["tier_" + t]) for t in ("local",) + TIERS])
        family("sclmd_tier_demoted_total", "counter", "requests of a verified address given the unverified tier, by the condition it broke",
               [({"condition": c}, self.n["demoted_" + c]) for c in ("rate", "requests", "tokens", "size", "spread")])
        with self.lock:
            kept = len(self.sizes)
        family("sclmd_tier_addresses_kept", "gauge", "addresses whose recent prompt sizes are kept (for the size conditions)", [({}, kept)])


# concurrency

class Slots:
    """--concurrency: each tenant's requests at once (this PC is not limited). full() is looked at before a body is
    read (a 429 then costs nothing), take() once the body has come, give() when its HTTP thread is done. A place is
    never held by a body still on its way: 12 connections sending headers and then a byte every 2 s held every
    place of an address before (verify-P1-tiers), and on a shared mobile address that locks out everyone behind it."""

    def __init__(self, bands=BANDS):
        self.bands = tuple(int(b) for b in bands)
        self.most = sum(self.bands)
        self.lock = threading.Lock()
        self.held = {}  # tenant -> requests that hold a place (only tenants with one: bounded by the connections)
        self.refused = 0

    def full(self, tenant: str) -> bool:
        with self.lock:
            return self.held.get(tenant, 0) >= self.most

    def take(self, tenant: str) -> bool:
        with self.lock:
            n = self.held.get(tenant, 0)
            if n >= self.most:
                self.refused += 1
                return False
            self.held[tenant] = n + 1
            return True

    def give(self, tenant: str) -> None:
        with self.lock:
            n = self.held.get(tenant, 0) - 1
            if n > 0:
                self.held[tenant] = n
            else:
                self.held.pop(tenant, None)

    def body(self) -> dict:
        return {"error": {"message": f"Too many requests at once: {self.most} of this key's or address's requests are under way, the most "
                                     f"at once. Please try again in {BUSY_RETRY:g}s.", "type": "requests", "param": None, "code": "concurrency_limit"}}

    def headers(self) -> dict:
        return {"Retry-After": str(int(BUSY_RETRY)), "retry-after-ms": str(int(BUSY_RETRY * 1000))}


# the hooks (scripts/serve_admin.py)

def refused_busy(adm, handler) -> bool:
    """POST /v1/chat/completions, before its body is read: when every place of its tenant is taken, 429
    concurrency_limit at once (the body's bytes are then read and thrown away, as for the daily limit). The place
    itself is taken after the body (take_place). True: answered."""
    from scripts import serve_contract as C

    slots = adm.slots
    w = N.who_of(handler)
    if slots is None or w.tenant is None or w.tenant == N.LOCAL:
        return False
    if not slots.full(w.tenant):
        return False
    slots.refused += 1
    length = handler.headers.get("Content-Length")
    try:
        unread = int(length) if length is not None else None
    except ValueError:
        unread = None
    unread = None if unread is not None and unread < 0 else unread
    if unread != 0:
        handler.close_connection = True
    handler._r4_logged = True
    adm._write({"status": 429, "finish": "concurrency_limit", "client": handler.client_address[0], "tenant": w.tenant})
    handler._json(429, slots.body(), slots.headers())
    if unread != 0:
        C.linger(handler.connection, handler.rfile, unread)
    return True


def take_place(adm, handler) -> bool:
    """The body has come: the request takes one of its tenant's places. False: all taken meanwhile (the caller
    answers 429 concurrency_limit; nothing is left unread)."""
    slots = adm.slots
    w = N.who_of(handler)
    if slots is None or w.tenant is None or w.tenant == N.LOCAL:
        return True
    if slots.take(w.tenant):
        handler._p1_slot = w.tenant
        return True
    adm._write({"status": 429, "finish": "concurrency_limit", "client": handler.client_address[0], "tenant": w.tenant})
    handler._r4_logged = True
    handler._json(429, slots.body(), slots.headers())
    return False


def posted(adm, handler) -> None:
    """do_POST is over, however it ended: the request's place is given back."""
    tenant = getattr(handler, "_p1_slot", None)
    if tenant is not None:
        handler._p1_slot = None
        adm.slots.give(tenant)


def version() -> str:
    """The repository's short commit hash (once, at start), "unknown" when git cannot say."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        h = out.stdout.strip()
        return h if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{4,40}", h) else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class Status:
    """GET /v1/status (--public-status): cheap (no lock is held long, nothing is read from disk), the same for everyone
    but the caller's tier."""

    def __init__(self, adm, model: str, ver: str | None = None, sample_s: float = 1.0):
        self.adm, self.model, self.version = adm, model, ver if ver is not None else version()
        # live throughput: (time, reply tokens written so far, requests ended so far), one sample a second, the last 70 s
        self.samples = collections.deque(maxlen=int(70 / sample_s) + 2)
        self.sample_s = sample_s
        if hasattr(adm.server, "totals"):
            threading.Thread(target=self._sampler, name="status-live", daemon=True).start()

    def _written(self) -> tuple:
        """Reply tokens written so far (ended requests' and the ones under way) and requests ended. Read without the
        engine's lock: a request ending between the two reads is missed for one sample, never counted twice."""
        s = self.adm.server
        t = s.totals
        done, ended = t["completion_tokens"], t["requests"]
        under = 0
        for req in list(s.active):
            job = getattr(req, "job", None)
            if job is not None and job.done is None:
                under += len(job.out)
        return done + under, ended

    def _sampler(self) -> None:
        last = 0
        while True:
            try:
                written, ended = self._written()
                last = max(last, written)  # never backwards (a request between "under way" and "ended")
                self.samples.append((time.monotonic(), last, ended))
            except Exception:  # noqa: BLE001  (a sample missed; the status still answers)
                pass
            time.sleep(self.sample_s)

    def live(self) -> dict | None:
        xs = list(self.samples)
        if len(xs) < 2:
            return None
        now, tok, req = xs[-1]

        def since(seconds: float) -> tuple:
            for t, k, r in xs:
                if now - t <= seconds + self.sample_s / 2:
                    return t, k, r
            return xs[0]

        t10, k10, _ = since(10)
        t60, k60, r60 = since(60)
        return {"tok_s": round((tok - k10) / max(now - t10, 1e-9), 1), "tok_s_1m": round((tok - k60) / max(now - t60, 1e-9), 1),
                "requests_1m": int(req - r60)}

    def body(self, handler) -> dict:
        adm = self.adm
        s, q, slots = adm.server, adm.quota, adm.slots
        if hasattr(s, "active"):
            under_way, waiting = len(s.active), len(s.waiting) + s.inbox.qsize()
        else:  # a stand-in engine
            r = s.about()["requests"]
            under_way, waiting = r["under_way"], r["waiting"]
        ops = getattr(s, "ops", None)
        ok = ops is None or ops.state() == "ok"
        cache = getattr(s, "cache", None)
        disk = cache is not None and getattr(cache, "disk", None) is not None
        apart = cache is not None and adm.users.per_tenant and adm.users.on

        def tier(lim):
            return {"tph": lim.tph, "tpd": lim.tpd} if lim is not None else None

        bands = slots.bands if slots is not None else (0, 0, 0)
        w = N.who_of(handler)
        caller = w.tier if w.tier is not None else (adm.tiers.tier_of(w) if adm.tiers is not None else ("local" if w.local else "ip_unverified"))
        return {"ok": bool(ok), "model": self.model, "version": self.version, "uptime_s": int(time.time() - adm.t0),
                "queue": {"under_way": int(under_way), "waiting": int(waiting)}, "live": self.live(),
                "zdr": {"prompt_logging": False, "cache_ram_minutes": 15 if cache is not None else None, "cache_disk_hours": 1 if disk else 0,
                        "disk_encryption": "AES-GCM" if disk else "none", "shared_between_users": not apart},
                "limits": {"ip_unverified": tier(q.ip), "ip_verified": tier(q.verified), "key": tier(q.key),
                           "output_weight": _num(q.output), "cached_weight": 0.1,
                           "concurrency": {"priority": bands[0], "normal": bands[1], "low": bands[2]}},
                "caller": {"tier": caller}}


# start-up

def add_options(ap) -> None:
    g = ap.add_argument_group("tiers, concurrency and the public status (scripts/serve_tiers.py; all off by default; design.md 6.21)")
    g.add_argument("--tiers", action="store_true", help="each request is in a tier, decided as it comes: key; ip_verified (no key, a Korean "
                   "home or mobile ISP's address in DB-IP Lite whose behaviour keeps to the --verify-* conditions); ip_unverified (any other "
                   "address). Each has its own quota (--tph..., --verified-tph..., --ip-tph...) and, with --priorities, its own priority "
                   "(this PC, then key, verified, unverified; within one, in quota then over). Needs --quotas and --ipdb")
    g.add_argument("--ipdb", default="", help="the folder of DB-IP Lite's dbip-country-lite-YYYY-MM.mmdb and dbip-asn-lite-YYYY-MM.mmdb "
                   "(CC BY 4.0, a new month each month; the newest of each is read at start), e.g. C:/models/sclmd-ipdb")
    g.add_argument("--unverified-prefixes", default="", help="a file of networks (one CIDR a line, '#' comments) inside the consumer "
                   "ASNs that are IDC, cloud or hosting (e.g. KT Cloud, Cafe24 in AS4766): never verified. DB-IP Lite has no usage type")
    g.add_argument("--kr-consumer-asns", default="", help="the ASNs of Korean home and mobile ISPs, by commas, in the place of the built-in "
                   "list (serve_tiers.KR_CONSUMER; \"\": that list)")
    g.add_argument("--verified-tph", type=int, default=VERIFIED[0], help="tokens an hour of a verified address (0: no limit)")
    g.add_argument("--verified-rph", type=int, default=VERIFIED[1], help="requests an hour of a verified address (0: no limit)")
    g.add_argument("--verified-tpd", type=int, default=VERIFIED[2], help="tokens a day of a verified address (0: no limit)")
    g.add_argument("--tier-steps", type=int, default=0, help="--priorities: steps from one tier's class to the next (0: --over-quota-steps "
                   "+ the bands' 2 + 1 with --concurrency, 15 by default, else + 1: a tier's over-quota requests stay ahead of the next tier's)")
    for name, kind, text in (("rpm", int, "the most requests in its last minute bucket and the one before"),
                             ("rph", int, "the most requests in its last hour"), ("tph", int, "the most tokens in its last hour"),
                             ("least", int, "tokens: a prompt shorter than this is outside the range"),
                             ("most", int, "tokens: a prompt longer than this is outside the range"),
                             ("outside", float, "the share of its last --verify-window prompts that may be outside the range"),
                             ("spread", float, "the most the 90th percentile of its last prompts may be, in times the 10th"),
                             ("window", int, "prompts kept of each address for the range and the spread")):
        g.add_argument(f"--verify-{name}", type=kind, default=RULES[name], help=f"--tiers' behaviour: {text} (a verified address past it is "
                       "unverified for that request; 0: no such condition; put the owner's own in serve.toml, not on a public page)")
    g.add_argument("--concurrency", default="", help="P,N,L: a tenant's first P requests at once are served at priority, the next N normal, "
                   "the next L low (with --priorities), and the next one is refused before its body is read: 429 concurrency_limit, "
                   "Retry-After 2 s (this PC is not limited). \"\": no limit, as before; 1,3,8 in serve.public.example.toml")
    g.add_argument("--public-status", action="store_true", help="GET /v1/status, open without a key: ok, model, version, uptime, the queue's "
                   "two counts, what is kept (zdr), the tiers' limits and the caller's own tier (needs --tiers)")


def bands_of(text: str) -> tuple:
    parts = [p for p in str(text or "").replace(" ", "").split(",") if p]
    try:
        bands = tuple(int(p) for p in parts)
    except ValueError:
        bands = ()
    if len(bands) != 3 or any(b < 0 for b in bands) or not sum(bands):
        raise SystemExit("--concurrency is P,N,L: three whole numbers (e.g. 1,3,8), not all 0")
    return bands


def ladder_of(cli):
    """--concurrency's bands, or None (sclmd/order.py: a step for each earlier request, as N2)."""
    text = getattr(cli, "concurrency", "") or ""
    return bands_of(text) if text else None


def verified_limits(cli):
    """The verified tier's (tph, rph, tpd) for serve_quota.Quota, None without --tiers."""
    if not getattr(cli, "tiers", False):
        return None
    return (cli.verified_tph, cli.verified_rph, cli.verified_tpd)


def asns_of(cli):
    text = getattr(cli, "kr_consumer_asns", "") or ""
    if not text:
        return None
    try:
        return frozenset(int(p.strip().removeprefix("AS").removeprefix("as")) for p in text.split(",") if p.strip())
    except ValueError:
        raise SystemExit("--kr-consumer-asns: ASNs by commas (4766,9318 or AS4766,AS9318)")


def rules_of(cli) -> dict:
    return {name: getattr(cli, f"verify_{name}", RULES[name]) for name in RULES}


def check(cli) -> None:
    """Checks made before the model loads."""
    if getattr(cli, "tiers", False):
        if not getattr(cli, "quotas", False):
            raise SystemExit("--tiers gives each tier its quota: it needs --quotas")
        if not getattr(cli, "ipdb", ""):
            raise SystemExit("--tiers needs --ipdb (the folder of DB-IP Lite's country and ASN files)")
        for kind in ("country", "asn"):
            AddressBook.newest(cli.ipdb, kind)
        asns_of(cli)
        for name in ("verified_tph", "verified_rph", "verified_tpd", "tier_steps") + tuple(f"verify_{n}" for n in RULES):
            if getattr(cli, name, 0) < 0:
                raise SystemExit(f"--{name.replace('_', '-')} is 0 or more")
        if cli.verify_window < LEAST_SAMPLES:
            raise SystemExit(f"--verify-window is at least {LEAST_SAMPLES}")
    if getattr(cli, "public_status", False) and not getattr(cli, "tiers", False):
        raise SystemExit("--public-status tells the tiers' limits: it needs --tiers")
    if getattr(cli, "concurrency", ""):
        bands_of(cli.concurrency)


def denied_of(cli) -> tuple:
    """--unverified-prefixes: the networks of its file (raises SystemExit on a line that is not one)."""
    path = getattr(cli, "unverified_prefixes", "") or ""
    if not path:
        return ()
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                s = line.split("#", 1)[0].strip()
                if s:
                    out.append(ipaddress.ip_network(s, strict=False))
    except (OSError, ValueError) as e:
        raise SystemExit(f"--unverified-prefixes {path}: {type(e).__name__} (line {i if 'i' in locals() else 0})")
    return tuple(out)


def install(adm, server, cli, app) -> None:
    """--tiers, --concurrency and --public-status onto a running server (after scripts/serve_quota.py's install)."""
    if getattr(cli, "tiers", False):
        book = AddressBook(cli.ipdb, asns_of(cli))
        denied = denied_of(cli)
        if denied:
            book.denied = denied
        tiers = Tiers(book, adm.quota, rules_of(cli))
        adm.tiers = tiers
        adm.users.tier_of = tiers.decide
        on = sum(1 for k, v in tiers.rules.items() if k not in ("outside", "window") and v)
        q = adm.quota
        print(f"[serve] tiers: a key {q.key.tph:,} tokens an hour and {q.key.tpd:,} a day, a verified address {q.verified.tph:,} and "
              f"{q.verified.tpd:,}, an unverified one {q.ip.tph:,} and {q.ip.tpd:,}; verified: a Korean home or mobile ISP "
              f"({len(book.asns)} ASNs) in {book.about()['country']} and {book.about()['asn']}, and {on} behaviour conditions "
              "(their thresholds are the options')", flush=True)
    if getattr(cli, "concurrency", ""):
        adm.slots = Slots(bands_of(cli.concurrency))
        print(f"[serve] concurrency: a tenant's requests at once {'/'.join(map(str, adm.slots.bands))} at priority/normal/low, then "
              f"429 concurrency_limit (this PC is not limited)", flush=True)
    if getattr(cli, "public_status", False):
        adm.status_api = Status(adm, app.card()["id"])
        print(f"[serve] GET /v1/status, open without a key (version {adm.status_api.version})", flush=True)
