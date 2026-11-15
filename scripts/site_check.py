"""The public pages' check (deploy/site, deploy/Caddyfile.edge; design.md 6.23), without Caddy, the GPU or Google.

    python -m scripts.site_check                     every check below; exit 1 on a failure
    python -m scripts.site_check --serve 8120        the stand-in edge with the mock API on 127.0.0.1:8120 (and its
                                                     stand-in Google on the next port), to open in a browser:
                                                     http://localhost:8120/keys (localhost: a secure context for passkeys)

The stand-in edge reads deploy/Caddyfile.edge's ssalmuck-ai block (its path matchers, its header snippets and each
handle's Cache-Control) and routes as that block does: the page paths to their .html files without a redirect, /keys/
and the like 308 to the page, /assets/*, the API paths to the mock API, anything else 404 with /404.html. The mock API
keeps to the contract of P1 (GET /v1/status) and P2 (/keys/google/start, /keys/callback/google, POST /keys/api/*, with
the X-Requested-With and Origin checks); it does not verify passkey signatures (P2's server does), only the parts the
page builds (base64url fields, clientDataJSON's type, challenge and origin). POST /_mock/next-error {"path", "status",
"code", "retry_after"} makes the next answer on that path an error (the page's error texts). It holds no real key and
reads no secret.

Checks: each page is well-formed HTML (a strict stdlib parse: every element closed, no duplicate id or attribute; not
a full HTML5 validation) with lang, charset, viewport and title; no inline script, style or event handler (the CSP
forbids them); every link and fragment resolves through the stand-in edge; no page, stylesheet or script names another
origin; the edge's routes, status codes and headers; the mock flow from Google to a key, rotate, revoke and logout; and
the scripts' own checks under node (tests/site_keys_check.js: base64url, the WebAuthn JSON forms, every error text,
the limits' fallback numbers), when node is installed."""
import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from html.parser import HTMLParser
from http.client import HTTPConnection
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "deploy" / "site"
CADDY = ROOT / "deploy" / "Caddyfile.edge"
NODE_CHECK = ROOT / "tests" / "site_keys_check.js"
PAGES = {"/": "index.html", "/keys": "keys.html", "/guide": "guide.html", "/trust": "trust.html", "/terms": "terms.html",
         "/privacy": "privacy.html"}
CONTACT = "mailto:rsb2890.sasm@gmail.com"

# GET /v1/status as the contract has it (P1); 6.20's numbers
STATUS = {
    "ok": True, "model": "gemma-4-31b-it", "version": "mock000", "uptime_s": 3725, "queue": {"under_way": 2, "waiting": 1},
    "zdr": {"prompt_logging": False, "cache_ram_minutes": 15, "cache_disk_hours": 1, "disk_encryption": "AES-GCM",
            "shared_between_users": False},
    "limits": {"ip_unverified": {"tph": 300000, "tpd": 7500000}, "ip_verified": {"tph": 750000, "tpd": 15000000},
               "key": {"tph": 1250000, "tpd": 40000000}, "output_weight": 5, "cached_weight": 0.1,
               "concurrency": {"priority": 1, "normal": 3, "low": 8}},
    "caller": {"tier": "ip_unverified"},
}


# ---------- deploy/Caddyfile.edge ----------

def site_block(text: str, name: str = "ssalmuck-ai.duckdns.org") -> str:
    """The body of a top-level block (`name {` ... the matching `}`)."""
    m = re.search(r"^" + re.escape(name) + r"\s*\{\s*$", text, re.M)
    if not m:
        raise ValueError(f"no block {name}")
    depth, i = 1, m.end()
    while depth:
        j = min((k for k in (text.find("{", i), text.find("}", i)) if k >= 0), default=-1)
        if j < 0:
            raise ValueError(f"block {name} is not closed")
        depth += 1 if text[j] == "{" else -1
        i = j + 1
    return text[m.end():i - 1]


def caddy_config(path: Path = CADDY) -> dict:
    """What the stand-in edge needs of the ssalmuck-ai block: matchers, header snippets, Cache-Control per handle."""
    text = path.read_text(encoding="utf-8")
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    block = site_block(text)
    matchers = {}
    for m in re.finditer(r"^\s*@(\w+)\s+(path|path_regexp)\s+(.+)$", block, re.M):
        name, kind, rest = m.group(1), m.group(2), m.group(3).split()
        matchers[name] = ("regexp", re.compile(rest[-1])) if kind == "path_regexp" else ("path", rest)
    headers = {}
    for snip in ("sclmd_headers", "sclmd_page"):
        body = site_block(text, "(" + snip + ")")
        hs = {}
        for m in re.finditer(r'^\s*(-?[A-Za-z][\w-]*)\s+"([^"]*)"\s*$|^\s*(-[A-Za-z][\w-]*)\s*$', body, re.M):
            if m.group(3):
                hs[m.group(3)] = None
            else:
                hs[m.group(1)] = m.group(2)
        headers[snip] = hs
    cache = {}
    for m in re.finditer(r"^\thandle @(\w+) \{(.*?)^\t\}", block, re.M | re.S):
        c = re.search(r'header Cache-Control "([^"]*)"', m.group(2))
        cache[m.group(1)] = c.group(1) if c else None
        cache.setdefault("_imports_" + m.group(1), re.findall(r"import (\w+)", m.group(2)))
    errors = re.search(r"handle_errors \{(.*)", block, re.S)
    cache["errors"] = re.search(r'header Cache-Control "([^"]*)"', errors.group(1)).group(1) if errors else None
    proxy = re.search(r"reverse_proxy\s+(\S+)", block)
    return {"matchers": matchers, "headers": headers, "cache": cache, "upstream": proxy.group(1) if proxy else None,
            "imports": re.findall(r"^\timport (\w+)", block, re.M)}


def caddy_match(kind_rest, path: str) -> bool:
    kind, rest = kind_rest
    if kind == "regexp":
        return bool(rest.search(path))
    p = path.lower()  # Caddy's path matcher ignores case
    for r in rest:
        r = r.lower()
        if r.endswith("*") and p.startswith(r[:-1]) or p == r:
            return True
    return False


# ---------- the mock API (the contract of P1 and P2) ----------

def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def unb64u(s: str) -> bytes:
    if not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9_-]*", s):
        raise ValueError("not base64url")
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class Mock:
    def __init__(self, google: str):
        self.google = google  # the stand-in Google's origin
        self.lock = threading.Lock()
        self.states, self.sessions, self.logins = {}, {}, set()
        self.accounts = {}  # account hmac -> {"name", "created", "key_hash", "creds"}
        self.creds = {}  # credential id -> account hmac
        self.calls = []  # (method, path, status): what the tests look at; never a key
        self.next_error = {}
        self.secret = secrets.token_bytes(32)

    def account_of(self, sub: str) -> str:
        return hmac.new(self.secret, sub.encode(), hashlib.sha256).hexdigest()

    def new_key(self, acct: str) -> str:
        key = "sk-mock-" + secrets.token_urlsafe(24)
        self.accounts[acct]["key_hash"] = hashlib.sha256(key.encode()).hexdigest()
        return key

    def usage(self) -> dict:
        def part(lt, lr):
            return {"tokens": 1234.5, "requests": 7, "limit_tokens": lt, "limit_requests": lr, "remaining_tokens": lt - 1234,
                    "remaining_requests": lr - 7, "reset_s": 1800}
        return {"object": "usage", "user": "key:mock", "exempt": False, "hour": part(1250000, 1500), "day": part(40000000, 36000),
                "over_quota": False, "back_in_quota_s": 0}


class Edge(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, google_port: int, tier: str = "ip_unverified"):
        super().__init__(("127.0.0.1", port), EdgeHandler)
        self.conf = caddy_config()
        self.mock = Mock(f"http://127.0.0.1:{google_port}")
        self.tier = tier
        self.google = ThreadingHTTPServer(("127.0.0.1", google_port), GoogleHandler)
        self.google.daemon_threads = True

    def start(self):
        for s in (self, self.google):
            threading.Thread(target=s.serve_forever, daemon=True).start()
        return self

    def stop(self):
        for s in (self, self.google):
            s.shutdown()
            s.server_close()


class GoogleHandler(BaseHTTPRequestHandler):
    """Google's consent screen in one step: back to redirect_uri with a code (or ?deny=1: access_denied)."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.path).query))
        back = q.get("redirect_uri", "")
        if q.get("deny"):
            to = back + "?" + urllib.parse.urlencode({"error": "access_denied", "state": q.get("state", "")})
        else:
            to = back + "?" + urllib.parse.urlencode({"code": "mock-code-" + secrets.token_hex(4), "state": q.get("state", "")})
        self.send_response(302)
        self.send_header("Location", to)
        self.send_header("Content-Length", "0")
        self.end_headers()


class EdgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = ""
    sys_version = ""

    def log_message(self, *a):
        pass

    def send_response(self, code, message=None):  # Caddyfile.edge drops Server
        self.log_request(code)
        self.send_response_only(code, message)
        self.send_header("Date", self.date_time_string())

    # ----- answers

    def answer(self, status: int, body: bytes = b"", ctype: str = "", extra: dict | None = None, page: bool = False,
               cache: str | None = None, head: bool = False):
        self.send_response(status)
        conf = self.server.conf["headers"]
        for k, v in conf["sclmd_headers"].items():
            if v is not None:
                self.send_header(k, v)
        if page:
            for k, v in conf["sclmd_page"].items():
                if v is not None:
                    self.send_header(k, v)
        if cache:
            self.send_header("Cache-Control", cache)
        if ctype:
            self.send_header("Content-Type", ctype)
        for k, v in (extra or {}).items():
            for one in v if isinstance(v, list) else [v]:
                self.send_header(k, one)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def file(self, rel: str, cache: str | None, status: int = 200, head: bool = False) -> bool:
        """A file under deploy/site, its name's case exact (the edge is Linux: /KEYS.html is not /keys.html)."""
        parts = [p for p in rel.split("/") if p]
        d = SITE
        for p in parts:
            if p in (".", "..") or not d.is_dir() or p not in os.listdir(d):
                return False
            d = d / p
        if not d.is_file():
            return False
        ext = d.suffix.lower()
        ctype = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".svg": "image/svg+xml"}.get(ext, "application/octet-stream")
        self.answer(status, d.read_bytes(), ctype, page=True, cache=cache, head=head)
        return True

    def not_found(self, head=False):
        if not self.file("404.html", self.server.conf["cache"].get("errors"), 404, head):
            self.answer(404, head=head)

    def route(self, method: str):
        url = urllib.parse.urlsplit(self.path)
        path = urllib.parse.unquote(url.path)
        m, cache = self.server.conf["matchers"], self.server.conf["cache"]
        head = method == "HEAD"
        if path == "/_mock/next-error" and method == "POST":  # the mock's own control, not a public path
            n = int(self.headers.get("Content-Length") or 0)
            want = json.loads(self.rfile.read(n) or b"{}")
            self.server.mock.next_error[str(want.get("path", ""))] = want
            return self.answer(204)
        if caddy_match(m["api"], path):
            return self.api(method, path, url.query)
        if method not in ("GET", "HEAD"):  # Caddy's file_server: 405 for other methods
            return self.answer(405, extra={"Allow": "GET, HEAD"}, page=True, head=head)
        if caddy_match(m["home"], path):
            return self.file("index.html", cache["home"], head=head) or self.not_found(head)
        if caddy_match(m["pages"], path):
            return self.file(path.lstrip("/") + ".html", cache["pages"], head=head) or self.not_found(head)
        if caddy_match(m["slash"], path):
            return self.answer(308, extra={"Location": "/" + m["slash"][1].search(path).group(1)}, head=head)
        if caddy_match(m["assets"], path):
            return self.file(path, cache["assets"], head=head) or self.not_found(head)
        return self.not_found(head)

    def do_GET(self):
        self.route("GET")

    def do_HEAD(self):
        self.route("HEAD")

    def do_POST(self):
        self.route("POST")

    def do_PUT(self):
        self.route("PUT")

    def do_DELETE(self):
        self.route("DELETE")

    # ----- the mock API

    def origin(self) -> str:
        return "http://" + (self.headers.get("Host") or f"127.0.0.1:{self.server.server_port}")

    def json(self, status: int, obj, extra: dict | None = None):
        self.server.mock.calls.append((self.command, urllib.parse.urlsplit(self.path).path, status))
        self.answer(status, json.dumps(obj).encode(), "application/json", extra=dict({"Cache-Control": "no-store"}, **(extra or {})))

    def error(self, status: int, code: str, message: str = "", extra: dict | None = None):
        etype = "invalid_request_error" if status < 500 else "server_error"
        self.json(status, {"error": {"message": message or code.replace("_", " "), "type": etype, "code": code}}, extra)

    def redirect(self, to: str, cookie: str | None = None):
        self.server.mock.calls.append((self.command, urllib.parse.urlsplit(self.path).path, 302))
        extra = {"Location": to, "Cache-Control": "no-store"}
        if cookie:
            extra["Set-Cookie"] = cookie
        self.answer(302, extra=extra)

    def session(self):
        c = SimpleCookie()
        try:
            c.load(self.headers.get("Cookie", ""))
        except Exception:  # noqa: BLE001
            return None, None
        sid = c["sclmd_keys"].value if "sclmd_keys" in c else None
        return sid, self.server.mock.sessions.get(sid) if sid else None

    @staticmethod
    def cookie(sid: str, age: int = 900) -> str:
        return f"sclmd_keys={sid}; HttpOnly; Secure; SameSite=Lax; Path=/keys; Max-Age={age}"

    def api(self, method: str, path: str, query: str):
        mock = self.server.mock
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(min(n, 1 << 20)) if n else b""
        if method == "GET" and path == "/v1/status":
            st = json.loads(json.dumps(STATUS))
            st["caller"]["tier"] = self.server.tier
            return self.json(200, st)
        if method == "GET" and path == "/health":
            return self.json(200, {"status": "ok"})
        if method == "GET" and path == "/keys/google/start":
            state, nonce = secrets.token_urlsafe(16), secrets.token_urlsafe(16)
            with mock.lock:
                mock.states[state] = {"nonce": nonce, "t": time.time()}
            back = self.origin() + "/keys/callback/google"
            return self.redirect(mock.google + "/auth?" + urllib.parse.urlencode(
                {"response_type": "code", "scope": "openid", "state": state, "nonce": nonce, "redirect_uri": back,
                 "code_challenge": b64u(hashlib.sha256(b"verifier").digest()), "code_challenge_method": "S256"}))
        if method == "GET" and path == "/keys/callback/google":
            q = dict(urllib.parse.parse_qsl(query))
            with mock.lock:
                st = mock.states.pop(q.get("state", ""), None)
            if q.get("error"):
                return self.redirect("/keys#error=" + ("access_denied" if q["error"] == "access_denied" else "google_error"))
            if st is None:
                return self.redirect("/keys#error=state_invalid")
            if time.time() - st["t"] > 600:
                return self.redirect("/keys#error=state_expired")
            sid = secrets.token_urlsafe(24)
            with mock.lock:
                mock.sessions[sid] = {"account": mock.account_of("sub-" + q.get("code", "")[-4:]), "verified": True, "login": False}
            return self.redirect("/keys#verified", self.cookie(sid))
        if not path.startswith("/keys/api/"):
            return self.error(404, "not_found")
        if method != "POST":
            return self.error(405, "method_not_allowed")
        if self.headers.get("X-Requested-With") != "sclmd" or self.headers.get("Origin") != self.origin():
            return self.error(403, "origin_mismatch")
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            return self.error(400, "invalid_json")
        name = path[len("/keys/api/"):]
        forced = mock.next_error.pop(name, None)
        if forced:
            extra = {"Retry-After": str(forced["retry_after"])} if forced.get("retry_after") else None
            return self.error(int(forced.get("status", 400)), forced.get("code", ""), extra=extra)
        sid, sess = self.session()
        if name == "passkey/register/options":
            if not sess or not sess.get("verified"):
                return self.error(401, "not_verified")
            sess["challenge"] = b64u(secrets.token_bytes(32))
            host = (self.headers.get("Host") or "localhost").rsplit(":", 1)[0]
            return self.json(200, {
                "rp": {"id": host, "name": "ssalmuck-ai"},
                "user": {"id": b64u(secrets.token_bytes(16)), "name": "ssalmuck-ai 키", "displayName": "ssalmuck-ai 키"},
                "challenge": sess["challenge"], "pubKeyCredParams": [{"type": "public-key", "alg": -7}, {"type": "public-key", "alg": -257}],
                "timeout": 300000, "excludeCredentials": [], "attestation": "none",
                "authenticatorSelection": {"residentKey": "required", "requireResidentKey": True, "userVerification": "preferred"}})
        if name == "passkey/register/verify":
            if not sess or not sess.get("verified") or not sess.get("challenge"):
                return self.error(401, "not_verified")
            why = self.check_credential(body.get("credential"), "webauthn.create", sess.pop("challenge"))
            if why:
                return self.error(400, why)
            cred = body["credential"]
            with mock.lock:
                acct = sess["account"]
                replaced = acct in mock.accounts
                if replaced:
                    for c in mock.accounts[acct]["creds"]:
                        mock.creds.pop(c, None)
                mock.accounts[acct] = {"name": "key-" + acct[:6], "created": int(time.time()), "creds": [cred["id"]]}
                mock.creds[cred["id"]] = acct
                key = mock.new_key(acct)
                sess.update(verified=False, login=True)
            return self.json(200, {"key": key, "name": mock.accounts[acct]["name"], "replaced": replaced})
        if name == "passkey/login/options":
            ch = b64u(secrets.token_bytes(32))
            with mock.lock:
                mock.logins.add(ch)
            return self.json(200, {"challenge": ch, "timeout": 300000, "rpId": (self.headers.get("Host") or "localhost").rsplit(":", 1)[0],
                                   "allowCredentials": [], "userVerification": "preferred"})
        if name == "passkey/login/verify":
            cred = body.get("credential")
            try:
                ch = json.loads(unb64u(cred["response"]["clientDataJSON"]))["challenge"]
            except Exception:  # noqa: BLE001
                return self.error(400, "credential_invalid")
            with mock.lock:
                known = ch in mock.logins
                mock.logins.discard(ch)
            if not known:
                return self.error(400, "challenge_expired")
            why = self.check_credential(cred, "webauthn.get", ch)
            if why:
                return self.error(400, why)
            if cred.get("id") not in mock.creds:
                return self.error(400, "credential_unknown")
            sid = secrets.token_urlsafe(24)
            with mock.lock:
                mock.sessions[sid] = {"account": mock.creds[cred["id"]], "verified": False, "login": True}
            return self.json(200, {"ok": True}, {"Set-Cookie": self.cookie(sid)})
        if name == "logout":
            with mock.lock:
                mock.sessions.pop(sid, None)
            return self.json(200, {"ok": True}, {"Set-Cookie": self.cookie("", 0)})
        if not sess or not sess.get("login") or sess["account"] not in mock.accounts:
            return self.error(401, "session_required")
        acct = mock.accounts[sess["account"]]
        if name == "me":
            return self.json(200, {"name": acct["name"], "created": acct["created"], "tier": "key", "usage": mock.usage()})
        if name == "rotate":
            with mock.lock:
                key = mock.new_key(sess["account"])
            return self.json(200, {"key": key})
        if name == "revoke":
            with mock.lock:
                for c in acct["creds"]:
                    mock.creds.pop(c, None)
                mock.accounts.pop(sess["account"], None)
                mock.sessions.pop(sid, None)
            return self.json(200, {"revoked": True}, {"Set-Cookie": self.cookie("", 0)})
        return self.error(404, "not_found")

    def check_credential(self, cred, kind: str, challenge: str) -> str:
        """The parts the page builds: the JSON form's fields and clientDataJSON (P2 verifies the signatures)."""
        try:
            if not isinstance(cred, dict) or cred.get("type") != "public-key":
                return "credential_invalid"
            if unb64u(cred["rawId"]) != unb64u(cred["id"]):
                return "credential_invalid"
            r = cred["response"]
            client = json.loads(unb64u(r["clientDataJSON"]))
            need = ["attestationObject"] if kind == "webauthn.create" else ["authenticatorData", "signature"]
            for f in need:
                unb64u(r[f])
            if client.get("type") != kind or client.get("challenge") != challenge:
                return "challenge_invalid"
            if client.get("origin") != self.origin():
                return "origin_invalid"
        except (KeyError, TypeError, ValueError):
            return "credential_invalid"
        return ""


# ---------- the HTML ----------

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
KNOWN = VOID | {"html", "head", "body", "title", "script", "noscript", "style", "header", "footer", "main", "nav", "section",
                "article", "aside", "h1", "h2", "h3", "h4", "p", "a", "span", "strong", "em", "code", "pre", "kbd", "ul", "ol", "li",
                "dl", "dt", "dd", "table", "thead", "tbody", "tr", "th", "td", "div", "button", "output", "label", "small", "svg",
                "path", "details", "summary", "abbr", "time", "blockquote", "figure", "figcaption", "sup", "sub"}


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.problems, self.ids, self.refs, self.links, self.order = [], [], set(), [], [], []
        self.doctype = False
        self.first = None

    def handle_decl(self, decl):
        if decl.lower() == "doctype html" and self.first is None:
            self.doctype = True
        self.first = self.first or "decl"

    def handle_starttag(self, tag, attrs):
        self.first = self.first or tag
        self.order.append((tag, dict(attrs)))
        names = [a for a, _ in attrs]
        if len(names) != len(set(names)):
            self.problems.append(f"<{tag}> repeats an attribute")
        a = dict(attrs)
        if tag not in KNOWN:
            self.problems.append(f"unknown element <{tag}>")
        if "id" in a:
            if a["id"] in self.ids:
                self.problems.append(f"duplicate id {a['id']}")
            self.ids.add(a["id"])
        for k, v in attrs:
            if k.startswith("on"):
                self.problems.append(f"<{tag} {k}>: an inline event handler (the CSP forbids it)")
            if k == "style":
                self.problems.append(f"<{tag} style>: an inline style (the CSP forbids it)")
            if k in ("href", "src", "action", "formaction", "poster", "data") and v is not None:
                self.links.append((tag, k, v))
            if k in ("aria-labelledby", "aria-controls", "aria-describedby", "for", "data-copy") and v:
                self.refs.extend((k, x) for x in v.split())
        if tag == "script" and "src" not in a:
            self.problems.append("an inline <script> (the CSP forbids it)")
        if tag == "style":
            self.problems.append("a <style> element (the CSP forbids it)")
        if tag == "img" and "alt" not in a:
            self.problems.append("<img> without alt")
        if tag == "a" and "href" not in a:
            self.problems.append("<a> without href")
        if tag == "button" and a.get("type") not in ("button", "submit", "reset"):
            self.problems.append("<button> without a type")
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        if tag not in VOID:
            self.problems.append(f"<{tag}/> is not a void element")
        self.handle_starttag(tag, attrs)
        if tag not in VOID and self.stack and self.stack[-1] == tag:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in VOID:
            self.problems.append(f"</{tag}>: a void element has no end tag")
            return
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"</{tag}> does not close <{self.stack[-1] if self.stack else '-'}>")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
            return
        self.stack.pop()


def check_html(name: str, text: str) -> tuple[list, Page]:
    p = Page()
    p.feed(text)
    p.close()
    out = [f"{name}: {x}" for x in p.problems]
    if not p.doctype:
        out.append(f"{name}: no <!doctype html> first")
    if p.stack:
        out.append(f"{name}: not closed: {p.stack}")
    tags = [t for t, _ in p.order]
    html = next((a for t, a in p.order if t == "html"), {})
    if html.get("lang") != "ko":
        out.append(f"{name}: <html lang> is not ko")
    metas = [a for t, a in p.order if t == "meta"]
    if not metas or metas[0].get("charset", "").lower() != "utf-8":
        out.append(f"{name}: the first <meta> is not charset=utf-8")
    if not any(a.get("name") == "viewport" and "width=device-width" in a.get("content", "") for a in metas):
        out.append(f"{name}: no viewport meta")
    for t in ("title", "main", "h1"):
        if tags.count(t) != 1:
            out.append(f"{name}: {tags.count(t)} <{t}> (want 1)")
    for kind, ref in p.refs:
        if ref not in p.ids:
            out.append(f"{name}: {kind}={ref} names no element")
    return out, p


EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.I)


def check_sources() -> list:
    """No stylesheet or script names another origin, and the scripts keep away from storage and the console."""
    out = []
    for f in sorted((SITE / "assets").iterdir()):
        text = f.read_text(encoding="utf-8")
        if f.suffix == ".css":
            if re.search(r"@import|url\(", text):
                out.append(f"{f.name}: @import or url() (the pages load nothing but their own files)")
        if f.suffix == ".js":
            code = re.sub(r"/\*.*?\*/", "", text, flags=re.S)  # the comments (no script has /* in a string)
            code = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith("//"))
            for m in re.finditer(r"https?://|wss?://|[\"'`]//", code):
                around = code[max(0, m.start() - 60):m.end() + 40]
                if "ssalmuck-ai.duckdns.org/keys" not in around:  # the error text that names this site's own address
                    out.append(f"{f.name}: names an origin: ...{around.strip()}...")
            for bad in ("localStorage", "sessionStorage", "indexedDB", "document.cookie", "console.", "eval(", "new Function",
                        "innerHTML", "outerHTML", "document.write", "insertAdjacentHTML", "XMLHttpRequest", "import("):
                if bad in code:
                    out.append(f"{f.name}: uses {bad}")
        if f.suffix == ".svg":
            if re.search(r"(?:href|src)\s*=\s*[\"'](?!#)", text) or "<script" in text:
                out.append(f"{f.name}: a link or script in the icon")
    return out


# ---------- through the stand-in edge ----------

def get(port: int, path: str, method: str = "GET", body: bytes | None = None, headers: dict | None = None):
    c = HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, body=body, headers=dict({"Host": f"localhost:{port}"}, **(headers or {})))
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, {k.lower(): v for k, v in r.getheaders()}, data, r.msg.get_all("Set-Cookie") or []


def check_links(port: int) -> list:
    out, pages = [], {}
    for route, name in list(PAGES.items()) + [("/404-check", "404.html")]:
        text = (SITE / name).read_text(encoding="utf-8")
        problems, p = check_html(name, text)
        out += problems
        pages[name] = p
    for name, p in pages.items():
        for tag, attr, v in p.links:
            if v.startswith("mailto:"):
                if v != CONTACT:
                    out.append(f"{name}: <{tag} {attr}={v}>: not the contact address")
                continue
            if EXTERNAL.match(v):
                out.append(f"{name}: <{tag} {attr}={v}>: another origin")
                continue
            path, _, frag = v.partition("#")
            if not path:
                if frag and frag not in p.ids:
                    out.append(f"{name}: #{frag} names no element")
                continue
            status, h, data, _ = get(port, path)
            if path in ("/keys/google/start",):
                if status != 302:
                    out.append(f"{name}: {v} -> {status} (want the mock's 302 to Google)")
                continue
            if status != 200:
                out.append(f"{name}: {v} -> {status}")
                continue
            if frag:
                target = next((pp for nn, pp in pages.items() if PAGES.get(path) == nn), None)
                if target is None or frag not in target.ids:
                    out.append(f"{name}: {v}: no element #{frag} there")
    return out


def check_routes(port: int) -> list:
    out = []
    conf = caddy_config()
    csp = conf["headers"]["sclmd_page"]["Content-Security-Policy"]
    want = [  # path, method, status, a header that must be there (name, value or None), body fragment
        ("/", "GET", 200, ("cache-control", "no-cache"), "<h1>"),
        ("/keys", "GET", 200, ("content-security-policy", csp), 'id="keys-app"'),
        ("/guide", "GET", 200, ("permissions-policy", None), "사용 안내"),
        ("/trust", "GET", 200, ("x-frame-options", "DENY"), "신뢰"),
        ("/terms", "GET", 200, ("referrer-policy", "same-origin"), "이용약관"),
        ("/privacy", "GET", 200, ("strict-transport-security", "max-age=31536000"), "개인정보"),
        ("/keys/", "GET", 308, ("location", "/keys"), None),
        ("/terms/", "GET", 308, ("location", "/terms"), None),
        ("/assets/site.css", "GET", 200, ("cache-control", "public, max-age=3600"), None),
        ("/assets/keys.js", "GET", 200, ("content-type", "text/javascript; charset=utf-8"), None),
        ("/assets/", "GET", 404, None, None),
        ("/index.html", "GET", 404, ("content-security-policy", csp), "찾을 수 없는"),
        ("/keys.html", "GET", 404, None, None),
        ("/404.html", "GET", 404, None, None),
        ("/KEYS", "GET", 404, None, None),
        ("/metrics", "GET", 404, None, None),
        ("/debug/audit", "GET", 404, None, None),
        ("/v1", "GET", 404, None, None),
        ("/robots.txt", "GET", 404, None, None),
        ("/assets/../keys.html", "GET", 404, None, None),
        ("/keys", "POST", 405, None, None),
        ("/v1/status", "GET", 200, ("content-type", "application/json"), '"limits"'),
        ("/health", "GET", 200, None, None),
        ("/keys/google/start", "GET", 302, ("location", None), None),
        ("/keys/api/me", "GET", 405, None, None),
    ]
    for path, method, status, header, frag in want:
        s, h, data, _ = get(port, path, method)
        if s != status:
            out.append(f"{method} {path}: {s} (want {status})")
            continue
        if "server" in h:
            out.append(f"{method} {path}: a Server header")
        if header:
            k, v = header
            if k not in h or (v is not None and h[k] != v):
                out.append(f"{method} {path}: {k} = {h.get(k)!r} (want {v!r})")
        if frag and frag.encode() not in data:
            out.append(f"{method} {path}: no {frag!r} in the body")
        if s == 200 and "text/html" in h.get("content-type", ""):
            for k in ("content-security-policy", "strict-transport-security", "x-content-type-options", "referrer-policy",
                      "permissions-policy", "cross-origin-opener-policy"):
                if k not in h:
                    out.append(f"{method} {path}: no {k}")
    # the CSP itself: this origin only, nothing inline
    for part in ("default-src 'none'", "script-src 'self'", "style-src 'self'", "connect-src 'self'", "frame-ancestors 'none'",
                 "base-uri 'none'"):
        if part not in csp:
            out.append(f"the CSP has no {part!r}")
    if "unsafe" in csp or "http" in csp or "*" in csp:
        out.append("the CSP allows more than this origin")
    if "-Server" not in conf["headers"]["sclmd_headers"]:
        out.append("the edge does not drop the Server header")
    if conf["upstream"] != "192.168.0.2:8001":
        out.append(f"reverse_proxy {conf['upstream']} (want 192.168.0.2:8001)")
    for path in ("/v1/chat/completions", "/v1/models", "/health", "/keys/google/start", "/keys/callback/google", "/keys/api/me"):
        if not caddy_match(conf["matchers"]["api"], path):
            out.append(f"{path} is not proxied")
    for path in ("/metrics", "/debug", "/", "/keys", "/keys/google", "/v1"):
        if caddy_match(conf["matchers"]["api"], path):
            out.append(f"{path} is proxied")
    return out


def check_flow(port: int) -> list:
    """The mock contract from Google to a key and back, as a browser would walk it (the page's JS: node and a browser)."""
    out = []
    origin = f"http://localhost:{port}"
    api = lambda name, body=None, cookie="", hdr=None: get(  # noqa: E731
        port, "/keys/api/" + name, "POST", json.dumps(body or {}).encode(),
        dict({"Content-Type": "application/json", "X-Requested-With": "sclmd", "Origin": origin, "Cookie": cookie}, **(hdr or {})))

    def cred(kind, challenge, cid="cred-1"):
        client = b64u(json.dumps({"type": kind, "challenge": challenge, "origin": origin}).encode())
        r = {"clientDataJSON": client}
        if kind == "webauthn.create":
            r.update(attestationObject=b64u(b"attestation"), transports=["internal"])
        else:
            r.update(authenticatorData=b64u(b"data"), signature=b64u(b"sig"), userHandle=None)
        return {"id": b64u(cid.encode()), "rawId": b64u(cid.encode()), "type": "public-key", "response": r, "clientExtensionResults": {}}

    s, h, _, _ = get(port, "/keys/google/start")
    g = urllib.parse.urlsplit(h.get("location", ""))
    q = dict(urllib.parse.parse_qsl(g.query))
    if s != 302 or q.get("scope") != "openid" or q.get("code_challenge_method") != "S256" or not q.get("nonce"):
        return [f"/keys/google/start: {s} {h.get('location')}"]
    c = HTTPConnection(g.hostname, g.port, timeout=10)
    c.request("GET", g.path + "?" + g.query)
    back = urllib.parse.urlsplit(c.getresponse().getheader("Location"))
    c.close()
    s, h, _, cookies = get(port, back.path + "?" + back.query)
    if s != 302 or h.get("location") != "/keys#verified" or not cookies:
        return [f"callback: {s} {h.get('location')}"]
    ck = cookies[0]
    for flag in ("HttpOnly", "Secure", "SameSite=Lax", "Path=/keys", "Max-Age=900"):
        if flag not in ck:
            out.append(f"session cookie without {flag}")
    cookie = ck.split(";")[0]
    # CSRF: no X-Requested-With, another Origin, no Origin
    for hdr in ({"X-Requested-With": ""}, {"Origin": "https://evil.example"}, {"Origin": "null"}):
        if api("passkey/register/options", cookie=cookie, hdr=hdr)[0] != 403:
            out.append(f"register/options with {hdr}: not 403")
    s, _, data, _ = api("passkey/register/options", cookie=cookie)
    opts = json.loads(data)
    if s != 200 or opts["authenticatorSelection"]["residentKey"] != "required":
        return out + [f"register/options: {s}"]
    s, _, data, _ = api("passkey/register/verify", {"credential": cred("webauthn.create", opts["challenge"])}, cookie)
    res = json.loads(data)
    if s != 200 or not res.get("key", "").startswith("sk-") or res.get("replaced") is not False:
        return out + [f"register/verify: {s} {sorted(res)}"]
    s, _, data, _ = api("me", cookie=cookie)
    if s != 200 or json.loads(data).get("tier") != "key":
        out.append(f"me after register: {s}")
    s, _, data, _ = api("rotate", cookie=cookie)
    if s != 200 or json.loads(data).get("key") == res["key"]:
        out.append(f"rotate: {s}")
    if api("logout", cookie=cookie)[0] != 200 or api("me", cookie=cookie)[0] != 401:
        out.append("logout did not end the session")
    s, _, data, _ = api("passkey/login/options")
    ch = json.loads(data)["challenge"]
    s, h, data, cookies = api("passkey/login/verify", {"credential": cred("webauthn.get", ch)})
    if s != 200 or not cookies:
        return out + [f"login/verify: {s} {data[:80]!r}"]
    cookie = cookies[0].split(";")[0]
    if api("me", cookie=cookie)[0] != 200:
        out.append("me after login: not 200")
    if api("revoke", cookie=cookie)[0] != 200 or api("me", cookie=cookie)[0] != 401:
        out.append("revoke did not end the key")
    s, _, data, _ = api("passkey/login/options")
    s, _, data, _ = api("passkey/login/verify", {"credential": cred("webauthn.get", json.loads(data)["challenge"])})
    if s != 400 or json.loads(data)["error"]["code"] != "credential_unknown":
        out.append(f"login with a revoked passkey: {s}")
    return out


def check_node() -> list:
    node = shutil.which("node")
    if not node:
        return []
    r = subprocess.run([node, str(NODE_CHECK), str(SITE / "assets")], capture_output=True, text=True, encoding="utf-8", timeout=60)
    out = [line for line in r.stdout.splitlines() if line.startswith("FAIL")]
    if r.returncode and not out:
        out.append(f"node: exit {r.returncode}: {r.stderr.strip()[-300:]}")
    # the fallback numbers in site.js are the guide's and the mock's (6.20)
    m = re.search(r"^FALLBACK (.*)$", r.stdout, re.M)
    if not m or json.loads(m.group(1)) != STATUS["limits"]:
        out.append("site.js's FALLBACK is not 6.20's numbers")
    return out


def check_fallback_text() -> list:
    """The guide's numbers before GET /v1/status answers are 6.20's."""
    text = (SITE / "guide.html").read_text(encoding="utf-8")
    want = {"key.tph": "125만", "key.tpd": "4,000만", "ip_verified.tph": "75만", "ip_verified.tpd": "1,500만",
            "ip_unverified.tph": "30만", "ip_unverified.tpd": "750만", "output_weight": "5", "cached_weight": "0.1",
            "concurrency.priority": "1", "concurrency.normal": "3", "concurrency.low": "8"}
    out = []
    for k, v in want.items():
        m = re.search(r'data-limit="' + re.escape(k) + r'"[^>]*>([^<]*)<', text)
        if not m or m.group(1) != v:
            out.append(f"guide.html: {k} shows {m.group(1) if m else None!r} (want {v!r})")
    if not re.search(r"data-limit-refused>13<", text):
        out.append("guide.html: the refused request is not the 13th")
    return out


def run_all(port: int = 8124) -> dict:
    edge = Edge(port, port + 1).start()
    try:
        results = {"html and links": check_links(port), "sources": check_sources(), "routes": check_routes(port),
                   "flow": check_flow(port), "fallback text": check_fallback_text(), "node": check_node()}
    finally:
        edge.stop()
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", type=int, metavar="PORT", help="serve the stand-in edge on 127.0.0.1:PORT (Google on PORT+1) until Ctrl+C")
    ap.add_argument("--port", type=int, default=8124, help="the checks' port (and the next one), default 8124")
    ap.add_argument("--tier", default="ip_unverified", help="the caller's tier GET /v1/status reports (--serve)")
    a = ap.parse_args(argv)
    if a.serve:
        edge = Edge(a.serve, a.serve + 1, a.tier).start()
        print(f"[site] the stand-in edge on http://localhost:{a.serve} (Google stand-in on {a.serve + 1}); Ctrl+C stops", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            edge.stop()
        return 0
    results = run_all(a.port)
    bad = 0
    for name, problems in results.items():
        print(f"{name}: {'ok' if not problems else str(len(problems)) + ' problem(s)'}")
        for p in problems:
            print("  " + p)
        bad += len(problems)
    if not shutil.which("node"):
        print("node: not installed, its checks did not run")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
