# Key issuance (scripts/serve_keys.py; P2, design.md 6.22) without a model, a GPU or Google: a stand-in Google
# (an OpenID provider on this PC that signs id_tokens with a test RSA key and serves its JWKS) and software
# authenticators (ES256, RS256, Ed25519 passkeys made here with cryptography) against the stand-in server.
import base64
import hashlib
import http.client
import http.server
import json
import os
import re
import socket
import struct
import threading
import time
import urllib.parse

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

from scripts import serve as S
from scripts import serve_admin as A
from scripts import serve_contract as C
from scripts import serve_keys as KS
from test_api_contract import Tok
from test_serve_admin import HI, Inbox, answer, chat_async, options

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTS = range(8110, 8120)  # this task's ports (P2-keys)
ORIGIN = "http://localhost:8110"  # the keys page's origin as the browser would send it (rp id localhost)
RP = "localhost"
CLIENT_ID = "p2-test-client.apps.example"
CLIENT_SECRET = "GOCSPX-canary-client-secret-q7z1"  # a fake client file's secret: must appear in no log or file
FILE_KEYS = [("owner", b"sk-owner-0123456789abcdef")]


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def enc(x) -> bytes:
    """CBOR, enough for an authenticator's output."""

    def head(major, n):
        if n < 24:
            return bytes([major << 5 | n])
        for info, size in ((24, 1), (25, 2), (26, 4), (27, 8)):
            if n < 1 << (8 * size):
                return bytes([major << 5 | info]) + n.to_bytes(size, "big")
        raise ValueError(n)

    if isinstance(x, bool):
        return bytes([0xF5 if x else 0xF4])
    if isinstance(x, int):
        return head(0, x) if x >= 0 else head(1, -1 - x)
    if isinstance(x, bytes):
        return head(2, len(x)) + x
    if isinstance(x, str):
        b = x.encode()
        return head(3, len(b)) + b
    if isinstance(x, list):
        return head(4, len(x)) + b"".join(enc(v) for v in x)
    if isinstance(x, dict):
        return head(5, len(x)) + b"".join(enc(k) + enc(v) for k, v in x.items())
    raise TypeError(type(x))


def bind(server_class, handler):
    """A server on the first free port of this task's (a port something listens on is skipped: on Windows the
    servers' SO_REUSEADDR would let a second one bind it too)."""
    for port in PORTS:
        probe = socket.socket()
        probe.settimeout(0.5)
        try:
            taken = probe.connect_ex(("127.0.0.1", port)) == 0
        finally:
            probe.close()
        if taken:
            continue
        try:
            return server_class(("127.0.0.1", port), handler)
        except OSError:
            continue
    raise RuntimeError("no free port in 8110-8119")


# the stand-in Google

class FakeGoogle:
    """An OpenID provider: /token exchanges a code made by consent() (client id and secret, redirect URI, PKCE S256),
    /certs serves the JWKS of its RSA key."""

    def __init__(self):
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.kid = "test-kid-1"
        self.codes, self.certs_fetched, self.tokens = {}, 0, []
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, status, obj, headers=()):
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in headers:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/certs":
                    fake.certs_fetched += 1
                    n = fake.key.public_key().public_numbers()
                    return self.reply(200, {"keys": [{"kty": "RSA", "alg": "RS256", "use": "sig", "kid": fake.kid, "n": b64u(n.n.to_bytes(256, "big")),
                                                      "e": b64u(n.e.to_bytes(3, "big"))}]}, [("Cache-Control", "public, max-age=3600")])
                self.reply(404, {})

            def do_POST(self):
                form = urllib.parse.parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
                one = {k: v[0] for k, v in form.items()}
                entry = fake.codes.pop(one.get("code"), None)
                ok = (entry is not None and one.get("grant_type") == "authorization_code" and one.get("client_id") == CLIENT_ID
                      and one.get("client_secret") == CLIENT_SECRET and one.get("redirect_uri") == entry["redirect_uri"]
                      and b64u(hashlib.sha256(one.get("code_verifier", "").encode()).digest()) == entry["challenge"])
                if not ok:
                    return self.reply(400, {"error": "invalid_grant"})
                self.reply(200, {"id_token": entry["token"], "access_token": "ya29.not-kept", "token_type": "Bearer", "expires_in": 3599})

        self.httpd = bind(http.server.ThreadingHTTPServer, H)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def jwt(self, claims, key=None, kid=None, alg="RS256") -> str:
        head = b64u(json.dumps({"alg": alg, "kid": kid or self.kid, "typ": "JWT"}).encode())
        body = b64u(json.dumps(claims).encode())
        sig = (key or self.key).sign(f"{head}.{body}".encode(), padding.PKCS1v15(), hashes.SHA256()) if alg == "RS256" else b""
        return f"{head}.{body}.{b64u(sig)}"

    def consent(self, location: str, who: str, token=None, **claims) -> str:
        """The person signs in at Google and consents: -> the code Google sends back to the redirect URI."""
        q = {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(location).query).items()}
        assert q["scope"] == "openid" and q["response_type"] == "code" and q["code_challenge_method"] == "S256" and q["client_id"] == CLIENT_ID
        now = int(time.time())
        c = {"iss": "https://accounts.google.com", "aud": CLIENT_ID, "azp": CLIENT_ID, "sub": who, "iat": now, "exp": now + 3600, "nonce": q["nonce"]}
        c.update(claims)
        tok = token if token is not None else self.jwt({k: v for k, v in c.items() if v is not None})
        self.tokens.append(tok)
        code = "4/0A-canary-code-" + os.urandom(8).hex()
        self.codes[code] = {"redirect_uri": q["redirect_uri"], "challenge": q["code_challenge"], "token": tok}
        return code


# a software authenticator

class Authenticator:
    """A passkey provider: one credential an rp id, discoverable, signing with ES256, RS256 or Ed25519."""

    def __init__(self, alg=-7, counting=False):
        self.alg, self.counting = alg, counting
        self.key = {-7: lambda: ec.generate_private_key(ec.SECP256R1()), -257: lambda: rsa.generate_private_key(65537, 2048),
                    -8: ed25519.Ed25519PrivateKey.generate}[alg]()
        self.cred, self.user, self.count = None, None, 0

    def cose(self) -> dict:
        pub = self.key.public_key()
        if self.alg == -7:
            n = pub.public_numbers()
            return {1: 2, 3: -7, -1: 1, -2: n.x.to_bytes(32, "big"), -3: n.y.to_bytes(32, "big")}
        if self.alg == -257:
            n = pub.public_numbers()
            return {1: 3, 3: -257, -1: n.n.to_bytes(256, "big"), -2: n.e.to_bytes(3, "big")}
        return {1: 1, 3: -8, -1: 6, -2: pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)}

    def sign(self, data: bytes) -> bytes:
        if self.alg == -7:
            return self.key.sign(data, ec.ECDSA(hashes.SHA256()))
        if self.alg == -257:
            return self.key.sign(data, padding.PKCS1v15(), hashes.SHA256())
        return self.key.sign(data)

    @staticmethod
    def client(kind, challenge, origin, **more) -> bytes:
        return json.dumps({"type": kind, "challenge": challenge, "origin": origin, "crossOrigin": False, **more}).encode()

    def create(self, options: dict, origin=ORIGIN, rp=None, flags=0x45, challenge=None, fmt="none") -> dict:
        assert options["authenticatorSelection"]["residentKey"] == "required" and options["attestation"] == "none"
        assert {p["alg"] for p in options["pubKeyCredParams"]} >= {-7, -257}
        self.cred, self.user = os.urandom(32), options["user"]["id"]
        if self.counting:
            self.count += 1
        auth = (hashlib.sha256((rp or options["rp"]["id"]).encode()).digest() + bytes([flags]) + struct.pack(">I", self.count) + bytes(16)
                + struct.pack(">H", len(self.cred)) + self.cred + enc(self.cose()))
        cdj = self.client("webauthn.create", challenge or options["challenge"], origin)
        att = enc({"fmt": fmt, "attStmt": {}, "authData": auth})
        return {"credential": {"id": b64u(self.cred), "rawId": b64u(self.cred), "type": "public-key",
                               "response": {"clientDataJSON": b64u(cdj), "attestationObject": b64u(att), "transports": ["internal"]},
                               "clientExtensionResults": {}, "authenticatorAttachment": "platform"}}

    def get(self, options: dict, origin=ORIGIN, rp=None, flags=0x05, tamper=False, count=None, user=None) -> dict:
        assert options["allowCredentials"] == [] and options["userVerification"] == "preferred"
        if self.counting:
            self.count += 1
        auth = hashlib.sha256((rp or options["rpId"]).encode()).digest() + bytes([flags]) + struct.pack(">I", self.count if count is None else count)
        cdj = self.client("webauthn.get", options["challenge"], origin)
        sig = self.sign(auth + hashlib.sha256(cdj).digest())
        if tamper:
            sig = sig[:-1] + bytes([sig[-1] ^ 1])
        return {"credential": {"id": b64u(self.cred), "rawId": b64u(self.cred), "type": "public-key",
                               "response": {"clientDataJSON": b64u(cdj), "authenticatorData": b64u(auth), "signature": b64u(sig),
                                            "userHandle": self.user if user is None else user}}}


# the stand-in server with --keys-store

class Server:
    def __init__(self, tmp, fake, *argv, keys=FILE_KEYS, per_minute=1000):
        self.tmp = tmp
        client = tmp / "client.json"
        client.write_text(json.dumps({"web": {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "project_id": "p2-test",
                                               "auth_uri": f"{fake.url}/auth", "token_uri": f"{fake.url}/token",
                                               "redirect_uris": [ORIGIN + KS.REDIRECT_PATH]}}), encoding="utf-8")
        self.cli = options("--anonymous", "--trusted-proxies", "127.0.0.1", "--keys-store", str(tmp / "keys" / "store.json"), "--oauth-file", str(client),
                           "--public-origin", ORIGIN, "--oauth-jwks", f"{fake.url}/certs", "--request-log", str(tmp / "requests.jsonl"),
                           "--status-file", str(tmp / "status.json"), *argv)
        KS.check(self.cli, list(keys))
        self.httpd = bind(C.BoundedHTTPServer, S.Handler)
        self.inbox = Inbox()
        self.app = self.httpd.app = S.App(self.inbox, Tok(), False, 64, {"bits_per_weight": 3.0})
        A.install(self.app, self.inbox, self.cli, self.httpd, list(keys))
        self.issuer = self.app.admin.issuer
        self.issuer.limiter.per_minute = per_minute
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def call(self, method, path, body=None, cookie=None, origin=ORIGIN, xrw="sclmd", xff=None, raw=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=20)
        h = {**({"Content-Type": "application/json"} if method == "POST" else {}), **(headers or {})}
        if origin is not None and method == "POST":
            h["Origin"] = origin
        if xrw is not None and method == "POST":
            h["X-Requested-With"] = xrw
        if cookie:
            h["Cookie"] = cookie
        if xff:
            h["X-Forwarded-For"] = xff
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None))
        conn.request(method, path, data, h)
        r = conn.getresponse()
        out = r.read()
        cookies = r.msg.get_all("Set-Cookie") or []
        heads = {k.lower(): v for k, v in r.getheaders()}
        conn.close()
        return r.status, heads, cookies, (json.loads(out) if out and heads.get("content-type", "").startswith("application/json") else out)

    def chat(self, key: str | None) -> int:
        """POST /v1/chat/completions with this key -> the status (the inbox answers what comes)."""
        th, out = chat_async(self.httpd, {"messages": HI}, {"Authorization": f"Bearer {key}"} if key else {})
        t = time.time()
        while th.is_alive():
            if self.inbox.reqs:
                req = self.inbox.reqs.pop(0)
                self.last = req
                answer(req)
            assert time.time() - t < 15
            time.sleep(0.01)
        return out["status"]


def cookie_of(set_cookies, name):
    for c in set_cookies:
        if c.startswith(name + "="):
            return c.split(";")[0].split("=", 1)[1], c
    return None, None


@pytest.fixture()
def google():
    fake = FakeGoogle()
    yield fake
    fake.close()


@pytest.fixture()
def made(tmp_path, google):
    servers = []

    def make(*argv, tmp=None, **kw):
        servers.append(Server(tmp or tmp_path, google, *argv, **kw))
        return servers[-1]

    yield make
    for s in servers:
        s.close()


def check_with_google(srv, google, who, xff=None, **claims):
    """GET /keys/google/start, Google's consent, the callback -> (the session cookie or None, the redirect's fragment)."""
    st, h, sc, _ = srv.call("GET", "/keys/google/start", xff=xff)
    assert st == 302, (st, h)
    loc = h["location"]
    if not loc.startswith(google.url):
        return None, loc.split("#", 1)[1].removeprefix("error=")
    flow, line = cookie_of(sc, KS.FLOW_COOKIE)
    assert "HttpOnly" in line and "Secure" in line and "SameSite=Lax" in line and "Path=/keys/callback/google" in line
    state = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(loc).query))["state"]
    code = google.consent(loc, who, **claims)
    st, h, sc, _ = srv.call("GET", f"/keys/callback/google?state={state}&code={urllib.parse.quote(code)}&scope=openid", cookie=f"{KS.FLOW_COOKIE}={flow}", xff=xff)
    assert st == 302 and h["cache-control"] == "no-store"
    sid, line = cookie_of(sc, KS.COOKIE)
    frag = h["location"].split("#", 1)[1].removeprefix("error=")
    assert h["location"].startswith(ORIGIN + "/keys#")
    if sid:
        assert line.endswith("Path=/keys; Max-Age=900; HttpOnly; Secure; SameSite=Lax") and frag == "verified"
    return (f"{KS.COOKIE}={sid}" if sid else None), frag


def register(srv, cookie, auth, xff=None, **kw):
    st, _, _, opts = srv.call("POST", "/keys/api/passkey/register/options", cookie=cookie, xff=xff)
    assert st == 200, opts
    assert opts["rp"]["id"] == RP and opts["authenticatorSelection"]["userVerification"] == "preferred"
    return srv.call("POST", "/keys/api/passkey/register/verify", auth.create(opts, **kw), cookie=cookie, xff=xff)


def sign_in(srv, auth, xff=None, **kw):
    st, _, _, opts = srv.call("POST", "/keys/api/passkey/login/options", xff=xff)
    assert st == 200 and opts["rpId"] == RP
    st, _, sc, out = srv.call("POST", "/keys/api/passkey/login/verify", auth.get(opts, **kw), xff=xff)
    sid, _ = cookie_of(sc, KS.COOKIE)
    return st, out, (f"{KS.COOKIE}={sid}" if sid else None), opts


# the whole flow

@pytest.mark.parametrize("alg", [-7, -257, -8])
def test_the_whole_flow_google_once_then_the_passkey(made, google, alg):
    srv = made()
    cookie, frag = check_with_google(srv, google, "1000000000000000000001")
    assert frag == "verified"
    auth = Authenticator(alg)
    st, _, _, out = register(srv, cookie, auth)
    assert st == 200 and out["replaced"] is False and re.fullmatch(r"g-[0-9a-f]{12}", out["name"])
    key = out["key"]
    assert re.fullmatch(r"sk-[A-Za-z0-9_-]{43}", key)  # 32 random bytes
    assert srv.chat(key) == 200 and srv.last.tenant == f"key:{out['name']}"  # the server's key check takes it, no restart
    assert srv.chat("sk-owner-0123456789abcdef") == 200 and srv.last.tenant == "key:owner"  # the file's keys too
    assert srv.chat("sk-" + "A" * 43) == 401
    st, _, _, me = srv.call("POST", "/keys/api/me", cookie=cookie)
    assert st == 200 and me["name"] == out["name"] and me["tier"] == "key" and isinstance(me["created"], int)
    # sign out, then in with the passkey alone (Google not asked again)
    assert srv.call("POST", "/keys/api/logout", cookie=cookie)[0] == 200
    assert srv.call("POST", "/keys/api/me", cookie=cookie)[0] == 401
    tokens = len(google.tokens)
    st, ok, cookie, _ = sign_in(srv, auth)
    assert st == 200 and ok == {"ok": True} and cookie and len(google.tokens) == tokens
    assert srv.call("POST", "/keys/api/me", cookie=cookie)[3]["name"] == out["name"]
    # rotate: the old key refused at once, the new one served
    st, _, _, rot = srv.call("POST", "/keys/api/rotate", cookie=cookie)
    assert st == 200 and set(rot) == {"key"} and rot["key"] != key
    assert srv.chat(key) == 401 and srv.chat(rot["key"]) == 200 and srv.last.tenant == f"key:{out['name']}"  # the same name: the same quota
    # revoke: key, passkeys and the account's HMAC deleted
    st, _, sc, rev = srv.call("POST", "/keys/api/revoke", cookie=cookie)
    assert st == 200 and rev == {"revoked": True} and cookie_of(sc, KS.COOKIE)[1].startswith(f"{KS.COOKIE}=;")
    assert srv.chat(rot["key"]) == 401
    store = json.loads((srv.tmp / "keys" / "store.json").read_text())
    assert store["accounts"] == {}
    assert sign_in(srv, auth)[0] == 400  # the passkey is gone too
    assert srv.call("POST", "/keys/api/me", cookie=cookie)[0] == 401  # and the session


def test_a_second_google_check_revokes_the_old_key_and_passkey(made, google):
    srv = made()
    cookie, _ = check_with_google(srv, google, "sub-two")
    a1 = Authenticator()
    first = register(srv, cookie, a1)[3]
    lost = sign_in(srv, a1)[2]  # the device that is lost next, still signed in
    assert srv.call("POST", "/keys/api/me", cookie=lost)[0] == 200
    cookie2, _ = check_with_google(srv, google, "sub-two")  # the same Google account again (a lost device)
    a2 = Authenticator()
    st, _, _, second = register(srv, cookie2, a2)
    assert st == 200 and second["replaced"] is True and second["name"] == first["name"]
    # the lost device's session ends with the replacement: it cannot rotate (a fresh key) or revoke (verify-P2-keys)
    assert srv.call("POST", "/keys/api/rotate", cookie=lost)[0] != 200 and srv.call("POST", "/keys/api/revoke", cookie=lost)[0] != 200
    assert srv.chat(second["key"]) == 200
    assert srv.chat(first["key"]) == 401 and srv.chat(second["key"]) == 200
    rec = next(iter(json.loads((srv.tmp / "keys" / "store.json").read_text())["accounts"].values()))
    assert len(rec["passkeys"]) == 1 and rec["passkeys"][0]["id"] == b64u(a2.cred)
    # another Google account is another name
    cookie3, _ = check_with_google(srv, google, "sub-three")
    assert register(srv, cookie3, Authenticator())[3]["name"] != first["name"]


def test_the_store_survives_a_restart(tmp_path, made, google):
    srv = made()
    cookie, _ = check_with_google(srv, google, "sub-restart")
    auth = Authenticator(counting=True)
    out = register(srv, cookie, auth)[3]
    srv.close()
    again = made()  # the same store and secret, a new process's memory
    assert again.chat(out["key"]) == 200 and again.last.tenant == f"key:{out['name']}"
    st, _, cookie, _ = sign_in(again, auth)
    assert st == 200
    assert again.call("POST", "/keys/api/me", cookie=cookie)[3]["name"] == out["name"]
    secret = (tmp_path / "keys" / "store.json.secret").read_bytes()
    assert len(secret) == 32 and again.issuer.store.secret == secret
    again.close()
    (tmp_path / "keys" / "store.json.secret").unlink()  # a store without its secret: no start (a new one would split the accounts)
    with pytest.raises(SystemExit):
        made()


# refusals

def test_registration_refusals_replay_origin_rp_and_session(made, google):
    srv = made()
    st, _, _, err = srv.call("POST", "/keys/api/passkey/register/options")
    assert st == 401 and err["error"]["code"] == "no_session" and set(err["error"]) == {"message", "type", "param", "code"}
    cookie, _ = check_with_google(srv, google, "sub-refusals")
    auth = Authenticator()

    def options():
        return srv.call("POST", "/keys/api/passkey/register/options", cookie=cookie)[3]

    for kw, why in (({"origin": "https://evil.example"}, "another origin"), ({"rp": "evil.example"}, "another rp id"),
                    ({"flags": 0x44}, "present"), ({"challenge": b64u(b"x" * 32)}, "challenge")):
        st, _, _, err = srv.call("POST", "/keys/api/passkey/register/verify", auth.create(options(), **kw), cookie=cookie)
        assert st == 400 and err["error"]["code"] == "invalid_credential" and why in err["error"]["message"], (kw, err)
    opts = options()
    good = auth.create(opts)
    assert srv.call("POST", "/keys/api/passkey/register/verify", good, cookie=cookie)[0] == 200
    st, _, _, err = srv.call("POST", "/keys/api/passkey/register/verify", good, cookie=cookie)  # the same registration again
    assert st == 400 and "no registration under way" in err["error"]["message"]
    # a session from a passkey sign-in cannot register (the Google check comes first)
    st, _, pcookie, _ = sign_in(srv, auth)
    st, _, _, err = srv.call("POST", "/keys/api/passkey/register/options", cookie=pcookie)
    assert st == 403 and err["error"]["code"] == "google_check_needed"


def test_sign_in_refusals_replay_tamper_origin_user_and_count(made, google):
    srv = made()
    cookie, _ = check_with_google(srv, google, "sub-signin")
    auth = Authenticator(counting=True)
    register(srv, cookie, auth)
    st, _, _, opts = srv.call("POST", "/keys/api/passkey/login/options")
    good = auth.get(opts)
    assert srv.call("POST", "/keys/api/passkey/login/verify", good)[0] == 200
    st, _, _, err = srv.call("POST", "/keys/api/passkey/login/verify", good)  # replayed
    assert st == 400 and "challenge" in err["error"]["message"]
    for kw, why in (({"tamper": True}, "signature"), ({"origin": "https://evil.example"}, "another origin"), ({"rp": "evil.example"}, "rp id"),
                    ({"count": 1}, "sign count"), ({"user": b64u(b"someone else" * 3)}, "user"), ({"flags": 0x04}, "present")):
        st, out, c, _ = sign_in(srv, auth, **kw)
        assert st == 400 and why in out["error"]["message"] and c is None, (kw, out)
    unknown = Authenticator()
    unknown.cred, unknown.user = os.urandom(32), auth.user
    st, out, _, _ = sign_in(srv, unknown)
    assert st == 400 and "not known" in out["error"]["message"]


def test_csrf_origin_and_x_requested_with(made, google):
    srv = made()
    for kw in ({"origin": "https://evil.example"}, {"origin": None}, {"xrw": None}, {"xrw": "XMLHttpRequest"}):
        st, _, _, err = srv.call("POST", "/keys/api/passkey/login/options", **kw)
        assert st == 403 and err["error"]["code"] == "csrf", kw
    st, _, _, err = srv.call("POST", "/keys/api/passkey/login/options", headers={"Content-Type": "text/plain"})
    assert st == 415
    st, h, _, _ = srv.call("GET", "/keys/api/me")
    assert st == 405 and h["cache-control"] == "no-store"
    assert srv.call("POST", "/keys/api/nothing")[0] == 404
    st, h, _, _ = srv.call("POST", "/keys/api/me", raw=b"x" * (KS.BODY_MOST + 1))
    assert st == 413


def test_google_refusals_state_browser_nonce_aud_iss_exp_signature(made, google):
    srv = made()
    other = rsa.generate_private_key(65537, 2048)
    for claims, frag in (({"nonce": "another"}, "token"), ({"aud": "someone-else"}, "token"), ({"iss": "https://evil.example"}, "token"),
                         ({"exp": int(time.time()) - 3600}, "token"), ({"sub": None}, "token"), ({"aud": ["x", CLIENT_ID], "azp": "x"}, "token")):
        assert check_with_google(srv, google, "sub-g", **claims) == (None, frag), claims
    assert check_with_google(srv, google, "sub-g", iss="accounts.google.com")[1] == "verified"  # Google's other form
    assert check_with_google(srv, google, "sub-g", aud=[CLIENT_ID, "x"], azp=CLIENT_ID)[1] == "verified"
    now = int(time.time())
    for tok in (google.jwt({"iss": "https://accounts.google.com", "aud": CLIENT_ID, "sub": "x", "exp": now + 60}, key=other),  # not Google's key
                google.jwt({"iss": "https://accounts.google.com", "aud": CLIENT_ID, "sub": "x", "exp": now + 60}, alg="none"),
                google.jwt({"iss": "https://accounts.google.com", "aud": CLIENT_ID, "sub": "x", "exp": now + 60}, kid="unknown")):
        assert check_with_google(srv, google, "x", token=tok) == (None, "token")
    # the state: once, from the browser that started it
    st, h, sc, _ = srv.call("GET", "/keys/google/start")
    loc = h["location"]
    flow, _ = cookie_of(sc, KS.FLOW_COOKIE)
    state = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(loc).query))["state"]
    code = google.consent(loc, "sub-state")
    st, h, _, _ = srv.call("GET", f"/keys/callback/google?state={state}&code={code}", cookie=f"{KS.FLOW_COOKIE}=not-this-browser")
    assert h["location"].endswith("#error=browser")
    st, h, _, _ = srv.call("GET", f"/keys/callback/google?state={state}&code={code}", cookie=f"{KS.FLOW_COOKIE}={flow}")
    assert h["location"].endswith("#error=expired")  # used up by the refusal: no replay
    for q in ("", "?state=nothing&code=x", "?" + "&".join(f"a{i}=1" for i in range(40))):
        st, h, _, _ = srv.call("GET", f"/keys/callback/google{q}")
        assert st == 302 and h["location"].startswith(ORIGIN + "/keys#error="), q
    st, h, sc, _ = srv.call("GET", "/keys/google/start")
    state = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(h["location"]).query))["state"]
    flow, _ = cookie_of(sc, KS.FLOW_COOKIE)
    st, h, _, _ = srv.call("GET", f"/keys/callback/google?state={state}&error=access_denied", cookie=f"{KS.FLOW_COOKIE}={flow}")
    assert h["location"] == ORIGIN + "/keys#error=denied"
    assert google.certs_fetched == 1  # the JWKS cached (an unknown kid fetches again at most once a minute)


def test_keys_per_address_a_day(made, google):
    srv = made()
    ip = "198.51.100.7"
    names = set()
    for i in range(3):
        cookie, _ = check_with_google(srv, google, f"sub-ip-{i}", xff=ip)
        st, _, _, out = register(srv, cookie, Authenticator(), xff=ip)
        assert st == 200
        names.add(out["name"])
    cookie, frag = check_with_google(srv, google, "sub-ip-9", xff=ip)  # the fourth: refused at the start already
    assert (cookie, frag) == (None, "rate_limited")
    stranger = Authenticator()
    stranger.cred, stranger.user = os.urandom(32), b64u(os.urandom(32))
    st, _, _, out = sign_in(srv, stranger, xff=ip)  # (an unknown passkey: 400; signing in is not issuing)
    assert st == 400
    # a rotation is a key too
    cookie, _ = check_with_google(srv, google, "sub-ip-0", xff="198.51.100.8")  # another address checks the first account
    st, _, _, err = srv.call("POST", "/keys/api/rotate", cookie=cookie, xff=ip)
    assert st == 429 and err["error"]["code"] == "rate_limit_exceeded" and err["error"]["type"] == "requests"
    st, h, _, err = srv.call("POST", "/keys/api/passkey/register/options", cookie=cookie, xff=ip)
    assert st == 429 and 0 < int(h["retry-after"]) <= 86400
    assert srv.call("POST", "/keys/api/rotate", cookie=cookie, xff="198.51.100.8")[0] == 200  # another address is not counted with it
    assert srv.call("POST", "/keys/api/rotate", cookie=cookie, xff="2001:db8:1:2::5")[0] == 200
    assert srv.call("POST", "/keys/api/rotate", cookie=cookie, xff="2001:db8:1:2::6")[0] == 200
    assert srv.call("POST", "/keys/api/rotate", cookie=cookie, xff="2001:db8:1:2:ffff::1")[0] == 200  # (a /64 is one address: 3 then refused)
    st, _, _, err = srv.call("POST", "/keys/api/rotate", cookie=cookie, xff="2001:db8:1:2::7")
    assert st == 429
    # the count survives a restart (the store keeps HMACs of the addresses, 24 hours)
    srv.close()
    again = made()
    assert check_with_google(again, google, "sub-ip-10", xff=ip) == (None, "rate_limited")
    store = (srv.tmp / "keys" / "store.json").read_text()
    assert ip not in store and "2001:db8" not in store


def test_bounded_memory_flows_sessions_challenges_and_the_minute(made, google, monkeypatch):
    srv = made(per_minute=5)
    for _ in range(5):
        assert srv.call("POST", "/keys/api/passkey/login/options", xff="203.0.113.9")[0] == 200
    st, h, _, err = srv.call("POST", "/keys/api/passkey/login/options", xff="203.0.113.9")
    assert st == 429 and h["retry-after"] == "60"
    assert srv.call("GET", "/keys/google/start", xff="203.0.113.9")[1]["location"].endswith("#error=rate_limited")
    assert srv.call("POST", "/keys/api/passkey/login/options", xff="203.0.113.10")[0] == 200  # another address
    srv.issuer.limiter.per_minute = 0
    # a full table never refuses a newcomer: the address group holding the most loses its oldest (verify-P2-keys:
    # 4,096 challenges from 137 /64s locked every sign-in out)
    ch = srv.issuer.challenges
    for i in range(12):  # one address asks again and again: it holds PER_OWNER at most
        assert srv.call("POST", "/keys/api/passkey/login/options", xff="198.51.100.7")[0] == 200
    assert len(ch.owners["ip:198.51.100.7"]) == KS.PER_OWNER
    for n in range(40):  # an IPv6 site's /64s are one owner (its /48)
        assert srv.call("POST", "/keys/api/passkey/login/options", xff=f"2001:db8:7:{n:x}::1")[0] == 200
    assert len(ch.owners["ip:2001:db8:7::/48"]) == KS.PER_OWNER
    ch.most = len(ch)
    st, _, body, _ = srv.call("POST", "/keys/api/passkey/login/options", xff="203.0.113.77")
    assert st == 200 and len(ch) == ch.most and "ip:203.0.113.77" in ch.owners
    flows = srv.issuer.flows
    flows.most = max(1, len(flows))
    assert "accounts.google" in srv.call("GET", "/keys/google/start", xff="203.0.113.78")[1]["location"] or         "#error" not in srv.call("GET", "/keys/google/start", xff="203.0.113.79")[1]["location"]
    # expired entries make room
    t = [time.time()]
    for m in (srv.issuer.challenges, srv.issuer.flows, srv.issuer.sessions):
        m.clock = lambda: t[0]
    t[0] += KS.CHALLENGE_S + 1
    assert srv.call("POST", "/keys/api/passkey/login/options")[0] == 200
    # sessions: at most SESSIONS_AN_ACCOUNT an account, at most MOST_SESSIONS in all (the oldest evicted)
    s = srv.issuer
    for i in range(KS.SESSIONS_AN_ACCOUNT + 3):
        s.new_session("acct-a", "passkey")
    assert len(s.sessions.keys_where(lambda v: v["account"] == "acct-a")) == KS.SESSIONS_AN_ACCOUNT
    s.sessions.most = 6
    for i in range(10):
        s.new_session(f"acct-{i}", "passkey")
    assert len(s.sessions) == 6
    lim = KS.Limiter(1, most=3)
    assert [lim.allow(f"ip:{i}") for i in range(5)] == [True] * 5  # past `most` groups a newcomer is let through uncounted
    assert [lim.allow("ip:0"), lim.allow("ip:0")] == [False, False]  # a counted one keeps its minute's limit


# what is written (ZDR)

def test_no_secret_in_any_log_or_file(tmp_path, made, google, capsys):
    srv = made("--quotas", "--quota-file", str(tmp_path / "quota.json"))
    cookie, _ = check_with_google(srv, google, "sub-canary-118833")
    auth = Authenticator()
    reg = register(srv, cookie, auth)[3]
    srv.chat(reg["key"])
    st, _, pcookie, _ = sign_in(srv, auth)
    rot = srv.call("POST", "/keys/api/rotate", cookie=pcookie)[3]["key"]
    srv.chat(rot)
    sign_in(srv, auth, tamper=True)
    srv.call("GET", "/keys/callback/google?state=canary-state-55&code=canary-code-66")
    check_with_google(srv, google, "sub-canary-118833", nonce="wrong")  # a refused id_token
    # the exchange refused by Google (a valid flow, a code Google does not know): the console's failure line
    st, h, sc, _ = srv.call("GET", "/keys/google/start")
    state = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(h["location"]).query))["state"]
    flow, _ = cookie_of(sc, KS.FLOW_COOKIE)
    st, h, _, _ = srv.call("GET", f"/keys/callback/google?state={state}&code=4/0A-canary-code-unknown", cookie=f"{KS.FLOW_COOKIE}={flow}")
    assert h["location"].endswith("#error=google")
    # Google's keys not reachable (an unknown kid makes it fetch them again): the other failure line
    jwks = srv.issuer.provider.jwks
    jwks.url, jwks.tried = "http://127.0.0.1:1/certs", float("-inf")
    assert check_with_google(srv, google, "sub-canary-118833", token=google.jwt({"iss": "https://accounts.google.com", "aud": CLIENT_ID,
                                                                                  "sub": "sub-canary-118833", "exp": int(time.time()) + 60}, kid="rotated"))[1] == "token"
    metrics = srv.call("GET", "/metrics")[3]
    srv.app.admin.quota.save()
    t = time.time()
    while not (tmp_path / "status.json").exists() or (tmp_path / "status.json").stat().st_mtime < t - 0.5:
        assert time.time() - t < 8, "the status file was not written"
        time.sleep(0.2)
    srv.close()
    time.sleep(0.2)
    printed = capsys.readouterr()
    assert "the code's exchange with Google failed (HTTPError)" in printed.out and "signing keys could not be fetched (URLError)" in printed.out
    sid = cookie.split("=", 1)[1]
    secrets_ = {"key": reg["key"], "rotated key": rot, "client secret": CLIENT_SECRET, "sub": "sub-canary-118833", "id_token": google.tokens[0],
                "id_token's signature": google.tokens[0].rsplit(".", 1)[1][:40], "session": sid, "state": "canary-state-55",
                "code": "canary-code-66", "Google's code": "4/0A-canary-code-"}
    texts = {"console": printed.out + printed.err, "/metrics": metrics.decode("utf-8", "replace")}
    for root, _, files in os.walk(tmp_path):
        for f in files:
            if f != "client.json":
                texts[os.path.join(root, f)] = open(os.path.join(root, f), "rb").read().decode("utf-8", "replace")
    for name in ("requests.jsonl", "status.json", "quota.json", "store.json", "store.json.secret"):
        assert any(p.endswith(os.sep + name) for p in texts), name
    assert "g-" in texts[str(tmp_path / "quota.json")]  # (the store key's tenant was counted there: the search saw its file)
    for where, text in texts.items():
        for what, s in secrets_.items():
            assert s not in text, (what, where)
        if not where.endswith("store.json"):
            assert b64u(auth.cred) not in text, where  # the credential id: in the store only
    store = texts[str(tmp_path / "keys" / "store.json")]
    assert hashlib.sha256(rot.encode()).hexdigest() in store and b64u(auth.cred) in store


def test_without_the_option_keys_paths_are_unknown_as_before(tmp_path):
    cli = options("--anonymous")
    httpd = bind(C.BoundedHTTPServer, S.Handler)
    app = httpd.app = S.App(Inbox(), Tok(), False, 64, {"bits_per_weight": 3.0})
    A.install(app, app.server, cli, httpd, [])
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        assert getattr(app.admin, "issuer", None) is None
        for method in ("GET", "POST"):
            conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
            conn.request(method, "/keys/google/start", b"{}" if method == "POST" else None, {"Content-Type": "application/json"})
            r = conn.getresponse()
            assert r.status == 404 and json.loads(r.read())["error"]["message"] == "unknown path: GET / lists the endpoints"
            conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_checks_at_start(tmp_path, google, capsys):
    client = tmp_path / "client.json"
    client.write_text(json.dumps({"web": {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}}), encoding="utf-8")
    base = ["--anonymous", "--oauth-file", str(client), "--public-origin", "https://ssalmuck-ai.duckdns.org"]
    ok = options("--keys-store", str(tmp_path / "s.json"), *base)
    KS.check(ok, [])
    assert KS.origin_and_rp(ok) == ("https://ssalmuck-ai.duckdns.org", "ssalmuck-ai.duckdns.org")
    for argv, keys in ((["--keys-store", os.path.join(ROOT, "logs", "store.json"), *base], []),  # inside the repository
                       (["--keys-store", str(tmp_path / "s.json"), "--oauth-file", str(client), "--public-origin", ORIGIN], []),  # no users told apart
                       (["--keys-store", str(tmp_path / "s.json"), *base], [("g-owner", b"x" * 20)]),
                       (["--keys-store", str(tmp_path / "s.json"), "--anonymous", "--public-origin", ORIGIN], []),
                       (["--keys-store", str(tmp_path / "s.json"), "--anonymous", "--oauth-file", str(client)], []),
                       (["--keys-store", str(tmp_path / "s.json"), *base[:3], "--public-origin", "http://ssalmuck-ai.duckdns.org"], []),
                       (["--keys-store", str(tmp_path / "s.json"), *base, "--keys-rp-id", "duckdns.com"], [])):
        with pytest.raises(SystemExit):
            KS.check(options(*argv), keys)
    for content in ("{not json", json.dumps({"web": {"client_id": CLIENT_ID}}), json.dumps({"other": {}}),
                    json.dumps({"web": {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "token_uri": "http://example.com/token"}})):
        client.write_text(content, encoding="utf-8")
        with pytest.raises(SystemExit) as e:
            KS.check(ok, [])
        assert CLIENT_SECRET not in str(e.value) and CLIENT_ID not in str(e.value)
    assert CLIENT_SECRET not in capsys.readouterr().out
    assert CLIENT_SECRET not in repr(KS.Google(CLIENT_ID, CLIENT_SECRET))
    # a store that is not one is left as it is
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(SystemExit):
        KS.Store(str(bad))
    assert bad.read_text() == "[1, 2]"


def test_the_public_example_turns_it_on():
    cli = A.parse_args(S.parser(), ["--config", os.path.join(ROOT, "serve.public.example.toml")])
    assert cli.keys_store and not KS._inside(cli.keys_store, ROOT) and cli.oauth_file and cli.keys_per_ip_day == 3
    assert KS.origin_and_rp(cli) == ("https://ssalmuck-ai.duckdns.org", "ssalmuck-ai.duckdns.org") and cli.anonymous


def test_the_store_is_written_whole_and_narrowed_to_this_user(tmp_path):
    store = KS.Store(str(tmp_path / "k" / "store.json"))
    acct = store.account_id("google", "sub-x")
    assert acct != store.account_id("google", "sub-y") and "sub-x" not in acct and len(acct) == 64
    key, replaced = store.issue(acct, "ip:203.0.113.1", 3, passkey={"id": b64u(b"c" * 16), "key": "AA", "alg": -7, "count": 0})
    assert not replaced and store.name_of(key.encode()) == store.name_for(acct) and store.name_of(b"sk-other") is None
    leftovers = [f for f in os.listdir(tmp_path / "k") if f.endswith(".tmp")]
    assert leftovers == []
    # a failed write leaves the store in memory and on disk as it was
    before = (tmp_path / "k" / "store.json").read_bytes()

    def broken(data):
        raise OSError("disk full")

    store._save = broken
    with pytest.raises(OSError):
        store.issue(acct, "ip:203.0.113.1", 3, rotate=True)
    assert store.name_of(key.encode()) is not None and (tmp_path / "k" / "store.json").read_bytes() == before
    if os.name == "nt":
        import subprocess

        for f in ("store.json", "store.json.secret"):
            acl = subprocess.run(["icacls", str(tmp_path / "k" / f)], capture_output=True, text=True).stdout
            aces = re.findall(r"(\S+):\(", acl)  # DOMAIN\user:(F), one an entry
            assert aces == [f"{os.environ.get('USERDOMAIN', '')}\\{os.environ['USERNAME']}"] and ":(F)" in acl, acl


def test_cbor_and_authenticator_data_refuse_garbage_without_crashing():
    import random

    rnd = random.Random(7)
    for _ in range(3000):
        raw = bytes(rnd.randrange(256) for _ in range(rnd.randrange(0, 80)))
        try:
            KS.cbor(raw)
        except ValueError:
            pass
        try:
            KS.authenticator_data(hashlib.sha256(b"localhost").digest() + raw, "localhost", attested=bool(raw and raw[0] & 1))
        except ValueError:
            pass
    nested = b"\x81" * 40 + b"\x00"
    with pytest.raises(ValueError):
        KS.cbor(nested)
    assert KS.cbor(enc({"fmt": "none", 1: [-7, b"\x01", True]}))[0] == {"fmt": "none", 1: [-7, b"\x01", True]}
    with pytest.raises(ValueError):
        KS.unb64u("not base64!")
