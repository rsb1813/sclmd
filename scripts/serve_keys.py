# Key issuance (design.md 6.20 and 6.22; P2): a person proves once with Google that they are someone (OpenID Connect,
# scope openid alone: no name, e-mail or profile), registers a passkey (WebAuthn), and gets an API key, shown once.
# Later they sign in with the passkey alone to see their usage, rotate the key or revoke it. Off unless --keys-store
# is given; scripts/serve_admin.py calls in through four hooks (options, start, install, and the GET and POST hooks).
#
# What the server keeps (the store, a JSON file outside the repository, written whole and atomically):
# - for each Google account: HMAC(server secret, "google:" + sub) as its id, never the sub itself;
# - its API key as SHA-256, the key's name ("g-" and 12 hex digits of another HMAC of the account: stable, so a
#   re-issued key keeps the account's quota counters), and when it was made;
# - its passkeys: the credential id, the COSE public key, its algorithm and its sign count;
# - the keys issued in the last 24 hours, by an HMAC of the address (IPv4, IPv6's /64): 3 an address a day.
# The secret of those HMACs is made once beside the store (<store>.secret, 32 random bytes; icacls: only this user).
#
# Endpoints (the edge forwards them; the pages are the edge's own static files):
#   GET  /keys/google/start            302 to Google: authorization code + PKCE S256, state, nonce (server side, 10 min),
#                                      and a short cookie that binds the state to this browser (sclmd_flow)
#   GET  /keys/callback/google         the code exchanged, the id_token checked (RS256 against Google's JWKS, iss, aud,
#                                      exp, nonce) -> the session cookie sclmd_keys and 302 /keys#verified (/keys#error=..)
#   POST /keys/api/passkey/register/options | register/verify | login/options | login/verify, /keys/api/me | rotate |
#        revoke | logout: JSON, X-Requested-With: sclmd and Origin = --public-origin, OpenAI-shaped errors.
# Nothing secret is ever printed or written: no key, id_token, code, state, sub, client secret, credential or
# signature; a failure prints its type.
import base64
import getpass
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import secrets
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

from scripts import serve_contract as C
from scripts import serve_net as N

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COOKIE = "sclmd_keys"  # the session: an opaque random id; the server's session holds only the account's HMAC
FLOW_COOKIE = "sclmd_flow"  # the Google check under way: binds its state to the browser that started it
SESSION_S = 15 * 60
FLOW_S = 10 * 60
CHALLENGE_S = 5 * 60
MOST_SESSIONS = MOST_FLOWS = MOST_CHALLENGES = 4096  # in memory, whatever comes
PER_OWNER = 4  # Google checks or sign-in challenges under way at once for one address (IPv4) or one IPv6 /48
SESSIONS_AN_ACCOUNT = 4  # a fifth sign-in ends the account's oldest session
MOST_ADDRESSES = 65536  # addresses counted a minute (past it a new address is refused until the minute turns)
PER_MINUTE = 30  # requests to /keys/* a minute from one address
KEYS_A_DAY = 3  # keys issued to one address in 24 hours (--keys-per-ip-day)
BODY_MOST = 64 << 10
DAY = 86400.0
REDIRECT_PATH = "/keys/callback/google"
GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
GOOGLE_JWKS = "https://www.googleapis.com/oauth2/v3/certs"
ISSUERS = ("https://accounts.google.com", "accounts.google.com")  # Google's id_tokens carry either
SKEW_S = 60  # an id_token's exp may be this far past
ALGS = (-7, -8, -257)  # ES256, EdDSA (Ed25519), RS256 (Windows Hello gives RS256)
ERRORS = ("denied", "expired", "browser", "google", "token", "rate_limited", "busy", "bad_request", "server")  # /keys#error=...
_B64U = re.compile(r"^[A-Za-z0-9_-]*={0,2}$")


class Bad(ValueError):
    """A request that is refused; its message is a fixed string of this module (never a request's bytes)."""


class TooMany(Exception):
    def __init__(self, retry_after: float):
        super().__init__("too many keys from this address today")
        self.retry_after = retry_after


def b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def unb64u(text, most: int = BODY_MOST) -> bytes:
    """base64url (padding optional) -> bytes; Bad for anything else."""
    if not isinstance(text, str) or len(text) > most * 4 // 3 + 4 or not _B64U.match(text):
        raise Bad("a field is not base64url")
    s = text.rstrip("=")
    if len(s) % 4 == 1:
        raise Bad("a field is not base64url")
    try:
        return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    except (ValueError, TypeError):
        raise Bad("a field is not base64url") from None


def sha256(data) -> bytes:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).digest()


# CBOR (RFC 8949), only what WebAuthn's authenticators write: definite lengths, at most 16 deep, 1,024 items a level

def cbor(data: bytes, i: int = 0, depth: int = 0) -> tuple:
    """The item at data[i:] -> (value, the offset after it). ValueError (Bad) for anything else."""
    if depth > 16 or i >= len(data):
        raise Bad("the CBOR is cut short or too deep")
    b = data[i]
    major, info = b >> 5, b & 31
    i += 1
    if major == 7:
        if info == 20:
            return False, i
        if info == 21:
            return True, i
        if info in (22, 23):
            return None, i
        size = {25: 2, 26: 4, 27: 8}.get(info)
        if size is None or i + size > len(data):
            raise Bad("a CBOR simple value that is not read")
        return struct.unpack(">" + {2: "e", 4: "f", 8: "d"}[size], data[i:i + size])[0], i + size
    if info < 24:
        n = info
    elif info in (24, 25, 26, 27):
        size = 1 << (info - 24)
        if i + size > len(data):
            raise Bad("the CBOR is cut short")
        n = int.from_bytes(data[i:i + size], "big")
        i += size
    else:
        raise Bad("CBOR of indefinite length")
    if major == 0:
        return n, i
    if major == 1:
        return -1 - n, i
    if major in (2, 3):
        if i + n > len(data):
            raise Bad("the CBOR is cut short")
        raw = bytes(data[i:i + n])
        if major == 3:
            try:
                return raw.decode("utf-8"), i + n
            except UnicodeDecodeError:
                raise Bad("CBOR text that is not UTF-8") from None
        return raw, i + n
    if major == 4:
        if n > 1024:
            raise Bad("a CBOR array too long")
        out = []
        for _ in range(n):
            v, i = cbor(data, i, depth + 1)
            out.append(v)
        return out, i
    if major == 5:
        if n > 1024:
            raise Bad("a CBOR map too long")
        out = {}
        for _ in range(n):
            k, i = cbor(data, i, depth + 1)
            if isinstance(k, (list, dict)):
                raise Bad("a CBOR map key that is not a value")
            v, i = cbor(data, i, depth + 1)
            out[k] = v
        return out, i
    v, i = cbor(data, i, depth + 1)  # major 6: a tag, its value
    return v, i


# WebAuthn (level 3), verified here with cryptography

def public_key(cose: dict):
    """A COSE key -> (its algorithm, the public key). Bad unless ES256 on P-256, Ed25519, or RS256 of 2,048 bits or more."""
    if not isinstance(cose, dict):
        raise Bad("the credential's public key is not a COSE key")
    kty, alg = cose.get(1), cose.get(3)
    try:
        if alg == -7 and kty == 2 and cose.get(-1) == 1:
            x, y = cose.get(-2), cose.get(-3)
            if not (isinstance(x, bytes) and isinstance(y, bytes) and len(x) == len(y) == 32):
                raise Bad("the credential's public key is not a P-256 point")
            return alg, ec.EllipticCurvePublicNumbers(int.from_bytes(x, "big"), int.from_bytes(y, "big"), ec.SECP256R1()).public_key()
        if alg == -8 and kty == 1 and cose.get(-1) == 6:
            x = cose.get(-2)
            if not (isinstance(x, bytes) and len(x) == 32):
                raise Bad("the credential's public key is not an Ed25519 key")
            return alg, ed25519.Ed25519PublicKey.from_public_bytes(x)
        if alg == -257 and kty == 3:
            n, e = cose.get(-1), cose.get(-2)
            if not (isinstance(n, bytes) and isinstance(e, bytes) and 256 <= len(n) <= 1024 and 0 < len(e) <= 8):
                raise Bad("the credential's public key is not an RSA key of 2,048 bits or more")
            return alg, rsa.RSAPublicNumbers(int.from_bytes(e, "big"), int.from_bytes(n, "big")).public_key()
    except Bad:
        raise
    except (ValueError, TypeError):
        raise Bad("the credential's public key does not load") from None
    raise Bad("the credential's algorithm is not ES256, EdDSA or RS256")


def signed(alg: int, key, signature: bytes, data: bytes) -> bool:
    try:
        if alg == -7:
            key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
        elif alg == -8:
            key.verify(signature, data)
        elif alg == -257:
            key.verify(signature, data, padding.PKCS1v15(), hashes.SHA256())
        else:
            return False
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


UP, UV, BE, BS, AT, ED = 0x01, 0x04, 0x08, 0x10, 0x40, 0x80


def authenticator_data(raw: bytes, rp_id: str, attested: bool) -> dict:
    """authenticatorData -> {"flags", "count", and with attested: "credential" (id), "cose" (the key's CBOR), "alg",
    "key"}. Checks: the rp id's hash, user presence, BS only with BE, the attested data there exactly when asked,
    nothing after the extensions."""
    if not isinstance(raw, bytes) or len(raw) < 37:
        raise Bad("the authenticator data is cut short")
    if not hmac.compare_digest(raw[:32], sha256(rp_id)):
        raise Bad("the authenticator data is for another rp id")
    flags, count = raw[32], int.from_bytes(raw[33:37], "big")
    if not flags & UP:
        raise Bad("the authenticator did not see the user present")
    if flags & BS and not flags & BE:
        raise Bad("the authenticator data's flags are not consistent")
    if bool(flags & AT) != attested:
        raise Bad("the authenticator data's attested credential is missing or not expected")
    out, i = {"flags": flags, "count": count}, 37
    if attested:
        if len(raw) < i + 18:
            raise Bad("the authenticator data is cut short")
        length = int.from_bytes(raw[i + 16:i + 18], "big")
        i += 18
        if not 16 <= length <= 1023 or len(raw) < i + length:
            raise Bad("the credential id is not of 16 to 1,023 bytes")
        out["credential"] = raw[i:i + length]
        i += length
        cose, j = cbor(raw, i)
        out["cose"] = raw[i:j]
        out["alg"], out["key"] = public_key(cose)
        i = j
    if flags & ED:
        _, i = cbor(raw, i)
    if i != len(raw):
        raise Bad("the authenticator data has bytes after its end")
    return out


def client_data(raw: bytes, kind: str, origin: str) -> bytes:
    """clientDataJSON -> its challenge. Checks: the type, the origin (exactly --public-origin), not cross-origin."""
    try:
        d = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise Bad("the client data is not JSON") from None
    if not isinstance(d, dict) or d.get("type") != kind:
        raise Bad("the client data is not of this ceremony")
    if not isinstance(d.get("origin"), str) or not hmac.compare_digest(d["origin"].encode(), origin.encode()):
        raise Bad("the client data is from another origin")
    if d.get("crossOrigin") not in (None, False) or d.get("topOrigin") not in (None, origin):
        raise Bad("the client data is from a frame of another origin")
    return unb64u(d.get("challenge"), 1024)


def _credential(body) -> tuple:
    """The PublicKeyCredential JSON of a body -> (raw id, its response dict)."""
    cred = body.get("credential") if isinstance(body, dict) else None
    if not isinstance(cred, dict) or cred.get("type") != "public-key" or not isinstance(cred.get("response"), dict):
        raise Bad('the body is {"credential": <PublicKeyCredential JSON>}')
    raw_id = unb64u(cred.get("rawId", cred.get("id")), 1024)
    if "id" in cred and unb64u(cred["id"], 1024) != raw_id:
        raise Bad("the credential's id and rawId differ")
    if not 16 <= len(raw_id) <= 1023:
        raise Bad("the credential id is not of 16 to 1,023 bytes")
    return raw_id, cred["response"]


def verify_registration(body, challenge: bytes, origin: str, rp_id: str) -> dict:
    """A registration (navigator.credentials.create) -> the passkey to keep: {"id", "key" (COSE), "alg", "count"}.
    The attestation statement is not checked (attestation "none" is asked for: no authenticator is trusted for its
    make); its authenticator data is."""
    raw_id, resp = _credential(body)
    cdj = unb64u(resp.get("clientDataJSON"))
    got = client_data(cdj, "webauthn.create", origin)
    if not hmac.compare_digest(got, challenge):
        raise Bad("the challenge is not the one given")
    raw_att = unb64u(resp.get("attestationObject"))
    att, end = cbor(raw_att)
    if end != len(raw_att) or not isinstance(att, dict) or not isinstance(att.get("fmt"), str) or not isinstance(att.get("authData"), bytes) or not isinstance(att.get("attStmt"), dict):
        raise Bad("the attestation object is not one")
    data = authenticator_data(att["authData"], rp_id, attested=True)
    if data["credential"] != raw_id:
        raise Bad("the credential's id is not the authenticator data's")
    return {"id": b64u(raw_id), "key": b64u(data["cose"]), "alg": data["alg"], "count": data["count"]}


def verify_assertion(body, origin: str, rp_id: str, challenge_of, passkey_of) -> tuple:
    """A sign-in (navigator.credentials.get). challenge_of(challenge) -> True once for a challenge given (and not
    used); passkey_of(raw id) -> (account, passkey dict, user handle) or None. -> (account, credential id, new count)."""
    raw_id, resp = _credential(body)
    cdj = unb64u(resp.get("clientDataJSON"))
    got = client_data(cdj, "webauthn.get", origin)
    if not challenge_of(got):
        raise Bad("the challenge is not one given, or was used")
    found = passkey_of(raw_id)
    if found is None:
        raise Bad("this passkey is not known here (revoked, or of another site)")
    account, pk, handle = found
    data = unb64u(resp.get("authenticatorData"))
    ad = authenticator_data(data, rp_id, attested=False)
    user = resp.get("userHandle")
    if user not in (None, "") and not hmac.compare_digest(unb64u(user, 1024), handle):
        raise Bad("the passkey's user is not its account's")
    alg, key = public_key(cbor(unb64u(pk["key"]))[0])
    if alg != pk["alg"] or not signed(alg, key, unb64u(resp.get("signature")), data + sha256(cdj)):
        raise Bad("the passkey's signature does not verify")
    if (ad["count"] or pk["count"]) and ad["count"] <= pk["count"]:
        raise Bad("the passkey's sign count went back (a cloned authenticator?)")
    return account, pk["id"], ad["count"]


# Google's OpenID Connect

def _https_or_loopback(url: str, what: str) -> str:
    u = urllib.parse.urlsplit(url)
    host = u.hostname or ""
    try:
        loop = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loop = host == "localhost"
    if u.scheme != "https" and not (u.scheme == "http" and loop):
        raise SystemExit(f"{what}: an https URL (http only to this PC)")
    return url


def _fetch(url: str, data: bytes | None = None, most: int = 1 << 18, timeout: float = 10.0) -> tuple:
    """(status, headers, body) of a GET (or a form POST with data). The body at most `most` bytes. Loopback URLs go
    through no proxy (the tests' stand-in)."""
    head = {"Accept": "application/json", "User-Agent": "sclmd"}
    if data is not None:
        head["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, headers=head, method="POST" if data is not None else "GET")
    host = urllib.parse.urlsplit(url).hostname or ""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if host in ("127.0.0.1", "localhost", "::1") else urllib.request.build_opener()
    with opener.open(req, timeout=timeout) as r:
        body = r.read(most + 1)
        if len(body) > most:
            raise ValueError("the answer is too long")
        return r.status, r.headers, body


class Jwks:
    """Google's signing keys, cached for the max-age its answer gives (5 min to a day); a kid not known is fetched
    again at most once a minute; a key a day past its age is still used when Google cannot be reached."""

    def __init__(self, url: str, fetch=_fetch, clock=time.time):
        self.url, self.fetch, self.clock = url, fetch, clock
        self.keys, self.until, self.tried = {}, 0.0, -math.inf
        self.lock = threading.Lock()

    def key(self, kid: str):
        with self.lock:
            now = self.clock()
            if now < self.until and kid in self.keys:
                return self.keys[kid]
            if now - self.tried >= 60:
                self.tried = now
                try:
                    self._load(now)
                except Exception as e:  # noqa: BLE001
                    print(f"[serve] keys: Google's signing keys could not be fetched ({C.why(e)})", flush=True)
            if kid in self.keys and now < self.until + DAY:
                return self.keys[kid]
            return None

    def _load(self, now: float) -> None:
        status, headers, body = self.fetch(self.url)
        data = json.loads(body)
        keys = {}
        for k in (data.get("keys") or [])[:32]:
            if not isinstance(k, dict) or k.get("kty") != "RSA" or k.get("alg", "RS256") != "RS256" or k.get("use", "sig") != "sig":
                continue
            if not isinstance(k.get("kid"), str):
                continue
            n, e = unb64u(k.get("n"), 1024), unb64u(k.get("e"), 16)
            keys[k["kid"]] = rsa.RSAPublicNumbers(int.from_bytes(e, "big"), int.from_bytes(n, "big")).public_key()
        if not keys:
            raise ValueError("no RS256 key")
        age = 3600
        m = re.search(r"max-age=(\d+)", (headers.get("Cache-Control") or "") if headers is not None else "")
        if m:
            age = int(m.group(1))
        self.keys, self.until = keys, now + min(max(age, 300), DAY)


class Google:
    """The OpenID provider: authorization URL, the code's exchange, the id_token's check. kind: the account HMAC's
    label ("google:" + sub; another provider, e.g. Kakao, would be another kind)."""

    kind = "google"

    def __init__(self, client_id: str, client_secret: str, auth_uri: str = GOOGLE_AUTH, token_uri: str = GOOGLE_TOKEN,
                 jwks_uri: str = GOOGLE_JWKS, fetch=_fetch, clock=time.time):
        self.client_id, self._secret = client_id, client_secret
        self.auth_uri, self.token_uri = auth_uri, token_uri
        self.fetch, self.clock = fetch, clock
        self.jwks = Jwks(jwks_uri, fetch, clock)

    def __repr__(self) -> str:  # never the secret
        return f"Google(auth={self.auth_uri!r}, token={self.token_uri!r})"

    @classmethod
    def from_file(cls, path: str, jwks_uri: str = GOOGLE_JWKS) -> "Google":
        """Google's client file ({"web": {...}} or {"installed": {...}}): its client_id and client_secret, and its
        auth_uri and token_uri when it has them. An error names the file and the fields, never a value."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            raise SystemExit(f"--oauth-file {path}: not read ({type(e).__name__})") from None
        block = (data.get("web") or data.get("installed")) if isinstance(data, dict) else None
        if not isinstance(block, dict):
            raise SystemExit(f"--oauth-file {path}: no \"web\" (or \"installed\") client in it")
        missing = [k for k in ("client_id", "client_secret") if not isinstance(block.get(k), str) or not block[k].strip()]
        if missing:
            raise SystemExit(f"--oauth-file {path}: {', '.join(missing)} missing")
        auth = _https_or_loopback(block.get("auth_uri") or GOOGLE_AUTH, f"--oauth-file {path}: auth_uri")
        token = _https_or_loopback(block.get("token_uri") or GOOGLE_TOKEN, f"--oauth-file {path}: token_uri")
        return cls(block["client_id"].strip(), block["client_secret"].strip(), auth, token, _https_or_loopback(jwks_uri, "--oauth-jwks"))

    def authorize(self, redirect: str, state: str, nonce: str, verifier: str) -> str:
        q = {"client_id": self.client_id, "redirect_uri": redirect, "response_type": "code", "scope": "openid", "state": state,
             "nonce": nonce, "code_challenge": b64u(sha256(verifier)), "code_challenge_method": "S256", "prompt": "select_account"}
        return self.auth_uri + ("&" if "?" in self.auth_uri else "?") + urllib.parse.urlencode(q)

    def exchange(self, code: str, verifier: str, redirect: str) -> str:
        """The authorization code -> the id_token (the access token is not kept)."""
        form = urllib.parse.urlencode({"grant_type": "authorization_code", "code": code, "client_id": self.client_id,
                                       "client_secret": self._secret, "redirect_uri": redirect, "code_verifier": verifier}).encode()
        _, _, body = self.fetch(self.token_uri, form, 1 << 16)
        token = json.loads(body).get("id_token")
        if not isinstance(token, str):
            raise ValueError("no id_token")
        return token

    def subject(self, token: str, nonce: str) -> str:
        """The id_token checked -> its sub. Bad: the signature (RS256, a key of the JWKS), iss, aud (azp), exp, iat or nonce."""
        if len(token) > 16384 or token.count(".") != 2:
            raise Bad("the id_token is not a JWT")
        head64, body64, sig64 = token.split(".")
        try:
            head, claims = json.loads(unb64u(head64)), json.loads(unb64u(body64))
        except (ValueError, UnicodeDecodeError):
            raise Bad("the id_token is not a JWT") from None
        if not isinstance(head, dict) or head.get("alg") != "RS256" or not isinstance(head.get("kid"), str) or not isinstance(claims, dict):
            raise Bad("the id_token is not signed with RS256 by a key named")
        key = self.jwks.key(head["kid"])
        if key is None:
            raise Bad("the id_token's key is not one of Google's")
        try:
            key.verify(unb64u(sig64), f"{head64}.{body64}".encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
        except (InvalidSignature, ValueError):
            raise Bad("the id_token's signature does not verify") from None
        now = self.clock()
        aud = claims.get("aud")
        if claims.get("iss") not in ISSUERS:
            raise Bad("the id_token is not Google's")
        if not (aud == self.client_id or (isinstance(aud, list) and self.client_id in aud and claims.get("azp") == self.client_id)):
            raise Bad("the id_token is for another client")
        exp, iat = claims.get("exp"), claims.get("iat")
        if not isinstance(exp, (int, float)) or now > exp + SKEW_S:
            raise Bad("the id_token has expired")
        if isinstance(iat, (int, float)) and iat > now + 300:
            raise Bad("the id_token is from the future")
        if not isinstance(claims.get("nonce"), str) or not hmac.compare_digest(claims["nonce"].encode(), nonce.encode()):
            raise Bad("the id_token's nonce is not this check's")
        sub = claims.get("sub")
        if not isinstance(sub, str) or not 0 < len(sub) <= 255:
            raise Bad("the id_token has no subject")
        return sub


# the store

def tighten(path: str) -> bool:
    """Only this user may read or write the file: icacls on Windows (inheritance off, this user alone), 0600 elsewhere."""
    if os.name != "nt":
        try:
            os.chmod(path, 0o600)
            return True
        except OSError:
            return False
    try:
        r = subprocess.run(["icacls", path, "/inheritance:r", "/grant:r", f"{getpass.getuser()}:(F)"], capture_output=True, timeout=20)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class Store:
    """The key store (above). Every change is written to a new file, flushed, and moved over the old one: a crash
    leaves the old store or the new, never half of one. In memory: the store and its indexes, under one lock."""

    def __init__(self, path: str, clock=time.time):
        self.path, self.clock = os.path.abspath(path), clock
        self.lock = threading.RLock()
        self.told = False
        self.secret = self._secret()
        self.data = self._load()
        self._index()

    def _secret(self) -> bytes:
        path = self.path + ".secret"
        folder = os.path.dirname(self.path)
        os.makedirs(folder, exist_ok=True)
        if not os.path.exists(path) and os.path.exists(self.path):  # a new secret would split every account from its HMAC
            raise SystemExit(f"--keys-store: {self.path} is there but its secret {path} is not: restore it (from the backup taken with the store); without it Google accounts and address counts cannot be matched again")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        except FileExistsError:
            fd = None
        if fd is not None:
            with os.fdopen(fd, "wb") as f:
                f.write(secrets.token_bytes(32))
                f.flush()
                os.fsync(f.fileno())
            if not tighten(path):
                print(f"[serve] keys: warning: the permissions of {path} could not be narrowed to this user", flush=True)
        with open(path, "rb") as f:
            s = f.read()
        if len(s) != 32:
            raise SystemExit(f"--keys-store: {path} is not a secret of 32 bytes")
        return s

    def _load(self) -> dict:
        try:
            with open(self.path, "rb") as f:
                data = json.loads(f.read())
        except FileNotFoundError:
            return {"version": 1, "accounts": {}, "issued": []}
        except (OSError, ValueError) as e:
            raise SystemExit(f"--keys-store {self.path}: not read ({type(e).__name__}); it is left as it is") from None
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("accounts"), dict) or not isinstance(data.get("issued"), list):
            raise SystemExit(f"--keys-store {self.path}: not a key store of version 1; it is left as it is")
        return data

    def _index(self) -> None:
        self.by_key, self.by_passkey = {}, {}
        for acct, rec in self.data["accounts"].items():
            if rec.get("key"):
                self.by_key[rec["key"]] = acct
            for pk in rec.get("passkeys", ()):
                self.by_passkey[pk["id"]] = acct

    def _save(self, data: dict) -> None:
        """data written in the store's place (and only then made the store in memory, by the caller)."""
        tmp = f"{self.path}.{os.getpid()}.tmp"
        raw = json.dumps(data, separators=(",", ":"), sort_keys=True).encode()
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        if not tighten(tmp) and not self.told:
            self.told = True
            print(f"[serve] keys: warning: the permissions of {self.path} could not be narrowed to this user", flush=True)
        for attempt in range(6):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:  # (Windows: a reader holds the old file for a moment)
                if attempt == 5:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def _commit(self, data: dict) -> None:
        self._save(data)
        self.data = data
        self._index()

    def _copy(self) -> dict:
        return json.loads(json.dumps(self.data))

    # ids

    def _mac(self, label: str, text: str) -> bytes:
        return hmac.new(self.secret, f"{label}:{text}".encode("utf-8"), hashlib.sha256).digest()

    def account_id(self, kind: str, subject: str) -> str:
        """The only form a provider's account id is kept in: HMAC(secret, "google:" + sub)."""
        return hmac.new(self.secret, f"{kind}:{subject}".encode("utf-8"), hashlib.sha256).hexdigest()

    def name_for(self, account: str) -> str:
        return "g-" + self._mac("name", account).hex()[:12]

    def user_handle(self, account: str) -> bytes:
        """WebAuthn's user.id of the account: the same each time, so the authenticator replaces its old passkey."""
        return self._mac("webauthn-user", account)

    def address_id(self, ident: str) -> str:
        return self._mac("address", ident).hex()[:32]

    # what the server asks

    def name_of(self, got: bytes) -> str | None:
        """The name of a store key (its SHA-256 looked up, then compared in constant time); None: not one."""
        h = hashlib.sha256(got).hexdigest()
        with self.lock:
            acct = self.by_key.get(h)
            rec = self.data["accounts"].get(acct) if acct is not None else None
            stored = rec.get("key") if rec else None
        return rec["name"] if stored and hmac.compare_digest(stored, h) else None

    def account(self, account: str) -> dict | None:
        with self.lock:
            rec = self.data["accounts"].get(account)
            return dict(rec) if rec else None

    def passkey_of(self, raw_id: bytes):
        """-> (account, passkey dict, user handle) or None."""
        cid = b64u(raw_id)
        with self.lock:
            acct = self.by_passkey.get(cid)
            if acct is None:
                return None
            pk = next((p for p in self.data["accounts"][acct]["passkeys"] if p["id"] == cid), None)
        return (acct, dict(pk), self.user_handle(acct)) if pk else None

    def _today(self, data: dict, now: float) -> list:
        return [x for x in data["issued"] if isinstance(x, list) and len(x) == 2 and now - x[1] < DAY]

    def full_today(self, ident: str, most: int) -> float:
        """0: the address may get a key today; else the seconds until it may."""
        if not most:
            return 0.0
        a, now = self.address_id(ident), self.clock()
        with self.lock:
            times = sorted(t for x, t in self._today(self.data, now) if x == a)
        return 0.0 if len(times) < most else max(1.0, times[len(times) - most] + DAY - now)

    def issue(self, account: str, ident: str, most: int, passkey: dict | None = None, rotate: bool = False) -> tuple:
        """A new key for the account -> (the key, whether an older one was revoked). With a passkey (a Google check
        and a registration): the account is made if new and its passkeys become this one. rotate: only an account that
        has a key. TooMany: the address has had `most` keys in 24 hours; LookupError: no key to rotate; Bad: the
        passkey belongs to another account."""
        with self.lock:
            wait = self.full_today(ident, most)
            if wait:
                raise TooMany(wait)
            data, now = self._copy(), self.clock()
            data["issued"] = self._today(data, now)
            rec = data["accounts"].get(account)
            if rotate and not (rec and rec.get("key")):
                raise LookupError("no key")
            if passkey is not None and self.by_passkey.get(passkey["id"], account) != account:
                raise Bad("this passkey is another account's")
            if rec is None:
                rec = data["accounts"][account] = {"name": self.name_for(account), "created": int(now), "passkeys": []}
            replaced = bool(rec.get("key"))
            key = "sk-" + b64u(secrets.token_bytes(32))
            rec["key"], rec["key_created"] = hashlib.sha256(key.encode()).hexdigest(), int(now)
            if passkey is not None:
                rec["passkeys"] = [dict(passkey, created=int(now))]
            data["issued"].append([self.address_id(ident), now])
            self._commit(data)
        return key, replaced

    def counted(self, account: str, cid: str, count: int) -> None:
        """A sign-in's new sign count (written only when it moved: synced passkeys stay at 0)."""
        with self.lock:
            rec = self.data["accounts"].get(account)
            pk = next((p for p in (rec or {}).get("passkeys", ()) if p["id"] == cid), None)
            if pk is None or pk["count"] == count:
                return
            data = self._copy()
            next(p for p in data["accounts"][account]["passkeys"] if p["id"] == cid)["count"] = count
            self._commit(data)

    def revoke(self, account: str) -> bool:
        """The account's key, passkeys and HMAC deleted."""
        with self.lock:
            if account not in self.data["accounts"]:
                return False
            data = self._copy()
            del data["accounts"][account]
            self._commit(data)
            return True

    def count(self) -> int:
        with self.lock:
            return sum(1 for r in self.data["accounts"].values() if r.get("key"))


# bounded memory

class Expiring:
    """Values that live `ttl` seconds, at most `most` of them, each with an owner (the address group that asked).
    An owner holds at most `per_owner` (its oldest goes for a new one). Past `most`, the oldest entry of the owner
    that holds the most goes: a full table never refuses a newcomer, and filling it pushes out the filler's own
    entries first (verify-P2-keys: 4,096 flows from 137 /64s locked everyone out for 10 minutes). Memory stays
    bounded whatever comes."""

    def __init__(self, ttl: float, most: int, clock=time.time, per_owner: int = 0):
        self.ttl, self.most, self.clock, self.per_owner = ttl, most, clock, per_owner
        self.d = OrderedDict()  # k -> (time, value, owner)
        self.owners = {}  # owner -> OrderedDict of its keys (oldest first)
        self.lock = threading.Lock()

    def _del(self, k) -> None:
        _, _, owner = self.d.pop(k)
        mine = self.owners.get(owner)
        if mine is not None:
            mine.pop(k, None)
            if not mine:
                del self.owners[owner]

    def _prune(self, now: float) -> None:
        while self.d:
            k, (t, _, _) = next(iter(self.d.items()))
            if t + self.ttl > now:
                break
            self._del(k)

    def put(self, k, v, evict: bool = True, owner=None) -> bool:
        with self.lock:
            now = self.clock()
            self._prune(now)
            if k in self.d:
                self._del(k)
            mine = self.owners.get(owner)
            if self.per_owner and mine is not None and len(mine) >= self.per_owner:
                self._del(next(iter(mine)))
            if len(self.d) >= self.most:
                if not evict:
                    return False
                heaviest = max(self.owners.values(), key=len)
                self._del(next(iter(heaviest)))
            self.d[k] = (now, v, owner)
            self.owners.setdefault(owner, OrderedDict())[k] = True
            return True

    def get(self, k):
        with self.lock:
            x = self.d.get(k)
            return x[1] if x is not None and x[0] + self.ttl > self.clock() else None

    def pop(self, k):
        with self.lock:
            x = self.d.get(k)
            if x is None:
                return None
            self._del(k)
            return x[1] if x[0] + self.ttl > self.clock() else None

    def drop(self, keep) -> None:
        """Every entry for which keep(value) is False goes."""
        with self.lock:
            for k in [k for k, (_, v, _) in self.d.items() if not keep(v)]:
                self._del(k)

    def keys_where(self, pred) -> list:
        with self.lock:
            return [k for k, (_, v, _) in self.d.items() if pred(v)]

    def __len__(self) -> int:
        with self.lock:
            return len(self.d)


def group(ident: str) -> str:
    """The owner of a flow or a sign-in challenge: an IPv4 address, or an IPv6 address's /48 (one site; a /64 is one
    of 65,536 a site may hold)."""
    if ident.startswith("ip:") and ":" in ident[3:] and "/64" in ident:
        try:
            net = ipaddress.ip_network(ident[3:], strict=False)
            return "ip:" + str(net.supernet(new_prefix=48))
        except ValueError:
            return ident
    return ident


class Limiter:
    """Requests a minute by address group (group(): IPv4, IPv6 /48), at most MOST_ADDRESSES groups counted a minute.
    Past that a new group is let through uncounted until the minute turns: refusing it would let 65,536 addresses
    lock everyone else out (the tables behind it are bounded per group anyway)."""

    def __init__(self, per_minute: int, most: int = MOST_ADDRESSES, clock=time.time):
        self.per_minute, self.most, self.clock = per_minute, most, clock
        self.minute, self.counts = None, {}
        self.lock = threading.Lock()

    def allow(self, ident: str) -> bool:
        if not self.per_minute:
            return True
        m = int(self.clock() // 60)
        with self.lock:
            if m != self.minute:
                self.minute, self.counts = m, {}
            ident = group(ident)
            n = self.counts.get(ident)
            if n is None and len(self.counts) >= self.most:
                return True
            if (n or 0) >= self.per_minute:
                return False
            self.counts[ident] = (n or 0) + 1
            return True


# the endpoints

def cookies(handler) -> dict:
    out = {}
    for header in handler.headers.get_all("Cookie") or ():
        for part in str(header).split(";"):
            k, eq, v = part.strip().partition("=")
            if eq and k and k not in out:
                out[k] = v.strip().strip('"')
    return out


def _cookie(name: str, value: str, path: str, age: int) -> str:
    return f"{name}={value}; Path={path}; Max-Age={age}; HttpOnly; Secure; SameSite=Lax"


NO_STORE = [("Cache-Control", "no-store"), ("Pragma", "no-cache"), ("X-Content-Type-Options", "nosniff"), ("Referrer-Policy", "no-referrer")]


def _send(handler, status: int, obj=None, headers=()) -> None:
    data = b"" if obj is None else json.dumps(obj, ensure_ascii=False).encode()
    handler.send_response(status)
    if obj is not None:
        handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    for k, v in [*NO_STORE, *headers]:
        handler.send_header(k, v)
    if handler.close_connection:
        handler.send_header("Connection", "close")
    handler.end_headers()
    if data:
        handler.wfile.write(data)


class Refusal(Exception):
    def __init__(self, status: int, code: str, message: str, headers=()):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, list(headers)


KIND = {429: "requests"}


def _error(handler, r: Refusal) -> None:
    from scripts import serve_admin as A

    A.message(handler, None, r.code)  # the request log keeps the code, not the message
    kind = KIND.get(r.status, "invalid_request_error" if r.status < 500 else "server_error")
    _send(handler, r.status, {"error": {"message": r.message, "type": kind, "param": None, "code": r.code}}, r.headers)


def _too_many(wait: float) -> Refusal:
    return Refusal(429, "rate_limit_exceeded", f"this address has had its keys for today: try again in {math.ceil(wait)} s",
                   [("Retry-After", str(math.ceil(wait)))])


class Issuer:
    """The endpoints' state: the store, the provider, the flows (Google checks under way), the sessions and the
    sign-in challenges, and the per-address limits."""

    def __init__(self, store: Store, provider, origin: str, rp_id: str, per_ip_day: int = KEYS_A_DAY, per_minute: int = PER_MINUTE,
                 clock=time.time):
        self.store, self.provider, self.origin, self.rp_id = store, provider, origin.rstrip("/"), rp_id
        self.per_ip_day, self.clock = per_ip_day, clock
        self.redirect = self.origin + REDIRECT_PATH
        self.flows = Expiring(FLOW_S, MOST_FLOWS, clock, per_owner=PER_OWNER)
        self.sessions = Expiring(SESSION_S, MOST_SESSIONS, clock)
        self.challenges = Expiring(CHALLENGE_S, MOST_CHALLENGES, clock, per_owner=PER_OWNER)
        self.limiter = Limiter(per_minute, clock=clock)
        self.lock = threading.Lock()
        self.n = {k: 0 for k in ("checked", "issued", "rotated", "revoked", "signed_in", "refused")}

    # who

    def ident(self, handler) -> str:
        """The address a request is counted by: IPv4, IPv6's /64 (X-Forwarded-For from a trusted proxy), this PC."""
        w = N.who_of(handler)
        if w.local:
            return N.LOCAL
        return N.tenant_of(w.address) if w.address is not None else "ip:unknown"

    def session(self, handler):
        sid = cookies(handler).get(COOKIE)
        if not sid or len(sid) > 128:
            return None, None
        h = sha256(sid).hex()
        return h, self.sessions.get(h)

    def new_session(self, account: str, how: str) -> str:
        sid = secrets.token_urlsafe(32)
        mine = self.sessions.keys_where(lambda s: s["account"] == account)
        for h in mine[: max(0, len(mine) - SESSIONS_AN_ACCOUNT + 1)]:
            self.sessions.pop(h)
        self.sessions.put(sha256(sid).hex(), {"account": account, "how": how, "reg": None}, evict=True)
        return sid

    def _count(self, what: str) -> None:
        with self.lock:
            self.n[what] += 1

    # GET

    def _redirect(self, handler, where: str, headers=()) -> None:
        _send(handler, 302, None, [("Location", where), *headers])

    def _back(self, handler, code: str, headers=()) -> None:
        assert code in ERRORS
        if code != "denied":
            self._count("refused")
        self._redirect(handler, f"{self.origin}/keys#error={code}", headers)

    def start(self, handler) -> None:
        ident = self.ident(handler)
        if not self.limiter.allow(ident) or self.store.full_today(ident, self.per_ip_day):
            return self._back(handler, "rate_limited")
        state, nonce, verifier, browser = secrets.token_urlsafe(32), secrets.token_urlsafe(32), secrets.token_urlsafe(64), secrets.token_urlsafe(32)
        self.flows.put(state, {"nonce": nonce, "verifier": verifier, "browser": sha256(browser)}, owner=group(ident))
        self._redirect(handler, self.provider.authorize(self.redirect, state, nonce, verifier),
                       [("Set-Cookie", _cookie(FLOW_COOKIE, browser, REDIRECT_PATH, FLOW_S))])

    def callback(self, handler, query: str) -> None:
        clear = [("Set-Cookie", _cookie(FLOW_COOKIE, "", REDIRECT_PATH, 0))]
        if not self.limiter.allow(self.ident(handler)):
            return self._back(handler, "rate_limited", clear)
        try:
            q = urllib.parse.parse_qs(query, keep_blank_values=True, max_num_fields=12)
        except ValueError:
            return self._back(handler, "bad_request", clear)
        one = lambda k: (q.get(k) or [""])[0]  # noqa: E731
        state = one("state")
        flow = self.flows.pop(state) if 0 < len(state) <= 128 else None
        if flow is None:
            return self._back(handler, "expired", clear)
        browser = cookies(handler).get(FLOW_COOKIE) or ""
        if not hmac.compare_digest(sha256(browser), flow["browser"]) or not browser:
            return self._back(handler, "browser", clear)
        code = one("code")
        if one("error") or not 0 < len(code) <= 2048:
            return self._back(handler, "denied", clear)
        try:
            token = self.provider.exchange(code, flow["verifier"], self.redirect)
        except Exception as e:  # noqa: BLE001
            print(f"[serve] keys: the code's exchange with Google failed ({C.why(e)})", flush=True)
            return self._back(handler, "google", clear)
        try:
            sub = self.provider.subject(token, flow["nonce"])
        except Bad:
            return self._back(handler, "token", clear)
        account = self.store.account_id(self.provider.kind, sub)
        sid = self.new_session(account, "google")
        self._count("checked")
        self._redirect(handler, f"{self.origin}/keys#verified", [*clear, ("Set-Cookie", _cookie(COOKIE, sid, "/keys", SESSION_S))])

    # POST

    def _body(self, handler) -> dict:
        refusal = C.body_refusal(handler.headers, BODY_MOST)
        if refusal is not None:
            handler.close_connection = True
            handler.unread = refusal[1]
            status = refusal[0].args[0]
            raise Refusal(status, "body_too_large" if status == 413 else "bad_body", f"a JSON body of at most {BODY_MOST:,} bytes, with its Content-Length")
        raw = handler.rfile.read(int(handler.headers["Content-Length"]))
        if not raw.strip():
            return {}
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise Refusal(400, "bad_body", "the body is not JSON") from None
        if not isinstance(body, dict):
            raise Refusal(400, "bad_body", "the body is not a JSON object")
        return body

    def _csrf(self, handler) -> None:
        if handler.headers.get("X-Requested-With") != "sclmd":
            raise Refusal(403, "csrf", "this endpoint answers the keys page only (X-Requested-With: sclmd)")
        got = handler.headers.get("Origin") or ""
        if not hmac.compare_digest(got.encode(), self.origin.encode()):
            raise Refusal(403, "csrf", "this endpoint answers the keys page only (Origin)")
        kind = (handler.headers.get("Content-Type") or "application/json").split(";")[0].strip().lower()
        if kind != "application/json":
            raise Refusal(415, "bad_body", "the body is application/json")

    def _signed_in(self, handler) -> tuple:
        h, s = self.session(handler)
        if s is None:
            raise Refusal(401, "no_session", "sign in first: the Google check, or your passkey")
        return h, s

    def _key_of(self, account: str) -> dict:
        rec = self.store.account(account)
        if not rec or not rec.get("key"):
            raise Refusal(404, "no_key", "this account has no key: check with Google and register a passkey")
        return rec

    def _issued(self, f):
        try:
            return f()
        except TooMany as e:
            raise _too_many(e.retry_after) from None

    def api(self, handler, adm, path: str) -> None:
        body = self._body(handler)
        self._csrf(handler)
        ident = self.ident(handler)
        if not self.limiter.allow(ident):
            raise Refusal(429, "rate_limit_exceeded", "too many requests to the keys endpoints from this address: try again in a minute",
                          [("Retry-After", "60")])
        what = path[len("/keys/api/"):]
        if what == "passkey/register/options":
            h, s = self._signed_in(handler)
            if s["how"] != "google":
                raise Refusal(403, "google_check_needed", "a passkey is registered right after the Google check")
            wait = self.store.full_today(ident, self.per_ip_day)
            if wait:
                raise _too_many(wait)
            challenge = secrets.token_bytes(32)
            s["reg"] = (challenge, self.clock())
            handle = self.store.user_handle(s["account"])
            name = self.store.name_for(s["account"])
            host = urllib.parse.urlsplit(self.origin).hostname or self.rp_id
            return _send(handler, 200, {
                "rp": {"id": self.rp_id, "name": host},
                "user": {"id": b64u(handle), "name": f"{host} {name}", "displayName": f"{host} {name}"},
                "challenge": b64u(challenge),
                "pubKeyCredParams": [{"type": "public-key", "alg": a} for a in ALGS],
                "timeout": CHALLENGE_S * 1000,
                "attestation": "none",
                "authenticatorSelection": {"residentKey": "required", "requireResidentKey": True, "userVerification": "preferred"},
                "excludeCredentials": [],
            })
        if what == "passkey/register/verify":
            h, s = self._signed_in(handler)
            with self.lock:
                reg, s["reg"] = s.get("reg"), None  # one try a challenge
            if s["how"] != "google" or reg is None or self.clock() - reg[1] > CHALLENGE_S:
                raise Refusal(400, "invalid_credential", "no registration under way: ask for its options first")
            try:
                passkey = verify_registration(body, reg[0], self.origin, self.rp_id)
                key, replaced = self._issued(lambda: self.store.issue(s["account"], ident, self.per_ip_day, passkey=passkey))
            except Bad as e:
                raise Refusal(400, "invalid_credential", f"the passkey was not taken: {e}") from None
            if replaced:  # a new device after a new Google check: the account's other sessions (a lost device signed in) end now
                self.sessions.drop(lambda v, a=s["account"], mine=s: v["account"] != a or v is mine)
            self._count("issued")
            return _send(handler, 200, {"key": key, "name": self.store.name_for(s["account"]), "replaced": replaced})
        if what == "passkey/login/options":
            challenge = secrets.token_bytes(32)
            self.challenges.put(b64u(challenge), True, owner=group(ident))
            return _send(handler, 200, {"challenge": b64u(challenge), "rpId": self.rp_id, "timeout": CHALLENGE_S * 1000,
                                        "userVerification": "preferred", "allowCredentials": []})
        if what == "passkey/login/verify":
            try:
                account, cid, count = verify_assertion(body, self.origin, self.rp_id, lambda c: self.challenges.pop(b64u(c)) is not None,
                                                       self.store.passkey_of)
            except Bad as e:
                raise Refusal(400, "invalid_credential", f"the sign-in did not verify: {e}") from None
            self.store.counted(account, cid, count)
            sid = self.new_session(account, "passkey")
            self._count("signed_in")
            return _send(handler, 200, {"ok": True}, [("Set-Cookie", _cookie(COOKIE, sid, "/keys", SESSION_S))])
        if what == "me":
            _, s = self._signed_in(handler)
            rec = self._key_of(s["account"])
            quota = getattr(adm, "quota", None)
            usage = quota.usage_of(f"key:{rec['name']}") if quota is not None else None
            return _send(handler, 200, {"name": rec["name"], "created": rec.get("key_created"), "tier": "key", "usage": usage})
        if what == "rotate":
            _, s = self._signed_in(handler)
            self._key_of(s["account"])
            try:
                key, _ = self._issued(lambda: self.store.issue(s["account"], ident, self.per_ip_day, rotate=True))
            except LookupError:
                raise Refusal(404, "no_key", "this account has no key: check with Google and register a passkey") from None
            self._count("rotated")
            return _send(handler, 200, {"key": key})
        if what == "revoke":
            _, s = self._signed_in(handler)
            account = s["account"]
            if not self.store.revoke(account):
                raise Refusal(404, "no_key", "this account has no key")
            self.sessions.drop(lambda v: v["account"] != account)
            self._count("revoked")
            return _send(handler, 200, {"revoked": True}, [("Set-Cookie", _cookie(COOKIE, "", "/keys", 0))])
        if what == "logout":
            h, _ = self.session(handler)
            if h is not None:
                self.sessions.pop(h)
            return _send(handler, 200, {"ok": True}, [("Set-Cookie", _cookie(COOKIE, "", "/keys", 0))])
        raise Refusal(404, "unknown_path", "unknown path under /keys/api")

    def handle(self, handler, adm) -> None:
        path, _, query = handler.path.partition("?")
        path = path.rstrip("/")
        try:
            if handler.command == "GET" and path == "/keys/google/start":
                return self.start(handler)
            if handler.command == "GET" and path == REDIRECT_PATH:
                return self.callback(handler, query)
            if handler.command == "POST" and path.startswith("/keys/api/"):
                return self.api(handler, adm, path)
            if handler.command == "POST":  # a body that is not read: thrown away after the answer
                handler.close_connection, handler.unread = True, _announced(handler)
            if path in ("/keys/google/start", REDIRECT_PATH) or path.startswith("/keys/api/"):
                raise Refusal(405, "method_not_allowed", "GET /keys/google/start and /keys/callback/google; POST /keys/api/*")
            raise Refusal(404, "unknown_path", "unknown path under /keys")
        except Refusal as r:
            if r.status >= 400:
                self._count("refused")
            _error(handler, r)
            unread = getattr(handler, "unread", 0)
            if handler.command == "POST" and unread != 0 and handler.close_connection:
                C.linger(handler.connection, handler.rfile, unread)
                handler.unread = 0


def _announced(handler):
    """The body's announced length (None: not known, read until it stops)."""
    try:
        n = int(handler.headers.get("Content-Length"))
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


# the hooks (scripts/serve_admin.py)

def answer(adm, handler) -> bool:
    """GET or POST under /keys/ with --keys-store: answered here (no API key needed). True: answered."""
    issuer = getattr(adm, "issuer", None)
    if issuer is None or not handler.path.startswith("/keys/"):
        return False
    handler._n1_who = adm.users.who(handler)  # the address (X-Forwarded-For from a trusted proxy); a key it carries is ignored
    handler._r4_key = None
    handler.unread = 0
    try:
        issuer.handle(handler, adm)
    except (ConnectionError, TimeoutError):
        handler.close_connection = True
    except Exception as e:  # noqa: BLE001  (its type only: ZDR)
        print(f"[serve] keys: {handler.command} under /keys failed ({C.why(e)})", flush=True)
        handler.close_connection = True
        try:
            if handler.command == "GET":  # a browser's navigation (the Google check): back to the page
                issuer._back(handler, "server")
            else:
                _error(handler, Refusal(500, "server_error", "the server failed while answering"))
        except Exception:  # noqa: BLE001
            pass
    return True


def add_options(ap) -> None:
    g = ap.add_argument_group("key issuance (scripts/serve_keys.py; off by default; design.md 6.20, 6.22)")
    g.add_argument("--keys-store", default="", help="issue API keys at /keys/*: one Google check (OpenID Connect, scope openid), then a "
                   "passkey (WebAuthn); the key is shown once. The store (a JSON file OUTSIDE the repository; its HMAC secret is made "
                   "beside it as <store>.secret) keeps the keys' SHA-256, HMACs of the Google accounts and the passkeys' public keys. "
                   "Needs --oauth-file and --public-origin, and API keys or --anonymous (\"\": off, as before)")
    g.add_argument("--oauth-file", default="", help="Google's OAuth client file (JSON, \"web\" client): read at start, kept in memory, "
                   "never printed. Its redirect URI must be --public-origin + /keys/callback/google")
    g.add_argument("--public-origin", default="", help="the origin the keys page is served at, e.g. https://ssalmuck-ai.duckdns.org: "
                   "POSTs to /keys/api/* must carry it as Origin, passkeys are bound to it, and Google returns to it")
    g.add_argument("--keys-rp-id", default="", help="the passkeys' rp id (\"\": the host of --public-origin)")
    g.add_argument("--keys-per-ip-day", type=int, default=KEYS_A_DAY, help="keys issued (a Google check's or a rotation's) to one address "
                   "(IPv4, IPv6's /64) in 24 hours; past it 429 rate_limit_exceeded (0: no limit)")
    g.add_argument("--oauth-jwks", default=GOOGLE_JWKS, help="the URL of the id_tokens' signing keys (Google's; the tests give a stand-in's)")


def _inside(path: str, folder: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(os.path.abspath(path)), os.path.normcase(folder)]) == os.path.normcase(folder)
    except ValueError:  # another drive
        return False


def check(cli, keys) -> None:
    """Checks made before the model loads (the client file read and let go; install reads it again)."""
    path = getattr(cli, "keys_store", "")
    if not path:
        return
    if _inside(path, ROOT):
        raise SystemExit(f"--keys-store {path} is inside the repository (pushed to a mirror): put it outside")
    if not (keys or getattr(cli, "anonymous", False)):
        raise SystemExit("--keys-store needs API keys or --anonymous: without either the server tells no users apart and reads no key")
    if any(name.startswith("g-") for name, _ in keys):
        raise SystemExit("--keys-store: the names g-... are the store's keys': rename the API key that has one")
    if not getattr(cli, "oauth_file", ""):
        raise SystemExit("--keys-store needs --oauth-file (Google's OAuth client)")
    if getattr(cli, "keys_per_ip_day", 0) < 0:
        raise SystemExit("--keys-per-ip-day is 0 (no limit) or more")
    origin_and_rp(cli)
    Google.from_file(cli.oauth_file, getattr(cli, "oauth_jwks", GOOGLE_JWKS))


def origin_and_rp(cli) -> tuple:
    origin = (getattr(cli, "public_origin", "") or "").rstrip("/")
    u = urllib.parse.urlsplit(origin)
    if not origin or u.path or u.query or u.fragment or not u.hostname:
        raise SystemExit("--keys-store needs --public-origin: the scheme and host the keys page is served at, e.g. https://ssalmuck-ai.duckdns.org")
    _https_or_loopback(origin, "--public-origin")
    rp = getattr(cli, "keys_rp_id", "") or u.hostname
    if not (u.hostname == rp or u.hostname.endswith("." + rp)):
        raise SystemExit(f"--keys-rp-id {rp}: the host of --public-origin or a domain it is under")
    return origin, rp


def install(adm, cli) -> None:
    """adm.issuer, and the server's key check takes the store's keys beside the file's (no restart: a revoked key is
    refused at once)."""
    path = getattr(cli, "keys_store", "")
    if not path:
        return
    origin, rp = origin_and_rp(cli)
    provider = Google.from_file(cli.oauth_file, getattr(cli, "oauth_jwks", GOOGLE_JWKS))
    store = Store(path)
    adm.issuer = Issuer(store, provider, origin, rp, per_ip_day=cli.keys_per_ip_day)
    users = adm.users
    own = users.name_of

    def name_of(keys, got):
        name = own(keys, got) if own is not None else None
        return name if name is not None else store.name_of(got)

    users.name_of = name_of
    print(f"[serve] keys at {origin}/keys: a Google check (openid only), then a passkey for {rp}; store {store.path} "
          f"({store.count()} keys); at most {cli.keys_per_ip_day or 'any number of'} keys an address a day", flush=True)
