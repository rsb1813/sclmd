# The internet server's base (N1, design.md 6.16 and 6.17; scripts/serve.py and scripts/serve_admin.py call in
# through a few hooks). All of it is off unless its option is given: without them a request is answered, and logged,
# as before.
#
# Who a request is (who(): one function of the connection's address, the request's headers, the API keys and the
# options; it reads no body):
# - a key (Authorization: Bearer, or x-api-key) the server has -> "key:<its name>"; a key it does not have -> 401,
#   never the address;
# - no key -> the client's address: "ip:203.0.113.7", or for IPv6 its /64, "ip:2001:db8:1:2::/64" (an IPv4 address
#   written as IPv6, ::ffff:203.0.113.7, is the IPv4 one). Served even when keys are set with --anonymous (without it,
#   as R4: 401). The address is the connection's; for a connection from --trusted-proxies (the proxy PC), the first
#   address of X-Forwarded-For from the right that is not a trusted proxy (every X-Forwarded-For header, in order);
# - a loopback connection that came through no proxy -> "local" (this PC). An address read from X-Forwarded-For is
#   never "local", whatever it says, and neither is a loopback connection that carries X-Forwarded-For or Forwarded
#   (a proxy or tunnel on this PC that is not in --trusted-proxies): it is "ip:127.0.0.1".
# Tenancy is on when the server has API keys or --anonymous; off (None) otherwise, as before N1.
#
# The seams of the stages after N1 (quotas and priority; prompt caches): every request object gets
#   req.tenant    the id above, None when tenancy is off;
#   req.priority  an int, 0 for everyone here (Users.priority_of: the quota stage fills it);
#   req.scope     the scope of its prompt caches: the tenant when caches are per tenant (Users.per_tenant), None
#                 (shared, as today) otherwise (Users.scope_of: the cache stage reads it);
# and who_of(handler) is the Who of the request a handler is answering (GET or POST, refused or not).
#
# Limits: --max-prompt (a prompt above it: 400 context_length_exceeded, OpenAI's shape), --max-output (max_tokens above
# it is cut to it and named in x-sclmd-ignored), --stream-ping (": ping" every S seconds until the first token),
# --metrics-local (GET /metrics only from this PC). Nothing here imports torch.
import ipaddress
import threading

LOCAL = "local"
METRICS_PATH = "/metrics"


class Who:
    """Who a request is. tenant: the id (None: tenancy is off); key: the API key's name (None: none); address: the
    client's address as read (an ipaddress object, None if the connection's is not an address); local: this PC
    (loopback, no proxy); proxied: the address came from X-Forwarded-For; refusal: None, "unknown_key" (a key the
    server does not have) or "key_needed" (keys are set, no key given, and no --anonymous)."""

    __slots__ = ("tenant", "key", "address", "local", "proxied", "refusal")

    def __init__(self, tenant=None, key=None, address=None, local=False, proxied=False, refusal=None):
        self.tenant, self.key, self.address, self.local, self.proxied, self.refusal = tenant, key, address, local, proxied, refusal

    def __repr__(self) -> str:
        return f"Who(tenant={self.tenant!r}, key={self.key!r}, local={self.local}, proxied={self.proxied}, refusal={self.refusal!r})"


NOBODY = Who()  # what a request is without the operator's side (the tests' stand-ins): as before N1


def address(text):
    """An address as a connection or X-Forwarded-For gives it -> an ipaddress object, or None. Brackets and a port
    ([2001:db8::1]:443, 203.0.113.7:5000) are taken off; ::ffff:203.0.113.7 is the IPv4 address."""
    if not isinstance(text, str):
        return None
    s = text.strip().strip('"')
    if s.startswith("["):
        if "]" not in s:
            return None
        s = s[1:].split("]", 1)[0]
    elif s.count(":") == 1:  # an IPv4 address and a port
        s = s.split(":", 1)[0]
    try:
        ip = ipaddress.ip_address(s)
    except ValueError:
        return None
    if ip.version == 6 and ip.scope_id:  # fe80::1%eth0: the address without its zone
        ip = ipaddress.IPv6Address(int(ip))
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip


def tenant_of(ip) -> str:
    """The id of a keyless client: IPv4 its address, IPv6 its /64."""
    if ip.version == 4:
        return f"ip:{ip}"
    return f"ip:{ipaddress.IPv6Network((int(ip) >> 64 << 64, 64))}"


def networks(spec) -> tuple:
    """--trusted-proxies: addresses or networks, by commas or spaces -> ip_network objects (raises ValueError)."""
    if isinstance(spec, (list, tuple)):
        spec = ",".join(map(str, spec))
    out = []
    for part in str(spec or "").replace(",", " ").split():
        net = ipaddress.ip_network(part, strict=False)
        out.append(net)
        if net.version == 6 and net.prefixlen >= 96 and net.network_address.ipv4_mapped is not None:  # ::ffff:a.b.c.d
            out.append(ipaddress.ip_network(f"{net.network_address.ipv4_mapped}/{net.prefixlen - 96}", strict=False))
    return tuple(out)


def _trusted(ip, proxies) -> bool:
    return ip is not None and any(ip.version == n.version and ip in n for n in proxies)


def forwarded(headers) -> list:
    """The addresses of every X-Forwarded-For header, left to right as the proxies wrote them."""
    values = headers.get_all("X-Forwarded-For") if hasattr(headers, "get_all") else [headers.get("X-Forwarded-For")]
    out = []
    for v in values or ():
        if v:
            out += [p.strip() for p in str(v).split(",") if p.strip()]
    return out


def origin(peer, headers, proxies=()) -> tuple:
    """(the client's address, whether it is this PC, whether it came from X-Forwarded-For). peer: the connection's
    address. From a trusted proxy with X-Forwarded-For: its addresses from the right, the first that is not a trusted
    proxy; when every one is (the proxy PC itself), the leftmost; when one cannot be read (not an address), the proxy's
    own: a header that a trusted proxy wrote wrongly does not give an address it did not see. From anywhere else
    X-Forwarded-For is not read. this PC: a loopback connection that carries neither X-Forwarded-For nor Forwarded. A
    loopback connection with one of them came through a proxy or tunnel on this PC (one not in --trusted-proxies): it
    is its loopback address as a keyless user, not this PC, so a tunnel here does not make every internet user local."""
    ip = address(peer)
    chain = forwarded(headers)
    if _trusted(ip, proxies) and chain:
        for raw in reversed(chain):
            hop = address(raw)
            if hop is None:
                return ip, False, False
            if not _trusted(hop, proxies):
                return hop, False, True
        return address(chain[0]), False, True
    through = bool(chain) or bool(headers.get("Forwarded"))
    return ip, bool(ip is not None and ip.is_loopback and not through), False


def presented(headers) -> bytes | None:
    """The key a request gives: Authorization: Bearer <key>, else x-api-key (as scripts/serve_admin.py reads it)."""
    auth = headers.get("Authorization")
    if auth:
        scheme, _, value = auth.strip().partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip().encode()
    other = headers.get("x-api-key")
    return other.strip().encode() if other and other.strip() else None


def who(peer, headers, keys=(), anonymous: bool = False, proxies=(), name_of=None) -> Who:
    """Who a request is (above). keys: [(name, key bytes)]; name_of(keys, got) -> the name of the key got, compared in
    constant time (scripts/serve_admin.py: key_name)."""
    ip, local, proxied = origin(peer, headers, proxies)
    on = bool(keys) or anonymous
    got = presented(headers)
    if got is not None and on:
        name = name_of(keys, got) if name_of is not None else None
        if name is None:
            return Who(None, None, ip, local, proxied, "unknown_key")
        return Who(f"key:{name}", name, ip, local, proxied)
    if not on:  # no keys and no --anonymous: everything is served, nobody is told apart (as before N1)
        return Who(None, None, ip, local, proxied)
    if keys and not anonymous:
        return Who(None, None, ip, local, proxied, "key_needed")
    if local:
        return Who(LOCAL, None, ip, True, False)
    return Who(tenant_of(ip) if ip is not None else "ip:unknown", None, ip, False, proxied)


class Users:
    """The operator's choices about who a request is, and the seams of the later stages."""

    def __init__(self, keys=(), anonymous: bool = False, proxies=(), metrics_local: bool = False, name_of=None):
        self.keys, self.anonymous, self.proxies, self.metrics_local, self.name_of = list(keys), anonymous, tuple(proxies), metrics_local, name_of
        self.per_tenant = False  # prompt caches apart for each tenant (the cache stage turns it on)
        self._told = False
        self._lock = threading.Lock()

    @property
    def on(self) -> bool:
        return bool(self.keys) or self.anonymous

    def who(self, handler) -> Who:
        peer = handler.client_address[0] if getattr(handler, "client_address", None) else None
        try:
            w = who(peer, handler.headers, self.keys, self.anonymous, self.proxies, self.name_of)
        except Exception as e:  # noqa: BLE001  (should not happen: the request is then read without its headers, so as keyless)
            print(f"[serve] who a request is could not be read ({type(e).__name__}): it is taken as one without a key", flush=True)
            w = who(peer, {}, self.keys, self.anonymous, (), self.name_of)
        if (self.on and not self._told and handler.headers.get("X-Forwarded-For") is not None and not w.local
                and not _trusted(address(peer), self.proxies)):
            with self._lock:  # once a process: a proxy that is not in --trusted-proxies (every user would be the proxy)
                told, self._told = self._told, True
            if not told:
                print(f"[serve] a connection from {peer} carries X-Forwarded-For, and {peer} is not in --trusted-proxies: the header is "
                      "not read (the connection's address is the user). Told once", flush=True)
        return w

    # the later stages' seams: the quota stage replaces priority_of, the cache stage sets per_tenant (or scope_of)

    def priority_of(self, req, w: Who) -> int:
        return 0

    def scope_of(self, req, w: Who):
        return w.tenant if self.per_tenant else None


def who_of(handler) -> Who:
    """The Who of the request this handler is answering (set by serve_admin's get and post hooks), NOBODY without the
    operator's side."""
    return getattr(handler, "_n1_who", None) or NOBODY


def tag(handler, req) -> None:
    """The request object gets who sent it: req.tenant, req.priority, req.scope (the later stages' seams)."""
    adm = getattr(getattr(handler.server, "app", None), "admin", None)
    users = getattr(adm, "users", None)
    if users is None:
        return
    w = who_of(handler)
    req.tenant = w.tenant
    req.priority = users.priority_of(req, w)
    req.scope = users.scope_of(req, w)


def metrics_refused(handler) -> bool:
    """--metrics-local: GET /metrics from anywhere but this PC is answered as an unknown path."""
    adm = getattr(getattr(handler.server, "app", None), "admin", None)
    users = getattr(adm, "users", None)
    return bool(users is not None and users.metrics_local and not who_of(handler).local)


def context_exceeded(most: int, got: int, what: str = "prompt", sure: bool = False) -> tuple:
    """OpenAI's answer to a prompt that is too long: (400, message, param, code) for Refused. sure: refused before the
    tokenizer, by sure_too_long (got is then unknown)."""
    over = (f"your messages have more than {most * TOKEN_BYTES:,} bytes of text, more than {most} tokens can hold" if sure
            else f"your messages resulted in {got} tokens")
    return (400, f"This model's maximum {what} length is {most} tokens. However, {over}. "
                 "Please reduce the length of the messages.", "messages", "context_length_exceeded")


TOKEN_BYTES = 48  # the most UTF-8 bytes of text one Gemma 4 token stands for (measured over all 262,144; added tokens 16 at most)


def text_bytes(x, stop: int = 0) -> int:
    """The UTF-8 bytes of every string in x (messages after sclmd/tools.py's messages_of: dicts and lists), keys left
    out. stop > 0: counting ends once the count passes it."""
    n, todo = 0, [x]
    while todo:
        y = todo.pop()
        if isinstance(y, str):
            n += len(y.encode("utf-8", "surrogatepass"))
            if stop and n > stop:
                return n
        elif isinstance(y, dict):
            todo.extend(y.values())
        elif isinstance(y, (list, tuple)):
            todo.extend(y)
    return n


def sure_too_long(messages, most: int) -> bool:
    """--max-prompt before the template and the tokenizer: messages with more text than `most` tokens can hold (each
    token at most TOKEN_BYTES bytes) are over it whatever they tokenize to (False: not sure, tokenize). The template
    only adds tokens, so no prompt within the limit is refused. A big body is refused at once instead of holding the
    prompt lock for its whole tokenization (37 s for 29.6 MiB, measured, every other request waiting)."""
    return bool(most) and text_bytes(messages, stop=most * TOKEN_BYTES) > most * TOKEN_BYTES


# start-up

def add_options(ap) -> None:
    g = ap.add_argument_group("the internet server (scripts/serve_net.py; all off by default; design.md 6.17)")
    g.add_argument("--anonymous", action="store_true", help="serve a request without an API key even when keys are set, as its "
                   "client's address (IPv4: the address, IPv6: its /64; --trusted-proxies: the address X-Forwarded-For gives); a "
                   "loopback request through no proxy is this PC. A key the server does not have is still 401. Without it: as before "
                   "(with keys, a request without one is 401)")
    g.add_argument("--trusted-proxies", default="", help="addresses or networks (commas) of the proxies whose X-Forwarded-For is read "
                   "(the proxy PC's LAN address): the client is the first address from the right that is not one of them. From any "
                   "other connection the header is not read (\"\": none)")
    g.add_argument("--metrics-local", action="store_true", help="GET /metrics only from this PC (a loopback connection through no "
                   "proxy); from elsewhere it is an unknown path (404)")
    g.add_argument("--max-prompt", type=int, default=0, help="a prompt of more tokens than this is refused: 400 with code "
                   "context_length_exceeded and param messages, as OpenAI answers (0: as before, the context alone bounds it; 32768 in "
                   "serve.public.example.toml)")
    g.add_argument("--max-output", type=int, default=0, help="the most reply tokens a request may ask for: a larger max_tokens "
                   "(or max_completion_tokens) is cut to it and named in x-sclmd-ignored (0: as before, the context alone bounds it; "
                   "8192 in serve.public.example.toml)")
    g.add_argument("--stream-ping", type=float, default=0.0, help="seconds between \": ping\" SSE comments a stream gets until its "
                   "first token, so that a proxy does not close it while a long prompt is read (0: none, as before; 15 in "
                   "serve.public.example.toml). With --queue-timeout the stream's headers, and so the pings, start once it has a sequence")


def users_of(cli, keys, name_of=None) -> Users:
    """The Users of serve.py's options (a namespace without them: all off)."""
    try:
        proxies = networks(getattr(cli, "trusted_proxies", "") or "")
    except ValueError as e:
        raise SystemExit(f"--trusted-proxies: {e}")
    return Users(keys, bool(getattr(cli, "anonymous", False)), proxies, bool(getattr(cli, "metrics_local", False)), name_of)


def check(cli) -> None:
    """Checks made before the model loads."""
    if getattr(cli, "max_prompt", 0) < 0 or getattr(cli, "max_output", 0) < 0 or getattr(cli, "stream_ping", 0.0) < 0:
        raise SystemExit("--max-prompt, --max-output and --stream-ping are 0 (off) or more")
    most = getattr(cli, "max_output", 0)
    if most and getattr(cli, "max_tokens", 0) > most:
        raise SystemExit(f"--max-tokens {cli.max_tokens} is more than --max-output {most}: a request that does not say would get more "
                         "than one that asks")
    try:
        networks(getattr(cli, "trusted_proxies", "") or "")
    except ValueError as e:
        raise SystemExit(f"--trusted-proxies: {e}")


def apply(cli, app) -> None:
    """The limits onto the App (serve.py reads app.max_prompt, app.max_output, app.ping_s)."""
    app.max_prompt = max(0, int(getattr(cli, "max_prompt", 0) or 0))
    app.max_output = max(0, int(getattr(cli, "max_output", 0) or 0))
    app.ping_s = max(0.0, float(getattr(cli, "stream_ping", 0.0) or 0.0))


def line(cli, users: Users) -> str | None:
    """The start-up line of what is on (None: nothing, and the start-up says nothing new)."""
    on = []
    if users.anonymous:
        on.append("a request without a key is served as its address (--anonymous)" + ("" if users.keys else "; no API key is set"))
    elif users.keys:
        on.append("each API key is a user")
    if users.proxies:
        on.append("X-Forwarded-For read from " + ", ".join(str(n) for n in users.proxies))
    if users.metrics_local:
        on.append("/metrics from this PC only")
    if getattr(cli, "max_prompt", 0):
        on.append(f"prompts up to {cli.max_prompt:,} tokens")
    if getattr(cli, "max_output", 0):
        on.append(f"replies up to {cli.max_output:,} tokens")
    if getattr(cli, "stream_ping", 0):
        on.append(f"a stream pings every {cli.stream_ping:g} s until its first token")
    if not (users.anonymous or users.proxies or users.metrics_local or getattr(cli, "max_prompt", 0) or getattr(cli, "max_output", 0)
            or getattr(cli, "stream_ping", 0)):
        return None
    return "[serve] users and limits: " + "; ".join(on)
