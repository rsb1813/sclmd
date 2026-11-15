# What the server (scripts/serve.py) promises its clients, apart from the reply itself:
# - request fields it does not serve: by default refused (400, naming the field) only where no usable answer can be
#   given (n other than 1, logprobs, top_logprobs above 0, a strict json_schema); every other field it would ignore is
#   accepted and named in the response header x-sclmd-ignored and the log. A neutral value (n: 1, logprobs: false,
#   penalties 0, tools: [], response_format text, modalities text) is what the server does anyway and passes.
#   --lenient-fields refuses none (as before R1); --strict-fields refuses every one it would ignore.
# - stop: a string or a list of strings; the answer a stop string cut is what the conversation is kept with. A stop
#   that is not text (false, 0, 5, an object, a list with other things in it) is ignored for what is not text and named.
# - the body: its Content-Length is needed (411), and it is read in pieces of 1 MB, so that a length announced and
#   not sent takes no memory. Options, off by default because each refuses something the server served before R1:
#   a cap on the body (--max-body-mb, 413), a bound on connections at once (--max-connections, 503), and closing
#   connections idle for --idle-timeout seconds. A request answered without its body being read (413, 411, 503) is
#   answered first and its body read and thrown away before the connection closes, so that a client that sends its
#   whole body before it reads (http.client, the openai library) gets the answer instead of a reset. Also off by
#   default: --look-gone, looking every second while the tokens of a reply that is not streamed come whether its client
#   has left (a client that closes its sending side after the request looks the same, and would get no answer).
# - usage.completion_tokens_details.reasoning_tokens: the reply's tokens from <|channel> to <channel|>.
# - ZDR (design.md 6.16, 6.17): nothing of a prompt or a reply reaches a log. An error the server logs is written by
#   why() and trace(): the exception's type and where it was raised, never its message (a ValueError of int() carries
#   the value it could not read, a template's error the text it could not write). quiet_hooks() does the same for an
#   exception no except clause catches (one that escapes a thread, the main program or a __del__). The client's own
#   error answers are as before.
import socket
import sys
import threading
import time
import traceback
from http.server import ThreadingHTTPServer

MAX_BODY = 0  # bytes of a request body at most (--max-body-mb; 0: no cap, as before R1; 32 MB suggested: a prompt the context could hold is far less)
IDLE_TIMEOUT = None  # seconds a connection may sit idle between requests (--idle-timeout; None: for ever, as before R1; 120 suggested)
LOOK_S = 0.0  # seconds between looks whether the client of a reply that is not streamed has left, while its tokens come (--look-gone; 0: off, as before R1)
PIECE = 1 << 20  # a body is read in pieces of this many bytes
LINGER_S = 30.0  # seconds at most that a body which is refused unread is read and thrown away after the answer
LINGER_IDLE_S = 2.0  # ... and how long the client may send nothing before the connection is closed anyway
MODES = ("lenient", "default", "strict")


class Refused(Exception):
    """A request that cannot be served: (HTTP status, message[, the field it is about[, the error's code]])."""


def why(e: BaseException) -> str:
    """An exception for a log line (ZDR): its type, never its message, which may carry a piece of a prompt or a
    reply. One kind of message is kept: a RuntimeError (torch's) whose first line is the CUDA driver's or allocator's
    own words ("CUDA error: an illegal memory access ...", "CUDA out of memory ..."), which no request writes."""
    name = type(e).__name__
    first = str(e).splitlines()[0][:200] if isinstance(e, RuntimeError) and str(e) else ""
    return f"{name}: {first}" if first.startswith(("CUDA error", "CUDA out of memory", "CUDA driver")) else name


def trace(e: BaseException, file=None) -> None:
    """A traceback for the log without the exception's message (ZDR): the frames (file, line, function and the line of
    code), then the exception's type (why()); the exceptions it came from as well, as Python prints them."""
    out = file if file is not None else sys.stderr
    chain, seen = [], set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        chain.append(e)
        e = e.__cause__ if e.__cause__ is not None else (None if e.__suppress_context__ else e.__context__)
    lines = []
    for k, x in enumerate(reversed(chain)):
        if k:
            lines.append("\nDuring the handling of the exception above, another one was raised:\n")
        lines.append("Traceback (most recent call last; the message is left out: ZDR):")
        for frame in traceback.extract_tb(x.__traceback__):
            lines.append(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}')
            if frame.line:
                lines.append(f"    {frame.line.strip()}")
        lines.append(why(x))
    try:
        print("\n".join(lines), file=out, flush=True)
    except (OSError, ValueError):
        pass


def quiet_hooks() -> None:
    """ZDR for what no except clause catches: an exception that escapes the main program, a thread, or a __del__ (or a
    callback Python cannot raise from) is printed as trace() prints it, the frames and the type, never the message.
    Python's default hooks print the message, which may carry a piece of a prompt or a reply."""
    def main_hook(kind, value, tb):
        trace(value.with_traceback(tb) if value is not None else kind())

    def thread_hook(args):
        if args.exc_type is SystemExit:  # as Python's own hook: a thread that exits is not an error
            return
        name = args.thread.name if args.thread is not None else "?"
        print(f"Exception in thread {name}:", file=sys.stderr, flush=True)
        value = args.exc_value if args.exc_value is not None else args.exc_type()
        trace(value.with_traceback(args.exc_traceback))

    def unraisable_hook(u):  # u.err_msg and u.object are not printed: Python's text for them may hold an object's repr
        print("Exception ignored (raised where it could not go on, such as a __del__):", file=sys.stderr, flush=True)
        value = u.exc_value if u.exc_value is not None else u.exc_type()
        trace(value.with_traceback(u.exc_traceback))

    sys.excepthook = main_hook
    threading.excepthook = thread_hook
    sys.unraisablehook = unraisable_hook


def _number(v):
    """v as a number, when it is one or a string of one (true and false are not) -> float or None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _yes(v) -> bool:
    """Whether v says yes: true, a number other than 0, or the string "true"."""
    if isinstance(v, bool):
        return v
    n = _number(v)
    return n != 0 if n is not None else isinstance(v, str) and v.strip().lower() == "true"


def _no(v) -> bool:
    """Whether v says no, or nothing: null, false, 0, "", "false"."""
    if v is None or v is False or v == "":
        return True
    n = _number(v)
    return n == 0 if n is not None else isinstance(v, str) and v.strip().lower() == "false"


def _strict_schema(v) -> bool:
    return isinstance(v, dict) and v.get("type") == "json_schema" and isinstance(v.get("json_schema"), dict) and _yes(v["json_schema"].get("strict"))


def _text_only(v) -> bool:
    return v is None or v == "text" or (isinstance(v, (list, tuple)) and all(m == "text" for m in v))


# Ignored, these would leave the client with no usable answer (another number of choices, no logprobs it asked for, a
# reply its schema may not parse): refused unless --lenient-fields. Each: (refused, named, why).
SHAPE = {
    "n": (lambda v: _number(v) is not None and _number(v) != 1, lambda v: v is not None and v is not True and _number(v) is None,
          "one choice a request is served: send n: 1 (or ask again)"),
    "logprobs": (_yes, lambda v: not _yes(v) and not _no(v), "logprobs are not served"),
    "top_logprobs": (lambda v: (_number(v) or 0) > 0, lambda v: _number(v) is None and not _no(v), "logprobs are not served"),
    "response_format": (_strict_schema, lambda v: False,  # the others: STEER
                        "a strict json_schema is not enforced here (without strict, or as json_object, it is accepted and not enforced)"),
}
# Ignored, these change only how the reply is sampled, steered or carried (the reply itself is a usable text):
# accepted and named (x-sclmd-ignored, the log); refused with --strict-fields.
STEER = {
    "presence_penalty": lambda v: not _no(v),
    "frequency_penalty": lambda v: not _no(v),
    "repetition_penalty": lambda v: v is not None and _number(v) != 1,
    "logit_bias": lambda v: bool(v),
    "min_p": lambda v: not _no(v),
    "response_format": lambda v: not (v is None or v == "text" or (isinstance(v, dict) and v.get("type") in (None, "text"))) and not _strict_schema(v),
    "modalities": lambda v: not _text_only(v),  # audio as well as text: the text alone comes
    "audio": lambda v: v is not None,
    # tools are served (sclmd/tools.py); a tool_choice of required or a function named is not enforced (no grammar)
    "tool_choice": lambda v: v not in (None, "none", "auto"),
    "functions": lambda v: bool(v),
    "function_call": lambda v: v not in (None, "none", "auto"),
    "web_search_options": lambda v: v is not None,
    "stream_options": lambda v: v is not None and not isinstance(v, dict),
    "thinking": lambda v: isinstance(v, dict) and v.get("budget_tokens") is not None,  # thinking is on, without a budget
}
NAMES = {"thinking": "thinking.budget_tokens"}  # what a field is called in the answer, where it is a part of one


def check_fields(body, mode: str = "default") -> list:
    """The fields of a request that the server would ignore -> their names (sorted); raises Refused (400) for the ones
    `mode` refuses. A body that is not an object passes here (the request's own checks refuse it)."""
    if not isinstance(body, dict):
        return []
    ignored = []
    for name, (refused, odd, why) in SHAPE.items():
        if name in body and refused(body[name]):
            if mode != "lenient":
                raise Refused(400, f"{name}: {why}", name)
            ignored.append(name)
        elif name in body and odd(body[name]):  # a value that asks for nothing the server knows: ignored like the others
            if mode == "strict":
                raise Refused(400, f"{name}: {body[name]!r:.40} is not a value served (--strict-fields)", name)
            ignored.append(name)
    for key, bad in STEER.items():
        name = NAMES.get(key, key)
        if key in body and bad(body[key]) and name not in ignored:
            if mode == "strict":
                raise Refused(400, f"{name} is not served (this server refuses the fields it would ignore: --strict-fields)", name)
            ignored.append(name)
    return sorted(ignored)


NAMES_MOST = 40  # names written in x-sclmd-ignored at most (the rest counted): a header line has room for 64 KB


def names_of(ignored: list) -> str:
    """The names of what a request gave and is not served, for the header x-sclmd-ignored and the log."""
    if len(ignored) <= NAMES_MOST:
        return ", ".join(ignored)
    return ", ".join(ignored[:NAMES_MOST]) + f", and {len(ignored) - NAMES_MOST} more"


def stops_of(body, mode: str = "default") -> tuple:
    """The request's stop strings (empty ones left out) -> (strings, whether the stop was not a string or a list of them).
    Before R1 any value that is not true (false, 0) meant no stop, and the strings of a list (or the keys of an object)
    were taken and the rest dropped; those requests are served as then, with "stop" named. A number or true, which
    dropped the connection then, is no stop. --strict-fields refuses (400) whatever is not a string or a list of them."""
    stops = body.get("stop") if isinstance(body, dict) else None
    if stops is None or stops == "" or stops == []:
        return [], False
    if isinstance(stops, str):
        return [stops], False
    kept = [s for s in stops if isinstance(s, str)] if isinstance(stops, (list, tuple, dict)) else []
    odd = not isinstance(stops, (list, tuple)) or len(kept) < len(stops)
    if odd and mode == "strict":
        raise Refused(400, "stop: a string or a list of strings (--strict-fields)", "stop")
    return [s for s in kept if s], odd


def include_usage(body) -> bool:
    options = body.get("stream_options") if isinstance(body, dict) else None
    return isinstance(options, dict) and bool(options.get("include_usage"))


def cut_at_stops(text: str, stops) -> str:
    """The text before the first stop string in it."""
    cut = min((text.find(s) for s in stops if s and s in text), default=-1)
    return text if cut < 0 else text[:cut]


def body_refusal(headers, most: int = MAX_BODY):
    """Why a body of these headers is not read -> (Refused, the bytes the client will send: None when not known), or None."""
    length = headers.get("Content-Length")
    if length is None:
        return Refused(411, "Content-Length is needed (a chunked body is not read)"), None
    try:
        n = int(length)
    except ValueError:
        n = -1
    if n < 0:
        return Refused(400, f"Content-Length is not a number of bytes: {length[:40]!r}"), None
    if most and n > most:
        return Refused(413, f"the body is {n:,} bytes: at most {most:,} are read (--max-body-mb)"), n
    return None


def read_body(handler, most: int = MAX_BODY):
    """The request's body, of the length its Content-Length gives (at most `most` bytes when that is not 0), read in
    pieces: memory for what has come, not for what was announced (reading it at once allocates the whole length first:
    2 GB for a Content-Length of 2 GB). A body that is refused here is not read into memory: the connection is closed
    after the answer (handler.unread: what is thrown away, linger). -> bytes or bytearray (json.loads takes both)."""
    refusal = body_refusal(handler.headers, most)
    if refusal is not None:
        handler.close_connection = True
        handler.unread = refusal[1]
        raise refusal[0]
    n = int(handler.headers["Content-Length"])
    if n <= PIECE:
        return handler.rfile.read(n)
    body = bytearray()
    while len(body) < n:
        piece = handler.rfile.read(min(PIECE, n - len(body)))
        if not piece:  # the client stopped sending: what came (as before R1, the request is then not JSON)
            break
        body += piece
    return body


def linger(sock, rfile=None, unread=None, seconds: float = LINGER_S, idle: float = LINGER_IDLE_S) -> int:
    """After the answer to a request whose body was not read: the connection's sending side is closed (the client has
    the whole answer), then what the client still sends is read and thrown away, `unread` bytes (None: until it stops),
    for at most `seconds`, and at most `idle` seconds without a byte. Closing a socket with bytes unread resets the
    connection, and a client that sends its whole body before it reads (http.client, the openai library) then loses the
    answer it was sent. rfile: the connection's buffered reader, which may hold bytes read with the headers.
    -> the bytes thrown away. Memory: 64 KB whatever the body."""
    try:
        sock.shutdown(socket.SHUT_WR)
    except OSError:
        return 0
    end, thrown = time.monotonic() + seconds, 0
    buf = bytearray(1 << 16)
    try:
        while unread is None or thrown < unread:
            left = end - time.monotonic()
            if left <= 0:
                break
            sock.settimeout(min(idle, left))
            n = len(buf) if unread is None else min(len(buf), unread - thrown)
            got = len(rfile.read1(n)) if rfile is not None else sock.recv_into(buf, n)
            if not got:
                break
            thrown += got
    except OSError:  # the time is up, or the client is gone
        pass
    return thrown


class Reasoning:
    """Counts a reply's thinking tokens as they come: from <|channel> to <channel|>, both counted."""

    def __init__(self, opening: int, closing: int):
        self.opening, self.closing, self.inside, self.count = opening, closing, False, 0

    def add(self, token: int) -> None:
        if token == self.opening:
            self.inside = True
        if self.inside:
            self.count += 1
        if token == self.closing:
            self.inside = False


def with_reasoning(usage: dict, reasoning: int) -> dict:
    return {**usage, "completion_tokens_details": {**usage.get("completion_tokens_details", {}), "reasoning_tokens": reasoning}}


BUSY = b'{"error": {"message": "the server has as many connections as it serves at once (--max-connections): try again", "type": "server_error", "param": null, "code": null}}'


def add_options(ap) -> None:
    g = ap.add_argument_group("the API's contract (scripts/serve_contract.py)")
    g.add_argument("--lenient-fields", action="store_true", help="refuse no request for fields that are not served (n > 1, "
                   "logprobs, a strict json_schema): ignore them and name them in the header x-sclmd-ignored, as for penalties")
    g.add_argument("--strict-fields", action="store_true", help="refuse (400) every request with a field that is not served "
                   "and would be ignored (penalties, logit_bias, tool_choice required, response_format json_object ...), not only those whose answer "
                   "would have another shape")
    g.add_argument("--max-body-mb", type=float, default=MAX_BODY / 2**20, help="a request body larger than this is refused (413) "
                   "without being read into memory (0: no cap, as before; 32 suggested: a prompt the context can hold is far less)")
    g.add_argument("--idle-timeout", type=float, default=IDLE_TIMEOUT or 0, help="seconds an idle connection is kept open between "
                   "requests (0: for ever, as before; 120 suggested). A client that uses a connection again after that "
                   "(http.client does not look first) gets an error for that request")
    g.add_argument("--max-connections", type=int, default=0, help="connections served at once (a thread each); one past them "
                   "is answered 503 at once (0: no bound, as before; 256 suggested)")
    g.add_argument("--look-gone", type=float, default=LOOK_S, help="seconds between looks, while the tokens of a reply that is not "
                   "streamed keep coming, whether its client has left, so that the engine stops writing it (0: off, as before: the "
                   "connection is looked at only after a second without a token; 1 suggested). A client that closes its sending "
                   "side after its request (shutdown) looks the same as one that has left, and gets no answer")


def apply(cli, app, handler) -> None:
    if cli.lenient_fields and cli.strict_fields:
        raise SystemExit("--lenient-fields and --strict-fields: one or the other")
    app.fields = "lenient" if cli.lenient_fields else "strict" if cli.strict_fields else "default"
    app.max_body = int(cli.max_body_mb * 2**20)
    app.look_s = max(0.0, cli.look_gone)
    handler.timeout = cli.idle_timeout if cli.idle_timeout > 0 else None


BUSY_ANSWER = (b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json; charset=utf-8\r\nRetry-After: 1\r\n"
               b"Connection: close\r\nContent-Length: " + str(len(BUSY)).encode() + b"\r\n\r\n" + BUSY)
REFUSERS = 16  # connections past the bound that are answered at once, each by a thread of its own for up to 2 s
REFUSE_S = 2.0


class BoundedHTTPServer(ThreadingHTTPServer):
    """A thread for each connection, as ThreadingHTTPServer, but at most `most` at once (0: no bound): a connection
    past them is answered 503 at once and closed. The answer is written before the request is read; what the client
    sends is then read and thrown away for up to 2 s (linger) by a short-lived thread, at most REFUSERS at once, so
    that the client reads the 503 rather than a reset. Past those too, the 503 is written and the connection closed."""

    daemon_threads = True
    # connections the OS holds for accept(): socketserver's 5 reset a burst (100 clients at once on Windows: 63 to 67
    # of them got WinError 10054 before the server saw them; R4's flood test)
    request_queue_size = 128

    def __init__(self, address, handler, most: int = 0):
        super().__init__(address, handler)
        self.most = most
        self.slots = threading.BoundedSemaphore(most) if most > 0 else None
        self.refusers = threading.BoundedSemaphore(REFUSERS)
        self.turned_away = 0
        self.live, self._count = 0, threading.Lock()  # connections being served now (a thread each)

    def _refuse(self, request, linger_s: float) -> None:
        try:
            request.settimeout(1.0)
            request.sendall(BUSY_ANSWER)
            if linger_s > 0:
                linger(request, None, None, seconds=linger_s, idle=min(LINGER_IDLE_S, linger_s))
        except OSError:
            pass
        self.shutdown_request(request)

    def _refuse_thread(self, request) -> None:
        try:
            self._refuse(request, REFUSE_S)
        finally:
            self.refusers.release()

    def process_request(self, request, client_address):
        if self.slots is not None and not self.slots.acquire(blocking=False):
            self.turned_away += 1
            if self.refusers.acquire(blocking=False):
                try:
                    threading.Thread(target=self._refuse_thread, args=(request,), daemon=True).start()
                    return
                except RuntimeError:  # no thread to be had: answered here, as below
                    self.refusers.release()
            self._refuse(request, 0)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            if self.slots is not None:
                self.slots.release()
            raise

    def handle_error(self, request, client_address) -> None:
        """A handler that failed outside its own error answers (socketserver prints the whole traceback, its message
        too): the trace without the message (ZDR)."""
        e = sys.exc_info()[1]
        if isinstance(e, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):  # the client's doing: a line, no trace
            print(f"[serve] a connection ended under its handler: {why(e)}", file=sys.stderr, flush=True)
            return
        print(f"[serve] a connection's handler failed: {why(e) if e is not None else 'unknown'}", file=sys.stderr, flush=True)
        if e is not None:
            trace(e)

    def process_request_thread(self, request, client_address):
        with self._count:
            self.live += 1
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._count:
                self.live -= 1
            if self.slots is not None:
                self.slots.release()
