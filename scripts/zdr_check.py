# The ZDR gate (design.md 6.16, 6.17): requests that carry a unique canary string everywhere a request can put text
# (messages, a system prompt, tools, their calls and results, stop strings, fields that are ignored or refused, headers,
# an API key, the path, a body that is too large, a prompt that is too long, a stream the client leaves, a failure of
# the server's own through POST /debug/fault), then every file under the folders given (the request log, the status
# file, logs\supervise.log, the server's console captured) and GET /metrics are searched for it: zero hits, or it fails.
#
#   python -m scripts.zdr_check --url http://127.0.0.1:8070 --search <folder or file> ... [--key K] [--debug]
#          [--canned] [--proxy] [--max-prompt N] [--body-mb N]
#
# --canned: the stand-in (scripts/serve_canned.py), whose replies echo the prompt and whose "call NAME" and
# "malformed WORD" put the canary into the model's text where the server logs about tool calls. --proxy: the server
# trusts 127.0.0.1 as a proxy (--trusted-proxies), so a request with X-Forwarded-For is read as one through it.
# The canary is printed only as its length: the gate's own output may be kept anywhere. Nothing here imports the project.
import argparse
import http.client
import json
import os
import secrets
import socket
import sys
import time
from urllib.parse import urlparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True, help="the server, e.g. http://127.0.0.1:8070")
    ap.add_argument("--search", nargs="+", required=True, help="folders and files the server writes (searched whole, recursively)")
    ap.add_argument("--key", default="", help="an API key the server has (the requests are sent with it; a few without)")
    ap.add_argument("--debug", action="store_true", help="the server has --debug: a handler and an engine fault are injected")
    ap.add_argument("--canned", action="store_true", help="the stand-in: its tool-call triggers too")
    ap.add_argument("--proxy", action="store_true", help="the server trusts 127.0.0.1 as a proxy: requests with X-Forwarded-For")
    ap.add_argument("--max-prompt", type=int, default=0, help="the server's --max-prompt: a prompt above it is sent")
    ap.add_argument("--body-mb", type=float, default=0.0, help="the server's --max-body-mb: a body above it is sent")
    ap.add_argument("--wait", type=float, default=7.0, help="seconds to wait before searching (the status file is rewritten every 5 s)")
    ap.add_argument("--canary", default="", help="the string (default: a new random one of 14 characters)")
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args()
    canary = args.canary or "Zdr" + secrets.token_hex(5) + "q"  # 14 characters: no log line's [:40] or [:300] cuts it
    u = urlparse(args.url)
    host, port = u.hostname, u.port
    key = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    sent, statuses, echoed = [], {}, 0

    def raw(method: str, path: str, body: bytes | None = None, headers: dict | None = None, timeout: float = 300, read: bool = True):
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request(method, path, body, {"Content-Type": "application/json", **(headers or {})})
            r = conn.getresponse()
            data = r.read() if read else r.read(200)
            return r.status, data
        except (OSError, http.client.HTTPException) as e:
            return type(e).__name__, b""
        finally:
            conn.close()

    def ask(name: str, body, headers: dict | None = None, path: str = "/v1/chat/completions", method: str = "POST", plain: bool = False):
        nonlocal echoed
        data = body if isinstance(body, (bytes, bytearray)) else (json.dumps(body, ensure_ascii=False).encode() if body is not None else None)
        status, out = raw(method, path, data, {**({} if plain else key), **(headers or {})})
        sent.append(name)
        statuses[name] = status
        echoed += canary.encode() in out
        return status, out

    C = canary
    chat = lambda content, **more: {"model": "gemma-4-31b-it", "messages": [{"role": "system", "content": f"You are the test {C}."},  # noqa: E731
                                                                            {"role": "user", "content": content}],
                                    "max_tokens": args.max_tokens, "temperature": 0, **more}
    repeat = f"Repeat this word exactly, and nothing else: {C}"
    tool = {"type": "function", "function": {"name": f"look_{C}", "description": f"Looks {C} up.",
                                             "parameters": {"type": "object", "properties": {"q": {"type": "string", "description": f"what {C}"}},
                                                            "required": ["q"]}}}

    # served
    ask("chat", chat(repeat))
    ask("stream", chat(repeat, stream=True, stream_options={"include_usage": True}))
    ask("thinking stream", chat(repeat, stream=True, thinking=True, max_tokens=args.max_tokens + 40))
    ask("tools", chat(f"Look {C} up with the tool.", tools=[tool]))
    ask("tools stream", chat(f"Look {C} up with the tool.", tools=[tool], stream=True))
    step = {"model": "x", "max_tokens": args.max_tokens, "temperature": 0, "tools": [tool], "messages": [
        {"role": "user", "content": f"Look {C} up."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": f"look_{C}",
                                                                                                             "arguments": json.dumps({"q": C})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": f"The answer is {C}."}]}
    ask("a tool's result", step)
    ask("a tool's result as an object", {**step, "messages": step["messages"][:2] + [{"role": "tool", "tool_call_id": "call_1", "content": {"x": C}}]})
    ask("arguments that are not JSON", {**step, "messages": [step["messages"][0], {**step["messages"][1], "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": f"look_{C}", "arguments": "{" + C}}]}, step["messages"][2]]})
    ask("a result after no call", {**step, "messages": [{"role": "tool", "content": C}, {"role": "user", "content": C}]})
    ask("tool_choice naming no tool", chat(repeat, tools=[tool], tool_choice={"type": "function", "function": {"name": C}}))
    if args.canned:
        ask("an undeclared call", chat(f"call {C}", tools=[tool]))
        ask("a malformed call block", chat(f"malformed {C}", tools=[tool]))
        ask("a malformed call block, streamed", chat(f"malformed {C}", tools=[tool], stream=True))
    ask("stop strings", chat(repeat, stop=[C, f"x{C}"]))
    ask("a stop that is not text", chat(repeat, stop={C: 1}))
    ask("fields ignored", chat(repeat, presence_penalty=0.5, logit_bias={C: 1}, metadata={"k": C}, user=C, response_format={"type": "json_object"}))
    ask("the model's name", {**chat(repeat), "model": C})
    ask("headers", chat(repeat), {"User-Agent": C, "x-request-id": C, "X-Title": C, "Referer": f"http://{C}/"})
    ask("the path's query", chat(repeat), path=f"/v1/chat/completions?x={C}")
    if args.proxy:
        ask("through the proxy", chat(repeat), {"X-Forwarded-For": "198.51.100.7"}, plain=True)
        ask("an X-Forwarded-For that is not an address", chat(repeat), {"X-Forwarded-For": f"{C}, 198.51.100.8"}, plain=True)
        ask("an X-Forwarded-For whose last hop is not one", chat(repeat), {"X-Forwarded-For": f"198.51.100.9, {C}"}, plain=True)
        ask("IPv6 through the proxy", chat(repeat), {"X-Forwarded-For": "2001:db8:1:2::77"}, plain=True)
        ask("a refusal through the proxy", chat(repeat, max_tokens=C), {"X-Forwarded-For": "198.51.100.7"}, plain=True)
    ask("anonymous", chat(repeat), plain=True)

    # refused: each answer quotes what it could not read; the log must not
    for field in ("max_tokens", "temperature", "top_p", "top_k", "seed"):
        ask(f"{field} that is not a number", chat(repeat, **{field: C}))
    ask("n that is not a number", chat(repeat, n=C))
    ask("n: 2", chat(repeat, n=2))
    ask("logprobs", chat(repeat, logprobs=True))
    ask("a strict json_schema", chat(repeat, response_format={"type": "json_schema", "json_schema": {"name": C, "strict": True, "schema": {}}}))
    ask("the last message is the model's", {"model": "x", "messages": [{"role": "user", "content": C}, {"role": "assistant", "content": C}]})
    ask("a role that is not one", {"model": "x", "messages": [{"role": C, "content": C}]})
    ask("an image", {"model": "x", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"http://{C}/a.png"}}]}]})
    ask("no messages", {"model": C, "messages": []})
    ask("a body that is not JSON", f'{{"messages": [{{"role": "user", "content": "{C}"'.encode())
    ask("a body that is not UTF-8", b'{"messages": [{"role": "user", "content": "' + C.encode() + b'\xff\xfe"}]}')
    ask("a body that is not an object", json.dumps([C]).encode())
    ask("system after user", {"model": "x", "max_tokens": args.max_tokens, "messages": [{"role": "user", "content": C}, {"role": "system", "content": C},
                                                                                          {"role": "user", "content": C}]})
    ask("an unknown path", chat(repeat), path=f"/v1/{C}")
    ask("an unknown path, GET", None, path=f"/{C}", method="GET")
    ask("a method not served", chat(repeat), method=C.upper()[:8])
    ask("an API key not known", chat(repeat), {"Authorization": f"Bearer {C}"}, plain=True)
    ask("an x-api-key not known", chat(repeat), {"x-api-key": C}, plain=True)
    ask("a GET with a key not known", None, {"Authorization": f"Bearer {C}"}, path="/v1/models", method="GET", plain=True)
    if args.max_prompt:
        words = (f"{C} " * (args.max_prompt // 2 + 64)).strip()
        ask("a prompt that is too long", chat(words))
    ask("max_tokens above the cap", chat(repeat, max_tokens=10 ** 6))
    if args.body_mb:
        big = json.dumps(chat(f"{C} " * int(args.body_mb * 2**20 / (len(C) + 1) + 4096))).encode()
        ask("a body that is too large", big)
    # Content-Length longer than the body: the server waits, then reads what came (not JSON)
    try:
        s = socket.create_connection((host, port), timeout=30)
        part = json.dumps(chat(repeat)).encode()[:-5]
        auth = f"Authorization: Bearer {args.key}\r\n" if args.key else ""
        s.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n{auth}Content-Type: application/json\r\nContent-Length: {len(part) + 5}\r\n\r\n".encode() + part)
        s.shutdown(socket.SHUT_WR)
        got = s.recv(400)
        statuses["a body cut short"] = got.split(b" ")[1].decode() if got.startswith(b"HTTP/") else "none"
        sent.append("a body cut short")
        s.close()
    except OSError as e:
        statuses["a body cut short"] = type(e).__name__
    # a stream the client leaves after its first event
    try:
        conn = http.client.HTTPConnection(host, port, timeout=60)
        conn.request("POST", "/v1/chat/completions", json.dumps(chat(repeat, stream=True, max_tokens=200)), {"Content-Type": "application/json", **key})
        r = conn.getresponse()
        r.read1(64)
        conn.close()
        statuses["a stream left"] = r.status
        sent.append("a stream left")
    except (OSError, http.client.HTTPException) as e:
        statuses["a stream left"] = type(e).__name__
    if args.debug:  # the server's own failures (--debug): a handler fault, then an engine fault, each under a request with the canary
        def idle() -> None:
            """Until the engine has nothing under way (GET /debug/audit; a stand-in has none): the fault armed next is
            the canary request's, not the end of a stream the client left."""
            t = time.time()
            while time.time() - t < 20:
                st, out = raw("GET", "/debug/audit", None, key)
                if st != 409:
                    return
                time.sleep(0.2)

        for kind, stream in (("handler", False), ("engine", False), ("engine", True)):
            idle()
            st, _ = raw("POST", "/debug/fault", json.dumps({"kind": kind, "count": 1}).encode(), key)
            statuses[f"/debug/fault {kind}"] = st
            ask(f"a {kind} fault" + (", streamed" if stream else ""), chat(repeat, stream=stream))
        raw("POST", "/debug/fault", json.dumps({"kind": "clear"}).encode(), key)

    time.sleep(args.wait)  # the request log's thread, and the status file's next write
    st, metrics = raw("GET", "/metrics", None, key)
    in_metrics = metrics.count(canary.encode())
    hits, files = {}, 0
    needles = (canary.encode(), canary.encode("utf-16-le"))
    for root in args.search:
        paths = [root] if os.path.isfile(root) else [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
        for p in paths:
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            files += 1
            n = sum(data.count(x) for x in needles)
            if n:
                hits[p] = n
    print(f"[zdr] {len(sent)} requests with a canary of {len(canary)} characters; answers: "
          + ", ".join(f"{k} {v}" for k, v in sorted(_count(statuses).items())))
    print(f"[zdr] the canary came back to the client in {echoed} answers (it reached the server)")
    print(f"[zdr] GET /metrics: {st}, {len(metrics)} bytes, {in_metrics} hits")
    print(f"[zdr] {files} files searched under {', '.join(args.search)}: {sum(hits.values())} hits")
    for p, n in sorted(hits.items()):
        print(f"[zdr]   HIT {n} in {p}")
    for name in sent:
        print(f"[zdr]   {statuses.get(name)}  {name}")
    ok = not hits and not in_metrics and st == 200
    print("[zdr] ZDR OK: no hit" if ok else "[zdr] ZDR FAILED")
    sys.exit(0 if ok else 1)


def _count(statuses: dict) -> dict:
    out = {}
    for v in statuses.values():
        out[str(v)] = out.get(str(v), 0) + 1
    return out


if __name__ == "__main__":
    main()
