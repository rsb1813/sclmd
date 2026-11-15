"""The server's HTTP side (scripts/serve.py: Handler and App, with the real tokenizer and chat template) with a canned
reply in the engine's place: for trying a client against the server's protocol without the GPU or the 15 s of loading.

    python -m scripts.serve_canned 8017            (and the real server's --max-body-mb, --max-connections, ... fields options)
    uv run --no-project --with openai python scripts/openai_client_check.py 8017

and the operator's options (scripts/serve_admin.py: keys, --max-waiting, --queue-timeout, the request log, --debug), and
the internet server's (scripts/serve_net.py: --anonymous, --trusted-proxies, --max-prompt, --max-output, --stream-ping,
--metrics-local; --first-ms holds the first token back, for the pings):
    python -m scripts.serve_canned 8024 --max-waiting 1 --token-ms 300

A user message "fail" makes the server fail after three tokens, "fail first" before any. With tools, "call NAME" is
answered by a call of the function NAME (declared or not), and "malformed WORD" by a call block that does not parse
(call:F{WORD}): the model's text that the server's log must not quote (scripts/zdr_check.py)."""
import argparse
import queue
import threading
import time
from types import SimpleNamespace

import transformers
from transformers import AutoTokenizer

from sclmd import ckpt
from scripts import serve_admin as A
from scripts import serve_contract as C
from scripts.serve import MODEL, App, Handler


class Canned:
    def __init__(self, tok, stop):
        self.tok, self.stop, self.canon = tok, stop, None
        self.eng = SimpleNamespace(capacity=8192, rows=32)  # (--context)
        self.token_s = 0.01  # between the canned reply's tokens (--token-ms)
        self.first_s = 0.0  # before the first one (--first-ms)
        self.inbox = queue.Queue()
        self.opening, self.closing = tok.convert_tokens_to_ids("<|channel>"), tok.convert_tokens_to_ids("<channel|>")

    def submit(self, req) -> None:
        self.inbox.put(req)

    def about(self) -> dict:
        return {"requests": {"under_way": 0, "waiting": self.inbox.qsize()}, "conversations_kept": 0}

    def run(self) -> None:
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)  # noqa: E731
        while True:
            req = self.inbox.get()
            said = req.messages[-1]["content"]
            opened, tools = getattr(req, "opened", False), getattr(req, "tools", None)  # (a server before tool calls has neither)
            out = ([self.opening] + enc("thought\nThe user greets me. I greet back.") + [self.closing] if req.think and not opened else [])
            out += enc("I go on.") + [self.closing] if opened else []
            results = sum(m["role"] == "tool" for m in req.messages)
            text = said if isinstance(said, str) else ""
            if tools and text.startswith("call "):  # a call of the function named (scripts/zdr_check.py: an undeclared one)
                out += enc(f"<|tool_call>call:{text[5:].strip()}{{page:1}}<tool_call|>") + [self.stop[2]]
            elif tools and text.startswith("malformed "):  # a block that does not parse: "no ':' after the key WORD"
                out += enc(f"<|tool_call>call:{tools[0]['function']['name']}{{{text[10:].strip()}}}<tool_call|>") + [self.stop[1]]
            elif tools and results < 8:  # tools: a call of the first with page: the results so far + 1 (R2's tool_loop_bench)
                name = tools[0]["function"]["name"]
                out += enc(f'<|tool_call>call:{name}{{page:{results + 1},note:<|"|>say "hi", {{x}}\n서울<|"|>}}<tool_call|>') + [self.stop[2]]
            else:
                out += enc(f"Hello! You said: {said} 안녕하세요.")[: max(req.max_new - len(out) - 1, 1)] + [self.stop[1]]
            if said == "fail first":
                req.events.put(("error", 500, "the server failed while answering: RuntimeError"))
                continue
            failed = False
            if self.first_s:
                time.sleep(self.first_s)
            for k, token in enumerate(out):
                if req.cancelled:  # the real engine ends a cancelled job and says so ("done", ended "cancelled")
                    break
                if said == "fail" and k == 3:
                    req.events.put(("error", 500, "the server failed while answering: RuntimeError"))
                    failed = True
                    break
                req.events.put(("token", token))
                time.sleep(self.token_s)
            if not failed:
                usage = {"prompt_tokens": len(req.full), "completion_tokens": len(out), "total_tokens": len(req.full) + len(out),
                         "prompt_tokens_details": {"cached_tokens": 0}}
                timings = {"total_ms": 10 * len(out), "first_token_ms": 10, "tokens_per_second": 100.0, "prompt_read": len(req.full), "rows_per_pass": 1.0}
                req.events.put(("done", {"finish_reason": "stop", "usage": usage, "timings": timings, "ended": "stop"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("port", type=int, nargs="?")
    ap.add_argument("--port", type=int, dest="port_option", help="the port, as scripts/serve.py takes it (scripts/soak.py --serve-module)")
    ap.add_argument("--token-ms", type=float, default=10.0, help="milliseconds between the canned reply's tokens")
    ap.add_argument("--first-ms", type=float, default=0.0, help="milliseconds before the canned reply's first token (a long prompt read)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="the most tokens of a reply when the request does not say")
    ap.add_argument("--context", type=int, default=8192, help="tokens a conversation can hold")
    C.add_options(ap)  # the real server's: fields, body cap, idle timeout, connections
    A.add_options(ap)  # and the operator's (keys, --max-waiting, --queue-timeout, the request log, --debug; scripts/serve_net.py's)
    cli = ap.parse_args()
    cli.host, cli.port = "127.0.0.1", cli.port_option if cli.port_option is not None else cli.port
    if cli.port is None:
        ap.error("the port")
    keys = A.start(cli)
    transformers.logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
    server = Canned(tok, stop)
    server.token_s, server.first_s = cli.token_ms / 1000, cli.first_ms / 1000
    server.eng.capacity = cli.context
    app = App(server, tok, False, cli.max_tokens, {"bits_per_weight": 3.0})
    server.canon = None
    C.apply(cli, app, Handler)
    threading.Thread(target=server.run, daemon=True).start()
    httpd = C.BoundedHTTPServer(("127.0.0.1", cli.port), Handler, cli.max_connections)
    httpd.app = app
    A.install(app, server, cli, httpd, keys)
    print(f"[canned] {MODEL} at http://127.0.0.1:{cli.port}/v1", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
