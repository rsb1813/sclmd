"""The server's HTTP side (scripts/serve.py: Handler and App, with the real tokenizer and chat template) with a canned
reply in the engine's place: for trying a client against the server's protocol without the GPU or the 15 s of loading.

    python -m scripts.serve_canned 8017            (and the real server's --max-body-mb, --max-connections, ... fields options)
    uv run --no-project --with openai python scripts/openai_client_check.py 8017

A user message "fail" makes the server fail after three tokens, "fail first" before any."""
import argparse
import queue
import threading
import time
from types import SimpleNamespace

import transformers
from transformers import AutoTokenizer

from sclmd import ckpt
from scripts import serve_contract as C
from scripts.serve import MODEL, App, Handler


class Canned:
    def __init__(self, tok, stop):
        self.tok, self.stop, self.canon = tok, stop, None
        self.eng = SimpleNamespace(capacity=8192, rows=32)
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
            if tools and results < 8:  # tools: a call of the first with page: the results so far + 1 (R2's tool_loop_bench)
                name = tools[0]["function"]["name"]
                out += enc(f'<|tool_call>call:{name}{{page:{results + 1},note:<|"|>say "hi", {{x}}\n서울<|"|>}}<tool_call|>') + [self.stop[2]]
            else:
                out += enc(f"Hello! You said: {said} 안녕하세요.")[: max(req.max_new - len(out) - 1, 1)] + [self.stop[1]]
            if said == "fail first":
                req.events.put(("error", 500, "the server failed while answering: RuntimeError"))
                continue
            failed = False
            for k, token in enumerate(out):
                if req.cancelled:  # the real engine ends a cancelled job and says so ("done", ended "cancelled")
                    break
                if said == "fail" and k == 3:
                    req.events.put(("error", 500, "the server failed while answering: RuntimeError"))
                    failed = True
                    break
                req.events.put(("token", token))
                time.sleep(0.01)
            if not failed:
                usage = {"prompt_tokens": len(req.full), "completion_tokens": len(out), "total_tokens": len(req.full) + len(out),
                         "prompt_tokens_details": {"cached_tokens": 0}}
                timings = {"total_ms": 10 * len(out), "first_token_ms": 10, "tokens_per_second": 100.0, "prompt_read": len(req.full), "rows_per_pass": 1.0}
                req.events.put(("done", {"finish_reason": "stop", "usage": usage, "timings": timings, "ended": "stop"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("port", type=int)
    C.add_options(ap)  # the real server's: fields, body cap, idle timeout, connections
    cli = ap.parse_args()
    transformers.logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
    server = Canned(tok, stop)
    app = App(server, tok, False, 4096, {"bits_per_weight": 3.0})
    server.canon = None
    C.apply(cli, app, Handler)
    threading.Thread(target=server.run, daemon=True).start()
    httpd = C.BoundedHTTPServer(("127.0.0.1", cli.port), Handler, cli.max_connections)
    httpd.app = app
    print(f"[canned] {MODEL} at http://127.0.0.1:{cli.port}/v1", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
