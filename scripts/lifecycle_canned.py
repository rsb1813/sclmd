"""The server's placeholders over the canned server of scripts/serve_canned.py (no GPU): the port answers 503 for a
while, then the canned server answers on the same port, for trying a client that keeps its connection (R3-land).

    python -m scripts.lifecycle_canned 8019 --starting 3    serve_ops.Starting (as --crash-only binds the port before the
                                                            engine loads) for 3 s, then the server (serve_ops.serving)
    python -m scripts.lifecycle_canned 8020 --holder 3      the supervisor's Holder (503 "restarting") for 3 s, then it is
                                                            stopped and the server takes the port
    --old: the placeholder as R3 first wrote it (no Connection: close): what a kept connection got then
    uv run --offline --no-project --with openai python scripts/lifecycle_client_check.py 8019 8020 3"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import transformers
from transformers import AutoTokenizer

from sclmd import ckpt
from scripts import serve_contract as C
from scripts import serve_ops
from scripts.serve import MODEL, App, Handler
from scripts.serve_canned import Canned


class Old(BaseHTTPRequestHandler):
    """R3's first placeholder: 503 on a connection that stays open (the bug the check found)."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def answer(self) -> None:
        if self.command == "POST":
            self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        data = b'{"error": {"message": "the server is starting", "type": "server_error", "param": null, "code": null}}'
        self.send_response(503)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Retry-After", "15")
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = answer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("port", type=int)
    ap.add_argument("--starting", type=float, default=0.0)
    ap.add_argument("--holder", type=float, default=0.0)
    ap.add_argument("--old", action="store_true")
    C.add_options(ap)
    cli = ap.parse_args()
    transformers.logging.set_verbosity_error()
    if cli.starting:  # the port first, as --crash-only takes it
        httpd = C.BoundedHTTPServer(("127.0.0.1", cli.port), Old if cli.old else serve_ops.Starting, cli.max_connections)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        holder = None
    else:
        from scripts.supervise import Holder

        holder = Holder("127.0.0.1", cli.port, print)
        if cli.old:
            holder.httpd = ThreadingHTTPServer(("127.0.0.1", cli.port), Old)
            holder.httpd.daemon_threads = True
            threading.Thread(target=holder.httpd.serve_forever, daemon=True).start()
        else:
            holder.start(cli.holder)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
    server = Canned(tok, stop)
    app = App(server, tok, False, 4096, {"bits_per_weight": 3.0})
    server.canon = None
    C.apply(cli, app, Handler)
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(max(0.0, t0 + (cli.starting or cli.holder) - time.time()))
    if holder is None:
        serve_ops.serving(httpd, Handler, app)
    else:
        holder.stop()
        httpd = C.BoundedHTTPServer(("127.0.0.1", cli.port), Handler, cli.max_connections)
        httpd.app = app
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[canned] {MODEL} at http://127.0.0.1:{cli.port}/v1 after {time.time() - t0:.1f} s of 503", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
