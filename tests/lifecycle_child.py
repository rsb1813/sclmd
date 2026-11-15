# A stand-in for scripts/serve.py under scripts/supervise.py (tests/test_lifecycle.py), without torch: it takes the
# port, says 503 "starting" for a moment, then "ok"; each start is a line in R3_CHILD_LOG (its own pid). The first
# R3_CHILD_FAILS starts end with exit 75 after R3_CHILD_AFTER seconds of "ok"; the later ones serve until Ctrl+Break,
# and then end with 0 (R3_CHILD_STUCK=1: they stop answering /health instead).
#   python -m tests.lifecycle_child --port P [--crash-only ...]
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import serve_ops  # noqa: E402


def main() -> None:
    port = int(sys.argv[sys.argv.index("--port") + 1])
    log = os.environ["R3_CHILD_LOG"]
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{os.getpid()} {time.time()} {' '.join(sys.argv[1:])}\n")
    with open(log, encoding="utf-8") as f:
        run = len(f.readlines())
    state = {"ok": False, "stuck": False}

    class Health(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            if state["stuck"]:
                time.sleep(60)
            data = json.dumps({"status": "ok" if state["ok"] else "starting"}).encode()
            self.send_response(200 if state["ok"] else 503)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Health)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    serve_ops.ctrl_break_stops()
    try:
        time.sleep(0.5)  # "loading"
        state["ok"] = True
        if run <= int(os.environ.get("R3_CHILD_FAILS", "0")):
            time.sleep(float(os.environ.get("R3_CHILD_AFTER", "0.5")))
            print(f"[child] run {run}: exit {serve_ops.EXIT_FATAL}", flush=True)
            os._exit(serve_ops.EXIT_FATAL)
        if os.environ.get("R3_CHILD_STUCK"):
            time.sleep(3.0)  # "ok" long enough for the supervisor's poll (every 2 s) to see it
            state["stuck"] = True
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print(f"[child] run {run}: stopping gracefully", flush=True)
        time.sleep(0.3)
        httpd.server_close()
        sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
