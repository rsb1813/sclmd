# One GPU job with a real server: starts scripts/serve.py (the options given, on --port), waits for "ok" on /health,
# runs a client command, then stops the server gracefully (Ctrl+Break; it needs --crash-only for that, else by its PID
# and its children) and checks that nothing it started is left.
#   python -m scripts.with_server --port 8041 [serve.py's options ...] -- <client command ...>
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    cut = sys.argv.index("--")
    serve_args, client = sys.argv[1:cut], sys.argv[cut + 1 :]
    port = serve_args[serve_args.index("--port") + 1]
    log = open(os.environ.get("SERVER_LOG", os.path.join(ROOT, "logs", f"with_server_{port}.log")), "w", encoding="utf-8")
    t0 = time.perf_counter()
    server = subprocess.Popen([sys.executable, "-u", "-m", "scripts.serve"] + serve_args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              env=dict(os.environ, PYTHONIOENCODING="utf-8"), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    code = 1
    try:
        while True:
            if server.poll() is not None:
                raise SystemExit(f"[with] the server ended ({server.returncode})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                pass
            if time.perf_counter() - t0 > 180:
                raise SystemExit("[with] no ok on /health after 180 s")
            time.sleep(0.2)
        print(f"[with] the server says ok {time.perf_counter() - t0:.1f} s after its start (pid {server.pid})", flush=True)
        code = subprocess.call(client, cwd=ROOT)
        print(f"[with] the client ended ({code})", flush=True)
    finally:
        if "--crash-only" in serve_args:
            os.kill(server.pid, signal.CTRL_BREAK_EVENT)
            try:
                print(f"[with] the server ended ({server.wait(60)}) after Ctrl+Break", flush=True)
            except subprocess.TimeoutExpired:
                pass
        if server.poll() is None:
            subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], capture_output=True)
            print(f"[with] the server (pid {server.pid}) and its children were stopped", flush=True)
        log.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
