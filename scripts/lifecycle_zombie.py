# R3's kill test (2), no GPU: the real Server and Handler (scripts/serve.py) over the stand-in engine of
# tests/test_lifecycle.py. One request is served; then every pass fails as after a sticky CUDA error. What do the
# following requests and /health get?   python -m scripts.lifecycle_zombie [--out FILE]
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

from test_lifecycle import StandIn, call, chat, serve_standin  # noqa: E402


def main() -> None:
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    s = StandIn()
    httpd = serve_standin(s)
    port = httpd.server_address[1]
    rows = {"before": {}, "after": []}
    status, body, _ = chat(port)
    rows["before"]["chat"] = [status, body["choices"][0]["message"]["content"] if status == 200 else body]
    rows["before"]["health"] = list(call(port, "GET", "/health")[:2])
    s.fault()  # from here every pass raises
    t0 = time.time()
    for k in range(5):
        status, body, _ = chat(port)
        hs, hb, _ = call(port, "GET", "/health")
        rows["after"].append({"t_s": round(time.time() - t0, 2), "chat": [status, body.get("error", {}).get("message") if isinstance(body, dict) else body],
                              "health": [hs, hb]})
    rows["process_alive"] = True
    rows["engine_thread_resets"] = "Server.run catches, tells, reset()s and turns on"
    print(json.dumps(rows, indent=1, ensure_ascii=False))
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1, ensure_ascii=False)
    httpd.shutdown()


if __name__ == "__main__":
    main()
