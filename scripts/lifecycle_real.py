# R3's phase B on the real model, in one GPU job: scripts/supervise.py starts the server (--crash-only, the options
# given), a fault is set in its first run (SCLMD_FAULT: "tile:N" a sticky CUDA error, "hang:S:N" a hung engine thread),
# and a client watches what happens: how soon the process ends after the failure, what /health and requests get while
# it is down, how soon it says "ok" again, and that a request is then answered. Then the supervisor is stopped with
# Ctrl+Break (gracefully), and nothing it started may be left.
#   python -m scripts.lifecycle_real --fault tile:30 --port 8041 --out FILE [serve.py's options ...]
import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get(url: str, body: dict | None = None, timeout: float = 120) -> tuple:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read())
        except ValueError:
            data = None
        return e.code, data, dict(e.headers)
    except OSError as e:
        return type(e).__name__, None, {}


def alive(pid: int) -> bool:
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value == 259


def children_of(pid: int) -> list:
    """The PIDs under a process (its tree), from the processes' parent links."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process | ForEach-Object { \"$($_.ProcessId) $($_.ParentProcessId)\" }"],
                         capture_output=True, text=True).stdout.split()
    pairs = [(int(out[k]), int(out[k + 1])) for k in range(0, len(out) - 1, 2)]
    tree, grew = {pid}, True
    while grew:
        grew = False
        for p, parent in pairs:
            if parent in tree and p not in tree:
                tree.add(p)
                grew = True
    return sorted(tree - {pid})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fault", required=True)
    ap.add_argument("--port", type=int, default=8041)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=os.path.join(ROOT, "logs", "lifecycle_real_supervise.log"))
    args, serve_args = ap.parse_known_args()
    base = f"http://127.0.0.1:{args.port}"
    if os.path.exists(args.log):
        os.remove(args.log)
    marker = os.path.join(tempfile.gettempdir(), f"sclmd_fault_{os.getpid()}")
    open(marker, "w").close()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", SCLMD_FAULT=args.fault, SCLMD_FAULT_FILE=marker)
    cmd = [sys.executable, "-u", "-m", "scripts.supervise", "--log", args.log, "--port", str(args.port)] + serve_args
    out_file = open(args.out + ".server.log", "w", encoding="utf-8")
    t0 = time.perf_counter()
    sup = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out_file, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    rows = {"cmd": cmd, "fault": args.fault, "health": []}
    ask = {"messages": [{"role": "user", "content": "Translate into Korean: The weather is nice today, so let's take a walk."}],
           "max_tokens": 40, "temperature": 0}

    def health() -> tuple:
        s, b, h = get(base + "/health", timeout=10)
        return s, (b or {}).get("status"), h.get("Retry-After")

    try:
        while True:  # the first start
            s, st, _ = health()
            rows["health"].append([round(time.perf_counter() - t0, 2), s, st])
            if s == 200:
                break
            if time.perf_counter() - t0 > 240:
                raise SystemExit("no ok after 240 s")
            time.sleep(0.2)
        rows["first_ok_s"] = round(time.perf_counter() - t0, 2)
        replies = []
        while True:  # requests until the fault
            t = time.perf_counter()
            s, b, h = get(base + "/v1/chat/completions", ask)
            replies.append([round(t - t0, 2), s, (b or {}).get("choices", [{}])[0].get("message", {}).get("content") if s == 200 else (b or {}).get("error", {}).get("message")])
            if s != 200:
                failed_at = time.perf_counter()
                break
            if len(replies) > 40:
                raise SystemExit("the fault did not come")
        rows["replies_before"] = replies
        rows["failed_request"] = replies[-1]
        # how soon the process ends: the supervisor's log says so
        ended = None
        while ended is None and time.perf_counter() - failed_at < 60:
            with open(args.log, encoding="utf-8") as f:
                text = f.read()
            if text.count("the server ended (") >= 1:
                ended = time.perf_counter()
            s, st, retry = health()
            rows["health"].append([round(time.perf_counter() - t0, 2), s, st, retry])
            time.sleep(0.1)
        rows["exit_after_failure_s"] = None if ended is None else round(ended - failed_at, 2)
        down = []
        while True:  # while it is down: /health and a request
            s, st, retry = health()
            rows["health"].append([round(time.perf_counter() - t0, 2), s, st, retry])
            if s == 200:
                break
            if not down or time.perf_counter() - down[-1][0] > 2:
                rs, rb, rh = get(base + "/v1/chat/completions", ask, timeout=10)
                down.append([time.perf_counter(), rs, rh.get("Retry-After"), (rb or {}).get("error", {}).get("message")])
            if time.perf_counter() - failed_at > 240:
                raise SystemExit("no ok again after 240 s")
            time.sleep(0.2)
        rows["ok_again_after_failure_s"] = round(time.perf_counter() - failed_at, 2)
        rows["requests_while_down"] = [[round(d[0] - t0, 2)] + d[1:] for d in down]
        s, b, _ = get(base + "/v1/chat/completions", ask)
        rows["after"] = [s, (b or {}).get("choices", [{}])[0].get("message", {}).get("content") if s == 200 else b]
        rows["same_reply_as_before"] = s == 200 and rows["after"][1] == replies[0][2]
    finally:
        tree = children_of(sup.pid)
        rows["tree_before_stop"] = tree
        os.kill(sup.pid, signal.CTRL_BREAK_EVENT)
        try:
            rows["supervisor_exit"] = sup.wait(60)
        except subprocess.TimeoutExpired:
            rows["supervisor_exit"] = "did not end: stopped by its PID"
            subprocess.run(["taskkill", "/PID", str(sup.pid), "/T", "/F"], capture_output=True)
        time.sleep(1.0)
        rows["left_running"] = [p for p in tree + [sup.pid] if alive(p)]
        for p in rows["left_running"]:  # only what this script started
            subprocess.run(["taskkill", "/PID", str(p), "/T", "/F"], capture_output=True)
        try:
            os.remove(marker)
        except OSError:
            pass
        with open(args.log, encoding="utf-8") as f:
            rows["supervisor_log"] = f.read().splitlines()[-12:]
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1, ensure_ascii=False)
        print(json.dumps({k: v for k, v in rows.items() if k not in ("health", "cmd")}, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
