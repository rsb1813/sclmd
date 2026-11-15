# The supervisor's manual sleep on the real model, in one GPU job: scripts/supervise.py starts the server (the options
# given); once it says ok, the sleep file is made and the card's used memory is read (nvidia-smi) while it sleeps; then
# the file is removed and the time until "ok" again is taken, and a request is answered. The supervisor is stopped with
# Ctrl+Break, and nothing it started may be left.
#   python -m scripts.lifecycle_sleep --port 8041 --out FILE [serve.py's options ...]
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

from scripts.lifecycle_real import ROOT, alive, children_of, get


def used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.split()[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8041)
    ap.add_argument("--out", required=True)
    args, serve_args = ap.parse_known_args()
    base = f"http://127.0.0.1:{args.port}"
    sleep = os.path.join(tempfile.gettempdir(), f"sclmd_sleep_{os.getpid()}")
    log = args.out + ".supervise.log"
    cmd = [sys.executable, "-u", "-m", "scripts.supervise", "--log", log, "--sleep-file", sleep, "--port", str(args.port)] + serve_args
    rows = {"cmd": cmd, "used_mib_before_start": used_mib()}
    out_file = open(args.out + ".server.log", "w", encoding="utf-8")
    sup = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"), stdout=out_file, stderr=subprocess.STDOUT,
                           creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    ask = {"messages": [{"role": "user", "content": "Translate into English: 오늘은 날씨가 좋아서 산책을 하고 싶어요."}], "max_tokens": 30, "temperature": 0}

    def wait(status: str, most: float = 120) -> float:
        t = time.perf_counter()
        while time.perf_counter() - t < most:
            s, b, _ = get(base + "/health", timeout=5)
            if (b or {}).get("status") == status:
                return time.perf_counter() - t
            time.sleep(0.1)
        raise SystemExit(f"no {status} after {most} s")

    try:
        rows["first_ok_s"] = round(wait("ok"), 2)
        s, b, _ = get(base + "/v1/chat/completions", ask)
        rows["reply_before"] = b["choices"][0]["message"]["content"] if s == 200 else [s, b]
        rows["used_mib_serving"] = used_mib()
        open(sleep, "w").close()
        t = time.perf_counter()
        rows["asleep_after_s"] = round(wait("asleep"), 2)
        readings = []
        for _ in range(10):  # the process has ended: its memory is given back
            readings.append(used_mib())
            time.sleep(0.3)
        rows["used_mib_asleep"] = readings
        rows["freed_gib"] = round((rows["used_mib_serving"] - min(readings)) / 1024, 2)
        s, b, h = get(base + "/v1/chat/completions", ask, timeout=10)
        rows["request_asleep"] = [s, h.get("Retry-After"), (b or {}).get("error", {}).get("message")]
        os.remove(sleep)
        t = time.perf_counter()
        rows["ok_after_wake_s"] = round(wait("ok"), 2)
        s, b, _ = get(base + "/v1/chat/completions", ask)
        rows["reply_after"] = b["choices"][0]["message"]["content"] if s == 200 else [s, b]
        rows["same_reply"] = rows["reply_after"] == rows["reply_before"]
    finally:
        tree = children_of(sup.pid)
        os.kill(sup.pid, signal.CTRL_BREAK_EVENT)
        try:
            rows["supervisor_exit"] = sup.wait(60)
        except subprocess.TimeoutExpired:
            rows["supervisor_exit"] = "did not end: stopped by its PID"
            subprocess.run(["taskkill", "/PID", str(sup.pid), "/T", "/F"], capture_output=True)
        time.sleep(1.0)
        rows["left_running"] = [p for p in tree + [sup.pid] if alive(p)]
        for p in rows["left_running"]:
            subprocess.run(["taskkill", "/PID", str(p), "/T", "/F"], capture_output=True)
        if os.path.exists(sleep):
            os.remove(sleep)
        with open(log, encoding="utf-8") as f:
            rows["supervisor_log"] = f.read().splitlines()
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1, ensure_ascii=False)
        print(json.dumps({k: v for k, v in rows.items() if k != "cmd"}, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
