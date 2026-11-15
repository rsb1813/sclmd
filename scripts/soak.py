# A soak of the real server (R4): hours of mixed traffic with faults on a timer, and the books checked. It builds on
# B8's load bench (scripts/task_load_bench.py: its task items and FLORES documents) and serve_bench's conversations.
#
#   python -m scripts.soak --state DIR --serve --seconds 300 --deadline 460      one piece: starts the server, runs, stops it
#   python -m scripts.soak --state DIR --url http://127.0.0.1:8001/v1 ...        against a server started elsewhere (--debug on)
#   python -m scripts.soak --report DIR                                         the invariants and trends of every piece
#
# Resumable: DIR/state.json counts the traffic seconds done against --target; calling again runs the next piece (each
# piece is one server lifetime, so trends are fitted within a piece). logs/pause (the main copy's, read only) is
# honoured: when it appears (or is touched again) the piece stops, the server is stopped and the VRAM freed, and the
# exit code is 3. A pause file that is already there at the start (the benchmark queue's) does not stop it, unless
# --strict-pause.
#
# Traffic (--profile mixed): short task items and documents of 500-6,000 tokens (greedy, 40 tokens), conversations of
# two turns (seeded, 200-600 tokens), long replies (1,000-1,500), thinking, stop strings, streamed or not, 10% left in
# the middle of a stream, some malformed bodies. Closed loop: --clients threads with a think time that keeps the
# engine busy 70-90% of the time. --profile gate: serve_bench's conversations at 8 clients (the brief's 1-hour gate).
# Chaos (mixed and gate): 200 idle keep-alive connections for 60 s, an injected handler fault, a VRAM ballast for 60 s
# (scripts/ballast.py), an injected engine fault, 50 clients that leave in the middle of a stream, one request with
# stream_options: true; again every --chaos-every seconds. Each event runs at its own time in a thread of its own (none
# waits behind another), none starts once the traffic has stopped, and when it stops the idle connections and the
# ballast end and every fault still armed is cleared (so that none can hit the replay).
# Invariants: replies with a fixed seed (a --replay share of them, asked alone at the end) are the same text; no 5xx
# but the injected ones (and those inside a ballast's window, counted apart); every stream ends with [DONE]; at idle
# GET /debug/audit holds; trends (RSS, private bytes, pinned RAM, VRAM reserved, threads, the queue) per hour; p99 of
# the time to the first token by hour.
import argparse
import http.client
import json
import os
import random
import shlex
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

from scripts import serve_bench as SB
from scripts import task_load_bench as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAUSE = "H:/projects/sclm/sclmd/logs/pause"  # the owner's (and the benchmark queue's) stop sign: read, never written
LEAST_TRAFFIC = 30  # seconds of traffic a piece has at least: --deadline must leave them besides --tail
KEY = {}  # the API key the soak's requests give (the first of SCLMD_API_KEY, when it is set: the server then asks for it)
SERIES = {  # what the sampler keeps from /metrics: (metric, labels)
    "resident_bytes": ("sclmd_process_resident_bytes", {}), "private_bytes": ("sclmd_process_private_bytes", {}),
    "python_threads": ("sclmd_process_threads", {"kind": "python"}), "os_threads": ("sclmd_process_threads", {"kind": "os"}),
    "cuda_reserved_bytes": ("sclmd_cuda_reserved_bytes", {}), "cuda_allocated_bytes": ("sclmd_cuda_allocated_bytes", {}),
    "host_pinned_bytes": ("sclmd_host_pinned_bytes", {"state": "held"}), "snapshot_bytes": ("sclmd_pinned_bytes", {"kind": "snapshots"}),
    "parked_bytes": ("sclmd_parked_bytes", {}), "waiting": ("sclmd_requests_waiting", {}), "under_way": ("sclmd_requests_under_way", {}),
    "kept": ("sclmd_conversations_kept", {}), "in_ram": ("sclmd_conversations_in_ram", {}), "kv_free_bytes": ("sclmd_kv_free_bytes", {}),
    "connections": ("sclmd_connections", {}), "busy_s": ("sclmd_busy_seconds_total", {}), "completion_tokens": ("sclmd_completion_tokens_total", {}),
    "requests": ("sclmd_requests_finished_total", {}),
}
TREND_S = 2700  # traffic seconds of one server lifetime from which the trend bar (under 1%/h) and the p99 bound are judged
TRENDS = ("resident_bytes", "private_bytes", "host_pinned_bytes", "snapshot_bytes", "cuda_reserved_bytes", "python_threads", "os_threads", "waiting")
LONG = ["Write a long story, in many paragraphs, about a lighthouse keeper who finds a message in a bottle and follows it across the sea.",
        "조선 시대의 과거 제도의 역사와 운영, 문과·무과·잡과의 차이, 그리고 그 사회적 영향을 아주 자세히, 여러 문단으로 설명해 줘.",
        "Explain, in great detail and with code, how a relational database executes a query: parsing, planning, indexes, joins and caching."]
THINK = ["A shop sells pens at 3 for 2 dollars. How much do 27 pens cost? Explain briefly.", "두 수의 합이 20이고 차가 6이면 두 수는 무엇인가?",
         "Is 221 a prime number?", "If it is 10:40 now, what time will it be after 150 minutes?"]
MALFORMED = [b"{not json", b'{"messages": "hi"}', b'{"messages": [{"role": "user", "content": "hi"}], "n": 2}',
             b'{"messages": [{"role": "user", "content": "hi"}], "logprobs": true}', b'{"model": "x"}']  # each 400 (a stop of 5 is served since R1)


def now() -> float:
    return time.time()


# the HTTP client: one request, streamed or not, maybe left in the middle

def request(url: str, body: dict | None, stream: bool, raw: bytes | None = None, leave_after: int | None = None, timeout: float = 600) -> dict:
    u = urlparse(url)
    out = {"status": None, "content": "", "reasoning": "", "finish": None, "usage": None, "first": None, "done": False, "error": None}
    t0 = time.perf_counter()
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        data = raw if raw is not None else json.dumps(body).encode()
        conn.request("POST", u.path.rstrip("/") + "/chat/completions", data, {"Content-Type": "application/json", **KEY})
        resp = conn.getresponse()
        out["status"] = resp.status
        if resp.status != 200 or not stream:
            text = resp.read().decode("utf-8", "replace")
            try:
                got = json.loads(text)
            except ValueError:
                got = {"error": {"message": text[:200]}}
            if resp.status == 200:
                msg = got["choices"][0]["message"]
                out.update(content=msg.get("content") or "", reasoning=msg.get("reasoning_content") or "", finish=got["choices"][0]["finish_reason"],
                           usage=got.get("usage"))
            else:
                out["error"] = (got.get("error") or {}).get("message", text[:200])
            out["seconds"] = time.perf_counter() - t0
            return out
        pieces = 0
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                out["done"] = True
                continue
            data = json.loads(line[6:])
            if "error" in data:
                out["error"] = data["error"].get("message")
                continue
            if data.get("usage"):
                out["usage"] = data["usage"]
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    pieces += 1
                    if out["first"] is None:
                        out["first"] = time.perf_counter() - t0
                out["content"] += delta.get("content") or ""
                out["reasoning"] += delta.get("reasoning_content") or ""
                if choice.get("finish_reason"):
                    out["finish"] = choice["finish_reason"]
            if leave_after is not None and pieces >= leave_after:
                out["left"] = True
                break
    except (OSError, http.client.HTTPException, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        conn.close()
    out["seconds"] = time.perf_counter() - t0
    return out


def get(url: str, path: str, timeout: float = 10) -> tuple:
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        conn.request("GET", path, headers=KEY)
        r = conn.getresponse()
        return r.status, r.read()
    finally:
        conn.close()


def post(url: str, path: str, body: dict, timeout: float = 10) -> tuple:
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json", **KEY})
        r = conn.getresponse()
        return r.status, r.read()
    finally:
        conn.close()


def metrics_values(text: str) -> dict:
    """Prometheus text -> {name: [(labels, value)]}."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        name, _, rest = head.partition("{")
        labels = dict(p.split("=", 1) for p in rest.rstrip("}").split(",") if "=" in p) if rest else {}
        labels = {k: v.strip('"') for k, v in labels.items()}
        out.setdefault(name, []).append((labels, float(value.replace("Inf", "inf"))))
    return out


def pick(values: dict, name: str, labels: dict):
    for lab, v in values.get(name, ()):
        if all(lab.get(k) == x for k, x in labels.items()):
            return v
    return None


# the soak

class Soak:
    def __init__(self, args, piece: int, url: str):
        self.args, self.piece, self.url = args, piece, url
        self.dir = args.state
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.t0 = now()
        self.think = 0.5  # seconds a client waits between requests: set by the sampler for 70-90% busy
        self.windows = []  # (start, end, kind) of chaos that may cause errors that are not faults (the ballast)
        self.replay = []  # requests with a fixed seed that ended well: candidates for the replay alone
        self.counts = {}
        with open(os.path.join(ROOT, "data", "task_eval", "items.jsonl"), encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f]
        self.sents = L.sentences(self.items)
        self.rng = random.Random(1000 + piece)
        self.serial = 0

    def write(self, name: str, record: dict) -> None:
        with self.lock:
            with open(os.path.join(self.dir, name), "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"piece": self.piece, "t": round(now() - self.t0, 2), **record}, ensure_ascii=False) + "\n")

    def classify(self, scenario: str, out: dict, stream: bool, t_start: float) -> str:
        status, err = out["status"], out.get("error") or ""
        if scenario == "malformed":
            return "expected_400" if status == 400 else "unexpected"
        if out.get("left"):
            return "left"
        if "InjectedFault" in err:
            return "injected"
        in_window = any(a - 2 <= t_start <= b + 10 or a - 2 <= now() <= b + 10 for a, b, _ in self.windows)
        if status in (429, 503) and ("--max-waiting" in err or "--queue-timeout" in err):
            return "refused"
        if status == 200 and not err and (not stream or out["done"]) and out["finish"] is not None:
            return "ok"
        return "chaos_window" if in_window else "unexpected"

    def one(self, client: int, scenario: str, body: dict | None, stream: bool, raw: bytes | None = None, leave_after: int | None = None,
            seeded: bool = False) -> dict:
        t_start = now()
        out = request(self.url, body, stream, raw, leave_after)
        kind = self.classify(scenario, out, stream, t_start)
        usage = out.get("usage") or {}
        rec = {"client": client, "scenario": scenario, "stream": stream, "status": out["status"], "class": kind,
               "first_s": round(out["first"], 3) if out["first"] is not None else None, "seconds": round(out.get("seconds", 0), 3),
               "prompt": usage.get("prompt_tokens"), "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "reply": usage.get("completion_tokens"), "finish": out["finish"], "done": out["done"]}
        if kind not in ("ok", "left", "expected_400"):
            rec["error"] = (out.get("error") or "")[:300]
        self.write("requests.jsonl", rec)
        with self.lock:
            self.counts[kind] = self.counts.get(kind, 0) + 1
            if seeded and kind == "ok":
                self.replay.append({"scenario": scenario, "body": body, "stream": stream, "content": out["content"], "reasoning": out["reasoning"]})
        return out

    # traffic

    def scenario(self) -> str:
        if self.args.profile == "gate":
            return "chat"
        weights = {"task": 0.27, "doc": 0.12, "chat": 0.25, "long": 0.04, "think": 0.06, "stop": 0.06, "cancel": 0.10, "malformed": 0.02}
        r, acc = self.rng.random() * sum(weights.values()), 0.0
        for k, w in weights.items():
            acc += w
            if r <= acc:
                return k
        return "task"

    def client(self, i: int) -> None:
        rng = random.Random(self.piece * 1000 + i)
        while not self.stop.is_set():
            with self.lock:
                kind, self.serial = self.scenario(), self.serial + 1
                n = self.serial
            stream = rng.random() < 0.7
            seed = (self.piece * 100003 + n) & 0x3FFFFFFF
            greedy = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 0, "thinking": False}
            if kind == "task":
                _, _, text = L.requests_for(50, 1, self.items, self.sents, rng, n)[0]
                self.one(i, kind, {"messages": [{"role": "user", "content": text}], "max_tokens": 40, "stream": stream, **greedy}, stream, seeded=True)
            elif kind == "doc":
                size = rng.choice((500, 1000, 2000, 4000, 6000))
                _, _, text = L.requests_for(size, 1, self.items, self.sents, rng, n)[0]
                self.one(i, kind, {"messages": [{"role": "user", "content": text}], "max_tokens": 40, "stream": stream, **greedy}, stream, seeded=True)
            elif kind in ("chat", "cancel", "stop"):
                k = rng.randrange(len(SB.PROMPTS))
                first, follow = SB.PROMPTS[k]
                gate = self.args.profile == "gate"
                most = 200 if gate else rng.choice((200, 400, 600))
                body = {"messages": [{"role": "user", "content": first}], "max_tokens": most, "stream": True if gate else stream, "seed": seed,
                        "thinking": False, "stream_options": {"include_usage": True}}
                if kind == "stop":
                    body["stop"] = ["\n\n"]
                if kind == "cancel":
                    self.one(i, kind, dict(body, stream=True), True, leave_after=rng.randrange(2, 12))
                    continue
                out = self.one(i, kind, body, body["stream"], seeded=True)
                if out["status"] == 200 and not self.stop.is_set():
                    time.sleep(rng.uniform(0.5, 3.0))  # a user reads and types
                    body2 = dict(body, messages=body["messages"] + [{"role": "assistant", "content": out["content"]}, {"role": "user", "content": follow}],
                                 seed=seed + 1)
                    self.one(i, kind + "2", body2, body2["stream"], seeded=True)
            elif kind == "long":
                body = {"messages": [{"role": "user", "content": rng.choice(LONG)}], "max_tokens": rng.choice((1000, 1500)), "stream": stream,
                        "seed": seed, "thinking": False}
                self.one(i, kind, body, stream, seeded=True)
            elif kind == "think":
                body = {"messages": [{"role": "user", "content": rng.choice(THINK)}], "max_tokens": 600, "stream": stream, "seed": seed, "thinking": True}
                self.one(i, kind, body, stream, seeded=True)
            elif kind == "malformed":
                self.one(i, kind, None, False, raw=rng.choice(MALFORMED))
            pause = rng.uniform(0.5, 1.5) * self.think
            self.stop.wait(pause)

    # chaos

    def chaos_plan(self, seconds: float) -> list:
        base = [(30, "idle"), (60, "handler"), (100, "ballast"), (190, "engine"), (240, "leave50"), (280, "stream_options")]
        every = self.args.chaos_every or seconds
        plan, p = [], 0.0
        while p < seconds - 20:
            span = min(every, seconds - p)
            plan += [(p + off * span / 330, kind) for off, kind in base if (p + off * span / 330) < seconds - 15 and kind in self.args.chaos]
            p += every
        return sorted(plan)

    def chaos(self, plan: list) -> list:
        """Each event of the plan in a thread of its own, started at its time unless the traffic has stopped by then
        (a long event, the 60 s of idle connections, delays no other) -> the threads."""
        threads = []
        for at, kind in plan:
            th = threading.Thread(target=self._event, args=(at, kind), daemon=True, name=f"chaos-{kind}")
            th.start()
            threads.append(th)
        return threads

    def _event(self, at: float, kind: str) -> None:
        if self.stop.wait(max(0.0, self.t0 + at - now())):  # the traffic stopped first: not run
            self.write("events.jsonl", {"event": kind, "skipped": "the traffic had stopped", "at_s": round(at, 1)})
            return
        t = now()
        try:
            detail = getattr(self, "chaos_" + kind)()
        except Exception as e:  # noqa: BLE001
            detail = f"failed: {type(e).__name__}: {e}"
        self.write("events.jsonl", {"event": kind, "at_s": round(at, 1), "took_s": round(now() - t, 2), "detail": detail})

    def clear_faults(self) -> str:
        """Every fault still armed (POST /debug/fault {"kind": "clear"}): none may hit the replay."""
        status, body = post(self.url.rsplit("/v1", 1)[0], "/debug/fault", {"kind": "clear"})
        return f"{status} {body.decode()[:120]}"

    def chaos_idle(self) -> str:
        """200 keep-alive connections that ask GET /health once and then sit idle for 60 s (or until the traffic stops)."""
        u = urlparse(self.url)
        socks, ok = [], 0
        for _ in range(200):
            try:
                s = socket.create_connection((u.hostname, u.port), timeout=5)
                s.sendall(b"GET /health HTTP/1.1\r\nHost: soak\r\n\r\n")
                ok += b"200" in s.recv(4096)
                socks.append(s)
            except OSError:
                pass
        held = len(socks)
        t = now()
        self.stop.wait(60)
        held_s = now() - t
        closed_by_server = 0
        for s in socks:
            try:
                s.settimeout(0.05)
                while True:  # what is left of the answer first (its body may have come after the first read), then the end
                    if s.recv(65536) == b"":
                        closed_by_server += 1
                        break
            except (ConnectionResetError, ConnectionAbortedError):  # Windows tells a closed peer so too
                closed_by_server += 1
            except OSError:  # a timeout: still open
                pass
            s.close()
        return f"{held} connections held {held_s:.0f} s ({ok} answered); {closed_by_server} already closed by the server"

    def chaos_handler(self) -> str:
        status, body = post(self.url.rsplit("/v1", 1)[0], "/debug/fault", {"kind": "handler", "count": 1})
        return f"{status} {body.decode()[:120]}"

    def chaos_engine(self) -> str:
        status, body = post(self.url.rsplit("/v1", 1)[0], "/debug/fault", {"kind": "engine", "count": 1})
        return f"{status} {body.decode()[:120]}"

    def chaos_ballast(self) -> str:
        t = now()
        self.windows.append((t, t + 75, "ballast"))
        p = subprocess.Popen([sys.executable, "-m", "scripts.ballast", "--gib", str(self.args.ballast_gib), "--seconds", "60"], cwd=ROOT,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        CHILDREN.append(p)  # stopped with the server, whatever ends the piece (its VRAM never outlives it)
        why = ""
        while p.poll() is None:  # it ends by itself after its 60 s; the traffic stopping, or 120 s, end it sooner
            if self.stop.wait(1.0) or now() - t > 120:
                why = " (stopped: the traffic had stopped)" if self.stop.is_set() else " (did not end in 120 s: stopped)"
                stop_tree(p)
                break
        out = p.stdout.read() if p.stdout is not None else ""
        self.windows[-1] = (t, now(), "ballast")
        return ((out or "").strip() + why)[-400:]

    def chaos_leave50(self) -> str:
        outs = [None] * 50

        def go(k):
            body = {"messages": [{"role": "user", "content": SB.PROMPTS[k % len(SB.PROMPTS)][0]}], "max_tokens": 200, "stream": True, "seed": k}
            outs[k] = self.one(100 + k, "leave50", body, True, leave_after=1 + k % 4)

        ths = [threading.Thread(target=go, args=(k,)) for k in range(50)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(300)
        return f"{sum(1 for o in outs if o and o.get('left'))} of 50 left in the middle of their stream"

    def chaos_stream_options(self) -> str:
        body = {"messages": [{"role": "user", "content": "Say hello in Korean."}], "max_tokens": 20, "stream": True, "stream_options": True, "seed": 3}
        out = self.one(200, "stream_options", body, True, seeded=True)
        return f"status {out['status']}, [DONE] {out['done']}, finish {out['finish']}"

    # the sampler

    def sampler(self) -> None:
        last = None
        while not self.stop.is_set():
            try:
                status, body = get(self.url.rsplit("/v1", 1)[0], "/metrics")
                values = metrics_values(body.decode())
                s = {k: pick(values, *v) for k, v in SERIES.items()}
                s["errors_5xx"] = sum(v for lab, v in values.get("sclmd_http_responses_total", ()) if lab.get("code", "").startswith("5"))
                if last is not None and s["busy_s"] is not None and last["busy_s"] is not None:
                    busy = (s["busy_s"] - last["busy_s"]) / max(now() - last["_t"], 1e-6)
                    s["busy_share"] = round(busy, 3)
                    if self.args.profile == "gate":  # serve_bench's clients do not wait between conversations
                        self.think = 0.0
                    elif busy > 0.9:
                        self.think = min(self.think * 1.25 + 0.05, 20.0)
                    elif busy < 0.7:
                        self.think = max(self.think / 1.25 - 0.02, 0.0)
                s["think_s"] = round(self.think, 2)
                s["_t"] = now()
                self.write("samples.jsonl", {k: v for k, v in s.items() if k != "_t"})
                last = s
            except Exception as e:  # noqa: BLE001
                self.write("samples.jsonl", {"error": f"{type(e).__name__}: {e}"})
            self.stop.wait(self.args.sample_every)


class Pause:
    """logs/pause, read only. It also holds the benchmark queue for days, so a file that is there when the piece
    starts does not stop it: what stops it is the file appearing, or being touched again, after that (--strict-pause:
    its being there at all)."""

    def __init__(self, path: str = PAUSE, strict: bool = False):
        self.path, self.strict = path, strict
        self.at_start = self._mtime()

    def _mtime(self):
        try:
            return os.stat(self.path).st_mtime
        except OSError:
            return None

    def __call__(self) -> bool:
        m = self._mtime()
        if m is None:
            return False
        return self.strict or self.at_start is None or m > self.at_start


paused = Pause()


def wait_idle_audit(base: str, most_s: float) -> dict:
    t, got = now(), {}
    while now() - t < most_s:
        try:
            status, body = get(base, "/debug/audit")
            got = json.loads(body)
            if status == 200:
                return got
            if status == 404:
                stand_in = (got.get("error") or {}).get("code") == "no_engine"
                return {"idle": None, "error": "the server has no engine to audit (a stand-in)" if stand_in else
                        "the server has no /debug/audit (start it with --debug)"}
        except (OSError, ValueError) as e:
            got = {"error": str(e)}
        time.sleep(1.0)
    return {"idle": False, "waited_s": most_s, **(got if isinstance(got, dict) else {})}


def start_server(args, piece: int) -> tuple:
    log = open(os.path.join(args.state, f"serve_{piece}.log"), "w", encoding="utf-8")
    cmd = [sys.executable, "-u", "-m", args.serve_module, "--port", str(args.port), "--debug", "--config", "",
           "--request-log", os.path.join(args.state, f"server_requests_{piece}.jsonl"), "--status-file", os.path.join(args.state, "status.json")]
    cmd += shlex.split(args.serve_args)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    SERVERS.append(proc)  # held before it is polled: the watchdog stops it even while it starts
    url = f"http://127.0.0.1:{args.port}/v1"
    t = now()
    while now() - t < 150:
        if proc.poll() is not None:
            return proc, None, f"the server ended with {proc.returncode} while starting"
        try:
            if get(url.rsplit("/v1", 1)[0], "/health", timeout=2)[0] == 200:
                return proc, url, f"up after {now() - t:.0f} s"
        except OSError:
            pass
        if paused():
            return proc, None, "logs/pause appeared while the server started"
        time.sleep(2)
    return proc, None, "the server did not answer /health within 150 s"


SERVERS = []  # the servers the piece started, from the moment each was started
CHILDREN = []  # processes of the piece on the GPU besides the server (the ballast)


def stop_tree(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    if sys.platform == "win32":  # the tree: the venv's launcher and the interpreter it starts
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(20)
    except subprocess.TimeoutExpired:
        proc.kill()


def stop_server(proc=None) -> None:
    """Every process of the piece: the ballast, and every server it started (proc among them, if given)."""
    for p in list(CHILDREN):
        stop_tree(p)
    for p in list(SERVERS) + ([proc] if proc is not None and proc not in SERVERS else []):
        stop_tree(p)


def load_state(path: str, target: float) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"target_s": target, "done_s": 0.0, "pieces": []}


def save_state(path: str, state: dict) -> None:
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(path + ".tmp", path)


def run_piece(args) -> int:
    os.makedirs(args.state, exist_ok=True)
    path = os.path.join(args.state, "state.json")
    state = load_state(path, args.target)
    if state["target_s"] - state["done_s"] < 10:  # what is left is shorter than a piece can be: done
        print(f"[soak] {state['done_s']:.0f} of {state['target_s']:.0f} s done: nothing left (python -m scripts.soak --report {args.state})")
        return 0
    if paused():
        print("[soak] logs/pause is there: not started")
        return 3
    t_begin, deadline = now(), now() + args.deadline
    piece = len(state["pieces"]) + 1
    proc, url = None, args.url

    def watchdog():  # the last word: the server never outlives the deadline (gpu.sh's limit comes after it)
        while now() < deadline + 15:
            time.sleep(1)
        print("[soak] the deadline passed: stopping the server and ending", flush=True)
        stop_server()  # every server started (SERVERS holds one from its start, before it answers)
        os._exit(124)

    threading.Thread(target=watchdog, daemon=True).start()
    info = {"piece": piece, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "profile": args.profile, "clients": args.clients}
    try:
        if args.serve:
            proc, url, how = start_server(args, piece)
            info["server"] = how
            print(f"[soak] piece {piece}: server {how}", flush=True)
            if url is None:
                info["ended"] = "no server"
                state["pieces"].append(info)
                save_state(path, state)
                return 3 if paused() else 1
        seconds = min(args.seconds, state["target_s"] - state["done_s"], deadline - now() - args.tail)
        if seconds < 10:  # the server's start took the time: no piece (not counted as one that ran)
            info.update(ended="no time", left_s=round(seconds, 1))
            state["pieces"].append(info)
            save_state(path, state)
            print(f"[soak] piece {piece}: {seconds:.0f} s left for traffic before --deadline less --tail: not run", flush=True)
            return 1
        soak = Soak(args, piece, url)
        seconds = min(seconds, deadline - soak.t0 - args.tail)  # the traffic ends by deadline - tail, and is then "done"
        threads = [threading.Thread(target=soak.client, args=(i,), daemon=True) for i in range(args.clients)]
        threads.append(threading.Thread(target=soak.sampler, daemon=True))
        plan = soak.chaos_plan(seconds)
        print(f"[soak] piece {piece}: {seconds:.0f} s of traffic ({args.profile}, {args.clients} clients), chaos {[(round(a), k) for a, k in plan]}", flush=True)
        for th in threads:
            th.start()
        events = soak.chaos(plan)  # each at its own time
        ended = "done"
        while now() - soak.t0 < seconds:
            if paused():
                ended = "paused"
                break
            if now() > deadline - args.tail:
                ended = "deadline"
                break
            time.sleep(1)
        traffic = now() - soak.t0
        soak.stop.set()
        drain = now()
        until = drain + max(5, min(60, deadline - drain - 60))  # the requests under way end (one deadline for all clients)
        for th in threads[: args.clients]:
            th.join(max(0.0, until - now()))
        info["drain_s"] = round(now() - drain, 1)
        info["clients_still_waiting"] = sum(th.is_alive() for th in threads[: args.clients])
        for th in events:  # the idle connections and the ballast end with the traffic; leave50 is short
            th.join(max(0.0, min(until, deadline - args.tail / 2) - now()))
        info["chaos_still_running"] = [th.name for th in events if th.is_alive()]
        base = url.rsplit("/v1", 1)[0]
        if "handler" in args.chaos or "engine" in args.chaos:
            try:
                info["faults_cleared"] = soak.clear_faults()
            except OSError as e:
                info["faults_cleared"] = f"failed: {type(e).__name__}: {e}"
        if ended != "paused":
            audit = wait_idle_audit(base, max(5, min(60, deadline - now() - 40)))
            soak.write("audits.jsonl", audit)
            info["audit"] = {k: audit.get(k) for k in ("idle", "ok", "problems")}
            # the replay: a share of the seeded replies that ended well, asked again alone
            cands = list(soak.replay)
            random.Random(piece).shuffle(cands)
            want = max(1, round(len(cands) * args.replay)) if cands else 0
            same = done = 0
            for c in cands[:want]:
                if paused() or now() > deadline - 25:
                    break
                out = request(url, c["body"], c["stream"], timeout=max(5.0, deadline - now() - 20))
                done += 1
                eq = out["status"] == 200 and out["content"] == c["content"] and out["reasoning"] == c["reasoning"]
                same += eq
                rec = {"scenario": c["scenario"], "same": eq, "status": out["status"]}
                if not eq:
                    a, b = c["reasoning"] + c["content"], out["reasoning"] + out["content"]
                    at = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                    rec.update(at=at, under_load=a[at: at + 60], alone=b[at: at + 60], error=out.get("error"))
                soak.write("replays.jsonl", rec)
            info["replay"] = {"candidates": len(cands), "wanted": want, "asked": done, "same": same}
            try:
                status, body = get(base, "/metrics")
                with open(os.path.join(args.state, f"metrics_{piece}.txt"), "w", encoding="utf-8") as f:
                    f.write(body.decode())
            except OSError:
                pass
        info.update(ended=ended, traffic_s=round(traffic, 1), counts=dict(soak.counts), wall_s=round(now() - t_begin, 1))
        state["done_s"] += traffic
        state["pieces"].append(info)
        save_state(path, state)
        print(f"[soak] piece {piece} {ended}: {json.dumps(info, ensure_ascii=False)}", flush=True)
        print(f"[soak] {state['done_s']:.0f} of {state['target_s']:.0f} s done", flush=True)
        return 3 if ended == "paused" else 0
    finally:
        stop_server(proc)


# the report

def slope(points: list) -> tuple:
    """Least squares of (t s, value) -> (per hour, mean)."""
    n = len(points)
    if n < 3:
        return None, None
    mt = sum(t for t, _ in points) / n
    mv = sum(v for _, v in points) / n
    var = sum((t - mt) ** 2 for t, _ in points)
    if var == 0:
        return None, mv
    return sum((t - mt) * (v - mv) for t, v in points) / var * 3600, mv


def read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def report(dir_: str) -> dict:
    if not os.path.exists(os.path.join(dir_, "state.json")):
        raise SystemExit(f"no soak in {dir_} (no state.json)")
    state = load_state(os.path.join(dir_, "state.json"), 0)
    reqs, samples = read_jsonl(os.path.join(dir_, "requests.jsonl")), read_jsonl(os.path.join(dir_, "samples.jsonl"))
    replays, audits, events = (read_jsonl(os.path.join(dir_, n)) for n in ("replays.jsonl", "audits.jsonl", "events.jsonl"))
    out = {"target_s": state["target_s"], "done_s": round(state["done_s"], 1), "pieces": []}
    for info in state["pieces"]:
        p = info["piece"]
        rs = [r for r in reqs if r["piece"] == p]
        ss = [s for s in samples if s["piece"] == p and "error" not in s]
        classes = {}
        for r in rs:
            classes[r["class"]] = classes.get(r["class"], 0) + 1
        ok = [r for r in rs if r["class"] == "ok"]
        ttft = sorted(r["first_s"] for r in ok if r["first_s"] is not None)
        hours = {}
        for r in ok:
            if r["first_s"] is not None:
                hours.setdefault(int(r["t"] // 3600), []).append(r["first_s"])
        p99 = {h: round(L.pct(sorted(v), 0.99), 3) for h, v in sorted(hours.items())}
        trends = {}
        for k in TRENDS:
            pts = [(s["t"], s[k]) for s in ss if s.get(k) is not None]
            per_h, mean = slope(pts)
            trends[k] = {"first": pts[0][1] if pts else None, "last": pts[-1][1] if pts else None, "max": max((v for _, v in pts), default=None),
                         "per_hour": None if per_h is None else round(per_h, 1),
                         "per_hour_share": None if per_h is None or not mean else round(per_h / mean, 4), "samples": len(pts)}
        traffic = info.get("traffic_s") or 1
        tokens = sum(r["reply"] or 0 for r in rs if r["class"] in ("ok", "left"))
        busy = [s["busy_share"] for s in ss if s.get("busy_share") is not None]
        rp = [r for r in replays if r["piece"] == p]
        au = [a for a in audits if a["piece"] == p]
        piece = {"piece": p, "ended": info.get("ended"), "traffic_s": info.get("traffic_s"), "requests": len(rs), "classes": classes,
                 "reply_tok_s": round(tokens / traffic, 1), "busy_share_mean": round(sum(busy) / len(busy), 3) if busy else None,
                 "ttft_p50": round(L.pct(ttft, 0.5), 3) if ttft else None, "ttft_p90": round(L.pct(ttft, 0.9), 3) if ttft else None,
                 "ttft_p99": round(L.pct(ttft, 0.99), 3) if ttft else None, "ttft_p99_by_hour": p99,
                 "replays": {"asked": len(rp), "same": sum(r["same"] for r in rp)}, "audits": [{k: a.get(k) for k in ("idle", "ok", "problems")} for a in au],
                 "streams_without_done": sum(1 for r in rs if r["stream"] and r["class"] == "unexpected" and not r["done"]),
                 "events": [(e["event"], e.get("detail")) for e in events if e["piece"] == p], "trends": trends}
        breaks = []
        if classes.get("unexpected"):
            breaks.append(f"{classes['unexpected']} unexpected failures")
        if piece["replays"]["asked"] != piece["replays"]["same"]:
            breaks.append(f"{piece['replays']['asked'] - piece['replays']['same']} replays differ")
        if not au:
            breaks.append("no audit ran")
        elif any(a.get("idle") is None for a in au):  # a stand-in (scripts/serve_canned) has no engine: nothing was checked
            breaks.append(f"no audit: {next(a.get('error') for a in au if a.get('idle') is None)}")
        elif not all(a.get("idle") for a in au):
            breaks.append("the audit found no idle moment (it did not run)")
        elif not all(a.get("ok") for a in au):
            breaks.append(f"the audit did not hold: {[p for a in au for p in a.get('problems') or []][:5]}")
        if info.get("clients_still_waiting"):
            breaks.append(f"{info['clients_still_waiting']} clients still waited for a reply when the drain ended (a hang?)")
        if info.get("ended") not in ("done", "paused"):
            breaks.append(f"the piece ended: {info.get('ended')}")
        if piece["streams_without_done"]:
            breaks.append(f"{piece['streams_without_done']} streams ended without [DONE]")
        if piece["replays"]["asked"] == 0 and info.get("ended") == "done":
            breaks.append("no replay was asked")
        long_enough = (info.get("traffic_s") or 0) >= TREND_S
        flat = [k for k, v in trends.items() if v["per_hour_share"] is not None and abs(v["per_hour_share"]) >= 0.01]
        if long_enough and flat:
            breaks.append(f"trends of 1%/h or more: {flat}")
        if long_enough and len(p99) > 1 and max(p99.values()) > 2 * p99[min(p99)]:
            breaks.append("p99 TTFT more than twice the first hour's")
        piece["invariant_breaks"] = breaks
        piece["trend_gate_applies"] = long_enough
        out["pieces"].append(piece)
    with open(os.path.join(dir_, "report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", help="the soak's folder (resumable)")
    ap.add_argument("--report", help="a soak's folder: print and write its report.json")
    ap.add_argument("--url", default="", help="a running server (started with --debug) instead of --serve")
    ap.add_argument("--serve", action="store_true", help="start the server (scripts/serve.py --debug ...) for the piece, and stop it after")
    ap.add_argument("--port", type=int, default=8124)
    ap.add_argument("--serve-args", default="", help="more options for the server")
    ap.add_argument("--serve-module", default="scripts.serve", help="the server --serve starts (scripts.serve_canned: the stand-in, "
                    "no GPU, to try the harness)")
    ap.add_argument("--profile", choices=("mixed", "gate"), default="mixed")
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=300, help="traffic seconds of this piece at most")
    ap.add_argument("--target", type=float, default=3600, help="traffic seconds of the whole soak (a new state only)")
    ap.add_argument("--deadline", type=float, default=460, help="seconds from the start of the piece by which the server is stopped")
    ap.add_argument("--tail", type=float, default=110, help="seconds kept before the deadline for the drain, the audit and the replay")
    ap.add_argument("--replay", type=float, default=0.05, help="share of the seeded replies asked again alone")
    ap.add_argument("--chaos", default=None, help="the kinds, with commas (mixed: idle,handler,ballast,engine,leave50,stream_options; "
                    "gate: idle,leave50,stream_options, the brief's three)")
    ap.add_argument("--chaos-every", type=float, default=0, help="seconds between rounds of chaos (0: one round a piece)")
    ap.add_argument("--ballast-gib", type=float, default=1.0)
    ap.add_argument("--sample-every", type=float, default=10.0)
    ap.add_argument("--strict-pause", action="store_true", help="do not run while logs/pause is there at all (without it: stop when it "
                    "appears or is touched again after the start)")
    args = ap.parse_args()
    global paused
    paused = Pause(PAUSE, args.strict_pause)
    if paused.at_start is not None:
        print(f"[soak] logs/pause is there (since {time.strftime('%m-%d %H:%M', time.localtime(paused.at_start))}): "
              + ("not run (--strict-pause)" if args.strict_pause else "the piece runs, and stops when it is touched again"), flush=True)
    if args.report:
        got = report(args.report)
        print(json.dumps(got, indent=1, ensure_ascii=False))
        for p in got["pieces"]:  # the verdict, plainly: a break is a break, whatever its cause
            print(f"[soak] piece {p['piece']}: " + (f"{len(p['invariant_breaks'])} INVARIANT BREAK(S): " + "; ".join(p["invariant_breaks"])
                                                    if p["invariant_breaks"] else "no invariant break"))
        return
    if not args.state or not (args.serve or args.url):
        ap.error("--state and one of --serve, --url")
    if args.deadline < args.tail + LEAST_TRAFFIC:  # else the piece would end at once, with no traffic, and look done
        ap.error(f"--deadline {args.deadline:g} leaves no time for traffic: it must be at least --tail ({args.tail:g} s) + {LEAST_TRAFFIC} s"
                 + (" + the server's start (15-60 s)" if args.serve else ""))
    from scripts import serve_admin as A

    keys = A.parse_keys(os.environ.get(A.ENV_KEY, ""), "key")
    if keys:  # the server started here (or at --url) reads the same variable: its requests give the first key
        KEY["Authorization"] = "Bearer " + keys[0][1].decode()
    if args.chaos is None:  # with --url the ballast only when it is asked for: the server may be a stand-in, outside the GPU lock
        args.chaos = ("idle,leave50,stream_options" if args.profile == "gate" else
                      "idle,handler,ballast,engine,leave50,stream_options" if args.serve else "idle,handler,engine,leave50,stream_options")
    args.chaos = [c for c in args.chaos.split(",") if c]
    if args.serve and args.serve_module != "scripts.serve" and "ballast" in args.chaos:  # the stand-in holds no GPU lock: no ballast
        args.chaos.remove("ballast")
        print("[soak] no ballast with a stand-in server (it would take VRAM outside the GPU lock)", flush=True)
    sys.exit(run_piece(args))


if __name__ == "__main__":
    main()
