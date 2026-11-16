# Keeps the server up (R3): starts scripts/serve.py with --crash-only and the options given, and starts it again
# whenever it ends with an error (exit 75: its engine died or hung; anything else not 0: a crash, a failed load), after
# 1, 2, 4 ... 60 s. While it is down the port answers 503 with Retry-After and closes the connection after each answer
# (a client that keeps its connection reaches the server when it is up again). A server that has once said "ok" on
# /health and then does not answer it, or says it is dead or hung, for --health-grace seconds is stopped and started
# again. Ctrl+C (or Ctrl+Break) stops the server gracefully (what is under way finishes, up to its --drain-s) and then
# the supervisor. It stops only the process it started, by its PID (and that process's children: the venv's
# python.exe is a launcher with the real interpreter as its child), never by name or path. It imports no torch.
#
# A manual sleep (for a game): while the file logs/serve_sleep is there (--sleep-file) the server is stopped gracefully,
# so all of its VRAM (14.0 GiB measured, the CUDA context with it) is free, and the port answers 503 "asleep"; when the file is
# removed the server starts again (a start-up from its image with --image auto). Kept conversations are read again at
# their next turn.
#
#   python -m scripts.supervise [supervisor options] [serve.py's options ...]      (serve.bat --supervise ..., or
#                                                                                  supervise = true in serve.toml)
#   log: logs/supervise.log (the supervisor's own lines: starts, exits, waits; nothing of a request)
import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

from scripts import serve_ops

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "logs", "supervise.log")


class Log:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, what: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {what}"
        print(f"[supervise] {what}", flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def option(args: list, name: str, default: str) -> str:
    """The value of serve.py's option `name` among its arguments."""
    for k, a in enumerate(args):
        if a == name and k + 1 < len(args):
            return args[k + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def configured(args: list, config: str = os.path.join(ROOT, "serve.toml")) -> dict:
    """The server's serve.toml (--config another, --config "" none; scripts/serve_admin.py), names with _: {} when there
    is none or it cannot be read (the server reports that)."""
    path = option(args, "--config", None)
    path = path if path is not None else (config if os.path.exists(config) else "")
    if not path:
        return {}
    import tomllib

    try:
        with open(path, "rb") as f:
            return {k.replace("-", "_"): v for k, v in tomllib.load(f).items()}
    except (OSError, ValueError):
        return {}


def wanted(args: list, config: str = os.path.join(ROOT, "serve.toml")) -> bool:
    """Whether serve.bat runs the server through the supervisor: --supervise among its arguments, or supervise = true
    in its serve.toml (serve.public.example.toml)."""
    return any(a.lower() == "--supervise" for a in args) or configured(args, config).get("supervise") is True


def served_at(args: list, config: str = os.path.join(ROOT, "serve.toml")) -> tuple:
    """(host, port) the server will take: its command line, then its serve.toml (--config another, --config "" none;
    scripts/serve_admin.py), then its defaults. A file that cannot be read is left to the server to report."""
    data = configured(args, config)
    host = option(args, "--host", None) or str(data.get("host") or "127.0.0.1")
    port = int(option(args, "--port", None) or data.get("port") or 8001)
    return host, port


def running_images() -> set:
    """The image names of the processes running now, lower case (tasklist: names only; no process is opened)."""
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.split('","', 1)[0].strip('"').lower() for line in out.splitlines() if line.startswith('"')}


class Sleeper:
    """Whether the server should sleep now: the sleep file is there, or one of the --sleep-for programs runs (a game).
    The process list is read every `every` seconds; once the last of them has ended, the server stays asleep
    `wake_after` seconds more (a game that restarts itself, a launcher's second step)."""

    def __init__(self, path: str, names: str = "", every: float = 3.0, wake_after: float = 20.0, images=running_images, clock=time.monotonic):
        self.path, self.every, self.wake_after, self.images, self.clock = path, every, wake_after, images, clock
        self.names = {n.strip().lower() for n in (names or "").split(",") if n.strip()}
        self.checked, self.seen, self.last_seen = None, None, None

    def game(self):
        """The watched program running now (or within wake_after), None if none."""
        if not self.names:
            return None
        now = self.clock()
        if self.checked is None or now - self.checked >= self.every:
            self.checked = now
            found = sorted(self.names & self.images())
            if found:
                self.seen, self.last_seen = found[0], now
        if self.last_seen is not None and now - self.last_seen < self.every + self.wake_after:
            return self.seen
        return None

    def why(self):
        """Why the server sleeps now ("the sleep file", "BF6.exe runs"), None when it should run."""
        if os.path.exists(self.path):
            return f"{self.path} is there"
        g = self.game()
        return f"{g} runs" if g else None


def kill_tree(pid: int) -> None:
    """The process we started and its children, by PID."""
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)


class Holder:
    """The port while the server is down: 503 "restarting" (or "asleep") with Retry-After to every request, and the
    connection closed after each answer (scripts/serve_ops.py: answer_503), so that a client that keeps its connection
    reaches the server once it is up again."""

    def __init__(self, host: str, port: int, log):
        self.host, self.port, self.log, self.httpd, self.until, self.state = host, port, log, None, 0.0, "restarting"

    def answer(self) -> tuple:
        """(state, reason, Retry-After): what is left of the wait, and a start-up."""
        why = {"restarting": "the server ended and is started again", "asleep": "the server sleeps (the sleep file is there)"}[self.state]
        return self.state, why, max(1, round(self.until - time.time())) + 15

    def start(self, seconds: float, state: str = "restarting") -> None:
        """Answers for `seconds` (the Retry-After it gives: what is left of them and a start-up); state "asleep": until
        the sleep file is gone."""
        self.until, self.state = time.time() + seconds, state
        try:
            self.httpd = ThreadingHTTPServer((self.host, self.port), serve_ops.placeholder(self.answer))
            self.httpd.daemon_threads = True
            threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        except OSError as e:
            self.httpd = None
            self.log(f"the port {self.port} could not be held while the server is down: {e}")

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


class Watch:
    """Polls the server's /health. bad(): seconds it has been failing or not "ok" since it last was (None: it never
    said "ok" yet, or it is fine)."""

    def __init__(self, url: str, every: float = 2.0):
        self.url, self.every, self.seen_ok, self.bad_since, self.stopped = url, every, False, None, False
        self.last = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self.stopped:
            try:
                with urllib.request.urlopen(self.url, timeout=5) as r:
                    ok = r.status == 200
                    self.last = r.status
            except Exception as e:  # noqa: BLE001  (503, refused, timed out)
                ok = False
                self.last = getattr(e, "code", type(e).__name__)
            if ok:
                self.seen_ok, self.bad_since = True, None
            elif self.seen_ok and self.bad_since is None:
                self.bad_since = time.time()
            time.sleep(self.every)

    def bad(self) -> float | None:
        return None if self.bad_since is None else time.time() - self.bad_since


def main() -> None:
    if sys.argv[1:2] == ["--wanted"]:  # serve.bat: through the supervisor? (yes or no; serve.toml's supervise = true)
        print("yes" if wanted(sys.argv[2:]) else "no")
        return
    ap = argparse.ArgumentParser(description="Keeps scripts/serve.py up; options it does not know go to the server.", allow_abbrev=False)
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--backoff-max", type=float, default=60.0, help="the longest wait before a restart (seconds)")
    ap.add_argument("--healthy-s", type=float, default=60.0, help="a run at least this long starts the back-off from 1 s again")
    ap.add_argument("--health-grace", type=float, default=30.0, help="seconds /health may fail, once it said ok, before the server is "
                    "stopped and started again (the server's own watchdog acts first)")
    ap.add_argument("--start-grace", type=float, default=300.0, help="seconds a start may take before it counts as hung")
    ap.add_argument("--max-runs", type=int, default=0, help="stop after this many starts (0: never; for tests)")
    ap.add_argument("--module", default="scripts.serve", help="what is run (tests: a stand-in)")
    ap.add_argument("--sleep-file", default=os.path.join(ROOT, "logs", "serve_sleep"), help="while this file is there (for a game), the "
                    "server is stopped gracefully and its VRAM is free; the port answers 503 'asleep'; when it is removed the server starts "
                    "again (from its image with --image auto)")
    ap.add_argument("--sleep-for", default=None, help="program names, by commas (e.g. BF6.exe): while one runs, the server sleeps as "
                    "with the sleep file; it wakes --wake-after seconds after the last has ended (serve.toml: sleep-for)")
    ap.add_argument("--wake-after", type=float, default=20.0, help="seconds after a --sleep-for program has ended before the server starts")
    sup, rest = ap.parse_known_args()
    rest = [a for a in rest if a.lower() != "--supervise"]  # serve.bat --supervise (anywhere, any case) passes it on
    if "--crash-only" not in rest:
        rest.append("--crash-only")
    log = Log(sup.log)
    host, port = served_at(rest)  # serve.toml's host and port too: the port held and watched is the server's
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    cmd = [sys.executable, "-u", "-m", sup.module] + rest
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    stopping = threading.Event()

    def stop(*_):
        stopping.set()

    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, stop)
    holder, backoff, runs = Holder(host, port, log), 1.0, 0
    games = sup.sleep_for if sup.sleep_for is not None else str(configured(rest).get("sleep_for") or "")
    sleeper = Sleeper(sup.sleep_file, games, wake_after=sup.wake_after)
    log(f"supervising: {' '.join(cmd)}")
    if sleeper.names:
        log(f"sleeps while one of these runs: {', '.join(sorted(sleeper.names))}")

    def stop_child(child, why: str):
        """A graceful stop: the server lets what is under way finish (Ctrl+Break), else it is stopped by its PID."""
        log(f"{why}: stopping the server (pid {child.pid}) gracefully")
        try:
            os.kill(child.pid, signal.CTRL_BREAK_EVENT)
        except OSError as e:
            log(f"could not signal it ({e})")
        try:
            return child.wait(timeout=float(option(rest, "--drain-s", "30")) + 15)
        except subprocess.TimeoutExpired:
            log(f"it did not end: stopping pid {child.pid} and its children")
            kill_tree(child.pid)
            return child.wait()

    while not stopping.is_set():
        why_sleep = sleeper.why()
        if why_sleep:  # asleep: no server, the port says so, until the file is gone and no watched program runs
            t_sleep = time.time()
            log(f"asleep while {why_sleep}")
            holder.start(30, "asleep")
            while sleeper.why() and not stopping.is_set():
                stopping.wait(0.5)
                holder.until = time.time() + 30
            holder.stop()
            if stopping.is_set():
                break
            log(f"awake after {time.time() - t_sleep:.0f} s asleep")
            backoff = 1.0
        holder.stop()
        runs += 1
        t0 = time.time()
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        log(f"run {runs}: started the server, pid {child.pid}")
        watch = Watch(f"http://{probe_host}:{port}/health")
        code, why = None, None
        while code is None:
            try:
                code = child.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                pass
            if stopping.is_set():
                code = stop_child(child, "the supervisor stops")
                break
            why_sleep = sleeper.why()
            if why_sleep:
                why = "sleep"
                code = stop_child(child, why_sleep)
                break
            bad = watch.bad()
            if bad is not None and bad > sup.health_grace:
                why = f"/health has not said ok for {bad:.0f} s (last: {watch.last})"
            elif not watch.seen_ok and time.time() - t0 > sup.start_grace:
                why = f"no ok on /health {sup.start_grace:.0f} s after the start"
            if why:
                log(f"{why}: stopping pid {child.pid} and its children")
                kill_tree(child.pid)
                code = child.wait()
        watch.stopped = True
        ran = time.time() - t0
        if stopping.is_set():
            log(f"the server ended ({code}); the supervisor stops")
            break
        if why == "sleep":
            log(f"the server ended ({code}) to sleep after {ran:.0f} s")
            continue
        if code == 0 and why is None:
            log(f"the server ended normally after {ran:.0f} s; the supervisor stops")
            break
        if sup.max_runs and runs >= sup.max_runs:
            log(f"the server ended ({code}) after {ran:.0f} s; {runs} runs: the supervisor stops")
            break
        if ran >= sup.healthy_s:
            backoff = 1.0
        log(f"the server ended ({code}{', ' + why if why else ''}) after {ran:.0f} s: starting it again in {backoff:.0f} s")
        holder.start(backoff)
        stopping.wait(backoff)
        backoff = min(backoff * 2, sup.backoff_max)
    holder.stop()


if __name__ == "__main__":
    main()
