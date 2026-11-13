# An OpenAI-compatible server for Gemma 4 31B on the engine, for several users at once: every pass of the main model
# carries tokens of all the requests that are under way (sclmd/batch.py), and a conversation's keys and values stay
# on the GPU after its reply, so its next turn reads only what is new.
#
#   POST /v1/chat/completions   messages, stream, max_tokens, temperature, top_p, top_k, seed, stop,
#                               reasoning_effort or thinking (the reply's thinking comes as reasoning_content)
#   GET  /v1/models, /health, / (what the server is doing)
#
# One thread owns the engine; the HTTP threads only hand it requests and turn the tokens it settles into text.
import argparse
import json
import os
import queue
import select
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from sclmd import ckpt
from sclmd import engine as E
from sclmd import model as M
from sclmd import plan as P
from sclmd.batch import Batcher, Job
from sclmd.generate import Picker
from sclmd.stream import VRAM_MARGIN, cap_vram

DEPTH = 6  # the drafter's generation_config.json: num_assistant_tokens
SHARES = "own_v1"  # the measurements a plan is fitted from (scripts/chat.py)
LOAD_ROOM = 320 << 20
MODEL = "gemma-4-31b-it"


class Refused(Exception):
    """A request that cannot be served: (HTTP status, message)."""


class Request:
    """A chat completion under way: what the HTTP thread and the engine's thread know of it."""

    def __init__(self, full: list, start: int, max_new: int, pick: Picker, think: bool, messages: list | None = None):
        self.full, self.start, self.max_new, self.pick, self.think, self.messages = full, start, max_new, pick, think, messages
        self.events = queue.Queue()  # ("token", id), then ("done", {...}) or ("error", status, message)
        self.cancelled = False
        self.job = None
        self.reused = 0
        self.t_in, self.t_first = time.perf_counter(), None


class Kept:
    """A conversation whose reply is over, kept on its sequence. base: the conversation up to where the model's last
    turn began, which the sequence can always go back to (snapped: through the copy in RAM, Engine.restore).
    later: the conversation with that reply as a later turn writes it, which the sequence is reading in or has read
    (job), so that the next turn finds it there."""

    def __init__(self, seq: int, base: list, snapped: bool):
        self.seq, self.base, self.snapped, self.used = seq, base, snapped, time.time()
        self.later, self.job = None, None

    def now(self, eng: E.Engine) -> list:
        """What the sequence holds."""
        return self.base if self.later is None else self.later[: eng.lengths[self.seq]]


class Parked:
    """A kept conversation that has left the GPU: its tokens and the engine's state of it in RAM (Engine.park)."""

    def __init__(self, tokens: list, state: dict, used: float):
        self.tokens, self.state, self.used = tokens, state, used


class Server:
    """The engine's side: requests come in through `inbox`, each becomes a job on a sequence, and run() makes passes
    while there is anything to do. Conversations whose reply is over stay on their sequences (kept); when sequences or
    keys and values run short, the one used longest ago moves to RAM (parked, up to `ram` bytes), and comes back
    when its next turn arrives."""

    def __init__(self, tok, eng: E.Engine, drafter, stop: tuple, ram: int = 0):
        self.tok, self.eng, self.stop, self.ram = tok, eng, stop, ram
        self.parked = []
        self.batch = Batcher(eng, drafter, DEPTH)
        self.batch.make_room = self._make_room
        self.canon = None  # (request, the reply's tokens) -> the conversation's tokens with the reply as a later turn writes it
        self.inbox, self.waiting, self.active = queue.Queue(), [], []
        self.kept = {}  # sequence -> Kept
        self.unused = list(range(eng.sequences - 1, -1, -1))
        self.totals = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
        self.t0, self.busy_s = time.time(), 0.0

    def submit(self, req: Request) -> None:
        self.inbox.put(req)

    def _make_room(self) -> bool:
        """The pool of keys and values is empty, or every sequence is taken: the kept conversation that was used
        longest ago leaves the GPU (to RAM while there is room for it there). False: none is kept."""
        if not self.kept:
            return False
        old = min(self.kept.values(), key=lambda k: k.used)
        if self.ram <= 0:
            self._forget(old)
            return True
        if old.job is not None:
            self.batch.drop(old.job)
        del self.kept[old.seq]
        self.unused.append(old.seq)
        if old.later is None and not self._to_start(old):  # nothing of it can be gone on from
            self.eng.release(old.seq)
            return True
        self.parked.append(Parked(old.now(self.eng), self.eng.park(old.seq), old.used))
        while sum(p.state["bytes"] for p in self.parked) > self.ram:  # RAM is full: the oldest parked ones are forgotten
            self.parked.remove(min(self.parked, key=lambda p: p.used))
        return True

    def _to_start(self, kept: Kept) -> bool:
        """Takes a kept conversation's sequence back to where its last turn began (its base): past it the sequence
        holds the reply as it was written, which a later turn does not begin with. False: the sliding layers no
        longer hold what that position needs (should not happen: the batcher copies a turn start to RAM before a
        reply writes over it)."""
        eng, seq, n = self.eng, kept.seq, len(kept.base)
        if eng.lengths[seq] == n:
            return True
        if kept.snapped or (kept.job is not None and kept.job.snapped):
            eng.restore(seq)
        if not eng._kept(n - eng.cfg.window + 1, seq):
            return False
        eng.crop(n, seq)
        eng.trim(seq)
        return True

    def _forget(self, kept: Kept) -> None:
        if kept.job is not None:
            self.batch.drop(kept.job)
        del self.kept[kept.seq]
        self.eng.release(kept.seq)
        self.unused.append(kept.seq)

    def _place(self, req: Request) -> bool:
        """A sequence for the request: the kept conversation it continues, else an unused one, else the one kept
        longest. False: every sequence is writing, or texts under way are waiting for keys and values (the request waits)."""
        eng, best, most = self.eng, None, 0
        if self.batch.starved and self.active:
            return False
        for k in self.kept.values():
            for held in (k.now(eng), k.base):  # as far as the sequence has got, else back at its last turn's start
                n = len(held)
                if most < n < len(req.full) and req.full[:n] == held:
                    best, most = k, n
        back = max((p for p in self.parked if most < len(p.tokens) < len(req.full) and req.full[: len(p.tokens)] == p.tokens),
                   key=lambda p: len(p.tokens), default=None)
        reused = 0
        if back is not None:  # the conversation is in RAM: back onto a sequence, making room as needed
            self.parked.remove(back)
            if not self.unused and not self._make_room():
                self.parked.append(back)
                return False
            seq = self.unused.pop()
            while True:
                try:
                    eng.unpark(seq, back.state)
                    reused = len(back.tokens)
                    break
                except E.KVFull:
                    if not self._make_room():  # the texts under way hold too much for it now: it starts anew
                        break
        elif best is not None:
            seq = best.seq
            if best.job is not None:
                self.batch.drop(best.job)
            del self.kept[seq]
            reused = most
            if eng.lengths[seq] != most and not self._to_start(best):  # back to the last turn's start
                eng.release(seq)
                reused = 0
        elif self.unused:
            seq = self.unused.pop()
        elif self._make_room():
            seq = self.unused.pop()
        else:
            return False
        # the sequence holds exactly what is counted as kept: nothing of another conversation
        assert eng.lengths[seq] == reused, f"sequence {seq} holds {eng.lengths[seq]} positions where {reused} were expected"
        req.reused = reused
        req.job = Job(seq, req.full[reused:], req.max_new, req.pick, self.stop, emit=lambda t, q=req.events: q.put(("token", t)),
                      start=req.start, base=req.full[:reused])
        self.batch.add(req.job)
        self.active.append(req)
        return True

    def _finish(self, req: Request) -> None:
        eng, job = self.eng, req.job
        seq = job.seq
        if job.done in ("stop", "length") and job.out:
            kept = self.kept[seq] = Kept(seq, req.full[: req.start], job.snapped)
            try:
                later = self.canon(req, job.out) if self.canon is not None else None
            except Exception as e:  # noqa: BLE001  (a template that cannot write the reply: nothing is read ahead)
                print(f"[serve] seq {seq}: the reply's later form could not be made ({type(e).__name__}: {e})", flush=True)
                later = None
            if later is not None and later[: req.start] == kept.base and req.start < len(later) <= eng.capacity - 2 * eng.rows:
                # the reply as a later turn writes it is read in from the turn's start, with rows nothing else needs
                if job.snapped:
                    eng.restore(seq)
                if eng._kept(req.start - eng.cfg.window + 1, seq):
                    eng.crop(req.start, seq)
                    eng.trim(seq)
                    kept.later = later
                    kept.job = Job(seq, later[req.start :], 0, None, start=req.start, only_read=True, base=later[: req.start])
                    kept.job.snapped = job.snapped
                    self.batch.add(kept.job)
        else:  # cancelled or out of memory: nothing of it is kept
            eng.release(seq)
            self.unused.append(seq)
        now = time.perf_counter()
        usage = {"prompt_tokens": len(req.full), "completion_tokens": len(job.out), "total_tokens": len(req.full) + len(job.out),
                 "prompt_tokens_details": {"cached_tokens": req.reused}}
        self.totals["requests"] += 1
        self.totals["prompt_tokens"] += len(req.full)
        self.totals["cached_tokens"] += req.reused
        self.totals["completion_tokens"] += len(job.out)
        first = req.t_first or now
        timings = {"total_ms": round(1000 * (now - req.t_in)), "first_token_ms": round(1000 * (first - req.t_in)),
                   "tokens_per_second": round(len(job.out) / max(now - first, 1e-9), 1), "prompt_read": job.read,
                   "rows_per_pass": round(job.rows / max(job.passes, 1), 2)}
        reason = {"stop": "stop", "length": "length", "full": "length", "cancelled": "stop"}[job.done]
        req.events.put(("done", {"finish_reason": reason, "usage": usage, "timings": timings, "ended": job.done}))
        print(f"[serve] seq {seq}: prompt {len(req.full)} ({req.reused} kept), reply {len(job.out)} tokens, first after "
              f"{timings['first_token_ms'] / 1000:.2f} s, {timings['tokens_per_second']} tok/s, {job.done}; "
              f"{len(self.active)} under way, {len(self.kept)} kept, {len(self.parked)} in RAM", flush=True)

    @torch.no_grad()
    def turn(self, wait: bool = True) -> bool:
        """Takes in the requests that have come and makes one pass. With nothing under way it waits for a request
        (wait) or returns False."""
        try:
            idle = not self.active and not self.waiting and not self.batch.jobs
            if idle and not wait and self.inbox.empty():
                return False
            self.waiting.append(self.inbox.get(timeout=None if idle else 0))
            while True:
                self.waiting.append(self.inbox.get_nowait())
        except queue.Empty:
            pass
        self.waiting = [r for r in self.waiting if not r.cancelled]
        while self.waiting and self._place(self.waiting[0]):
            self.waiting.pop(0)
        for req in self.active:
            if req.cancelled:
                req.job.cancel = True
        t = time.perf_counter()
        self.batch.step()
        self.busy_s += time.perf_counter() - t
        for req in self.active:
            if req.t_first is None and req.job.out:
                req.t_first = time.perf_counter()
        for req in [r for r in self.active if r.job.done is not None]:
            self.active.remove(req)
            try:
                self._finish(req)
            except Exception:  # it is in no list any more: told here, the others by run()
                req.events.put(("error", 500, "the server failed while ending the reply"))
                raise
        return True

    def run(self) -> None:
        """Turns for ever. If one fails, every request that is under way or waiting is told (the HTTP threads answer
        with an error instead of waiting for tokens that will not come), and the engine starts from nothing."""
        import traceback

        while True:
            try:
                self.turn()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                print(f"[serve] the engine's thread failed: {type(e).__name__}: {e}; {len(self.active) + len(self.waiting)} requests are told", flush=True)
                for req in self.active + self.waiting:
                    req.events.put(("error", 500, f"the server failed while answering: {type(e).__name__}"))
                self.reset()

    def reset(self) -> None:
        """Forgets every conversation and job: the engine as it was when it was loaded."""
        self.active, self.waiting, self.kept, self.parked = [], [], {}, []
        self.batch.jobs, self.batch.starved = [], 0
        for seq in range(self.eng.sequences):
            self.eng.release(seq)
        self.unused = list(range(self.eng.sequences - 1, -1, -1))

    def about(self) -> dict:
        eng = self.eng
        tile = {t: E.tile_bytes(eng.cfg, t) for t in eng.ring}
        free = eng.free_tiles()
        return {"requests": {"under_way": len(self.active), "waiting": len(self.waiting) + self.inbox.qsize(), **self.totals},
                "conversations_kept": len(self.kept), "conversations_in_ram": len(self.parked),
                "ram_gib": round(sum(p.state["bytes"] for p in self.parked) / 2**30, 2), "sequences": eng.sequences,
                "keys_and_values": {"gib": round(sum(tile[t] * eng.pool[t] for t in tile) / 2**30, 2),
                                    "free_gib": round(sum(tile[t] * free[t] for t in tile) / 2**30, 2)},
                "started_over_for_memory": self.batch.restarts, "passes": self.batch.passes, "passes_by_width": dict(self.batch.widths), "rows_per_pass": round(self.batch.rows / max(self.batch.passes, 1), 2),
                "guesses_kept": round(self.batch.kept_rows / max(self.batch.guess_rows, 1), 2),
                "tokens_per_second_while_busy": round(self.totals["completion_tokens"] / max(self.busy_s, 1e-9), 1)}


class Pieces:
    """A reply's tokens as pieces of text, thinking and answer apart: add(token) -> (kind, text) or None."""

    def __init__(self, tok, stop: tuple, opening: int, closing: int):
        self.tok, self.stop, self.opening, self.closing = tok, stop, opening, closing
        self.kind, self.ids, self.done, self.shown, self.name = "answer", [], 0, 0, False

    def add(self, token: int):
        if token in self.stop:
            return None
        if token == self.opening or token == self.closing:
            self.kind, self.ids, self.done, self.shown, self.name = ("think" if token == self.opening else "answer"), [], 0, 0, token == self.opening
            return None
        self.ids.append(token)
        # the text of the tokens from `done` on, less what of it was shown: a few tokens are decoded, not the reply
        before = self.tok.decode(self.ids[self.done : self.shown], skip_special_tokens=False)
        text = self.tok.decode(self.ids[self.done :], skip_special_tokens=False)
        if text.endswith("\ufffd") or len(text) <= len(before):  # a character is still missing bytes
            return None
        piece = text[len(before):]
        self.done, self.shown = self.shown, len(self.ids)
        if self.name:  # the thinking channel's name comes first: "thought\n"
            whole = self.tok.decode(self.ids, skip_special_tokens=False)
            if "thought\n".startswith(whole):
                return None
            self.name = False
            if whole.startswith("thought\n"):
                piece = whole[len("thought\n"):]
        return (self.kind, piece) if piece else None


class App:
    def __init__(self, server: Server, tok, think: bool, max_tokens: int, facts: dict):
        self.server, self.tok, self.think, self.max_tokens, self.facts = server, tok, think, max_tokens, facts
        self.created, self.started = int(time.time()), time.time()
        self.turn = tok.encode("<|turn>model\n", add_special_tokens=False)
        self.opening, self.closing = tok.convert_tokens_to_ids("<|channel>"), tok.convert_tokens_to_ids("<channel|>")
        self.fingerprint = f"sclmd-{facts['bits_per_weight']}bit"
        self.lock = threading.Lock()  # the tokenizer's template is not known to be safe from several threads
        server.canon = self.canonical

    def canonical(self, req: Request, tokens: list):
        """The conversation with the reply just given as a later turn writes it: the answer as the client got it
        (without the thinking), through the template."""
        pieces, text = Pieces(self.tok, self.server.stop, self.opening, self.closing), ""
        for token in tokens:
            piece = pieces.add(token)
            if piece is not None and piece[0] == "answer":
                text += piece[1]
        with self.lock:
            out = self.tok.apply_chat_template(req.messages + [{"role": "assistant", "content": text}], tokenize=True,
                                               add_generation_prompt=False, enable_thinking=req.think)
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    def card(self) -> dict:
        eng = self.server.eng
        return {"id": MODEL, "object": "model", "created": self.created, "owned_by": "sclmd", "root": MODEL, "parent": None,
                "max_model_len": eng.capacity, "context_length": eng.capacity, "max_output_tokens": self.max_tokens,
                "quantization": f"{self.facts['bits_per_weight']} bits a weight",
                "capabilities": {"streaming": True, "reasoning": True, "tools": False, "prompt_cache": True, "vision": False},
                "defaults": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "thinking": self.think, "max_tokens": self.max_tokens}}

    def about(self) -> dict:
        return {"server": "sclmd serve", "model": MODEL, "system_fingerprint": self.fingerprint, "uptime_s": round(time.time() - self.started),
                "endpoints": ["GET /", "GET /health", "GET /v1/models", "GET /v1/models/{id}", "POST /v1/chat/completions"],
                "context_length": self.server.eng.capacity, "max_output_tokens": self.max_tokens, **self.facts, **self.server.about()}

    def request(self, body: dict) -> Request:
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            raise Refused(400, "messages: a list of {role, content} is needed")
        messages = []
        for m in body["messages"]:
            role, content = m.get("role"), m.get("content")
            if isinstance(content, list):  # parts: the text ones
                if any(p.get("type") not in ("text", None) for p in content):
                    raise Refused(400, "only text content is served")
                content = "".join(p.get("text", "") for p in content)
            if role == "developer":
                role = "system"
            if role not in ("system", "user", "assistant") or not isinstance(content, str):
                raise Refused(400, "a message is {role: system | user | assistant, content: text}")
            messages.append({"role": role, "content": content})
        if messages[-1]["role"] == "assistant":
            raise Refused(400, "the last message is the one to answer: a user's")
        think = self.think
        if "thinking" in body:
            t = body["thinking"]
            think = t.get("type") == "enabled" if isinstance(t, dict) else bool(t)
        elif body.get("reasoning_effort") is not None:
            think = body["reasoning_effort"] not in ("none", "minimal")
        with self.lock:
            ids = lambda prompt: self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=prompt, enable_thinking=think)  # noqa: E731
            before, full = ids(False), ids(True)
        before, full = (list(x["input_ids"] if hasattr(x, "keys") else x) for x in (before, full))
        start = len(before) + len(self.turn)
        if full[: len(before)] != before or full[len(before) : start] != self.turn:
            raise Refused(500, "the chat template did not write the turn as expected")
        eng = self.server.eng
        room = eng.capacity - len(full) - eng.rows
        if room < 16:
            raise Refused(400, f"the conversation is {len(full)} tokens: the context holds {eng.capacity}")
        asked = body.get("max_completion_tokens", body.get("max_tokens"))
        max_new = min(int(asked) if asked else self.max_tokens, room)
        if max_new < 1:
            raise Refused(400, "max_tokens is at least 1")
        temperature = float(body.get("temperature", 1.0) if body.get("temperature") is not None else 1.0)
        top_p = float(body.get("top_p", 0.95) if body.get("top_p") is not None else 0.95)
        top_k = int(body.get("top_k", 64) or 64)
        seed = body.get("seed")
        pick = Picker(temperature, max(1, min(top_k, 1024)), top_p, seed=int(seed) & 0x3FFFFFFF if seed is not None else uuid.uuid4().int & 0x3FFFFFFF)
        return Request(full, start, max_new, pick, think, messages)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "sclmd"

    def log_message(self, *args) -> None:  # the engine's thread prints a line per request
        pass

    def _json(self, status: int, obj: dict, headers: dict | None = None) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        kind = "invalid_request_error" if status < 500 else "server_error"
        self._json(status, {"error": {"message": message, "type": kind, "param": None, "code": None}})

    def _gone(self) -> bool:
        """Whether the client has closed the connection."""
        try:
            if not select.select([self.connection], [], [], 0)[0]:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def do_GET(self) -> None:
        app = self.server.app
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/models":
            self._json(200, {"object": "list", "data": [app.card()]})
        elif path.startswith("/v1/models/"):
            if path[len("/v1/models/"):] == MODEL:
                self._json(200, app.card())
            else:
                self._error(404, f"the model served is {MODEL}")
        elif path == "/health":
            about = app.server.about()
            self._json(200, {"status": "ok", "model": MODEL, "under_way": about["requests"]["under_way"],
                             "waiting": about["requests"]["waiting"], "kept": about["conversations_kept"]})
        elif path in ("", "/v1"):
            self._json(200, app.about())
        else:
            self._error(404, "unknown path: GET / lists the endpoints")

    def do_POST(self) -> None:
        app = self.server.app
        if self.path.split("?")[0].rstrip("/") != "/v1/chat/completions":
            return self._error(404, "unknown path: GET / lists the endpoints")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            req = app.request(body)
        except Refused as e:
            return self._error(*e.args)
        except (ValueError, TypeError, AttributeError) as e:
            return self._error(400, f"the body is not a chat completion request: {e}")
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else [s for s in stops if isinstance(s, str) and s]
        app.server.submit(req)
        head = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": MODEL, "system_fingerprint": app.fingerprint}
        stream = bool(body.get("stream"))
        pieces = Pieces(app.tok, app.server.stop, app.opening, app.closing)
        text = {"think": "", "answer": ""}
        sent = 0  # of the answer: what has gone out (a stop string may still be forming at its end)
        hold = max((len(s) for s in stops), default=1) - 1
        stopped = False

        def chunk(delta: dict, finish: str | None = None, **more) -> None:
            data = {**head, "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}], **more}
            self.wfile.write(b"data: " + json.dumps(data, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

        try:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("x-request-id", head["id"])
                self.end_headers()
                self.close_connection = True
                chunk({"role": "assistant", "content": ""})
            while True:
                try:
                    event = req.events.get(timeout=1.0)
                except queue.Empty:
                    if self._gone():
                        raise ConnectionError
                    continue
                if event[0] == "error":
                    if stream:
                        self.wfile.write(b"data: " + json.dumps({"error": {"message": event[2], "type": "server_error"}}).encode() + b"\n\n")
                        break
                    return self._error(event[1], event[2])
                if event[0] == "done":
                    result = event[1]
                    break
                if stopped:
                    continue
                # the tokens that have come by now are taken together (waiting 30 ms for more made no difference
                # to the speed of eight clients: 293 tok/s with it, 300 without)
                tokens, last = [event[1]], None
                try:
                    while last is None:
                        more = req.events.get_nowait()
                        if more[0] == "token":
                            tokens.append(more[1])
                        else:
                            last = more
                except queue.Empty:
                    pass
                new = {"think": "", "answer": ""}
                for token in tokens:
                    piece = pieces.add(token)
                    if piece is not None:
                        new[piece[0]] += piece[1]
                        text[piece[0]] += piece[1]
                if new["think"] and stream:
                    chunk({"reasoning_content": new["think"]})
                if new["answer"]:
                    cut = min((text["answer"].find(s) for s in stops if s in text["answer"]), default=-1)
                    if cut >= 0:  # a stop string: the answer ends before it, and the engine is told to stop
                        text["answer"], stopped, req.cancelled = text["answer"][:cut], True, True
                    upto = len(text["answer"]) if stopped else len(text["answer"]) - hold
                    if stream and upto > sent:
                        chunk({"content": text["answer"][sent:upto]})
                        sent = upto
                if last is not None:
                    event = last
                    if event[0] == "error":
                        if stream:
                            self.wfile.write(b"data: " + json.dumps({"error": {"message": event[2], "type": "server_error"}}).encode() + b"\n\n")
                            break
                        return self._error(event[1], event[2])
                    result = event[1]
                    break
            if event[0] == "done":
                if stream and len(text["answer"]) > sent:
                    chunk({"content": text["answer"][sent:]})
                finish = "stop" if stopped else result["finish_reason"]
                if not stream:
                    message = {"role": "assistant", "content": text["answer"]}
                    if text["think"]:
                        message["reasoning_content"] = text["think"]
                    choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": finish}
                    return self._json(200, {**head, "object": "chat.completion", "choices": [choice], "usage": result["usage"],
                                            "timings": result["timings"]},
                                      {"x-request-id": head["id"], "openai-processing-ms": str(result["timings"]["total_ms"])})
                chunk({}, finish, timings=result["timings"])
                if (body.get("stream_options") or {}).get("include_usage"):
                    data = {**head, "object": "chat.completion.chunk", "choices": [], "usage": result["usage"]}
                    self.wfile.write(b"data: " + json.dumps(data).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (ConnectionError, OSError):  # the client left: its text ends at the next pass
            req.cancelled = True
            self.close_connection = True


def pool_for(cfg: M.Config, budget: int, full_per_sliding: float) -> dict:
    """Tiles of keys and values for `budget` bytes: `full_per_sliding` tiles of the full-attention layers for each of
    the sliding ones (a conversation needs one of each for every 16 tokens up to the sliding window, 1,024, and only
    full-attention ones after that)."""
    unit = E.tile_bytes(cfg, M.SLIDING) + full_per_sliding * E.tile_bytes(cfg, M.FULL)
    n = int(budget // unit)
    return {M.SLIDING: n, M.FULL: int(n * full_per_sliding)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on (this machine only by default; 0.0.0.0 lets others in, "
                    "and there is no password)")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--name", default="v1", help="the codes' folder under C:\\models\\sclmd-codes")
    ap.add_argument("--kv-gib", type=float, default=0.0, help="VRAM for the conversations' keys and values (0: all that 3 bits a weight "
                    "leaves, about 2.4 GiB). A conversation takes 0.42 MB a token up to 1,024 tokens and 41 KB a token after; what is "
                    "not set aside here goes to 4-bit matrices (quality)")
    ap.add_argument("--context", type=int, default=8192, help="tokens one conversation can hold (prompt and reply)")
    ap.add_argument("--sequences", type=int, default=32, help="conversations held at once, those kept for their next turn included")
    ap.add_argument("--ram-gib", type=float, default=24.0, help="RAM for conversations that wait for their next turn after the GPU "
                    "has run short (0: they are forgotten instead and read again from the start)")
    ap.add_argument("--typical", type=int, default=2048, help="tokens of a typical conversation: how the keys and values are split "
                    "between the two kinds of layer")
    ap.add_argument("--max-tokens", type=int, default=4096, help="the most tokens of a reply when the request does not say (thinking included)")
    ap.add_argument("--think", action="store_true", help="think unless the request says otherwise")
    ap.add_argument("--head", type=int, default=8)
    ap.add_argument("--no-drafter", action="store_true")
    ap.add_argument("--rows", type=int, default=32, help="the most rows of a pass (8, 16 or 32): tokens of all requests together")
    ap.add_argument("--reserve", type=float, default=VRAM_MARGIN / 2**30, help="GiB of the VRAM that is free now to leave to other programs")
    cli = ap.parse_args()
    import sys

    import transformers
    from transformers import AutoTokenizer

    sys.setswitchinterval(0.0005)  # the engine's thread does not wait 5 ms for an HTTP thread that holds the interpreter

    transformers.logging.set_verbosity_error()
    reserve = int(cli.reserve * 2**30)
    print("[serve] loading (about 15 s) ...", flush=True)
    cap_vram(margin=reserve)
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    cfg = M.Config.load(ckpt("orig"))
    context = -(-cli.context // E.TILE) * E.TILE
    # the VRAM that is free now, less what is left to other programs, the head, the work memory, the drafter and room
    # to load in: that much is shared between the layers' codes and the conversations' keys and values
    free = torch.cuda.mem_get_info()[0] - reserve
    rest = E.fixed_bytes(cfg, cli.head, context, pool={M.SLIDING: 0, M.FULL: 0}, rows=cli.rows) + (0 if cli.no_drafter else E.DRAFTER_BYTES)
    both = free - rest - LOAD_ROOM
    least = P.code_bytes(cfg, P.uniform(cfg, 6))
    if both - least < (256 << 20):
        raise SystemExit(f"not enough VRAM: {(free + reserve) / 2**30:.1f} GiB are free, and 3 bits a weight with a little room for "
                         f"conversations needs {(least + rest + LOAD_ROOM + reserve) / 2**30 + 0.25:.1f} GiB")
    kv = min(int(cli.kv_gib * 2**30), both - least) if cli.kv_gib > 0 else both - least
    plan, estimate = P.fit(cfg, cli.name, both - kv, of=SHARES) if both - kv > least else (P.uniform(cfg, 6), None)
    typical = max(cli.typical, cfg.window)
    pool = pool_for(cfg, kv, typical / (cfg.window + E.SPARE))
    with torch.no_grad():
        eng = E.Engine.load(cli.name, plan, cli.head, context, sequences=cli.sequences, pool=pool, rows=cli.rows)
        drafter = None if cli.no_drafter else E.EngineDrafter.load(eng, ckpt("drafter"), cli.name, depth=DEPTH)
    stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
    server = Server(tok, eng, drafter, stop, ram=int(cli.ram_gib * 2**30))
    bits = round(P.bits_per_weight(cfg, plan), 2)
    one = E.tile_bytes(cfg, M.SLIDING) * E.ring_tiles(cfg, M.SLIDING, context, cli.rows) + E.tile_bytes(cfg, M.FULL) * typical // E.TILE
    facts = {"bits_per_weight": bits, "kl_estimate": None if estimate is None else round(estimate, 3),
             "vram_gib": round(torch.cuda.memory_allocated() / 2**30, 2), "drafter": drafter is not None}
    app = App(server, tok, cli.think, cli.max_tokens, facts)
    threading.Thread(target=server.run, daemon=True).start()
    httpd = ThreadingHTTPServer((cli.host, cli.port), Handler)
    httpd.daemon_threads = True
    httpd.app = app
    print(f"[serve] {MODEL} at http://{cli.host}:{cli.port}/v1 · {bits} bits a weight"
          + (f" (KL about {estimate:.3f} on its own text)" if estimate else " (KL about 0.10 on its own text)")
          + f" · VRAM {facts['vram_gib']} GiB\n[serve] keys and values {kv / 2**30:.2f} GiB: about {kv // one} conversations of {typical} tokens "
          f"at once, or {kv // (E.tile_bytes(cfg, M.SLIDING) + E.tile_bytes(cfg, M.FULL)) * E.TILE} tokens of shorter ones; "
          f"a conversation holds up to {context} tokens; thinking {'on' if cli.think else 'off'} unless asked", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
