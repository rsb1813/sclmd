# The prompt caches (scripts/serve_cache.py) without a GPU: the names, the RAM store's scopes, budget and times, the
# disk cache's files (encrypted, named by an HMAC, deleted at start, at their time and past the budget), and where
# cache_control's marks fall in Gemma 4's rendered prompt (the real tokenizer and template; App.request with a
# stand-in for the engine).
import array
import http.client
import json
import os
import threading
import time

import pytest
import torch

from sclmd import ckpt
from scripts import serve as S
from scripts import serve_cache as K
from scripts import serve_contract as C


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def state_of(n_tiles: int, seed: int, length: int = 40) -> dict:
    """A state in Engine.park's form, of CPU tensors: two kinds of layer, int8 values and float scales."""
    g = torch.Generator().manual_seed(seed)
    kinds = {}
    for t, layers in (("sliding_attention", 3), ("full_attention", 2)):
        data = K.alloc([((layers, n_tiles, 16, 2, 8), torch.int8), ((layers, n_tiles, 16, 2), torch.float32)])
        data[0].copy_(torch.randint(-127, 127, data[0].shape, generator=g, dtype=torch.int8))
        data[1].copy_(torch.rand(data[1].shape, generator=g))
        kinds[t] = {"js": list(range(n_tiles)), "data": data}
    return {"length": length, "high": length, "kinds": kinds, "bytes": sum(d.numel() * d.element_size() for k in kinds.values() for d in k["data"])}


def same(a: dict, b: dict) -> bool:
    return (a["length"], a["high"], a["bytes"]) == (b["length"], b["high"], b["bytes"]) and all(
        a["kinds"][t]["js"] == b["kinds"][t]["js"] and all(torch.equal(x, y) for x, y in zip(a["kinds"][t]["data"], b["kinds"][t]["data"]))
        for t in a["kinds"])


def test_names_are_the_scopes_and_the_tokens_and_one_pass_gives_each_prefixs():
    names = K.Names()
    toks = list(range(5, 400))
    got = names.of("key:a", toks, [10, 100, 395])
    assert got[100] == names.one("key:a", toks[:100]) and len(got[10]) == 64
    assert names.one("key:a", toks[:100]) != names.one("key:b", toks[:100]) != names.one(None, toks[:100])
    assert names.one("key:a", toks[:100]) != names.one("key:a", toks[:101])
    assert K.Names().one("key:a", toks[:100]) != names.one("key:a", toks[:100])  # another process: another key


def test_the_store_finds_only_the_scopes_own_and_the_longest():
    store = K.Store(1 << 30, K.Names(), Clock())
    a, b = list(range(10, 90)), list(range(10, 60)) + [7] * 30
    store.put(a[:40], state_of(1, 1), "key:a")
    store.put(a[:70], state_of(2, 2), "key:a")
    store.put(b[:40], state_of(1, 3), None)
    assert store.find(a, 0, "key:a")[0] == a[:70]
    assert store.find(a, 70, "key:a") is None  # longer than what is held
    assert store.find(a, 0, "key:b") is None and store.find(a, 0, None)[0] == a[:40]  # None's own (the same 40 tokens)
    assert store.find(a[:70], 0, "key:a")[0] == a[:40]  # shorter than full only
    assert not store.wants(a[:70], "key:a") and store.wants(a[:70], "key:b") and not store.wants(a[:5], "key:a")
    # an entry keeps its name and length, never the prompt's tokens (ZDR: they would sit in pageable RAM)
    for e in store.held.values():
        assert not any(isinstance(getattr(e, s), (list, tuple, array.array)) for s in K.Entry.__slots__ if s != "state")
    assert store.get("key:a", a[:70]).n == 70 and store.get("key:a", a[:69]) is None and store.get("key:b", a[:70]) is None


def test_cache_turns_saves_no_state_a_few_tokens_past_one_already_there():
    """verify-N3-cache: a document sent as a marked part, then as its own message, got a second 0.4-0.9 GiB state two
    tokens past the first. --cache-turns now skips a point within TURN_NEAR past what the request read or its user holds."""
    from types import SimpleNamespace

    def arm(cache, n, reused):
        job = SimpleNamespace(seq=0, pause=None)
        req = SimpleNamespace(full=list(range(10, 10 + n + 50)), scope="key:a", job=job, cut=0, points=[(n, K.IMPLICIT_S, "turn")],
                              reused=reused, went_on=False)
        cache.arm(SimpleNamespace(eng=SimpleNamespace(highs={0: 0}), parked=[]), req)
        return job.pause

    cache = K.Cache(1 << 30, turns=True)
    assert arm(cache, 40, 0) == 40 and cache.near == 0  # nothing read (reused 0) is no state before it, however short
    assert arm(cache, 2000, 1998) is None and cache.near == 1  # read from a state 2 tokens before it
    cache.store.put(list(range(10, 10 + 1990)), state_of(1, 1), "key:a")
    assert arm(cache, 2000, 0) is None and cache.near == 2  # this user holds one 10 tokens before it
    assert arm(cache, 3000, 0) == 3000 and cache.near == 2  # far from both: saved as the prompt is read
    job = SimpleNamespace(seq=0, pause=None)  # a scope of one request (an unsure address): nothing is saved, nobody could find it
    req = SimpleNamespace(full=list(range(10, 4000)), scope="unsure:7", job=job, cut=0, points=[(3000, K.LONG_S, "mark")], reused=0, went_on=False)
    cache.arm(SimpleNamespace(eng=SimpleNamespace(highs={0: 0}), parked=[]), req)
    assert job.pause is None


def test_the_budget_is_one_for_parked_conversations_and_entries_and_the_oldest_goes():
    clock = Clock()
    one = state_of(1, 0)["bytes"]

    class P:
        def __init__(self, used):
            self.state, self.used = state_of(1, 9), used

    parked = [P(clock.t - 30), P(clock.t - 10)]
    store = K.Store(3 * one, K.Names(), clock)
    toks = list(range(10, 90))
    clock.t += 1
    store.put(toks[:20], state_of(1, 1), "s", parked=parked)  # three of one size: fits
    assert len(parked) == 2 and len(store.held) == 1
    clock.t += 1
    store.put(toks[:30], state_of(1, 2), "s", parked=parked)  # four: the oldest (a parked one) goes
    assert len(parked) == 1 and parked[0].used == 1000 - 10 and len(store.held) == 2 and store.evicted == 1
    clock.t += 1
    store.find(toks[:25], 0, "s")  # the 20 tokens' entry is used now: the parked one is the oldest
    clock.t += 1
    store.put(toks[:40], state_of(1, 3), "s", parked=parked)
    assert not parked and len(store.held) == 3
    clock.t += 1
    store.put(toks[:50], state_of(1, 4), "s", parked=parked)  # the 30 tokens' entry was used longest ago
    assert sorted(e.n for e in store.held.values()) == [20, 40, 50]
    big = store.put(toks[:60], state_of(4, 5), "s")  # larger than the whole budget: never held
    assert big is None and K.Names  # noqa: B015
    assert not store.wants(toks[:60], "s")


def test_entries_go_at_their_time_since_last_use():
    clock = Clock()
    store = K.Store(1 << 30, K.Names(), clock)
    toks = list(range(10, 90))
    store.put(toks[:20], state_of(1, 1), "s", ttl=900)
    store.put(toks[:30], state_of(1, 2), "s", ttl=3600)
    clock.t += 800
    assert store.find(toks, 25, "s") is not None  # the 30 tokens' entry used now
    clock.t += 200
    assert store.sweep(clock.t) == 1 and [e.n for e in store.held.values()] == [30]
    clock.t += 3300
    assert store.sweep(clock.t) == 0
    clock.t += 200
    assert store.sweep(clock.t) == 1 and not store.held and store.expired == 2


@pytest.fixture()
def folder(tmp_path):
    return str(tmp_path / "cache")


def wait_written(disk: K.Disk, n: int, seconds: float = 20.0) -> None:
    t = time.time()
    while disk.written < n:
        assert time.time() - t < seconds, "not written"
        time.sleep(0.02)


def test_the_disk_files_are_encrypted_named_by_an_hmac_and_read_back_the_same(folder):
    names = K.Names()
    disk = K.Disk(folder, 1 << 30, names, Clock())
    canary = "QZXJ-canary-7741"
    toks = [ord(c) for c in canary] * 8 + list(range(300, 340))  # token ids that spell the canary, and others
    st = state_of(3, 1, length=len(toks))
    # the canary's text and its token ids in the tensors would show up in a file written in the clear: put both there
    raw = (canary.encode() + array.array("i", toks).tobytes() + array.array("q", toks).tobytes())
    flat = st["kinds"]["sliding_attention"]["data"][0].view(-1).view(torch.uint8)
    flat[: len(raw)] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    name = names.one("key:a", toks)
    assert disk.put(name, "key:a", len(toks), st, 3600)
    wait_written(disk, 1)
    files = os.listdir(folder)
    assert files == [name + ".kv"] and K.FILE.match(files[0])
    blob = open(os.path.join(folder, files[0]), "rb").read()
    assert blob.startswith(K.MAGIC) and len(blob) > st["bytes"]
    for needle in (canary.encode(), canary.encode("utf-16-le"), array.array("i", toks[:8]).tobytes(), array.array("q", toks[:8]).tobytes(),
                   array.array("i", toks[-8:]).tobytes(), raw[:16], b"sliding_attention", b"length"):
        assert needle not in blob
    assert disk.longest("key:a", toks + [1, 2]) == (name, len(toks)) and disk.longest("key:b", toks + [1, 2]) is None
    back = disk.read(name)
    assert same(back, st)
    # a file of another process (other keys): not read, deleted
    other = K.Disk(folder + "2", 1 << 30, names, Clock())
    os.makedirs(other.folder, exist_ok=True)
    with open(other.path(name), "wb") as f:
        f.write(blob)
    other._add(name, "key:a", len(toks), len(blob), 3600)
    assert other.read(name) is None and not os.path.exists(other.path(name)) and other.failed == 1


def test_the_folder_is_emptied_of_its_own_files_only_at_start(folder):
    os.makedirs(folder)
    ours, tmp = "ab" * 32 + ".kv", "cd" * 32 + ".kv.tmp"
    for name, data in ((ours, K.MAGIC + b"x" * 40), (tmp, b"partial"), ("ef" * 32 + ".kv", b"not ours"), ("notes.txt", b"keep"),
                       ("model.safetensors", b"keep")):
        with open(os.path.join(folder, name), "wb") as f:
            f.write(data)
    os.makedirs(os.path.join(folder, "sub"))
    with open(os.path.join(folder, "sub", "12" * 32 + ".kv"), "wb") as f:
        f.write(K.MAGIC)
    disk = K.Disk(folder, 1 << 30, K.Names(), Clock())
    assert (disk.removed, disk.others) == (2, 4)
    assert sorted(os.listdir(folder)) == sorted(["ef" * 32 + ".kv", "notes.txt", "model.safetensors", "sub"])
    assert os.listdir(os.path.join(folder, "sub")) == ["12" * 32 + ".kv"]


def test_files_go_at_their_time_and_past_the_budget(folder):
    clock = Clock()
    names = K.Names()
    st = state_of(2, 1)
    size = st["bytes"]
    disk = K.Disk(folder, int(2.5 * size), names, clock)
    toks = list(range(10, 90))
    for i, n in enumerate((20, 30, 40)):
        clock.t += 10
        disk.put(names.one("s", toks[:n]), "s", n, state_of(2, i), 3600)
        wait_written(disk, i + 1)
    assert sorted(v[1] for v in disk.index.values()) == [30, 40] and len(os.listdir(folder)) == 2  # the first went past the budget
    clock.t += 100
    disk.touch(names.one("s", toks[:30]))
    clock.t += 3505
    assert disk.sweep() == 1 and [v[1] for v in disk.index.values()] == [30]  # the 40 tokens' file: an hour since its last use
    clock.t += 100
    assert disk.sweep() == 1 and not disk.index and os.listdir(folder) == []
    disk.put(names.one("s", toks[:50]), "s", 50, state_of(2, 5), 3600)
    wait_written(disk, 4)
    disk.close()
    assert os.listdir(folder) == []


# where the marks fall in the rendered prompt (the real tokenizer and template)

needs_tok = pytest.mark.skipif(not os.path.exists(os.path.join(ckpt("orig"), "tokenizer.json")), reason="needs Gemma 4's tokenizer")


@pytest.fixture(scope="module")
def tok():
    import transformers
    from transformers import AutoTokenizer

    transformers.logging.set_verbosity_error()
    return AutoTokenizer.from_pretrained(ckpt("orig"))


class Inbox:
    """In the engine's place: the requests that came (with a Cache, or None: as before)."""

    def __init__(self, tok, cache=None, turns=False):
        self.stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
        self.reqs, self.canon = [], None
        self.eng = type("Eng", (), {"capacity": 8192, "rows": 32})()
        self.prefixes = S.Prefixes(0)
        self.cache = cache

    def submit(self, req) -> None:
        self.reqs.append(req)


def app_of(tok, on: bool, turns: bool = False, fields: str = "default") -> S.App:
    cache = K.Cache(1 << 30, turns=turns) if on else None
    app = S.App(Inbox(tok, cache), tok, False, 64, {"bits_per_weight": 3.0})
    app.fields = fields
    return app


DOC = "The engine reads prompts in wide passes. " * 160  # about 1,300 tokens


def ids(tok, messages, prompt=True) -> list:
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=prompt, enable_thinking=False)
    return list(out["input_ids"] if hasattr(out, "keys") else out)


@needs_tok
def test_a_parts_mark_is_where_the_part_ends_whatever_follows(tok):
    app = app_of(tok, True)
    cuts = []
    for q in ("What is it about?", "\n\nName three things it does.", " Another question."):
        body = {"messages": [{"role": "user", "content": [{"type": "text", "text": DOC, "cache_control": {"type": "ephemeral"}},
                                                          {"type": "text", "text": q}]}]}
        req = app.request(body)
        assert req.ignored == [] and len(req.points) == 1 and req.points[0][1:] == (900, "mark")
        n = req.points[0][0]
        assert n >= 1024 and req.full[:n] == ids(tok, [{"role": "user", "content": DOC}], False)[:n]
        cuts.append(n)
    assert len(set(cuts)) == 1
    # an hour's, on the message: its end (the turn's close included)
    body = {"messages": [{"role": "system", "content": DOC, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                         {"role": "user", "content": "Hi"}]}
    req = app.request(body)
    whole = ids(tok, body["messages"][:1], False)
    assert req.points == [(len(whole), 3600, "mark")] and req.full[: len(whole)] == whole
    # a field of cache_control that is not served: the mark is served, the field named (R1); --strict-fields refuses
    body["messages"][0]["cache_control"] = {"type": "ephemeral", "ttl": "1h", "extra": 1}
    req = app.request(body)
    assert req.points == [(len(whole), 3600, "mark")] and req.ignored == ["messages[0].cache_control.extra"]
    with pytest.raises(C.Refused) as e:
        app_of(tok, True, fields="strict").request(body)
    assert e.value.args[2] == "messages[0].cache_control.extra"


@needs_tok
def test_short_prefixes_odd_marks_and_a_fifth_are_named_and_strict_refuses(tok):
    app = app_of(tok, True)
    parts = [{"type": "text", "text": DOC, "cache_control": {"type": "ephemeral"}},
             {"type": "text", "text": DOC, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
             {"type": "text", "text": DOC, "cache_control": {"type": "ephemeral", "ttl": "2h"}},  # not a ttl served
             {"type": "text", "text": DOC, "cache_control": {"type": "persistent"}},
             {"type": "text", "text": DOC, "cache_control": {"type": "ephemeral"}},
             {"type": "text", "text": DOC, "cache_control": {"type": "ephemeral"}},
             {"type": "text", "text": "end", "cache_control": {"type": "ephemeral"}}]  # the fifth and sixth served ones: named
    body = {"messages": [{"role": "user", "content": "short", "cache_control": {"type": "ephemeral"}},  # under 1,024 tokens
                         {"role": "assistant", "content": "ok"}, {"role": "user", "content": parts}]}
    req = app.request(body)
    # (the short one counts among the four, as a breakpoint does at Anthropic's)
    assert req.ignored == ["messages[0].cache_control", "messages[2].content[2].cache_control", "messages[2].content[3].cache_control",
                           "messages[2].content[5].cache_control", "messages[2].content[6].cache_control"]
    assert [(t, k) for _, t, k in req.points] == [(900, "mark"), (3600, "mark"), (900, "mark")]
    assert sorted(n for n, _, _ in req.points) == [n for n, _, _ in req.points] and len({n for n, _, _ in req.points}) == 3
    strict = app_of(tok, True, fields="strict")
    with pytest.raises(C.Refused) as e:
        strict.request(body)
    assert e.value.args[0] == 400 and e.value.args[2] == "messages[2].content[2].cache_control"


@needs_tok
def test_without_the_option_cache_control_is_read_nowhere(tok):
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": DOC, "cache_control": {"type": "ephemeral", "ttl": "2h"}},
                                                      {"type": "text", "text": "Q?"}]}]}
    off = app_of(tok, False).request(body)
    on = app_of(tok, True).request(body)
    assert off.ignored == [] and off.points == [] and on.ignored == ["messages[0].content[0].cache_control"]
    assert off.full == on.full


@needs_tok
def test_the_boundary_before_the_last_user_message_with_cache_turns(tok):
    ms = [{"role": "user", "content": DOC}, {"role": "assistant", "content": "Read."}, {"role": "user", "content": "Its first word?"}]
    req = app_of(tok, True, turns=True).request({"messages": ms})
    before = ids(tok, ms[:2], False)
    assert req.points == [(len(before), 900, "turn")] and req.full[: len(before)] == before
    assert app_of(tok, True).request({"messages": ms}).points == []  # without --cache-turns
    short = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}, {"role": "user", "content": "again"}]
    assert app_of(tok, True, turns=True).request({"messages": short}).points == []  # under 1,024 tokens
    marked = [{"role": "user", "content": DOC, "cache_control": {"type": "ephemeral"}}, {"role": "user", "content": "Q?"}]
    got = app_of(tok, True, turns=True).request({"messages": marked}).points
    assert len(got) == 1 and got[0][2] == "mark"  # the same place: the mark's


@needs_tok
def test_the_header_names_whats_not_cached(tok):
    httpd = C.BoundedHTTPServer(("127.0.0.1", 0), S.Handler)
    httpd.app = app_of(tok, True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
        body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "short", "cache_control": {"type": "ephemeral"}}]}], "max_tokens": 4}
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        t = time.time()
        while not httpd.app.server.reqs:
            assert time.time() - t < 10
            time.sleep(0.01)
        req = httpd.app.server.reqs.pop()
        req.events.put(("done", {"finish_reason": "stop", "usage": {"prompt_tokens": len(req.full), "completion_tokens": 0, "total_tokens": len(req.full),
                                                                    "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}},
                                 "timings": {"total_ms": 1}}))
        resp = conn.getresponse()
        data = json.loads(resp.read())
        assert resp.status == 200 and resp.getheader("x-sclmd-ignored") == "messages[0].content[0].cache_control"
        assert data["usage"]["prompt_tokens_details"] == {"cached_tokens": 0, "cache_write_tokens": 0}
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_options_are_checked_before_the_model_loads(tmp_path):
    ap = S.parser()
    for argv in (["--cache-dir", str(tmp_path)], ["--cache-turns"], ["--prompt-cache", "--cache-ram-gib", "-1"],
                 ["--prompt-cache", "--cache-dir", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]):
        with pytest.raises(SystemExit):
            K.check(ap.parse_args(argv))
    K.check(ap.parse_args([]))
    K.check(ap.parse_args(["--prompt-cache", "--cache-dir", str(tmp_path / "c"), "--cache-turns"]))
    assert K.install(None, None, ap.parse_args([])) is None
