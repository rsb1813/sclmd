# Tool calls (sclmd/tools.py and its hooks in scripts/serve.py) without a model or a GPU: the real App and Handler with
# Gemma 4's own tokenizer and chat template, and the test in the engine's place, putting the token ids a model would
# write. What a client sends (tools, tool_choice, role tool, assistant tool_calls) and gets (tool_calls, streamed or
# not), and that a conversation's later form is a prefix of the agent's next step (only the tool's result is read).
import http.client
import json
import os
import re
import threading
import time

import pytest

from sclmd import ckpt
from sclmd import tools as T
from scripts import serve as S
from scripts import serve_contract as C

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ckpt("orig"), "tokenizer.json")), reason="needs Gemma 4's tokenizer")

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                                                           "required": ["city"]}}}
TIME = {"type": "function", "function": {"name": "get_time", "description": "Local time in a zone.",
                                         "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}, "required": ["zone"]}}}
ASK = [{"role": "user", "content": "Weather in Seoul?"}]


@pytest.fixture(scope="module")
def tok():
    import transformers
    from transformers import AutoTokenizer

    transformers.logging.set_verbosity_error()
    return AutoTokenizer.from_pretrained(ckpt("orig"))


class Inbox:
    """In the engine's place: the requests that came, for the test to answer."""

    def __init__(self, tok):
        self.stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
        self.reqs, self.canon = [], None
        self.eng = type("Eng", (), {"capacity": 8192, "rows": 32})()
        self.prefixes = S.Prefixes(0)

    def submit(self, req) -> None:
        self.reqs.append(req)

    def next(self):
        t = time.time()
        while not self.reqs:
            assert time.time() - t < 10, "the request did not come"
            time.sleep(0.01)
        return self.reqs.pop(0)


@pytest.fixture()
def served(tok):
    httpd = C.BoundedHTTPServer(("127.0.0.1", 0), S.Handler)
    httpd.app = S.App(Inbox(tok), tok, False, 256, {"bits_per_weight": 3.0})
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def enc(tok, s: str) -> list:
    return tok.encode(s, add_special_tokens=False)


def call(name: str, body: str) -> str:
    """A call as the model writes it."""
    return f"<|tool_call>call:{name}{{{body}}}<tool_call|>"


def q(s: str) -> str:
    return '<|"|>' + s + '<|"|>'


def answer(httpd, body: dict, tokens: list, finish: str = "stop") -> tuple:
    """Sends a request and answers it with `tokens` -> (status, the parsed body or a stream's events, the request)."""
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=20)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    req = httpd.app.server.next()
    for token in tokens:
        req.events.put(("token", token))
    usage = {"prompt_tokens": len(req.full), "completion_tokens": len(tokens), "total_tokens": len(req.full) + len(tokens),
             "prompt_tokens_details": {"cached_tokens": 0}}
    req.events.put(("done", {"finish_reason": finish, "usage": usage, "timings": {"total_ms": 5}, "ended": finish}))
    resp = conn.getresponse()
    data = resp.read()
    got = json.loads(data) if not body.get("stream") else events_of(data)
    return resp.status, got, req


def refused(httpd, body: dict) -> tuple:
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=20)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read())


def events_of(body: bytes) -> list:
    out = []
    for line in body.decode().split("\n\n"):
        if line.startswith("data: "):
            out.append(line[6:] if line[6:] == "[DONE]" else json.loads(line[6:]))
    return out


def streamed(events: list) -> tuple:
    """A stream's content, its tool calls (by index) and its finish_reason."""
    content, calls, finish = "", {}, None
    for e in events[:-1]:
        for c in e.get("choices", []):
            content += c["delta"].get("content") or ""
            for tc in c["delta"].get("tool_calls") or []:
                calls[tc["index"]] = tc
            finish = c.get("finish_reason") or finish
    return content, [calls[k] for k in sorted(calls)], finish


# -- the parser ------------------------------------------------------------------------------------------------------

HOSTILE = ['say "hi"', "{braces}, [lists]: and, commas", "line one\nline two", "서울특별시 강남구", "  spaces  ", "", "back\\slash", "100%"]


def test_calls_written_by_the_template_are_read_back_on_token_ids(tok):
    ids = T.Ids(tok)
    for k, s in enumerate(HOSTILE):
        args = {"text": s, "n": k, "x": -1.5, "ok": k % 2 == 0, "none": None, "list": [s, 1, {"in": s}], "obj": {"deep": {"s": s}}}
        msg = {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "do_it", "arguments": args}}]}
        rendered = tok.apply_chat_template(ASK + [msg], tokenize=True, add_generation_prompt=False)
        rendered = list(rendered["input_ids"] if hasattr(rendered, "keys") else rendered)
        block = rendered[rendered.index(ids.call) + 1: rendered.index(ids.end)]
        assert T.parse_call(tok, block, ids.quote, lenient=False) == ("do_it", args)


def test_a_block_that_is_not_a_call_is_malformed(tok):
    ids = T.Ids(tok)
    for bad in ["call:f{a:" + q("x"), "call:f{a " + q("x") + "}", "f{a:1}", "call:f{a:1}}", "call:f{a:" + q("x")[:-5] + "}"]:
        with pytest.raises(T.Malformed):
            T.parse_call(tok, enc(tok, bad), ids.quote, lenient=False)
    assert T.parse_call(tok, enc(tok, "call:f{unit:celsius}"), ids.quote) == ("f", {"unit": "celsius"})  # a bare word: lenient


# -- the request -----------------------------------------------------------------------------------------------------

def test_tools_are_declared_and_a_request_without_them_is_written_as_before(served, tok):
    app = served.app
    req = app.request({"messages": ASK, "tools": [WEATHER, TIME]})
    text = tok.decode(req.full)
    assert "<|tool>declaration:get_weather{" in text and "<|tool>declaration:get_time{" in text and req.tools and req.ignored == []
    plain = app.request({"messages": ASK})
    want = tok.apply_chat_template(ASK, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    assert plain.full == list(want["input_ids"] if hasattr(want, "keys") else want) and plain.tools is None and not plain.opened
    before = tok.apply_chat_template(ASK, tokenize=True, add_generation_prompt=False, enable_thinking=False)
    assert plain.start == len(before["input_ids"] if hasattr(before, "keys") else before) + len(app.turn) and plain.stop_ids is None
    assert app.request({"messages": ASK, "tools": [WEATHER], "tool_choice": "none"}).tools is None  # none: nothing declared
    named = app.request({"messages": ASK, "tools": [WEATHER, TIME], "tool_choice": {"type": "function", "function": {"name": "get_time"}}})
    assert [t["function"]["name"] for t in named.tools] == ["get_time"] and named.ignored == ["tool_choice"]  # declared alone, not forced
    assert app.request({"messages": ASK, "tools": [WEATHER], "tool_choice": "required"}).ignored == ["tool_choice"]
    other = app.request({"messages": ASK, "tools": [WEATHER, {"type": "web_search"}]})
    assert len(other.tools) == 1 and other.ignored == ["tools[1].type"]
    strict = app.request({"messages": ASK, "tools": [{"type": "function", "function": {**WEATHER["function"], "strict": True}}]})
    assert strict.tools and strict.ignored == ["tools[0].function.strict"]  # declared, its schema not enforced
    one = app.request({"messages": ASK, "tools": [WEATHER], "parallel_tool_calls": False})
    assert one.stop_ids == tuple(app.server.stop) + (T.Ids(tok).end,)


def test_tool_messages_and_calls_in_the_conversation_are_accepted(served, tok):
    app = served.app
    history = ASK + [{"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                                                                             "function": {"name": "get_weather", "arguments": '{"city": "Seoul"}'}}]},
                     {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": '{"temp": 3}'}]}]
    for think in (False, True):
        req = app.request({"messages": history, "tools": [WEATHER], "thinking": think})
        text = tok.decode(req.full)
        assert text.endswith('response:get_weather{value:<|"|>{"temp": 3}<|"|>}<tool_response|>' + ("<|channel>thought\n" if think else ""))
        assert req.start == len(req.full) - (3 if think else 0) and req.opened == think  # the reply goes on in the model's turn
    for bad, where in ((["not json"], "not JSON"), (["[1, 2]"], "a JSON object")):  # a call that cannot be written: left out, named
        h = json.loads(json.dumps(history))
        h[1]["tool_calls"][0]["function"]["arguments"] = bad[0]
        req = app.request({"messages": h, "tools": [WEATHER]})
        assert req.ignored == ["messages[1].tool_calls[0].function.arguments", "messages[2]"] and "<|tool_call>" not in tok.decode(req.full)
        app.fields = "strict"
        try:
            status, got = refused(served, {"messages": h, "tools": [WEATHER]})
            assert status == 400 and where in got["error"]["message"] and got["error"]["param"] == "messages"
        finally:
            app.fields = "default"
    status, got = refused(served, {"messages": ASK + [{"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})
    assert status == 400


def test_a_tool_result_given_as_an_object_is_written_as_its_json_text(served, tok):
    history = ASK + [{"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                                                                             "function": {"name": "get_weather", "arguments": '{"city": "Seoul"}'}}]},
                     {"role": "tool", "tool_call_id": "c1", "content": {"temp": 3, "sky": "맑음"}}]
    for content, text in (({"temp": 3, "sky": "맑음"}, '{"temp": 3, "sky": "맑음"}'), (7, "7"), (None, ""),
                          ([{"temp": 3}, {"temp": 4}], '[{"temp": 3}, {"temp": 4}]'), (["a", 1], '["a", 1]')):
        history[2]["content"] = content
        status, got, req = answer(served, {"messages": history, "tools": [WEATHER]}, enc(tok, "It is 3 C.<turn|>"))
        assert status == 200 and got["choices"][0]["message"]["content"] == "It is 3 C." and req.ignored == []
        assert tok.decode(req.full).endswith('response:get_weather{value:<|"|>' + text + '<|"|>}<tool_response|>')


def test_a_tool_result_that_follows_no_call_is_named(served, tok):
    """The template writes nothing for it: by default the request is served and the message named (strict: 400)."""
    app = served.app
    orphan = ASK + [{"role": "tool", "tool_call_id": "c9", "content": "3 C"}]
    req = app.request({"messages": orphan, "tools": [WEATHER]})
    assert req.ignored == ["messages[1]"] and "3 C" not in tok.decode(req.full) and not req.opened
    later = ASK + [{"role": "assistant", "content": "Hi."}, {"role": "tool", "tool_call_id": "c9", "content": "3 C"}, {"role": "user", "content": "And?"}]
    assert app.request({"messages": later, "tools": [WEATHER]}).ignored == ["messages[2]"]
    paired = ASK + [{"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": "{}"}},
                                                                          {"id": "c2", "function": {"name": "get_time", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "3 C"}, {"role": "tool", "tool_call_id": "c2", "content": "09:00"}]
    assert app.request({"messages": paired, "tools": [WEATHER, TIME]}).ignored == []  # results that follow their calls
    app.fields = "strict"
    try:
        status, got = refused(served, {"messages": orphan, "tools": [WEATHER]})
        assert status == 400 and got["error"]["message"].startswith("messages[1]:")
    finally:
        app.fields = "default"


BAD_TOOLS = [("x", ["tools"], 0), ([5, WEATHER], ["tools[0]"], 1), ([{"type": "function", "function": {"description": "no name"}}, WEATHER], ["tools[0].function.name"], 1),
             ([{"type": "function", "function": {"name": ""}}], ["tools[0].function.name"], 0),
             ([{"type": "function", "function": {"name": "f", "parameters": "{}"}}], ["tools[0].function.parameters"], 1),
             ([WEATHER, {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": "x"}}}], ["tools[1]"], 1),
             ([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "required": 5}}}, TIME], ["tools[0]"], 1)]


@pytest.mark.parametrize("tools, names, declared", BAD_TOOLS)
def test_a_tool_that_cannot_be_declared_is_left_out_and_named(served, tok, tools, names, declared):
    """Before tool calls were served, every tool was ignored and the request served: one that cannot be declared still
    is (left out, named in x-sclmd-ignored); --strict-fields refuses it."""
    app = served.app
    status, got, req = answer(served, {"messages": ASK, "tools": tools}, enc(tok, "It is sunny.<turn|>"))
    assert status == 200 and req.ignored == names and len(req.tools or []) == declared
    assert tok.decode(req.full).count("<|tool>declaration:") == declared
    app.fields = "strict"
    try:
        status, got = refused(served, {"messages": ASK, "tools": tools})
        assert status == 400 and got["error"]["param"] == "tools"
    finally:
        app.fields = "default"


def test_what_the_template_cannot_write_or_the_context_cannot_hold_is_left_out_as_before(served, tok):
    """Before tool calls were served, a request with tools (or with calls in its history) was served without them:
    calls the template cannot write (nested past its depth), tools nested past Python's, and tools that leave no room
    in the context are left out and named, and the request served; strict: 400."""
    app = served.app
    deep = 1
    for _ in range(300):
        deep = {"k": deep}
    history = ASK + [{"role": "assistant", "content": "ok", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps(deep)}}]},
                     {"role": "user", "content": "go on"}]
    plain = ASK + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go on"}]
    for tools in (None, [WEATHER]):
        req = app.request({"messages": history, **({"tools": tools} if tools else {})})
        assert req.ignored == ["messages[1].tool_calls"] and "<|tool_call>" not in tok.decode(req.full)
        assert req.full == app.request({"messages": plain, **({"tools": tools} if tools else {})}).full  # as master wrote it
    schema = {"type": "string"}
    for _ in range(600):
        schema = {"type": "object", "properties": {"a": schema}}
    req = app.request({"messages": ASK, "tools": [WEATHER, {"type": "function", "function": {"name": "f", "parameters": schema}}]})
    assert req.ignored == ["tools[1]"] and len(req.tools) == 1
    many = [{"type": "function", "function": {"name": f"f{k}", "description": "Does thing number %d." % k,
                                              "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}} for k in range(400)]
    req = app.request({"messages": ASK, "tools": many})
    assert req.tools is None and req.ignored == ["tools"] and len(req.full) < 100
    app.fields = "strict"
    try:
        assert refused(served, {"messages": history})[0] == 400
        assert refused(served, {"messages": ASK, "tools": many})[0] == 400
    finally:
        app.fields = "default"


def test_the_names_header_is_bounded(served, tok):
    status, got, req = answer(served, {"messages": ASK, "tools": [5] * 6000}, enc(tok, "Hi.<turn|>"))
    assert status == 200 and len(req.ignored) == 6000
    assert C.names_of(req.ignored).endswith(", and 5960 more") and len(C.names_of(req.ignored)) < 1000


def test_a_tool_choice_that_names_no_tool_given_is_as_auto(served):
    app = served.app
    for choice in ({"type": "function", "function": {"name": "nope"}}, {"type": "function"}, "any", 5):
        req = app.request({"messages": ASK, "tools": [WEATHER, TIME], "tool_choice": choice})
        assert len(req.tools) == 2 and req.ignored == ["tool_choice"]


def test_a_stand_in_without_a_prefix_store_serves_tools(tok):
    """master 6bd9bbf: stand-ins for the engine (serve_canned, tests) have no `prefixes`."""
    class Bare:
        stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
        canon = None
        eng = type("Eng", (), {"capacity": 8192, "rows": 32})()

    app = S.App(Bare(), tok, False, 256, {"bits_per_weight": 3.0})
    assert app.request({"messages": ASK}).tools is None
    assert app.request({"messages": ASK, "tools": [WEATHER]}).tools


def test_schemas_are_rewritten_into_the_templates_forms(tok):
    """A pydantic schema (a nested model by $ref, optional fields by anyOf with null or a type list) is declared with
    its nested properties and nullable; a plain schema is declared as it is."""
    plain = WEATHER["function"]["parameters"]
    assert T.schema_of(plain) is plain
    pyd = {"$defs": {"Addr": {"properties": {"city": {"type": "string"}, "zip": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}},
                              "required": ["city"], "type": "object"},
                     "Node": {"properties": {"name": {"type": "string"}, "child": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]}}, "type": "object"}},
           "properties": {"to": {"$ref": "#/$defs/Addr", "description": "where"}, "note": {"type": ["string", "null"]},
                          "tags": {"type": "array", "items": {"$ref": "#/$defs/Addr"}}, "tree": {"$ref": "#/$defs/Node"},
                          "both": {"allOf": [{"$ref": "#/$defs/Addr"}]}},
           "required": ["to"], "type": "object"}
    got = T.schema_of(pyd)
    assert "$defs" in pyd and "$defs" not in got and "$ref" not in json.dumps(got) and "anyOf" not in json.dumps(got)
    assert got["properties"]["to"]["description"] == "where" and got["properties"]["to"]["properties"]["zip"] == {"type": "string", "default": None, "nullable": True}
    assert got["properties"]["note"] == {"type": "string", "nullable": True} and got["properties"]["both"]["required"] == ["city"]
    assert got["properties"]["tree"]["properties"]["child"] == {"type": "object", "nullable": True}  # a model inside itself
    tools, ignored, _ = T.tools_of({"tools": [{"type": "function", "function": {"name": "send", "description": "Sends.", "parameters": pyd}}]})
    text = tok.apply_chat_template(ASK, tools=tools, tokenize=False, add_generation_prompt=True)
    assert "to:{description:<|\"|>where<|\"|>,properties:{city:{type:<|\"|>STRING<|\"|>},zip:{nullable:true,type:<|\"|>STRING<|\"|>}}" in text
    assert "note:{nullable:true,type:<|\"|>STRING<|\"|>}" in text and "NULL" not in text and ignored == []


@pytest.mark.parametrize("stream", [False, True])
def test_a_stop_string_before_a_call_cuts_the_call_however_the_tokens_come(served, tok, stream):
    """The answer a stop string cuts keeps the calls that had closed before the stop string was whole, as when the
    tokens come one at a time: all at once or one by one, the client gets the same (and the conversation is kept so)."""
    tokens = enc(tok, "Let me look. " + call("get_weather", "city:" + q("Seoul")) + "<|tool_response>")
    before = enc(tok, "Seoul first: " + call("get_weather", "city:" + q("Seoul")) + " then look" + call("get_time", "zone:" + q("x")) + "<|tool_response>")
    for toks, stop, names, text in ((tokens, ["look"], [], "Let me "), (before, ["look"], ["get_weather"], "Seoul first:  then ")):
        seen = []
        for gap in (0.0, 0.02):
            conn = http.client.HTTPConnection("127.0.0.1", served.server_address[1], timeout=20)
            conn.request("POST", "/v1/chat/completions", json.dumps({"messages": ASK, "tools": [WEATHER, TIME], "stop": stop, "stream": stream}),
                         {"Content-Type": "application/json"})
            req = served.app.server.next()
            for t in toks:
                if gap and req.cancelled:
                    break  # the engine stops a cancelled job at its next pass
                req.events.put(("token", t))
                time.sleep(gap)
            req.events.put(("done", {"finish_reason": "stop", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                                     "timings": {"total_ms": 1}, "ended": "stop"}))
            resp = conn.getresponse()
            data = resp.read()
            if stream:
                content, calls, finish = streamed(events_of(data))
            else:
                m = json.loads(data)["choices"][0]
                content, calls, finish = m["message"]["content"] or "", m["message"].get("tool_calls") or [], m["finish_reason"]
            seen.append((content, [c["function"]["name"] for c in calls], finish, [c["function"]["name"] for c in req.stop_calls], req.stop_text))
        assert seen[0] == seen[1] == (text, names, "tool_calls" if names else "stop", names, text), seen
    # the engine's thread, when the reply ended before the HTTP thread saw the stop string: the same calls
    req = served.app.request({"messages": ASK, "tools": [WEATHER, TIME]})
    req.stops = ["look"]
    later = tok.decode(served.app.canonical(req, before))
    assert later.count("<|tool_call>") == 1 and "call:get_weather" in later


def test_a_stop_string_in_a_block_the_end_left_open_cuts_the_answer(served, tok):
    """A reply that ends (here at its length) inside a call block that does not parse: the block's text goes to the
    content, cut at a stop string in it as the kept conversation is."""
    tokens = enc(tok, "Hi " + "<|tool_call>call:get_weather{city:" + q("look and more"))
    for stream in (False, True):
        status, got, req = answer(served, {"messages": ASK, "tools": [WEATHER], "stop": ["look"], "stream": stream}, tokens, finish="length")
        if not stream:
            content, finish = got["choices"][0]["message"]["content"], got["choices"][0]["finish_reason"]
        else:
            content, _, finish = streamed(got)
        assert content.startswith("Hi ") and "look" not in content and finish == "stop", (content, finish)
        assert "look" not in tok.decode(served.app.canonical(req, tokens))  # the kept conversation is cut there too


def test_text_beside_a_call_reopens_a_turn_the_template_closed(served, tok):
    history = ASK + [{"role": "assistant", "content": "Let me check.", "tool_calls": [{"id": "c1", "type": "function",
                                                                                     "function": {"name": "get_weather", "arguments": {"city": "Seoul"}}}]},
                     {"role": "tool", "tool_call_id": "c1", "content": "3 C"}]
    req = served.app.request({"messages": history, "tools": [WEATHER]})
    assert tok.decode(req.full).endswith("Let me check.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>") and not req.opened
    assert tok.decode(req.full[: req.start]).endswith("Let me check.")  # its start: before the close


# -- the reply -------------------------------------------------------------------------------------------------------

def test_a_call_is_answered_as_tool_calls_streamed_or_not(served, tok):
    tokens = enc(tok, call("get_weather", "city:" + q("Seoul") + ",unit:" + q("c")) + "<|tool_response>")
    for stream in (False, True):
        status, got, req = answer(served, {"messages": ASK, "tools": [WEATHER], "stream": stream}, tokens)
        assert status == 200
        if not stream:
            msg, finish = got["choices"][0]["message"], got["choices"][0]["finish_reason"]
            content, calls = msg["content"], msg["tool_calls"]
        else:
            assert got[-1] == "[DONE]"
            content, calls, finish = streamed(got)
            content = content or None
            assert calls[0]["index"] == 0
        assert content is None and finish == "tool_calls" and len(calls) == 1
        c = calls[0]
        assert re.fullmatch(r"call_[0-9a-f]{24}", c["id"]) and c["type"] == "function" and c["function"]["name"] == "get_weather"
        assert json.loads(c["function"]["arguments"]) == {"city": "Seoul", "unit": "c"}


def test_parallel_calls_with_hostile_strings_and_text_before_them(served, tok):
    s1, s2 = 'he said "{x}, [y]:"\nnext line', "강남역 근처 일식집"
    tokens = enc(tok, "Checking both." + call("get_weather", "city:" + q(s1)) + call("get_time", "zone:" + q(s2)) + "<|tool_response>")
    for stream in (False, True):
        status, got, req = answer(served, {"messages": ASK, "tools": [WEATHER, TIME], "stream": stream}, tokens)
        if not stream:
            msg = got["choices"][0]["message"]
            content, calls, finish = msg["content"], msg["tool_calls"], got["choices"][0]["finish_reason"]
        else:
            content, calls, finish = streamed(got)
            assert [c["index"] for c in calls] == [0, 1]
        assert content == "Checking both." and finish == "tool_calls"
        assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == [("get_weather", {"city": s1}), ("get_time", {"zone": s2})]
        assert calls[0]["id"] != calls[1]["id"]


def test_a_malformed_block_stays_in_the_content(served, tok):
    tokens = enc(tok, "Sure. " + "<|tool_call>call:get_weather{city " + q("Seoul") + "}<tool_call|>" + "<turn|>")
    for stream in (False, True):
        status, got, req = answer(served, {"messages": ASK, "tools": [WEATHER], "stream": stream}, tokens)
        if not stream:
            msg = got["choices"][0]["message"]
            content, finish, calls = msg["content"], got["choices"][0]["finish_reason"], msg.get("tool_calls")
        else:
            content, calls, finish = streamed(got)
        assert content.startswith("Sure. <|tool_call>call:get_weather{city") and finish == "stop" and not calls


def test_a_text_reply_with_tools_is_as_before(served, tok):
    tokens = enc(tok, "It is sunny in Seoul.<turn|>")
    status, got, req = answer(served, {"messages": ASK, "tools": [WEATHER]}, tokens)
    msg = got["choices"][0]["message"]
    assert msg == {"role": "assistant", "content": "It is sunny in Seoul."} and got["choices"][0]["finish_reason"] == "stop"
    status, got, req = answer(served, {"messages": ASK}, tokens)  # no tools: the answer has no tool_calls at all
    assert got["choices"][0]["message"] == {"role": "assistant", "content": "It is sunny in Seoul."}


def test_a_thought_the_prompt_opened_goes_to_reasoning_content(served, tok):
    history = ASK + [{"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                                                                             "function": {"name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"}}]},
                     {"role": "tool", "tool_call_id": "c1", "content": "3 C"}]
    tokens = enc(tok, "It is cold.<channel|>It is 3 C in Seoul.<turn|>")
    for stream in (False, True):
        status, got, req = answer(served, {"messages": history, "tools": [WEATHER], "thinking": True, "stream": stream,
                                           "stream_options": {"include_usage": True}}, tokens)
        assert req.opened
        if not stream:
            msg = got["choices"][0]["message"]
            assert msg["reasoning_content"] == "It is cold." and msg["content"] == "It is 3 C in Seoul." and "tool_calls" not in msg
            assert got["usage"]["completion_tokens_details"]["reasoning_tokens"] == len(enc(tok, "It is cold.")) + 1  # and <channel|>
        else:
            assert "".join(e["choices"][0]["delta"].get("reasoning_content", "") for e in got[:-2] if e.get("choices")) == "It is cold."


def test_the_model_card_says_tools(served):
    assert served.app.card()["capabilities"]["tools"] is True


def test_the_tools_declarations_are_in_the_shared_part(served, tok):
    """--prefix-gib (B4): the part kept for other requests holds the declarations, with a system message or without one."""
    app = served.app
    app.server.prefixes = S.Prefixes(1 << 30)
    try:
        for system in ([], [{"role": "system", "content": "Use the tools."}]):
            req = app.request({"messages": system + ASK, "tools": [WEATHER, TIME]})
            part = tok.decode(req.full[: req.cut])
            assert part.endswith("<tool|><turn|>\n") and part.count("<|tool>declaration:") == 2 and "Weather in Seoul" not in part
            assert ("Use the tools." in part) == bool(system)
        assert app.request({"messages": ASK, "tools": [WEATHER], "sclmd_prefix_messages": 0}).cut == 0  # the client's mark: nothing shared
    finally:
        app.server.prefixes = S.Prefixes(0)


def test_strict_fields_accept_tools_and_refuse_a_tool_choice_that_is_not_enforced(served):
    served.app.fields = "strict"
    req = served.app.request({"messages": ASK, "tools": [WEATHER], "tool_choice": "auto"})
    assert req.tools and req.ignored == []
    status, got = refused(served, {"messages": ASK, "tools": [WEATHER], "tool_choice": "required"})
    assert status == 400 and got["error"]["param"] == "tool_choice"


# -- the agent loop: the later form is what the next step begins with --------------------------------------------------

@pytest.mark.parametrize("think", [False, True])
def test_each_steps_later_form_is_a_prefix_of_the_next_request(served, tok, think):
    """Three steps through the real App: the reply the client gets, sent back as an OpenAI client does (arguments as
    JSON text, its ids, the tools' results), begins with canonical()'s later form, and the turn start is kept."""
    app = served.app
    messages = [{"role": "system", "content": "Use the tools."}] + ASK
    writes = [call("get_weather", "city:" + q("Seoul")) + "<|tool_response>",
              call("get_time", "zone:" + q("Asia/Seoul")) + call("get_weather", "city:" + q('Busan "port"') + ",unit:" + q("c")) + "<|tool_response>",
              "Seoul: 3 C, 09:00. 부산: 5 C.<turn|>"]
    for step, write in enumerate(writes):
        body = {"messages": messages, "tools": [WEATHER, TIME], "thinking": think}
        thought = ("" if step == 0 or not think else "") + ("<|channel>thought\nI call it.<channel|>" if think and step == 0 else
                                                        ("Now the next.<channel|>" if think else ""))
        status, got, req = answer(served, body, enc(tok, thought + write))
        later = app.canonical(req, enc(tok, thought + write))
        assert later[: req.start] == req.full[: req.start] and req.start < len(later)  # the server reads the later form in
        msg = got["choices"][0]["message"]
        sent = {"role": "assistant", "content": msg["content"]}
        if msg.get("tool_calls"):
            sent["tool_calls"] = msg["tool_calls"]
        if msg.get("reasoning_content"):
            sent["reasoning_content"] = msg["reasoning_content"]  # sent back: the server leaves it out, as before tools
        messages = messages + [sent]
        if msg.get("tool_calls"):
            messages += [{"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"result": k})} for k, c in enumerate(msg["tool_calls"])]
        else:
            messages += [{"role": "user", "content": "Thanks!"}]
        nxt = app.request({"messages": messages, "tools": [WEATHER, TIME], "thinking": think})
        assert nxt.full[: len(later)] == later, f"step {step}: the later form is not where the next request begins"
