# The official openai client against the server's HTTP side with a canned reply (python -m scripts.serve_canned PORT):
# what the library makes of each kind of answer. It does not import the project (the library need not be installed
# in its environment): uv run --no-project --with openai python scripts/openai_client_check.py PORT
import sys

import openai
from openai import OpenAI

client = OpenAI(base_url=f"http://127.0.0.1:{sys.argv[1]}/v1", api_key="none", max_retries=0, timeout=20)
print("openai", openai.__version__)
ok = True


def check(name: str, cond: bool, detail="") -> None:
    global ok
    ok = ok and bool(cond)
    print(("ok  " if cond else "FAIL"), name, detail)


models = client.models.list()
model = models.data[0].id
check("models.list", len(models.data) == 1, model)
check("models.retrieve", client.models.retrieve(model).id == model)

r = client.chat.completions.create(model=model, messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}])
check("a reply, not streamed", r.choices[0].message.content == "Hello! You said: hi 안녕하세요." and r.choices[0].finish_reason == "stop", repr(r.choices[0].message.content))
check("  usage", r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens and r.usage.prompt_tokens_details.cached_tokens == 0, str(r.usage.total_tokens))
check("  role", r.choices[0].message.role == "assistant")

r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": [{"type": "text", "text": "parts"}]}])
check("content given as parts", r.choices[0].message.content.startswith("Hello! You said: parts"))

r = client.chat.completions.create(model=model, messages=[{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "hi"}],
                                   max_completion_tokens=64, temperature=0.7, top_p=0.9, seed=3, stop=["안녕"])
check("a stop string, developer role, sampling options", r.choices[0].message.content == "Hello! You said: hi " and r.choices[0].finish_reason == "stop", repr(r.choices[0].message.content))

r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], reasoning_effort="high")
extra = getattr(r.choices[0].message, "reasoning_content", None) or (r.choices[0].message.model_extra or {}).get("reasoning_content")
check("thinking asked with reasoning_effort", r.choices[0].message.content == "Hello! You said: hi 안녕하세요." and extra == "The user greets me. I greet back.", repr(extra))

text, think, finish, usage, roles = "", "", None, None, []
for chunk in client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], stream=True,
                                            stream_options={"include_usage": True}, extra_body={"thinking": True}):
    if chunk.usage is not None:
        usage = chunk.usage
    for c in chunk.choices:
        if c.delta.role:
            roles.append(c.delta.role)
        text += c.delta.content or ""
        think += (c.delta.model_extra or {}).get("reasoning_content") or ""
        finish = c.finish_reason or finish
check("a stream", text == "Hello! You said: hi 안녕하세요." and finish == "stop" and roles == ["assistant"], repr(text))
check("  its thinking", think == "The user greets me. I greet back.", repr(think))
check("  its usage", usage is not None and usage.completion_tokens > 0)

with client.chat.completions.stream(model=model, messages=[{"role": "user", "content": "hi"}]) as s:
    final = s.get_final_completion()
check("the library's stream helper", final.choices[0].message.content == "Hello! You said: hi 안녕하세요.")

try:
    client.chat.completions.create(model=model, messages=[{"role": "user", "content": "fail"}])
    check("a failure in the middle, not streamed", False, "no error raised")
except openai.InternalServerError as e:
    check("a failure in the middle, not streamed", e.status_code == 500 and "RuntimeError" in str(e), type(e).__name__)

for said in ("fail", "fail first"):
    got = ""
    try:
        for chunk in client.chat.completions.create(model=model, messages=[{"role": "user", "content": said}], stream=True):
            for c in chunk.choices:
                got += c.delta.content or ""
        check(f"a stream of {said!r}", False, f"no error raised; text {got!r}")
    except openai.APIError as e:
        check(f"a stream of {said!r}", "RuntimeError" in str(e), f"{type(e).__name__} after {got!r}")

try:
    client.chat.completions.create(model=model, messages=[])
    check("a request without messages", False)
except openai.BadRequestError as e:
    check("a request without messages", e.status_code == 400, str(e)[:80])

try:
    client.chat.completions.create(model=model, messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}])
    check("an image", False)
except openai.BadRequestError as e:
    check("an image", "only text" in str(e), str(e)[:80])

raw = client.chat.completions.with_raw_response.create(model=model, messages=[{"role": "user", "content": "hi"}], presence_penalty=1.0,
                                                       extra_body={"stop": 5})
check("fields that are not served are named", raw.headers.get("x-sclmd-ignored") == "presence_penalty, stop"
      and raw.parse().choices[0].message.content == "Hello! You said: hi 안녕하세요.", repr(raw.headers.get("x-sclmd-ignored")))

try:
    client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], n=2)
    check("n: 2", False, "no error raised")
except openai.BadRequestError as e:
    check("n: 2", e.status_code == 400 and (e.body or {}).get("param") == "n", str(e)[:80])

if len(sys.argv) > 2:  # a second server, started with --max-body-mb 1 --max-connections 1
    import socket
    import time

    busy = OpenAI(base_url=f"http://127.0.0.1:{sys.argv[2]}/v1", api_key="none", max_retries=0, timeout=20)
    try:  # sent whole before the answer is read: the answer must not be lost to a reset
        busy.chat.completions.create(model=model, messages=[{"role": "user", "content": "a " * (3 << 20)}])
        check("a body over --max-body-mb 1 (6 MB)", False, "no error raised")
    except openai.APIStatusError as e:
        check("a body over --max-body-mb 1 (6 MB)", e.status_code == 413 and "--max-body-mb" in str(e), f"{type(e).__name__} {e.status_code}")
    except openai.APIConnectionError as e:
        check("a body over --max-body-mb 1 (6 MB)", False, f"{type(e).__name__}: {e}")
    time.sleep(0.5)  # that connection's thread lets its place go
    held = socket.create_connection(("127.0.0.1", int(sys.argv[2])))  # one connection held, then the library's requests
    time.sleep(0.3)
    outcomes = {}
    for _ in range(20):
        try:
            busy.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi " * 20000}])
            outcomes["200"] = outcomes.get("200", 0) + 1
        except openai.APIStatusError as e:
            outcomes[str(e.status_code)] = outcomes.get(str(e.status_code), 0) + 1
        except openai.APIConnectionError as e:
            outcomes[type(e).__name__] = outcomes.get(type(e).__name__, 0) + 1
    check("past --max-connections: 503 (20 requests)", outcomes == {"503": 20}, str(outcomes))
    held.close()
    time.sleep(0.3)
    check("  and served once a connection is free", busy.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}])
          .choices[0].message.content == "Hello! You said: hi 안녕하세요.")

# tool calls (R2): the canned server answers a request with tools by a call of the first one, page: the tool results so far + 1
import json  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "read_page", "description": "Reads a page of the report.",
                                           "parameters": {"type": "object", "properties": {"page": {"type": "integer"}, "note": {"type": "string"}},
                                                          "required": ["page"]}}}]
NOTE = 'say "hi", {x}\n서울'
ask = [{"role": "user", "content": "read the report"}]
r = client.chat.completions.create(model=model, messages=ask, tools=TOOLS)
m = r.choices[0].message
check("a tool call, not streamed", r.choices[0].finish_reason == "tool_calls" and m.content is None and len(m.tool_calls or []) == 1
      and m.tool_calls[0].type == "function" and m.tool_calls[0].function.name == "read_page"
      and json.loads(m.tool_calls[0].function.arguments) == {"page": 1, "note": NOTE}, repr(m.tool_calls))
step2 = ask + [m.model_dump(exclude_none=True), {"role": "tool", "tool_call_id": m.tool_calls[0].id, "content": "page one"}]
r = client.chat.completions.create(model=model, messages=step2, tools=TOOLS)
check("  the next step, with the library's own message and the tool's result",
      r.choices[0].message.tool_calls and json.loads(r.choices[0].message.tool_calls[0].function.arguments)["page"] == 2,
      repr(r.choices[0].message.tool_calls))
calls, finish = {}, None
for chunk in client.chat.completions.create(model=model, messages=ask, tools=TOOLS, stream=True):
    for c in chunk.choices:
        for tc in c.delta.tool_calls or []:
            got = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            got["id"] += tc.id or ""
            got["name"] += (tc.function.name or "") if tc.function else ""
            got["arguments"] += (tc.function.arguments or "") if tc.function else ""
        finish = c.finish_reason or finish
check("a tool call, streamed", finish == "tool_calls" and len(calls) == 1 and calls[0]["name"] == "read_page"
      and json.loads(calls[0]["arguments"]) == {"page": 1, "note": NOTE} and calls[0]["id"].startswith("call_"), repr(calls))
with client.chat.completions.stream(model=model, messages=ask, tools=TOOLS) as s:
    final = s.get_final_completion()
check("  the library's stream helper", final.choices[0].message.tool_calls[0].function.name == "read_page"
      and final.choices[0].finish_reason == "tool_calls", repr(final.choices[0].message.tool_calls))
r = client.chat.completions.create(model=model, messages=ask, tools=TOOLS, tool_choice="none")
check("tool_choice none: no tools declared, a text reply", r.choices[0].message.content.startswith("Hello!") and not r.choices[0].message.tool_calls)
check("the model card says tools", (client.models.retrieve(model).model_extra or {}).get("capabilities", {}).get("tools") is True)

print("ALL OK" if ok else "SOMETHING FAILED")
