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

print("ALL OK" if ok else "SOMETHING FAILED")
