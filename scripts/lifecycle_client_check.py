# The official openai client against one of the server's placeholders (python -m scripts.lifecycle_canned PORT
# --starting S or --holder S, started just before): while the port answers 503 the client is told so, with Retry-After
# and the connection closed; once the server answers, the same client object (its pooled connection) gets the reply;
# and a client with the library's own retries (2, after Retry-After) gets it from its first try. It does not import the
# project:   uv run --offline --no-project --with openai python scripts/lifecycle_client_check.py PORT S [--old]
# --old: the server runs the placeholder as R3 first wrote it (scripts/lifecycle_canned.py --old); the check then
# expects what a kept connection got then: the 503 after the server is up.
import sys
import threading
import time

import openai
from openai import OpenAI

t0 = time.time()
port, s, old = sys.argv[1], float(sys.argv[2]), "--old" in sys.argv
print("openai", openai.__version__, "port", port, "old" if old else "")
ok = True
ask = {"model": "gemma-4-31b-it", "messages": [{"role": "user", "content": "hi"}]}
want = "Hello! You said: hi 안녕하세요."
url = f"http://127.0.0.1:{port}/v1"


def check(name: str, cond: bool, detail="") -> None:
    global ok
    ok = ok and bool(cond)
    print(("ok  " if cond else "FAIL"), name, detail, flush=True)


def reply(client) -> str:
    try:
        return client.chat.completions.create(**ask).choices[0].message.content
    except openai.APIStatusError as e:
        return f"{e.status_code}"
    except openai.APIConnectionError as e:
        return type(e).__name__


retried = {}
retrying = OpenAI(base_url=url, api_key="none", max_retries=2, timeout=30)  # the library's default retries
th = threading.Thread(target=lambda: retried.update(got=reply(retrying), t=time.time() - t0))
th.start()
client = OpenAI(base_url=url, api_key="none", max_retries=0, timeout=10)
try:
    client.chat.completions.create(**ask)
    check("503 while the placeholder holds the port", False, "a reply")
except openai.APIStatusError as e:
    close = e.response.headers.get("connection", "")
    check("503 while the placeholder holds the port", e.status_code == 503 and int(e.response.headers.get("retry-after", 0)) > 0,
          f"{e.status_code}, Retry-After {e.response.headers.get('retry-after')}")
    check("  Connection: close", (close == "close") != old, repr(close))
try:
    client.models.list()
    check("models.list: 503 too", False)
except openai.APIStatusError as e:
    check("models.list: 503 too", e.status_code == 503)
time.sleep(max(0.0, t0 + s + 1.5 - time.time()))  # the server is up now
for k in range(3):  # the same client: its pool holds the connection it used, if the placeholder left it open
    got = reply(client)
    check(f"the same client once the server is up, request {k + 1}", (got == want) != old, repr(got))
try:
    m = client.models.list().data[0].id
except openai.APIStatusError as e:
    m = e.status_code
check("  models.list", (m == "gemma-4-31b-it") != old, repr(m))
th.join(120)
# (the library's own retries got through with R3's first placeholder too: measured, scripts/lifecycle_canned.py --old)
check("a client with the library's retries, first sent while the port answered 503", retried.get("got") == want,
      f"{retried.get('got')!r} after {retried.get('t', 0):.1f} s")
print("ALL OK" if ok else "SOME FAILED", flush=True)
sys.exit(0 if ok else 1)
