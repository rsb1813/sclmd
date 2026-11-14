# R2's kill test, first half (CPU, the tokenizer only): do tool calls keep a conversation's state through an agent loop?
# 1. Thirty synthetic agent loops (thinking on and off, one to three calls a step, text beside the calls or not, the
#    thinking sent back or not, the tool message named or not, a user's next question after the final answer). At each
#    step the server's side is played with sclmd/tools.py, the code scripts/serve.py calls: the prompt and where the
#    reply begins (the turn check), a reply as the model writes it (thought, text, calls in its own format, then
#    <|tool_response>), the calls read back on token ids, the later form canonical() writes; then the client's next
#    request, as an OpenAI client builds it from the answer (arguments as JSON text, the ids it got). Checked on token
#    ids: the later form is a prefix of the next request (the step reads only the tool results), and the turn start is
#    where the sequence goes back to (kept.base).
# 2. Two hundred calls with hostile strings rendered by the template and read back by the parser.
#
#   python -m scripts.tool_prefix_check [--out logs/exp/R2-tool-calls/prefix_check.json]
import argparse
import json
import os
import random

from sclmd import ROOT, ckpt
from sclmd import tools as T

TOOLS = {
    "weather": [
        {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                                          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                                                                                          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                                                         "required": ["city"]}}},
        {"type": "function", "function": {"name": "get_time", "description": "Local time in a time zone.",
                                          "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}, "required": ["zone"]}}}],
    "files": [
        {"type": "function", "function": {"name": "read_file", "description": "Reads a text file.",
                                          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"},
                                                                                          "lines": {"type": "integer"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "search", "description": "Searches the files for a pattern.",
                                          "parameters": {"type": "object", "properties": {"pattern": {"type": "string"},
                                                                                          "globs": {"type": "array", "items": {"type": "string"}},
                                                                                          "case": {"type": "boolean"}}, "required": ["pattern"]}}},
        {"type": "function", "function": {"name": "write_file", "description": "Writes a text file.",
                                          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
                                                         "required": ["path", "text"]}}}],
    "shop": [
        {"type": "function", "function": {"name": "find_product", "description": "상품을 찾는다.",
                                          "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "검색어"},
                                                                                          "max_price": {"type": "number"},
                                                                                          "filters": {"type": "object", "properties": {"color": {"type": "string"}, "in_stock": {"type": "boolean"}}}},
                                                         "required": ["query"]}}},
        {"type": "function", "function": {"name": "add_to_cart", "description": "Adds a product to the cart.",
                                          "parameters": {"type": "object", "properties": {"product_id": {"type": "string"}, "quantity": {"type": "integer"}},
                                                         "required": ["product_id", "quantity"]}}}],
}
HOSTILE = ['say "hi"', "it's", "{braces}", "[1, 2]", "a:b,c", "line one\nline two", "tab\there", "back\\slash", "서울특별시 강남구",
           "日本語のテキスト", "emoji 🙂", "  spaces  ", "", '{"json": [1, {"x": null}]}', "trailing\n", "C:\\Users\\x\\file.txt",
           "<b>html</b>", "100%", "null", "true", "3.5", "a\r\nb", "질문: \"무엇?\"", "end}"]
SPECIAL = ['quote <|"|> inside', "a <tool_call|> b", "x <|tool_response> y", "<|tool_call>call:f{}"]  # what the format cannot carry


def ids_of(x):
    return list(x["input_ids"] if hasattr(x, "keys") else x)


class Side:
    """The server's side of one request, as scripts/serve.py does it with sclmd/tools.py."""

    def __init__(self, tok):
        self.tok = tok
        self.turn = tok.encode("<|turn>model\n", add_special_tokens=False)
        self.thought = tok.encode("<|channel>thought\n", add_special_tokens=False)
        self.empty = tok.encode("<|channel>thought\n<channel|>", add_special_tokens=False)
        self.closed = tok.encode("<turn|>\n", add_special_tokens=False)
        self.ids = T.Ids(tok)
        self.opening, self.closing = tok.convert_tokens_to_ids("<|channel>"), tok.convert_tokens_to_ids("<channel|>")
        self.stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), self.ids.response)

    def render(self, messages, tools, think, prompt):
        kw = {"tools": tools} if tools else {}
        return ids_of(self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=prompt, enable_thinking=think, **kw))

    def request(self, body, think, keep_reasoning=False):
        messages = T.messages_of(body["messages"])[0]
        if keep_reasoning:  # not what the server does (it leaves the thinking a client sends back out, as before tools)
            for m, raw in zip(messages, body["messages"]):
                if raw.get("reasoning_content"):
                    m["reasoning_content"] = raw["reasoning_content"]
        tools = T.tools_of(body)[0]
        before, full = self.render(messages, tools, think, False), self.render(messages, tools, think, True)
        suffix = full[len(before):]
        if full[: len(before)] == before and suffix[: len(self.turn)] == self.turn:
            start, opened, kind = len(before) + len(self.turn), False, "turn"
        else:
            got = T.continuation(before, full, self.turn, self.thought, self.closed, self.empty) if messages[-1]["role"] == "tool" else None
            if got is None:
                raise AssertionError(f"turn check fails: suffix {self.tok.decode(suffix)!r}")
            start, full, opened = got
            kind = "reopened" if len(full) > len(before) + len(suffix) or start < len(before) else ("thought" if opened else "empty")
        return {"messages": messages, "tools": tools, "full": full, "start": start, "opened": opened, "kind": kind,
                "suffix": self.tok.decode(suffix, skip_special_tokens=False)}

    def canonical(self, req, reply, think):
        text, calls, reader = T.read_reply(self.tok, reply, self.stop, self.opening, self.closing, self.ids, req["opened"])
        later = self.render(req["messages"] + [T.later_message(text, calls)], req["tools"], think, False)
        return text, calls, reader, later


def model_call(name, args, order):
    """A call as the model writes it: its own format, keys in the order given (the template sorts them)."""
    def arg(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return '<|"|>' + v + '<|"|>'
        if isinstance(v, dict):
            return "{" + ",".join(f"{k}:{arg(x)}" for k, x in v.items()) + "}"
        if isinstance(v, list):
            return "[" + ",".join(arg(x) for x in v) + "]"
        return str(v)
    keys = list(args) if order == "given" else sorted(args)
    return "<|tool_call>call:" + name + "{" + ",".join(f"{k}:{arg(args[k])}" for k in keys) + "}<tool_call|>"


def random_args(rng, tool):
    props = tool["function"]["parameters"]["properties"]
    out = {}
    for key, p in props.items():
        if key not in tool["function"]["parameters"].get("required", []) and rng.random() < 0.4:
            continue
        t = p.get("type")
        if t == "string":
            out[key] = rng.choice(p["enum"]) if p.get("enum") else rng.choice(HOSTILE + ["Seoul", "src/main.py", "레드 셔츠"])
        elif t == "integer":
            out[key] = rng.choice([0, 1, 7, 120, -3, 100000])
        elif t == "number":
            out[key] = rng.choice([0.5, 19.99, 1e-07, 3, 12000.0])
        elif t == "boolean":
            out[key] = rng.random() < 0.5
        elif t == "array":
            out[key] = [rng.choice(HOSTILE) for _ in range(rng.randint(0, 3))]
        elif t == "object":
            out[key] = {"color": rng.choice(["red", "파랑", 'say "hi"']), "in_stock": rng.random() < 0.5}
    return out


def loops(side, n, rng):
    """n synthetic agent loops -> their records."""
    records = []
    for k in range(n):
        think, parallel = k % 2 == 1, (k // 2) % 3 + 1  # thinking off/on; one, two or three calls a step
        send_reasoning, with_text, named = k % 4 >= 2, k % 5 == 0, k % 3 != 0
        kit = list(TOOLS)[k % len(TOOLS)]
        tools = TOOLS[kit]
        system = [{"role": "system", "content": "You are a helpful agent. Use the tools."}] if k % 7 else []
        messages = system + [{"role": "user", "content": rng.choice(["What is the weather in Seoul and the time there?",
                                                                     "src/main.py 파일을 읽고 버그를 찾아줘.",
                                                                     "Find a red shirt under 30 dollars and add two to my cart."])}]
        steps = []
        for step in range(rng.randint(2, 4) + 1):
            body = {"messages": messages, "tools": tools}
            req = side.request(body, think)
            final = step >= 2 and rng.random() < 0.5 or step == 4
            reply_text = ""
            thought = "thought\nI should call the tools step by step." if think else None
            reply = ""
            if think:
                reply += ("" if req["opened"] else "<|channel>thought\n") + "Next I call the tool.\n<channel|>"
            if final:
                reply_text = "모두 끝났습니다. Here is the result: 3°C, {ok}."
                reply += reply_text + "<turn|>"
                calls_made = []
            else:
                if with_text:
                    reply_text = "Let me check that."
                    reply += reply_text
                calls_made = []
                for _ in range(parallel):
                    tool = rng.choice(tools)
                    args = random_args(rng, tool)
                    calls_made.append((tool["function"]["name"], args))
                    reply += model_call(tool["function"]["name"], args, rng.choice(["given", "sorted"]))
                reply += "<|tool_response>"
            reply_ids = side.tok.encode(reply, add_special_tokens=False)
            text, calls, reader, later = side.canonical(req, reply_ids, think)
            parsed_ok = [(c["function"]["name"], c["args"]) for c in calls] == [(nm, json.loads(json.dumps(a))) for nm, a in calls_made]
            rec = {"step": step, "kind": req["kind"], "suffix": req["suffix"], "prompt": len(req["full"]), "start": req["start"],
                   "base_ok": later[: req["start"]] == req["full"][: req["start"]] and req["start"] < len(later),
                   "parsed_ok": parsed_ok, "text_ok": text.strip() == reply_text.strip(), "calls": len(calls)}
            # the client's next request: the answer as it got it, then the tools' results (or a user's next question)
            answer = {"role": "assistant", "content": (text or None) if calls else text}
            if calls:
                answer["tool_calls"] = [T.public(c) for c in calls]
            if think and send_reasoning:
                answer["reasoning_content"] = "Next I call the tool."
            nxt = messages + [answer]
            if calls:
                for c in calls:
                    result = {"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"ok": True, "data": "결과 " + "x" * rng.randint(0, 40)}, ensure_ascii=False)}
                    if named:
                        result["name"] = c["function"]["name"]
                    nxt.append(result)
            else:
                nxt.append({"role": "user", "content": "Thanks. One more question: why?"})
            nreq = side.request({"messages": nxt, "tools": tools}, think)
            rec["next_prompt"] = len(nreq["full"])
            rec["prefix"] = later == nreq["full"][: len(later)]
            rec["reused"] = len(later) if rec["prefix"] else (req["start"] if req["full"][: req["start"]] == nreq["full"][: req["start"]] else 0)
            if think and send_reasoning:  # what passing the thinking a client sends back to the template would cost
                kreq = side.request({"messages": nxt, "tools": tools}, think, keep_reasoning=True)
                rec["prefix_if_kept_reasoning"] = later == kreq["full"][: len(later)]
                rec["reused_if_kept_reasoning"] = len(later) if rec["prefix_if_kept_reasoning"] else (
                    req["start"] if req["full"][: req["start"]] == kreq["full"][: req["start"]] else 0)
            steps.append(rec)
            messages = nxt
            if not calls:
                break
        records.append({"loop": k, "think": think, "parallel": parallel, "send_reasoning": send_reasoning, "text_beside_calls": with_text,
                        "named_results": named, "tools": kit, "system": bool(system), "steps": steps})
    return records


def round_trip(side, n, rng):
    """n calls with hostile strings, written by the template and read back -> (right, of, failures, special results)."""
    def value(depth=0):
        r = rng.random()
        if depth < 2 and r < 0.15:
            return {rng.choice(["a", "key", "도시", "x_1", "nested"]): value(depth + 1) for _ in range(rng.randint(0, 3))}
        if depth < 2 and r < 0.3:
            return [value(depth + 1) for _ in range(rng.randint(0, 3))]
        if r < 0.75:
            return rng.choice(HOSTILE) + (rng.choice(HOSTILE) if rng.random() < 0.3 else "")
        return rng.choice([0, -1, 42, 3.25, -0.5, 1e-07, 12345678901234, True, False, None])

    def read(args, name="do_it"):
        msg = {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}
        ids = side.render([{"role": "user", "content": "go"}, msg], None, False, False)
        a, b = ids.index(side.ids.call), ids.index(side.ids.end)
        return T.parse_call(side.tok, ids[a + 1:b], side.ids.quote, lenient=False)

    right, fails = 0, []
    for _ in range(n):
        args = {rng.choice(["city", "query", "path", "note", "q2", "이름"]) + str(j): value() for j in range(rng.randint(1, 4))}
        args = json.loads(json.dumps(args))  # as the client sends them back
        try:
            got = read(args)
            ok = got == ("do_it", args)
        except T.Malformed as e:
            got, ok = str(e), False
        right += ok
        if not ok:
            fails.append({"args": args, "got": got})
    special = []
    for s in SPECIAL:
        try:
            got = read({"text": s})
            special.append({"string": s, "read_back": got[1].get("text"), "same": got == ("do_it", {"text": s})})
        except T.Malformed as e:
            special.append({"string": s, "read_back": None, "error": str(e), "same": False})
    return right, n, fails, special


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loops", type=int, default=30)
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "exp", "R2-tool-calls", "prefix_check.json"))
    args = ap.parse_args()
    import transformers
    from transformers import AutoTokenizer

    transformers.logging.set_verbosity_error()
    side = Side(AutoTokenizer.from_pretrained(ckpt("orig")))
    rng = random.Random(args.seed)
    recs = loops(side, args.loops, rng)
    steps = [s for r in recs for s in r["steps"]]
    later_steps = [s for s in steps if s["next_prompt"]]
    suffixes = sorted({(s["kind"], s["suffix"]) for s in steps})
    # does the call's id reach the tokens? (the template uses it only to find a result's function)
    a = side.render([{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_aaa", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}}]}], None, False, False)
    b = side.render([{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_bbb", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}}]}], None, False, False)
    right, of, fails, special = round_trip(side, args.calls, rng)
    with_reasoning = [s for r in recs if r["think"] and r["send_reasoning"] for s in r["steps"]]
    report = {
        "loops": len(recs), "steps": len(steps),
        "prefix_holds": f"{sum(s['prefix'] for s in steps)} of {len(steps)}",
        "base_ok": f"{sum(s['base_ok'] for s in steps)} of {len(steps)}",
        "calls_read_back": f"{sum(s['parsed_ok'] for s in steps)} of {len(steps)}",
        "text_read_back": f"{sum(s['text_ok'] for s in steps)} of {len(steps)}",
        "reused_share_of_next_prompt": {"min": round(min(s["reused"] / s["next_prompt"] for s in later_steps), 3),
                                        "mean": round(sum(s["reused"] / s["next_prompt"] for s in later_steps) / len(later_steps), 3)},
        "turn_kinds": [list(x) for x in suffixes],
        "id_rendered": a != b,
        "with_reasoning_sent_back": {"steps": len(with_reasoning), "prefix_holds_as_served": sum(s["prefix"] for s in with_reasoning),
                                     "prefix_holds_if_passed_to_template": sum(s.get("prefix_if_kept_reasoning", False) for s in with_reasoning),
                                     "reused_share_if_passed": round(sum(s.get("reused_if_kept_reasoning", 0) / s["next_prompt"] for s in with_reasoning) / max(len(with_reasoning), 1), 3)},
        "round_trip": f"{right} of {of}", "round_trip_failures": fails[:10], "special_strings": special,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"report": report, "loops": recs}, f, ensure_ascii=False, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
