# Tool calls in Gemma 4's own format, as its chat template writes them (C:/models/gemma-4-31B-it/chat_template.jinja):
# - the tools' declarations at the top of the system turn: <|tool>declaration:NAME{description:...,parameters:{...}}<tool|>;
# - a call in the model's turn: <|tool_call>call:NAME{key:value,...}<tool_call|>, a string between two <|"|> tokens,
#   numbers, true, false and null as they are, objects {key:value} and lists [...];
# - its result in the same turn, right after the calls: <|tool_response>response:NAME{value:<|"|>...<|"|>}<tool_response|>,
#   after which the model goes on in that turn (the template opens no new one: thinking on, it opens a thought).
# The server (scripts/serve.py) takes a request's tools and tool messages to the template, and reads the calls of a
# reply on its token ids, so that a quote, a brace, a newline or Korean inside a string cannot end it early (the text
# of a string goes from one <|"|> token to the next). Limit of the format itself: the template does not escape, so a
# string whose text holds <|"|>, <|tool_call>, <tool_call|> or <|tool_response> becomes those tokens and cannot be
# written or read back as it was.
import json
import uuid

from sclmd.text import Pieces

PUNCT = "{}[]:,"


class BadRequest(ValueError):
    """A request whose tools or tool messages cannot be served: (message, the field it is about)."""

    def __init__(self, message: str, param: str | None = None):
        super().__init__(message)
        self.param = param


class Malformed(ValueError):
    """A call block that is not call:NAME{key:value,...}."""


class Ids:
    """The format's special tokens."""

    def __init__(self, tok):
        get = tok.convert_tokens_to_ids
        self.call, self.end, self.response, self.quote = get("<|tool_call>"), get("<tool_call|>"), get("<|tool_response>"), get('<|"|>')


# -- the request ---------------------------------------------------------------------------------------------------

class Named:
    """What a request gives that is not served: by default left out and named (the header x-sclmd-ignored), as the
    server did with every tool before it served them; with --strict-fields (strict) refused (400)."""

    def __init__(self, strict: bool = False):
        self.strict, self.names = strict, []

    def __call__(self, name: str, why: str, param: str = "messages") -> None:
        if self.strict:
            raise BadRequest(f"{name}: {why} (--strict-fields refuses what is left out)", param)
        self.names.append(name)


def tools_of(body: dict, strict: bool = False) -> tuple[list | None, list, list]:
    """The tools to declare to the model -> (tools as the template takes them or None, fields given and not served,
    the index in the request's tools of each one declared). tool_choice none: none are declared. required or a
    function named: not enforced (no grammar here; named in the answer as ignored); a named function is the only one
    declared. A tool the server cannot declare (not an object, no name, parameters that are not a schema) is left out
    and named; so is a tool_choice that names no tool given. strict: those are refused (400)."""
    tools, choice = body.get("tools"), body.get("tool_choice")
    if not tools:
        return None, [], []
    named = Named(strict)
    if not isinstance(tools, list):
        named("tools", "a list of {type: function, function: {name, description, parameters}}", "tools")
        return None, named.names, []
    out, at = [], []
    for k, t in enumerate(tools):
        if not isinstance(t, dict):
            named(f"tools[{k}]", "an object {type: function, function: {...}}", "tools")
            continue
        if t.get("type", "function") != "function":  # a tool of a kind this server cannot run for the client
            named(f"tools[{k}].type", "only function tools are served", "tools")
            continue
        f = t.get("function")
        if not isinstance(f, dict) or not isinstance(f.get("name"), str) or not f["name"]:
            named(f"tools[{k}].function.name", "a function's name is needed", "tools")
            continue
        params = f.get("parameters")
        if params is not None and not isinstance(params, dict):
            named(f"tools[{k}].function.parameters", "a JSON schema object (declared without it)", "tools")
            params = None
        if f.get("strict"):  # the arguments are not held to the schema here (no grammar)
            named(f"tools[{k}].function.strict", "the arguments are not held to the schema (no grammar)", "tools")
        description = f.get("description")
        if params:
            try:
                params = schema_of(params)
            except RecursionError:  # nested past Python's depth: as it came (the template tells whether it can write it)
                pass
        out.append({"type": "function", "function": {"name": f["name"], "description": description if isinstance(description, str) else "",
                                                     **({"parameters": params} if params else {})}})
        at.append(k)
    if choice in (None, "auto"):
        pass
    elif choice == "none":
        return None, named.names, []
    elif choice == "required":
        named.names.append("tool_choice")  # (--strict-fields refuses it in serve_contract.check_fields)
    elif isinstance(choice, dict) and isinstance(choice.get("function"), dict) and isinstance(choice["function"].get("name"), str):
        alone = [(t, k) for t, k in zip(out, at) if t["function"]["name"] == choice["function"]["name"]]
        if alone:
            out, at = [t for t, _ in alone], [k for _, k in alone]
            named.names.append("tool_choice")
        else:  # as auto: every tool declared
            named("tool_choice", f"no tool is named {choice['function']['name']!r}", "tool_choice")
    else:  # as auto
        named("tool_choice", "none, auto, required or {type: function, function: {name}}", "tool_choice")
    return (out or None), sorted(set(named.names)), at


# -- schemas as the template writes them ------------------------------------------------------------------------------
# The template writes a parameter's description, type (STRING, NUMBER, ..., an enum for a string, items for an array,
# properties and required for an object) and nullable; nothing else. A schema made by pydantic (the Agents SDK,
# LangChain) says the same things in forms the template does not know: a nested model is a $ref to $defs, an optional
# field is anyOf [X, {type: null}] or type [X, "null"]. Those are rewritten here into the template's forms (the
# $ref inlined, the optional made nullable), so that the declaration the model reads keeps them; a schema without any
# of these forms is declared as it is (the same object). What has no form in the template (anyOf of two real types,
# additionalProperties, a pattern, ...) is passed as it comes, and the template leaves it out.
_FORMS = ("$ref", "anyOf", "oneOf", "allOf")


def _has_forms(s) -> bool:
    if isinstance(s, dict):
        return any(k in s for k in _FORMS) or isinstance(s.get("type"), list) or any(_has_forms(v) for v in s.values())
    if isinstance(s, list):
        return any(_has_forms(v) for v in s)
    return False


def _resolve(s, defs: dict, path: tuple):
    """s with its $refs inlined (path: the models being inlined around it; a model inside itself, a recursive one, is
    declared there as an object), a nullable anyOf, oneOf or type list made nullable, a single allOf merged."""
    if isinstance(s, list):
        return [_resolve(v, defs, path) for v in s]
    if not isinstance(s, dict):
        return s
    s = dict(s)
    ref = s.get("$ref")
    name = ref.split("/")[-1] if isinstance(ref, str) and ref.startswith(("#/$defs/", "#/definitions/")) else None
    if name in defs:
        del s["$ref"]
        target = {"type": "object"} if name in path else _resolve(defs[name], defs, path + (name,))
        s = {**target, **s}  # the field's own description, ... over the model's
    for key in ("anyOf", "oneOf"):
        alts = s.get(key)
        if isinstance(alts, list):
            real = [a for a in alts if not (isinstance(a, dict) and a.get("type") == "null")]
            if len(real) == 1 and isinstance(real[0], dict):
                del s[key]
                s = {**_resolve(real[0], defs, path), **s}
                if len(real) < len(alts):
                    s["nullable"] = True
    alls = s.get("allOf")
    if isinstance(alls, list) and len(alls) == 1 and isinstance(alls[0], dict):
        del s["allOf"]
        s = {**_resolve(alls[0], defs, path), **s}
    kind = s.get("type")
    if isinstance(kind, list):
        real = [t for t in kind if t != "null"]
        if len(real) == 1:
            s["type"] = real[0]
            if len(real) < len(kind):
                s["nullable"] = True
    if isinstance(s.get("properties"), dict):
        s["properties"] = {k: _resolve(v, defs, path) for k, v in s["properties"].items()}
    if isinstance(s.get("items"), (dict, list)):
        s["items"] = _resolve(s["items"], defs, path)
    return s


def schema_of(params: dict) -> dict:
    """A tool's parameters as the template can write them (above): the same object when it has none of the forms."""
    if not _has_forms(params):
        return params
    defs = {}
    for key in ("definitions", "$defs"):
        if isinstance(params.get(key), dict):
            defs.update({k: v for k, v in params[key].items() if isinstance(v, dict)})
    out = _resolve({k: v for k, v in params.items() if k not in ("$defs", "definitions")}, defs, ())
    return out


def _text_of(content, where: str):
    """A message's content as the template takes it: text parts joined (others refused, as for every message)."""
    if isinstance(content, list):
        if any(not isinstance(p, dict) or p.get("type") not in ("text", None) for p in content):
            raise BadRequest("only text content is served", where)
        return "".join(p.get("text", "") for p in content)
    return content


def _arguments(f: dict):
    """A call's arguments as the template takes them (an object, or None) -> (them, why they cannot be: None if they can)."""
    args = f.get("arguments")
    if isinstance(args, str):
        if not args.strip():
            return {}, None
        try:
            args = json.loads(args)
        except (ValueError, RecursionError) as e:
            return None, f"not JSON ({str(e)[:80]})"
    if args is not None and not isinstance(args, dict):
        return None, "a JSON object"
    return args, None


def messages_of(raw: list, strict: bool = False) -> tuple[list, list]:
    """A request's messages as the template takes them -> (them, what was left out of them, named). Role, content (text)
    and, for tool calls, the calls with their arguments as objects (the client sends them as JSON text) and the tool
    messages with their call's id and name. A message without any of these fields comes out as it did before tool calls
    were served: {role, content}.
    - A tool's result given as an object (or a number), or as a list that is not text parts (values, objects without a
      type), is written as its JSON text: the template takes text or parts. Parts of another type are refused as before.
    - A call that cannot be written (not {function: {name, arguments}}, arguments that are not a JSON object) is left out
      of its message and named, as calls were before they were served; so are tool_calls that are not a list.
    - A tool's result that follows no assistant message with calls (the template writes nothing for it) is named.
    strict (--strict-fields): those are refused (400)."""
    out, named = [], Named(strict)
    for i, m in enumerate(raw):
        role, content = m.get("role"), m.get("content")
        if role == "tool" and isinstance(content, list) and content and not all(
                isinstance(p, dict) and p.get("type") in ("text", None) and isinstance(p.get("text"), str) for p in content) and not any(
                isinstance(p, dict) and p.get("type") not in ("text", None) for p in content):
            content = json.dumps(content, ensure_ascii=False)  # a result that is a list of values or objects, not text parts
        content = _text_of(content, "messages")
        if role == "developer":
            role = "system"
        if role == "tool":
            if content is not None and not isinstance(content, str):  # an object: its JSON text
                content = json.dumps(content, ensure_ascii=False)
            msg = {"role": "tool", "content": "" if content is None else content}
            for key in ("tool_call_id", "name"):
                if isinstance(m.get(key), str):
                    msg[key] = m[key]
            j = len(out) - 1
            while j >= 0 and out[j]["role"] == "tool":
                j -= 1
            if j < 0 or not out[j].get("tool_calls"):  # its call is not in the conversation: the template leaves it out
                named(f"messages[{i}]", "a tool's result after no assistant message with tool_calls (the template writes nothing for it)")
            out.append(msg)
            continue
        calls = m.get("tool_calls") if role == "assistant" else None
        if calls:
            got = []
            if not isinstance(calls, list):
                named(f"messages[{i}].tool_calls", "a list of {id, type: function, function: {name, arguments}}")
                calls = []
            for k, c in enumerate(calls):
                f = c.get("function") if isinstance(c, dict) else None
                if not isinstance(f, dict) or not isinstance(f.get("name"), str) or not f["name"]:
                    named(f"messages[{i}].tool_calls[{k}]", "{id, type: function, function: {name, arguments}}")
                    continue
                args, why = _arguments(f)
                if why is not None:
                    named(f"messages[{i}].tool_calls[{k}].function.arguments", why)
                    continue
                call = {"type": "function", "function": {"name": f["name"], "arguments": args}}
                if isinstance(c.get("id"), str):
                    call["id"] = c["id"]
                got.append(call)
            if content is not None and not isinstance(content, str):
                raise BadRequest("a message is {role: system | user | assistant | tool, content: text}", "messages")
            if got:
                out.append({"role": "assistant", "content": content or "", "tool_calls": got})
                continue
            if content is None:  # no call left: a message of its text
                content = ""
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            raise BadRequest("a message is {role: system | user | assistant | tool, content: text}", "messages")
        out.append({"role": role, "content": content})
    return out, sorted(set(named.names))


def ids_of(got) -> list:
    """apply_chat_template's token ids (a list, or a mapping with input_ids)."""
    return list(got["input_ids"] if hasattr(got, "keys") else got)


def uses_tools(messages: list) -> bool:
    return any(m["role"] == "tool" or m.get("tool_calls") for m in messages)


def continuation(before: list, full: list, turn: list, thought: list, closed: list, empty: list):
    """Where the model's reply begins in `full` (the prompt), after a tool's result: the template writes no turn there,
    so the reply goes on in the model's turn (nothing more with thinking off; with thinking on, an open thought:
    `thought`). -> (start, the prompt, whether the reply begins inside a thought), or None (not such a prompt).

    When the model's last message had text besides its calls, the template closes that turn after the results
    (`closed`: <turn|> and a newline) and opens none: a model's turn is then opened as for any reply (`turn`, and with
    thinking off the `empty` thought the template writes there). Its start is before the close: a later turn writes
    the reply on in the same turn, after that text."""
    if full[: len(before)] != before:
        return None
    suffix = full[len(before):]
    if suffix not in ([], thought):
        return None
    if before[-len(closed):] == closed:
        return len(before) - len(closed), before + turn + ([] if suffix else empty), False
    return len(before), full, bool(suffix)


# -- the reply -------------------------------------------------------------------------------------------------------

def _lex(tok, ids: list, quote: int) -> list:
    """A call block's tokens -> its lexemes: ("s", the text of a string), (c, c) for c in PUNCT, ("w", a bare word)."""
    out, run, inside = [], [], False

    def words(run):
        s = tok.decode(run, skip_special_tokens=False)
        i = 0
        while i < len(s):
            c = s[i]
            if c in PUNCT:
                out.append((c, c))
                i += 1
            elif c.isspace():
                i += 1
            else:
                j = i
                while j < len(s) and s[j] not in PUNCT and not s[j].isspace():
                    j += 1
                out.append(("w", s[i:j]))
                i = j

    for t in ids:
        if t == quote:
            if inside:
                out.append(("s", tok.decode(run, skip_special_tokens=False)))
            elif run:
                words(run)
            run, inside = [], not inside
        else:
            run.append(t)
    if inside:
        raise Malformed("a string is not closed")
    if run:
        words(run)
    return out


def _value(lx: list, i: int, lenient: bool):
    if i >= len(lx):
        raise Malformed("a value is missing")
    kind, x = lx[i]
    if kind == "s":
        return x, i + 1
    if kind == "{":
        return _object(lx, i, lenient)
    if kind == "[":
        items, i = [], i + 1
        if i < len(lx) and lx[i][0] == "]":
            return items, i + 1
        while True:
            v, i = _value(lx, i, lenient)
            items.append(v)
            if i < len(lx) and lx[i][0] == ",":
                i += 1
            elif i < len(lx) and lx[i][0] == "]":
                return items, i + 1
            else:
                raise Malformed("a list is not closed")
    if kind == "w":
        try:
            v = json.loads(x)  # a number, true, false, null
        except ValueError:
            if lenient:  # a bare word where a string was meant
                return x, i + 1
            raise Malformed(f"{x!r} is not a value") from None
        if isinstance(v, (dict, list)):
            raise Malformed(f"{x!r} is not a value")
        return v, i + 1
    raise Malformed(f"{x!r} where a value was expected")


def _object(lx: list, i: int, lenient: bool):
    obj, i = {}, i + 1
    if i < len(lx) and lx[i][0] == "}":
        return obj, i + 1
    while True:
        if i >= len(lx) or lx[i][0] not in ("w", "s"):
            raise Malformed("a key is missing")
        key = lx[i][1]
        if i + 1 >= len(lx) or lx[i + 1][0] != ":":
            raise Malformed(f"no ':' after the key {key!r}")
        obj[key], i = _value(lx, i + 2, lenient)
        if i < len(lx) and lx[i][0] == ",":
            i += 1
        elif i < len(lx) and lx[i][0] == "}":
            return obj, i + 1
        else:
            raise Malformed("an object is not closed")


def parse_call(tok, ids: list, quote: int, lenient: bool = True) -> tuple[str, dict]:
    """The tokens between <|tool_call> and <tool_call|> -> (the function's name, its arguments)."""
    lx = _lex(tok, ids, quote)
    if len(lx) < 4 or lx[0] != ("w", "call") or lx[1][0] != ":" or lx[2][0] != "w" or lx[3][0] != "{":
        raise Malformed("not call:NAME{...}")
    args, i = _object(lx, 3, lenient)
    if i != len(lx):
        raise Malformed("text after the call's arguments")
    return lx[2][1], args


def new_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


class Reader:
    """A reply's tokens as they come, for a request with tools: add(token) -> ("think" | "answer", text) or None, as
    Pieces gives them; the calls whose block has closed are in `calls` (and those not yet taken in take()), each
    {"id", "type": "function", "function": {"name", "arguments": JSON text}} with `args` (the object) beside it.
    A block that does not parse stays in the answer as its text (and in `malformed`). think: the reply begins inside a
    thought the prompt opened. Each call has beside it `at`: how many characters of the answer had come before it, so
    that a stop string of the client's that cuts the answer cuts the calls after it too (kept_before)."""

    def __init__(self, tok, stop: tuple, opening: int, closing: int, ids: Ids, think: bool = False, lenient: bool = True):
        self.tok, self.ids, self.lenient, self.stop = tok, ids, lenient, stop
        self.pieces = Pieces(tok, stop, opening, closing)
        if think:
            self.pieces.kind, self.pieces.name = "think", False
        self.block = None  # the tokens of a call being written
        self.calls, self.malformed, self.unclosed = [], [], 0
        self._taken = 0
        self.said = 0  # characters of the answer given so far

    def _close(self, closed: bool):
        block, self.block = self.block, None
        try:
            name, args = parse_call(self.tok, block, self.ids.quote, self.lenient)
        except Malformed as e:
            text = self.tok.decode([self.ids.call] + block + ([self.ids.end] if closed else []), skip_special_tokens=False)
            self.malformed.append((str(e), text))
            return ("answer", text)
        self.unclosed += not closed
        self.calls.append({"id": new_id(), "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                           "args": args, "at": self.said})
        return None

    def _said(self, piece):
        if piece is not None and piece[0] == "answer":
            self.said += len(piece[1])
        return piece

    def add(self, token: int):
        if self.block is not None:
            if token == self.ids.end:
                return self._said(self._close(True))
            if token in self.stop:  # the reply ended inside the block (no <tool_call|>): read what is there
                return self._said(self._close(False))
            self.block.append(token)
            return None
        if token == self.ids.call and self.pieces.kind == "answer":
            self.block = []
            return None
        return self._said(self.pieces.add(token))

    def end(self):
        """The reply is over: a block still open (the reply reached its length) is read as it is."""
        return self._said(self._close(False)) if self.block is not None else None

    def take(self) -> list:
        """The calls that have closed since the last take()."""
        new = self.calls[self._taken:]
        self._taken = len(self.calls)
        return new


def kept_before(calls: list, answer: str, stops) -> list:
    """The calls a reply keeps when a stop string cuts its answer (`answer`: the answer before the cut): those that had
    closed before the first stop string was whole, as they come when the tokens are read one at a time (the client
    of a stream has had them), whatever the batches the tokens come in."""
    whole = min((answer.find(s) + len(s) for s in stops if s and s in answer), default=None)
    return list(calls) if whole is None else [c for c in calls if c["at"] < whole]


def public(call: dict) -> dict:
    """A call as the client gets it (without the parsed arguments and the place kept beside it)."""
    return {k: v for k, v in call.items() if k not in ("args", "at")}


def later_message(text: str, calls: list) -> dict:
    """The reply as an assistant message for the template, the way the next request will send it back. With calls, the
    text is left out: the template writes a message's text after its tools' results, so the part of the next
    request that is the same as the reply ends with the calls (and <|tool_response>)."""
    if calls:
        return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": c["function"]["name"], "arguments": c["args"]}}
                                                                   for c in calls]}
    return {"role": "assistant", "content": text}


def read_reply(tok, tokens: list, stop: tuple, opening: int, closing: int, ids: Ids, think: bool = False) -> tuple[str, list, Reader]:
    """A whole reply -> (its answer's text, its calls, the reader)."""
    r, text = Reader(tok, stop, opening, closing, ids, think), ""
    for token in tokens:
        piece = r.add(token)
        if piece is not None and piece[0] == "answer":
            text += piece[1]
    piece = r.end()
    if piece is not None:
        text += piece[1]
    return text, r.calls, r
