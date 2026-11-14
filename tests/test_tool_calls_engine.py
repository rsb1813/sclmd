# An agent loop on the server's conversations (scripts/serve.py: Server) with the small random model (GPU), as tool
# calls make it: a step's prompt is the last step's later form (the call as the template writes it) and the tool's
# result, with no turn of its own (the reply goes on in the model's turn: start = the prompt's end, or before an opened
# thought), and a step may end at a token of its own (parallel_tool_calls: false ends it at <tool_call|>). Each later
# step reads only the result, and its reply is the one a fresh engine gives for the whole prompt.
import pytest
import torch

from scripts.serve import Request, Server
from sclmd import engine as E
from sclmd.generate import Picker, generate
from test_engine import engine_drafter, ids_of
from test_engine import drafter_e, small_e  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not __import__("os").path.exists(E.DLL), reason="needs a GPU and the engine")
STOP = (7,)
CALL_END = 9  # in the small model's place of <tool_call|>
CAP = 160  # positions: three steps with 44-token replies


def hot(seed: int = 0) -> Picker:
    return Picker(temperature=5.0, top_k=16, top_p=1.0, seed=seed)  # as tests/test_serve.py: the text depends on what came before


def fresh(small, full: list, max_new: int, pick: Picker, stop=STOP) -> list:
    return generate(small.engine(capacity=CAP), full, max_new, pick, stop=stop)


def run(server: Server, req: Request) -> tuple:
    server.submit(req)
    while server.turn(wait=False):
        pass
    tokens, result = [], None
    while not req.events.empty():
        event = req.events.get()
        if event[0] == "token":
            tokens.append(event[1])
        else:
            result = event[1]
    return tokens, result


@pytest.mark.parametrize("reply_room", [12, 44])  # 44: past what the sliding ring keeps of a turn start
def test_an_agent_loop_reads_only_each_tools_result(small_e, drafter_e, reply_room):  # noqa: F811
    eng = small_e.engine(capacity=CAP, sequences=3)
    server = Server(None, eng, engine_drafter(eng, drafter_e), STOP)
    written = {}  # step -> the later form's tokens after the turn start (the call as the template writes it)

    def canon(req, tokens):
        if req is other:
            return None  # nothing read ahead for it
        step = len(written)
        written[step] = ids_of(5 + step, 300 + step) + [50 % 320]  # ... and <|tool_response> in its place
        return req.full[: req.start] + written[step]

    server.canon = canon
    full = ids_of(30, 301) + [5, 6]
    start = len(full) - 1  # the first step: a turn of the model's own
    other = Request(ids_of(20, 309), 19, 10, hot(9), False)  # another conversation at the same time
    server.submit(other)
    for step in range(3):
        stop = STOP if step != 1 else STOP + (CALL_END,)  # step 1: one call, the reply ends at <tool_call|>
        req = Request(full, start, reply_room, hot(step + 1), False)
        if step == 1:
            req.stop_ids = stop
        reply, result = run(server, req)
        assert reply == fresh(small_e, full, reply_room, hot(step + 1), stop), f"step {step}"
        if step:
            assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == later_len, f"step {step}: read more than the result"
        if result["ended"] not in ("stop", "length"):
            pytest.fail(f"step {step} ended {result['ended']}")
        later = full[:start] + written[step]
        later_len = len(later)
        result_tokens = ids_of(8, 310 + step)  # the tool's result, rendered
        if step == 1:  # with thinking on, the template opens a thought after the result: the reply begins inside it
            full, start = later + result_tokens + [11, 12, 13], len(later) + len(result_tokens)
        else:  # thinking off: nothing after the result
            full = later + result_tokens
            start = len(full)
    assert not server.active and len(server.kept) == 2  # the loop's conversation and the other one
