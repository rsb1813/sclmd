# A terminal chat with Gemma 4 31B on the engine: the conversation lives in the engine's keys and values, a reply is
# streamed as the main model settles its tokens (the drafter guesses ahead, the main model verifies), and slash
# commands change the settings. The context is always what the model's chat template writes for the conversation.
import argparse
import ctypes
import os
import random
import signal
import threading
import time

import torch
from rich.console import Console

from sclmd import ckpt
from sclmd import engine as E
from sclmd import model as M
from sclmd import plan as P
from sclmd.generate import Picker, extend
from sclmd.stream import VRAM_MARGIN, cap_vram
from sclmd.text import Pieces

# Tokens the drafter guesses ahead. The drafter's generation_config.json says 6 (num_assistant_tokens); with 7 a pass
# has 8 rows, which costs what 7 rows cost (rows are in groups of eight), and more guesses are kept than the extra
# guess costs: 160 -> 165 tok/s over scripts/speed_check.py's replies. With 8 the pass has 9 rows and costs 3 ms more
# (140 tok/s then). The text does not depend on it.
DEPTH = 7
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARES = "own_v1"  # the measurements the plan is fitted from: on text the model wrote itself (on unseen text of that kind a plan
# fitted from them is KL 0.055 where one of the same size fitted from other people's text is 0.066)
# Rows of a pass: a prompt is read that many tokens at a time. On the chat's engine (scripts/prefill_check.py): 16 rows
# 668 tok/s for a prompt of 512 tokens and 543 for one of 12,000; 32 rows 957 and 768, for 62 MB more of work memory
# (86 MB at a context of 16,384) that the plan cannot give to 4-bit matrices (--rows).
ROWS = 16
LOAD_ROOM = 320 << 20  # VRAM left over the engine: what loading uses for a moment, and the allocator's leftovers

HELP = """[bold]명령[/]
  /?, /help              이 도움말
  /bye, /exit            끝내기 (Ctrl+D도 같음)
  /clear                 대화를 지우고 새 문맥에서 시작
  /show                  지금 설정과 문맥 사용량
  /think, /nothink       생각 과정을 쓸지 (다음 답부터; 바꾸면 대화를 처음부터 다시 읽음)
  /showthink, /hidethink 생각 과정을 펼쳐 보일지, 한 줄로 접을지 (기본: 접음)
  /system <글>           시스템 프롬프트 (다음 답부터; 대화 중에 바꾸면 처음부터 다시 읽음)
  /temperature <수>      기본 1.0 (모델 공식값). 0이면 가장 확률 높은 토큰만 고름
  /top_p <수>, /top_k <수>  temperature가 0보다 클 때 고르는 범위 (기본 0.95, 64: 모델 공식값)
  /num_predict <수>      답 하나의 최대 토큰 수 (기본 4096)
  /drafter, /nodrafter   드래프터로 몇 토큰씩 미리 짐작할지 (기본: 켬. 글은 같고 속도만 다름)
  /verbose, /quiet       답 뒤에 속도를 보일지
[bold]입력[/]
  \"\"\"로 시작하면 여러 줄을 쓸 수 있고, \"\"\"로 끝냅니다. 답이 나오는 중에 Ctrl+C를 누르면 멈춥니다."""


class ContextFull(Exception):
    """The turn does not fit in what is left of the context."""


def looping(tokens: list, repeats: int = 6, span: int = 96) -> bool:
    """Whether a reply has run into a loop: its last `span` tokens or more are one piece of at most 32 tokens, said
    at least `repeats` times in a row."""
    for period in range(1, 33):
        need = max(span, period * repeats)
        if len(tokens) >= need and tokens[-1] == tokens[-1 - period] and tokens[-need + period:] == tokens[-need:-period]:
            return True
    return False


class Session:
    """One conversation. The engine holds `ctx`, which is always a beginning of what the chat template writes for the
    messages. A reply is generated after the template's prompt; the template writes an earlier reply differently
    (without its thinking, and without the empty thinking a reply starts with when thinking is off), so after a reply
    the engine goes back to where the model's turn began and reads the reply again in that form (settle)."""

    def __init__(self, tok, eng: E.Engine, drafter):
        self.tok, self.eng, self.drafter = tok, eng, drafter
        self.stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<|tool_response>"))
        self.open, self.close = tok.convert_tokens_to_ids("<|channel>"), tok.convert_tokens_to_ids("<channel|>")
        self.turn = tok.encode("<|turn>model\n", add_special_tokens=False)
        self.system, self.think, self.use_drafter = "", False, True
        self.clear()

    def clear(self) -> None:
        self.messages, self.ctx, self.owed = [], [], None
        self.eng.crop(0)

    def render(self, prompt: bool) -> list:
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + self.messages
        out = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=prompt, enable_thinking=self.think)
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    def settle(self) -> None:
        """The last reply as the template writes an earlier turn: back to where the model's turn began, and from
        there the reply's text and the end of the turn. The chat does this while the next message is typed."""
        if self.owed is None:
            return
        back, tokens = self.owed
        self.eng.restore()
        if tokens:
            self.eng.feed(tokens, keep="last")
        self.ctx, self.owed = self.ctx[:back] + tokens, None

    def ask(self, message: str, opts: dict, on_text, halt=lambda: False) -> dict:
        """Feeds the user's message and generates the reply, handing each piece of text to on_text(kind, text) with
        kind "think" or "answer". Returns the counts and times of the turn."""
        eng, tok = self.eng, self.tok
        t0 = time.perf_counter()
        self.settle()
        self.messages.append({"role": "user", "content": message})
        before, full = self.render(False), self.render(True)
        start = len(before) + len(self.turn)  # the model's turn begins here: what a later turn's context keeps up to
        assert full[:len(before)] == before and full[len(before):start] == self.turn
        room = eng.capacity - len(full) - eng.rows
        if room < 16:
            self.messages.pop()
            raise ContextFull
        same = 0
        while same < min(len(self.ctx), len(full)) and self.ctx[same] == full[same]:
            same += 1
        if same < len(self.ctx):  # the beginning changed (thinking switched, another system prompt): read it all again
            same = 0
            eng.crop(0)
        logits, hidden = eng.feed(full[same:start], keep="last")
        eng.snapshot()
        if full[start:]:
            logits, hidden = eng.feed(full[start:], keep="last")
        self.ctx = list(full)
        t1 = time.perf_counter()

        said, looped, answer = [], [False], []
        # the text of a few tokens at a time: decoding the reply so far at every token was 15% of a long thinking pass
        pieces = Pieces(tok, self.stop, self.open, self.close)

        def emit(token: int) -> None:
            said.append(token)
            if looping(said):
                looped[0] = True
            if token in self.stop:
                return
            piece = pieces.add(token)
            if token == self.open or token == self.close:
                on_text(pieces.kind, "")
            elif pieces.kind == "answer":
                answer.append(token)
            if piece:
                on_text(*piece)

        pick = Picker(opts["temperature"], opts["top_k"], opts["top_p"], seed=random.getrandbits(30))
        guess = self.drafter.guess if self.drafter is not None and self.use_drafter else None
        stats = {}
        extend(eng, len(full), logits[-1], hidden[-1], min(opts["num_predict"], room), pick, guess, self.stop, stats, emit,
               lambda: halt() or looped[0])
        t2 = time.perf_counter()
        # what the context keeps of this turn: the template's own rendering of it as an earlier turn
        self.messages.append({"role": "assistant", "content": tok.decode(answer, skip_special_tokens=True)})
        later = self.render(False)
        assert later[:start] == full[:start]
        self.owed = (start, later[start:])
        return {"prompt_tokens": len(full) - same, "prompt_s": t1 - t0, "tokens": len(said), "seconds": t2 - t1,
                "per_pass": stats["tokens"] / max(stats["passes"], 1), "looped": looped[0]}

    def used(self) -> int:
        return len(self.ctx) if self.owed is None else self.owed[0] + len(self.owed[1])


def read_message(prompt: str) -> str:
    """One message from the terminal; a line starting with three double quotes opens a message of several lines."""
    line = input(prompt)
    if not line.startswith('"""'):
        return line
    lines = [line[3:]]
    while not lines[-1].endswith('"""'):
        lines.append(input("... "))
    lines[-1] = lines[-1][:-3]
    return "\n".join(lines)


def settings(session: Session, opts: dict) -> str:
    return (f"생각 {'켬' if session.think else '끔'}({'펼침' if opts['showthink'] else '접음'}) · temperature {opts['temperature']:g} · "
            f"top_p {opts['top_p']:g} · top_k {opts['top_k']} · num_predict {opts['num_predict']} · "
            f"드래프터 {'켬' if session.use_drafter and session.drafter is not None else '끔'}")


def command(line: str, session: Session, opts: dict, console: Console) -> bool:
    """Runs a slash command; False ends the chat."""
    name, _, value = line.strip().partition(" ")
    key, value = name[1:], value.strip()
    try:
        if key in ("bye", "exit"):
            return False
        if key in ("?", "help"):
            console.print(HELP)
        elif key == "clear":
            session.clear()
            console.print("[grey62]대화를 지웠습니다.[/]")
        elif key == "show":
            console.print(f"[grey62]문맥 {session.used()} / {session.eng.capacity} 토큰 · {settings(session, opts)}\n"
                          f"시스템 프롬프트: {session.system or '(없음)'}[/]", highlight=False)
        elif key == "system":
            session.system = value
            console.print("[grey62]시스템 프롬프트를 바꿨습니다" + ("." if not session.messages else " (다음 답부터 적용: 대화를 처음부터 다시 읽습니다).") + "[/]")
        elif key in ("think", "nothink"):
            session.think = key == "think"
        elif key in ("showthink", "hidethink"):
            opts["showthink"] = key == "showthink"
        elif key in ("drafter", "nodrafter"):
            session.use_drafter = key == "drafter"
        elif key in ("temperature", "top_p"):
            opts[key] = float(value)
        elif key in ("num_predict", "top_k"):
            opts[key] = int(value)
        elif key in ("verbose", "quiet"):
            opts["verbose"] = key == "verbose"
        else:
            console.print("[yellow]모르는 명령입니다. /? 를 보세요.[/]")
    except ValueError:
        console.print("[yellow]값을 읽지 못했습니다.[/]")
    return True


def echo_typing(on: bool) -> None:
    """On a Windows console, whether keys typed are shown. Off while a reply streams, or they would land in the
    middle of it; turning it on again also throws away what was typed meanwhile."""
    if os.name != "nt":
        return
    kernel32, mode = ctypes.windll.kernel32, ctypes.c_uint()
    stdin = kernel32.GetStdHandle(-10)
    if not kernel32.GetConsoleMode(stdin, ctypes.byref(mode)):  # input is not a console
        return
    if on:
        kernel32.FlushConsoleInputBuffer(stdin)
    kernel32.SetConsoleMode(stdin, mode.value | 4 if on else mode.value & ~4)  # ENABLE_ECHO_INPUT


def reply(line: str, session: Session, opts: dict, console: Console) -> None:
    """Streams the reply to one message, then a line with the speed. The thinking is streamed in grey, or folded
    into one line that counts its pieces (opts["showthink"])."""
    shown, stopped, thought = [None], [False], [0, time.perf_counter()]

    def on_text(kind: str, text: str) -> None:
        if kind == "think" and not opts["showthink"]:
            if shown[0] != "think":
                thought[1] = time.perf_counter()
            shown[0], thought[0] = kind, thought[0] + 1
            if thought[0] % 8 == 1:  # rewritten in place
                console.print(f"생각 중… {thought[0]}", end="\r", style="grey50", highlight=False)
            return
        if kind != shown[0]:
            if kind == "think":
                console.print("생각 중…", style="grey50")
            elif shown[0] == "think" and opts["showthink"]:
                console.print("\n…생각 끝\n", style="grey50")
            elif shown[0] == "think":
                console.print(f"생각 끝 · {time.perf_counter() - thought[1]:.0f}초            \n", style="grey50", highlight=False)
            shown[0] = kind
        console.print(text, end="", style="grey50" if kind == "think" else None, markup=False, highlight=False, soft_wrap=True)

    # Ctrl+C only raises a flag that is read between passes
    before = signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    echo_typing(False)
    try:
        stats = session.ask(line, opts, on_text, lambda: stopped[0])
    except ContextFull:
        console.print("[yellow]문맥이 가득 찼습니다. /clear 로 새로 시작하세요.[/]")
        return
    finally:
        echo_typing(True)
        signal.signal(signal.SIGINT, before)
    console.print("\n")
    if stats["looped"]:
        console.print("[yellow]같은 구절이 되풀이되어 멈췄습니다. /clear 로 새로 시작하는 편이 좋습니다.[/]\n")
    if opts["verbose"]:
        note = " · 멈춤" if stopped[0] else ""
        console.print(f"[grey50]{stats['tokens']} 토큰 · {stats['tokens'] / max(stats['seconds'], 1e-9):.1f} tok/s · 검증 한 번에 "
                      f"{stats['per_pass']:.1f} 토큰 · 입력 {stats['prompt_tokens']} 토큰 {stats['prompt_s']:.1f}초 · "
                      f"문맥 {session.used()}/{session.eng.capacity}{note}[/]\n", highlight=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1", help="the codes' folder under C:\\models\\sclmd-codes")
    ap.add_argument("--plan", default="auto", help="which matrices are 3 bits and which 4: auto (as many at 4 bits as the VRAM "
                    "that is free now holds, the most useful first), c6 (3 bits everywhere), or a plan file")
    ap.add_argument("--head", type=int, default=8)
    ap.add_argument("--context", type=int, default=8192, help="tokens a conversation can hold (41 KB of VRAM each, 21 KB with --keyless, taken at the start: what it takes, the plan cannot spend on 4-bit matrices)")
    ap.add_argument("--no-drafter", action="store_true")
    ap.add_argument("--keyless", action="store_true", help="for long contexts: the full-attention layers keep their values only (21 KB a token "
                    "for 41) and the keys are made of them when read; other bits than without it, of the same precision")
    ap.add_argument("--rows", type=int, default=ROWS, choices=(8, 16, 32), help="tokens of a prompt read in one pass: 32 reads a prompt about 45%% "
                    "faster than 16 and takes 62 MB more VRAM (86 MB at a context of 16,384), which the plan cannot spend on 4-bit matrices")
    ap.add_argument("--reserve", type=float, default=VRAM_MARGIN / 2**30, help="GiB of the VRAM that is free now to leave to other "
                    "programs. Less means more matrices at 4 bits; if other programs then take more, Windows pages VRAM out and everything slows down")
    E.add_int8_option(ap)
    cli = ap.parse_args()
    E.use_int8_option(cli)
    reserve = int(cli.reserve * 2**30)
    console = Console()
    console.print("[grey62]모델을 올리는 중입니다 (약 15초)…[/]")
    import transformers
    from transformers import AutoTokenizer

    transformers.logging.set_verbosity_error()
    cap_vram(margin=reserve)
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    cfg = M.Config.load(ckpt("orig"))
    head, estimate = cli.head, None
    if cli.plan == "auto":
        # what is free now, less what PyTorch is kept from (cap_vram), the rest of the engine, and room to load in
        free = torch.cuda.mem_get_info()[0] - reserve
        rest = E.fixed_bytes(cfg, head, cli.context, rows=cli.rows, keyless=cli.keyless) + (0 if cli.no_drafter else E.DRAFTER_BYTES)
        budget = free - rest - LOAD_ROOM
        if budget < P.code_bytes(cfg, P.uniform(cfg, 6)):
            console.print(f"[yellow]VRAM이 모자랍니다: 지금 비어 있는 것은 {(free + reserve) / 2**30:.1f}GiB이고, 전부 3bit로도 "
                          f"{(P.code_bytes(cfg, P.uniform(cfg, 6)) + rest + LOAD_ROOM + reserve) / 2**30:.1f}GiB가 필요합니다. "
                          f"다른 프로그램을 닫거나 --context 를 줄여 보세요.[/]")
            return
        plan, estimate = P.fit(cfg, cli.name, budget, of=SHARES)
    elif cli.plan == "c6":
        plan = P.uniform(cfg, 6)
    else:
        plan, head = P.read(cli.plan)
        head = head or cli.head
    with torch.no_grad():
        eng = E.Engine.load(cli.name, plan, head, -(-cli.context // E.TILE) * E.TILE, rows=cli.rows, keyless=cli.keyless)
        drafter = None if cli.no_drafter else E.EngineDrafter.load(eng, ckpt("drafter"), cli.name, depth=DEPTH)
        opts = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "num_predict": 4096, "verbose": True, "showthink": False}
        session = Session(tok, eng, drafter)
        console.print(f"\n[bold]sclmd chat[/] · gemma-4-31B-it · VRAM {torch.cuda.memory_allocated() / 2**30:.1f}GiB · "
                      f"가중치 평균 {P.bits_per_weight(cfg, plan):.2f}bit" + (f" (직접 쓴 글에서 원본 대비 KL 약 {estimate:.3f})" if estimate else "")
                      + f"\n[grey62]{settings(session, opts)}\n/? 도움말 · /show 설정 보기 · /bye 끝내기[/]\n", highlight=False)
        while True:
            # the last reply goes into the context in its later form while the next message is typed
            ahead = threading.Thread(target=torch.no_grad()(session.settle))
            ahead.start()
            try:
                line = read_message(">>> ")
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                continue
            finally:
                ahead.join()
            if line.startswith("/"):
                if not command(line, session, opts, console):
                    break
            elif line.strip():
                reply(line, session, opts, console)


if __name__ == "__main__":
    main()
