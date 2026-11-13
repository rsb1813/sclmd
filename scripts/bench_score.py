# Scores the stored benchmark replies (scripts/bench.py) and writes the summary (data/bench_summary.json).
# Stands alone (no project imports): run it as  python -I scripts/bench_score.py
# Only the answer counts (what stands outside the thinking); a reply that ran out of context or into a loop is scored
# as it stands, which is nearly always wrong, and counted as cut.
#
# Programs the model wrote (HumanEval, LiveCodeBench) are run here: each in a process of its own, in an empty
# temporary folder, with a time limit, a memory limit and no child processes (a Windows job object), and with the
# calls that delete or move files or start programs taken away first. Results are kept by the program's hash, so
# scoring again does not run them again.
import argparse
import ctypes
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation

LETTERS = "ABCDEFGHIJ"
# the model card's scores for Gemma 4 31B (C:\models\gemma-4-31B-it\README.md); None: the card has no such row
CARD = {"mmlu_pro": 85.2, "aime_2026": 89.2, "lcb_v6": 80.0, "bbeh": 74.4, "bbeh_mini": 74.4, "mmmlu_ko": 88.4}
NOTE = {"bbeh_mini": "the card's score is on all 4,520 items; this is the benchmark's own 460-item slice",
        "mmmlu_ko": "the card's score is over all of MMMLU's languages; this is Korean only",
        "bbeh": "scored after the benchmark's evaluate.py", "lcb_v6": "the 175 problems release 6 added, as the card's counts are out of 175",
        "aime_2026": "the card's scores are counts out of 120: four runs of the 30 problems"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


# ---- answers out of text

def letter(text: str, options: int) -> str | None:
    ok = LETTERS[:options]
    for pattern in (r"(?i:answer)\s*:\s*[\*\$\(\\\s]*(?:text\{)?([A-J])(?![A-Za-z])", r"(?i:answer\s+is)\s*:?\s*[\*\$\(\\\s]*(?:text\{)?([A-J])(?![A-Za-z])",
                    r"정답[은는]?\s*:?\s*[\*\$\(\s]*([A-J])(?![A-Za-z])", r"boxed\{\s*(?:\\text\{)?\(?([A-J])\)?"):
        found = [m for m in re.findall(pattern, text) if m in ok]
        if found:
            return found[-1]
    return None


def number(text: str) -> str | None:
    """The number of the last 'Answer: ...' line, or of \\boxed{...}, or the last number in the text."""
    for pattern in (r"(?i:answer)\s*:\s*([^\n]*)", r"boxed\{([^{}]*)\}"):
        lines = re.findall(pattern, text)
        for line in reversed(lines):
            nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", line.replace("\\!", "").replace("\\,", ""))
            if nums:
                return nums[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same_number(a: str | None, b: str) -> bool:
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


# BigBench Extra Hard's own matching (bbeh/evaluate.py; compared with the repository's file on 2026-11-14, and its eight
# examples give the same results here)
def bbeh_answer(sample: str) -> str:
    answer = sample.strip()
    for prefix in ("The answer is:", "The final answer is ", "The final answer is: ", "The answer is "):
        if prefix in answer:
            answer = answer.split(prefix)[-1].strip()
    answer = answer[:-1] if answer.endswith(".") else answer
    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1]
    for opener in ("boxed{", "text{", "texttt{"):
        if opener in answer and answer.endswith("}"):
            answer = answer[:-1].split(opener)[1]
    answer = answer.lower().replace(", ", ",").replace("**", "").split("\n")[0]
    return answer[:-1] if answer.endswith(".") else answer


def bbeh_match(prediction: str, reference: str) -> bool:
    reference = reference.strip().lower().replace(", ", ",")
    if prediction == reference:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    return prediction.endswith("?") and prediction[:-1] == reference


# ---- running programs

GUARD = '''import os as _os, shutil as _shutil, subprocess as _subprocess
for _n in ("remove", "unlink", "rmdir", "removedirs", "rename", "renames", "replace", "system", "kill", "startfile", "truncate", "chdir",
           "spawnl", "spawnv", "spawnve", "execv", "execve", "popen"):
    if hasattr(_os, _n):
        setattr(_os, _n, None)
_shutil.rmtree = _shutil.move = _shutil.copytree = None
_subprocess.Popen = _subprocess.run = _subprocess.call = _subprocess.check_output = None
del _os, _shutil, _subprocess, _n
'''


# the interpreter itself: a virtual environment's python.exe is a launcher that starts it as a child, which the job forbids
PYTHON = getattr(sys, "_base_executable", None) or sys.executable


class _Basic(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32)]


class _Extended(ctypes.Structure):
    _fields_ = [("Basic", _Basic), ("IoInfo", ctypes.c_ulonglong * 6), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _job(memory: int):
    """A Windows job object: `memory` bytes a process, one process, everything killed when the handle closes."""
    if os.name != "nt":
        return None
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = ctypes.c_void_p
    k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    job = k32.CreateJobObjectW(None, None)
    info = _Extended()
    info.Basic.LimitFlags = 0x100 | 0x2000 | 0x8  # process memory, kill on close, active processes
    info.Basic.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = memory
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    return k32, job


RUNNER = "r2"  # in the key of a cached result: results of an earlier way of running programs are not used


def run_program(source: str, stdin: str, timeout: float, memory: int = 4 << 30) -> tuple[bool, str]:
    """Runs a Python program the model wrote -> (ended without error in time, its output)."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "program.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(GUARD + source)
        env = {k: v for k, v in os.environ.items() if k.upper() in ("SYSTEMROOT", "PATH", "TEMP", "TMP", "WINDIR")}
        proc = subprocess.Popen([PYTHON, "-I", "-X", "utf8", path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, cwd=folder, env=env)
        job = _job(memory)
        if job:
            job[0].AssignProcessToJobObject(job[1], int(proc._handle))
        try:
            out, _ = proc.communicate(stdin.encode("utf-8"), timeout=timeout)
            return proc.returncode == 0, out.decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return False, ""
        finally:
            if job:
                job[0].CloseHandle(job[1])


def code_of(answer: str, must_have: str | None = None) -> str | None:
    blocks = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", answer, flags=re.S)
    if must_have:
        blocks = [b for b in blocks if must_have in b] or blocks
    return blocks[-1] if blocks else None


def humaneval_passes(code: str, t: dict) -> bool:
    program = t["prompt"] + "\n\n" + code + "\n\n" + t["test"] + f"\n\ncheck({t['entry_point']})\n"
    return run_program(program, "", 20)[0]


def same_output(got: str, want: str) -> bool:
    """LiveCodeBench's comparison: line by line, each line stripped; else the lines' numbers as decimals."""
    a, b = [x.strip() for x in got.strip().split("\n")], [x.strip() for x in want.strip().split("\n")]
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            if [Decimal(v) for v in x.split()] != [Decimal(v) for v in y.split()]:
                return False
        except InvalidOperation:
            return False
    return True


DRIVER = '''
import json as _json, sys as _sys
_cases = _json.loads(_sys.stdin.read())
_ok = True
for _args, _want in _cases:
    _got = getattr(Solution(), {name!r})(*_args)
    if isinstance(_got, tuple):
        _got = list(_got)
    if _got != _want:
        _ok = False
        break
print("PASSED" if _ok else "FAILED")
'''
# What LiveCodeBench's own runner puts before every program (lcb_runner/evaluation/testing_util.py: import_string).
# The order matters: builtins come after math and operator, so that pow() is the built-in one with its three arguments.
LCB_IMPORTS = (
    'from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\n'
    'from copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n'
    'from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\n'
    'import string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\n'
    'import copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\n'
    'import operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n'
)
# ... and how it runs one: an exit() is not an error. The stack is 64 MB here: recursion through @cache needs more than
# the 2 MB of a Windows main thread (theirs has Linux's 8 MB), and Python's own limit is the header's 50,000.
LCB_RUN = '''
import sys as _sys, threading as _threading
_failed = []


def _run():
    try:
        exec(compile(_SOURCE, "solution.py", "exec"), {"__name__": "__main__", "__builtins__": __builtins__})
    except SystemExit:
        pass
    except BaseException:
        import traceback
        traceback.print_exc()
        _failed.append(True)


_threading.stack_size(64 << 20)
_thread = _threading.Thread(target=_run)
_thread.start()
_thread.join()
_sys.stdout.flush()
_sys.exit(1 if _failed else 0)
'''


def lcb_program(source: str) -> str:
    return f"_SOURCE = {LCB_IMPORTS + chr(10) + chr(10) + source!r}\n" + LCB_RUN


def lcb_passes(code: str, t: dict, per_case: float = 6.0) -> bool:
    cases = t["cases"]
    if cases and cases[0]["testtype"] == "functional":
        try:
            pairs = [([json.loads(line) for line in c["input"].split("\n")], json.loads(c["output"])) for c in cases]
        except json.JSONDecodeError:
            return False
        ok, out = run_program(lcb_program(code + DRIVER.format(name=t["func_name"])), json.dumps(pairs), per_case * len(cases))
        return ok and out.strip().endswith("PASSED")
    program = lcb_program(code)
    for c in cases:
        ok, out = run_program(program, c["input"], per_case)
        if not ok or not same_output(out, c["output"]):
            return False
    return True


# ---- tasks

def read(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_task(bench: str, task: str, tag: str, files: list, run_code: bool) -> dict:
    gold = {it["id"]: it for it in read(os.path.join(bench, "items", f"{task}.jsonl"))}
    tests = None
    cache_path = os.path.join(bench, "exec", f"{task}.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    rows, by_stratum = [], {}
    for path in files:
        # an item's last reply: bench.py answers again what the chat's loop guard had ended
        for r in {r["id"]: r for r in read(path)}.values():
            it = gold[r["id"]]
            answer = r.get("answer", "")
            got, right = None, False
            if "error" in r:
                pass
            elif task in ("mmlu_pro", "mmlu", "mmmlu_ko", "kmmlu"):
                got = letter(answer, it["options"])
                right = got == it["gold"]
            elif task == "gsm8k":
                got = number(answer)
                right = same_number(got, it["gold"])
            elif task == "aime_2026":
                got = number(answer)
                right = got is not None and got.isdigit() and int(got) == int(it["gold"])
            elif task.startswith("bbeh"):
                got = bbeh_answer(answer)
                right = bbeh_match(got, it["gold"])
            elif task in ("humaneval", "lcb_v6"):
                if tests is None:
                    tests = {t["id"]: t for t in read(os.path.join(bench, "items", f"{task}_tests.jsonl"))}
                t = tests[r["id"]]
                got = code_of(answer, f"def {t['entry_point']}" if task == "humaneval" else None)
                if got is not None:
                    key = f"{RUNNER}:{r['id']}:{hashlib.sha256(got.encode()).hexdigest()[:16]}"
                    if key not in cache and run_code:
                        cache[key] = humaneval_passes(got, t) if task == "humaneval" else lcb_passes(got, t)
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(cache, f)
                    if key not in cache:
                        continue  # not run yet: left out of the count
                    right = cache[key]
                    got = "code"
            rows.append((r, got, right))
            s = by_stratum.setdefault(it["stratum"], [0, 0])
            s[0] += right
            s[1] += 1
    n, k = len(rows), sum(right for _, _, right in rows)
    lo, hi = wilson(k, n)
    ok = [r for r, _, _ in rows if "error" not in r]
    tokens, seconds = sum(r["tokens"] for r in ok), sum(r["seconds"] for r in ok)
    out = {"task": task + tag, "runs": len(files), "items": n, "of": len(gold) * max(len(files), 1), "correct": k,
           "score": round(100 * k / max(n, 1), 1), "low": round(100 * lo, 1), "high": round(100 * hi, 1), "card": CARD.get(task),
           "no_answer": sum(got is None for r, got, _ in rows), "cut": sum(not r.get("ended", False) and not r.get("looped", False) for r in ok),
           "looped": sum(r.get("looped", False) for r in ok), "errors": n - len(ok),
           "over_8k": sum(r["prompt_tokens"] + r["tokens"] > 8192 - 8 for r in ok),
           "score_at_8k": round(100 * sum(right and r.get("prompt_tokens", 0) + r.get("tokens", 0) <= 8192 - 8 for r, _, right in rows) / max(n, 1), 1),
           "mean_tokens": round(tokens / max(len(ok), 1)), "mean_thinking": round(sum(r["think_tokens"] for r in ok) / max(len(ok), 1)),
           "tok_s": round(tokens / max(seconds, 1e-9), 1), "hours": round(seconds / 3600, 2)}
    if task in NOTE:
        out["note"] = NOTE[task]
    if len(by_stratum) > 1:
        out["by"] = {s: [c, m] for s, (c, m) in sorted(by_stratum.items())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bench"))
    ap.add_argument("--no-code", action="store_true", help="do not run programs now (score only those already run)")
    ap.add_argument("--smoke", action="store_true", help="the smoke runs (*.smoke.run*.jsonl) instead of the real ones")
    ap.add_argument("tasks", nargs="*")
    args = ap.parse_args()
    groups = {}
    for path in sorted(glob.glob(os.path.join(args.bench, "replies", "*.run*.jsonl"))):
        name = os.path.basename(path).rsplit(".run", 1)[0]
        task, _, tag = name.partition(".")
        if (tag == "smoke") != args.smoke or (args.tasks and task not in args.tasks):
            continue
        groups.setdefault((task, "." + tag if tag else ""), []).append(path)
    results = [score_task(args.bench, task, tag, files, not args.no_code) for (task, tag), files in groups.items()]
    for r in results:
        card = f"{r['card']:.1f}" if r["card"] is not None else "  - "
        print(f"{r['task']:12s} {r['score']:5.1f}% ({r['low']:.1f}-{r['high']:.1f}) card {card} | {r['correct']}/{r['items']} of {r['of']}, "
              f"no answer {r['no_answer']}, cut {r['cut']}, looped {r['looped']}, errors {r['errors']}, over 8k {r['over_8k']} "
              f"(score with an 8,192 context {r['score_at_8k']}%) | {r['mean_tokens']} tokens a reply "
              f"({r['mean_thinking']} thinking), {r['tok_s']} tok/s, {r['hours']} h", flush=True)
    if not args.smoke and not args.tasks:
        path = os.path.join(os.path.dirname(args.bench), "bench_summary.json")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"plan": "fit_own_v1_11.9.json (3.49 bits a weight, int8 keys and values: what the chat loads)", "context": 16384,
                       "sampling": "temperature 1.0, top_p 0.95, top_k 64, thinking on, one reply an item", "interval": "Wilson 95%",
                       "cut": "replies that ran out of the 16,384 tokens; over_8k: replies that would not have fitted the chat's default 8,192",
                       "tasks": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
