# The benchmarks' questions as prompts, written once to <out>/items/<task>.jsonl so that the runs themselves read only
# local files (scripts/bench.py). Stands alone (no project imports): run it as  python -I scripts/bench_prepare.py
# The order of a task's items is fixed (shuffled within each category with a fixed seed, then one category after
# another in turn), so the first N items are a balanced slice and a longer run extends a shorter one.
#
# Tasks (model card: MMLU-Pro, AIME 2026, LiveCodeBench v6, BigBench Extra Hard, MMMLU; others: MMLU, KMMLU, GSM8K,
# HumanEval). Not here: GPQA Diamond and HLE (gated datasets this account has no access to), Tau2 (needs a second
# model that plays the user), MRCR 128k (the context does not fit the VRAM), the vision and audio sets.
import argparse
import base64
import io
import json
import os
import pickle
import random
import zlib

from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download

LETTERS = "ABCDEFGHIJ"
MC = ("Answer the following multiple choice question{about}. The last line of your response should be of the following format: "
      "'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.\n\n{question}\n\n{options}")
NUMBER = ("Solve the following math problem. Think step by step. The last line of your response should be of the form "
          "'Answer: $NUMBER' (without quotes), where NUMBER is the final numeric answer without units.\n\n{question}")
AIME = ("Solve the following math problem. The answer is an integer from 0 to 999. Think step by step. The last line of your "
        "response should be of the form 'Answer: $INTEGER' (without quotes).\n\n{question}")
HUMANEVAL = ("Complete the following Python function. Write the whole function, with any imports it needs, in a single "
             "```python code block.\n\n```python\n{prompt}\n```")
# LiveCodeBench's own prompt (lcb_runner/prompts/code_generation.py)
LCB_SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will generate a "
              "correct Python program that matches the specification and passes all tests.")
LCB_STARTER = ("You will use the following starter code to write the solution to the problem and enclose your code within "
               "delimiters.\n```python\n{starter}\n```\n\n")
LCB_STDIN = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample "
             "inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the "
             "inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n")
# BigBench Extra Hard's suggested suffix (its README)
BBEH = ("{input}\n\nThink step by step, and when you provide the final answer, please use the prefix \"The answer is:\" without any "
        "modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer "
        "is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must "
        "only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\"")


def mc(question: str, options: list, about: str = "") -> str:
    return MC.format(about=f" about {about}" if about else "", letters=LETTERS[: len(options)], question=question.strip(),
                     options="\n".join(f"{LETTERS[i]}) {str(o).strip()}" for i, o in enumerate(options)))


def in_turn(items: list) -> list:
    """Shuffled within each stratum with a fixed seed, then the strata in turn."""
    by = {}
    for it in items:
        by.setdefault(it["stratum"], []).append(it)
    for s in sorted(by):
        random.Random(f"{s}").shuffle(by[s])
    out, depth = [], 0
    while len(out) < len(items):
        out += [by[s][depth] for s in sorted(by) if depth < len(by[s])]
        depth += 1
    return out


class NoGlobals(pickle.Unpickler):
    """Unpickles plain data only: LiveCodeBench's private tests are a pickled string, and nothing else is let through."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def lcb_tests(text: str) -> list:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raw = zlib.decompress(base64.b64decode(text.encode("utf-8")))
        return json.loads(NoGlobals(io.BytesIO(raw)).load())


def mmlu_pro():
    for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test"):
        yield {"id": str(r["question_id"]), "stratum": r["category"], "prompt": mc(r["question"], r["options"], r["category"]),
               "gold": r["answer"], "options": len(r["options"])}


def mmlu():
    for n, r in enumerate(load_dataset("cais/mmlu", "all", split="test")):
        yield {"id": str(n), "stratum": r["subject"], "prompt": mc(r["question"], r["choices"]), "gold": LETTERS[r["answer"]], "options": 4}


def mmmlu_ko():
    for n, r in enumerate(load_dataset("openai/MMMLU", "KO_KR", split="test")):
        yield {"id": str(n), "stratum": r["Subject"], "prompt": mc(r["Question"], [r[c] for c in "ABCD"]), "gold": r["Answer"], "options": 4}


def kmmlu():
    for cfg in get_dataset_config_names("HAERAE-HUB/KMMLU"):
        for n, r in enumerate(load_dataset("HAERAE-HUB/KMMLU", cfg, split="test")):
            yield {"id": f"{cfg}/{n}", "stratum": cfg, "prompt": mc(r["question"], [r[c] for c in "ABCD"]),
                   "gold": LETTERS[int(r["answer"]) - 1], "options": 4}


def gsm8k():
    for n, r in enumerate(load_dataset("openai/gsm8k", "main", split="test")):
        yield {"id": str(n), "stratum": "all", "prompt": NUMBER.format(question=r["question"].strip()),
               "gold": r["answer"].split("####")[-1].strip().replace(",", "")}


def aime_2026():
    for r in load_dataset("MathArena/aime_2026", split="train"):
        yield {"id": str(r["problem_idx"]), "stratum": "all", "prompt": AIME.format(question=r["problem"].strip()), "gold": str(r["answer"]).strip()}


def humaneval():
    for r in load_dataset("openai/openai_humaneval", split="test"):
        yield {"id": r["task_id"], "stratum": "all", "prompt": HUMANEVAL.format(prompt=r["prompt"].rstrip()), "gold": None,
               "tests": {"prompt": r["prompt"], "test": r["test"], "entry_point": r["entry_point"]}}


def lcb_v6():
    """The 175 problems release 6 added (the model card's scores are counts out of 175)."""
    path = hf_hub_download("livecodebench/code_generation_lite", "test6.jsonl", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fmt = LCB_STARTER.format(starter=r["starter_code"]) if r["starter_code"] else LCB_STDIN
            prompt = f"### Question:\n{r['question_content']}\n\n### Format: {fmt}### Answer: (use the provided format with backticks)\n\n"
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
            yield {"id": r["question_id"], "stratum": r["difficulty"], "system": LCB_SYSTEM, "prompt": prompt, "gold": None,
                   "platform": r["platform"], "date": r["contest_date"],
                   "tests": {"func_name": meta.get("func_name"), "cases": json.loads(r["public_test_cases"]) + lcb_tests(r["private_test_cases"])}}


def bbeh(mini: bool):
    def rows():
        for n, r in enumerate(load_dataset("BBEH/bbeh", split="train")):
            if not mini or str(r["mini"]) == "1":
                yield {"id": str(n), "stratum": r["task"], "prompt": BBEH.format(input=r["input"]), "gold": r["target"]}
    return rows


TASKS = {"mmlu_pro": mmlu_pro, "mmlu": mmlu, "mmmlu_ko": mmmlu_ko, "kmmlu": kmmlu, "gsm8k": gsm8k, "aime_2026": aime_2026,
         "humaneval": humaneval, "lcb_v6": lcb_v6, "bbeh_mini": bbeh(True), "bbeh": bbeh(False)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="the benchmark folder (items land in <out>/items)")
    ap.add_argument("tasks", nargs="*", default=list(TASKS))
    args = ap.parse_args()
    folder = os.path.join(args.out, "items")
    os.makedirs(folder, exist_ok=True)
    for name in args.tasks:
        try:
            items = in_turn(list(TASKS[name]()))
        except Exception as e:  # noqa: BLE001  (a set that cannot be read must not stop the others)
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        tests = [{"id": it["id"], **it.pop("tests")} for it in items if "tests" in it]
        with open(os.path.join(folder, f"{name}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        if tests:  # kept apart: the runs do not need them, and some are large
            with open(os.path.join(folder, f"{name}_tests.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                for t in tests:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"{name}: {len(items)} items in {len({it['stratum'] for it in items})} strata"
              + (f", tests for {len(tests)}" if tests else ""), flush=True)


if __name__ == "__main__":
    main()
