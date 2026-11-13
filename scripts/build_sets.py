# Builds a token set (evaluation or calibration): fixed-length sequences from licensed public sources, plus a manifest.
# Chats go through the model's own chat template; the math category carries its worked solution as the thinking.
import argparse
import glob
import json
import os
import re

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from sclmd import DATA, ckpt

SEQ_LEN = 2048
DOC_MAX = 1024  # cap tokens per document so one long document cannot fill a category

# name -> (number of SEQ_LEN sequences at scale 1, source description, license)
PLAN = {
    "en_web": (2, "HuggingFaceFW/fineweb-edu sample-10BT", "ODC-By"),
    "ko_wiki": (2, "wikimedia/wikipedia 20231101.ko", "CC BY-SA 3.0"),
    "multi_wiki": (1, "wikimedia/wikipedia 20231101.{ja,zh,de,fr}", "CC BY-SA 3.0"),
    "chat_en": (2, "HuggingFaceH4/ultrachat_200k train_sft", "MIT"),
    "chat_ko": (2, "nlpai-lab/kullm-v2", "Apache-2.0"),
    "code_inst": (2, "m-a-p/CodeFeedback-Filtered-Instruction", "Apache-2.0"),
    "code_raw": (1, "local CPython 3.14 standard library sources", "PSF-2.0"),
    "math": (2, "openai/gsm8k main", "MIT"),
    "tools": (2, "glaiveai/glaive-function-calling-v2", "Apache-2.0"),
}


def stream(name: str, config: str | None = None, split: str = "train"):
    return load_dataset(name, config, split=split, streaming=True)


def docs(category: str, tok):
    """Yield token-id lists, one per document, for `category`."""

    def text(t: str):
        return [tok.bos_token_id] + tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]

    def chat(messages, thinking: bool = False):
        out = tok.apply_chat_template(messages, tokenize=True, enable_thinking=thinking)
        return list(out["input_ids"] if isinstance(out, dict) or hasattr(out, "keys") else out)

    if category == "en_web":
        for r in stream("HuggingFaceFW/fineweb-edu", "sample-10BT"):
            yield text(r["text"])
    elif category == "ko_wiki":
        for r in stream("wikimedia/wikipedia", "20231101.ko"):
            yield text(r["text"])
    elif category == "multi_wiki":
        sources = [iter(stream("wikimedia/wikipedia", f"20231101.{lang}")) for lang in ("ja", "zh", "de", "fr")]
        while True:  # round-robin so every language gets a share
            for s in sources:
                yield text(next(s)["text"])
    elif category == "chat_en":
        for r in stream("HuggingFaceH4/ultrachat_200k", split="train_sft"):
            yield chat(r["messages"])
    elif category == "chat_ko":
        for r in stream("nlpai-lab/kullm-v2"):
            prompt = r["instruction"] + ("\n\n" + r["input"] if r.get("input") else "")
            yield chat([{"role": "user", "content": prompt}, {"role": "assistant", "content": r["output"]}])
    elif category == "code_inst":
        for r in stream("m-a-p/CodeFeedback-Filtered-Instruction"):
            yield chat([{"role": "user", "content": r["query"]}, {"role": "assistant", "content": r["answer"]}])
    elif category == "code_raw":
        for path in sorted(glob.glob(r"C:\Python314\Lib\*.py")):
            with open(path, encoding="utf-8", errors="replace") as f:
                yield text(f.read())
    elif category == "math":
        for r in stream("openai/gsm8k", "main"):
            steps, _, result = r["answer"].partition("####")
            yield chat([{"role": "user", "content": r["question"]},
                        {"role": "assistant", "reasoning_content": steps.strip(), "content": result.strip()}], thinking=True)
    elif category == "tools":
        for r in stream("glaiveai/glaive-function-calling-v2"):
            messages = [{"role": "system", "content": r["system"].removeprefix("SYSTEM:").strip()}]
            for role, content in re.findall(r"(USER|ASSISTANT|FUNCTION RESPONSE):(.*?)(?=(?:USER|ASSISTANT|FUNCTION RESPONSE):|$)", r["chat"], re.S):
                content = content.replace("<|endoftext|>", "").strip()
                if role == "FUNCTION RESPONSE":  # kept as a tagged user turn: the source has no structured tool calls
                    role, content = "USER", "[function response]\n" + content
                messages.append({"role": "user" if role == "USER" else "assistant", "content": content})
            yield chat(messages)
    else:
        raise KeyError(category)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="eval_v1")
    ap.add_argument("--scale", type=int, default=1, help="multiply every category's sequence count")
    ap.add_argument("--skip", help="earlier set (e.g. eval_v1) whose documents to skip, so the two are disjoint")
    cli = ap.parse_args()
    out = os.path.join(DATA, cli.name)
    skip = {}
    if cli.skip:
        with open(os.path.join(DATA, cli.skip + ".json"), encoding="utf-8") as f:
            skip = {c: v.get("skipped", 0) + v["documents"] for c, v in json.load(f)["categories"].items()}
    tok = AutoTokenizer.from_pretrained(ckpt("orig"))
    os.makedirs(out, exist_ok=True)
    seqs, cats, manifest = [], [], {"seq_len": SEQ_LEN, "doc_max": DOC_MAX, "skip": cli.skip, "categories": {}}
    for ci, (category, (n_seq, source, license_)) in enumerate(PLAN.items()):
        n_seq *= cli.scale
        need, buf, n_docs, n_skip = n_seq * SEQ_LEN, [], 0, 0
        for ids in docs(category, tok):
            if len(ids) < 32:
                continue
            if n_skip < skip.get(category, 0):
                n_skip += 1
                continue
            buf.extend(ids[:DOC_MAX])
            n_docs += 1
            if len(buf) >= need:
                break
        arr = np.array(buf[:need], dtype=np.int32).reshape(n_seq, SEQ_LEN)
        seqs.append(arr)
        cats += [ci] * n_seq
        manifest["categories"][category] = {"source": source, "license": license_, "sequences": n_seq, "documents": n_docs,
                                            "skipped": n_skip}
        print(f"{category:10s} {n_seq} seqs from {n_docs} docs", flush=True)
    tokens = np.concatenate(seqs)
    np.savez(os.path.join(out, "tokens.npz"), tokens=tokens, category=np.array(cats, dtype=np.int8), names=np.array(list(PLAN)))
    manifest["total_tokens"] = int(tokens.size)
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("total tokens", tokens.size)


if __name__ == "__main__":
    main()
