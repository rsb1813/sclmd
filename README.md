# sclmd

Gemma 4 31B (dense) on a single RTX 5080 16 GB: as close to the original as the VRAM allows, as fast as the memory bandwidth allows.
Every weight sits in VRAM as 3- or 4-bit pair codes and is multiplied straight from the codes on the tensor cores; a small drafter guesses seven tokens ahead and the main model verifies them in one pass, which leaves the text unchanged.

- `chat.bat`: the terminal chat (`/?` lists its commands). It picks how many matrices get 4 bits from the VRAM that is free when it starts.
- `serve.bat`: an OpenAI-compatible server for several users at once (`http://127.0.0.1:8001/v1`; `-h` lists its options): every pass of the model carries tokens of all requests under way, 137 tok/s for one reply and 546 together for eight, and a prompt is read at about 900 tokens a second (3 bits a weight; `--kv-gib 0.68` gives the chat's 3.48 bits and room for fewer conversations).
- `design.md`: the model, the budget, and what was measured (Korean)
- `checklist.md`: what is done and what is being done now (Korean)

From scratch, in this order (Python runs from the parent project's environment, `..\.venv`; weights are never committed):

| Step | Command | Takes |
|---|---|---|
| Checkpoints into `C:\models` (`SCLMD_MODELS` to change) | `python -m scripts.download orig drafter` (without names: Google's QAT checkpoints too, which nothing here needs any more) | 64 GB (126 GB) |
| The token sets | `python -m scripts.build_sets` | |
| The original's outputs over the evaluation set (the reference) | `python -m scripts.reference`, and again with `--quant fp32` | 2 min each |
| Pair codes for every layer, at 3 and 4 bits | `python -m scripts.quantize` | 23 min, 28 GB |
| The output head's and the drafter's codes | `python -m scripts.quantize_head`, `python -m scripts.quantize_drafter` | 3 min each |
| The engine | `engine\build.bat` (CUDA toolkit and MSVC) | |
| What each matrix is worth at 4 bits, on text the model writes itself (the reports are already in `data/`): a plan to write with, the text, the original over it, the measurements | `python -m scripts.plans`, `sh scripts/sweep.sh`, `python -m scripts.allocate --gib 11.1`, `python -m scripts.own_text`, `python -m scripts.reference --set own_v1` (and again with `--quant fp32`), `sh scripts/sweep_own.sh` | 50 + 25 min |
| Chat | `chat.bat` | 15 s to load |

Checks: `python -m pytest` (small random models, a GPU for the engine's tests; the server's HTTP side without a model), `python -m scripts.serve_canned 8017` with `uv run --no-project --with openai python scripts/openai_client_check.py 8017` (the official openai client against the server's protocol, a canned reply in the engine's place), `python -m scripts.engine_check` (the real model on the engine against the same codes in PyTorch and against the original), `python -m scripts.say "..." --check` (text with the drafter equals text without), `python -m scripts.long_recall` (facts from the start and middle of a 7,000-token prompt), `python -m scripts.batch_bench` (several texts at once, each the text it is alone), `python scripts/serve_bench.py` (clients against a running server), `python -m scripts.speed_check --against base` (fixed prompts and seeds: the same tokens as before a change, and how fast; `--chat` through the chat's own loop), `python -m scripts.bits_check --against base` (two builds of the engine give the same bits: `SCLMD_DLL` picks the build), `python -m scripts.prefill_check` (how fast a prompt is read, and that what it leaves is the same bits; `python scripts/serve_bench.py --long 1500,6000` for the same through a running server).

Benchmarks: `python -I scripts/bench_prepare.py --out data/bench` (the questions as local prompt files), `python -m scripts.bench --task mmlu_pro --limit 140` (the replies; `sh scripts/bench_queue.sh <this folder>` runs `data/bench/queue.txt` and can be paused with `logs/pause`), `python -I scripts/bench_score.py` (scores, `data/bench_summary.json`), `python -m scripts.bench_paired --side ours | orig | score` (the original and the engine on the same MMLU-Pro questions, the answer's letter without thinking: 71.3% and 70.0% on 2,800).
