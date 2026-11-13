# sclmd

Gemma 4 31B (dense) on a single RTX 5080 16 GB: as close to the original as the VRAM allows, as fast as the memory bandwidth allows.
Every weight sits in VRAM as 3- or 4-bit pair codes and is multiplied straight from the codes on the tensor cores; a small drafter guesses six tokens ahead and the main model verifies them in one pass, which leaves the text unchanged.

- `chat.bat`: the terminal chat (`/?` lists its commands). It picks how many matrices get 4 bits from the VRAM that is free when it starts.
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

Checks: `python -m pytest` (small random models, a GPU for the engine's tests), `python -m scripts.engine_check` (the real model on the engine against the same codes in PyTorch and against the original), `python -m scripts.say "..." --check` (text with the drafter equals text without), `python -m scripts.long_recall` (facts from the start and middle of a 7,000-token prompt).
