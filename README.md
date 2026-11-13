# sclmd

Gemma 4 31B (dense) on a single RTX 5080 16 GB: as close to the original as the VRAM allows, as fast as the memory bandwidth allows.

- `design.md`: the model, the budget and the order of work (Korean)
- `checklist.md`: what is done and what is being done now (Korean)
- `python -m scripts.download`: fetches the checkpoints into `C:\models` (set `SCLMD_MODELS` to change)

Python runs from the parent project's environment (`..\.venv`). Weights are never committed.
