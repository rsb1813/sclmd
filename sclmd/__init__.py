# sclmd: Gemma 4 31B (dense) on one 16 GB GPU. Paths shared by the package and the scripts.
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MODELS = os.environ.get("SCLMD_MODELS", r"C:\models")

# Hugging Face repo -> folder under MODELS. The original is what loss is measured against.
REPOS = {
    "orig": "google/gemma-4-31B-it",
    "drafter": "google/gemma-4-31B-it-assistant",
    "qat": "google/gemma-4-31B-it-qat-q4_0-unquantized",
    "qat_drafter": "google/gemma-4-31B-it-qat-q4_0-unquantized-assistant",
}


def ckpt(name: str) -> str:
    """Local folder of one of REPOS."""
    return os.path.join(MODELS, REPOS[name].split("/")[1])
