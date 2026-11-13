# Reads tensors out of a safetensors checkpoint folder by name, a few at a time.
import json
import os

import torch
from safetensors import safe_open

from . import model as M


class Checkpoint:
    def __init__(self, folder: str):
        self.folder = folder
        index = os.path.join(folder, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index, encoding="utf-8") as f:
                self.where = json.load(f)["weight_map"]
        else:
            with safe_open(os.path.join(folder, "model.safetensors"), framework="pt") as f:
                self.where = {name: "model.safetensors" for name in f.keys()}
        self.files = {}
        # the text decoder sits under "model.language_model." in the multimodal checkpoint and under "model." in the drafter
        self.prefix = next(p for p in ("model.language_model.", "model.") if f"{p}layers.0.input_layernorm.weight" in self.where)
        self.cfg = M.Config.load(folder)

    def get(self, name: str, device="cpu") -> torch.Tensor:
        file = self.where[name]
        if file not in self.files:
            self.files[file] = safe_open(os.path.join(self.folder, file), framework="pt")
        return self.files[file].get_tensor(name).to(device)

    def layer(self, i: int, device="cpu", skip: tuple = ()) -> dict:
        """Layer i's weights by the short names of model.LAYER_KEYS, without those in `skip`."""
        names = {short: f"{self.prefix}layers.{i}.{name}" for short, name in M.LAYER_KEYS.items() if short not in skip}
        return {short: self.get(name, device) for short, name in names.items() if name in self.where}

    def embed(self, device="cpu") -> torch.Tensor:
        return self.get(f"{self.prefix}embed_tokens.weight", device)

    def norm(self, device="cpu") -> torch.Tensor:
        return self.get(f"{self.prefix}norm.weight", device)

    def close(self):
        self.files.clear()
