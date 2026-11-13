# A layer's matrices as pair codes: fitting them, keeping them on disk, and turning them back into weights.
#
# A matrix W [rows, cols] is stored in a rotated input basis (W R, R a block Hadamard transform with fixed signs; the
# input is rotated by R^T when the matrix is used). There each pair of adjacent columns is one code naming an entry of
# the matrix's codebook of pairs, and each row has one scale. 6 bits a code is 3 bits a weight, 8 bits is 4.
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from . import MODELS
from . import quant as Q

CODES = os.path.join(MODELS, "sclmd-codes")
GROUP = {"q": "attn_in", "k": "attn_in", "v": "attn_in", "o": "o_in", "gate": "mlp_in", "up": "mlp_in", "down": "down_in"}


def rotation(cols: int, device="cuda") -> Q.BlockHadamard:
    """The input rotation of every matrix with `cols` columns (the same for all of them)."""
    return Q.BlockHadamard(cols, seed=17 + cols, device=device)


def rotated_hessian(rot: Q.BlockHadamard, h: torch.Tensor) -> torch.Tensor:
    """R^T H R. Uses up h."""
    h = rot.rotate(h).t().contiguous()
    return rot.rotate(h)


def pair_codebook(w: torch.Tensor, bits: int) -> torch.Tensor:
    """k-means over the row-normalized weight's adjacent pairs: [2**bits, 2]."""
    torch.manual_seed(0)
    wn, _ = Q.row_normalize(w)
    pairs = wn.reshape(-1, 2)
    sample = pairs[torch.randperm(pairs.shape[0], device=w.device)[: 1 << 21]]
    return Q.kmeans(sample, 1 << bits, iters=25)


def fit(w: torch.Tensor, factor: tuple, bits: int, book: torch.Tensor | None = None) -> dict:
    """Pair codes of w [rows, cols] under a rotated Hessian given as Q.gptq_factor's result (shared by the matrices
    with that input). book: the codebook to use (a matrix fitted a part of its rows at a time has one for all of them).
    Returns codes (uint8 [rows, cols / 2]), scales (fp32 [rows]), book (fp32 [2**bits, 2]), error: the share of the
    output's energy that is lost, tr(D H D^T) / tr(W H W^T) with D the difference, and energy: tr(W H W^T). Both are
    under the damped Hessian the fit used and on the inputs it was fitted to: good for ranking, not a held-out error."""
    rows, cols = w.shape
    hr = factor[1]
    wr = rotation(cols, w.device).rotate(w)
    if book is None:
        book = pair_codebook(wr, bits)
    codes, scales = Q.gptq_vq(wr, None, book, cols, refit=True, factor=factor)
    d = Q.vq_codes_dequant(codes, scales, book) - wr
    energy = ((wr @ hr) * wr).sum().item()
    error = ((d @ hr) * d).sum().item() / energy
    return {"codes": codes.to(torch.uint8), "scales": scales.reshape(rows).float(), "book": book.float(), "error": error, "energy": energy}


def weight(codes: torch.Tensor, scales: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
    """The matrix the codes stand for, back in the model's own input basis (fp32)."""
    wr = Q.vq_codes_dequant(codes.long(), scales.float()[:, None], book.float())
    return rotation(wr.shape[1], wr.device).unrotate(wr)


def layer_path(name: str, layer: int) -> str:
    return os.path.join(CODES, name, f"layer_{layer:02d}.safetensors")


def save_layer(name: str, layer: int, fits: dict, info: dict):
    """fits: (matrix, bits) -> fit()'s result. info goes to a JSON file beside the tensors."""
    path = layer_path(name, layer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensors = {}
    for (mat, bits), f in fits.items():
        for key in ("codes", "scales", "book"):
            tensors[f"{mat}.c{bits}.{key}"] = f[key].cpu().contiguous()
    save_file(tensors, path + ".tmp")
    with open(path[: -len(".safetensors")] + ".json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    os.replace(path + ".tmp", path)  # the layer counts as done once this file is there


def load_layer(name: str, layer: int, bits: dict, device="cuda") -> dict:
    """bits: matrix -> 6 or 8. Returns matrix -> weight (fp32) for the matrices named."""
    t = load_file(layer_path(name, layer), device=device)
    return {mat: weight(t[f"{mat}.c{b}.codes"], t[f"{mat}.c{b}.scales"], t[f"{mat}.c{b}.book"]) for mat, b in bits.items()}


def layer_info(name: str, layer: int) -> dict | None:
    path = layer_path(name, layer)
    if not os.path.exists(path):
        return None
    with open(path[: -len(".safetensors")] + ".json", encoding="utf-8") as fh:
        return json.load(fh)


def stored(name: str, layer: int | str, mats: dict, device="cpu") -> dict:
    """A layer's (or with layer "head", the output head's) matrices as stored: matrix -> (codes, scales, book, bits),
    for mats: matrix -> bits. The head is the one matrix "head"."""
    path = head_path(name) if layer == "head" else layer_path(name, layer)
    with safe_open(path, framework="pt", device=device) as f:
        key = (lambda m, b: f"c{b}") if layer == "head" else (lambda m, b: f"{m}.c{b}")
        return {m: (f.get_tensor(key(m, b) + ".codes"), f.get_tensor(key(m, b) + ".scales"), f.get_tensor(key(m, b) + ".book"), b)
                for m, b in mats.items()}


# The output head (the embedding table used as a matrix): one file, both rates.

def head_path(name: str) -> str:
    return os.path.join(CODES, name, "head.safetensors")


def save_head(name: str, fits: dict, info: dict):
    """fits: bits -> {codes, scales, book}."""
    path = head_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({f"c{bits}.{key}": f[key].cpu().contiguous() for bits, f in fits.items() for key in ("codes", "scales", "book")}, path + ".tmp")
    with open(path[: -len(".safetensors")] + ".json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    os.replace(path + ".tmp", path)


def load_head(name: str, bits: int, device="cuda", dtype=torch.float32, rows: int = 32768) -> torch.Tensor:
    """The head's weight as its codes give it, in the model's own basis ([vocab, hidden]; built a part at a time: the
    float32 matrix alone is 5.6 GB)."""
    codes, scales, book, _ = stored(name, "head", {"head": bits})["head"]
    out = torch.empty(codes.shape[0], codes.shape[1] * 2, dtype=dtype, device=device)
    for a in range(0, codes.shape[0], rows):
        out[a : a + rows] = weight(codes[a : a + rows].to(device), scales[a : a + rows].to(device), book.to(device)).to(dtype)
    return out


# The drafter's matrices: one file, both rates. Names: pre, post, head, layers.<i>.<q|o|gate|up|down>.

def drafter_path(name: str) -> str:
    return os.path.join(CODES, name, "drafter.safetensors")


def save_drafter(name: str, fits: dict, info: dict):
    """fits: (matrix, bits) -> {codes, scales, book}."""
    path = drafter_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({f"{m}.c{bits}.{key}": f[key].cpu().contiguous() for (m, bits), f in fits.items() for key in ("codes", "scales", "book")},
              path + ".tmp")
    with open(path[: -len(".safetensors")] + ".json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    os.replace(path + ".tmp", path)


def drafter_stored(name: str, bits: int, device="cpu") -> dict:
    """Every matrix of the drafter as stored: name -> (codes, scales, book, bits)."""
    out = {}
    with safe_open(drafter_path(name), framework="pt", device=device) as f:
        for key in f.keys():
            m, rate, part = key.rsplit(".", 2)
            if rate == f"c{bits}" and part == "codes":
                out[m] = (f.get_tensor(key), f.get_tensor(f"{m}.{rate}.scales"), f.get_tensor(f"{m}.{rate}.book"), bits)
    return out


def load_drafter_head(name: str, bits: int, device="cuda", dtype=torch.float32, rows: int = 32768) -> torch.Tensor:
    """The drafter's head as its codes give it ([vocab, hidden]), built a part at a time."""
    with safe_open(drafter_path(name), framework="pt", device="cpu") as f:
        codes, scales, book = (f.get_tensor(f"head.c{bits}.{part}") for part in ("codes", "scales", "book"))
    out = torch.empty(codes.shape[0], codes.shape[1] * 2, dtype=dtype, device=device)
    for a in range(0, codes.shape[0], rows):
        out[a : a + rows] = weight(codes[a : a + rows].to(device), scales[a : a + rows].to(device), book.to(device)).to(dtype)
    return out
