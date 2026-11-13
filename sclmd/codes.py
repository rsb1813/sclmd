# A layer's matrices as pair codes: fitting them, keeping them on disk, and turning them back into weights.
#
# A matrix W [rows, cols] is stored in a rotated input basis (W R, R a block Hadamard transform with fixed signs; the
# input is rotated by R^T when the matrix is used). There each pair of adjacent columns is one code naming an entry of
# the matrix's codebook of pairs, and each row has one scale. 6 bits a code is 3 bits a weight, 8 bits is 4.
import json
import os

import torch
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


def fit(w: torch.Tensor, hr: torch.Tensor, bits: int) -> dict:
    """Pair codes of w [rows, cols] under the rotated Hessian hr (not modified).
    Returns codes (uint8 [rows, cols / 2]), scales (fp32 [rows]), book (fp32 [2**bits, 2]) and error: the share of the
    output's energy that is lost, tr(D H D^T) / tr(W H W^T) with D the difference."""
    rows, cols = w.shape
    wr = rotation(cols, w.device).rotate(w)
    book = pair_codebook(wr, bits)
    codes, scales = Q.gptq_vq(wr, hr.clone(), book, cols, refit=True, own_h=True)
    d = Q.vq_codes_dequant(codes, scales, book) - wr
    error = ((d @ hr) * d).sum().item() / ((wr @ hr) * wr).sum().item()
    return {"codes": codes.to(torch.uint8), "scales": scales.reshape(rows).float(), "book": book.float(), "error": error}


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
