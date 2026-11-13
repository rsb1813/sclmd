# Which matrices get 4 bits a weight instead of 3: a plan is layer -> {matrix: bits a pair (6 or 8)}.
#
# What an upgrade is worth is measured, not assumed (scripts/sweep.sh): the whole model's KL with everything at 3
# bits, with one matrix type at 4 in every layer, and with one block of ten layers at 4. Two things the numbers say:
#  - KL goes with the square root of the error: at 4 bits every matrix's error is a quarter, and the KL is half.
#    So a matrix's share of the squared KL is what counts, and shares add up.
#  - A measured KL moves by about 0.005 from one plan to the next for no reason (a few positions tip over), so the
#    shares are rough: a type's share times a block's share is as fine as they are taken.
import json
import os

from . import DATA
from . import model as M

BLOCK = 10  # layers a measured block


def sizes(cfg: M.Config, i: int) -> dict:
    """Weights of layer i's matrices."""
    t = cfg.layer_types[i]
    qo, kv = cfg.heads * cfg.head_dim[t] * cfg.hidden, cfg.kv_heads[t] * cfg.head_dim[t] * cfg.hidden
    out = {"q": qo, "k": kv, "v": kv, "o": qo, "gate": cfg.mlp * cfg.hidden, "up": cfg.mlp * cfg.hidden, "down": cfg.mlp * cfg.hidden}
    if t == M.FULL:
        del out["v"]
    return out


def uniform(cfg: M.Config, bits: int) -> dict:
    return {i: {m: bits for m in sizes(cfg, i)} for i in range(cfg.n_layers)}


def code_bytes(cfg: M.Config, plan: dict) -> int:
    """VRAM of the layers' codes under a plan."""
    return sum(size * plan[i][m] // 16 for i in range(cfg.n_layers) for m, size in sizes(cfg, i).items())


def bits_per_weight(cfg: M.Config, plan: dict) -> float:
    return code_bytes(cfg, plan) * 8 / sum(s for i in range(cfg.n_layers) for s in sizes(cfg, i).values())


def read(path: str) -> tuple[dict, int | None]:
    """(plan, the head's bits or None) from a plan file."""
    with open(path, encoding="utf-8") as f:
        p = json.load(f)
    return {int(k): v for k, v in p["layers"].items()}, p.get("head")


def write(path: str, plan: dict, head: int | None = None, note: str = "") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"note": note, "head": head, "layers": {str(i): mats for i, mats in plan.items()}}, f, indent=1)
    return path


def measured(name: str, tag: str, against: str = "orig_fp32") -> float:
    with open(os.path.join(DATA, f"codes_{name}_{tag}.json"), encoding="utf-8") as f:
        return json.load(f)["against"][against]["all"]["kl"]


def shares(cfg: M.Config, name: str, against: str = "orig_fp32") -> tuple[dict, float]:
    """((layer, matrix) -> its share of the squared KL at 3 bits everywhere, that KL).
    A type's (or block's) share follows from the KL with it alone at 4 bits: three quarters of its share go away."""
    base = measured(name, f"l0-{cfg.n_layers - 1}_c6", against)
    share = lambda tag: max((1 - (measured(name, tag, against) / base) ** 2) / 0.75, 0.0)  # noqa: E731
    by_type = {m: share(f"c6_{m}8") for m in M.LINEARS}
    by_block = {b: share(f"c6_l{b}-{b + BLOCK - 1}_8") for b in range(0, cfg.n_layers, BLOCK)}
    t_sum, b_sum = sum(by_type.values()), sum(by_block.values())
    out = {}
    for b in by_block:
        layers = range(b, min(b + BLOCK, cfg.n_layers))
        for m in M.LINEARS:
            have = [i for i in layers if m in sizes(cfg, i)]
            for i in have:
                out[(i, m)] = by_type[m] / t_sum * by_block[b] / b_sum / len(have)
    return out, base


def fit(cfg: M.Config, name: str, budget: int, against: str = "orig_fp32") -> tuple[dict, float]:
    """The plan for `budget` bytes of layer codes: 3 bits everywhere, then 4 bits for the matrices with the largest
    share a byte, while they fit -> (plan, the KL the shares predict for it)."""
    share, base = shares(cfg, name, against)
    plan = uniform(cfg, 6)
    used = code_bytes(cfg, plan)
    gone = 0.0
    for (i, m), s in sorted(share.items(), key=lambda kv: -kv[1] / sizes(cfg, kv[0][0])[kv[0][1]]):
        extra = sizes(cfg, i)[m] // 8
        if s > 0 and used + extra <= budget:
            plan[i][m] = 8
            used += extra
            gone += 0.75 * s
    return plan, base * max(1 - gone, 0.0) ** 0.5
