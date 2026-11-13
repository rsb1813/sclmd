# Which matrices get 4 bits a weight instead of 3: a plan is layer -> {matrix: bits a pair (6 or 8)}.
#
# What an upgrade is worth is measured, not assumed (scripts/sweep.sh): the whole model's KL with everything at 3
# bits, everything at 4, one matrix type at 4 in every layer, and one block of ten layers at 4.
#  - How KL goes with the error differs by text. At 4 bits every matrix's error is a quarter. On text the model wrote
#    itself the KL is then about a fifth (KL goes with the error); on other people's text it is half (with the
#    error's square root: there a few sharp positions tipping over make most of the KL). So the power is taken from
#    the set's own two ends, and a matrix's share is of KL^(1/power), in which shares add up.
#  - On other people's text a measured KL moves by about 0.005 from one plan to the next for no reason, so the
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


# The measurements' report tags by set: (everything at 3 bits, everything at 4, a type at 4, a block at 4).
# The evaluation set's were made with the original head; the model's own text's with the head at 4 bits throughout.
TAGS = {"eval_v1": ("l0-{last}_c6", "l0-{last}_c8", "c6_{m}8", "c6_l{a}-{b}_8"),
        "own_v1": ("c6_head8", "c8_head8", "h8_{m}8", "h8_l{a}-{b}_8")}


def measured(name: str, tag: str, against: str = "orig_fp32", of: str = "eval_v1") -> float:
    where = "" if of == "eval_v1" else of + "_"
    with open(os.path.join(DATA, f"{where}codes_{name}_{tag}.json"), encoding="utf-8") as f:
        return json.load(f)["against"][against]["all"]["kl"]


def shares(cfg: M.Config, name: str, against: str = "orig_fp32", of: str = "eval_v1") -> tuple[dict, float, float]:
    """((layer, matrix) -> its share at 3 bits everywhere, the KL there, the power) from the measurements on set `of`.
    KL goes with error^power; the power is what makes everything at 4 bits (a quarter of every error) come out as
    measured. A type's (or block's) share follows from the KL with it alone at 4 bits: three quarters of it go away."""
    import math

    t_base, t_all, t_type, t_block = TAGS[of]
    kl = lambda tag: measured(name, tag, against, of)  # noqa: E731
    base = kl(t_base.format(last=cfg.n_layers - 1))
    power = math.log(base / kl(t_all.format(last=cfg.n_layers - 1))) / math.log(4)
    share = lambda tag: max((1 - (kl(tag) / base) ** (1 / power)) / 0.75, 0.0)  # noqa: E731
    by_type = {m: share(t_type.format(m=m)) for m in M.LINEARS}
    by_block = {b: share(t_block.format(a=b, b=b + BLOCK - 1)) for b in range(0, cfg.n_layers, BLOCK)}
    t_sum, b_sum = sum(by_type.values()), sum(by_block.values())
    out = {}
    for b in by_block:
        layers = range(b, min(b + BLOCK, cfg.n_layers))
        for m in M.LINEARS:
            have = [i for i in layers if m in sizes(cfg, i)]
            for i in have:
                out[(i, m)] = by_type[m] / t_sum * by_block[b] / b_sum / len(have)
    return out, base, power


def fit(cfg: M.Config, name: str, budget: int, against: str = "orig_fp32", of: str = "eval_v1") -> tuple[dict, float]:
    """The plan for `budget` bytes of layer codes: 3 bits everywhere, then 4 bits for the matrices with the largest
    share a byte, while they fit -> (plan, the KL the shares predict for it on the set `of` they were measured on)."""
    share, base, power = shares(cfg, name, against, of)
    plan = uniform(cfg, 6)
    used = code_bytes(cfg, plan)
    gone = 0.0
    for (i, m), s in sorted(share.items(), key=lambda kv: -kv[1] / sizes(cfg, kv[0][0])[kv[0][1]]):
        extra = sizes(cfg, i)[m] // 8
        if s > 0 and used + extra <= budget:
            plan[i][m] = 8
            used += extra
            gone += 0.75 * s
    return plan, base * max(1 - gone, 0.0) ** power
