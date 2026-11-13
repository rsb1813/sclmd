# Writes plan files (which rate each matrix gets) into data/plans: the ones that measure what an upgrade from 3 to 4
# bits is worth, by matrix type and by block of ten layers.
import json
import os

from sclmd import DATA, ckpt
from sclmd import model as M

FOLDER = os.path.join(DATA, "plans")


def sizes(cfg: M.Config, i: int) -> dict:
    """Weights of layer i's matrices."""
    t = cfg.layer_types[i]
    qo, kv = cfg.heads * cfg.head_dim[t] * cfg.hidden, cfg.kv_heads[t] * cfg.head_dim[t] * cfg.hidden
    out = {"q": qo, "k": kv, "v": kv, "o": qo, "gate": cfg.mlp * cfg.hidden, "up": cfg.mlp * cfg.hidden, "down": cfg.mlp * cfg.hidden}
    if t == M.FULL:
        del out["v"]
    return out


def write(name: str, layers: dict, head: int | None = None, note: str = "") -> str:
    os.makedirs(FOLDER, exist_ok=True)
    path = os.path.join(FOLDER, name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"note": note, "head": head, "layers": {str(i): mats for i, mats in layers.items()}}, f, indent=1)
    return path


def uniform(cfg: M.Config, bits: int) -> dict:
    return {i: {m: bits for m in sizes(cfg, i)} for i in range(cfg.n_layers)}


def main():
    cfg = M.Config.load(ckpt("orig"))
    for m in M.LINEARS:
        plan = uniform(cfg, 6)
        for i in plan:
            if m in plan[i]:
                plan[i][m] = 8
        write(f"c6_{m}8", plan, note=f"every matrix at 3 bits, {m} at 4")
    for b in range(0, cfg.n_layers, 10):
        plan = uniform(cfg, 6)
        for i in range(b, b + 10):
            plan[i] = {m: 8 for m in plan[i]}
        write(f"c6_l{b}-{b + 9}_8", plan, note=f"every matrix at 3 bits, layers {b}-{b + 9} at 4")
    print(sorted(os.listdir(FOLDER)))


if __name__ == "__main__":
    main()
