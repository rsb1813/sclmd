# Writes plan files (which rate each matrix gets) into data/plans: the ones that measure what an upgrade from 3 to 4
# bits is worth, by matrix type and by block of ten layers.
import json
import os

from sclmd import DATA, ckpt
from sclmd import model as M
from sclmd import plan as P

FOLDER = os.path.join(DATA, "plans")


def write(name: str, layers: dict, head: int | None = None, note: str = "") -> str:
    return P.write(os.path.join(FOLDER, name + ".json"), layers, head, note)


uniform = P.uniform


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
