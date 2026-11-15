# R3's kill test (3), on the GPU with the small random model of tests/test_engine.py (run it in a process of its own:
# a sticky CUDA error spoils the whole process). The real Server: one request is served; then one pass reads a tile
# whose number is far out of the pool (an illegal address in the kernels); then requests come as before. Is the error
# sticky, and does the server go on answering 500 to every request?
#   python -m scripts.lifecycle_sticky [--out FILE] [--serve-ops]
# --serve-ops: with the crash-only handling of scripts/serve_ops.py (the device probe; no exit: what it decides is printed)
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import torch  # noqa: E402

from scripts.serve import Request, Server  # noqa: E402
from sclmd import model as M  # noqa: E402
from sclmd.generate import Picker  # noqa: E402

STOP = (7,)


def ask(server: Server, full: list, timeout: float = 30) -> dict:
    req = Request(full, len(full) - 1, 8, Picker(temperature=5.0, top_k=16, top_p=1.0, seed=1), False)
    t = time.time()
    server.submit(req)
    tokens = []
    while True:
        event = req.events.get(timeout=timeout)
        if event[0] == "token":
            tokens.append(event[1])
            continue
        return {"t_s": round(time.time() - t, 3), "tokens": tokens, "end": list(event[:2]) if event[0] == "error" else ["done", event[1]["ended"]],
                "message": event[2] if event[0] == "error" else None}


def main() -> None:
    from test_engine import Small, ids_of

    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    small = Small()
    eng = small.engine(sequences=2)
    server = Server(None, eng, None, STOP)
    ops = None
    if "--serve-ops" in sys.argv:
        from scripts import serve_ops

        ops = serve_ops.install(server, exits=False)
    threading.Thread(target=server.run, daemon=True).start()
    rows = {"before": ask(server, ids_of(20, 1))}
    provide = eng._provide

    def bad(need):  # the tables as the pass needs them, then one entry far out of the pool
        provide(need)
        seq = need[0][0]
        eng.tables[M.SLIDING][seq, 0] = 1 << 27

    eng._provide = bad
    rows["fault"] = ask(server, ids_of(24, 2))
    eng._provide = provide  # the fault is gone from the bookkeeping: what is left is the CUDA context's
    rows["after"] = [ask(server, ids_of(18 + k, 3 + k)) for k in range(4)]
    try:
        torch.cuda.synchronize()
        x = torch.ones(1, device="cuda")
        rows["probe"] = f"ok: {float((x + 1).item())}"
    except Exception as e:  # noqa: BLE001
        rows["probe"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
    if ops is not None:
        rows["serve_ops"] = ops.state()
    print(json.dumps(rows, indent=1), flush=True)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
