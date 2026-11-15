# The server's start-up phase by phase: scripts/serve.py's main() as it is, with timers around the parts of the
# load (a CUDA synchronize at both ends of every GPU part, so that its time is its own), stopped where the server
# would begin to serve. R3's kill test.
#
#   T_LAUNCH=$(date +%s.%N) python -m scripts.startup_timers [--out FILE] [serve.py's options ...]
#
# T_LAUNCH (seconds since the epoch, when the shell started the interpreter) adds the interpreter's own start.
import functools
import os
import sys
import time

T0 = time.time()
LAUNCH = float(os.environ.get("T_LAUNCH") or T0)
MARKS = [("interpreter start", T0 - LAUNCH)]


def _mark(name: str, since: float) -> float:
    now = time.time()
    MARKS.append((name, now - since))
    return now


t = _mark("-", T0)
MARKS.pop()
import torch  # noqa: E402

t = _mark("import torch", t)
import transformers  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

t = _mark("import transformers", t)
from sclmd import codes as CD  # noqa: E402
from sclmd import engine as E  # noqa: E402
from sclmd import model as M  # noqa: E402
from sclmd import plan as P  # noqa: E402
from sclmd.checkpoint import Checkpoint  # noqa: E402
from scripts import serve as S  # noqa: E402

t = _mark("import sclmd, scripts.serve", t)

PARTS = {}  # name -> [seconds, calls, bytes]
INNER = {}  # parts timed inside others (R3_DETAIL=1): name -> [seconds, calls, bytes]
_local = __import__("threading").local()


def inner(name: str, fn, gpu: bool = False, size=None):
    """fn with its time added to INNER[name] whatever part it is in (a finer split of the outer parts)."""

    @functools.wraps(fn)  # inspect.getsource (the image's fingerprint) sees the function itself
    def wrapper(*a, **k):
        if gpu:
            _sync()
        s = time.perf_counter()
        out = fn(*a, **k)
        if gpu:
            _sync()
        p = INNER.setdefault(name, [0.0, 0, 0])
        p[0] += time.perf_counter() - s
        p[1] += 1
        if size is not None:
            p[2] += size(a, out)
        return out

    return wrapper


def _sync():
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def timed(name: str, fn, gpu: bool = False, size=None):
    """fn with its time added to PARTS[name]; parts inside another timed part count only to the outer one."""

    @functools.wraps(fn)  # inspect.getsource (the image's fingerprint) sees the function itself
    def wrapper(*a, **k):
        depth = getattr(_local, "depth", 0)  # by thread: the tokenizer may load in a thread of its own
        if depth:
            return fn(*a, **k)
        if gpu:
            _sync()
        _local.depth = 1
        s = time.perf_counter()
        try:
            out = fn(*a, **k)
            if gpu:
                _sync()
            return out
        finally:
            _local.depth = 0
            p = PARTS.setdefault(name, [0.0, 0, 0])
            p[0] += time.perf_counter() - s
            p[1] += 1
            if size is not None:
                try:
                    p[2] += size(out)  # noqa: F821  (set when fn returned)
                except Exception:  # noqa: BLE001
                    pass

    return wrapper


def nbytes(x) -> int:
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, dict):
        return sum(nbytes(v) for v in x.values())
    if isinstance(x, (tuple, list)):
        return sum(nbytes(v) for v in x)
    return 0


def main() -> None:
    import json

    out_path = None
    if "--out" in sys.argv:
        i = sys.argv.index("--out")
        out_path = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    # what main() does, part by part (serve.py imported these names itself: patched where it looks them up)
    S.cap_vram = timed("cap_vram (CUDA context)", S.cap_vram, size=None)
    tok_load = AutoTokenizer.from_pretrained
    AutoTokenizer.from_pretrained = staticmethod(timed("tokenizer", tok_load))
    M.Config.load = staticmethod(timed("config", M.Config.load))
    P.fit = timed("plan fit", P.fit)
    CD.stored = timed("codes read (CD.stored)", CD.stored, size=nbytes)
    CD.drafter_stored = timed("drafter codes read", CD.drafter_stored, size=nbytes)
    CD.rotation = timed("rotations", CD.rotation, gpu=True)
    Checkpoint.__init__ = timed("checkpoint index", Checkpoint.__init__)
    Checkpoint.layer = timed("small weights read", Checkpoint.layer, size=nbytes)
    Checkpoint.embed = timed("embedding read (RAM)", Checkpoint.embed, size=nbytes)
    Checkpoint.norm = timed("norm read", Checkpoint.norm)
    E.Engine._mat = timed("pack + upload (Engine._mat)", E.Engine._mat, gpu=True)
    E.EngineDrafter._mat = timed("drafter pack + upload", E.EngineDrafter._mat, gpu=True)
    torch.cuda.empty_cache = timed("empty_cache", torch.cuda.empty_cache, gpu=True)
    E.lib = timed("DLL load", E.lib)
    from sclmd import image as IM

    IM.fingerprint = timed("image fingerprint", IM.fingerprint)
    IM.Filler.fill = timed("image fill (read, copy, check)", IM.Filler.fill, gpu=True)
    IM.write = lambda *a, **k: print("[timers] no image is written here (the process ends at serving)", flush=True)
    if os.environ.get("R3_DETAIL"):
        # inside pack(): the pack kernels (_pack) apart from the rest (the codes' pages read from the file and the copy
        # to the GPU); inside _mat: the codebook and the scales; inside Engine(): the keys and values' zeros
        E._pack = inner("  of which: pack kernels (_pack)", E._pack, gpu=True)
        E.pack = inner("  of which: pack() all", E.pack, gpu=True, size=lambda a, out: a[0].numel())
        E.lut = inner("  of which: lut", E.lut, gpu=True)
        real_zeros = torch.zeros
        torch.zeros = inner("  zeros (keys/values, work)", real_zeros, gpu=True,
                            size=lambda a, out: out.numel() * out.element_size() if out.is_cuda else 0)
    totals = {}

    inside = set()

    def whole(name, fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            if name in inside:  # Engine.load within Engine.load (an image that is not there): counted once
                return fn(*a, **k)
            inside.add(name)
            _sync()
            s = time.perf_counter()
            try:
                out = fn(*a, **k)
                _sync()
            finally:
                inside.discard(name)
            totals[name] = totals.get(name, 0.0) + time.perf_counter() - s
            return out
        return wrapper

    E.Engine.load = classmethod(whole("Engine.load", E.Engine.load.__func__))
    E.EngineDrafter.load = classmethod(whole("EngineDrafter.load", E.EngineDrafter.load.__func__))
    S.Server.__init__ = whole("Server()", S.Server.__init__)
    S.App.__init__ = whole("App()", S.App.__init__)
    began = time.time()

    def serve_forever(self, *a, **k):  # where the server would begin to serve: the start-up is over
        end = time.time()
        rows = {"launch_to_serving_s": end - LAUNCH, "main_s": end - began, "marks": MARKS, "totals": totals,
                "parts": {n: {"s": round(p[0], 3), "calls": p[1], "gb": round(p[2] / 1e9, 3)} for n, p in PARTS.items()},
                "vram_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2), "argv": sys.argv[1:]}
        print("[timers] start-up phase by phase", flush=True)
        for name, s in MARKS:
            print(f"  {name:34s} {s:7.2f} s", flush=True)
        for name, p in sorted(PARTS.items(), key=lambda kv: -kv[1][0]):
            extra = f"  {p[2] / 1e9:6.2f} GB  {p[2] / 1e9 / max(p[0], 1e-9):5.2f} GB/s" if p[2] else ""
            print(f"  {name:34s} {p[0]:7.2f} s  x{p[1]}{extra}", flush=True)
        for name, p in INNER.items():
            extra = f"  {p[2] / 1e9:6.2f} GB  {p[2] / 1e9 / max(p[0], 1e-9):5.2f} GB/s" if p[2] else ""
            print(f"  {name:34s} {p[0]:7.2f} s  x{p[1]}{extra}", flush=True)
        rows["inner"] = {n: {"s": round(p[0], 3), "calls": p[1], "gb": round(p[2] / 1e9, 3)} for n, p in INNER.items()}
        for name, s in totals.items():
            print(f"  [total] {name:26s} {s:7.2f} s", flush=True)
        print(f"  [total] main() to serving          {end - began:7.2f} s", flush=True)
        print(f"  [total] launch to serving          {end - LAUNCH:7.2f} s", flush=True)
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=1)
        sys.stdout.flush()
        os._exit(0)  # the engine's thread and the CUDA context go with the process

    if "--crash-only" in sys.argv:
        raise SystemExit("startup_timers: not with --crash-only (its port serves from the start)")
    S.C.BoundedHTTPServer.serve_forever = serve_forever
    sys.argv = ["scripts.serve"] + sys.argv[1:]
    S.main()


if __name__ == "__main__":
    main()
