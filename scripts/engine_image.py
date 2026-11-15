# Engine images (sclmd/image.py) by hand.
#   python -m scripts.engine_image build [--plan c6|FILE] [--head 8] [--int8]   the image of a plan (c6: the server's 3 bits
#                                                                       everywhere; --int8: with the int8 codebooks, another image)
#   python -m scripts.engine_image list                                 the images there are, their plans and sizes
#   python -m scripts.engine_image speed [FILE]                         how fast an image reads past the file cache (no GPU):
#                                                                       what a load after a reboot would find
import argparse
import ctypes as C
import ctypes.wintypes as W
import json
import os
import time

from sclmd import image as IM


def plan_of(arg: str, head: int):
    from sclmd import ckpt
    from sclmd import model as M
    from sclmd import plan as P

    if arg == "c6":
        return P.uniform(M.Config.load(ckpt("orig")), 6), head
    plan, h = P.read(arg)
    return plan, h or head


def build(args) -> None:
    import torch

    from sclmd import engine as E
    from sclmd.stream import cap_vram

    int8 = E.use_int8_option(args)
    cap_vram(margin=int(0.4 * 2**30))
    plan, head = plan_of(args.plan, args.head)
    digest, parts = IM.fingerprint(args.name, plan, head, int8=bool(int8))
    if os.path.exists(IM.path_of(digest)) and not args.again:
        print(f"[image] there already is {IM.path_of(digest)}", flush=True)
        return
    t = time.perf_counter()
    with torch.no_grad():
        eng = E.Engine.load(args.name, plan, head, 8192, rows=8, image="off")
    print(f"[image] loaded from the codes in {time.perf_counter() - t:.1f} s", flush=True)
    IM.write(eng, digest, parts)


def listing(args) -> None:
    if not os.path.isdir(IM.FOLDER):
        print("no images")
        return
    for n in sorted(os.listdir(IM.FOLDER)):
        p = os.path.join(IM.FOLDER, n)
        if not n.endswith(".img"):
            print(f"{n}: {os.path.getsize(p) / 1e9:.2f} GB (not an image)")
            continue
        try:
            head, _ = IM.header(p)
            plan = head["parts"]["plan"]
            eights = sum(b == 8 for mats in plan.values() for b in mats.values())
            print(f"{n}: {os.path.getsize(p) / 1e9:.2f} GB, codes {head['parts']['codes']}, head {head['parts']['head_bits']} bits, "
                  f"{eights} matrices at 4 bits{', int8 codebooks' if head['parts'].get('int8') else ''}, written {head['written']}, "
                  f"last used {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p)))}")
        except Exception as e:  # noqa: BLE001
            print(f"{n}: {type(e).__name__}: {e}")


def speed(args) -> None:
    """Reads the file with FILE_FLAG_NO_BUFFERING (the device, not the file cache) and through the cache."""
    path = args.file or max((os.path.join(IM.FOLDER, n) for n in os.listdir(IM.FOLDER) if n.endswith(".img")), key=os.path.getmtime)
    size, chunk = os.path.getsize(path), 64 << 20
    k32 = C.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = W.HANDLE
    k32.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, C.c_void_p, W.DWORD, W.DWORD, W.HANDLE]
    k32.ReadFile.argtypes = [W.HANDLE, C.c_void_p, W.DWORD, C.POINTER(W.DWORD), C.c_void_p]
    k32.VirtualAlloc.restype = C.c_void_p
    k32.VirtualAlloc.argtypes = [C.c_void_p, C.c_size_t, W.DWORD, W.DWORD]
    buf = k32.VirtualAlloc(None, chunk, 0x3000, 0x04)
    out = {"file": path, "gb": round(size / 1e9, 3)}
    h = k32.CreateFileW(path, 0x80000000, 1, None, 3, 0x20000000 | 0x08000000, None)  # GENERIC_READ, NO_BUFFERING | SEQUENTIAL
    got, total, t = W.DWORD(), 0, time.perf_counter()
    while k32.ReadFile(h, C.c_void_p(buf), chunk, C.byref(got), None) and got.value:
        total += got.value
    k32.CloseHandle(h)
    out["unbuffered_gb_s"] = round(total / 1e9 / (time.perf_counter() - t), 2)
    mv, t = memoryview(bytearray(chunk)), time.perf_counter()
    with open(path, "rb", buffering=0) as f:
        while f.readinto(mv):
            pass
    out["through_the_cache_gb_s"] = round(size / 1e9 / (time.perf_counter() - t), 2)
    print(json.dumps(out), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=("build", "list", "speed"))
    ap.add_argument("file", nargs="?")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--plan", default="c6")
    ap.add_argument("--head", type=int, default=8)
    ap.add_argument("--again", action="store_true", help="build: write it again even if it is there")
    IM.E.add_int8_option(ap)
    args = ap.parse_args()
    {"build": build, "list": listing, "speed": speed}[args.what](args)


if __name__ == "__main__":
    main()
