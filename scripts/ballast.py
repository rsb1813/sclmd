# A VRAM ballast (scripts/soak.py's chaos: what a game started beside the server does to it): takes GiB of the card in
# a process of its own for some seconds, then gives them back and ends. It runs only inside the GPU lock of the job
# that started it (the soak's), never on its own.
#   python -m scripts.ballast --gib 1.0 --seconds 60      (--leave 0.5: all that is free but 0.5 GiB, at most --gib)
import argparse
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gib", type=float, default=1.0, help="GiB to take at most")
    ap.add_argument("--leave", type=float, default=0.5, help="GiB of what is free to leave free")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()
    free, total = torch.cuda.mem_get_info()
    want = max(0, min(int(args.gib * 2**30), free - int(args.leave * 2**30)))
    chunks, got, t = [], 0, time.time()
    while got < want:  # in pieces of 256 MiB: as much as the card gives
        n = min(256 << 20, want - got)
        try:
            chunks.append(torch.ones(n, dtype=torch.uint8, device="cuda"))  # written, so the pages are really taken
        except torch.OutOfMemoryError:
            break
        got += n
    torch.cuda.synchronize()
    print(f"[ballast] took {got / 2**30:.2f} GiB of {free / 2**30:.2f} GiB free ({total / 2**30:.1f} GiB card) in {time.time() - t:.1f} s; "
          f"holding {args.seconds:.0f} s", flush=True)
    time.sleep(args.seconds)
    del chunks
    torch.cuda.empty_cache()
    print("[ballast] given back", flush=True)


if __name__ == "__main__":
    main()
