# Downloads the checkpoints one after another. Safe to stop and run again: finished files are skipped.
import argparse
import os
import time

from huggingface_hub import snapshot_download

from sclmd import REPOS, ckpt

ORDER = ["drafter", "qat_drafter", "orig", "qat"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", default=ORDER, help=f"which of {ORDER} (default: all, in this order)")
    args = ap.parse_args()
    for name in args.names:
        dest = ckpt(name)
        os.makedirs(dest, exist_ok=True)
        t = time.time()
        print(f"[download] {REPOS[name]} -> {dest}", flush=True)
        snapshot_download(REPOS[name], local_dir=dest, max_workers=8)
        size = sum(os.path.getsize(os.path.join(dest, f)) for f in os.listdir(dest) if os.path.isfile(os.path.join(dest, f)))
        print(f"[download] {name} done: {size / 1e9:.1f} GB in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
