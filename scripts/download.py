# Downloads the checkpoints one after another, each large file in parallel byte ranges.
# A range that stops receiving is dropped and asked for again from where it was; a run that is stopped picks up from
# the ranges already on disk. Finished files are checked against the repository's SHA-256 and then skipped.
import argparse
import hashlib
import json
import os
import threading
import time

import httpx2 as httpx

from sclmd import REPOS, ckpt

ORDER = ["drafter", "qat_drafter", "orig", "qat"]
SEGMENT = 512 << 20
WORKERS = 8
STALL = 30.0  # seconds without a byte before a range is asked for again


def listing(repo: str) -> list:
    r = httpx.get(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=30, follow_redirects=True)
    r.raise_for_status()
    return [(s["rfilename"], s["size"], (s.get("lfs") or {}).get("sha256")) for s in r.json()["siblings"]]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, path: str, size: int, label: str):
    """Fill `path` (created at full size) range by range; `path`.ranges remembers how far each range got."""
    state_path = path + ".ranges"
    starts = list(range(0, size, SEGMENT)) or [0]
    done = {str(s): 0 for s in starts}
    if os.path.exists(state_path) and os.path.exists(path):
        with open(state_path) as f:
            done.update(json.load(f))
    else:
        with open(path, "wb") as f:
            f.truncate(size)
    lock, todo = threading.Lock(), [s for s in starts if done[str(s)] < min(SEGMENT, size - s)]
    t0, got0 = time.time(), sum(done.values())

    def worker():
        client = httpx.Client(follow_redirects=True, timeout=httpx.Timeout(STALL, connect=20))
        with open(path, "r+b") as f:
            while True:
                with lock:
                    if not todo:
                        return
                    s = todo.pop()
                end = min(s + SEGMENT, size)
                while s + done[str(s)] < end:
                    at = s + done[str(s)]
                    try:
                        with client.stream("GET", url, headers={"Range": f"bytes={at}-{end - 1}"}) as r:
                            r.raise_for_status()
                            for chunk in r.iter_bytes(1 << 20):
                                with lock:
                                    f.seek(at)
                                    f.write(chunk)
                                    at += len(chunk)
                                    done[str(s)] = at - s
                    except (httpx.HTTPError, OSError) as e:
                        print(f"[download] {label}: range {s >> 20} MiB again after {type(e).__name__}", flush=True)
                        time.sleep(2)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(WORKERS, len(todo)))]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        with lock:
            got = sum(done.values())
            with open(state_path, "w") as f:
                json.dump(done, f)
        print(f"[download] {label}: {got / 1e9:.1f} / {size / 1e9:.1f} GB, {(got - got0) / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)
    with open(state_path, "w") as f:
        json.dump(done, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", default=ORDER, help=f"which of {ORDER} (default: all, in this order)")
    args = ap.parse_args()
    for name in args.names:
        repo, dest = REPOS[name], ckpt(name)
        t = time.time()
        print(f"[download] {repo} -> {dest}", flush=True)
        for rel, size, digest in listing(repo):
            path = os.path.join(dest, rel)
            part = path + ".part"
            if os.path.exists(path) and os.path.getsize(path) == size:
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fetch(f"https://huggingface.co/{repo}/resolve/main/{rel}", part, size, rel)
            if digest and sha256(part) != digest:
                os.remove(part + ".ranges")
                raise SystemExit(f"[download] {rel}: SHA-256 differs from the repository's; run again to fetch it anew")
            os.replace(part, path)
            if os.path.exists(part + ".ranges"):
                os.remove(part + ".ranges")
        print(f"[download] {name} done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
