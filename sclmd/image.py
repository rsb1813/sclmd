# A packed engine image: the bytes Engine.load makes on the GPU of the stored codes (every matrix's packed words, its
# codebook as fp16 pairs, its scales, and with int8 its int8 codebook), kept in one file and read back raw into the
# same tensors. Engine.load reads each layer's codes from the codes folder (a byte a pair, both rates in one file) and
# packs them on the GPU: about 9 s of the server's 13 s start-up (R3's phase timers). The image holds the packed form
# only (3 bits a weight: 6 of 8 bits), so a load is one sequential read through pinned buffers, with the copies to the
# GPU under the next read.
#
# It is a cache. Its name is the fingerprint of everything that decides its bytes: the plan, the head's bits, the codes
# files (sizes and times), the packing code, the engine's DLL and whether the engine has int8 codebooks (--int8; its
# clip, SIGMA, is not in the bytes). kv4, keyless, the capacity, the pool and the rows change no byte of it: the same
# image serves them all. Any change gives another name, and the load is the one without an image (and writes a new
# image, with "auto"). A file that does not match what it should hold (its header's checksum, its size, a checksum of
# every record taken on the GPU when it was written) is not used. At most KEEP images are kept: the oldest go. The
# small weights, the rotations, the keys and values and the drafter are made as without it.
#
# The file: MAGIC, FORMAT (4 bytes), the header's length (4), the header's sha256 (32), the header (JSON: the
# fingerprint, what it was made of, the records), then from the next 4096 bytes the records one after the other, and
# zeros to a whole 4096. A record: [name, bytes, rows, pairs, bits, its two GPU checksums, extra]; extra is the
# matrix's book_scale (the float the int8 codebook is multiplied by) on its lut8 record, else None.
import gc
import hashlib
import inspect
import json
import os
import threading
import time

import torch

from . import DATA, ckpt
from . import codes as CD
from . import engine as E
from . import model as M

FORMAT = 2  # the file's layout; part of the fingerprint (1: R3's first, without the header's checksum and lut8)
MAGIC = b"SCLMDIMG"
LEAD = len(MAGIC) + 4 + 4 + 32  # magic, format, the header's length, its sha256
ALIGN = 4096
FOLDER = os.environ.get("SCLMD_IMAGES") or os.path.join(DATA, "images")
KEEP = 2  # images kept on disk (12-13 GB each)
CHUNK = 64 << 20  # bytes a read, and a pinned buffer
# How an image is read: "direct" past the file cache (Windows: FILE_FLAG_NO_BUFFERING), READERS reads under way at
# once into BUFFERS pinned buffers; "buffered" one read after the other through the file cache (Python's readinto).
# R3 measured the 11.7 GB image of 3 bits everywhere at 4.8 GB/s past the cache and 2.5-3.1 through it, one read at a time.
READER = os.environ.get("SCLMD_IMAGE_READER", "direct")
READERS, BUFFERS = 2, 4


class Mismatch(Exception):
    """The image is not the one these codes, this plan and this build would make."""


class Shape:
    """A matrix's codes as far as an image needs them: their shape [rows, pairs]."""

    def __init__(self, rows: int, pairs: int):
        self.shape = (rows, pairs)


def parts_of(int8: bool) -> tuple:
    """The records of a matrix, in the order Engine._mat makes its tensors."""
    return ("codes", "lut", "scale") + (("lut8",) if int8 else ())


# ---- what decides the bytes

def _dll_hash() -> str:
    with open(os.path.abspath(E.DLL), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fingerprint(name: str, plan: dict, head: int, int8: bool = False) -> tuple[str, dict]:
    """(the image's name, what it was made of). Capacity, the pool, rows, keyless and kv4 change no byte of the codes
    and are not in it; nor are the checkpoint's small weights, which are read as without an image."""
    files = [CD.layer_path(name, i) for i in sorted(plan)] + [CD.head_path(name)]
    stats = []
    for p in files:
        st = os.stat(p)
        stats.append([os.path.basename(p), st.st_size, st.st_mtime_ns])
    code = "".join(inspect.getsource(f) for f in (E.words, E.pack, E._pack, E.lut, E.lut8, E.Engine._mat))
    lanes = (E._ROW.numpy().tobytes() + E._PAIR.numpy().tobytes()).hex()
    parts = {"format": FORMAT, "codes": name, "head_bits": int(head), "int8": bool(int8),
             "plan": {str(i): {m: int(b) for m, b in sorted(plan[i].items())} for i in sorted(plan)},
             "files": stats, "packing_sha256": hashlib.sha256((code + lanes).encode()).hexdigest(), "dll": os.path.abspath(E.DLL),
             "dll_sha256": _dll_hash(), "torch": torch.__version__}
    keyed = dict(parts, dll=None)  # the DLL's place does not matter, its bytes do
    digest = hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()[:24]
    return digest, parts


def path_of(digest: str) -> str:
    return os.path.join(FOLDER, f"{digest}.img")


# ---- the records: per matrix (layer by layer in M.LINEARS order, then the head) its packed words, lut, scales (lut8)

def _records(eng: E.Engine) -> list:
    """[(name, device tensor (a view), rows, pairs, bits, extra)] of an engine, in the order Engine.__init__ makes them."""
    keep = {t.data_ptr(): t for t in eng.keep}
    out = []

    def add(prefix: str, mat: E.Mat, whole: torch.Tensor | None):
        pairs, bits = mat.cols // 2, mat.bits
        n = E.words(Shape(mat.rows, pairs), bits)
        if whole is None:
            codes = keep[mat.codes][:n]
        else:
            at = (mat.codes - whole.data_ptr()) // 4
            codes = whole[at : at + n]
        assert codes.data_ptr() == mat.codes and codes.numel() == n
        lut, scale = keep[mat.lut], keep[mat.scale]
        assert lut.numel() == 1 << bits and scale.numel() == mat.rows
        out.append((prefix + ".codes", codes, mat.rows, pairs, bits, None))
        out.append((prefix + ".lut", lut, mat.rows, pairs, bits, None))
        out.append((prefix + ".scale", scale, mat.rows, pairs, bits, None))
        if eng.int8:  # the int8 codebook (Engine._mat with int8) and the float it is multiplied by
            lut8 = keep[mat.lut8]
            assert lut8.numel() == 1 << bits
            out.append((prefix + ".lut8", lut8, mat.rows, pairs, bits, float(mat.book_scale)))

    for i, cl in enumerate(eng.c_layers):
        mats = [m for m in M.LINEARS if getattr(cl, m).codes]
        whole = keep[getattr(cl, mats[0]).codes]  # the layer's codes are one tensor, its first matrix's at its start
        for m in mats:
            add(f"L{i}.{m}", getattr(cl, m), whole)
    add("head", eng.model.head, None)
    return out


def _checksums(tensors: list) -> list:
    """Two sums a tensor over its 32-bit words, the second weighted by where they are (as int64, on the GPU): any torn or
    misplaced part shows."""
    sums, piece = [], 1 << 22  # words at a time: the temporaries stay small (the VRAM is full of the engine)
    for t in tensors:
        v = t.view(torch.int32).view(-1)
        plain = weighted = torch.zeros((), dtype=torch.int64, device=v.device)
        for a in range(0, v.numel(), piece):
            p = v[a : a + piece]
            n = p.numel() // 1024 * 1024
            rows = p[:n].view(-1, 1024).sum(1, dtype=torch.int64)
            w = torch.arange(a // 1024 + 1, a // 1024 + 1 + rows.numel(), device=v.device, dtype=torch.int64)
            plain = plain + p.sum(dtype=torch.int64)
            weighted = weighted + (rows * w).sum() + p[n:].sum(dtype=torch.int64) * (a // 1024 + rows.numel() + 1)
        sums.append(torch.stack((plain, weighted)))
    return torch.stack(sums).cpu().tolist() if sums else []


def _data_at(n: int) -> int:
    """Where the records begin, after a header of n bytes."""
    return -(-(LEAD + n) // ALIGN) * ALIGN


# ---- writing

def write(eng: E.Engine, digest: str, parts: dict, background: bool = False) -> str | threading.Thread | None:
    """Writes the engine's image (to a temporary file, then renamed). background: in a thread, with the copies off
    the GPU on a stream of their own (the engine serves meanwhile; its codes never change)."""
    os.makedirs(FOLDER, exist_ok=True)
    target = path_of(digest)

    def run():
        t0 = time.perf_counter()
        tmp = f"{target}.{os.getpid()}.{threading.get_ident()}.tmp"
        stream = torch.cuda.Stream() if background else torch.cuda.current_stream()
        try:
            with torch.cuda.stream(stream):
                recs = _records(eng)
                sums = _checksums([r[1] for r in recs])
            head = {"format": FORMAT, "fingerprint": digest, "parts": parts, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "records": [[name, t.numel() * t.element_size(), rows, pairs, bits, s, extra]
                                for (name, t, rows, pairs, bits, extra), s in zip(recs, sums)]}
            blob = json.dumps(head).encode()
            data_at = _data_at(len(blob))
            total = sum(r[1] for r in head["records"])
            staging = [torch.empty(CHUNK, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
            with open(tmp, "wb") as f:
                f.write(MAGIC + FORMAT.to_bytes(4, "little") + len(blob).to_bytes(4, "little") + hashlib.sha256(blob).digest() + blob)
                f.write(b"\0" * (data_at - f.tell()))
                k, have = 0, 0  # buffers filled, bytes in the one in use
                buf = staging[0]
                with torch.cuda.stream(stream):
                    for _, t, *_ in recs:
                        src = t.view(torch.uint8).view(-1)
                        at = 0
                        while at < src.numel():
                            take = min(CHUNK - have, src.numel() - at)
                            buf[have : have + take].copy_(src[at : at + take], non_blocking=True)
                            have, at = have + take, at + take
                            if have == CHUNK:
                                stream.synchronize()
                                f.write(memoryview(buf.numpy()))
                                k, have = k + 1, 0
                                buf = staging[k % 2]
                    stream.synchronize()
                    if have:
                        f.write(memoryview(buf.numpy())[:have])
                f.write(b"\0" * (-(data_at + total) % ALIGN))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
            _evict(keep=target)
            gb = (data_at + total) / 1e9
            print(f"[image] wrote {target}: {gb:.2f} GB in {time.perf_counter() - t0:.1f} s", flush=True)
        except BaseException as e:  # noqa: BLE001  (a cache: the engine goes on without it)
            print(f"[image] could not write {target}: {type(e).__name__}: {e}", flush=True)
            try:
                os.remove(tmp)
            except OSError:
                pass
            return None
        return target

    if background:
        th = threading.Thread(target=run, daemon=True, name="image-writer")
        th.start()
        return th
    return run()


def is_image(path: str) -> bool:
    """Whether a file begins as an engine image does (of any FORMAT): what _evict may remove."""
    try:
        with open(path, "rb") as f:
            return f.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


def _evict(keep: str) -> None:
    """At most KEEP images (the newest by use; `keep` always stays), and no temporary file of an image older than an
    hour. Only files that begin as an image does are removed (another .img put in the folder stays)."""
    now = time.time()
    names = [os.path.join(FOLDER, n) for n in os.listdir(FOLDER)]
    for p in names:
        if p.endswith(".tmp") and ".img." in os.path.basename(p) and now - os.path.getmtime(p) > 3600:
            try:
                os.remove(p)
            except OSError:
                pass
    images = sorted((p for p in names if p.endswith(".img") and os.path.exists(p) and is_image(p)), key=os.path.getmtime, reverse=True)
    images = [keep] + [p for p in images if os.path.normcase(p) != os.path.normcase(keep)]
    for p in images[KEEP:]:
        try:
            os.remove(p)
            print(f"[image] removed the older image {p}", flush=True)
        except OSError:
            pass


# ---- reading

def header(path: str) -> tuple[dict, int]:
    """(the image's header, where its data begins). Mismatch when it is not an image of this format, its header is
    not the one written, or the file is cut short."""
    with open(path, "rb") as f:
        lead = f.read(LEAD)
        if len(lead) < len(MAGIC) + 4 or lead[: len(MAGIC)] != MAGIC:
            raise Mismatch("not an engine image")
        fmt = int.from_bytes(lead[len(MAGIC) : len(MAGIC) + 4], "little")
        if fmt != FORMAT:
            raise Mismatch(f"format {fmt}, not {FORMAT}")
        if len(lead) < LEAD:
            raise Mismatch("the image ends inside its header")
        n, sha = int.from_bytes(lead[len(MAGIC) + 4 : len(MAGIC) + 8], "little"), lead[len(MAGIC) + 8 :]
        blob = f.read(n)
        if len(blob) != n or hashlib.sha256(blob).digest() != sha:
            raise Mismatch("the header is not the one written (its checksum)")
        head = json.loads(blob)
    data_at = _data_at(n)
    total = sum(r[1] for r in head["records"])
    size = os.path.getsize(path)
    if size != data_at + total + (-(data_at + total) % ALIGN):
        raise Mismatch(f"{size} bytes where {data_at + total} are expected")
    return head, data_at


def expected(cfg: M.Config, plan: dict, head_bits: int, int8: bool = False) -> list:
    """The records an engine of this plan has, in order: [(name, bits)]."""
    out = []
    for i in range(cfg.n_layers):
        for m in M.LINEARS:
            if m in plan[i] and not (m == "v" and cfg.layer_types[i] == M.FULL):
                out += [(f"L{i}.{m}.{part}", plan[i][m]) for part in parts_of(int8)]
    return out + [(f"head.{part}", head_bits) for part in parts_of(int8)]


class Filler:
    """Engine._mat's other way (Engine._image): the same tensors, empty, and noted in the image's order; fill() then
    reads the file into them."""

    def __init__(self, path: str, head: dict, data_at: int):
        self.path, self.head, self.data_at = path, head, data_at
        self.records = head["records"]
        self.targets = []  # (tensor, its record)

    def mat(self, eng: E.Engine, codes, bits: int, out: torch.Tensor | None) -> E.Mat:
        rows, pairs = codes.shape
        n = E.words(codes, bits)
        # as Engine._mat makes them, in its order (eng.keep is the same list)
        packed = out if out is not None else eng._own(torch.empty(n, dtype=torch.int32, device=eng.device))
        table = eng._own(torch.empty(1 << bits, dtype=torch.int32, device=eng.device))
        scale = eng._own(torch.empty(rows, dtype=torch.float32, device=eng.device))
        tensors = [packed, table, scale]
        if eng.int8:
            tensors.append(eng._own(torch.empty(1 << bits, dtype=torch.int32, device=eng.device)))
        for part, t in zip(parts_of(eng.int8), tensors):
            k = len(self.targets)
            if k >= len(self.records):
                raise Mismatch("the engine has more matrices than the image")
            name, nbytes, r_rows, r_pairs, r_bits = self.records[k][:5]
            if not name.endswith("." + part) or (nbytes, r_rows, r_pairs, r_bits) != (t.numel() * t.element_size(), rows, pairs, bits):
                raise Mismatch(f"record {k} ({name}): {nbytes} bytes of [{r_rows}, {r_pairs}] at {r_bits} bits, the engine's "
                               f"{part} is {t.numel() * t.element_size()} bytes of [{rows}, {pairs}] at {bits}")
            self.targets.append((t, self.records[k]))
        mat = E.Mat(packed.data_ptr(), table.data_ptr(), scale.data_ptr(), rows, pairs * 2, bits, 0)
        if eng.int8:
            scale8 = self.records[len(self.targets) - 1][6]
            if not isinstance(scale8, float):
                raise Mismatch(f"record {len(self.targets) - 1}: no book_scale")
            mat.lut8, mat.book_scale = tensors[3].data_ptr(), scale8
        return mat

    def fill(self) -> dict:
        """Reads the records into their tensors: a part at a time into pinned buffers, each copied to the GPU while the
        next ones are read; then every record's checksums are taken on the GPU and compared. -> timings."""
        if len(self.targets) != len(self.records):
            raise Mismatch(f"the engine has {len(self.targets)} records, the image {len(self.records)}")
        t0 = time.perf_counter()
        how = "direct" if READER == "direct" and os.name == "nt" and CHUNK % ALIGN == 0 else "buffered"
        count = BUFFERS if how == "direct" else 2
        raw = [torch.empty(CHUNK + ALIGN, dtype=torch.uint8, pin_memory=True) for _ in range(count)]
        pinned = [r[-r.data_ptr() % ALIGN :][:CHUNK] for r in raw]  # whole sectors from an aligned address: reads past the cache need it
        t_pin = time.perf_counter() - t0
        flat = [(t.view(torch.uint8).view(-1), rec[1]) for t, rec in self.targets]
        total = sum(n for _, n in flat)
        chunks = -(-total // CHUNK)
        stream = torch.cuda.current_stream()
        copied = [threading.Event() for _ in range(chunks)]  # chunk k's copies are queued (its CUDA event is in done[k])
        done = [None] * chunks
        reads = (_read_direct if how == "direct" else _read_buffered)(self.path, self.data_at, total, pinned, copied, done)
        ti, tat = 0, 0
        try:
            for k in range(chunks):
                b, n = next(reads)  # chunk k is in pinned[b]: n bytes
                off = 0
                while off < n:
                    dst, size = flat[ti]
                    take = min(size - tat, n - off)
                    dst[tat : tat + take].copy_(pinned[b][off : off + take], non_blocking=True)
                    off, tat = off + take, tat + take
                    if tat == size:
                        ti, tat = ti + 1, 0
                ev = torch.cuda.Event()
                ev.record(stream)
                done[k] = ev
                copied[k].set()
        finally:
            for e in copied:  # a reader that waits gives up
                e.set()
            reads.close()
        torch.cuda.synchronize()
        t_copy = time.perf_counter() - t0
        sums = _checksums([t for t, _ in self.targets])
        bad = [rec[0] for (t, rec), s in zip(self.targets, sums) if list(rec[5]) != s]
        if bad:
            raise Mismatch(f"{len(bad)} records differ from their checksums, the first {bad[0]}")
        return {"gb": round(total / 1e9, 3), "reader": how, "pin_s": round(t_pin, 3), "read_and_copy_s": round(t_copy, 3),
                "check_s": round(time.perf_counter() - t0 - t_copy, 3)}


def _free(k: int, count: int, copied: list, done: list) -> None:
    """Waits until the buffer of chunk k is free: the copies of chunk k - count out of it are done."""
    if k >= count:
        copied[k - count].wait()
        if done[k - count] is not None:
            done[k - count].synchronize()


def _read_buffered(path: str, data_at: int, total: int, pinned: list, copied: list, done: list):
    """Chunk after chunk through the file cache (readinto): yields (buffer, bytes in it)."""
    views = [memoryview(p.numpy()) for p in pinned]
    with open(path, "rb", buffering=0) as f:
        f.seek(data_at)
        for k in range(-(-total // CHUNK)):
            b, n = k % len(pinned), min(CHUNK, total - k * CHUNK)
            _free(k, len(pinned), copied, done)
            got = 0
            while got < n:
                r = f.readinto(views[b][got:n])
                if not r:
                    raise Mismatch(f"the image ends after {k * CHUNK + got} of {total} bytes")
                got += r
            yield b, n


def _read_direct(path: str, data_at: int, total: int, pinned: list, copied: list, done: list):
    """Past the file cache (FILE_FLAG_NO_BUFFERING), READERS threads reading alternate chunks at once (the device is
    faster with more than one read under way): yields (buffer, bytes in it) in order."""
    import ctypes as C
    import ctypes.wintypes as W

    k32 = C.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = W.HANDLE
    k32.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, C.c_void_p, W.DWORD, W.DWORD, W.HANDLE]
    k32.ReadFile.argtypes = [W.HANDLE, C.c_void_p, W.DWORD, C.POINTER(W.DWORD), C.c_void_p]
    k32.CloseHandle.argtypes = [W.HANDLE]

    class Overlapped(C.Structure):
        _fields_ = [("Internal", C.c_void_p), ("InternalHigh", C.c_void_p), ("Offset", W.DWORD), ("OffsetHigh", W.DWORD), ("hEvent", W.HANDLE)]

    assert CHUNK % ALIGN == 0 and data_at % ALIGN == 0
    chunks, count = -(-total // CHUNK), len(pinned)
    ready = [threading.Event() for _ in range(chunks)]
    errors, stop = [], threading.Event()

    def work(r: int) -> None:
        h = k32.CreateFileW(os.path.abspath(path), 0x80000000, 1, None, 3, 0x20000000 | 0x08000000, None)  # read, no buffering, sequential
        try:
            if h in (None, W.HANDLE(-1).value):
                raise OSError(f"CreateFileW: error {C.get_last_error()}")
            for k in range(r, chunks, READERS):
                _free(k, count, copied, done)
                if stop.is_set():
                    return
                n = min(CHUNK, total - k * CHUNK)
                want = -(-n // ALIGN) * ALIGN  # whole sectors: the file is padded to them
                at = data_at + k * CHUNK
                ov, got = Overlapped(), W.DWORD()
                ov.Offset, ov.OffsetHigh = at & 0xFFFFFFFF, at >> 32
                if not k32.ReadFile(h, C.c_void_p(pinned[k % count].data_ptr()), want, C.byref(got), C.byref(ov)) or got.value < n:
                    raise OSError(f"ReadFile at {at}: {got.value} of {want} bytes, error {C.get_last_error()}")
                ready[k].set()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            for e2 in ready:
                e2.set()
        finally:
            if h not in (None, W.HANDLE(-1).value):
                k32.CloseHandle(h)

    threads = [threading.Thread(target=work, args=(r,), daemon=True) for r in range(min(READERS, chunks))]
    for t in threads:
        t.start()
    try:
        for k in range(chunks):
            ready[k].wait()
            if errors:
                raise Mismatch(f"the image could not be read: {errors[0]}")
            yield k % count, min(CHUNK, total - k * CHUNK)
    finally:
        stop.set()
        for e in copied:
            e.set()
        for t in threads:
            t.join()


def from_image(cls, path: str, head: dict, data_at: int, cfg: M.Config, plan: dict, head_bits: int, small, norm: torch.Tensor,
               embed: torch.Tensor, **kw) -> E.Engine:
    """An engine whose packed codes come from the image at `path` (header(path) gave head and data_at); small(i): layer
    i's small weights (the norms, the scalar), which the image does not hold; kw: Engine()'s other arguments (capacity,
    device, sequences, pool, rows, keyless, int8, kv4)."""
    kw["int8"] = E.int8_default() if kw.get("int8") is None else float(kw["int8"])  # as Engine() takes it: the records follow it
    want = expected(cfg, plan, head_bits, bool(kw["int8"]))
    have = [(r[0], r[4]) for r in head["records"]]
    if want != have:
        raise Mismatch("the image's matrices are not the plan's" + (" (int8 codebooks)" if [w[0] for w in want] != [h[0] for h in have] else ""))
    shapes = {r[0].rsplit(".", 1)[0]: (r[2], r[3], r[4]) for r in head["records"]}

    def layers():
        for i in range(cfg.n_layers):
            mats = {m: (Shape(*shapes[f"L{i}.{m}"][:2]), None, None, shapes[f"L{i}.{m}"][2]) for m in M.LINEARS if f"L{i}.{m}" in shapes}
            yield small(i) | {"mats": mats}

    filler = Filler(path, head, data_at)
    eng = cls.__new__(cls)
    eng._image = filler
    hr, hp, hb = shapes["head"]
    try:
        cls.__init__(eng, cfg, layers(), (Shape(hr, hp), None, None, hb), norm, embed, **kw)
    finally:
        eng._image = None
    eng.image_timings = filler.fill()  # on a Mismatch the engine's tensors go when the caller lets go of the error (run)
    filler.targets = []
    torch.cuda.empty_cache()
    os.utime(path)  # the newest in use: the last to be removed
    return eng


def run(mode: str, digest: str, parts: dict, from_file, from_codes) -> E.Engine:
    """A load with an image, for any engine maker: from_file(path, header, data_at) -> the engine from the image (raises
    Mismatch and the like when it does not fit); from_codes() -> the engine as without an image. mode: "auto" (read it,
    or load without it and write it), "read" (never writes); ":background" after it: the write is left to the caller,
    in a thread once the engine is loaded (eng.image["write"]). The engine's .image says what happened."""
    mode, _, how = mode.partition(":")
    assert mode in ("auto", "read") and how in ("", "background"), f"image mode {mode}"
    t0 = time.perf_counter()
    path = path_of(digest)
    why = None
    if os.path.exists(path):
        try:
            h, data_at = header(path)
            if h.get("fingerprint") != digest:
                raise Mismatch("another fingerprint inside")
            eng = from_file(path, h, data_at)
            eng.image = {"path": path, "used": True, "load_s": round(time.perf_counter() - t0, 3), **eng.image_timings}
            print(f"[image] the engine from {path}: {eng.image['gb']:.2f} GB in {eng.image['load_s']:.2f} s", flush=True)
            return eng
        except (Mismatch, OSError, ValueError, KeyError) as e:
            why = f"{type(e).__name__}: {e}"
            remove = isinstance(e, Mismatch) and mode == "auto"  # it is written again below ("read" leaves the file alone)
        # the error (and with it the frames that hold a half-filled engine) is gone here: its tensors go before the codes load
        gc.collect()
        torch.cuda.empty_cache()
        held = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        print(f"[image] {path} is not used ({why}); the engine is loaded from the codes (VRAM held now {held / 2**30:.2f} GiB)", flush=True)
        if remove:
            try:
                os.remove(path)
            except OSError:
                pass
    eng = from_codes()
    eng.image = {"path": path, "used": False, "load_s": round(time.perf_counter() - t0, 3), "why": why or "there is none"}
    if mode == "auto" and how == "background":  # the caller starts it once the rest is loaded (scripts/serve_ops.py)
        eng.image["write"] = lambda: write(eng, digest, parts, background=True)
    elif mode == "auto":
        write(eng, digest, parts)
    return eng


def load(cls, name: str, plan: dict, head: int, mode: str, **kw) -> E.Engine:
    """Engine.load with an image (Engine.load(image=...) or SCLMD_IMAGE): kw are Engine.load's other arguments."""
    kw["int8"] = E.int8_default() if kw.get("int8") is None else float(kw["int8"])  # the key and the engine take the same

    def from_codes():
        return cls.load(name, plan, head, image="off", **kw)

    try:
        digest, parts = fingerprint(name, plan, head, int8=bool(kw["int8"]))
    except Exception as e:  # noqa: BLE001  (a cache: whatever keeps it from a name, the load is the one without it)
        print(f"[image] no fingerprint ({type(e).__name__}: {e}); the engine is loaded from the codes", flush=True)
        return from_codes()

    def from_file(path, h, data_at):
        from .checkpoint import Checkpoint

        ck = Checkpoint(ckpt("orig"))
        return from_image(cls, path, h, data_at, ck.cfg, plan, head, lambda i: ck.layer(i, "cpu", skip=M.LINEARS), ck.norm("cpu"),
                          ck.embed("cpu"), **kw)

    return run(mode, digest, parts, from_file, from_codes)
