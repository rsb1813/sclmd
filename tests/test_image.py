# The packed engine image (sclmd/image.py): an engine filled from an image holds the same bytes as one packed from the
# codes and computes the same bits (with int8's codebooks too, and kv4 from the same image); a spoilt image is noticed,
# removed, the engine loaded from the codes and the image written again, and the half-filled engine's VRAM is given back
# before the codes load; at most KEEP images stay; the fingerprint moves with what decides the bytes.
import gc
import os
import shutil
import time
from types import SimpleNamespace

import pytest
import torch

from sclmd import codes as CD
from sclmd import engine as E
from sclmd import image as IM
from sclmd import model as M
from sclmd import plan as P
from test_engine import ids_of
from test_engine import small_e  # noqa: F401  (fixture)

GPU = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.exists(E.DLL), reason="needs a GPU and the engine")
SIGMA = E.INT8_SIGMA


@pytest.fixture()
def images(tmp_path, monkeypatch):
    """Images go to a folder of the test's own."""
    monkeypatch.setattr(IM, "FOLDER", str(tmp_path))
    return tmp_path


def plan_of(small) -> dict:
    return {i: {m: lay["mats"][m][3] for m in lay["mats"]} for i, lay in enumerate(small.eng_layers)}


def small_of(small):
    return lambda i: {k: v for k, v in small.eng_layers[i].items() if k != "mats"}


def from_file_of(small, **more):
    """IM.run's from_file for the small model (what IM.load does with the checkpoint)."""
    kw = dict(capacity=more.get("capacity", 96), device="cuda", sequences=more.get("sequences", 1), pool=more.get("pool"),
              rows=more.get("rows", 8), keyless=small.keyless, int8=more.get("int8"), kv4=more.get("kv4"))
    return lambda path, head, data_at: IM.from_image(E.Engine, path, head, data_at, small.cfg, plan_of(small), small.head.bits, small_of(small),
                                                     small.norm, small.embed, **kw)


def small_engine_and_image(small, **more):
    """(an engine of the small model packed from its codes, the engine filled from its image, the image's path, what
    from_image needs again)."""
    eng = small.engine(**more)
    digest = f"test_{small.keyless}_{more.get('rows', 8)}_{bool(more.get('int8'))}"
    path = IM.write(eng, digest, {"test": True})
    assert path == IM.path_of(digest) and os.path.exists(path)
    head, data_at = IM.header(path)
    again = from_file_of(small, **more)(path, head, data_at)
    return eng, again, path, (head, data_at, plan_of(small), small_of(small))


def same_engines(a: E.Engine, b: E.Engine, small, seq: int = 1, fresh: bool = True) -> None:
    """The same records (and, fresh: every tensor the engines hold), the same structs and the same bits of a text."""
    if fresh:  # neither has computed anything yet: every tensor, the keys and values and tables too
        assert len(a.keep) == len(b.keep)
        for x, y in zip(a.keep, b.keep):
            assert x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y)
    ra, rb = IM._records(a), IM._records(b)
    assert [r[0] for r in ra] == [r[0] for r in rb] and all(torch.equal(x[1], y[1]) and x[5] == y[5] for x, y in zip(ra, rb))
    mats = [(getattr(a.c_layers[i], n), getattr(b.c_layers[i], n)) for i in range(small.cfg.n_layers) for n in M.LINEARS] + [(a.model.head, b.model.head)]
    for ma, mb in mats:  # the structs say the same of every matrix (int8: the same book_scale)
        assert (bool(ma.codes), ma.rows, ma.cols, ma.bits, bool(ma.lut8), ma.book_scale) == (bool(mb.codes), mb.rows, mb.cols, mb.bits, bool(mb.lut8), mb.book_scale)
    ids = ids_of(40, 7)
    a.release(seq)
    b.release(seq)
    la, ha = a.feed(ids, seq=seq)
    lb, hb = b.feed(ids, seq=seq)
    assert torch.equal(la, lb) and torch.equal(ha, hb)


@GPU
@pytest.mark.parametrize("chunk,reader", [(64 << 20, "direct"), (8192, "direct"), (12_800, "buffered"), (64 << 20, "buffered")])
def test_an_engine_from_its_image_holds_the_same_bytes_and_computes_the_same_bits(small_e, images, monkeypatch, chunk, reader):  # noqa: F811
    # one read; and records cut across many small reads, past the file cache by two threads or through it
    monkeypatch.setattr(IM, "CHUNK", chunk)
    monkeypatch.setattr(IM, "READER", reader)
    eng, again, path, _ = small_engine_and_image(small_e, sequences=2, rows=16)
    assert again.image_timings["reader"] == (reader if os.name == "nt" else "buffered")
    same_engines(eng, again, small_e)
    assert again.image_timings["gb"] > 0 and not eng.int8 and not again.int8


@GPU
def test_with_int8_the_image_holds_the_int8_codebooks_and_their_scales(small_e, images, monkeypatch):  # noqa: F811
    monkeypatch.setattr(IM, "CHUNK", 8192)  # records cut across reads
    eng, again, path, (head, data_at, plan, small_of_) = small_engine_and_image(small_e, sequences=2, int8=SIGMA)
    assert eng.int8 == again.int8 == SIGMA
    assert all(again.c_layers[i].q.lut8 and again.c_layers[i].q.book_scale > 0 for i in range(small_e.cfg.n_layers)) and again.model.head.lut8
    assert sum(r[0].endswith(".lut8") for r in head["records"]) == len(IM.expected(small_e.cfg, plan, small_e.head.bits)) // 3
    same_engines(eng, again, small_e)
    # an engine without int8 does not take an image with them, nor one with int8 an image without them
    with pytest.raises(IM.Mismatch, match="int8"):
        from_file_of(small_e, sequences=2, int8=0.0)(path, head, data_at)
    plain = small_engine_and_image(small_e)[2]
    with pytest.raises(IM.Mismatch, match="int8"):
        from_file_of(small_e, int8=SIGMA)(plain, *IM.header(plain))


@GPU
def test_kv4_takes_the_same_image_and_computes_kv4s_bits(small_e, images):  # noqa: F811
    """kv4 changes the keys and values, which no record holds: an image written by an engine without kv4 fills a kv4
    engine that computes the same bits as one packed from the codes."""
    plain = small_e.engine(sequences=2)
    path = IM.write(plain, "kv4test", {"test": True})
    del plain
    four = small_e.engine(sequences=2, kv4=True)
    again = from_file_of(small_e, sequences=2, kv4=True)(path, *IM.header(path))
    assert four.kv4 and again.kv4 and four.kv4_set == again.kv4_set and four.kv4_set
    same_engines(four, again, small_e)


@GPU
def test_a_spoilt_image_is_not_used(small_e, images):  # noqa: F811
    eng, again, path, (head, data_at, plan, small_of_) = small_engine_and_image(small_e)
    del again
    size = os.path.getsize(path)
    with open(path, "r+b") as f:  # one bit in the middle of the data
        f.seek(data_at + (size - data_at) // 2)
        b = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([b[0] ^ 0x40]))
    kw = dict(capacity=96, device="cuda", sequences=1, pool=None, rows=8, keyless=small_e.keyless)
    with pytest.raises(IM.Mismatch, match="checksum"):
        IM.from_image(E.Engine, path, head, data_at, small_e.cfg, plan, small_e.head.bits, small_of_, small_e.norm, small_e.embed, **kw)
    other = {i: dict(m) for i, m in plan.items()}
    other[0]["q"] = 14 - other[0]["q"]  # another plan than the image's
    with pytest.raises(IM.Mismatch, match="plan"):
        IM.from_image(E.Engine, path, head, data_at, small_e.cfg, other, small_e.head.bits, small_of_, small_e.norm, small_e.embed, **kw)
    with open(path, "r+b") as f:  # a byte of the header (a record's size, a book_scale, a checksum: none is used)
        f.seek(IM.LEAD + 40)
        b = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([b[0] ^ 0x01]))
    with pytest.raises(IM.Mismatch, match="header"):
        IM.header(path)
    with open(path, "r+b") as f:  # cut short
        f.truncate(size - 4096)
    with pytest.raises(IM.Mismatch):
        IM.header(path)


@GPU
def test_the_load_falls_back_to_the_codes_removes_a_spoilt_image_writes_it_again_and_frees_the_vram_first(small_e, images):  # noqa: F811
    """IM.run, as Engine.load(image="auto") drives it: no image -> from the codes, and written; then from the image;
    a spoilt one -> from the codes with the half-filled engine's VRAM given back first, removed and written again;
    "read" leaves a spoilt one alone and writes nothing."""
    entry = []  # the VRAM held when the codes' load begins

    def from_codes():
        torch.cuda.synchronize()
        entry.append(torch.cuda.memory_allocated())
        return small_e.engine(sequences=2)

    from_file = from_file_of(small_e, sequences=2)
    path = IM.path_of("runtest")
    gc.collect()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    first = IM.run("auto", "runtest", {"test": True}, from_file, from_codes)
    assert not first.image["used"] and first.image["why"] == "there is none" and os.path.exists(path) and entry == [base]
    second = IM.run("auto", "runtest", {"test": True}, from_file, from_codes)
    assert second.image["used"] and len(entry) == 1
    same_engines(first, second, small_e)
    del second
    size = os.path.getsize(path)
    _, data_at = IM.header(path)
    with open(path, "r+b") as f:  # one bit in the last record's part
        f.seek(size - 4096 - 100)
        b = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([b[0] ^ 0x10]))
    spoilt = open(path, "rb").read()
    gc.collect()
    torch.cuda.empty_cache()
    held = torch.cuda.memory_allocated()  # the first engine only
    # "read": not used, not removed, not written again
    third = IM.run("read", "runtest", {"test": True}, from_file, from_codes)
    assert not third.image["used"] and "checksum" in third.image["why"] and entry[-1] == held
    assert open(path, "rb").read() == spoilt
    del third
    gc.collect()
    torch.cuda.empty_cache()
    # "auto": not used (its VRAM back before the codes load), removed, written again from the codes' engine
    fourth = IM.run("auto", "runtest", {"test": True}, from_file, from_codes)
    assert not fourth.image["used"] and "checksum" in fourth.image["why"] and entry[-1] == held, (entry, held)
    assert os.path.exists(path) and open(path, "rb").read() != spoilt and IM.header(path)[0]["fingerprint"] == "runtest"
    fifth = IM.run("auto", "runtest", {"test": True}, from_file, from_codes)
    assert fifth.image["used"] and len(entry) == 3
    same_engines(first, fifth, small_e, fresh=False)


@GPU
def test_a_background_write_from_an_engine_that_serves_meanwhile(small_e, images):  # noqa: F811
    eng = small_e.engine(sequences=2)
    ids = ids_of(40, 7)
    want = eng.feed(ids, seq=1)
    eng.release(1)
    th = IM.write(eng, "bg", {"test": True}, background=True)
    got = eng.feed(ids, seq=1)  # the engine computes while the image is written on a stream of its own
    th.join(60)
    assert not th.is_alive() and torch.equal(want[0], got[0]) and torch.equal(want[1], got[1])
    again = from_file_of(small_e, sequences=2)(IM.path_of("bg"), *IM.header(IM.path_of("bg")))
    same_engines(eng, again, small_e, fresh=False)


def test_at_most_keep_images_stay_and_only_images_are_removed(images, monkeypatch):
    monkeypatch.setattr(IM, "KEEP", 2)
    now = time.time()

    def make(name: str, age_s: float, image: bool = True) -> str:
        p = os.path.join(str(images), name)
        with open(p, "wb") as f:
            f.write((IM.MAGIC if image else b"NOTANIMG") + b"\0" * 64)
        os.utime(p, (now - age_s, now - age_s))
        return p

    kept = make("a.img", 300)  # the one just written or used: stays, however old its time says it is
    old, newest, other = make("b.img", 200), make("c.img", 100), make("other.img", 400, image=False)
    tmp_old, tmp_new, tmp_other = make("d.img.1.2.tmp", 7200), make("e.img.1.2.tmp", 60), make("notes.tmp", 7200, image=False)
    IM._evict(keep=kept)
    left = set(os.listdir(str(images)))
    assert left == {"a.img", "c.img", "other.img", "e.img.1.2.tmp", "notes.tmp"}, left
    assert not os.path.exists(old) and not os.path.exists(tmp_old)
    del newest, other, tmp_new, tmp_other


def test_engine_load_hands_int8_kv4_and_the_rest_to_the_image(monkeypatch):
    got = {}

    def fake(cls, name, plan, head, mode, **kw):
        got.update(kw, cls=cls, name=name, mode=mode)
        return "engine"

    monkeypatch.setattr(IM, "load", fake)
    monkeypatch.delenv("SCLMD_IMAGE", raising=False)
    assert E.Engine.load("v1", {}, 8, 1024, sequences=3, rows=16, keyless=True, int8=5.0, kv4=True, image="read") == "engine"
    assert got == {"cls": E.Engine, "name": "v1", "mode": "read", "capacity": 1024, "device": "cuda", "sequences": 3, "pool": None, "rows": 16,
                   "keyless": True, "int8": 5.0, "kv4": True}
    monkeypatch.setenv("SCLMD_IMAGE", "auto")
    got.clear()
    assert E.Engine.load("v1", {}, 8) == "engine" and got["mode"] == "auto" and got["int8"] is None and got["kv4"] is None


def test_the_fingerprint_follows_what_decides_the_bytes(tmp_path, monkeypatch):
    if not os.path.exists(CD.head_path("v1")) or not os.path.exists(E.DLL):
        pytest.skip("needs the stored codes v1 and the engine's DLL")
    from sclmd import ckpt

    cfg = M.Config.load(ckpt("orig"))
    c6 = P.uniform(cfg, 6)
    base, parts = IM.fingerprint("v1", c6, 8)
    assert parts["head_bits"] == 8 and len(parts["files"]) == cfg.n_layers + 1 and parts["int8"] is False
    assert IM.fingerprint("v1", P.uniform(cfg, 6), 8)[0] == base  # the plan's contents, not the object
    assert IM.fingerprint("v1", c6, 8, int8=True)[0] != base  # int8's codebooks are in the image
    up = P.uniform(cfg, 6)
    up[30]["q"] = 8
    assert IM.fingerprint("v1", up, 8)[0] != base
    assert IM.fingerprint("v1", c6, 6)[0] != base
    # another build of the engine: the DLL's bytes count, not its place
    dll = tmp_path / "sclmd.dll"
    shutil.copy(E.DLL, dll)
    with monkeypatch.context() as m:
        m.setattr(E, "DLL", str(dll))
        assert IM.fingerprint("v1", c6, 8)[0] == base
        with open(dll, "r+b") as f:
            f.seek(1000)
            b = f.read(1)
            f.seek(-1, 1)
            f.write(bytes([b[0] ^ 1]))
        assert IM.fingerprint("v1", c6, 8)[0] != base
    # the packing code's lane order
    with monkeypatch.context() as m:
        m.setattr(E, "_PAIR", E._PAIR.flip(0))
        assert IM.fingerprint("v1", c6, 8)[0] != base
    # a codes file written again (another time)
    real, path = os.stat, CD.layer_path("v1", 7)

    def stat(p, *a, **k):
        st = real(p, *a, **k)
        if os.path.normcase(str(p)) == os.path.normcase(path):
            return SimpleNamespace(st_size=st.st_size, st_mtime_ns=st.st_mtime_ns + 1)
        return st

    with monkeypatch.context() as m:
        m.setattr(os, "stat", stat)
        assert IM.fingerprint("v1", c6, 8)[0] != base
    assert IM.fingerprint("v1", c6, 8)[0] == base
