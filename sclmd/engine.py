# The engine from Python: the model's pair codes laid out for the GPU kernels (engine/sclmd.cu), the memory they work
# in, and a pass as one call. All memory is PyTorch tensors; the DLL gets their addresses.
import ctypes as C
import os

import torch

from . import codes as CD
from . import model as M

# SCLMD_DLL: another build of the engine (to compare builds: scripts/bits_check.py)
DLL = os.environ.get("SCLMD_DLL") or os.path.join(os.path.dirname(__file__), "..", "engine", "build", "sclmd.dll")
MAX_TOKENS = 32  # the most rows the kernels take in a pass; an engine is made for 8, 16 or 32 (Engine.rows)
TILE = 16   # positions of keys and values a sequence is given at a time
SPARE = 16  # positions a sliding layer keeps past its window at least: a pass writes its rows while its first row still reads a window back
DRAFTER_BYTES = 240 << 20  # the drafter at 4 bits a weight (0.47B weights) and its work memory


class KVFull(Exception):
    """The pool of keys and values has no tile left for a sequence's next positions."""


def ring_tiles(cfg: M.Config, kind: str, capacity: int, rows: int = 8) -> int:
    """Tiles a sequence can hold of a kind of layer: a sliding layer keeps the last window + SPARE positions (in
    whole tiles; with passes of 32 rows, window + 32), a full-attention layer every position up to the capacity."""
    return capacity // TILE if kind == M.FULL else -(-(cfg.window + max(SPARE, rows)) // TILE)


def tile_bytes(cfg: M.Config, kind: str) -> int:
    """VRAM of one tile of a kind: 16 positions of keys and values (int8, and a float32 scale a head) in every layer
    of the kind."""
    return cfg.layer_types.count(kind) * TILE * cfg.kv_heads[kind] * (cfg.head_dim[kind] + 4) * 2


def fixed_bytes(cfg: M.Config, head_bits: int, capacity: int, pool: dict | None = None, rows: int = 8) -> int:
    """VRAM an Engine takes besides the layers' codes: the head's codes, the keys and values, the work memory.
    pool: tiles of keys and values by kind of layer (default: what one sequence of `capacity` positions can hold);
    rows: the rows of its passes."""
    head = cfg.vocab * cfg.hidden * head_bits // 16
    kinds = dict.fromkeys(cfg.layer_types)
    kv = sum(tile_bytes(cfg, t) * (ring_tiles(cfg, t, capacity, rows) if pool is None else pool[t]) for t in kinds)
    # the logits of the main model and the drafter, attention's scores and weights, and the rest
    work = 2 * rows * cfg.vocab * 4 + cfg.heads * rows * (max(capacity, cfg.window + rows) + 32) * 6 + (32 << 20)
    return head + kv + work


class Mat(C.Structure):
    _fields_ = [("codes", C.c_void_p), ("lut", C.c_void_p), ("scale", C.c_void_p), ("rows", C.c_int), ("cols", C.c_int),
                ("bits", C.c_int), ("pad", C.c_int)]


class Rot(C.Structure):
    _fields_ = [("signs", C.c_void_p), ("block", C.c_int), ("pad", C.c_int)]


class Layer(C.Structure):
    _fields_ = ([(m, Mat) for m in ("q", "k", "v", "o", "gate", "up", "down")] +
                [(r, Rot) for r in ("rot_in", "rot_o", "rot_mlp", "rot_down")] +
                [(p, C.c_void_p) for p in ("in_ln", "post_attn_ln", "pre_ff_ln", "post_ff_ln", "q_norm", "k_norm", "inv_freq", "keys", "values",
                                           "tiles", "kscale", "vscale")] +
                [("scalar", C.c_float), ("full", C.c_int), ("head_dim", C.c_int), ("kv_heads", C.c_int), ("ring", C.c_int), ("pad", C.c_int)])


class Segs(C.Structure):
    """The rows of a pass by sequence (engine/sclmd.cu)."""
    _fields_ = [("n", C.c_int)] + [(name, C.c_int * MAX_TOKENS) for name in ("row0", "cnt", "pos0", "seq")]


class Model(C.Structure):
    _fields_ = ([("layers", C.c_void_p)] + [(n, C.c_int) for n in ("n_layers", "hidden", "mlp", "heads", "window", "vocab")] +
                [("eps", C.c_float), ("softcap", C.c_float), ("head", Mat), ("rot_head", Rot), ("norm", C.c_void_p)] +
                [(p, C.c_void_p) for p in ("h", "h2", "fps", "xf", "y", "y2", "qhat", "scores", "probs", "attn", "hn", "logits")])


_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = C.CDLL(os.path.abspath(DLL))  # finds the CUDA runtime torch has already loaded
        sizes = [C.c_int() for _ in range(5)]
        assert _lib.sclmd_sizes(*[C.byref(s) for s in sizes]) == MAX_TOKENS
        assert [s.value for s in sizes] == [C.sizeof(Mat), C.sizeof(Rot), C.sizeof(Layer), C.sizeof(Model), C.sizeof(Segs)], "struct layouts differ"
        _lib.sclmd_pass.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
        if hasattr(_lib, "sclmd_pass_from"):  # an older build (SCLMD_DLL, to compare builds) computes every row's logits
            _lib.sclmd_pass_from.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_int]
        _lib.sclmd_linear.argtypes = [C.c_void_p] * 5 + [C.c_int]
    return _lib


# lane 4g + t of a tile holds the pairs (row g, pair t), (row g + 8, pair t), (row g, pair t + 4), (row g + 8, pair t + 4)
_LANE = torch.arange(32)
_ROW = (_LANE // 4)[:, None] + torch.tensor([0, 8, 0, 8])[None, :]
_PAIR = (_LANE % 4)[:, None] + torch.tensor([0, 0, 4, 4])[None, :]


def words(codes: torch.Tensor, bits: int) -> int:
    """int32 words pack() makes of codes [rows, pairs]: four codes a word at 8 bits, sixteen in three at 6."""
    rows, pairs = codes.shape
    return rows // 16 * (4 * pairs if bits == 8 else 3 * pairs)


def pack(codes: torch.Tensor, bits: int, strips_at_once: int = 64, device=None, out: torch.Tensor | None = None) -> torch.Tensor:
    """codes (uint8 [rows, cols / 2], a code per pair of columns) in the order the kernel reads them (int32), on
    `device` (default: where the codes are), or into `out` (words(codes, bits) long). Done a part of the rows at a
    time, each part moved to the device by itself: the temporaries are 16 times the codes, and the head's codes
    alone are 0.7 GB."""
    rows, pairs = codes.shape
    assert rows % 16 == 0 and pairs % 32 == 0 and bits in (6, 8)
    per_strip = words(codes, bits) // (rows // 16)
    if out is None:
        out = torch.empty(rows // 16 * per_strip, dtype=torch.int32, device=codes.device if device is None else device)
    assert out.numel() == rows // 16 * per_strip
    device = out.device
    step = 16 * strips_at_once
    for a in range(0, rows, step):
        out[a // 16 * per_strip : (a + step) // 16 * per_strip] = _pack(codes[a : a + step].to(device), bits)
    return out


def _pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    rows, pairs = codes.shape
    strips, tiles = rows // 16, pairs // 8
    c = codes.view(strips, 16, tiles, 8).permute(0, 2, 1, 3)  # [strip, tile, row in strip, pair in tile]
    sel = c[:, :, _ROW.to(codes.device), _PAIR.to(codes.device)].long()  # [strip, tile, lane, 4]
    if bits == 8:
        word = sel[..., 0] | sel[..., 1] << 8 | sel[..., 2] << 16 | sel[..., 3] << 24
        return word.to(torch.int32).contiguous().view(-1)  # values past 2^31 wrap to the same bits
    # 6 bits: a lane's 16 codes of four tiles are 96 bits, three words stored [strip, four tiles, word, lane]
    s = sel.view(strips, tiles // 4, 4, 32, 4).permute(0, 1, 3, 2, 4).reshape(strips, tiles // 4, 32, 16)
    lo = torch.zeros_like(s[..., 0])
    for n in range(10):
        lo |= s[..., n] << (6 * n)
    lo |= (s[..., 10] & 15) << 60  # may set the sign bit: only the bits matter
    hi = s[..., 10] >> 4
    for n in range(11, 16):
        hi |= s[..., n] << (6 * n - 64)
    words = torch.stack((lo & 0xFFFFFFFF, (lo >> 32) & 0xFFFFFFFF, hi), dim=2)  # [strip, four tiles, 3, lane]
    return words.to(torch.int32).contiguous().view(-1)


def lut(book: torch.Tensor) -> torch.Tensor:
    """A codebook of pairs [codes, 2] as one word a pair: two fp16 (int32)."""
    h = book.half().contiguous().view(torch.int16).long() & 0xFFFF
    return (h[:, 0] | h[:, 1] << 16).to(torch.int32)


class Engine:
    """The main model on the GPU kernels. The same four things generation asks of RefTarget (sclmd/generate.py).

    It holds `sequences` sequences (conversations) side by side; a pass may carry tokens of several (run). A
    sequence's keys and values sit in tiles of 16 positions handed out from a pool as it grows (KVFull when the pool
    is empty); `pool` is the tiles by kind of layer, by default what `sequences` sequences of `capacity` positions
    can hold at most. Where a method takes `seq`, it is the sequence's number, 0 by default.

    layers: per layer a dict with
      "mats": matrix -> (codes uint8 [rows, cols / 2], scales [rows], book [2**bits, 2], bits)
      and the layer's small weights by the names of model.LAYER_KEYS (in_ln, q_norm, ..., scalar)
    head: the output matrix as (codes, scales, book, bits); norm: the final norm's weight
    embed: the input embedding table [vocab, hidden] (any device and type; rows are fetched as needed)"""

    def __init__(self, cfg: M.Config, layers: list, head: tuple, norm: torch.Tensor, embed: torch.Tensor, capacity: int = 8192,
                 device: str = "cuda", sequences: int = 1, pool: dict | None = None, rows: int = 8):
        assert capacity % TILE == 0, "the capacity is a multiple of 16"
        assert rows in (8, 16, 32), "a pass has 8, 16 or 32 rows"
        self.cfg, self.device, self.capacity, self.sequences, self.rows = cfg, device, capacity, sequences, rows
        self.lengths = [0] * sequences  # positions each sequence holds
        self.highs = [0] * sequences  # one past the furthest position written since its start: what a sliding layer's ring has lost follows from it
        self._snaps = {}
        self.embed_table, self.embed_scale = embed, float(M.embed_scale(cfg, embed.dtype))
        self.keep = []  # every tensor the DLL has the address of
        self.rots = {}
        f32 = lambda t: self._own(t.to(device=device, dtype=torch.float32).contiguous())  # noqa: E731
        self.c_layers = (Layer * cfg.n_layers)()
        kinds = list(dict.fromkeys(cfg.layer_types))
        # The tiles: per kind of layer, how many a sequence's table has (ring), how many there are (pool), which are
        # free, which each sequence holds (tiles[kind][seq][j]: its positions 16j.., j modulo ring; -1: none), and
        # the same table on the GPU for the kernels. Tiles are handed out lowest first: one sequence alone has
        # tile j for its positions 16j..
        self.ring = {t: ring_tiles(cfg, t, capacity, rows) for t in kinds}
        self.pool = {t: sequences * self.ring[t] for t in kinds} if pool is None else {t: int(pool[t]) for t in kinds}
        self.free = {t: list(range(self.pool[t] - 1, -1, -1)) for t in kinds}
        self.tiles = {t: [[-1] * self.ring[t] for _ in range(sequences)] for t in kinds}
        self.tables = {t: self._own(torch.zeros(sequences, self.ring[t], dtype=torch.int32, device=device)) for t in kinds}
        # Few, large allocations: PyTorch rounds each up to 2 MiB, and a tensor a matrix and a layer's keys wasted 0.57 GiB.
        # The keys and values of all layers of a kind are one tensor each: [layers of the kind, pool slots, heads, head_dim],
        # int8, and their scales [layers, pool slots, heads]: (keys, values, keys' scales, values' scales)
        self.kv_all = {t: tuple(self._own(torch.zeros(cfg.layer_types.count(t), self.pool[t] * TILE, cfg.kv_heads[t], cfg.head_dim[t],
                                                      dtype=torch.int8, device=device)) for _ in range(2))
                       + tuple(self._own(torch.zeros(cfg.layer_types.count(t), self.pool[t] * TILE, cfg.kv_heads[t], device=device)) for _ in range(2))
                       for t in kinds}
        self.kv = []
        for i, (lay, cl) in enumerate(zip(layers, self.c_layers)):
            t = cfg.layer_types[i]
            hd, kvh = cfg.head_dim[t], cfg.kv_heads[t]
            mats = {name: lay["mats"][name] for name in M.LINEARS if name in lay["mats"]}
            codes_all = self._own(torch.empty(sum(words(c, b) for c, _, _, b in mats.values()), dtype=torch.int32, device=device))
            at = 0
            for name, (codes, scales, book, bits) in mats.items():  # the layer's codes are one tensor
                setattr(cl, name, self._mat(codes, scales, book, bits, codes_all[at : at + words(codes, bits)]))
                at += words(codes, bits)
            cl.rot_in, cl.rot_o = self._rot(cfg.hidden), self._rot(cfg.heads * hd)
            cl.rot_mlp, cl.rot_down = self._rot(cfg.hidden), self._rot(cfg.mlp)
            for name in ("in_ln", "post_attn_ln", "pre_ff_ln", "post_ff_ln", "q_norm", "k_norm"):
                setattr(cl, name, f32(lay[name]).data_ptr())
            theta, rotated = cfg.rope[t]
            inv = torch.zeros(hd // 2)
            inv[:rotated] = 1.0 / (theta ** (torch.arange(0, 2 * rotated, 2).float() / hd))
            cl.inv_freq = f32(inv).data_ptr()
            nth = cfg.layer_types[:i].count(t)
            keys, values, kscale, vscale = (x[nth] for x in self.kv_all[t])
            self.kv.append((keys, values, kscale, vscale))
            cl.keys, cl.values, cl.tiles = keys.data_ptr(), values.data_ptr(), self.tables[t].data_ptr()
            cl.kscale, cl.vscale = kscale.data_ptr(), vscale.data_ptr()
            cl.scalar, cl.full, cl.head_dim, cl.kv_heads, cl.ring = float(lay["scalar"]), int(t == M.FULL), hd, kvh, self.ring[t]
            if device != "cpu":
                torch.cuda.empty_cache()  # packing's temporaries, or the allocator keeps them beside the codes
        widest_head = cfg.heads * max(cfg.head_dim.values())
        y_size = max(max(cfg.heads * cfg.head_dim[t] + 2 * cfg.kv_heads[t] * cfg.head_dim[t] for t in cfg.head_dim), cfg.mlp, cfg.hidden)
        x_size = max(cfg.hidden, widest_head, cfg.mlp)
        # attention takes the positions a pass reads in whole tiles of 16: up to 15 more at either end
        scores = cfg.heads * rows * (max(capacity, cfg.window + rows) + 32)
        z = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
        # the kernels' work memory, by the names of the DLL's Model (what a pass leaves in them is the last layer's)
        self.work = dict(h=z(rows, cfg.hidden), hn=z(rows, cfg.hidden), logits=z(rows, cfg.vocab),
                         h2=z(rows, cfg.hidden), fps=z(3 * 32), xf=z(rows // 8, x_size // 16, 32, dtype=torch.int64),  # fps: scales, then prep_k's sums (3 * MAX_TOKENS)
                         y=z(rows, y_size), y2=z(rows, cfg.mlp), qhat=z(rows, widest_head, dtype=torch.float16),
                         attn=z(rows, widest_head), scores=z(scores), probs=z(scores, dtype=torch.float16))
        self.h, self.hn, self.logits = self.work["h"], self.work["hn"], self.work["logits"]
        m = self.model = Model()
        m.layers = C.cast(self.c_layers, C.c_void_p)
        m.n_layers, m.hidden, m.mlp, m.heads, m.window, m.vocab = cfg.n_layers, cfg.hidden, cfg.mlp, cfg.heads, cfg.window, cfg.vocab
        m.eps, m.softcap = cfg.eps, cfg.softcap or 0.0
        m.head, m.rot_head, m.norm = self._mat(*head), self._rot(cfg.hidden), f32(norm).data_ptr()
        for name, t in self.work.items():
            setattr(m, name, t.data_ptr())
        self.lib = lib()

    @classmethod
    def load(cls, name: str, plan: dict, head: int, capacity: int = 8192, device: str = "cuda", sequences: int = 1,
             pool: dict | None = None, rows: int = 8) -> "Engine":
        """The original model with the stored codes `name`: plan is layer -> {matrix: bits a pair}, head the output
        head's bits. The layers come off the disk one at a time; the embedding table stays in RAM."""
        from . import ckpt
        from .checkpoint import Checkpoint

        ck = Checkpoint(ckpt("orig"))
        cfg = ck.cfg

        def layers():
            for i in range(cfg.n_layers):
                has = [m for m in M.LINEARS if not (m == "v" and cfg.layer_types[i] == M.FULL)]
                yield ck.layer(i, "cpu", skip=M.LINEARS) | {"mats": CD.stored(name, i, {m: plan[i][m] for m in has})}

        eng = cls(cfg, layers(), CD.stored(name, "head", {"head": head})["head"], ck.norm("cpu"), ck.embed("cpu"), capacity, device,
                  sequences, pool, rows)
        torch.cuda.empty_cache()  # packing left about 1 GiB reserved and unused: whatever loads next needs it
        return eng

    def bytes(self) -> int:
        """VRAM held: the codes, the keys and values, the work memory."""
        return sum(t.numel() * t.element_size() for t in self.keep + list(self.work.values()) if t.is_cuda)

    def _own(self, t: torch.Tensor) -> torch.Tensor:
        self.keep.append(t)
        return t

    def _mat(self, codes: torch.Tensor, scales: torch.Tensor, book: torch.Tensor, bits: int, out: torch.Tensor | None = None) -> Mat:
        dev = self.device
        packed = pack(codes, bits, out=out) if out is not None else self._own(pack(codes, bits, device=dev))
        table = self._own(lut(book.to(dev)))
        scale = self._own(scales.to(device=dev, dtype=torch.float32).contiguous())
        return Mat(packed.data_ptr(), table.data_ptr(), scale.data_ptr(), codes.shape[0], codes.shape[1] * 2, bits, 0)

    def _rot(self, cols: int) -> Rot:
        if cols not in self.rots:
            r = CD.rotation(cols, self.device)
            self.rots[cols] = (self._own(r.signs.float().contiguous()), r.b)
        signs, block = self.rots[cols]
        return Rot(signs.data_ptr(), block, 0)

    def _check(self, status: int):
        assert status == 0, f"GPU kernel launch failed ({status})"

    def linear(self, mat: Mat, x: torch.Tensor) -> torch.Tensor:
        """One matrix times x [tokens <= rows, cols] (tests)."""
        x = x.to(device=self.device, dtype=torch.float32).contiguous()
        y = torch.zeros(x.shape[0], mat.rows, device=self.device)
        rot = self._rot(mat.cols)
        self._check(self.lib.sclmd_linear(C.byref(self.model), C.byref(mat), C.byref(rot), x.data_ptr(), y.data_ptr(), x.shape[0]))
        return y

    # what generation needs (sclmd/generate.py)

    @property
    def length(self) -> int:
        return self.lengths[0]

    @length.setter
    def length(self, value: int):
        self.lengths[0] = value

    @property
    def high(self) -> int:
        return self.highs[0]

    def embed(self, ids: list) -> torch.Tensor:
        rows = self.embed_table[torch.as_tensor(ids, device=self.embed_table.device)]
        return rows.to(device=self.device, dtype=torch.float32) * self.embed_scale

    def lacks(self, seq: int, pos: int, n: int) -> dict:
        """Tiles by kind the sequence would have to be given to write positions pos .. pos + n - 1."""
        return {t: sum(self.tiles[t][seq][j] < 0 for j in dict.fromkeys((q >> 4) % ring for q in range(pos, pos + n)))
                for t, ring in self.ring.items()}

    def _provide(self, need: list):
        """Tiles for the positions a pass is about to write: need is [(sequence, first position, count)]. All or none."""
        want = {t: [] for t in self.ring}
        for t, ring in self.ring.items():
            for seq, pos, n in need:
                for j in dict.fromkeys((q >> 4) % ring for q in range(pos, pos + n)):
                    if self.tiles[t][seq][j] < 0:
                        want[t].append((seq, j))
        for t in want:
            if len(want[t]) > len(self.free[t]):
                raise KVFull(f"no tile of {t} keys and values is free ({self.pool[t]} in all)")
        for t in want:
            for seq, j in want[t]:
                tile = self.free[t].pop()
                self.tiles[t][seq][j] = tile
                self.tables[t][seq, j] = tile

    def advance(self, parts: list, logits_from: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """One pass over the rows that are in self.h: parts is [(sequence, rows)] in row order, each sequence once,
        up to self.rows rows in all; a sequence's rows are its next positions. -> (logits [rows, vocab], outputs
        after the final norm): views of the engine's memory, which the next pass writes over.
        logits_from: the first row whose logits are wanted (the rows before it are a prompt's: their logits are not
        computed and hold what an earlier pass left; the number of rows: none are)."""
        total = sum(n for _, n in parts)
        assert 0 < total <= self.rows and len({seq for seq, _ in parts}) == len(parts) and all(n > 0 for _, n in parts)
        segs, row = Segs(), 0
        for i, (seq, n) in enumerate(parts):
            at = self.lengths[seq]
            assert at + n <= self.capacity, "past the engine's capacity"
            assert self._kept(at - self.cfg.window + 1, seq), "cropped further back than the sliding layers keep"
            segs.row0[i], segs.cnt[i], segs.pos0[i], segs.seq[i] = row, n, at, seq
            row += n
        segs.n = len(parts)
        self._provide([(seq, self.lengths[seq], n) for seq, n in parts])
        if hasattr(self.lib, "sclmd_pass_from"):
            self._check(self.lib.sclmd_pass_from(C.byref(self.model), total, C.byref(segs), logits_from))
        else:
            self._check(self.lib.sclmd_pass(C.byref(self.model), total, C.byref(segs)))
        for seq, n in parts:
            self.lengths[seq] += n
            self.highs[seq] = max(self.highs[seq], self.lengths[seq])
        return self.logits[:total], self.hn[:total]

    def run(self, parts: list, logits_from: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """One pass over tokens of several sequences: parts is [(sequence, token ids)] (see advance)."""
        flat = [t for _, ids in parts for t in ids]
        self.h[: len(flat)] = self.embed(flat)
        return self.advance([(seq, len(ids)) for seq, ids in parts], logits_from)

    def feed(self, ids: list, keep: str = "all", seq: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Run tokens at a sequence's next positions, a pass's rows at a time -> (logits [n, vocab], last outputs after
        the final norm). keep="last": only the last pass's rows (a long prompt's logits would not fit: 1 MB a token)."""
        assert self.lengths[seq] + len(ids) <= self.capacity, "past the engine's capacity"
        logits, hidden = [], []
        for a in range(0, len(ids), self.rows):
            last = a + self.rows >= len(ids)
            # with keep="last" only the last pass's logits are looked at: the passes before compute none
            lg, hn = self.run([(seq, ids[a : a + self.rows])], 0 if keep == "all" or last else self.rows)
            if keep == "all" or last:
                logits.append(lg.clone())
                hidden.append(hn.clone())
        return torch.cat(logits), torch.cat(hidden)

    def _tiles_view(self, t: torch.Tensor, kind: str) -> torch.Tensor:
        """[layers, pool slots, heads (, head_dim)] as [layers, pool tiles, 16, heads (, head_dim)]."""
        return t.view(t.shape[0], self.pool[kind], TILE, *t.shape[2:])

    def kv_float(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A layer's keys and values as numbers: [pool slots, heads, head_dim] (the int8 values times their scales)."""
        k, v, ks, vs = self.kv[layer]
        return k.float() * ks[..., None], v.float() * vs[..., None]

    def snapshot(self, seq: int = 0):
        """Remembers a sequence's present length so that restore() can come back to it however far the text goes on.
        (crop() cannot: a sliding layer's ring is written over as the text goes on. Its keys and values are copied
        to RAM here, 0.43 GB for the real model; one snapshot a sequence.)"""
        if M.SLIDING not in self.kv_all:
            self._snaps[seq] = {"at": (self.lengths[seq], self.highs[seq]), "tiles": [], "data": []}
            return
        own = [x for x in self.tiles[M.SLIDING][seq] if x >= 0]
        snap = self._snaps.get(seq)
        if snap is None or not snap["data"]:
            shape = lambda t: (t.shape[0], self.ring[M.SLIDING], TILE, *t.shape[2:])  # noqa: E731
            snap = self._snaps[seq] = {"data": [torch.empty(shape(t), dtype=t.dtype).pin_memory() for t in self.kv_all[M.SLIDING]]}
        self._copy_tiles(M.SLIDING, own, snap["data"], out=True)
        snap["tiles"], snap["at"] = own, (self.lengths[seq], self.highs[seq])

    def _copy_tiles(self, kind: str, own: list, data: list, out: bool):
        """The tiles `own` of a kind (keys, values and their scales) to (out) or from the RAM copies `data`, a layer at
        a time (a copy of all layers at once would need its size again in VRAM when the tiles are not next to each other)."""
        if not own:
            return
        run = own == list(range(own[0], own[0] + len(own)))  # next to each other in the pool: a slice
        idx = None if run else torch.tensor(own, device=self.device)
        for t, ram in zip(self.kv_all[kind], data):
            v = self._tiles_view(t, kind)
            for lay in range(v.shape[0]):
                if out:
                    ram[lay, : len(own)].copy_(v[lay, own[0] : own[0] + len(own)] if run else v[lay, idx])
                elif run:
                    v[lay, own[0] : own[0] + len(own)].copy_(ram[lay, : len(own)])
                else:
                    v[lay, idx] = ram[lay, : len(own)].to(self.device)

    def restore(self, seq: int = 0):
        """Back to the snapshot's length: every position from there on is forgotten."""
        snap = self._snaps[seq]
        self._copy_tiles(M.SLIDING, snap["tiles"], snap["data"], out=False)
        self.lengths[seq], self.highs[seq] = snap["at"]

    def park(self, seq: int) -> dict:
        """Takes a sequence off the GPU: what it holds comes back as a state in RAM (its keys and values up to its
        length: 0.42 MB a token for the last 1,040 tokens and 41 KB a token), and its tiles go back to the pool.
        unpark() puts it on any free sequence later. A snapshot of it is given up."""
        self.trim(seq)
        state = {"length": self.lengths[seq], "high": self.highs[seq], "kinds": {}, "bytes": 0}
        for t in self.ring:
            js = [j for j, tile in enumerate(self.tiles[t][seq]) if tile >= 0]
            data = [torch.empty(k.shape[0], len(js), TILE, *k.shape[2:], dtype=k.dtype) for k in self.kv_all[t]]
            self._copy_tiles(t, [self.tiles[t][seq][j] for j in js], data, out=True)
            state["kinds"][t] = {"js": js, "data": data}
            state["bytes"] += sum(d.numel() * d.element_size() for d in data)
        self.release(seq)
        return state

    def unpark(self, seq: int, state: dict):
        """Puts a parked sequence on the free sequence `seq`. KVFull, and nothing changed, when the pool is short."""
        assert self.lengths[seq] == 0 and all(x < 0 for t in self.ring for x in self.tiles[t][seq]), "the sequence is in use"
        for t, part in state["kinds"].items():
            if len(part["js"]) > len(self.free[t]):
                raise KVFull(f"no room for a parked sequence's {len(part['js'])} tiles of {t} keys and values")
        for t, part in state["kinds"].items():
            own = [self.free[t].pop() for _ in part["js"]]
            for j, tile in zip(part["js"], own):
                self.tiles[t][seq][j] = tile
            if own:
                self.tables[t][seq, torch.tensor(part["js"], device=self.device)] = torch.tensor(own, dtype=torch.int32, device=self.device)
            self._copy_tiles(t, own, part["data"], out=False)
        self.lengths[seq], self.highs[seq] = state["length"], state["high"]

    def kept(self) -> int:
        """Positions a sliding layer keeps of a sequence (its ring); without sliding layers, every position."""
        return self.ring[M.SLIDING] * TILE if M.SLIDING in self.ring else self.capacity

    def _kept(self, first: int, seq: int = 0) -> bool:
        """Whether the sliding layers still hold every position of the sequence from `first` on: their ring has been
        written up to position high - 1, so it holds the positions from high - its size."""
        return max(first, 0) >= self.highs[seq] - self.kept()

    def crop(self, length: int, seq: int = 0):
        """Forget every position from `length` on (a later pass writes over them before anything reads them)."""
        self.lengths[seq] = length
        if length == 0:
            self.highs[seq] = 0

    def release(self, seq: int):
        """The sequence is over: its tiles go back to the pool."""
        for t in self.ring:
            self.free[t] += sorted((x for x in self.tiles[t][seq] if x >= 0), reverse=True)
            self.tiles[t][seq] = [-1] * self.ring[t]
        self.lengths[seq] = self.highs[seq] = 0
        self._snaps.pop(seq, None)

    def trim(self, seq: int):
        """Gives back the tiles past the sequence's present length (after a crop): those of the full-attention
        layers, and those of the sliding layers while their ring has not gone round."""
        last = (self.lengths[seq] - 1) >> 4  # the last tile still in use (-1: none)
        for t in self.ring:
            if t != M.FULL and self.highs[seq] > self.kept():
                continue
            own = self.tiles[t][seq]
            for j in range(last + 1, self.ring[t]):
                if own[j] >= 0:
                    self.free[t].append(own[j])
                    own[j] = -1
            self.free[t].sort(reverse=True)
            if t != M.FULL:
                self.highs[seq] = self.lengths[seq]

    def free_tiles(self) -> dict:
        return {t: len(f) for t, f in self.free.items()}

    def slots(self, kind: str, positions: torch.Tensor, seq: int = 0) -> torch.Tensor:
        """Where a sequence's positions sit in the pool of a kind of layer."""
        own = torch.tensor(self.tiles[kind][seq], device=positions.device)
        return own[(positions >> 4) % self.ring[kind]] * TILE + (positions & 15)

    def keys(self, length: int, seq: int = 0) -> dict:
        """What the drafter attends to: per layer type the keys and values of the last layer of that type, for the
        positions the drafter may read, as (keys [1, heads, m, head_dim], values, positions): the numbers as they are
        kept (int8 times the scale)."""
        assert length <= self.lengths[seq] and self._kept(length - (self.cfg.window + 1), seq), "further back than the sliding layers keep"
        out = {}
        for t in (M.SLIDING, M.FULL):
            k, v, ks, vs = self.kv[self.cfg.last_of(t)]
            first = 0 if t == M.FULL else max(0, length - (self.cfg.window + 1))
            pos = torch.arange(first, length, device=self.device)
            slot = self.slots(t, pos, seq)
            out[t] = ((k[slot].float() * ks[slot][..., None]).transpose(0, 1)[None], (v[slot].float() * vs[slot][..., None]).transpose(0, 1)[None], pos)
        return out


# ---- The drafter on the engine's kernels (engine/sclmd.cu: sclmd_draft)

class DLayer(C.Structure):
    _fields_ = ([(m, Mat) for m in ("q", "o", "gate", "up", "down")] + [(r, Rot) for r in ("rot_in", "rot_o", "rot_mlp", "rot_down")] +
                [(p, C.c_void_p) for p in ("in_ln", "post_attn_ln", "pre_ff_ln", "post_ff_ln", "q_norm", "inv_freq")] +
                [("scalar", C.c_float), ("full", C.c_int), ("head_dim", C.c_int), ("kv_heads", C.c_int)])


class DrafterC(C.Structure):
    _fields_ = ([("layers", C.c_void_p)] + [(n, C.c_int) for n in ("n_layers", "hidden", "mlp", "heads", "vocab", "backbone")] +
                [("eps", C.c_float), ("pad", C.c_int), ("pre", Mat), ("post", Mat), ("head", Mat), ("rot_pre", Rot), ("rot_out", Rot),
                 ("norm", C.c_void_p)] +
                [(p, C.c_void_p) for p in ("inp", "h", "hn", "logits", "next", "h2", "fps", "xf", "y", "y2", "qhat", "scores", "probs", "attn")])


class EngineDrafter:
    """The drafter beside an Engine: guess(token, hidden, length) as generation asks for it (sclmd/generate.py).
    Its matrices are pair codes like the main model's (it only guesses: what the codes lose shows as fewer guesses
    kept, never in the text); its work memory is the engine's, which is idle while the drafter guesses.

    layers: per layer a dict with "mats": q, o, gate, up, down -> (codes, scales, book, bits), and the layer's small
    weights by the names of model.LAYER_KEYS. pre, post, head: the same four things for the three other matrices."""

    def __init__(self, eng: Engine, cfg: M.Config, layers: list, pre: tuple, post: tuple, norm: torch.Tensor, head: tuple, depth: int = 6):
        self.eng, self.cfg, self.depth, dev = eng, cfg, depth, eng.device
        main = eng.cfg
        assert cfg.kv_shared == cfg.n_layers and cfg.head_dim == main.head_dim and cfg.kv_heads == main.kv_heads
        assert cfg.window == main.window and pre[0].shape[1] == main.hidden  # the joined input: a code a pair of columns
        self.keep = []
        f32 = lambda t: self._own(t.to(device=dev, dtype=torch.float32).contiguous())  # noqa: E731
        every = [m for lay in layers for m in lay["mats"].values()] + [pre, post, head]
        self._codes = self._own(torch.empty(sum(words(c, b) for c, _, _, b in every), dtype=torch.int32, device=dev))  # one allocation
        self._at = 0
        self.c_layers = (DLayer * cfg.n_layers)()
        for i, (lay, cl) in enumerate(zip(layers, self.c_layers)):
            t = cfg.layer_types[i]
            hd = cfg.head_dim[t]
            for name in ("q", "o", "gate", "up", "down"):
                setattr(cl, name, self._mat(*lay["mats"][name]))
            cl.rot_in, cl.rot_o, cl.rot_mlp, cl.rot_down = self._rot(cfg.hidden), self._rot(cfg.heads * hd), self._rot(cfg.hidden), self._rot(cfg.mlp)
            for name in ("in_ln", "post_attn_ln", "pre_ff_ln", "post_ff_ln", "q_norm"):
                setattr(cl, name, f32(lay[name]).data_ptr())
            theta, rotated = cfg.rope[t]
            inv = torch.zeros(hd // 2)
            inv[:rotated] = 1.0 / (theta ** (torch.arange(0, 2 * rotated, 2).float() / hd))
            cl.inv_freq = f32(inv).data_ptr()
            cl.scalar, cl.full, cl.head_dim, cl.kv_heads = float(lay["scalar"]), int(t == M.FULL), hd, cfg.kv_heads[t]
        z = lambda *shape: self._own(torch.zeros(eng.rows, *shape, dtype=torch.float32, device=dev))  # noqa: E731  (a row a sequence)
        self.inp, self.h, self.hn, self.logits, self.next = z(2 * main.hidden), z(cfg.hidden), z(cfg.hidden), z(cfg.vocab), z(main.hidden)
        self.h2 = z(cfg.hidden)
        c = self.c = DrafterC()
        c.layers = C.cast(self.c_layers, C.c_void_p)
        c.n_layers, c.hidden, c.mlp, c.heads, c.vocab, c.backbone = cfg.n_layers, cfg.hidden, cfg.mlp, cfg.heads, cfg.vocab, main.hidden
        c.eps = cfg.eps
        c.pre, c.post, c.head, c.norm = self._mat(*pre), self._mat(*post), self._mat(*head), f32(norm).data_ptr()
        c.rot_pre, c.rot_out = self._rot(2 * main.hidden), self._rot(cfg.hidden)
        for name in ("inp", "h", "hn", "logits", "next", "h2"):
            setattr(c, name, getattr(self, name).data_ptr())
        work = eng.work  # wide enough for the main model's matrices: checked here for the drafter's
        widest = max(2 * main.hidden, cfg.mlp, cfg.heads * max(cfg.head_dim.values()))
        assert work["xf"].shape[1] * 16 >= widest and work["y"].shape[1] >= widest and work["y2"].shape[1] >= cfg.mlp
        assert cfg.heads <= main.heads
        for name in ("fps", "xf", "y", "y2", "qhat", "scores", "probs", "attn"):
            setattr(c, name, work[name].data_ptr())
        self.sliding, self.full = eng.c_layers[main.last_of(M.SLIDING)], eng.c_layers[main.last_of(M.FULL)]
        self.lib = lib()
        sizes = [C.c_int() for _ in range(2)]
        self.lib.sclmd_draft_sizes(*[C.byref(s) for s in sizes])
        assert [s.value for s in sizes] == [C.sizeof(DLayer), C.sizeof(DrafterC)], "struct layouts differ"
        self.lib.sclmd_draft.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_void_p]

    @classmethod
    def load(cls, eng: Engine, folder: str, name: str, bits: int = 8, depth: int = 6) -> "EngineDrafter":
        """The drafter in `folder` with its matrices as the stored codes `name` (scripts/quantize_drafter.py)."""
        from .checkpoint import Checkpoint

        ck = Checkpoint(folder)
        mats = CD.drafter_stored(name, bits)
        layers = [ck.layer(i, "cpu", skip=M.LINEARS) | {"mats": {m: mats[f"layers.{i}.{m}"] for m in ("q", "o", "gate", "up", "down")}}
                  for i in range(ck.cfg.n_layers)]
        return cls(eng, ck.cfg, layers, mats["pre"], mats["post"], ck.norm(), mats["head"], depth)

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.keep)

    def _own(self, t: torch.Tensor) -> torch.Tensor:
        self.keep.append(t)
        return t

    def _mat(self, codes: torch.Tensor, scales: torch.Tensor, book: torch.Tensor, bits: int) -> Mat:
        dev = self.eng.device
        packed = pack(codes, bits, out=self._codes[self._at : self._at + words(codes, bits)])
        self._at += words(codes, bits)
        table = self._own(lut(book.to(dev)))
        scale = self._own(scales.to(device=dev, dtype=torch.float32).contiguous())
        return Mat(packed.data_ptr(), table.data_ptr(), scale.data_ptr(), codes.shape[0], codes.shape[1] * 2, bits, 0)

    def _rot(self, cols: int) -> Rot:
        return self.eng._rot(cols)  # the same rotation as any matrix of the main model with that many columns

    def steps(self, token_embeds: torch.Tensor, hiddens: torch.Tensor, lengths: list, seqs: list) -> tuple[torch.Tensor, torch.Tensor]:
        """One guess for each of several sequences (up to the engine's rows): row r guesses position lengths[r] + 1 of sequence seqs[r] from the
        token at lengths[r] (its embedding in the main model) and a hidden state -> (logits [rows, vocab], the hidden
        states for the guesses after). Both are the drafter's own memory: the next step writes over them."""
        eng, rows = self.eng, len(lengths)
        for length, seq in zip(lengths, seqs):
            assert length <= eng.lengths[seq] and eng._kept(length - (eng.cfg.window + 1), seq), "further back than the sliding layers keep"
        n = eng.cfg.hidden
        self.inp[:rows, :n], self.inp[:rows, n:] = token_embeds, hiddens
        ints = C.c_int * rows
        status = self.lib.sclmd_draft(C.byref(self.c), C.byref(self.sliding), C.byref(self.full), eng.cfg.window, rows, ints(*lengths), ints(*seqs))
        assert status == 0, f"GPU kernel launch failed ({status})"
        return self.logits[:rows], self.next[:rows]

    def step(self, token_embed: torch.Tensor, hidden: torch.Tensor, length: int, seq: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """One guess for position length + 1 of a sequence (see steps)."""
        logits, nxt = self.steps(token_embed[None], hidden[None], [length], [seq])
        return logits[0], nxt[0]

    def guess(self, token: int, hidden: torch.Tensor, length: int, seq: int = 0) -> list:
        """Up to `depth` tokens after `token`, which sits at position `length` (generate.py's guess)."""
        out = []
        for _ in range(self.depth):
            logits, hidden = self.step(self.eng.embed([token])[0], hidden, length, seq)
            token = int(logits.argmax())
            out.append(token)
        return out

    def guess_many(self, asks: list) -> list:
        """Guesses for several sequences at once (up to the engine's rows): asks is [(token, hidden, length, sequence,
        how many)] -> each one's tokens. A step costs about the same for eight sequences as for one."""
        tokens, hidden = [a[0] for a in asks], torch.stack([a[1] for a in asks])
        lengths, seqs, out = [a[2] for a in asks], [a[3] for a in asks], [[] for _ in asks]
        for step in range(max(a[4] for a in asks)):
            logits, hidden = self.steps(self.eng.embed(tokens), hidden, lengths, seqs)
            tokens = logits.argmax(-1).tolist()
            for r, a in enumerate(asks):
                if step < a[4]:
                    out[r].append(tokens[r])
        return out
