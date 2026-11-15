# The engine from Python: the model's pair codes laid out for the GPU kernels (engine/sclmd.cu), the memory they work
# in, and a pass as one call. All memory is PyTorch tensors; the DLL gets their addresses.
import ctypes as C
import os

import torch

from . import codes as CD
from . import model as M

# SCLMD_DLL: another build of the engine (to compare builds: scripts/bits_check.py)
DLL = os.environ.get("SCLMD_DLL") or os.path.join(os.path.dirname(__file__), "..", "engine", "build", "sclmd.dll")
MAX_TOKENS = 256  # the most rows the kernels take in a pass; an engine is made for 8, 16, 32, 64, 128 or 256 (Engine.rows)
ROWS = (8, 16, 32, 64, 128, 256)
MAX_SEGS = 64  # segments of a pass (engine/sclmd.cu)
# a wider pass's attention takes as many rows at a time as its memory holds (sized for this many at the whole context),
# and the pass has the logits of its last ATT_ROWS rows at most
ATT_ROWS = 32
TILE = 16   # positions of keys and values a sequence is given at a time
SPARE = 16  # positions a sliding layer keeps past its window at least: a pass writes its rows while its first row still reads a window back
DRAFTER_BYTES = 240 << 20  # the drafter at 4 bits a weight (0.47B weights) and its work memory


def logits_first(rows: int, logits_from: int) -> int:
    """The first row a pass of `rows` rows has logits of (engine/sclmd.cu): from logits_from's group of eight on, and
    in a pass of more than ATT_ROWS rows of its last ATT_ROWS at most."""
    asked = max(0, logits_from) // 8 * 8
    return max(asked, (rows - ATT_ROWS + 7) // 8 * 8) if rows > ATT_ROWS else asked


class KVFull(Exception):
    """The pool of keys and values has no tile left for a sequence's next positions."""


def ring_tiles(cfg: M.Config, kind: str, capacity: int, rows: int = 8) -> int:
    """Tiles a sequence can hold of a kind of layer: a sliding layer keeps the last window + SPARE positions (in
    whole tiles; with passes of 32 rows, window + 32), a full-attention layer every position up to the capacity."""
    return capacity // TILE if kind == M.FULL else -(-(cfg.window + max(SPARE, rows)) // TILE)


def keyless_kind(kind: str, keyless: bool) -> bool:
    """Whether the layers of a kind keep their values only in an engine made with keyless=True. A full-attention
    layer's value is its key's projection (normalized; the key is that times k_norm, turned by the position), so the
    scores can make the key of the kept value where they read it (engine/sclmd.cu: scores_made32_k), and a position
    costs half the memory: 21 KB a token for 41. The key is then made of the value as it was rounded to int8, not
    rounded by itself: other bits than an engine with keys."""
    return keyless and kind == M.FULL


def turned(cfg: M.Config, kind: str) -> int:
    """The rotated pairs of a kind's heads in whole 16s: the first that many values of each half of a head are turned."""
    return -(-cfg.rope[kind][1] // 16) * 16


def kv4_layers(cfg: M.Config) -> list:
    """The layers an engine made with kv4=True keeps at 4 bits (brief K3, sclmd/kvsim.py's k4v4t0): the sliding layers
    other than the last one, whose keys and values the drafter reads."""
    if M.SLIDING not in cfg.layer_types:
        return []
    last = cfg.last_of(M.SLIDING)
    return [i for i, t in enumerate(cfg.layer_types) if t == M.SLIDING and i != last]


def extra_tiles(cfg: M.Config, kind: str, kv4: bool = False) -> int:
    """Tiles a sequence holds of a kind besides its ring: with kv4, a sliding tile for the int8 of its positions 8..15
    (its first tile is kept at int8, which takes two 4-bit tiles' bytes: positions 0..7 sit in its ring tile 0)."""
    return 1 if kv4 and kind == M.SLIDING and kv4_layers(cfg) else 0


def tile_bytes(cfg: M.Config, kind: str, keyless: bool = False, kv4: bool = False) -> int:
    """VRAM of one tile of a kind: 16 positions of keys and values (int8, and a float32 scale a head) in every layer
    of the kind; of the values only where the layers keep no keys (keyless); at 4 bits in kv4's layers."""
    n, kvh, hd = cfg.layer_types.count(kind), cfg.kv_heads[kind], cfg.head_dim[kind]
    if kv4 and kind == M.SLIDING:
        n4 = len(kv4_layers(cfg))
        return (n - n4) * TILE * kvh * (hd + 4) * 2 + n4 * TILE * kvh * (hd // 2 + 4) * 2
    return n * TILE * kvh * (hd + 4) * (1 if keyless_kind(kind, keyless) else 2)


def fixed_bytes(cfg: M.Config, head_bits: int, capacity: int, pool: dict | None = None, rows: int = 8, keyless: bool = False,
                kv4: bool = False) -> int:
    """VRAM an Engine takes besides the layers' codes: the head's codes, the keys and values, the work memory.
    pool: tiles of keys and values by kind of layer (default: what one sequence of `capacity` positions can hold);
    rows: the rows of its passes."""
    head = cfg.vocab * cfg.hidden * head_bits // 16
    kinds = dict.fromkeys(cfg.layer_types)
    kv = sum(tile_bytes(cfg, t, keyless, kv4) * (ring_tiles(cfg, t, capacity, rows) + extra_tiles(cfg, t, kv4) if pool is None else pool[t])
             for t in kinds)
    # the logits of the main model and the drafter, attention's scores and weights, and the rest
    narrow = min(rows, ATT_ROWS)  # the logits and attention's scores are of ATT_ROWS rows at most
    work = 2 * narrow * cfg.vocab * 4 + cfg.heads * narrow * (max(capacity, cfg.window + rows) + 32) * 6 + (32 << 20)
    # the rest (32 MiB) holds the rows' own work memory of up to ATT_ROWS rows (0.36 MiB a row for the real model); the
    # rows of a wider engine past those are counted (256 rows: 80 MiB more)
    work += max(0, rows - ATT_ROWS) * row_bytes(cfg, kv4)
    work += sum(capacity * turned(cfg, t) * 4 for t in kinds if keyless_kind(t, keyless))  # cos and sin of every position
    return head + kv + work


def work_sizes(cfg: M.Config, kv4: bool = False) -> tuple:
    """(widest head, a matrix's widest output, its widest input): the values a row of an Engine's work memory holds.
    kv4: with 4-bit layers (kv4_layers), whose queries are kept twice (turned and as they are) and their output twice
    (turned and back), every row's (the real model's widest head, 512, is already twice its sliding one)."""
    widest_head = cfg.heads * max(cfg.head_dim.values())
    if kv4 and kv4_layers(cfg):
        widest_head = max(widest_head, 2 * cfg.heads * cfg.head_dim[M.SLIDING])
    y_size = max(max(cfg.heads * cfg.head_dim[t] + 2 * cfg.kv_heads[t] * cfg.head_dim[t] for t in cfg.head_dim), cfg.mlp, cfg.hidden)
    return widest_head, y_size, max(cfg.hidden, widest_head, cfg.mlp)


def row_bytes(cfg: M.Config, kv4: bool = False) -> int:
    """The bytes of the work memory an Engine has for each row of its passes: h, h2, hn, y, y2, attn (float32), qhat
    (fp16) and xf (fp16 inputs)."""
    widest_head, y_size, x_size = work_sizes(cfg, kv4)
    return 4 * (3 * cfg.hidden + y_size + cfg.mlp + widest_head) + 2 * widest_head + 2 * x_size


class Mat(C.Structure):
    _fields_ = [("codes", C.c_void_p), ("lut", C.c_void_p), ("scale", C.c_void_p), ("rows", C.c_int), ("cols", C.c_int),
                ("bits", C.c_int), ("pad", C.c_int), ("lut8", C.c_void_p), ("book_scale", C.c_float), ("pad2", C.c_int)]


class Rot(C.Structure):
    _fields_ = [("signs", C.c_void_p), ("block", C.c_int), ("pad", C.c_int)]


class Layer(C.Structure):
    _fields_ = ([(m, Mat) for m in ("q", "k", "v", "o", "gate", "up", "down")] +
                [(r, Rot) for r in ("rot_in", "rot_o", "rot_mlp", "rot_down")] +
                [(p, C.c_void_p) for p in ("in_ln", "post_attn_ln", "pre_ff_ln", "post_ff_ln", "q_norm", "k_norm", "inv_freq", "keys", "values",
                                           "tiles", "kscale", "vscale", "trig")] +
                [("scalar", C.c_float), ("full", C.c_int), ("head_dim", C.c_int), ("kv_heads", C.c_int), ("ring", C.c_int), ("turned", C.c_int)] +
                [("kv4", C.c_int), ("kv4_pad", C.c_int), ("kv4_signs", C.c_void_p), ("first", C.c_void_p)])


class Segs(C.Structure):
    """The rows of a pass by sequence (engine/sclmd.cu)."""
    _fields_ = [("n", C.c_int)] + [(name, C.c_int * MAX_SEGS) for name in ("row0", "cnt", "pos0", "seq")]


class Model(C.Structure):
    _fields_ = ([("layers", C.c_void_p)] + [(n, C.c_int) for n in ("n_layers", "hidden", "mlp", "heads", "window", "vocab")] +
                [("eps", C.c_float), ("softcap", C.c_float), ("head", Mat), ("rot_head", Rot), ("norm", C.c_void_p)] +
                [(p, C.c_void_p) for p in ("h", "h2", "fps", "xf", "y", "y2", "qhat", "scores", "probs", "attn", "hn", "logits")] +
                [("int8", C.c_float), ("pad_int8", C.c_int), ("scores_room", C.c_longlong)])


# The integer products (brief B1, lossy, off by default): the layers' matrices take int8 inputs, clipped at INT8_SIGMA
# times their rms, and int8 codebooks, on the s8 tensor-core instruction. SCLMD_INT8=<sigma> turns it on for every
# Engine made without int8= (the scripts' --int8 sets it); 0 or unset: off.
INT8_SIGMA = 5.0


def int8_default() -> float:
    v = os.environ.get("SCLMD_INT8")
    return float(v) if v else 0.0


def add_int8_option(ap):
    """--int8 [SIGMA] on a script's arguments; use_int8_option(args) after parsing turns it on for the Engines it makes."""
    ap.add_argument("--int8", type=float, nargs="?", const=INT8_SIGMA, default=0.0, metavar="SIGMA",
                    help=f"integer products (lossy option, off by default): int8 inputs clipped at SIGMA times their rms (default {INT8_SIGMA:g}) "
                         "and int8 codebooks on the s8 tensor-core instruction")


def use_int8_option(args) -> float:
    if args.int8:
        os.environ["SCLMD_INT8"] = repr(float(args.int8))
    return int8_default()


# Keys and values of the sliding layers at 4 bits (brief K3, lossy, off by default): kv4_layers() keep theirs as 4-bit
# codes in a turned basis, a sequence's first tile at int8 (engine/sclmd.cu: kv4_heads_k). SCLMD_KV4=1 turns it on
# for every Engine made without kv4= (the scripts' --kv4 sets it); 0 or unset: off.
def kv4_default() -> bool:
    return os.environ.get("SCLMD_KV4", "") not in ("", "0")


def add_kv4_option(ap):
    """--kv4 on a script's arguments; use_kv4_option(args) after parsing turns it on for the Engines it makes."""
    ap.add_argument("--kv4", action="store_true",
                    help="the sliding layers (but the drafter's) keep their keys and values at 4 bits (lossy option, off by default): "
                         "0.52 of the memory a position up to 1,024 tokens")


def use_kv4_option(args) -> bool:
    if getattr(args, "kv4", False):
        os.environ["SCLMD_KV4"] = "1"
    return kv4_default()


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


def lut8(book: torch.Tensor) -> tuple[torch.Tensor, float]:
    """The codebook rounded to int8 for the integer products: (one word a pair: the two int8 twice, b0 b1 b0 b1 (int32),
    the scale they are multiplied by: the largest entry / 127)."""
    b = book.float()
    s = float(b.abs().max()) / 127.0 or 1.0
    q = torch.clamp(torch.round(b / s), -127, 127).to(torch.int64) & 0xFF
    two = q[:, 0] | q[:, 1] << 8
    return (two | two << 16).to(torch.int32), s


class Engine:
    """The main model on the GPU kernels. The same four things generation asks of RefTarget (sclmd/generate.py).

    keyless: the full-attention layers keep their values only (keyless_kind: half the memory a position, other bits).

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
                 device: str = "cuda", sequences: int = 1, pool: dict | None = None, rows: int = 8, keyless: bool = False,
                 int8: float | None = None, kv4: bool | None = None):
        assert capacity % TILE == 0, "the capacity is a multiple of 16"
        self.int8 = int8_default() if int8 is None else float(int8)  # 0: off; else the clip in rms of the inputs (INT8_SIGMA)
        self.kv4 = kv4_default() if kv4 is None else bool(kv4)  # brief K3: kv4_layers() keep their keys and values at 4 bits
        assert rows in ROWS, "a pass has 8, 16, 32, 64, 128 or 256 rows"
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
        # with kv4 a sequence's tile list has one more entry past its ring (extra_tiles): the tile of its positions 8..15
        # at int8, which the kernels find in self.firsts
        self.extra = {t: extra_tiles(cfg, t, self.kv4) for t in kinds}
        self.width = {t: self.ring[t] + self.extra[t] for t in kinds}
        self.pool = {t: sequences * self.width[t] for t in kinds} if pool is None else {t: int(pool[t]) for t in kinds}
        self.free = {t: list(range(self.pool[t] - 1, -1, -1)) for t in kinds}
        self.tiles = {t: [[-1] * self.width[t] for _ in range(sequences)] for t in kinds}
        self.tables = {t: self._own(torch.zeros(sequences, self.ring[t], dtype=torch.int32, device=device)) for t in kinds}
        self.kv4_set = set(kv4_layers(cfg)) if self.kv4 else set()
        self.firsts = self._own(torch.zeros(sequences, dtype=torch.int32, device=device)) if self.kv4_set else None
        # Few, large allocations: PyTorch rounds each up to 2 MiB, and a tensor a matrix and a layer's keys wasted 0.57 GiB.
        # The keys and values of all layers of a kind are one tensor each: [layers of the kind, pool slots, heads, head_dim],
        # int8, and their scales [layers, pool slots, heads]: (keys, values, keys' scales, values' scales); of a kind
        # that keeps no keys (keyless): (values, values' scales)
        self.lean = keyless  # the option; self.keyless: by kind of layer
        self.keyless = {t: keyless_kind(t, keyless) for t in kinds}

        def kept_of(t):
            n, slots = cfg.layer_types.count(t), self.pool[t] * TILE
            n4 = sum(cfg.layer_types[i] == t for i in self.kv4_set)
            vectors = lambda n: self._own(torch.zeros(n, slots, cfg.kv_heads[t], cfg.head_dim[t], dtype=torch.int8, device=device))  # noqa: E731
            scales = lambda n: self._own(torch.zeros(n, slots, cfg.kv_heads[t], device=device))  # noqa: E731
            if self.keyless[t]:
                return vectors(n), scales(n)
            if not n4:
                return vectors(n), vectors(n), scales(n), scales(n)
            # kv4: the kind's other layers as above, then the 4-bit ones (two codes a byte; kv4_heads_k's order in a tile)
            codes = lambda: self._own(torch.zeros(n4, slots, cfg.kv_heads[t], cfg.head_dim[t] // 2, dtype=torch.uint8, device=device))  # noqa: E731
            return vectors(n - n4), vectors(n - n4), scales(n - n4), scales(n - n4), codes(), codes(), scales(n4), scales(n4)

        self.kv_all = {t: kept_of(t) for t in kinds}
        if self.kv4_set:
            from . import kvsim as KV

            fmt = KV.Kept(cfg, "k4v4t0")  # the simulated format: which layers, and each layer's signs of the turn
            assert self.kv4_set == {i for i in range(cfg.n_layers) if fmt.bits[i]["k"] == 4 and fmt.bits[i]["v"] == 4}
            hd = cfg.head_dim[M.SLIDING]
            assert hd in (32, 64, 128, 256), "kv4: a sliding head of 32, 64, 128 or 256"
            self.kv4_signs = {i: self._own(fmt.signs[i].float().contiguous().to(device)) for i in sorted(self.kv4_set)}
            self.kv4_h = KV.hadamard(hd).to(device)
            lv, top = KV.grid_levels(4)
            self.kv4_levels, self.kv4_unit = lv.to(device), float(torch.tensor(top / 127, dtype=torch.float32))
        # what the scores make a keyless kind's keys with: the rotated pairs in whole 16s, and cos and sin (fp16) of
        # every position's angle for those pairs [capacity, pairs, 2] (the angle as heads_k and model.rope_tables
        # compute it: position * inv_freq in float32)
        self.turn, self.trig, self.k_norm = {}, {}, []
        for t in kinds:
            if self.keyless[t]:
                hd, n = cfg.head_dim[t], turned(cfg, t)
                assert hd % 32 == 0 and n <= min(hd // 2, 128), "a keyless layer's head is whole 32s, with at most 128 turned pairs"
                theta, rotated = cfg.rope[t]
                inv = torch.zeros(n)
                inv[:rotated] = 1.0 / (theta ** (torch.arange(0, 2 * rotated, 2).float() / hd))
                ang = torch.arange(capacity).float()[:, None] * inv[None, :]
                self.turn[t], self.trig[t] = n, self._own(torch.stack((ang.cos(), ang.sin()), -1).half().contiguous().to(device))
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
            for name in ("in_ln", "post_attn_ln", "pre_ff_ln", "post_ff_ln", "q_norm"):
                setattr(cl, name, f32(lay[name]).data_ptr())
            self.k_norm.append(f32(lay["k_norm"]))
            cl.k_norm = self.k_norm[-1].data_ptr()
            theta, rotated = cfg.rope[t]
            inv = torch.zeros(hd // 2)
            inv[:rotated] = 1.0 / (theta ** (torch.arange(0, 2 * rotated, 2).float() / hd))
            cl.inv_freq = f32(inv).data_ptr()
            nth = cfg.layer_types[:i].count(t)
            if self.keyless[t]:
                assert "v" not in mats, "a keyless layer's value is its key's projection"
                (values, vscale), keys, kscale = (x[nth] for x in self.kv_all[t]), None, None
                cl.keys, cl.kscale, cl.trig, cl.turned = None, None, self.trig[t].data_ptr(), self.turn[t]
            elif i in self.kv4_set:  # 4-bit codes: keys and values as bytes, their scales as ever
                keys, values, kscale, vscale = (x[sum(j in self.kv4_set for j in range(i))] for x in self.kv_all[t][4:])
                cl.keys, cl.kscale, cl.trig, cl.turned = keys.data_ptr(), kscale.data_ptr(), None, 0
                cl.kv4, cl.kv4_signs, cl.first = 1, self.kv4_signs[i].data_ptr(), self.firsts.data_ptr()
            else:
                nth -= sum(j in self.kv4_set for j in range(i) if cfg.layer_types[j] == t)  # (kv4: the kind's int8 layers by themselves)
                keys, values, kscale, vscale = (x[nth] for x in self.kv_all[t][:4])
                cl.keys, cl.kscale, cl.trig, cl.turned = keys.data_ptr(), kscale.data_ptr(), None, 0
            self.kv.append((keys, values, kscale, vscale))  # keys and their scales: None of a keyless layer
            cl.values, cl.vscale, cl.tiles = values.data_ptr(), vscale.data_ptr(), self.tables[t].data_ptr()
            cl.scalar, cl.full, cl.head_dim, cl.kv_heads, cl.ring = float(lay["scalar"]), int(t == M.FULL), hd, kvh, self.ring[t]
            if device != "cpu":
                torch.cuda.empty_cache()  # packing's temporaries, or the allocator keeps them beside the codes
        widest_head, y_size, x_size = work_sizes(cfg, bool(self.kv4_set))  # (row_bytes: what a row of the work memory below takes)
        # attention takes the positions a pass reads in whole tiles of 16: up to 15 more at either end
        # (sized for ATT_ROWS rows at most: a wider pass's attention takes as many rows at a time as this holds, at least
        # ATT_ROWS: Model.scores_room)
        narrow = min(rows, ATT_ROWS)
        scores = cfg.heads * narrow * (max(capacity, cfg.window + rows) + 32)
        z = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
        # the kernels' work memory, by the names of the DLL's Model (what a pass leaves in them is the last layer's)
        self.work = dict(h=z(rows, cfg.hidden), hn=z(rows, cfg.hidden), logits=z(narrow, cfg.vocab),
                         h2=z(rows, cfg.hidden), fps=z(3 * MAX_TOKENS), xf=z(rows // 8, x_size // 16, 32, dtype=torch.int64),  # fps: scales, then prep_k's sums (3 * MAX_TOKENS)
                         y=z(rows, y_size), y2=z(rows, cfg.mlp), qhat=z(rows, widest_head, dtype=torch.float16),
                         attn=z(rows, widest_head), scores=z(scores), probs=z(scores, dtype=torch.float16))
        self.h, self.hn, self.logits = self.work["h"], self.work["hn"], self.work["logits"]
        m = self.model = Model()
        m.layers = C.cast(self.c_layers, C.c_void_p)
        m.n_layers, m.hidden, m.mlp, m.heads, m.window, m.vocab = cfg.n_layers, cfg.hidden, cfg.mlp, cfg.heads, cfg.window, cfg.vocab
        m.eps, m.softcap = cfg.eps, cfg.softcap or 0.0
        m.int8 = self.int8
        m.scores_room = scores  # a wide pass's attention takes as many rows at a time as fit (engine/sclmd.cu)
        m.head, m.rot_head, m.norm = self._mat(*head), self._rot(cfg.hidden), f32(norm).data_ptr()
        for name, t in self.work.items():
            setattr(m, name, t.data_ptr())
        self.lib = lib()
        if self.kv4_set:  # the kernels' levels, midpoints and unit are the simulation's
            levels, cuts, unit = (C.c_int * 16)(), (C.c_float * 15)(), C.c_float()
            assert self.lib.sclmd_kv4_format(levels, cuts, C.byref(unit)) == TILE
            lv = self.kv4_levels.cpu()
            assert list(levels) == lv.int().tolist() and list(cuts) == ((lv[1:] + lv[:-1]) / 2).tolist() and unit.value == self.kv4_unit

    @classmethod
    def load(cls, name: str, plan: dict, head: int, capacity: int = 8192, device: str = "cuda", sequences: int = 1,
             pool: dict | None = None, rows: int = 8, keyless: bool = False, int8: float | None = None, kv4: bool | None = None,
             image: str | None = None) -> "Engine":
        """The original model with the stored codes `name`: plan is layer -> {matrix: bits a pair}, head the output
        head's bits. The layers come off the disk one at a time; the embedding table stays in RAM.
        image: "off" (default; also SCLMD_IMAGE), "auto" or "read": the packed codes from an engine image (sclmd/image.py)."""
        image = image or os.environ.get("SCLMD_IMAGE") or "off"
        if image != "off":
            from . import image as IM

            return IM.load(cls, name, plan, head, image, capacity=capacity, device=device, sequences=sequences, pool=pool, rows=rows,
                           keyless=keyless, int8=int8, kv4=kv4)
        from . import ckpt
        from .checkpoint import Checkpoint

        ck = Checkpoint(ckpt("orig"))
        cfg = ck.cfg

        def layers():
            for i in range(cfg.n_layers):
                has = [m for m in M.LINEARS if not (m == "v" and cfg.layer_types[i] == M.FULL)]
                yield ck.layer(i, "cpu", skip=M.LINEARS) | {"mats": CD.stored(name, i, {m: plan[i][m] for m in has})}

        eng = cls(cfg, layers(), CD.stored(name, "head", {"head": head})["head"], ck.norm("cpu"), ck.embed("cpu"), capacity, device,
                  sequences, pool, rows, keyless, int8, kv4=kv4)
        torch.cuda.empty_cache()  # packing left about 1 GiB reserved and unused: whatever loads next needs it
        return eng

    def bytes(self) -> int:
        """VRAM held: the codes, the keys and values, the work memory."""
        return sum(t.numel() * t.element_size() for t in self.keep + list(self.work.values()) if t.is_cuda)

    def _own(self, t: torch.Tensor) -> torch.Tensor:
        self.keep.append(t)
        return t

    _image = None  # while an image fills the engine: where _mat takes its tensors from (sclmd/image.py)

    def _mat(self, codes: torch.Tensor, scales: torch.Tensor, book: torch.Tensor, bits: int, out: torch.Tensor | None = None) -> Mat:
        if self._image is not None:  # the same tensors, filled from an image afterwards (sclmd/image.py)
            return self._image.mat(self, codes, bits, out)
        dev = self.device
        packed = pack(codes, bits, out=out) if out is not None else self._own(pack(codes, bits, device=dev))
        table = self._own(lut(book.to(dev)))
        scale = self._own(scales.to(device=dev, dtype=torch.float32).contiguous())
        mat = Mat(packed.data_ptr(), table.data_ptr(), scale.data_ptr(), codes.shape[0], codes.shape[1] * 2, bits, 0)
        if self.int8:
            t8, s8 = lut8(book.to(dev))
            mat.lut8, mat.book_scale = self._own(t8).data_ptr(), s8
        return mat

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

    def _entries(self, kind: str, pos: int, n: int) -> list:
        """The entries of a sequence's tile list of a kind that positions pos .. pos + n - 1 are written to: its ring
        tiles, and with kv4 the extra one (the int8 of positions 8..15) when the first tile is among them."""
        ring = self.ring[kind]
        return list(dict.fromkeys((q >> 4) % ring for q in range(pos, pos + n))) + ([ring] if self.extra[kind] and pos < TILE else [])

    def _put(self, kind: str, seq: int, js: list, tiles: list):
        """The kernels' tables: a sequence's entries js of a kind are the pool tiles `tiles` (the ring's in its row of
        self.tables, the extra one in self.firsts)."""
        ring = self.ring[kind]
        at = [(j, x) for j, x in zip(js, tiles) if j < ring]
        if at:
            self.tables[kind][seq, torch.tensor([j for j, _ in at], device=self.device)] = torch.tensor([x for _, x in at], dtype=torch.int32, device=self.device)
        for j, x in zip(js, tiles):
            if j >= ring:
                self.firsts[seq] = x

    def lacks(self, seq: int, pos: int, n: int) -> dict:
        """Tiles by kind the sequence would have to be given to write positions pos .. pos + n - 1."""
        return {t: sum(self.tiles[t][seq][j] < 0 for j in self._entries(t, pos, n)) for t in self.ring}

    def _provide(self, need: list):
        """Tiles for the positions a pass is about to write: need is [(sequence, first position, count)]. All or none."""
        want = {t: [] for t in self.ring}
        for t in self.ring:
            for seq, pos, n in need:
                for j in self._entries(t, pos, n):
                    if self.tiles[t][seq][j] < 0:
                        want[t].append((seq, j))
        for t in want:
            if len(want[t]) > len(self.free[t]):
                raise KVFull(f"no tile of {t} keys and values is free ({self.pool[t]} in all)")
        for t in want:
            for seq, j in want[t]:
                tile = self.free[t].pop()
                self.tiles[t][seq][j] = tile
                if j < self.ring[t]:
                    self.tables[t][seq, j] = tile
                else:
                    self.firsts[seq] = tile

    def advance(self, parts: list, logits_from: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """One pass over the rows that are in self.h: parts is [(sequence, rows)] in row order, each sequence once,
        up to self.rows rows in all; a sequence's rows are its next positions. -> (logits [rows, vocab], outputs
        after the final norm): views of the engine's memory, which the next pass writes over.
        logits_from: the first row whose logits are wanted (the rows before it are a prompt's: their logits are not
        computed and hold what an earlier pass left; the number of rows: none are).
        A pass of more than ATT_ROWS rows returns the logits of its rows from logits_first(rows, logits_from) on
        only (the last ATT_ROWS at most)."""
        total = sum(n for _, n in parts)
        assert 0 < total <= self.rows and len({seq for seq, _ in parts}) == len(parts) and all(n > 0 for _, n in parts)
        assert len(parts) <= MAX_SEGS, f"a pass has {MAX_SEGS} sequences at most"
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
        if total > ATT_ROWS:  # a wide pass: the logits of the rows from logits_first() on only
            return self.logits[: total - logits_first(total, logits_from)], self.hn[:total]
        return self.logits[:total], self.hn[:total]

    def run(self, parts: list, logits_from: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """One pass over tokens of several sequences: parts is [(sequence, token ids)] (see advance)."""
        flat = [t for _, ids in parts for t in ids]
        self.h[: len(flat)] = self.embed(flat)
        return self.advance([(seq, len(ids)) for seq, ids in parts], logits_from)

    def feed(self, ids: list, keep: str = "all", seq: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Run tokens at a sequence's next positions, a pass's rows at a time -> (logits [n, vocab], last outputs after
        the final norm). keep="last": only the last pass's rows (a long prompt's logits would not fit: 1 MB a token);
        of a pass of more than ATT_ROWS rows the logits of its last ATT_ROWS (logits_first) and the outputs of all.
        keep="all" on an engine of more than ATT_ROWS rows reads ATT_ROWS rows a pass."""
        assert self.lengths[seq] + len(ids) <= self.capacity, "past the engine's capacity"
        logits, hidden = [], []
        width = self.rows if keep == "last" else min(self.rows, ATT_ROWS)
        for a in range(0, len(ids), width):
            last = a + width >= len(ids)
            # with keep="last" only the last pass's logits are looked at: the passes before compute none
            lg, hn = self.run([(seq, ids[a : a + width])], 0 if keep == "all" or last else width)
            if keep == "all" or last:
                logits.append(lg.clone())
                hidden.append(hn.clone())
        return torch.cat(logits), torch.cat(hidden)

    def _tiles_view(self, t: torch.Tensor, kind: str) -> torch.Tensor:
        """[layers, pool slots, heads (, head_dim)] as [layers, pool tiles, 16, heads (, head_dim)]."""
        return t.view(t.shape[0], self.pool[kind], TILE, *t.shape[2:])

    def kv_float(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A layer's keys and values as numbers: [pool slots, heads, head_dim] (the int8 values times their scales).
        A keyless layer's keys are made as the scores make them, taking a slot's number for its position (which it is
        while the engine holds one sequence); so are a 4-bit layer's positions (kv4_float of sequence 0)."""
        k, v, ks, vs = self.kv[layer]
        if layer in self.kv4_set:
            return self.kv4_float(layer, torch.arange(v.shape[0], device=v.device))
        if k is None:
            k, ks = self._made(layer, v, torch.arange(v.shape[0], device=v.device) % self.capacity), vs
        return k.float() * ks[..., None], v.float() * vs[..., None]

    def _made(self, layer: int, v: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The keys a keyless layer's scores make of its kept values v (int8 [m, heads, head_dim], at `positions`), in
        the values' units (times the values' scales: the keys). As engine/sclmd.cu: where a value is not turned the
        key is the value times k_norm (the kernel has k_norm on the query there: folded); the turned ones are
        value * k_norm * cos -+ partner * k_norm * sin, rounded to fp16."""
        t = self.cfg.layer_types[layer]
        half, n, kn = self.cfg.head_dim[t] // 2, self.turn[t], self.k_norm[layer]
        x = v.float()
        out = x * kn
        trig = self.trig[t][positions].float()  # [m, pairs, 2]
        cos, sin = trig[:, None, :, 0], trig[:, None, :, 1]
        a, b = x[..., :n] * kn[:n], x[..., half : half + n] * kn[half : half + n]
        out[..., :n] = (a * cos - b * sin).half().float()
        out[..., half : half + n] = (b * cos + a * sin).half().float()
        return out

    def kv4_codes(self, layer: int, positions: torch.Tensor, seq: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """A 4-bit layer's codes (0..15, the levels' order) of the keys and values at positions of a sequence past its
        first tile: [m, heads, head_dim] each, read where engine/sclmd.cu's kv4_heads_k writes them."""
        hd, kvh = self.cfg.head_dim[M.SLIDING], self.cfg.kv_heads[M.SLIDING]
        k4, v4, _, _ = self.kv[layer]
        slot = self.slots(M.SLIDING, positions, seq).long()
        d = torch.arange(hd, device=k4.device)
        c = d & 15
        kbyte = 2 * (((c & 7) >> 1) * (hd // 16) + (d >> 4)) + (c >> 3)  # a key's value d: that byte of its head
        keys = (k4[slot][..., kbyte].long() >> (4 * (d & 1))) & 15
        values = (v4[slot][..., d >> 1].long() >> (4 * (d & 1))) & 15  # a value's byte: values 2b and 2b + 1 of the head
        return keys, values

    def kv4_float(self, layer: int, positions: torch.Tensor, seq: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """A 4-bit layer's keys and values at positions of a sequence as numbers [m, heads, head_dim]: the codes' levels
        times their scales, turned back (signs * H (level * scale)); the first tile's int8 times its scales."""
        hd, kvh = self.cfg.head_dim[M.SLIDING], self.cfg.kv_heads[M.SLIDING]
        k4, v4, ks, vs = self.kv[layer]
        positions = positions.to(k4.device)
        slot = self.slots(M.SLIDING, positions, seq).long()
        out = []
        signs = self.kv4_signs[layer][None]
        first = positions < TILE
        own = self.tiles[M.SLIDING][seq]
        for codes, data, scale in zip(self.kv4_codes(layer, positions, seq), (k4, v4), (ks, vs)):
            x = ((self.kv4_levels[codes] * scale[slot][..., None]) @ self.kv4_h) * signs
            if first.any():
                p = positions[first].long()
                tile = torch.where(p < 8, own[0], own[self.ring[M.SLIDING]])
                raw = data.view(-1, 8, kvh, hd).view(torch.int8)[tile, p & 7]
                x[first] = raw.float() * scale[tile * TILE + (p & 7)][..., None]
            out.append(x)
        return out[0], out[1]

    def folded(self, layer: int) -> torch.Tensor | None:
        """What a keyless layer's queries are multiplied by before they are kept [head_dim]: k_norm where the scores
        read the value itself as the key, 1 at the turned values. None of a layer with keys."""
        t = self.cfg.layer_types[layer]
        if not self.keyless[t]:
            return None
        half, n = self.cfg.head_dim[t] // 2, self.turn[t]
        f = self.k_norm[layer].clone()
        f[:n] = f[half : half + n] = 1.0
        return f

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
            shape = lambda t: (t.shape[0], self.width[M.SLIDING], TILE, *t.shape[2:])  # noqa: E731
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
        length: 0.42 MB a token for the last 1,040 tokens and 41 KB a token, 21 KB when keyless), and its tiles go back to the pool.
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

    def save(self, seq: int, pin: bool = True) -> dict:
        """A copy in RAM of what a sequence holds, in park()'s form, the sequence left as it is (a prefix many texts
        begin with: unpark() puts it on any free sequence, as often as wanted). Nothing may have been written past its
        length (save right when the prefix is in; a sliding layer's ring then holds what the next positions read). pin: page-locked RAM (copies to the GPU at full speed)."""
        assert self.highs[seq] == self.lengths[seq], "positions past its length were written: crop and trim first, or save earlier"
        state = {"length": self.lengths[seq], "high": self.highs[seq], "kinds": {}, "bytes": 0}
        for t in self.ring:
            js = [j for j, tile in enumerate(self.tiles[t][seq]) if tile >= 0]
            data = [torch.empty(k.shape[0], len(js), TILE, *k.shape[2:], dtype=k.dtype, pin_memory=pin) for k in self.kv_all[t]]
            self._copy_tiles(t, [self.tiles[t][seq][j] for j in js], data, out=True)
            state["kinds"][t] = {"js": js, "data": data}
            state["bytes"] += sum(d.numel() * d.element_size() for d in data)
        return state

    def fork(self, src: int, dst: int):
        """The free sequence `dst` gets a copy of what sequence `src` holds, GPU to GPU (src: a prefix kept on its own
        sequence; nothing written past its length). KVFull, and nothing changed, when the pool is short."""
        assert self.lengths[dst] == 0 and all(x < 0 for t in self.ring for x in self.tiles[t][dst]), "the sequence is in use"
        assert self.highs[src] == self.lengths[src], "positions past its length were written"
        js = {t: [j for j, tile in enumerate(self.tiles[t][src]) if tile >= 0] for t in self.ring}
        for t in self.ring:
            if len(js[t]) > len(self.free[t]):
                raise KVFull(f"no room for a copy of {len(js[t])} tiles of {t} keys and values")
        for t in self.ring:
            if not js[t]:
                continue
            own = [self.free[t].pop() for _ in js[t]]
            frm = [self.tiles[t][src][j] for j in js[t]]
            for j, tile in zip(js[t], own):
                self.tiles[t][dst][j] = tile
            self._put(t, dst, js[t], own)
            a, b = torch.tensor(frm, device=self.device), torch.tensor(own, device=self.device)
            for x in self.kv_all[t]:
                v = self._tiles_view(x, t)
                for lay in range(v.shape[0]):  # a layer at a time: the temporary is one layer's tiles
                    v[lay].index_copy_(0, b, v[lay].index_select(0, a))
        self.lengths[dst], self.highs[dst] = self.lengths[src], self.highs[src]

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
            self._put(t, seq, part["js"], own)
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
            self.tiles[t][seq] = [-1] * self.width[t]
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
            # (kv4: the extra tile is the first tile's, in use while a position is)
            for j in list(range(last + 1, self.ring[t])) + ([self.ring[t]] if self.extra[t] and last < 0 else []):
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
            key = self._made(self.cfg.last_of(t), v[slot], pos) * vs[slot][..., None] if k is None else k[slot].float() * ks[slot][..., None]
            out[t] = (key.transpose(0, 1)[None], (v[slot].float() * vs[slot][..., None]).transpose(0, 1)[None], pos)
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
        # a row a sequence, ATT_ROWS at most (the main model's scores' memory, which the drafter uses, is that wide)
        self.rows = min(eng.rows, ATT_ROWS)
        z = lambda *shape: self._own(torch.zeros(self.rows, *shape, dtype=torch.float32, device=dev))  # noqa: E731
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
        assert rows <= self.rows, f"the drafter guesses for {self.rows} sequences at a time at most"
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
