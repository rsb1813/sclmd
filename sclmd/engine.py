# The engine from Python: the model's pair codes laid out for the GPU kernels (engine/sclmd.cu), the memory they work
# in, and a pass as one call. All memory is PyTorch tensors; the DLL gets their addresses.
import ctypes as C
import os

import torch

from . import codes as CD
from . import model as M

DLL = os.path.join(os.path.dirname(__file__), "..", "engine", "build", "sclmd.dll")
MAX_TOKENS = 8
SPARE = 16  # positions a sliding layer keeps past its window: a pass writes up to 8, the drafter reads window + 1
DRAFTER_BYTES = 240 << 20  # the drafter at 4 bits a weight (0.47B weights) and its work memory


def fixed_bytes(cfg: M.Config, head_bits: int, capacity: int) -> int:
    """VRAM an Engine takes besides the layers' codes: the head's codes, the keys and values, the work memory."""
    head = cfg.vocab * cfg.hidden * head_bits // 16
    kv = sum((capacity if t == M.FULL else cfg.window + SPARE) * cfg.kv_heads[t] * cfg.head_dim[t] * 2 * 2 for t in cfg.layer_types)
    work = MAX_TOKENS * cfg.vocab * 4 + cfg.heads * MAX_TOKENS * (max(capacity, cfg.window + MAX_TOKENS) + 32) * 6 + (32 << 20)
    return head + kv + work


class Mat(C.Structure):
    _fields_ = [("codes", C.c_void_p), ("lut", C.c_void_p), ("scale", C.c_void_p), ("rows", C.c_int), ("cols", C.c_int),
                ("bits", C.c_int), ("pad", C.c_int)]


class Rot(C.Structure):
    _fields_ = [("signs", C.c_void_p), ("block", C.c_int), ("pad", C.c_int)]


class Layer(C.Structure):
    _fields_ = ([(m, Mat) for m in ("q", "k", "v", "o", "gate", "up", "down")] +
                [(r, Rot) for r in ("rot_in", "rot_o", "rot_mlp", "rot_down")] +
                [(p, C.c_void_p) for p in ("in_ln", "post_attn_ln", "pre_ff_ln", "post_ff_ln", "q_norm", "k_norm", "inv_freq", "keys", "values")] +
                [("scalar", C.c_float), ("full", C.c_int), ("head_dim", C.c_int), ("kv_heads", C.c_int), ("slots", C.c_int), ("pad", C.c_int)])


class Model(C.Structure):
    _fields_ = ([("layers", C.c_void_p)] + [(n, C.c_int) for n in ("n_layers", "hidden", "mlp", "heads", "window", "vocab")] +
                [("eps", C.c_float), ("softcap", C.c_float), ("head", Mat), ("rot_head", Rot), ("norm", C.c_void_p)] +
                [(p, C.c_void_p) for p in ("h", "h2", "fps", "xf", "y", "y2", "qhat", "scores", "probs", "attn", "hn", "logits")])


_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = C.CDLL(os.path.abspath(DLL))  # finds the CUDA runtime torch has already loaded
        sizes = [C.c_int() for _ in range(4)]
        assert _lib.sclmd_sizes(*[C.byref(s) for s in sizes]) == MAX_TOKENS
        assert [s.value for s in sizes] == [C.sizeof(Mat), C.sizeof(Rot), C.sizeof(Layer), C.sizeof(Model)], "struct layouts differ"
        _lib.sclmd_pass.argtypes = [C.c_void_p, C.c_int, C.c_int]
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

    layers: per layer a dict with
      "mats": matrix -> (codes uint8 [rows, cols / 2], scales [rows], book [2**bits, 2], bits)
      and the layer's small weights by the names of model.LAYER_KEYS (in_ln, q_norm, ..., scalar)
    head: the output matrix as (codes, scales, book, bits); norm: the final norm's weight
    embed: the input embedding table [vocab, hidden] (any device and type; rows are fetched as needed)"""

    def __init__(self, cfg: M.Config, layers: list, head: tuple, norm: torch.Tensor, embed: torch.Tensor, capacity: int = 8192,
                 device: str = "cuda"):
        self.cfg, self.device, self.capacity, self.length = cfg, device, capacity, 0
        self.high = 0  # one past the furthest position written since the start: what a sliding layer's ring has lost follows from it
        self._snap, self._snap_at = None, None
        self.embed_table, self.embed_scale = embed, float(M.embed_scale(cfg, embed.dtype))
        self.keep = []  # every tensor the DLL has the address of
        self.rots = {}
        f32 = lambda t: self._own(t.to(device=device, dtype=torch.float32).contiguous())  # noqa: E731
        self.c_layers = (Layer * cfg.n_layers)()
        # Few, large allocations: PyTorch rounds each up to 2 MiB, and a tensor a matrix and a layer's keys wasted 0.57 GiB.
        # The keys and values of all layers of a kind are one tensor each: [layers of the kind, slots, heads, head_dim]
        self.kv_all = {t: tuple(self._own(torch.zeros(cfg.layer_types.count(t), capacity if t == M.FULL else cfg.window + SPARE,
                                                      cfg.kv_heads[t], cfg.head_dim[t], dtype=torch.float16, device=device)) for _ in range(2))
                       for t in dict.fromkeys(cfg.layer_types)}
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
            slots = capacity if t == M.FULL else cfg.window + SPARE
            nth = cfg.layer_types[:i].count(t)
            keys, values = self.kv_all[t][0][nth], self.kv_all[t][1][nth]
            self.kv.append((keys, values))
            cl.keys, cl.values = keys.data_ptr(), values.data_ptr()
            cl.scalar, cl.full, cl.head_dim, cl.kv_heads, cl.slots = float(lay["scalar"]), int(t == M.FULL), hd, kvh, slots
            if device != "cpu":
                torch.cuda.empty_cache()  # packing's temporaries, or the allocator keeps them beside the codes
        widest_head = cfg.heads * max(cfg.head_dim.values())
        y_size = max(max(cfg.heads * cfg.head_dim[t] + 2 * cfg.kv_heads[t] * cfg.head_dim[t] for t in cfg.head_dim), cfg.mlp, cfg.hidden)
        x_size = max(cfg.hidden, widest_head, cfg.mlp)
        # attention takes the positions a pass reads in whole tiles of 16: up to 15 more at either end
        scores = cfg.heads * MAX_TOKENS * (max(capacity, cfg.window + MAX_TOKENS) + 32)
        z = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
        # the kernels' work memory, by the names of the DLL's Model (what a pass leaves in them is the last layer's)
        self.work = dict(h=z(MAX_TOKENS, cfg.hidden), hn=z(MAX_TOKENS, cfg.hidden), logits=z(MAX_TOKENS, cfg.vocab),
                         h2=z(MAX_TOKENS, cfg.hidden), fps=z(MAX_TOKENS), xf=z(x_size // 16, 32, dtype=torch.int64),
                         y=z(MAX_TOKENS, y_size), y2=z(MAX_TOKENS, cfg.mlp), qhat=z(MAX_TOKENS, widest_head, dtype=torch.float16),
                         attn=z(MAX_TOKENS, widest_head), scores=z(scores), probs=z(scores, dtype=torch.float16))
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
    def load(cls, name: str, plan: dict, head: int, capacity: int = 8192, device: str = "cuda") -> "Engine":
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

        eng = cls(cfg, layers(), CD.stored(name, "head", {"head": head})["head"], ck.norm("cpu"), ck.embed("cpu"), capacity, device)
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
        """One matrix times x [tokens <= 8, cols] (tests)."""
        x = x.to(device=self.device, dtype=torch.float32).contiguous()
        y = torch.zeros(x.shape[0], mat.rows, device=self.device)
        rot = self._rot(mat.cols)
        self._check(self.lib.sclmd_linear(C.byref(self.model), C.byref(mat), C.byref(rot), x.data_ptr(), y.data_ptr(), x.shape[0]))
        return y

    # what generation needs (sclmd/generate.py)

    def embed(self, ids: list) -> torch.Tensor:
        rows = self.embed_table[torch.as_tensor(ids, device=self.embed_table.device)]
        return rows.to(device=self.device, dtype=torch.float32) * self.embed_scale

    def feed(self, ids: list, keep: str = "all") -> tuple[torch.Tensor, torch.Tensor]:
        """Run tokens at the next positions, eight a pass -> (logits [n, vocab], last outputs after the final norm).
        keep="last": only the last pass's rows (a long prompt's logits would not fit: 1 MB a token)."""
        assert self.length + len(ids) <= self.capacity, "past the engine's capacity"
        assert self._kept(self.length - self.cfg.window + 1), "cropped further back than the sliding layers keep"
        logits, hidden = [], []
        for a in range(0, len(ids), MAX_TOKENS):
            part = ids[a : a + MAX_TOKENS]
            self.h[: len(part)] = self.embed(part)
            self._check(self.lib.sclmd_pass(C.byref(self.model), len(part), self.length))
            self.length += len(part)
            self.high = max(self.high, self.length)
            if keep == "all" or a + MAX_TOKENS >= len(ids):
                logits.append(self.logits[: len(part)].clone())
                hidden.append(self.hn[: len(part)].clone())
        return torch.cat(logits), torch.cat(hidden)

    def snapshot(self):
        """Remembers the present length so that restore() can come back to it however far the text goes on.
        (crop() cannot: a sliding layer's ring is written over as the text goes on. Its keys and values are copied
        to RAM here, 0.85 GB for the real model; one snapshot at a time.)"""
        sliding = self.kv_all.get(M.SLIDING, ())
        if self._snap is None:
            self._snap = [torch.empty(t.shape, dtype=t.dtype).pin_memory() for t in sliding]
        for t, to in zip(sliding, self._snap):
            to.copy_(t)
        self._snap_at = (self.length, self.high)

    def restore(self):
        """Back to the snapshot's length: every position from there on is forgotten."""
        for t, src in zip(self.kv_all.get(M.SLIDING, ()), self._snap):
            t.copy_(src)
        self.length, self.high = self._snap_at

    def _kept(self, first: int) -> bool:
        """Whether the sliding layers still hold every position from `first` on: their ring of window + SPARE slots has
        been written up to position high - 1, so it holds the positions from high - slots."""
        return max(first, 0) >= self.high - (self.cfg.window + SPARE)

    def crop(self, length: int):
        """Forget every position from `length` on (a later pass writes over them before anything reads them)."""
        self.length = length
        if length == 0:
            self.high = 0

    def keys(self, length: int) -> dict:
        """What the drafter attends to: per layer type the keys and values of the last layer of that type, for the
        positions the drafter may read, as (keys [1, heads, m, head_dim], values, positions), in fp16 as they are kept
        (the full-attention layer's are views of the engine's memory, not copies)."""
        assert length <= self.length and self._kept(length - (self.cfg.window + 1)), "further back than the sliding layers keep"
        out = {}
        for t in (M.SLIDING, M.FULL):
            k, v = self.kv[self.cfg.last_of(t)]
            first = 0 if t == M.FULL else max(0, length - (self.cfg.window + 1))
            pos = torch.arange(first, length, device=self.device)
            if t == M.FULL:
                out[t] = (k[:length].transpose(0, 1)[None], v[:length].transpose(0, 1)[None], pos)
            else:
                slot = pos % k.shape[0]
                out[t] = (k[slot].transpose(0, 1)[None], v[slot].transpose(0, 1)[None], pos)
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
        z = lambda *shape: self._own(torch.zeros(*shape, dtype=torch.float32, device=dev))  # noqa: E731
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
        assert work["xf"].shape[0] * 16 >= widest and work["y"].shape[1] >= widest and work["y2"].shape[1] >= cfg.mlp
        assert cfg.heads <= main.heads
        for name in ("fps", "xf", "y", "y2", "qhat", "scores", "probs", "attn"):
            setattr(c, name, work[name].data_ptr())
        self.sliding, self.full = eng.c_layers[main.last_of(M.SLIDING)], eng.c_layers[main.last_of(M.FULL)]
        self.lib = lib()
        sizes = [C.c_int() for _ in range(2)]
        self.lib.sclmd_draft_sizes(*[C.byref(s) for s in sizes])
        assert [s.value for s in sizes] == [C.sizeof(DLayer), C.sizeof(DrafterC)], "struct layouts differ"
        self.lib.sclmd_draft.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int]

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

    def step(self, token_embed: torch.Tensor, hidden: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One guess for position length + 1 from the token at `length` (its embedding in the main model) and a hidden
        state -> (logits [vocab], the hidden state for the guess after it). Both are the drafter's own memory:
        the next step writes over them."""
        eng = self.eng
        assert length <= eng.length and eng._kept(length - (eng.cfg.window + 1)), "further back than the sliding layers keep"
        n = eng.cfg.hidden
        self.inp[:n], self.inp[n:] = token_embed, hidden
        status = self.lib.sclmd_draft(C.byref(self.c), C.byref(self.sliding), C.byref(self.full), eng.cfg.window, length)
        assert status == 0, f"GPU kernel launch failed ({status})"
        return self.logits, self.next

    def guess(self, token: int, hidden: torch.Tensor, length: int) -> list:
        """Up to `depth` tokens after `token`, which sits at position `length` (generate.py's guess)."""
        out = []
        for _ in range(self.depth):
            logits, hidden = self.step(self.eng.embed([token])[0], hidden, length)
            token = int(logits.argmax())
            out.append(token)
        return out
