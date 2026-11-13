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
                [(p, C.c_void_p) for p in ("h", "x", "fps", "xf", "y", "y2", "qhat", "scores", "attn", "hn", "logits")])


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


def pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """codes (uint8 [rows, cols / 2], a code per pair of columns) in the order the kernel reads them (int32)."""
    rows, pairs = codes.shape
    assert rows % 16 == 0 and pairs % 32 == 0 and bits in (6, 8)
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
        self.embed_table, self.embed_scale = embed, float(M.embed_scale(cfg, embed.dtype))
        self.keep = []  # every tensor the DLL has the address of
        self.rots = {}
        f32 = lambda t: self._own(t.to(device=device, dtype=torch.float32).contiguous())  # noqa: E731
        self.c_layers = (Layer * cfg.n_layers)()
        self.kv = []
        for i, (lay, cl) in enumerate(zip(layers, self.c_layers)):
            t = cfg.layer_types[i]
            hd, kvh = cfg.head_dim[t], cfg.kv_heads[t]
            for name in M.LINEARS:
                if name in lay["mats"]:
                    setattr(cl, name, self._mat(*lay["mats"][name]))
            cl.rot_in, cl.rot_o = self._rot(cfg.hidden), self._rot(cfg.heads * hd)
            cl.rot_mlp, cl.rot_down = self._rot(cfg.hidden), self._rot(cfg.mlp)
            for name in ("in_ln", "post_attn_ln", "pre_ff_ln", "post_ff_ln", "q_norm", "k_norm"):
                setattr(cl, name, f32(lay[name]).data_ptr())
            theta, rotated = cfg.rope[t]
            inv = torch.zeros(hd // 2)
            inv[:rotated] = 1.0 / (theta ** (torch.arange(0, 2 * rotated, 2).float() / hd))
            cl.inv_freq = f32(inv).data_ptr()
            slots = capacity if t == M.FULL else cfg.window + SPARE
            keys = self._own(torch.zeros(slots, kvh, hd, dtype=torch.float16, device=device))
            values = self._own(torch.zeros(slots, kvh, hd, dtype=torch.float16, device=device))
            self.kv.append((keys, values))
            cl.keys, cl.values = keys.data_ptr(), values.data_ptr()
            cl.scalar, cl.full, cl.head_dim, cl.kv_heads, cl.slots = float(lay["scalar"]), int(t == M.FULL), hd, kvh, slots
        widest_head = cfg.heads * max(cfg.head_dim.values())
        y_size = max(max(cfg.heads * cfg.head_dim[t] + 2 * cfg.kv_heads[t] * cfg.head_dim[t] for t in cfg.head_dim), cfg.mlp, cfg.hidden)
        x_size = max(cfg.hidden, widest_head, cfg.mlp)
        scores = max(cfg.kv_heads[t] * (cfg.heads // cfg.kv_heads[t]) * MAX_TOKENS * (capacity if t == M.FULL else cfg.window + MAX_TOKENS)
                     for t in cfg.head_dim)
        z = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
        # the kernels' work memory, by the names of the DLL's Model (what a pass leaves in them is the last layer's)
        self.work = dict(h=z(MAX_TOKENS, cfg.hidden), hn=z(MAX_TOKENS, cfg.hidden), logits=z(MAX_TOKENS, cfg.vocab),
                         x=z(MAX_TOKENS, x_size), fps=z(MAX_TOKENS), xf=z(x_size // 16, 32, dtype=torch.int64),
                         y=z(MAX_TOKENS, y_size), y2=z(MAX_TOKENS, cfg.mlp), qhat=z(MAX_TOKENS, widest_head),
                         attn=z(MAX_TOKENS, widest_head), scores=z(scores))
        self.h, self.hn, self.logits = self.work["h"], self.work["hn"], self.work["logits"]
        m = self.model = Model()
        m.layers = C.cast(self.c_layers, C.c_void_p)
        m.n_layers, m.hidden, m.mlp, m.heads, m.window, m.vocab = cfg.n_layers, cfg.hidden, cfg.mlp, cfg.heads, cfg.window, cfg.vocab
        m.eps, m.softcap = cfg.eps, cfg.softcap or 0.0
        m.head, m.rot_head, m.norm = self._mat(*head), self._rot(cfg.hidden), f32(norm).data_ptr()
        for name, t in self.work.items():
            setattr(m, name, t.data_ptr())
        self.lib = lib()

    def _own(self, t: torch.Tensor) -> torch.Tensor:
        self.keep.append(t)
        return t

    def _mat(self, codes: torch.Tensor, scales: torch.Tensor, book: torch.Tensor, bits: int) -> Mat:
        dev = self.device
        packed = self._own(pack(codes.to(dev), bits))
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

    def feed(self, ids: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Run tokens at the next positions, eight a pass -> (logits [n, vocab], last outputs after the final norm)."""
        assert self.length + len(ids) <= self.capacity, "past the engine's capacity"
        assert self._kept(self.length - self.cfg.window + 1), "cropped further back than the sliding layers keep"
        logits, hidden = [], []
        for a in range(0, len(ids), MAX_TOKENS):
            part = ids[a : a + MAX_TOKENS]
            self.h[: len(part)] = self.embed(part)
            self._check(self.lib.sclmd_pass(C.byref(self.model), len(part), self.length))
            self.length += len(part)
            self.high = max(self.high, self.length)
            logits.append(self.logits[: len(part)].clone())
            hidden.append(self.hn[: len(part)].clone())
        return torch.cat(logits), torch.cat(hidden)

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
        positions the drafter may read, as (keys [1, heads, m, head_dim], values, positions)."""
        assert length <= self.length and self._kept(length - (self.cfg.window + 1)), "further back than the sliding layers keep"
        out = {}
        for t in (M.SLIDING, M.FULL):
            k, v = self.kv[self.cfg.last_of(t)]
            first = 0 if t == M.FULL else max(0, length - (self.cfg.window + 1))
            pos = torch.arange(first, length, device=self.device)
            slot = pos % k.shape[0]
            out[t] = (k[slot].transpose(0, 1)[None].float(), v[slot].transpose(0, 1)[None].float(), pos)
        return out
