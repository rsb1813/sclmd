# Weight quantizers: the pair-codebook codes the GPU kernel multiplies (bench/mma_gemm.cu), fitted with GPTQ, and
# llama.cpp's Q4_0 grid for comparison. The GPTQ and k-means code is the parent project's (sclm/quant.py).
import torch


def largest_pow2_divisor(n: int, cap: int = 1024) -> int:
    b = 1
    while n % (b * 2) == 0 and b * 2 <= cap:
        b *= 2
    return b


def fwht_(x: torch.Tensor) -> torch.Tensor:
    """In-place unnormalized fast Walsh-Hadamard transform over the last dim (a power of 2)."""
    n = x.size(-1)
    h = 1
    while h < n:
        y = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :].clone(), y[..., 1, :]
        y[..., 0, :] += b
        y[..., 1, :] = a - b
        h *= 2
    return x


class BlockHadamard:
    """Orthogonal transform R = diag(signs) * blockdiag(H_b / sqrt(b)) along the input dim of a [N, K] weight.
    W @ x == (W R) @ (R^T x), so quantizing W R and rotating activations by R^T is lossless before quantization."""

    def __init__(self, k: int, seed: int, device="cuda"):
        self.b = largest_pow2_divisor(k)
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.signs = (torch.randint(0, 2, (k,), generator=g, device="cpu") * 2 - 1).float().to(device)

    def _h(self, w: torch.Tensor) -> torch.Tensor:
        n, k = w.shape
        return fwht_(w.reshape(n, k // self.b, self.b).clone()).reshape(n, k) / self.b**0.5

    def rotate(self, w: torch.Tensor) -> torch.Tensor:
        return self._h(w.float() * self.signs)

    def unrotate(self, w: torch.Tensor) -> torch.Tensor:
        return self._h(w.float()) * self.signs


def gptq_vq(
    w: torch.Tensor, h: torch.Tensor, codebook: torch.Tensor, group: int, damp: float = 0.01, block: int = 128,
    refit: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPTQ with vector rounding: d columns at a time go to the row of `codebook` [C, d] (times the group's scale)
    that is nearest in the metric the error feedback leaves on them, |delta U_BB^-1|^2 with U the Cholesky factor
    gptq_codes uses, i.e. the cost that remains after the later columns have compensated. A group's scale is the rms
    of its columns once earlier feedback has landed (fp16; group >= K: one per row, the rms of the row as given, or
    with `refit` the least-squares scale of the chosen codes under h). Batch dims as in gptq_codes.
    Returns codes (long [..., N, K/d]) and scales (fp32 [..., N, K/group or 1])."""
    d = codebook.size(-1)
    *batch, n, k = w.shape
    per_row = group >= k
    assert block % d == 0 and k % d == 0 and (per_row or (block % group == 0 and group % d == 0))
    assert per_row or not refit
    W, H = w.float().clone(), h.float().clone()
    diag = torch.diagonal(H, dim1=-2, dim2=-1)  # a view: edits below land in H
    dead = diag == 0
    diag.masked_fill_(dead, 1)
    W.masked_fill_(dead.unsqueeze(-2), 0)
    diag += damp * diag.mean(-1, keepdim=True)
    u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    w0 = W.clone() if refit else None
    cb = codebook.float()
    # every code's d x d diagonal block of U inverted at once, and the codewords in each block's metric: one
    # linalg.inv per code was a fifth of the run time
    blocks = u.unflatten(-2, (k // d, d)).unflatten(-1, (k // d, d)).diagonal(dim1=-4, dim2=-2).movedim(-1, -3)
    inv_all = torch.linalg.inv(blocks)  # [..., K/d, d, d]
    ct_all = cb @ inv_all  # [..., K/d, C, d]
    ct2_all = ct_all.pow(2).sum(-1)
    del blocks
    rms = lambda g: g.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12).half().float()  # noqa: E731
    codes = torch.empty(*batch, n, k // d, dtype=torch.long, device=w.device)
    scales = torch.empty(*batch, n, 1 if per_row else k // group, device=w.device, dtype=torch.float32)
    if per_row:
        scale = rms(W)
        scales[..., 0] = scale.squeeze(-1)
    for b0 in range(0, k, block):
        b1 = min(b0 + block, k)
        w1, err = W[..., b0:b1].clone(), torch.zeros(*batch, n, b1 - b0, device=w.device, dtype=torch.float32)
        ub = u[..., b0:b1, b0:b1]
        for i in range(0, b1 - b0, d):
            col = b0 + i
            if not per_row and col % group == 0:
                scale = rms(w1[..., i : i + group])
                scales[..., col // group] = scale.squeeze(-1)
            # targets [..., N, d] and codewords [..., C, d] in the metric
            t, ct = w1[..., i : i + d] @ inv_all[..., col // d, :, :], ct_all[..., col // d, :, :]
            # |t - scale * c|^2 up to terms and a factor that do not depend on the codeword
            j = (scale / 2 * ct2_all[..., col // d, :].unsqueeze(-2) - t @ ct.transpose(-1, -2)).argmin(-1)
            codes[..., col // d] = j
            e = t - scale * ct.gather(-2, j.unsqueeze(-1).expand(*j.shape, d))
            w1[..., i + d :] -= e @ ub[..., i : i + d, i + d :]
            err[..., i : i + d] = e
        W[..., b1:] -= err @ u[..., b0:b1, b1:]
    if refit:
        del W, u, inv_all, ct_all  # the refit's [..., N, K] temporaries were the VRAM peak
        q = cb[codes].flatten(-2)
        qh = q @ H
        num = w0.mul_(qh).sum(-1, keepdim=True)
        scales = (num / qh.mul_(q).sum(-1, keepdim=True).clamp_min(1e-12)).half().float()
    return codes, scales


def vq_codes_dequant(codes: torch.Tensor, scales: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Weights [..., N, K] of gptq_vq's codes and scales."""
    q = codebook.float()[codes]  # [..., N, K/d, d]
    return (q.unflatten(-2, (scales.size(-1), -1)) * scales[..., None, None]).flatten(-3)


def kmeanspp_init(x: torch.Tensor, n_codes: int, seed: int = 0, max_points: int = 64) -> torch.Tensor:
    """k-means++ seeding on a random subsample of at most max_points * n_codes rows."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    sub = x[torch.randperm(x.size(0), generator=g, device="cpu")[: max_points * n_codes].to(x.device)]
    codes = [sub[torch.randint(0, sub.size(0), (1,), generator=g, device="cpu").item()]]
    d2 = (sub - codes[0]).pow(2).sum(-1)
    for _ in range(n_codes - 1):
        nxt = sub[torch.multinomial(d2 / d2.sum(), 1).item()]
        codes.append(nxt)
        d2 = torch.minimum(d2, (sub - nxt).pow(2).sum(-1))
    return torch.stack(codes)


def kmeans(x: torch.Tensor, n_codes: int, iters: int = 15, chunk: int = 1 << 16, seed: int = 0) -> torch.Tensor:
    """k-means (k-means++ seeded) on rows of x [m, d] (fp32, GPU). Returns the codebook [n_codes, d]."""
    codebook = kmeanspp_init(x, n_codes, seed)
    for _ in range(iters):
        assign = vq_assign(x, codebook, chunk)
        sums = torch.zeros_like(codebook).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=n_codes).float().unsqueeze(-1)
        empty = counts.squeeze(-1) == 0
        codebook = torch.where(empty.unsqueeze(-1), codebook, sums / counts.clamp_min(1))
        if empty.any():  # re-seed dead codes from random points
            codebook[empty] = x[torch.randint(0, x.size(0), (int(empty.sum()),), device=x.device)]
    return codebook


def vq_assign(x: torch.Tensor, codebook: torch.Tensor, chunk: int = 1 << 16) -> torch.Tensor:
    """Index of the nearest code for each row of x."""
    c2 = codebook.pow(2).sum(-1)
    out = torch.empty(x.size(0), dtype=torch.long, device=x.device)
    for s in range(0, x.size(0), chunk):
        xs = x[s : s + chunk]
        out[s : s + chunk] = (c2 - 2 * xs @ codebook.t()).argmin(-1)
    return out


def row_normalize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w / rms(row) and the per-row rms (stored as fp16: 16/K bpw, negligible)."""
    s = w.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    return w.float() / s, s


def q4_0(w: torch.Tensor) -> torch.Tensor:
    """The weight as llama.cpp's Q4_0 stores it (4.5 bits a weight): per 32 weights along a row, the value of largest
    magnitude maps to code 0 (-8 steps), the step is kept as fp16, the others round to the nearest of 16 codes."""
    n, k = w.shape
    g = w.float().reshape(n, k // 32, 32)
    peak = g.gather(-1, g.abs().argmax(-1, keepdim=True))
    d = peak / -8
    inv = torch.where(d != 0, 1 / d, torch.zeros_like(d))
    q = torch.clamp(torch.round(g * inv) + 8, 0, 15) - 8
    return (q * d.half().float()).reshape(n, k)


def vq_dequant(w_norm: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """A row-normalized [N, K] weight with each group of d columns moved to its nearest codeword."""
    d = codebook.size(1)
    return codebook[vq_assign(w_norm.reshape(-1, d), codebook)].reshape(w_norm.shape)
