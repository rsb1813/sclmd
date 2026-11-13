# The quantizers on small matrices.
import torch

from sclmd import quant as Q


def test_q4_0_keeps_the_block_peak_and_uses_16_levels():
    torch.manual_seed(0)
    w = torch.randn(4, 64)
    q = Q.q4_0(w)
    for row in range(4):
        for b in range(2):
            blk, qb = w[row, 32 * b : 32 * b + 32], q[row, 32 * b : 32 * b + 32]
            peak = blk.abs().argmax()
            assert torch.isclose(qb[peak], blk[peak], rtol=2e-3)  # the largest value is a level (its step is fp16)
            assert len(torch.unique(qb)) <= 16
    assert (q - w).pow(2).mean() < 0.02 * w.pow(2).mean()


def test_error_feedback_beats_nearest_codeword_on_correlated_inputs():
    torch.manual_seed(0)
    k, n = 64, 48
    mix = torch.randn(k, k) * 0.3 + torch.eye(k)
    x = torch.randn(4000, k) @ mix  # correlated inputs: where feedback has something to use
    h = x.t() @ x
    w = torch.randn(n, k)
    cb = Q.kmeans(Q.row_normalize(w)[0].reshape(-1, 2), 64)
    wn, s = Q.row_normalize(w)
    near = Q.vq_dequant(wn, cb) * s
    codes, scales = Q.gptq_vq(w, h, cb, k, refit=True)
    fed = Q.vq_codes_dequant(codes, scales, cb)
    err = lambda q: (x @ (q - w).t()).pow(2).sum() / (x @ w.t()).pow(2).sum()  # noqa: E731
    assert err(fed) < 0.8 * err(near)


def test_rotation_round_trip_and_hessian():
    torch.manual_seed(0)
    k = 96  # blocks of 32
    rot = Q.BlockHadamard(k, seed=3, device="cpu")
    w, x = torch.randn(5, k), torch.randn(200, k)
    assert torch.allclose(rot.unrotate(rot.rotate(w)), w, atol=1e-5)
    xr = rot.rotate(x)  # inputs in the rotated basis: W x = (W R)(R^T x), and x R is R^T x written as rows
    assert torch.allclose(xr @ rot.rotate(w).t(), x @ w.t(), atol=1e-4)
    hr = rot.rotate(rot.rotate(x.t() @ x).t().contiguous())
    assert torch.allclose(hr, xr.t() @ xr, atol=1e-2, rtol=1e-4)
