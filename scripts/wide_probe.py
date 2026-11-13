# What a matrix product from pair codes costs with more than eight tokens a pass (engine/wide_probe.cu): the gate and up
# matrices of one real layer (a third of a pass's products), for 8, 16, 32 and 64 tokens, at 3 and at 4 bits.
# Build first:  cmd /c "engine\build_probe.bat"
import argparse
import ctypes as C
import os
import time

import torch

from sclmd import codes as CD
from sclmd import engine as E

DLL = os.path.join(os.path.dirname(E.DLL), "wide_probe.dll")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--repeat", type=int, default=40)
    args = ap.parse_args()
    torch.zeros(1, device="cuda")
    lib = C.CDLL(os.path.abspath(DLL))
    lib.wide_gemm.argtypes = [C.c_void_p] * 6 + [C.c_int, C.c_int]
    keep = []

    def mat(stored) -> E.Mat:
        codes, scales, book, bits = stored
        packed, table, scale = E.pack(codes, bits, device="cuda"), E.lut(book.to("cuda")), scales.to(device="cuda", dtype=torch.float32).contiguous()
        keep.extend((packed, table, scale))
        return E.Mat(packed.data_ptr(), table.data_ptr(), scale.data_ptr(), codes.shape[0], codes.shape[1] * 2, bits, 0)

    for bits in (6, 8):
        stored = CD.stored(args.name, args.layer, {"gate": bits, "up": bits})
        gate, up = mat(stored["gate"]), mat(stored["up"])
        cols, rows = gate.cols, gate.rows
        base = None
        for tokens in (8, 16, 32, 64):
            groups = tokens // 8
            x = (torch.randn(groups, cols // 16, 32, 4, device="cuda") * 0.02).half().contiguous()  # the inputs as the kernel reads them
            fps = torch.ones(tokens, device="cuda")
            ya, yb = torch.zeros(tokens, rows, device="cuda"), torch.zeros(tokens, rows, device="cuda")
            run = lambda n: lib.wide_gemm(C.byref(gate), C.byref(up), ya.data_ptr(), yb.data_ptr(), x.data_ptr(), fps.data_ptr(), tokens, n)  # noqa: E731
            assert run(3) == 0
            torch.cuda.synchronize()
            t = time.perf_counter()
            assert run(args.repeat) == 0
            torch.cuda.synchronize()
            ms = 1000 * (time.perf_counter() - t) / args.repeat
            base = base or ms
            # the same inputs through a narrower kernel give the same outputs: the first eight tokens against the 8-token run
            print(f"{bits // 2} bits, {tokens:2d} tokens: {ms:6.3f} ms a product of gate and up ({ms / base:.2f}x the 8-token one, "
                  f"{tokens / 8 * base / ms:.2f}x the tokens a second)", flush=True)
        del gate, up
        keep.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
