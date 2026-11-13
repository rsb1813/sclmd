// A probe: what a matrix product from pair codes costs with 16 or 32 tokens a pass instead of 8.
// The kernel is the engine's gemm_k with G groups of eight tokens: the codes of a tile are decoded once and multiplied
// with each group's inputs. xf: [group][tile][32 lanes]. (scripts/wide_probe.py times it on a real layer's matrices.)
#define NOMINMAX
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define API extern "C" __declspec(dllexport)

struct Mat {
    const uint32_t* codes;
    const uint32_t* lut;
    const float* scale;
    int rows, cols, bits, pad;
};

__device__ __forceinline__ void mma(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

template <int G>
__global__ void gemm_k(Mat m0, Mat m1, float* y0, float* y1, const uint64_t* xf, const float* fps, int segs, int tokens) {
    __shared__ uint32_t slut[256];
    __shared__ float part[8][32][4 * G];
    int strip = blockIdx.x;
    Mat m = m0;
    float* y = y0;
    if (strip >= m0.rows / 16) strip -= m0.rows / 16, m = m1, y = y1;
    for (int i = threadIdx.x; i < (1 << m.bits); i += blockDim.x) slut[i] = m.lut[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tiles = m.cols / 16, per = tiles / segs, first = warp * per;
    float c[G][4];
    for (int g = 0; g < G; ++g) c[g][0] = c[g][1] = c[g][2] = c[g][3] = 0;
    const uint64_t* xw = xf + size_t(first) * 32 + lane;
    const size_t group = size_t(tiles) * 32;  // a group's inputs follow the last one's
    if (m.bits == 8) {
        const uint32_t* cw = m.codes + (size_t(strip) * tiles + first) * 32 + lane;
        for (int j = 0; j < per; j += 4) {
            uint32_t w[4];
            for (int i = 0; i < 4; ++i) w[i] = cw[size_t(j + i) * 32];
            for (int i = 0; i < 4; ++i) {
                const uint32_t a0 = slut[w[i] & 255], a1 = slut[w[i] >> 8 & 255], a2 = slut[w[i] >> 16 & 255], a3 = slut[w[i] >> 24];
                for (int g = 0; g < G; ++g) {
                    const uint64_t b = xw[g * group + size_t(j + i) * 32];
                    mma(c[g], a0, a1, a2, a3, uint32_t(b), uint32_t(b >> 32));
                }
            }
        }
    } else {
        const uint32_t* cw = m.codes + (size_t(strip) * (tiles / 4) + first / 4) * 96 + lane;
        for (int j = 0; j < per; j += 4) {
            const uint32_t* q = cw + size_t(j / 4) * 96;
            const uint64_t lo = uint64_t(q[0]) | uint64_t(q[32]) << 32, hi = q[64];
            const uint32_t k10 = uint32_t(lo >> 60 | hi << 4) & 63;
            const uint32_t a[16] = {slut[lo & 63], slut[lo >> 6 & 63], slut[lo >> 12 & 63], slut[lo >> 18 & 63],
                                    slut[lo >> 24 & 63], slut[lo >> 30 & 63], slut[lo >> 36 & 63], slut[lo >> 42 & 63],
                                    slut[lo >> 48 & 63], slut[lo >> 54 & 63], slut[k10], slut[hi >> 2 & 63],
                                    slut[hi >> 8 & 63], slut[hi >> 14 & 63], slut[hi >> 20 & 63], slut[hi >> 26 & 63]};
            for (int g = 0; g < G; ++g) {
                const uint64_t* x = xw + g * group + size_t(j) * 32;
                const uint64_t b0 = x[0], b1 = x[32], b2 = x[64], b3 = x[96];
                mma(c[g], a[0], a[1], a[2], a[3], uint32_t(b0), uint32_t(b0 >> 32));
                mma(c[g], a[4], a[5], a[6], a[7], uint32_t(b1), uint32_t(b1 >> 32));
                mma(c[g], a[8], a[9], a[10], a[11], uint32_t(b2), uint32_t(b2 >> 32));
                mma(c[g], a[12], a[13], a[14], a[15], uint32_t(b3), uint32_t(b3 >> 32));
            }
        }
    }
    for (int g = 0; g < G; ++g)
        for (int i = 0; i < 4; ++i) part[warp][lane][4 * g + i] = c[g][i];
    __syncthreads();
    if (warp) return;
    for (int g = 0; g < G; ++g)
        for (int i = 0; i < 4; ++i) {
            float sum = 0;
            for (int w = 0; w < segs; ++w) sum += part[w][lane][4 * g + i];
            const int row = strip * 16 + (lane >> 2) + (i >= 2 ? 8 : 0), token = 8 * g + 2 * (lane & 3) + (i & 1);
            if (token < tokens) y[size_t(token) * m.rows + row] = m.scale[row] * fps[token] * sum;
        }
}

static int segs_for(int cols) {
    const int tiles = cols / 16;
    for (int s = 8; s > 1; s >>= 1)
        if (tiles % (4 * s) == 0 && tiles / s >= 32) return s;
    return 1;
}

// `repeat` launches of one product of two matrices that share the input, for `tokens` tokens (8, 16 or 32).
API int wide_gemm(const Mat* a, const Mat* b, float* ya, float* yb, const uint64_t* xf, const float* fps, int tokens, int repeat) {
    const int segs = segs_for(a->cols), blocks = (a->rows + b->rows) / 16, groups = (tokens + 7) / 8;
    for (int r = 0; r < repeat; ++r) {
        if (groups == 1) gemm_k<1><<<blocks, 32 * segs>>>(*a, *b, ya, yb, xf, fps, segs, tokens);
        else if (groups == 2) gemm_k<2><<<blocks, 32 * segs>>>(*a, *b, ya, yb, xf, fps, segs, tokens);
        else if (groups <= 4) gemm_k<4><<<blocks, 32 * segs>>>(*a, *b, ya, yb, xf, fps, segs, tokens);
        else gemm_k<8><<<blocks, 32 * segs>>>(*a, *b, ya, yb, xf, fps, segs, tokens);
    }
    return int(cudaGetLastError());
}
