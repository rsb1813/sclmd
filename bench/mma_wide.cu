// Kill test for wide passes (brief B2-wide-passes): the matrix kernel past four groups of eight tokens.
//
// One kernel, wide_k<S8, G, SETS, SHARE>, on the engine's codes and input layouts (engine/sclmd.cu gemm_k / gemm8_k):
//   S8:    false = today's fp16 instruction (m16n8k16, f32 sums), true = B1's s8 instruction (m16n8k32, s32 sums,
//          the input as prep_k writes it with sigma: a lane's words of four tiles one uint4).
//   G:     the groups a warp holds in its registers (c[G][4] + b[G][4]; the engine's cap is 4).
//   SETS:  warps of a block that take different group sets of the same strip and tiles (groups = G * SETS).
//   SHARE: false = form (A): every set's warp decodes the codes itself (the codes come from L1 after the first);
//          true  = form (B): the set warps of a (strip, seg) decode one quad of tiles each into shared memory, and
//          every set multiplies all of them (one decoded run shared by the group-set warps).
// SETS = 1, G = 1 / 2 / 4 is exactly today's gemm_k / gemm8_k.
// The time per group at 8 / 16 / 32 groups is set against the time per group at 4 (the stop bar: > 1.1x), and the
// cost a group adds against the instruction's floor (bench/mma_rate.cu: f32-sum f16 118.9, s8 474.8 T ops/s).
#define NOMINMAX
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#define CHECK(call)                                                                   \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(err_), __LINE__); \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

__device__ __forceinline__ void mma(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void mma8(int c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t join8(uint32_t p, uint32_t q) { return __byte_perm(p, q, 0x5410); }

struct Mat {
    const uint32_t* codes;
    const uint32_t* lut;  // fp16 form: two fp16; s8 form: the two int8 twice (b0 b1 b0 b1)
    const float* scale;
    int rows, cols, bits, pad;
};

#define GEMM_STRIPS 4
#define MAX_WARPS 32

// a quad of tiles' A registers: fp16 16 (a tile's four), s8 8 (two tiles an instruction)
template <bool S8>
struct Frag {
    static constexpr int R = S8 ? 8 : 16;
};

// The quad of tiles at tile j of strip `strip`, decoded (the engine's loops, unchanged)
template <bool S8>
__device__ __forceinline__ void decode(const Mat& m, const uint32_t* slut, int strip, int tiles, int j, int lane, uint32_t* a) {
    if (m.bits == 8) {
        const uint32_t* cw = m.codes + (size_t(strip) * tiles + j) * 32 + lane;
        uint32_t w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = cw[size_t(i) * 32];
        if constexpr (S8) {
#pragma unroll
            for (int i = 0; i < 4; i += 2) {
                const uint32_t u = w[i], v = w[i + 1];
                a[2 * i + 0] = join8(slut[u & 255], slut[u >> 16 & 255]), a[2 * i + 1] = join8(slut[u >> 8 & 255], slut[u >> 24]);
                a[2 * i + 2] = join8(slut[v & 255], slut[v >> 16 & 255]), a[2 * i + 3] = join8(slut[v >> 8 & 255], slut[v >> 24]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                a[4 * i] = slut[w[i] & 255], a[4 * i + 1] = slut[w[i] >> 8 & 255], a[4 * i + 2] = slut[w[i] >> 16 & 255],
                          a[4 * i + 3] = slut[w[i] >> 24];
        }
    } else {
        const uint32_t* q = m.codes + (size_t(strip) * (tiles / 4) + j / 4) * 96 + lane;
        const uint32_t q0 = q[0], q1 = q[32], q2 = q[64];
        const uint64_t lo = uint64_t(q0) | uint64_t(q1) << 32, hi = q2;
        const uint32_t k10 = uint32_t(lo >> 60 | hi << 4) & 63;
        const uint32_t t[16] = {slut[lo & 63],       slut[lo >> 6 & 63],  slut[lo >> 12 & 63], slut[lo >> 18 & 63],
                                slut[lo >> 24 & 63], slut[lo >> 30 & 63], slut[lo >> 36 & 63], slut[lo >> 42 & 63],
                                slut[lo >> 48 & 63], slut[lo >> 54 & 63], slut[k10],           slut[hi >> 2 & 63],
                                slut[hi >> 8 & 63],  slut[hi >> 14 & 63], slut[hi >> 20 & 63], slut[hi >> 26 & 63]};
        if constexpr (S8) {
            a[0] = join8(t[0], t[2]), a[1] = join8(t[1], t[3]), a[2] = join8(t[4], t[6]), a[3] = join8(t[5], t[7]);
            a[4] = join8(t[8], t[10]), a[5] = join8(t[9], t[11]), a[6] = join8(t[12], t[14]), a[7] = join8(t[13], t[15]);
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = t[i];
        }
    }
}

// a group's inputs for a quad: fp16 four 64-bit words (xf[group][tile][lane]); s8 one uint4 (prep_k's int8 layout)
template <bool S8>
struct In4 {
    uint64_t w[4];
};
template <>
struct In4<true> {
    uint4 w;
};

template <bool S8>
__device__ __forceinline__ In4<S8> load_in(const void* xf, int tiles, int g, int j, int lane) {
    In4<S8> r;
    if constexpr (S8) {
        r.w = reinterpret_cast<const uint4*>(xf)[size_t(g) * tiles * 8 + size_t(j / 4) * 32 + lane];
    } else {
        const uint64_t* x = reinterpret_cast<const uint64_t*>(xf) + (size_t(g) * tiles + j) * 32 + lane;
        r.w[0] = x[0], r.w[1] = x[32], r.w[2] = x[64], r.w[3] = x[96];
    }
    return r;
}

template <bool S8, class C>
__device__ __forceinline__ void mult(C c[4], const uint32_t* a, const In4<S8>& b) {
    if constexpr (S8) {
        mma8(c, a[0], a[1], a[2], a[3], b.w.x, b.w.y);
        mma8(c, a[4], a[5], a[6], a[7], b.w.z, b.w.w);
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) mma(c, a[4 * i], a[4 * i + 1], a[4 * i + 2], a[4 * i + 3], uint32_t(b.w[i]), uint32_t(b.w[i] >> 32));
    }
}

// block: SETS x strips x segs warps (warp id = (set * strips + local) * segs + seg)
template <bool S8, int G, int SETS, bool SHARE>
__global__ void wide_k(Mat m, float book_scale, float* y, const void* xf, const float* fps, int segs, int tokens, int strips) {
    using C = std::conditional_t<S8, int, float>;
    constexpr int R = Frag<S8>::R;
    extern __shared__ uint4 shared_frags[];  // SHARE: [warp][R / 4][32]
    __shared__ uint32_t luts[GEMM_STRIPS][256];
    __shared__ C part[MAX_WARPS][32][4];
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, team = segs * strips;
    const int set = wid / team, rest = wid % team, local = rest / segs, warp = rest % segs;
    const int strip = blockIdx.x * strips + local;
    const bool idle = strip >= m.rows / 16;
    for (int i = threadIdx.x; i < strips * 256; i += blockDim.x)
        if ((i & 255) < (1 << m.bits)) luts[i >> 8][i & 255] = m.lut[i & 255];
    __syncthreads();
    const uint32_t* slut = luts[local];
    const int tiles = m.cols / 16, per = tiles / segs, first = warp * per, g0 = set * G;
    C c[G][4];
#pragma unroll
    for (int g = 0; g < G; ++g) c[g][0] = c[g][1] = c[g][2] = c[g][3] = 0;
    if constexpr (!SHARE) {
        if (!idle)
            for (int j = first; j < first + per; j += 4) {
                In4<S8> b[G];
#pragma unroll
                for (int g = 0; g < G; ++g) b[g] = load_in<S8>(xf, tiles, g0 + g, j, lane);
                uint32_t a[R];
                decode<S8>(m, slut, strip, tiles, j, lane, a);
#pragma unroll
                for (int g = 0; g < G; ++g) mult<S8>(c[g], a, b[g]);
            }
    } else {
        const int nq = per / 4;
        for (int r = 0; r < nq; r += SETS) {
            const int q = r + set;
            if (!idle && q < nq) {
                uint32_t a[R];
                decode<S8>(m, slut, strip, tiles, first + 4 * q, lane, a);
#pragma unroll
                for (int k = 0; k < R / 4; ++k) shared_frags[(wid * (R / 4) + k) * 32 + lane] = make_uint4(a[4 * k], a[4 * k + 1], a[4 * k + 2], a[4 * k + 3]);
            }
            __syncthreads();
            const int here = min(SETS, nq - r);
            if (!idle)
                for (int s = 0; s < here; ++s) {
                    const int j = first + 4 * (r + s), src = s * team + rest;
                    In4<S8> b[G];
#pragma unroll
                    for (int g = 0; g < G; ++g) b[g] = load_in<S8>(xf, tiles, g0 + g, j, lane);
                    uint32_t a[R];
#pragma unroll
                    for (int k = 0; k < R / 4; ++k) {
                        const uint4 v = shared_frags[(src * (R / 4) + k) * 32 + lane];
                        a[4 * k] = v.x, a[4 * k + 1] = v.y, a[4 * k + 2] = v.z, a[4 * k + 3] = v.w;
                    }
#pragma unroll
                    for (int g = 0; g < G; ++g) mult<S8>(c[g], a, b[g]);
                }
            __syncthreads();
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        for (int i = 0; i < 4; ++i) part[wid][lane][i] = c[g][i];
        __syncthreads();
        if (warp == 0 && !idle)
            for (int i = 0; i < 4; ++i) {
                C sum = 0;
                for (int w = 0; w < segs; ++w) sum += part[wid + w][lane][i];
                const int row = strip * 16 + (lane >> 2) + (i >= 2 ? 8 : 0), token = 8 * (g0 + g) + 2 * (lane & 3) + (i & 1);
                if (token < tokens) y[size_t(token) * m.rows + row] = m.scale[row] * (S8 ? book_scale : 1.0f) * fps[token] * float(sum);
            }
        __syncthreads();
    }
}

// ---- engine/sclmd.cu's gemm_k and gemm8_k, verbatim (the denominator: today's kernels at 1 / 2 / 4 groups)
struct EMat {
    const uint32_t* codes;
    const uint32_t* lut;
    const float* scale;
    int rows, cols, bits, pad;
    const uint32_t* lut8;
    float book_scale;
    int pad2;
};
template <int G>
__global__ void eng_gemm_k(EMat m0, EMat m1, EMat m2, float* y0, float* y1, float* y2, const uint64_t* xf, const float* fps, int segs, int tokens,
                       int strips) {
    __shared__ uint32_t luts[GEMM_STRIPS][256];
    __shared__ float part[GEMM_STRIPS][8][32][4];
    const int local = (threadIdx.x >> 5) / segs, warp = (threadIdx.x >> 5) % segs, lane = threadIdx.x & 31;  // the block's strip, and its warp
    int strip = blockIdx.x * strips + local;
    EMat m = m0;
    float* y = y0;
    if (strip >= m0.rows / 16) {
        strip -= m0.rows / 16, m = m1, y = y1;
        if (strip >= m1.rows / 16) strip -= m1.rows / 16, m = m2, y = y2;
    }
    const bool idle = strip >= m.rows / 16;  // past the last strip (the last block may not be full)
    uint32_t* slut = luts[local];
    if (!idle)
        for (int i = threadIdx.x - local * segs * 32; i < (1 << m.bits); i += segs * 32) slut[i] = m.lut[i];
    __syncthreads();
    const int tiles = m.cols / 16, per = idle ? 0 : tiles / segs, first = warp * per;
    float c[G][4];
    for (int g = 0; g < G; ++g) c[g][0] = c[g][1] = c[g][2] = c[g][3] = 0;
    const uint64_t* xw = xf + size_t(first) * 32 + lane;
    const size_t group = size_t(tiles) * 32;  // a group's inputs follow the group before
    if (m.bits == 8) {
        const uint32_t* cw = m.codes + (size_t(strip) * tiles + first) * 32 + lane;
        for (int j = 0; j < per; j += 4) {
            // the four tiles' codes and every group's inputs for them are asked for together, before the first is
            // used: a warp waits at the first value that has not come, and an input read when its product is due was
            // a wait a group and tile (what a wide pass paid for its rows)
            uint32_t w[4];
            uint64_t b[G][4];
            for (int i = 0; i < 4; ++i) w[i] = cw[size_t(j + i) * 32];
#pragma unroll
            for (int g = 0; g < G; ++g)
#pragma unroll
                for (int i = 0; i < 4; ++i) b[g][i] = xw[g * group + size_t(j + i) * 32];
            for (int i = 0; i < 4; ++i) {
                const uint32_t a0 = slut[w[i] & 255], a1 = slut[w[i] >> 8 & 255], a2 = slut[w[i] >> 16 & 255], a3 = slut[w[i] >> 24];
                for (int g = 0; g < G; ++g) mma(c[g], a0, a1, a2, a3, uint32_t(b[g][i]), uint32_t(b[g][i] >> 32));
            }
        }
    } else {  // 6 bits a pair: four tiles of a lane are 16 codes in 96 bits, stored as three words
        const uint32_t* cw = m.codes + (size_t(strip) * (tiles / 4) + first / 4) * 96 + lane;
        for (int j = 0; j < per; j += 4) {
            const uint32_t* q = cw + size_t(j / 4) * 96;
            const uint32_t q0 = q[0], q1 = q[32], q2 = q[64];
            uint64_t b[G][4];  // every group's inputs, asked for with the codes (see the 8-bit loop)
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const uint64_t* x = xw + g * group + size_t(j) * 32;
                b[g][0] = x[0], b[g][1] = x[32], b[g][2] = x[64], b[g][3] = x[96];
            }
            const uint64_t lo = uint64_t(q0) | uint64_t(q1) << 32, hi = q2;
            const uint32_t k10 = uint32_t(lo >> 60 | hi << 4) & 63;
            const uint32_t a[16] = {slut[lo & 63],       slut[lo >> 6 & 63],  slut[lo >> 12 & 63], slut[lo >> 18 & 63],
                                    slut[lo >> 24 & 63], slut[lo >> 30 & 63], slut[lo >> 36 & 63], slut[lo >> 42 & 63],
                                    slut[lo >> 48 & 63], slut[lo >> 54 & 63], slut[k10],           slut[hi >> 2 & 63],
                                    slut[hi >> 8 & 63],  slut[hi >> 14 & 63], slut[hi >> 20 & 63], slut[hi >> 26 & 63]};
            for (int g = 0; g < G; ++g) {
                mma(c[g], a[0], a[1], a[2], a[3], uint32_t(b[g][0]), uint32_t(b[g][0] >> 32));
                mma(c[g], a[4], a[5], a[6], a[7], uint32_t(b[g][1]), uint32_t(b[g][1] >> 32));
                mma(c[g], a[8], a[9], a[10], a[11], uint32_t(b[g][2]), uint32_t(b[g][2] >> 32));
                mma(c[g], a[12], a[13], a[14], a[15], uint32_t(b[g][3]), uint32_t(b[g][3] >> 32));
            }
        }
    }
    // a strip's warps hand their sums to its first warp, a group of tokens at a time (the shared memory a block needs
    // does not grow with the groups: it is taken from what the block's cache has)
    for (int g = 0; g < G; ++g) {
        for (int i = 0; i < 4; ++i) part[local][warp][lane][i] = c[g][i];
        __syncthreads();
        if (warp == 0 && !idle)
            for (int i = 0; i < 4; ++i) {
                float sum = 0;
                for (int w = 0; w < segs; ++w) sum += part[local][w][lane][i];
                const int row = strip * 16 + (lane >> 2) + (i >= 2 ? 8 : 0), token = 8 * g + 2 * (lane & 3) + (i & 1);
                if (token < tokens) y[size_t(token) * m.rows + row] = m.scale[row] * fps[token] * sum;
            }
        __syncthreads();
    }
}

// gemm_k on the integer instruction (Model.int8): the same codes, strips, tiles and launch, the input int8 (xf read as
// uint32, a word a lane and tile: prep_k with sigma), the codebook int8 (lut8), two tiles an instruction. Instruction
// columns 0..15 are tile j (k = 4t + i: column {2t, 2t+1, 2t+8, 2t+9}[i]), 16..31 tile j + 1; so, from the word w of
// tile j (pairs (g, 2t), (g+8, 2t), (g, 2t+8), (g+8, 2t+8) a byte each), A's registers are
// pair(w & 255) | pair(w >> 16 & 255) << 16 (row g) and pair(w >> 8 & 255) | pair(w >> 24) << 16 (row g + 8), and the
// same of tile j + 1's word. The int32 sums are exact: a row's result has the same bits at any width and order.
// y = scale[row] * book_scale * fps[token] * sum.
template <int G>
__global__ void eng_gemm8_k(EMat m0, EMat m1, EMat m2, float* y0, float* y1, float* y2, const uint32_t* xf, const float* fps, int segs, int tokens,
                        int strips) {
    __shared__ uint32_t luts[GEMM_STRIPS][256];
    __shared__ int part[GEMM_STRIPS][8][32][4];
    const int local = (threadIdx.x >> 5) / segs, warp = (threadIdx.x >> 5) % segs, lane = threadIdx.x & 31;
    int strip = blockIdx.x * strips + local;
    EMat m = m0;
    float* y = y0;
    if (strip >= m0.rows / 16) {
        strip -= m0.rows / 16, m = m1, y = y1;
        if (strip >= m1.rows / 16) strip -= m1.rows / 16, m = m2, y = y2;
    }
    const bool idle = strip >= m.rows / 16;
    uint32_t* slut = luts[local];
    if (!idle)
        for (int i = threadIdx.x - local * segs * 32; i < (1 << m.bits); i += segs * 32) slut[i] = m.lut8[i];
    __syncthreads();
    const int tiles = m.cols / 16, per = idle ? 0 : tiles / segs, first = warp * per;
    int c[G][4];
    for (int g = 0; g < G; ++g) c[g][0] = c[g][1] = c[g][2] = c[g][3] = 0;
    // a lane's input words of four tiles are one 16-byte load (prep_k's int8 layout)
    const uint4* xw = reinterpret_cast<const uint4*>(xf) + size_t(first / 4) * 32 + lane;
    const size_t group = size_t(tiles) * 8;
    if (m.bits == 8) {
        const uint32_t* cw = m.codes + (size_t(strip) * tiles + first) * 32 + lane;
        for (int j = 0; j < per; j += 4) {
            uint32_t w[4], b[G][4];
            for (int i = 0; i < 4; ++i) w[i] = cw[size_t(j + i) * 32];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const uint4 x = xw[g * group + size_t(j / 4) * 32];
                b[g][0] = x.x, b[g][1] = x.y, b[g][2] = x.z, b[g][3] = x.w;
            }
#pragma unroll
            for (int i = 0; i < 4; i += 2) {
                const uint32_t u = w[i], v = w[i + 1];
                const uint32_t a0 = join8(slut[u & 255], slut[u >> 16 & 255]), a1 = join8(slut[u >> 8 & 255], slut[u >> 24]);
                const uint32_t a2 = join8(slut[v & 255], slut[v >> 16 & 255]), a3 = join8(slut[v >> 8 & 255], slut[v >> 24]);
                for (int g = 0; g < G; ++g) mma8(c[g], a0, a1, a2, a3, b[g][i], b[g][i + 1]);
            }
        }
    } else {  // 6 bits a pair (see gemm_k)
        const uint32_t* cw = m.codes + (size_t(strip) * (tiles / 4) + first / 4) * 96 + lane;
        for (int j = 0; j < per; j += 4) {
            const uint32_t* q = cw + size_t(j / 4) * 96;
            const uint32_t q0 = q[0], q1 = q[32], q2 = q[64];
            uint32_t b[G][4];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const uint4 x = xw[g * group + size_t(j / 4) * 32];
                b[g][0] = x.x, b[g][1] = x.y, b[g][2] = x.z, b[g][3] = x.w;
            }
            const uint64_t lo = uint64_t(q0) | uint64_t(q1) << 32, hi = q2;
            const uint32_t k10 = uint32_t(lo >> 60 | hi << 4) & 63;
            const uint32_t a[16] = {slut[lo & 63],       slut[lo >> 6 & 63],  slut[lo >> 12 & 63], slut[lo >> 18 & 63],
                                    slut[lo >> 24 & 63], slut[lo >> 30 & 63], slut[lo >> 36 & 63], slut[lo >> 42 & 63],
                                    slut[lo >> 48 & 63], slut[lo >> 54 & 63], slut[k10],           slut[hi >> 2 & 63],
                                    slut[hi >> 8 & 63],  slut[hi >> 14 & 63], slut[hi >> 20 & 63], slut[hi >> 26 & 63]};
            const uint32_t r0 = join8(a[0], a[2]), r1 = join8(a[1], a[3]), r2 = join8(a[4], a[6]), r3 = join8(a[5], a[7]);
            const uint32_t s0 = join8(a[8], a[10]), s1 = join8(a[9], a[11]), s2 = join8(a[12], a[14]), s3 = join8(a[13], a[15]);
            for (int g = 0; g < G; ++g) {
                mma8(c[g], r0, r1, r2, r3, b[g][0], b[g][1]);
                mma8(c[g], s0, s1, s2, s3, b[g][2], b[g][3]);
            }
        }
    }
    for (int g = 0; g < G; ++g) {
        for (int i = 0; i < 4; ++i) part[local][warp][lane][i] = c[g][i];
        __syncthreads();
        if (warp == 0 && !idle)
            for (int i = 0; i < 4; ++i) {
                int sum = 0;
                for (int w = 0; w < segs; ++w) sum += part[local][w][lane][i];
                const int row = strip * 16 + (lane >> 2) + (i >= 2 ? 8 : 0), token = 8 * g + 2 * (lane & 3) + (i & 1);
                if (token < tokens) y[size_t(token) * m.rows + row] = m.scale[row] * m.book_scale * fps[token] * float(sum);
            }
        __syncthreads();
    }
}


static int segs_for(int cols, int most) {
    const int tiles = cols / 16;
    for (int s = 8; s > 1; s >>= 1)
        if (s <= most && tiles % (4 * s) == 0 && tiles / s >= 32) return s;
    return 1;
}

static uint16_t float_to_half(float f) {
    if (f == 0) return 0;
    int e;
    const float mm = std::frexp(std::fabs(f), &e);
    int mant = int(std::lround(std::ldexp(mm, 11))) - 1024, ex = e + 14;
    if (mant == 1024) mant = 0, ++ex;
    return uint16_t((f < 0 ? 0x8000 : 0) | ex << 10 | mant);
}

static std::vector<uint32_t> pack(const std::vector<uint8_t>& pairs, int rows, int cols, int bits) {
    const int tiles = cols / 16, half = cols / 2;
    std::vector<uint32_t> out(bits == 8 ? size_t(rows / 16) * tiles * 32 : size_t(rows / 16) * (tiles / 4) * 96, 0);
    for (int s = 0; s < rows / 16; ++s)
        for (int j = 0; j < tiles; ++j)
            for (int lane = 0; lane < 32; ++lane) {
                const int g = lane >> 2, t = lane & 3;
                const int r0 = 16 * s + g, r1 = r0 + 8, p0 = 8 * j + t, p1 = p0 + 4;
                const uint32_t k[4] = {pairs[size_t(r0) * half + p0], pairs[size_t(r1) * half + p0], pairs[size_t(r0) * half + p1],
                                       pairs[size_t(r1) * half + p1]};
                if (bits == 8) {
                    out[(size_t(s) * tiles + j) * 32 + lane] = k[0] | k[1] << 8 | k[2] << 16 | k[3] << 24;
                } else {
                    uint32_t* q = out.data() + (size_t(s) * (tiles / 4) + j / 4) * 96 + lane;
                    for (int i = 0; i < 4; ++i) {
                        const int bit = 6 * (4 * (j % 4) + i);
                        const uint64_t v = uint64_t(k[i]) << (bit % 32);
                        q[size_t(bit / 32) * 32] |= uint32_t(v);
                        if (bit % 32 > 26) q[size_t(bit / 32 + 1) * 32] |= uint32_t(v >> 32);
                    }
                }
            }
    return out;
}

// One matrix's data on the device, for up to MAXG groups of tokens
struct Problem {
    int rows, cols, bits, half, n_codes, tiles, ntok;
    std::vector<uint8_t> pairs;
    std::vector<int8_t> book, x;
    std::vector<float> scale;
    size_t mat_bytes, copies;
    uint32_t *d_codes, *d_l16, *d_l8, *d_xf8;
    uint64_t* d_xf16;
    float *d_scale, *d_fps, *d_y;
};

static void make(Problem& p, int rows, int cols, int bits, int maxg, size_t budget) {
    std::mt19937 rng(7);
    p.rows = rows, p.cols = cols, p.bits = bits, p.half = cols / 2, p.n_codes = 1 << bits, p.tiles = cols / 16, p.ntok = 8 * maxg;
    p.pairs.resize(size_t(rows) * p.half);
    for (auto& v : p.pairs) v = uint8_t(rng() % p.n_codes);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    p.book.resize(2 * p.n_codes);
    for (auto& v : p.book) v = int8_t(std::max(-127.0f, std::min(127.0f, std::round(normal(rng) * 40.0f))));
    std::vector<uint32_t> lut16(p.n_codes), lut8(p.n_codes);
    for (int i = 0; i < p.n_codes; ++i) {
        lut16[i] = uint32_t(float_to_half(p.book[2 * i])) | uint32_t(float_to_half(p.book[2 * i + 1])) << 16;
        const uint32_t two = uint32_t(uint8_t(p.book[2 * i])) | uint32_t(uint8_t(p.book[2 * i + 1])) << 8;
        lut8[i] = two | two << 16;
    }
    p.scale.resize(rows);
    for (auto& s : p.scale) s = 0.5f + 0.001f * float(rng() % 1000);
    p.x.resize(size_t(p.ntok) * cols);
    for (auto& v : p.x) v = int8_t(std::max(-127.0f, std::min(127.0f, std::round(normal(rng) * 30.0f))));
    std::vector<float> fps(p.ntok, 1.0f);
    const int tiles = p.tiles;
    std::vector<uint64_t> xf16(size_t(maxg) * tiles * 32, 0);
    std::vector<uint32_t> xf8(size_t(maxg) * tiles * 32, 0);
    for (int gr = 0; gr < maxg; ++gr)
        for (int j = 0; j < tiles; ++j)
            for (int lane = 0; lane < 32; ++lane) {
                const int g = lane >> 2, t = lane & 3, token = 8 * gr + g;
                const int8_t* q = p.x.data() + size_t(token) * cols + 16 * j + 2 * t;
                xf16[(size_t(gr) * tiles + j) * 32 + lane] = uint64_t(float_to_half(q[0])) | uint64_t(float_to_half(q[1])) << 16 |
                                                             uint64_t(float_to_half(q[8])) << 32 | uint64_t(float_to_half(q[9])) << 48;
                // prep_k's int8 layout: tile j's word of lane L at (group * tiles + j - j % 4) * 32 + 4 L + j % 4
                xf8[(size_t(gr) * tiles + (j & ~3)) * 32 + 4 * lane + (j & 3)] =
                    uint32_t(uint8_t(q[0])) | uint32_t(uint8_t(q[1])) << 8 | uint32_t(uint8_t(q[8])) << 16 | uint32_t(uint8_t(q[9])) << 24;
            }
    const std::vector<uint32_t> codes = pack(p.pairs, rows, cols, bits);
    p.mat_bytes = codes.size() * 4, p.copies = std::max<size_t>(1, budget / p.mat_bytes);
    CHECK(cudaMalloc(&p.d_codes, p.mat_bytes * p.copies));
    CHECK(cudaMalloc(&p.d_l16, p.n_codes * 4));
    CHECK(cudaMalloc(&p.d_l8, p.n_codes * 4));
    CHECK(cudaMalloc(&p.d_xf16, xf16.size() * 8));
    CHECK(cudaMalloc(&p.d_xf8, xf8.size() * 4));
    CHECK(cudaMalloc(&p.d_scale, rows * 4));
    CHECK(cudaMalloc(&p.d_fps, p.ntok * 4));
    CHECK(cudaMalloc(&p.d_y, size_t(p.ntok) * rows * 4));
    for (size_t i = 0; i < p.copies; ++i)
        CHECK(cudaMemcpy((char*)p.d_codes + i * p.mat_bytes, codes.data(), p.mat_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p.d_l16, lut16.data(), p.n_codes * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p.d_l8, lut8.data(), p.n_codes * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p.d_xf16, xf16.data(), xf16.size() * 8, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p.d_xf8, xf8.data(), xf8.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p.d_scale, p.scale.data(), rows * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p.d_fps, fps.data(), p.ntok * 4, cudaMemcpyHostToDevice));
}

static void release(Problem& p) {
    cudaFree(p.d_codes), cudaFree(p.d_l16), cudaFree(p.d_l8), cudaFree(p.d_xf16), cudaFree(p.d_xf8), cudaFree(p.d_scale), cudaFree(p.d_fps),
        cudaFree(p.d_y);
}

struct Result {
    double us, err;
    int regs, threads, segs, strips;
    bool ok;
};

template <bool S8, int G, int SETS, bool SHARE>
static Result run(Problem& p, std::mt19937& rng, int seg_cap = 8) {
    auto kernel = wide_k<S8, G, SETS, SHARE>;
    cudaFuncAttributes attr;
    CHECK(cudaFuncGetAttributes(&attr, kernel));
    constexpr int R = Frag<S8>::R;
    const int tokens = 8 * G * SETS;
    // warps a block: SETS x strips x segs <= what the kernel's registers allow (and 32)
    const int most_warps = std::min(MAX_WARPS, attr.maxThreadsPerBlock / 32);
    const int segs = segs_for(p.cols, std::min(seg_cap, std::max(1, most_warps / SETS)));
    const int strips = std::min(GEMM_STRIPS, most_warps / (segs * SETS));
    Result res{0, 0, attr.numRegs, 0, segs, strips, false};
    if (strips < 1) return res;
    const int threads = 32 * segs * strips * SETS, blocks = (p.rows / 16 + strips - 1) / strips;
    const size_t dyn = SHARE ? size_t(threads / 32) * (R / 4) * 32 * 16 : 0;
    if (dyn > 0) CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(dyn)));
    res.threads = threads;
    auto launch = [&](size_t i) {
        Mat m{(const uint32_t*)((const char*)p.d_codes + i * p.mat_bytes), S8 ? p.d_l8 : p.d_l16, p.d_scale, p.rows, p.cols, p.bits, 0};
        kernel<<<blocks, threads, dyn>>>(m, 1.0f, p.d_y, S8 ? (const void*)p.d_xf8 : (const void*)p.d_xf16, p.d_fps, segs, tokens, strips);
    };
    CHECK(cudaMemset(p.d_y, 0, size_t(tokens) * p.rows * 4));
    launch(0);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    std::vector<float> y(size_t(tokens) * p.rows);
    CHECK(cudaMemcpy(y.data(), p.d_y, y.size() * 4, cudaMemcpyDeviceToHost));
    double worst = 0;
    for (int trial = 0; trial < 300; ++trial) {
        const int row = rng() % p.rows, token = rng() % tokens;
        int64_t want = 0;
        for (int q = 0; q < p.half; ++q) {
            const int k = p.pairs[size_t(row) * p.half + q];
            want += int64_t(p.book[2 * k]) * p.x[size_t(token) * p.cols + 2 * q] + int64_t(p.book[2 * k + 1]) * p.x[size_t(token) * p.cols + 2 * q + 1];
        }
        const double w = double(want) * p.scale[row], got = y[size_t(token) * p.rows + row];
        worst = std::max(worst, std::fabs(w - got) / std::max(1.0, std::fabs(w)));
    }
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    float best = 1e30f;
    for (int r = 0; r < 4; ++r) {
        CHECK(cudaEventRecord(e0));
        for (size_t i = 0; i < p.copies; ++i) launch(i);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        best = std::min(best, ms);
    }
    cudaEventDestroy(e0), cudaEventDestroy(e1);
    res.us = best * 1e3 / p.copies, res.err = worst, res.ok = true;
    return res;
}


// today's engine kernel at G groups (one matrix: the second and third have no rows)
template <bool S8, int G>
static Result run_engine(Problem& p) {
    const int segs = segs_for(p.cols, 8), strips = GEMM_STRIPS * segs * 32 <= 1024 ? GEMM_STRIPS : 1024 / (segs * 32);
    const int blocks = (p.rows / 16 + strips - 1) / strips, tokens = 8 * G;
    cudaFuncAttributes attr;
    CHECK(cudaFuncGetAttributes(&attr, S8 ? (const void*)eng_gemm8_k<G> : (const void*)eng_gemm_k<G>));
    auto launch = [&](size_t i) {
        EMat m{(const uint32_t*)((const char*)p.d_codes + i * p.mat_bytes), p.d_l16, p.d_scale, p.rows, p.cols, p.bits, 0, p.d_l8, 1.0f, 0};
        if (S8)
            eng_gemm8_k<G><<<blocks, 32 * segs * strips>>>(m, EMat{}, EMat{}, p.d_y, nullptr, nullptr, p.d_xf8, p.d_fps, segs, tokens, strips);
        else
            eng_gemm_k<G><<<blocks, 32 * segs * strips>>>(m, EMat{}, EMat{}, p.d_y, nullptr, nullptr, p.d_xf16, p.d_fps, segs, tokens, strips);
    };
    launch(0);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    float best = 1e30f;
    for (int r = 0; r < 4; ++r) {
        CHECK(cudaEventRecord(e0));
        for (size_t i = 0; i < p.copies; ++i) launch(i);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        best = std::min(best, ms);
    }
    cudaEventDestroy(e0), cudaEventDestroy(e1);
    return Result{best * 1e3 / p.copies, 0.0, attr.numRegs, 32 * segs * strips, segs, strips, true};
}

static void show(const Problem& p, const char* form, int groups, const Result& r, double per4, double floor_us) {
    if (!r.ok) {
        printf("  %-5s %-10s groups %2d: does not fit (regs %d)\n", form, "", groups, r.regs);
        return;
    }
    const double per = r.us / groups;
    printf("  %-14s groups %2d: %8.1f us  %6.2f us/group  (x%.3f of G4's per group)  %5.0f GB/s  err %.1e  regs %3d  threads %4d segs %d strips %d  floor %.0f%%\n",
           form, groups, r.us, per, per4 > 0 ? per / per4 : 0.0, p.mat_bytes / (r.us * 1e-6) / 1e9, r.err, r.regs, r.threads, r.segs, r.strips,
           100.0 * floor_us * groups / r.us);
}

template <bool S8>
static void family(Problem& p, std::mt19937& rng) {
    const char* inst = S8 ? "s8" : "fp16";
    // the instruction's floor for one group: rows x cols x 8 tokens x 2 ops
    const double ops = double(p.rows) * p.cols * 8 * 2, floor_us = ops / (S8 ? 474.8e12 : 118.9e12) * 1e6;
    printf(" %s (instruction floor %.1f us a group)\n", inst, floor_us);
    char name[64];
    const Result e1 = run_engine<S8, 1>(p), e2 = run_engine<S8, 2>(p), e4 = run_engine<S8, 4>(p);
    const Result g1 = run<S8, 1, 1, false>(p, rng), g2 = run<S8, 2, 1, false>(p, rng), g4 = run<S8, 4, 1, false>(p, rng);
    const double per4 = e4.us / 4;  // the denominator: the engine's own kernel at four groups
    snprintf(name, sizeof name, "%s engine", inst);
    show(p, name, 1, e1, per4, floor_us);
    show(p, name, 2, e2, per4, floor_us);
    show(p, name, 4, e4, per4, floor_us);
    snprintf(name, sizeof name, "%s today", inst);
    show(p, name, 1, g1, per4, floor_us);
    show(p, name, 2, g2, per4, floor_us);
    show(p, name, 4, g4, per4, floor_us);
    snprintf(name, sizeof name, "%s A", inst);
    const Result a8 = run<S8, 4, 2, false>(p, rng), a16 = run<S8, 4, 4, false>(p, rng), a32 = run<S8, 4, 8, false>(p, rng);
    show(p, name, 8, a8, per4, floor_us);
    show(p, name, 16, a16, per4, floor_us);
    show(p, name, 32, a32, per4, floor_us);
    snprintf(name, sizeof name, "%s B", inst);
    const Result b8 = run<S8, 4, 2, true>(p, rng), b16 = run<S8, 4, 4, true>(p, rng), b32 = run<S8, 4, 8, true>(p, rng);
    show(p, name, 8, b8, per4, floor_us);
    show(p, name, 16, b16, per4, floor_us);
    show(p, name, 32, b32, per4, floor_us);
    snprintf(name, sizeof name, "%s A(G2)", inst);
    const Result c8 = run<S8, 2, 4, false>(p, rng), c16 = run<S8, 2, 8, false>(p, rng);
    show(p, name, 8, c8, per4, floor_us);
    show(p, name, 16, c16, per4, floor_us);
    // the K split over fewer warps, so that a block has more strips on the same input (input traffic a strip)
    for (int cap : {1, 2}) {
        snprintf(name, sizeof name, "%s A seg<=%d", inst, cap);
        show(p, name, 16, run<S8, 4, 4, false>(p, rng, cap), per4, floor_us);
        show(p, name, 32, run<S8, 4, 8, false>(p, rng, cap), per4, floor_us);
        snprintf(name, sizeof name, "%s B seg<=%d", inst, cap);
        show(p, name, 16, run<S8, 4, 4, true>(p, rng, cap), per4, floor_us);
        show(p, name, 32, run<S8, 4, 8, true>(p, rng, cap), per4, floor_us);
    }
    snprintf(name, sizeof name, "%s reg G8", inst);
    show(p, name, 8, run<S8, 8, 1, false>(p, rng), per4, floor_us);
    // cost a group adds: today 1 -> 4, wide 4 -> 32
    printf("  added a group (from the engine's G4): engine (1->4) %.2f us; A (4->8) %.2f, (4->16) %.2f, (4->32) %.2f; B (4->8) %.2f, (4->16) %.2f, "
           "(4->32) %.2f\n",
           (e4.us - e1.us) / 3, (a8.us - e4.us) / 4, (a16.us - e4.us) / 12, (a32.us - e4.us) / 28, (b8.us - e4.us) / 4, (b16.us - e4.us) / 12,
           (b32.us - e4.us) / 28);
}

int main(int argc, char** argv) {
    const size_t budget = size_t((argc > 1 ? atof(argv[1]) : 1.0) * double(size_t(1) << 30));
    const char* only = argc > 2 ? argv[2] : nullptr;
    struct Shape {
        const char* name;
        int rows, cols;
    } shapes[] = {{"gate+up", 43008, 5376}, {"down", 5376, 21504}, {"q", 8192, 5376}};
    std::mt19937 rng(11);
    for (const auto& sh : shapes)
        for (int bits : {6, 8}) {
            if (only && !strstr(only, sh.name)) continue;
            Problem p;
            make(p, sh.rows, sh.cols, bits, 32, budget);
            printf("%s %dx%d, %d bits a pair, %.1f MB of codes, %zu copies\n", sh.name, sh.rows, sh.cols, bits, p.mat_bytes / 1e6, p.copies);
            family<false>(p, rng);
            family<true>(p, rng);
            release(p);
            fflush(stdout);
        }
    return 0;
}
