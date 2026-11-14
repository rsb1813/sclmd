// Kill test for integer products from the stored pair codes (brief B1-s8-products).
//
// The engine's gemm_k (engine/sclmd.cu) against an s8 variant of it, on the same codes, the same strip / tile order
// and the same launch shape (segs_for, GEMM_STRIPS), at G = 1, 2 and 4 groups of eight tokens:
//   fp16: today's form, mma.sync m16n8k16 f16 x f16 with f32 sums, a tile an instruction, lut[code] = two fp16.
//   s8:   mma.sync m16n8k32 s8 x s8 with s32 sums, two tiles an instruction. t8[code] = the pair as two int8, held
//         twice in one word (so one byte_perm makes a register of two pairs); the input is int8 with the same lane
//         layout as today's xf (lane 4g + t of tile j: token g's columns 16j + 2t, + 1, + 8, + 9), one word a tile.
// Fragments of m16n8k32 .s8 (PTX ISA): A reg0 = row g, k 4t..4t+3; reg1 = row g+8, k 4t..; reg2 = row g, k 16+4t..;
// reg3 = row g+8, k 16+4t... B reg0 = k 4t..4t+3 at column (token) g; reg1 = k 16+4t... C as in m16n8k16.
// Instruction k 0..15 is tile j (k 4t+i = column {2t, 2t+1, 2t+8, 2t+9}[i]), k 16..31 is tile j+1. The weight word of
// tile j holds (row g, 2t) | (row g+8, 2t) << 8 | (row g, 2t+8) << 16 | (row g+8, 2t+8) << 24, so
//   reg0 = pair(w & 255) | pair(w >> 16 & 255) << 16, reg1 = pair(w >> 8 & 255) | pair(w >> 24) << 16, for tile j, and
//   reg2, reg3 the same from tile j+1's word.
// The int32 sums are exact, so the check against the host is exact up to the final fp32 scaling.
#define NOMINMAX
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
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

// two pairs (each held twice in its word) as one register: the low pair from p, the high from q
__device__ __forceinline__ uint32_t join(uint32_t p, uint32_t q) { return __byte_perm(p, q, 0x5410); }

struct Mat {
    const uint32_t* codes;
    const uint32_t* lut;  // fp16 form: two fp16; s8 form: the two int8 twice (b0 b1 b0 b1)
    const float* scale;
    int rows, cols, bits, pad;
};

#define GEMM_STRIPS 4

// today's gemm_k, one matrix (engine/sclmd.cu, unchanged in its loop)
template <int G>
__global__ void gemm_k(Mat m, float* y, const uint64_t* xf, const float* fps, int segs, int tokens, int strips) {
    __shared__ uint32_t luts[GEMM_STRIPS][256];
    __shared__ float part[GEMM_STRIPS][8][32][4];
    const int local = (threadIdx.x >> 5) / segs, warp = (threadIdx.x >> 5) % segs, lane = threadIdx.x & 31;
    int strip = blockIdx.x * strips + local;
    const bool idle = strip >= m.rows / 16;
    uint32_t* slut = luts[local];
    if (!idle)
        for (int i = threadIdx.x - local * segs * 32; i < (1 << m.bits); i += segs * 32) slut[i] = m.lut[i];
    __syncthreads();
    const int tiles = m.cols / 16, per = idle ? 0 : tiles / segs, first = warp * per;
    float c[G][4];
    for (int g = 0; g < G; ++g) c[g][0] = c[g][1] = c[g][2] = c[g][3] = 0;
    const uint64_t* xw = xf + size_t(first) * 32 + lane;
    const size_t group = size_t(tiles) * 32;
    if (m.bits == 8) {
        const uint32_t* cw = m.codes + (size_t(strip) * tiles + first) * 32 + lane;
        for (int j = 0; j < per; j += 4) {
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
    } else {
        const uint32_t* cw = m.codes + (size_t(strip) * (tiles / 4) + first / 4) * 96 + lane;
        for (int j = 0; j < per; j += 4) {
            const uint32_t* q = cw + size_t(j / 4) * 96;
            const uint32_t q0 = q[0], q1 = q[32], q2 = q[64];
            uint64_t b[G][4];
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

// the s8 form: xf8[group][tile][lane] one word (four int8); y = scale[row] * book_scale * fps[token] * sum
template <int G>
__global__ void gemm8_k(Mat m, float book_scale, float* y, const uint32_t* xf, const float* fps, int segs, int tokens, int strips) {
    __shared__ uint32_t luts[GEMM_STRIPS][256];
    __shared__ int part[GEMM_STRIPS][8][32][4];
    const int local = (threadIdx.x >> 5) / segs, warp = (threadIdx.x >> 5) % segs, lane = threadIdx.x & 31;
    int strip = blockIdx.x * strips + local;
    const bool idle = strip >= m.rows / 16;
    uint32_t* slut = luts[local];
    if (!idle)
        for (int i = threadIdx.x - local * segs * 32; i < (1 << m.bits); i += segs * 32) slut[i] = m.lut[i];
    __syncthreads();
    const int tiles = m.cols / 16, per = idle ? 0 : tiles / segs, first = warp * per;
    int c[G][4];
    for (int g = 0; g < G; ++g) c[g][0] = c[g][1] = c[g][2] = c[g][3] = 0;
    const uint32_t* xw = xf + size_t(first) * 32 + lane;
    const size_t group = size_t(tiles) * 32;
    if (m.bits == 8) {
        const uint32_t* cw = m.codes + (size_t(strip) * tiles + first) * 32 + lane;
        for (int j = 0; j < per; j += 4) {
            uint32_t w[4], b[G][4];
            for (int i = 0; i < 4; ++i) w[i] = cw[size_t(j + i) * 32];
#pragma unroll
            for (int g = 0; g < G; ++g)
#pragma unroll
                for (int i = 0; i < 4; ++i) b[g][i] = xw[g * group + size_t(j + i) * 32];
#pragma unroll
            for (int i = 0; i < 4; i += 2) {
                const uint32_t u = w[i], v = w[i + 1];
                const uint32_t a0 = join(slut[u & 255], slut[u >> 16 & 255]), a1 = join(slut[u >> 8 & 255], slut[u >> 24]);
                const uint32_t a2 = join(slut[v & 255], slut[v >> 16 & 255]), a3 = join(slut[v >> 8 & 255], slut[v >> 24]);
                for (int g = 0; g < G; ++g) mma8(c[g], a0, a1, a2, a3, b[g][i], b[g][i + 1]);
            }
        }
    } else {
        const uint32_t* cw = m.codes + (size_t(strip) * (tiles / 4) + first / 4) * 96 + lane;
        for (int j = 0; j < per; j += 4) {
            const uint32_t* q = cw + size_t(j / 4) * 96;
            const uint32_t q0 = q[0], q1 = q[32], q2 = q[64];
            uint32_t b[G][4];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const uint32_t* x = xw + g * group + size_t(j) * 32;
                b[g][0] = x[0], b[g][1] = x[32], b[g][2] = x[64], b[g][3] = x[96];
            }
            const uint64_t lo = uint64_t(q0) | uint64_t(q1) << 32, hi = q2;
            const uint32_t k10 = uint32_t(lo >> 60 | hi << 4) & 63;
            // tile i's four pairs are a[4i..4i+3] = (g, 2t), (g+8, 2t), (g, 2t+8), (g+8, 2t+8)
            const uint32_t a[16] = {slut[lo & 63],       slut[lo >> 6 & 63],  slut[lo >> 12 & 63], slut[lo >> 18 & 63],
                                    slut[lo >> 24 & 63], slut[lo >> 30 & 63], slut[lo >> 36 & 63], slut[lo >> 42 & 63],
                                    slut[lo >> 48 & 63], slut[lo >> 54 & 63], slut[k10],           slut[hi >> 2 & 63],
                                    slut[hi >> 8 & 63],  slut[hi >> 14 & 63], slut[hi >> 20 & 63], slut[hi >> 26 & 63]};
            const uint32_t r0 = join(a[0], a[2]), r1 = join(a[1], a[3]), r2 = join(a[4], a[6]), r3 = join(a[5], a[7]);
            const uint32_t s0 = join(a[8], a[10]), s1 = join(a[9], a[11]), s2 = join(a[12], a[14]), s3 = join(a[13], a[15]);
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
                if (token < tokens) y[size_t(token) * m.rows + row] = m.scale[row] * book_scale * fps[token] * float(sum);
            }
        __syncthreads();
    }
}

static int segs_for(int cols) {
    const int tiles = cols / 16;
    for (int s = 8; s > 1; s >>= 1)
        if (tiles % (4 * s) == 0 && tiles / s >= 32) return s;
    return 1;
}

static float half_to_float(uint16_t h) {
    const int e = h >> 10 & 31, m = h & 1023;
    const float v = e == 0 ? std::ldexp(float(m), -24) : std::ldexp(float(m | 1024), e - 25);
    return h & 0x8000 ? -v : v;
}

static uint16_t float_to_half(float f) {
    if (f == 0) return 0;
    int e;
    const float m = std::frexp(std::fabs(f), &e);
    int mant = int(std::lround(std::ldexp(m, 11))) - 1024, ex = e + 14;
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

struct Result {
    double us, gbps, err;
};

// rows x cols at `bits` a pair, `tokens` tokens (G = ceil(tokens / 8) groups), both forms
template <int G>
static void run(const char* name, int rows, int cols, int bits, int tokens, size_t budget) {
    std::mt19937 rng(7);
    const int half = cols / 2, n_codes = 1 << bits, tiles = cols / 16;
    std::vector<uint8_t> pairs(size_t(rows) * half);
    for (auto& p : pairs) p = uint8_t(rng() % n_codes);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    // the codebook as int8 (one scale for the matrix) and the same values as fp16 (exact: |v| <= 127)
    std::vector<int8_t> book(2 * n_codes);
    for (auto& v : book) v = int8_t(std::max(-127.0f, std::min(127.0f, std::round(normal(rng) * 40.0f))));
    std::vector<uint32_t> lut16(n_codes), lut8(n_codes);
    for (int i = 0; i < n_codes; ++i) {
        lut16[i] = uint32_t(float_to_half(book[2 * i])) | uint32_t(float_to_half(book[2 * i + 1])) << 16;
        const uint32_t two = uint32_t(uint8_t(book[2 * i])) | uint32_t(uint8_t(book[2 * i + 1])) << 8;
        lut8[i] = two | two << 16;
    }
    std::vector<float> scale(rows);
    for (auto& s : scale) s = 0.5f + 0.001f * float(rng() % 1000);
    // the input as int8 values (fp16 form: the same integers as fp16, exact)
    const int ntok = 8 * G;
    std::vector<int8_t> x(size_t(ntok) * cols);
    for (auto& v : x) v = int8_t(std::max(-127.0f, std::min(127.0f, std::round(normal(rng) * 30.0f))));
    std::vector<float> fps(ntok, 1.0f);
    // fragments
    std::vector<uint64_t> xf16(size_t(G) * tiles * 32, 0);
    std::vector<uint32_t> xf8(size_t(G) * tiles * 32, 0);
    for (int gr = 0; gr < G; ++gr)
        for (int j = 0; j < tiles; ++j)
            for (int lane = 0; lane < 32; ++lane) {
                const int g = lane >> 2, t = lane & 3, token = 8 * gr + g;
                if (token >= tokens) continue;
                const int8_t* p = x.data() + size_t(token) * cols + 16 * j + 2 * t;
                xf16[(size_t(gr) * tiles + j) * 32 + lane] = uint64_t(float_to_half(p[0])) | uint64_t(float_to_half(p[1])) << 16 |
                                                              uint64_t(float_to_half(p[8])) << 32 | uint64_t(float_to_half(p[9])) << 48;
                xf8[(size_t(gr) * tiles + j) * 32 + lane] = uint32_t(uint8_t(p[0])) | uint32_t(uint8_t(p[1])) << 8 | uint32_t(uint8_t(p[8])) << 16 |
                                                             uint32_t(uint8_t(p[9])) << 24;
            }
    const std::vector<uint32_t> codes = pack(pairs, rows, cols, bits);
    const size_t mat_bytes = codes.size() * 4, copies = std::max<size_t>(1, budget / mat_bytes);
    uint32_t *d_codes, *d_l16, *d_l8, *d_xf8;
    uint64_t* d_xf16;
    float *d_scale, *d_fps, *d_y;
    CHECK(cudaMalloc(&d_codes, mat_bytes * copies));
    CHECK(cudaMalloc(&d_l16, n_codes * 4));
    CHECK(cudaMalloc(&d_l8, n_codes * 4));
    CHECK(cudaMalloc(&d_xf16, xf16.size() * 8));
    CHECK(cudaMalloc(&d_xf8, xf8.size() * 4));
    CHECK(cudaMalloc(&d_scale, rows * 4));
    CHECK(cudaMalloc(&d_fps, ntok * 4));
    CHECK(cudaMalloc(&d_y, size_t(ntok) * rows * 4));
    for (size_t i = 0; i < copies; ++i) CHECK(cudaMemcpy((char*)d_codes + i * mat_bytes, codes.data(), mat_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_l16, lut16.data(), n_codes * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_l8, lut8.data(), n_codes * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_xf16, xf16.data(), xf16.size() * 8, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_xf8, xf8.data(), xf8.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_scale, scale.data(), rows * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_fps, fps.data(), ntok * 4, cudaMemcpyHostToDevice));

    const int segs = segs_for(cols), strips = GEMM_STRIPS * segs * 32 <= 1024 ? GEMM_STRIPS : 1024 / (segs * 32);
    const int blocks = (rows / 16 + strips - 1) / strips, threads = 32 * segs * strips;
    const float book_scale = 1.0f;
    Result res[2];
    for (int form = 0; form < 2; ++form) {
        auto launch = [&](size_t i) {
            Mat m{(const uint32_t*)((const char*)d_codes + i * mat_bytes), form ? d_l8 : d_l16, d_scale, rows, cols, bits, 0};
            if (form == 0)
                gemm_k<G><<<blocks, threads>>>(m, d_y, d_xf16, d_fps, segs, tokens, strips);
            else
                gemm8_k<G><<<blocks, threads>>>(m, book_scale, d_y, d_xf8, d_fps, segs, tokens, strips);
        };
        CHECK(cudaMemset(d_y, 0, size_t(ntok) * rows * 4));
        launch(0);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        std::vector<float> y(size_t(ntok) * rows);
        CHECK(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost));
        double worst = 0;
        for (int trial = 0; trial < 300; ++trial) {
            const int row = rng() % rows, token = rng() % tokens;
            int64_t want = 0;
            for (int p = 0; p < half; ++p) {
                const int k = pairs[size_t(row) * half + p];
                want += int64_t(book[2 * k]) * x[size_t(token) * cols + 2 * p] + int64_t(book[2 * k + 1]) * x[size_t(token) * cols + 2 * p + 1];
            }
            const double w = double(want) * scale[row], got = y[size_t(token) * rows + row];
            worst = std::max(worst, std::fabs(w - got) / std::max(1.0, std::fabs(w)));
        }
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0));
        CHECK(cudaEventCreate(&e1));
        float best = 1e30f;
        for (int r = 0; r < 4; ++r) {
            CHECK(cudaEventRecord(e0));
            for (size_t i = 0; i < copies; ++i) launch(i);
            CHECK(cudaEventRecord(e1));
            CHECK(cudaEventSynchronize(e1));
            float ms;
            CHECK(cudaEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms);
        }
        cudaEventDestroy(e0), cudaEventDestroy(e1);
        const double s = best * 1e-3 / copies;
        res[form] = {s * 1e6, mat_bytes / s / 1e9, worst};
    }
    printf("%-8s %5dx%-5d %d bits/pair G=%d (tokens %2d) segs %d: fp16 %8.1f us %5.0f GB/s err %.1e | s8 %8.1f us %5.0f GB/s err %.1e | s8/fp16 %.3f\n",
           name, rows, cols, bits, G, tokens, segs, res[0].us, res[0].gbps, res[0].err, res[1].us, res[1].gbps, res[1].err, res[1].us / res[0].us);
    cudaFree(d_codes), cudaFree(d_l16), cudaFree(d_l8), cudaFree(d_xf16), cudaFree(d_xf8), cudaFree(d_scale), cudaFree(d_fps), cudaFree(d_y);
}

int main(int argc, char** argv) {
    const size_t budget = size_t((argc > 1 ? atof(argv[1]) : 2.0) * double(size_t(1) << 30));
    struct Shape {
        const char* name;
        int rows, cols;
    } shapes[] = {{"gate+up", 43008, 5376}, {"down", 5376, 21504}, {"q", 8192, 5376}};
    for (const auto& sh : shapes)
        for (int bits : {6, 8}) {
            run<1>(sh.name, sh.rows, sh.cols, bits, 8, budget);
            run<2>(sh.name, sh.rows, sh.cols, bits, 16, budget);
            run<4>(sh.name, sh.rows, sh.cols, bits, 32, budget);
        }
    return 0;
}
