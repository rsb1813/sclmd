// How fast the GPU multiplies a low-bit matrix by up to eight token vectors at once, straight from the stored codes.
//
// y[token][row] = scale[row] * sum over columns of W[row][col] * x[token][col], W stored as one code per pair of
// adjacent columns: 8 bits a pair (4 bits a weight) or 6 bits a pair (3 bits a weight), each code naming an entry
// of a codebook of pairs. The product runs on the tensor cores (PTX mma.sync m16n8k16, fp16 inputs, fp32 sums):
// a warp takes 16 rows by 16 columns at a time, and the eight columns of the instruction's other operand are the
// tokens, so one token or eight cost the same reads and the same instructions.
//
// The codes are stored in the order the instruction wants them: for rows 16s..16s+15 (a strip) and columns
// 16j..16j+15 (a tile), lane L = 4g + t of the warp holds the pairs
//   (row 16s+g, cols 16j+2t), (row 16s+g+8, cols 16j+2t), (row 16s+g, cols 16j+2t+8), (row 16s+g+8, cols 16j+2t+8),
// so a warp's read of a tile is 32 adjacent words (8 bits a pair) and nothing is shuffled on the way.
#define NOMINMAX
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;

__device__ __forceinline__ void mma(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// x[tokens][cols] (fp32, tokens <= 8) into the order the instruction reads it: for tile j, lane 4g + t holds token g's
// values at columns 16j + 2t, + 1, + 8, + 9 as four fp16 in one 64-bit word. Tokens past `tokens` are zero.
__global__ void x_fragments_k(const float* x, uint64_t* xf, int cols, int tokens) {
    const int j = blockIdx.x, lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    uint64_t w = 0;
    if (g < tokens) {
        const float* p = x + size_t(g) * cols + 16 * j + 2 * t;
        w = uint64_t(__half_as_ushort(__float2half(p[0]))) | uint64_t(__half_as_ushort(__float2half(p[1]))) << 16 |
            uint64_t(__half_as_ushort(__float2half(p[8]))) << 32 | uint64_t(__half_as_ushort(__float2half(p[9]))) << 48;
    }
    xf[size_t(j) * 32 + lane] = w;
}

// One block per strip of 16 rows; its `segs` warps split the strip's tiles between them and add their sums up at the
// end. BITS is the size of a pair's code. lut[code] is the pair as two fp16 in one word.
template <int BITS>
__global__ void pair_gemm_k(const uint32_t* codes, const uint32_t* lut, const float* scale, const uint64_t* xf, float* y, int rows,
                            int cols, int segs, int tokens) {
    __shared__ uint32_t slut[1 << BITS];
    __shared__ float part[8][32][4];
    for (int i = threadIdx.x; i < (1 << BITS); i += blockDim.x) slut[i] = lut[i];
    __syncthreads();
    const int strip = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tiles = cols / 16, per = tiles / segs, first = warp * per;  // per is a multiple of 4
    float c[4] = {0, 0, 0, 0};
    if (BITS == 8) {
        const uint32_t* cw = codes + (size_t(strip) * tiles + first) * 32 + lane;
        const uint64_t* xw = xf + size_t(first) * 32 + lane;
        for (int j = 0; j < per; j += 4) {
            uint32_t w[4];
            uint64_t b[4];
            for (int i = 0; i < 4; ++i) w[i] = cw[size_t(j + i) * 32], b[i] = xw[size_t(j + i) * 32];
            for (int i = 0; i < 4; ++i)
                mma(c, slut[w[i] & 255], slut[w[i] >> 8 & 255], slut[w[i] >> 16 & 255], slut[w[i] >> 24], uint32_t(b[i]), uint32_t(b[i] >> 32));
        }
    } else {  // 6 bits a pair: four tiles of a lane are 16 codes in 96 bits, stored as three words
        const uint32_t* cw = codes + (size_t(strip) * (tiles / 4) + first / 4) * 96 + lane;
        const uint64_t* xw = xf + size_t(first) * 32 + lane;
        for (int j = 0; j < per; j += 4) {
            const uint32_t* q = cw + size_t(j / 4) * 96;
            const uint64_t lo = uint64_t(q[0]) | uint64_t(q[32]) << 32, hi = q[64];
            uint64_t b[4];
            for (int i = 0; i < 4; ++i) b[i] = xw[size_t(j + i) * 32];
            const uint32_t k10 = uint32_t(lo >> 60 | hi << 4) & 63;
            mma(c, slut[lo & 63], slut[lo >> 6 & 63], slut[lo >> 12 & 63], slut[lo >> 18 & 63], uint32_t(b[0]), uint32_t(b[0] >> 32));
            mma(c, slut[lo >> 24 & 63], slut[lo >> 30 & 63], slut[lo >> 36 & 63], slut[lo >> 42 & 63], uint32_t(b[1]), uint32_t(b[1] >> 32));
            mma(c, slut[lo >> 48 & 63], slut[lo >> 54 & 63], slut[k10], slut[hi >> 2 & 63], uint32_t(b[2]), uint32_t(b[2] >> 32));
            mma(c, slut[hi >> 8 & 63], slut[hi >> 14 & 63], slut[hi >> 20 & 63], slut[hi >> 26 & 63], uint32_t(b[3]), uint32_t(b[3] >> 32));
        }
    }
    for (int i = 0; i < 4; ++i) part[warp][lane][i] = c[i];
    __syncthreads();
    if (warp) return;
    for (int i = 0; i < 4; ++i) {
        float sum = 0;
        for (int w = 0; w < segs; ++w) sum += part[w][lane][i];
        const int row = strip * 16 + (lane >> 2) + (i >= 2 ? 8 : 0), token = 2 * (lane & 3) + (i & 1);
        if (token < tokens) y[size_t(token) * rows + row] = scale[row] * sum;
    }
}

static float half_to_float(uint16_t h) {
    const int e = h >> 10 & 31, m = h & 1023;
    const float v = e == 0 ? std::ldexp(float(m), -24) : std::ldexp(float(m | 1024), e - 25);
    return h & 0x8000 ? -v : v;
}

static uint16_t float_to_half(float f) {  // nearest; the bench's values are far from fp16's limits
    if (f == 0) return 0;
    int e;
    const float m = std::frexp(std::fabs(f), &e);  // |f| = m * 2^e, m in [0.5, 1)
    int mant = int(std::lround(std::ldexp(m, 11))) - 1024, ex = e + 14;
    if (mant == 1024) mant = 0, ++ex;
    return uint16_t((f < 0 ? 0x8000 : 0) | ex << 10 | mant);
}

#define CHECK(call)                                                                      \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(err_), __LINE__);    \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

// pairs[row][cols / 2] (a code per pair) into the stored order.
static std::vector<uint32_t> pack(const std::vector<uint8_t>& pairs, int rows, int cols, int bits) {
    const int tiles = cols / 16, half = cols / 2;
    std::vector<uint32_t> out(bits == 8 ? size_t(rows / 16) * tiles * 32 : size_t(rows / 16) * (tiles / 4) * 96, 0);
    for (int s = 0; s < rows / 16; ++s)
        for (int j = 0; j < tiles; ++j)
            for (int lane = 0; lane < 32; ++lane) {
                const int g = lane >> 2, t = lane & 3;
                const int r0 = 16 * s + g, r1 = r0 + 8, p0 = 8 * j + t, p1 = p0 + 4;  // pair index = column / 2
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

template <int BITS>
static void run(int rows, int cols, int segs, int tokens, size_t budget_bytes, const char* name) {
    std::mt19937 rng(1);
    const int half = cols / 2, n_codes = 1 << BITS;
    std::vector<uint8_t> pairs(size_t(rows) * half);
    for (auto& p : pairs) p = uint8_t(rng() % n_codes);
    std::vector<uint32_t> lut(n_codes);
    std::vector<float> book(2 * n_codes), scale(rows), x(size_t(8) * cols);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int i = 0; i < n_codes; ++i) {
        const uint16_t a = float_to_half(normal(rng)), b = float_to_half(normal(rng));
        lut[i] = uint32_t(a) | uint32_t(b) << 16;
        book[2 * i] = half_to_float(a), book[2 * i + 1] = half_to_float(b);
    }
    for (auto& s : scale) s = 0.5f + 0.001f * float(rng() % 1000);
    for (auto& v : x) v = half_to_float(float_to_half(normal(rng)));  // fp16 values, so the check below is exact up to the sums
    const std::vector<uint32_t> codes = pack(pairs, rows, cols, BITS);
    const size_t mat_bytes = codes.size() * 4, copies = std::max<size_t>(1, budget_bytes / mat_bytes);

    uint32_t *d_codes, *d_lut;
    float *d_scale, *d_x, *d_y;
    uint64_t* d_xf;
    CHECK(cudaMalloc(&d_codes, mat_bytes * copies));
    CHECK(cudaMalloc(&d_lut, lut.size() * 4));
    CHECK(cudaMalloc(&d_scale, scale.size() * 4));
    CHECK(cudaMalloc(&d_x, x.size() * 4));
    CHECK(cudaMalloc(&d_y, size_t(8) * rows * 4));
    CHECK(cudaMalloc(&d_xf, size_t(cols / 16) * 32 * 8));
    for (size_t i = 0; i < copies; ++i) CHECK(cudaMemcpy((char*)d_codes + i * mat_bytes, codes.data(), mat_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_lut, lut.data(), lut.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_scale, scale.data(), scale.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));

    x_fragments_k<<<cols / 16, 32>>>(d_x, d_xf, cols, tokens);
    pair_gemm_k<BITS><<<rows / 16, 32 * segs>>>(d_codes, d_lut, d_scale, d_xf, d_y, rows, cols, segs, tokens);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> y(size_t(8) * rows);
    CHECK(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost));
    double worst = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const int row = rng() % rows, token = rng() % tokens;
        double want = 0, mag = 0;
        for (int p = 0; p < half; ++p) {
            const int k = pairs[size_t(row) * half + p];
            const double a = double(book[2 * k]) * x[size_t(token) * cols + 2 * p], b = double(book[2 * k + 1]) * x[size_t(token) * cols + 2 * p + 1];
            want += a + b, mag += std::fabs(a) + std::fabs(b);
        }
        worst = std::max(worst, std::fabs(want * scale[row] - y[size_t(token) * rows + row]) / (mag * scale[row]));
    }

    const int rounds = 3;
    double best = 1e9;
    for (int r = 0; r < rounds; ++r) {
        const auto t0 = clk::now();
        for (size_t i = 0; i < copies; ++i)
            pair_gemm_k<BITS><<<rows / 16, 32 * segs>>>((const uint32_t*)((const char*)d_codes + i * mat_bytes), d_lut, d_scale, d_xf, d_y, rows, cols, segs, tokens);
        CHECK(cudaDeviceSynchronize());
        best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count());
    }
    const double gbps = mat_bytes * copies / best / 1e9, per_weight_ns = best / (double(rows) * cols * copies) * 1e9;
    printf("%-5s %d bits/pair %5dx%-5d segs %d tokens %d: %6.1f us a matrix, %6.0f GB/s, %5.1f ms per 29.29e9 weights, error %.1e%s\n", name, BITS,
           rows, cols, segs, tokens, best / copies * 1e6, gbps, per_weight_ns * 29.29e9 * 1e-6, worst, worst > 1e-5 ? "  <-- WRONG" : "");
    cudaFree(d_codes), cudaFree(d_lut), cudaFree(d_scale), cudaFree(d_x), cudaFree(d_y), cudaFree(d_xf);
}

int main(int argc, char** argv) {
    const size_t budget = (argc > 1 ? atof(argv[1]) : 4.0) * (size_t(1) << 30);
    for (int tokens : {1, 8})
        for (int segs : {1, 2, 4, 7}) {  // a warp's share of the tiles is a multiple of 4
            run<8>(21504, 5376, segs, tokens, budget, "gate");
            run<6>(21504, 5376, segs, tokens, budget, "gate");
        }
    for (int segs : {4, 8}) {
        run<8>(5376, 21504, segs, 8, budget, "down");
        run<6>(5376, 21504, segs, 8, budget, "down");
    }
    for (int segs : {4, 7}) {
        run<8>(8192, 5376, segs, 8, budget, "q");
        run<6>(8192, 5376, segs, 8, budget, "q");
    }
    return 0;
}
