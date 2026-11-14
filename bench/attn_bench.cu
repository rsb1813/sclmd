// How long a full-attention layer's scores, softmax and weighted sum take in a wide pass, on synthetic int8 keys and
// values (no model): today's kernels (copied from engine/sclmd.cu) against variants.
//   S0 / M0 / X / A0: today's scores32_k, scores_made32_k, softmax_k, attend32_k.
//   S1 / A1: the same kernels with the int8 -> fp16 table replaced by arithmetic (same bits).
//   S2 / M2: a block decodes a key tile once into shared memory (fp16) and its warps, one query group each with the
//            group's queries in registers, read it with ldmatrix (same bits).
//   A2:      a block decodes a value tile once into shared memory and its warps (4 groups x 2 halves of the head)
//            read it with ldmatrix.trans; positions split into fixed chunks added in order by a second kernel (other
//            bits unless one chunk; with one chunk the bits are today's).
//   S3:      s8 scores (mma m16n8k32) on the stored int8 keys with int8 queries (a scale per query head): lossy.
// One layer: P positions, 4 key heads, head 512, 32 query heads, the pass's R rows at the last R positions.
// Usage: attn_bench.exe [P=65536] [reps=5]
#define NOMINMAX
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e = (x);                                                                     \
        if (e != cudaSuccess) {                                                                  \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);      \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

static const int MAX_SEGS = 32;
struct Att {
    int n, most;
    int row0[MAX_SEGS], cnt[MAX_SEGS], pos0[MAX_SEGS], seq[MAX_SEGS], base[MAX_SEGS], stride[MAX_SEGS], tile0[MAX_SEGS], off[MAX_SEGS];
};

// ---------------------------------------------------------------------------------------------- copied from engine
__device__ __forceinline__ void mma(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ int seg_of_row(const Att& a, int row) {
    int s = 0;
    for (int i = 1; i < a.n; ++i)
        if (row >= a.row0[i]) s = i;
    return s;
}

__device__ __forceinline__ size_t slot_of(const int32_t* tiles, int ring, int seq, int pos) {
    return size_t(tiles[size_t(seq) * ring + (pos >> 4) % ring]) * 16 + (pos & 15);
}

#define BOOST 4096.0f

__device__ __forceinline__ void fill_table(uint16_t* table) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) table[i] = __half_as_ushort(__float2half(float(int8_t(i))));
    __syncthreads();
}

__device__ __forceinline__ uint32_t two(const uint16_t* table, uint16_t w) {
    return uint32_t(table[w & 255]) | uint32_t(table[w >> 8]) << 16;
}

// the same as two() by arithmetic: byte b as fp16 1024 + (b ^ 0x80), minus 1152 (exact)
__device__ __forceinline__ uint32_t cvt2(uint32_t w16) {
    const uint32_t x = __byte_perm(w16 ^ 0x8080u, 0x64646464u, 0x4140);
    const __half2 h = __hsub2(*(const __half2*)&x, __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480)));
    return *(const uint32_t*)&h;
}

template <bool CVT>
__device__ __forceinline__ uint32_t two_k(const uint16_t* table, uint16_t w) {
    return CVT ? cvt2(w) : two(table, w);
}

__device__ float block_sum(float v) {
    __shared__ float warp_sums[32];
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0;
    for (int w = 0; w < blockDim.x / 32; ++w) total += warp_sums[w];
    __syncthreads();
    return total;
}

__device__ float block_max(float v) {
    __shared__ float warp_max[32];
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = -INFINITY;
    for (int w = 0; w < blockDim.x / 32; ++w) total = fmaxf(total, warp_max[w]);
    __syncthreads();
    return total;
}

template <int STEPS, bool CVT>
__device__ __forceinline__ void scores_steps(float c[4], int& j, int to, const uint16_t* table, const int8_t* ka, const int8_t* kb,
                                             const uint16_t* qa, bool has) {
    for (; j + 16 * STEPS <= to; j += 16 * STEPS) {
        uint16_t w[STEPS][4];
        uint32_t b0[STEPS], b1[STEPS];
#pragma unroll
        for (int u = 0; u < STEPS; ++u) {
            const int at = j + 16 * u;
            w[u][0] = *(const uint16_t*)(ka + at), w[u][1] = *(const uint16_t*)(kb + at);
            w[u][2] = *(const uint16_t*)(ka + at + 8), w[u][3] = *(const uint16_t*)(kb + at + 8);
            b0[u] = has ? *(const uint32_t*)(qa + at) : 0, b1[u] = has ? *(const uint32_t*)(qa + at + 8) : 0;
        }
#pragma unroll
        for (int u = 0; u < STEPS; ++u)
            mma(c, two_k<CVT>(table, w[u][0]), two_k<CVT>(table, w[u][1]), two_k<CVT>(table, w[u][2]), two_k<CVT>(table, w[u][3]), b0[u], b1[u]);
    }
}

template <int AHEAD, bool CVT>
__device__ __forceinline__ void scores_plain(float c[4], int& j, int to, const uint16_t* table, const int8_t* ka, const int8_t* kb,
                                             const uint16_t* qa, bool has) {
    scores_steps<AHEAD, CVT>(c, j, to, table, ka, kb, qa, has);
    if (AHEAD > 8) scores_steps<8, CVT>(c, j, to, table, ka, kb, qa, has);
    if (AHEAD > 2) scores_steps<2, CVT>(c, j, to, table, ka, kb, qa, has);
    scores_steps<1, CVT>(c, j, to, table, ka, kb, qa, has);
}

__device__ __forceinline__ uint32_t made(uint16_t own, uint16_t part, uint64_t tr, const float* ko, const float* kp, float sign) {
    const float c0 = __half2float(__ushort_as_half(uint16_t(tr))), s0 = __half2float(__ushort_as_half(uint16_t(tr >> 16)));
    const float c1 = __half2float(__ushort_as_half(uint16_t(tr >> 32))), s1 = __half2float(__ushort_as_half(uint16_t(tr >> 48)));
    const float e0 = float(int8_t(own & 255)) * ko[0] * c0 + sign * (float(int8_t(part & 255)) * kp[0] * s0);
    const float e1 = float(int8_t(own >> 8)) * ko[1] * c1 + sign * (float(int8_t(part >> 8)) * kp[1] * s1);
    return uint32_t(__half_as_ushort(__float2half(e0))) | uint32_t(__half_as_ushort(__float2half(e1))) << 16;
}

template <int STEPS>
__device__ __forceinline__ void scores_made(float c[4], int& j, int to, int first, int half, int turned, const int8_t* va, const int8_t* vb,
                                            const uint64_t* ta, const uint64_t* tb, const float* kn, int t, const uint16_t* qa, bool has) {
    const int other = first ? -half : half;
    const float sign = first ? 1.0f : -1.0f;
    const float *ko = kn + (first ? turned : 0) + 2 * t, *kp = kn + (first ? 0 : turned) + 2 * t;
    for (; j + 16 * STEPS <= to; j += 16 * STEPS) {
        uint16_t w[STEPS][4], x[STEPS][4];
        uint64_t tr[STEPS][4];
        uint32_t b0[STEPS], b1[STEPS];
#pragma unroll
        for (int u = 0; u < STEPS; ++u) {
            const int at = j + 16 * u, i = ((at - first) >> 1) + t;
            w[u][0] = *(const uint16_t*)(va + at), w[u][1] = *(const uint16_t*)(vb + at);
            w[u][2] = *(const uint16_t*)(va + at + 8), w[u][3] = *(const uint16_t*)(vb + at + 8);
            x[u][0] = *(const uint16_t*)(va + at + other), x[u][1] = *(const uint16_t*)(vb + at + other);
            x[u][2] = *(const uint16_t*)(va + at + other + 8), x[u][3] = *(const uint16_t*)(vb + at + other + 8);
            tr[u][0] = ta[i], tr[u][1] = tb[i], tr[u][2] = ta[i + 4], tr[u][3] = tb[i + 4];
            b0[u] = has ? *(const uint32_t*)(qa + at) : 0, b1[u] = has ? *(const uint32_t*)(qa + at + 8) : 0;
        }
#pragma unroll
        for (int u = 0; u < STEPS; ++u) {
            const int p = j + 16 * u - first;
            mma(c, made(w[u][0], x[u][0], tr[u][0], ko + p, kp + p, sign), made(w[u][1], x[u][1], tr[u][1], ko + p, kp + p, sign),
                made(w[u][2], x[u][2], tr[u][2], ko + p + 8, kp + p + 8, sign), made(w[u][3], x[u][3], tr[u][3], ko + p + 8, kp + p + 8, sign),
                b0[u], b1[u]);
        }
    }
}

template <int AHEAD, bool MADE, bool CVT>
__device__ __forceinline__ void scores_body(const uint16_t* qhat, const int8_t* keys, const float* kscale, float* scores, const int32_t* tiles,
                                            const Att& a, int heads, int kvh, int hd, int ring, const uint16_t* trig = nullptr,
                                            const float* k_norm = nullptr, int turned = 0) {
    __shared__ uint16_t table[256];
    __shared__ float kn[MADE ? 256 : 1];
    if (MADE)
        for (int i = threadIdx.x; i < 2 * turned; i += blockDim.x) kn[i] = k_norm[i < turned ? i : hd / 2 + i - turned];
    fill_table(table);
    int seg = 0;
    for (int i = 1; i < a.n; ++i)
        if (int(blockIdx.x) >= a.tile0[i]) seg = i;
    const int rep = heads / kvh, nq = rep * a.cnt[seg], stride = a.stride[seg];
    const int tile = blockIdx.x - a.tile0[seg], kh = blockIdx.y, group = blockIdx.z * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    if (group * 8 >= nq) return;
    const size_t slot = slot_of(tiles, ring, a.seq[seg], a.base[seg] + 16 * tile);
    const int8_t* ka = keys + ((slot + g) * kvh + kh) * hd + 2 * t;
    const int8_t* kb = keys + ((slot + g + 8) * kvh + kh) * hd + 2 * t;
    const int query = group * 8 + g;
    const bool has = query < nq;
    const uint16_t* qa = qhat + (size_t(a.row0[seg] + (has ? query / rep : 0)) * heads + kh * rep + (has ? query % rep : 0)) * hd + 2 * t;
    float c[4] = {0, 0, 0, 0};
    int j = 0;
    if (MADE) {
        const int half = hd / 2, pos = a.base[seg] + 16 * tile + g;
        const uint64_t* ta = (const uint64_t*)(trig + size_t(pos) * turned * 2);
        const uint64_t* tb = (const uint64_t*)(trig + size_t(pos + 8) * turned * 2);
        for (int first = 0; first < hd; first += half) {
            scores_made<4>(c, j, first + turned, first, half, turned, ka, kb, ta, tb, kn, t, qa, has);
            scores_made<1>(c, j, first + turned, first, half, turned, ka, kb, ta, tb, kn, t, qa, has);
            scores_plain<AHEAD, CVT>(c, j, first + half, table, ka, kb, qa, has);
        }
    } else {
        scores_plain<AHEAD, CVT>(c, j, hd, table, ka, kb, qa, has);
    }
    float* out = scores + a.off[seg];
    const float sa = kscale[(slot + g) * kvh + kh], sb = kscale[(slot + g + 8) * kvh + kh];
    for (int i = 0; i < 4; ++i) {
        const int q = group * 8 + 2 * t + (i & 1), r = 16 * tile + g + (i >= 2 ? 8 : 0);
        if (q < nq) out[(size_t(kh) * nq + q) * stride + r] = c[i] * (i >= 2 ? sb : sa);
    }
}

template <bool CVT>
__global__ void scores32_k(const uint16_t* qhat, const int8_t* keys, const float* kscale, float* scores, const int32_t* tiles, Att a, int heads,
                           int kvh, int hd, int ring) {
    scores_body<32, false, CVT>(qhat, keys, kscale, scores, tiles, a, heads, kvh, hd, ring);
}

__global__ void scores_made32_k(const uint16_t* qhat, const int8_t* values, const float* vscale, float* scores, const int32_t* tiles, Att a,
                                int heads, int kvh, int hd, int ring, const uint16_t* trig, const float* k_norm, int turned) {
    scores_body<32, true, false>(qhat, values, vscale, scores, tiles, a, heads, kvh, hd, ring, trig, k_norm, turned);
}

__global__ void softmax_k(const float* scores, uint16_t* probs, const float* vscale, const int32_t* tiles, Att a, int heads, int kvh,
                          int window, int ring) {
    const int n = blockDim.x, rep = heads / kvh, kh = blockIdx.y, row = blockIdx.x / rep, seg = seg_of_row(a, row);
    const int nq = rep * a.cnt[seg], query = blockIdx.x - rep * a.row0[seg], pos = a.pos0[seg] + row - a.row0[seg];
    const int base = a.base[seg], stride = a.stride[seg];
    const size_t at = a.off[seg] + (size_t(kh) * nq + query) * stride;
    const float* s = scores + at;
    uint16_t* p = probs + at;
    const int first = (window > 0 && pos - window + 1 > 0 ? pos - window + 1 : 0) - base, last = pos - base;
    const int mine = first + ((int(threadIdx.x) - (base + first) % n + n) % n);
    float mx = -INFINITY;
    for (int r = mine; r <= last; r += n) mx = fmaxf(mx, s[r]);
    mx = block_max(mx);
    float sum = 0;
    for (int r = mine; r <= last; r += n) sum += expf(s[r] - mx);
    sum = block_sum(sum);
    const int32_t* own = tiles + size_t(a.seq[seg]) * ring;
    const int j0 = (base >> 4) % ring;
    for (int r = threadIdx.x; r < stride; r += n) {
        if (r < first || r > last) {
            p[r] = 0;
            continue;
        }
        const int j = j0 + (r >> 4);
        const float scale = vscale[(size_t(own[j < ring ? j : j - ring]) * 16 + (r & 15)) * kvh + kh];
        p[r] = __half_as_ushort(__float2half(expf(s[r] - mx) / sum * (scale * BOOST)));
    }
}

struct Walk {
    const int32_t* mine;
    const uint8_t* values;
    const uint16_t* pq;
    size_t tile_values, in_tile, row;
    int j0, ring;
    bool has;
};

template <int TILES, bool CVT>
__device__ __forceinline__ void attend_tiles(float c[4], int& r, int stride, const uint16_t* table, const Walk& k) {
#define PAIR(x, y) (CVT ? cvt2(uint32_t(x) | uint32_t(y) << 8) : (uint32_t(table[x]) | uint32_t(table[y]) << 16))
#define LOW(w) ((w) & 255)
#define HIGH(w) ((w) >> 8)
    for (; r + 16 * TILES <= stride; r += 16 * TILES) {
        int32_t where[TILES];
        uint16_t x[TILES][4];
        uint32_t b0[TILES], b1[TILES];
#pragma unroll
        for (int u = 0; u < TILES; ++u) {
            const int j = k.j0 + (r >> 4) + u;
            where[u] = k.mine[j < k.ring ? j : j - k.ring];
            b0[u] = k.has ? *(const uint32_t*)(k.pq + r + 16 * u) : 0, b1[u] = k.has ? *(const uint32_t*)(k.pq + r + 16 * u + 8) : 0;
        }
#pragma unroll
        for (int u = 0; u < TILES; ++u) {
            const uint8_t* va = k.values + size_t(where[u]) * k.tile_values + k.in_tile;
            x[u][0] = *(const uint16_t*)va, x[u][1] = *(const uint16_t*)(va + k.row);
            x[u][2] = *(const uint16_t*)(va + 8 * k.row), x[u][3] = *(const uint16_t*)(va + 9 * k.row);
        }
#pragma unroll
        for (int u = 0; u < TILES; ++u)
            mma(c, PAIR(LOW(x[u][0]), LOW(x[u][1])), PAIR(HIGH(x[u][0]), HIGH(x[u][1])), PAIR(LOW(x[u][2]), LOW(x[u][3])),
                PAIR(HIGH(x[u][2]), HIGH(x[u][3])), b0[u], b1[u]);
    }
#undef PAIR
#undef LOW
#undef HIGH
}

template <int AHEAD, bool CVT>
__global__ void attend32_k(const uint16_t* probs, const int8_t* values, float* out, const int32_t* tiles, Att a, int heads, int kvh, int hd,
                           int ring) {
    __shared__ uint16_t table[256];
    fill_table(table);
    const int seg = blockIdx.z, rep = heads / kvh, nq = rep * a.cnt[seg], stride = a.stride[seg], base = a.base[seg];
    const int dt = blockIdx.x % (hd / 16), group = (blockIdx.x / (hd / 16)) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int kh = blockIdx.y, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    if (group * 8 >= nq) return;
    const int query = group * 8 + g;
    const bool has = query < nq;
    const size_t col = size_t(kh) * hd + 16 * dt + 2 * g, row = size_t(kvh) * hd;
    const Walk k{tiles + size_t(a.seq[seg]) * ring, (const uint8_t*)values, probs + a.off[seg] + (size_t(kh) * nq + (has ? query : 0)) * stride + 2 * t,
                 16 * row, 2 * t * row + col, row, (base >> 4) % ring, ring, has};
    float c[4] = {0, 0, 0, 0};
    int r = 0;
    attend_tiles<AHEAD, CVT>(c, r, stride, table, k);
    if (AHEAD > 8) attend_tiles<8, CVT>(c, r, stride, table, k);
    if (AHEAD > 2) attend_tiles<2, CVT>(c, r, stride, table, k);
    attend_tiles<1, CVT>(c, r, stride, table, k);
    for (int i = 0; i < 4; ++i) {
        const int q = group * 8 + 2 * t + (i & 1), d = 16 * dt + 2 * g + (i >= 2 ? 1 : 0);
        if (q < nq) out[(size_t(a.row0[seg] + q / rep) * heads + kh * rep + q % rep) * hd + d] = c[i] * (1.0f / BOOST);
    }
}

// ---------------------------------------------------------------------------------------------- variants
// A3: today's weighted sum with a warp taking NDT sixteens of the head (its weights loaded once for all of them, and
// its values one load of 2 NDT bytes a position); every sum still adds the tiles in order: same bits.
// Block (NDT-sixteens chunk of the head x blocks of groups, key head, segment); lane's rows g, g + 8 of sixteen k are
// the values c0 + 2 NDT g + 2k, + 1.
template <int NDT>
struct Wide;
template <> struct Wide<1> { using T = uint16_t; };
template <> struct Wide<2> { using T = uint32_t; };
template <> struct Wide<4> { using T = uint64_t; };

template <int AHEAD, int NDT>
__global__ void attend3_k(const uint16_t* probs, const int8_t* values, float* out, const int32_t* tiles, Att a, int heads, int kvh, int hd,
                          int ring) {
    using W = typename Wide<NDT>::T;
    const int seg = blockIdx.z, rep = heads / kvh, nq = rep * a.cnt[seg], stride = a.stride[seg], base = a.base[seg];
    const int chunks = hd / (16 * NDT), dc = blockIdx.x % chunks, group = (blockIdx.x / chunks) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int kh = blockIdx.y, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    if (group * 8 >= nq) return;
    const int query = group * 8 + g;
    const bool has = query < nq;
    const size_t c0 = size_t(16 * NDT) * dc, col = size_t(kh) * hd + c0 + 2 * NDT * g, row = size_t(kvh) * hd;
    const int32_t* mine = tiles + size_t(a.seq[seg]) * ring;
    const uint8_t* vals = (const uint8_t*)values + 2 * t * row + col;
    const uint16_t* pq = probs + a.off[seg] + (size_t(kh) * nq + (has ? query : 0)) * stride + 2 * t;
    const int j0 = (base >> 4) % ring;
    float c[NDT][4];
#pragma unroll
    for (int k = 0; k < NDT; ++k) c[k][0] = c[k][1] = c[k][2] = c[k][3] = 0;
    int r = 0;
    auto run = [&](auto steps) {
        constexpr int S = decltype(steps)::value;
        for (; r + 16 * S <= stride; r += 16 * S) {
            int32_t where[S];
            W x[S][4];
            uint32_t b0[S], b1[S];
#pragma unroll
            for (int u = 0; u < S; ++u) {
                const int j = j0 + (r >> 4) + u;
                where[u] = mine[j < ring ? j : j - ring];
                b0[u] = has ? *(const uint32_t*)(pq + r + 16 * u) : 0, b1[u] = has ? *(const uint32_t*)(pq + r + 16 * u + 8) : 0;
            }
#pragma unroll
            for (int u = 0; u < S; ++u) {
                const uint8_t* va = vals + size_t(where[u]) * 16 * row;
                x[u][0] = *(const W*)va, x[u][1] = *(const W*)(va + row), x[u][2] = *(const W*)(va + 8 * row), x[u][3] = *(const W*)(va + 9 * row);
            }
#pragma unroll
            for (int u = 0; u < S; ++u)
#pragma unroll
                for (int k = 0; k < NDT; ++k) {
                    const uint32_t y0 = uint32_t(x[u][0] >> (16 * k)) & 0xffff, y1 = uint32_t(x[u][1] >> (16 * k)) & 0xffff;
                    const uint32_t y2 = uint32_t(x[u][2] >> (16 * k)) & 0xffff, y3 = uint32_t(x[u][3] >> (16 * k)) & 0xffff;
                    mma(c[k], cvt2(__byte_perm(y0, y1, 0x40)), cvt2(__byte_perm(y0, y1, 0x51)), cvt2(__byte_perm(y2, y3, 0x40)),
                        cvt2(__byte_perm(y2, y3, 0x51)), b0[u], b1[u]);
                }
        }
    };
    run(std::integral_constant<int, AHEAD>{});
    if (AHEAD > 8) run(std::integral_constant<int, 8>{});
    if (AHEAD > 2) run(std::integral_constant<int, 2>{});
    run(std::integral_constant<int, 1>{});
#pragma unroll
    for (int k = 0; k < NDT; ++k)
        for (int i = 0; i < 4; ++i) {
            const int q = group * 8 + 2 * t + (i & 1), d = int(c0) + 2 * NDT * g + 2 * k + (i >= 2 ? 1 : 0);
            if (q < nq) out[(size_t(a.row0[seg] + q / rep) * heads + kh * rep + q % rep) * hd + d] = c[k][i] * (1.0f / BOOST);
        }
}

__device__ __forceinline__ void ldsm4(uint32_t r[4], const void* p) {
    const uint32_t a = uint32_t(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t r[4], const void* p) {
    const uint32_t a = uint32_t(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

static const int HD = 512, SROW = HD + 8;  // a shared tile row: 512 fp16 and 16 bytes of padding (no bank conflicts)

// 16 positions x 512 values of key head kh from int8 memory into the shared tile as fp16 (MADE: the keys made of the
// values, as scores_made does). 256 threads: thread i takes 32 bytes, row i / 16, values 32 (i % 16) on.
template <bool MADE>
__device__ __forceinline__ void decode_tile(uint16_t* sh, const int8_t* src, int kvh, int kh, int pos, const uint16_t* trig, const float* kn,
                                            int turned) {
    const int i = threadIdx.x, p = i >> 4, d0 = (i & 15) * 32;
    const int8_t* row = src + (size_t(p) * kvh + kh) * HD;
    uint4 w[2];
    w[0] = *(const uint4*)(row + d0), w[1] = *(const uint4*)(row + d0 + 16);
    const uint32_t* ws = (const uint32_t*)w;
    uint32_t o[16];
    if (!MADE) {
#pragma unroll
        for (int k = 0; k < 8; ++k) o[2 * k] = cvt2(ws[k] & 0xffff), o[2 * k + 1] = cvt2(ws[k] >> 16);
    } else {
        const int half = HD / 2, h = d0 >= half, e0 = d0 - h * half;
        if (e0 < turned) {  // all 32 values of the run are turned (turned is a multiple of 16; 32-runs: needs turned % 32 == 0)
            const int8_t* prow = row + d0 + (h ? -half : half);
            uint4 pw[2];
            pw[0] = *(const uint4*)prow, pw[1] = *(const uint4*)(prow + 16);
            const uint32_t* ps = (const uint32_t*)pw;
            const uint64_t* tr = (const uint64_t*)(trig + size_t(pos + p) * turned * 2);
            const float sign = h ? 1.0f : -1.0f;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const int e = e0 + 2 * k;
                const uint32_t ow = (ws[k >> 1] >> (16 * (k & 1))) & 0xffff, pp = (ps[k >> 1] >> (16 * (k & 1))) & 0xffff;
                o[k] = made(uint16_t(ow), uint16_t(pp), tr[e >> 1], kn + h * turned + e, kn + (1 - h) * turned + e, sign);
            }
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) o[2 * k] = cvt2(ws[k] & 0xffff), o[2 * k + 1] = cvt2(ws[k] >> 16);
        }
    }
    uint4* dst = (uint4*)(sh + p * SROW + d0);
#pragma unroll
    for (int k = 0; k < 4; ++k) dst[k] = make_uint4(o[4 * k], o[4 * k + 1], o[4 * k + 2], o[4 * k + 3]);
}

// S2 / M2: block (chunk of CT tiles, key head, eight query groups); 256 threads, a warp a group with its queries in
// registers. One segment (the bench's), so the segment search is left out.
template <bool MADE>
__global__ void __launch_bounds__(256) scores_once_k(const uint16_t* qhat, const int8_t* keys, const float* kscale, float* scores,
                                                     const int32_t* tiles, Att a, int heads, int kvh, int ring, int ct, const uint16_t* trig,
                                                     const float* k_norm, int turned) {
    __shared__ __align__(16) uint16_t sh[16 * SROW];
    __shared__ float kn[256];
    if (MADE)
        for (int i = threadIdx.x; i < 2 * turned; i += blockDim.x) kn[i] = k_norm[i < turned ? i : HD / 2 + i - turned];
    const int seg = 0, rep = heads / kvh, nq = rep * a.cnt[seg], stride = a.stride[seg], ntiles = stride / 16;
    const int kh = blockIdx.y, warp = threadIdx.x >> 5, group = blockIdx.z * 8 + warp;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int query = group * 8 + g;
    const bool has = query < nq, any = group * 8 < nq;
    const uint16_t* qa = qhat + (size_t(a.row0[seg] + (has ? query / rep : 0)) * heads + kh * rep + (has ? query % rep : 0)) * HD + 2 * t;
    uint32_t b0[32], b1[32];
#pragma unroll
    for (int u = 0; u < 32; ++u) b0[u] = has ? *(const uint32_t*)(qa + 16 * u) : 0, b1[u] = has ? *(const uint32_t*)(qa + 16 * u + 8) : 0;
    float* out = scores + a.off[seg];
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;
    const int t0 = blockIdx.x * ct, t1 = min(ntiles, t0 + ct);
    for (int tile = t0; tile < t1; ++tile) {
        const size_t slot = slot_of(tiles, ring, a.seq[seg], a.base[seg] + 16 * tile);
        __syncthreads();
        decode_tile<MADE>(sh, keys + slot * kvh * HD, kvh, kh, a.base[seg] + 16 * tile, trig, kn, turned);
        __syncthreads();
        if (!any) continue;
        float c[4] = {0, 0, 0, 0};
#pragma unroll
        for (int u = 0; u < 32; ++u) {
            uint32_t r[4];
            ldsm4(r, sh + lrow * SROW + 16 * u + lcol);
            mma(c, r[0], r[1], r[2], r[3], b0[u], b1[u]);
        }
        const float sa = kscale[(slot + g) * kvh + kh], sb = kscale[(slot + g + 8) * kvh + kh];
        for (int i = 0; i < 4; ++i) {
            const int q = group * 8 + 2 * t + (i & 1), r = 16 * tile + g + (i >= 2 ? 8 : 0);
            if (q < nq) out[(size_t(kh) * nq + q) * stride + r] = c[i] * (i >= 2 ? sb : sa);
        }
    }
}

// A2: block (chunk of CT tiles, key head, four query groups); 256 threads: warp w takes group w / 2 and values
// 256 (w % 2) .. + 255 (sixteen 16s, 64 sums). Partial sums part[chunk][kh][query][512] (fp32, the BOOSTed units).
__global__ void __launch_bounds__(256) attend_once_k(const uint16_t* probs, const int8_t* values, float* part, const int32_t* tiles, Att a,
                                                     int heads, int kvh, int ring, int ct) {
    __shared__ __align__(16) uint16_t sh[16 * SROW];
    const int seg = 0, rep = heads / kvh, nq = rep * a.cnt[seg], stride = a.stride[seg], ntiles = stride / 16;
    const int kh = blockIdx.y, warp = threadIdx.x >> 5, group = blockIdx.z * 4 + (warp >> 1), dh = (warp & 1) * 256;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int query = group * 8 + g;
    const bool has = query < nq, any = group * 8 < nq;
    const uint16_t* pq = probs + a.off[seg] + (size_t(kh) * nq + (has ? query : 0)) * stride + 2 * t;
    const int lpos = (lane & 7) + ((lane >> 4) & 1) * 8, ldim = ((lane >> 3) & 1) * 8;
    float c[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0;
    const int t0 = blockIdx.x * ct, t1 = min(ntiles, t0 + ct);
    for (int tile = t0; tile < t1; ++tile) {
        const size_t slot = slot_of(tiles, ring, a.seq[seg], a.base[seg] + 16 * tile);
        const int r = 16 * tile;
        const uint32_t pb0 = has ? *(const uint32_t*)(pq + r) : 0, pb1 = has ? *(const uint32_t*)(pq + r + 8) : 0;
        __syncthreads();
        decode_tile<false>(sh, values + slot * kvh * HD, kvh, kh, 0, nullptr, nullptr, 0);
        __syncthreads();
        if (!any) continue;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint32_t q[4];
            ldsm4t(q, sh + lpos * SROW + dh + 16 * i + ldim);
            mma(c[i], q[0], q[1], q[2], q[3], pb0, pb1);
        }
    }
    if (!any) return;
    float* o = part + ((size_t(blockIdx.x) * kvh + kh) * nq) * HD;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        for (int k = 0; k < 4; ++k) {
            const int q = group * 8 + 2 * t + (k & 1), d = dh + 16 * i + g + (k >= 2 ? 8 : 0);
            if (q < nq) o[size_t(q) * HD + d] = c[i][k];
        }
}

// the chunks' sums in order -> out[row][head][512]
__global__ void attend_add_k(const float* part, float* out, Att a, int heads, int kvh, int chunks) {
    const int rep = heads / kvh, nq = rep * a.cnt[0];
    const size_t n = size_t(kvh) * nq * HD;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = part[i];
    for (int k = 1; k < chunks; ++k) s += part[size_t(k) * n + i];
    const int d = i % HD, q = (i / HD) % nq, kh = i / HD / nq;
    out[(size_t(a.row0[0] + q / rep) * heads + kh * rep + q % rep) * HD + d] = s * (1.0f / BOOST);
}

// S3: s8 scores. q8[row][head][512] int8, qs[row][head] its scale. Block (chunk of CT tiles, key head, eight groups),
// a warp a group with its queries in registers (32 words), the keys straight from memory as int8 words.
__device__ __forceinline__ void mma8(int c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

__global__ void __launch_bounds__(256) scores8_k(const int8_t* q8, const float* qs, const int8_t* keys, const float* kscale, float* scores,
                                                 const int32_t* tiles, Att a, int heads, int kvh, int ring, int ct) {
    const int seg = 0, rep = heads / kvh, nq = rep * a.cnt[seg], stride = a.stride[seg], ntiles = stride / 16;
    const int kh = blockIdx.y, warp = threadIdx.x >> 5, group = blockIdx.z * 8 + warp;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    if (group * 8 >= nq) return;
    const int query = group * 8 + g;
    const bool has = query < nq;
    const int8_t* qa = q8 + (size_t(a.row0[seg] + (has ? query / rep : 0)) * heads + kh * rep + (has ? query % rep : 0)) * HD + 4 * t;
    uint32_t b0[16], b1[16];
#pragma unroll
    for (int u = 0; u < 16; ++u) b0[u] = has ? *(const uint32_t*)(qa + 32 * u) : 0, b1[u] = has ? *(const uint32_t*)(qa + 32 * u + 16) : 0;
    float qsc[2];
    for (int i = 0; i < 2; ++i) {
        const int q = group * 8 + 2 * t + i;
        qsc[i] = q < nq ? qs[size_t(a.row0[seg] + q / rep) * heads + kh * rep + q % rep] : 0;
    }
    float* out = scores + a.off[seg];
    const int t0 = blockIdx.x * ct, t1 = min(ntiles, t0 + ct);
    for (int tile = t0; tile < t1; ++tile) {
        const size_t slot = slot_of(tiles, ring, a.seq[seg], a.base[seg] + 16 * tile);
        const int8_t* ka = keys + ((slot + g) * kvh + kh) * HD + 4 * t;
        const int8_t* kb = keys + ((slot + g + 8) * kvh + kh) * HD + 4 * t;
        uint32_t w[16][4];
#pragma unroll
        for (int u = 0; u < 16; ++u)
            w[u][0] = *(const uint32_t*)(ka + 32 * u), w[u][1] = *(const uint32_t*)(kb + 32 * u), w[u][2] = *(const uint32_t*)(ka + 32 * u + 16),
            w[u][3] = *(const uint32_t*)(kb + 32 * u + 16);
        int c[4] = {0, 0, 0, 0};
#pragma unroll
        for (int u = 0; u < 16; ++u) mma8(c, w[u][0], w[u][1], w[u][2], w[u][3], b0[u], b1[u]);
        const float sa = kscale[(slot + g) * kvh + kh], sb = kscale[(slot + g + 8) * kvh + kh];
        for (int i = 0; i < 4; ++i) {
            const int q = group * 8 + 2 * t + (i & 1), r = 16 * tile + g + (i >= 2 ? 8 : 0);
            if (q < nq) out[(size_t(kh) * nq + q) * stride + r] = float(c[i]) * (i >= 2 ? sb : sa) * qsc[i & 1];
        }
    }
}

// queries fp16 -> int8 with a scale per (row, head): block (head, row), 512 threads
__global__ void quant_q_k(const uint16_t* qhat, int8_t* q8, float* qs) {
    const size_t at = (size_t(blockIdx.y) * gridDim.x + blockIdx.x) * HD + threadIdx.x;
    const float v = __half2float(__ushort_as_half(qhat[at]));
    const float most = block_max(fabsf(v)), scale = most > 0 ? most / 127.0f : 1.0f;
    q8[at] = int8_t(rintf(v / scale));
    if (threadIdx.x == 0) qs[size_t(blockIdx.y) * gridDim.x + blockIdx.x] = scale;
}

// ---------------------------------------------------------------------------------------------- host
template <class F>
static float timed(int reps, F f) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    f();
    CK(cudaDeviceSynchronize());
    cudaEventRecord(e0);
    for (int i = 0; i < reps; ++i) f();
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    CK(cudaGetLastError());
    return ms / reps;
}

template <class T>
static std::vector<T> fetch(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

int main(int argc, char** argv) {
    const int P = argc > 1 ? atoi(argv[1]) : 65536, reps = argc > 2 ? atoi(argv[2]) : 5;
    const int kvh = 4, heads = 32, rep = heads / kvh, turned = 64, ntiles = P / 16, ring = ntiles;
    std::mt19937 rng(1);
    std::vector<int8_t> hk(size_t(P) * kvh * HD), hv(size_t(P) * kvh * HD);
    {
        std::normal_distribution<float> nd(0, 40);
        for (auto& x : hk) x = int8_t(std::max(-127.f, std::min(127.f, rintf(nd(rng)))));
        for (auto& x : hv) x = int8_t(std::max(-127.f, std::min(127.f, rintf(nd(rng)))));
    }
    std::vector<float> hks(size_t(P) * kvh), hvs(size_t(P) * kvh), hkn(HD);
    {
        std::uniform_real_distribution<float> ud(0.5f / 127, 1.5f / 127);
        for (auto& x : hks) x = ud(rng);
        for (auto& x : hvs) x = ud(rng);
        std::uniform_real_distribution<float> un(0.7f, 1.3f);
        for (auto& x : hkn) x = un(rng);
    }
    std::vector<int32_t> htiles(ntiles);
    for (int i = 0; i < ntiles; ++i) htiles[i] = i;
    std::vector<uint16_t> htrig(size_t(P) * turned * 2);
    for (int p = 0; p < P; ++p)
        for (int e = 0; e < turned; ++e) {
            const float ang = float(p) * powf(1e6f, -float(e) / 256.0f);
            htrig[(size_t(p) * turned + e) * 2] = __half_as_ushort(__float2half(cosf(ang)));
            htrig[(size_t(p) * turned + e) * 2 + 1] = __half_as_ushort(__float2half(sinf(ang)));
        }
    int8_t *dk, *dv, *dq8;
    float *dks, *dvs, *dkn, *dqs, *dsc, *dsc2, *dout, *dout2, *dpart;
    int32_t* dtiles;
    uint16_t *dtrig, *dq, *dpr;
    const int RMAX = 32;
    const size_t nsc = size_t(kvh) * rep * RMAX * P;
    CK(cudaMalloc(&dk, hk.size())); CK(cudaMalloc(&dv, hv.size()));
    CK(cudaMalloc(&dks, hks.size() * 4)); CK(cudaMalloc(&dvs, hvs.size() * 4)); CK(cudaMalloc(&dkn, HD * 4));
    CK(cudaMalloc(&dtiles, ntiles * 4)); CK(cudaMalloc(&dtrig, htrig.size() * 2));
    CK(cudaMalloc(&dq, size_t(RMAX) * heads * HD * 2)); CK(cudaMalloc(&dq8, size_t(RMAX) * heads * HD)); CK(cudaMalloc(&dqs, RMAX * heads * 4));
    CK(cudaMalloc(&dsc, nsc * 4)); CK(cudaMalloc(&dsc2, nsc * 4)); CK(cudaMalloc(&dpr, nsc * 2));
    CK(cudaMalloc(&dout, size_t(RMAX) * heads * HD * 4)); CK(cudaMalloc(&dout2, size_t(RMAX) * heads * HD * 4));
    const int MAXCH = 256;
    CK(cudaMalloc(&dpart, size_t(MAXCH) * kvh * rep * RMAX * HD * 4));
    CK(cudaMemcpy(dk, hk.data(), hk.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv, hv.data(), hv.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dks, hks.data(), hks.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dvs, hvs.data(), hvs.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkn, hkn.data(), HD * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dtiles, htiles.data(), ntiles * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dtrig, htrig.data(), htrig.size() * 2, cudaMemcpyHostToDevice));
    {  // queries: normalized-ish fp16 (values ~ N(0, 1))
        std::vector<uint16_t> hq(size_t(RMAX) * heads * HD);
        std::normal_distribution<float> nd(0, 1);
        for (auto& x : hq) x = __half_as_ushort(__float2half(nd(rng) * 0.08f));
        CK(cudaMemcpy(dq, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
    }
    printf("P = %d positions, one full layer (4 key heads x 512, 32 query heads); ms per layer, then x10 layers per 1,000 positions\n", P);
    const double per = 10.0 * 1000.0 / P;
    for (int R : {8, 16, 32}) {
        Att a{};
        a.n = 1, a.most = R, a.row0[0] = 0, a.cnt[0] = R, a.pos0[0] = P - R, a.seq[0] = 0, a.base[0] = 0, a.stride[0] = P, a.tile0[0] = 0, a.off[0] = 0;
        const int nq = rep * R, groups = (nq + 7) / 8, perb = groups > 8 ? 8 : groups, zb = (groups + perb - 1) / perb;
        const size_t ns = size_t(kvh) * nq * P, no = size_t(R) * heads * HD;
        const dim3 sgrid(ntiles, kvh, zb);
        auto S0 = [&] { scores32_k<false><<<sgrid, 32 * perb>>>(dq, dk, dks, dsc, dtiles, a, heads, kvh, HD, ring); };
        auto S1 = [&] { scores32_k<true><<<sgrid, 32 * perb>>>(dq, dk, dks, dsc2, dtiles, a, heads, kvh, HD, ring); };
        auto M0 = [&] { scores_made32_k<<<sgrid, 32 * perb>>>(dq, dv, dvs, dsc, dtiles, a, heads, kvh, HD, ring, dtrig, dkn, turned); };
        auto X = [&] { softmax_k<<<dim3(rep * R, kvh), 256>>>(dsc, dpr, dvs, dtiles, a, heads, kvh, 0, ring); };
        auto A0 = [&] { attend32_k<32, false><<<dim3(HD / 16 * zb, kvh, 1), 32 * perb>>>(dpr, dv, dout, dtiles, a, heads, kvh, HD, ring); };
        auto A1 = [&] { attend32_k<32, true><<<dim3(HD / 16 * zb, kvh, 1), 32 * perb>>>(dpr, dv, dout2, dtiles, a, heads, kvh, HD, ring); };
        auto A3 = [&](int ndt, float* o) {
            const dim3 gr(HD / (16 * ndt) * zb, kvh, 1);
            if (ndt == 1) attend3_k<16, 1><<<gr, 32 * perb>>>(dpr, dv, o, dtiles, a, heads, kvh, HD, ring);
            if (ndt == 2) attend3_k<16, 2><<<gr, 32 * perb>>>(dpr, dv, o, dtiles, a, heads, kvh, HD, ring);
            if (ndt == 4) attend3_k<8, 4><<<gr, 32 * perb>>>(dpr, dv, o, dtiles, a, heads, kvh, HD, ring);
        };
        const int zs = (groups + 7) / 8;
        auto S2 = [&](int ct, float* o) {
            scores_once_k<false><<<dim3((ntiles + ct - 1) / ct, kvh, zs), 256>>>(dq, dk, dks, o, dtiles, a, heads, kvh, ring, ct, nullptr, nullptr, 0);
        };
        auto M2 = [&](int ct, float* o) {
            scores_once_k<true><<<dim3((ntiles + ct - 1) / ct, kvh, zs), 256>>>(dq, dv, dvs, o, dtiles, a, heads, kvh, ring, ct, dtrig, dkn, turned);
        };
        const int za = (groups + 3) / 4;
        auto A2 = [&](int ct, float* o) {
            const int ch = (ntiles + ct - 1) / ct;
            attend_once_k<<<dim3(ch, kvh, za), 256>>>(dpr, dv, dpart, dtiles, a, heads, kvh, ring, ct);
            attend_add_k<<<unsigned((size_t(kvh) * nq * HD + 255) / 256), 256>>>(dpart, o, a, heads, kvh, ch);
        };
        auto S3 = [&](int ct) {
            quant_q_k<<<dim3(heads, R), HD>>>(dq, dq8, dqs);
            scores8_k<<<dim3((ntiles + ct - 1) / ct, kvh, zs), 256>>>(dq8, dqs, dk, dks, dsc2, dtiles, a, heads, kvh, ring, ct);
        };
        printf("\n%d rows (%d query groups a key head)\n", R, groups);
        // correctness: bits of S1, S2 against S0; M2 against M0; A1, A2(one chunk) against A0; S3's error
        auto same = [&](const float* x, const float* y, size_t n) {
            auto u = fetch(x, n), v = fetch(y, n);
            size_t diff = 0;
            for (size_t i = 0; i < n; ++i) diff += memcmp(&u[i], &v[i], 4) != 0;
            return diff;
        };
        S0(); CK(cudaDeviceSynchronize());
        S1(); CK(cudaDeviceSynchronize());
        printf("  bits: S1 vs S0 %zu differ of %zu\n", same(dsc, dsc2, ns), ns);
        S2(16, dsc2); CK(cudaDeviceSynchronize());
        printf("  bits: S2 vs S0 %zu differ\n", same(dsc, dsc2, ns));
        S3(16); CK(cudaDeviceSynchronize());
        {
            auto u = fetch(dsc, ns), v = fetch(dsc2, ns);
            double e2 = 0, s2 = 0;
            for (size_t i = 0; i < ns; ++i) e2 += double(u[i] - v[i]) * (u[i] - v[i]), s2 += double(u[i]) * u[i];
            printf("  S3 (s8) vs S0: relative error of the scores %.2e (rms %.3g, scores rms %.3g)\n", sqrt(e2 / s2), sqrt(e2 / ns), sqrt(s2 / ns));
        }
        M0(); CK(cudaDeviceSynchronize());
        M2(16, dsc2); CK(cudaDeviceSynchronize());
        printf("  bits: M2 vs M0 %zu differ\n", same(dsc, dsc2, ns));
        // probs from S0's scores (key mode) for the weighted sums
        S0(); X(); A0(); CK(cudaDeviceSynchronize());
        A1(); CK(cudaDeviceSynchronize());
        printf("  bits: A1 vs A0 %zu differ of %zu\n", same(dout, dout2, no), no);
        for (int ndt : {1, 2, 4}) {
            A3(ndt, dout2);
            CK(cudaDeviceSynchronize());
            printf("  bits: A3 (%d sixteens a warp) vs A0 %zu differ\n", ndt, same(dout, dout2, no));
        }
        A2(ntiles, dout2); CK(cudaDeviceSynchronize());
        printf("  bits: A2 (one chunk) vs A0 %zu differ\n", same(dout, dout2, no));
        A2((ntiles + 15) / 16, dout2); CK(cudaDeviceSynchronize());
        {
            auto u = fetch(dout, no), v = fetch(dout2, no);
            double e2 = 0, s2 = 0;
            size_t diff = 0;
            for (size_t i = 0; i < no; ++i) e2 += double(u[i] - v[i]) * (u[i] - v[i]), s2 += double(u[i]) * u[i], diff += u[i] != v[i];
            printf("  A2 (16 chunks) vs A0: %zu differ, relative %.2e\n", diff, sqrt(e2 / s2));
        }
        const float t_s0 = timed(reps, S0), t_s1 = timed(reps, S1), t_m0 = timed(reps, M0), t_x = timed(reps, X);
        S0(), X();
        const float t_a0 = timed(reps, A0), t_a1 = timed(reps, A1);
        printf("  today:  scores %.3f  made scores %.3f  softmax %.3f  attend %.3f ms   key mode total %.3f  keyless %.3f   (x10/1k: %.3f / %.3f)\n", t_s0,
               t_m0, t_x, t_a0, t_s0 + t_x + t_a0, t_m0 + t_x + t_a0, (t_s0 + t_x + t_a0) * per, (t_m0 + t_x + t_a0) * per);
        printf("  arith:  scores %.3f  attend %.3f\n", t_s1, t_a1);
        float ba3 = 1e9;
        printf("  attend by sixteens a warp:");
        for (int ndt : {1, 2, 4}) {
            const float x = timed(reps, [&] { A3(ndt, dout2); });
            printf(" %d: %.3f", ndt, x);
            ba3 = std::min(ba3, x);
        }
        printf("\n");
        float bs2 = 1e9, bm2 = 1e9, ba2 = 1e9, bs3 = 1e9;
        int cs2 = 0, cm2 = 0, ca2 = 0, cs3 = 0;
        for (int ct : {4, 8, 16, 32, 64}) {
            const float x = timed(reps, [&] { S2(ct, dsc2); });
            if (x < bs2) bs2 = x, cs2 = ct;
            const float y = timed(reps, [&] { M2(ct, dsc2); });
            if (y < bm2) bm2 = y, cm2 = ct;
            const float z = timed(reps, [&] { S3(ct); });
            if (z < bs3) bs3 = z, cs3 = ct;
        }
        printf("  attend once by tiles a chunk:");
        for (int ct : {16, 32, 64, 128, 256}) {
            if ((ntiles + ct - 1) / ct > MAXCH) continue;
            const float x = timed(reps, [&] { A2(ct, dout2); });
            printf(" %d: %.3f", ct, x);
            if (x < ba2) ba2 = x, ca2 = ct;
        }
        printf("\n");
        printf("  once:   scores %.3f (%d tiles a block)  made scores %.3f (%d)  attend %.3f (%d tiles a chunk)\n", bs2, cs2, bm2, cm2, ba2, ca2);
        printf("  s8:     scores %.3f (%d tiles a block, with the query rounding)\n", bs3, cs3);
        printf("  totals x10 layers per 1,000 positions: today key %.3f keyless %.3f | same bits (S2+X+A1) key %.3f keyless(M2) %.3f | "
               "+A2 key %.3f keyless %.3f | s8 S3+X+A2 %.3f | same bits S2+X+A3 key %.3f keyless %.3f\n",
               (t_s0 + t_x + t_a0) * per, (t_m0 + t_x + t_a0) * per, (bs2 + t_x + std::min(t_a0, t_a1)) * per, (bm2 + t_x + std::min(t_a0, t_a1)) * per,
               (bs2 + t_x + ba2) * per, (bm2 + t_x + ba2) * per, (bs3 + t_x + ba2) * per, (bs2 + t_x + ba3) * per, (bm2 + t_x + ba3) * per);
    }
    return 0;
}
