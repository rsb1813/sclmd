// The engine: one pass of the Gemma 4 decoder over up to 32 new tokens (rows), on the GPU, from pair codes.
// The rows may belong to several sequences (conversations): a pass is made of segments, each a run of rows that are
// consecutive positions of one sequence, and a row attends only to its own sequence.
// A pass costs little more for 8 rows than for one, 1.1 times as much for 16 and 1.6 times for 32 (the codes are
// read and decoded once whatever the rows; scripts/wide_probe.py).
//
// All memory is PyTorch's (the caller allocates it and hands over addresses); the kernels run in its CUDA context on
// the default stream. A pass is one call (sclmd_pass): no Python between the layers.
//
// Matrices are pair codes in a rotated input basis (sclmd/codes.py): the input is rotated by a block Hadamard
// transform with fixed signs, converted to fp16 in the order the tensor-core instruction reads it, and multiplied
// straight from the stored codes (bench/mma_gemm.cu has the layout and the measurements).
#define NOMINMAX
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <utility>

#define API extern "C" __declspec(dllexport)

static const int MAX_TOKENS = 32;  // rows of a pass: up to four groups of the eight columns of the tensor-core instruction
static const int TWO_PHASE = 9;    // rows from which a matrix's input is prepared in two kernels (prep_k)
static const int MAX_SEGS = 32;    // segments of a pass: at most a sequence a row

// The rows of a pass by sequence: segment i is rows row0[i] .. row0[i] + cnt[i] - 1 (the segments follow each other
// and cover every row), positions pos0[i] .. of sequence seq[i].
struct Segs {
    int n;
    int row0[MAX_SEGS], cnt[MAX_SEGS], pos0[MAX_SEGS], seq[MAX_SEGS];
};

// What the attention kernels of a layer are told of each segment besides: the positions its rows read, in whole
// tiles of 16 (from `base`, `stride` positions), where its tiles start among scores_k's blocks, and where its scores
// start in the scores' memory. most: the rows of the largest segment.
struct Att {
    int n, most;
    int row0[MAX_SEGS], cnt[MAX_SEGS], pos0[MAX_SEGS], seq[MAX_SEGS], base[MAX_SEGS], stride[MAX_SEGS], tile0[MAX_SEGS], off[MAX_SEGS];
};

// A matrix as pair codes. codes: the stored order (bench/mma_gemm.cu); lut[code]: the pair as two fp16 in one word;
// scale[row]. bits: 6 or 8 a pair. rows is a multiple of 16, cols of 64.
struct Mat {
    const uint32_t* codes;
    const uint32_t* lut;
    const float* scale;
    int rows, cols, bits, pad;
};

// The rotation of an input of `cols` values: signs[cols] (+1 / -1), then a Hadamard transform over blocks of `block`.
struct Rot {
    const float* signs;
    int block, pad;
};

struct Layer {
    Mat q, k, v, o, gate, up, down;  // v.rows == 0: the value is the key projection (full attention)
    Rot rot_in, rot_o, rot_mlp, rot_down;
    const float *in_ln, *post_attn_ln, *pre_ff_ln, *post_ff_ln;  // [hidden]
    const float *q_norm, *k_norm;                                 // [head_dim]
    const float* inv_freq;                                        // [head_dim / 2], zero past the rotated pairs
    // This layer's keys and values in tiles of 16 positions: [pool tiles][16][kv_heads * head_dim], as int8 with a
    // scale for each head's vector (kscale, vscale: [pool tiles][16][kv_heads]): a vector is its int8 values times
    // its scale, the largest of them 127. A sequence is given tiles from the pool as it grows: tiles[sequence * ring
    // + j] is the pool tile that holds its positions 16j .. 16j + 15, j taken modulo `ring` (the table is shared by
    // the layers of a kind).
    int8_t *keys, *values;
    const int32_t* tiles;
    float *kscale, *vscale;
    float scalar;
    int full;      // 1: attends to every earlier position; 0: to the last `window`
    int head_dim, kv_heads;
    int ring;      // sliding: (window + spare) / 16, so positions go round; full: capacity / 16
    int pad;
};

struct Model {
    Layer* layers;
    int n_layers, hidden, mlp, heads, window, vocab;
    float eps, softcap;
    Mat head;
    Rot rot_head;
    const float* norm;  // the final norm [hidden]
    // work memory; [rows]: the most rows a pass of this model may have (8, 16 or 32: the caller sizes the memory)
    float* h;        // [rows][hidden] the residual stream of the tokens of this pass (in: the embeddings; out: the last layer's output)
    float* h2;       // [rows][hidden] where the residual stream's next step is written (the two take turns)
    float* fps;      // [rows] the power of two an input was divided by to fit fp16
    uint64_t* xf;    // [rows / 8][widest input / 16][32] a matrix's input as the instruction reads it, eight rows a group
    float* y;        // [rows][widest output] a matrix's output
    float* y2;       // [rows][mlp] (up's output beside gate's)
    uint16_t* qhat;  // [rows][heads * widest head] the queries, normalized and rotated (fp16)
    float* scores;   // [kv heads * queries][positions in whole tiles] attention scores, segment after segment
    uint16_t* probs; // the same layout: attention weights (fp16)
    float* attn;     // [rows][heads * widest head] attention's output
    float* hn;       // [rows][hidden] the last output after the final norm (what the drafter is handed)
    float* logits;   // [rows][vocab]
};

__device__ __forceinline__ void mma(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// The segment a row belongs to (the segments are in row order).
__device__ __forceinline__ int seg_of_row(const Att& a, int row) {
    int s = 0;
    for (int i = 1; i < a.n; ++i)
        if (row >= a.row0[i]) s = i;
    return s;
}

// The slot (position in the pool) of a sequence's position `pos`.
__device__ __forceinline__ size_t slot_of(const int32_t* tiles, int ring, int seq, int pos) {
    return size_t(tiles[size_t(seq) * ring + (pos >> 4) % ring]) * 16 + (pos & 15);
}

// An attention weight is kept as fp16 of weight * the value's scale * BOOST (attend_k divides the sum by BOOST): a
// scale is at most sqrt(head_dim) / 127, under 0.2, so the largest is under 800, and a weight of 1e-6 is still an
// ordinary fp16 number (without the boost a weight under 2e-3 would fall among the smallest, which have few bits).
#define BOOST 4096.0f

// int8 values as the fp16 the tensor-core instruction takes: table[b] is the half-precision bits of int8(b). Filled by
// the block that uses it (256 entries in its shared memory).
__device__ __forceinline__ void fill_table(uint16_t* table) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) table[i] = __half_as_ushort(__float2half(float(int8_t(i))));
    __syncthreads();
}

// Two neighbouring int8 values (read as one 16-bit word) as two fp16 in one register.
__device__ __forceinline__ uint32_t two(const uint16_t* table, uint16_t w) {
    return uint32_t(table[w & 255]) | uint32_t(table[w >> 8]) << 16;
}

__device__ __forceinline__ uint32_t two(const uint16_t* table, const int8_t* p) { return two(table, *(const uint16_t*)p); }

// The attention kernels ask for the loads of several steps of their loops before the first is used. A warp waits at
// the first value that has not come from memory, so a step's loads at a time is a wait a step; with the loads of many
// steps on their way together the wait is shared (attend_k at 12,288 positions, a pass of 7 rows: 5.4 ms a pass one
// tile at a time, 3.8 ms by eights, 2.8 ms by sixteens). The loaded values are held in registers, and a block's
// threads share its registers (65,536): by 32 steps a block may have 256 threads, which is eight groups of queries;
// a pass with more groups has more blocks (GROUPS_A_BLOCK). The sums are the same bits however many steps go
// together: the steps are added in the same order.
// The kernels carry no bound on a block's threads: with one, the compiler kept them to 64 registers and put the rest
// on the stack, which was as slow as a step at a time.

// The sum of v over the block's threads (a multiple of 32, at most 1024), returned to every thread.
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

__device__ __forceinline__ float gelu_tanh(float v) {
    return 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

// What prep_k reads of one element of a matrix's input: the stream, the output owed to it, the norm's weight there,
// and the value's own weight or factor.
struct In {
    float a, y, p, x;
};

// use(load(j)) for j = first, first + step, ... below n, in that order, with the loads of STEPS elements asked for
// before the first is used.
template <int STEPS, class Load, class Use>
__device__ __forceinline__ void strided_by(int& j, int n, int step, Load load, Use use) {
    for (; j + (STEPS - 1) * step < n; j += STEPS * step) {
        decltype(load(0)) e[STEPS];
#pragma unroll
        for (int u = 0; u < STEPS; ++u) e[u] = load(j + u * step);
#pragma unroll
        for (int u = 0; u < STEPS; ++u) use(e[u]);
    }
}

// MOST at a time (what a thread can hold: a block of 1,024 threads has 64 registers a thread), then by fours and ones.
template <int MOST, class Load, class Use>
__device__ __forceinline__ void strided(int first, int n, int step, Load load, Use use) {
    int j = first;
    strided_by<MOST>(j, n, step, load, use);
    if (MOST > 4) strided_by<4>(j, n, step, load, use);
    strided_by<1>(j, n, step, load, use);
}

// The input of a matrix with n columns, in one kernel: block (run of blockDim values, token), where blockDim is the
// rotation's block, or without a rotation (signs null) a power of two that divides n. Three steps:
//  1. the residual step still owed, if any: with y, h = (a + RMS norm of y times post) * scalar, and the block
//     writes its own run of h to h_out (which is not a). Without y, h = a.
//  2. the value: mode 0: RMS norm of h times w; 1: h as it is; 2: gelu(h) * b. Written to v_out if given.
//  3. signs and the Hadamard transform over the run (with a rotation), over the power of two that brings the whole
//     input's length under 2^14, so that no value leaves fp16's range (fps[token]); then fp16 in the order the
//     instruction reads: for tile j, lane 4 * (token % 8) + t of the token's group of eight in xf holds the values
//     at columns 16j + 2t, + 1, + 8, + 9 (a group is the n / 16 tiles of the input).
// The sums over a token's whole input (the two norms and the length) are made by every block for itself when the
// pass is narrow (phase 0: a kernel less is worth more than the sums cost). In a wide pass they are most of the
// kernel's work, n / blockDim times over, so one block a token makes them first (phase 1, into sums[2 * token] and
// fps[token]) and the blocks read them (phase 2). The three phases are this one kernel: the same instructions make
// the sums and the values in each, so a row's result has the same bits whichever way its pass went.
__global__ void __launch_bounds__(1024) prep_k(const float* a, const float* b, const float* w, const float* y, const float* post, float scalar, float* h_out,
                       float* v_out, const float* signs, float* fps, uint64_t* xf, int n, float eps, int mode, int phase, float* sums) {
    __shared__ float buf[1024];
    const int bd = blockDim.x, token = blockIdx.y, i = threadIdx.x, col = blockIdx.x * bd + i;
    const size_t at = size_t(token) * n;
    float ry = 0, r = 1.0f, fp = 1.0f;
#define STREAM_OF(av, yv, pv) (y ? ((av) + (yv) * ry * (pv)) * scalar : (av))
#define VALUE_OF(hv, wv, bv) (mode == 0 ? (hv) * r * (wv) : mode == 1 ? (hv) : gelu_tanh(hv) * (bv))
#define STREAM(j) STREAM_OF(a[at + (j)], y[at + (j)], post[j])
#define VALUE(j, hv) VALUE_OF(hv, w[j], b[at + (j)])
    if (phase == 2) {
        ry = sums[2 * token], r = sums[2 * token + 1], fp = fps[token];
    } else {
        // a thread's elements are a block apart: each a load that waits for memory, so they are asked for several at
        // a time (strided) and added in the same order as one by one
        if (y) {
            float ss = 0;
            strided<8>(i, n, bd, [&](int j) { return y[at + j]; }, [&](float e) { ss += e * e; });
            ry = powf(block_sum(ss) / n + eps, -0.5f);
        }
        const auto stream = [&](int j) { return In{a[at + j], y ? y[at + j] : 0.0f, y ? post[j] : 0.0f, mode == 0 ? w[j] : mode == 2 ? b[at + j] : 0.0f}; };
        if (mode == 0) {
            float ss = 0;
            strided<4>(i, n, bd, stream, [&](const In& e) {
                const float hv = STREAM_OF(e.a, e.y, e.p);
                ss += hv * hv;
            });
            r = powf(block_sum(ss) / n + eps, -0.5f);
        }
        float len = 0;
        strided<4>(i, n, bd, stream, [&](const In& e) {
            const float hv = STREAM_OF(e.a, e.y, e.p), v = VALUE_OF(hv, e.x, e.x);
            len += v * v;
        });
        len = sqrtf(block_sum(len));
        fp = len > 16384.0f ? exp2f(ceilf(log2f(len / 16384.0f))) : 1.0f;
        if (col == 0) fps[token] = fp;
        if (phase == 1) {
            if (i == 0) sums[2 * token] = ry, sums[2 * token + 1] = r;
            return;
        }
    }
    const float hv = STREAM(col), v = VALUE(col, hv);
#undef STREAM
#undef VALUE
#undef STREAM_OF
#undef VALUE_OF
    if (y) h_out[at + col] = hv;
    if (v_out) v_out[at + col] = v;
    buf[i] = signs ? v * signs[col] : v;
    __syncthreads();
    if (signs)
        for (int h = 1; h < bd; h <<= 1) {
            const float lo = buf[i & ~h], hi = buf[i | h];
            __syncthreads();
            buf[i] = (i & h) ? lo - hi : lo + hi;
            __syncthreads();
        }
    const float k = (signs ? rsqrtf(float(bd)) : 1.0f) / fp;
    if ((i & 9) == 0) {  // i % 16 in {0, 2, 4, 6}: the first column of a lane's word
        const int t = (i & 15) >> 1;
        const uint64_t word = uint64_t(__half_as_ushort(__float2half(buf[i] * k))) | uint64_t(__half_as_ushort(__float2half(buf[i + 1] * k))) << 16 |
                              uint64_t(__half_as_ushort(__float2half(buf[i + 8] * k))) << 32 |
                              uint64_t(__half_as_ushort(__float2half(buf[i + 9] * k))) << 48;
        xf[(size_t(token >> 3) * (n >> 4) + (col >> 4)) * 32 + 4 * (token & 7) + t] = word;
    }
}

// y[token][row] = scale[row] * fps[token] * sum over columns of W[row][col] * x[token][col]. A strip is 16 rows; its
// `segs` warps split the strip's tiles (a multiple of 4 each) between them. A block is `strips` strips that follow
// each other (up to GEMM_STRIPS): their warps walk the same tiles of the input, so what one reads of it the others
// find in the block's cache. Up to three matrices that share the input go in one launch: the strips are the first
// one's, then the second's, then the third's.
// G: the groups of eight tokens. A tile's codes are read and decoded once and multiplied with each group's inputs.
#define GEMM_STRIPS 4
template <int G>
__global__ void gemm_k(Mat m0, Mat m1, Mat m2, float* y0, float* y1, float* y2, const uint64_t* xf, const float* fps, int segs, int tokens,
                       int strips) {
    __shared__ uint32_t luts[GEMM_STRIPS][256];
    __shared__ float part[GEMM_STRIPS][8][32][4];
    const int local = (threadIdx.x >> 5) / segs, warp = (threadIdx.x >> 5) % segs, lane = threadIdx.x & 31;  // the block's strip, and its warp
    int strip = blockIdx.x * strips + local;
    Mat m = m0;
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

// What attention does to the projections' outputs, block (vector, token) with head_dim threads. Vector v of a token is
// one of its `heads` query heads (v < heads), `kvh` key heads or `kvh` value heads:
//   query: RMS norm times q_norm, rotary position -> qhat[token][head][head_dim] (fp16)
//   key:   RMS norm times k_norm, rotary position -> keys[slot of the position][head] (int8), kscale[slot][head]
//   value: RMS norm                               -> values[slot][head] (int8), vscale[slot][head]; its source is yv (or yk when null)
// The rotary position turns pair (d, d + head_dim / 2) by pos * inv_freq[d]. The token's position and sequence are
// its segment's (a.pos0: the position of the segment's first row). A key or value is kept as int8: the vector divided
// by its scale (its largest size over 127) and rounded.
__global__ void heads_k(const float* yq, const float* yk, const float* yv, const float* q_norm, const float* k_norm, const float* inv_freq,
                        uint16_t* qhat, int8_t* keys, int8_t* values, float* kscale, float* vscale, const int32_t* tiles, Att a, int heads,
                        int kvh, int ring, float eps) {
    const int hd = blockDim.x, v = blockIdx.x, token = blockIdx.y, d = threadIdx.x, half = hd / 2;
    const int kind = v < heads ? 0 : v < heads + kvh ? 1 : 2, head = kind == 0 ? v : kind == 1 ? v - heads : v - heads - kvh;
    const float* src = kind == 0 ? yq + (size_t(token) * heads + head) * hd
                                 : (kind == 1 || !yv ? yk : yv) + (size_t(token) * kvh + head) * hd;
    __shared__ float buf[1024];
    const float val = src[d];
    const float r = powf(block_sum(val * val) / hd + eps, -0.5f);
    const int seg = seg_of_row(a, token), pos = a.pos0[seg] + token - a.row0[seg];
    const size_t slot = kind == 0 ? 0 : slot_of(tiles, ring, a.seq[seg], pos);
    if (kind == 2) {
        const float v = val * r, most = block_max(fabsf(v)), scale = most > 0 ? most / 127.0f : 1.0f;
        values[(slot * kvh + head) * hd + d] = int8_t(rintf(v / scale));
        if (d == 0) vscale[slot * kvh + head] = scale;
        return;
    }
    buf[d] = val * r * (kind == 0 ? q_norm[d] : k_norm[d]);
    __syncthreads();
    const float ang = float(pos) * inv_freq[d % half];
    const float turned = d < half ? -buf[d + half] : buf[d - half];
    const float out = buf[d] * cosf(ang) + turned * sinf(ang);
    if (kind == 0) {
        qhat[(size_t(token) * heads + head) * hd + d] = __half_as_ushort(__float2half(out));
        return;
    }
    const float most = block_max(fabsf(out)), scale = most > 0 ? most / 127.0f : 1.0f;
    keys[(slot * kvh + head) * hd + d] = int8_t(rintf(out / scale));
    if (d == 0) kscale[slot * kvh + head] = scale;
}

// Attention runs on the tensor cores too, in three kernels. The positions a segment's rows read are taken in tiles of
// 16 that start at multiples of 16 (`base` is the first tile's position, `stride` the number of positions in the
// tiles), so a query's sums are made of the same tiles in the same order whichever pass it is computed in (alone,
// among eight, or beside other sequences). Positions in the tiles that a query may not read, or that hold nothing
// yet, get weight zero.
// A key head's queries are numbered, within a segment, query = row * rep + (head within the key head's `rep` query
// heads); eight of them are the eight columns of the instruction, and each group of eight has a warp.
// A segment's scores: scores[off + (key head * nq + query) * stride + (position - base)], nq its queries a key head.

// Scores = query . key: block (position tile of a segment, key head). The keys are int8: the sums are of the int8
// values (as fp16), and a position's sum is multiplied by its key's scale.
// STEPS steps of a query group's sum over a head's values, from value j on, while as many are left: the loads of all
// of them, then the products in order.
template <int STEPS>
__device__ __forceinline__ void scores_steps(float c[4], int& j, int hd, const uint16_t* table, const int8_t* ka, const int8_t* kb,
                                             const uint16_t* qa, bool has) {
    for (; j + 16 * STEPS <= hd; j += 16 * STEPS) {
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
        for (int u = 0; u < STEPS; ++u) mma(c, two(table, w[u][0]), two(table, w[u][1]), two(table, w[u][2]), two(table, w[u][3]), b0[u], b1[u]);
    }
}

template <int AHEAD>
__device__ __forceinline__ void scores_body(const uint16_t* qhat, const int8_t* keys, const float* kscale, float* scores, const int32_t* tiles,
                                            const Att& a, int heads, int kvh, int hd, int ring) {
    __shared__ uint16_t table[256];
    fill_table(table);
    int seg = 0;
    for (int i = 1; i < a.n; ++i)
        if (int(blockIdx.x) >= a.tile0[i]) seg = i;
    const int rep = heads / kvh, nq = rep * a.cnt[seg], stride = a.stride[seg];
    // a block's warps are groups of queries; with more groups than a block takes, blockIdx.z counts the blocks before
    const int tile = blockIdx.x - a.tile0[seg], kh = blockIdx.y, group = blockIdx.z * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    if (group * 8 >= nq) return;
    // the left operand: rows are the tile's positions (this lane: g and g + 8), columns sixteen of the head's values
    const size_t slot = slot_of(tiles, ring, a.seq[seg], a.base[seg] + 16 * tile);
    const int8_t* ka = keys + ((slot + g) * kvh + kh) * hd + 2 * t;
    const int8_t* kb = keys + ((slot + g + 8) * kvh + kh) * hd + 2 * t;
    // the right operand: column g is query group * 8 + g
    const int query = group * 8 + g;
    const bool has = query < nq;
    const uint16_t* qa = qhat + (size_t(a.row0[seg] + (has ? query / rep : 0)) * heads + kh * rep + (has ? query % rep : 0)) * hd + 2 * t;
    float c[4] = {0, 0, 0, 0};
    int j = 0;
    scores_steps<AHEAD>(c, j, hd, table, ka, kb, qa, has);
    if (AHEAD > 8) scores_steps<8>(c, j, hd, table, ka, kb, qa, has);
    if (AHEAD > 2) scores_steps<2>(c, j, hd, table, ka, kb, qa, has);
    scores_steps<1>(c, j, hd, table, ka, kb, qa, has);
    float* out = scores + a.off[seg];
    const float sa = kscale[(slot + g) * kvh + kh], sb = kscale[(slot + g + 8) * kvh + kh];
    for (int i = 0; i < 4; ++i) {  // c[i]: position of the tile g (+ 8), query group * 8 + 2t (+ 1)
        const int q = group * 8 + 2 * t + (i & 1), r = 16 * tile + g + (i >= 2 ? 8 : 0);
        if (q < nq) out[(size_t(kh) * nq + q) * stride + r] = c[i] * (i >= 2 ? sb : sa);
    }
}

#define SCORES_K(NAME, AHEAD, BOUND)                                                                                                    \
    __global__ void BOUND NAME(const uint16_t* qhat, const int8_t* keys, const float* kscale, float* scores, const int32_t* tiles, Att a, \
                               int heads, int kvh, int hd, int ring) {                                                                  \
        scores_body<AHEAD>(qhat, keys, kscale, scores, tiles, a, heads, kvh, hd, ring);                                                 \
    }
SCORES_K(scores32_k, 32, )
#undef SCORES_K

// The softmax of a query's scores over the positions it may read, as fp16 in the same layout: block (query of the
// pass, key head). A row at position pos reads positions up to its own, and with a window the last `window` of them.
// What is written is the weight times the scale of the position's value (the values are int8: attend_k sums those).
__global__ void softmax_k(const float* scores, uint16_t* probs, const float* vscale, const int32_t* tiles, Att a, int heads, int kvh,
                          int window, int ring) {
    const int n = blockDim.x, rep = heads / kvh, kh = blockIdx.y, row = blockIdx.x / rep, seg = seg_of_row(a, row);
    const int nq = rep * a.cnt[seg], query = blockIdx.x - rep * a.row0[seg], pos = a.pos0[seg] + row - a.row0[seg];
    const int base = a.base[seg], stride = a.stride[seg];
    const size_t at = a.off[seg] + (size_t(kh) * nq + query) * stride;
    const float* s = scores + at;
    uint16_t* p = probs + at;
    const int first = (window > 0 && pos - window + 1 > 0 ? pos - window + 1 : 0) - base, last = pos - base;  // inclusive, as indices
    // a thread takes the positions that are its own number modulo the block, so the sums add up in the same order in any pass
    const int mine = first + ((int(threadIdx.x) - (base + first) % n + n) % n);
    float mx = -INFINITY;
    for (int r = mine; r <= last; r += n) mx = fmaxf(mx, s[r]);
    mx = block_max(mx);
    float sum = 0;
    for (int r = mine; r <= last; r += n) sum += expf(s[r] - mx);
    sum = block_sum(sum);
    const int32_t* own = tiles + size_t(a.seq[seg]) * ring;  // the sequence's table (see attend_k)
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

// The weighted sum of the values: block (sixteen of the head's values, key head, segment); a warp walks the position
// tiles. -> out[row][head][head_dim]. The values are int8 (as fp16 through the table); their scales are in the weights.
// A warp's time is its own chain of steps (the tiles are added in order): a block that took several sixteens, so that
// its warps shared what they read, was no faster (measured).
// What attend_k's steps are told: where the sequence's tiles are, and a lane's place in a tile's values and weights.
struct Walk {
    const int32_t* mine;    // the sequence's table of tiles
    const uint8_t* values;
    const uint16_t* pq;     // the lane's weights of the first tile
    size_t tile_values, in_tile, row;
    int j0, ring;
    bool has;
};

// TILES position tiles of a query group's weighted sum, from position r on, while as many are left: the loads of all
// of them (first where the tiles are, then their values and weights), then the products in order.
template <int TILES>
__device__ __forceinline__ void attend_tiles(float c[4], int& r, int stride, const uint16_t* table, const Walk& k) {
#define PAIR(x, y) (uint32_t(table[x]) | uint32_t(table[y]) << 16)
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
            const uint8_t* va = k.values + size_t(where[u]) * k.tile_values + k.in_tile;  // position 2t of the tile
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

template <int AHEAD>
__device__ __forceinline__ void attend_body(const uint16_t* probs, const int8_t* values, float* out, const int32_t* tiles, const Att& a,
                                            int heads, int kvh, int hd, int ring) {
    __shared__ uint16_t table[256];
    fill_table(table);
    const int seg = blockIdx.z, rep = heads / kvh, nq = rep * a.cnt[seg], stride = a.stride[seg], base = a.base[seg];
    // a block's warps are groups of queries; with more groups than a block takes, the blocks of a sixteen follow each other
    const int dt = blockIdx.x % (hd / 16), group = (blockIdx.x / (hd / 16)) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int kh = blockIdx.y, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    if (group * 8 >= nq) return;
    // the right operand: rows are the tile's positions, column g is query group * 8 + g
    const int query = group * 8 + g;
    const bool has = query < nq;
    // the left operand: columns are the tile's positions 2t, 2t + 1, 2t + 8, 2t + 9; its rows g and g + 8 are the
    // values 16 dt + 2g and 16 dt + 2g + 1 of the head: neighbours in memory, so a position is one 16-bit load (which
    // row of the instruction a value sits in does not change its sum)
    const size_t col = size_t(kh) * hd + 16 * dt + 2 * g, row = size_t(kvh) * hd;
    // the sequence's table; a pass reads at most ring + 1 tiles, so the table's index goes round at most once
    const Walk k{tiles + size_t(a.seq[seg]) * ring, (const uint8_t*)values, probs + a.off[seg] + (size_t(kh) * nq + (has ? query : 0)) * stride + 2 * t,
                 16 * row, 2 * t * row + col, row, (base >> 4) % ring, ring, has};
    float c[4] = {0, 0, 0, 0};
    int r = 0;
    attend_tiles<AHEAD>(c, r, stride, table, k);
    if (AHEAD > 8) attend_tiles<8>(c, r, stride, table, k);
    if (AHEAD > 2) attend_tiles<2>(c, r, stride, table, k);
    attend_tiles<1>(c, r, stride, table, k);
    for (int i = 0; i < 4; ++i) {  // c[i]: value 16 dt + 2g (+ 1), query group * 8 + 2t (+ 1)
        const int q = group * 8 + 2 * t + (i & 1), d = 16 * dt + 2 * g + (i >= 2 ? 1 : 0);
        if (q < nq) out[(size_t(a.row0[seg] + q / rep) * heads + kh * rep + q % rep) * hd + d] = c[i] * (1.0f / BOOST);
    }
}

#define ATTEND_K(NAME, AHEAD, BOUND)                                                                                                    \
    __global__ void BOUND NAME(const uint16_t* probs, const int8_t* values, float* out, const int32_t* tiles, Att a, int heads, int kvh, \
                               int hd, int ring) {                                                                                      \
        attend_body<AHEAD>(probs, values, out, tiles, a, heads, kvh, hd, ring);                                                         \
    }
ATTEND_K(attend32_k, 32, )
#undef ATTEND_K

// The soft cap on the logits: tanh(v / cap) * cap.
__global__ void softcap_k(float* logits, size_t n, float cap) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) logits[i] = tanhf(logits[i] / cap) * cap;
}

static int segs_for(int cols) {
    const int tiles = cols / 16;
    for (int s = 8; s > 1; s >>= 1)
        if (tiles % (4 * s) == 0 && tiles / s >= 32) return s;
    return 1;
}

// One launch for a matrix, or for up to three that share the input in xf (b and c may have no rows).
static void gemm(const uint64_t* xf, const float* fps, int tokens, const Mat& a, float* ya, const Mat& b = Mat{}, float* yb = nullptr,
                 const Mat& c = Mat{}, float* yc = nullptr) {
    const int segs = segs_for(a.cols), all = (a.rows + b.rows + c.rows) / 16;
    // as many strips a block as its threads allow, up to four. Measured on the real model (a pass at 64 positions):
    // 32 rows 36.3 ms with a strip a block, 31.4 ms with four; 16 rows 25.2 -> 22.6 ms; 8 rows the same (20.3 ms)
    const int strips = GEMM_STRIPS * segs * 32 <= 1024 ? GEMM_STRIPS : 1024 / (segs * 32), blocks = (all + strips - 1) / strips;
    if (tokens <= 8)
        gemm_k<1><<<blocks, 32 * segs * strips>>>(a, b, c, ya, yb, yc, xf, fps, segs, tokens, strips);
    else if (tokens <= 16)
        gemm_k<2><<<blocks, 32 * segs * strips>>>(a, b, c, ya, yb, yc, xf, fps, segs, tokens, strips);
    else  // a pass of 17 to 24 rows multiplies a fourth group that nothing has written and nothing reads the products of
        gemm_k<4><<<blocks, 32 * segs * strips>>>(a, b, c, ya, yb, yc, xf, fps, segs, tokens, strips);
}

// What prep_k is told of the residual step still owed: the matrix output it adds, the norm's weight and the layer's scalar.
struct Owed {
    const float *y = nullptr, *post = nullptr;
    float scalar = 1.0f;
};

// The input of a matrix with n columns from a (and b, w: see prep_k), rotated, in m.xf with its scales in m.fps.
// Rows from which prep_k goes in two phases (SCLMD_TWO_PHASE sets it for measurements; a number past MAX_TOKENS: never).
static int two_phase_from() {
    static const int from = getenv("SCLMD_TWO_PHASE") ? atoi(getenv("SCLMD_TWO_PHASE")) : TWO_PHASE;
    return from;
}

// prep_k for `rows` rows; the sums of the two-phase way go after the rows' scales in fps (which holds 3 * MAX_TOKENS).
static void launch_prep(int rows, const float* a, const float* b, const float* w, int n, const Rot& rot, int mode, const Owed& owed,
                        float* h_out, float* v_out, float* fps, uint64_t* xf, float eps) {
    const dim3 grid(n / rot.block, rows);
    float* sums = fps + MAX_TOKENS;
    if (rows >= two_phase_from() && grid.x > 1) {
        prep_k<<<dim3(1, rows), rot.block>>>(a, b, w, owed.y, owed.post, owed.scalar, h_out, v_out, rot.signs, fps, xf, n, eps, mode, 1, sums);
        prep_k<<<grid, rot.block>>>(a, b, w, owed.y, owed.post, owed.scalar, h_out, v_out, rot.signs, fps, xf, n, eps, mode, 2, sums);
    } else {
        prep_k<<<grid, rot.block>>>(a, b, w, owed.y, owed.post, owed.scalar, h_out, v_out, rot.signs, fps, xf, n, eps, mode, 0, sums);
    }
}

static void prep(const Model& m, const float* a, const float* b, const float* w, int n, const Rot& rot, int mode, int tokens,
                 const Owed& owed = Owed{}, float* h_out = nullptr, float* v_out = nullptr) {
    launch_prep(tokens, a, b, w, n, rot, mode, owed, h_out, v_out, m.fps, m.xf, m.eps);
}

// What a layer's attention kernels are told of a pass's segments. A row reads the positions up to its own (the
// drafter: up to `pos0`, the last the main model has run) and in a sliding layer `back` positions before the
// segment's first.
static Att att_for(const Segs& g, bool full, int heads, int back) {
    Att a{};
    a.n = g.n, a.most = 1;
    int tiles = 0, off = 0;
    for (int i = 0; i < g.n; ++i) {
        const int last = g.pos0[i] + g.cnt[i] - 1, lo = full ? 0 : (g.pos0[i] - back > 0 ? g.pos0[i] - back : 0);
        a.row0[i] = g.row0[i], a.cnt[i] = g.cnt[i], a.pos0[i] = g.pos0[i], a.seq[i] = g.seq[i];
        a.base[i] = lo & ~15, a.stride[i] = ((last >> 4) + 1) * 16 - a.base[i];
        a.tile0[i] = tiles, a.off[i] = off;
        tiles += a.stride[i] / 16, off += heads * g.cnt[i] * a.stride[i];
        if (g.cnt[i] > a.most) a.most = g.cnt[i];
    }
    return a;
}

static int tiles_of(const Att& a) { return a.tile0[a.n - 1] + a.stride[a.n - 1] / 16; }

// Groups of queries a block of the attention kernels takes at most: eight (256 threads), so that the form that asks
// for 32 steps' loads has its registers whatever the rows of the pass; a pass with more groups has more blocks.
static const int GROUPS_A_BLOCK = 8;

// The two attention kernels with a warp for each of `groups` groups of eight queries.
static void scores(int tiles, int kvh, int groups, const uint16_t* qhat, const int8_t* keys, const float* kscale, float* out, const int32_t* table,
                   const Att& a, int heads, int hd, int ring) {
    const int per = groups > GROUPS_A_BLOCK ? GROUPS_A_BLOCK : groups;
    scores32_k<<<dim3(tiles, kvh, (groups + per - 1) / per), 32 * per>>>(qhat, keys, kscale, out, table, a, heads, kvh, hd, ring);
}

static void attend(int groups, const uint16_t* probs, const int8_t* values, float* out, const int32_t* table, const Att& a, int heads, int kvh,
                   int hd, int ring) {
    const int per = groups > GROUPS_A_BLOCK ? GROUPS_A_BLOCK : groups;
    attend32_k<<<dim3(hd / 16 * ((groups + per - 1) / per), kvh, a.n), 32 * per>>>(probs, values, out, table, a, heads, kvh, hd, ring);
}

// Where a pass's time goes (sclmd_profile): with a table set, the pass waits for the GPU after every launch and adds
// the time to the launch's kind. The waits make the pass slower as a whole; the shares are what is looked at.
enum { P_INPUT, P_QKV, P_HEADS, P_SCORES, P_SOFTMAX, P_ATTEND, P_O, P_GATE_UP, P_DOWN, P_HEAD, P_KINDS };
static double* g_prof = nullptr;
static std::chrono::steady_clock::time_point g_prof_t;

static inline void tick(int kind) {
    if (!g_prof) return;
    cudaDeviceSynchronize();
    const auto now = std::chrono::steady_clock::now();
    g_prof[kind] += std::chrono::duration<double>(now - g_prof_t).count();
    g_prof_t = std::chrono::steady_clock::now();
}

// times: P_KINDS seconds to add to, or null to stop.
API int sclmd_profile(double* times) {
    g_prof = times;
    cudaDeviceSynchronize();
    g_prof_t = std::chrono::steady_clock::now();
    return P_KINDS;
}

// One matrix times a given input (tests): x[tokens][cols] -> y[tokens][rows], with the model's work memory.
API int sclmd_linear(const Model* m, const Mat* mat, const Rot* rot, const float* x, float* y, int tokens) {
    prep(*m, x, nullptr, nullptr, mat->cols, *rot, 1, tokens);
    gemm(m->xf, m->fps, tokens, *mat, y);
    return int(cudaGetLastError());
}

// The pass: m.h holds the scaled embeddings of `tokens` tokens, which are the rows of `segs`; afterwards it holds the
// last layer's output, m.hn that output after the final norm, m.logits the logits, and every layer's keys and
// values have the new positions of each sequence (whose tiles the caller has provided). Launches only; nothing waits.
// A layer's two residual steps are not kernels of their own: each is taken by the kernel that prepares the next
// matrix input from the residual stream, which writes the stream's next state to the other of m.h and m.h2.
static int pass(const Model* mp, int tokens, const Segs* segs, int logits_from);

API int sclmd_pass(const Model* mp, int tokens, const Segs* segs) { return pass(mp, tokens, segs, 0); }

// The pass with the logits of the rows from `logits_from` on only (of its group of eight rows on; from `tokens`: of
// none): a prompt's rows need no logits, and the head is 0.66 GiB to read and a multiplication a group of rows.
// m.logits of the rows before is whatever an earlier pass left there.
API int sclmd_pass_from(const Model* mp, int tokens, const Segs* segs, int logits_from) { return pass(mp, tokens, segs, logits_from); }

static int pass(const Model* mp, int tokens, const Segs* segs, int logits_from) {
    const Model& m = *mp;
    const int n = m.hidden;
    float *h = m.h, *spare = m.h2;
    Owed owed;
    for (int i = 0; i < m.n_layers; ++i) {
        const Layer& l = m.layers[i];
        const int hd = l.head_dim, kvh = l.kv_heads, nqo = m.heads * hd, nkv = kvh * hd;
        // attention
        prep(m, h, nullptr, l.in_ln, n, l.rot_in, 0, tokens, owed, spare);
        if (owed.y) std::swap(h, spare);
        tick(P_INPUT);
        float *yq = m.y, *yk = m.y + size_t(tokens) * nqo, *yv = yk + size_t(tokens) * nkv;
        gemm(m.xf, m.fps, tokens, l.q, yq, l.k, yk, l.v, yv);  // a full-attention layer's v has no rows
        tick(P_QKV);
        const Att a = att_for(*segs, l.full, m.heads, m.window - 1);
        const int rep = m.heads / kvh, groups = (rep * a.most + 7) / 8;
        heads_k<<<dim3(m.heads + 2 * kvh, tokens), hd>>>(yq, yk, l.v.rows ? yv : nullptr, l.q_norm, l.k_norm, l.inv_freq, m.qhat, l.keys,
                                                       l.values, l.kscale, l.vscale, l.tiles, a, m.heads, kvh, l.ring, m.eps);
        tick(P_HEADS);
        scores(tiles_of(a), kvh, groups, m.qhat, l.keys, l.kscale, m.scores, l.tiles, a, m.heads, hd, l.ring);
        tick(P_SCORES);
        softmax_k<<<dim3(rep * tokens, kvh), 256>>>(m.scores, m.probs, l.vscale, l.tiles, a, m.heads, kvh, l.full ? 0 : m.window, l.ring);
        tick(P_SOFTMAX);
        attend(groups, m.probs, l.values, m.attn, l.tiles, a, m.heads, kvh, hd, l.ring);
        tick(P_ATTEND);
        prep(m, m.attn, nullptr, nullptr, nqo, l.rot_o, 1, tokens);
        tick(P_INPUT);
        gemm(m.xf, m.fps, tokens, l.o, m.y);
        tick(P_O);
        owed = Owed{m.y, l.post_attn_ln, 1.0f};
        // feed-forward
        prep(m, h, nullptr, l.pre_ff_ln, n, l.rot_mlp, 0, tokens, owed, spare);
        std::swap(h, spare);
        tick(P_INPUT);
        gemm(m.xf, m.fps, tokens, l.gate, m.y, l.up, m.y2);
        tick(P_GATE_UP);
        prep(m, m.y, m.y2, nullptr, m.mlp, l.rot_down, 2, tokens);
        tick(P_INPUT);
        gemm(m.xf, m.fps, tokens, l.down, m.y);
        tick(P_DOWN);
        owed = Owed{m.y, l.post_ff_ln, l.scalar};
    }
    // the last residual step, the final norm (kept in m.hn) and the head. Two steps a layer: the stream ends in m.h
    prep(m, h, nullptr, m.norm, n, m.rot_head, 0, tokens, owed, spare, m.hn);
    const int first = logits_from < 0 ? 0 : logits_from / 8 * 8;  // the first row of the group of eight the logits start in
    if (first < tokens) {
        float* logits = m.logits + size_t(first) * m.vocab;
        gemm(m.xf + size_t(first / 8) * (n / 16) * 32, m.fps + first, tokens - first, m.head, logits);
        if (m.softcap > 0) {
            const size_t count = size_t(tokens - first) * m.vocab;
            softcap_k<<<unsigned((count + 1023) / 1024), 1024>>>(logits, count, m.softcap);
        }
    }
    tick(P_HEAD);
    return int(cudaGetLastError());
}

// ---- The drafter (Gemma4Assistant): a few layers that guess the next token from the main model's last output.
// Its matrices are pair codes like the main model's. It has no keys of its own: every layer attends to the keys and
// values of the main model's last sliding or last full-attention layer.

struct DLayer {
    Mat q, o, gate, up, down;
    Rot rot_in, rot_o, rot_mlp, rot_down;
    const float *in_ln, *post_attn_ln, *pre_ff_ln, *post_ff_ln, *q_norm, *inv_freq;
    float scalar;
    int full, head_dim, kv_heads;
};

struct Drafter {
    DLayer* layers;
    int n_layers, hidden, mlp, heads, vocab, backbone;
    float eps;
    int pad;
    Mat pre, post, head;
    Rot rot_pre, rot_out;  // of the joined input, and of the final norm's output (the head's and post's input)
    const float* norm;
    // work memory, a row a sequence (as many as the main model's passes have rows)
    float* in;      // [rows][2 * backbone] the token's embedding in the main model, then a hidden state: the main model's last output, or `next`
    float* h;       // [rows][hidden]
    float* hn;      // [rows][hidden] after the final norm
    float* logits;  // [rows][vocab]
    float* next;    // [rows][backbone] the hidden state for the guess after this one
    float* h2;      // [rows][hidden] the residual stream's other state (see sclmd_pass)
    float* fps;     // the rest as in Model (it may be the main model's own work memory)
    uint64_t* xf;
    float *y, *y2;
    uint16_t* qhat;
    float* scores;
    uint16_t* probs;
    float* attn;
};

// The input of one of the drafter's matrices with n columns from a (and b, w: see prep_k), in d.xf with its scales in d.fps.
static void prep(const Drafter& d, int rows, const float* a, const float* b, const float* w, int n, const Rot& rot, int mode,
                 const Owed& owed = Owed{}, float* h_out = nullptr, float* v_out = nullptr) {
    launch_prep(rows, a, b, w, n, rot, mode, owed, h_out, v_out, d.fps, d.xf, d.eps);
}

// One guess for each of `rows` sequences (row r: sequence seqs[r]): d.in[r] holds the embedding of the token at
// position lengths[r] and a hidden state; the main model has run the positions before it, and `sliding` and `full`
// are its last layers of each kind (their keys and values are read: in a sliding layer the last window + 1 of them,
// as upstream lets the drafter). Afterwards d.logits[r] holds the guess for position lengths[r] + 1 and d.next[r]
// the hidden state for the guess after it. Launches only.
API int sclmd_draft(const Drafter* dp, const Layer* sliding, const Layer* full, int window, int rows, const int* lengths, const int* seqs) {
    const Drafter& d = *dp;
    const int n = d.hidden;
    Segs at{}, read{};  // where the queries sit, and the last position each may read
    at.n = read.n = rows;
    for (int r = 0; r < rows; ++r) {
        at.row0[r] = read.row0[r] = r, at.cnt[r] = read.cnt[r] = 1, at.seq[r] = read.seq[r] = seqs[r];
        at.pos0[r] = lengths[r], read.pos0[r] = lengths[r] - 1;
    }
    prep(d, rows, d.in, nullptr, nullptr, 2 * d.backbone, d.rot_pre, 1);
    gemm(d.xf, d.fps, rows, d.pre, d.h);
    float *h = d.h, *spare = d.h2;
    Owed owed;
    for (int i = 0; i < d.n_layers; ++i) {
        const DLayer& l = d.layers[i];
        const Layer& src = l.full ? *full : *sliding;
        const int hd = l.head_dim, kvh = l.kv_heads, rep = d.heads / kvh, groups = (rep + 7) / 8;
        // attention: the query sits at position `length` and reads the main model's keys up to length - 1
        prep(d, rows, h, nullptr, l.in_ln, n, l.rot_in, 0, owed, spare);
        if (owed.y) std::swap(h, spare);
        gemm(d.xf, d.fps, rows, l.q, d.y);
        heads_k<<<dim3(d.heads, rows), hd>>>(d.y, nullptr, nullptr, l.q_norm, nullptr, l.inv_freq, d.qhat, nullptr, nullptr, nullptr, nullptr,
                                             nullptr, att_for(at, true, d.heads, 0), d.heads, kvh, 1, d.eps);
        const Att a = att_for(read, l.full, d.heads, window);
        scores(tiles_of(a), kvh, groups, d.qhat, src.keys, src.kscale, d.scores, src.tiles, a, d.heads, hd, src.ring);
        softmax_k<<<dim3(rep * rows, kvh), 256>>>(d.scores, d.probs, src.vscale, src.tiles, a, d.heads, kvh, l.full ? 0 : window + 1, src.ring);
        attend(groups, d.probs, src.values, d.attn, src.tiles, a, d.heads, kvh, hd, src.ring);
        prep(d, rows, d.attn, nullptr, nullptr, d.heads * hd, l.rot_o, 1);
        gemm(d.xf, d.fps, rows, l.o, d.y);
        owed = Owed{d.y, l.post_attn_ln, 1.0f};
        // feed-forward
        prep(d, rows, h, nullptr, l.pre_ff_ln, n, l.rot_mlp, 0, owed, spare);
        std::swap(h, spare);
        gemm(d.xf, d.fps, rows, l.gate, d.y, l.up, d.y2);
        prep(d, rows, d.y, d.y2, nullptr, d.mlp, l.rot_down, 2);
        gemm(d.xf, d.fps, rows, l.down, d.y);
        owed = Owed{d.y, l.post_ff_ln, l.scalar};
    }
    prep(d, rows, h, nullptr, d.norm, n, d.rot_out, 0, owed, spare, d.hn);
    gemm(d.xf, d.fps, rows, d.head, d.logits, d.post, d.next);
    return int(cudaGetLastError());
}

API int sclmd_draft_sizes(int* dlayer, int* drafter) {
    *dlayer = sizeof(DLayer), *drafter = sizeof(Drafter);
    return 0;
}

API int sclmd_sizes(int* mat, int* rot, int* layer, int* model, int* segs) {
    *mat = sizeof(Mat), *rot = sizeof(Rot), *layer = sizeof(Layer), *model = sizeof(Model), *segs = sizeof(Segs);
    return MAX_TOKENS;
}
