// The engine: one pass of the Gemma 4 decoder over up to 256 new tokens (rows; past 32 for reading prompts, brief B2),
// on the GPU, from pair codes.
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

// rows of a pass: up to 32 groups of the eight columns of the tensor-core instruction. Up to 32 rows a warp holds every
// group (gemm_k / gemm8_k). A wider pass: with the integer option its groups are split over the warps of a block
// (gemm8_wide_k, brief B2), with fp16 products gemm_k goes 32 rows a launch; its attention takes as many rows at a time
// as the scores' memory holds (Model.scores_room, sized for ATT_ROWS rows), so that memory is what a 32-row pass needs.
static const int MAX_TOKENS = 256;
static const int ATT_ROWS = 32;    // rows of a pass attention takes at a time at least (and the most rows a pass has logits of)
static const int TWO_PHASE = 9;    // rows from which a matrix's input is prepared in two kernels (prep_k)
static const int MAX_SEGS = 64;    // segments of a pass

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
// For the integer products (Model.int8, an option): lut8[code] is the pair rounded to int8 (the codebook divided by
// book_scale, its largest entry / 127), the two bytes held twice in the word (b0 b1 b0 b1).
struct Mat {
    const uint32_t* codes;
    const uint32_t* lut;
    const float* scale;
    int rows, cols, bits, pad;
    const uint32_t* lut8;
    float book_scale;
    int pad2;
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
    // keys == null: the layer keeps its values only (its value is its key's projection), and the scores make its
    // keys of them (scores_made32_k) with `trig` and `turned`.
    int8_t *keys, *values;
    const int32_t* tiles;
    float *kscale, *vscale;
    const uint16_t* trig;  // [capacity][turned][2] cos and sin (fp16) of position * inv_freq[pair]
    float scalar;
    int full;      // 1: attends to every earlier position; 0: to the last `window`
    int head_dim, kv_heads;
    int ring;      // sliding: (window + spare) / 16, so positions go round; full: capacity / 16
    int turned;    // the rotated pairs, in whole 16s: the first `turned` values of each half of a head are turned
    // An option, off when 0 (brief K3, lossy): kv4 1: this sliding layer keeps its keys and values at 4 bits in a
    // turned basis (see kv4_heads_k): keys and values are then 4-bit codes, kscale and vscale their scales, and a
    // sequence's first tile is int8, its positions 8..15 in the pool tile first[sequence].
    int kv4, kv4_pad;
    const float* kv4_signs;  // [kv_heads][head_dim] +1 / -1: the signs of the turn (sclmd/kvsim.py)
    const int32_t* first;    // [sequences]
};

struct Model {
    Layer* layers;
    int n_layers, hidden, mlp, heads, window, vocab;
    float eps, softcap;
    Mat head;
    Rot rot_head;
    const float* norm;  // the final norm [hidden]
    // work memory; [rows]: the most rows a pass of this model may have (8, 16, 32, 64, 128 or 256: the caller sizes
    // the memory)
    float* h;        // [rows][hidden] the residual stream of the tokens of this pass (in: the embeddings; out: the last layer's output)
    float* h2;       // [rows][hidden] where the residual stream's next step is written (the two take turns)
    float* fps;      // [3 * MAX_TOKENS] the power of two an input was divided by to fit fp16 (then prep_k's sums)
    uint64_t* xf;    // [rows / 8][widest input / 16][32] a matrix's input as the instruction reads it, eight rows a group
    float* y;        // [rows][widest output] a matrix's output
    float* y2;       // [rows][mlp] (up's output beside gate's)
    uint16_t* qhat;  // [rows][heads * widest head] the queries, normalized and rotated (fp16)
    float* scores;   // [kv heads * queries][positions in whole tiles] attention scores, segment after segment, of the
                     // rows attention takes at a time (scores_room values: ATT_ROWS rows at the whole context)
    uint16_t* probs; // the same layout: attention weights (fp16)
    float* attn;     // [rows][heads * widest head] attention's output
    float* hn;       // [rows][hidden] the last output after the final norm (what the drafter is handed)
    float* logits;   // [min(rows, ATT_ROWS)][vocab] (a pass of more rows has the logits of its last ATT_ROWS at most: pass())
    // An option, off when 0 (brief B1, lossy): the layers' matrices multiply int8 inputs by int8 codebooks on the
    // integer tensor-core instruction (gemm8_k). A matrix's input is rounded per token with the clip at int8 times
    // its rms. The head stays fp16 (and the drafter, which has its own prep).
    float int8;
    int pad_int8;
    // the values m.scores (and m.probs) have room for: a pass of more than ATT_ROWS rows runs attention over as many
    // rows at a time as fit (ATT_ROWS at least; 0: ATT_ROWS)
    long long scores_room;
};

__device__ __forceinline__ void mma(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// s8 x s8 with int32 sums, 32 columns an instruction (two tiles); the sums are exact (bench/mma_s8.cu has the
// fragment layout and the measurements)
__device__ __forceinline__ void mma8(int c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// two int8 pairs (each held twice in its word, lut8) as one register: the low pair from p, the high from q
__device__ __forceinline__ uint32_t join8(uint32_t p, uint32_t q) { return __byte_perm(p, q, 0x5410); }

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

// Two int8 values (the low two bytes of w: the first in the low half) as the two fp16 the tensor-core instruction
// takes, in one register: a byte b is put in an fp16 as 1024 + (b ^ 0x80) and 1152 is taken away, which is exact.
// (A table of the 256 fp16 in shared memory, which this replaced, gave the same bits two lookups a word slower:
// bench/attn_bench.cu, 1.3-1.5x on the scores and the weighted sum.)
__device__ __forceinline__ uint32_t i8x2(uint32_t w) {
    const uint32_t x = __byte_perm(w ^ 0x8080u, 0x64646464u, 0x4140);
    const __half2 h = __hsub2(*(const __half2*)&x, __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480)));
    return *(const uint32_t*)&h;
}

// Two neighbouring int8 values (read as one 16-bit word) as two fp16 in one register.
__device__ __forceinline__ uint32_t two(uint16_t w) { return i8x2(w); }

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
// sigma > 0 (Model.int8): the values go out as int8 instead, one 32-bit word of four a lane and tile (the same
// columns as the fp16 word), xf read as uint32 with a lane's words of four tiles side by side (gemm8_k reads them as
// one uint4), and fps[token] is what they were divided by:
// sigma times the rms of the input / 127 (the rotation keeps the length), values past it clipped.
__global__ void __launch_bounds__(1024) prep_k(const float* a, const float* b, const float* w, const float* y, const float* post, float scalar, float* h_out,
                       float* v_out, const float* signs, float* fps, uint64_t* xf, int n, float eps, int mode, int phase, float* sums,
                       float sigma) {
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
        if (sigma > 0)
            fp = len > 0 ? sigma * len * rsqrtf(float(n)) / 127.0f : 1.0f;
        else
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
    if (sigma > 0) {
        if ((i & 9) == 0) {
            const int t = (i & 15) >> 1;
            const auto q = [&](float v) { return uint32_t(uint8_t(int8_t(fminf(fmaxf(rintf(v * k), -127.0f), 127.0f)))); };
            const uint32_t word = q(buf[i]) | q(buf[i + 1]) << 8 | q(buf[i + 8]) << 16 | q(buf[i + 9]) << 24;
            // tile j's word of lane L at (group * tiles + j - j % 4) * 32 + 4 L + j % 4: a lane's four tiles are one uint4
            const int j = col >> 4;
            reinterpret_cast<uint32_t*>(xf)[(size_t(token >> 3) * (n >> 4) + (j & ~3)) * 32 + 4 * (4 * (token & 7) + t) + (j & 3)] = word;
        }
        return;
    }
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

// gemm_k on the integer instruction (Model.int8): the same codes, strips, tiles and launch, the input int8 (xf read as
// uint32, a word a lane and tile: prep_k with sigma), the codebook int8 (lut8), two tiles an instruction. Instruction
// columns 0..15 are tile j (k = 4t + i: column {2t, 2t+1, 2t+8, 2t+9}[i]), 16..31 tile j + 1; so, from the word w of
// tile j (pairs (g, 2t), (g+8, 2t), (g, 2t+8), (g+8, 2t+8) a byte each), A's registers are
// pair(w & 255) | pair(w >> 16 & 255) << 16 (row g) and pair(w >> 8 & 255) | pair(w >> 24) << 16 (row g + 8), and the
// same of tile j + 1's word. The int32 sums are exact: a row's result has the same bits at any width and order.
// y = scale[row] * book_scale * fps[token] * sum.
template <int G>
__global__ void gemm8_k(Mat m0, Mat m1, Mat m2, float* y0, float* y1, float* y2, const uint32_t* xf, const float* fps, int segs, int tokens,
                        int strips) {
    __shared__ uint32_t luts[GEMM_STRIPS][256];
    __shared__ int part[GEMM_STRIPS][8][32][4];
    const int local = (threadIdx.x >> 5) / segs, warp = (threadIdx.x >> 5) % segs, lane = threadIdx.x & 31;
    int strip = blockIdx.x * strips + local;
    Mat m = m0;
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

// gemm8_k for a pass of more than four groups (brief B2; bench/mma_wide.cu is the kill test and the form's choice).
// One warp holds four groups in its registers (c[4][4] + b[4][4]), so the groups are split over SETS warps of a block
// that take the same strip and tiles: each decodes a quad of tiles of its strip in turn into shared memory, and each
// multiplies all the decoded quads with its own four groups (a decoded run shared by the group-set warps, form B).
// Keeping many strips a block on the same input matters most: a block's input is every group's, and with the
// engine's split of a strip's tiles over 4-8 warps the blocks had one strip and read the input once a strip
// (gate+up at 32 groups 1,152 us against 461 with segs 1). So `segs` is 1 here and a block has up to 4 strips.
// The int32 sums are exact: a row's result has the same bits as in gemm8_k (y is the same expression).
#define WIDE_WARPS 32
template <int SETS>
__global__ void __launch_bounds__(1024) gemm8_wide_k(Mat m0, Mat m1, Mat m2, float* y0, float* y1, float* y2, const uint32_t* xf, const float* fps, int segs, int tokens,
                             int strips) {
    extern __shared__ uint4 frags[];  // [warp][2][32]: a quad's eight A registers a lane
    __shared__ uint32_t luts[3][256];
    __shared__ int part[WIDE_WARPS][32][4];
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, team = segs * strips;
    const int set = wid / team, rest = wid % team, local = rest / segs, warp = rest % segs;
    if (m0.rows)
        for (int i = threadIdx.x; i < (1 << m0.bits); i += blockDim.x) luts[0][i] = m0.lut8[i];
    if (m1.rows)
        for (int i = threadIdx.x; i < (1 << m1.bits); i += blockDim.x) luts[1][i] = m1.lut8[i];
    if (m2.rows)
        for (int i = threadIdx.x; i < (1 << m2.bits); i += blockDim.x) luts[2][i] = m2.lut8[i];
    __syncthreads();
    int strip = blockIdx.x * strips + local, which = 0;
    Mat m = m0;
    float* y = y0;
    if (strip >= m0.rows / 16) {
        strip -= m0.rows / 16, m = m1, y = y1, which = 1;
        if (strip >= m1.rows / 16) strip -= m1.rows / 16, m = m2, y = y2, which = 2;
    }
    const bool idle = strip >= m.rows / 16;
    const uint32_t* slut = luts[which];
    // the matrices share the input: its tiles are m0's (an idle warp's m may have none, and every warp takes part in
    // the block's waits)
    const int tiles = m0.cols / 16, per = tiles / segs, first = warp * per, g0 = 4 * set, nq = per / 4;
    int c[4][4];
#pragma unroll
    for (int g = 0; g < 4; ++g) c[g][0] = c[g][1] = c[g][2] = c[g][3] = 0;
    const uint4* xw = reinterpret_cast<const uint4*>(xf) + size_t(first / 4) * 32 + lane;
    const size_t group = size_t(tiles) * 8;
    for (int r = 0; r < nq; r += SETS) {
        const int q = r + set;
        if (!idle && q < nq) {
            const int j = first + 4 * q;
            uint32_t a[8];
            if (m.bits == 8) {
                const uint32_t* cw = m.codes + (size_t(strip) * tiles + j) * 32 + lane;
                uint32_t w[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) w[i] = cw[size_t(i) * 32];
#pragma unroll
                for (int i = 0; i < 4; i += 2) {
                    const uint32_t u = w[i], v = w[i + 1];
                    a[2 * i] = join8(slut[u & 255], slut[u >> 16 & 255]), a[2 * i + 1] = join8(slut[u >> 8 & 255], slut[u >> 24]);
                    a[2 * i + 2] = join8(slut[v & 255], slut[v >> 16 & 255]), a[2 * i + 3] = join8(slut[v >> 8 & 255], slut[v >> 24]);
                }
            } else {
                const uint32_t* cq = m.codes + (size_t(strip) * (tiles / 4) + j / 4) * 96 + lane;
                const uint32_t q0 = cq[0], q1 = cq[32], q2 = cq[64];
                const uint64_t lo = uint64_t(q0) | uint64_t(q1) << 32, hi = q2;
                const uint32_t k10 = uint32_t(lo >> 60 | hi << 4) & 63;
                const uint32_t t[16] = {slut[lo & 63],       slut[lo >> 6 & 63],  slut[lo >> 12 & 63], slut[lo >> 18 & 63],
                                        slut[lo >> 24 & 63], slut[lo >> 30 & 63], slut[lo >> 36 & 63], slut[lo >> 42 & 63],
                                        slut[lo >> 48 & 63], slut[lo >> 54 & 63], slut[k10],           slut[hi >> 2 & 63],
                                        slut[hi >> 8 & 63],  slut[hi >> 14 & 63], slut[hi >> 20 & 63], slut[hi >> 26 & 63]};
                a[0] = join8(t[0], t[2]), a[1] = join8(t[1], t[3]), a[2] = join8(t[4], t[6]), a[3] = join8(t[5], t[7]);
                a[4] = join8(t[8], t[10]), a[5] = join8(t[9], t[11]), a[6] = join8(t[12], t[14]), a[7] = join8(t[13], t[15]);
            }
            frags[(wid * 2) * 32 + lane] = make_uint4(a[0], a[1], a[2], a[3]);
            frags[(wid * 2 + 1) * 32 + lane] = make_uint4(a[4], a[5], a[6], a[7]);
        }
        __syncthreads();
        const int here = nq - r < SETS ? nq - r : SETS;
        if (!idle)
            for (int s = 0; s < here; ++s) {
                const int src = s * team + rest, jq = (r + s);  // the quad's place among the warp's
                uint4 b[4];
#pragma unroll
                for (int g = 0; g < 4; ++g) b[g] = xw[(g0 + g) * group + size_t(jq) * 32];
                const uint4 lo = frags[(src * 2) * 32 + lane], hi = frags[(src * 2 + 1) * 32 + lane];
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    mma8(c[g], lo.x, lo.y, lo.z, lo.w, b[g].x, b[g].y);
                    mma8(c[g], hi.x, hi.y, hi.z, hi.w, b[g].z, b[g].w);
                }
            }
        __syncthreads();
    }
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        for (int i = 0; i < 4; ++i) part[wid][lane][i] = c[g][i];
        __syncthreads();
        if (warp == 0 && !idle)
            for (int i = 0; i < 4; ++i) {
                int sum = 0;
                for (int w = 0; w < segs; ++w) sum += part[wid + w][lane][i];
                const int row = strip * 16 + (lane >> 2) + (i >= 2 ? 8 : 0), token = 8 * (g0 + g) + 2 * (lane & 3) + (i & 1);
                if (token < tokens) y[size_t(token) * m.rows + row] = m.scale[row] * m.book_scale * fps[token] * float(sum);
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
// fold (a layer that keeps no keys: its k_norm, or the k_norm of the layer the drafter reads): the key is the value
// times k_norm, turned; where scores read the value itself as the key (the 16s of a head past the first `nturn` values
// of each half, which are not turned) k_norm is put on the query instead.
__global__ void heads_k(const float* yq, const float* yk, const float* yv, const float* q_norm, const float* k_norm, const float* inv_freq,
                        uint16_t* qhat, int8_t* keys, int8_t* values, float* kscale, float* vscale, const int32_t* tiles, Att a, int heads,
                        int kvh, int ring, float eps, const float* fold, int nturn) {
    const int hd = blockDim.x, v = blockIdx.x, token = blockIdx.y, d = threadIdx.x, half = hd / 2;
    const int kind = v < heads ? 0 : v < heads + kvh ? 1 : 2, head = kind == 0 ? v : kind == 1 ? v - heads : v - heads - kvh;
    const float* src = kind == 0 ? yq + (size_t(token) * heads + head) * hd
                                 : (kind == 1 || !yv ? yk : yv) + (size_t(token) * kvh + head) * hd;
    __shared__ float buf[1024];
    if (kind == 1 && !keys) return;  // the layer keeps no keys
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
        const float q = fold && (d & ~15) % half >= nturn ? out * fold[d] : out;
        qhat[(size_t(token) * heads + head) * hd + d] = __half_as_ushort(__float2half(q));
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
// STEPS steps of a query group's sum over a head's values, from value j on up to value `to`, while as many are left:
// the loads of all of them, then the products in order.
template <int STEPS>
__device__ __forceinline__ void scores_steps(float c[4], int& j, int to, const int8_t* ka, const int8_t* kb,
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
        for (int u = 0; u < STEPS; ++u) mma(c, two(w[u][0]), two(w[u][1]), two(w[u][2]), two(w[u][3]), b0[u], b1[u]);
    }
}

// The steps from value j up to `to` with the loads of up to AHEAD steps on their way together.
template <int AHEAD>
__device__ __forceinline__ void scores_plain(float c[4], int& j, int to, const int8_t* ka, const int8_t* kb,
                                             const uint16_t* qa, bool has) {
    scores_steps<AHEAD>(c, j, to, ka, kb, qa, has);
    if (AHEAD > 8) scores_steps<8>(c, j, to, ka, kb, qa, has);
    if (AHEAD > 2) scores_steps<2>(c, j, to, ka, kb, qa, has);
    scores_steps<1>(c, j, to, ka, kb, qa, has);
}

// A layer that keeps its values only (Layer::keys is null): a key is the value times k_norm, turned by its position
// (heads_k: out = x * cos + (the value half a head away, negated in the first half) * sin). Two of a turned key's
// values as the instruction takes them, made of the two int8 values (own), their partners half a head away (part),
// k_norm of each (ko, kp) and the cos and sin of the position for the two pairs (tr: cos, sin, cos, sin as fp16).
// sign: -1 in the first half of the head, +1 in the second.
__device__ __forceinline__ uint32_t made(uint16_t own, uint16_t part, uint64_t tr, const float* ko, const float* kp, float sign) {
    const float c0 = __half2float(__ushort_as_half(uint16_t(tr))), s0 = __half2float(__ushort_as_half(uint16_t(tr >> 16)));
    const float c1 = __half2float(__ushort_as_half(uint16_t(tr >> 32))), s1 = __half2float(__ushort_as_half(uint16_t(tr >> 48)));
    const float e0 = float(int8_t(own & 255)) * ko[0] * c0 + sign * (float(int8_t(part & 255)) * kp[0] * s0);
    const float e1 = float(int8_t(own >> 8)) * ko[1] * c1 + sign * (float(int8_t(part >> 8)) * kp[1] * s1);
    return uint32_t(__half_as_ushort(__float2half(e0))) | uint32_t(__half_as_ushort(__float2half(e1))) << 16;
}

// STEPS steps over turned values, from value j on up to `to` (all in the half of the head that starts at `first`):
// va, vb: the values of the lane's two positions at the lane's columns (2t on); ta, tb: the positions' cos and sin,
// two pairs a word; kn: k_norm of the turned values (the first half's, then the second's).
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

// MADE: the layer keeps no keys: `keys` and `kscale` are its values and their scales, and the keys are made of them.
template <int AHEAD, bool MADE>
__device__ __forceinline__ void scores_body(const uint16_t* qhat, const int8_t* keys, const float* kscale, float* scores, const int32_t* tiles,
                                            const Att& a, int heads, int kvh, int hd, int ring, const uint16_t* trig = nullptr,
                                            const float* k_norm = nullptr, int turned = 0) {
    __shared__ float kn[MADE ? 256 : 1];
    if (MADE) {
        for (int i = threadIdx.x; i < 2 * turned; i += blockDim.x) kn[i] = k_norm[i < turned ? i : hd / 2 + i - turned];
        __syncthreads();
    }
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
    if (MADE) {
        const int half = hd / 2, pos = a.base[seg] + 16 * tile + g;
        const uint64_t* ta = (const uint64_t*)(trig + size_t(pos) * turned * 2);
        const uint64_t* tb = (const uint64_t*)(trig + size_t(pos + 8) * turned * 2);
        for (int first = 0; first < hd; first += half) {
            scores_made<4>(c, j, first + turned, first, half, turned, ka, kb, ta, tb, kn, t, qa, has);
            scores_made<1>(c, j, first + turned, first, half, turned, ka, kb, ta, tb, kn, t, qa, has);
            scores_plain<AHEAD>(c, j, first + half, ka, kb, qa, has);
        }
    } else {
        scores_plain<AHEAD>(c, j, hd, ka, kb, qa, has);
    }
    float* out = scores + a.off[seg];
    const float sa = kscale[(slot + g) * kvh + kh], sb = kscale[(slot + g + 8) * kvh + kh];
    for (int i = 0; i < 4; ++i) {  // c[i]: position of the tile g (+ 8), query group * 8 + 2t (+ 1)
        const int q = group * 8 + 2 * t + (i & 1), r = 16 * tile + g + (i >= 2 ? 8 : 0);
        if (q < nq) out[(size_t(kh) * nq + q) * stride + r] = c[i] * (i >= 2 ? sb : sa);
    }
}

__global__ void scores32_k(const uint16_t* qhat, const int8_t* keys, const float* kscale, float* scores, const int32_t* tiles, Att a, int heads,
                           int kvh, int hd, int ring) {
    scores_body<32, false>(qhat, keys, kscale, scores, tiles, a, heads, kvh, hd, ring);
}

__global__ void scores_made32_k(const uint16_t* qhat, const int8_t* values, const float* vscale, float* scores, const int32_t* tiles, Att a,
                                int heads, int kvh, int hd, int ring, const uint16_t* trig, const float* k_norm, int turned) {
    scores_body<32, true>(qhat, values, vscale, scores, tiles, a, heads, kvh, hd, ring, trig, k_norm, turned);
}

// The scores of a pass with many query groups: the kernels above have every warp (a group) make the fp16 of a key
// tile for itself, so a tile is made as many times as there are groups. Here a block of eight warps (eight groups,
// each with its queries in registers) makes each tile once, in shared memory, and the warps read it with ldmatrix:
// the products are the same, in the same order, so the scores are the same bits (bench/attn_bench.cu: a full layer
// at 16,384 positions, 32 rows 1.23 -> 0.24 ms, 16 rows 0.62 -> 0.13; keyless 1.50 -> 0.44, 0.75 -> 0.23).
// Block (ONCE_TILES tiles of a segment, key head, eight groups); the segments' runs of blocks follow each other.
static const int ONCE_TILES = 16;
// Query groups a key head from which scores() takes this way. The real model at a 2,048-position context, scores a
// pass: 8 rows (sliding layers 2 groups) 1.49 ms from 2 groups, 1.31 from 3; 16 rows (4 groups) 1.53 from 3, 1.86 from 5.
static const int ONCE_FROM = 3;

__device__ __forceinline__ void ldsm4(uint32_t r[4], const void* p) {
    const uint32_t s = uint32_t(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

// 16 positions x HD values of key head kh from int8 memory (the tile's first position's row: src) into the shared
// tile (rows of HD + 8 fp16) as the fp16 scores_body would make: 256 threads, thread i takes row i / 16 and HD / 16
// values from HD / 16 * (i % 16). MADE: the keys made of the values (a run is all turned or none: turned % 32 == 0).
template <int HD, bool MADE>
__device__ __forceinline__ void key_tile(uint16_t* sh, const int8_t* src, int kvh, int kh, int pos, const uint16_t* trig, const float* kn,
                                         int turned) {
    constexpr int NB = HD / 16, NW = NB / 4;
    const int i = threadIdx.x, p = i >> 4, d0 = (i & 15) * NB;
    const int8_t* row = src + (size_t(p) * kvh + kh) * HD;
    uint32_t ws[NW], o[NB / 2];
#pragma unroll
    for (int k = 0; k < NB / 16; ++k) ((uint4*)ws)[k] = ((const uint4*)(row + d0))[k];
    const int half = HD / 2, h = d0 >= half, e0 = d0 - h * half;
    if (MADE && e0 < turned) {
        uint32_t ps[NW];
        const int8_t* prow = row + d0 + (h ? -half : half);
#pragma unroll
        for (int k = 0; k < NB / 16; ++k) ((uint4*)ps)[k] = ((const uint4*)prow)[k];
        const uint64_t* tr = (const uint64_t*)(trig + size_t(pos + p) * turned * 2);
        const float sign = h ? 1.0f : -1.0f;
#pragma unroll
        for (int k = 0; k < NB / 2; ++k) {
            const int e = e0 + 2 * k;
            o[k] = made(uint16_t(ws[k >> 1] >> (16 * (k & 1))), uint16_t(ps[k >> 1] >> (16 * (k & 1))), tr[e >> 1], kn + h * turned + e,
                        kn + (1 - h) * turned + e, sign);
        }
    } else {
#pragma unroll
        for (int k = 0; k < NW; ++k) o[2 * k] = i8x2(ws[k] & 0xffff), o[2 * k + 1] = i8x2(ws[k] >> 16);
    }
    uint4* dst = (uint4*)(sh + p * (HD + 8) + d0);
#pragma unroll
    for (int k = 0; k < NB / 8; ++k) dst[k] = make_uint4(o[4 * k], o[4 * k + 1], o[4 * k + 2], o[4 * k + 3]);
}

template <int HD, bool MADE>
__global__ void __launch_bounds__(256) scores_once_k(const uint16_t* qhat, const int8_t* keys, const float* kscale, float* scores,
                                                     const int32_t* tiles, Att a, int heads, int kvh, int ring, const uint16_t* trig,
                                                     const float* k_norm, int turned) {
    constexpr int SROW = HD + 8, STEPS = HD / 16;
    __shared__ __align__(16) uint16_t sh[16 * SROW];
    __shared__ float kn[MADE ? 256 : 1];
    int seg = 0, c0 = 0;
    for (int i = 0, c = 0; i < a.n; c += (a.stride[i] / 16 + ONCE_TILES - 1) / ONCE_TILES, ++i)
        if (int(blockIdx.x) >= c) seg = i, c0 = c;
    const int rep = heads / kvh, nq = rep * a.cnt[seg], stride = a.stride[seg];
    if (int(blockIdx.z) * 64 >= nq) return;  // the whole block: no group of it has a query in this segment
    if (MADE)
        for (int i = threadIdx.x; i < 2 * turned; i += blockDim.x) kn[i] = k_norm[i < turned ? i : HD / 2 + i - turned];
    const int kh = blockIdx.y, group = blockIdx.z * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int query = group * 8 + g;
    const bool has = query < nq, any = group * 8 < nq;
    const uint16_t* qa = qhat + (size_t(a.row0[seg] + (has ? query / rep : 0)) * heads + kh * rep + (has ? query % rep : 0)) * HD + 2 * t;
    uint32_t b0[STEPS], b1[STEPS];
#pragma unroll
    for (int u = 0; u < STEPS; ++u) b0[u] = has ? *(const uint32_t*)(qa + 16 * u) : 0, b1[u] = has ? *(const uint32_t*)(qa + 16 * u + 8) : 0;
    float* out = scores + a.off[seg];
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;
    const int t0 = (blockIdx.x - c0) * ONCE_TILES, t1 = min(stride / 16, t0 + ONCE_TILES);
    for (int tile = t0; tile < t1; ++tile) {
        const size_t slot = slot_of(tiles, ring, a.seq[seg], a.base[seg] + 16 * tile);
        __syncthreads();  // (the first: kn is written)
        key_tile<HD, MADE>(sh, keys + slot * kvh * HD, kvh, kh, a.base[seg] + 16 * tile, trig, kn, turned);
        __syncthreads();
        if (!any) continue;
        float c[4] = {0, 0, 0, 0};
#pragma unroll
        for (int u = 0; u < STEPS; ++u) {
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
// tiles. -> out[row][head][head_dim]. The values are int8 (as fp16 by i8x2); their scales are in the weights.
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
__device__ __forceinline__ void attend_tiles(float c[4], int& r, int stride, const Walk& k) {
#define PAIR(x, y) i8x2(uint32_t(x) | uint32_t(y) << 8)
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
    attend_tiles<AHEAD>(c, r, stride, k);
    if (AHEAD > 8) attend_tiles<8>(c, r, stride, k);
    if (AHEAD > 2) attend_tiles<2>(c, r, stride, k);
    attend_tiles<1>(c, r, stride, k);
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

// The weighted sum with a warp taking NDT sixteens of the head: its weights are loaded once for all of them and its
// values are one load of 2 NDT bytes a position, so a pass with many groups reads the weights NDT times less often
// (the cost of attend32_k at long contexts: every warp reads its group's weights from L2). Each sum still adds the
// tiles in order and which row of the instruction a value sits in does not change its sum, so the bits are
// attend32_k's (bench/attn_bench.cu). Block (NDT-sixteens chunk of the head x blocks of groups, key head, segment);
// a lane's rows g and g + 8 of sixteen k are the values c0 + 2 NDT g + 2k and + 1.
// Query groups a key head from which attend() takes two / four sixteens a warp. bench/attn_bench.cu, a full layer at
// 65,536 positions: 8 groups 1.25 ms (attend32_k) / 0.98 (one sixteen, i8x2) / 0.55 (two) / 1.40 (four); 16 groups
// 2.48 / 1.90 / 0.85 / 1.40; 32 groups 4.31 / 3.28 / 1.68 / 1.56 (6,144 positions: 0.41 / 0.31 / 0.16 / 0.135).
static const int ATTEND_NDT2_FROM = 2, ATTEND_NDT4_FROM = 32;
template <int NDT>
struct Wide;
template <>
struct Wide<2> {
    using T = uint32_t;
};
template <>
struct Wide<4> {
    using T = uint64_t;
};

template <int S, int NDT>
__device__ __forceinline__ void attend_wide_steps(float c[NDT][4], int& r, int stride, const int32_t* mine, int j0, int ring, const uint8_t* vals,
                                                  size_t row, const uint16_t* pq, bool has) {
    using W = typename Wide<NDT>::T;
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
                mma(c[k], i8x2(__byte_perm(y0, y1, 0x40)), i8x2(__byte_perm(y0, y1, 0x51)), i8x2(__byte_perm(y2, y3, 0x40)),
                    i8x2(__byte_perm(y2, y3, 0x51)), b0[u], b1[u]);
            }
    }
}

template <int AHEAD, int NDT>
__global__ void attend_wide_k(const uint16_t* probs, const int8_t* values, float* out, const int32_t* tiles, Att a, int heads, int kvh, int hd,
                              int ring) {
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
    attend_wide_steps<AHEAD, NDT>(c, r, stride, mine, j0, ring, vals, row, pq, has);
    if (AHEAD > 8) attend_wide_steps<8, NDT>(c, r, stride, mine, j0, ring, vals, row, pq, has);
    if (AHEAD > 2) attend_wide_steps<2, NDT>(c, r, stride, mine, j0, ring, vals, row, pq, has);
    attend_wide_steps<1, NDT>(c, r, stride, mine, j0, ring, vals, row, pq, has);
#pragma unroll
    for (int k = 0; k < NDT; ++k)
        for (int i = 0; i < 4; ++i) {
            const int q = group * 8 + 2 * t + (i & 1), d = int(c0) + 2 * NDT * g + 2 * k + (i >= 2 ? 1 : 0);
            if (q < nq) out[(size_t(a.row0[seg] + q / rep) * heads + kh * rep + q % rep) * hd + d] = c[k][i] * (1.0f / BOOST);
        }
}

// ---- Brief K3 (an option, off by default; lossy): a sliding layer's keys and values at 4 bits (Layer::kv4).
// The format is sclmd/kvsim.py's k4v4t0, whose cost was simulated first (logs/exp/K3-kv4-sliding). A key (after the
// rotary turn) or a value (RMS-normed) x of key head h is kept as
//   y = H (s * x)           s: the layer's signs of the head (Layer::kv4_signs), H: the Walsh-Hadamard matrix / sqrt(hd)
//   scale = RMS(y) * KV4_UNIT   (the top level, 127, is 2.73 RMS: the Gaussian Lloyd-Max levels), in today's scale slot
//   code = the nearest of 16 levels on the int8 grid (+-6, 18, 31, 44, 58, 75, 96, 127): how many of the 15 midpoints are below y / scale
// so that the kept vector is s * H (level * scale). H is symmetric and orthonormal, so q . that = (H (s * q)) . (level *
// scale): the scores take the query turned the same way against the levels, and the weighted sum of the values is
// turned back once a row and head (kv4_unrot_k). A sequence's first tile (positions 0..15, where attention's sinks sit)
// is kept as without the option: int8 of x itself, its positions 0..7 in the bytes of its ring tile 0 and 8..15 in
// those of a second tile, Layer::first[sequence] (an int8 vector is two 4-bit ones' bytes). The scores read that tile
// with the query as it is, and kv4_unrot_k adds its share of the weighted sum. Every 4-bit position's bytes lie within
// the int8 rows of position p / 2 there, so when the ring goes round, writing positions 16 ring + p over the ring tile 0
// takes only first-tile positions no row of the pass (nor a turn cropped back to, Engine._kept) still reads, as int8 does.
// A layer's tile holds 16 x kvh x hd / 2 bytes of keys and as many of values (and today's scales, a float a vector):
//   keys:   [16 positions][kvh][hd / 2]: lane t's step u of the scores (values 16u + 2t, 2t+1, 2t+8, 2t+9, two to a
//           byte, the lower value in the low nibble) at bytes 2 (t hd / 16 + u) and + 1 of the position's head
//   values: [16 positions][kvh][hd / 2]: byte b of a position's head holds its values 2b (low nibble) and 2b + 1
//   the first tile's int8: [8 positions][kvh][hd] in each of its two tiles' bytes
static const int KV4_FIRST = 16;
__host__ __device__ constexpr uint32_t kv4_word(int a, int b, int c, int d) {
    return uint32_t(uint8_t(a)) | uint32_t(uint8_t(b)) << 8 | uint32_t(uint8_t(c)) << 16 | uint32_t(uint8_t(d)) << 24;
}
// the levels of codes 0..15, four to a word (sclmd/kvsim.py grid_levels(4))
#define KV4_W0 kv4_word(-127, -96, -75, -58)
#define KV4_W1 kv4_word(-44, -31, -18, -6)
#define KV4_W2 kv4_word(6, 18, 31, 44)
#define KV4_W3 kv4_word(58, 75, 96, 127)
#define KV4_CUTS {-111.5f, -85.5f, -66.5f, -51.0f, -37.5f, -24.5f, -12.0f, 0.0f, 12.0f, 24.5f, 37.5f, 51.0f, 66.5f, 85.5f, 111.5f}
__constant__ float kv4_cut[15] = KV4_CUTS;
static const float kv4_cut_host[15] = KV4_CUTS;
constexpr float KV4_UNIT = 0.021516453474760056f;  // the top Lloyd-Max level of 16 (2.7326 RMS) / 127, as float32

// Four codes (the low 16 bits of w, a nibble each, the first lowest) as their levels, an int8 a byte in the same order.
__device__ __forceinline__ uint32_t kv4_bytes(uint32_t w) {
    const uint32_t s = w & 0x7777u;  // a code's low three bits pick a level among eight, its fourth which eight
    const uint32_t lo = __byte_perm(KV4_W0, KV4_W1, s), hi = __byte_perm(KV4_W2, KV4_W3, s);
    const uint32_t x = (w >> 3) & 0x1111u;
    const uint32_t m = (__byte_perm(x, x >> 4, 0x5140) & 0x01010101u) * 0xFFu;  // 0xff in the bytes of codes 8..15
    return (lo & ~m) | (hi & m);
}

// Two int8 bytes of w, picked by sel (0x4140: bytes 0 and 1; 0x4342: 2, 3; 0x4240: 0, 2; 0x4341: 1, 3), as two fp16
// in one register (as i8x2).
__device__ __forceinline__ uint32_t i8x2_of(uint32_t w, uint32_t sel) {
    const uint32_t x = __byte_perm(w ^ 0x80808080u, 0x64646464u, sel);
    const __half2 h = __hsub2(*(const __half2*)&x, __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480)));
    return *(const uint32_t*)&h;
}

// The code of y in a vector of this scale: how many midpoints are below y / scale (torch.bucketize, as kvsim does).
__device__ __forceinline__ uint32_t kv4_code(float y, float scale) {
    const float x = y / scale;
    uint32_t c = 0;
#pragma unroll
    for (int i = 0; i < 15; ++i) c += kv4_cut[i] < x;
    return c;
}

// heads_k for a 4-bit layer (same grid and threads): the queries go out twice, as they are (qplain, for the first
// tile) and turned (qhat); a key or value at a position past the first tile as 4-bit codes, in the first tile as int8.
__global__ void kv4_heads_k(const float* yq, const float* yk, const float* yv, const float* q_norm, const float* k_norm, const float* inv_freq,
                            uint16_t* qhat, uint16_t* qplain, uint8_t* keys, uint8_t* values, float* kscale, float* vscale, const int32_t* tiles,
                            const int32_t* first, const float* signs, Att a, int heads, int kvh, int ring, float eps) {
    const int hd = blockDim.x, v = blockIdx.x, token = blockIdx.y, d = threadIdx.x, half = hd / 2;
    const int kind = v < heads ? 0 : v < heads + kvh ? 1 : 2, head = kind == 0 ? v : kind == 1 ? v - heads : v - heads - kvh;
    const float* src = kind == 0 ? yq + (size_t(token) * heads + head) * hd : (kind == 1 ? yk : yv) + (size_t(token) * kvh + head) * hd;
    __shared__ float buf[1024];
    const float val = src[d];
    const float r = powf(block_sum(val * val) / hd + eps, -0.5f);
    const int seg = seg_of_row(a, token), pos = a.pos0[seg] + token - a.row0[seg], seq = a.seq[seg];
    float out = val * r;  // a value; a query or key is normed with its weight and turned by its position, as heads_k does
    if (kind != 2) {
        buf[d] = val * r * (kind == 0 ? q_norm[d] : k_norm[d]);
        __syncthreads();
        const float ang = float(pos) * inv_freq[d % half];
        const float turned = d < half ? -buf[d + half] : buf[d - half];
        out = buf[d] * cosf(ang) + turned * sinf(ang);
    }
    const int kh = kind == 0 ? head / (heads / kvh) : head;
    const size_t tile_bytes = size_t(16) * kvh * hd / 2;
    if (kind == 0) qplain[(size_t(token) * heads + head) * hd + d] = __half_as_ushort(__float2half(out));
    if (kind != 0 && pos < KV4_FIRST) {  // the sequence's first tile: int8 of the vector itself, as heads_k keeps it
        const float most = block_max(fabsf(out)), scale = most > 0 ? most / 127.0f : 1.0f;
        const int tile = pos < 8 ? tiles[size_t(seq) * ring] : first[seq], sub = pos & 7;
        (kind == 1 ? keys : values)[tile * tile_bytes + (size_t(sub) * kvh + kh) * hd + d] = uint8_t(int8_t(rintf(out / scale)));
        if (d == 0) (kind == 1 ? kscale : vscale)[(size_t(tile) * 16 + sub) * kvh + kh] = scale;
        return;
    }
    __syncthreads();  // (the rotary turn has read buf)
    buf[d] = out * signs[kh * hd + d];
    __syncthreads();
    for (int h = 1; h < hd; h <<= 1) {
        const float lo = buf[d & ~h], hi = buf[d | h];
        __syncthreads();
        buf[d] = (d & h) ? lo - hi : lo + hi;
        __syncthreads();
    }
    const float y = buf[d] * (1.0f / sqrtf(float(hd)));
    if (kind == 0) {
        qhat[(size_t(token) * heads + head) * hd + d] = __half_as_ushort(__float2half(y));
        return;
    }
    const float rms = sqrtf(block_sum(y * y) / hd), scale = rms > 0 ? rms * KV4_UNIT : 1.0f;
    const uint32_t code = kv4_code(y, scale), other = __shfl_down_sync(0xffffffffu, code, 1);
    const size_t slot = slot_of(tiles, ring, seq, pos);
    if (d == 0) (kind == 1 ? kscale : vscale)[slot * kvh + kh] = scale;
    if (d & 1) return;
    const uint8_t byte = uint8_t(code | other << 4);  // values d and d + 1 of the position
    if (kind == 1) {
        const int u = d >> 4, c = d & 15, t = (c & 7) >> 1;
        keys[(slot * kvh + kh) * (hd / 2) + 2 * (t * (hd / 16) + u) + (c >> 3)] = byte;
    } else {
        values[(slot * kvh + kh) * (hd / 2) + (d >> 1)] = byte;
    }
}

// The first tile's scores, int8 in the plain basis, as scores_body takes a tile of int8 keys: its positions 0..7 from
// the ring tile 0, 8..15 from the sequence's second tile. c: the sums; sa, sb: the scales of positions g and g + 8.
template <int HD>
__device__ __forceinline__ void kv4_first_scores(float c[4], float& sa, float& sb, const uint8_t* keys, const float* kscale, const int32_t* tiles,
                                                 const int32_t* first, int seq, int ring, int kvh, int kh, int g, int t, const uint16_t* qa, bool has) {
    const int ta = tiles[size_t(seq) * ring], tb = first[seq];
    const size_t bytes = size_t(16) * kvh * HD / 2;
    const int8_t* ka = (const int8_t*)keys + ta * bytes + (size_t(g) * kvh + kh) * HD + 2 * t;
    const int8_t* kb = (const int8_t*)keys + tb * bytes + (size_t(g) * kvh + kh) * HD + 2 * t;
    int j = 0;
    scores_plain<HD / 16>(c, j, HD, ka, kb, qa, has);
    sa = kscale[(size_t(ta) * 16 + g) * kvh + kh], sb = kscale[(size_t(tb) * 16 + g) * kvh + kh];
}

// The scores of a 4-bit layer, a warp a group as scores32_k: block (position tile of a segment, key head, blocks of
// groups). A lane reads its two positions' codes of the whole head at once (hd / 8 bytes each) and makes the
// instruction's fp16 of them step by step.
template <int HD>
__global__ void kv4_scores_k(const uint16_t* qhat, const uint16_t* qplain, const uint8_t* keys, const float* kscale, float* scores,
                             const int32_t* tiles, const int32_t* first, Att a, int heads, int kvh, int ring) {
    constexpr int NW = HD / 32, STEPS = HD / 16;  // a lane's words of a position, and the instruction's steps over a head
    int seg = 0;
    for (int i = 1; i < a.n; ++i)
        if (int(blockIdx.x) >= a.tile0[i]) seg = i;
    const int rep = heads / kvh, nq = rep * a.cnt[seg], stride = a.stride[seg];
    const int tile = blockIdx.x - a.tile0[seg], kh = blockIdx.y, group = blockIdx.z * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    if (group * 8 >= nq) return;
    const int query = group * 8 + g, at = a.base[seg] + 16 * tile, seq = a.seq[seg];
    const bool has = query < nq;
    const size_t qo = (size_t(a.row0[seg] + (has ? query / rep : 0)) * heads + kh * rep + (has ? query % rep : 0)) * HD + 2 * t;
    float c[4] = {0, 0, 0, 0}, sa, sb;
    if (at == 0) {
        kv4_first_scores<HD>(c, sa, sb, keys, kscale, tiles, first, seq, ring, kvh, kh, g, t, qplain + qo, has);
    } else {
        const size_t slot = slot_of(tiles, ring, seq, at);
        const uint8_t* ka = keys + ((slot + g) * kvh + kh) * (HD / 2) + t * (HD / 8);
        const uint8_t* kb = keys + ((slot + g + 8) * kvh + kh) * (HD / 2) + t * (HD / 8);
        uint32_t wa[NW], wb[NW], b0[STEPS], b1[STEPS];
        if constexpr (NW % 4 == 0) {
#pragma unroll
            for (int k = 0; k < NW / 4; ++k) ((uint4*)wa)[k] = ((const uint4*)ka)[k], ((uint4*)wb)[k] = ((const uint4*)kb)[k];
        } else {
#pragma unroll
            for (int k = 0; k < NW; ++k) wa[k] = ((const uint32_t*)ka)[k], wb[k] = ((const uint32_t*)kb)[k];
        }
#pragma unroll
        for (int u = 0; u < STEPS; ++u)
            b0[u] = has ? *(const uint32_t*)(qhat + qo + 16 * u) : 0, b1[u] = has ? *(const uint32_t*)(qhat + qo + 16 * u + 8) : 0;
#pragma unroll
        for (int u = 0; u < STEPS; ++u) {
            const uint32_t xa = kv4_bytes(wa[u >> 1] >> (16 * (u & 1))), xb = kv4_bytes(wb[u >> 1] >> (16 * (u & 1)));
            mma(c, i8x2_of(xa, 0x4140), i8x2_of(xb, 0x4140), i8x2_of(xa, 0x4342), i8x2_of(xb, 0x4342), b0[u], b1[u]);
        }
        sa = kscale[(slot + g) * kvh + kh], sb = kscale[(slot + g + 8) * kvh + kh];
    }
    float* out = scores + a.off[seg];
    for (int i = 0; i < 4; ++i) {
        const int q = group * 8 + 2 * t + (i & 1), r = 16 * tile + g + (i >= 2 ? 8 : 0);
        if (q < nq) out[(size_t(kh) * nq + q) * stride + r] = c[i] * (i >= 2 ? sb : sa);
    }
}

// kv4_scores_k with each tile's fp16 made once a block, as scores_once_k (the same products in the same order: the
// same bits as kv4_scores_k). Thread i makes position i / 16's values of lane (i % 16) / 4's steps 4 (i % 4) .. + 3.
template <int HD>
__global__ void __launch_bounds__(256) kv4_scores_once_k(const uint16_t* qhat, const uint16_t* qplain, const uint8_t* keys, const float* kscale,
                                                         float* scores, const int32_t* tiles, const int32_t* first, Att a, int heads, int kvh,
                                                         int ring) {
    static_assert(HD == 256, "a thread makes 16 values of a position: 8 bytes");
    constexpr int SROW = HD + 8, STEPS = HD / 16;
    __shared__ __align__(16) uint16_t sh[16 * SROW];
    int seg = 0, c0 = 0;
    for (int i = 0, c = 0; i < a.n; c += (a.stride[i] / 16 + ONCE_TILES - 1) / ONCE_TILES, ++i)
        if (int(blockIdx.x) >= c) seg = i, c0 = c;
    const int rep = heads / kvh, nq = rep * a.cnt[seg], stride = a.stride[seg], seq = a.seq[seg];
    if (int(blockIdx.z) * 64 >= nq) return;  // the whole block: no group of it has a query in this segment
    const int kh = blockIdx.y, group = blockIdx.z * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int query = group * 8 + g;
    const bool has = query < nq, any = group * 8 < nq;
    const size_t qo = (size_t(a.row0[seg] + (has ? query / rep : 0)) * heads + kh * rep + (has ? query % rep : 0)) * HD + 2 * t;
    uint32_t b0[STEPS], b1[STEPS];
#pragma unroll
    for (int u = 0; u < STEPS; ++u)
        b0[u] = has ? *(const uint32_t*)(qhat + qo + 16 * u) : 0, b1[u] = has ? *(const uint32_t*)(qhat + qo + 16 * u + 8) : 0;
    float* out = scores + a.off[seg];
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;
    const int p = threadIdx.x >> 4, ch = threadIdx.x & 15, tq = ch >> 2, u0 = 4 * (ch & 3);
    const int t0 = (blockIdx.x - c0) * ONCE_TILES, t1 = min(stride / 16, t0 + ONCE_TILES);
    for (int tile = t0; tile < t1; ++tile) {
        const int at = a.base[seg] + 16 * tile;
        float c[4] = {0, 0, 0, 0}, sa, sb;
        if (at == 0) {  // (the whole block) the first tile, int8: each warp for itself
            if (!any) continue;
            kv4_first_scores<HD>(c, sa, sb, keys, kscale, tiles, first, seq, ring, kvh, kh, g, t, qplain + qo, has);
        } else {
            const size_t slot = slot_of(tiles, ring, seq, at);
            __syncthreads();
            const uint2 w = *(const uint2*)(keys + ((slot + p) * kvh + kh) * (HD / 2) + 8 * ch);
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const uint32_t x = kv4_bytes((s < 2 ? w.x : w.y) >> (16 * (s & 1)));
                uint32_t* at2 = (uint32_t*)(sh + p * SROW + 16 * (u0 + s) + 2 * tq);
                at2[0] = i8x2_of(x, 0x4140), at2[4] = i8x2_of(x, 0x4342);  // values 2tq, 2tq + 1 of the step, and 8 on
            }
            __syncthreads();
            if (!any) continue;
#pragma unroll
            for (int u = 0; u < STEPS; ++u) {
                uint32_t r[4];
                ldsm4(r, sh + lrow * SROW + 16 * u + lcol);
                mma(c, r[0], r[1], r[2], r[3], b0[u], b1[u]);
            }
            sa = kscale[(slot + g) * kvh + kh], sb = kscale[(slot + g + 8) * kvh + kh];
        }
        for (int i = 0; i < 4; ++i) {
            const int q = group * 8 + 2 * t + (i & 1), r = 16 * tile + g + (i >= 2 ? 8 : 0);
            if (q < nq) out[(size_t(kh) * nq + q) * stride + r] = c[i] * (i >= 2 ? sb : sa);
        }
    }
}

// softmax_k for a 4-bit layer: the same sums; the scales of the first tile's positions are where it keeps them.
__global__ void kv4_softmax_k(const float* scores, uint16_t* probs, const float* vscale, const int32_t* tiles, const int32_t* first, Att a, int heads,
                              int kvh, int window, int ring) {
    const int n = blockDim.x, rep = heads / kvh, kh = blockIdx.y, row = blockIdx.x / rep, seg = seg_of_row(a, row);
    const int nq = rep * a.cnt[seg], query = blockIdx.x - rep * a.row0[seg], pos = a.pos0[seg] + row - a.row0[seg];
    const int base = a.base[seg], stride = a.stride[seg];
    const size_t at = a.off[seg] + (size_t(kh) * nq + query) * stride;
    const float* s = scores + at;
    uint16_t* p = probs + at;
    const int lo = (window > 0 && pos - window + 1 > 0 ? pos - window + 1 : 0) - base, last = pos - base;  // inclusive, as indices
    const int mine = lo + ((int(threadIdx.x) - (base + lo) % n + n) % n);
    float mx = -INFINITY;
    for (int r = mine; r <= last; r += n) mx = fmaxf(mx, s[r]);
    mx = block_max(mx);
    float sum = 0;
    for (int r = mine; r <= last; r += n) sum += expf(s[r] - mx);
    sum = block_sum(sum);
    const int32_t* own = tiles + size_t(a.seq[seg]) * ring;
    const int j0 = (base >> 4) % ring;
    for (int r = threadIdx.x; r < stride; r += n) {
        if (r < lo || r > last) {
            p[r] = 0;
            continue;
        }
        const int j = j0 + (r >> 4), q = base + r;
        const float scale = q < KV4_FIRST ? vscale[(size_t(q < 8 ? own[0] : first[a.seq[seg]]) * 16 + (q & 7)) * kvh + kh]
                                          : vscale[(size_t(own[j < ring ? j : j - ring]) * 16 + (r & 15)) * kvh + kh];
        p[r] = __half_as_ushort(__float2half(expf(s[r] - mx) / sum * (scale * BOOST)));
    }
}

// The weighted sum of a 4-bit layer's values, in the turned basis, as attend_wide_k: block (NDT-sixteens chunk of the
// head x blocks of groups, key head, segment), a warp NDT sixteens (NDT 1: as attend32_k), a lane's rows g and g + 8 of
// sixteen k the values c0 + 2 NDT g + 2k and + 1: its NDT bytes of a position are one load, four positions a tile
// (2t, 2t + 1, 2t + 8, 2t + 9), and byte k of the four is its left operand's eight codes of sixteen k. The tiles are
// added in order. A segment that reads the sequence's first tile leaves it out: kv4_unrot_k adds it.
// -> turned[row][head][head_dim]
template <int NDT>
struct Bytes;
template <>
struct Bytes<1> {
    using T = uint8_t;
};
template <>
struct Bytes<2> {
    using T = uint16_t;
};
template <>
struct Bytes<4> {
    using T = uint32_t;
};

template <int S, int NDT>
__device__ __forceinline__ void kv4_attend_steps(float c[NDT][4], int& r, int stride, const int32_t* mine, int j0, int ring, const uint8_t* vals,
                                                 size_t row, const uint16_t* pq, bool has) {
    using W = typename Bytes<NDT>::T;
    for (; r + 16 * S <= stride; r += 16 * S) {
        int32_t where[S];
        uint32_t x[S][4], b0[S], b1[S];
#pragma unroll
        for (int u = 0; u < S; ++u) {
            const int j = j0 + (r >> 4) + u;
            where[u] = mine[j < ring ? j : j - ring];
            b0[u] = has ? *(const uint32_t*)(pq + r + 16 * u) : 0, b1[u] = has ? *(const uint32_t*)(pq + r + 16 * u + 8) : 0;
        }
#pragma unroll
        for (int u = 0; u < S; ++u) {
            const uint8_t* va = vals + size_t(where[u]) * 16 * row;  // position 2t of the tile
            x[u][0] = *(const W*)va, x[u][1] = *(const W*)(va + row), x[u][2] = *(const W*)(va + 8 * row), x[u][3] = *(const W*)(va + 9 * row);
        }
#pragma unroll
        for (int u = 0; u < S; ++u)
#pragma unroll
            for (int k = 0; k < NDT; ++k) {
                const uint32_t sel = k | (4 + k) << 4;  // byte k of two positions
                const uint32_t w = (__byte_perm(x[u][0], x[u][1], sel) & 0xffffu) | __byte_perm(x[u][2], x[u][3], sel) << 16;
                const uint32_t lo = kv4_bytes(w), hi = kv4_bytes(w >> 16);  // positions 2t, 2t + 1; 2t + 8, 2t + 9
                mma(c[k], i8x2_of(lo, 0x4240), i8x2_of(lo, 0x4341), i8x2_of(hi, 0x4240), i8x2_of(hi, 0x4341), b0[u], b1[u]);
            }
    }
}

template <int AHEAD, int NDT>
__global__ void kv4_attend_k(const uint16_t* probs, const uint8_t* values, float* out, const int32_t* tiles, Att a, int heads, int kvh, int hd,
                             int ring) {
    const int seg = blockIdx.z, rep = heads / kvh, nq = rep * a.cnt[seg], stride = a.stride[seg], base = a.base[seg];
    const int chunks = hd / (16 * NDT), dc = blockIdx.x % chunks, group = (blockIdx.x / chunks) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int kh = blockIdx.y, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    if (group * 8 >= nq) return;
    const int query = group * 8 + g;
    const bool has = query < nq;
    const size_t c0 = size_t(16 * NDT) * dc, row = size_t(kvh) * hd / 2;  // a position's bytes
    const uint8_t* vals = values + 2 * t * row + size_t(kh) * hd / 2 + c0 / 2 + NDT * g;
    const int32_t* mine = tiles + size_t(a.seq[seg]) * ring;
    const uint16_t* pq = probs + a.off[seg] + (size_t(kh) * nq + (has ? query : 0)) * stride + 2 * t;
    const int j0 = (base >> 4) % ring;
    float c[NDT][4];
#pragma unroll
    for (int k = 0; k < NDT; ++k) c[k][0] = c[k][1] = c[k][2] = c[k][3] = 0;
    int r = base == 0 ? 16 : 0;  // the first tile is kv4_unrot_k's
    kv4_attend_steps<AHEAD, NDT>(c, r, stride, mine, j0, ring, vals, row, pq, has);
    if (AHEAD > 8) kv4_attend_steps<8, NDT>(c, r, stride, mine, j0, ring, vals, row, pq, has);
    if (AHEAD > 2) kv4_attend_steps<2, NDT>(c, r, stride, mine, j0, ring, vals, row, pq, has);
    kv4_attend_steps<1, NDT>(c, r, stride, mine, j0, ring, vals, row, pq, has);
#pragma unroll
    for (int k = 0; k < NDT; ++k)
        for (int i = 0; i < 4; ++i) {
            const int q = group * 8 + 2 * t + (i & 1), d = int(c0) + 2 * NDT * g + 2 * k + (i >= 2 ? 1 : 0);
            if (q < nq) out[(size_t(a.row0[seg] + q / rep) * heads + kh * rep + q % rep) * hd + d] = c[k][i] * (1.0f / BOOST);
        }
}

// Attention's output of a 4-bit layer: the weighted sum turned back (s * H sum / sqrt(hd)), and the first tile's share
// (its weights times its int8 values, in order) where the row's segment reads it. A warp a (query head, row), eight to a
// block: lane l holds values V l .. V l + V - 1 of the head (V = hd / 32), so the transform's first steps are a lane's
// own and the others go between lanes; no shared memory, no waits for the block.
template <int HD>
__global__ void __launch_bounds__(256) kv4_unrot_k(const float* turned, float* out, const uint16_t* probs, const uint8_t* values,
                                                   const int32_t* tiles, const int32_t* first, const float* signs, Att a, int heads, int kvh,
                                                   int ring, int pairs) {
    constexpr int V = HD / 32;
    const int pair = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (pair >= pairs) return;
    const int head = pair % heads, row = pair / heads, rep = heads / kvh, kh = head / rep, seg = seg_of_row(a, row), d0 = V * lane;
    const size_t at = (size_t(row) * heads + head) * HD + d0;
    float x[V];
#pragma unroll
    for (int i = 0; i < V; ++i) x[i] = turned[at + i];
#pragma unroll
    for (int h = 1; h < V; h <<= 1)
#pragma unroll
        for (int i = 0; i < V; ++i)
            if (!(i & h)) {
                const float lo = x[i], hi = x[i | h];
                x[i] = lo + hi, x[i | h] = lo - hi;
            }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {  // value pairs h = V m apart: the lanes l and l ^ m
        const bool upper = lane & m;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            const float other = __shfl_xor_sync(0xffffffffu, x[i], m);
            x[i] = upper ? other - x[i] : x[i] + other;
        }
    }
    float o[V];
#pragma unroll
    for (int i = 0; i < V; ++i) o[i] = x[i] * (1.0f / sqrtf(float(HD))) * signs[kh * HD + d0 + i];
    if (a.base[seg] == 0) {
        const int nq = rep * a.cnt[seg], query = (row - a.row0[seg]) * rep + head % rep;
        const uint16_t* p = probs + a.off[seg] + (size_t(kh) * nq + query) * a.stride[seg];
        const size_t bytes = size_t(16) * kvh * HD / 2;
        const int8_t* va = (const int8_t*)values + tiles[size_t(a.seq[seg]) * ring] * bytes + size_t(kh) * HD + d0;
        const int8_t* vb = (const int8_t*)values + first[a.seq[seg]] * bytes + size_t(kh) * HD + d0;
        float f[V];
#pragma unroll
        for (int i = 0; i < V; ++i) f[i] = 0;
        for (int r = 0; r < KV4_FIRST; ++r) {
            const float w = __half2float(__ushort_as_half(p[r]));
            const int8_t* v = (r < 8 ? va : vb) + size_t(r & 7) * kvh * HD;
#pragma unroll
            for (int i = 0; i < V; ++i) f[i] += w * float(v[i]);
        }
#pragma unroll
        for (int i = 0; i < V; ++i) o[i] += f[i] * (1.0f / BOOST);
    }
#pragma unroll
    for (int i = 0; i < V; ++i) out[at + i] = o[i];
}

// The format's constants (sclmd/engine.py checks them against sclmd/kvsim.py): the 16 levels, the 15 midpoints, the unit.
API int sclmd_kv4_format(int* levels, float* cuts, float* unit) {
    const uint32_t w[4] = {KV4_W0, KV4_W1, KV4_W2, KV4_W3};
    for (int i = 0; i < 16; ++i) levels[i] = int8_t(uint8_t(w[i >> 2] >> (8 * (i & 3))));
    for (int i = 0; i < 15; ++i) cuts[i] = kv4_cut_host[i];
    *unit = KV4_UNIT;
    return KV4_FIRST;
}

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
    if (tokens > 32) {
        // more than four groups on the fp16 instruction: 32 rows a launch (the same bits as a 32-row pass; a wider
        // kernel was no faster a group there: bench/mma_wide.cu, 0.82-1.0 of four groups' time a group at best, and
        // only with another order of the sums)
        const size_t group = size_t(a.cols / 16) * 32;  // a group's words in xf
        for (int t0 = 0; t0 < tokens; t0 += 32) {
            const int n = tokens - t0 < 32 ? tokens - t0 : 32;
            gemm(xf + size_t(t0 / 8) * group, fps + t0, n, a, ya + size_t(t0) * a.rows, b, yb ? yb + size_t(t0) * b.rows : nullptr, c,
                 yc ? yc + size_t(t0) * c.rows : nullptr);
        }
        return;
    }
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

// Strips a block of gemm8_wide_k: 4 (bench/mma_wide.cu), or SCLMD_WIDE_STRIPS (measurements), within its 32 warps
// (`warps`: a strip's warps, segs * sets).
static int wide_strips(int warps) {
    static const int want = getenv("SCLMD_WIDE_STRIPS") ? atoi(getenv("SCLMD_WIDE_STRIPS")) : GEMM_STRIPS;
    const int most = WIDE_WARPS / warps;
    return want < 1 ? 1 : want > most ? most : want;
}

// The warps a launch of gemm8_wide_k should have at least (SCLMD_WIDE_TARGET for measurements): 4,096, 49 for each
// of the card's 84 multiprocessors.
static int wide_target() {
    static const int want = getenv("SCLMD_WIDE_TARGET") ? atoi(getenv("SCLMD_WIDE_TARGET")) : 4096;
    return want;
}

// gemm on the integer instruction (xf as prep_k wrote it with sigma), with the same launch
static void gemm8(const uint64_t* xf64, const float* fps, int tokens, const Mat& a, float* ya, const Mat& b = Mat{}, float* yb = nullptr,
                  const Mat& c = Mat{}, float* yc = nullptr) {
    const uint32_t* xf = reinterpret_cast<const uint32_t*>(xf64);
    if (tokens > 32) {  // more than four groups: gemm8_wide_k, the groups four a warp
        const int groups = (tokens + 7) / 8, sets = groups <= 8 ? 2 : groups <= 16 ? 4 : 8, all = (a.rows + b.rows + c.rows) / 16;
        // a strip's tiles split over as few warps as keep the card busy: a matrix of few strips and long rows (down,
        // o) on one warp a strip had too few warps (down at 64 rows 14.1 ms a pass against 3.9 at 32)
        int segs = 1;
        while (segs < segs_for(a.cols) && all * segs * sets < wide_target() && 2 * segs * sets <= WIDE_WARPS) segs *= 2;
        const int strips = wide_strips(segs * sets), blocks = (all + strips - 1) / strips, threads = 32 * segs * sets * strips;
        const size_t frags = size_t(threads / 32) * 2 * 32 * 16;
        static bool raised = false;  // the shared memory past 48 KB a block is asked for once
        if (!raised) {
            cudaFuncSetAttribute(gemm8_wide_k<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * 1024);
            cudaFuncSetAttribute(gemm8_wide_k<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * 1024);
            cudaFuncSetAttribute(gemm8_wide_k<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * 1024);
            raised = true;
        }
        if (sets == 2)
            gemm8_wide_k<2><<<blocks, threads, frags>>>(a, b, c, ya, yb, yc, xf, fps, segs, tokens, strips);
        else if (sets == 4)
            gemm8_wide_k<4><<<blocks, threads, frags>>>(a, b, c, ya, yb, yc, xf, fps, segs, tokens, strips);
        else
            gemm8_wide_k<8><<<blocks, threads, frags>>>(a, b, c, ya, yb, yc, xf, fps, segs, tokens, strips);
        return;
    }
    const int segs = segs_for(a.cols), all = (a.rows + b.rows + c.rows) / 16;
    const int strips = GEMM_STRIPS * segs * 32 <= 1024 ? GEMM_STRIPS : 1024 / (segs * 32), blocks = (all + strips - 1) / strips;
    if (tokens <= 8)
        gemm8_k<1><<<blocks, 32 * segs * strips>>>(a, b, c, ya, yb, yc, xf, fps, segs, tokens, strips);
    else if (tokens <= 16)
        gemm8_k<2><<<blocks, 32 * segs * strips>>>(a, b, c, ya, yb, yc, xf, fps, segs, tokens, strips);
    else
        gemm8_k<4><<<blocks, 32 * segs * strips>>>(a, b, c, ya, yb, yc, xf, fps, segs, tokens, strips);
}

// the layers' matrices: on the integer instruction when the model's option is on
static void gemm_of(const Model& m, int tokens, const Mat& a, float* ya, const Mat& b = Mat{}, float* yb = nullptr, const Mat& c = Mat{},
                    float* yc = nullptr) {
    if (m.int8 > 0)
        gemm8(m.xf, m.fps, tokens, a, ya, b, yb, c, yc);
    else
        gemm(m.xf, m.fps, tokens, a, ya, b, yb, c, yc);
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
                        float* h_out, float* v_out, float* fps, uint64_t* xf, float eps, float sigma = 0.0f) {
    const dim3 grid(n / rot.block, rows);
    float* sums = fps + MAX_TOKENS;
    if (rows >= two_phase_from() && grid.x > 1) {
        prep_k<<<dim3(1, rows), rot.block>>>(a, b, w, owed.y, owed.post, owed.scalar, h_out, v_out, rot.signs, fps, xf, n, eps, mode, 1, sums, sigma);
        prep_k<<<grid, rot.block>>>(a, b, w, owed.y, owed.post, owed.scalar, h_out, v_out, rot.signs, fps, xf, n, eps, mode, 2, sums, sigma);
    } else {
        prep_k<<<grid, rot.block>>>(a, b, w, owed.y, owed.post, owed.scalar, h_out, v_out, rot.signs, fps, xf, n, eps, mode, 0, sums, sigma);
    }
}

// A layer matrix's input: int8 when the model's option is on (the head's input goes through prep_head, fp16)
static void prep(const Model& m, const float* a, const float* b, const float* w, int n, const Rot& rot, int mode, int tokens,
                 const Owed& owed = Owed{}, float* h_out = nullptr, float* v_out = nullptr) {
    launch_prep(tokens, a, b, w, n, rot, mode, owed, h_out, v_out, m.fps, m.xf, m.eps, m.int8);
}

static void prep_head(const Model& m, const float* a, const float* b, const float* w, int n, const Rot& rot, int mode, int tokens,
                      const Owed& owed = Owed{}, float* h_out = nullptr, float* v_out = nullptr) {
    launch_prep(tokens, a, b, w, n, rot, mode, owed, h_out, v_out, m.fps, m.xf, m.eps, 0.0f);
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
// Query groups from which the scores make each key tile once for a block's groups (scores_once_k; SCLMD_ONCE_FROM
// sets it for measurements, a large number: never). The bits are the same either way.
static int once_from() {
    static const int from = getenv("SCLMD_ONCE_FROM") ? atoi(getenv("SCLMD_ONCE_FROM")) : ONCE_FROM;
    return from;
}

static void scores(int tiles, int groups, const uint16_t* qhat, const Layer& l, float* out, const Att& a, int heads) {
    const int hd = l.head_dim;
    if (groups >= once_from() && (hd == 256 || hd == 512) && (l.keys || (hd == 512 && l.turned % 32 == 0 && l.turned <= 128))) {
        int blocks = 0;
        for (int i = 0; i < a.n; ++i) blocks += (a.stride[i] / 16 + ONCE_TILES - 1) / ONCE_TILES;
        const dim3 grid(blocks, l.kv_heads, (groups + 7) / 8);
        if (!l.keys)
            scores_once_k<512, true><<<grid, 256>>>(qhat, l.values, l.vscale, out, l.tiles, a, heads, l.kv_heads, l.ring, l.trig, l.k_norm, l.turned);
        else if (hd == 512)
            scores_once_k<512, false><<<grid, 256>>>(qhat, l.keys, l.kscale, out, l.tiles, a, heads, l.kv_heads, l.ring, nullptr, nullptr, 0);
        else
            scores_once_k<256, false><<<grid, 256>>>(qhat, l.keys, l.kscale, out, l.tiles, a, heads, l.kv_heads, l.ring, nullptr, nullptr, 0);
        return;
    }
    const int per = groups > GROUPS_A_BLOCK ? GROUPS_A_BLOCK : groups;
    const dim3 grid(tiles, l.kv_heads, (groups + per - 1) / per);
    if (l.keys)
        scores32_k<<<grid, 32 * per>>>(qhat, l.keys, l.kscale, out, l.tiles, a, heads, l.kv_heads, l.head_dim, l.ring);
    else
        scores_made32_k<<<grid, 32 * per>>>(qhat, l.values, l.vscale, out, l.tiles, a, heads, l.kv_heads, l.head_dim, l.ring, l.trig, l.k_norm,
                                            l.turned);
}

static void attend(int groups, const uint16_t* probs, const int8_t* values, float* out, const int32_t* table, const Att& a, int heads, int kvh,
                   int hd, int ring) {
    const int per = groups > GROUPS_A_BLOCK ? GROUPS_A_BLOCK : groups, blocks = (groups + per - 1) / per;
    // sixteens a warp (SCLMD_ATTEND_NDT sets it for measurements: 1, 2 or 4); the bits are the same
    static const int force = getenv("SCLMD_ATTEND_NDT") ? atoi(getenv("SCLMD_ATTEND_NDT")) : 0;
    const int ndt = force ? force : groups >= ATTEND_NDT4_FROM ? 4 : groups >= ATTEND_NDT2_FROM ? 2 : 1;
    if (ndt == 4 && hd % 64 == 0) {
        attend_wide_k<8, 4><<<dim3(hd / 64 * blocks, kvh, a.n), 32 * per>>>(probs, values, out, table, a, heads, kvh, hd, ring);
        return;
    }
    if (ndt == 2 && hd % 32 == 0) {
        attend_wide_k<16, 2><<<dim3(hd / 32 * blocks, kvh, a.n), 32 * per>>>(probs, values, out, table, a, heads, kvh, hd, ring);
        return;
    }
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
    gemm_of(*m, tokens, *mat, y);
    return int(cudaGetLastError());
}

// A 4-bit layer's (Layer::kv4) attention, in two calls a pass. kv4_write: every row's queries (turned into qhat, as
// they are into qplain: m.qhat past the rows' turned ones) and keys and values. kv4_read: the attention of the rows of
// `a` (their queries at qhat and qplain), into attn; `turned` is room for as much (m.attn past the rows' output).
static void kv4_write(const Model& m, const Layer& l, const Att& a, int tokens, const float* yq, const float* yk, const float* yv) {
    const int hd = l.head_dim, kvh = l.kv_heads;
    kv4_heads_k<<<dim3(m.heads + 2 * kvh, tokens), hd>>>(yq, yk, yv, l.q_norm, l.k_norm, l.inv_freq, m.qhat, m.qhat + size_t(tokens) * m.heads * hd,
                                                       (uint8_t*)l.keys, (uint8_t*)l.values, l.kscale, l.vscale, l.tiles, l.first, l.kv4_signs, a,
                                                       m.heads, kvh, l.ring, m.eps);
}

static void kv4_read(const Layer& l, const Att& a, int groups, const uint16_t* qhat, const uint16_t* qplain, float* attn, float* turned, float* scores,
                     uint16_t* probs, int heads, int window) {
    const int hd = l.head_dim, kvh = l.kv_heads, rep = heads / kvh, rows = a.row0[a.n - 1] + a.cnt[a.n - 1];
    const uint8_t *keys = (const uint8_t*)l.keys, *values = (const uint8_t*)l.values;
    if (groups >= once_from() && hd == 256) {
        int blocks = 0;
        for (int i = 0; i < a.n; ++i) blocks += (a.stride[i] / 16 + ONCE_TILES - 1) / ONCE_TILES;
        kv4_scores_once_k<256><<<dim3(blocks, kvh, (groups + 7) / 8), 256>>>(qhat, qplain, keys, l.kscale, scores, l.tiles, l.first, a, heads, kvh, l.ring);
    } else {
        const int per = groups > GROUPS_A_BLOCK ? GROUPS_A_BLOCK : groups;
        const dim3 grid(tiles_of(a), kvh, (groups + per - 1) / per);
#define KV4_SCORES(HD) kv4_scores_k<HD><<<grid, 32 * per>>>(qhat, qplain, keys, l.kscale, scores, l.tiles, l.first, a, heads, kvh, l.ring)
        switch (hd) {
            case 32: KV4_SCORES(32); break;
            case 64: KV4_SCORES(64); break;
            case 128: KV4_SCORES(128); break;
            default: KV4_SCORES(256); break;  // sclmd/engine.py takes these four
        }
#undef KV4_SCORES
    }
    tick(P_SCORES);
    kv4_softmax_k<<<dim3(rep * rows, kvh), 256>>>(scores, probs, l.vscale, l.tiles, l.first, a, heads, kvh, window, l.ring);
    tick(P_SOFTMAX);
    const int per = groups > GROUPS_A_BLOCK ? GROUPS_A_BLOCK : groups, blocks = (groups + per - 1) / per;
    static const int force = getenv("SCLMD_ATTEND_NDT") ? atoi(getenv("SCLMD_ATTEND_NDT")) : 0;
    const int ndt = force ? force : groups >= ATTEND_NDT4_FROM ? 4 : groups >= ATTEND_NDT2_FROM ? 2 : 1;
    if (ndt == 4 && hd % 64 == 0)
        kv4_attend_k<8, 4><<<dim3(hd / 64 * blocks, kvh, a.n), 32 * per>>>(probs, values, turned, l.tiles, a, heads, kvh, hd, l.ring);
    else if (ndt == 2 && hd % 32 == 0)
        kv4_attend_k<16, 2><<<dim3(hd / 32 * blocks, kvh, a.n), 32 * per>>>(probs, values, turned, l.tiles, a, heads, kvh, hd, l.ring);
    else
        kv4_attend_k<32, 1><<<dim3(hd / 16 * blocks, kvh, a.n), 32 * per>>>(probs, values, turned, l.tiles, a, heads, kvh, hd, l.ring);
    const int pairs = heads * rows;
#define KV4_UNROT(HD) kv4_unrot_k<HD><<<(pairs + 7) / 8, 256>>>(turned, attn, probs, values, l.tiles, l.first, l.kv4_signs, a, heads, kvh, l.ring, pairs)
    switch (hd) {
        case 32: KV4_UNROT(32); break;
        case 64: KV4_UNROT(64); break;
        case 128: KV4_UNROT(128); break;
        default: KV4_UNROT(256); break;
    }
#undef KV4_UNROT
    tick(P_ATTEND);
}

// The pass: m.h holds the scaled embeddings of `tokens` tokens, which are the rows of `segs`; afterwards it holds the
// last layer's output, m.hn that output after the final norm, m.logits the logits, and every layer's keys and
// values have the new positions of each sequence (whose tiles the caller has provided). Launches only; nothing waits.
// A layer's two residual steps are not kernels of their own: each is taken by the kernel that prepares the next
// matrix input from the residual stream, which writes the stream's next state to the other of m.h and m.h2.
static int pass(const Model* mp, int tokens, const Segs* segs, int logits_from);

// Rows from .. from + count - 1 of a pass as segments of their own (rows counted from `from`): a segment cut at an
// edge keeps its sequence, with the position of its first row here.
static Segs rows_of(const Segs& g, int from, int count) {
    Segs s{};
    for (int i = 0; i < g.n; ++i) {
        const int lo = g.row0[i] > from ? g.row0[i] : from, end = g.row0[i] + g.cnt[i] < from + count ? g.row0[i] + g.cnt[i] : from + count;
        if (lo >= end) continue;
        s.row0[s.n] = lo - from, s.cnt[s.n] = end - lo, s.pos0[s.n] = g.pos0[i] + lo - g.row0[i], s.seq[s.n] = g.seq[i];
        ++s.n;
    }
    return s;
}

// The first row a pass of more than ATT_ROWS rows has logits of: from the group of eight logits_from is in, and of
// the last ATT_ROWS rows at most (m.logits has ATT_ROWS rows); a pass of up to ATT_ROWS rows keeps them at their rows.
static int logits_first(int tokens, int logits_from) {
    const int asked = logits_from < 0 ? 0 : logits_from / 8 * 8, last = (tokens - ATT_ROWS + 7) / 8 * 8;
    return asked > last ? asked : last;
}

API int sclmd_logits_first(int tokens, int logits_from) { return tokens > ATT_ROWS ? logits_first(tokens, logits_from) : (logits_from < 0 ? 0 : logits_from / 8 * 8); }

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
        gemm_of(m, tokens, l.q, yq, l.k, yk, l.v, yv);  // a full-attention layer's v has no rows
        tick(P_QKV);
        const Att a = att_for(*segs, l.full, m.heads, m.window - 1);
        const int rep = m.heads / kvh;
        // every row's keys and values first, then attention over as many rows at a time as the scores' memory holds,
        // ATT_ROWS at least (a row reads only positions up to its own, whose keys are written by then; a sliding
        // layer's ring holds window + rows positions, so what the pass writes is not where its first row still reads).
        // The chunks go to the kernels of a pass of their width (scores_once_k, attend_wide_k from enough groups),
        // which give a query the same bits in any chunk (tests/test_wide.py against passes of one row).
        // A 4-bit layer (brief K3, Layer::kv4) goes the same way: kv4_write for every row, then kv4_read a chunk at a
        // time. Its plain queries sit past every row's turned ones (m.qhat + tokens rows) and its turned sums past every
        // row's output (m.attn + tokens rows), a chunk's at its own rows there (the Engine sizes both for twice its rows).
        // When the ring goes round within the pass, writing position 16 ring + p over the ring tile 0 takes the first
        // tile's int8 position p / 2 and scale p; every row of the pass reads from p + 2 on, as its ring holds window +
        // rows (tests/test_kv4_wide.py).
        if (l.kv4)
            kv4_write(m, l, a, tokens, yq, yk, yv);
        else
            heads_k<<<dim3(m.heads + 2 * kvh, tokens), hd>>>(yq, yk, l.v.rows ? yv : nullptr, l.q_norm, l.k_norm, l.inv_freq, m.qhat, l.keys,
                                                           l.values, l.kscale, l.vscale, l.tiles, a, m.heads, kvh, l.ring, m.eps,
                                                           l.keys ? nullptr : l.k_norm, l.turned);
        tick(P_HEADS);
        for (int c0 = 0, cn; c0 < tokens; c0 += cn) {
            cn = tokens - c0;
            Att ac = a;
            if (tokens > ATT_ROWS)
                for (;;) {  // as many rows as the scores' memory holds (in steps of ATT_ROWS)
                    ac = att_for(rows_of(*segs, c0, cn), l.full, m.heads, m.window - 1);
                    const long long need = ac.off[ac.n - 1] + (long long)m.heads * ac.cnt[ac.n - 1] * ac.stride[ac.n - 1];
                    if (cn <= ATT_ROWS || need <= m.scores_room) break;
                    cn = cn - ATT_ROWS > ATT_ROWS ? cn - ATT_ROWS : ATT_ROWS;
                }
            const int groups = (rep * ac.most + 7) / 8;
            uint16_t* qhat = m.qhat + size_t(c0) * nqo;
            if (l.kv4) {
                kv4_read(l, ac, groups, qhat, m.qhat + size_t(tokens + c0) * nqo, m.attn + size_t(c0) * nqo, m.attn + size_t(tokens + c0) * nqo,
                         m.scores, m.probs, m.heads, m.window);
                continue;
            }
            scores(tiles_of(ac), groups, qhat, l, m.scores, ac, m.heads);
            tick(P_SCORES);
            softmax_k<<<dim3(rep * cn, kvh), 256>>>(m.scores, m.probs, l.vscale, l.tiles, ac, m.heads, kvh, l.full ? 0 : m.window, l.ring);
            tick(P_SOFTMAX);
            attend(groups, m.probs, l.values, m.attn + size_t(c0) * nqo, l.tiles, ac, m.heads, kvh, hd, l.ring);
            tick(P_ATTEND);
        }
        prep(m, m.attn, nullptr, nullptr, nqo, l.rot_o, 1, tokens);
        tick(P_INPUT);
        gemm_of(m, tokens, l.o, m.y);
        tick(P_O);
        owed = Owed{m.y, l.post_attn_ln, 1.0f};
        // feed-forward
        prep(m, h, nullptr, l.pre_ff_ln, n, l.rot_mlp, 0, tokens, owed, spare);
        std::swap(h, spare);
        tick(P_INPUT);
        gemm_of(m, tokens, l.gate, m.y, l.up, m.y2);
        tick(P_GATE_UP);
        prep(m, m.y, m.y2, nullptr, m.mlp, l.rot_down, 2, tokens);
        tick(P_INPUT);
        gemm_of(m, tokens, l.down, m.y);
        tick(P_DOWN);
        owed = Owed{m.y, l.post_ff_ln, l.scalar};
    }
    // the last residual step, the final norm (kept in m.hn) and the head. Two steps a layer: the stream ends in m.h
    prep_head(m, h, nullptr, m.norm, n, m.rot_head, 0, tokens, owed, spare, m.hn);
    int first = logits_from < 0 ? 0 : logits_from / 8 * 8;  // the first row of the group of eight the logits start in
    // a pass of more than ATT_ROWS rows: the logits of its last ATT_ROWS at most (from a group's first row on), from
    // m.logits' first row (logits_first() says which)
    const int lbase = tokens > ATT_ROWS ? (first = logits_first(tokens, logits_from)) : 0;
    if (first < tokens) {
        float* logits = m.logits + size_t(first - lbase) * m.vocab;
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
                                             nullptr, att_for(at, true, d.heads, 0), d.heads, kvh, 1, d.eps, src.keys ? nullptr : src.k_norm,
                                             src.turned);
        const Att a = att_for(read, l.full, d.heads, window);
        scores(tiles_of(a), groups, d.qhat, src, d.scores, a, d.heads);
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
