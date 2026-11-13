// The engine: one pass of the Gemma 4 decoder over up to eight new tokens, on the GPU, from pair codes.
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
#include <cstdint>
#include <utility>

#define API extern "C" __declspec(dllexport)

static const int MAX_TOKENS = 8;  // the eight columns of the tensor-core instruction

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
    uint16_t *keys, *values;  // [slots][kv_heads * head_dim] fp16: this layer's keys and values by position
    float scalar;
    int full;      // 1: attends to every earlier position; 0: to the last `window`
    int head_dim, kv_heads;
    int slots;     // sliding: positions are kept modulo this; full: the capacity
    int pad;
};

struct Model {
    Layer* layers;
    int n_layers, hidden, mlp, heads, window, vocab;
    float eps, softcap;
    Mat head;
    Rot rot_head;
    const float* norm;  // the final norm [hidden]
    // work memory
    float* h;        // [8][hidden] the residual stream of the tokens of this pass (in: the embeddings; out: the last layer's output)
    float* h2;       // [8][hidden] where the residual stream's next step is written (the two take turns)
    float* fps;      // [8] the power of two an input was divided by to fit fp16
    uint64_t* xf;    // [widest input / 16][32] a matrix's input as the instruction reads it
    float* y;        // [8][widest output] a matrix's output
    float* y2;       // [8][mlp] (up's output beside gate's)
    uint16_t* qhat;  // [8][heads * widest head] the queries, normalized and rotated (fp16)
    float* scores;   // [kv heads * queries][positions in whole tiles] attention scores
    uint16_t* probs; // the same layout: attention weights (fp16)
    float* attn;     // [8][heads * widest head] attention's output
    float* hn;       // [8][hidden] the last output after the final norm (what the drafter is handed)
    float* logits;   // [8][vocab]
};

__device__ __forceinline__ void mma(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

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

// The input of a matrix with n columns, in one kernel: block (run of blockDim values, token), where blockDim is the
// rotation's block, or without a rotation (signs null) a power of two that divides n. Three steps:
//  1. the residual step still owed, if any: with y, h = (a + RMS norm of y times post) * scalar, and the block
//     writes its own run of h to h_out (which is not a). Without y, h = a.
//  2. the value: mode 0: RMS norm of h times w; 1: h as it is; 2: gelu(h) * b. Written to v_out if given.
//  3. signs and the Hadamard transform over the run (with a rotation), over the power of two that brings the whole
//     input's length under 2^14, so that no value leaves fp16's range (fps[token]); then fp16 in the order the
//     instruction reads: for tile j, lane 4 * token + t of xf holds the values at columns 16j + 2t, + 1, + 8, + 9.
// Sums over the whole input are made by every block for itself: they are small next to the matrix that follows,
// and a kernel less is worth more.
__global__ void prep_k(const float* a, const float* b, const float* w, const float* y, const float* post, float scalar, float* h_out,
                       float* v_out, const float* signs, float* fps, uint64_t* xf, int n, float eps, int mode) {
    __shared__ float buf[1024];
    const int bd = blockDim.x, token = blockIdx.y, i = threadIdx.x, col = blockIdx.x * bd + i;
    const size_t at = size_t(token) * n;
    float ry = 0;
    if (y) {
        float ss = 0;
        for (int j = i; j < n; j += bd) ss += y[at + j] * y[at + j];
        ry = powf(block_sum(ss) / n + eps, -0.5f);
    }
#define STREAM(j) (y ? (a[at + (j)] + y[at + (j)] * ry * post[j]) * scalar : a[at + (j)])
    float r = 1.0f;
    if (mode == 0) {
        float ss = 0;
        for (int j = i; j < n; j += bd) {
            const float hv = STREAM(j);
            ss += hv * hv;
        }
        r = powf(block_sum(ss) / n + eps, -0.5f);
    }
#define VALUE(j, hv) (mode == 0 ? (hv) * r * w[j] : mode == 1 ? (hv) : gelu_tanh(hv) * b[at + (j)])
    float len = 0;
    for (int j = i; j < n; j += bd) {
        const float hv = STREAM(j), v = VALUE(j, hv);
        len += v * v;
    }
    len = sqrtf(block_sum(len));
    const float fp = len > 16384.0f ? exp2f(ceilf(log2f(len / 16384.0f))) : 1.0f;
    if (col == 0) fps[token] = fp;
    const float hv = STREAM(col), v = VALUE(col, hv);
#undef STREAM
#undef VALUE
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
        xf[size_t(col >> 4) * 32 + 4 * token + t] = word;
    }
}

// y[token][row] = scale[row] * fps[token] * sum over columns of W[row][col] * x[token][col], one block a strip of 16
// rows, its `segs` warps splitting the strip's tiles (a multiple of 4 each) between them. Up to three matrices that
// share the input go in one launch: the blocks are the first one's strips, then the second's, then the third's.
__global__ void gemm_k(Mat m0, Mat m1, Mat m2, float* y0, float* y1, float* y2, const uint64_t* xf, const float* fps, int segs, int tokens) {
    __shared__ uint32_t slut[256];
    __shared__ float part[8][32][4];
    int strip = blockIdx.x;
    Mat m = m0;
    float* y = y0;
    if (strip >= m0.rows / 16) {
        strip -= m0.rows / 16, m = m1, y = y1;
        if (strip >= m1.rows / 16) strip -= m1.rows / 16, m = m2, y = y2;
    }
    for (int i = threadIdx.x; i < (1 << m.bits); i += blockDim.x) slut[i] = m.lut[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tiles = m.cols / 16, per = tiles / segs, first = warp * per;
    float c[4] = {0, 0, 0, 0};
    const uint64_t* xw = xf + size_t(first) * 32 + lane;
    if (m.bits == 8) {
        const uint32_t* cw = m.codes + (size_t(strip) * tiles + first) * 32 + lane;
        for (int j = 0; j < per; j += 4) {
            uint32_t w[4];
            uint64_t b[4];
            for (int i = 0; i < 4; ++i) w[i] = cw[size_t(j + i) * 32], b[i] = xw[size_t(j + i) * 32];
            for (int i = 0; i < 4; ++i)
                mma(c, slut[w[i] & 255], slut[w[i] >> 8 & 255], slut[w[i] >> 16 & 255], slut[w[i] >> 24], uint32_t(b[i]), uint32_t(b[i] >> 32));
        }
    } else {  // 6 bits a pair: four tiles of a lane are 16 codes in 96 bits, stored as three words
        const uint32_t* cw = m.codes + (size_t(strip) * (tiles / 4) + first / 4) * 96 + lane;
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
        if (token < tokens) y[size_t(token) * m.rows + row] = m.scale[row] * fps[token] * sum;
    }
}

// What attention does to the projections' outputs, block (vector, token) with head_dim threads. Vector v of a token is
// one of its `heads` query heads (v < heads), `kvh` key heads or `kvh` value heads:
//   query: RMS norm times q_norm, rotary position -> qhat[token][head][head_dim] (fp16)
//   key:   RMS norm times k_norm, rotary position -> keys[slot of the position][head] (fp16)
//   value: RMS norm                               -> values[slot][head] (fp16); the value's source is yv (or yk when null)
// The rotary position turns pair (d, d + head_dim / 2) by pos * inv_freq[d].
__global__ void heads_k(const float* yq, const float* yk, const float* yv, const float* q_norm, const float* k_norm, const float* inv_freq,
                        uint16_t* qhat, uint16_t* keys, uint16_t* values, int heads, int kvh, int pos0, int slots, float eps) {
    const int hd = blockDim.x, v = blockIdx.x, token = blockIdx.y, d = threadIdx.x, half = hd / 2;
    const int kind = v < heads ? 0 : v < heads + kvh ? 1 : 2, head = kind == 0 ? v : kind == 1 ? v - heads : v - heads - kvh;
    const float* src = kind == 0 ? yq + (size_t(token) * heads + head) * hd
                                 : (kind == 1 || !yv ? yk : yv) + (size_t(token) * kvh + head) * hd;
    __shared__ float buf[1024];
    const float val = src[d];
    const float r = powf(block_sum(val * val) / hd + eps, -0.5f);
    const int slot = (pos0 + token) % slots;
    if (kind == 2) {
        values[(size_t(slot) * kvh + head) * hd + d] = __half_as_ushort(__float2half(val * r));
        return;
    }
    buf[d] = val * r * (kind == 0 ? q_norm[d] : k_norm[d]);
    __syncthreads();
    const float ang = float(pos0 + token) * inv_freq[d % half];
    const float turned = d < half ? -buf[d + half] : buf[d - half];
    const float out = buf[d] * cosf(ang) + turned * sinf(ang);
    if (kind == 0)
        qhat[(size_t(token) * heads + head) * hd + d] = __half_as_ushort(__float2half(out));
    else
        keys[(size_t(slot) * kvh + head) * hd + d] = __half_as_ushort(__float2half(out));
}

// Attention runs on the tensor cores too, in three kernels. The positions a pass reads are taken in tiles of 16 that
// start at multiples of 16 (`base` is the first tile's position, `stride` the number of positions in the tiles), so a
// query's sums are made of the same tiles in the same order whichever pass it is computed in (alone, or among
// eight). Positions in the tiles that a query may not read, or that hold nothing yet, get weight zero.
// A key head's queries are numbered query = token * rep + (head within the key head's `rep` query heads); eight of
// them are the eight columns of the instruction, and each group of eight has a warp.

// Scores: block (position tile, key head). scores[(key head * nq + query) * stride + (position - base)] = query . key.
__global__ void scores_k(const uint16_t* qhat, const uint16_t* keys, float* scores, int heads, int kvh, int hd, int tokens, int base,
                         int stride, int slots) {
    const int rep = heads / kvh, nq = rep * tokens;
    const int tile = blockIdx.x, kh = blockIdx.y, group = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int p0 = base + 16 * tile;
    // the left operand: rows are the tile's positions (this lane: g and g + 8), columns sixteen of the head's values
    const uint16_t* ka = keys + (size_t((p0 + g) % slots) * kvh + kh) * hd + 2 * t;
    const uint16_t* kb = keys + (size_t((p0 + g + 8) % slots) * kvh + kh) * hd + 2 * t;
    // the right operand: column g is query group * 8 + g
    const int query = group * 8 + g;
    const bool has = query < nq;
    const uint16_t* qa = qhat + (size_t(has ? query / rep : 0) * heads + kh * rep + (has ? query % rep : 0)) * hd + 2 * t;
    float c[4] = {0, 0, 0, 0};
    for (int j = 0; j < hd; j += 16) {
        const uint32_t b0 = has ? *(const uint32_t*)(qa + j) : 0, b1 = has ? *(const uint32_t*)(qa + j + 8) : 0;
        mma(c, *(const uint32_t*)(ka + j), *(const uint32_t*)(kb + j), *(const uint32_t*)(ka + j + 8), *(const uint32_t*)(kb + j + 8), b0, b1);
    }
    for (int i = 0; i < 4; ++i) {  // c[i]: position p0 + g (+ 8), query group * 8 + 2t (+ 1)
        const int q = group * 8 + 2 * t + (i & 1), r = 16 * tile + g + (i >= 2 ? 8 : 0);
        if (q < nq) scores[(size_t(kh) * nq + q) * stride + r] = c[i];
    }
}

// The softmax of a query's scores over the positions it may read, as fp16 in the same layout: block (query, key head).
// Token t at position pos0 + t reads positions up to its own, and in a sliding layer the last `window` of them.
__global__ void softmax_k(const float* scores, uint16_t* probs, int heads, int kvh, int tokens, int pos0, int base, int stride, int window) {
    const int n = blockDim.x, rep = heads / kvh, nq = rep * tokens, query = blockIdx.x, kh = blockIdx.y, pos = pos0 + query / rep;
    const float* s = scores + (size_t(kh) * nq + query) * stride;
    uint16_t* p = probs + (size_t(kh) * nq + query) * stride;
    const int first = (window > 0 && pos - window + 1 > 0 ? pos - window + 1 : 0) - base, last = pos - base;  // inclusive, as indices
    // a thread takes the positions that are its own number modulo the block, so the sums add up in the same order in any pass
    const int mine = first + ((int(threadIdx.x) - (base + first) % n + n) % n);
    float mx = -INFINITY;
    for (int r = mine; r <= last; r += n) mx = fmaxf(mx, s[r]);
    mx = block_max(mx);
    float sum = 0;
    for (int r = mine; r <= last; r += n) sum += expf(s[r] - mx);
    sum = block_sum(sum);
    for (int r = threadIdx.x; r < stride; r += n)
        p[r] = r >= first && r <= last ? __half_as_ushort(__float2half(expf(s[r] - mx) / sum)) : 0;
}

// The weighted sum of the values: block (sixteen of the head's values, key head); a warp walks the position tiles.
// -> out[token][head][head_dim]
__global__ void attend_k(const uint16_t* probs, const uint16_t* values, float* out, int heads, int kvh, int hd, int tokens, int base,
                         int stride, int slots) {
    const int rep = heads / kvh, nq = rep * tokens;
    const int dt = blockIdx.x, kh = blockIdx.y, group = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    // the right operand: rows are the tile's positions, column g is query group * 8 + g
    const int query = group * 8 + g;
    const bool has = query < nq;
    const uint16_t* pq = probs + (size_t(kh) * nq + (has ? query : 0)) * stride + 2 * t;
    // the left operand: rows are values 16 dt + g and + 8 of the head, columns the tile's positions 2t, 2t + 1, 2t + 8, 2t + 9
    const size_t col = size_t(kh) * hd + 16 * dt + g, row = size_t(kvh) * hd;
    int sa = (base + 2 * t) % slots;  // the slot of position 2t of the tile; a sliding layer's slots go round
    float c[4] = {0, 0, 0, 0};
    for (int r = 0; r < stride; r += 16) {
        const int sb = sa + 1 < slots ? sa + 1 : sa + 1 - slots, sc = sa + 8 < slots ? sa + 8 : sa + 8 - slots,
                  sd = sc + 1 < slots ? sc + 1 : sc + 1 - slots;
        const uint16_t *va = values + sa * row + col, *vb = values + sb * row + col, *vc = values + sc * row + col, *vd = values + sd * row + col;
        const uint32_t b0 = has ? *(const uint32_t*)(pq + r) : 0, b1 = has ? *(const uint32_t*)(pq + r + 8) : 0;
        mma(c, uint32_t(va[0]) | uint32_t(vb[0]) << 16, uint32_t(va[8]) | uint32_t(vb[8]) << 16, uint32_t(vc[0]) | uint32_t(vd[0]) << 16,
            uint32_t(vc[8]) | uint32_t(vd[8]) << 16, b0, b1);
        sa += 16;
        while (sa >= slots) sa -= slots;
    }
    for (int i = 0; i < 4; ++i) {  // c[i]: value 16 dt + g (+ 8), query group * 8 + 2t (+ 1)
        const int q = group * 8 + 2 * t + (i & 1), d = 16 * dt + g + (i >= 2 ? 8 : 0);
        if (q < nq) out[(size_t(q / rep) * heads + kh * rep + q % rep) * hd + d] = c[i];
    }
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
    const int segs = segs_for(a.cols);
    gemm_k<<<(a.rows + b.rows + c.rows) / 16, 32 * segs>>>(a, b, c, ya, yb, yc, xf, fps, segs, tokens);
}

// What prep_k is told of the residual step still owed: the matrix output it adds, the norm's weight and the layer's scalar.
struct Owed {
    const float *y = nullptr, *post = nullptr;
    float scalar = 1.0f;
};

// The input of a matrix with n columns from a (and b, w: see prep_k), rotated, in m.xf with its scales in m.fps.
static void prep(const Model& m, const float* a, const float* b, const float* w, int n, const Rot& rot, int mode, int tokens,
                 const Owed& owed = Owed{}, float* h_out = nullptr, float* v_out = nullptr) {
    prep_k<<<dim3(n / rot.block, tokens), rot.block>>>(a, b, w, owed.y, owed.post, owed.scalar, h_out, v_out, rot.signs, m.fps, m.xf, n, m.eps, mode);
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

// The pass: m.h holds the scaled embeddings of `tokens` tokens at positions pos0.. ; afterwards it holds the last
// layer's output, m.hn that output after the final norm, m.logits the logits, and every layer's keys and values have
// the new positions. Launches only; nothing waits.
// A layer's two residual steps are not kernels of their own: each is taken by the kernel that prepares the next
// matrix input from the residual stream, which writes the stream's next state to the other of m.h and m.h2.
API int sclmd_pass(const Model* mp, int tokens, int pos0) {
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
        float *yq = m.y, *yk = m.y + size_t(MAX_TOKENS) * nqo, *yv = yk + size_t(MAX_TOKENS) * nkv;
        gemm(m.xf, m.fps, tokens, l.q, yq, l.k, yk, l.v, yv);  // a full-attention layer's v has no rows
        tick(P_QKV);
        heads_k<<<dim3(m.heads + 2 * kvh, tokens), hd>>>(yq, yk, l.v.rows ? yv : nullptr, l.q_norm, l.k_norm, l.inv_freq, m.qhat, l.keys,
                                                       l.values, m.heads, kvh, pos0, l.slots, m.eps);
        tick(P_HEADS);
        const int last = pos0 + tokens - 1, lo = l.full ? 0 : (pos0 - m.window + 1 > 0 ? pos0 - m.window + 1 : 0);
        const int base = lo & ~15, stride = ((last >> 4) + 1) * 16 - base;  // the positions read, in whole tiles of 16
        const int nq = m.heads / kvh * tokens, groups = (nq + 7) / 8;
        scores_k<<<dim3(stride / 16, kvh), 32 * groups>>>(m.qhat, l.keys, m.scores, m.heads, kvh, hd, tokens, base, stride, l.slots);
        tick(P_SCORES);
        softmax_k<<<dim3(nq, kvh), 256>>>(m.scores, m.probs, m.heads, kvh, tokens, pos0, base, stride, l.full ? 0 : m.window);
        tick(P_SOFTMAX);
        attend_k<<<dim3(hd / 16, kvh), 32 * groups>>>(m.probs, l.values, m.attn, m.heads, kvh, hd, tokens, base, stride, l.slots);
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
    gemm(m.xf, m.fps, tokens, m.head, m.logits);
    if (m.softcap > 0) {
        const size_t count = size_t(tokens) * m.vocab;
        softcap_k<<<unsigned((count + 1023) / 1024), 1024>>>(m.logits, count, m.softcap);
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
    // work memory
    float* in;      // [2 * backbone] the token's embedding in the main model, then a hidden state: the main model's last output, or `next`
    float* h;       // [hidden]
    float* hn;      // [hidden] after the final norm
    float* logits;  // [vocab]
    float* next;    // [backbone] the hidden state for the guess after this one
    float* h2;      // [hidden] the residual stream's other state (see sclmd_pass)
    float* fps;     // the rest as in Model (it may be the main model's own work memory)
    uint64_t* xf;
    float *y, *y2;
    uint16_t* qhat;
    float* scores;
    uint16_t* probs;
    float* attn;
};

// The input of one of the drafter's matrices with n columns from a (and b, w: see prep_k), in d.xf with its scale in d.fps.
static void prep(const Drafter& d, const float* a, const float* b, const float* w, int n, const Rot& rot, int mode, const Owed& owed = Owed{},
                 float* h_out = nullptr, float* v_out = nullptr) {
    prep_k<<<dim3(n / rot.block, 1), rot.block>>>(a, b, w, owed.y, owed.post, owed.scalar, h_out, v_out, rot.signs, d.fps, d.xf, n, d.eps, mode);
}

// One guess: d.in holds the embedding of the token at position `length` and a hidden state; the main model has run
// the positions before it, and `sliding` and `full` are its last layers of each kind (their keys and values are
// read: in a sliding layer the last window + 1 of them, as upstream lets the drafter). Afterwards d.logits holds the
// guess for position length + 1 and d.next the hidden state for the guess after it. Launches only.
API int sclmd_draft(const Drafter* dp, const Layer* sliding, const Layer* full, int window, int length) {
    const Drafter& d = *dp;
    const int n = d.hidden, last = length - 1;
    prep(d, d.in, nullptr, nullptr, 2 * d.backbone, d.rot_pre, 1);
    gemm(d.xf, d.fps, 1, d.pre, d.h);
    float *h = d.h, *spare = d.h2;
    Owed owed;
    for (int i = 0; i < d.n_layers; ++i) {
        const DLayer& l = d.layers[i];
        const Layer& src = l.full ? *full : *sliding;
        const int hd = l.head_dim, kvh = l.kv_heads;
        // attention: the query sits at position `length` and reads the main model's keys up to length - 1
        prep(d, h, nullptr, l.in_ln, n, l.rot_in, 0, owed, spare);
        if (owed.y) std::swap(h, spare);
        gemm(d.xf, d.fps, 1, l.q, d.y);
        heads_k<<<dim3(d.heads, 1), hd>>>(d.y, nullptr, nullptr, l.q_norm, nullptr, l.inv_freq, d.qhat, nullptr, nullptr, d.heads, kvh, length, 1, d.eps);
        const int lo = l.full ? 0 : (last - window > 0 ? last - window : 0);
        const int base = lo & ~15, stride = ((last >> 4) + 1) * 16 - base;
        const int nq = d.heads / kvh, groups = (nq + 7) / 8;
        scores_k<<<dim3(stride / 16, kvh), 32 * groups>>>(d.qhat, src.keys, d.scores, d.heads, kvh, hd, 1, base, stride, src.slots);
        softmax_k<<<dim3(nq, kvh), 256>>>(d.scores, d.probs, d.heads, kvh, 1, last, base, stride, l.full ? 0 : window + 1);
        attend_k<<<dim3(hd / 16, kvh), 32 * groups>>>(d.probs, src.values, d.attn, d.heads, kvh, hd, 1, base, stride, src.slots);
        prep(d, d.attn, nullptr, nullptr, d.heads * hd, l.rot_o, 1);
        gemm(d.xf, d.fps, 1, l.o, d.y);
        owed = Owed{d.y, l.post_attn_ln, 1.0f};
        // feed-forward
        prep(d, h, nullptr, l.pre_ff_ln, n, l.rot_mlp, 0, owed, spare);
        std::swap(h, spare);
        gemm(d.xf, d.fps, 1, l.gate, d.y, l.up, d.y2);
        prep(d, d.y, d.y2, nullptr, d.mlp, l.rot_down, 2);
        gemm(d.xf, d.fps, 1, l.down, d.y);
        owed = Owed{d.y, l.post_ff_ln, l.scalar};
    }
    prep(d, h, nullptr, d.norm, n, d.rot_out, 0, owed, spare, d.hn);
    gemm(d.xf, d.fps, 1, d.head, d.logits, d.post, d.next);
    return int(cudaGetLastError());
}

API int sclmd_draft_sizes(int* dlayer, int* drafter) {
    *dlayer = sizeof(DLayer), *drafter = sizeof(Drafter);
    return 0;
}

API int sclmd_sizes(int* mat, int* rot, int* layer, int* model) {
    *mat = sizeof(Mat), *rot = sizeof(Rot), *layer = sizeof(Layer), *model = sizeof(Model);
    return MAX_TOKENS;
}
