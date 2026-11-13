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
#include <cmath>
#include <cstdint>

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
    float* x;        // [8][widest input] a matrix's input before the rotation
    float* fps;      // [8] the power of two an input was divided by to fit fp16
    uint64_t* xf;    // [widest input / 16][32] a matrix's input as the instruction reads it
    float* y;        // [8][widest output] a matrix's output
    float* y2;       // [8][mlp] (up's output beside gate's)
    float* qhat;     // [8][heads * widest head] the queries, normalized and rotated
    float* scores;   // [kv heads * queries][positions] attention scores, then weights
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

// A matrix's input, one block a token: mode 0: the RMS norm of a[token] times w; 1: a[token] as it is; 2: gelu(a) * b.
// -> x[token][n], and fps[token]: the power of two that brings the input's length under 2^14, so that no value of
// it, rotated or not, leaves fp16's range.
__global__ void input_k(const float* a, const float* b, const float* w, float* x, float* fps, int n, float eps, int mode) {
    const int token = blockIdx.x;
    const float* at = a + size_t(token) * n;
    float r = 1.0f;
    if (mode == 0) {
        float ss = 0;
        for (int i = threadIdx.x; i < n; i += blockDim.x) ss += at[i] * at[i];
        r = powf(block_sum(ss) / n + eps, -0.5f);
    }
    float len = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = mode == 0 ? at[i] * r * w[i] : mode == 1 ? at[i] : gelu_tanh(at[i]) * b[size_t(token) * n + i];
        x[size_t(token) * n + i] = v;
        len += v * v;
    }
    len = sqrtf(block_sum(len));
    if (threadIdx.x == 0) fps[token] = len > 16384.0f ? exp2f(ceilf(log2f(len / 16384.0f))) : 1.0f;
}

// x[token][n] rotated and laid out for the instruction: block (m, token) takes the token's m-th run of `block` values
// (block threads, a power of two from 16 to 1024): signs, Hadamard transform, 1 / sqrt(block), over fps[token], fp16.
// For tile j, lane 4 * token + t of xf holds the values at columns 16j + 2t, + 1, + 8, + 9.
__global__ void rotate_k(const float* x, const float* signs, const float* fps, uint64_t* xf, int n) {
    __shared__ float buf[1024];
    const int b = blockDim.x, m = blockIdx.x, token = blockIdx.y, i = threadIdx.x, col = m * b + i;
    buf[i] = x[size_t(token) * n + col] * signs[col];
    __syncthreads();
    for (int h = 1; h < b; h <<= 1) {
        const float lo = buf[i & ~h], hi = buf[i | h];
        __syncthreads();
        buf[i] = (i & h) ? lo - hi : lo + hi;
        __syncthreads();
    }
    const float k = rsqrtf(float(b)) / fps[token];
    if ((i & 9) == 0) {  // i % 16 in {0, 2, 4, 6}: the first column of a lane's word
        const int t = (i & 15) >> 1;
        const uint64_t w = uint64_t(__half_as_ushort(__float2half(buf[i] * k))) | uint64_t(__half_as_ushort(__float2half(buf[i + 1] * k))) << 16 |
                           uint64_t(__half_as_ushort(__float2half(buf[i + 8] * k))) << 32 |
                           uint64_t(__half_as_ushort(__float2half(buf[i + 9] * k))) << 48;
        xf[size_t(col >> 4) * 32 + 4 * token + t] = w;
    }
}

// y[token][row] = scale[row] * fps[token] * sum over columns of W[row][col] * x[token][col], one block a strip of 16
// rows, its `segs` warps splitting the strip's tiles (a multiple of 4 each) between them.
template <int BITS>
__global__ void gemm_k(const uint32_t* codes, const uint32_t* lut, const float* scale, const uint64_t* xf, const float* fps, float* y,
                       int rows, int cols, int segs, int tokens) {
    __shared__ uint32_t slut[1 << BITS];
    __shared__ float part[8][32][4];
    for (int i = threadIdx.x; i < (1 << BITS); i += blockDim.x) slut[i] = lut[i];
    __syncthreads();
    const int strip = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tiles = cols / 16, per = tiles / segs, first = warp * per;
    float c[4] = {0, 0, 0, 0};
    const uint64_t* xw = xf + size_t(first) * 32 + lane;
    if (BITS == 8) {
        const uint32_t* cw = codes + (size_t(strip) * tiles + first) * 32 + lane;
        for (int j = 0; j < per; j += 4) {
            uint32_t w[4];
            uint64_t b[4];
            for (int i = 0; i < 4; ++i) w[i] = cw[size_t(j + i) * 32], b[i] = xw[size_t(j + i) * 32];
            for (int i = 0; i < 4; ++i)
                mma(c, slut[w[i] & 255], slut[w[i] >> 8 & 255], slut[w[i] >> 16 & 255], slut[w[i] >> 24], uint32_t(b[i]), uint32_t(b[i] >> 32));
        }
    } else {  // 6 bits a pair: four tiles of a lane are 16 codes in 96 bits, stored as three words
        const uint32_t* cw = codes + (size_t(strip) * (tiles / 4) + first / 4) * 96 + lane;
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
        if (token < tokens) y[size_t(token) * rows + row] = scale[row] * fps[token] * sum;
    }
}

// h[token] = (h[token] + RMS norm of y[token] times w) * scalar, one block a token.
__global__ void residual_k(float* h, const float* y, const float* w, int n, float eps, float scalar) {
    const int token = blockIdx.x;
    const float* yt = y + size_t(token) * n;
    float* ht = h + size_t(token) * n;
    float ss = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += yt[i] * yt[i];
    const float r = powf(block_sum(ss) / n + eps, -0.5f);
    for (int i = threadIdx.x; i < n; i += blockDim.x) ht[i] = (ht[i] + yt[i] * r * w[i]) * scalar;
}

// What attention does to the projections' outputs, block (vector, token) with head_dim threads. Vector v of a token is
// one of its `heads` query heads (v < heads), `kvh` key heads or `kvh` value heads:
//   query: RMS norm times q_norm, rotary position -> qhat[token][head][head_dim]
//   key:   RMS norm times k_norm, rotary position -> keys[slot of the position][head] (fp16)
//   value: RMS norm                               -> values[slot][head] (fp16); the value's source is yv (or yk when null)
// The rotary position turns pair (d, d + head_dim / 2) by pos * inv_freq[d].
__global__ void heads_k(const float* yq, const float* yk, const float* yv, const float* q_norm, const float* k_norm, const float* inv_freq,
                        float* qhat, uint16_t* keys, uint16_t* values, int heads, int kvh, int pos0, int slots, float eps) {
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
        qhat[(size_t(token) * heads + head) * hd + d] = out;
    else
        keys[(size_t(slot) * kvh + head) * hd + d] = __half_as_ushort(__float2half(out));
}

// Attention scores: block (chunk, key head) takes blockDim / nq of the positions lo + r (r from chunk * that many) and
// all nq = rep * tokens queries of the key head's `rep` query heads; a thread one (position, query) pair.
// -> scores[(key head * nq + query)][r], query = token * rep + head within the group.
__global__ void scores_k(const float* qhat, const uint16_t* keys, float* scores, int heads, int kvh, int hd, int tokens, int lo, int npos,
                         int slots) {
    const int rep = heads / kvh, nq = rep * tokens, per = blockDim.x / nq;
    const int r = blockIdx.x * per + threadIdx.x / nq, query = threadIdx.x % nq, kh = blockIdx.y;
    if (r >= npos || threadIdx.x >= per * nq) return;
    const int token = query / rep, head = kh * rep + query % rep;
    const float* q = qhat + (size_t(token) * heads + head) * hd;
    const uint16_t* k = keys + (size_t((lo + r) % slots) * kvh + kh) * hd;
    float acc = 0;
    for (int d = 0; d < hd; ++d) acc += q[d] * __half2float(__ushort_as_half(k[d]));
    scores[(size_t(kh) * nq + query) * npos + r] = acc;
}

// The softmax over the positions a query may read and the weighted sum of their values, block (query, key head) with
// head_dim threads. Token t at position pos0 + t reads positions up to its own, and in a sliding layer the last
// `window` of them. -> out[token][head][head_dim]
__global__ void attend_k(float* scores, const uint16_t* values, float* out, int heads, int kvh, int tokens, int pos0, int lo, int npos,
                         int slots, int window) {
    const int hd = blockDim.x, rep = heads / kvh, nq = rep * tokens, query = blockIdx.x, kh = blockIdx.y;
    const int token = query / rep, head = kh * rep + query % rep, pos = pos0 + token;
    float* s = scores + (size_t(kh) * nq + query) * npos;
    const int first = (window > 0 && pos - window + 1 > lo ? pos - window + 1 : lo) - lo, last = pos - lo;  // inclusive, as indices
    // a thread takes the positions that are its own number modulo the block: the sums then add up in the same order
    // whichever pass a position is computed in (alone, or as one of eight)
    const int mine = first + ((int(threadIdx.x) - (lo + first) % hd + hd) % hd);
    float mx = -INFINITY;
    for (int r = mine; r <= last; r += hd) mx = fmaxf(mx, s[r]);
    mx = block_max(mx);
    float sum = 0;
    for (int r = mine; r <= last; r += hd) sum += expf(s[r] - mx);
    sum = block_sum(sum);
    for (int r = mine; r <= last; r += hd) s[r] = expf(s[r] - mx) / sum;
    __syncthreads();
    float acc = 0;
    const int d = threadIdx.x;
    for (int r = first; r <= last; ++r)
        acc += s[r] * __half2float(__ushort_as_half(values[(size_t((lo + r) % slots) * kvh + kh) * hd + d]));
    out[(size_t(token) * heads + head) * hd + d] = acc;
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

static void gemm(const Mat& m, const uint64_t* xf, const float* fps, float* y, int tokens) {
    const int segs = segs_for(m.cols);
    if (m.bits == 8)
        gemm_k<8><<<m.rows / 16, 32 * segs>>>(m.codes, m.lut, m.scale, xf, fps, y, m.rows, m.cols, segs, tokens);
    else
        gemm_k<6><<<m.rows / 16, 32 * segs>>>(m.codes, m.lut, m.scale, xf, fps, y, m.rows, m.cols, segs, tokens);
}

// a (and b, w: see input_k) -> the rotated input of a matrix with n columns, in m.xf with its scales in m.fps.
static void input(const Model& m, const float* a, const float* b, const float* w, int n, const Rot& rot, int mode, int tokens) {
    input_k<<<tokens, 1024>>>(a, b, w, m.x, m.fps, n, m.eps, mode);
    rotate_k<<<dim3(n / rot.block, tokens), rot.block>>>(m.x, rot.signs, m.fps, m.xf, n);
}

// One matrix times a given input (tests): x[tokens][cols] -> y[tokens][rows], with the model's work memory.
API int sclmd_linear(const Model* m, const Mat* mat, const Rot* rot, const float* x, float* y, int tokens) {
    input(*m, x, nullptr, nullptr, mat->cols, *rot, 1, tokens);
    gemm(*mat, m->xf, m->fps, y, tokens);
    return int(cudaGetLastError());
}

// The pass: m.h holds the scaled embeddings of `tokens` tokens at positions pos0.. ; afterwards it holds the last
// layer's output, m.hn that output after the final norm, m.logits the logits, and every layer's keys and values have
// the new positions. Launches only; nothing waits.
API int sclmd_pass(const Model* mp, int tokens, int pos0) {
    const Model& m = *mp;
    const int n = m.hidden;
    for (int i = 0; i < m.n_layers; ++i) {
        const Layer& l = m.layers[i];
        const int hd = l.head_dim, kvh = l.kv_heads, nqo = m.heads * hd, nkv = kvh * hd;
        // attention
        input(m, m.h, nullptr, l.in_ln, n, l.rot_in, 0, tokens);
        float *yq = m.y, *yk = m.y + size_t(MAX_TOKENS) * nqo, *yv = yk + size_t(MAX_TOKENS) * nkv;
        gemm(l.q, m.xf, m.fps, yq, tokens);
        gemm(l.k, m.xf, m.fps, yk, tokens);
        if (l.v.rows) gemm(l.v, m.xf, m.fps, yv, tokens);
        heads_k<<<dim3(m.heads + 2 * kvh, tokens), hd>>>(yq, yk, l.v.rows ? yv : nullptr, l.q_norm, l.k_norm, l.inv_freq, m.qhat, l.keys,
                                                       l.values, m.heads, kvh, pos0, l.slots, m.eps);
        const int last = pos0 + tokens - 1, lo = l.full ? 0 : (pos0 - m.window + 1 > 0 ? pos0 - m.window + 1 : 0), npos = last - lo + 1;
        const int rep = m.heads / kvh, nq = rep * tokens, per = 1024 / nq;
        scores_k<<<dim3((npos + per - 1) / per, kvh), 1024>>>(m.qhat, l.keys, m.scores, m.heads, kvh, hd, tokens, lo, npos, l.slots);
        attend_k<<<dim3(nq, kvh), hd>>>(m.scores, l.values, m.attn, m.heads, kvh, tokens, pos0, lo, npos, l.slots, l.full ? 0 : m.window);
        input(m, m.attn, nullptr, nullptr, nqo, l.rot_o, 1, tokens);
        gemm(l.o, m.xf, m.fps, m.y, tokens);
        residual_k<<<tokens, 1024>>>(m.h, m.y, l.post_attn_ln, n, m.eps, 1.0f);
        // feed-forward
        input(m, m.h, nullptr, l.pre_ff_ln, n, l.rot_mlp, 0, tokens);
        gemm(l.gate, m.xf, m.fps, m.y, tokens);
        gemm(l.up, m.xf, m.fps, m.y2, tokens);
        input(m, m.y, m.y2, nullptr, m.mlp, l.rot_down, 2, tokens);
        gemm(l.down, m.xf, m.fps, m.y, tokens);
        residual_k<<<tokens, 1024>>>(m.h, m.y, l.post_ff_ln, n, m.eps, l.scalar);
    }
    input_k<<<tokens, 1024>>>(m.h, nullptr, m.norm, m.hn, m.fps, n, m.eps, 0);
    rotate_k<<<dim3(n / m.rot_head.block, tokens), m.rot_head.block>>>(m.hn, m.rot_head.signs, m.fps, m.xf, n);
    gemm(m.head, m.xf, m.fps, m.logits, tokens);
    if (m.softcap > 0) {
        const size_t count = size_t(tokens) * m.vocab;
        softcap_k<<<unsigned((count + 1023) / 1024), 1024>>>(m.logits, count, m.softcap);
    }
    return int(cudaGetLastError());
}

API int sclmd_sizes(int* mat, int* rot, int* layer, int* model) {
    *mat = sizeof(Mat), *rot = sizeof(Rot), *layer = sizeof(Layer), *model = sizeof(Model);
    return MAX_TOKENS;
}
