// The tensor cores' instruction rate on this card, register-only (no memory in the loop): every warp runs CHAINS
// independent accumulators, each a chain of dependent mma.sync instructions, so the issue rate and not the latency
// bounds the time. Three instructions:
//   f16 x f16, f32 sums   m16n8k16  (today's, engine/sclmd.cu mma)      4,096 operations an instruction
//   f16 x f16, f16 sums   m16n8k16                                        4,096
//   s8 x s8, s32 sums     m16n8k32                                        8,192
// Printed: tera-operations a second over the whole card, for 4 and 8 warps an SM's block.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                                                   \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(err_), __LINE__); \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

#define CHAINS 8

__global__ void f32sum_k(float* out, int iters, uint32_t seed) {
    float c[CHAINS][4];
    for (int k = 0; k < CHAINS; ++k) c[k][0] = c[k][1] = c[k][2] = c[k][3] = 0;
    const uint32_t a0 = seed ^ threadIdx.x, a1 = a0 * 3u, a2 = a0 * 5u, a3 = a0 * 7u, b0 = a0 * 11u, b1 = a0 * 13u;
    for (int it = 0; it < iters; ++it)
#pragma unroll
        for (int k = 0; k < CHAINS; ++k)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(c[k][0]), "+f"(c[k][1]), "+f"(c[k][2]), "+f"(c[k][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    float s = 0;
    for (int k = 0; k < CHAINS; ++k) s += c[k][0] + c[k][1] + c[k][2] + c[k][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void f16sum_k(float* out, int iters, uint32_t seed) {
    uint32_t c[CHAINS][2];
    for (int k = 0; k < CHAINS; ++k) c[k][0] = c[k][1] = 0;
    const uint32_t a0 = seed ^ threadIdx.x, a1 = a0 * 3u, a2 = a0 * 5u, a3 = a0 * 7u, b0 = a0 * 11u, b1 = a0 * 13u;
    for (int it = 0; it < iters; ++it)
#pragma unroll
        for (int k = 0; k < CHAINS; ++k)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                         : "+r"(c[k][0]), "+r"(c[k][1])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    uint32_t s = 0;
    for (int k = 0; k < CHAINS; ++k) s ^= c[k][0] ^ c[k][1];
    out[blockIdx.x * blockDim.x + threadIdx.x] = float(s);
}

__global__ void s8_k(float* out, int iters, uint32_t seed) {
    int c[CHAINS][4];
    for (int k = 0; k < CHAINS; ++k) c[k][0] = c[k][1] = c[k][2] = c[k][3] = 0;
    const uint32_t a0 = seed ^ threadIdx.x, a1 = a0 * 3u, a2 = a0 * 5u, a3 = a0 * 7u, b0 = a0 * 11u, b1 = a0 * 13u;
    for (int it = 0; it < iters; ++it)
#pragma unroll
        for (int k = 0; k < CHAINS; ++k)
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+r"(c[k][0]), "+r"(c[k][1]), "+r"(c[k][2]), "+r"(c[k][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    int s = 0;
    for (int k = 0; k < CHAINS; ++k) s += c[k][0] + c[k][1] + c[k][2] + c[k][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = float(s);
}

typedef void (*Kern)(float*, int, uint32_t);

static double time_it(Kern k, int blocks, int threads, int iters, float* out) {
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    k<<<blocks, threads>>>(out, 10, 1);  // warm-up
    CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < 5; ++r) {
        CHECK(cudaEventRecord(e0));
        k<<<blocks, threads>>>(out, iters, 1 + r);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        best = ms < best ? ms : best;
    }
    cudaEventDestroy(e0), cudaEventDestroy(e1);
    return best * 1e-3;
}

int main() {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    printf("%s: %d SMs\n", p.name, p.multiProcessorCount);
    float* out;
    CHECK(cudaMalloc(&out, size_t(p.multiProcessorCount) * 8 * 1024 * 4));
    const int iters = 20000;
    struct {
        const char* name;
        Kern k;
        double ops;
    } ks[] = {{"m16n8k16 f32.f16.f16.f32 (today)", f32sum_k, 4096}, {"m16n8k16 f16.f16.f16.f16", f16sum_k, 4096}, {"m16n8k32 s32.s8.s8.s32", s8_k, 8192}};
    double rate[3][3] = {};
    for (int w : {4, 8}) {
        for (int bps : {1, 2}) {
            const int blocks = p.multiProcessorCount * bps, threads = 32 * w;
            for (int i = 0; i < 3; ++i) {
                const double s = time_it(ks[i].k, blocks, threads, iters, out);
                const double tops = double(blocks) * w * iters * CHAINS * ks[i].ops / s / 1e12;
                printf("%-34s %d warps x %d blocks an SM: %8.1f T ops/s  (%.2f ms)\n", ks[i].name, w, bps, tops, s * 1e3);
                if (tops > rate[i][0]) rate[i][0] = tops;
            }
        }
    }
    printf("best: f32-sum %.1f, f16-sum %.1f, s8 %.1f T ops/s; s8 / f32-sum = %.2fx, f16-sum / f32-sum = %.2fx\n", rate[0][0], rate[1][0], rate[2][0],
           rate[2][0] / rate[0][0], rate[1][0] / rate[0][0]);
    cudaFree(out);
    return 0;
}
