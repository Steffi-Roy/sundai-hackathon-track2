#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// ---- Configuration ----
static constexpr int BLOCK_M    = 128;
static constexpr int BLOCK_N    = 128;
static constexpr int BLOCK_K    = 64;
static constexpr int WARP_SZ    = 32;
static constexpr int NUM_WARPS  = 8;
static constexpr int WARP_M     = BLOCK_M / NUM_WARPS;  // 16
static constexpr int TILES_N    = BLOCK_N / 16;          // 8
static constexpr int NUM_STAGES = 3;

// Shared memory stride: K/2 bytes + padding (must be 16-byte aligned for cp.async)
static constexpr int SMEM_STRIDE = BLOCK_K / 2 + 16;   // 48 bytes per row

// ── Quantize kernel configuration ──
static constexpr int WARPS_PER_BLOCK_Q = 8;
static constexpr int GROUPS_PER_WARP   = 4;
static constexpr int THREADS_PER_GROUP = WARP_SZ / GROUPS_PER_WARP;  // 8

// ── Optimized activation INT4 quantization kernel ─────────────────────────
// float4 loads (8 halves/thread), 4 groups/warp, 3-step sub-group reduction.
// Scales stored in transposed [num_groups, M] layout for coalesced GEMM reads.
// group_size must be 64 (8 threads × 8 halves).
__global__ void quantize_int4_opt_kernel(
    const half*  __restrict__ input,
    const half*  __restrict__ smooth,
    uint8_t*     __restrict__ output,
    half*        __restrict__ scales,  // [num_groups, M] transposed
    int M, int K, int group_size
) {
    const int warpId        = threadIdx.x / WARP_SZ;
    const int lane          = threadIdx.x % WARP_SZ;
    const int group_in_warp = lane / THREADS_PER_GROUP;   // 0..3
    const int lane_in_group = lane % THREADS_PER_GROUP;   // 0..7

    const int row    = blockIdx.x;
    const int g_base = (blockIdx.y * WARPS_PER_BLOCK_Q + warpId) * GROUPS_PER_WARP;
    const int g      = g_base + group_in_warp;
    if (g >= K / group_size) return;
    const int k0 = g * group_size;

    // float4 load: 16 bytes = 8 half values per thread
    const float4* ptr4 = reinterpret_cast<const float4*>(
        input + row * K + k0 + lane_in_group * 8);
    float4 chunk = *ptr4;

    half2 p0 = *reinterpret_cast<const half2*>(&chunk.x);
    half2 p1 = *reinterpret_cast<const half2*>(&chunk.y);
    half2 p2 = *reinterpret_cast<const half2*>(&chunk.z);
    half2 p3 = *reinterpret_cast<const half2*>(&chunk.w);

    float a0 = __half2float(p0.x), a1 = __half2float(p0.y);
    float a2 = __half2float(p1.x), a3 = __half2float(p1.y);
    float a4 = __half2float(p2.x), a5 = __half2float(p2.y);
    float a6 = __half2float(p3.x), a7 = __half2float(p3.y);

    if (smooth != nullptr) {
        const float4* sptr4 = reinterpret_cast<const float4*>(
            smooth + k0 + lane_in_group * 8);
        float4 sv = *sptr4;
        half2 s0 = *reinterpret_cast<const half2*>(&sv.x);
        half2 s1 = *reinterpret_cast<const half2*>(&sv.y);
        half2 s2 = *reinterpret_cast<const half2*>(&sv.z);
        half2 s3 = *reinterpret_cast<const half2*>(&sv.w);
        a0 /= __half2float(s0.x); a1 /= __half2float(s0.y);
        a2 /= __half2float(s1.x); a3 /= __half2float(s1.y);
        a4 /= __half2float(s2.x); a5 /= __half2float(s2.y);
        a6 /= __half2float(s3.x); a7 /= __half2float(s3.y);
    }

    // Sub-group max reduction: 8 threads per group, 3 shuffles (log2(8)=3)
    float local_max = fmaxf(fmaxf(fmaxf(fabsf(a0), fabsf(a1)),
                            fmaxf(fabsf(a2), fabsf(a3))),
                     fmaxf(fmaxf(fabsf(a4), fabsf(a5)),
                            fmaxf(fabsf(a6), fabsf(a7))));
    unsigned sub_mask = 0xFFu << (group_in_warp * THREADS_PER_GROUP);
    for (int off = 4; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(sub_mask, local_max, off));

    float rscale = (local_max > 0.f) ? (7.5f / local_max) : 0.f;
    // Transposed [num_groups, M] layout: consecutive m values are adjacent in memory
    if (lane_in_group == 0)
        scales[g * M + row] = __float2half(local_max / 7.5f);

    auto quant = [&](float v) -> int {
        return max(-8, min(7, __float2int_rn(v * rscale)));
    };
    int q0 = quant(a0), q1 = quant(a1), q2 = quant(a2), q3 = quant(a3);
    int q4 = quant(a4), q5 = quant(a5), q6 = quant(a6), q7 = quant(a7);

    // Pack 8 INT4 values into 1 uint32_t (4 bytes): low nibble = even index
    uint32_t packed =
        ((uint32_t)(q0 & 0xF))        |
        ((uint32_t)(q1 & 0xF) << 4)   |
        ((uint32_t)(q2 & 0xF) << 8)   |
        ((uint32_t)(q3 & 0xF) << 12)  |
        ((uint32_t)(q4 & 0xF) << 16)  |
        ((uint32_t)(q5 & 0xF) << 20)  |
        ((uint32_t)(q6 & 0xF) << 24)  |
        ((uint32_t)(q7 & 0xF) << 28);

    const int out_base = row * (K / 2) + k0 / 2;
    *reinterpret_cast<uint32_t*>(output + out_base + lane_in_group * 4) = packed;
}

std::vector<torch::Tensor> quantize_int4_custom(torch::Tensor input, int group_size) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kHalf);
    int M = input.size(0), K = input.size(1);
    TORCH_CHECK(K % group_size == 0 && group_size % 2 == 0);
    TORCH_CHECK(group_size == 64, "quantize_int4_custom requires group_size=64");

    auto output = torch::empty({M, K / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    int num_groups = K / group_size;
    // Transposed layout [num_groups, M] for coalesced reads in GEMM kernel
    auto scales = torch::empty({num_groups, M},
        torch::TensorOptions().dtype(torch::kHalf).device(input.device()));

    const int groups_per_block = WARPS_PER_BLOCK_Q * GROUPS_PER_WARP;   // 32
    dim3 grid(M, (num_groups + groups_per_block - 1) / groups_per_block);
    dim3 block(WARP_SZ * WARPS_PER_BLOCK_Q);   // 256 threads

    quantize_int4_opt_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        nullptr,
        output.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
        M, K, group_size
    );
    return {output, scales};
}


// ---- MMA wrapper: m16n8k64 INT4×INT4 → INT32 ----
__device__ __forceinline__ void mma_s4(uint4 a, uint2 b, int (&c)[4]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        : "r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b.x),"r"(b.y));
#else
    asm volatile("{"
        ".reg .b32 t0,t1,t2,t3;\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {t0,t1},{%4},{%8},{%0,%1};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {t2,t3},{%5},{%8},{%2,%3};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1},{%6},{%9},{t0,t1};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%2,%3},{%7},{%9},{t2,t3};\n"
        "}\n"
        : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        : "r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b.x),"r"(b.y));
#endif
}


// ---- cp.async: 16-byte async global→shared copy ----
__device__ __forceinline__ void cp_async_16(void *dst, const void *src, bool pred) {
    unsigned s = __cvta_generic_to_shared(dst);
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,%2,0;\n"
        "  @p cp.async.cg.shared.global [%0],[%1],16;\n"
        "  @!p st.shared.v4.u32 [%0],{0,0,0,0}; }\n"
        :: "r"(s),"l"(src),"r"((int)pred));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait(int n) {
    if      (n == 0) asm volatile("cp.async.wait_group 0;\n");
    else if (n == 1) asm volatile("cp.async.wait_group 1;\n");
    else             asm volatile("cp.async.wait_group 2;\n");
}


// ---- Load MMA A-fragment directly from shared memory ----
__device__ __forceinline__ uint4 load_a_frag(const uint8_t *base, int stride) {
    int lane   = threadIdx.x % WARP_SZ;
    int row_lo = lane / 4;
    int row_hi = row_lo + 8;
    int col    = (lane % 4) * 4;
    uint4 a;
    a.x = *(const uint32_t*)(base + row_lo * stride + col);
    a.y = *(const uint32_t*)(base + row_hi * stride + col);
    a.z = *(const uint32_t*)(base + row_lo * stride + 16 + col);
    a.w = *(const uint32_t*)(base + row_hi * stride + 16 + col);
    return a;
}

// ---- Load MMA B-fragment from shared memory ----
__device__ __forceinline__ uint2 load_b_frag(const uint8_t *base, int stride) {
    int lane = threadIdx.x % WARP_SZ;
    int row  = lane / 4;
    int col  = (lane % 4) * 4;
    uint2 b;
    b.x = *(const uint32_t*)(base + row * stride + col);
    b.y = *(const uint32_t*)(base + row * stride + 16 + col);
    return b;
}


// ---- Main GEMM kernel ----
// Baseline post-revert per CLAUDE_revert.md:
//   - MMA m16n8k64, Triple buffering (NUM_STAGES=3), cp.async.cg
//   - __ldg for scale reads (NO SMEM scale staging)
//   - half2 epilogue
//   - BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, NUM_WARPS=8
//   - NO __launch_bounds__, NO cudaFuncSetAttribute
// scales_A is in transposed [num_groups, M] layout produced by the quantize kernel;
// scales_B is in [N, num_groups] layout produced by quantize.py.
__global__ void gemm_int4_kernel(
    const uint8_t *__restrict__ A,
    const uint8_t *__restrict__ B,
    const half    *__restrict__ scales_A,  // [num_groups, M] transposed
    const half    *__restrict__ scales_B,  // [N, num_groups]
    half          *__restrict__ C,
    int M, int N, int K, int group_size)
{
    const int bm = blockIdx.y * BLOCK_M;
    const int bn = blockIdx.x * BLOCK_N;
    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SZ;
    const int laneId = tid % WARP_SZ;
    const int halfK = K / 2;
    const int num_groups = K / group_size;
    const int num_k_tiles = K / BLOCK_K;

    // Triple-buffered shared memory (3 stages)
    extern __shared__ uint8_t smem[];
    const int tileA  = BLOCK_M * SMEM_STRIDE;
    const int tileB  = BLOCK_N * SMEM_STRIDE;
    const int tileAB = tileA + tileB;
    uint8_t *sA[NUM_STAGES], *sB[NUM_STAGES];
    for (int i = 0; i < NUM_STAGES; i++) {
        sA[i] = smem + i * tileAB;
        sB[i] = smem + i * tileAB + tileA;
    }

    // FP32 accumulators: [n_tile][mma_half=0,1][4 values]
    float acc[TILES_N][2][4];
    for (int j = 0; j < TILES_N; j++)
        for (int h = 0; h < 2; h++)
            acc[j][h][0] = acc[j][h][1] = acc[j][h][2] = acc[j][h][3] = 0.f;

    // ---- Cooperative tile loader ----
    auto load_tile = [&](int kt, int s) {
        int kb = kt * (BLOCK_K / 2);
        {
            int row = tid / 2, half = tid % 2;
            bool p = (bm + row < M) && (kb + half * 16 < halfK);
            cp_async_16(sA[s] + row * SMEM_STRIDE + half * 16,
                        A + (size_t)(bm + row) * halfK + kb + half * 16, p);
        }
        {
            int row = tid / 2, half = tid % 2;
            bool p = (bn + row < N) && (kb + half * 16 < halfK);
            cp_async_16(sB[s] + row * SMEM_STRIDE + half * 16,
                        B + (size_t)(bn + row) * halfK + kb + half * 16, p);
        }
        cp_commit();
    };

    // Prefetch first two tiles before the loop
    if (num_k_tiles > 0) load_tile(0, 0);
    if (num_k_tiles > 1) load_tile(1, 1);

    // ---- Main K-loop ----
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int s = kt % NUM_STAGES;
        if (kt + 2 < num_k_tiles)
            load_tile(kt + 2, (kt + 2) % NUM_STAGES);
        cp_wait(kt + 2 < num_k_tiles ? NUM_STAGES - 1 : 0);
        __syncthreads();

        int g = (kt * BLOCK_K) / group_size;

        int m_lo = bm + warpId * WARP_M + laneId / 4;
        int m_hi = m_lo + 8;
        // scales_A is [num_groups, M] transposed — read as scales_A[g*M + m]
        float sa_lo = (m_lo < M) ? __half2float(__ldg(&scales_A[g * M + m_lo])) : 0.f;
        float sa_hi = (m_hi < M) ? __half2float(__ldg(&scales_A[g * M + m_hi])) : 0.f;

        uint4 af = load_a_frag(sA[s] + warpId * WARP_M * SMEM_STRIDE, SMEM_STRIDE);

        #pragma unroll
        for (int nt = 0; nt < TILES_N; nt++) {
            int n_off = nt * 16;

            uint2 bf0 = load_b_frag(sB[s] + (n_off + 0) * SMEM_STRIDE, SMEM_STRIDE);
            uint2 bf1 = load_b_frag(sB[s] + (n_off + 8) * SMEM_STRIDE, SMEM_STRIDE);

            int p0[4] = {0,0,0,0}, p1[4] = {0,0,0,0};
            mma_s4(af, bf0, p0);
            mma_s4(af, bf1, p1);

            int c0 = bn + n_off + (laneId % 4) * 2;
            int c1 = c0 + 1;
            int c2 = c0 + 8;
            int c3 = c2 + 1;
            // scales_B is [N, num_groups] — read as scales_B[n*num_groups + g]
            float sb0 = (c0 < N) ? __half2float(__ldg(&scales_B[c0 * num_groups + g])) : 0.f;
            float sb1 = (c1 < N) ? __half2float(__ldg(&scales_B[c1 * num_groups + g])) : 0.f;
            float sb2 = (c2 < N) ? __half2float(__ldg(&scales_B[c2 * num_groups + g])) : 0.f;
            float sb3 = (c3 < N) ? __half2float(__ldg(&scales_B[c3 * num_groups + g])) : 0.f;

            acc[nt][0][0] += (float)p0[0] * sa_lo * sb0;
            acc[nt][0][1] += (float)p0[1] * sa_lo * sb1;
            acc[nt][0][2] += (float)p0[2] * sa_hi * sb0;
            acc[nt][0][3] += (float)p0[3] * sa_hi * sb1;
            acc[nt][1][0] += (float)p1[0] * sa_lo * sb2;
            acc[nt][1][1] += (float)p1[1] * sa_lo * sb3;
            acc[nt][1][2] += (float)p1[2] * sa_hi * sb2;
            acc[nt][1][3] += (float)p1[3] * sa_hi * sb3;
        }
        __syncthreads();
    }

    // Flush any remaining outstanding async copies before epilogue
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    // ---- Epilogue: half2 vectorized stores ----
    int m_lo = bm + warpId * WARP_M + laneId / 4;
    int m_hi = m_lo + 8;
    for (int nt = 0; nt < TILES_N; nt++) {
        int c0 = bn + nt * 16 + (laneId % 4) * 2;
        int c1 = c0 + 1, c2 = c0 + 8, c3 = c2 + 1;
        if (m_lo < M) {
            if (c0 < N && c1 < N)
                *reinterpret_cast<half2*>(&C[m_lo * N + c0]) =
                    __floats2half2_rn(acc[nt][0][0], acc[nt][0][1]);
            if (c2 < N && c3 < N)
                *reinterpret_cast<half2*>(&C[m_lo * N + c2]) =
                    __floats2half2_rn(acc[nt][1][0], acc[nt][1][1]);
        }
        if (m_hi < M) {
            if (c0 < N && c1 < N)
                *reinterpret_cast<half2*>(&C[m_hi * N + c0]) =
                    __floats2half2_rn(acc[nt][0][2], acc[nt][0][3]);
            if (c2 < N && c3 < N)
                *reinterpret_cast<half2*>(&C[m_hi * N + c2]) =
                    __floats2half2_rn(acc[nt][1][2], acc[nt][1][3]);
        }
    }
}


// ---- Host wrapper ----
torch::Tensor gemm_int4_custom(
    torch::Tensor A_packed, torch::Tensor B_packed,
    torch::Tensor scales_A, torch::Tensor scales_B, int group_size)
{
    TORCH_CHECK(A_packed.is_cuda() && B_packed.is_cuda());
    TORCH_CHECK(A_packed.dtype() == torch::kUInt8);
    int M = A_packed.size(0), K = A_packed.size(1) * 2, N = B_packed.size(0);

    auto C = torch::zeros({M, N},
        torch::TensorOptions().dtype(torch::kHalf).device(A_packed.device()));

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(WARP_SZ * NUM_WARPS);
    // Triple-buffered shared memory (A/B tiles only — no SMEM scale staging)
    int smem = NUM_STAGES * (BLOCK_M * SMEM_STRIDE + BLOCK_N * SMEM_STRIDE);

    gemm_int4_kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
        A_packed.data_ptr<uint8_t>(), B_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, group_size);
    return C;
}
