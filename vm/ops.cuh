/* ===========================================================================================
 * AutoMegaKernel, THE CORE INSTRUCTION SET (Layer 1, device functions)
 * ===========================================================================================
 * Each opcode is exactly the abi.h Layer-1 archetype:
 *     __device__ void amk_inst_<name>(const amk_device_program&, const amk_instruction_t&);
 *   - reads prog.buffers[inst.inputs[i]], writes prog.buffers[inst.outputs[j]], reads inst.params,
 *     reads shape/stride from the buffer records;
 *   - PURE COMPUTE: no counters, no other buffers, no launches;
 *   - the WHOLE threadblock cooperates on one instruction (__syncthreads within is fine).
 *
 * Numerics MUST match instructions/reference.py (the ground truth the ReferenceVM oracle uses):
 *   COPY      : out = in (cast to out dtype)
 *   ADD       : out = (f32(a) + f32(b)) cast
 *   RMSNORM   : out = x * rsqrt(mean(x^2)+eps) * w           (mean over last dim, fp32 accumulate)
 *   GEMV_TILE : out[..., n_off:n_off+N_tile] = x @ W[n_off:n_off+N_tile, :].T   (W is [N,K])
 *   SILU_MUL  : out = silu(gate) * up = (gate * sigmoid(gate)) * up             (fp32, then cast)
 * fp32 + fp16 storage paths both go through amk_load_f / amk_store_f (always fp32 math), exactly
 * mirroring the reference's "accumulate in fp32, cast to output dtype" rule.
 * =========================================================================================== */
#ifndef AMK_OPS_CUH
#define AMK_OPS_CUH

#include <math_constants.h>   /* CUDART_INF_F (attention / argmax running max) */
#if (__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__)
#include <cuda_pipeline.h>    /* __pipeline_memcpy_async / commit / wait_prior (cp.async, sm_80+) */
#endif

#include "pages.cuh"

/* OCCUPANCY EXPERIMENT (GOAL 2): the megakernel inlines every opcode, so its register frame is the
 * WORST opcode's. Define -DAMK_OP_NOINLINE to mark the per-opcode device functions __noinline__:
 * the compiler then emits one body + a call, so the kernel's register frame can shrink toward the
 * cheap-opcode common case (potentially raising cudaOccupancyMaxActiveBlocksPerMultiprocessor).
 * Default keeps __forceinline__ (fewer call overheads on the hot single-stream path). This is an
 * HONEST A/B knob: vm/loader.py threads it as the `op_noinline` build flag; the bench reports
 * blocks/SM + numRegs both ways. NOTE: occupancy on sm_120 is ALSO capped by the GEMV's static
 * SMEM x-cache (AMK_GEMV_MAX_K), registers alone are not the only binder (see r2_vm_perf.json). */
#ifdef AMK_OP_NOINLINE
#define AMK_OP_QUAL __noinline__
#else
#define AMK_OP_QUAL __forceinline__
#endif

/* forward decl: the watchdog trap (defined with the dispatch table below) is used by the dynamic
 * head_dim guard in amk_inst_attention_tile to abort loudly rather than read OOB shared memory. */
__device__ __forceinline__ void amk_trap_unimplemented(const amk_device_program& prog, int op);

/* ---- COPY: outputs[0] = inputs[0] -----------------------------------------------------------
 * Elementwise over numel; dtypes may differ (cast through fp32). */
__device__ AMK_OP_QUAL void amk_inst_copy(const amk_device_program& prog,
                                              const amk_instruction_t& inst) {
    const amk_buffer_t& in  = amk_buf(prog, inst.inputs[0]);
    const amk_buffer_t& out = amk_buf(prog, inst.outputs[0]);
    int64_t n = out.numel;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        amk_store_f(out, i, amk_load_f(in, i));
    }
}

/* ---- ADD: outputs[0] = inputs[0] + inputs[1] (residual add) --------------------------------- */
__device__ AMK_OP_QUAL void amk_inst_add(const amk_device_program& prog,
                                             const amk_instruction_t& inst) {
    const amk_buffer_t& a   = amk_buf(prog, inst.inputs[0]);
    const amk_buffer_t& b   = amk_buf(prog, inst.inputs[1]);
    const amk_buffer_t& out = amk_buf(prog, inst.outputs[0]);
    int64_t n = out.numel;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        amk_store_f(out, i, amk_load_f(a, i) + amk_load_f(b, i));
    }
}

/* ---- RMSNORM: out = x / sqrt(mean(x^2)+eps) * w --------------------------------------------
 * inputs = [x (1,H or H), weight (H)]; reduction over the LAST dim (hidden). For the decode case
 * M=1 so we reduce one row of length H. We support M rows generally (each row independently),
 * cooperating across the whole block per row with a shared-memory reduction. */
__device__ AMK_OP_QUAL void amk_inst_rmsnorm(const amk_device_program& prog,
                                                 const amk_instruction_t& inst) {
    const amk_buffer_t& x   = amk_buf(prog, inst.inputs[0]);
    const amk_buffer_t& w   = amk_buf(prog, inst.inputs[1]);
    const amk_buffer_t& out = amk_buf(prog, inst.outputs[0]);
    const int   H   = inst.params.hidden;
    const float eps = inst.params.eps;
    const int64_t M = (H > 0) ? (out.numel / H) : 1;   /* rows (M=1 for decode gemv stream) */

    __shared__ float s_partial[1024 / 32];             /* one slot per warp (blockDim<=1024)  */

    for (int64_t r = 0; r < M; ++r) {
        const int64_t base = (int64_t)r * H;
        /* sum of squares over the row, fp32 */
        float local = 0.f;
        for (int j = threadIdx.x; j < H; j += blockDim.x) {
            float v = amk_load_f(x, base + j);
            local += v * v;
        }
        /* warp reduce then block reduce via shared memory */
        for (int o = warpSize / 2; o > 0; o >>= 1)
            local += __shfl_down_sync(0xffffffffu, local, o);
        int lane = threadIdx.x & (warpSize - 1);
        int warp = threadIdx.x / warpSize;
        if (lane == 0) s_partial[warp] = local;
        __syncthreads();
        float ssum = 0.f;
        int nwarps = (blockDim.x + warpSize - 1) / warpSize;
        if (warp == 0) {
            float v = (lane < nwarps) ? s_partial[lane] : 0.f;
            for (int o = warpSize / 2; o > 0; o >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, o);
            if (lane == 0) s_partial[0] = v;
        }
        __syncthreads();
        ssum = s_partial[0];
        float rms = rsqrtf(ssum / (float)H + eps);
        for (int j = threadIdx.x; j < H; j += blockDim.x) {
            float v = amk_load_f(x, base + j);
            amk_store_f(out, base + j, v * rms * amk_load_f(w, j));
        }
        __syncthreads();   /* reuse s_partial safely on the next row */
    }
}

/* ---- GEMV_TILE: out[..., n_off:n_off+N_tile] = x @ W[n_off:n_off+N_tile, :].T ----------------
 * inputs = [x (M,K), W (N,K) torch-Linear layout]; this task computes the column slice
 * [n_off, n_off+N_tile) of the output. Reference (_gemv_gemm):
 *     w_tile = W[n_off:n_off+N_tile, :]   # [N_tile, K]
 *     out_tile = x @ w_tile.T             # [M, N_tile]
 *     out[..., n_off:n_off+N_tile] = out_tile
 * fp32 accumulate (matches the reference + tensor-core fp32-accumulate rule).
 *
 * BANDWIDTH-EFFICIENT LAYOUT (the #1 decode win):
 *   Decode is HBM-bandwidth bound. W is row-major [N,K], so weight row n == output column n is a
 *   CONTIGUOUS run of K elements. We assign ONE WARP per output column (per weight row): the 32
 *   lanes stream that row in 128-bit vector loads, lane L reading the vector at index
 *   (kv*32 + L), so at every step the warp's 32 loads cover 32 ADJACENT 16-byte vectors == one
 *   contiguous 512-byte span -> fully COALESCED peak-bandwidth bursts (float4 for fp32, an 8-wide
 *   bf16/fp16 burst). Each lane fp32-accumulates; a warp-shuffle tree reduces to lane 0, which
 *   writes y[n]. The activation row x (length K) is cached once in shared memory as fp32, so the
 *   weight stream is the ONLY HBM traffic and the kernel reads W at (near) peak HBM bandwidth.
 *
 *   The v1 one-thread-per-column scheme made adjacent threads read addresses K apart (one element
 *   per 128B sector), ~1% of bandwidth. This warp-per-row + vectorized scheme is the coalesced fix.
 *
 *   Fast path requires x/W contiguous on K (stride[last]==1, the lowering's case) and K within the
 *   SMEM x-cache (<= AMK_GEMV_MAX_K); otherwise a general strided warp-per-row fallback via
 *   amk_load_f keeps correctness for any layout. M=1 for decode; the outer M loop is for generality. */

/* x (the activation row, length K) is cached in shared memory as fp32 once per warp-pass so every
 * warp's dot reads it from SMEM (not HBM) and the vectorized weight load is the only HBM traffic.
 *
 * COALESCING: lane L of a warp reads the 128-bit vector at vector-index (kv*warpSize + L). At each
 * kv step the warp's 32 lanes therefore touch 32 ADJACENT 128-bit vectors == one contiguous
 * 32*16 = 512-byte span -> fully coalesced peak-bandwidth bursts. Each lane unpacks its VEC scalars
 * (4 fp32 / 8 bf16) and fp32-accumulates against the cached x; a warp shuffle reduces to lane 0.
 * The fp32 product order (sum over k ascending within each lane's stride) matches the reference's
 * x @ w.T elementwise-then-sum closely enough for the frozen tolerance (verified by the tests). */
/* ---- AUTOTUNE KNOBS (compile-time, threaded by vm/loader.py as -D options) -------------------
 * These are the AutoKernel-style search dimensions for the bandwidth-bound decode GEMV. Each knob
 * is a macro consumed here and built into a DISTINCT extension variant (distinct -D set + distinct
 * extension name) by the loader, so vm/autotune.py can enumerate a grid, build each variant,
 * correctness-gate it, and keep the fastest. Defaults reproduce the prior v1-coalesced kernel.
 *
 *   AMK_GEMV_COLS_PER_WARP : output columns (== weight rows) a single warp computes per pass. The
 *       activation row x is cached ONCE in SMEM and REUSED across all C columns of the warp, so a
 *       single x-read serves C in-flight weight streams -> more memory-level parallelism per warp
 *       (the real lever for a bandwidth-bound GEMV). C accumulators live in registers.
 *   AMK_GEMV_KUNROLL       : how many float4/bf16x8 vectors each lane loads per loop iteration
 *       (instruction-level parallelism over the K reduction; more independent loads in flight).
 *   __launch_bounds__(AMK_LB_MAXTHREADS, AMK_LB_MINBLOCKS) on the megakernel (scheduler.cu) caps
 *       registers to RAISE the number of co-resident blocks/warps per SM == more MLP == higher
 *       achieved HBM bandwidth for this register-heavy persistent kernel. */
#ifndef AMK_GEMV_COLS_PER_WARP
#define AMK_GEMV_COLS_PER_WARP 1
#endif
#ifndef AMK_GEMV_KUNROLL
#define AMK_GEMV_KUNROLL 1
#endif
/* SMEM x-cache budget shared by the fp and quantized GEMV (8192 floats = 32KB static). */
#ifndef AMK_GEMV_MAX_K
#define AMK_GEMV_MAX_K 8192
#endif

/* ---- cp.async SOFTWARE-PIPELINED DOUBLE-BUFFER (the latency-hiding decode lever) -------------
 * The default register-load GEMV above is memory-LATENCY bound: each lane issues a float4 weight
 * load and immediately consumes it in the FMA, so only ~COLS_PER_WARP * KUNROLL loads are ever in
 * flight per warp, far below the ~300KB Little's-law working set needed to hide ~400ns HBM
 * latency, so it plateaus at ~48% of measured HBM bandwidth. The cp.async path DECOUPLES load
 * from compute: each warp asynchronously streams its weight-row K-chunks SMEM<-HBM with
 * __pipeline_memcpy_async into a multi-stage ring; while it computes the dot over stage s it has
 * STAGES-1 future chunks already in flight. That keeps many independent loads outstanding per warp
 * -> high memory-level parallelism -> approaches peak HBM bandwidth.
 *
 *   AMK_GEMV_CPASYNC_COLS : output columns (weight rows) one warp streams+computes at once. Each is
 *       an independent cp.async stream sharing the SMEM-cached x -> more MLP per warp.
 *   AMK_GEMV_CPA_STAGES   : ring depth (>=2; 3-4 gives the deepest in-flight pipeline that still
 *       fits the megakernel's co-resident SMEM budget).
 *   AMK_GEMV_CPA_VPL      : 16-byte vectors (float4 == 4 fp32 / 8 bf16-fp16) each lane copies per
 *       chunk. chunk width = warpSize*VPL*(elems per vec). Bigger chunk = more bytes per cp.async.
 * The staging ring lives in DYNAMIC shared memory provisioned by vm/loader.py (it cannot afford a
 * 2nd large static array atop the 32KB x-cache). fp32 elementwise-then-sum order is preserved, so
 * the result is bit-equal to the register path and matches the frozen reference tolerance. */
#ifndef AMK_GEMV_CPASYNC_COLS
#define AMK_GEMV_CPASYNC_COLS 2
#endif
#ifndef AMK_GEMV_CPA_STAGES
#define AMK_GEMV_CPA_STAGES 4
#endif
#ifndef AMK_GEMV_CPA_VPL
#define AMK_GEMV_CPA_VPL 1
#endif

__device__ __forceinline__ float amk_gemv_row_dot_f32(const float* __restrict__ xs,
                                                      const float* __restrict__ wrow,
                                                      int K, int lane) {
    float acc = 0.f;
    const int Kv = K / 4;                                   /* float4 vectors */
    const float4* w4 = (const float4*)wrow;
    for (int kv = lane; kv < Kv; kv += warpSize) {
        const float4 wv = w4[kv];
        const int kb = kv * 4;
        acc += xs[kb + 0] * wv.x + xs[kb + 1] * wv.y
             + xs[kb + 2] * wv.z + xs[kb + 3] * wv.w;
    }
    for (int k = Kv * 4 + lane; k < K; k += warpSize)
        acc += xs[k] * wrow[k];
    return acc;
}

/* ---- COLS-PER-WARP dot kernels (x-reuse): one warp streams C contiguous weight rows at once and
 * fp32-accumulates C independent dot products against the SHARED cached x. Each lane reads C float4
 * vectors per kv step (C*512B coalesced spans), unrolled AMK_GEMV_KUNROLL kv-steps deep. The C
 * accumulators are reduced to lane 0 by the caller. Same fp32 elementwise-then-sum order as the
 * 1-column path, so the result is bit-equal to it and matches the reference tolerance. */
template <int C>
__device__ __forceinline__ void amk_gemv_rows_dot_f32(const float* __restrict__ xs,
                                                      const float* __restrict__ wbase,
                                                      int K, int64_t wstride, int lane,
                                                      float acc[C]) {
    #pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = 0.f;
    const int Kv = K / 4;
    const int step = warpSize * AMK_GEMV_KUNROLL;
    int kv = lane;
    for (; kv + warpSize * (AMK_GEMV_KUNROLL - 1) < Kv; kv += step) {
        #pragma unroll
        for (int u = 0; u < AMK_GEMV_KUNROLL; ++u) {
            const int kvu = kv + u * warpSize;
            const int kb  = kvu * 4;
            #pragma unroll
            for (int c = 0; c < C; ++c) {
                const float4 wv = ((const float4*)(wbase + c * wstride))[kvu];
                acc[c] += xs[kb + 0] * wv.x + xs[kb + 1] * wv.y
                        + xs[kb + 2] * wv.z + xs[kb + 3] * wv.w;
            }
        }
    }
    for (; kv < Kv; kv += warpSize) {
        const int kb = kv * 4;
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            const float4 wv = ((const float4*)(wbase + c * wstride))[kv];
            acc[c] += xs[kb + 0] * wv.x + xs[kb + 1] * wv.y
                    + xs[kb + 2] * wv.z + xs[kb + 3] * wv.w;
        }
    }
    for (int k = Kv * 4 + lane; k < K; k += warpSize) {
        #pragma unroll
        for (int c = 0; c < C; ++c) acc[c] += xs[k] * (wbase + c * wstride)[k];
    }
}

template <int C>
__device__ __forceinline__ void amk_gemv_rows_dot_bf16(const float* __restrict__ xs,
                                                       const __nv_bfloat16* __restrict__ wbase,
                                                       int K, int64_t wstride, int lane,
                                                       float acc[C]) {
    #pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = 0.f;
    const int Kv = K / 8;                                   /* 8 bf16 == 16 bytes == float4 */
    for (int kv = lane; kv < Kv; kv += warpSize) {
        const int kb = kv * 8;
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            const float4 raw = ((const float4*)(wbase + c * wstride))[kv];
            const __nv_bfloat16* h = (const __nv_bfloat16*)&raw;
            #pragma unroll
            for (int e = 0; e < 8; ++e) acc[c] += xs[kb + e] * __bfloat162float(h[e]);
        }
    }
    for (int k = Kv * 8 + lane; k < K; k += warpSize) {
        #pragma unroll
        for (int c = 0; c < C; ++c) acc[c] += xs[k] * __bfloat162float((wbase + c * wstride)[k]);
    }
}

template <int C>
__device__ __forceinline__ void amk_gemv_rows_dot_f16(const float* __restrict__ xs,
                                                      const __half* __restrict__ wbase,
                                                      int K, int64_t wstride, int lane,
                                                      float acc[C]) {
    #pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = 0.f;
    const int Kv = K / 8;
    for (int kv = lane; kv < Kv; kv += warpSize) {
        const int kb = kv * 8;
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            const float4 raw = ((const float4*)(wbase + c * wstride))[kv];
            const __half* h = (const __half*)&raw;
            #pragma unroll
            for (int e = 0; e < 8; ++e) acc[c] += xs[kb + e] * __half2float(h[e]);
        }
    }
    for (int k = Kv * 8 + lane; k < K; k += warpSize) {
        #pragma unroll
        for (int c = 0; c < C; ++c) acc[c] += xs[k] * __half2float((wbase + c * wstride)[k]);
    }
}

/* 8-wide (128-bit) coalesced bf16/fp16 weight load: lane reads a float4 of packed halves. */
__device__ __forceinline__ float amk_gemv_row_dot_bf16(const float* __restrict__ xs,
                                                       const __nv_bfloat16* __restrict__ wrow,
                                                       int K, int lane) {
    float acc = 0.f;
    const int Kv = K / 8;                                   /* 8 bf16 == 16 bytes == float4 */
    const float4* w4 = (const float4*)wrow;
    for (int kv = lane; kv < Kv; kv += warpSize) {
        const float4 raw = w4[kv];
        const __nv_bfloat16* h = (const __nv_bfloat16*)&raw;
        const int kb = kv * 8;
        #pragma unroll
        for (int e = 0; e < 8; ++e)
            acc += xs[kb + e] * __bfloat162float(h[e]);
    }
    for (int k = Kv * 8 + lane; k < K; k += warpSize)
        acc += xs[k] * __bfloat162float(wrow[k]);
    return acc;
}
__device__ __forceinline__ float amk_gemv_row_dot_f16(const float* __restrict__ xs,
                                                      const __half* __restrict__ wrow,
                                                      int K, int lane) {
    float acc = 0.f;
    const int Kv = K / 8;
    const float4* w4 = (const float4*)wrow;
    for (int kv = lane; kv < Kv; kv += warpSize) {
        const float4 raw = w4[kv];
        const __half* h = (const __half*)&raw;
        const int kb = kv * 8;
        #pragma unroll
        for (int e = 0; e < 8; ++e)
            acc += xs[kb + e] * __half2float(h[e]);
    }
    for (int k = Kv * 8 + lane; k < K; k += warpSize)
        acc += xs[k] * __half2float(wrow[k]);
    return acc;
}

/* ---- DEQUANT-FUSED QUANTIZED GEMV (int4 / int8 weight-only) ---------------------------------
 * out[..., n] = sum_k x[k] * dequant(W[n,k]) , where dequant(W[n,k]) = (q - zero[n,g]) * scale[n,g]
 * and g = k / group. inputs = [x, qW(I4 packed uint8 / I8 int8), scales(fp16 [N,n_groups])
 * (, zeros fp16 [N,n_groups])]. params.qdtype = AMK_I4/AMK_I8, params.group = group size.
 *
 * This is the WHOLE POINT of the bandwidth lever: we read the PACKED int4 weight rows coalesced
 * (one byte holds two columns), unpack + apply the per-group scale in REGISTERS, fp32-accumulate,
 * and write the fp16/bf16/fp32 output, the fp weight is NEVER materialized in HBM. int4 weight
 * traffic is ~4x less than bf16, dropping the decode roofline floor ~4x.
 *
 * One warp owns one output column n (= weight row n). The x row is cached in SMEM (shared across
 * all columns the block computes). fp32 elementwise-then-sum order matches the reference exactly
 * (the per-group scale is applied per element, like instructions/reference.py dequant-then-matmul:
 * x@(q*scale) == sum_k x[k]*q[k]*scale[g(k)]). Correctness is bit/ulp-checked vs the ReferenceVM. */
__device__ AMK_OP_QUAL void amk_inst_gemv_tile_quant(const amk_device_program& prog,
                                                         const amk_instruction_t& inst) {
    const amk_buffer_t& x   = amk_buf(prog, inst.inputs[0]);   /* [M,K] fp */
    const amk_buffer_t& W   = amk_buf(prog, inst.inputs[1]);   /* packed I4 / I8 weight [N,K] */
    const amk_buffer_t& S   = amk_buf(prog, inst.inputs[2]);   /* scales fp16 [N, n_groups] */
    const bool has_zeros    = (inst.n_inputs > 3);
    const int K      = inst.params.K;
    const int N_tile = inst.params.N_tile;
    const int n_off  = inst.params.n_off;
    const int group  = (inst.params.group > 0) ? inst.params.group : K;
    const int qdtype = inst.params.qdtype;
    const int n_groups = (K + group - 1) / group;

    const amk_buffer_t& out = amk_buf(prog, inst.outputs[0]);  /* [M,N] */
    const int64_t Nf = out.shape[1];                          /* full output width */
    const int64_t M  = (Nf > 0) ? (out.numel / Nf) : 1;

    const int lane   = threadIdx.x & (warpSize - 1);
    const int warp   = threadIdx.x / warpSize;
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;

    /* scales/zeros pointers (fp16). zeros optional (asymmetric). */
    const __half* __restrict__ Sp = (const __half*)S.ptr;
    const __half* __restrict__ Zp = has_zeros ? (const __half*)amk_buf(prog, inst.inputs[3]).ptr
                                              : (const __half*)nullptr;

    /* DYNAMIC-SMEM x-cache: the loader (vm/loader.py) opts the kernel into >= min(maxK,8192)*4 bytes
     * of dynamic shared memory for the quantized GEMV (it CANNOT afford a 2nd large static array on
     * top of the fp GEMV's 32KB). We cache the x row once per block so every warp's weight stream
     * multiplies against SMEM x (not repeated global reads). K>cap falls back to L2 global x. */
#ifndef AMK_QGEMV_SMEM_MAX_K
#define AMK_QGEMV_SMEM_MAX_K 16384   /* quant GEMV DYNAMIC-smem x-cache cap (floats); must match vm/loader.py */
#endif
    extern __shared__ float s_dyn[];
    const int64_t xls = x.stride[x.rank - 1];
    /* K<=cap: x cached in dynamic SMEM (fast). K>cap: x read from (L2-resident) global below, the
     * vectorized SMEM loops are GUARDED on use_smem_x so they never read past the cached region. */
    const bool use_smem_x = (xls == 1) && (K <= AMK_QGEMV_SMEM_MAX_K);
    float* const xs = s_dyn;

    const uint8_t* __restrict__ Wbytes = (const uint8_t*)W.ptr;
    const int64_t row_bytes_i4 = (int64_t)((K + 1) / 2);     /* packed bytes per row */
    const int8_t*  __restrict__ Wi8 = (const int8_t*)W.ptr;
    const int64_t row_elems_i8 = (int64_t)K;
    const int Kv4 = K / 32;                                  /* uint4 = 16 bytes = 32 int4 nibbles */
    const int Ki8v = K / 16;                                 /* uint4 = 16 bytes = 16 int8 weights */

#if defined(AMK_QGEMV_CPASYNC) && ((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
    /* INT8 cp.async SOFTWARE-PIPELINED ring (sm_80+, opt-in via knob 'qcpasync'), mirror of the fp
     * amk_gemv_tile_cpasync, but it stages RAW int8 weight granules into a per-warp SMEM ring and
     * decodes the per-GROUP fp16 scale in the compute phase. The synchronous int8 path below issues
     * one uint4 weight load and immediately consumes it (only ~QC loads in flight -> HBM-latency
     * bound); the ring keeps STAGES-1 future chunks in flight -> deep MLP -> hides the load latency.
     * Needs K%16==0 (16B cp.async granule alignment of each row) else falls to the sync path. fp32
     * accumulate (per-granule scale folded out); correct vs the dequant reference (NOT bit-identical
     * to the sync accumulation order, fine for this opt-in experimental variant). SMEM ring lives
     * after the x-cache; vm/loader.py _qgemv_cpasync_smem provisions x_bytes + ring_bytes. */
    if (qdtype == AMK_I8 && use_smem_x && (row_elems_i8 % 16 == 0)) {
        const int CPC = AMK_GEMV_CPASYNC_COLS, STAGES = AMK_GEMV_CPA_STAGES, VPL = AMK_GEMV_CPA_VPL;
        const int VEC = 16;                                   /* int8 weights per uint4 granule */
        const int CHUNK = warpSize * VPL * VEC;               /* int8 per column per chunk */
        const int x_floats = (K + 3) & ~3;                    /* 16B-align the ring after the x-cache */
        int8_t* const ring = (int8_t*)(s_dyn + x_floats);
        const int rper = STAGES * CPC * CHUNK;                /* int8 ring elems per warp */
        const int stage_stride = CPC * CHUNK;
        const int nchunks = K / CHUNK;
        const int tail0 = nchunks * CHUNK;
        for (int64_t m = 0; m < M; ++m) {
            const int64_t x_row = m * x.stride[0];
            for (int j = threadIdx.x; j < K; j += blockDim.x) xs[j] = amk_load_f(x, x_row + j);
            __syncthreads();
            for (int t = warp * CPC; t < N_tile; t += nwarps * CPC) {
                const int cols = (t + CPC <= N_tile) ? CPC : (N_tile - t);
                const int n0 = n_off + t;
                float acc[AMK_GEMV_CPASYNC_COLS];
                #pragma unroll
                for (int c = 0; c < AMK_GEMV_CPASYNC_COLS; ++c) acc[c] = 0.f;
                int8_t* const wring = ring + (int64_t)warp * rper;
                #define AMK_QCPA_LOAD(ci) do {                                                        \
                    const int _st = (ci) % STAGES; int8_t* _sb = wring + (int64_t)_st * stage_stride; \
                    const int _kb = (ci) * CHUNK;                                                     \
                    _Pragma("unroll") for (int _c = 0; _c < AMK_GEMV_CPASYNC_COLS; ++_c) {            \
                        if (_c >= cols) break;                                                        \
                        const uint4* _src = (const uint4*)(Wi8 + (int64_t)(n0 + _c) * row_elems_i8 + _kb); \
                        uint4* _dst = (uint4*)(_sb + _c * CHUNK);                                     \
                        _Pragma("unroll") for (int _v = 0; _v < AMK_GEMV_CPA_VPL; ++_v)               \
                            __pipeline_memcpy_async(_dst + (_v * warpSize + lane),                    \
                                                    _src + (_v * warpSize + lane), sizeof(uint4));    \
                    } __pipeline_commit(); } while (0)
                int committed = 0; const int prime = (STAGES - 1 < nchunks) ? (STAGES - 1) : nchunks;
                #pragma unroll 1
                for (int ci = 0; ci < prime; ++ci) { AMK_QCPA_LOAD(ci); ++committed; }
                #pragma unroll 1
                for (int ci = 0; ci < nchunks; ++ci) {
                    const int nx = ci + (STAGES - 1);
                    if (nx < nchunks) { AMK_QCPA_LOAD(nx); ++committed; }
                    const int remain = committed - ci - 1;
                    __pipeline_wait_prior(remain > 0 ? remain : 0);
                    const int st = ci % STAGES; const int8_t* sb = wring + (int64_t)st * stage_stride;
                    const int kbase = ci * CHUNK;
                    #pragma unroll
                    for (int v = 0; v < AMK_GEMV_CPA_VPL; ++v) {
                        const int eoff = (v * warpSize + lane) * VEC;        /* int8 offset within chunk */
                        const int g = (kbase + eoff) / group;               /* one group per 16-granule */
                        #pragma unroll
                        for (int c = 0; c < AMK_GEMV_CPASYNC_COLS; ++c) {
                            if (c >= cols) break;
                            const int64_t srow = (int64_t)(n0 + c) * n_groups;
                            const float sc = __half2float(Sp[srow + g]);
                            const float z  = has_zeros ? __half2float(Zp[srow + g]) : 0.f;
                            const int8_t* col = sb + c * CHUNK + eoff;
                            float p = 0.f;
                            #pragma unroll
                            for (int e = 0; e < VEC; ++e) p += xs[kbase + eoff + e] * ((float)col[e] - z);
                            acc[c] += p * sc;
                        }
                    }
                }
                __pipeline_wait_prior(0);
                #undef AMK_QCPA_LOAD
                for (int k = tail0 + lane; k < K; k += warpSize) {           /* ragged K tail */
                    const float xvk = xs[k];
                    #pragma unroll
                    for (int c = 0; c < AMK_GEMV_CPASYNC_COLS; ++c) {
                        if (c >= cols) break;
                        const int64_t srow = (int64_t)(n0 + c) * n_groups;
                        const int g = k / group; const float sc = __half2float(Sp[srow + g]);
                        float qv = (float)(Wi8 + (int64_t)(n0 + c) * row_elems_i8)[k];
                        if (has_zeros) qv -= __half2float(Zp[srow + g]);
                        acc[c] += xvk * (qv * sc);
                    }
                }
                #pragma unroll
                for (int c = 0; c < AMK_GEMV_CPASYNC_COLS; ++c) {
                    float a = acc[c];
                    #pragma unroll
                    for (int o = warpSize / 2; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
                    if (lane == 0 && c < cols) {
                        const int n = n0 + c;
                        amk_store_f(out, m * out.stride[0] + (int64_t)n * out.stride[1], a);
                    }
                }
                __syncwarp();
            }
            __syncthreads();
        }
        return;
    }
#endif

    for (int64_t m = 0; m < M; ++m) {
        const int64_t x_row = m * x.stride[0];
        if (use_smem_x) {
            for (int j = threadIdx.x; j < K; j += blockDim.x)
                xs[j] = amk_load_f(x, x_row + j);
            __syncthreads();
        }
        #define AMK_QX(k) (use_smem_x ? xs[(k)] : amk_load_f(x, x_row + (int64_t)(k) * xls))
        /* COLS-PER-WARP x-reuse: each warp owns a run of QC consecutive output columns (weight
         * rows). The SMEM-cached x is read once and reused across all QC weight streams, and the
         * QC accumulators are independent (ILP that hides the dequant ALU + load latency). This is
         * the same lever that made the fp GEMV bandwidth-bound; here it amortizes the unpack work. */
        #ifndef AMK_QGEMV_COLS
        #define AMK_QGEMV_COLS 4   /* quantized GEMV output cols/warp; overridable via -D (knob 'qc') */
        #endif
        const int QC = AMK_QGEMV_COLS;
        /* group must be a multiple of the vector chunk (32 for int4, 16 for int8) so that all
         * weights of a 128-bit load share ONE group scale (decoded once per chunk). The quantizer
         * uses group multiples of 64, so this holds; if a future group breaks it, the vectorized
         * chunk would mix two groups' scales, assert it in the loader. The K-tail handles the
         * non-multiple remainder per element. */
        for (int t = warp * QC; t < N_tile; t += nwarps * QC) {
            const int cols = (t + QC <= N_tile) ? QC : (N_tile - t);
            float acc[AMK_QGEMV_COLS];
            #pragma unroll
            for (int c = 0; c < QC; ++c) acc[c] = 0.f;
            const int n0 = n_off + t;
            if (qdtype == AMK_I8) {
                /* INT8 weight-only GEMV, the headline lossless lever.
                 * Each lane reads a 128-bit uint4 == 16 contiguous int8 weights (coalesced; adjacent
                 * lanes read adjacent 16B sectors). Reinterpreting the uint4 as char4x4 lets the
                 * compiler keep the bytes in registers. The per-GROUP fp16 scale is decoded ONCE per
                 * 16-chunk (group=128 is a multiple of 16, so all 16 weights of a chunk share one
                 * group, no per-element scale lookup). x is read straight from SMEM (xs[kbase+e]),
                 * NOT staged into a per-lane register array (that array was the register-pressure /
                 * spill source that made the old kernel SLOWER than bf16). The QC independent column
                 * accumulators give ILP that hides the load latency + the (cheap) int->float + FMA. */
                if (use_smem_x) for (int kv = lane; kv < Ki8v; kv += warpSize) {
                    const int kbase = kv * 16;
                    const int g = kbase / group;                 /* one group per 16-chunk */
                    #pragma unroll
                    for (int c = 0; c < QC; ++c) {
                        if (c >= cols) break;
                        const int n = n0 + c; const int64_t srow = (int64_t)n * n_groups;
                        const uint4 raw = ((const uint4*)(Wi8 + (int64_t)n * row_elems_i8))[kv];
                        const char4 b0 = *(const char4*)&raw.x;
                        const char4 b1 = *(const char4*)&raw.y;
                        const char4 b2 = *(const char4*)&raw.z;
                        const char4 b3 = *(const char4*)&raw.w;
                        const float sc = __half2float(Sp[srow + g]);
                        const float z  = has_zeros ? __half2float(Zp[srow + g]) : 0.f;
                        /* fp32 dot of the 16 (q*scale) against the SMEM x; scale applied to the
                         * partial sum once (sum(x*q)*sc), identical fp32 order to the reference's
                         * x@(q*scale) within a group (a group-constant scale factors out). A 4-way
                         * independent-accumulator variant was measured and REVERTED: the extra fp32
                         * registers cut occupancy and it was neutral-to-slower (the kernel is
                         * occupancy/memory-bound here, not FMA-latency-bound). */
                        float p = 0.f;
                        p += xs[kbase +  0] * ((float)b0.x - z); p += xs[kbase +  1] * ((float)b0.y - z);
                        p += xs[kbase +  2] * ((float)b0.z - z); p += xs[kbase +  3] * ((float)b0.w - z);
                        p += xs[kbase +  4] * ((float)b1.x - z); p += xs[kbase +  5] * ((float)b1.y - z);
                        p += xs[kbase +  6] * ((float)b1.z - z); p += xs[kbase +  7] * ((float)b1.w - z);
                        p += xs[kbase +  8] * ((float)b2.x - z); p += xs[kbase +  9] * ((float)b2.y - z);
                        p += xs[kbase + 10] * ((float)b2.z - z); p += xs[kbase + 11] * ((float)b2.w - z);
                        p += xs[kbase + 12] * ((float)b3.x - z); p += xs[kbase + 13] * ((float)b3.y - z);
                        p += xs[kbase + 14] * ((float)b3.z - z); p += xs[kbase + 15] * ((float)b3.w - z);
                        acc[c] += p * sc;
                    }
                }
                for (int k = (use_smem_x ? Ki8v * 16 : 0) + lane; k < K; k += warpSize) {  /* tail (or full GEMV when K > smem cap) */
                    const float xvk = AMK_QX(k);
                    #pragma unroll
                    for (int c = 0; c < QC; ++c) {
                        if (c >= cols) break;
                        const int n = n0 + c; const int64_t srow = (int64_t)n * n_groups;
                        const int g = k / group; const float sc = __half2float(Sp[srow + g]);
                        float qv = (float)(Wi8 + (int64_t)n * row_elems_i8)[k];
                        if (has_zeros) qv -= __half2float(Zp[srow + g]);
                        acc[c] += xvk * (qv * sc);
                    }
                }
            } else { /* AMK_I4 */
                /* INT4 weight-only GEMV. Each lane reads a 128-bit uint4 == 16 packed bytes == 32
                 * nibbles (coalesced). Unpack with masks/shifts directly into the FMA against SMEM x;
                 * the per-group scale is decoded ONCE per 32-chunk (group=128 is a multiple of 32) and
                 * applied to the partial sum. No xv[] register staging (same fix as int8). */
                if (use_smem_x) for (int kv = lane; kv < Kv4; kv += warpSize) {
                    const int kbase = kv * 32;
                    const int g = kbase / group;                 /* one group per 32-chunk */
                    #pragma unroll
                    for (int c = 0; c < QC; ++c) {
                        if (c >= cols) break;
                        const int n = n0 + c; const int64_t srow = (int64_t)n * n_groups;
                        const uint4 raw = ((const uint4*)(Wbytes + (int64_t)n * row_bytes_i4))[kv];
                        const uint32_t w0 = raw.x, w1 = raw.y, w2 = raw.z, w3 = raw.w;
                        const float sc = __half2float(Sp[srow + g]);
                        const float z  = has_zeros ? __half2float(Zp[srow + g]) : 8.f; /* sym: q-8 */
                        float p = 0.f;
                        #pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            p += xs[kbase +      e] * ((float)((w0 >> (e*4)) & 0xF) - z);
                            p += xs[kbase +  8 + e] * ((float)((w1 >> (e*4)) & 0xF) - z);
                            p += xs[kbase + 16 + e] * ((float)((w2 >> (e*4)) & 0xF) - z);
                            p += xs[kbase + 24 + e] * ((float)((w3 >> (e*4)) & 0xF) - z);
                        }
                        acc[c] += p * sc;
                    }
                }
                for (int k = (use_smem_x ? Kv4 * 32 : 0) + lane; k < K; k += warpSize) {  /* tail (or full GEMV when K > smem cap) */
                    const float xvk = AMK_QX(k);
                    #pragma unroll
                    for (int c = 0; c < QC; ++c) {
                        if (c >= cols) break;
                        const int n = n0 + c; const int64_t srow = (int64_t)n * n_groups;
                        const int g = k / group; const float sc = __half2float(Sp[srow + g]);
                        int q = amk_unpack_i4_u(Wbytes + (int64_t)n * row_bytes_i4, k);
                        float qv = has_zeros ? ((float)q - __half2float(Zp[srow + g])) : (float)(q - 8);
                        acc[c] += xvk * (qv * sc);
                    }
                }
            }
            #undef AMK_QX
            /* warp-shuffle reduce each column to lane 0 (fp32) */
            #pragma unroll
            for (int c = 0; c < QC; ++c) {
                float a = acc[c];
                #pragma unroll
                for (int o = warpSize / 2; o > 0; o >>= 1)
                    a += __shfl_down_sync(0xffffffffu, a, o);
                if (lane == 0 && c < cols) {
                    const int n = n0 + c;
                    amk_store_f(out, m * out.stride[0] + (int64_t)n * out.stride[1], a);
                }
            }
        }
        #undef AMK_QGEMV_COLS
        if (use_smem_x) __syncthreads();
    }
}

/* =============================================================================================
 * cp.async SOFTWARE-PIPELINED DOUBLE-BUFFERED fp GEMV (the decode latency-hiding kernel)
 * =============================================================================================
 * out[..,n] = x @ W[n,:]^T for n in [n_off, n_off+N_tile).  W is row-major [N,K], fp32/bf16/fp16.
 *
 * Mapping: one warp owns a run of CPC = AMK_GEMV_CPASYNC_COLS consecutive output columns (weight
 * rows). It streams each row's K dimension in CHUNK-element pieces. A STAGES-deep ring of SMEM
 * buffers (one [CPC][CHUNK] slab per stage, per warp) is filled by __pipeline_memcpy_async: at the
 * top we PRIME the first STAGES-1 chunks (commit one pipeline group per chunk), then for each
 * compute chunk we wait on the OLDEST in-flight group, FMA its CHUNK elements against the cached x,
 * and immediately issue the async copy for the next-not-yet-requested chunk. So while the warp
 * computes chunk c it already has STAGES-1 future chunks' loads outstanding -> deep MLP that hides
 * HBM latency (the whole point: many independent loads in flight, Little's-law working set met).
 *
 * cp.async copies 16-byte (float4) granules straight HBM->SMEM, bypassing the register file; each
 * lane copies AMK_GEMV_CPA_VPL granules per chunk, so CHUNK = warpSize*VPL*VEC elements (VEC = 4
 * fp32 / 8 half). The compute reads the staged weights from SMEM and the activation from the static
 * x-cache; fp32 accumulate, same elementwise-then-sum order as the register path (bit-equal result).
 *
 * The ring is carved from DYNAMIC shared memory (vm/loader.py provisions
 * ring_bytes = STAGES*nwarps*CPC*CHUNK_bytes for the program's GEMV tiles); x stays in the static
 * 32KB cache. Requires sm_80+ (cp.async); the loader only routes here when the device supports it.
 * Any non-fast-path case (strided operands, K below one full chunk, dyn-smem not provisioned) falls
 * back to the proven register warp-per-row path for correctness. */
#if (__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__)
/* decode one staged weight element (SMEM) to fp32; one overload per storage type. */
__device__ __forceinline__ float amk_w_to_f32(const float*         p, int e) { return p[e]; }
__device__ __forceinline__ float amk_w_to_f32(const __nv_bfloat16* p, int e) { return __bfloat162float(p[e]); }
__device__ __forceinline__ float amk_w_to_f32(const __half*        p, int e) { return __half2float(p[e]); }

template <typename WT, int VEC>
__device__ __forceinline__ void amk_gemv_tile_cpasync(const amk_device_program& prog,
                                                      const amk_instruction_t& inst,
                                                      const float* __restrict__ xs,
                                                      WT* __restrict__ ring, int ring_elems_per_warp,
                                                      int K, int N_tile, int n_off) {
    const int lane   = threadIdx.x & (warpSize - 1);
    const int warp   = threadIdx.x / warpSize;
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;

    const amk_buffer_t& W   = amk_buf(prog, inst.inputs[1]);
    const amk_buffer_t& out = amk_buf(prog, inst.outputs[0]);
    const int64_t N_full = out.shape[1];
    const int64_t M      = (N_full > 0) ? (out.numel / N_full) : 1;
    const int64_t wstride = W.stride[0];                 /* elements per weight row (==K, row-major) */
    const WT* __restrict__ Wp = (const WT*)W.ptr;

    const int CPC    = AMK_GEMV_CPASYNC_COLS;
    const int STAGES = AMK_GEMV_CPA_STAGES;
    const int VPL    = AMK_GEMV_CPA_VPL;
    const int CHUNK  = warpSize * VPL * VEC;              /* weight elems per column per chunk */
    const int nchunks = K / CHUNK;                        /* whole chunks; tail handled per-element */
    const int tail0   = nchunks * CHUNK;                  /* first K index of the ragged tail */

    /* this warp's private ring region: [STAGES][CPC][CHUNK] of WT */
    WT* const wring = ring + (int64_t)warp * ring_elems_per_warp;
    const int stage_stride = CPC * CHUNK;                 /* elems per stage in this warp's ring */

    for (int64_t m = 0; m < M; ++m) {
        for (int t = warp * CPC; t < N_tile; t += nwarps * CPC) {
            const int cols = (t + CPC <= N_tile) ? CPC : (N_tile - t);
            const int n0   = n_off + t;
            float acc[AMK_GEMV_CPASYNC_COLS];
            #pragma unroll
            for (int c = 0; c < AMK_GEMV_CPASYNC_COLS; ++c) acc[c] = 0.f;

            /* --- async copy of chunk `ci` (cols [0,cols)) into ring stage `ci % STAGES` --- */
            #define AMK_CPA_LOAD(ci)                                                              \
            do {                                                                                  \
                const int _st = (ci) % STAGES;                                                    \
                WT* _sb = wring + (int64_t)_st * stage_stride;                                    \
                const int _kbase = (ci) * CHUNK;                                                  \
                _Pragma("unroll")                                                                 \
                for (int _c = 0; _c < AMK_GEMV_CPASYNC_COLS; ++_c) {                              \
                    if (_c >= cols) break;                                                        \
                    const float4* _src = (const float4*)(Wp + (int64_t)(n0 + _c) * wstride + _kbase); \
                    float4* _dst = (float4*)(_sb + _c * CHUNK);                                   \
                    _Pragma("unroll")                                                             \
                    for (int _v = 0; _v < AMK_GEMV_CPA_VPL; ++_v)                                 \
                        __pipeline_memcpy_async(_dst + (_v * warpSize + lane),                    \
                                                _src + (_v * warpSize + lane), sizeof(float4));   \
                }                                                                                 \
                __pipeline_commit();                                                             \
            } while (0)

            /* prime the pipeline: issue the first min(STAGES-1, nchunks) chunk copies. `committed`
             * tracks how many cp.async groups we have committed; groups COMPLETE in FIFO order, so
             * before consuming chunk ci (the ci-th to complete) we wait until exactly
             * (committed - ci - 1) groups remain in flight. That is correct even for tiny nchunks
             * (where committed < STAGES-1), unlike a fixed wait_prior(STAGES-1). */
            int committed = 0;
            const int prime = (STAGES - 1 < nchunks) ? (STAGES - 1) : nchunks;
            #pragma unroll 1
            for (int ci = 0; ci < prime; ++ci) { AMK_CPA_LOAD(ci); ++committed; }

            /* steady state: for each compute chunk, wait oldest, FMA, then request next chunk */
            #pragma unroll 1
            for (int ci = 0; ci < nchunks; ++ci) {
                const int next = ci + (STAGES - 1);
                if (next < nchunks) { AMK_CPA_LOAD(next); ++committed; }
                /* ensure chunk ci has landed: leave (committed - ci - 1) groups in flight */
                const int remain = committed - ci - 1;
                __pipeline_wait_prior(remain > 0 ? remain : 0);
                const int st = ci % STAGES;
                const WT* sb = wring + (int64_t)st * stage_stride;
                const int kbase = ci * CHUNK;
                /* lane L owns the SAME granules it cp.async-copied: float4 vector v*warpSize+L for
                 * v in [0,VPL) -> chunk elements [(v*warpSize+L)*VEC, +VEC). Disjoint over lanes,
                 * covering the whole chunk exactly once. fp32 accumulate (per-lane partials reduced
                 * by the warp shuffle below); ascending-k within a lane matches the reg path order. */
                #pragma unroll
                for (int v = 0; v < AMK_GEMV_CPA_VPL; ++v) {
                    const int eoff = (v * warpSize + lane) * VEC;   /* element offset within chunk */
                    #pragma unroll
                    for (int c = 0; c < AMK_GEMV_CPASYNC_COLS; ++c) {
                        if (c >= cols) break;
                        const WT* col = sb + c * CHUNK + eoff;
                        #pragma unroll
                        for (int e = 0; e < VEC; ++e)
                            acc[c] += xs[kbase + eoff + e] * amk_w_to_f32(col, e);
                    }
                }
            }
            __pipeline_wait_prior(0);   /* drain any still-in-flight copies before ring reuse */
            #undef AMK_CPA_LOAD

            /* ragged K tail (K not a multiple of CHUNK): plain coalesced register loads from HBM */
            for (int k = tail0 + lane; k < K; k += warpSize) {
                const float xv = xs[k];
                #pragma unroll
                for (int c = 0; c < AMK_GEMV_CPASYNC_COLS; ++c) {
                    if (c >= cols) break;
                    acc[c] += xv * amk_w_to_f32(Wp + (int64_t)(n0 + c) * wstride + k, 0);
                }
            }

            /* warp-reduce each column to lane 0 and store */
            #pragma unroll
            for (int c = 0; c < AMK_GEMV_CPASYNC_COLS; ++c) {
                float a = acc[c];
                #pragma unroll
                for (int o = warpSize / 2; o > 0; o >>= 1)
                    a += __shfl_down_sync(0xffffffffu, a, o);
                if (lane == 0 && c < cols) {
                    const int n = n0 + c;
                    amk_store_f(out, m * out.stride[0] + (int64_t)n * out.stride[1], a);
                }
            }
            __syncwarp();
        }
    }
}
#endif /* sm_80+ cp.async */

/* ---- GEMV ABLATION TOGGLE (reproducible coalescing A/B) -------------------------------------
 * Define AMK_GEMV_SCALAR at compile time (-DAMK_GEMV_SCALAR, threaded through vm/loader.py as the
 * `gemv_scalar=True` kwarg / AMK_GEMV_SCALAR=1 env) to select the OLD naive v1 GEMV: ONE THREAD per
 * output column (per weight row). Adjacent threads then read weight addresses K elements apart, one
 * element per 128B sector == ~1% of HBM bandwidth (UNCOALESCED). This is the pre-optimization path
 * the headline coalescing win is measured against. The DEFAULT (toggle undefined) is the current
 * warp-per-column vectorized float4 / bf16x8 / fp16x8 COALESCED path above. Both are
 * correctness-checked against the reference; the only difference is the memory access pattern, so
 * the latency delta is a clean ablation of coalescing alone (same fp32 math, same result).
 *
 * The scalar path still fp32-accumulates and matches the reference's x@w.T elementwise-then-sum, so
 * it passes the same frozen tolerance, it is simply slow because of the access pattern. */
#ifdef AMK_GEMV_SCALAR
__device__ AMK_OP_QUAL void amk_inst_gemv_tile(const amk_device_program& prog,
                                                   const amk_instruction_t& inst,
                                                   int res_id = -1) {
    /* res_id >= 0 (set by amk_inst_fused for AMK_FUSED_GEMV_ADD) adds residual[res_id] to each
     * output element in-register before the single store, eliminating the intermediate's global
     * round-trip + one ADD instruction's dispatch/sync. -1 == the ordinary unfused GEMV. */
    /* weight-only quantized tile? (params.qdtype set to I4/I8 by the quantized lowering) */
    if (inst.params.qdtype == AMK_I4 || inst.params.qdtype == AMK_I8) {
        amk_inst_gemv_tile_quant(prog, inst); return;
    }
    const amk_buffer_t& x   = amk_buf(prog, inst.inputs[0]);   /* [M,K] */
    const amk_buffer_t& W   = amk_buf(prog, inst.inputs[1]);   /* [N,K] */
    const amk_buffer_t& out = amk_buf(prog, inst.outputs[0]);  /* [M,N] */
    const int K      = inst.params.K;
    const int N_tile = inst.params.N_tile;
    const int n_off  = inst.params.n_off;

    const int64_t N_full = out.shape[1];                      /* full output width */
    const int64_t M      = (N_full > 0) ? (out.numel / N_full) : 1;

    /* v1 NAIVE: one thread owns one output column (= one weight row). Thread t loops k=0..K and
     * reads W[n, k]; adjacent threads t,t+1 own rows n,n+1 whose elements at the same k are K apart
     * in memory -> each 128B sector serves a single thread -> ~1% of peak HBM bandwidth. No SMEM
     * x-cache, no vectorization, no warp reduction: the deliberately-uncoalesced baseline. */
    for (int64_t m = 0; m < M; ++m) {
        const int64_t x_row = m * x.stride[0];
        for (int t = threadIdx.x; t < N_tile; t += blockDim.x) {
            const int n = n_off + t;                          /* output column = weight row */
            const int64_t w_row = (int64_t)n * W.stride[0];
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += amk_load_f(x, x_row + (int64_t)k * x.stride[x.rank - 1])
                     * amk_load_f(W, w_row + (int64_t)k * W.stride[W.rank - 1]);
            }
            amk_store_f(out, m * out.stride[0] + (int64_t)n * out.stride[1], acc);
        }
        __syncthreads();   /* match the coalesced path's per-m barrier discipline */
    }
}
#else
__device__ AMK_OP_QUAL void amk_inst_gemv_tile(const amk_device_program& prog,
                                                   const amk_instruction_t& inst,
                                                   int res_id = -1) {
    /* res_id >= 0 (set by amk_inst_fused for AMK_FUSED_GEMV_ADD) adds residual[res_id] to each
     * output element in-register before the single store, eliminating the intermediate's global
     * round-trip + one ADD instruction's dispatch/sync. -1 == the ordinary unfused GEMV. */
    /* weight-only quantized tile? (params.qdtype set to I4/I8 by the quantized lowering) */
    if (inst.params.qdtype == AMK_I4 || inst.params.qdtype == AMK_I8) {
        amk_inst_gemv_tile_quant(prog, inst); return;
    }
    const amk_buffer_t& x   = amk_buf(prog, inst.inputs[0]);   /* [M,K] */
    const amk_buffer_t& W   = amk_buf(prog, inst.inputs[1]);   /* [N,K] */
    const amk_buffer_t& out = amk_buf(prog, inst.outputs[0]);  /* [M,N] */
    const int K      = inst.params.K;
    const int N_tile = inst.params.N_tile;
    const int n_off  = inst.params.n_off;

    const int64_t N_full = out.shape[1];                      /* full output width */
    const int64_t M      = (N_full > 0) ? (out.numel / N_full) : 1;

    const int lane    = threadIdx.x & (warpSize - 1);
    const int warp    = threadIdx.x / warpSize;
    const int nwarps  = (blockDim.x + warpSize - 1) / warpSize;

    /* contiguous-along-K iff both operands have unit stride on their last axis (the lowering's
     * case: x is [1,K] contiguous, W is row-major [N,K]). Then a weight row is a contiguous K-run
     * and we take the vectorized fast path; otherwise a general strided warp-per-row fallback. */
    const bool fast = (x.stride[x.rank - 1] == 1) && (W.stride[W.rank - 1] == 1);

#if defined(AMK_GEMV_CPASYNC) && ((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
    /* ===== cp.async DOUBLE-BUFFERED PATH (the production decode kernel on sm_80+) =============
     * DYNAMIC SMEM layout: [ x: K fp32 | ring: nwarps*STAGES*CPC*CHUNK weight-elems ]. The loader
     * provisions x_bytes + ring_bytes (see vm/loader.py _gemv_cpasync_smem). x is cached fp32 once
     * per block; each warp streams its CPC weight rows K-chunk-by-K-chunk with cp.async into the
     * ring, computing on stage s while STAGES-1 future chunks load -> deep MLP -> hides HBM latency.
     * The ring's WT granule is 16 bytes (float4) regardless of dtype, so x_floats*4 must be 16-byte
     * aligned (K is even here; we round the x region up to 16B). Fast path only; else falls through
     * to the register path below (which uses the same dynamic x region). */
    extern __shared__ float s_cpa[];
    const int    CPC    = AMK_GEMV_CPASYNC_COLS;
    const int    STAGES = AMK_GEMV_CPA_STAGES;
    const int    VPL    = AMK_GEMV_CPA_VPL;
    const int    VEC_BF = 8, VEC_F32 = 4;
    const int    chunk_bf  = warpSize * VPL * VEC_BF;     /* bf16/fp16 chunk elems */
    const int    chunk_f32 = warpSize * VPL * VEC_F32;    /* fp32 chunk elems */
    /* x region rounded up so the ring (float4-aligned) starts 16-byte aligned */
    const int    x_floats  = (K + 3) & ~3;
    float* const xs = s_cpa;
    const bool   cpa_ok = fast && (chunk_bf > 0) && (K >= chunk_bf || K >= chunk_f32);
    if (cpa_ok) {
        for (int j = threadIdx.x; j < K; j += blockDim.x)
            xs[j] = amk_load_f(x, j);                     /* M==1 decode: x row 0 */
        __syncthreads();
        switch (W.dtype) {
            case AMK_F16: {
                __half* ring = (__half*)(s_cpa + x_floats);
                const int rper = STAGES * CPC * chunk_bf;
                amk_gemv_tile_cpasync<__half, 8>(prog, inst, xs, ring, rper, K, N_tile, n_off);
                break; }
            case AMK_BF16: {
                __nv_bfloat16* ring = (__nv_bfloat16*)(s_cpa + x_floats);
                const int rper = STAGES * CPC * chunk_bf;
                amk_gemv_tile_cpasync<__nv_bfloat16, 8>(prog, inst, xs, ring, rper, K, N_tile, n_off);
                break; }
            case AMK_F32:
            default: {
                float* ring = (float*)(s_cpa + x_floats);
                const int rper = STAGES * CPC * chunk_f32;
                amk_gemv_tile_cpasync<float, 4>(prog, inst, xs, ring, rper, K, N_tile, n_off);
                break; }
        }
        __syncthreads();
        return;
    }
    /* cpa_ok==false -> fall through to the register path, reusing s_cpa as the x-cache */
    const bool use_smem_x = fast && (K <= AMK_GEMV_MAX_K);
    (void)use_smem_x;
#else
    /* shared fp32 cache of the current x row (K up to AMK_GEMV_MAX_K; larger -> strided fallback) */
    __shared__ float xs[AMK_GEMV_MAX_K];
    const bool use_smem_x = fast && (K <= AMK_GEMV_MAX_K);
#endif

    for (int64_t m = 0; m < M; ++m) {
        const int64_t x_row = m * x.stride[0];

        if (use_smem_x) {
            for (int j = threadIdx.x; j < K; j += blockDim.x)
                xs[j] = amk_load_f(x, x_row + j);
            __syncthreads();
        }

        /* COLS-PER-WARP (x-reuse): each warp owns a run of C = AMK_GEMV_COLS_PER_WARP consecutive
         * output columns (weight rows). The cached x is reused across all C, so one x-read feeds C
         * in-flight weight streams. C==1 keeps the original single-row dots (bit-identical). */
        const int C = AMK_GEMV_COLS_PER_WARP;
        const int64_t wstride = W.stride[0];
        for (int t = warp * C; t < N_tile; t += nwarps * C) {
            float acc[AMK_GEMV_COLS_PER_WARP];
            const int n0 = n_off + t;
            const int cols = (t + C <= N_tile) ? C : (N_tile - t);   /* tail-safe */
            if (use_smem_x && cols == C) {
                const int64_t w_row = (int64_t)n0 * wstride;
                switch (W.dtype) {
                    case AMK_F16:
                        amk_gemv_rows_dot_f16<AMK_GEMV_COLS_PER_WARP>(
                            xs, (const __half*)W.ptr + w_row, K, wstride, lane, acc);
                        break;
                    case AMK_BF16:
                        amk_gemv_rows_dot_bf16<AMK_GEMV_COLS_PER_WARP>(
                            xs, (const __nv_bfloat16*)W.ptr + w_row, K, wstride, lane, acc);
                        break;
                    case AMK_F32:
                    default:
                        amk_gemv_rows_dot_f32<AMK_GEMV_COLS_PER_WARP>(
                            xs, (const float*)W.ptr + w_row, K, wstride, lane, acc);
                        break;
                }
            } else {
                /* tail (cols<C) or strided fallback: per-column, correctness over speed. */
                #pragma unroll
                for (int c = 0; c < C; ++c) acc[c] = 0.f;
                for (int c = 0; c < cols; ++c) {
                    const int64_t w_row = (int64_t)(n0 + c) * wstride;
                    if (use_smem_x) {
                        switch (W.dtype) {
                            case AMK_F16:
                                acc[c] = amk_gemv_row_dot_f16(xs, (const __half*)W.ptr + w_row, K, lane); break;
                            case AMK_BF16:
                                acc[c] = amk_gemv_row_dot_bf16(xs, (const __nv_bfloat16*)W.ptr + w_row, K, lane); break;
                            case AMK_F32:
                            default:
                                acc[c] = amk_gemv_row_dot_f32(xs, (const float*)W.ptr + w_row, K, lane); break;
                        }
                    } else {
                        for (int k = lane; k < K; k += warpSize)
                            acc[c] += amk_load_f(x, x_row + (int64_t)k * x.stride[x.rank - 1])
                                    * amk_load_f(W, w_row + (int64_t)k * W.stride[W.rank - 1]);
                    }
                }
            }
            /* warp-shuffle reduce each column's accumulator to lane 0 (fp32) */
            #pragma unroll
            for (int c = 0; c < C; ++c) {
                float a = acc[c];
                #pragma unroll
                for (int o = warpSize / 2; o > 0; o >>= 1)
                    a += __shfl_down_sync(0xffffffffu, a, o);
                if (lane == 0 && c < cols) {
                    const int n = n0 + c;
                    float val = a;
                    if (res_id >= 0) {   /* FUSED GEMV_ADD: residual added in-register, no round-trip */
                        const amk_buffer_t& res = amk_buf(prog, res_id);
                        val += amk_load_f(res, m * res.stride[0] + (int64_t)n * res.stride[1]);
                    }
                    amk_store_f(out, m * out.stride[0] + (int64_t)n * out.stride[1], val);
                }
            }
        }
        if (use_smem_x) __syncthreads();   /* reuse xs safely for the next m row */
    }
}
#endif /* AMK_GEMV_SCALAR */

/* ---- GEMM_TILE: out[..., n_off:n_off+N_tile] = x @ W[n_off:n_off+N_tile, :].T  (batched, M>=1) --
 * The THROUGHPUT-mode projection (the keystone of batched decode): x is [M,K], W is [N,K] (torch
 * nn.Linear layout), out is [M,N]. This task writes the output COLUMN slice [n_off, n_off+N_tile)
 * for ALL M rows, exactly instructions/reference.py ref_gemm_tile (_gemv_gemm):
 *     w_tile   = W[n_off:n_off+N_tile, :]      # [N_tile, K]
 *     out_tile = x @ w_tile.T                  # [M, N_tile]
 *     out[..., n_off:n_off+N_tile] = out_tile  # (+ optional bias[n_off:n_off+N_tile])
 * i.e. out[m, n] = sum_k x[m,k] * W[n,k]. It is the decode GEMV (M==1) GENERALIZED to M output
 * rows: we reuse the GEMV's coalesced warp-per-column dot (one warp owns output column n == weight
 * row n; its 32 lanes stride the contiguous K run of W[n,:], each lane fp32-accumulates, a
 * warp-shuffle tree reduces to lane 0) and OUTER-LOOP over the M rows. fp32 accumulate (like the
 * GEMV and the reference's "accumulate in fp32, cast to output dtype" rule) keeps bf16/fp16
 * faithful; the per-lane-then-shuffle reduction order is IDENTICAL to the GEMV, which is
 * correctness-gated against this same reference, so the GEMM clears the same frozen tolerance.
 *
 * M (output rows) is read from the OUTPUT buffer's leading extent (== x.shape[0]), exactly the
 * reference, which computes EVERY row of x for this column tile (it never subsets rows by
 * m_off/M_tile). In a well-formed task this equals params.M_tile.
 *
 * CORRECTNESS-FIRST (throughput milestone 1): a clean, obviously-correct tiled GEMM. The throughput
 * WIN proper - reading each weight element ONCE and reusing it across all M rows (a register
 * row-tile), plus vectorized float4 / bf16x8 loads and an SMEM x-cache - is a LATER perf milestone;
 * this version reads operands straight from (L2-resident) global so it needs NO extra shared memory
 * (a GEMM-only program is not provisioned any) and is trivially correct for any dtype/stride. The
 * weight read IS coalesced (adjacent lanes read adjacent K). ADDITIVE: the decode GEMV path above is
 * untouched and byte-identical; this is a separate opcode reached only by AMK_OP_GEMM_TILE. */
__device__ AMK_OP_QUAL void amk_inst_gemm_tile(const amk_device_program& prog,
                                                   const amk_instruction_t& inst) {
    const amk_buffer_t& x   = amk_buf(prog, inst.inputs[0]);   /* [M,K] activations */
    const amk_buffer_t& W   = amk_buf(prog, inst.inputs[1]);   /* [N,K] torch-Linear weight */
    const amk_buffer_t& out = amk_buf(prog, inst.outputs[0]);  /* [M,N] */
    const int K        = inst.params.K;
    const int N_tile   = inst.params.N_tile;
    const int n_off    = inst.params.n_off;
    const bool has_bias = (inst.n_inputs > 2);

    const int64_t N_full = out.shape[1];                       /* full output width */
    const int64_t M      = (N_full > 0) ? (out.numel / N_full) : 1;

    const int lane   = threadIdx.x & (warpSize - 1);
    const int warp   = threadIdx.x / warpSize;
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;

    const int64_t xks = x.stride[x.rank - 1];                  /* x element stride along K */
    const int64_t wks = W.stride[W.rank - 1];                  /* W element stride along K */
    const int64_t wrs = W.stride[0];                           /* W stride between rows (== K) */

    for (int64_t m = 0; m < M; ++m) {
        const int64_t x_row = m * x.stride[0];
        /* one warp per output column (= weight row); strided so every warp stays busy. */
        for (int t = warp; t < N_tile; t += nwarps) {
            const int n = n_off + t;
            const int64_t w_row = (int64_t)n * wrs;
            float acc = 0.f;
            /* coalesced K reduction: adjacent lanes touch adjacent K elements of x and W[n,:]. */
            for (int k = lane; k < K; k += warpSize) {
                acc += amk_load_f(x, x_row + (int64_t)k * xks)
                     * amk_load_f(W, w_row + (int64_t)k * wks);
            }
            /* warp-shuffle reduce the per-lane K-partials to lane 0 (fp32) */
            #pragma unroll
            for (int o = warpSize / 2; o > 0; o >>= 1)
                acc += __shfl_down_sync(0xffffffffu, acc, o);
            if (lane == 0) {
                if (has_bias)                                  /* bias[n] over the column tile */
                    acc += amk_load_f(amk_buf(prog, inst.inputs[2]), n);
                amk_store_f(out, m * out.stride[0] + (int64_t)n * out.stride[1], acc);
            }
        }
    }
}

/* ---- SILU_MUL (SwiGLU): out = silu(gate) * up = gate*sigmoid(gate) * up ----------------------
 * inputs = [gate, up]; fp32 math then cast. */
__device__ AMK_OP_QUAL void amk_inst_silu_mul(const amk_device_program& prog,
                                                  const amk_instruction_t& inst) {
    const amk_buffer_t& g   = amk_buf(prog, inst.inputs[0]);
    const amk_buffer_t& u   = amk_buf(prog, inst.inputs[1]);
    const amk_buffer_t& out = amk_buf(prog, inst.outputs[0]);
    int64_t n = out.numel;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        float gate = amk_load_f(g, i);
        float silu = gate / (1.0f + __expf(-gate));   /* gate * sigmoid(gate) */
        amk_store_f(out, i, silu * amk_load_f(u, i));
    }
}

/* ---- EMBED: gather rows of a table by ids, then view_as(out) --------------------------------
 * Matches instructions/reference.py `ref_embed` EXACTLY:
 *     ids = inputs[0].to(long).view(-1)          # [S] integer ids
 *     out = table.index_select(0, ids).view_as(out)
 * inputs = [ids (S ints), table (V, row_len)]; output = out (any shape, numel == S*row_len).
 * The gather is row-major-flat: row r of `out` (conceptually) is table[ids[r]] of length row_len,
 * and the whole result is copied flat into `out` (the view_as reshape is a no-op on contiguous
 * storage). schedule/lower.py uses two forms, both covered:
 *   (a) real embed : ids=[token], table=[V,H]      -> out=[1,H]              (one gather row)
 *   (b) reshape    : ids=[0],     table=[1,dim]     -> out=head-shaped[dim]  (flat->head bridge) */
__device__ AMK_OP_QUAL void amk_inst_embed(const amk_device_program& prog,
                                               const amk_instruction_t& inst) {
    const amk_buffer_t& ids   = amk_buf(prog, inst.inputs[0]);
    const amk_buffer_t& table = amk_buf(prog, inst.inputs[1]);
    const amk_buffer_t& out   = amk_buf(prog, inst.outputs[0]);
    /* row_len = elements per table row = table.numel / table.shape[0] (rank>=2 always here). */
    const int64_t V       = (table.rank > 0) ? table.shape[0] : 1;
    const int64_t row_len = (V > 0) ? (table.numel / V) : table.numel;
    const int64_t S       = ids.numel;                 /* number of gather rows */
    for (int64_t r = 0; r < S; ++r) {
        long id = amk_load_i(ids, r);
        if (id < 0) id = 0;                             /* defensive clamp (ref assumes valid) */
        if (id >= V) id = V - 1;
        const int64_t src_base = id * row_len;
        const int64_t dst_base = r * row_len;
        for (int64_t j = threadIdx.x; j < row_len; j += blockDim.x) {
            amk_store_f(out, dst_base + j, amk_load_f(table, src_base + j));
        }
    }
}

/* ---- ROPE: Llama rotate-half rotary embedding (in place over head shape) --------------------
 * Matches instructions/reference.py `ref_rope` EXACTLY:
 *     half     = head_dim/2
 *     inv_freq[i] = theta^(-(i/half))                # i in [0,half)
 *     ang      = pos * inv_freq[i]
 *     c,s      = cos(ang), sin(ang)
 *     out[..,:half] = x1*c - x2*s ;  out[..,half:] = x2*c + x1*s
 * inputs = [x (.., head_dim), pos (S ints)]; output = same shape as x. cos/sin broadcast over
 * heads at one position (decode S==1). We treat x as a flat array of `n_vec = numel/head_dim`
 * vectors of length head_dim; every vector at the SAME sequence position uses the same angle.
 * lower.py only ever ropes a single position (q: [n_heads,head_dim]; k: [1,n_kv,head_dim]), so
 * we use pos[0] for all vectors, exactly the reference's cos[0]/sin[0] broadcast. */
__device__ AMK_OP_QUAL void amk_inst_rope(const amk_device_program& prog,
                                              const amk_instruction_t& inst) {
    const amk_buffer_t& x   = amk_buf(prog, inst.inputs[0]);
    const amk_buffer_t& pb  = amk_buf(prog, inst.inputs[1]);
    const amk_buffer_t& out = amk_buf(prog, inst.outputs[0]);
    const int   head_dim = inst.params.head_dim;
    const float theta    = inst.params.theta;
    const int   half     = head_dim / 2;
    const long  pos      = amk_load_i(pb, 0);           /* single decode position (S==1) */
    const int64_t n_vec  = (head_dim > 0) ? (x.numel / head_dim) : 0;

    for (int64_t vec = 0; vec < n_vec; ++vec) {
        const int64_t base = vec * head_dim;
        for (int i = threadIdx.x; i < half; i += blockDim.x) {
            float inv = __powf(theta, -((float)i / (float)half));
            float ang = (float)pos * inv;
            float c = __cosf(ang), s = __sinf(ang);
            float x1 = amk_load_f(x, base + i);
            float x2 = amk_load_f(x, base + i + half);
            amk_store_f(out, base + i,        x1 * c - x2 * s);
            amk_store_f(out, base + i + half, x2 * c + x1 * s);
        }
    }
}

/* ---- KV_APPEND: write this step's k or v into the cache at position `pos` ---------------------
 * Matches instructions/reference.py `ref_kv_append` EXACTLY:
 *     cache[pos:pos+new.shape[0]] = new.view(-1, *cache.shape[1:])
 * inputs = [new_kv, cache]; output = cache (in place). For decode new has leading dim 1, so we
 * copy `new.numel` contiguous elements (== n_kv*head_dim) to cache flat offset pos*row_stride,
 * where row_stride = cache.numel/cache.shape[0] (== n_kv*head_dim). The k path passes the roped
 * [1,n_kv,head_dim]; the v path passes flat [1,kv_dim], both flatten to the same row. */
__device__ AMK_OP_QUAL void amk_inst_kv_append(const amk_device_program& prog,
                                                   const amk_instruction_t& inst) {
    const amk_buffer_t& nw    = amk_buf(prog, inst.inputs[0]);
    const amk_buffer_t& cache = amk_buf(prog, inst.outputs[0]);
    const int     pos        = inst.params.pos;
    /* BATCHED DECODE: cache is [B, max_seq, n_kv, head_dim] (one history per sequence) and the new
     * k/v is [B, n_kv, head_dim] (roped k) or [B, kv_dim] (flat v), both nw.numel == B*n_kv*head_dim.
     * Write every sequence's position-`pos` slot. Matches ref_kv_append's rank-4 branch. Reached
     * only for a batch>1 program (rank-4 cache); the rank-3 single-sequence path below is
     * byte-identical (same flat copy as before). */
    if (cache.rank == 4) {
        const int64_t Bsz      = cache.shape[0];
        const int64_t kv_elems = cache.shape[2] * cache.shape[3];        /* n_kv*head_dim (1 pos)  */
        const int64_t seq_str  = cache.shape[1] * kv_elems;              /* per-sequence stride    */
        for (int64_t b = 0; b < Bsz; ++b) {
            const int64_t dst = b * seq_str + (int64_t)pos * kv_elems;   /* cache[b, pos, :, :]     */
            const int64_t src = b * kv_elems;                            /* nw[b] (contiguous)      */
            for (int64_t j = threadIdx.x; j < kv_elems; j += blockDim.x)
                amk_store_f(cache, dst + j, amk_load_f(nw, src + j));
        }
        return;
    }
    const int64_t max_seq    = (cache.rank > 0) ? cache.shape[0] : 1;
    const int64_t row_stride = (max_seq > 0) ? (cache.numel / max_seq) : cache.numel;
    const int64_t n          = nw.numel;               /* == new.shape[0]*row_stride for decode */
    const int64_t dst_base   = (int64_t)pos * row_stride;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        amk_store_f(cache, dst_base + j, amk_load_f(nw, j));
    }
}

/* ---- ATTENTION_TILE: GQA whole-window decode attention (numerically-stable flash softmax) ----
 * Matches instructions/reference.py `ref_attention_tile` EXACTLY:
 *     rep         = n_heads / n_kv_heads        (head h reads kv head h/rep == repeat_interleave)
 *     scores[h,j] = (sum_d q[h,d]*k[kv_start+j, h/rep, d]) * scale
 *     probs       = softmax_j(scores[h,:])      (decode 1-query attends all cached -> no masking)
 *     out[h,d]    = sum_j probs[h,j]*v[kv_start+j, h/rep, d]
 * inputs = [q (n_heads,head_dim), k_cache (kv_total,n_kv,head_dim), v_cache (same)];
 * output = (1, n_heads*head_dim) via the reference's view_as. fp32 math throughout. Each query
 * head is processed by the whole block via an online (flash) softmax over the KV window; we use
 * shared memory for the q-vector cache, the running accumulator, and the cross-block reductions.
 * Heads are looped (one block owns the whole instruction, like every other op). */
#define AMK_ATTN_MAX_HEAD_DIM 512

/* ---- ATTENTION_TILE (BATCHED decode): B independent decode-attentions in one task --------------
 * q is [B, n_heads, head_dim], the caches are [B, max_seq, n_kv, head_dim] (one KV history per
 * sequence), out is [B, n_heads*head_dim]. Each of the B queries is a single token attending to its
 * OWN cached window [kv_start, kv_start+kv_len). Mirrors instructions/reference.py ref_attention_tile
 * rank-3 branch EXACTLY (same online flash softmax, fp32). It is the single-sequence warp-parallel
 * kernel wrapped in a batch loop: each WARP still owns a disjoint set of heads, and we REUSE the same
 * per-warp q_s|acc SMEM slice across the B sequences (so the loader provisions no extra SMEM). One
 * block barrier at op end. Split-KV PARTIAL_WRITE is single-sequence only; the lowerer never batches
 * it (P==1 forced for B>1), so this path always writes the normalized output. */
__device__ AMK_OP_QUAL void amk_inst_attention_tile_batched(const amk_device_program& prog,
                                                                const amk_instruction_t& inst) {
    const amk_buffer_t& q   = amk_buf(prog, inst.inputs[0]);   /* [B, n_heads, head_dim]          */
    const amk_buffer_t& k   = amk_buf(prog, inst.inputs[1]);   /* [B, max_seq, n_kv, head_dim]    */
    const amk_buffer_t& v   = amk_buf(prog, inst.inputs[2]);
    const amk_buffer_t& out = amk_buf(prog, inst.outputs[0]);  /* [B, n_heads*head_dim]           */
    const int   head_dim = inst.params.head_dim;
    const int   kv_start = inst.params.kv_start;
    const int   kv_len   = inst.params.kv_len;
    const int   n_heads  = inst.params.n_heads;
    const int   n_kv     = inst.params.n_kv_heads;
    const float scale    = inst.params.scale;
    const int   rep      = (n_kv > 0) ? (n_heads / n_kv) : 1;
    const int64_t kv_row_stride = (int64_t)n_kv * head_dim;    /* one kv position's elements      */
    if (head_dim > AMK_ATTN_MAX_HEAD_DIM) { amk_trap_unimplemented(prog, AMK_OP_ATTENTION_TILE); return; }
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp = threadIdx.x / warpSize;
    extern __shared__ float s_dyn[];
    float* const q_s = s_dyn + (int64_t)warp * 2 * head_dim;   /* this warp's private q_s|acc slice */
    float* const acc = q_s + head_dim;
    const int64_t Bsz        = (q.rank > 0) ? q.shape[0] : 1;
    const int64_t qd         = (int64_t)n_heads * head_dim;    /* q/out row width                 */
    const int64_t kv_seq_str = (k.rank >= 2 ? k.shape[1] : 1) * kv_row_stride;   /* per-seq cache  */
    for (int64_t b = 0; b < Bsz; ++b) {
        const int64_t q_seq  = b * qd;                         /* q[b] base (== out[b] base)       */
        const int64_t kv_seq = b * kv_seq_str;                 /* this sequence's cache base       */
        for (int h = warp; h < n_heads; h += nwarps) {
            const int   kvh    = h / rep;
            const int64_t q_base   = q_seq + (int64_t)h * head_dim;
            const int64_t kv_h_off = kv_seq + (int64_t)kv_start * kv_row_stride + (int64_t)kvh * head_dim;
            for (int d = lane; d < head_dim; d += warpSize) { q_s[d] = amk_load_f(q, q_base + d); acc[d] = 0.f; }
            float m = -CUDART_INF_F, l = 0.f;
            for (int j = 0; j < kv_len; ++j) {
                const int64_t krow = kv_h_off + (int64_t)j * kv_row_stride;
                float dot = 0.f;
                for (int d = lane; d < head_dim; d += warpSize) dot += q_s[d] * amk_load_f(k, krow + d);
                for (int o = warpSize / 2; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
                const float sj = dot * scale;
                const float m_new = fmaxf(m, sj);
                const float corr  = __expf(m - m_new);
                const float p     = __expf(sj - m_new);
                for (int d = lane; d < head_dim; d += warpSize)
                    acc[d] = acc[d] * corr + p * amk_load_f(v, krow + d);   /* v shares k's layout  */
                l = l * corr + p;
                m = m_new;
            }
            const float inv_l = 1.0f / l;                      /* out[b, h, d] flat == q_base + d  */
            for (int d = lane; d < head_dim; d += warpSize) amk_store_f(out, q_base + d, acc[d] * inv_l);
        }
    }
    __syncthreads();   /* one barrier at op end: all warps finish `out` before the scheduler signals */
}

__device__ AMK_OP_QUAL void amk_inst_attention_tile(const amk_device_program& prog,
                                                        const amk_instruction_t& inst) {
    const amk_buffer_t& q   = amk_buf(prog, inst.inputs[0]);
    /* BATCHED decode dispatch: a rank-3 q (== [B, n_heads, head_dim]) means B independent
     * decode-attentions over rank-4 caches; hand off to the batched kernel. The rank-2 single-token
     * path below is byte-identical. */
    if (q.rank == 3) { amk_inst_attention_tile_batched(prog, inst); return; }
    const amk_buffer_t& k   = amk_buf(prog, inst.inputs[1]);
    const amk_buffer_t& v   = amk_buf(prog, inst.inputs[2]);
    const amk_buffer_t& out = amk_buf(prog, inst.outputs[0]);
    const int   head_dim = inst.params.head_dim;
    const int   kv_start = inst.params.kv_start;
    const int   kv_len   = inst.params.kv_len;
    const int   n_heads  = inst.params.n_heads;
    const int   n_kv     = inst.params.n_kv_heads;
    const float scale    = inst.params.scale;
    const int   rep      = (n_kv > 0) ? (n_heads / n_kv) : 1;
    const int64_t kv_row_stride = (int64_t)n_kv * head_dim;   /* one kv position's elements */

    /* PARTIAL_WRITE (flags bit1): this is a split-KV shard -> write the un-normalized flash partial
     * {acc | m | l} per head into the [n_heads, head_dim+2] partial buffer instead of the normalized
     * output; ATTENTION_COMBINE then merges the P shards. (flags bit0 = causal is unused at decode:
     * a single query attends all cached keys.) */
    const bool partial = (inst.params.flags & 0x2) != 0;
    if (head_dim > AMK_ATTN_MAX_HEAD_DIM) {
        amk_trap_unimplemented(prog, AMK_OP_ATTENTION_TILE);
        return;
    }
    /* WARP-PARALLEL decode attention with per-warp SMEM q/acc. Each WARP owns a disjoint set of
     * query heads (h = warp, warp+nwarps, ...) so heads run concurrently; within a warp each LANE
     * owns the slots d = lane, lane+32, ... of its warp's PRIVATE q_s|acc slice of dynamic smem
     * (the loader provisions nwarps*2*head_dim floats, see vm/loader.py). Each lane touches only
     * its own slots, so there is NO cross-lane smem race and NO per-token __syncthreads; the q-dot-k
     * is a warp butterfly all-reduce. SMEM (not registers) holds q/acc so this does NOT compete with
     * the GEMV's register budget at threads_per_block=512. Math identical to ref_attention_tile
     * (online flash softmax); ONE block barrier at op end so the scheduler signals only after every
     * warp has written `out`. This removes the single-SM serial-over-KV barrier storm that collapsed
     * pos>0 decode. */
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp = threadIdx.x / warpSize;
    extern __shared__ float s_dyn[];
    float* const q_s = s_dyn + (int64_t)warp * 2 * head_dim;   /* this warp's private q_s|acc slice */
    float* const acc = q_s + head_dim;
    for (int h = warp; h < n_heads; h += nwarps) {
        const int   kvh    = h / rep;
        const int64_t q_base = (int64_t)h * head_dim;
        const int64_t kv_h_off = (int64_t)kv_start * kv_row_stride + (int64_t)kvh * head_dim;
        for (int d = lane; d < head_dim; d += warpSize) { q_s[d] = amk_load_f(q, q_base + d); acc[d] = 0.f; }
        float m = -CUDART_INF_F, l = 0.f;
        for (int j = 0; j < kv_len; ++j) {
            const int64_t krow = kv_h_off + (int64_t)j * kv_row_stride;
            float dot = 0.f;
            for (int d = lane; d < head_dim; d += warpSize) dot += q_s[d] * amk_load_f(k, krow + d);
            for (int o = warpSize / 2; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            const float sj = dot * scale;
            const float m_new = fmaxf(m, sj);
            const float corr  = __expf(m - m_new);
            const float p     = __expf(sj - m_new);
            for (int d = lane; d < head_dim; d += warpSize)    /* v shares k's [kv,n_kv,head_dim] layout */
                acc[d] = acc[d] * corr + p * amk_load_f(v, krow + d);
            l = l * corr + p;
            m = m_new;
        }
        if (partial) {                                     /* split-KV shard: emit {acc | m | l} */
            const int64_t pbase = (int64_t)h * (head_dim + 2);
            for (int d = lane; d < head_dim; d += warpSize) amk_store_f(out, pbase + d, acc[d]);
            if (lane == 0) { amk_store_f(out, pbase + head_dim, m); amk_store_f(out, pbase + head_dim + 1, l); }
        } else {
            const float inv_l = 1.0f / l;
            for (int d = lane; d < head_dim; d += warpSize) amk_store_f(out, q_base + d, acc[d] * inv_l);
        }
    }
    __syncthreads();   /* one barrier at op end: all warps finish `out` before the scheduler signals */
}

/* ---- ATTENTION_COMBINE: flash-decoding merge of P split-KV partials -------------------------
 * Matches instructions/reference.py `ref_attention_combine`. inputs[0..P-1] are per-shard partials
 * [n_heads, head_dim+2] = {acc | m | l}; output[0] is the normalized attention [.., n_heads*head_dim].
 * P = inst.n_inputs. Each warp owns disjoint heads; online-softmax reduce across shards, fp32. */
__device__ AMK_OP_QUAL void amk_inst_attention_combine(const amk_device_program& prog,
                                                           const amk_instruction_t& inst) {
    const amk_buffer_t& out = amk_buf(prog, inst.outputs[0]);
    const int head_dim = inst.params.head_dim;
    const int n_heads  = inst.params.n_heads;
    const int P        = inst.n_inputs;                /* number of shard partials (<= AMK_MAX_INPUTS) */
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp = threadIdx.x / warpSize;
    const int stride = head_dim + 2;                   /* per-head partial row width */
    for (int h = warp; h < n_heads; h += nwarps) {
        float w[AMK_MAX_INPUTS];                        /* per-shard online-softmax weight (P<=8) */
        float m_g = -CUDART_INF_F;
        for (int p = 0; p < P; ++p)
            m_g = fmaxf(m_g, amk_load_f(amk_buf(prog, inst.inputs[p]), (int64_t)h * stride + head_dim));
        float l_g = 0.f;
        for (int p = 0; p < P; ++p) {
            const amk_buffer_t& pb = amk_buf(prog, inst.inputs[p]);
            const float m_p = amk_load_f(pb, (int64_t)h * stride + head_dim);
            const float l_p = amk_load_f(pb, (int64_t)h * stride + head_dim + 1);
            w[p] = (l_p > 0.f) ? __expf(m_p - m_g) : 0.f;   /* empty shard -> zero weight (no exp(-inf)) */
            l_g += l_p * w[p];
        }
        const float inv_l = 1.0f / l_g;
        for (int d = lane; d < head_dim; d += warpSize) {
            float a = 0.f;
            for (int p = 0; p < P; ++p)
                a += amk_load_f(amk_buf(prog, inst.inputs[p]), (int64_t)h * stride + d) * w[p];
            amk_store_f(out, (int64_t)h * head_dim + d, a * inv_l);
        }
    }
    __syncthreads();
}

/* ---- SAMPLE_ARGMAX: greedy token selection logits[..,V] -> argmax id ------------------------
 * Matches instructions/reference.py `ref_sample_argmax`:
 *     out = argmax(logits, dim=-1)
 * inputs = [logits (.., V)]; output = (..,) integer ids. lower.py does NOT emit this for the
 * decode acceptance program (the host samples from the returned logits), but we implement it so
 * the VM covers every opcode the lowering could emit. Single-block argmax over the last dim; for
 * the decode case there is one row. fp32 compare, ties resolve to the lowest index (torch). */
__device__ AMK_OP_QUAL void amk_inst_sample_argmax(const amk_device_program& prog,
                                                       const amk_instruction_t& inst) {
    const amk_buffer_t& logits = amk_buf(prog, inst.inputs[0]);
    const amk_buffer_t& out    = amk_buf(prog, inst.outputs[0]);
    const int64_t V    = (logits.rank > 0) ? logits.shape[logits.rank - 1] : logits.numel;
    const int64_t rows = (V > 0) ? (logits.numel / V) : 1;

    __shared__ float s_val[1024 / 32];
    __shared__ int   s_idx[1024 / 32];
    const int lane   = threadIdx.x & (warpSize - 1);
    const int warp   = threadIdx.x / warpSize;
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;

    for (int64_t r = 0; r < rows; ++r) {
        const int64_t base = r * V;
        float best_v = -CUDART_INF_F;
        int   best_i = 0;
        for (int64_t j = threadIdx.x; j < V; j += blockDim.x) {
            float val = amk_load_f(logits, base + j);
            if (val > best_v) { best_v = val; best_i = (int)j; }
        }
        /* warp reduce (lowest index wins ties to match torch argmax) */
        for (int o = warpSize / 2; o > 0; o >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, best_v, o);
            int   oi = __shfl_down_sync(0xffffffffu, best_i, o);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0) { s_val[warp] = best_v; s_idx[warp] = best_i; }
        __syncthreads();
        if (warp == 0) {
            float v = (lane < nwarps) ? s_val[lane] : -CUDART_INF_F;
            int   i = (lane < nwarps) ? s_idx[lane] : 0;
            for (int o = warpSize / 2; o > 0; o >>= 1) {
                float ov = __shfl_down_sync(0xffffffffu, v, o);
                int   oi = __shfl_down_sync(0xffffffffu, i, o);
                if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
            }
            if (lane == 0) { s_val[0] = v; s_idx[0] = i; }
        }
        __syncthreads();
        if (threadIdx.x == 0) amk_store_f(out, r, (float)s_idx[0]);
        __syncthreads();
    }
}

/* ===========================================================================================
 * amk_dispatch, the opcode switch. Pure compute; the VM (scheduler.cu) wraps each call with
 * amk_wait_all (before) and amk_signal (after).
 *
 * ROBUSTNESS (abi.h watchdog contract): an unimplemented / unknown opcode is NOT a silent no-op.
 * Thread 0 records the offending opcode into abort_flag (encoded != 0) and raises a device-scope
 * fence so every spinning block sees the abort and exits cleanly via amk_wait_all's abort path,
 * rather than the program hanging forever on a counter the trapped instruction never signals. The
 * encoding (-(op+1)) is negative & nonzero so it never collides with a host-set positive flag and
 * the host can decode which opcode trapped. =================================================== */
__device__ __forceinline__ void amk_trap_unimplemented(const amk_device_program& prog, int op) {
    if (threadIdx.x == 0) {
        /* record which opcode trapped (negative, nonzero) and publish it grid-wide */
        atomicExch(prog.abort_flag, -(op + 1));
        __threadfence();
    }
    __syncthreads();
}

__device__ AMK_OP_QUAL void amk_inst_fused(const amk_device_program& prog,
                                           const amk_instruction_t& inst) {
    /* Run a recognized fused PATTERN on device (the variable-size recipe stays host-side as the CPU
     * oracle/synthesizer; only the pattern id in params.flags bits 16-31 + the constituent op's
     * scalar params + buffer ids cross the ABI). Only the register-path build (no cp.async, no
     * scalar ablation) carries the GEMV_ADD residual epilogue; any other build or an unknown
     * pattern TRAPS loudly rather than silently miscomputing. */
    const int pat = (inst.params.flags >> 16) & 0xFFFF;
#if !defined(AMK_GEMV_CPASYNC) && !defined(AMK_GEMV_SCALAR)
    if (pat == AMK_FUSED_GEMV_ADD) { amk_inst_gemv_tile(prog, inst, inst.inputs[2]); return; }
#endif
    amk_trap_unimplemented(prog, inst.op);
}

__device__ __forceinline__ void amk_dispatch(const amk_device_program& prog,
                                             const amk_instruction_t& inst) {
    switch (inst.op) {
        case AMK_OP_NOP:            break;
        case AMK_OP_COPY:           amk_inst_copy(prog, inst);           break;
        case AMK_OP_EMBED:          amk_inst_embed(prog, inst);          break;
        case AMK_OP_RMSNORM:        amk_inst_rmsnorm(prog, inst);        break;
        case AMK_OP_GEMV_TILE:      amk_inst_gemv_tile(prog, inst);      break;
        case AMK_OP_GEMM_TILE:      amk_inst_gemm_tile(prog, inst);      break;
        case AMK_OP_ROPE:           amk_inst_rope(prog, inst);           break;
        case AMK_OP_ATTENTION_TILE: amk_inst_attention_tile(prog, inst); break;
        case AMK_OP_ATTENTION_COMBINE: amk_inst_attention_combine(prog, inst); break;
        case AMK_OP_KV_APPEND:      amk_inst_kv_append(prog, inst);      break;
        case AMK_OP_ADD:            amk_inst_add(prog, inst);            break;
        case AMK_OP_SILU_MUL:       amk_inst_silu_mul(prog, inst);       break;
        case AMK_OP_SAMPLE_ARGMAX:  amk_inst_sample_argmax(prog, inst);  break;
        case AMK_OP_FUSED:          amk_inst_fused(prog, inst);          break;
        default:                    /* unknown/unimplemented opcode -> TRAP (abort, don't hang) */
                                    amk_trap_unimplemented(prog, inst.op); break;
    }
}

#endif /* AMK_OPS_CUH */
