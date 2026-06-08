/* ===========================================================================================
 * AMK Layer-1 micro-kernel, ATTENTION_TILE  (opcode AMK_OP_ATTENTION_TILE = 7)
 * ===========================================================================================
 * Whole-window decode attention with GQA + (decode) causal, matching instructions/reference.py
 * `ref_attention_tile` EXACTLY:
 *
 *     q : [n_heads, head_dim]                       (current step query, one token)
 *     k : [kv_total, n_kv_heads, head_dim]          (KV cache; we read [kv_start, kv_start+kv_len))
 *     v : [kv_total, n_kv_heads, head_dim]
 *     rep   = n_heads // n_kv_heads                 (grouped-query attention)
 *     head h reads kv head (h // rep)               (== repeat_interleave(rep, dim=1))
 *     scores[h,j] = (sum_d q[h,d] * k[kv_start+j, h//rep, d]) * scale
 *     probs       = softmax_j(scores[h, :])         (causal: decode 1-query attends all cached)
 *     out[h,d]    = sum_j probs[h,j] * v[kv_start+j, h//rep, d]
 *
 * Compute in fp32 (matches reference). One threadblock per query head; a numerically-stable
 * online (flash) softmax streams the KV window so we never materialise the full score row. The
 * causal flag (params.flags bit0) is honoured but, exactly as the reference notes, a single decode
 * query attends to every cached key (all <= current pos) so no extra masking is applied here.
 *
 * ABI plug-in: `amk_attention_head_core` computes one head's output given resolved k/v base
 * pointers + window; the VM's amk_inst_attention_tile maps heads to blocks identically.
 * =========================================================================================== */
// ATen + pybind directly (NOT <torch/extension.h>; see add.cu note on MSVC C2872).
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>   // CUDART_INF_F

namespace {

template <typename T> __device__ __forceinline__ float to_f32(T v) { return static_cast<float>(v); }
template <> __device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
template <typename T> __device__ __forceinline__ T from_f32(float v) { return static_cast<T>(v); }
template <> __device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float block_reduce_max(float v, float* s) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    if (lane == 0) s[warp] = v;
    __syncthreads();
    v = (threadIdx.x < warps) ? s[lane] : -CUDART_INF_F;
    if (warp == 0)
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) s[0] = v;
    __syncthreads();
    return s[0];
}

__device__ __forceinline__ float block_reduce_sum(float v, float* s) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) s[warp] = v;
    __syncthreads();
    v = (threadIdx.x < warps) ? s[lane] : 0.f;
    if (warp == 0)
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) s[0] = v;
    __syncthreads();
    return s[0];
}

/* ABI device core: one query head. q_head[head_dim], k/v point at the WINDOW start row for the
 * mapped kv head; row stride between consecutive kv positions is `kv_row_stride` elements. */
template <typename QT, typename KT, typename VT, typename OT>
__device__ void amk_attention_head_core(const QT* __restrict__ q_head, const KT* __restrict__ k0,
                                        const VT* __restrict__ v0, OT* __restrict__ out_head,
                                        int head_dim, int kv_len, long kv_row_stride, float scale,
                                        float* smem) {
    // smem layout: [head_dim] q cache | [head_dim] accumulator | [reduce scratch]
    float* q_s = smem;
    float* acc = smem + head_dim;
    float* red = smem + 2 * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) { q_s[d] = to_f32(q_head[d]); acc[d] = 0.f; }
    __syncthreads();

    float m = -CUDART_INF_F;   // running max
    float l = 0.f;             // running denom
    for (int j = 0; j < kv_len; ++j) {
        const KT* krow = k0 + (long)j * kv_row_stride;
        // dot(q, k_j) reduced across the block
        float partial = 0.f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) partial += q_s[d] * to_f32(krow[d]);
        // block_reduce_sum broadcasts the reduced value to every thread (returns s[0]).
        float s_j = block_reduce_sum(partial, red) * scale;
        // online (flash) softmax update, every thread holds the same m/l so this stays consistent
        float m_new = fmaxf(m, s_j);
        float corr = __expf(m - m_new);     // rescale factor for existing acc/l
        float p = __expf(s_j - m_new);
        const VT* vrow = v0 + (long)j * kv_row_stride;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            acc[d] = acc[d] * corr + p * to_f32(vrow[d]);
        l = l * corr + p;
        m = m_new;
        __syncthreads();                    // red scratch reuse barrier before next j
    }
    float inv_l = 1.0f / l;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_head[d] = from_f32<OT>(acc[d] * inv_l);
}

template <typename T>
__global__ void attention_kernel(const T* __restrict__ q, const T* __restrict__ k,
                                 const T* __restrict__ v, T* __restrict__ out,
                                 int head_dim, int kv_len, int kv_start, int n_heads, int n_kv,
                                 int rep, float scale) {
    extern __shared__ float smem[];
    const int h = blockIdx.x;                       // query head
    const int kvh = h / rep;                        // mapped kv head (repeat_interleave)
    const long kv_row_stride = (long)n_kv * head_dim;
    const T* k0 = k + (long)kv_start * kv_row_stride + (long)kvh * head_dim;
    const T* v0 = v + (long)kv_start * kv_row_stride + (long)kvh * head_dim;
    amk_attention_head_core<T, T, T, T>(q + (long)h * head_dim, k0, v0, out + (long)h * head_dim,
                                        head_dim, kv_len, kv_row_stride, scale, smem);
}

}  // namespace

// q:[n_heads,head_dim]  k,v:[kv_total,n_kv,head_dim]  out:[n_heads,head_dim] (pre-allocated).
void attention_tile(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out,
                    int64_t head_dim, int64_t kv_start, int64_t kv_len,
                    int64_t n_heads, int64_t n_kv_heads, double scale, int64_t flags) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda(),
                "attention_tile: tensors must be CUDA");
    TORCH_CHECK(n_heads % n_kv_heads == 0, "attention_tile: n_heads must be divisible by n_kv_heads");
    const int rep = (int)(n_heads / n_kv_heads);
    auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
    const int threads = head_dim < 32 ? 32 : (head_dim > 256 ? 256 : (int)head_dim);
    const int warps = (threads + 31) / 32;
    // smem: q cache + acc + max(reduce scratch, warps)
    size_t smem_bytes = (2 * head_dim + std::max((int)head_dim, warps)) * sizeof(float);
    (void)flags;  // decode single-query causal == attend-all; no masking (matches reference)
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "attention_tile", [&] {
        attention_kernel<scalar_t><<<(int)n_heads, threads, smem_bytes>>>(
            qc.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(), vc.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), (int)head_dim, (int)kv_len, (int)kv_start,
            (int)n_heads, (int)n_kv_heads, rep, (float)scale);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_tile", &attention_tile,
          "AMK ATTENTION_TILE: GQA whole-window decode attention (flash online softmax, fp32)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("out"), py::arg("head_dim"),
          py::arg("kv_start"), py::arg("kv_len"), py::arg("n_heads"), py::arg("n_kv_heads"),
          py::arg("scale"), py::arg("flags"));
}
