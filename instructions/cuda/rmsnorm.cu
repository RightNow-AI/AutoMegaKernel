/* ===========================================================================================
 * AMK Layer-1 micro-kernel, RMSNORM  (opcode AMK_OP_RMSNORM = 3)
 * ===========================================================================================
 * Matches instructions/reference.py `ref_rmsnorm` EXACTLY:
 *
 *     rms = rsqrt(mean(x^2, dim=-1) + eps)
 *     out = (x * rms * w)             # all math in fp32, then cast to out dtype
 *
 * inputs = [x (shape [.., H]), weight (shape [H])]. The reduction is over the last axis (H).
 * One threadblock per row: block-wide fp32 reduction of sum(x^2), then a fused scale-and-weight
 * pass. fp32 accumulate exactly as the reference (`_f32`). H is the model hidden / params.hidden.
 *
 * ABI plug-in shape: `amk_rmsnorm_core` is the __device__ routine the VM calls per row from
 * amk_inst_rmsnorm; the torch wrapper just maps each row to a block.
 * =========================================================================================== */
// ATen + pybind directly (NOT <torch/extension.h>; see add.cu note on MSVC C2872).
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef AMK_THREADS
#define AMK_THREADS 256
#endif

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

__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    v = (threadIdx.x < warps) ? smem[lane] : 0.f;
    if (warp == 0) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    }
    return v;  // valid in thread 0
}

/* ABI device core: normalise one row x[0..H) by RMS, scale by weight w[0..H), write out. */
template <typename XT, typename WT, typename OT>
__device__ void amk_rmsnorm_core(const XT* __restrict__ x, const WT* __restrict__ w,
                                 OT* __restrict__ out, int H, float eps, float* smem) {
    float local = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = to_f32(x[i]);
        local += v * v;
    }
    float ssum = block_reduce_sum(local, smem);
    __shared__ float rms;
    if (threadIdx.x == 0) rms = rsqrtf(ssum / (float)H + eps);
    __syncthreads();
    const float r = rms;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        out[i] = from_f32<OT>(to_f32(x[i]) * r * to_f32(w[i]));
}

template <typename XT, typename WT, typename OT>
__global__ void rmsnorm_kernel(const XT* x, const WT* w, OT* out, int H, float eps, int stride_row) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    amk_rmsnorm_core<XT, WT, OT>(x + (long)row * stride_row, w, out + (long)row * stride_row, H,
                                 eps, smem);
}

}  // namespace

void rmsnorm(at::Tensor x, at::Tensor w, at::Tensor out, double eps) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "rmsnorm: tensors must be CUDA");
    const int H = x.size(-1);
    TORCH_CHECK(w.numel() == H, "rmsnorm: weight length must equal H");
    const int rows = x.numel() / H;
    auto xc = x.contiguous(), wc = w.contiguous();
    const int threads = AMK_THREADS;
    const int warps = (threads + 31) / 32;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rmsnorm", [&] {
        rmsnorm_kernel<scalar_t, scalar_t, scalar_t><<<rows, threads, warps * sizeof(float)>>>(
            xc.data_ptr<scalar_t>(), wc.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
            H, (float)eps, H);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm, "AMK RMSNORM: out = x * rsqrt(mean(x^2)+eps) * w (fp32 accum)",
          py::arg("x"), py::arg("w"), py::arg("out"), py::arg("eps"));
}
