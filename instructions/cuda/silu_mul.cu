/* ===========================================================================================
 * AMK Layer-1 micro-kernel, SILU_MUL  (opcode AMK_OP_SILU_MUL = 9)  [SwiGLU]
 * ===========================================================================================
 * Matches instructions/reference.py `ref_silu_mul` EXACTLY:
 *
 *     out = silu(gate) * up           # silu(g) = g * sigmoid(g);  all math fp32 then cast
 *
 * inputs = [gate, up] (same shape); fully elementwise. fp32 accumulate/compute per the reference
 * (`_f32`). ABI plug-in: `amk_silu_mul_core` is the per-element __device__ math; the kernel is a
 * flat grid-stride loop. This is the activation between the gate/up GEMVs and the down GEMV.
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

__device__ __forceinline__ float amk_silu_mul_core(float g, float u) {
    return (g / (1.0f + __expf(-g))) * u;   // silu(g) * u
}

template <typename T>
__global__ void silu_mul_kernel(const T* __restrict__ gate, const T* __restrict__ up,
                                T* __restrict__ out, long n) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x)
        out[i] = from_f32<T>(amk_silu_mul_core(to_f32(gate[i]), to_f32(up[i])));
}

}  // namespace

void silu_mul(at::Tensor gate, at::Tensor up, at::Tensor out) {
    TORCH_CHECK(gate.is_cuda() && up.is_cuda() && out.is_cuda(), "silu_mul: tensors must be CUDA");
    TORCH_CHECK(gate.numel() == up.numel() && gate.numel() == out.numel(),
                "silu_mul: gate/up/out must have equal numel");
    auto g = gate.contiguous(), u = up.contiguous();
    const long n = g.numel();
    const int threads = AMK_THREADS;
    const int blocks = (int)std::min<long>((n + threads - 1) / threads, 65535);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, gate.scalar_type(), "silu_mul", [&] {
        silu_mul_kernel<scalar_t><<<blocks, threads>>>(
            g.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_mul", &silu_mul, "AMK SILU_MUL (SwiGLU): out = silu(gate) * up (fp32 compute)",
          py::arg("gate"), py::arg("up"), py::arg("out"));
}
