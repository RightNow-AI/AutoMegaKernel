/* ===========================================================================================
 * AMK Layer-1 micro-kernel, ADD  (opcode AMK_OP_ADD = 11)   [residual add]
 * ===========================================================================================
 * Matches instructions/reference.py `ref_add` EXACTLY:  out = (a + b) in fp32, then cast.
 * inputs = [a, b] (same shape), elementwise. This is the residual-stream add. ABI plug-in:
 * `amk_add_core` is the per-element math; the kernel is a flat grid-stride loop.
 * =========================================================================================== */
// NOTE: include ATen + pybind directly (NOT <torch/extension.h>), on Windows the latter pulls
// in torch/csrc/dynamo/compiled_autograd.h which fails under MSVC 14.4x with nvcc
// ("error C2872: 'std' ambiguous"). ATen gives us Tensor + AT_DISPATCH + TORCH_CHECK.
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

__device__ __forceinline__ float amk_add_core(float a, float b) { return a + b; }

template <typename T>
__global__ void add_kernel(const T* __restrict__ a, const T* __restrict__ b,
                           T* __restrict__ out, long n) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x)
        out[i] = from_f32<T>(amk_add_core(to_f32(a[i]), to_f32(b[i])));
}

}  // namespace

void add(at::Tensor a, at::Tensor b, at::Tensor out) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(), "add: tensors must be CUDA");
    TORCH_CHECK(a.numel() == b.numel() && a.numel() == out.numel(),
                "add: a/b/out must have equal numel");
    auto ac = a.contiguous(), bc = b.contiguous();
    const long n = ac.numel();
    const int threads = AMK_THREADS;
    const int blocks = (int)std::min<long>((n + threads - 1) / threads, 65535);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, a.scalar_type(), "add", [&] {
        add_kernel<scalar_t><<<blocks, threads>>>(
            ac.data_ptr<scalar_t>(), bc.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "AMK ADD: out = a + b (fp32 accum)",
          py::arg("a"), py::arg("b"), py::arg("out"));
}
