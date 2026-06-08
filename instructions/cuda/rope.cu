/* ===========================================================================================
 * AMK Layer-1 micro-kernel, ROPE  (opcode AMK_OP_ROPE = 8)
 * ===========================================================================================
 * Matches instructions/reference.py `ref_rope` EXACTLY (GPT-NeoX / Llama rotate-half):
 *
 *     x        : [S, n_heads, head_dim]    (or [n_heads, head_dim] for one step)
 *     pos      : [S] integer positions
 *     half     = head_dim // 2
 *     inv_freq = 1 / theta^(arange(0,half)/half)            # length `half`
 *     ang      = pos[:,None] * inv_freq[None,:]             # [S, half]
 *     cos,sin  = cos(ang), sin(ang)                         # [S, half]  (NOT duplicated)
 *     x1,x2    = x[..., :half], x[..., half:]
 *     out[..., :half]  = x1*cos - x2*sin
 *     out[..., half:]  = x2*cos + x1*sin
 *
 * cos/sin broadcast over heads (same per position). All compute in fp32, cast to out dtype.
 * ABI plug-in: `amk_rope_core` applies the rotation to one [n_heads,head_dim] step given its
 * position; the kernel maps (seq, head) pairs to blocks/threads. theta = params.theta.
 * =========================================================================================== */
// ATen + pybind directly (NOT <torch/extension.h>; see add.cu note on MSVC C2872).
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef AMK_THREADS
#define AMK_THREADS 128
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

/* ABI device core: rotate one head's head_dim vector at sequence position `pos`. */
template <typename T>
__device__ void amk_rope_core(const T* __restrict__ x, T* __restrict__ out,
                              int head_dim, long pos, float theta) {
    const int half = head_dim / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        // inv_freq[i] = theta^(-(i/half))  => ang = pos * inv_freq[i]
        float inv = __powf(theta, -((float)i / (float)half));
        float ang = (float)pos * inv;
        float c = __cosf(ang), s = __sinf(ang);
        float x1 = to_f32(x[i]);
        float x2 = to_f32(x[i + half]);
        out[i]        = from_f32<T>(x1 * c - x2 * s);
        out[i + half] = from_f32<T>(x2 * c + x1 * s);
    }
}

template <typename T, typename IT>
__global__ void rope_kernel(const T* __restrict__ x, const IT* __restrict__ pos,
                            T* __restrict__ out, int head_dim, int n_heads, float theta) {
    const int s = blockIdx.y;          // sequence index
    const int h = blockIdx.x;          // head index
    const long base = ((long)s * n_heads + h) * head_dim;
    amk_rope_core<T>(x + base, out + base, head_dim, (long)pos[s], theta);
}

template <typename scalar_t>
void launch(const at::Tensor& x, const at::Tensor& pos, at::Tensor& out,
            int head_dim, int n_heads, int S, float theta) {
    dim3 grid(n_heads, S);
    const int threads = head_dim / 2 < 32 ? 32 : (head_dim / 2 > 256 ? 256 : head_dim / 2);
    if (pos.scalar_type() == at::kLong) {
        rope_kernel<scalar_t, int64_t><<<grid, threads>>>(
            x.data_ptr<scalar_t>(), pos.data_ptr<int64_t>(), out.data_ptr<scalar_t>(),
            head_dim, n_heads, theta);
    } else {
        auto pi = pos.to(at::kInt);
        rope_kernel<scalar_t, int32_t><<<grid, threads>>>(
            x.data_ptr<scalar_t>(), pi.data_ptr<int32_t>(), out.data_ptr<scalar_t>(),
            head_dim, n_heads, theta);
    }
}

}  // namespace

// x: [S, n_heads, head_dim] (or [n_heads, head_dim] -> S=1). out same shape, pre-allocated.
void rope(at::Tensor x, at::Tensor pos, at::Tensor out, int64_t head_dim, double theta) {
    TORCH_CHECK(x.is_cuda() && pos.is_cuda() && out.is_cuda(), "rope: tensors must be CUDA");
    TORCH_CHECK(x.size(-1) == head_dim, "rope: x last dim must equal head_dim");
    int S, n_heads;
    if (x.dim() == 3) { S = x.size(0); n_heads = x.size(1); }
    else { S = 1; n_heads = x.size(0); }
    auto xc = x.contiguous();
    auto pc = pos.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rope", [&] {
        launch<scalar_t>(xc, pc, out, (int)head_dim, n_heads, S, (float)theta);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rope", &rope, "AMK ROPE (Llama rotate-half): out = rotate(x, pos, theta)",
          py::arg("x"), py::arg("pos"), py::arg("out"), py::arg("head_dim"), py::arg("theta"));
}
