/* ===========================================================================================
 * AMK Layer-1 micro-kernel, GEMV_TILE  (opcode AMK_OP_GEMV_TILE = 5)
 * ===========================================================================================
 * Computes one output-column tile of a decode matvec, matching instructions/reference.py
 * `_gemv_gemm` EXACTLY:
 *
 *     x : [M, K]        (M==1 for decode gemv; we support M>=1 row-by-row)
 *     W : [N, K]        (torch nn.Linear layout, weight is [out, in])
 *     out[..., n_off : n_off+N_tile] = x @ W[n_off:n_off+N_tile, :].T   (+ optional bias tile)
 *
 * Accumulation is in fp32 (matches reference + tensor-core fp32-accumulate), output cast to the
 * out dtype. One threadblock computes the whole tile for all M rows: each warp owns one output
 * column n (n in [n_off, n_off+N_tile)) and reduces the length-K dot product, vectorising the
 * K loop by 4 when alignment allows. This is the bandwidth-bound decode kernel: each weight row
 * is read exactly once.
 *
 * ABI plug-in shape: the math lives in `amk_gemv_tile_core`, a __device__ routine that takes the
 * already-resolved base pointers + (K, N, N_tile, n_off, M), i.e. exactly what the VM extracts
 * from amk_buffer_t/amk_params_t for amk_inst_gemv_tile. The torch wrapper below is only a test
 * harness around the same core.
 * =========================================================================================== */
// ATen + pybind directly (NOT <torch/extension.h>; see add.cu note on MSVC C2872).
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Tunable block size (gen.py search knob). Absent -> built-in default. Must be a multiple of 32.
#ifndef AMK_THREADS
#define AMK_THREADS 256
#endif

namespace {

// ---- dtype-generic element load to fp32 (the VM passes typed buffers; here templated) --------
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

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

/* -----------------------------------------------------------------------------------------
 * ABI-conformant device core. Computes out[m, n_off+j] for j in [0,N_tile), m in [0,M).
 * Layout: x is row-major [M,K] (stride_x_row=K), W is row-major [N,K] (stride_w_row=K),
 * out is row-major [M,Nfull] (stride_o_row=Nfull). bias may be null (length N, indexed at n).
 * One warp per (m, output-column) pair, walking n in a grid-stride over the tile.
 * --------------------------------------------------------------------------------------- */
template <typename XT, typename WT, typename OT, typename BT>
__device__ void amk_gemv_tile_core(const XT* __restrict__ x, const WT* __restrict__ W,
                                   OT* __restrict__ out, const BT* __restrict__ bias,
                                   int K, int Nfull, int N_tile, int n_off, int M,
                                   int stride_x_row, int stride_w_row, int stride_o_row) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    for (int m = 0; m < M; ++m) {
        const XT* xrow = x + (long)m * stride_x_row;
        // each warp handles a strided set of tile columns
        for (int j = warp; j < N_tile; j += warps) {
            const int n = n_off + j;
            const WT* wrow = W + (long)n * stride_w_row;
            float acc = 0.f;
            for (int k = lane; k < K; k += 32)
                acc += to_f32(xrow[k]) * to_f32(wrow[k]);
            acc = warp_reduce_sum(acc);
            if (lane == 0) {
                if (bias) acc += to_f32(bias[n]);
                out[(long)m * stride_o_row + n] = from_f32<OT>(acc);
            }
        }
    }
}

template <typename XT, typename WT, typename OT, typename BT>
__global__ void gemv_tile_kernel(const XT* x, const WT* W, OT* out, const BT* bias,
                                 int K, int Nfull, int N_tile, int n_off, int M,
                                 int sxr, int swr, int sor) {
    amk_gemv_tile_core<XT, WT, OT, BT>(x, W, out, bias, K, Nfull, N_tile, n_off, M, sxr, swr, sor);
}

template <typename scalar_t>
void launch(const at::Tensor& x, const at::Tensor& W, at::Tensor& out,
            const at::Tensor* bias, int K, int Nfull, int N_tile, int n_off, int M) {
    const int threads = AMK_THREADS;  // warps -> output cols in flight per block (tunable)
    const int blocks = 1;             // one block computes the whole tile (decode is tiny)
    using T = scalar_t;
    const T* bptr = bias ? bias->data_ptr<T>() : nullptr;
    gemv_tile_kernel<T, T, T, T><<<blocks, threads>>>(
        x.data_ptr<T>(), W.data_ptr<T>(), out.data_ptr<T>(), bptr,
        K, Nfull, N_tile, n_off, M, K, K, Nfull);
}

}  // namespace

// ---- torch entry: out is pre-allocated [M, Nfull]; writes only the [.., n_off:n_off+N_tile] slice
void gemv_tile(at::Tensor x, at::Tensor W, at::Tensor out,
               c10::optional<at::Tensor> bias, int64_t n_off, int64_t N_tile) {
    TORCH_CHECK(x.is_cuda() && W.is_cuda() && out.is_cuda(), "gemv_tile: tensors must be CUDA");
    TORCH_CHECK(x.dim() == 2 && W.dim() == 2 && out.dim() == 2, "gemv_tile: expect 2D x,W,out");
    const int M = x.size(0), K = x.size(1), Nfull = out.size(1);
    TORCH_CHECK(W.size(1) == K, "gemv_tile: W.size(1) must equal K");
    auto xc = x.contiguous(), Wc = W.contiguous();
    at::Tensor bc;
    const at::Tensor* bptr = nullptr;
    if (bias.has_value()) { bc = bias->contiguous(); bptr = &bc; }
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "gemv_tile", [&] {
        launch<scalar_t>(xc, Wc, out, bptr, K, Nfull, (int)N_tile, (int)n_off, M);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_tile", &gemv_tile,
          "AMK GEMV_TILE: out[..,n_off:n_off+N_tile] = x @ W[n_off:n_off+N_tile,:].T (+bias)",
          py::arg("x"), py::arg("W"), py::arg("out"), py::arg("bias"),
          py::arg("n_off"), py::arg("N_tile"));
}
