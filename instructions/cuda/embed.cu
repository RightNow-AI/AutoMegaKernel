/* ===========================================================================================
 * AMK Layer-1 micro-kernel, EMBED  (opcode AMK_OP_EMBED = 2)
 * ===========================================================================================
 * Matches instructions/reference.py `ref_embed` EXACTLY:
 *
 *     ids   : [S] integer token ids
 *     table : [V, H]  embedding matrix
 *     out   : [S, H] = table[ids]              (gather rows)
 *
 * One threadblock per output row copies table[id, :] into out[row, :]. ABI plug-in:
 * `amk_embed_core` copies a single row given a resolved id; the kernel maps rows to blocks.
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

/* ABI device core: copy embedding row `id` of the [V,H] table into out[0..H). */
template <typename T>
__device__ void amk_embed_core(const T* __restrict__ table, long id, T* __restrict__ out, int H) {
    const T* src = table + id * (long)H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) out[i] = src[i];
}

template <typename T, typename IT>
__global__ void embed_kernel(const IT* __restrict__ ids, const T* __restrict__ table,
                             T* __restrict__ out, int H, long V) {
    const int row = blockIdx.x;
    long id = (long)ids[row];
    if (id < 0) id = 0;
    if (id >= V) id = V - 1;             // defensive clamp; reference assumes valid ids
    amk_embed_core<T>(table, id, out + (long)row * H, H);
}

template <typename scalar_t>
void launch(const at::Tensor& ids, const at::Tensor& table, at::Tensor& out,
            int H, long V, int S) {
    const int threads = AMK_THREADS;
    if (ids.scalar_type() == at::kLong) {
        embed_kernel<scalar_t, int64_t><<<S, threads>>>(
            ids.data_ptr<int64_t>(), table.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), H, V);
    } else {
        auto idi = ids.to(at::kInt);
        embed_kernel<scalar_t, int32_t><<<S, threads>>>(
            idi.data_ptr<int32_t>(), table.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), H, V);
    }
}

}  // namespace

void embed(at::Tensor ids, at::Tensor table, at::Tensor out) {
    TORCH_CHECK(ids.is_cuda() && table.is_cuda() && out.is_cuda(), "embed: tensors must be CUDA");
    TORCH_CHECK(table.dim() == 2, "embed: table must be [V,H]");
    const long V = table.size(0);
    const int H = table.size(1);
    const int S = ids.numel();
    auto idsc = ids.contiguous(), tc = table.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, table.scalar_type(), "embed", [&] {
        launch<scalar_t>(idsc, tc, out, H, V, S);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embed", &embed, "AMK EMBED: out[s,:] = table[ids[s], :]",
          py::arg("ids"), py::arg("table"), py::arg("out"));
}
