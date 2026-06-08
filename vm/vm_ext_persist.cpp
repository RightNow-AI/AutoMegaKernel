/* ===========================================================================================
 * AutoMegaKernel, torch/pybind HOST WRAPPER for the single-launch K-token persistent decode.
 * ===========================================================================================
 * Thin host layer (host compiler, NOT nvcc) that exposes the plain-C launcher in
 * scheduler_persist.cu to python. Keeping torch/extension.h out of the .cu avoids the nvcc + CCCL
 * 'std ambiguous' Windows parse error; this file does the python<->C marshalling and uses the
 * current torch CUDA stream so the launch lands on the same stream as the rest of AMK.
 * =========================================================================================== */
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <string>

#include "amk_persist.cuh"   /* amk_persist_launch_args + the *_impl signatures (CUDA-runtime side) */

int64_t persist_max_coresident_blocks(int64_t threads_per_block, int64_t dyn_smem_bytes) {
    return amk_persist_max_coresident_blocks_impl(threads_per_block, dyn_smem_bytes);
}

bool persist_supports_cooperative() {
    return amk_persist_supports_cooperative_impl();
}

py::dict persist_struct_sizes() {
    int64_t p = 0, b = 0, i = 0;
    amk_persist_struct_sizes_impl(&p, &b, &i);
    py::dict d;
    d["amk_params_t"]      = p;
    d["amk_buffer_t"]      = b;
    d["amk_instruction_t"] = i;
    return d;
}

py::dict persist_launch(int64_t buffers_ptr,        int64_t n_buffers,
                        int64_t counters_ptr,       int64_t n_counters,
                        int64_t instructions_ptr,   int64_t n_instructions,
                        int64_t sm_queue_flat_ptr,
                        int64_t sm_queue_off_ptr,
                        int64_t sm_queue_len_ptr,
                        int64_t num_sms,
                        int64_t scratch_ptr,        int64_t scratch_bytes,
                        int64_t abort_flag_ptr,
                        int64_t grid_dim,           int64_t threads_per_block,
                        int64_t dyn_smem_bytes,
                        int64_t K,                  int64_t base_pos,
                        int64_t pos_cell_ptr,       int64_t token_cell_ptr,
                        int64_t buffers_table_ptr,  int64_t logits_buf_idx,
                        int64_t generated_ptr,      int64_t first_token) {
    amk_persist_launch_args a{};
    a.buffers_ptr = buffers_ptr;           a.n_buffers = n_buffers;
    a.counters_ptr = counters_ptr;         a.n_counters = n_counters;
    a.instructions_ptr = instructions_ptr; a.n_instructions = n_instructions;
    a.sm_queue_flat_ptr = sm_queue_flat_ptr;
    a.sm_queue_off_ptr = sm_queue_off_ptr;
    a.sm_queue_len_ptr = sm_queue_len_ptr;
    a.num_sms = num_sms;
    a.scratch_ptr = scratch_ptr;           a.scratch_bytes = scratch_bytes;
    a.abort_flag_ptr = abort_flag_ptr;
    a.grid_dim = grid_dim;                 a.threads_per_block = threads_per_block;
    a.dyn_smem_bytes = dyn_smem_bytes;
    a.K = K;                               a.base_pos = base_pos;
    a.pos_cell_ptr = pos_cell_ptr;         a.token_cell_ptr = token_cell_ptr;
    a.buffers_table_ptr = buffers_table_ptr; a.logits_buf_idx = logits_buf_idx;
    a.generated_ptr = generated_ptr;       a.first_token = first_token;

    char errbuf[256] = {0};
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int rc = amk_persist_launch_impl(&a, (void*)stream, errbuf, (int)sizeof(errbuf));

    py::dict result;
    const char* status = (rc == 0) ? "OK" : (rc == 1) ? "TIMEOUT" : (rc == 3) ? "REJECTED" : "ERROR";
    result["status"] = std::string(status);
    result["error"]  = std::string(errbuf);
    if (rc == 3) throw std::runtime_error(std::string("AMK persist REJECTED: ") + errbuf);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("persist_launch", &persist_launch,
          "Launch the AMK single-launch K-token persistent decode kernel (cooperative)");
    m.def("persist_max_coresident_blocks", &persist_max_coresident_blocks,
          "Max co-resident blocks for the persistent-decode kernel at (threads, dyn_smem)");
    m.def("persist_supports_cooperative", &persist_supports_cooperative,
          "Whether the device supports cudaLaunchCooperativeKernel");
    m.def("persist_struct_sizes", &persist_struct_sizes,
          "C sizeof of the ABI PODs (byte-layout drift guard)");
}
