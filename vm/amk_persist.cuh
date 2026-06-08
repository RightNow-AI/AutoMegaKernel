/* ===========================================================================================
 * AutoMegaKernel, shared types for the SINGLE-LAUNCH K-TOKEN persistent decode path.
 * ===========================================================================================
 * Split out so the device kernel (scheduler_persist.cu, nvcc) and the torch/pybind host wrapper
 * (vm_ext_persist.cpp, host compiler) agree on the POD layout WITHOUT the .cu ever including
 * torch/extension.h (nvcc + torch headers on Windows hits a CCCL/MSVC 'std ambiguous' parse error).
 * =========================================================================================== */
#ifndef AMK_PERSIST_CUH
#define AMK_PERSIST_CUH

#include <cstdint>
#include "amk_vm.cuh"   /* amk_device_program, amk_buffer_t, amk_counter_t, amk_instruction_t */

/* The persistent-decode device program: the SAME amk_device_program the baseline runs, plus the
 * extra device pointers/scalars the in-kernel autoregressive loop needs. pos/token cells are the
 * data_ptr()s of the existing `pos`/`token_id` IO_INPUT buffers; logits is the `logits` IO_OUTPUT. */
struct amk_persist_program {
    amk_device_program base;     /* buffers/counters/instructions/queues/scratch/abort (by value) */
    int32_t  K;                  /* number of decode steps to run in this single launch */
    int32_t  base_pos;           /* absolute position of step 0 (== prompt length so far) */
    int32_t* pos_cell;           /* device int32 cell read by ROPE (the `pos` IO_INPUT buffer)     */
    int32_t* token_cell;         /* device int32 cell read by EMBED (the `token_id` IO_INPUT buf)  */
    const amk_buffer_t* logits;  /* the `logits` IO_OUTPUT buffer record (for in-kernel argmax)    */
    int32_t* generated;          /* [K] device int32 output: the K sampled token ids (host reads)  */
    int32_t  first_token;        /* token id fed at step 0 (the last prompt token)                 */
};

/* Plain-POD launch arguments the host wrapper fills from python int64 device addresses. Kept as a
 * flat struct (not a long arg list) so the host .cpp and the device .cu stay trivially in sync. */
struct amk_persist_launch_args {
    int64_t buffers_ptr, n_buffers;
    int64_t counters_ptr, n_counters;
    int64_t instructions_ptr, n_instructions;
    int64_t sm_queue_flat_ptr, sm_queue_off_ptr, sm_queue_len_ptr;
    int64_t num_sms;
    int64_t scratch_ptr, scratch_bytes;
    int64_t abort_flag_ptr;
    int64_t grid_dim, threads_per_block, dyn_smem_bytes;
    int64_t K, base_pos;
    int64_t pos_cell_ptr, token_cell_ptr;
    int64_t buffers_table_ptr, logits_buf_idx;
    int64_t generated_ptr, first_token;
};

/* Implemented in scheduler_persist.cu (CUDA-runtime only), called from vm_ext_persist.cpp (torch). */
int64_t amk_persist_max_coresident_blocks_impl(int64_t threads_per_block, int64_t dyn_smem_bytes);
bool    amk_persist_supports_cooperative_impl();
void    amk_persist_struct_sizes_impl(int64_t* params, int64_t* buffer, int64_t* instr);
/* returns 0=OK 1=TIMEOUT 2=ERROR 3=REJECT; errbuf gets the message on nonzero. */
int     amk_persist_launch_impl(const amk_persist_launch_args* a, void* stream_handle,
                                char* errbuf, int errbuf_len);

#endif /* AMK_PERSIST_CUH */
