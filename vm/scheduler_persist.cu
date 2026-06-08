/* ===========================================================================================
 * AutoMegaKernel, THE SINGLE-LAUNCH K-TOKEN PERSISTENT DECODE KERNEL (Layer 0, novel path)
 * ===========================================================================================
 * THE NOVEL CAPABILITY: run an ENTIRE K-token greedy decode loop inside ONE persistent
 * cooperative kernel launch, no per-token host relaunch, no per-token host marshalling/sync.
 *
 * Production engines (vLLM / TRT-LLM) relaunch (or per-step CUDA-graph replay) for EVERY decoded
 * token, paying host launch + marshalling + sync K times. AMK's baseline already does one launch
 * per token (vm/scheduler.cu). THIS kernel does the whole greedy loop in ONE launch: the host
 * launches once and reads back K generated token ids. Every per-token host/launch/sync cost is
 * eliminated, paid once instead of K times.
 *
 * It reuses the EXISTING per-step decode program verbatim (the same amk_device_program the
 * baseline runs: one forward pass == one token). The driver below wraps that program in an outer
 * step loop and threads the autoregressive feedback ENTIRELY in-kernel:
 *
 *   for step in [0, K):
 *     (A) SETUP   (block 0): write this step's absolute position into the `pos` IO_INPUT buffer
 *                 cell (ROPE reads it) and the prior step's sampled token into the `token_id`
 *                 IO_INPUT buffer cell (EMBED reads it). grid.sync() so every block sees them.
 *     (B) RUN     (all blocks): execute this SM's queue of the per-step task-DAG in global topo
 *                 order, amk_wait_all -> amk_dispatch -> amk_signal, EXACTLY as scheduler.cu,
 *                 with a per-instruction register-local POSITION PATCH (see below). grid.sync().
 *     (C) SAMPLE  (block 0): in-kernel SAMPLE_ARGMAX over the logits buffer -> next token id,
 *                 written to generated[step] (host reads this back) AND fed back as token_id for
 *                 the next step. grid.sync().
 *     (D) RESET   (all blocks, grid-strided): zero the per-step counters + abort flag for the
 *                 next iteration. grid.sync().
 *
 * POSITION PATCH (the one subtlety): the per-step program was lowered at a fixed base position, so
 * three position-dependent fields must advance each step. We NEVER mutate the shared device tables
 * (that would race across the 82 blocks); instead each block patches a REGISTER-LOCAL copy of the
 * instruction before dispatch, derived purely from `step`:
 *     ROPE          : reads `pos` from the pos IO_INPUT buffer  -> handled by (A), no patch.
 *     KV_APPEND     : params.pos   := base_pos + step           (write index into the KV cache)
 *     ATTENTION_TILE: params.kv_len := base_pos + step + 1      (causal window = pos+1)
 * These mirror schedule/lower.py's per-position lowering EXACTLY, so the K tokens this single
 * launch produces are token-for-token identical to the per-token-relaunch path (which re-lowers
 * the program at each pos). kv_start stays 0; everything else is position-independent.
 *
 * KV lives in HBM and persists across the in-kernel steps for free (same buffers, never realloc).
 * Counters are zeroed in-kernel between steps (the baseline host-zeroes them between launches).
 *
 * WDDM/TDR: this single launch runs K forward passes back-to-back, so K must be small enough to
 * stay under the ~2s local watchdog (K=8..16 on the laptop). On a no-TDR datacenter GPU K can be
 * large, that is where one-launch-for-the-whole-sequence pays off most.
 *
 * This file is ADDITIVE: it #includes the existing read-only headers (abi.h via amk_vm.cuh, the
 * frozen ops.cuh dispatch, sync.cuh, pages.cuh) and defines a NEW kernel + a plain-C launcher.
 * It does not touch scheduler.cu / ops.cuh / loader.py.
 *
 * STRUCTURE (mirrors scheduler.cu + vm_ext.cpp): this .cu holds ONLY the device kernel + a plain
 * launcher (no torch headers, nvcc + torch/extension.h on Windows hits a CCCL/MSVC 'std ambiguous'
 * parse error). The pybind/torch wrapper lives in vm/vm_ext_persist.cpp (host compiler); the shared
 * amk_persist_program POD lives in vm/amk_persist.cuh.
 * =========================================================================================== */
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "amk_persist.cuh"   /* amk_persist_program + amk_persist_launch_raw signature */
#include "amk_vm.cuh"
#include "sync.cuh"
#include "pages.cuh"
#include "ops.cuh"

namespace cg = cooperative_groups;

#if defined(AMK_LB_MAXTHREADS) && defined(AMK_LB_MINBLOCKS)
#define AMK_PERSIST_LAUNCH_BOUNDS __launch_bounds__(AMK_LB_MAXTHREADS, AMK_LB_MINBLOCKS)
#else
#define AMK_PERSIST_LAUNCH_BOUNDS
#endif

/* ---- in-kernel greedy argmax over the logits row, executed by block 0 only --------------------
 * Mirrors amk_inst_sample_argmax / instructions/reference.py ref_sample_argmax EXACTLY: argmax
 * over the last dim, ties resolve to the LOWEST index (torch semantics). The decode logits buffer
 * is a single row [1, vocab]; we reduce it with the whole block and thread 0 returns the id. */
__device__ __forceinline__ int amk_persist_argmax(const amk_buffer_t& logits) {
    const int64_t V = (logits.rank > 0) ? logits.shape[logits.rank - 1] : logits.numel;
    __shared__ float s_val[1024 / 32];
    __shared__ int   s_idx[1024 / 32];
    __shared__ int   s_result;
    const int lane   = threadIdx.x & (warpSize - 1);
    const int warp   = threadIdx.x / warpSize;
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;

    float best_v = -CUDART_INF_F;
    int   best_i = 0;
    for (int64_t j = threadIdx.x; j < V; j += blockDim.x) {
        float val = amk_load_f(logits, j);
        if (val > best_v) { best_v = val; best_i = (int)j; }
    }
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
        if (lane == 0) s_result = i;
    }
    __syncthreads();
    return s_result;
}

/* The single-launch K-token persistent decode kernel. Cooperative: every block co-resident, all
 * grid.sync()s are whole (out-of-range blocks still participate so the barrier count is exact). */
extern "C" __global__ void AMK_PERSIST_LAUNCH_BOUNDS
amk_megakernel_persist(amk_persist_program pp) {
    cg::grid_group grid = cg::this_grid();
    /* non-const local copy: amk_signal/amk_wait_all take a non-const amk_device_program& (they ++ a
     * counter / read abort), exactly as scheduler.cu passes its by-value `prog`. */
    amk_device_program prog = pp.base;           /* register-local copy of the table bases */
    const int sm = blockIdx.x;

    const int32_t  qlen  = (sm < prog.num_sms) ? prog.sm_queue_len[sm] : 0;
    const int32_t  qoff  = (sm < prog.num_sms) ? prog.sm_queue_off[sm] : 0;
    const int32_t* queue = prog.sm_queue_flat + qoff;

    /* The token fed into THIS step's EMBED. Step 0 uses the host-provided first_token; every later
     * step uses the token the previous step sampled (threaded in-kernel via pp.token_cell). */
    int cur_token = pp.first_token;

    grid.sync();   /* entry barrier: host-zeroed counters + all tables globally visible */

    for (int step = 0; step < pp.K; ++step) {
        const int abs_pos = pp.base_pos + step;

        /* ---- (A) SETUP: publish this step's position + input token into the IO_INPUT cells ---- */
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *pp.pos_cell   = abs_pos;        /* ROPE reads this (the `pos` IO_INPUT buffer)      */
            *pp.token_cell = cur_token;      /* EMBED reads this (the `token_id` IO_INPUT buffer)*/
        }
        grid.sync();   /* every block now sees the step's pos + token before any dispatch */

        /* ---- (B) RUN: this SM's queue of the per-step DAG, in global topo order -------------- */
        bool aborted = false;
        for (int32_t qi = 0; qi < qlen; ++qi) {
            const int32_t inst_idx = queue[qi];
            /* REGISTER-LOCAL copy so the per-step POSITION PATCH never mutates shared tables. */
            amk_instruction_t inst = prog.instructions[inst_idx];
            if (inst.op == AMK_OP_KV_APPEND) {
                inst.params.pos = abs_pos;                 /* write index into the KV cache */
            } else if (inst.op == AMK_OP_ATTENTION_TILE) {
                inst.params.kv_len = abs_pos + 1;          /* causal window = pos + 1 */
            }
            if (!amk_wait_all(prog, inst)) { aborted = true; break; }
            amk_dispatch(prog, inst);
            amk_signal(prog, inst.out_counter);
        }
        grid.sync();   /* all producers retired: the logits row is complete & visible */
        if (aborted) break;
        /* a device-side TRAP set the abort flag during dispatch -> stop the whole loop cleanly */
        if (*prog.abort_flag != 0) break;

        /* ---- (C) SAMPLE: in-kernel greedy argmax -> generated[step] + next step's token ------ */
        if (blockIdx.x == 0) {
            int next = amk_persist_argmax(*pp.logits);
            if (threadIdx.x == 0) pp.generated[step] = next;
        }
        /* every block needs the sampled token for the next step; recompute it locally after the
         * barrier so we don't depend on a second global broadcast write. The sampled value was
         * written to generated[step] by block 0; publish + read it grid-wide. */
        grid.sync();
        cur_token = pp.generated[step];

        /* ---- (D) RESET: zero the per-step counters + abort flag for the next iteration -------- */
        for (int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
             c < prog.n_counters; c += gridDim.x * blockDim.x) {
            prog.counters[c] = 0;
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) *prog.abort_flag = 0;
        grid.sync();   /* counters are zero everywhere before the next step spins on them */
    }

    grid.sync();   /* exit barrier */
}

/* ===========================================================================================
 * PLAIN-C LAUNCHER (no torch), assembles amk_persist_program and fires ONE cooperative launch.
 * The torch/pybind wrapper lives in vm/vm_ext_persist.cpp and calls these. Status is returned as
 * an enum + a CUDA error string so the .cpp can build the python dict without nvcc seeing torch.
 * =========================================================================================== */
int64_t amk_persist_max_coresident_blocks_impl(int64_t threads_per_block, int64_t dyn_smem_bytes) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 0;
    if (dyn_smem_bytes > 0) {
        cudaFuncSetAttribute((const void*)amk_megakernel_persist,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_bytes);
    }
    int blocks_per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, (const void*)amk_megakernel_persist,
            (int)threads_per_block, (size_t)dyn_smem_bytes) != cudaSuccess)
        return 0;
    return (int64_t)blocks_per_sm * (int64_t)prop.multiProcessorCount;
}

bool amk_persist_supports_cooperative_impl() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    int coop = 0;
    if (cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev) != cudaSuccess)
        return false;
    return coop != 0;
}

/* Launch the persistent K-token decode kernel on `stream`. Returns:
 *   0 = OK, 1 = TIMEOUT (WDDM TDR), 2 = launch/sync ERROR, 3 = co-residency/contract REJECT.
 * On a nonzero return, `errbuf` (>= errbuf_len) holds a human-readable message. */
int amk_persist_launch_impl(const amk_persist_launch_args* a, void* stream_handle,
                            char* errbuf, int errbuf_len) {
    auto setmsg = [&](const char* s) {
        if (errbuf && errbuf_len > 0) {
            int i = 0; for (; s[i] && i < errbuf_len - 1; ++i) errbuf[i] = s[i]; errbuf[i] = '\0';
        }
    };
    if (!amk_persist_supports_cooperative_impl()) {
        setmsg("device does not support cudaLaunchCooperativeKernel"); return 3;
    }
    if (a->dyn_smem_bytes > 0) {
        cudaFuncSetAttribute((const void*)amk_megakernel_persist,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)a->dyn_smem_bytes);
    }
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, (const void*)amk_megakernel_persist,
        (int)a->threads_per_block, (size_t)a->dyn_smem_bytes);
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
    int64_t max_grid = (int64_t)blocks_per_sm * (int64_t)prop.multiProcessorCount;
    if (blocks_per_sm <= 0) { setmsg("kernel has ZERO occupancy at the requested config"); return 3; }
    if (a->grid_dim > max_grid) {
        setmsg("requested gridDim exceeds cooperative co-residency limit, REFUSING to launch");
        return 3;
    }

    amk_persist_program pp{};
    pp.base.buffers       = (const amk_buffer_t*)a->buffers_ptr;
    pp.base.n_buffers     = (int32_t)a->n_buffers;
    pp.base.counters      = (amk_counter_t*)a->counters_ptr;
    pp.base.n_counters    = (int32_t)a->n_counters;
    pp.base.instructions  = (const amk_instruction_t*)a->instructions_ptr;
    pp.base.n_instructions= (int32_t)a->n_instructions;
    pp.base.sm_queue_flat = (const int32_t*)a->sm_queue_flat_ptr;
    pp.base.sm_queue_off  = (const int32_t*)a->sm_queue_off_ptr;
    pp.base.sm_queue_len  = (const int32_t*)a->sm_queue_len_ptr;
    pp.base.num_sms       = (int32_t)a->num_sms;
    pp.base.scratch       = (unsigned char*)a->scratch_ptr;
    pp.base.scratch_bytes = a->scratch_bytes;
    pp.base.abort_flag    = (int32_t*)a->abort_flag_ptr;

    pp.K          = (int32_t)a->K;
    pp.base_pos   = (int32_t)a->base_pos;
    pp.pos_cell   = (int32_t*)a->pos_cell_ptr;
    pp.token_cell = (int32_t*)a->token_cell_ptr;
    pp.logits     = (const amk_buffer_t*)a->buffers_table_ptr + (int32_t)a->logits_buf_idx;
    pp.generated  = (int32_t*)a->generated_ptr;
    pp.first_token= (int32_t)a->first_token;

    dim3 grid((unsigned)a->grid_dim), block((unsigned)a->threads_per_block);
    void* kargs[] = { (void*)&pp };
    cudaStream_t stream = (cudaStream_t)stream_handle;

    cudaError_t launch_err = cudaLaunchCooperativeKernel(
        (const void*)amk_megakernel_persist, grid, block, kargs,
        (size_t)a->dyn_smem_bytes, stream);
    if (launch_err != cudaSuccess) {
        cudaGetLastError(); setmsg(cudaGetErrorString(launch_err)); return 2;
    }
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err == cudaErrorLaunchTimeout) {
        cudaGetLastError();
        setmsg("cudaErrorLaunchTimeout (Windows WDDM TDR ~2s), shrink K or raise TdrDelay");
        return 1;
    }
    if (sync_err != cudaSuccess) { cudaGetLastError(); setmsg(cudaGetErrorString(sync_err)); return 2; }
    return 0;
}

void amk_persist_struct_sizes_impl(int64_t* params, int64_t* buffer, int64_t* instr) {
    *params = (int64_t)sizeof(amk_params_t);
    *buffer = (int64_t)sizeof(amk_buffer_t);
    *instr  = (int64_t)sizeof(amk_instruction_t);
}
