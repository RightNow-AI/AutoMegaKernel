/* ===========================================================================================
 * AutoMegaKernel, THE PERSISTENT MEGAKERNEL VM (Layer 0)
 * ===========================================================================================
 * One cooperative launch == one forward pass == one decoded token (abi.h DECODE MODEL).
 *
 *   - Launched with cudaLaunchCooperativeKernel so every block is co-resident and makes forward
 *     progress (a persistent block that is never scheduled = guaranteed deadlock). The host
 *     (vm_ext.cpp) proves gridDim <= cudaOccupancyMaxActiveBlocksPerMultiprocessor * num_sms and
 *     REFUSES to launch otherwise.
 *   - gridDim.x == prog.num_sms (one block per SM is the target). Block b owns SM-queue b.
 *   - Kernel entry: grid.sync() after each block caches its queue (and the host-zeroed counters
 *     are confirmed visible). Then for each instruction in its queue, in GLOBAL TOPOLOGICAL ORDER:
 *         amk_wait_all(inst)  ->  amk_dispatch(inst)  ->  amk_signal(inst.out_counter)
 *     honoring abort_flag (WDDM/TDR escape). Kernel exit: grid.sync().
 *
 * The per-op compute lives in ops.cuh; sync in sync.cuh; buffer resolution in pages.cuh. All are
 * __forceinline__ so the whole VM collapses into this single kernel.
 * =========================================================================================== */
#include <cooperative_groups.h>

#include "amk_vm.cuh"
#include "sync.cuh"
#include "pages.cuh"
#include "ops.cuh"

namespace cg = cooperative_groups;

/* ===========================================================================================
 * SOFTWARE-PIPELINED WEIGHT PREFETCH (acts on ScheduleConfig.pipelining_depth)
 * -------------------------------------------------------------------------------------------
 * Decode is HBM-bandwidth bound and the single biggest megakernel win is hiding the inter-op
 * latency of streaming the NEXT GEMV tile's weight rows from HBM. While this block computes the
 * current instruction, we issue a non-binding L2 prefetch of the weight tile that the next
 * GEMV_TILE in THIS SM's queue will read. Walking only this SM's own queue keeps it correct under
 * the topo-order contract (no cross-SM assumption). It is a pure hint: never changes results, and
 * lowers to a no-op on archs without prefetch PTX (see amk_prefetch_l2).
 *
 * Depth is carried per GEMV_TILE in params.M_tile (unused by the GEMV op, which reads only
 * K/N_tile/n_off). The loader injects ScheduleConfig.pipelining_depth there; 0 disables prefetch.
 * We look ahead up to `depth` queue slots for the next GEMV and prefetch its [N_tile x K] weight
 * rows (the bandwidth-dominant operand). =========================================================*/
__device__ __forceinline__ void amk_prefetch_gemv_weights(const amk_device_program& prog,
                                                          const amk_instruction_t& inst) {
    if (inst.op != AMK_OP_GEMV_TILE) return;
    const amk_buffer_t& W = amk_buf(prog, inst.inputs[1]);   /* weight [N,K] */
    const int   K      = inst.params.K;
    const int   N_tile = inst.params.N_tile;
    const int   n_off  = inst.params.n_off;
    const int   esz    = amk_dtype_bytes(W.dtype);
    /* contiguous weight rows [n_off, n_off+N_tile) of length K (row-major Linear layout) */
    const int64_t byte_off = (int64_t)n_off * W.stride[0] * esz;
    const int64_t nbytes   = (int64_t)N_tile * (int64_t)K * esz;
    amk_prefetch_l2(W.ptr, byte_off, nbytes);
}

__device__ __forceinline__ void amk_prefetch_next_gemv(const amk_device_program& prog,
                                                       const int32_t* queue, int32_t qi,
                                                       int32_t qlen, int depth) {
    if (depth <= 0) return;
    const int32_t end = (qi + 1 + depth < qlen) ? (qi + 1 + depth) : qlen;
    for (int32_t j = qi + 1; j < end; ++j) {
        const amk_instruction_t& nxt = prog.instructions[queue[j]];
        if (nxt.op != AMK_OP_GEMV_TILE) continue;
        amk_prefetch_gemv_weights(prog, nxt);
        return;   /* prefetch the FIRST upcoming GEMV only (one tile in flight) */
    }
}

/* OCCUPANCY KNOB (autotune): __launch_bounds__(maxThreads, minBlocksPerSM) caps per-thread
 * registers so MORE blocks/warps co-reside per SM. For a bandwidth-bound persistent decode kernel
 * the achieved HBM bandwidth is gated by memory-level parallelism (resident warps), so raising
 * occupancy is the real lever. The loader threads these as -DAMK_LB_MAXTHREADS / -DAMK_LB_MINBLOCKS
 * and builds a distinct extension variant; undefined => no launch bound (compiler's own reg alloc,
 * the prior default behavior). */
#if defined(AMK_LB_MAXTHREADS) && defined(AMK_LB_MINBLOCKS)
#define AMK_LAUNCH_BOUNDS __launch_bounds__(AMK_LB_MAXTHREADS, AMK_LB_MINBLOCKS)
#else
#define AMK_LAUNCH_BOUNDS
#endif

/* The device program is passed BY VALUE (a small POD of pointers + counts) so every block has its
 * own register copy of the table bases, no extra indirection through global memory. */
extern "C" __global__ void AMK_LAUNCH_BOUNDS amk_megakernel(amk_device_program prog) {
    cg::grid_group grid = cg::this_grid();

    const int sm = blockIdx.x;             /* this block owns SM-queue `sm` */

    /* ---- ENTRY BARRIER --------------------------------------------------------------------
     * Guarantees the host-zeroed counters and all table copies are globally visible before any
     * block starts spinning on a counter (otherwise a fast block could read a stale, nonzero
     * counter from a previous launch's memory). */
    grid.sync();

    /* Out-of-range blocks (gridDim > num_sms should never happen, the loader caps it, but be
     * defensive) still participate in the two grid.sync()s so the cooperative barrier is whole. */
    if (sm < prog.num_sms) {
        const int32_t  qlen = prog.sm_queue_len[sm];
        const int32_t  qoff = prog.sm_queue_off[sm];
        const int32_t* queue = prog.sm_queue_flat + qoff;

        for (int32_t qi = 0; qi < qlen; ++qi) {
            const int32_t inst_idx = queue[qi];
            const amk_instruction_t& inst = prog.instructions[inst_idx];

            /* 1a) software pipeline (depth>0): if THIS op is a GEMV, kick its weight rows toward L2
             *     BEFORE the input-counter wait. The weights are HBM (always ready); the wait spins
             *     on the activation dependency. Issuing the prefetch first overlaps the weight HBM
             *     latency with the dependency spin, a real producer/consumer overlap, not extra
             *     serial work. Pure hint; never affects results. M_tile carries the depth gate. */
            if (inst.params.M_tile > 0) {
                amk_prefetch_gemv_weights(prog, inst);
            }

            /* 1b) wait on all input counters (acquire spin w/ backoff, abort-aware) */
            if (!amk_wait_all(prog, inst)) {
                break;                      /* watchdog fired: stop, head for the exit barrier */
            }

            /* 1c) also warm the NEXT GEMV tile's weights so they overlap THIS op's compute. */
            amk_prefetch_next_gemv(prog, queue, qi, qlen, inst.params.M_tile);

            /* 2) pure compute (block-cooperative) */
            amk_dispatch(prog, inst);

            /* 3) release-fence outputs, then ++ out_counter (signal completion) */
            amk_signal(prog, inst.out_counter);
        }
    }

    /* ---- EXIT BARRIER ---------------------------------------------------------------------- */
    grid.sync();
}
