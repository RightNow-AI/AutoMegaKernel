/* ===========================================================================================
 * AutoMegaKernel, THE CANONICAL SYNC PRIMITIVES (Layer 0)
 * ===========================================================================================
 * Implements the SYNC CONTRACT frozen in vm/abi.h, verbatim. These are the ONLY way the VM
 * crosses SM boundaries; instructions never touch counters themselves.
 *
 *   amk_signal(prog, c):
 *       thread 0 does a DEVICE-scope release fence ordering ALL output-buffer stores, THEN the
 *       atomic increment; finally a block barrier so the whole block agrees the inst is retired.
 *           __threadfence();                  // .gl / device scope release
 *           atomicAdd(&counters[c], 1u);      // monotonic ++ (producers only)
 *           __syncthreads();
 *
 *   amk_wait_all(prog, inst):
 *       thread 0 spins on a NON-HOISTABLE ACQUIRE load (atomicAdd(&c,0), RMW, cannot be hoisted
 *       or cached) until every (counter,threshold) pair holds, with exponential __nanosleep
 *       backoff (~32ns..~few us) after a short busy window, polling abort_flag for the WDDM/TDR
 *       watchdog escape. Returns false if aborted (the caller must then exit the queue loop),
 *       true once all waits are satisfied. A trailing __syncthreads() publishes the acquired data
 *       block-wide (acquire applies after the barrier). A plain non-volatile load is FORBIDDEN.
 * =========================================================================================== */
#ifndef AMK_SYNC_CUH
#define AMK_SYNC_CUH

#include "amk_vm.cuh"

/* Backoff schedule: short busy window for the already-satisfied fast path, then exponential
 * __nanosleep from ~32ns capped at a few microseconds. 82 hot-spinning blocks would otherwise
 * starve the producers of HBM bandwidth (the whole point of the megakernel is to BE bandwidth
 * bound), so this is required, not optional. */
#define AMK_SPIN_BUSY_ITERS   64u      /* busy-poll this many times before sleeping at all      */
#define AMK_BACKOFF_MIN_NS    32u      /* first sleep                                           */
#define AMK_BACKOFF_MAX_NS    4096u    /* cap (~4us)                                            */
#define AMK_ABORT_POLL_MASK   0x3Fu    /* poll abort_flag every 64 spins (cheap, still prompt)  */

/* ---- non-hoistable acquire load of a counter ------------------------------------------------
 * atomicAdd(addr, 0) is a read-modify-write that returns the current value with .gpu acquire
 * semantics and CANNOT be hoisted out of the loop or served from a stale register/L1 line, which
 * a plain `counters[c]` load could be (=> spin forever). This is the abi.h-blessed primitive. */
__device__ __forceinline__ uint32_t amk_acquire_load(amk_counter_t* c) {
    return atomicAdd(c, 0u);
}

/* ---- the watchdog escape: read abort_flag without it being optimized into a constant --------- */
__device__ __forceinline__ bool amk_aborted(const amk_device_program& prog) {
    /* volatile load of the device int, the host (or a sibling block) may set it asynchronously. */
    return *((volatile int*)prog.abort_flag) != 0;
}

/* ===========================================================================================
 * amk_signal, retire an instruction: release-fence its outputs, then ++ its out_counter.
 * Whole block calls this; only thread 0 performs the atomic so the counter advances by exactly 1
 * per instruction (the abi.h "producers only ++ by 1" invariant).
 * =========================================================================================== */
__device__ __forceinline__ void amk_signal(amk_device_program& prog, int32_t counter_id) {
    __syncthreads();                       /* every thread's output stores are issued first      */
    if (threadIdx.x == 0) {
        __threadfence();                   /* DEVICE-scope release: order ALL output stores < ++  */
        atomicAdd(&prog.counters[counter_id], 1u);
    }
    __syncthreads();                       /* block agrees the instruction is retired             */
}

/* ===========================================================================================
 * amk_wait_all, block until every (counter,threshold) precondition holds. Returns false if the
 * abort_flag fired (the caller must stop processing its queue and head for the grid-exit barrier).
 * =========================================================================================== */
__device__ __forceinline__ bool amk_wait_all(amk_device_program& prog,
                                             const amk_instruction_t& inst) {
    /* Shared so the spin result (satisfied / aborted) is broadcast to the whole block in one
     * barrier instead of every thread hammering global memory. */
    __shared__ int s_ready;     /* 1 once all waits satisfied                                    */
    __shared__ int s_abort;     /* 1 if the watchdog fired while waiting                          */

    if (inst.n_waits == 0) {
        return true;            /* no preconditions: the fast path costs nothing                 */
    }

    if (threadIdx.x == 0) {
        s_ready = 0;
        s_abort = 0;
        uint32_t spins   = 0;
        uint32_t backoff = AMK_BACKOFF_MIN_NS;
        for (;;) {
            /* check every wait with a fresh acquire load each pass */
            bool all_ok = true;
            #pragma unroll
            for (int w = 0; w < AMK_MAX_WAITS; ++w) {
                if (w >= inst.n_waits) break;
                uint32_t cur = amk_acquire_load(&prog.counters[inst.wait_counter[w]]);
                if (cur < (uint32_t)inst.wait_threshold[w]) { all_ok = false; break; }
            }
            if (all_ok) { s_ready = 1; break; }

            /* watchdog escape, poll cheaply but promptly */
            if ((spins & AMK_ABORT_POLL_MASK) == 0u && amk_aborted(prog)) { s_abort = 1; break; }

            /* backoff: busy for the first AMK_SPIN_BUSY_ITERS, then exponential __nanosleep */
            ++spins;
            if (spins > AMK_SPIN_BUSY_ITERS) {
                __nanosleep(backoff);
                backoff = backoff < AMK_BACKOFF_MAX_NS ? (backoff << 1) : AMK_BACKOFF_MAX_NS;
            }
        }
    }
    __syncthreads();            /* publish s_ready/s_abort AND the acquired data block-wide       */
    return s_abort == 0;
}

#endif /* AMK_SYNC_CUH */
