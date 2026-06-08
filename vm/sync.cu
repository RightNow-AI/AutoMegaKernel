/* ===========================================================================================
 * AutoMegaKernel, sync.cu
 * ===========================================================================================
 * Translation unit home of the canonical sync primitives. The definitions live in sync.cuh as
 * `__device__ __forceinline__` so the persistent kernel in scheduler.cu inlines them and so each
 * TU in the cpp_extension file list compiles cleanly without duplicate-symbol link errors.
 *
 * This file exists to satisfy the abi.h module layout ("vm/sync.cu : the canonical sync
 * primitives") and to be independently compilable; it pulls in the header so a standalone build of
 * this TU still type-checks the primitives. See sync.cuh for the full SYNC CONTRACT implementation
 * (amk_signal: device-scope __threadfence then atomicAdd + __syncthreads; amk_wait_all:
 * non-hoistable acquire spin via atomicAdd(&c,0), exponential __nanosleep backoff, abort poll).
 * =========================================================================================== */
#include "sync.cuh"

/* No additional definitions: sync.cuh is header-complete. A dummy device symbol keeps the TU
 * non-empty for toolchains that warn on empty .cu objects. */
__device__ int amk_sync_tu_anchor = 0;
