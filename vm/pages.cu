/* ===========================================================================================
 * AutoMegaKernel, pages.cu
 * ===========================================================================================
 * Translation unit home of the page/scratch model. As with sync.cu, the helpers live in pages.cuh
 * as `__device__ __forceinline__` (buffer-id -> device pointer resolution, typed fp32/fp16
 * load/store, row-major stride addressing) so they inline into the persistent kernel and the
 * cpp_extension multi-file build links cleanly.
 *
 * v1 page model: activations live in the GLOBAL_SCRATCH arena; the HOST resolves every buffer id to
 * a device pointer (already offset for paged activations) at load time, so the device resolver is a
 * direct read of amk_buffer_t.ptr. SMEM paging of hot activations (the megakernel bandwidth win) is
 * a documented TODO, correctness against the reference VM comes first.
 * =========================================================================================== */
#include "pages.cuh"

__device__ int amk_pages_tu_anchor = 0;
