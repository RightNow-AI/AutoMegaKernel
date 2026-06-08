/* ===========================================================================================
 * AutoMegaKernel, THE PAGE / SCRATCH MODEL (Layer 0)
 * ===========================================================================================
 * v1 page model: every buffer's device pointer is RESOLVED BY THE HOST at load time and stored in
 * amk_buffer_t.ptr (abi.h says: "Host resolves every IR buffer id to a device pointer + LAYOUT
 * before launch ... activations point into the scratch arena at their page's offset"). So on the
 * device the resolver is trivial, read buffers[id].ptr. We keep activations in GLOBAL_SCRATCH for
 * v1 (correctness first); SMEM paging of hot activations is a documented TODO (the megakernel
 * bandwidth win), not required for conformance against the reference VM.
 *
 * This header also provides the small typed element load/store helpers (fp32 + fp16 paths) the
 * instructions use, and row-major tile addressing via amk_buffer_t.shape/stride exactly as abi.h
 * specifies ("base + m_off*stride[0] + n_off*stride[1]").
 * =========================================================================================== */
#ifndef AMK_PAGES_CUH
#define AMK_PAGES_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "amk_vm.cuh"

/* ---- resolve a buffer id to its (already host-offset) device base pointer -------------------- */
__device__ __forceinline__ const amk_buffer_t& amk_buf(const amk_device_program& prog, int32_t id) {
    return prog.buffers[id];
}
__device__ __forceinline__ void* amk_ptr(const amk_device_program& prog, int32_t id) {
    return prog.buffers[id].ptr;   /* HBM weight/IO/KV, or scratch-arena offset for activations */
}

/* ---- typed scalar load: read element #i of a buffer as fp32, honoring its dtype ---------------
 * The reference accumulates in fp32 then casts to the output dtype; we match that by reading every
 * operand as fp32 here regardless of storage dtype (fp32 / fp16 / bf16 covered; others fall through
 * to a fp32 reinterpret which the v1 acceptance programs never hit). Real Llama weights are bf16,
 * so the bf16 path (load -> fp32 compute) is a first-class case, not a fallback. */
__device__ __forceinline__ float amk_load_f(const amk_buffer_t& b, int64_t i) {
    switch (b.dtype) {
        case AMK_F16:  return __half2float(((const __half*)b.ptr)[i]);
        case AMK_BF16: return __bfloat162float(((const __nv_bfloat16*)b.ptr)[i]);
        case AMK_F32:
        default:       return ((const float*)b.ptr)[i];
    }
}

/* ---- typed scalar load as int64: read element #i of an integer buffer (ids / positions) -------
 * EMBED ids and ROPE/KV positions are I32 (schedule.lower emits DType.I32). We also tolerate I8 /
 * F32 / F16 storage by truncating toward zero, mirroring torch's ``.to(torch.long)`` on the
 * reference path (which only ever sees integer-valued tensors here). */
__device__ __forceinline__ long amk_load_i(const amk_buffer_t& b, int64_t i) {
    switch (b.dtype) {
        case AMK_I32: return (long)((const int32_t*)b.ptr)[i];
        case AMK_I8:  return (long)((const int8_t*)b.ptr)[i];
        case AMK_U8:  return (long)((const uint8_t*)b.ptr)[i];
        case AMK_F16: return (long)__half2float(((const __half*)b.ptr)[i]);
        case AMK_BF16:return (long)__bfloat162float(((const __nv_bfloat16*)b.ptr)[i]);
        case AMK_F32:
        default:      return (long)((const float*)b.ptr)[i];
    }
}

/* ---- typed scalar store: write fp32 value into element #i of a buffer in its storage dtype ----- */
__device__ __forceinline__ void amk_store_f(const amk_buffer_t& b, int64_t i, float v) {
    switch (b.dtype) {
        case AMK_F16:  ((__half*)b.ptr)[i] = __float2half(v); break;
        case AMK_BF16: ((__nv_bfloat16*)b.ptr)[i] = __float2bfloat16(v); break;
        case AMK_F32:
        default:       ((float*)b.ptr)[i] = v; break;
    }
}

/* ---- row-major linear offset of element (r,c) of a rank>=2 buffer using its element strides ----
 * abi.h: tile addressing reads buffer.stride (in elements), NOT params. For the decode programs
 * the leading dim is M=1 so r is usually 0, but we honor stride[] generally. */
__device__ __forceinline__ int64_t amk_off2(const amk_buffer_t& b, int64_t r, int64_t c) {
    return r * b.stride[0] + c * b.stride[1];
}

/* ---- element size (bytes) for a buffer's storage dtype -------------------------------------- */
__device__ __forceinline__ int amk_dtype_bytes(int32_t dtype) {
    switch (dtype) {
        case AMK_F32: case AMK_I32:                return 4;
        case AMK_F16: case AMK_BF16:               return 2;
        case AMK_F8E4M3: case AMK_F8E5M2:
        case AMK_I8: case AMK_U8: case AMK_BOOL:   return 1;
        default:                                    return 4;
    }
}

/* ---- int4 / int8 WEIGHT-ONLY QUANTIZATION unpack helpers -------------------------------------
 * Storage convention (frozen, matches schedule/quantize.py + instructions/reference.py):
 *   I8 : signed int8 weight, row-major [N, K]. real = q * scale[n, k/group].
 *   I4 : uint8 packed [N, K//2]; byte b of a row holds column 2b in its LOW nibble and 2b+1 in its
 *        HIGH nibble (little-endian within the byte). Symmetric int4 stores q+8 in [0,15]; we
 *        subtract 8 to recover q in [-8,7]. real = (q - zero) * scale (zero=0 if symmetric).
 *
 * These return the SIGNED integer level q (pre-scale); the GEMV multiplies by the group scale and
 * (optionally) subtracts the per-group zero-point before scaling, in fp32, matching the reference. */

/* Signed int4 level of column k in packed row base `row_bytes` (uint8*). Symmetric (q+8) decode. */
__device__ __forceinline__ int amk_unpack_i4_sym(const uint8_t* __restrict__ row_bytes, int k) {
    const uint8_t byte = row_bytes[k >> 1];
    const int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);   /* odd col -> high nibble */
    return nib - 8;                                          /* [0,15] -> [-8,7] */
}

/* Unsigned int4 nibble (asymmetric: a per-group zero-point is subtracted by the caller). */
__device__ __forceinline__ int amk_unpack_i4_u(const uint8_t* __restrict__ row_bytes, int k) {
    const uint8_t byte = row_bytes[k >> 1];
    return (k & 1) ? (byte >> 4) : (byte & 0xF);
}

/* ---- software prefetch of a byte range of a buffer into L2 (compute-capability agnostic) ------
 * Walks the [byte_off, byte_off+nbytes) span of a buffer's backing storage and issues a non-binding
 * L2 prefetch per cache line, striped across the block's threads. This hides the inter-op HBM
 * latency for the NEXT GEMV tile's weight rows while the current instruction still computes. It is
 * a pure hint, never changes results, safe on every arch (prefetch.global.L2 lowers to a no-op
 * where unsupported). Acted on only when ScheduleConfig.pipelining_depth > 0 (see scheduler.cu). */
__device__ __forceinline__ void amk_prefetch_l2(const void* base, int64_t byte_off, int64_t nbytes) {
    const char* p = (const char*)base + byte_off;
    const int64_t LINE = 128;                       /* HBM/L2 cache-line granularity */
    /* one thread per cache line, strided over the block */
    for (int64_t off = (int64_t)threadIdx.x * LINE; off < nbytes; off += (int64_t)blockDim.x * LINE) {
#if (__CUDA_ARCH__ >= 800)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(p + off));
#else
        /* pre-sm_80: no prefetch PTX, touch the line with a volatile load to warm L2.
         * (Discarded read; never affects correctness.) */
        volatile const char dummy = *(const volatile char*)(p + off);
        (void)dummy;
#endif
    }
}

#endif /* AMK_PAGES_CUH */
