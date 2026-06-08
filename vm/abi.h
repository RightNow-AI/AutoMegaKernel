/* ===========================================================================================
 * AutoMegaKernel, THE FROZEN INSTRUCTION ABI
 * ===========================================================================================
 * Hard contract between three parties:
 *   (1) the megakernel VM (vm/scheduler.cu, pages.cu, sync.cu), the trusted runtime,
 *   (2) every ABI-conformant micro-kernel (instructions/cuda/*.cu) , the Layer-1 ops,
 *   (3) the host-side schedule loader (driven by schedule/ir.py)    , fills the tables.
 *
 * Numeric enum codes AND the capacity/version constants here are CANONICAL and MUST match
 * schedule/ir.py (DType, MemSpace, InstructionKind, ABI_MAX_*, ABI_VERSION). tests/test_abi_sync.py
 * parses this header and the IR module and fails the build if they ever drift.
 *
 * DESIGN INVARIANTS (do not weaken):
 *   - An *instruction* is PURE COMPUTE: read inputs, compute, write outputs. NO synchronization,
 *     NO global side effects beyond its declared output buffers.
 *   - The *VM* owns synchronization: before dispatching it waits on the instruction's input
 *     counters reaching their static thresholds; after compute it issues a release fence that
 *     orders ALL output-buffer stores, then increments the single output counter by 1. So a
 *     counter reaching its threshold means "every output of every contributing producer is
 *     written and visible."
 *   - Counters are monotonic (producers only ++; consumers only wait on static thresholds).
 *     A counter with >1 producer is an ALL-JOIN (every consumer waits threshold == #producers);
 *     partial waits on a shared counter are a which-producer race and are rejected by the IR.
 *   - Per-SM instruction queues MUST be emitted in an order consistent with a GLOBAL topological
 *     sort, so no SM blocks on a counter only its own later queue entry could signal.
 *   - DECODE MODEL: one kernel launch == one forward pass == one decoded token. Counters are
 *     host-memset to zero before each launch; KV_CACHE persists in HBM across launches; the host
 *     drives the autoregressive loop. (This also keeps each launch under the Windows WDDM TDR
 *     ~2s watchdog on the laptop dev GPU, see the launch contract below.)
 * =========================================================================================== */
#ifndef AMK_ABI_H
#define AMK_ABI_H

#include <stdint.h>

#define AMK_ABI_VERSION_MAJOR 0
#define AMK_ABI_VERSION_MINOR 2   /* keep in sync with schedule.ir.ABI_VERSION = "0.2" */

/* ---- Fixed capacities, MUST equal schedule.ir.ABI_MAX_* (test_abi_sync enforces) ------- */
#define AMK_MAX_INPUTS   8
#define AMK_MAX_OUTPUTS  4
#define AMK_MAX_WAITS    8
#define AMK_MAX_RANK     4   /* covers attention's [head, seq, dim] + batched GEMM */

/* ---- Element types, codes MUST match schedule.ir.DType ---------------------------------- */
typedef enum {
    AMK_F32 = 0, AMK_F16 = 1, AMK_BF16 = 2, AMK_F8E4M3 = 3, AMK_F8E5M2 = 4,
    AMK_I32 = 5, AMK_I8 = 6, AMK_I4 = 7 /* packed 4-bit, two per byte */, AMK_U8 = 8, AMK_BOOL = 9
} amk_dtype_t;

/* ---- Memory spaces, codes MUST match schedule.ir.MemSpace ------------------------------- */
typedef enum {
    AMK_HBM = 0, AMK_GLOBAL_SCRATCH = 1, AMK_SMEM = 2, AMK_REGISTER = 3
} amk_memspace_t;

/* ---- Opcodes, codes MUST match schedule.ir.InstructionKind ------------------------------ */
typedef enum {
    AMK_OP_NOP            = 0,
    AMK_OP_COPY           = 1,
    AMK_OP_EMBED          = 2,
    AMK_OP_RMSNORM        = 3,
    AMK_OP_LAYERNORM      = 4,
    AMK_OP_GEMV_TILE      = 5,
    AMK_OP_GEMM_TILE      = 6,
    AMK_OP_ATTENTION_TILE = 7,
    AMK_OP_ROPE           = 8,
    AMK_OP_SILU_MUL       = 9,
    AMK_OP_GELU           = 10,
    AMK_OP_ADD            = 11,
    AMK_OP_MUL            = 12,
    AMK_OP_DEQUANT        = 13,
    AMK_OP_SOFTMAX        = 14,
    AMK_OP_ALLREDUCE_SHARD= 15,
    AMK_OP_KV_APPEND      = 16,
    AMK_OP_SAMPLE_ARGMAX  = 17,
    AMK_OP_ATTENTION_COMBINE = 18,   /* merge per-KV-block (out,m,l) partials (flash) */
    AMK_OP__COUNT
} amk_opcode_t;

/* ---- Counter type ------------------------------------------------------------------------
 * One 32-bit unsigned counter per sync point, resident in global memory, host-memset to zero
 * before EACH launch (one launch == one forward pass). uint32 is ample: a counter only needs to
 * reach max #producers within one pass. See the SYNC CONTRACT below for the exact primitives. */
typedef uint32_t amk_counter_t;

/* ---- Params blob, op-specific scalars. Append fields at the end; never reorder. ---------
 * Each opcode reads only the fields it documents in OP_REGISTRY (schedule/ir.py). int fields are
 * int32 (PARAM_FIELDS 'i'); float fields are 'f'. */
typedef struct {
    int32_t K, N, M;            /* full GEMM/GEMV dims */
    int32_t N_tile, M_tile;     /* output tile extent this task computes */
    int32_t n_off, m_off;       /* tile offset into the output */
    int32_t hidden, vocab;      /* model dims (embed / norm / sample) */
    int32_t head_dim, n_heads, n_kv_heads;
    int32_t kv_start, kv_len, pos;   /* attention / kv-cache window */
    int32_t qdtype;             /* amk_dtype_t of quantized operand for DEQUANT/GEMV */
    int32_t group;              /* quant group size */
    int32_t flags;              /* bitfield: bit0=causal, bit1=transpose_b, ... */
    int32_t dim;                /* generic reduction axis (softmax) */
    float   eps, scale, theta;  /* rmsnorm eps, attn softmax scale, rope base */
} amk_params_t;

/* ---- Buffer table -----------------------------------------------------------------------
 * Host resolves every IR buffer id to a device pointer + LAYOUT before launch: weights point
 * into the model's HBM tensors; activations point into the scratch arena at their page's offset.
 * shape/stride (row-major, in ELEMENTS, int64 to survive vocab*hidden) let a tiled instruction
 * address element (row,col) of its tile as base + m_off*stride[0] + n_off*stride[1]. The VM
 * never allocates HBM mid-flight; all pointers are fixed at load. */
typedef struct {
    void*    ptr;                 /* device pointer (already offset for paged activations) */
    int64_t  numel;
    int32_t  rank;
    int32_t  dtype;               /* amk_dtype_t */
    int32_t  space;               /* amk_memspace_t */
    int32_t  _pad;
    int64_t  shape [AMK_MAX_RANK];
    int64_t  stride[AMK_MAX_RANK]; /* element strides; tile addressing reads these, not params */
} amk_buffer_t;

/* ---- Instruction record (POD, fixed size), 1:1 with schedule.ir.Task ------------------- */
typedef struct {
    int32_t      op;                            /* amk_opcode_t */
    int32_t      n_inputs, n_outputs, n_waits;
    int32_t      inputs [AMK_MAX_INPUTS];       /* buffer ids read */
    int32_t      outputs[AMK_MAX_OUTPUTS];      /* buffer ids written */
    int32_t      wait_counter  [AMK_MAX_WAITS]; /* counter ids to wait on */
    int32_t      wait_threshold[AMK_MAX_WAITS]; /* static thresholds (>=1, ==#producers if shared) */
    int32_t      out_counter;                   /* the single counter incremented on completion */
    int32_t      sm;                            /* assigned SM/worker (>=0; loader rejects <0) */
    amk_params_t params;
} amk_instruction_t;

/* ---- The program loaded into the VM ----------------------------------------------------- */
typedef struct {
    const amk_buffer_t*      buffers;       /* [n_buffers] */
    int32_t                  n_buffers;
    amk_counter_t*           counters;      /* [n_counters], host-memset 0 before launch */
    int32_t                  n_counters;
    const amk_instruction_t* instructions;  /* [n_instructions] */
    int32_t                  n_instructions;
    const int32_t* const*    sm_queue;      /* [num_sms][...] instruction indices, topo order */
    const int32_t*           sm_queue_len;  /* [num_sms] */
    int32_t                  num_sms;
    void*                    scratch;       /* base of the global scratch arena (pages.cu) */
    int64_t                  scratch_bytes;
    int32_t*                 abort_flag;    /* grid-polled watchdog: set !=0 -> all SMs exit cleanly */
} amk_program_t;

/* =========================================================================================
 * THE LAUNCH CONTRACT (Layer 0), the VM kernel
 * -----------------------------------------------------------------------------------------
 *   - MUST be launched with cudaLaunchCooperativeKernel so all blocks are co-resident and make
 *     forward progress (a persistent block that is never scheduled = guaranteed deadlock).
 *   - gridDim MUST equal min(num_sms, cudaOccupancyMaxActiveBlocksPerMultiprocessor(kernel,
 *     threads_per_block, smem) * num_sms). The loader computes occupancy from the COMPILED kernel
 *     and REFUSES to launch if the requested grid exceeds the cooperative co-residency limit.
 *     One block per SM is the target; if occupancy>1 the loader either caps gridDim or drops the
 *     1:1-SM claim, it never assumes co-residency it did not verify.
 *   - Dynamic SMEM opt-in (cudaFuncAttributeMaxDynamicSharedMemorySize) MUST be <=
 *     GpuTarget.smem_bytes_per_block_optin (NOT smem_bytes_per_sm). Reject otherwise.
 *   - Kernel entry: grid.sync() after each block caches its queue pointer and the host-zeroed
 *     counters are confirmed visible. Kernel exit: grid.sync(). (Available via cooperative groups.)
 *   - WINDOWS WDDM TDR: the dev GPU is a display GPU with a ~2s OS watchdog (GpuTarget.wddm_tdr).
 *     Keep each launch < TDR by the one-launch-per-token model; raise HKLM TdrDelay for dev; the
 *     host treats cudaErrorLaunchTimeout as a distinct TIMEOUT (not a clean REJECTED).
 *
 * THE SYNC CONTRACT (frozen primitives, prose is NOT enough for a cross-SM contract)
 * -----------------------------------------------------------------------------------------
 *   PRIMITIVES: use proper acquire/release ordering, NOT a bare legacy atomicAdd with no fence.
 *   The device-scope (.gpu/.gl) is required for cross-SM (inter-CTA) visibility; block scope is
 *   INSUFFICIENT. There are TWO sanctioned, EQUIVALENT spellings of the same release/acquire
 *   contract on the supported archs (sm_80 / sm_90 / sm_120):
 *
 *     (A) CANONICAL on supported archs, fence + legacy device-scope atomic (what vm/sync.cuh
 *         implements). __threadfence() is the device-scope release fence; an unscoped CUDA
 *         atomicAdd is a device-scope RMW; and atomicAdd(c,0) is a non-hoistable device-scope
 *         acquire-RMW load. The fence supplies the release ordering and the RMW supplies the
 *         acquire ordering, so together they are a classically-correct release/acquire pair on
 *         sm_80/90/120, this is the BLESSED primitive the VM ships.
 *     (B) EQUIVALENT, cuda::atomic_ref<uint32_t, cuda::thread_scope_device> with
 *         fetch_add(release)/load(acquire), or PTX atom.add.release.gpu / ld.acquire.gpu.u32.
 *         Same memory model; use it on archs/toolchains where it lowers as well or better.
 *
 *   signal(c):  thread 0 of the block release-publishes the increment AFTER a device-scope fence
 *               that orders all output-buffer stores. CANONICAL (A) form, as in vm/sync.cuh:
 *                   __threadfence();                                  // device-scope release fence
 *                   atomicAdd(&prog.counters[c], 1u);                 // device-scope RMW (++ by 1)
 *               (EQUIVALENT (B): atomic_ref{prog.counters[c]}.fetch_add(1, release);)
 *               then __syncthreads() so the whole block agrees the instruction is retired.
 *               (Cross-GPU counters for ALLREDUCE_SHARD use system scope / __threadfence_system().)
 *   wait(c,t):  thread 0 spins on an ACQUIRE load (never a plain/hoistable load). CANONICAL (A)
 *               form, as in vm/sync.cuh, uses atomicAdd(c,0) as the non-hoistable acquire-RMW:
 *                   bool ab=false;
 *                   while (atomicAdd(&prog.counters[c], 0u) < t) {     // device-scope acquire-RMW
 *                       if (*(volatile int*)prog.abort_flag) { ab=true; break; }
 *                       backoff();
 *                   }
 *               (EQUIVALENT (B): atomic_ref{prog.counters[c]}.load(acquire).)
 *               then publish `ab` to shared memory and __syncthreads() so EVERY thread takes the
 *               same path, return-from-only-thread-0 would hang the block at the next barrier.
 *               amk_wait_all returns false (block must exit) iff aborted.
 *   backoff():  exponential __nanosleep starting ~32ns capped at a few microseconds, after a
 *               short initial busy window for the already-satisfied fast path. Required, not
 *               optional, 82 hot-spinning blocks would otherwise starve producers of bandwidth.
 *
 * THE INSTRUCTION ABI (Layer 1), every micro-kernel is exactly this device function, nothing more:
 *     __device__ void amk_inst_<name>(const amk_program_t& prog, const amk_instruction_t& inst);
 *   - reads prog.buffers[inst.inputs[i]] for i in [0,n_inputs); writes outputs[j] for j in
 *     [0,n_outputs); reads scalars from inst.params; reads shape/stride from the buffer records.
 *   - MUST NOT touch counters, MUST NOT read/write any other buffer, MUST NOT launch work.
 *   - The whole threadblock cooperates on one instruction; __syncthreads() within is fine.
 * ========================================================================================= */
#ifdef __CUDACC__
__device__ void amk_signal(amk_program_t& prog, int32_t counter_id);
__device__ bool amk_wait_all(const amk_program_t& prog, const amk_instruction_t& inst); /* false if aborted */
__device__ void amk_dispatch(amk_program_t& prog, const amk_instruction_t& inst);
#endif

#endif /* AMK_ABI_H */
