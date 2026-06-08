/* ===========================================================================================
 * AutoMegaKernel, DEVICE-SIDE VM TYPES (Layer 0)
 * ===========================================================================================
 * This header mirrors the FROZEN POD records in vm/abi.h into the device translation units, and
 * adds the small amount of glue the persistent kernel needs (a flattened sm_queue, the resolved
 * amk_program_t on device).
 *
 * IMPORTANT: nothing here weakens the abi.h contract. amk_params_t / amk_buffer_t /
 * amk_instruction_t / amk_program_t are exactly the abi.h structs (we #include it). The host
 * (vm/loader.py) memcpys byte-identical POD arrays onto the device; this header is how the kernel
 * reads them back.
 *
 * Multi-TU note: every device function in the VM is defined `__device__ __forceinline__` inside a
 * .cuh so each translation unit (sync.cu / pages.cu / scheduler.cu) gets its own private copy -
 * no duplicate-symbol link errors when torch.utils.cpp_extension compiles the file list together,
 * and the optimizer fully inlines them into the single persistent kernel in scheduler.cu.
 * =========================================================================================== */
#ifndef AMK_VM_CUH
#define AMK_VM_CUH

#include <cstdint>
#include "abi.h"   /* the frozen ABI: amk_dtype_t, amk_params_t, amk_buffer_t, amk_instruction_t,
                    * amk_program_t, opcode/dtype/memspace enums, AMK_MAX_* capacities. */

/* ----------------------------------------------------------------------------------------------
 * Device view of the program.
 *
 * abi.h's amk_program_t uses `const int32_t* const* sm_queue` (an array of per-SM pointer rows).
 * Passing a host array of device pointers is awkward across the cpp_extension boundary, so the
 * loader instead flattens the per-SM queues into ONE contiguous int32 array plus an offset table:
 *
 *     sm_queue_flat[ sm_queue_off[s] + k ]  == k-th instruction index of SM s
 *     sm_queue_len[s]                       == number of instructions for SM s
 *
 * Everything else is a straight copy of the abi.h fields. The kernel only ever reads from this.
 * -------------------------------------------------------------------------------------------- */
struct amk_device_program {
    const amk_buffer_t*      buffers;        /* [n_buffers]                        (device) */
    int32_t                  n_buffers;
    amk_counter_t*           counters;       /* [n_counters], host-zeroed pre-launch (device) */
    int32_t                  n_counters;
    const amk_instruction_t* instructions;   /* [n_instructions]                   (device) */
    int32_t                  n_instructions;
    const int32_t*           sm_queue_flat;  /* flattened per-SM instruction-index queues (device) */
    const int32_t*           sm_queue_off;   /* [num_sms] start offset into sm_queue_flat (device) */
    const int32_t*           sm_queue_len;   /* [num_sms] length of each SM's queue   (device) */
    int32_t                  num_sms;        /* == gridDim.x of the cooperative launch */
    unsigned char*           scratch;        /* base of GLOBAL_SCRATCH arena             (device) */
    int64_t                  scratch_bytes;
    int32_t*                 abort_flag;     /* grid-polled watchdog: !=0 -> all blocks exit  */
};

#endif /* AMK_VM_CUH */
