/* ===========================================================================================
 * AutoMegaKernel, HOST LAUNCHER (pybind11 / torch extension)
 * ===========================================================================================
 * The host entry the Python loader (vm/loader.py) calls. It receives the device pointers the
 * loader already resolved (buffer table, counters, instruction table, flattened per-SM queues,
 * scratch arena, abort flag) as integer addresses, assembles the amk_device_program POD, and
 * launches the persistent kernel via cudaLaunchCooperativeKernel after honoring the LAUNCH
 * CONTRACT in abi.h:
 *
 *   - set cudaFuncAttributeMaxDynamicSharedMemorySize if dynamic SMEM is requested
 *     (must be <= the per-block opt-in cap, the loader passes the value and we pass it on);
 *   - compute cudaOccupancyMaxActiveBlocksPerMultiprocessor for the COMPILED kernel and ASSERT
 *     gridDim fits the cooperative co-residency limit (num_sms * occupancy); REFUSE otherwise;
 *   - the loader host-memsets the counters to 0 before calling (one launch == one pass);
 *   - block until the launch finishes, then surface cudaErrorLaunchTimeout DISTINCTLY (Windows
 *     WDDM TDR) vs any other error.
 *
 * The function returns a small status dict so Python can branch on TIMEOUT vs OK vs REJECTED.
 * =========================================================================================== */
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include <string>
#include <stdexcept>

#include "amk_vm.cuh"

/* the kernel, defined in scheduler.cu */
extern "C" __global__ void amk_megakernel(amk_device_program prog);

/* helper: turn an int64 address from Python (tensor.data_ptr()) into a typed device pointer */
template <typename T>
static inline T* as_dptr(int64_t addr) { return reinterpret_cast<T*>(addr); }

/* Query how many blocks of this kernel co-reside per SM at (threads, dyn_smem). This is the hard
 * ceiling cudaLaunchCooperativeKernel enforces: gridDim must be <= occupancy * num_sms or the
 * launch deadlocks / errors. */
int64_t amk_max_coresident_blocks(int64_t threads_per_block, int64_t dyn_smem_bytes) {
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    if (dyn_smem_bytes > 0) {
        /* opt into the larger dynamic SMEM partition (abi.h: must be <= per-block opt-in cap;
         * the loader validates against GpuTarget.smem_bytes_per_block_optin before calling). */
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            (const void*)amk_megakernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)dyn_smem_bytes));
    }

    int blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, (const void*)amk_megakernel,
        (int)threads_per_block, (size_t)dyn_smem_bytes));
    return (int64_t)blocks_per_sm * (int64_t)prop.multiProcessorCount;
}

/* Report the COMPILED kernel's static resource footprint + occupancy at a launch config. This is
 * the honest instrument for the occupancy experiment: numRegs is the per-thread register frame
 * that determines how many blocks co-reside per SM. blocks_per_sm is the measured occupancy
 * (cudaOccupancyMaxActiveBlocksPerMultiprocessor) at (threads, dyn_smem), the number the
 * cooperative grid is capped to. No fabricated numbers: both come straight from the driver. */
py::dict amk_kernel_attributes(int64_t threads_per_block, int64_t dyn_smem_bytes) {
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    if (dyn_smem_bytes > 0) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            (const void*)amk_megakernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_bytes));
    }
    cudaFuncAttributes attr{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, (const void*)amk_megakernel));
    int blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, (const void*)amk_megakernel,
        (int)threads_per_block, (size_t)dyn_smem_bytes));
    py::dict d;
    d["num_regs"]            = (int64_t)attr.numRegs;
    d["static_smem_bytes"]   = (int64_t)attr.sharedSizeBytes;
    d["local_bytes"]         = (int64_t)attr.localSizeBytes;
    d["max_threads"]         = (int64_t)attr.maxThreadsPerBlock;
    d["const_bytes"]         = (int64_t)attr.constSizeBytes;
    d["blocks_per_sm"]       = (int64_t)blocks_per_sm;
    d["num_sms"]             = (int64_t)prop.multiProcessorCount;
    d["max_grid"]            = (int64_t)blocks_per_sm * (int64_t)prop.multiProcessorCount;
    return d;
}

/* Does this device support cooperative launch at all? */
bool amk_supports_cooperative() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    int coop = 0;
    if (cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev) != cudaSuccess)
        return false;
    return coop != 0;
}

/* Launch the persistent VM. All pointer args are device addresses (int64) the loader resolved.
 * Returns: {"status": "OK"|"TIMEOUT"|"ERROR", "error": "<cuda msg>"} . Raises only on a genuine
 * pre-launch contract violation (so Python sees a clean REJECTED). */
py::dict amk_launch(int64_t buffers_ptr,        int64_t n_buffers,
                    int64_t counters_ptr,       int64_t n_counters,
                    int64_t instructions_ptr,   int64_t n_instructions,
                    int64_t sm_queue_flat_ptr,
                    int64_t sm_queue_off_ptr,
                    int64_t sm_queue_len_ptr,
                    int64_t num_sms,
                    int64_t scratch_ptr,        int64_t scratch_bytes,
                    int64_t abort_flag_ptr,
                    int64_t grid_dim,           int64_t threads_per_block,
                    int64_t dyn_smem_bytes) {
    py::dict result;

    if (!amk_supports_cooperative()) {
        throw std::runtime_error("AMK: device does not support cudaLaunchCooperativeKernel");
    }

    /* opt into dynamic SMEM (abi.h: <= per-block opt-in cap; loader pre-validated the size) */
    if (dyn_smem_bytes > 0) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            (const void*)amk_megakernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)dyn_smem_bytes));
    }

    /* prove co-residency: gridDim MUST fit occupancy*num_sms or the cooperative launch deadlocks.
     * We REFUSE (raise) rather than silently exceed co-residency. */
    int blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, (const void*)amk_megakernel,
        (int)threads_per_block, (size_t)dyn_smem_bytes));
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int64_t max_grid = (int64_t)blocks_per_sm * (int64_t)prop.multiProcessorCount;
    if (blocks_per_sm <= 0) {
        throw std::runtime_error(
            "AMK: kernel has ZERO occupancy at the requested launch config "
            "(threads/smem too large), cooperative launch impossible");
    }
    if (grid_dim > max_grid) {
        throw std::runtime_error(
            "AMK: requested gridDim=" + std::to_string(grid_dim) +
            " exceeds cooperative co-residency limit " + std::to_string(max_grid) +
            " (blocks_per_sm=" + std::to_string(blocks_per_sm) +
            " * SMs=" + std::to_string(prop.multiProcessorCount) +
            "), REFUSING to launch (would deadlock)");
    }

    /* assemble the device program POD */
    amk_device_program prog{};
    prog.buffers       = as_dptr<const amk_buffer_t>(buffers_ptr);
    prog.n_buffers     = (int32_t)n_buffers;
    prog.counters      = as_dptr<amk_counter_t>(counters_ptr);
    prog.n_counters    = (int32_t)n_counters;
    prog.instructions  = as_dptr<const amk_instruction_t>(instructions_ptr);
    prog.n_instructions= (int32_t)n_instructions;
    prog.sm_queue_flat = as_dptr<const int32_t>(sm_queue_flat_ptr);
    prog.sm_queue_off  = as_dptr<const int32_t>(sm_queue_off_ptr);
    prog.sm_queue_len  = as_dptr<const int32_t>(sm_queue_len_ptr);
    prog.num_sms       = (int32_t)num_sms;
    prog.scratch       = as_dptr<unsigned char>(scratch_ptr);
    prog.scratch_bytes = scratch_bytes;
    prog.abort_flag    = as_dptr<int32_t>(abort_flag_ptr);

    /* cooperative launch (grid.sync requires it) */
    dim3 grid((unsigned)grid_dim), block((unsigned)threads_per_block);
    void* kargs[] = { (void*)&prog };
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    cudaError_t launch_err = cudaLaunchCooperativeKernel(
        (const void*)amk_megakernel, grid, block, kargs,
        (size_t)dyn_smem_bytes, stream);

    if (launch_err != cudaSuccess) {
        /* clear the sticky error so the next call starts clean */
        cudaGetLastError();
        result["status"] = "ERROR";
        result["error"]  = std::string("launch: ") + cudaGetErrorString(launch_err);
        return result;
    }

    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err == cudaErrorLaunchTimeout) {
        cudaGetLastError();
        result["status"] = "TIMEOUT";
        result["error"]  = "cudaErrorLaunchTimeout (Windows WDDM TDR ~2s watchdog), raise HKLM "
                           "TdrDelay or shrink the launch";
        return result;
    }
    if (sync_err != cudaSuccess) {
        cudaGetLastError();
        result["status"] = "ERROR";
        result["error"]  = std::string("sync: ") + cudaGetErrorString(sync_err);
        return result;
    }

    result["status"] = "OK";
    result["error"]  = std::string("");
    return result;
}

/* Drift guard: report the C sizeof each POD so the Python packer can assert byte-layout match. */
py::dict amk_struct_sizes() {
    py::dict d;
    d["amk_params_t"]      = (int64_t)sizeof(amk_params_t);
    d["amk_buffer_t"]      = (int64_t)sizeof(amk_buffer_t);
    d["amk_instruction_t"] = (int64_t)sizeof(amk_instruction_t);
    return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &amk_launch, "Launch the AMK persistent megakernel (cooperative)");
    m.def("max_coresident_blocks", &amk_max_coresident_blocks,
          "Max co-resident blocks for the VM kernel at (threads, dyn_smem)");
    m.def("kernel_attributes", &amk_kernel_attributes,
          "Compiled-kernel resource footprint (numRegs, smem) + occupancy at (threads, dyn_smem)");
    m.def("supports_cooperative", &amk_supports_cooperative,
          "Whether the device supports cudaLaunchCooperativeKernel");
    m.def("struct_sizes", &amk_struct_sizes, "C sizeof of the ABI PODs (byte-layout drift guard)");
}
