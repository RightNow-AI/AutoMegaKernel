"""
AMK, THE HOST LOADER for the persistent CUDA megakernel VM (Layer 0)
=====================================================================

``MegakernelVM(program, weights, device='cuda')`` is the GPU counterpart of
``vm/reference_vm.py``'s ``ReferenceVM`` and produces bit-equivalent results (within fp32
tolerance) on the SAME validated program + weights.

What it does, in order (mirroring the abi.h LAUNCH CONTRACT):

  1. ``validate(program)`` first and REFUSE to load a rejected program (agent-safety).
  2. Build the device POD tables EXACTLY as ``vm/abi.h`` lays them out:
       * ``amk_buffer_t[]``  , weights/IO from the caller's tensors (contiguous, right dtype),
                                activations from ONE scratch arena with per-buffer byte offsets
                                (dtype-sized), KV from provided tensors. ``ptr`` is the resolved
                                device address; ``shape``/``stride`` are row-major in elements.
       * ``amk_counter_t[]`` , host-zeroed uint32 (one launch == one pass).
       * ``amk_instruction_t[]``, opcode/params/inputs/outputs/waits/out_counter/sm per task.
       * flattened per-SM queues in GLOBAL TOPOLOGICAL ORDER (``program.topological_order()``);
         SMs assigned round-robin/load-balance when tasks lack an ``sm``.
  3. JIT-compile the extension (``cpp_extension.load``, cached) and verify its struct sizes match
     this packer's (drift guard).
  4. ``run(inputs, kv)`` zeroes counters, fills IO inputs into the arena, launches cooperatively
     (occupancy-checked; gridDim never exceeds measured co-residency), and returns
     ``{buffer_name: tensor}`` for every IO_OUTPUT and KV_CACHE, exactly like ReferenceVM.run.

Numpy structured dtypes pack the PODs so the byte layout is explicit and auditable; the compiled
extension reports its ``sizeof`` of each struct and we assert equality before any launch.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from schedule.ir import (
    Buffer, BufferKind, DType, InstructionKind, MegakernelProgram, validate,
)

_HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------------------------
# torch dtype mapping (mirrors reference_vm._TORCH_DTYPE for the dtypes the core ops support)
# ------------------------------------------------------------------------------------------------
_TORCH_DTYPE = {
    DType.F32: torch.float32, DType.F16: torch.float16, DType.BF16: torch.bfloat16,
    DType.I32: torch.int32, DType.I8: torch.int8, DType.U8: torch.uint8, DType.BOOL: torch.bool,
}


def _torch_dtype(dt: DType) -> torch.dtype:
    return _TORCH_DTYPE.get(dt, torch.float32)


# ------------------------------------------------------------------------------------------------
# numpy structured dtypes, the EXACT byte layout of the abi.h PODs (little-endian, natural C
# alignment on x86-64 / the NVCC host compiler). The compiled extension's sizeof is asserted equal.
# ------------------------------------------------------------------------------------------------
# amk_params_t: 19 int32 then 3 float32 -> 88 bytes, align 4.
_NP_PARAMS = np.dtype({
    "names": ["K", "N", "M", "N_tile", "M_tile", "n_off", "m_off", "hidden", "vocab",
              "head_dim", "n_heads", "n_kv_heads", "kv_start", "kv_len", "pos", "qdtype",
              "group", "flags", "dim", "eps", "scale", "theta"],
    "formats": ["<i4"] * 19 + ["<f4"] * 3,
    "offsets": [4 * i for i in range(19)] + [76, 80, 84],
    "itemsize": 88,
}, align=False)

# amk_buffer_t: void* ptr(8); int64 numel(8); int32 rank,dtype,space,_pad(16); int64 shape[4](32);
# int64 stride[4](32) -> 96 bytes, align 8.
_NP_BUFFER = np.dtype({
    "names": ["ptr", "numel", "rank", "dtype", "space", "_pad", "shape", "stride"],
    "formats": ["<u8", "<i8", "<i4", "<i4", "<i4", "<i4", ("<i8", 4), ("<i8", 4)],
    "offsets": [0, 8, 16, 20, 24, 28, 32, 64],
    "itemsize": 96,
}, align=False)

# amk_instruction_t: op,n_inputs,n_outputs,n_waits (16); inputs[8](32); outputs[4](16);
# wait_counter[8](32); wait_threshold[8](32); out_counter,sm(8); params(88) -> 224 bytes, align 4.
_NP_INSTR = np.dtype({
    "names": ["op", "n_inputs", "n_outputs", "n_waits", "inputs", "outputs",
              "wait_counter", "wait_threshold", "out_counter", "sm", "params"],
    "formats": ["<i4", "<i4", "<i4", "<i4", ("<i4", 8), ("<i4", 4),
                ("<i4", 8), ("<i4", 8), "<i4", "<i4", _NP_PARAMS],
    "offsets": [0, 4, 8, 12, 16, 48, 64, 96, 128, 132, 136],
    "itemsize": 224,
}, align=False)

# Param marshalling: which amk_params_t field each known IR param key maps to, and its C type.
_PARAM_KEYS_I = ("K", "N", "M", "N_tile", "M_tile", "n_off", "m_off", "hidden", "vocab",
                 "head_dim", "n_heads", "n_kv_heads", "kv_start", "kv_len", "pos", "qdtype",
                 "group", "flags", "dim")
_PARAM_KEYS_F = ("eps", "scale", "theta")


# ------------------------------------------------------------------------------------------------
# Extension build (cached). nvcc gencode for compute_120/sm_120; TORCH_CUDA_ARCH_LIST="12.0".
# ------------------------------------------------------------------------------------------------
_EXT_CACHE: dict[str, Any] = {}


def _ensure_msvc_on_path() -> None:
    """On Windows, nvcc needs the MSVC host compiler (cl.exe) on PATH. If it is missing (i.e. we
    were not launched from a 'x64 Native Tools' prompt), discover Visual Studio via vswhere and
    import the environment that vcvars64.bat sets, so `torch.utils.cpp_extension` can build."""
    if os.name != "nt":
        return
    import shutil
    import subprocess
    if shutil.which("cl"):
        return
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.exists(vswhere):
        raise RuntimeError("AMK: cl.exe not on PATH and vswhere.exe not found, install MSVC "
                           "build tools or launch from a 'x64 Native Tools Command Prompt'")
    install = subprocess.check_output(
        [vswhere, "-latest", "-products", "*",
         "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"], text=True).strip()
    vcvars = os.path.join(install, "VC", "Auxiliary", "Build", "vcvars64.bat")
    if not os.path.exists(vcvars):
        raise RuntimeError(f"AMK: vcvars64.bat not found at {vcvars}")
    # run vcvars64 and capture the resulting environment, then import it into this process
    out = subprocess.check_output(f'cmd /c ""{vcvars}" >nul 2>&1 && set"', text=True)
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            os.environ[k] = v
    if not shutil.which("cl"):
        raise RuntimeError("AMK: failed to put cl.exe on PATH via vcvars64.bat")


# ------------------------------------------------------------------------------------------------
# AUTOTUNE KNOBS, compile-time GEMV / occupancy macros (the AutoKernel search surface for the VM).
# Each maps to a -D<MACRO>=<value> consumed by vm/ops.cuh (GEMV) or vm/scheduler.cu (launch bound).
# A knob dict deterministically yields a DISTINCT extension name + build dir so every variant can
# coexist in one process (a real, reproducible A/B). vm/autotune.py enumerates a grid of these.
#   cols_per_warp : AMK_GEMV_COLS_PER_WARP  (output columns/weight-rows a warp computes -> x-reuse)
#   kunroll       : AMK_GEMV_KUNROLL        (float4/bf16x8 vectors loaded per lane per K-iteration)
#   lb_maxthreads : AMK_LB_MAXTHREADS  ) together -> __launch_bounds__(maxThreads, minBlocksPerSM)
#   lb_minblocks  : AMK_LB_MINBLOCKS   )  on the megakernel: cap regs -> RAISE occupancy (the lever)
# ------------------------------------------------------------------------------------------------
#   gemv_max_k    : AMK_GEMV_MAX_K          (static-SMEM x-cache size in floats; the OCCUPANCY lever:
#                                            8192 floats == 32KB static SMEM pins the megakernel to
#                                            2 blocks/SM because SMEM/SM (100KB on sm_120) / 35KB ==
#                                            2, registers are NOT the binder. Sizing this to the
#                                            program's real max K shrinks static SMEM and can lift
#                                            co-residency above 2. 0 -> compiler default 8192.)
#   cpasync       : AMK_GEMV_CPASYNC        (1 -> compile the cp.async DOUBLE-BUFFERED fp GEMV: each
#                                            warp streams its weight-row K-chunks SMEM<-HBM with
#                                            __pipeline_memcpy_async into a multi-stage ring and
#                                            computes on stage s while STAGES-1 future chunks load.
#                                            This keeps many independent loads in flight -> hides HBM
#                                            latency -> approaches peak bandwidth on the otherwise
#                                            latency-bound decode GEMV. Needs sm_80+; the loader only
#                                            enables it when the live device supports cp.async. 0 ->
#                                            the prior register warp-per-column path. The ring lives
#                                            in DYNAMIC smem provisioned by _gemv_cpasync_smem().)
#   cpa_cols      : AMK_GEMV_CPASYNC_COLS   (columns/weight-rows one warp streams at once -> MLP)
#   cpa_stages    : AMK_GEMV_CPA_STAGES     (ring depth, >=2; 3-4 = deepest pipeline that still fits)
#   cpa_vpl       : AMK_GEMV_CPA_VPL        (float4 granules per lane per chunk -> chunk width)
_KNOB_MACRO = {
    "cols_per_warp": "AMK_GEMV_COLS_PER_WARP",
    "kunroll":       "AMK_GEMV_KUNROLL",
    "lb_maxthreads": "AMK_LB_MAXTHREADS",
    "lb_minblocks":  "AMK_LB_MINBLOCKS",
    "gemv_max_k":    "AMK_GEMV_MAX_K",
    "cpasync":       "AMK_GEMV_CPASYNC",
    "cpa_cols":      "AMK_GEMV_CPASYNC_COLS",
    "cpa_stages":    "AMK_GEMV_CPA_STAGES",
    "cpa_vpl":       "AMK_GEMV_CPA_VPL",
    "qc":            "AMK_QGEMV_COLS",   # quantized (int4/int8) GEMV output columns per warp
    "qcpasync":      "AMK_QGEMV_CPASYNC",  # int8 quant GEMV cp.async ring (sm_80+; 0 -> synchronous path)
}
# cp.async is the production decode path (latency-hiding double-buffer); defaults here mirror the
# ops.cuh #defines (CPC=4, STAGES=3, VPL=2) and fit the megakernel SMEM budget at 2 blocks/SM.
_KNOB_DEFAULTS = {"cols_per_warp": 1, "kunroll": 1, "lb_maxthreads": 0, "lb_minblocks": 0,
                  "gemv_max_k": 0, "cpasync": 1, "cpa_cols": 2, "cpa_stages": 4, "cpa_vpl": 1,
                  "qc": 4, "qcpasync": 0}


def _normalize_knobs(knobs: dict | None) -> dict:
    # Guard: every searchable knob must appear in _KNOB_MACRO so autoresearch drift is caught at
    # import time rather than silently at the first GPU build.  We do this lazily (inside the
    # function that is always called before a build) to avoid a top-level circular import.
    try:
        from flywheel.prior import SEARCHABLE_KNOB_CHOICES as _skc
        _missing = [k for k in _skc if k not in _KNOB_MACRO]
        assert not _missing, (
            f"SEARCHABLE_KNOB_CHOICES keys {_missing} are not in vm.loader._KNOB_MACRO, "
            "update _KNOB_MACRO or fix the canonical knob list in flywheel/prior.py"
        )
    except ImportError:
        pass  # flywheel not importable in minimal test environments; skip the guard
    k = dict(_KNOB_DEFAULTS)
    if knobs:
        for key, val in knobs.items():
            if key not in _KNOB_MACRO:
                raise KeyError(f"unknown AMK GEMV knob {key!r}; valid: {sorted(_KNOB_MACRO)}")
            k[key] = int(val)
    # Range-guard every knob: an absurd value (e.g. cpa_stages=1000000) would overflow the cp.async
    # SMEM ring or the build. Fail fast with a clear message here, before any allocation. Bounds are
    # generous (cover the whole search space); they only catch values that physically cannot build.
    # Only the knobs that set SMEM/register footprint (the OOM/build risks); bounds are generous (the
    # whole search space fits), catching only values that physically cannot build. Flags + quant-mode
    # knobs (cpasync/qc/qcpasync/lb_*) carry no overflow risk and are validated elsewhere.
    _RANGES = {"cols_per_warp": (1, 16), "kunroll": (1, 64), "gemv_max_k": (0, 65536),
               "cpa_cols": (1, 16), "cpa_stages": (1, 16), "cpa_vpl": (1, 8)}
    for _key, (_lo, _hi) in _RANGES.items():
        if _key in k and not (_lo <= k[_key] <= _hi):
            raise ValueError(f"AMK knob {_key}={k[_key]} out of the validated range [{_lo}, {_hi}]; "
                             f"a value outside this can overflow SMEM or fail the build.")
    # launch_bounds is only emitted when BOTH parts are > 0 (else: compiler's own reg alloc)
    if (k["lb_maxthreads"] > 0) != (k["lb_minblocks"] > 0):
        raise ValueError("lb_maxthreads and lb_minblocks must be set together (both >0 or both 0)")
    return k


def _knob_suffix(knobs: dict) -> str:
    """Stable, filesystem-safe variant suffix from non-default knobs (empty for the default set)."""
    parts = [f"{k}{knobs[k]}" for k in
             ("cols_per_warp", "kunroll", "lb_maxthreads", "lb_minblocks", "gemv_max_k",
              "cpasync", "cpa_cols", "cpa_stages", "cpa_vpl", "qc", "qcpasync")
             if knobs[k] != _KNOB_DEFAULTS[k]]
    return ("_" + "_".join(parts)) if parts else ""


def _gemv_cpasync_smem(knobs: dict, program, threads_per_block: int) -> int:
    """Dynamic SMEM bytes the cp.async fp GEMV needs for this program: the largest GEMV's
    x-cache (K fp32, rounded to 16B) PLUS the cp.async weight ring
    (nwarps * STAGES * CPC * CHUNK * 16B). The ring's float4 granule is 16 bytes regardless of
    weight dtype, so CHUNK*sizeof(WT) == VPL*warpSize*16 for both fp32 and bf16/fp16. Returns 0
    when cp.async is disabled or the program has no fp (non-quant) GEMV tile."""
    if not knobs.get("cpasync"):
        return 0
    WARP = 32
    nwarps = (threads_per_block + WARP - 1) // WARP
    CPC, STAGES, VPL = knobs["cpa_cols"], knobs["cpa_stages"], knobs["cpa_vpl"]
    max_k = 0
    for t in program.tasks:
        if int(t.op) == int(InstructionKind.GEMV_TILE) and int(t.params.get("qdtype", 0)) == 0:
            max_k = max(max_k, int(t.params.get("K", 0)))
    if max_k <= 0:
        return 0
    x_floats = (max_k + 3) & ~3
    x_bytes = x_floats * 4
    ring_bytes = nwarps * STAGES * CPC * (VPL * WARP * 16)
    return x_bytes + ring_bytes


def _qgemv_cpasync_smem(knobs: dict, program, threads_per_block: int) -> int:
    """Dynamic SMEM the INT8 cp.async quant GEMV needs: the largest int8 GEMV's x-cache (K fp32,
    capped, 16B-rounded) PLUS the int8 weight ring (nwarps*STAGES*CPC*CHUNK bytes; the cp.async
    granule is a 16-byte uint4 == 16 int8, so CHUNK_bytes == VPL*warpSize*16). 0 when qcpasync is off
    or the program has no int8 (qdtype==AMK_I8==6) GEMV tile. The kernel carves the ring right after
    the x-cache (vm/ops.cuh), so the two are SUMMED (both live concurrently)."""
    if not knobs.get("qcpasync"):
        return 0
    WARP = 32
    nwarps = (threads_per_block + WARP - 1) // WARP
    CPC, STAGES, VPL = knobs["cpa_cols"], knobs["cpa_stages"], knobs["cpa_vpl"]
    QGEMV_SMEM_MAX_K = 16384   # must match vm/ops.cuh AMK_QGEMV_SMEM_MAX_K + the x-cache cap above
    max_k = 0
    for t in program.tasks:
        if int(t.op) == int(InstructionKind.GEMV_TILE) and int(t.params.get("qdtype", 0)) == 6:  # AMK_I8
            max_k = max(max_k, int(t.params.get("K", 0)))
    if max_k <= 0:
        return 0
    x_floats = (min(max_k, QGEMV_SMEM_MAX_K) + 3) & ~3
    ring_bytes = nwarps * STAGES * CPC * (VPL * WARP * 16)
    return x_floats * 4 + ring_bytes


def _build_extension(gemv_scalar: bool = False, knobs: dict | None = None,
                     op_noinline: bool = False):
    """JIT-build (cached) the megakernel extension.

    ``gemv_scalar`` selects the GEMV ablation variant: when True we compile with
    ``-DAMK_GEMV_SCALAR`` (the OLD naive one-thread-per-output-column, UNCOALESCED path in
    vm/ops.cuh) under a DISTINCT extension name + build dir, so the coalesced (default) and scalar
    variants can coexist in the same process, exactly what a reproducible A/B ablation needs. The
    default (False) is the current warp-per-column vectorized COALESCED path. Both are correctness-
    checked against the reference; the only difference is the memory access pattern.

    ``knobs`` selects an AUTOTUNE variant (cols_per_warp / kunroll / launch-bounds). Each distinct
    knob set compiles a distinct extension (distinct -D set + distinct name/build dir), so the
    on-hardware tune loop in vm/autotune.py can build, correctness-gate, and time each one."""
    knobs = _normalize_knobs(knobs)
    variant = ("scalar" if gemv_scalar else "coalesced") + _knob_suffix(knobs) \
              + ("_noinline" if op_noinline else "")
    if variant in _EXT_CACHE:
        return _EXT_CACHE[variant]

    # RETARGETING SURFACE: derive the gencode from the LIVE device, never hardcode an arch.
    # sm_120 (RTX 5090), sm_90 (H100), sm_80 (A100), sm_100 (B200) all fall out of this.
    cap = torch.cuda.get_device_capability()
    arch = f"{cap[0]}.{cap[1]}"      # e.g. "12.0", "9.0", "8.0"
    cc = f"{cap[0]}{cap[1]}"         # e.g. "120", "90", "80"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    if os.name == "nt":              # MSVC discovery is Windows-only; Linux/Modal uses gcc
        _ensure_msvc_on_path()
    from torch.utils.cpp_extension import load

    sources = [
        os.path.join(_HERE, "scheduler.cu"),
        os.path.join(_HERE, "sync.cu"),
        os.path.join(_HERE, "pages.cu"),
        os.path.join(_HERE, "vm_ext.cpp"),
    ]
    # -gencode for the live arch; correctness-first (no --use_fast_math); rdc not needed since
    # all device fns are __forceinline__ in headers (single-kernel TU in scheduler.cu).
    nvcc_flags = [
        f"-gencode=arch=compute_{cc},code=sm_{cc}",
        f"-gencode=arch=compute_{cc},code=compute_{cc}",
        "--extended-lambda",
        "-std=c++17",
    ]
    if os.name == "nt":
        # COLD-BUILD RELIABILITY (the #1 hardening fix). torch injects -D__CUDA_NO_*_CONVERSIONS__,
        # so scheduler.cu (which includes ops.cuh -> cuda_fp16.h / cuda_bf16.h) pulls in the deeply
        # nested CCCL <cuda/std/...> headers. Under MSVC's *traditional* (legacy) preprocessor -
        # which nvcc uses by default on Windows, CCCL's variadic-macro / token-paste expansion is
        # non-conforming and INTERMITTENTLY mis-parses, surfacing as spurious '__half' /
        # 'amk_gemv_row_dot' type errors on a from-scratch parallel ninja build (a re-run then
        # builds clean because the timing differs). nvcc/CCCL themselves warn: "MSVC/cl.exe with
        # traditional preprocessor is used. This may lead to unexpected compilation errors."
        # Forcing the standard-conforming preprocessor (/Zc:preprocessor) makes the macro expansion
        # deterministic, so a clean build succeeds on the FIRST try, every time. This mirrors what
        # instructions/_build.py already does for the Layer-1 micro-kernel builds.
        nvcc_flags += ["-Xcompiler", "/Zc:preprocessor",
                       "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"]
    # GEMV ABLATION: compile-time toggle for the uncoalesced (scalar) GEMV path. A distinct
    # extension name + build dir per variant lets both coexist (a real A/B in one process).
    name_suffix = ""
    if gemv_scalar:
        nvcc_flags.append("-DAMK_GEMV_SCALAR")
        name_suffix = "_gemvscalar"
    # AUTOTUNE knobs -> -D macros (only non-defaults emitted; launch-bounds only when both >0).
    for key in ("cols_per_warp", "kunroll", "gemv_max_k", "cpa_cols", "cpa_stages", "cpa_vpl", "qc"):
        if knobs[key] != _KNOB_DEFAULTS[key]:
            nvcc_flags.append(f"-D{_KNOB_MACRO[key]}={knobs[key]}")
    # cp.async double-buffered GEMV: enabled by default (the production decode path) ONLY when the
    # live device supports cp.async (sm_80+). On older arches it silently stays on the register path.
    if knobs.get("cpasync") and cap[0] >= 8:
        nvcc_flags.append(f"-DAMK_GEMV_CPASYNC={knobs['cpasync']}")
    if knobs.get("qcpasync") and cap[0] >= 8:
        nvcc_flags.append(f"-DAMK_QGEMV_CPASYNC={knobs['qcpasync']}")
    if knobs["lb_maxthreads"] > 0 and knobs["lb_minblocks"] > 0:
        nvcc_flags.append(f"-DAMK_LB_MAXTHREADS={knobs['lb_maxthreads']}")
        nvcc_flags.append(f"-DAMK_LB_MINBLOCKS={knobs['lb_minblocks']}")
    name_suffix += _knob_suffix(knobs)
    # OCCUPANCY EXPERIMENT: mark per-opcode device fns __noinline__ to shrink the kernel's register
    # frame (GOAL 2). Distinct build dir so the inline/noinline A/B coexist in one process.
    if op_noinline:
        nvcc_flags.append("-DAMK_OP_NOINLINE")
        name_suffix += "_noinline"
    cflags = (["/std:c++17", "/Zc:preprocessor"] if os.name == "nt" else ["-std=c++17"])
    ext = load(
        name=f"amk_vm_ext_sm{cc}{name_suffix}",  # per-arch + per-variant build dir (no stale SASS)
        sources=sources,
        extra_include_paths=[_HERE],
        extra_cuda_cflags=nvcc_flags,
        extra_cflags=cflags,
        verbose=True,
    )
    # drift guard: the numpy packer's byte layout MUST equal the C struct layout, or we marshal
    # garbage onto the device. Assert sizeof equality before any program is ever launched.
    sizes = ext.struct_sizes()
    _assert_layout(sizes)

    _EXT_CACHE[variant] = ext
    return ext


def _assert_layout(sizes: dict) -> None:
    want = {
        "amk_params_t": _NP_PARAMS.itemsize,
        "amk_buffer_t": _NP_BUFFER.itemsize,
        "amk_instruction_t": _NP_INSTR.itemsize,
    }
    for k, v in want.items():
        if int(sizes[k]) != int(v):
            raise RuntimeError(
                f"AMK POD layout drift: C sizeof({k})={sizes[k]} != numpy packer {v}. "
                f"The host loader would marshal wrong bytes onto the device.")


# ------------------------------------------------------------------------------------------------
class MegakernelVM:
    """GPU megakernel VM. Same contract as ReferenceVM; result equal within fp32 tolerance."""

    def __init__(self, program: MegakernelProgram, weights: dict[str, torch.Tensor],
                 device: str = "cuda", strict_validate: bool = True,
                 gemv_scalar: bool | None = None, sm_round_robin: bool = False,
                 disable_persistent: bool = False, knobs: dict | None = None,
                 op_noinline: bool = False):
        """ABLATION KNOBS (paper A/B; all default to the production config):
          * ``gemv_scalar``      , compile the UNCOALESCED v1 GEMV (one-thread-per-column) instead
                                    of the default warp-per-column vectorized coalesced path. None
                                    (default) honors the AMK_GEMV_SCALAR env var; True/False force it.
          * ``sm_round_robin``   , assign tasks to SMs round-robin over the topo order instead of
                                    the default greedy LPT cost-balance.
          * ``disable_persistent``- rebuild + re-upload ALL device tables on EVERY run() (no
                                    steady-state reuse), to measure the persistent-tables win.
          * ``knobs``            , AUTOTUNE dict {cols_per_warp, kunroll, lb_maxthreads,
                                    lb_minblocks}; compiles a distinct GEMV/occupancy variant. None
                                    => the default (cols_per_warp=1, kunroll=1, no launch bound),
                                    which is the prior production kernel. vm/autotune.py grids these.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("MegakernelVM requires a CUDA device")
        self.prog = program
        self.device = torch.device(device)
        self.weights = weights
        if gemv_scalar is None:
            gemv_scalar = os.environ.get("AMK_GEMV_SCALAR", "") not in ("", "0", "false", "False")
        self.gemv_scalar = bool(gemv_scalar)
        self.sm_round_robin = bool(sm_round_robin)
        self.disable_persistent = bool(disable_persistent)
        self.op_noinline = bool(op_noinline)
        self.knobs = _normalize_knobs(knobs)

        if strict_validate:
            res = validate(program)
            if not res.ok:
                raise ValueError(
                    "MegakernelVM refuses to load an invalid schedule:\n" + res.report())

        self.num_sms = program.target.num_sms if program.target else 82
        cfg = program.config
        self.threads_per_block = cfg.threads_per_block if cfg else 256
        # STABILITY GUARD: the cooperative grid-sync megakernel is validated up to 256 threads/block;
        # 512 deadlocks it (measured). Refuse it with a fast, clear error instead of hanging, BEFORE
        # the expensive build/launch - a production system must never hang. Raise the cap only after
        # the cooperative barrier is fixed + re-validated (then restore 512 to THREADS_PER_BLOCK_CHOICES).
        _SAFE_MAX_TPB = 256
        if self.threads_per_block > _SAFE_MAX_TPB:
            raise ValueError(
                f"threads_per_block={self.threads_per_block} exceeds the cooperative-kernel validated "
                f"maximum {_SAFE_MAX_TPB}; higher values deadlock the grid sync. Refused (fail-fast), "
                f"not launched, so the system never hangs.")
        self.dyn_smem_bytes = cfg.smem_bytes_per_block if cfg else 0
        # Software-pipelining depth (ScheduleConfig.pipelining_depth). Threaded to the kernel per
        # GEMV_TILE via the unused params.M_tile slot (GEMV reads only K/N_tile/n_off); the
        # scheduler prefetches the next GEMV tile's weights this many queue-slots ahead. 0 = off.
        self.pipelining_depth = int(cfg.pipelining_depth) if cfg else 0

        # WARP-PARALLEL ATTENTION: amk_inst_attention_tile gives every WARP a PRIVATE q_s[head_dim] |
        # acc[head_dim] slice of dynamic smem (heads run concurrently across warps, no per-token
        # barriers, see vm/ops.cuh). The loader MUST opt the kernel into nwarps*2*head_dim floats
        # for the LARGEST attention head_dim in the program. nwarps = ceil(threads_per_block/32).
        # e.g. 512 threads (16 warps) x head_dim 128 -> 16*2*128*4 = 16 KB; head_dim 512 -> 64 KB
        # (within the per-block opt-in cap; the guard below catches any overflow).
        _nwarps = (self.threads_per_block + 31) // 32
        max_attn_hd = 0
        for t in program.tasks:
            if int(t.op) == int(InstructionKind.ATTENTION_TILE):
                max_attn_hd = max(max_attn_hd, int(t.params.get("head_dim", 0)))
        if max_attn_hd > 0:
            attn_dyn = _nwarps * 2 * max_attn_hd * 4   # per-warp q_s|acc, all warps; floats -> bytes
            self.dyn_smem_bytes = max(self.dyn_smem_bytes, attn_dyn)
        self.max_attn_head_dim = max_attn_hd

        # QUANTIZED GEMV x-cache: the dequant-fused int4/int8 GEMV caches the x row in DYNAMIC
        # shared memory (it cannot afford a 2nd large STATIC array on top of the fp GEMV's). The
        # loader sizes the kernel's dynamic smem to hold the largest quantized GEMV's K floats
        # (capped at AMK_QGEMV_SMEM_MAX_K to respect the opt-in cap; larger K falls back to L2 x).
        AMK_QGEMV_SMEM_MAX_K = 16384   # covers real MLP intermediate dims (11008@7B, 13824@13B) so the
        #                                down-proj GEMV (K=intermediate) uses the fast SMEM-x path. Must
        #                                match vm/ops.cuh AMK_QGEMV_SMEM_MAX_K. Capped by the opt-in guard.
        max_qk = 0
        for t in program.tasks:
            if int(t.op) == int(InstructionKind.GEMV_TILE) and int(t.params.get("qdtype", 0)) in (6, 7):
                max_qk = max(max_qk, int(t.params.get("K", 0)))
        if max_qk > 0:
            qgemv_dyn = min(max_qk, AMK_QGEMV_SMEM_MAX_K) * 4   # floats -> bytes
            self.dyn_smem_bytes = max(self.dyn_smem_bytes, qgemv_dyn)
        self.max_quant_gemv_k = max_qk

        # cp.async DOUBLE-BUFFERED fp GEMV: the ring + x-cache live in DYNAMIC smem. Size it to the
        # program's largest fp GEMV K (x-cache) plus the cp.async weight ring. Only when cp.async is
        # the active variant AND the device supports it (the build only emits the cp.async path then).
        self._gemv_cpasync_bytes = 0
        cap0 = torch.cuda.get_device_capability()[0]
        if self.knobs.get("cpasync") and cap0 >= 8:
            self._gemv_cpasync_bytes = _gemv_cpasync_smem(self.knobs, program, self.threads_per_block)
            self.dyn_smem_bytes = max(self.dyn_smem_bytes, self._gemv_cpasync_bytes)

        # INT8 cp.async quant-GEMV ring (opt-in via knob 'qcpasync', sm_80+): x-cache + weight ring,
        # summed (the kernel carves the ring right after the x-cache). Becomes the dyn-smem max when on.
        self._qgemv_cpasync_bytes = 0
        if self.knobs.get("qcpasync") and cap0 >= 8:
            self._qgemv_cpasync_bytes = _qgemv_cpasync_smem(self.knobs, program, self.threads_per_block)
            self.dyn_smem_bytes = max(self.dyn_smem_bytes, self._qgemv_cpasync_bytes)

        # opt-in SMEM cap guard (abi.h: <= per-block opt-in, NOT per-sm)
        if program.target and self.dyn_smem_bytes > program.target.smem_bytes_per_block_optin:
            raise ValueError(
                f"smem_bytes_per_block={self.dyn_smem_bytes} exceeds target opt-in cap "
                f"{program.target.smem_bytes_per_block_optin}")

        self.ext = _build_extension(gemv_scalar=self.gemv_scalar, knobs=self.knobs,
                                    op_noinline=self.op_noinline)
        self.supports_coop = bool(self.ext.supports_cooperative())

        # ---- topological global order ----
        self._topo = program.topological_order()
        if self._topo is None:
            raise ValueError("MegakernelVM refuses: program has a dependency cycle")

        # ---- occupancy-checked cooperative grid_dim (abi.h: never exceed co-residency) ----
        # One block per SM is the target; if occupancy forces fewer co-resident blocks than
        # num_sms, we cap grid_dim and pack all work into [0, grid_dim). We compute this ONCE here
        # so SM assignment lands within the launchable grid.
        max_grid = int(self.ext.max_coresident_blocks(self.threads_per_block, self.dyn_smem_bytes))
        if max_grid <= 0:
            raise RuntimeError("AMK: kernel reports zero occupancy at this launch config")
        self.grid_dim = min(self.num_sms, max_grid)

        self._assign_sms()

    # --------------------------------------------------------------------------------------------
    def _assign_sms(self) -> None:
        """Resolve each task's SM into [0, grid_dim). If tasks already carry an sm in range (the
        validator checked per-SM queue order), keep it; otherwise COST-BALANCE the tasks across SMs.

        Decode is HBM-bandwidth bound, so a tile's wall-cost is dominated by the weight bytes it
        streams (est_bytes); est_flops is the tie-break for the few compute-bound ops. We walk the
        GLOBAL TOPOLOGICAL ORDER and greedily place each task on the currently-least-loaded SM
        (loaded by accumulated est_bytes, then est_flops, then count). Because every SM still
        processes its tasks in global-topo order, each SM's queue remains a linear extension of the
        DAG, the abi.h invariant the validator enforces (no SM blocks on a counter only its own
        later queue entry could signal) is preserved. Cost-balancing cuts stragglers vs the naive
        round-robin that ignored that GEMV tiles cost 100x an elementwise ADD."""
        import heapq
        prog = self.prog
        g = self.grid_dim
        has_sm = any(t.sm is not None for t in prog.tasks)
        if has_sm and all((t.sm is None or 0 <= t.sm < g) for t in prog.tasks):
            self._sm_of = {t.id: (t.sm if t.sm is not None else 0) for t in prog.tasks}
        elif self.sm_round_robin:
            # ABLATION: naive round-robin over the GLOBAL TOPO ORDER (ignores per-tile cost). Each
            # SM still gets a topo subsequence (assignment walks the topo stream), so the abi.h
            # per-SM queue-order invariant holds, this is the cost-blind baseline the default LPT
            # cost-balance is measured against. GEMV tiles cost ~100x an elementwise ADD, so RR
            # leaves stragglers the LPT scheme avoids.
            self._sm_of = {}
            for i, tid in enumerate(self._topo):
                self._sm_of[tid] = i % g
        else:
            cost_of = {t.id: (int(t.est_bytes), int(t.est_flops)) for t in prog.tasks}
            # min-heap keyed by (bytes_load, flops_load, count, sm); pop the least-loaded SM,
            # assign the next topo task to it, push it back with the updated load. Greedy LPT-style
            # balance over the topo stream keeps each SM's queue a topo subsequence by construction.
            heap = [(0, 0, 0, s) for s in range(g)]
            heapq.heapify(heap)
            self._sm_of = {}
            for tid in self._topo:
                bl, fl, cnt, s = heapq.heappop(heap)
                self._sm_of[tid] = s
                eb, ef = cost_of.get(tid, (0, 0))
                heapq.heappush(heap, (bl + eb, fl + ef, cnt + 1, s))

        # per-SM queues in global topo order (length num_sms; queues for SM>=grid_dim stay empty)
        self._queues: list[list[int]] = [[] for _ in range(self.num_sms)]
        for tid in self._topo:
            self._queues[self._sm_of[tid]].append(tid)

    # --------------------------------------------------------------------------------------------
    def _bind_external(self, b: Buffer, inputs: dict[str, torch.Tensor],
                       kv: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Return the device tensor for a NON-activation buffer (weight/const/io_input/kv), or None
        for ACTIVATION/IO_OUTPUT (which live in the scratch arena)."""
        dt = _torch_dtype(b.dtype)
        if b.kind in (BufferKind.WEIGHT, BufferKind.CONST):
            key = b.source or b.name
            if key not in self.weights:
                raise KeyError(f"weight/const buffer '{b.name}' (source='{b.source}') not in weights")
            # QUANTIZED weight buffers (I4 packed nibbles / I8) carry PACKED integer storage that
            # must NOT be dtype-cast: the device kernel unpacks/scales them. The packer stores I4 as
            # uint8 ([N, K//2]) and I8 as int8 ([N, K]); bind those raw, contiguous, on device.
            if b.dtype in (DType.I4, DType.I8):
                return self.weights[key].to(self.device).contiguous()
            t = self.weights[key].to(self.device, dt).contiguous()
            return t
        if b.kind == BufferKind.IO_INPUT:
            if b.name not in inputs:
                raise KeyError(f"IO_INPUT buffer '{b.name}' not provided to run()")
            return inputs[b.name].to(self.device, dt).contiguous()
        if b.kind == BufferKind.KV_CACHE:
            if b.name in kv:
                return kv[b.name].to(self.device, dt).contiguous()
            return torch.zeros(b.shape, dtype=dt, device=self.device)
        return None  # ACTIVATION / IO_OUTPUT -> arena

    # --------------------------------------------------------------------------------------------
    def _io_fingerprint(self, inputs: dict[str, torch.Tensor],
                        kv: dict[str, torch.Tensor]) -> tuple:
        """A hashable signature of the input/kv shapes+dtypes (NOT values). Two run() calls with
        the same fingerprint can reuse the device tables the first build produced, only the IO
        input *values* need re-copying. Distinct shapes/dtypes force a fresh build."""
        sig_in = tuple(sorted(
            (k, tuple(v.shape), str(v.dtype)) for k, v in inputs.items()))
        sig_kv = tuple(sorted(
            (k, tuple(v.shape), str(v.dtype)) for k, v in kv.items()))
        return (id(self.prog), sig_in, sig_kv)

    # --------------------------------------------------------------------------------------------
    def run(self, inputs: dict[str, torch.Tensor],
            kv: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Execute one forward pass (== one decoded token).

        STEADY-STATE (persistent tables): the FIRST run() for a given program/shape builds and
        uploads ALL device tables (buffers/instructions/queues/arena/counters) once. Every
        subsequent run() with the SAME program and the SAME input/kv shapes reuses those tables
        verbatim, it only host-memsets the counters to zero and copies the NEW IO-input values
        into their persistent device tensors, then relaunches the cooperative kernel. This pays the
        per-token host marshalling cost ONCE instead of every token (the abi.h DECODE MODEL: the
        host drives the autoregressive loop, KV persists in HBM, counters reset per launch).

        A shape/dtype change (or a different program object) transparently triggers a fresh build.
        validate() already gated the program at __init__; the persistent path never bypasses it."""
        kv = kv or {}

        fp = self._io_fingerprint(inputs, kv)
        # ABLATION: disable_persistent forces a full table rebuild+upload on EVERY run() (no
        # steady-state reuse), so a bench can measure the persistent-device-tables win directly.
        if not self.disable_persistent and getattr(self, "_built_fp", None) == fp:
            return self._run_steady_state(inputs, kv)
        return self._run_full_build(inputs, kv, fp)

    # --------------------------------------------------------------------------------------------
    def _run_steady_state(self, inputs: dict[str, torch.Tensor],
                          kv: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Fast per-token path: device tables already built and uploaded for this shape. Copy the
        new IO-input values into their persistent device tensors, optionally refresh KV inputs,
        zero the counters, relaunch the kernel, read outputs back. No host re-packing, no table
        H2D, no fresh allocations."""
        # 1) copy new input VALUES into the persistent device input tensors (same data_ptr). copy_
        #    handles dtype cast + H2D in one op; the device pointer in the buffer table is unchanged.
        for name, t in self._io_input_tensors.items():
            t.copy_(inputs[name])
        # 2) KV inputs: the caller may pass fresh KV tensors; copy their values into the persistent
        #    KV device tensors so the cache reflects prior steps (decode loop owns KV in HBM).
        for name, t in self._kv_tensors.items():
            if name in kv:
                t.copy_(kv[name])
        # 3) zero counters + trap flag (one launch == one pass) and relaunch the cooperative kernel.
        self._counters.zero_()
        self._abort.zero_()
        torch.cuda.synchronize()
        status = self.ext.launch(*self._launch_args)
        self.last_status = status
        if status["status"] == "TIMEOUT":
            raise TimeoutError("AMK launch hit WDDM TDR: " + status["error"])
        if status["status"] != "OK":
            raise RuntimeError("AMK launch failed: " + status["error"])
        torch.cuda.synchronize()
        self._check_abort()
        # 4) read outputs back (IO_OUTPUT + KV_CACHE) from the persistent buffer views.
        out: dict[str, torch.Tensor] = {}
        for b in self.prog.buffers:
            if b.kind in (BufferKind.IO_OUTPUT, BufferKind.KV_CACHE):
                out[b.name] = self._buf_tensor[b.id].clone()
        return out

    # --------------------------------------------------------------------------------------------
    def _run_full_build(self, inputs: dict[str, torch.Tensor],
                        kv: dict[str, torch.Tensor], fp: tuple) -> dict[str, torch.Tensor]:
        prog = self.prog
        dev = self.device

        # ---- 1. bind external tensors; keep refs alive so their device memory persists -----------
        # IO_INPUT / KV_CACHE tensors are made PERSISTENT so steady-state run()s can copy_ new
        # values into them without changing any device pointer (the whole point of persistent tables).
        self._keepalive: list[torch.Tensor] = []
        self._io_input_tensors: dict[str, torch.Tensor] = {}
        self._kv_tensors: dict[str, torch.Tensor] = {}
        buf_tensor: dict[int, torch.Tensor] = {}
        buf_ptr: dict[int, int] = {}
        arena_buffers: list[Buffer] = []
        for b in prog.buffers:
            ext = self._bind_external(b, inputs, kv)
            if ext is not None:
                self._keepalive.append(ext)
                buf_tensor[b.id] = ext
                buf_ptr[b.id] = ext.data_ptr()
                if b.kind == BufferKind.IO_INPUT:
                    self._io_input_tensors[b.name] = ext
                elif b.kind == BufferKind.KV_CACHE:
                    self._kv_tensors[b.name] = ext
            else:
                arena_buffers.append(b)

        # ---- 2. allocate ONE scratch arena for activations/IO_OUTPUT; per-buffer dtype-sized offset
        arena_offsets: dict[int, int] = {}
        cursor = 0
        for b in arena_buffers:
            elsize = _torch_dtype(b.dtype).itemsize
            # 256-byte align each buffer (safe for any vectorized access, matches CUDA malloc align)
            cursor = (cursor + 255) & ~255
            arena_offsets[b.id] = cursor
            cursor += b.numel * elsize
        arena_bytes = max(cursor, 256)
        arena = torch.zeros(arena_bytes, dtype=torch.uint8, device=dev)
        self._keepalive.append(arena)
        arena_base = arena.data_ptr()
        for b in arena_buffers:
            off = arena_offsets[b.id]
            buf_ptr[b.id] = arena_base + off
            # a typed view so we can read outputs back at the end
            dt = _torch_dtype(b.dtype)
            n = b.numel
            view = arena[off:off + n * dt.itemsize].view(dt).view(b.shape)
            buf_tensor[b.id] = view

        # ---- 3. pack amk_buffer_t[] -------------------------------------------------------------
        nb = len(prog.buffers)
        buf_arr = np.zeros(nb, dtype=_NP_BUFFER)
        for b in prog.buffers:
            strides = b.contiguous_strides()
            shape = list(b.shape) + [0] * (4 - len(b.shape))
            stride = list(strides) + [0] * (4 - len(strides))
            rec = buf_arr[b.id]
            rec["ptr"] = np.uint64(buf_ptr[b.id])
            rec["numel"] = b.numel
            rec["rank"] = len(b.shape)
            rec["dtype"] = int(b.dtype)
            rec["space"] = int(b.space)
            rec["shape"][:] = shape
            rec["stride"][:] = stride

        # ---- 4. counters (host-zeroed uint32) ---------------------------------------------------
        nc = len(prog.counters)
        counters = torch.zeros(max(nc, 1), dtype=torch.int32, device=dev)

        # ---- 5. pack amk_instruction_t[] --------------------------------------------------------
        ni = len(prog.tasks)
        instr_arr = np.zeros(max(ni, 1), dtype=_NP_INSTR)
        for t in prog.tasks:
            rec = instr_arr[t.id]
            rec["op"] = int(t.op)
            rec["n_inputs"] = len(t.inputs)
            rec["n_outputs"] = len(t.outputs)
            rec["n_waits"] = len(t.waits)
            for i, bid in enumerate(t.inputs):
                rec["inputs"][i] = bid
            for i, bid in enumerate(t.outputs):
                rec["outputs"][i] = bid
            for i, w in enumerate(t.waits):
                rec["wait_counter"][i] = w.counter
                rec["wait_threshold"][i] = w.threshold
            rec["out_counter"] = t.out_counter
            rec["sm"] = self._sm_of[t.id]
            p = rec["params"]
            for k in _PARAM_KEYS_I:
                if k in t.params:
                    p[k] = int(t.params[k])
            for k in _PARAM_KEYS_F:
                if k in t.params:
                    p[k] = float(t.params[k])
            # Pipelining: stash the prefetch depth in M_tile for GEMV tiles (unused by the op).
            # The scheduler reads inst.params.M_tile to decide how far ahead to prefetch weights.
            if int(t.op) == int(InstructionKind.GEMV_TILE):
                p["M_tile"] = int(self.pipelining_depth)

        # ---- 6. flattened per-SM queues in GLOBAL TOPO ORDER -----------------------------------
        q_off = np.zeros(self.num_sms, dtype=np.int32)
        q_len = np.zeros(self.num_sms, dtype=np.int32)
        flat: list[int] = []
        for s in range(self.num_sms):
            q_off[s] = len(flat)
            q_len[s] = len(self._queues[s])
            flat.extend(self._queues[s])
        flat_arr = np.asarray(flat if flat else [0], dtype=np.int32)

        # ---- 7. upload POD tables as device byte/int tensors ------------------------------------
        def _upload(np_arr: np.ndarray) -> torch.Tensor:
            raw = np.ascontiguousarray(np_arr).view(np.uint8)
            t = torch.from_numpy(raw.copy()).to(dev)
            self._keepalive.append(t)
            return t

        d_buffers = _upload(buf_arr)
        d_instr = _upload(instr_arr)
        d_qflat = torch.from_numpy(flat_arr.copy()).to(dev)
        self._keepalive.append(d_qflat)
        d_qoff = torch.from_numpy(q_off.copy()).to(dev)
        self._keepalive.append(d_qoff)
        d_qlen = torch.from_numpy(q_len.copy()).to(dev)
        self._keepalive.append(d_qlen)
        d_abort = torch.zeros(1, dtype=torch.int32, device=dev)
        self._keepalive.append(d_abort)

        # ---- 8. occupancy-checked cooperative gridDim (computed at init; assignment fits it) ----
        grid_dim = self.grid_dim
        # invariant: SM assignment landed in [0, grid_dim), so no non-empty queue is unreachable.
        for s in range(grid_dim, self.num_sms):
            assert not self._queues[s], "AMK loader bug: work assigned beyond launchable grid"
        self.last_grid_dim = grid_dim

        # ---- 9. zero counters (one launch == one pass) and LAUNCH ------------------------------
        counters.zero_()
        # Stash the exact (device-pointer) launch arguments + the counters tensor so a bench can
        # re-fire JUST the cooperative kernel (no host re-packing / re-upload) for clean kernel-only
        # timing via relaunch(). All device tensors are kept alive in self._keepalive.
        self._counters = counters
        self._abort = d_abort
        self._launch_args = (
            d_buffers.data_ptr(), nb,
            counters.data_ptr(), nc,
            d_instr.data_ptr(), ni,
            d_qflat.data_ptr(), d_qoff.data_ptr(), d_qlen.data_ptr(),
            grid_dim,
            arena.data_ptr(), arena_bytes,
            d_abort.data_ptr(),
            grid_dim, self.threads_per_block, self.dyn_smem_bytes,
        )
        d_abort.zero_()                         # clear the watchdog/trap flag before launching
        torch.cuda.synchronize()
        status = self.ext.launch(*self._launch_args)
        self.last_status = status
        if status["status"] == "TIMEOUT":
            raise TimeoutError("AMK launch hit WDDM TDR: " + status["error"])
        if status["status"] != "OK":
            raise RuntimeError("AMK launch failed: " + status["error"])

        torch.cuda.synchronize()
        self._check_abort()                     # surface a device-side opcode TRAP (robustness)

        # ---- 9b. record persistent state so the next run() with the same shapes is a fast
        #          steady-state relaunch (only counter-zero + IO-input copy_, no re-marshalling). --
        self._buf_tensor = buf_tensor
        self._built_fp = fp

        # ---- 10. read outputs (IO_OUTPUT + KV_CACHE) back ---------------------------------------
        out: dict[str, torch.Tensor] = {}
        for b in prog.buffers:
            if b.kind in (BufferKind.IO_OUTPUT, BufferKind.KV_CACHE):
                out[b.name] = buf_tensor[b.id].clone()
        return out

    # --------------------------------------------------------------------------------------------
    def _check_abort(self) -> None:
        """Surface a DEVICE-SIDE TRAP. amk_dispatch sets abort_flag to a negative, nonzero value
        when it hits an unimplemented/unknown opcode (encoded as -(op+1)) instead of silently
        no-oping and letting the program hang on a counter the trapped instruction never signals.
        A positive value is the host/TDR watchdog. Either way a nonzero flag means the pass did not
        complete cleanly; we raise so the caller never trusts a half-finished result."""
        flag = int(self._abort.item())
        if flag == 0:
            return
        if flag < 0:
            opcode = (-flag) - 1
            self.last_status = {"status": "TRAP", "error": f"unimplemented opcode {opcode}"}
            raise RuntimeError(
                f"AMK: kernel TRAPPED on unimplemented/unknown opcode {opcode} "
                f"(abort_flag={flag}); the VM aborted instead of hanging or returning garbage")
        self.last_status = {"status": "ABORTED", "error": f"abort_flag={flag}"}
        raise RuntimeError(f"AMK: kernel aborted (abort_flag={flag})")

    # --------------------------------------------------------------------------------------------
    def relaunch(self) -> dict:
        """Re-fire JUST the cooperative megakernel using the device tables a prior run() built -
        no host re-packing or H2D copies. For kernel-only benchmarking (e.g. cuda-event timing of
        the pipelining win). Requires a prior run(); zeroes counters + trap flag first (one launch
        == one pass). Returns the same status dict as launch()."""
        if not hasattr(self, "_launch_args"):
            raise RuntimeError("relaunch() requires a prior run() to build the device tables")
        self._counters.zero_()
        self._abort.zero_()
        status = self.ext.launch(*self._launch_args)
        self.last_status = status
        if status["status"] != "OK":
            raise RuntimeError("AMK relaunch failed: " + status["error"])
        return status
