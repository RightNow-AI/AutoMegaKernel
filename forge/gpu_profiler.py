"""AMK Forge -- the measurement engine.

JIT-compiles three torch-free microkernels on sm_120 (the exact CU/CPP split proven to build by
``forge/_scratch/probe_latency_proto.py``) and runs the drift-robust sweeps a :class:`GpuModel`
needs:

  * ``read_reduce_k`` -- grid-stride float4 read-reduce with compile-time ILP ``-DPROBE_U``; a
    never-touched dynamic-SMEM reservation throttles occupancy; the accumulator flows to ``out[]``
    so nothing is dead-code-eliminated. THIS is design-A (occupancy saturation sweep).
  * ``chase_k`` -- 1 block x 1 thread Hamiltonian-cycle pointer-chase, dependent loads, ``*out``
    sink; the random successor map defeats the prefetcher. THIS is design-B (HBM latency).
  * ``copy_k`` -- ``dst[i]=src[i]`` 2x-traffic anchor.

Self-retargets to any GPU by reading num_sms / smem-per-SM / threads-per-SM / L2 / max-blocks via
``cudaDeviceGetAttribute`` at runtime, so this same file runs on an H100 unchanged. Timing reuses
``eval.peak_bandwidth._event_time_ms`` (CUDA-event median) and ``_clock_state`` for the boost/
throttle annotation; the sweep adds an INTERLEAVED round-robin so run-to-run drift hits every
occupancy point equally within one window.

Run via the scratch driver:  ``./.venv/Scripts/python.exe forge/_scratch/run_profile_5090.py``
"""
from __future__ import annotations

import statistics
import sys

import numpy as np

sys.path.insert(0, "C:/Users/jaber/RightNow-Full/AutoMegaKernel")
from vm.loader import _ensure_msvc_on_path  # noqa: E402  (puts MSVC cl.exe on PATH)

_ensure_msvc_on_path()
import torch  # noqa: E402
from torch.utils.cpp_extension import load_inline  # noqa: E402

from eval.peak_bandwidth import _clock_state, _event_time_ms, measure_peak_bandwidth  # noqa: E402
from schedule.ir import TARGETS  # noqa: E402

from .gpu_model import (  # noqa: E402
    WARP,
    GpuModel,
    LatencyModel,
    OccupancyLimits,
    SaturationCurve,
    fit_hyperbolic,
)

# ======================================================================================
# CUDA / cpp source (the proven sm_120 split: torch-free kernels in CU, torch wrappers in CPP)
# ======================================================================================
CU = r"""
#include <cstdint>
#include <cuda_runtime.h>

#ifndef PROBE_U
#define PROBE_U 1
#endif

// ---- streaming read-reduce: grid-stride float4, compile-time ILP = PROBE_U ----
// The dynamic shared-memory reservation is a LAUNCH parameter (3rd <<<>>> arg); it is never
// touched by the code -- its only job is to throttle co-resident blocks/SM for the occupancy
// sweep. The accumulator flows to out[] under an impossible predicate (anti-DCE sink) so every
// load stays live; the buffer is read exactly once (grid-stride partition) => traffic = n*16 B.
__global__ void read_reduce_k(const float4* __restrict__ in, float* __restrict__ out,
                              long long n){
    long long stride = (long long)gridDim.x * blockDim.x;
    long long base   = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    float4 acc = make_float4(0.f,0.f,0.f,0.f);
    // unroll 1 on the OUTER loop so PROBE_U is the SOLE in-flight-load knob: without this -O3
    // auto-unrolls the grid stride and injects hidden ILP that masks the occupancy axis (the
    // probe then saturates below 2 warps/SM and the curve has no measurable rising limb to fit).
    #pragma unroll 1
    for(long long i = base; i < n; i += stride * PROBE_U){
        #pragma unroll
        for(int p = 0; p < PROBE_U; p++){
            long long j = i + (long long)p * stride;
            if(j < n){ float4 v = in[j]; acc.x+=v.x; acc.y+=v.y; acc.z+=v.z; acc.w+=v.w; }
        }
    }
    // impossible predicate: never stores, but the compiler cannot prove it, so all loads live.
    if(acc.x + acc.y + acc.z + acc.w == -123456.0f) out[base & 1023] = acc.x;
}

// ---- dependent pointer-chase: one thread, one outstanding load at a time (HBM latency) ----
__global__ void chase_k(const int64_t* __restrict__ nxt, int64_t* __restrict__ out,
                        int64_t start, long long hops){
    int64_t c = start;
    for(long long i = 0; i < hops; i++) c = nxt[c];  // each load depends on the previous
    *out = c;                                         // sink: defeat DCE
}

// ---- copy: dst[i]=src[i], 2x traffic anchor ----
__global__ void copy_k(const float* __restrict__ src, float* __restrict__ dst, long long n){
    long long stride = (long long)gridDim.x * blockDim.x;
    for(long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        dst[i] = src[i];
}

extern "C" {

// opt into a larger dynamic-SMEM ceiling so the throttle can reserve > 48 KB (returns cudaError).
int probe_opt_in(int bytes){
    return (int)cudaFuncSetAttribute((const void*)read_reduce_k,
            cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}
// driver occupancy: max active blocks/SM for read_reduce_k at (tpb, dyn_smem). Ground truth.
int probe_occ_blocks(int tpb, int dyn_smem){
    int b = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, (const void*)read_reduce_k, tpb,
                                                  (size_t)dyn_smem);
    return b;
}
void probe_launch(const void* in, void* out, long long n, int grid, int tpb, int dyn_smem){
    read_reduce_k<<<grid, tpb, (size_t)dyn_smem>>>((const float4*)in, (float*)out, n);
}
void probe_chase(const void* nxt, void* out, long long start, long long hops){
    chase_k<<<1,1>>>((const int64_t*)nxt, (int64_t*)out, start, hops);
}
void probe_copy(const void* src, void* dst, long long n, int grid, int tpb){
    copy_k<<<grid, tpb>>>((const float*)src, (float*)dst, n);
}
// device attributes straight from the driver (self-retargeting; same file runs on any GPU).
void dev_attr(int* o){
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&o[0], cudaDevAttrMultiProcessorCount,            dev);
    cudaDeviceGetAttribute(&o[1], cudaDevAttrMaxSharedMemoryPerMultiprocessor,dev);
    cudaDeviceGetAttribute(&o[2], cudaDevAttrMaxThreadsPerMultiProcessor,    dev);
    cudaDeviceGetAttribute(&o[3], cudaDevAttrL2CacheSize,                    dev);
    cudaDeviceGetAttribute(&o[4], cudaDevAttrMaxRegistersPerMultiprocessor,  dev);
    cudaDeviceGetAttribute(&o[5], cudaDevAttrMaxBlocksPerMultiprocessor,     dev);
    cudaDeviceGetAttribute(&o[6], cudaDevAttrMaxSharedMemoryPerBlockOptin,   dev);
    cudaDeviceGetAttribute(&o[7], cudaDevAttrWarpSize,                       dev);
}

} // extern "C"
"""

CPP = r"""
#include <torch/extension.h>
#include <cstdint>
extern "C" {
    int  probe_opt_in(int);
    int  probe_occ_blocks(int,int);
    void probe_launch(const void*, void*, long long, int, int, int);
    void probe_chase(const void*, void*, long long, long long);
    void probe_copy(const void*, void*, long long, int, int);
    void dev_attr(int*);
}
int64_t opt_in(int64_t bytes){ return (int64_t)probe_opt_in((int)bytes); }
int64_t occ(int64_t tpb, int64_t smem){ return (int64_t)probe_occ_blocks((int)tpb,(int)smem); }
void launch(torch::Tensor in, torch::Tensor out, int64_t n, int64_t grid, int64_t tpb, int64_t smem){
    probe_launch(in.data_ptr(), out.data_ptr(), (long long)n, (int)grid, (int)tpb, (int)smem);
}
void chase(torch::Tensor nxt, torch::Tensor out, int64_t start, int64_t hops){
    probe_chase(nxt.data_ptr(), out.data_ptr(), (long long)start, (long long)hops);
}
void copy(torch::Tensor src, torch::Tensor dst, int64_t n, int64_t grid, int64_t tpb){
    probe_copy(src.data_ptr(), dst.data_ptr(), (long long)n, (int)grid, (int)tpb);
}
torch::Tensor dev_attrs(){
    auto t = torch::zeros({8}, torch::TensorOptions().dtype(torch::kInt32));
    dev_attr(t.data_ptr<int>());
    return t;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("opt_in", &opt_in);
    m.def("occ", &occ);
    m.def("launch", &launch);
    m.def("chase", &chase);
    m.def("copy", &copy);
    m.def("dev_attrs", &dev_attrs);
}
"""

_EXT_CACHE: dict[int, object] = {}


def build_probe(U: int):
    """JIT-compile read_reduce/chase/copy with compile-time ILP ``-DPROBE_U=U`` on sm_120.
    Cached per U (a unique extension ``name`` per variant, as the proto requires)."""
    U = int(U)
    if U in _EXT_CACHE:
        return _EXT_CACHE[U]
    ext = load_inline(
        name=f"forge_bwocc_U{U}",
        cpp_sources=CPP,
        cuda_sources=CU,
        extra_cflags=["/Zc:preprocessor"],
        extra_cuda_cflags=["-gencode=arch=compute_120,code=sm_120", "-std=c++17",
                           "--expt-relaxed-constexpr", "-O3", f"-DPROBE_U={U}"],
        verbose=False,
    )
    _EXT_CACHE[U] = ext
    return ext


# ======================================================================================
# device attributes (runtime ground truth)
# ======================================================================================
def dev_attrs(ext) -> dict:
    """num_sms / smem_per_sm / max_threads_per_sm / l2_bytes / regs_per_sm / max_blocks_per_sm /
    smem_optin / warp via cudaDeviceGetAttribute. Live values, not the static GpuTarget record."""
    a = ext.dev_attrs().tolist()
    return {
        "num_sms": int(a[0]), "smem_per_sm": int(a[1]), "max_threads_per_sm": int(a[2]),
        "l2_bytes": int(a[3]), "regs_per_sm": int(a[4]), "max_blocks_per_sm": int(a[5]),
        "smem_optin": int(a[6]), "warp": int(a[7] or WARP),
    }


# ======================================================================================
# timing helpers (drift-robust; interleaved round-robin for A/B-in-one-window)
# ======================================================================================
def _sweep_interleaved(fns, reps: int, warmup: int = 6) -> list[float]:
    """Time many callables round-robin in ONE window. Returns the per-fn MEDIAN ms. Interleaving
    makes the 8-13% laptop drift hit every point equally within each rep, so the curve SHAPE is
    drift-robust even though absolute GB/s breathes with the boost clock."""
    for f in fns:
        for _ in range(max(0, warmup)):
            f()
    torch.cuda.synchronize()
    samples = [[] for _ in fns]
    for _ in range(max(1, reps)):
        for k, f in enumerate(fns):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); f(); e.record(); e.synchronize()
            samples[k].append(s.elapsed_time(e))
    return [statistics.median(x) for x in samples]


# ======================================================================================
# occupancy lattice (driver-verified)
# ======================================================================================
_TPB_GRID = [64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024]


def cap_smem_for_blocks(ext, tpb: int, target_b: int, optin_max: int) -> int | None:
    """Binary-search the smallest dynamic-SMEM reservation that forces driver occupancy to EXACTLY
    ``target_b`` blocks/SM. Returns the smem bytes, or ``None`` if ``target_b`` is unreachable at
    this tpb (register/thread cap lands above OR a granularity step skips it)."""
    if int(ext.occ(tpb, 0)) < target_b:
        return None  # even zero smem is already below target_b (reg/thread-bound lower)
    if int(ext.occ(tpb, optin_max)) > target_b:
        return None  # even max smem cannot throttle this tpb down to target_b
    lo, hi = 0, optin_max
    while lo < hi:                       # smallest smem with occ <= target_b
        mid = (lo + hi) // 2
        if int(ext.occ(tpb, mid)) <= target_b:
            hi = mid
        else:
            lo = mid + 1
    return lo if int(ext.occ(tpb, lo)) == target_b else None


def verified_occupancy_lattice(ext, attrs: dict) -> list[tuple[int, int, int, int]]:
    """Sweep ``tpb x target_blocks`` and keep ONLY points the driver confirms (occ == intended).
    Returns sorted ``(warps_per_sm, tpb, blocks, smem)``. The x-axis (warps/SM) is taken from the
    driver-verified occupancy, never guessed."""
    optin = attrs["smem_optin"]
    ext.opt_in(optin)  # allow the throttle to reserve the full opt-in partition
    max_threads = attrs["max_threads_per_sm"]
    max_blocks = attrs["max_blocks_per_sm"]
    lattice: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for tpb in _TPB_GRID:
        wpb = tpb // WARP
        if wpb <= 0:
            continue
        cap = min(max_threads // tpb, max_blocks)     # most blocks this tpb can co-reside
        nat = int(ext.occ(tpb, 0))                    # natural (reg/thread) occupancy at 0 smem
        cap = min(cap, nat)
        for b in range(1, cap + 1):
            smem = cap_smem_for_blocks(ext, tpb, b, optin)
            if smem is None:
                continue
            if int(ext.occ(tpb, smem)) != b:          # double-check the driver agrees
                continue
            warps = b * wpb
            key = (warps, tpb)
            if key in seen:
                continue
            seen.add(key)
            lattice.append((warps, tpb, b, smem))
    lattice.sort()
    return lattice


def sweep_saturation_curve(ext, lattice, buf, n_float4: int, num_sms: int,
                           reps: int = 80) -> list[tuple[float, float]]:
    """Persistent grid = blocks*num_sms (one steady wave, zero tail). Interleaved round-robin
    CUDA-event median over ``reps``. Returns ``(warps_per_sm, gbps)`` for every lattice point.
    GB/s = buffer_bytes / median_s (the grid-stride reads the buffer exactly once)."""
    buf_bytes = n_float4 * 16
    out = torch.zeros(1024, dtype=torch.float32, device="cuda")
    fns = []
    meta = []
    for (warps, tpb, b, smem) in lattice:
        grid = b * num_sms
        fns.append(lambda b=buf, o=out, n=n_float4, g=grid, t=tpb, s=smem:
                   ext.launch(b, o, n, g, t, s))
        meta.append((warps, tpb, b, smem))
    med_ms = _sweep_interleaved(fns, reps=reps, warmup=8)
    curve = []
    for (warps, tpb, b, smem), ms in zip(meta, med_ms):
        gbs = buf_bytes / (ms * 1e-3) / 1e9
        curve.append((float(warps), float(gbs)))
    return curve


def ilp_sweep(attrs: dict, buf_bytes: int = 768 * 1024 * 1024,
              us: tuple[int, ...] = (1, 2, 4, 8, 16), reps: int = 60) -> dict[int, float]:
    """Rebuild read_reduce with ``-DPROBE_U`` in ``us`` pinned NEAR-MAX occupancy; same interleaved
    timing. The orthogonal Little's-law axis: per-thread in-flight loads. Returns ``{U: gbps}``."""
    num_sms = attrs["num_sms"]
    n_float4 = (buf_bytes // 16)
    big = torch.empty(n_float4 * 4, dtype=torch.float32, device="cuda").normal_()
    out = torch.zeros(1024, dtype=torch.float32, device="cuda")
    tpb = 256
    grid = num_sms * max(1, attrs["max_threads_per_sm"] // tpb)  # fill to the thread cap
    fns, order = [], []
    for U in us:
        ext = build_probe(U)
        ext.opt_in(attrs["smem_optin"])
        fns.append(lambda e=ext, b=big, o=out, n=n_float4, g=grid, t=tpb: e.launch(b, o, n, g, t, 0))
        order.append(U)
    med = _sweep_interleaved(fns, reps=reps, warmup=8)
    res = {U: (n_float4 * 16) / (ms * 1e-3) / 1e9 for U, ms in zip(order, med)}
    del big
    torch.cuda.empty_cache()
    return res


def consistency_crosscheck(ext, attrs: dict, buf, n_float4: int,
                           target_warps: int = 16, reps: int = 80) -> dict:
    """M3: reach the SAME warps/SM via several (tpb, blocks) and time them in ONE interleaved
    window. If medians agree (~5%), OCCUPANCY (not raw block count/size) drives the curve."""
    num_sms = attrs["num_sms"]
    optin = attrs["smem_optin"]
    ext.opt_in(optin)
    out = torch.zeros(1024, dtype=torch.float32, device="cuda")
    buf_bytes = n_float4 * 16
    cfgs = []
    for tpb in (128, 256, 512, 768, 1024):
        wpb = tpb // WARP
        if target_warps % wpb != 0:
            continue
        b = target_warps // wpb
        if b < 1 or b > min(attrs["max_threads_per_sm"] // tpb, attrs["max_blocks_per_sm"]):
            continue
        smem = cap_smem_for_blocks(ext, tpb, b, optin)
        if smem is None or int(ext.occ(tpb, smem)) != b:
            continue
        cfgs.append((tpb, b, smem))
    fns = [lambda b=buf, o=out, n=n_float4, g=bk * num_sms, t=tpb, s=smem:
           ext.launch(b, o, n, g, t, s) for (tpb, bk, smem) in cfgs]
    med = _sweep_interleaved(fns, reps=reps, warmup=8) if fns else []
    gbs = [buf_bytes / (ms * 1e-3) / 1e9 for ms in med]
    spread = (max(gbs) - min(gbs)) / statistics.median(gbs) if gbs else float("nan")
    return {"target_warps": target_warps,
            "configs": [{"tpb": t, "blocks": b, "smem": s, "gbs": round(g, 1)}
                        for (t, b, s), g in zip(cfgs, gbs)],
            "spread_frac": spread}


# ======================================================================================
# HBM latency (pointer-chase) + Little's law
# ======================================================================================
def make_cycle(n: int, seed: int = 1) -> torch.Tensor:
    """Random single Hamiltonian cycle as an int64 successor map on the GPU. Linking
    perm[0]->perm[1]->...->perm[n-1]->perm[0] is always ONE cycle for any permutation, and a
    vectorized rng.permutation is ~C speed. The random successor map defeats the HW prefetcher."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).astype(np.int64)
    succ = np.empty(n, dtype=np.int64)
    succ[perm[:-1]] = perm[1:]
    succ[perm[-1]] = perm[0]
    return torch.from_numpy(succ).to("cuda")


def measure_hbm_latency(ext, attrs: dict, hops: int = 1 << 21, reps: int = 20) -> tuple[float, dict]:
    """Working-set sweep 32 KB -> 384 MB single-thread dependent chase -> L1/L2/HBM plateaus.
    HBM latency = median ns/hop over the largest 3 (> L2) working sets (Little's-law L).

    HOPS must be large enough that the chase's TOUCHED footprint (hops random 128 B lines)
    exceeds L2, else reps 2+ replay from L2 and report L2 latency, not HBM. hops=2^21 => ~256 MB
    touched > the 64 MB L2 (this was a real bug at 2^18: ~33 MB touched < L2 => 133 ns L2 number)."""
    l2_mb = attrs["l2_bytes"] / 1e6
    sizes_mb = [0.03, 0.25, 1, 4, 16, 48, 96, 192, 384]
    out = torch.zeros(1, dtype=torch.int64, device="cuda")
    table = {}
    for mb in sizes_mb:
        n = max(1024, int(mb * 1024 * 1024 / 8))
        succ = make_cycle(n, seed=1)
        med_ms, _ = _event_time_ms(lambda: ext.chase(succ, out, 0, hops), warmup=4, iters=reps)
        ns_hop = med_ms * 1e6 / hops
        table[mb] = ns_hop
        del succ
        torch.cuda.empty_cache()
    over_l2 = [mb for mb in sizes_mb if mb > l2_mb]
    big3 = sorted(over_l2)[-3:] if len(over_l2) >= 3 else over_l2
    latency = float(statistics.median([table[mb] for mb in big3]))
    return latency, {"ns_per_hop_by_mb": {str(k): round(v, 2) for k, v in table.items()},
                     "l2_mb": round(l2_mb, 1), "plateau_sets_mb": big3, "hops": hops}


# ======================================================================================
# copy anchor + L2 contamination guard
# ======================================================================================
def measure_copy_anchor(ext, attrs: dict, n_float4: int, reps: int = 80) -> dict:
    """copy_k (2x bytes) at near-max occupancy vs torch dst.copy_ vs eval.peak_bandwidth peak.
    Then the HALVE-buffer read guard: re-time the read plateau at half size (still > L2) -- GB/s
    must NOT jump (if it does, L2 is serving hits and the streaming number is rejected). Also a
    BELOW-L2 positive control that SHOULD jump, to prove the guard can see L2."""
    num_sms = attrs["num_sms"]
    buf_bytes = n_float4 * 16
    nf = n_float4 * 4  # fp32 element count
    src = torch.empty(nf, dtype=torch.float32, device="cuda").normal_()
    dst = torch.empty(nf, dtype=torch.float32, device="cuda")
    tpb = 256
    grid = num_sms * max(1, attrs["max_threads_per_sm"] // tpb)
    copy_med, _ = _event_time_ms(lambda: ext.copy(src, dst, nf, grid, tpb), warmup=8, iters=reps)
    copy_gbs = (2 * nf * 4) / (copy_med * 1e-3) / 1e9
    tcopy_med, _ = _event_time_ms(lambda: dst.copy_(src), warmup=8, iters=reps)
    tcopy_gbs = (2 * nf * 4) / (tcopy_med * 1e-3) / 1e9
    # peak_bandwidth reference (D2D / triad best-boosted)
    try:
        pb = measure_peak_bandwidth(buffer_bytes=1 << 30, warmup=20, iters=60).peak_gbs
    except Exception as e:  # pragma: no cover
        pb = float("nan")
        print(f"  [copy_anchor] peak_bandwidth ref failed: {e}")

    ext.opt_in(attrs["smem_optin"])
    out = torch.zeros(1024, dtype=torch.float32, device="cuda")

    def read_gbs(nbytes: int) -> float:
        nf4 = nbytes // 16
        b = torch.empty(nf4 * 4, dtype=torch.float32, device="cuda").normal_()
        g = num_sms * max(1, attrs["max_threads_per_sm"] // tpb)
        med, _ = _event_time_ms(lambda: ext.launch(b, out, nf4, g, tpb, 0), warmup=8, iters=reps)
        del b
        torch.cuda.empty_cache()
        return nbytes / (med * 1e-3) / 1e9

    full_gbs = read_gbs(buf_bytes)
    half_gbs = read_gbs(buf_bytes // 2)            # still > L2: must NOT jump
    sub_l2 = int(attrs["l2_bytes"] // 2)           # < L2: SHOULD jump (positive control)
    sub_l2 -= sub_l2 % 16
    subl2_gbs = read_gbs(max(16 * 1024, sub_l2))
    del src, dst
    torch.cuda.empty_cache()
    jump = (half_gbs - full_gbs) / full_gbs if full_gbs else float("nan")
    return {"copy_k_gbs": round(copy_gbs, 1), "torch_copy_gbs": round(tcopy_gbs, 1),
            "peak_bandwidth_ref_gbs": round(pb, 1) if pb == pb else None,
            "read_full_gbs": round(full_gbs, 1), "read_half_gbs": round(half_gbs, 1),
            "halve_jump_frac": round(jump, 3), "read_sub_l2_gbs": round(subl2_gbs, 1),
            "l2_contaminated": bool(jump > 0.15)}


# ======================================================================================
# orchestration
# ======================================================================================
def profile_gpu(reps: int = 80, knee_frac: float = 0.95, ilp_curve: int = 1,
                buf_mb: int = 768, save: str | None = None, verbose: bool = True) -> GpuModel:
    """Orchestrate every probe, fit the saturation curve, fill OccupancyLimits LIVE, compose a
    :class:`GpuModel`, optionally save, and return it.

    ``ilp_curve`` is the compile-time ILP used for the saturation sweep (default 1 = minimal,
    most-conservative MLP per thread; the orthogonal ILP axis is reported separately)."""
    assert torch.cuda.is_available(), "profile_gpu needs a CUDA GPU (no fabrication)"
    p = torch.cuda.get_device_properties(0)
    if verbose:
        print(f"[forge] GPU={p.name} cc={p.major}.{p.minor} clocks: {_clock_state()}")

    ext = build_probe(ilp_curve)
    attrs = dev_attrs(ext)
    if verbose:
        print(f"[forge] driver attrs: {attrs}")
    l2 = attrs["l2_bytes"]
    buf_bytes = buf_mb * 1024 * 1024
    assert buf_bytes >= 8 * l2, f"buffer {buf_bytes} < 8x L2 {8*l2} (would measure L2, not HBM)"
    n_float4 = buf_bytes // 16

    # ---- M1: occupancy saturation sweep -------------------------------------------
    lattice = verified_occupancy_lattice(ext, attrs)
    if verbose:
        print(f"[forge] verified lattice: {len(lattice)} points, "
              f"warps/SM {lattice[0][0]}..{lattice[-1][0]}")
    big = torch.empty(n_float4 * 4, dtype=torch.float32, device="cuda").normal_()
    curve_samples = sweep_saturation_curve(ext, lattice, big, n_float4, attrs["num_sms"], reps=reps)
    peak, warps_half, r2 = fit_hyperbolic(curve_samples)
    achieved_peak = max(g for _, g in curve_samples)
    if verbose:
        print(f"[forge] fit: peak={peak:.1f} GB/s warps_half={warps_half:.3f} r2={r2:.4f} "
              f"(achieved max sample {achieved_peak:.1f})")

    # ---- M3: occupancy-consistency cross-check ------------------------------------
    cross = consistency_crosscheck(ext, attrs, big, n_float4, target_warps=16, reps=reps)
    if verbose:
        print(f"[forge] M3 crosscheck @16 warps/SM spread={cross['spread_frac']:.3f}: "
              f"{cross['configs']}")
    del big
    torch.cuda.empty_cache()

    # ---- M2: ILP sweep ------------------------------------------------------------
    ilp = ilp_sweep(attrs, buf_bytes=buf_bytes, reps=max(40, reps // 2))
    if verbose:
        print(f"[forge] M2 ILP sweep (near-max occ): {{U: GB/s}} = "
              f"{{{', '.join(f'{u}:{g:.0f}' for u,g in ilp.items())}}}")

    # ---- M4: copy anchor + L2 guard -----------------------------------------------
    anchor = measure_copy_anchor(ext, attrs, n_float4, reps=max(40, reps // 2))
    if verbose:
        print(f"[forge] M4 anchor: {anchor}")

    # ---- M5: HBM latency ----------------------------------------------------------
    latency_ns, lat_table = measure_hbm_latency(ext, attrs)
    if verbose:
        print(f"[forge] M5 latency={latency_ns:.1f} ns (over {lat_table['plateau_sets_mb']} MB); "
              f"table={lat_table['ns_per_hop_by_mb']}")

    # ---- live occupancy limits (driver truth; the 1536-vs-2048 record drift lives here) ----
    occ = OccupancyLimits(
        max_warps_per_sm=attrs["max_threads_per_sm"] // WARP,
        max_blocks_per_sm=attrs["max_blocks_per_sm"],
        regs_per_sm=attrs["regs_per_sm"],
        smem_bytes_per_sm=attrs["smem_per_sm"],
    )

    # ---- compose curve + Little's-law latency model -------------------------------
    curve = SaturationCurve(peak_bw_gbs=peak, warps_half=warps_half, knee_frac=knee_frac,
                            samples=curve_samples, r2=r2, achieved_peak_gbs=achieved_peak)
    knee = curve.warps_knee  # empirical: min warps reaching knee_frac of ACHIEVED peak
    # bytes_in_flight_per_warp calibrated so predicted_knee == the empirical curve knee on THIS GPU
    # (Little's law at the ACHIEVED bandwidth, the physical bytes actually in flight).
    per_sm_inflight = (achieved_peak * 1e9 / attrs["num_sms"]) * (latency_ns * 1e-9)
    bytes_per_warp = per_sm_inflight / knee if knee > 0 else float("nan")
    latency = LatencyModel(hbm_latency_ns=latency_ns, bytes_in_flight_per_warp=bytes_per_warp)
    mlp_total = latency.bytes_in_flight(achieved_peak)

    # ---- compose the model (target: prefer the matching live-named record) ---------
    name = p.name.lower()
    tgt_key = next((k for k in TARGETS if k.replace("rtx", "") in name.replace(" ", "")), None)
    if tgt_key is None:
        tgt_key = "rtx5090" if "5090" in name else None
    target = TARGETS.get(tgt_key) if tgt_key else None
    if target is None and TARGETS:
        target = TARGETS["rtx5090"]

    note = (f"measured probe={'float4_stream'} ILP={ilp_curve}; achieved_peak={achieved_peak:.0f} "
            f"GB/s; knee={knee:.1f} warps/SM; 8warp%={curve.fraction_at(8)*100:.1f}; "
            f"L2 contamination={anchor['l2_contaminated']}; ILP_sweep={ilp}; "
            f"M3_spread={cross['spread_frac']:.3f}")
    gm = GpuModel(target=target, occ=occ, curve=curve, latency=latency,
                  probe_access_pattern="float4_stream", probe_ilp=ilp_curve,
                  measured_on=f"{p.name} cc{p.major}.{p.minor}", note=note)

    # stash the raw probe detail alongside the model (for the run report / auditing)
    gm._probe_detail = {  # type: ignore[attr-defined]
        "attrs": attrs, "lattice_points": len(lattice), "achieved_peak_gbs": achieved_peak,
        "ilp_sweep": ilp, "anchor": anchor, "latency_table": lat_table, "crosscheck": cross,
        "mlp_bytes_total": mlp_total, "bytes_in_flight_per_warp": bytes_per_warp,
        "clock_state": _clock_state(),
    }

    if save:
        gm.save(save)
        if verbose:
            print(f"[forge] saved model -> {save}")
    return gm


__all__ = [
    "build_probe", "dev_attrs", "verified_occupancy_lattice", "cap_smem_for_blocks",
    "sweep_saturation_curve", "ilp_sweep", "consistency_crosscheck", "make_cycle",
    "measure_hbm_latency", "measure_copy_anchor", "profile_gpu", "CU", "CPP",
]
