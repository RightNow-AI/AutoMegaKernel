"""Prototype: HBM dependent-load latency + MLP (Little's law) probe on the live GPU.

Validates the Probe-B design before it is hardened into forge/. Measures:
  1) working-set sweep (pointer-chase over growing buffers) -> reveals L1/L2/HBM plateaus
  2) dependent-load latency on the HBM plateau (single thread, single in-flight load)
  3) read bandwidth vs (warps/SM x ILP) -> saturation knee
  4) Little's law: bytes_in_flight_to_saturate = peak_BW * latency, and the MLP requirement.

Run: ./.venv/Scripts/python.exe forge/_scratch/probe_latency_proto.py
"""
import sys, time, numpy as np
sys.path.insert(0, "C:/Users/jaber/RightNow-Full/AutoMegaKernel")
from vm.loader import _ensure_msvc_on_path
_ensure_msvc_on_path()
import torch
from torch.utils.cpp_extension import load_inline

assert torch.cuda.is_available()
dev = torch.device("cuda")
props = torch.cuda.get_device_properties(0)
print(f"GPU={props.name} SMs={props.multi_processor_count} "
      f"L2={getattr(props,'L2_cache_size',0)/1e6:.1f}MB "
      f"cc={props.major}.{props.minor}")

CUDA = r"""
#include <cstdint>
// ---- dependent pointer-chase: one thread, one outstanding load at a time ----
__global__ void chase_k(const int64_t* __restrict__ nxt, int64_t* __restrict__ out,
                        int64_t start, long long hops){
    int64_t c = start;
    for(long long i=0;i<hops;i++) c = nxt[c];   // each load depends on the previous
    *out = c;                                    // sink: defeat DCE
}
// ---- streaming read with compile-time ILP P: P independent loads in flight/thread ----
template<int P>
__global__ void stream_k(const float4* __restrict__ in, float* __restrict__ out, long long n){
    long long stride = (long long)gridDim.x * blockDim.x;
    long long base   = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    float4 acc = make_float4(0,0,0,0);
    for(long long i=base; i<n; i += stride*P){
        #pragma unroll
        for(int p=0;p<P;p++){
            long long j = i + (long long)p*stride;
            if(j<n){ float4 v = in[j]; acc.x+=v.x; acc.y+=v.y; acc.z+=v.z; acc.w+=v.w; }
        }
    }
    if(acc.x == -123456.0f) out[base] = acc.x;   // sink: never true, keeps all loads
}
void launch_chase(const void* nxt, void* out, long long start, long long hops){
    chase_k<<<1,1>>>((const int64_t*)nxt,(int64_t*)out,start,hops);
}
void launch_stream(int P, const void* in, void* out, long long n, int grid, int block){
    switch(P){
        case 1: stream_k<1><<<grid,block>>>((const float4*)in,(float*)out,n); break;
        case 2: stream_k<2><<<grid,block>>>((const float4*)in,(float*)out,n); break;
        case 4: stream_k<4><<<grid,block>>>((const float4*)in,(float*)out,n); break;
        case 8: stream_k<8><<<grid,block>>>((const float4*)in,(float*)out,n); break;
    }
}
"""
CPP = r"""
#include <torch/extension.h>
void launch_chase(const void*, void*, long long, long long);
void launch_stream(int, const void*, void*, long long, int, int);
void chase(torch::Tensor nxt, torch::Tensor out, long long start, long long hops){
    launch_chase(nxt.data_ptr(), out.data_ptr(), start, hops);
}
void stream(long long P, torch::Tensor in, torch::Tensor out, long long n, long long grid, long long block){
    launch_stream((int)P, in.data_ptr(), out.data_ptr(), n, (int)grid, (int)block);
}
"""
m = load_inline(
    name="forge_probe_b_proto",
    cpp_sources=CPP, cuda_sources=CUDA,
    functions=["chase","stream"],
    extra_cflags=["/Zc:preprocessor"],
    extra_cuda_cflags=["-gencode=arch=compute_120,code=sm_120","-std=c++17",
                       "--expt-relaxed-constexpr","-O3"],
    verbose=False,
)

def ev_time_ms(fn, reps=100, warmup=10):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))

def make_cycle(n, seed=0):
    """Random single Hamiltonian cycle as an int64 successor map, on GPU.
    Linking perm[0]->perm[1]->...->perm[n-1]->perm[0] is ALWAYS one cycle for any
    permutation perm, so a C-level rng.permutation (fast, no python loop) suffices."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).astype(np.int64)   # vectorized, ~C speed
    succ = np.empty(n, dtype=np.int64)
    succ[perm[:-1]] = perm[1:]
    succ[perm[-1]] = perm[0]
    return torch.from_numpy(succ).to(dev)

out = torch.zeros(1, dtype=torch.int64, device=dev)

# ---- 1) working-set sweep: latency vs buffer size (find HBM plateau) ----
print("\n[1] working-set sweep (ns/hop):")
HOPS = 1<<21
sizes_mb = [0.03, 0.25, 1, 4, 16, 48, 96, 192, 384]
lat_by_size = {}
for mb in sizes_mb:
    n = max(1024, int(mb*1024*1024/8))
    succ = make_cycle(n, seed=1)
    f = lambda: m.chase(succ, out, 0, HOPS)
    ms = ev_time_ms(f, reps=60, warmup=8)
    ns_hop = ms*1e6/HOPS
    lat_by_size[mb] = ns_hop
    print(f"   {mb:7.2f} MB  ({n:>10d} elts)  {ns_hop:7.2f} ns/hop")
    del succ; torch.cuda.empty_cache()

# plateau latency = median of the largest 3 working sets
plateau = float(np.median([lat_by_size[mb] for mb in sizes_mb[-3:]]))
print(f"   -> HBM plateau dependent-load latency ~= {plateau:.1f} ns")

# ---- 2) peak read bandwidth vs (block size x ILP) ----
print("\n[2] read bandwidth sweep GB/s  [rows=block, cols=ILP P]:")
NF4 = (512*1024*1024)//16            # float4 count in a 512MB buffer
big = torch.randn(NF4*4, dtype=torch.float32, device=dev)  # 512MB of float
sink = torch.zeros(1<<20, dtype=torch.float32, device=dev)
bytes_moved = NF4*16
SM = props.multi_processor_count
best = (0.0, None)
for block in [128, 256, 512, 1024]:
    row = []
    for P in [1,2,4,8]:
        grid = SM * max(1, 2048//block)  # fill the GPU; P adds in-flight loads/thread only
        grid = max(grid, SM)
        f = lambda P=P, grid=grid, block=block: m.stream(P, big, sink, NF4, grid, block)
        ms = ev_time_ms(f, reps=50, warmup=8)
        gbs = bytes_moved/(ms*1e-3)/1e9
        row.append(gbs)
        if gbs > best[0]: best = (gbs, (block,P,grid))
    print(f"   block={block:4d}: " + "  ".join(f"P{p}={g:6.1f}" for p,g in zip([1,2,4,8],row)))
peakbw, cfg = best
print(f"   -> peak read BW ~= {peakbw:.1f} GB/s at block={cfg[0]} ILP={cfg[1]} grid={cfg[2]}")

# ---- 3) Little's law / MLP requirement ----
print("\n[3] Little's law -> MLP requirement:")
bytes_in_flight = peakbw*1e9 * plateau*1e-9
print(f"   bytes_in_flight_to_saturate = {peakbw:.0f}e9 B/s * {plateau:.1f}e-9 s "
      f"= {bytes_in_flight/1024:.0f} KB")
for sector in (32, 128):
    req = bytes_in_flight/sector
    print(f"   ~= {req:,.0f} outstanding {sector}B requests "
          f"({req/SM:.0f}/SM over {SM} SMs)")
# AMK runs 8 warps/SM x 32 threads = 256 threads/SM. loads/thread @128B granule:
amk_threads_per_sm = 256
need_per_sm_128 = (bytes_in_flight/128)/SM
print(f"   AMK 8 warps/SM (256 thr): needs {need_per_sm_128/amk_threads_per_sm:.2f} "
      f"outstanding 128B loads/thread to saturate")
print("\nDONE")
