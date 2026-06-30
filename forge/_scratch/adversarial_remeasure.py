"""ADVERSARIAL independent re-measure of the bandwidth-vs-occupancy keystone.

Written from scratch (own kernel, own occupancy control, own timing, own analysis) to
INDEPENDENTLY reproduce the claimed RTX-5090-Laptop saturation curve, knee, 8-warp point,
HBM latency, and -- critically -- to falsify the "HBM-bound" claim (peak must not exceed the
~896 GB/s spec; halving the buffer must not raise GB/s => not serving from L2).

Run: ./.venv/Scripts/python.exe forge/_scratch/adversarial_remeasure.py
"""
import sys, statistics, json, time
import numpy as np
sys.path.insert(0, "C:/Users/jaber/RightNow-Full/AutoMegaKernel")
from vm.loader import _ensure_msvc_on_path
_ensure_msvc_on_path()
import torch
from torch.utils.cpp_extension import load_inline
import subprocess

WARP = 32

def clocks():
    try:
        q = "clocks.mem,clocks.sm,temperature.gpu,clocks_throttle_reasons.active"
        p = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15)
        return p.stdout.strip()
    except Exception as e:
        return f"err {e}"

# ---- my own torch-free kernel: float4 grid-stride read-reduce, ILP=1, anti-DCE sink ----
CU = r"""
#include <cstdint>
#include <cuda_runtime.h>
// grid-stride float4 read; accumulate; store only under an impossible predicate so every load
// stays live (no DCE). dynamic smem is a launch param, never touched -- it only throttles occ.
__global__ void rr(const float4* __restrict__ in, float* __restrict__ out, long long n){
    long long stride = (long long)gridDim.x * blockDim.x;
    long long base   = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    float4 acc = make_float4(0.f,0.f,0.f,0.f);
    #pragma unroll 1
    for(long long i = base; i < n; i += stride){
        float4 v = in[i]; acc.x+=v.x; acc.y+=v.y; acc.z+=v.z; acc.w+=v.w;
    }
    if(acc.x+acc.y+acc.z+acc.w == -123456.0f) out[base & 1023] = acc.x;
}
// dependent pointer-chase: 1 thread, 1 in-flight load => unloaded HBM latency
__global__ void chase(const int64_t* __restrict__ nxt, int64_t* __restrict__ out,
                      int64_t start, long long hops){
    int64_t c = start;
    for(long long i=0;i<hops;i++) c = nxt[c];
    *out = c;
}
extern "C" {
int rr_optin(int bytes){
    return (int)cudaFuncSetAttribute((const void*)rr,
            cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}
int rr_occ(int tpb, int dsmem){
    int b=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b,(const void*)rr,tpb,(size_t)dsmem);
    return b;
}
void rr_launch(const void* in, void* out, long long n, int grid, int tpb, int dsmem){
    rr<<<grid,tpb,(size_t)dsmem>>>((const float4*)in,(float*)out,n);
}
void chase_launch(const void* nxt, void* out, long long start, long long hops){
    chase<<<1,1>>>((const int64_t*)nxt,(int64_t*)out,start,hops);
}
void dattr(int* o){
    int d=0; cudaGetDevice(&d);
    cudaDeviceGetAttribute(&o[0],cudaDevAttrMultiProcessorCount,d);
    cudaDeviceGetAttribute(&o[1],cudaDevAttrMaxSharedMemoryPerMultiprocessor,d);
    cudaDeviceGetAttribute(&o[2],cudaDevAttrMaxThreadsPerMultiProcessor,d);
    cudaDeviceGetAttribute(&o[3],cudaDevAttrL2CacheSize,d);
    cudaDeviceGetAttribute(&o[4],cudaDevAttrMaxBlocksPerMultiprocessor,d);
    cudaDeviceGetAttribute(&o[5],cudaDevAttrMaxSharedMemoryPerBlockOptin,d);
}
}
"""
CPP = r"""
#include <torch/extension.h>
#include <cstdint>
extern "C" {
  int  rr_optin(int);
  int  rr_occ(int,int);
  void rr_launch(const void*,void*,long long,int,int,int);
  void chase_launch(const void*,void*,long long,long long);
  void dattr(int*);
}
int64_t optin(int64_t b){ return (int64_t)rr_optin((int)b); }
int64_t occ(int64_t tpb,int64_t s){ return (int64_t)rr_occ((int)tpb,(int)s); }
void launch(torch::Tensor in,torch::Tensor out,int64_t n,int64_t g,int64_t tpb,int64_t s){
    rr_launch(in.data_ptr(),out.data_ptr(),(long long)n,(int)g,(int)tpb,(int)s);
}
void chase(torch::Tensor nxt,torch::Tensor out,int64_t st,int64_t h){
    chase_launch(nxt.data_ptr(),out.data_ptr(),(long long)st,(long long)h);
}
torch::Tensor attrs(){
    auto t=torch::zeros({6},torch::TensorOptions().dtype(torch::kInt32));
    dattr(t.data_ptr<int>()); return t;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("optin",&optin); m.def("occ",&occ); m.def("launch",&launch);
    m.def("chase",&chase); m.def("attrs",&attrs);
}
"""

print(f"[adv] building kernel... clocks: {clocks()}")
ext = load_inline(name="adv_remeasure_v1", cpp_sources=CPP, cuda_sources=CU,
                  extra_cflags=["/Zc:preprocessor"],
                  extra_cuda_cflags=["-gencode=arch=compute_120,code=sm_120","-std=c++17",
                                     "--expt-relaxed-constexpr","-O3"], verbose=False)
a = ext.attrs().tolist()
NUM_SMS, SMEM_SM, THR_SM, L2, MAXB, OPTIN = a[0],a[1],a[2],a[3],a[4],a[5]
print(f"[adv] driver attrs: num_sms={NUM_SMS} smem/SM={SMEM_SM} thr/SM={THR_SM} "
      f"L2={L2/1e6:.1f}MB max_blocks/SM={MAXB} smem_optin={OPTIN}")
ext.optin(OPTIN)

# ---------- interleaved round-robin timer (drift hits every point equally) ----------
def sweep(fns, reps, warmup=8):
    for f in fns:
        for _ in range(warmup): f()
    torch.cuda.synchronize()
    samp = [[] for _ in fns]
    for _ in range(reps):
        for k,f in enumerate(fns):
            s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
            s.record(); f(); e.record(); e.synchronize()
            samp[k].append(s.elapsed_time(e))
    return [statistics.median(x) for x in samp]

def time1(f, reps, warmup=8):
    for _ in range(warmup): f()
    torch.cuda.synchronize()
    xs=[]
    for _ in range(reps):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); f(); e.record(); e.synchronize(); xs.append(s.elapsed_time(e))
    return statistics.median(xs), min(xs)

# ---------- find min dynamic smem that forces EXACTLY b blocks/SM (driver-verified) ----------
def smem_for_blocks(tpb, b):
    if ext.occ(tpb,0) < b: return None
    if ext.occ(tpb,OPTIN) > b: return None
    lo,hi=0,OPTIN
    while lo<hi:
        mid=(lo+hi)//2
        if ext.occ(tpb,mid) <= b: hi=mid
        else: lo=mid+1
    return lo if ext.occ(tpb,lo)==b else None

# ================= MAIN SWEEP: single block-shape tpb=64 (2 warps/block), vary blocks =========
BUF_MB = 768
buf_bytes = BUF_MB*1024*1024
assert buf_bytes >= 8*L2, "buffer must be >= 8x L2 to be HBM-bound"
nf4 = buf_bytes//16
print(f"[adv] buffer {BUF_MB}MB = {buf_bytes/1e6:.0f}MB ({buf_bytes/L2:.1f}x L2); nf4={nf4}")
big = torch.empty(nf4*4, dtype=torch.float32, device="cuda").normal_()
out = torch.zeros(1024, dtype=torch.float32, device="cuda")

TPB = 64; WPB = TPB//WARP  # 2 warps/block
nat = ext.occ(TPB,0)
print(f"[adv] tpb={TPB}: natural occ at 0 smem = {nat} blocks/SM (=> max {nat*WPB} warps/SM)")
points = []
for b in range(1, nat+1):
    s = smem_for_blocks(TPB, b)
    if s is None: continue
    assert ext.occ(TPB,s)==b
    points.append((b*WPB, TPB, b, s))
print(f"[adv] occupancy lattice (tpb=64): {[p[0] for p in points]} warps/SM")

REPS=100
fns=[(lambda B=big,O=out,n=nf4,g=b*NUM_SMS,t=tpb,s=sm: ext.launch(B,O,n,g,t,s))
     for (w,tpb,b,sm) in points]
t0=time.time()
med = sweep(fns, REPS)
print(f"[adv] main sweep {len(points)} pts x {REPS} reps in {time.time()-t0:.1f}s; clocks: {clocks()}")
curve=[]
for (w,tpb,b,sm),ms in zip(points, med):
    gbs = buf_bytes/(ms*1e-3)/1e9
    curve.append((w,gbs))
    print(f"     {w:3d} warps/SM (b={b:2d} smem={sm:6d})  {ms*1e3:7.1f} us  {gbs:7.1f} GB/s")

achieved = max(g for _,g in curve)
# empirical knee = min warps reaching 95% of achieved peak (linear interp), repo definition
def emp_knee(curve, frac=0.95):
    by={}
    for w,g in curve: by.setdefault(w,[]).append(g)
    med=sorted((w, statistics.median(gs)) for w,gs in by.items())
    thr=frac*achieved; prev=None
    for w,g in med:
        if g>=thr:
            if prev is None: return float(w)
            pw,pg=prev
            return pw+(w-pw)*(thr-pg)/(g-pg) if g!=pg else float(w)
        prev=(w,g)
    return float(med[-1][0])
knee=emp_knee(curve)
g8 = dict((w,g) for w,g in curve).get(8, float('nan'))
print(f"\n[adv] ACHIEVED PEAK = {achieved:.1f} GB/s")
print(f"[adv] EMPIRICAL KNEE (95% of achieved) = {knee:.2f} warps/SM")
print(f"[adv] 8-warp point = {g8:.1f} GB/s = {100*g8/achieved:.1f}% of achieved peak")

# ================= HBM-BOUND GUARDS (the falsification tests) =================
print("\n[adv] === HBM-bound falsification ===")
# (1) peak vs spec
print(f"  achieved {achieved:.1f} GB/s vs 896 spec => ratio {achieved/896:.3f} "
      f"({'OK <1.0x, HBM-plausible' if achieved<=896 else 'EXCEEDS SPEC' if achieved>1075 else 'slightly over, watch'})")
# (2) buffer-size invariance: read GB/s at 1536/768/384 MB (ALL > L2) must NOT rise as size shrinks
tpb=256; grid = NUM_SMS*max(1, THR_SM//tpb)  # near-max occ for the size test
def read_gbs(mb):
    n=(mb*1024*1024)//16
    b=torch.empty(n*4,dtype=torch.float32,device="cuda").normal_()
    m,_=time1(lambda: ext.launch(b,out,n,grid,tpb,0), reps=60)
    del b; torch.cuda.empty_cache()
    return (mb*1024*1024)/(m*1e-3)/1e9
sizes=[1536,768,384,64,16]   # 64MB ~= L2, 16MB < L2 (positive controls that SHOULD cache)
gz={mb:read_gbs(mb) for mb in sizes}
for mb in sizes:
    tag = "> L2" if mb*1e6>L2 else "<= L2 (cache control)"
    print(f"  read {mb:5d}MB ({mb*1024*1024/L2:4.1f}x L2): {gz[mb]:7.1f} GB/s   {tag}")
jump_768_384 = (gz[384]-gz[768])/gz[768]
print(f"  halve 768->384MB jump = {jump_768_384*100:+.1f}%  "
      f"({'NO jump => HBM-bound OK' if jump_768_384<0.15 else 'JUMP => L2 CONTAMINATION'})")
print(f"  sub-L2 16MB vs 768MB ratio = {gz[16]/gz[768]:.2f}x (positive control; >1.3 proves the test can see cache)")

# ================= M3: occupancy NOT block-shape (8 warps three ways) =================
print("\n[adv] === M3: 8 warps/SM via 3 block-shapes (interleaved) ===")
m3cfgs=[]
for tpb,b in [(64,4),(128,2),(256,1)]:
    s=smem_for_blocks(tpb,b)
    if s is not None and ext.occ(tpb,s)==b: m3cfgs.append((tpb,b,s))
m3fns=[(lambda B=big,O=out,n=nf4,g=b*NUM_SMS,t=tpb,s=s: ext.launch(B,O,n,g,t,s)) for (tpb,b,s) in m3cfgs]
m3med=sweep(m3fns, 100)
m3g=[buf_bytes/(ms*1e-3)/1e9 for ms in m3med]
for (tpb,b,s),g in zip(m3cfgs,m3g):
    print(f"  tpb={tpb:4d} b={b} smem={s:6d} -> {g:7.1f} GB/s")
m3spread=(max(m3g)-min(m3g))/statistics.median(m3g)
print(f"  spread = {m3spread*100:.2f}%  ({'occupancy drives it' if m3spread<0.05 else 'block-shape matters!'})")

# ================= HBM latency (pointer-chase) + Little's law =================
print("\n[adv] === HBM dependent-load latency (pointer-chase) ===")
def make_cycle(n, seed=7):
    rng=np.random.default_rng(seed); perm=rng.permutation(n).astype(np.int64)
    succ=np.empty(n,dtype=np.int64); succ[perm[:-1]]=perm[1:]; succ[perm[-1]]=perm[0]
    return torch.from_numpy(succ).to("cuda")
HOPS=1<<21; o64=torch.zeros(1,dtype=torch.int64,device="cuda")
lat={}
for mb in [0.03,0.25,1,4,16,48,96,192,384]:
    n=max(1024,int(mb*1024*1024/8)); succ=make_cycle(n)
    m,_=time1(lambda: ext.chase(succ,o64,0,HOPS), reps=20, warmup=4)
    lat[mb]=m*1e6/HOPS; del succ; torch.cuda.empty_cache()
    print(f"  {mb:7.2f}MB  {lat[mb]:7.2f} ns/hop")
over=[mb for mb in lat if mb*1e6>L2]
latency=statistics.median([lat[mb] for mb in sorted(over)[-3:]])
print(f"  HBM latency (median of >L2 sets) = {latency:.1f} ns")
# Little's law cross-check of knee: bytes_in_flight = peak*latency; per warp from MEASURED knee
inflight_total = achieved*1e9 * latency*1e-9
per_sm = inflight_total/NUM_SMS
print(f"  Little: bytes_in_flight = {achieved:.0f}e9 * {latency:.0f}e-9 = {inflight_total/1024:.0f} KB "
      f"({per_sm:.0f} B/SM); at knee {knee:.1f} warps => {per_sm/knee:.0f} B/warp")

# ================= compare to CLAIM =================
claim={"peak":826.7,"knee":11.2,"g8":698.4,"g8_pct":84.5,"latency":327.3}
print("\n[adv] === vs CLAIM ===")
print(f"  peak    : mine {achieved:6.1f}  claim {claim['peak']:6.1f}  (absolute drifts ~20% w/ power cap)")
print(f"  knee    : mine {knee:6.2f}  claim {claim['knee']:6.2f} warps/SM")
print(f"  8-warp% : mine {100*g8/achieved:6.1f}  claim {claim['g8_pct']:6.1f} %")
print(f"  latency : mine {latency:6.1f}  claim {claim['latency']:6.1f} ns")

summary={"clocks_end":clocks(),"achieved_peak_gbs":achieved,"empirical_knee_warps":knee,
         "g8_gbs":g8,"g8_pct":100*g8/achieved,"latency_ns":latency,
         "halve_jump_frac":jump_768_384,"subL2_ratio":gz[16]/gz[768],"m3_spread":m3spread,
         "curve":curve,"read_by_mb":gz,"lat_by_mb":lat}
open("C:/Users/jaber/RightNow-Full/AutoMegaKernel/forge/_scratch/adv_remeasure.json","w").write(
    json.dumps(summary,indent=2,default=float))
print("\n[adv] wrote forge/_scratch/adv_remeasure.json")
