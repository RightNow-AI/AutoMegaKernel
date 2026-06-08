# GEMV Optimization Plan, Closing the Datacenter cuBLAS Gap at Batch-1 Decode

**Status:** scoping / engineering plan. No kernel code here, this is the prioritized work list a
senior CUDA engineer can execute from. Every item cites the real code it touches and states which
regime/lever it targets, expected impact, effort/risk, the ABI/validator constraint it must respect,
and whether it needs `ncu` to confirm.

**Scope:** the whole-model **single-token, batch-1 decode** megakernel on **datacenter GPUs**:
A100 (sm_80, 108 SMs, 1383 GB/s measured peak) and H100 (sm_90, 132 SMs, 3089 GB/s measured peak).
The consumer RTX 5090 (sm_120) is the dev baseline.

---

## 1. Problem statement and honest current standing

### 1.1 The regime (this is the whole game)

Batch-1 decode is **memory-bandwidth bound**. Each decoded token streams the *entire* weight set
through HBM exactly once. The latency floor is

```
t_floor = weight_bytes / HBM_bandwidth
```

Tensor-core FLOPs do **not** help a memory-bound GEMV: the matvec arithmetic intensity is ~2 FLOP
per weight byte (`est_flops = 2*K*N_tile`, `est_bytes = K*N_tile*dtype_bytes`, `schedule/lower.py`
`_Lowerer._gemv_tiles`), orders of magnitude below the tensor-core ridge point. The only levers that
move a bandwidth-bound GEMV are:

1. **Achieving near-peak HBM bandwidth**, driven by *memory-level parallelism* (MLP): the number of
   independent loads in flight per SM, which by Little's Law must cover `latency × bandwidth`.
2. **Removing non-load overhead**, per-tile grid sync, dispatch, and barriers that do not shrink
   when bandwidth rises.

**The one exception is the int8 weight-only path.** It reads ~4× fewer weight bytes (lower roofline
floor) but is believed **ALU-bound on per-element dequant** *and* **load-serial** (no weight staging -
see INT8-P0). There, the levers are decoupling weight loads (INT8-P0) and making dequant cheaper
(P5, DP4A / IMMA), to convert it from ALU-bound + load-serial back to bandwidth-bound so the 4× byte
saving actually pays off.

> **ncu-gated hypothesis (not a derived figure).** The often-quoted "int8 sits at ~35–37% of its own
> lower floor" **cannot be computed from the committed measurements**:
> `paper/results/int8_search_datacenter.json` stores the cuBLAS/AMK *ratio* (`ratio_median` A100
> ~0.4467, H100 ~0.2481) and `int8_floor_us` (A100 244.9, H100 113.7) but **not the absolute AMK
> microseconds**, so AMK-us ÷ int8_floor_us is not obtainable from it. To state a "% of int8 floor"
> we need the absolute AMK decode time recorded alongside; until then treat the ~35–37% as an
> ncu-gated hypothesis, not a measured fact.

### 1.2 Where we are (measured, honest)

| Platform | Path | Achieved vs cuBLAS-graphed | Notes |
|---|---|---|---|
| RTX 5090 (sm_120) | cuBLAS bf16 graph | ~77% of measured peak | the bar |
| RTX 5090 (sm_120) | AMK bf16 | ~69% of measured peak | ~0.9× cuBLAS, close on consumer |
| A100 (sm_80) | AMK int8 self-tuned, 4 layers | ~0.45× cuBLAS | gap is large and structural |
| A100 (sm_80) | AMK int8 self-tuned, 24 layers | ~0.76× cuBLAS | rises with depth, **plateaus below parity** |
| H100 (sm_90) | AMK int8 self-tuned, 4 layers | ~0.26× cuBLAS | worst gap |
| H100 (sm_90) | AMK int8 self-tuned, 24 layers | ~0.62× cuBLAS | **plateaus below parity** |

**The scale trend is the key diagnostic.** The ratio rises with model depth (more independent tiles
to balance over more SMs → more MLP and better load balance) but **plateaus below 1.0×**. Crucially,
the existing int8 datacenter knob search (`qc` / `N_tile` / `threads_per_block`, the grid actually
swept in `paper/results/int8_search_datacenter.json`; **note it did NOT sweep the cp.async ring,
because the quantized path never executes the ring**, see INT8-P0 / P0's caveat) does **not** cross
cuBLAS on datacenter. That confirms the int8 gap is **structural** (no weight-load decoupling on the
quant path + dequant-ALU cost + insufficient MLP per SM + fixed per-tile sync overhead that grows as a
fraction of the now-shorter compute time), not a tuning problem the *current* (qc/N_tile/threads)
search surface can solve.

Higher bandwidth makes both structural costs worse relative to compute: H100's 3089 GB/s drains a
thin tile ~4× faster than the 731 GB/s dev part, so (a) the fixed per-tile sync is a larger fraction
of the shorter tile, and (b) Little's-Law MLP demand is ~4× higher, which the current ring depth and
occupancy do not meet.

### 1.3 The structural diagnosis (from roofline + scale trend + code)

> **No hardware-counter data exists.** `ncu`/Nsight is blocked on the Modal account
> (`LibraryNotLoaded`). The diagnosis below reasons from the roofline, the depth-scaling trend, and
> the code. Each work item is tagged with whether it **NEEDS ncu** to *confirm* (vs. items measurable
> end-to-end by the existing wall-clock paired-interleaved harness).

**Diagnosis A, insufficient MLP per SM (the bf16/fp bandwidth lever).**
The register fast path (`vm/ops.cuh` `amk_gemv_rows_dot_f32` / `_bf16` / `_f16`, ~lines 232–316, and
the single-column `amk_gemv_row_dot_*`) loads one 16-byte vector per lane per K-step and immediately
consumes it in the FMA, so only `~COLS_PER_WARP * KUNROLL` loads are ever outstanding per warp
(defaults `cols_per_warp=1, kunroll=1`, `vm/loader.py` `_KNOB_DEFAULTS` line 182). The cp.async path
(`amk_gemv_tile_cpasync<WT,VEC>`, ops.cuh ~582–703) decouples load from compute with a
`AMK_GEMV_CPA_STAGES`-deep ring, but the default ring is only `STAGES=4`, `CPC=2`, `VPL=1`, so a warp
has at most `STAGES-1 = 3` future chunks in flight (`__pipeline_wait_prior`, ops.cuh ~654). The
ops.cuh comment itself (lines 181–189) states the register path "plateaus at ~48% of measured HBM
bandwidth"; that was tuned for the 731 GB/s dev part. At 1.37–3.05 TB/s the same in-flight depth is
far short of the working set Little's Law demands.

**Diagnosis B, fixed per-tile grid sync (the sync-frequency lever).**
Every GEMV tile pays one `amk_signal` (`vm/sync.cuh` 55–62: `__syncthreads` → `__threadfence` →
`atomicAdd` → `__syncthreads`) plus the consumer's `amk_wait_all` acquire-spin (sync.cuh 68–108).
`vm/scheduler.cu` (101–127) runs these serially per SM: `wait → dispatch → signal` per queued
instruction, and the trailing `__syncthreads` in `amk_signal` gates the *next* queue entry even when
it has no dependency. The lowering emits one counter per projection shared across all its tiles
(`lower.py` `_gemv_tiles` line 231) and `N_tile≈32` (lower.py `_DEFAULT_N_TILE_AUTO`, line 76) yields
~1500+ tiles per forward pass. This sync cost is *fixed in wall-clock* but is a rising fraction of the
shrinking per-tile compute time as bandwidth grows.

**Diagnosis C, int8 dequant is ALU-bound (the int8 lever).**
`amk_inst_gemv_tile_quant` (ops.cuh 370–550) already uses `QC` independent column accumulators
(`AMK_QGEMV_COLS`, default `qc=4`) and coalesced `uint4` (16 int8) loads. The bottleneck is the inner
chunk: 16 serial `(float)bX.{x,y,z,w} - z` int→float casts + 16 FMAs per 16-byte load (ops.cuh
470–477). The code comment (463–468) notes a 4-way accumulator variant was reverted as
occupancy-neutral, consistent with the kernel being dequant-ALU/occupancy bound, not FMA-latency
bound. **It is also load-serial:** the chunk `uint4` load is synchronous, so there is no weight
staging / load-compute overlap (no `__pipeline_memcpy_async` in this op, see INT8-P0). The frequently
quoted "int8 sits at ~35–37% of its *own* (4× lower) floor" is an **ncu-gated hypothesis**, not a
derived number: the committed `int8_search_datacenter.json` records the ratio and `int8_floor_us` but
**not** absolute AMK us, so the % cannot be computed from it (see §1.1).

**Occupancy context.** The fp register path uses a 32 KB static x-cache (`AMK_GEMV_MAX_K=8192`
floats, ops.cuh 177) which pins co-residency; the cp.async path and the quantized path instead carve
x-cache + ring from **dynamic** SMEM provisioned by `vm/loader.py` (`_gemv_cpasync_smem` 209–229;
quantized x-cache 412–424), bounded by `smem_bytes_per_block_optin` (loader.py 436). The occupancy
math is reported by `ext.max_coresident_blocks(...)` (loader.py 454), every variant's blocks/SM is
already in the bench output and must be read for each change.

---

## 2. Prioritized work items

Priority order is by **(expected datacenter impact) × (confidence) ÷ (effort+risk)**. Items P0–P2 are
pure knob/config exploration on the *existing* search surface (zero/near-zero code), so they are the
honest first moves, they directly test whether MLP-saturation and occupancy are the binders before
any structural rewrite. **INT8-P0 is the exception: it is the top *int8* lever and requires a kernel
change, but it sits high in priority because it attacks the worst datacenter gap (int8) at the
mechanism the knob-only P0 cannot reach, the quant path never uses the cp.async ring.** P3–P6 are
code changes (validator/kernel) gated on what P0–P2 (and INT8-P0) reveal.

### P0, Deepen the cp.async ring on datacenter (MLP lever, bf16/fp bandwidth), **DO FIRST**

- **What:** sweep `cpa_stages` 4→{6,8} and re-balance `cpa_cols`/`cpa_vpl` so the ring stays within
  the opt-in SMEM cap and does not drop blocks/SM. Ring bytes per block =
  `nwarps * STAGES * CPC * (VPL*warpSize*16)` (`vm/loader.py:_gemv_cpasync_smem`, line 228) plus the
  x-cache `((K+3)&~3)*4`. Example trades that keep ring bytes ~constant while deepening the pipeline:
  `{STAGES=6, CPC=1, VPL=1}` or `{STAGES=8, CPC=1, VPL=1}` vs the `{STAGES=4, CPC=2, VPL=1}` default.
- **Code:** **no kernel change.** Knobs already exist: `AMK_GEMV_CPA_STAGES`/`_CPASYNC_COLS`/`_CPA_VPL`
  (`vm/ops.cuh` 200–208), threaded by `vm/loader.py` `_KNOB_MACRO` (168–179) → distinct `-D` build
  variant. The ring-sizing already follows `cpa_stages`/`cpa_cols`/`cpa_vpl` in `_gemv_cpasync_smem`.
- **Lever / regime:** MLP → bf16/fp bandwidth saturation. More `STAGES-1` chunks in flight per warp.
- **Expected impact, bf16/fp PATH ONLY (this is the path that uses the ring).**
  **Attribution caveat (verified in code):** the headline datacenter numbers (A100 ~0.45×,
  H100 ~0.26×) are the **int8** path, and the int8 path **never executes the cp.async ring** -
  `amk_inst_gemv_tile` dispatches the quantized kernel `amk_inst_gemv_tile_quant` at the top of
  the op (`vm/ops.cuh` ~722–724 / ~758–760, `qdtype==AMK_I4||AMK_I8`) **before** the cp.async block
  (~line 780). So **P0 cannot move the int8 0.45× / 0.26× headline numbers**, that gap is the
  separate int8-staging/dequant lever (see "INT8-P0" below and P5). P0's impact is on the **bf16/fp
  path** that actually streams through the ring: RTX5090 bf16 ~0.9× cuBLAS (~69%→? of measured peak,
  the path already at the ~48% HBM plateau the ops.cuh comment cites), and **bf16 on datacenter**
  (sm_80/sm_90), which has no measured ratio in the table above and must be measured directly. The
  mechanism, raise in-flight depth ~50–100%, is an **ncu-gated hypothesis** (no counter data yet),
  not a measured figure; confirm by measured bandwidth, not theory.
- **Effort / risk:** ~1 hr (grid build + measure). Risk: `CPC=1` reduces x-reuse/parallel columns per
  warp, the sweep reveals the trade. Watch blocks/SM (`max_coresident_blocks`) does not drop.
- **ABI / validator:** none, compile-time kernel structure only. ABI frozen, drift guard unaffected.
- **NCU:** end-to-end measurable now (paired-interleaved). `ncu` would *confirm* the in-flight-load
  count rose (e.g. `lts__t_sectors`, `l1tex` outstanding loads) and bandwidth utilization, **needed
  to attribute** a win/no-win to MLP rather than something else.

### INT8-P0, Give the int8/int4 quantized GEMV its own cp.async / double-buffered weight staging (the TOP int8 lever), **DO FIRST for int8**

> **This is the single biggest missing lever for the worst datacenter gap** (int8 A100 ~0.45×,
> H100 ~0.26×). It is **distinct from and higher-priority than P5** (cheaper dequant ALU). P5 is
> *necessary but not sufficient* without this staging: cheapening the dequant does not help if the
> path is starved on synchronous weight loads with zero load/compute decoupling.

- **What:** stream the packed int8/int4 weight chunks HBM→SMEM **ahead of** dequant with a
  double-buffered / `__pipeline_memcpy_async` ring, exactly as the fp path's `amk_gemv_tile_cpasync`
  already does for bf16/fp. Today `amk_inst_gemv_tile_quant` (`vm/ops.cuh` ~370–550) reads each
  16-byte `uint4` weight chunk with a **synchronous** load in a serial `kv`-loop (int8 inner loop
  ~449–480, int4 ~498–519) and consumes it immediately in the dequant FMA, **no
  `__pipeline_memcpy_async` anywhere in the quantized op**. So the path with the *worst* datacenter
  gap has **zero load/compute decoupling** (no MLP from staging; the only overlap is the `QC` column
  accumulators' ILP, ops.cuh ~433–437). Staging the packed bytes ahead lets the next chunk's HBM
  load fly while this chunk dequantizes, the same MLP win P0 brings to bf16, applied to the int8/int4
  path P0 cannot reach (because the quant dispatch precedes the cp.async block, see P0's caveat).
- **Code:** **kernel change**, add a staged (cp.async ring) variant inside / alongside
  `amk_inst_gemv_tile_quant` (`vm/ops.cuh` ~439–531), reusing the dynamic-SMEM x-cache that's already
  there (ops.cuh ~400–418) and a packed-byte ring sized like `_gemv_cpasync_smem`. Select it via a new
  build knob in `vm/loader.py:_KNOB_MACRO` (e.g. `qcpa`/`qstages`), so it is a distinct `-D` variant
  (lowering/compile-flag only).
- **Lever / regime:** MLP → **int8/int4 bandwidth saturation** (the path with the largest gap). Does
  not touch bf16 (bf16 already has the ring via P0).
- **Expected impact:** this is the lever that can actually move the int8 0.45× / 0.26× headline (P0
  cannot). Magnitude is an **ncu-gated hypothesis**, measure end-to-end and confirm in-flight loads
  with `ncu`. Stacks with P5 (which removes the dequant-ALU binder once loads are decoupled).
- **Effort / risk:** ~1–2 days (mirror the fp cp.async ring for packed bytes). Risk: SMEM for the
  packed-weight ring competes with the x-cache and can drop blocks/SM, watch `max_coresident_blocks`;
  the unpack must stay bit-identical to the current dequant order (correctness oracle: argmax-exact
  for quant).
- **ABI / validator:** none, still `GEMV_TILE` with `qdtype`/`group`; a compile-flag-selected inner
  variant. No `amk_params_t` change, no new opcode, drift guard (`tests/test_abi_sync.py`) unaffected.
- **NCU:** **needs ncu** to confirm the int8 path goes from load-serial to load-decoupled (outstanding
  loads up, memory throughput up). End-to-end ratio observable now.

### P1, Raise occupancy via launch-bounds + per-target SMEM sizing (MLP lever)

- **What:** on datacenter, force more co-resident blocks/SM to add resident warps (= more total
  in-flight loads). Two coupled knobs: (a) `lb_maxthreads`/`lb_minblocks` →
  `__launch_bounds__(maxThreads, minBlocks)` on the megakernel (`vm/scheduler.cu` 75–83); (b) shrink
  the static x-cache `gemv_max_k` toward the program's real max-K so SMEM stops capping co-residency
  (the static 32 KB is documented as pinning ~2 blocks/SM, loader.py 149–154). Sweep e.g.
  `{lb_maxthreads=512, lb_minblocks=3}` on sm_90 and verify blocks/SM actually rises without spilling.
- **Code:** **no kernel change.** Knobs already plumbed (`vm/loader.py` 168–184, 297–306). Add a
  datacenter default set in `_KNOB_DEFAULTS` *only after* the sweep finds a winner (see P2).
- **Lever / regime:** occupancy → MLP → bf16/fp + int8 bandwidth (helps both paths).
- **Expected impact (ncu-gated hypothesis):** *hypothesis*, ~+50% resident warps **if** 2→3 blocks/SM
  lands without register spill → proportional MLP. Blocks/SM is verifiable now from the loader; whether
  those warps translate to bandwidth is the ncu-gated part. Stacks with P0.
- **Effort / risk:** ~1–2 hr. Risk: launch-bounds too aggressive → register spill to local memory
  (slower) or occupancy *collapse* below 2 blocks/SM. Mitigation: read numRegs + blocks/SM per
  variant from the bench JSON (`ext.kernel_attributes`, loader.py); reject regressions.
- **ABI / validator:** none, launch config and compile flags only.
- **NCU:** blocks/SM and reg count are available without `ncu` (loader reports them). Whether the
  extra warps *translate to bandwidth*, **needs ncu** to confirm (achieved occupancy vs theoretical,
  memory throughput %).

### P2, Per-GpuTarget datacenter knob profile (config, makes P0/P1 automatic)

- **What:** add a `sm_arch`-keyed default override (`sm_80`, `sm_90`) selecting the
  P0/P1-winning `{cpa_stages, cpa_cols, cpa_vpl, qc, lb_maxthreads, lb_minblocks, gemv_max_k}` when
  the caller passes no explicit knobs. Today `_KNOB_DEFAULTS` (loader.py 182–184) is one global set
  tuned for the dev part; datacenter retargeting is manual.
- **Code:** `vm/loader.py` (a dict keyed by `cap[0]` / `cap` from `torch.cuda.get_device_capability()`,
  consulted in `_normalize_knobs` / `MegakernelVM.__init__` when `knobs is None`), or carry it on
  `GpuTarget` in `schedule/ir.py` (`TARGETS`, ir.py ~420–447 already has per-GPU records).
- **Lever / regime:** packaging, makes the MLP wins reproducible/default per arch.
- **Expected impact:** none on its own; it *captures* P0+P1 so they apply by default.
- **Effort / risk:** ~1 hr. Risk: config sprawl, keep ONE "datacenter" profile per arch, not
  per-SKU. Falls back to current defaults on unknown arch.
- **ABI / validator:** none. A default override only applied when no explicit config is given.
- **NCU:** not needed.

### P3, Reduce sync frequency via tile grouping (sync-frequency lever), **CONDITIONAL, do not start until P0/P1 prove MLP saturation**

> **Strictly conditional.** P3 trades parallelism/MLP for fewer sync points, and the **dev part
> measured the opposite direction as a loss**: `schedule/lower.py`'s `_DEFAULT_N_TILE_AUTO` note
> (~lines 62–76) records that an actual CUDA-event sweep (`eval/bench_fat_tile_gemv.py`) found fatter
> tiles **monotonically SLOWER**, the binder there was parallelism / SM load-balance / MLP, **not**
> per-tile overhead. So P3 must **not** be started until P0/P1 have *demonstrated* the datacenter MLP
> is actually saturated (i.e. deeper ring + higher occupancy stopped buying bandwidth). If MLP is not
> yet saturated, P3 will REGRESS, exactly as it did on dev.

- **What:** group `G` consecutive GEMV tiles of one projection so the **count of sync points** drops
  ~`G×`. Two designs, in order of increasing risk:
  - **P3a (lowering-only, lowest risk):** keep one tile = one task, but have the lowering assign `G`
    tiles to the **same SM in adjacent queue slots** and ensure the consumer's all-join threshold
    equals the true `#tiles`. This does not reduce `amk_signal` count but improves locality; modest.
  - **P3b (one task computes G tiles):** emit a single `GEMV_TILE` task whose `N_tile` spans `G`
    thin tiles' worth of columns, so the kernel's existing `for (t = warp*C; t < N_tile; ...)` column
    loop (ops.cuh 848 / 433) covers them and signals the counter **once**. This is just a larger
    `N_tile` chosen for *sync amortization* rather than parallelism, i.e. a sync-cost-aware override
    of `_gemv_n_tile` (`schedule/lower.py` 98–118) gated on `target` bandwidth/SM count.
- **Code:** `schedule/lower.py` `_auto_n_tile` / `_gemv_n_tile` (85–118), add a `sync_cost_aware`
  branch: when `target.num_sms` is large AND `target` bandwidth is high, raise the tile-width floor so
  `#tiles` per projection drops, *but never below the point where SM load-balance starves* (the
  measured dev optimum was thin tiles; this is a datacenter-specific re-balance, validated by P0/P1
  results, not a global revert).
- **Lever / regime:** sync-frequency. Targets Diagnosis B; most relevant on H100 where compute/tile is
  shortest.
- **Expected impact:** realistically **single-digit % at best, and may REGRESS via parallelism loss.**
  The "~`G×` fewer sync points" is the *mechanism count*, not the end-to-end gain, it is strictly
  subordinate to MLP and only converts to a (small) win once MLP is already saturated by P0/P1.
  **Tension:** fewer tiles = less parallelism/MLP, and the dev sweep measured fatter tiles as
  *monotonically slower* (`schedule/lower.py` ~62–76). Net effect is positive only if P0/P1 have first
  demonstrated MLP saturation; otherwise it regresses, as on dev. **Therefore do P3 only after P0/P1
  show MLP is the no-longer-binding constraint.**
- **Effort / risk:** P3a ~2 hr; P3b ~half day. Risk: coarser tiles regress on the dev part and on
  shallow models (fewer schedulable units than SMs). Gate strictly on `target`.
- **ABI / validator:** **none for P3b**, it reuses the existing `GEMV_TILE` opcode with a larger
  `N_tile`. The validator's all-join rule (`schedule/ir.py` 841–851: a multi-producer counter must
  have `threshold == #producers`) still holds because fewer/one producer(s) feed the counter with the
  matching threshold. No new opcode, no `amk_params_t` change, drift guard (`tests/test_abi_sync.py`)
  unaffected. P3a likewise. **Do NOT** introduce a separate `GEMV_TILE_GROUP` opcode, it would bump
  `ABI_VERSION`, require a ReferenceVM oracle, and weaken the proof for negligible additional gain
  over P3b (see §4).
- **NCU:** end-to-end measurable now. `ncu` would *confirm* the per-tile barrier/atomic time, but the
  wall-clock A/B (group vs no-group, same total columns) is a clean measurement without it.

### P4, Wider KUNROLL (ILP→MLP, fp path)

- **What:** sweep `kunroll` 1→{2,4}: each lane loads more independent float4/bf16x8 vectors per
  K-iteration before consuming them (more independent loads in flight from ILP). The loop already
  steps `warpSize*KUNROLL` (`amk_gemv_rows_dot_f32`, ops.cuh 240–254).
- **Code:** **no kernel change.** Knob `AMK_GEMV_KUNROLL` (ops.cuh 173) plumbed (loader.py 170).
- **Lever / regime:** ILP→MLP on the register fp path (a fallback when cp.async is off / K too small,
  ops.cuh 799, 826–832). Less central than P0 because cp.async is the production fp path on sm_80+.
- **Expected impact:** +5–10% bandwidth per doubling on the register path, with register-pressure risk
  above 4. Marginal where cp.async already runs.
- **Effort / risk:** ~30 min. Risk: register spill cuts occupancy; read numRegs per variant.
- **ABI / validator:** none.
- **NCU:** measurable end-to-end; `ncu` confirms in-flight-load / register effects.

### P5, DP4A / IMMA int8 dequant (int8 dequant lever), **necessary but NOT sufficient without INT8-P0**

> **Pair with INT8-P0.** P5 only cheapens the dequant *ALU*; it does nothing about the quant path's
> **synchronous, undecoupled weight loads** (INT8-P0). If the path is load-serial, cheaper dequant
> exposes the load stall rather than removing it. Do INT8-P0 (stage the weights) first or together;
> P5 then converts the now-load-decoupled path the rest of the way to bandwidth-bound.

- **What:** replace the 16 serial int→float casts + FMAs in the int8 inner chunk
  (`amk_inst_gemv_tile_quant`, ops.cuh 470–477) with `__dp4a` (sm_61+, 4×int8 dot → int32) or
  sm_80+ int8 IMMA, accumulating int32 then applying the per-group fp16 scale once per chunk (the
  group-constant scale already factors out, ops.cuh 463–468, 478). Goal: move int8 from
  **ALU-bound** (hypothesized ~35–37% of its floor, ncu-gated, see §1.1) to **bandwidth-bound**,
  where its 4× byte saving finally beats
  cuBLAS bf16. **Caveat:** plain weight-only DP4A needs *integer activations*; the current path
  multiplies fp32 `xs[k]` by int8 weights. Two sub-options:
  - **P5a (W8A8):** quantize activations to int8 too → true `dp4a(int8 x, int8 w)→int32`. Changes the
    numeric contract (int accumulate vs fp). Must gate on perplexity/accuracy and re-pass the
    correctness oracle (argmax-exact for quant).
  - **P5b (cheaper fp dequant):** keep fp activations but vectorize the int8→fp conversion (e.g.
    `__bytePerm`/packed convert, half2 FMA) to cut the ALU op count without changing numerics. Lower
    payoff than DP4A but numerically safe.
- **Code:** new inner-loop variant inside `amk_inst_gemv_tile_quant` (ops.cuh 439–492), or a sibling
  device function selected by a knob. Likely a new knob (e.g. `qmma`) in `vm/loader.py:_KNOB_MACRO`.
- **Lever / regime:** int8 dequant only. Does **not** help bf16.
- **Expected impact (ncu-gated hypothesis):** int8 from ~35% → ~45–55% of its floor *if* dequant stops
  being the binder (both endpoints are hypotheses pending ncu / an absolute-us measurement, see §1.1);
  then the same INT8-P0 weight-staging + P0/P1 MLP fixes apply to the now-bandwidth-bound int8 path.
- **Effort / risk:** ~1–2 days. Risk: **precision**, DP4A int32 intermediate vs the reference's fp
  `float(q)*scale`. P5a may shift argmax on some tokens → must pass `tests/test_cuda_int4.py` +
  quant evals with the frozen tolerance (argmax-exact). P5b is numerically conservative.
- **ABI / validator:** no ABI change if it stays a GEMV inner-loop variant (still `GEMV_TILE` with
  `qdtype`/`group`). W8A8 (P5a) would add an activation-scale input/param, justify and drift-guard in
  `tests/test_abi_sync.py` if so; prefer P5b's no-ABI route first.
- **NCU:** **needs ncu** to confirm the int8 path actually crosses from ALU-bound to memory-bound
  (`smsp__inst_executed` / integer-pipe utilization down, memory throughput up). Without it, only the
  end-to-end ratio is observable.

### P6, `op_noinline` to shrink the megakernel register frame (occupancy lever, conditional)

- **What:** the megakernel inlines every opcode, so its register frame is the *worst* opcode's
  (`AMK_OP_QUAL`, ops.cuh 30–42). `-DAMK_OP_NOINLINE` makes per-op device fns `__noinline__` so the
  frame can shrink toward the cheap-op case, potentially raising blocks/SM. Already an A/B knob
  (`op_noinline`, loader.py 232/310–312; reports blocks/SM + numRegs both ways).
- **Code:** **no kernel change.** Flip the flag in the datacenter profile *only if* the A/B shows more
  blocks/SM **and** a measured win.
- **Lever / regime:** occupancy → MLP. Conditional, ops.cuh 36–37 explicitly warns occupancy is also
  capped by the static SMEM x-cache, so registers may not be the binder (couple with P1's `gemv_max_k`
  shrink).
- **Expected impact:** unlocks higher blocks/SM only when registers (not SMEM) are the binder.
- **Effort / risk:** ~30 min A/B. Risk: call overhead on the hot single-stream path can offset the
  occupancy gain, measure.
- **ABI / validator:** none. Inlining vs calling is bit-identical numerically.
- **NCU:** blocks/SM/regs from loader; bandwidth attribution **wants ncu**.

---

## 3. ABI / validator constraints every item must respect

These are the invariants from `vm/abi.h` and `schedule/ir.py` that bound the whole plan:

1. **Instructions are pure compute** (`abi.h` 13–20, 192–197): an op reads inputs, computes, writes
   outputs, **no counters, no cross-buffer side effects, no launches**. All kernel-structure items
   (P0, INT8-P0, P4, P5, P6) stay inside this archetype, INT8-P0 stages weights into SMEM and
   computes from SMEM, still pure compute (the cp.async load is a memory op into the block's own SMEM,
   exactly as the existing fp cp.async path).
2. **The VM owns sync; counters are monotonic** (`abi.h` 21–26, 80–84): producers only `++` by 1;
   consumers only wait on static thresholds. A counter with >1 producer is an **all-join**
   (`threshold == #producers`); **partial waits are rejected**, `schedule/ir.py` validate, lines
   841–851. Any tile-grouping (P3) must keep `threshold == #tiles-feeding-the-counter`.
3. **Per-SM queues are global-topological subsequences** (`abi.h` 24–25; validator 919–930): an SM
   must not block on a counter only its own later queue entry could signal. The LPT/round-robin SM
   assignment already preserves this by walking the topo order (`vm/loader.py` `_assign_sms`
   462–506); P3's grouping must keep grouped tiles in topo order on their SM.
4. **The frozen ABI byte layout** (`abi.h` `amk_params_t`/`amk_buffer_t`/`amk_instruction_t`,
   89–131; `ABI_VERSION="0.2"`, ir.py 78): drift-guarded by the numpy packer vs C `sizeof`
   (`vm/loader.py` `_assert_layout` 331–341) and by `tests/test_abi_sync.py`. **P0–P4, INT8-P0, and P6
   require no ABI change** (INT8-P0 is a compile-flag-selected GEMV inner variant, still `GEMV_TILE`
   with `qdtype`/`group`, no new field). P5b requires none; **P5a (W8A8)** is the only item that could
   add a param/input -
   if taken, bump nothing silently: add the field at the *end* of `amk_params_t` (abi.h 86–87 "append
   at end, never reorder"), update the packer + `tests/test_abi_sync.py`, and justify it.
5. **Correctness is gated by the CPU ReferenceVM** (bit/ulp-exact for fp, argmax for quant). The fp
   GEMV variants preserve the fp32 elementwise-then-sum order (ops.cuh notes this is bit-equal across
   the register and cp.async paths). P5 (DP4A/W8A8) changes accumulation order/precision and must
   re-pass `tests/test_cuda_int4.py` + the quant evals before it can be kept.
6. **Cooperative-launch occupancy gate** (`abi.h` 150–162; loader 450–457): `gridDim ≤
   max_coresident_blocks × num_sms`, and dynamic SMEM ≤ `smem_bytes_per_block_optin`. P0/P1/INT8-P0/P6
   change SMEM/registers and **must re-check** that the loader still finds ≥1 block/SM and stays under
   the opt-in cap (loader.py 436), otherwise the launch is refused (INT8-P0's packed-weight ring adds
   dynamic SMEM on top of the quant x-cache, so its blocks/SM must be watched closely).

---

## 4. What we are NOT doing, and why

- **Tensor-core MMA / wmma for the bf16 GEMV.** Batch-1 decode is bandwidth-bound (~2 FLOP/byte). The
  bottleneck is feeding HBM bytes, not multiplying them, tensor cores would sit idle waiting on
  loads. MMA cannot raise achieved HBM bandwidth, so it cannot help here. (DP4A/IMMA in P5 is **not**
  used for throughput; it is used solely to make the *int8 dequant* cheaper so that path stops being
  ALU-bound. Different lever, different regime.)
- **A new `GEMV_TILE_GROUP` opcode for sync batching.** It would bump `ABI_VERSION`, demand a matching
  ReferenceVM oracle, and, because within-group tile ordering would no longer be expressed by
  counters, *weaken* the validator's race-freedom proof (the validator cannot see a kernel's internal
  loop). P3b achieves the same sync amortization with a larger `N_tile` on the *existing* opcode, ABI
  and proof intact. Not worth the ABI break.
- **SM-local "deferred/batched" counters or a relaxed `amk_signal` without the block barrier.** The
  frozen SYNC CONTRACT (`abi.h` 167–190, `vm/sync.cuh`) is what makes the deadlock/race-freedom proof
  hold across SMs. Weakening the release-fence/acquire pair or the publishing `__syncthreads` risks
  silent stale reads or hangs for, at best, tens of nanoseconds per tile, dominated by the
  higher-leverage MLP work. Reduce sync **frequency** (P3) instead of weakening sync **primitives**.
- **Counter pooling / liveness reuse.** Saves only L2 atomic-contention at the margin; the validator's
  transitive-provenance proof (ir.py 860–904) does not currently reason about counter reuse, so it
  would need new liveness checks to stay sound. Not justified versus the MLP items.
- **Multi-token / speculative batching to amortize VM overhead.** A real and large win, but it changes
  the *regime* (no longer batch-1 single-token) and the decode model in `abi.h` (one launch == one
  token). Out of scope for "make the batch-1 decode GEMV competitive". Note it as the strategic
  follow-on once the structural batch-1 gap is closed as far as it goes.
- **Chasing parity blindly.** Honest expectation: with P0–P4 stacked, A100 plausibly moves toward
  ~0.70–0.80× and H100 toward ~0.50–0.65× of cuBLAS. cuBLAS's GEMV is a single library call with
  effectively zero per-call cross-SM sync; AMK's VM pays a grid-cooperative sync per tile by design.
  **Full batch-1 parity may not be reachable** without the multi-token regime change above. The plan
  narrows the structural gap; it does not promise to cross cuBLAS at batch-1.

---

## 5. Measurement plan (existing harness, no new infra)

All measurement uses the **existing, correctness-gated, kernel-only, paired-interleaved** harness.

> **Note, the `paper/...` and `modal_app.py` drivers ship with the paper artifact, not the OSS
> tree.** `paper/exp_int8_search.py` and `modal_app.py` are gitignored paper RESULTS infrastructure
> and are **not** in the published repository, so commands against them will **not** run on a clean
> clone. The shipped equivalent for the local int8/quant GEMV sweep + correctness gate is
> `eval/bench_quant.py` (and the broader baselines/roofline probes `eval/bench_baselines.py` /
> `eval/roofline.py`). The line/function references below are kept so the methodology is reproducible
> from the paper artifact and so the shipped probes can mirror it.

1. **Local (dev RTX 5090) smoke + correctness gate:** `paper/exp_int8_search.py` (paper artifact;
   the shipped OSS equivalent is `eval/bench_quant.py`). It sweeps `qc` /
   `N_tile` / `threads_per_block`, builds each variant, gates argmax-exact vs bf16 eager, and measures
   `vm.relaunch()` (one cooperative launch = whole forward) vs cuBLAS CUDA-graph `g.replay()`
   **per-sample paired-interleaved** (`_paired`, exp_int8_search.py 59–70; ratio = cuBLAS_us/AMK_us,
   >1 ⇒ AMK faster) so clock drift cancels. Use it to (a) confirm every new variant is still
   correct, and (b) check no consumer-part regression before promoting a datacenter default.
   Extend `QC_CHOICES`/`NTILE_CHOICES`/`TPB_CHOICES` (lines 45–47) and add the cp.async ring +
   launch-bounds knobs to the swept grid for P0/P1/P4.
2. **Datacenter (A100, H100):** `modal_app.py::int8search` (`--gpu both --layers N --minutes M`,
   modal_app.py 477–491) for the per-size sweep, and `modal_app.py::overnight_pro`
   (`overnight_pro_a100`/`_h100`, 762–779) for the across-depth sustained run that produced the
   current 4→24-layer ratio table. Same `vm.relaunch` vs `g.replay()` paired-interleaved method
   (modal_app.py 289/309/399/425). Re-run before/after each item; a crossing is declared per size only
   if the best config's **p10 > 1.0** (the existing bar).
3. **Per-variant occupancy/register read-out:** the loader already reports blocks/SM
   (`max_coresident_blocks`, loader.py 454) and kernel attributes (numRegs, dyn smem). For P1/P4/P6,
   record these alongside the ratio so an occupancy/spill regression is caught without `ncu`.
4. **`ncu` gate (when unblocked):** items tagged **needs ncu** above require Nsight Compute to *prove*
   the mechanism (in-flight loads / memory throughput % for P0/P1/P4; integer-pipe vs memory bound
   crossover for P5). The wall-clock ratio tells us *whether* a variant is faster; `ncu` tells us
   *why*, which is required before declaring "MLP was the binder". **Resolving the Modal
   `LibraryNotLoaded` `ncu` block is itself a prerequisite task for confirming P0/P1/P5.**
5. **Additional no-code levers to fold into the existing sweeps (cheap, do not skip):**
   - **Co-sweep `threads_per_block` alongside the new ring/launch-bounds knobs.** Every
     measured-best int8 config in `paper/results/int8_search_datacenter.json` used `threads=512`
     (both A100 and H100 top-5), but that was found with the *old* `qc`/`N_tile` grid, the optimum
     TPB can move once `cpa_stages` / `lb_minblocks` / `gemv_max_k` change occupancy. Sweep TPB jointly
     with P0/P1's knobs, not in isolation.
   - **Re-sweep the existing L2 weight-prefetch pipeline at datacenter bandwidth.** The
     software-pipelined prefetch (`vm/scheduler.cu` `amk_prefetch_gemv_weights`, gated by
     `ScheduleConfig.pipelining_depth`) was ~break-even on the dev part and **was never re-swept at
     1.4–3.0 TB/s**, at higher bandwidth the inter-op HBM bubble it hides is a different fraction of
     the tile. Sweep `pipelining_depth` 0–4 on A100/H100. **No code, no ABI**, it is an existing
     `ScheduleConfig` knob.
   - **Name the int4 path as the most-likely-to-cross-cuBLAS candidate.** `qdtype==AMK_I4` has the
     **lowest roofline floor** (~8× fewer weight bytes than bf16, ~2× below int8), so once it is
     load-decoupled (INT8-P0 covers int4 too) and bandwidth-bound it is the path most likely to cross
     cuBLAS bf16 at batch-1. **Accuracy caveat:** int4 is lossier than int8/W8A16, it must clear the
     same argmax-exact quant oracle (`tests/test_cuda_int4.py` + quant evals) before any int4 crossing
     is claimed; a faster-but-wrong int4 kernel does not count.

### Suggested execution order

`P0 → P1` (measure MLP/occupancy on the existing knobs first; this is the make-or-break test of the
structural diagnosis) → if they move the needle, `P2` to lock the datacenter profile → `P4` (cheap
ILP top-up) → `P3` (sync-frequency, **only after** P0/P1 prove MLP is saturated; otherwise it
regresses) → **on the int8/int4 path (the worst gap): `INT8-P0` (stage the packed weights, the lever
P0 cannot reach) then `P5` (cheaper dequant, necessary but not sufficient without INT8-P0)** → `P6`
(conditional occupancy top-up). Unblock `ncu` in parallel to confirm mechanisms.

---

## 6. One-line summary

The datacenter gap is structural: the GEMV does not keep enough loads in flight to saturate 1.4–3.0
TB/s HBM, and pays a fixed per-tile grid sync that grows as a fraction of the now-shorter compute
time. Attack **MLP/occupancy first via the existing cp.async-ring + launch-bounds knobs (P0–P2, zero
code)** for the **bf16/fp** path, but note the headline int8 0.45×/0.26× numbers live on the
**quantized** path, which **never uses the cp.async ring** (the quant dispatch precedes it), so the
biggest int8 lever is **staging the packed weights (INT8-P0)** followed by cheaper dequant (P5).
Then sync-frequency (P3, only once MLP is saturated). Do **not** reach for tensor-core MMA, it cannot
help a bandwidth-bound batch-1 GEMV.
