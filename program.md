# AutoMegaKernel, Autonomous Operation Brain (`program.md`)

> This is the playbook a coding agent (Claude Code, Codex, …) reads to run AMK **unattended for
> 10+ hours**: propose a change, run a *fixed* eval, keep or revert on measured
> correctness-then-latency, log it, repeat, and never get stuck on a hung GPU. It is the most
> important document in the repo. Keep it current; if reality drifts from this file, fix this file.
>
> AMK has **two loops** with **different edit surfaces and different safety models**. Read the
> whole "Two Loops" section before you touch anything. The contracts (`schedule/ir.py`,
> `vm/abi.h`, `vm/reference_vm.py`, `instructions/reference.py`, `models/toy.py`) are **LOCKED**.
> You build *against* them; you never edit them.

---

## 0. Mission & North Star (stated honestly)

AMK becomes **the default toolchain for megakernel generation for the next decade**, the thing
people reach for the way they reach for AutoCAD's DWG format. That status comes from four
properties; treat them as the actual product spec, not marketing:

1. **Generality.** One command (`compile.py` → `amk compile <hf-model> --gpu <arch>`) lowers
   any supported HuggingFace **Llama-family** decoder into a megakernel on *any* registered GPU, with zero per-model
   hand-written CUDA. If a new model needs a human to write a kernel, AMK has failed its mission.
2. **Self-retargeting.** A GPU is *data*, never branches in code: it is a `GpuTarget` record in
   `schedule.ir.TARGETS` (`rtx5090`, `b200`, `h100` ship today). New silicon = a new record +
   search + on-hardware verification, in days, not the months hand-tuned libraries lag. This is
   the durable moat. No arch assumption may be hardcoded that cannot be searched over.
3. **A standard IR.** AMK owns the canonical megakernel IR, the SM-level task-DAG
   (`schedule.ir.MegakernelProgram`), the frozen instruction ABI (`vm/abi.h`), the schedule
   search object (`schedule.ir.ScheduleConfig`). Clean, documented, versioned by `IR_VERSION`
   / `ABI_VERSION`. This is the DWG-format lock-in. See `docs/IR_SPEC.md` for the standalone spec.
4. **A data flywheel.** Every experiment appends a row to `results.tsv`:
   `(model, gpu, schedule_or_kernel_id, regime, correctness, latency, pct_of_roofline)`. That
   corpus trains a learned prior over schedules so every future run starts smarter. Design the
   logging to feed this from day one.

**Performance target, stated so we never oversell.** As close to the
`weight_bytes / HBM_bandwidth` bound as the silicon allows, automatically, on every model and
chip. Single-stream / low-batch **decode latency** is the win regime (it is *bandwidth-bound*:
each token must stream the whole weight set through the SMs once;
`GpuTarget.bandwidth_bound_us(weight_bytes)` is the honest floor). We do **not** claim to beat
throughput-optimized serving at high batch, that is compute-bound and not our fight. Win on
generality, retargeting, trust, and being near the bound *everywhere*.

**The flywheel.** Propose → fixed eval → keep/revert → repeat, for hours, unattended. The eval
is full-model correctness (`eval/oracle.py` vs eager `models.toy.ToyLlama`) **then** latency
(`eval/bench.py`). This loop is the heartbeat; everything below is how to run it safely.

---

## 1. The Two Loops (DIFFERENT edit surfaces, internalize this)

A single kernel has a trivial oracle (compare op output) and a trivial benchmark (time it). A
megakernel does not: its oracle is the whole model, and its danger is the *scheduler* (a bad
schedule deadlocks the GPU, a hang, which yields *no clean FAIL signal*). So AMK splits work
into two loops that are evaluated and made safe completely differently.

### Loop 1, Instruction optimization (this *is* AutoKernel, transferred directly)

- **Agent command:** `amk tune-instruction <op> --gpu <arch> --budget N` (e.g.
  `uv run python amk_cli.py tune-instruction gemv_tile --gpu rtx5090 --budget 6`) drives the whole
  loop on ONE instruction: propose a kernel variant (`-D` knob set from
  `instructions/gen.SEARCH_SPACE`) → BUILD → verify correctness vs `instructions/reference.py` →
  microbench (CUDA events) → keep/revert (correctness first, then ≥1% latency win) → log each trial
  to `results.tsv` (`loop=instruction`) → return a JSON summary
  `{op, trials, best_variant, best_us, baseline_us, speedup, all_correct, pct_of_roofline}`. This is
  the single-instruction analog of `amk loop`. Code: `loop1.py` (+ `instructions/gen.py`,
  `instructions/verify_inst.py`). See HARNESS.md §1A.
- **Edit surface:** ONE ABI-conformant micro-kernel file at a time, under `instructions/triton/`
  (fast iteration) or `instructions/cuda/` (max perf). Same `KERNEL_TYPE` + `kernel_fn()`
  interface as AutoKernel, behind one backend switch. The agent edits the kernel body and/or adds a
  searchable `-D` knob to `SEARCH_SPACE`; variant 0 (no `-D`) is the incumbent baseline.
- **Contract:** an instruction is **pure compute** (read inputs, compute, write outputs). It
  MUST NOT touch counters, MUST NOT read/write any buffer it did not declare, MUST NOT launch
  work. See the Layer-1 ABI in `vm/abi.h`
  (`__device__ void amk_inst_<name>(const amk_program_t&, const amk_instruction_t&)`).
- **Eval:** isolated correctness vs the matching reference in `instructions/reference.py`
  (`REFERENCE[op](inputs, outputs, params, ctx)` writes outputs in place), then an isolated
  benchmark. ~seconds. A wrong instruction fails its **own unit test**, there is no persistent
  kernel, so **no GPU hang is possible** in this loop.
- **Keep/revert:** measured correctness-then-latency, exactly like AutoKernel (rules in §3).
- **The math the instruction must reproduce** (frozen conventions, copy them exactly):
  - Linear weight layout is torch `nn.Linear`: weight is `[N_out, K_in]`; a GEMV/GEMM computes
    `x @ W.T`. A tile writes the slice `out[..., n_off : n_off+N_tile]`.
  - Reductions/matmuls accumulate in **fp32** then cast to the output dtype.
  - RoPE = Llama rotate-half (`_rotate_half` splits into two halves). RMSNorm =
    `x * rsqrt(mean(x^2)+eps) * w`. SwiGLU = `silu(gate) * up`. Attention is GQA
    (`n_heads % n_kv_heads == 0`, `repeat_interleave`), fp32 scores, scale `1/sqrt(head_dim)`.

### Loop 2, Schedule optimization (the NEW loop, the agent's hands are deliberately moved)

The agent must **NEVER** freely edit the megakernel/VM CUDA (`vm/scheduler.cu`, `pages.cu`,
`sync.cu`). That would throw away correctness-by-construction and let it hand-write deadlocks.
Instead:

- **Agent commands:** `amk propose <model> --gpu <arch>` (read the editable `ScheduleConfig` +
  search space), `amk eval <model> --gpu <arch> --config cfg.json` (one structured verdict),
  `amk loop <model> --gpu <arch> --budget N` (the keep/revert autoresearch loop). Code: `harness.py`.
  See HARNESS.md §2–§6.
- **Edit surface = `schedule.ir.ScheduleConfig` ONLY**, a structured object, *not* kernel code:
  ```python
  ScheduleConfig(
      tiling = {"gemv": {"N_tile": 256}, "attention": {"kv_block": 128}},  # per-op-archetype
      fusion_grouping = [["rmsnorm", "gemv"], ...],   # adjacent ops -> one resident task group
      sm_assignment = "load_balance",                 # | "round_robin" | {task_id: sm}
      pipelining_depth = 2,                            # prefetch depth, THE biggest single win
      page_allocation = "graph_color",                # | "linear" | "none"
      threads_per_block = 256,                         # block size for the persistent VM kernel
      smem_bytes_per_block = 0,                        # dynamic SMEM opt-in (<= target opt-in cap)
  )
  ```
  The **frozen VM deterministically lowers this config** into a runnable megakernel
  (`MegakernelProgram` of `Task`s). The agent chooses a point in this search space; the VM knows
  how to realize it safely. The agent never writes the dangerous part.
- **Validate BEFORE launch (load-bearing).** Every proposed program goes through
  `schedule.ir.validate(prog) -> ValidationResult` first. A `REJECTED` result MUST prevent
  launch. `validate()` **never raises** on a malformed program, it always returns a result.
  `vm.reference_vm.ReferenceVM` and the CUDA loader **refuse to load** anything `validate()`
  rejects. This converts the overwhelming majority of "hung GPU" outcomes into a clean
  `REJECTED` signal, exactly like AutoKernel's `FAIL`. (See §2 for the two invariants it proves.)
- **Watchdog + abort flag for the rest.** Hangs static checks can't catch (data-dependent
  stalls, page exhaustion) are caught at runtime: the CUDA `amk_program_t.abort_flag` is
  grid-polled in `amk_wait_all`'s spin loop, set it `!= 0` and all SMs exit cleanly. The host
  sets it on a timeout. The software analogue is `vm.reference_vm.ReferenceVM.run(timeout_s=...)`
  raising `DeadlockError` when a full sweep makes no progress (the watchdog firing in software).

| | **Loop 1, Instruction** | **Loop 2, Schedule** |
|---|---|---|
| Edit surface | one micro-kernel file (`instructions/{triton,cuda}/`) | `ScheduleConfig` object **only** |
| Hand-write CUDA? | yes (the kernel body) | **never** (VM lowers the config) |
| Correctness gate | vs `instructions/reference.py` op, isolated | `validate()` + full-model oracle vs eager |
| Failure mode | unit-test FAIL (no hang possible) | `REJECTED` (no launch) or TIMEOUT (watchdog) |
| Eval cost | ~seconds | seconds (reference VM) → on-hardware |

---

## 2. Correctness from Structure, the two invariants `validate()` proves

These are the teeth behind "auto-generated schedules are safe to run unattended." Both are
enforced by `schedule.ir.validate()`; `tests/test_validator_races.py` pins the counterexamples.

### Invariant A, Deadlock-freedom
Producers only **increment** a counter (`out_counter += 1` once on completion, after a release
fence). Consumers only **wait** on static `(counter, threshold)` pairs (`schedule.ir.Wait`).
`validate()` requires `1 <= threshold <= #producers(counter)` and that the producer→consumer
graph (`MegakernelProgram.dependency_edges()`) is **acyclic** (`topological_order()` returns
`None` ⇒ REJECTED with a cycle witness). When SMs are assigned, each SM's serial queue must be a
linear extension of the DAG, else an SM blocks on a counter only its own later queue entry could
signal, also checked.

### Invariant B, Race-freedom (the subtle one)
A counter carries a **count, not which producer finished**. So:
- **A counter with >1 producer is an ALL-JOIN.** Every consumer must wait `threshold ==
  #producers`. A *partial* wait (`1 < t < #producers`) is a "first-k-of-N" race (the *wrong*
  producers can satisfy it) → **REJECTED**. NEVER put a partial threshold on a shared counter.
  (Tiled GEMV: N tiles share one counter; the consumer waits `threshold == n_tiles`. See the
  `_tiled_gemv` helper in `vm/verify_vm.py`.)
- **Every ACTIVATION / IO read must have a TRANSITIVE happens-before edge from its producer.**
  `validate()` walks the topo order computing, per task, the bitmask of buffers written by its
  transitive predecessors; a read with no such edge is a data RACE → **REJECTED**. Read-only
  kinds (`WEIGHT`, `CONST`, `IO_INPUT`) never need an edge.
- **KV_CACHE rule.** A `KV_CACHE` buffer written this pass (by `KV_APPEND`) may be read only by
  tasks ordered *after* the append. Prior-step KV state pre-exists, so the *writer* reading its
  own cache is fine; any *other* reader must happen-after the `KV_APPEND` or it is a RACE.

### Fixed ABI caps (`validate()` rejects violations; never silently truncates)
`<= 8 inputs` (`ABI_MAX_INPUTS`), `<= 4 outputs` (`ABI_MAX_OUTPUTS`), `<= 8 waits`
(`ABI_MAX_WAITS`), rank `<= 4` (`ABI_MAX_RANK`). For wide fan-in, build a **counter/reduction
tree** (e.g. `ATTENTION_COMBINE` merging per-KV-block partials), not one 100-wide wait.

### The all-join pattern (memorize, it is the core building block)
```python
c = p.new_counter().id          # one shared counter for the N tiles
for i in range(N):              # N producers, each increments c once
    p.add_task(InstructionKind.GEMV_TILE, [x, w], [out], out_counter=c,
               params={"K": K, "N_tile": tile, "n_off": i*tile})
p.add_task(InstructionKind.SILU_MUL, [out, u], [act], out_counter=c2,
           waits=[Wait(c, N)])  # threshold == N producers == ALL-JOIN (never < N)
```

> **Always sanity-check a freshly built program with `validate(prog).report()` and
> `prog.simulate_counters()` (stuck list must be empty) before you ever try to run it.** Use
> `prog.simulate_adversarial()` as the dynamic backstop for race hunts.

---

## 3. Keep / Revert Rules (apply strictly, in order)

Correctness is **always** first. A fast-but-wrong megakernel is reverted instantly. **Never
report a latency number without its paired correctness result** (§7).

| Condition | Action |
|---|---|
| `validate()` = REJECTED | **REVERT**, do not launch. Clean signal; log `correctness=REJECTED`, `latency_us=` (blank). |
| Runtime TIMEOUT / `DeadlockError` (watchdog fired) | **REVERT**, kill, set abort flag, log `correctness=TIMEOUT`, `latency_us=` (blank). |
| correctness = FAIL (oracle mismatch vs eager) | **REVERT** immediately. Never keep an incorrect schedule/instruction. |
| correctness = PASS, latency improved **≥ 1%** | **KEEP**. New baseline. |
| correctness = PASS, latency within ±1% (noise) | **REVERT**, *unless* the config/code is meaningfully simpler (tie-break below). |
| correctness = PASS, latency worse | **REVERT**. |

- **"Improved" = ≥ 1% reduction in `latency_us`.** Noise-level changes are reverted.
- **Simplicity tie-break.** If latency is equal (±1%) but the new schedule is meaningfully
  simpler (fewer tasks/pages, lower `pipelining_depth`, fewer fusion groups), **KEEP** it -
  simpler schedules are more robust to retargeting and easier for the flywheel prior to learn.
- **One focused change per experiment.** Change `pipelining_depth` *or* a tile size *or* the SM
  policy, not three at once. You must know what caused the delta.
- **Commit before you run** (Loop 1, on a campaign branch) so a regression is one
  `git reset --hard HEAD~1` away. Do **not** commit `results.tsv` / `run.log`, leave untracked.

### Move-on / diminishing-returns criteria (concrete thresholds)
Stop optimizing a region and move to the next-highest-impact one when **any** holds:
1. **Plateau:** the last **10–15** experiments on this region all failed to beat the best by ≥ 1%.
2. **Near-roofline:** measured `pct_of_roofline` (vs `GpuTarget.bandwidth_bound_us`) is **≥ 85%**
   for the bandwidth-bound decode regime, you are within ~15% of the hardware floor; accept it.
3. **Diminishing returns (Amdahl):** the next region would yield more end-to-end benefit. A 1.5×
   on a 60% region beats a 3× on a 5% region. Always optimize the largest `est_bytes` /
   wall-time fraction first.
4. **Time budget:** a per-region budget was set, respect it.

### Radical vs incremental
| Situation | Strategy |
|---|---|
| Early (exp 0–10) | Aggressive: large tile changes, different fusion groupings, prefetch depth sweeps. |
| Mid (exp 10–30) | Focused: systematic sweeps of the one promising knob. |
| Late (exp 30+) | Incremental: combine winners, fine-tune. |
| Plateau, `pct_of_roofline` < 50% | Radical: rethink fusion/pipelining, structurally bandwidth is being wasted. |
| Plateau, `pct_of_roofline` ≥ 80% | Accept, near the bound; move on. |

---

## 4. Crash / Deadlock / Timeout Handling (never let one failure stop the run)

This loop runs unattended. A single bad experiment must degrade to a logged failure, never a
stuck session.

### `REJECTED` is a clean signal, NOT a hang
`validate()` runs on the host with **no GPU and no launch**. A structurally invalid schedule
(cycle, partial wait on a shared counter, missing happens-before edge, capacity overflow,
KV-read-before-append) returns a `ValidationResult(ok=False, errors=[...])` *instantly*. Treat
it exactly like AutoKernel's compile-time `FAIL`: read the first error, revert, log, move on.
`validate()` is also robust to pathological input, a 5000-node cycle returns REJECTED via the
**iterative** `_describe_cycle`, never a `RecursionError`.

### A hung GPU = TIMEOUT via watchdog/abort flag
Some stalls only manifest at runtime. The defenses, in layers:
1. **Software (no GPU):** `ReferenceVM.run(inputs, kv, timeout_s=30.0)` raises `DeadlockError`
   the moment a full sweep makes no progress, or after `timeout_s`. This is the watchdog in
   software and the GPU-free correctness oracle in one.
2. **Hardware:** the host arms a wall-clock timeout per launch. On expiry it sets
   `amk_program_t.abort_flag != 0`; every SM's `amk_wait_all` spin loop polls it and returns
   `false` so all blocks exit cleanly (no orphaned persistent block = no permanent hang).
3. **Process:** if a launch wedges past the timeout, kill the process, **revert**, log
   `correctness=TIMEOUT`, **continue the run**. Same crash 3× in a row on one approach ⇒ stop
   trying that approach, try something structurally different.

### Windows WDDM TDR is real (this dev machine: RTX 5090 laptop, `wddm_tdr=True`)
The dev GPU is a **display** GPU with an OS watchdog (~2s TDR). The design defends against it:
- **One launch per token** (`decode model: one launch == one forward pass == one token`).
  Counters are host-memset to zero before each launch; `KV_CACHE` persists in HBM across
  launches; the host drives the autoregressive loop. Each launch stays well under the ~2s TDR.
- For dev, **raise `HKLM\System\CurrentControlSet\Control\GraphicsDrivers\TdrDelay`** so longer
  experimental launches don't get nuked mid-flight.
- The host treats `cudaErrorLaunchTimeout` as a **distinct TIMEOUT** (a TDR reset), not a clean
  `REJECTED`. Log it as TIMEOUT and revert.
- `GpuTarget.wddm_tdr` flags which targets need this; datacenter parts (`b200`, `h100`) are
  `False`.

### Generic crashes (typos, OOM, won't-compile)
1. Read the tail of `run.log` (redirect with `> run.log 2>&1`; do **not** `tee`/flood context).
2. Trivial bug (typo, missing param key) → fix, re-run.
3. Fundamentally broken (OOM, compile fail) → revert, log CRASH, move on. VRAM must stay under
   ~80% of `GpuTarget.hbm_bytes`; treat an overflow as a regression and revert.

---

## 5. Optimization Tiers (Amdahl impact-first ordering, for BOTH loops)

Work tiers roughly in order; earlier tiers give larger gains at lower risk. **Always spend
budget where it moves end-to-end latency most**, optimize the region with the largest
`est_bytes`/wall fraction first (Amdahl: `speedup = 1/((1-f)+f/s)`).

### 5A. Loop 1, Instruction tuning (per micro-kernel, à la AutoKernel)
1. **Tile / block-size sweep** (biggest single win). Sweep `N_tile`/`M_tile`/`K` blocks and
   `threads_per_block` through powers of two (64/128/256). Rectangular tiles for GEMM.
   `num_warps`/`num_stages` as secondary knobs.
2. **Memory access.** Coalesced (stride-1) loads; transpose an operand if needed; vectorized
   `float4`/`half2`; `__ldg`/`__restrict__`; 128-byte alignment.
3. **Tensor cores.** wmma/WGMMA 16×16×16 fp16 with **fp32 accumulate** (matches the reference
   conventions); cast to output dtype only in the epilogue. 2–4× on matmul-like ops.
4. **Software pipelining / `cp.async`.** Overlap global→shared loads with compute
   (`num_stages=3..4`). Online softmax for attention/softmax; Welford for norms.
5. **Arch-specific (the retargeting surface, search, never hardcode):** Blackwell/Hopper TMA
   (`cp.async.bulk`), DSMEM/cluster shared memory, FP8 tensor cores. Keyed off `GpuTarget`
   fields (`sm_arch`, `smem_bytes_per_block_optin`, `smem_bytes_per_sm`).
6. **Kernel-specific tricks:** epilogue fusion, split-K for tall-skinny, causal early-exit,
   `__sincosf` for RoPE.
   *Anti-patterns:* huge blocks (register spill), too many stages (SMEM overflow), bank
   conflicts, atomics in hot paths, warp divergence in inner loops.

### 5B. Loop 2, Schedule tuning (edit `ScheduleConfig` only)
1. **Software-pipelining / prefetch depth (`pipelining_depth`), THE BIGGEST WIN.** This hides
   the inter-op HBM bubble by prefetching the next instruction's weights while the current one
   computes, the entire reason a megakernel beats one-kernel-per-op. Sweep `0,1,2,3,4`.
   Diminishing past the depth where prefetch fully covers compute; too deep wastes SMEM/L2.
2. **Tiling (`tiling`).** Per-archetype tile extents (`{"gemv": {"N_tile": …}}`). Trades
   parallelism (more tiles = more SMs busy, finer load-balance) against per-tile overhead and
   counter-tree width. Co-tune with `threads_per_block`.
3. **Fusion grouping (`fusion_grouping`).** Fuse adjacent ops into one resident task group so the
   activation never round-trips to HBM (norm→gemv, gate/up→silu_mul). Bigger groups = fewer HBM
   bounces but larger SMEM live-set; watch `smem_bytes_per_block` vs the target opt-in cap.
4. **SM load-balance (`sm_assignment`).** `"load_balance"` over `"round_robin"`; explicit
   `{task_id: sm}` only when measurement shows a specific imbalance. Goal: no SM idle while
   another has a queue tail. Recall the per-SM-queue ordering invariant (§2/§6).
5. **Page allocation / reuse (`page_allocation`).** `"graph_color"` reuses a `Page` across
   non-overlapping activation live ranges (less scratch, better cache residency) vs `"linear"`
   (simplest, most scratch). Verify no WAR/WAW clobber, `validate()` warns on unordered
   page-aliasing reuse.
6. **Launch config (`threads_per_block`, `smem_bytes_per_block`).** Proven against the target's
   occupancy by the loader; the dynamic SMEM opt-in MUST be `<= GpuTarget.smem_bytes_per_block_optin`
   (NOT `smem_bytes_per_sm`), or the loader rejects it.

---

## 6. `results.tsv` schema & flywheel logging discipline

Plain TSV, **tab-separated** (commas break in `description`). Human-readable, git-friendly, no
DB. Append one row per experiment. **Do not commit `results.tsv`** (leave untracked); the
flywheel ingester copies it into `flywheel/` for the corpus.

**Header (exact columns, create with this first row):**
```
experiment	tag	loop	model	gpu	regime	kept	correctness	latency_us	pct_of_roofline	schedule_or_kernel_id	description
```

| column | meaning |
|---|---|
| `experiment` | monotonic integer id within the campaign |
| `tag` | campaign tag, e.g. `jun05-toyllama-rtx5090` |
| `loop` | `1` (instruction) or `2` (schedule), which loop produced this row |
| `model` | `meta['model']`, e.g. `toy-llama` / `Llama-3.2-1B` |
| `gpu` | `target.name` (flywheel derives GPU from `target.name`, not `meta['gpu']`) |
| `regime` | `single-stream` \| `decode` \| `prefill` \| `continuous-batch` |
| `kept` | `keep` \| `revert` (the §3 decision) |
| `correctness` | `PASS` \| `FAIL` \| `REJECTED` \| `TIMEOUT` \| `CRASH` |
| `latency_us` | measured per-token latency; **blank** if not PASS (never a number without correctness) |
| `pct_of_roofline` | `bandwidth_bound_us / measured_us * 100` (distance to the floor) |
| `schedule_or_kernel_id` | content hash / path of the `ScheduleConfig` (loop 2) or kernel file (loop 1) |
| `description` | one-line hypothesis + outcome (no tabs) |

**Discipline:**
- **Every** experiment logs a row, including REJECTED/TIMEOUT/CRASH (negative results train the
  prior on what *not* to propose). A REJECTED schedule still serializes via
  `MegakernelProgram.to_json()` into `flywheel/` keyed by `schedule_or_kernel_id`.
- `pct_of_roofline` uses `GpuTarget.bandwidth_bound_us(prog.total_weight_bytes())`, the honest
  single-stream decode floor. Always report distance to the bound.
- Measured numbers belong only to the GPU named in `gpu`. Never transcribe a number for a chip
  you did not run on (datacenter `GpuTarget`s are cost-model/roofline only, `note="Spec only."`).

---

## 7. Correctness Before Performance (non-negotiable)

- **Never a latency number without its paired correctness result.** `eval/bench.py` must refuse
  to emit a latency without a verdict from `eval/oracle.py`. A row with `latency_us` set and
  `correctness != PASS` is a bug in your logging.
- **The oracle is the whole model.** Correctness = full-model **logit equivalence** within
  tolerance vs eager `models.toy.ToyLlama.forward(input_ids) -> logits[S, vocab]`, **plus**
  generated-token divergence over a few hundred decoded tokens. The GPU-free proof is
  `ReferenceVM(prog, weights).run(inputs, kv)` matching eager (see `vm/verify_vm.py`:
  3-instruction DAG and full SwiGLU block both match eager within `rtol/atol ~2e-5`).
- **Tolerances:** fp32 reference path `rtol=atol≈1e-5..2e-5`; relax for fp16/bf16 but justify it
  in `description`. A divergence beyond tolerance is a FAIL → revert.
- **The reference is ground truth.** If the CUDA VM disagrees with `vm/reference_vm.py` +
  `instructions/reference.py`, the CUDA side is wrong by definition.

---

## 8. Build Roadmap (M0 → M∞) and the FIRST ACTIONS

From the spec. This is your plan, not a ceiling, build past it toward the north star.

- **M0, VM + ABI on one GPU.** Hand-schedule one small dense model (`models.toy.ToyLlama`, then
  Qwen3-0.6B / Llama-1B) into a correct megakernel vs eager. Proves the runtime. *Verify the VM
  to death first* (`vm/verify_vm.py`). **← we are here / next.**
- **M1, Instruction generation.** AutoKernel loop emits ABI-conformant instructions to
  `instructions/{triton,cuda}/`; `instructions/verify_inst.py` checks each vs
  `instructions/reference.py` + isolated bench; swap into M0.
- **M2, Schedule search.** `schedule/search.py` (cost-model explore + on-hardware exploit +
  evolve) beats the M0 hand-schedule and MPK on ≥1 (model, GPU) point, measured. Core thesis.
- **M3, Generality.** `schedule/graph.py` HF importer → arbitrary dense models, one command.
- **M4, Self-retargeting.** Add a second arch (H100 / MI300X) with *no* hand-written kernels -
  search generates them. Proves the moat.
- **M5, Dynamism.** `dynamism/`: continuous batching + dynamic shapes, then MoE routing.
- **M6, Multi-GPU.** Fuse compute + communication (`ALLREDUCE_SHARD`,
  `__threadfence_system()`) into the megakernel.
- **M∞, The flywheel.** Train a learned schedule prior from the `flywheel/` corpus so each run
  starts smarter. This is what compounds into the decade-long lead.

### FIRST ACTIONS for the agent (start here, in order)
1. **Read the contracts.** `schedule/ir.py` (esp. `validate`, `ScheduleConfig`,
   `InstructionKind`, `TARGETS`, the sync-model docstring), `vm/abi.h`, `vm/reference_vm.py`,
   `instructions/reference.py`, `models/toy.py`. Then `vm/verify_vm.py`, `tests/test_ir_smoke.py`,
   `tests/test_validator_races.py` for the *valid-program* patterns (`_tiled_gemv`, all-join).
2. **Prove the foundation (no GPU):**
   ```bash
   uv run python vm/verify_vm.py        # VM semantics: DAG executes, no deadlock, == eager
   uv run pytest tests/                 # IR smoke + validator races + ABI<->IR sync
   ```
3. **Build a campaign branch** `amk/<date>-<model>-<gpu>` and create `results.tsv` with the §6
   header.
4. **Stand up the fixed eval** (`eval/oracle.py` logit-equivalence vs `ToyLlama`,
   `eval/bench.py` per-token latency that refuses to emit latency without a correctness verdict,
   `eval/roofline.py` distance to `bandwidth_bound_us`).
5. **Hand-lower `ToyLlama` end-to-end** through `MegakernelProgram` (embed → per-layer
   {input_norm→q/k/v gemv tiles→rope→kv_append→attention→o_proj→residual; post_norm→gate/up
   tiles→silu_mul→down tiles→residual} → final_norm → lm_head tiles → sample_argmax),
   `validate()` it, run it through `ReferenceVM`, assert it matches eager. That is M0.
6. **Enter Loop 2.** Propose a `ScheduleConfig`, `validate()`, eval, keep/revert, log, repeat -
   sweeping `pipelining_depth` first (§5B.1). Don't stop.

---

## 8B. Unattended autoresearch, point it at a `(model, gpu)` and sleep

You do not have to drive the keep/revert loop by hand. `autoresearch.py` is the headless "run it
and sleep" driver: point it at a `(model, gpu)`, give it an iteration or wall-clock budget, and it
runs the whole §3 methodology unattended, propose → lower → `validate()` → correctness-gated eval
→ keep/revert → record → checkpoint, for hours, resumable, and crash-proof.

```bash
# 20 iterations on the cost model (fast, deterministic, good for a smoke run)
uv run python amk_cli.py autoresearch toy --gpu rtx5090 --iters 20 --device cpu
# or: sleep on it for 2 hours, measuring on the real GPU (correctness-gated)
uv run python amk_cli.py autoresearch toy --gpu rtx5090 --minutes 120 --device cuda
# ignore the flywheel prior (pure exploration; grows the corpus from scratch)
uv run python amk_cli.py autoresearch toy --gpu rtx5090 --iters 50 --cold
```

What it does every iteration (all of §3, automatically):

1. **Propose** a `ScheduleConfig`. Unless `--cold`, the proposal is biased by the flywheel prior
   (`flywheel/prior.py`): the *first* incumbent is the best `warm_start` seed from the corpus for
   the nearest `(model_shape, gpu)` shape, and exploitation proposals are ranked best-first by the
   learned/kNN prior (`prior.rank`). Exploration (mutation / fresh random) is mixed in
   epsilon-greedy so the search never collapses onto the prior. **The prior is advisory only**, a
   bad suggestion still has to pass the same gate, so it can only ever lose.
2. **Lower → `validate()` → correctness-gated evaluate** via `harness.evaluate` (the §4/§7 honesty
   gate is reused verbatim, a latency is never emitted without a correctness PASS; `latency_kind`
   is `measured-gpu` or `predicted`, never mislabeled).
3. **Keep/revert:** keep iff correct AND ≥ 1% faster than the incumbent (else revert). Kept correct
   points are appended to `flywheel/corpus.jsonl` (the moat) and recorded in the campaign state.
4. **Record + checkpoint:** every experiment (kept / revert / rejected / **crash** / **timeout**)
   is appended to `results.tsv` via `flywheel.log` with its real correctness, and the campaign
   state (`workspace/amk_orchestration_state.json`) is checkpointed. A crash / CUDA-error / timeout
   in one iteration is logged and the loop **continues**, one failure never stops the run.
   Re-running the same command **resumes** the campaign from the checkpoint.

### The orchestrator (`amk_orchestrate.py`), the AutoKernel-style campaign brain

The same state machine a coding agent talks to via `status` / `next` / `record` / `report`
(faithful to `autoresearch/orchestrate.py`), but in AMK's latency / pct-of-roofline units:

```bash
uv run python amk_orchestrate.py status   # baseline, best, speedup, region split, plateau counter
uv run python amk_orchestrate.py next      # continue-or-stop (MOVE_ON_CRITERIA), + hottest region
uv run python amk_orchestrate.py report    # the aggregate campaign report
```

`MOVE_ON_CRITERIA` mirror AutoKernel: **`consecutive_reverts ≥ 8`** (plateau, stop grinding),
**`pct_roofline ≤ 110%`** (within 10% of the weights/bandwidth floor, near-roofline, done),
**`max_minutes`** (wall-clock budget), **`speedup ≥ 3×`** vs the default baseline. When a region
plateaus, mutation naturally explores elsewhere; when the *campaign* trips a criterion the state
flips to `done` and `next` says STOP. The state also tracks the per-region (attention / mlp /
lm_head) share of the critical path so `next` can point you at the region worth optimizing.

The flywheel makes every future run start smarter: the cold-vs-warm proof is
`paper/exp_flywheel_learning.py` (**ships with the paper artifact, not this OSS tree, it is
gitignored**; a warm run begins from the corpus's best schedule, not the
default, and reaches the cold best in fewer iterations, honest about the gain size). See
`HARNESS.md` for the agent + headless contract.

---

## 9. Hard Constraints (violating any is a bug)

1. **Never edit the locked contracts:** `schedule/ir.py`, `vm/abi.h`, `vm/reference_vm.py`,
   `instructions/reference.py`, `models/toy.py`. Build against them.
2. **Loop 2 edits `ScheduleConfig` ONLY.** Never hand-write megakernel/VM CUDA in the schedule
   loop.
3. **Never launch a schedule `validate()` rejects.** The VM refuses; you must too.
4. **Never skip correctness.** Every experiment passes the oracle before its latency counts.
5. **Never a latency number without its correctness result.**
6. **No new heavy dependencies.** torch + numpy only.
7. **Simpler wins when performance is equal** (±1%).
8. **One focused change per experiment.** Log every experiment (incl. REJECTED/TIMEOUT/CRASH).
9. **A hung GPU never stops the run**, timeout, abort flag, kill, revert, log, continue.
10. **Don't commit `results.tsv` / `run.log` / `workspace/`** (gitignored runtime artifacts).
11. **Respect Amdahl / move-on criteria** (§3), don't grind a plateaued or near-roofline region.
12. **Honest provenance**, measured numbers only for the GPU you ran on.

---

*If anything here is wrong or stale, the fix is to edit THIS file. It is the soul of the repo:
the difference between an agent that runs for ten hours and one that hangs on the third.*
