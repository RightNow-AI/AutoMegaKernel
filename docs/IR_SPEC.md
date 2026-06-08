# AutoMegaKernel IR Specification (`docs/IR_SPEC.md`)

> The standalone specification of the **AMK megakernel intermediate representation**, the
> "DWG-format" asset AMK owns. It defines the SM-level task-DAG, the counter-synchronization
> model and its two safety invariants, the buffer/page memory model, the `ScheduleConfig` search
> surface, the on-disk JSON format, and the mapping to the on-device ABI in `vm/abi.h`. The
> normative Python reference is [`schedule/ir.py`](../schedule/ir.py); this document is written so
> an external team could build a compatible importer, validator, or VM. Where prose and code
> disagree, **`schedule/ir.py` wins.**
>
> Versions: `IR_VERSION = "0.2.0"` (on-disk format), `ABI_VERSION = "0.2"` (on-device ABI,
> mirrors `AMK_ABI_VERSION_{MAJOR,MINOR}` in `vm/abi.h`). Minor bumps are additive/compatible;
> `MegakernelProgram.from_dict` rejects a *major* mismatch. The format is dependency-free pure
> Python + stdlib, you can read, validate, diff, and version a schedule on a laptop with no GPU.

---

## 1. Overview & data model

A program is a `schedule.ir.MegakernelProgram`: a complete, runnable-after-validation megakernel
schedule. It is the artifact `compile.py` emits, the flywheel stores, and the VM loads. Its parts:

| Field | Type | Role |
|---|---|---|
| `meta` | `dict` | `{model, gpu, regime, dtype, notes…}`, provenance for `results.tsv`. |
| `target` | `GpuTarget \| None` | the GPU described **as data** (the retargeting surface). |
| `buffers` | `list[Buffer]` | every named tensor (weights, activations, KV, IO, consts). |
| `counters` | `list[Counter]` | synchronization counters (the only cross-task signalling). |
| `tasks` | `list[Task]` | the DAG nodes; each == one ABI instruction invocation on one SM. |
| `pages` | `PageAllocation \| None` | physical scratch-slot binding for activations (Layer 2 output). |
| `config` | `ScheduleConfig \| None` | the Loop-2 search point that produced this lowering. |
| `ir_version`, `abi_version` | `str` | format/ABI versions. |

The **task-DAG** is the heart: nodes are `Task`s, edges are producer→consumer relations induced
by counters (`MegakernelProgram.dependency_edges()`). A forward pass is a DAG; execution is a
topological walk of it with monotonic counters. There are **no locks and no arbitrary
signalling**, only counter increments and static-threshold waits.

### 1.1 Enums (numeric codes are CANONICAL, mirrored in `vm/abi.h`)

- **`DType`** (`schedule.ir.DType`): `F32=0, F16=1, BF16=2, F8E4M3=3, F8E5M2=4, I32=5, I8=6,
  I4=7` (packed 4-bit, two per byte), `U8=8, BOOL=9`. `.bits` / `.nbytes(count)` give sizes
  (ceil for sub-byte packed types).
- **`MemSpace`** (`schedule.ir.MemSpace`): `HBM=0` (weights/KV/IO), `GLOBAL_SCRATCH=1` (large
  activations), `SMEM=2` (hot activations, the megakernel win), `REGISTER=3` (lowering hint).
- **`BufferKind`** (`schedule.ir.BufferKind`): `WEIGHT=0`, `ACTIVATION=1`, `KV_CACHE=2`,
  `IO_INPUT=3`, `IO_OUTPUT=4`, `CONST=5`. The read-only kinds (`WEIGHT`, `CONST`, `IO_INPUT`,
  i.e. `_READONLY_KINDS`) never require a producer edge to read.
- **`InstructionKind`** (the opcodes, §7): `NOP=0 … ATTENTION_COMBINE=18`. Extend by appending;
  never renumber.

### 1.2 `Buffer`
```
Buffer(id, name, kind: BufferKind, dtype: DType, shape: tuple[int,...],
       space: MemSpace = HBM, source: str | None = None)
```
`source` is the dotted state-dict key for `WEIGHT`/`CONST` buffers (else `None`). Derived:
`numel`, `nbytes`, `contiguous_strides()` (row-major element strides → `amk_buffer_t.stride`).
Rank (`len(shape)`) MUST be `<= ABI_MAX_RANK (4)`.

### 1.3 `Task` (a DAG node == one ABI instruction)
```
Task(id, op: InstructionKind, inputs: list[int], outputs: list[int], out_counter: int,
     waits: list[Wait] = [], params: dict = {}, sm: int | None = None,
     est_bytes=0, est_flops=0, label="")
```
- `inputs`/`outputs` are **buffer ids** (read / written). Caps: `<= ABI_MAX_INPUTS (8)` inputs,
  `<= ABI_MAX_OUTPUTS (4)` outputs.
- `out_counter`: the **single** counter this task increments by **1** on completion, after a
  release fence ordering all its output-buffer writes. Meaning: "all my outputs are written and
  visible." Exactly one increment per task.
- `waits`: preconditions (`<= ABI_MAX_WAITS (8)`).
- `params`: op-specific scalars (§7); keys/types validated against `PARAM_FIELDS`.
- `sm`: SM/worker assignment, a **lowering OUTPUT** (assigned from `ScheduleConfig.sm_assignment`;
  `None` = unassigned). Agents/search edit the config, never this field.
- `est_bytes`/`est_flops`: cost-model hints (weights dominate `est_bytes`); `label` is human text.

### 1.4 `Counter` and `Wait`
```
Counter(id, init: int = 0, note: str = "")     # init is always 0 for a fresh forward pass
Wait(counter: int, threshold: int)             # do not execute until counters[counter] >= threshold
```
A counter is a `uint32` (see `amk_counter_t`), monotonically incremented by its producer tasks,
never decremented. `threshold` MUST be a statically-known positive int; for a counter with >1
producer it MUST equal the producer count (§3.2).

---

## 2. The synchronization model

Each task, on completion, does exactly one `out_counter += 1` (producers only **increment**)
after a release fence ordering ALL of its output-buffer writes before the increment. Each task,
before executing, **waits** on a set of `(counter, threshold)` pairs with statically-known
thresholds (consumers only **wait**, never signal).

Producer→consumer edges (`MegakernelProgram.dependency_edges()`): for every counter, each
producer task precedes every task that waits on it. The acyclicity of this graph guarantees a
topological order exists (`MegakernelProgram.topological_order()` via Kahn's algorithm; returns
`None` on a cycle). The reference VM (`vm/reference_vm.py`) and CUDA VM both execute by repeatedly
firing every task whose every wait is satisfied, incrementing counters, a counter-driven walk of
the DAG.

**Decode-loop model (frozen).** One kernel launch == one forward pass == one decoded token.
Counters are host-memset to zero before each launch (`Counter.init = 0`); `KV_CACHE` persists in
HBM across launches; positions advance via `params` (`pos`, `kv_start`, `kv_len`) between
launches. The host drives the autoregressive loop. This keeps each launch under the Windows WDDM
~2s TDR watchdog (`GpuTarget.wddm_tdr`) while preserving the megakernel-within-a-step win.

---

## 3. The two invariants `validate()` enforces

`schedule.ir.validate(prog) -> ValidationResult` statically proves a program is safe to load. A
`REJECTED` result MUST prevent launch; the VM refuses anything `validate()` rejects.
`validate()` **never raises** on a malformed program, it always returns a result (the
clean-signal contract). `ValidationResult` carries `ok`, `errors`, `warnings`, `stats` and
`report()`.

### 3.1 Deadlock-freedom
- Referential integrity + opcode arity (`OP_REGISTRY`) + required params + ABI caps + param
  type/key sanity (`PARAM_FIELDS`: `'i'`→int32, `'f'`→real).
- Every `Wait.threshold` satisfies `1 <= threshold <= #producers(counter)`. A wait on a counter
  with **no** producer, or `threshold > #producers`, is unsatisfiable → REJECTED.
- The producer→consumer graph is **acyclic** (`topological_order() != None`); a cycle → REJECTED
  with a witness from `_describe_cycle` (iterative DFS, never `RecursionError`, even at 5000+
  nodes).
- **Per-SM queue ordering** (once `sm` is assigned): each SM's serial queue (task-list order)
  must be a linear extension of the DAG, for every edge `a→b` with `sm[a]==sm[b]`, `a` must
  precede `b` in that SM's queue, else the SM blocks on a counter only its own later entry could
  signal. Assigned `sm` must be in `[0, target.num_sms)`.

### 3.2 Race-freedom (the subtle one)
A counter carries a **count, not which producer finished**. Therefore:
- **Shared-counter all-join rule.** A counter with `>1` producer is a true join: **every** wait
  on it MUST use `threshold == #producers`. A partial wait (`1 < t < #producers`) is a
  "first-k-of-N" race (the wrong producers can satisfy it) → REJECTED.
- **Transitive happens-before provenance.** For every ACTIVATION / IO read, there must be a
  transitive predecessor (through dependency edges) that *wrote* that buffer. `validate()` walks
  the topo order maintaining, per task, the bitmask of buffers written by transitive predecessors
  (`avail[t] = ext_mask ∪ over preds (avail[p] ∪ out_mask[p])`); a read whose bit is unset is a
  data RACE → REJECTED. Read-only kinds (`_READONLY_KINDS`) are pre-set in `ext_mask`.
- **KV_CACHE ordering.** A `KV_CACHE` written this pass (`KV_APPEND`) may be read only by tasks
  ordered *after* the append. The writer reading its own cache (prior-step state) is fine; any
  *other* reader without a happens-before edge from the `KV_APPEND` is a RACE → REJECTED.

### 3.3 Additional checks
- **Page-aliasing (WAR/WAW).** When `pages` is present (graphs `<= 4000` tasks), if a `Page` is
  reused by two activation buffers but a reader of the first and a writer of the second are
  unordered, a **warning** is emitted (possible clobber).
- **Output reachability.** Every `IO_OUTPUT` buffer must be produced by some task, else REJECTED.
- **GPU labeling.** `meta['gpu'] != target.name` is a warning (flywheel derives GPU from
  `target.name`).

### 3.4 Backstops (not part of `validate`, used by builders/tests)
- `MegakernelProgram.simulate_counters()`, static reachability cross-check; returns
  `(order, stuck)`; `stuck` non-empty iff a wait is permanently unsatisfiable.
- `MegakernelProgram.simulate_adversarial(seeds=16)`, fires ready tasks in varied worst-case
  interleavings and asserts every transient input was written by a prior-fired task; returns a
  list of race violations (the dynamic backstop to the static checks).

---

## 4. The buffer / page memory model

- **`WEIGHT` / `CONST`** live in HBM, read-only, bound by `Buffer.source` into the model
  state-dict. The bandwidth bound is on these (`MegakernelProgram.total_weight_bytes()`).
- **`KV_CACHE`** is persistent read/write HBM state (written by `KV_APPEND`, read by attention),
  surviving across launches.
- **`IO_INPUT` / `IO_OUTPUT`** are the model's input (token ids / embeds) and output (logits /
  sampled token) in HBM.
- **`ACTIVATION`** is transient and **logical**: it is bound to a physical `Page` for its live
  range by the allocator; non-overlapping live ranges may share a page (graph-coloring reuse).

`Page(id, space: MemSpace, nbytes, live_start=-1, live_end=-1)` is a physical scratch slot.
`PageAllocation(buffer_to_page: dict[int,int], pages: list[Page])` is the allocation output
(`total_scratch_bytes` sums the pages). Activations in `SMEM`/`GLOBAL_SCRATCH` do **not**
round-trip to HBM between ops, that is the megakernel win. On device, the host resolves every
buffer id to a fixed `void* ptr` (already offset for paged activations) before launch; the VM
never allocates HBM mid-flight.

---

## 5. The `ScheduleConfig` search surface (Layer-2 / Loop-2 edit surface)

`ScheduleConfig` is the structured object a coding agent proposes; the **frozen VM
deterministically lowers it** into a runnable `MegakernelProgram`. The agent never writes kernel
code, it only chooses a point in this search space the VM knows how to realize safely. It is also
exactly what is logged to the flywheel as the "schedule" column.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `tiling` | `dict[str, dict[str,int]]` | `{}` | per-op-archetype tile sizes, e.g. `{"gemv": {"N_tile": 256}, "attention": {"kv_block": 128}}`. |
| `fusion_grouping` | `list[list[str]]` | `[]` | adjacent op-name groups fused into one resident task group. |
| `sm_assignment` | `str \| dict[int,int]` | `"load_balance"` | `"round_robin"` \| `"load_balance"` \| explicit `{task_id: sm}`. INPUT only; lowerer resolves it into each `Task.sm`. |
| `pipelining_depth` | `int` | `2` | instructions-ahead to prefetch weights, **the biggest megakernel win** (hides the inter-op HBM bubble). `0` = no prefetch. |
| `page_allocation` | `str` | `"graph_color"` | `"linear"` \| `"graph_color"` \| `"none"`. |
| `threads_per_block` | `int` | `256` | block size of the persistent VM kernel (occupancy-proven by the loader). |
| `smem_bytes_per_block` | `int` | `0` | dynamic SMEM opt-in per block; MUST be `<= GpuTarget.smem_bytes_per_block_optin`. |

`ScheduleConfig.to_dict()` serializes int-keyed `sm_assignment` with string keys (re-parsed on
load). Search/agents tune these fields; the deterministic lowering + `validate()` guarantee the
result is safe regardless of the point chosen.

### 5.1 `GpuTarget` (the retargeting surface, a GPU as data, never branches)
`GpuTarget(name, sm_arch, num_sms, smem_bytes_per_sm, smem_bytes_per_block_optin, regs_per_sm,
max_threads_per_sm, max_regs_per_thread, l2_bytes, hbm_bytes, hbm_bandwidth_gbs, fp16_tflops,
clock_ghz=0.0, supports_cooperative=True, wddm_tdr=False, note="")`.
`bandwidth_bound_us(weight_bytes) = weight_bytes / (hbm_bandwidth_gbs*1e9) * 1e6`, the honest
single-stream decode floor. Built-in registry `schedule.ir.TARGETS`: `rtx5090` (sm_120, 82 SMs,
`wddm_tdr=True`, measured on this machine), `b200` (sm_100, spec only), `h100` (sm_90, spec
only). Add a chip by adding a record, never by editing the scheduler.

---

## 6. On-disk JSON format

Serialization is explicit, human-readable, **additive-compatible** JSON (git-friendly).
`MegakernelProgram.to_json(indent=2)` / `.save(path)` write it;
`MegakernelProgram.from_json(s)` / `.load(path)` / `.from_dict(d)` read it. Enums serialize **by
name** (`kind`, `dtype`, `space`, `op` are strings like `"WEIGHT"`, `"GEMV_TILE"`). Unknown
(newer) fields in `target`/`config` are dropped on load via `_filter_known` (forward
compatibility); a *major* `ir_version` mismatch raises. Round-trip is **stable**
(`from_json(to_json(p)).to_json() == to_json(p)`).

```json
{
  "ir_version": "0.2.0",
  "abi_version": "0.2",
  "meta": {"model": "toy", "gpu": "rtx5090"},
  "target": {"name": "rtx5090", "sm_arch": 120, "num_sms": 82, "...": "..."},
  "buffers": [
    {"id": 0, "name": "x", "kind": "IO_INPUT", "dtype": "F16",
     "shape": [1, 16], "space": "HBM", "source": null},
    {"id": 1, "name": "proj.w", "kind": "WEIGHT", "dtype": "F16",
     "shape": [16, 16], "space": "HBM", "source": "proj.weight"}
  ],
  "counters": [{"id": 0, "init": 0, "note": "rmsnorm done"}],
  "tasks": [
    {"id": 0, "op": "RMSNORM", "inputs": [0, 2], "outputs": [3], "out_counter": 0,
     "waits": [], "params": {"eps": 1e-06, "hidden": 16}, "sm": null,
     "est_bytes": 0, "est_flops": 0, "label": "rmsnorm"},
    {"id": 1, "op": "GEMV_TILE", "inputs": [3, 1], "outputs": [4], "out_counter": 1,
     "waits": [{"counter": 0, "threshold": 1}],
     "params": {"K": 16, "N_tile": 16, "n_off": 0}, "sm": null, "label": "gemv"}
  ],
  "pages": {"buffer_to_page": {"3": 0, "4": 0}, "pages": [
    {"id": 0, "space": "SMEM", "nbytes": 64, "live_start": 0, "live_end": 1}]},
  "config": {"tiling": {}, "fusion_grouping": [], "sm_assignment": "load_balance",
             "pipelining_depth": 2, "page_allocation": "graph_color",
             "threads_per_block": 256, "smem_bytes_per_block": 0}
}
```
`pages` and `config` are `null` until those passes run. `buffer_to_page` keys are JSON strings
(int-parsed on load).

---

## 7. The instruction set (opcodes) and `OP_REGISTRY`

Opcodes are the Layer-1 ABI archetypes. Numeric codes are canonical (mirrored in `vm/abi.h` as
`AMK_OP_*`). `OP_REGISTRY[kind] = OpSpec(kind, min_inputs, max_inputs, n_outputs,
required_params, note)` pins arity and the params that make each op mathematically correct;
`validate()` checks against it. `-1` means variadic/variable (still capped by ABI limits).

| Opcode (code) | inputs | outputs | required params | semantics (`instructions/reference.py`) |
|---|---|---|---|---|
| `NOP` (0) | 0 | 0 |, | no-op |
| `COPY` (1) | 1 | 1 |, | page→page move |
| `EMBED` (2) | 2 | 1 | `hidden` | `[ids, table[V,H]]` → gathered rows |
| `RMSNORM` (3) | 2 | 1 | `eps, hidden` | `x*rsqrt(mean(x²)+eps)*w` |
| `LAYERNORM` (4) | 2–3 | 1 | `eps, hidden` | LayerNorm(x, w[, b]) |
| `GEMV_TILE` (5) | 2–3 | 1 | `K, N_tile, n_off` | `out[..,n_off:n_off+N_tile] = x @ W[n_off:…].T` |
| `GEMM_TILE` (6) | 2–3 | 1 | `M_tile, K, N_tile, n_off` | tiled GEMM (prefill/batch) |
| `ATTENTION_TILE` (7) | 3–4 | 1 | `head_dim, kv_start, kv_len, scale, n_heads, n_kv_heads` | GQA attention over a KV window |
| `ROPE` (8) | 2 | 1 | `head_dim, theta` | Llama rotate-half rotary embedding |
| `SILU_MUL` (9) | 2 | 1 |, | SwiGLU `silu(gate)*up` |
| `GELU` (10) | 1 | 1 |, | GELU |
| `ADD` (11) | 2 | 1 |, | residual add |
| `MUL` (12) | 1–2 | 1 |, | elementwise / scale |
| `DEQUANT` (13) | 2–3 | 1 | `qdtype, group` | int4/int8 + scales → fp tile |
| `SOFTMAX` (14) | 1 | 1 |, | softmax over `dim` |
| `ALLREDUCE_SHARD` (15) | 1–8 | 1 |, | tensor-parallel shard reduce (multi-GPU) |
| `KV_APPEND` (16) | 2 | 1 | `pos` | append new k/v into the KV cache at `pos` |
| `SAMPLE_ARGMAX` (17) | 1 | 1 |, | greedy logits → next token |
| `ATTENTION_COMBINE` (18) | 2–8 | 1 |, | merge per-KV-block `(out,m,l)` flash partials |

**Frozen numeric conventions** (the backends MUST match `instructions/reference.py`): weight
layout `[N_out, K_in]`, GEMV/GEMM compute `x @ W.T`, a tile writes `out[..., n_off:n_off+N_tile]`;
reductions accumulate in fp32 then cast; RoPE rotate-half; GQA with `repeat_interleave` and scale
`1/sqrt(head_dim)`. Known scalar param fields and marshalled types are in `PARAM_FIELDS`
(int32 `'i'` or float `'f'`); unknown keys warn (they will not marshal to `amk_params_t`).

---

## 8. ABI mapping to `vm/abi.h`

Each `Task` maps **1:1** onto an `amk_instruction_t` (a fixed-size POD). The host loader resolves
the program into the on-device `amk_program_t` tables before launch. The IR enum values and
capacity/version constants are CANONICAL; `vm/abi.h` must match, and `tests/test_abi_sync.py`
parses both and fails the build on any drift.

| IR (`schedule/ir.py`) | ABI (`vm/abi.h`) |
|---|---|
| `Task.op` | `amk_instruction_t.op` (`amk_opcode_t`) |
| `Task.inputs` / `outputs` | `int32 inputs[AMK_MAX_INPUTS]` / `outputs[AMK_MAX_OUTPUTS]` + `n_inputs`/`n_outputs` |
| `Task.waits[i].counter` / `.threshold` | `wait_counter[AMK_MAX_WAITS]` / `wait_threshold[AMK_MAX_WAITS]` + `n_waits` |
| `Task.out_counter` | `amk_instruction_t.out_counter` |
| `Task.sm` | `amk_instruction_t.sm` (`>= 0`; loader rejects `< 0`) |
| `Task.params` | `amk_params_t` (keys/types per `PARAM_FIELDS`) |
| `Buffer` | `amk_buffer_t {ptr, numel, rank, dtype, space, shape[4], stride[4]}` (element strides) |
| `Counter` | `amk_counter_t` (`uint32`, host-memset 0 before each launch) |
| `MegakernelProgram` | `amk_program_t {buffers, counters, instructions, sm_queue[][], scratch, abort_flag}` |
| `ABI_MAX_INPUTS/OUTPUTS/WAITS/RANK` | `AMK_MAX_INPUTS/OUTPUTS/WAITS/RANK` (= 8/4/8/4) |
| `DType` / `MemSpace` / `InstructionKind` codes | `amk_dtype_t` / `amk_memspace_t` / `amk_opcode_t` |
| `ABI_VERSION = "0.2"` | `AMK_ABI_VERSION_MAJOR.MINOR` (= 0.2) |

### 8.1 On-device sync contract (the runtime realizing §2)
- **`signal(c)`**: thread 0 issues a device-scope release fence (`__threadfence()`) ordering all
  output stores, then `atomicAdd(&prog.counters[c], 1u)`, then `__syncthreads()`. Cross-GPU
  counters (`ALLREDUCE_SHARD`) use `__threadfence_system()`.
- **`wait(c,t)`**: thread 0 spins on an **acquire** load the compiler may not hoist
  (`while (atomicAdd(&counters[c],0u) < t) { backoff(); if (*abort_flag) return; }`), then
  `__syncthreads()`. A plain non-volatile load is FORBIDDEN (it would hoist and spin forever).
  `backoff()` = exponential `__nanosleep`. The `abort_flag` poll is the watchdog escape.
- **Launch contract**: `cudaLaunchCooperativeKernel` (co-resident blocks make forward progress);
  `gridDim` capped by verified cooperative occupancy; dynamic SMEM opt-in `<=
  GpuTarget.smem_bytes_per_block_optin`; one launch per token to stay under the WDDM TDR. The host
  treats `cudaErrorLaunchTimeout` as a distinct TIMEOUT, not a clean REJECTED.
- **Instruction contract (Layer 1)**: each micro-kernel is exactly
  `__device__ void amk_inst_<name>(const amk_program_t&, const amk_instruction_t&)`, pure
  compute, MUST NOT touch counters or any undeclared buffer, MUST NOT launch work.

---

## 9. Conformance checklist (to build a compatible tool)

A compatible implementation MUST:
1. Represent the data model of §1 with the canonical enum codes (§1.1) and ABI caps
   (8/4/8 i/o/waits, rank 4).
2. Implement `validate()` enforcing both invariants of §3 (deadlock-freedom **and**
   race-freedom, incl. the shared-counter all-join rule, transitive happens-before, and the
   KV_CACHE ordering rule) and **refuse to load** any rejected program.
3. Execute by counter-driven scheduling (§2): fire a task only when every wait
   `counter >= threshold`; on completion increment `out_counter` by exactly 1 after a release
   fence.
4. Read/write the JSON of §6 with name-encoded enums, additive forward-compatibility, and stable
   round-trip; reject a major `ir_version` mismatch.
5. Honor the opcode semantics and frozen numeric conventions of §7 (verified against
   `instructions/reference.py`) and the ABI mapping of §8 (verified against `vm/abi.h`).

The canonical conformance oracle is `vm/reference_vm.py` (`ReferenceVM`) executing real numerics;
if a CUDA VM disagrees with it, the CUDA side is wrong by definition.
