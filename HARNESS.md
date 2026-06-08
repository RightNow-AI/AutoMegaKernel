# AMK Coding-Agent Harness, Integration Contract

This document is the contract a coding agent (Claude Code, Codex), or an NVIDIA/Google engineer
driving one, uses to generate megakernels with AutoMegaKernel (AMK). It is the AMK analogue of
AutoKernel's `read program.md → propose → eval → keep/revert` loop.

AMK has **two** AutoKernel-style agent loops, each a `read surface → propose → eval → keep/revert
→ log → repeat` cycle with the *same discipline* (correctness ALWAYS first, then a strict ≥1%
latency win) but a **different edit surface**:

```
# Loop 1, INSTRUCTION tuning: edit/search ONE ABI micro-kernel; loop1.py + amk_cli.py
uv run python amk_cli.py tune-instruction <op> --gpu <arch> --budget N   # single-kernel loop

# Loop 2, SCHEDULE search: edit ONE ScheduleConfig object; harness.py + amk_cli.py
uv run python amk_cli.py propose <model> --gpu <arch>                  # read the edit surface
uv run python amk_cli.py eval    <model> --gpu <arch> --config cfg.json # one structured verdict
uv run python amk_cli.py loop    <model> --gpu <arch> --budget N        # keep/revert autoresearch
```

`<op>` is an ABI op name (`gemv_tile`, `rmsnorm`, `silu_mul`, `add`, `rope`, `attention_tile`,
`embed`). `<model>` is `toy` / `toy-2L` (fully supported) or a HuggingFace id (best-effort via
`schedule.graph.from_hf`). `<arch>` is a registered `GpuTarget`: `rtx5090`, `b200`, `h100`,
`a100`.

> **The two loops in one sentence each.** Loop 1 tunes the *body* of a single instruction (you/the
> search edit a CUDA micro-kernel and its `-D` knobs; the oracle is the per-op reference). Loop 2
> tunes how instructions are *scheduled* into one megakernel (you edit a `ScheduleConfig`; the
> frozen VM lowers it; the oracle is the whole model). **Do not confuse their edit surfaces.**

---

## Native coding-agent integration

This file is the loop contract. AMK is *also* exposed **natively** to coding agents, same
substrate, same honesty rules, zero behavior change. See **[`docs/AGENT_HARNESS.md`](docs/AGENT_HARNESS.md)**
for the full per-integration guide. Entry points:

- **MCP server**, `amk_mcp.py` (tools `amk_doctor` / `amk_propose` / `amk_eval` / `amk_loop` /
  `amk_autoresearch` / `amk_orchestrate_status|next|report|record`); enable with
  `uv sync --extra agent` and register via `.mcp.json` (Claude Code) or `~/.codex/config.toml` (Codex).
- **Claude Code**, the `megakernel-optimization` **skill**; the `/amk-optimize`, `/amk-autoresearch`,
  `/amk-compile` **slash commands**; the `amk-megakernel-optimizer` **subagent**; the ultracode
  **/workflow** (`.claude/workflows/megakernel-autoresearch.js`); and the **/goal**
  (`.claude/goals/optimize-megakernel.md`).
- **Codex**, `AGENTS.md` + the same MCP server.

A "which mode when" table (interactive vs subagent vs workflow/ultracode vs overnight autoresearch
vs /goal) is in [`docs/AGENT_HARNESS.md` §5](docs/AGENT_HARNESS.md).

---

## 1. The two loops and their DIFFERENT edit surfaces

AMK has two independent optimization loops, **both AutoKernel-style** (read surface → propose →
eval → keep/revert → log → repeat, correctness ALWAYS first, then a strict ≥1% latency win). **They
edit different things. Do not confuse them.**

| | Loop 1, instruction (`amk tune-instruction`) | Loop 2, schedule (`amk propose/eval/loop`) |
|---|---|---|
| What a coding agent edits | ONE ABI micro-kernel file `instructions/cuda/<op>.cu` (+ its `-D` knobs in `instructions/gen.SEARCH_SPACE`) | ONE structured object: `ScheduleConfig` (a JSON dict of typed knobs) |
| Surface | hand-written device code + compile-time macros | typed knobs only, never kernel code |
| Who realizes it | you write the kernel; `instructions/_build` JIT-builds it with the `-D` flags | the **frozen** VM lowers your config deterministically (`schedule.lower.lower`) |
| Correctness oracle | the per-op reference `instructions/reference.py`, **isolated** (a wrong variant fails its own unit test) | `schedule.ir.validate()` + whole-model `ReferenceVM` vs eager |
| Failure mode | build fail (`CRASH`) / numerics mismatch (`FAIL`), **no GPU hang possible** (no persistent kernel) | `REJECTED` (no launch) or runtime `TIMEOUT` (watchdog) |
| Keep/revert | correctness first, then a strict ≥1% latency win (identical to AutoKernel's single-kernel loop) | correctness first, then a strict ≥1% latency win, simplicity tie-break |
| Driven by | `loop1.tune_instruction` / `amk tune-instruction` | `harness.propose / evaluate / loop` |
| Eval cost | ~seconds (build + isolated bench) | seconds (reference VM) → on-hardware megakernel |

Both loops log every trial to `results.tsv` via `flywheel.log` (column `loop` is `instruction` vs
`schedule`), so the flywheel corpus learns from both.

### Loop 1 is documented in §1A below; Loop 2 (the `ScheduleConfig` harness) in §2–§6.

### Why Loop 2 is safe

A bad schedule cannot hang or corrupt the GPU. Before anything launches, AMK:

1. **Lowers** your config into a task-DAG (`schedule.lower.lower`).
2. **Validates** it (`schedule.ir.validate`): proves deadlock-freedom (acyclic DAG + satisfiable
   waits + per-SM serial-queue ordering) AND race-freedom (transitive happens-before provenance
   for every activation/KV read; shared counters must be true all-joins; page reuse needs
   separated live ranges). A failure is returned as `valid=False` + a `rejected_reason`.
3. **Checks launch-config feasibility** against the target (block size a positive multiple of 32
   within [32,1024]; dynamic SMEM opt-in within the target's per-block cap). A violation is also a
   clean `valid=False`.

Only a program that passes all three is ever run. **validate-before-launch is the whole point: a
bad schedule is a clean REJECTED, not a hung GPU.**

---

## 1A. Loop 1, instruction tuning (`amk tune-instruction`)

This is AutoKernel's single-kernel loop, transferred directly. A coding agent (Claude Code, Codex)
optimizes **ONE ABI micro-kernel at a time** under the *exact same discipline* as Loop 2.

### What the agent edits

- **The kernel body:** `instructions/cuda/<op>.cu`, a `__device__` core conforming to `vm/abi.h`
  plus a thin torch wrapper. This is hand-written CUDA. The same `.cu` is reused inside the
  megakernel VM later, so a Loop-1 win is a megakernel win.
- **The searchable knobs:** the per-op `-D` macros in `instructions/gen.SEARCH_SPACE`
  (today `AMK_THREADS` = block size; add knobs as kernels grow to honour them). The kernel reads a
  macro when defined and falls back to its built-in default when absent, so **variant 0 (the
  incumbent, no `-D` flags) is byte-identical to what `verify_inst.py` validates.**

The agent never touches `vm/`, `schedule/`, or the locked `instructions/reference.py` (the oracle).

### The loop (what `tune-instruction` runs, per variant)

1. **BUILD** the kernel with the variant's `nvcc -D` flags (`instructions/_build.load_kernel`;
   distinct flags get a distinct build dir, so variants are compared in one process).
2. **VERIFY** correctness against `instructions/reference.py` on the live GPU, isolated
   (`instructions/verify_inst` `Case.run_cuda` vs `Case.run_ref`, dtype tolerance). **The reference
   is ground truth**; a mismatch is a `FAIL`.
3. **MICROBENCH** latency with CUDA events (warmup + timed iters, median-ish), **only after a
   correctness PASS** (a wrong kernel is never timed; the honesty rule).
4. **KEEP / REVERT:** correctness ALWAYS first, then a strict **≥1% latency win** over the
   incumbent. A correct-but-not-faster variant is `tried` (logged, discarded); a build failure is
   `rejected`/`CRASH`; a numerics mismatch is `revert`/`FAIL`.
5. **LOG** one row to `results.tsv` (`loop=instruction`, `kernel_id=<op>[<variant>]`,
   `latency_us` blank unless PASS, `pct_of_roofline` from `GpuTarget.bandwidth_bound_us` of the
   op's measured byte traffic).

**No GPU hang is possible in this loop**, there is no persistent megakernel; a bad variant fails
its own isolated unit test. That is why Loop 1 has no `validate()`/watchdog machinery: it does not
need it.

### The JSON summary

`amk tune-instruction <op> --gpu <arch> --budget N` prints **only** JSON on stdout (build chatter →
stderr), safe to pipe into `jq`/`json.loads`:

```jsonc
{
  "op": "gemv_tile", "gpu": "rtx5090", "dtype": "fp32",
  "device": "NVIDIA GeForce RTX 5090 Laptop GPU",
  "trials": 4, "n_correct": 4,
  "best_variant": "AMK_THREADS=512",  // the kept winner (or null if none was correct)
  "best_us": 45.78,                   // real CUDA-event median of the best correct variant
  "baseline_us": 72.43,               // variant 0 (built-in defaults) measured latency
  "speedup": 1.582,                   // baseline_us / best_us, report this honestly, even ~1.0x
  "all_correct": true,                // EVERY built variant passed the reference oracle
  "pct_of_roofline": 2.57,            // best_us vs the op's HBM-traffic floor (bandwidth_bound_us)
  "bound_us": 1.17, "op_bytes": 1052672,
  "results_tsv": "workspace\\results.tsv",
  "trials_detail": [ /* per-variant: variant, built, correct, latency_us, kept, note */ ]
}
```

Exit code is `0` iff every built variant was correct **and** a correct best was found (gate your
agent loop on it). `all_correct=true` is the Loop-1 `correctness_preserved` guarantee: every kept
variant passed `verify_inst` vs the locked reference.

### Copy-paste agent loop (Loop 1)

```bash
# Tune one ABI micro-kernel end-to-end on the real GPU (build → verify → bench → keep/revert).
uv run python amk_cli.py tune-instruction gemv_tile --gpu rtx5090 --budget 6
uv run python amk_cli.py tune-instruction rmsnorm   --gpu rtx5090 --budget 4 --dtype fp16

# To add a NEW knob/variant: edit instructions/cuda/<op>.cu to read a -D macro, add that macro's
# candidate values to instructions/gen.SEARCH_SPACE["<op>"], then re-run tune-instruction.
# Variant 0 (no -D) stays the incumbent baseline that verify_inst validates.
```

```python
# Programmatic (Python)
import loop1
out = loop1.tune_instruction("gemv_tile", "rtx5090", budget=6)   # builds + times on local GPU
assert out["all_correct"]                                         # every variant matched reference
print(out["best_variant"], out["best_us"], "us", "speedup", out["speedup"])
```

**Keep/revert rules (identical to AutoKernel / Loop 2):** correctness first (a wrong variant is
reverted instantly, no latency recorded); then keep only on a strict ≥1% latency reduction; a
sub-1% change is reverted (logged as `tried`). One focused knob change per variant.

---

## 2. The edit surface, `ScheduleConfig`

The complete, machine-readable schema is `schemas/schedule_config.schema.json`. The knobs:

| knob | type | choices / range | meaning |
|---|---|---|---|
| `tiling.gemv.N_tile` | int | 64, 128, 256, 512 | GEMV output-column tile width (the one tiling knob the frozen lowerer consumes today). Wider = fewer tiles / less sync; narrower = more parallelism. |
| `tiling.attention.kv_block` | int | 64, 128, 256 | KV window block size. Reserved/searchable; recorded but not yet lowered (whole-window attention today). |
| `fusion_grouping` | list[list[str]] | `[]`, `[["gate","up"]]`, `[["gate","up","silu"]]`, `[["rmsnorm","gemv"]]` | op-name groups to co-resident. **Reserved**: recorded + searchable, but **not yet consumed by the frozen lowerer** (no effect on the emitted program today). |
| `sm_assignment` | str \| dict | `"round_robin"`, `"load_balance"`, or `{task_id: sm}` | SM placement policy or explicit map. **Reserved**: recorded + searchable, but **not yet consumed by the frozen lowerer/loader** (no effect on the emitted program today). |
| `pipelining_depth` | int | 0–4 typical (0–8 allowed) | instructions ahead to prefetch weights (hides the inter-op HBM bubble, the biggest megakernel win). 0 = none. |
| `page_allocation` | str | `"graph_color"`, `"linear"`, `"none"` | activation page reuse policy. **Reserved**: recorded + searchable, but **not yet consumed by the frozen lowerer** (no effect on the emitted program today). |
| `threads_per_block` | int | 128, 256, 512 (mult. of 32, ≤1024) | persistent VM kernel block size. Loader proves occupancy. |
| `smem_bytes_per_block` | int | 0, 16384, 49152 (≤ target opt-in cap) | dynamic SMEM opt-in per block. Loader rejects an over-cap value before launch. |

`harness.propose(model, gpu)["search_space"]` returns the same knobs+choices at runtime (with the
target's actual SMEM cap filled in), so an agent can read the surface programmatically.

---

## 3. The eval command + the JSON verdict schema

`evaluate(model_id, gpu, config_dict, device='auto')` (CLI: `eval <model> --gpu <arch> --config cfg.json`)
returns a structured JSON verdict. The CLI prints **only** the JSON on stdout (build/JIT chatter is
redirected to stderr), so it is safe to pipe into `jq` or `json.loads`.

```jsonc
{
  "valid": true,            // did it lower + validate + pass launch-config feasibility?
  "rejected_reason": null,  // string when valid=false; the exact reason (deadlock/race/arity/launch/...)
  "correct": true,          // ReferenceVM vs eager PyTorch, AUTHORITATIVE
  "max_abs_err": 0.0,       // max |amk - eager| over the logits (fp32)
  "top1_agreement": 1.0,    // fraction of positions whose argmax matches
  "latency_us": 1228.03,    // per-token latency; NEVER present without correct=true
  "latency_kind": "measured-gpu", // "measured-gpu" (real CUDA event timing) | "predicted" (cost model)
  "pct_of_roofline": 393960.77,   // 100 * latency / HBM-bandwidth floor (>100 = above the floor)
  "bound_us": 0.3117,       // the HBM-bandwidth floor: weight_bytes / target bandwidth
  "schedule_id": "sch_2f8b213192", // stable hash of the ScheduleConfig (the flywheel key)
  "tasks": 22, "weight_mb": 0.2793,
  "gpu": "rtx5090", "model": "...", "device": "cuda",
  "n_buffers": 37, "n_counters": 22,
  "notes": []               // human-readable context (e.g. why latency fell back to predicted)
}
```

### Honesty rules (enforced in code, not comments)

- **No latency without a correctness PASS.** `latency_us` and `latency_kind` are `null` unless
  `correct` is `true`. (`eval.bench.bench` physically refuses to time a wrong kernel.)
- **No fake measurements.** `latency_kind` is `"measured-gpu"` only for a real CUDA event-timed run
  whose GPU output also matched eager. Otherwise it is `"predicted"` (the analytic cost model),
  clearly labelled. On `device='cpu'` it is always `"predicted"` (a CPU reference time is not a GPU
  perf number).
- **No latency for a rejected config.** `valid=false` ⇒ `latency_us`/`latency_kind` are `null` and
  there is no correctness claim.
- **`evaluate` never crashes.** Malformed knobs (`pipelining_depth:"deep"`, negative `N_tile`,
  `tiling` not an object), over-cap SMEM, bad block sizes, lowering failures, all come back as a
  clean `valid=false` verdict with a `rejected_reason`.

> Note on `pct_of_roofline`: for the tiny `toy` model the weights are sub-megabyte, so the HBM
> floor (`bound_us`) is a fraction of a microsecond while a real launch is dominated by per-launch
> overhead, hence the huge percentage. This is honest: it is exactly "how far above the bandwidth
> floor this kernel runs," and it shrinks toward 100% as the model (and per-token weight traffic)
> grows.

---

## 4. keep/revert + move-on rules (`loop`)

`loop(model_id, gpu, budget, device='auto')` (CLI: `loop <model> --gpu <arch> --budget N`) runs the
autoresearch loop and logs **every** trial to `results.tsv` via `flywheel.log` (header + one row per
trial; the honesty rule forbids a row without a `correctness` verdict). It returns the best
VALID+CORRECT verdict and all rows.

The proposal schedule mirrors `schedule.search`:

1. trial 0 = the neutral `default_config` (the bar to beat),
2. a slice (`explore_fraction`, default 35%) of fresh **random** configs (explore),
3. the rest = **mutations** of the running best (exploit, one knob at a time).

The **keep/revert** decision (mirrors AutoKernel):

1. **Correctness first.** A candidate must be `valid` AND `correct` to be eligible. The incumbent is
   always a correct config; a correct candidate beats "no incumbent yet."
2. **Then latency.** A candidate dethrones the incumbent only on a **strict ≥1% latency gain**.
3. **Move-on.** We always evolve around whichever config is currently best; when a region stops
   improving, mutation naturally explores elsewhere (a sub-1% trial is logged as `tried`/`revert`
   and discarded, not kept).
4. **Tie-break = simplicity.** Within ±1% latency, a **measured** number outranks a **predicted**
   one (trust hardware over the model), and at equal kind the **simpler** config wins (fewer fusion
   groups, lower pipelining depth, no extra SMEM, default page/sm policy).

Kept correct points additionally enter the flywheel corpus (`flywheel/corpus.jsonl`), the learned
prior every future run starts from.

`results.tsv` row tags: `kept` (new incumbent), `tried` (valid+correct but not a ≥1% win),
`revert` (valid but incorrect), `rejected` (invalid/infeasible config).

---

## 5. The safety model (summary)

- **You edit `ScheduleConfig` only.** Never kernel code, never `Task.sm`, never `vm/*`.
- **The VM is frozen and deterministic.** The same config always lowers to the same program.
- **validate-before-launch.** Deadlock- and race-freedom are *proven* statically; launch-config
  feasibility is checked against the target. An unsafe config is a clean `REJECTED`, full stop.
- **Correctness is authoritative and free of the GPU.** The CPU `ReferenceVM` reproduces the exact
  counter-driven scheduling semantics and is compared to eager PyTorch every time. A GPU mismatch
  never produces a latency.
- **Local device only.** The harness uses the machine's local GPU (or CPU). It never touches Modal
  or any cloud GPU.

---

## 6. A copy-pasteable agent loop (Loop 2, schedule)

(For the Loop-1 copy-paste examples, see §1A above.)

### Programmatic (Python)

```python
import harness

MODEL, GPU = "toy", "rtx5090"

# 1. Read the edit surface (the "program.md" step).
p = harness.propose(MODEL, GPU)
cfg = p["schedule_config"]            # the incumbent ScheduleConfig dict
print("knobs:", list(p["search_space"]))

# 2. Evaluate the incumbent (authoritative correctness + honest latency).
base = harness.evaluate(MODEL, GPU, cfg)
assert base["valid"] and base["correct"]
best = base

# 3. Propose a few edits, keep/revert on correctness-then-latency.
for N_tile in (64, 128, 512):
    trial_cfg = {**cfg, "tiling": {**cfg["tiling"], "gemv": {"N_tile": N_tile}}}
    v = harness.evaluate(MODEL, GPU, trial_cfg)
    if v["valid"] and v["correct"] and v["latency_us"] is not None \
            and v["latency_us"] < best["latency_us"] * 0.99:   # >=1% gain
        best, cfg = v, trial_cfg
        print(f"kept N_tile={N_tile}: {v['latency_us']:.1f}us")

print("best:", best["schedule_id"], best["latency_us"], "us")

# 4. Or let the built-in keep/revert loop do all of the above:
out = harness.loop(MODEL, GPU, budget=16)
print(out["best_verdict"]["schedule_id"], out["best_verdict"]["latency_us"], "us")
```

### CLI (shell, for an agent that shells out)

```bash
# read the surface
uv run python amk_cli.py propose toy --gpu rtx5090 > surface.json

# write a candidate config (edit ONE knob), then evaluate it
cat > cfg.json <<'JSON'
{ "tiling": {"gemv": {"N_tile": 128}, "attention": {"kv_block": 128}},
  "fusion_grouping": [], "sm_assignment": "load_balance",
  "pipelining_depth": 3, "page_allocation": "graph_color",
  "threads_per_block": 256, "smem_bytes_per_block": 0 }
JSON
uv run python amk_cli.py eval toy --gpu rtx5090 --config cfg.json   # prints ONLY JSON on stdout
# exit code: 0 = valid+correct, 1 = rejected or incorrect (gate your loop on it)

# run the full keep/revert autoresearch loop
uv run python amk_cli.py loop toy --gpu rtx5090 --budget 16
```

That is the entire integration surface. Edit the config, read the verdict, keep on correctness then
a ≥1% latency win. The frozen VM and the validator guarantee you can never produce an unsafe or
dishonest result.

---

## Unattended autoresearch: `amk autoresearch` (run it and sleep)

`propose`/`eval`/`loop` are the hand-driven verbs. `autoresearch` is the **headless driver**: point
it at a `(model, gpu)`, give it a budget, and it runs the whole keep/revert methodology unattended -
for hours, resumable, crash-proof, and the flywheel makes every future run start smarter.

```bash
# 20 cost-model iterations (fast + deterministic)
uv run python amk_cli.py autoresearch toy --gpu rtx5090 --iters 20 --device cpu

# sleep on it: a 2-hour wall-clock budget, measuring on the real GPU (correctness-gated)
uv run python amk_cli.py autoresearch toy --gpu rtx5090 --minutes 120 --device cuda

# pure exploration (ignore the flywheel prior; grow the corpus from scratch)
uv run python amk_cli.py autoresearch toy --gpu rtx5090 --iters 50 --cold --seed 1

# RUN-IT-AND-SLEEP: ~8 hours on the real GPU, never stops on a plateau (basin-hops to fresh
# regions while always preserving the global best), writes a morning wake-up report.
uv run python amk_cli.py autoresearch small --gpu rtx5090 --minutes 480 --device cuda --overnight
# wake up and read:  workspace/amk_overnight_report.md   (best config + speedup-vs-baseline)
```

### `--overnight` (the "run it at night, wake up to better megakernels" mode)

`--overnight` (use with a long `--minutes`, e.g. `--minutes 480` for ~8h) makes the loop run for
hours and *keep improving* instead of quitting at the first plateau:

- **No plateau-stop.** Only the `--minutes`/`--iters` budget ends the run.
- **Basin-hopping.** After `--restart-after N` (default 6) consecutive non-improvements, the search
  jumps to a fresh random `(schedule, kernel_knobs)` region and resets the plateau counter, so it
  escapes stuck basins. The **global best is always preserved** (every candidate is kept/reverted
  against the resident global-best VM via interleaved, drift-robust measurement), so a restart can
  only ever *find* something better, never lose the best.
- **Bounded memory.** Reverted candidate VMs are freed and the CUDA cache is trimmed periodically.
- **Resumable + crash-proof.** Checkpointed every iteration (re-running the same command continues
  the campaign); a CUDA error / timeout in one iteration is logged and the run continues.
- **Wake-up report.** `workspace/amk_overnight_report.{json,md}` is refreshed periodically and at
  the end: best config, speedup-vs-baseline, the improvement milestones, restart count.
- **Flywheel compounding.** Kept points enter `flywheel/corpus.jsonl`; the *next* night's run
  warm-starts from them and goes further.

**Honest scope:** the morning "best" is a correct, measured, drift-robust win **over AMK's own
default schedule** (typically ~1.2–1.7× on the `small` model on this laptop GPU). It is **not** a
claim of beating cuBLAS/vLLM, AMK is honestly behind those at batch-1 (it carries the megakernel's
DAG-dispatch + cross-SM counter-sync that a single library GEMV does not). The loop never reports a
latency below the HBM roofline floor (physically impossible numbers are withheld as artifacts).

Flags: `--iters N` and/or `--minutes M` (whichever is hit first), `--device auto|cuda|cpu`
(`cpu` ⇒ analytic `predicted` fitness; `cuda`/`auto` ⇒ `measured-gpu`, correctness-gated),
`--cold` (prior off), `--seed S`, `--epsilon E` (explore rate), `--corpus`/`--results`/`--state`
to redirect the artifacts. It prints a trajectory line per iteration and a final report, then a
compact JSON summary (best config / latency / %roofline / speedup vs baseline / iters-to-best /
warm-seed + ranker status). Exit code 0 iff a correct schedule was found.

What it guarantees, every iteration:

- **Propose → lower → `validate()` → correctness-gated eval → keep/revert** (the same honest
  pipeline as `harness.evaluate`/`loop`; a latency never exists without a correctness PASS, and
  `latency_kind` is `measured-gpu` or `predicted`, never mislabeled).
- **Keep iff correct AND ≥ 1% faster** than the incumbent; kept points enter `flywheel/corpus.jsonl`.
- **Robust:** a crash / CUDA-error / timeout in one iteration is logged (`CRASH`/`TIMEOUT` in
  `results.tsv`) and the loop **continues**, one failure never stops an overnight run.
- **Resumable:** the campaign state (`workspace/amk_orchestration_state.json`) is checkpointed each
  iteration; re-running the same command **continues** the same campaign.
- **Flywheel-warm by default:** unless `--cold`, the first incumbent is the corpus's best
  `warm_start` seed for the nearest `(model_shape, gpu)` shape, and exploitation proposals are
  ranked by the learned prior (`flywheel.prior.rank`). Advisory only, a bad suggestion still has
  to clear the gate.

### The orchestrator (for a coding agent driving `next`/`record`)

The campaign brain is `amk_orchestrate.py`, the AutoKernel `status`/`next`/`record`/`report` state
machine, in AMK's latency / pct-of-roofline units:

```bash
uv run python amk_orchestrate.py status   # baseline, best, speedup, region split, plateau counter
uv run python amk_orchestrate.py next      # continue-or-stop decision (MOVE_ON_CRITERIA) + hot region
uv run python amk_orchestrate.py report    # the aggregate campaign report (markdown + terminal)
# an agent records its own experiment outcomes:
uv run python amk_orchestrate.py record kept --latency-us 24.31 --pct-roofline 7800 \
    --kind predicted --config cfg.json --description "fused rmsnorm+gemv"
```

`MOVE_ON_CRITERIA` mirror AutoKernel: `consecutive_reverts ≥ 8` (plateau), `pct_roofline ≤ 110%`
(near the weights/bandwidth floor), `max_minutes` (wall-clock budget), `speedup ≥ 3×` vs baseline.
`record kept` resets the plateau counter and may set a new best; `revert`/`failed`/`crash`/`timeout`
increment it; when a criterion trips, the campaign flips to `done` and `next` says STOP. Both the
headless driver and an agent talking to the orchestrator write the *same* `results.tsv` + corpus, so
the flywheel learns from every run regardless of who drove it.

The cross-run "it gets smarter" proof is `paper/exp_flywheel_learning.py`
(`paper/EXPERIMENTS_flywheel.md`): a warm run starts from the corpus's best schedule (not the
default) and reaches the cold best in fewer iterations, honest about the (model-dependent) gain.
(That experiment script ships with the **paper artifact**, not this OSS tree, it is gitignored.
On a clean clone you exercise the same cold-vs-warm flywheel directly via the shipped autoresearch
driver: `uv run python amk_cli.py autoresearch toy --gpu rtx5090 --iters 50 --cold` ignores the
prior corpus, while the default warm run seeds from `flywheel/corpus.jsonl`.)

> **Experimental decode path.** The default multi-token decoder is host-driven, one launch per token
> (`generate.py`). `vm/loader_persist.py::PersistentDecodeVM` is a separate **experimental/research**
> path that runs a whole K-token decode loop inside a single cooperative launch; it is **not** the
> default decode and is exercised only via `tests/test_persist_decode.py`.
