# Goal: compile MODEL into a correct megakernel and improve its decode latency

> Used via Claude Code: `/goal .claude/goals/optimize-megakernel.md`
> (set `MODEL` and `GPU` below; defaults are `toy` / `rtx5090`).

## Objective

Compile **MODEL** into a correct AutoMegaKernel (AMK) megakernel for **GPU** and improve its decode
latency **over AMK's own baseline schedule**, correctness-gated the whole way. Then report the
measured, drift-robust speedup and the best `ScheduleConfig` that produced it.

- `MODEL` = `toy`  (or `toy-2L` / a HuggingFace Llama-family id)
- `GPU`   = `rtx5090`  (a registered GpuTarget: `rtx5090` | `b200` | `h100` | `a100`)

## What you own, the edit surface (and ONLY this)

You search **one structured object**: a `ScheduleConfig` (typed JSON knobs), optionally with a
`kernel_knobs` sub-object. You NEVER write kernel code, never touch `vm/`, `Task.sm`, or the frozen
ABI. The frozen VM lowers your config deterministically and **proves it deadlock/race-free before
launch**, an unsafe config is a clean `REJECTED`, never a hung GPU. Read the live knob surface from
`amk_propose(MODEL, GPU)["search_space"]` (schema: `schemas/schedule_config.schema.json`).

## Canonical tools / CLI to use (these EXACT names)

MCP tools (server: `amk_mcp.py`):
- `amk_doctor()`, environment + registered GpuTargets.
- `amk_propose(MODEL, GPU)`, incumbent `ScheduleConfig` + editable `search_space` (incl. `kernel_knobs`).
- `amk_eval(MODEL, GPU, config, device="auto")`, verdict `{valid, correct, latency_us,
  latency_kind, pct_of_roofline, bound_us, schedule_id, ...}`. `config` is a JSON object.
- `amk_loop(MODEL, GPU, budget=8, device="auto")`, keep/revert loop → `{best_verdict, best_config, rows}`.
- `amk_autoresearch(MODEL, GPU, minutes=None, iters=None, device="auto", overnight=False, cold=False)`
 , unattended campaign.
- `amk_orchestrate_status() / amk_orchestrate_next() / amk_orchestrate_report()`, campaign state machine.
- `amk_orchestrate_record(status, latency_us=None, pct_roofline=None, kind=None, config=None, description="")`.

Equivalent CLI (shell out if preferred): `amk propose|eval|loop|autoresearch|compile|generate|doctor`
and `python amk_orchestrate.py status|next|record|report`. `eval` prints JSON-only on stdout and
exits `0` iff valid+correct, gate the loop on it.

## The loop to run

1. `amk_doctor()`, confirm CUDA + GpuTargets (CPU works too: `device="cpu"` ⇒ analytic `predicted`).
2. `amk_propose(MODEL, GPU)`, read the incumbent `ScheduleConfig` and the `search_space`.
3. `amk_eval(MODEL, GPU, incumbent)`, establish the baseline verdict (must be valid + correct).
4. Propose edits **one knob at a time**, `amk_eval` each. **KEEP** a candidate only if it is
   `valid` AND `correct` AND its `latency_us` is **≥1% lower** than the incumbent; otherwise REVERT.
5. To automate steps 3–4: `amk_loop(MODEL, GPU, budget=16)`; to run unattended for hours:
   `amk_autoresearch(MODEL, GPU, minutes=480, overnight=true)` (basin-hops, preserves the global best).
6. Track the campaign with `amk_orchestrate_status` / `amk_orchestrate_next`; finish with
   `amk_orchestrate_report`.

## Success criteria

- A **correct** megakernel for MODEL on GPU: `valid=true` AND `correct=true` (ReferenceVM vs eager).
- A kept best `ScheduleConfig` that is correct AND measurably **≥1% faster** than AMK's own baseline.
- Reported result includes: the best `schedule_id`, the best `ScheduleConfig`, `latency_us` with its
  `latency_kind` (`measured-gpu` or `predicted`), `pct_of_roofline`, and the **speedup vs AMK's own
  baseline** (NOT vs cuBLAS/vLLM).

## Honesty rules (non-negotiable, enforced in code)

- **Correctness FIRST.** Never report a latency without a correctness PASS vs the CPU ReferenceVM.
  Keep only if correct AND ≥1% faster than the incumbent.
- **validate-before-launch.** An unsafe `ScheduleConfig` is a clean `REJECTED`, never a hung GPU.
- **Edit surface = `ScheduleConfig` + `kernel_knobs` ONLY**, never raw kernel code, `vm/`, or the ABI.
- **Measured-gpu latency is drift-robust;** physically-impossible sub-roofline numbers are withheld.
- **All speedups are vs AMK's OWN baseline**, NOT a claim of beating cuBLAS/vLLM (AMK is within
  ~13% of cuBLAS at batch-1, behind it, state this plainly).
