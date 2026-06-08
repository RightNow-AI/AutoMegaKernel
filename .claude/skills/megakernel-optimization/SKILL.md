---
name: megakernel-optimization
description: Use when optimizing or generating a CUDA megakernel for a HuggingFace Llama-family model with AutoMegaKernel (AMK), drives the correctness-gated propose -> eval -> keep/revert loop (or hands off to the unattended autoresearch driver).
---

# AutoMegaKernel (AMK), megakernel schedule optimization

AMK compiles a HuggingFace Llama-family model into ONE persistent CUDA megakernel and tunes it
with an AutoKernel-style loop: **read the edit surface -> propose ONE knob change -> eval ->
keep/revert -> record -> repeat**. This skill drives Loop 2 (schedule + kernel_knobs search).
You never write kernel code; you only edit a structured `ScheduleConfig` (plus its reserved
`kernel_knobs` sub-object). The frozen VM lowers your config deterministically and the CPU
ReferenceVM judges correctness vs eager PyTorch.

## HARD HONESTY RULES (state and obey these every time)

- **Correctness FIRST.** A latency is NEVER reported without a correctness PASS vs the CPU
  ReferenceVM. Keep a candidate only if it is correct AND >= 1% faster than the incumbent.
- **validate-before-launch.** An unsafe `ScheduleConfig` is a clean REJECTED (a deadlock/race-free
  proof rejects it before launch), never a hung GPU.
- **The edit surface is `ScheduleConfig` + `kernel_knobs` ONLY**, never raw kernel code, never
  `vm/`, never the frozen ABI.
- **Measured-gpu latency is drift-robust;** physically-impossible sub-roofline latencies are
  withheld as artifacts.
- **All speedups are vs AMK's OWN baseline (default schedule), NOT a claim of beating
  cuBLAS/vLLM.** AMK is currently within ~13% of cuBLAS at batch-1, behind it.

## The edit surface (read it before proposing)

Read the surface programmatically, never guess knob names. Prefer the canonical MCP tool; fall
back to the CLI if MCP is unavailable.

- MCP: `amk_propose(model, gpu="rtx5090")` -> `{ schedule_config, schedule_id, search_space, ... }`.
  `search_space` includes the `kernel_knobs.*` sub-surface.
- CLI: `amk propose <model> --gpu <arch>` (or `uv run python amk_cli.py propose <model> --gpu <arch>`)
  prints the same surface as JSON on stdout.

The `ScheduleConfig` knobs (edit ONE per trial): `tiling.gemv.N_tile`,
`tiling.attention.kv_block`, `fusion_grouping`, `sm_assignment`, `pipelining_depth`,
`page_allocation`, `threads_per_block`, `smem_bytes_per_block`. The reserved `kernel_knobs`
object holds GEMV build knobs: `cols_per_warp`, `cpasync`, `cpa_stages`, `cpa_cols` (these move
MEASURED latency under `device=cuda`; the `predicted`/CPU path does not model them). A config
WITHOUT `kernel_knobs` is byte-identical to the production incumbent.

`<model>` is `toy` / `toy-2L` (fully supported) or a HuggingFace id (best-effort). `<arch>` is a
registered GpuTarget: `rtx5090`, `b200`, `h100`, `a100`.

## The eval verdict

- MCP: `amk_eval(model, gpu, config, device="auto")` where `config` is a JSON `ScheduleConfig`
  object (optionally carrying a `kernel_knobs` object).
- CLI: write `cfg.json`, then `amk eval <model> --gpu <arch> --config cfg.json` (JSON-only on
  stdout; exit code 0 = valid+correct, 1 = rejected or incorrect).

The verdict carries `valid`, `rejected_reason`, `correct`, `latency_us`, `latency_kind`
(`measured-gpu` | `predicted`), `pct_of_roofline`, `bound_us`, `schedule_id`. `latency_us` and
`latency_kind` are `null` unless `correct` is `true` and the config was `valid`. `eval` never
crashes, malformed knobs come back as a clean `valid=false` with a `rejected_reason`.

## The loop (drive this exactly)

1. **Read the surface** with `amk_propose` (or `amk propose`). Note the incumbent
   `schedule_config` and the editable `search_space`.
2. **Baseline.** `amk_eval` the incumbent. Require `valid` AND `correct`. This is the bar to beat;
   its `latency_us` is the incumbent latency.
3. **Propose ONE knob change** (one knob per trial, schedule knob OR one `kernel_knobs` field),
   building the candidate config from the incumbent.
4. **Eval the candidate** with `amk_eval`.
5. **Keep/revert.** Keep ONLY if `valid` AND `correct` AND
   `latency_us < incumbent_latency_us * 0.99` (a strict >= 1% win). Otherwise revert (keep the
   old incumbent). Tie-break: a `measured-gpu` number outranks a `predicted` one, then simpler
   config wins.
6. **Record** the outcome to the orchestrator: `amk_orchestrate_record(status, latency_us=...,
   pct_roofline=..., kind=..., config=..., description=...)` with `status` one of
   `kept`/`revert`/`failed`/`crash`/`timeout`/`rejected`. (CLI: `python amk_orchestrate.py record kept --latency-us
   ... --pct-roofline ... --kind ... --config cfg.json --description "..."`.)
7. **Repeat** from step 3 around the current best. Ask `amk_orchestrate_next()` (CLI:
   `python amk_orchestrate.py next`) whether to continue or STOP (plateau / near-roofline /
   budget / >=3x speedup), and `amk_orchestrate_status()` for baseline/best/speedup/plateau.

To run the whole keep/revert loop in one call, use `amk_loop(model, gpu, budget=8)` (CLI:
`amk loop <model> --gpu <arch> --budget N`). To run unattended for hours, hand off to
`amk_autoresearch(model, gpu, minutes=..., overnight=...)` (CLI: `amk autoresearch ...`); see the
`/amk-autoresearch` command.

## Worked example (toy on rtx5090)

```
1. amk_propose("toy", "rtx5090")
   -> incumbent schedule_config (pipelining_depth=0, N_tile default, no kernel_knobs),
      search_space lists N_tile in {64,128,256,512}, pipelining_depth 0-4, kernel_knobs.cpasync {0,1}, ...

2. amk_eval("toy", "rtx5090", <incumbent cfg>, device="cuda")
   -> { valid:true, correct:true, latency_us: 1228.0, latency_kind:"measured-gpu", ... }
      incumbent_latency = 1228.0 us

3. ONE knob change: set pipelining_depth = 3 (hides the inter-op HBM bubble).
   cfg = { ...incumbent, "pipelining_depth": 3 }

4. amk_eval("toy", "rtx5090", cfg, device="cuda")
   -> { valid:true, correct:true, latency_us: 1010.0, latency_kind:"measured-gpu", ... }

5. 1010.0 < 1228.0 * 0.99  -> KEEP. New incumbent latency = 1010.0 us.
   amk_orchestrate_record("kept", latency_us=1010.0, pct_roofline=..., kind="measured-gpu",
                          config=cfg, description="pipelining_depth 0->3")

6. Next ONE knob change off the new best, e.g. kernel_knobs.cpasync = 1 / N_tile = 128. Eval.
   If a candidate is correct but only 0.4% faster -> record "revert" (correct but not kept).
   If a candidate is invalid (e.g. over-cap smem) -> record "rejected" and revert.

7. amk_orchestrate_next() until it says STOP. Speedup is vs AMK's own default schedule, NOT a
   cuBLAS/vLLM claim.
```

## Canonical tool / CLI names (use these EXACT names)

- MCP: `amk_doctor`, `amk_propose`, `amk_eval`, `amk_loop`, `amk_autoresearch`,
  `amk_orchestrate_status`, `amk_orchestrate_next`, `amk_orchestrate_report`,
  `amk_orchestrate_record`.
- CLI: `amk propose|eval|loop|autoresearch|compile|generate|doctor` and
  `python amk_orchestrate.py status|next|record|report`.

Full contract: read `HARNESS.md` (terminology in `README.md`).
