---
description: Drive an interactive AMK propose -> eval -> keep/revert megakernel schedule session on $ARGUMENTS (model [gpu]).
---

Drive an interactive AutoMegaKernel (AMK) schedule-optimization session on `$ARGUMENTS`
(parse as `model [gpu]`; default `gpu` = `rtx5090`; default `model` = `toy`). Use the
`megakernel-optimization` skill's loop and obey its HARD HONESTY RULES.

HARD HONESTY RULES (state them, obey them):
- Correctness FIRST: NEVER report a latency without a correctness PASS vs the CPU ReferenceVM.
  Keep a candidate only if it is correct AND >= 1% faster than the incumbent.
- validate-before-launch: an unsafe ScheduleConfig is a clean REJECTED, never a hung GPU.
- Edit surface is ScheduleConfig + kernel_knobs ONLY, never kernel code, never vm/ or the ABI.
- Measured-gpu latency is drift-robust; impossible sub-roofline latencies are withheld.
- Speedups are vs AMK's OWN default schedule, NOT a claim of beating cuBLAS/vLLM.

Do this:
1. Read the edit surface with `amk_propose(model, gpu)` (CLI fallback:
   `amk propose <model> --gpu <gpu>`). Report the incumbent `schedule_config` and the editable
   `search_space` (including the `kernel_knobs.*` sub-surface).
2. Establish the baseline with `amk_eval(model, gpu, <incumbent cfg>, device="auto")` (CLI:
   write `cfg.json`, `amk eval <model> --gpu <gpu> --config cfg.json`). Require `valid` AND
   `correct`. Record its `latency_us` as the incumbent latency.
3. Loop: propose ONE knob change off the current best (one ScheduleConfig knob OR one
   `kernel_knobs` field per trial), `amk_eval` it, and keep/revert, keep ONLY if `valid` AND
   `correct` AND `latency_us < incumbent_latency_us * 0.99`. After each trial call
   `amk_orchestrate_record(status, latency_us=..., pct_roofline=..., kind=..., config=...,
   description=...)` with `status` in `kept`/`revert`/`failed`/`crash`/`timeout`/`rejected`
   (`revert` is the canonical token for any correct-but-not-kept candidate).
4. Between trials, consult `amk_orchestrate_next()` (continue-or-STOP) and
   `amk_orchestrate_status()` (baseline/best/speedup/plateau). Stop when `next` says STOP or the
   user asks. To run the full keep/revert loop in one shot instead, use `amk_loop(model, gpu,
   budget=N)` (CLI: `amk loop <model> --gpu <gpu> --budget N`).

Finish with a short report: best `schedule_id`, best `latency_us` + `latency_kind`,
`pct_of_roofline`, and the speedup vs the incumbent baseline (clearly stated as vs AMK's own
default, not vs cuBLAS/vLLM).

Use only the canonical names: MCP `amk_propose`/`amk_eval`/`amk_loop`/`amk_orchestrate_record`/
`amk_orchestrate_next`/`amk_orchestrate_status`; CLI `amk propose|eval|loop` and
`python amk_orchestrate.py record|next|status`.
