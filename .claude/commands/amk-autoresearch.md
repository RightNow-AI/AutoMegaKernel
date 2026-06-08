---
description: Launch the unattended / overnight AMK autoresearch driver on $ARGUMENTS (model [gpu] [minutes|iters]).
---

Launch the unattended AutoMegaKernel (AMK) autoresearch driver on `$ARGUMENTS` (parse as
`model [gpu] [minutes|iters]`; default `gpu` = `rtx5090`; default `model` = `toy`). This is the
headless keep/revert campaign, point it at a `(model, gpu)`, give it a budget, and it runs the
whole correctness-gated methodology unattended, resumable and crash-proof, growing the flywheel.

HARD HONESTY RULES (state them, obey them):
- Correctness FIRST: NEVER a latency without a correctness PASS vs the CPU ReferenceVM. Keep iff
  correct AND >= 1% faster than the incumbent.
- validate-before-launch: an unsafe ScheduleConfig is a clean REJECTED, never a hung GPU.
- Edit surface is ScheduleConfig + kernel_knobs ONLY, never kernel code, never vm/ or the ABI.
- Measured-gpu latency is drift-robust; impossible sub-roofline latencies are withheld.
- The morning "best" is a speedup vs AMK's OWN default schedule, NOT a claim of beating
  cuBLAS/vLLM (AMK is within ~13% of cuBLAS at batch-1, behind it).

Do this:
1. Confirm the environment first with `amk_doctor()` (CLI: `amk doctor`), torch/cuda
   availability, device name, registered targets. For a real `measured-gpu` campaign you need a
   CUDA GPU; otherwise the fitness is analytic `predicted` (use `device="cpu"`).
2. Launch the driver with `amk_autoresearch(model, gpu, minutes=<M>, iters=<N>, device="auto",
   overnight=<bool>, cold=<bool>)`. CLI fallback:
   `amk autoresearch <model> --gpu <gpu> --minutes <M> --device cuda [--overnight] [--cold]`
   (or `--iters <N> --device cpu` for a fast deterministic run). Use `overnight=true` with a long
   `minutes` (e.g. 480 for ~8h): no plateau-stop, basin-hops to fresh regions while always
   preserving the global best, bounded memory, checkpointed every iteration (re-run the same
   command to continue), and writes a wake-up report.
3. It is resumable + crash-proof: a CUDA error / timeout in one iteration is logged and the run
   continues. Re-running the same command continues the same campaign.
4. When it finishes (or to inspect progress), report from the orchestrator:
   `amk_orchestrate_status()`, `amk_orchestrate_report()` (CLI: `python amk_orchestrate.py status`
   / `report`), and for an overnight run read `workspace/amk_overnight_report.md` (best config +
   speedup-vs-baseline + milestones + restart count).

Report: best `schedule_id`, best `latency_us` + `latency_kind`, `pct_of_roofline`, and the
speedup vs AMK's own default schedule (NOT vs cuBLAS/vLLM).

Use only the canonical names: MCP `amk_doctor`/`amk_autoresearch`/`amk_orchestrate_status`/
`amk_orchestrate_report`; CLI `amk doctor|autoresearch` and `python amk_orchestrate.py
status|report`.
