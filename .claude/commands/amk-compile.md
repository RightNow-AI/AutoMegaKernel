---
description: One-shot compile + verify a model into a CUDA megakernel via `amk compile` on $ARGUMENTS (model [gpu]).
---

One-shot compile and verify a model into a single persistent CUDA megakernel with AutoMegaKernel
(AMK) on `$ARGUMENTS` (parse as `model [gpu]`; default `gpu` = `rtx5090`; default `model` =
`toy`). This is THE product path: import -> lower -> validate (deadlock + race free) -> verify vs
eager -> build the GPU megakernel -> measure against the HBM roofline -> emit a correct megakernel
+ report.

HARD HONESTY RULES (state them, obey them):
- Correctness FIRST: the compile path verifies vs eager PyTorch AND the CPU ReferenceVM; a
  latency is NEVER reported without a correctness PASS.
- validate-before-launch: an unsafe schedule is a clean REJECTED (deadlock/race-free proof),
  never a hung GPU.
- The edit surface (if you tune afterward) is ScheduleConfig + kernel_knobs ONLY, never kernel
  code, never vm/ or the frozen ABI.
- Measured-gpu latency is drift-robust; impossible sub-roofline latencies are withheld.
- Any speedup is vs AMK's OWN baseline, NOT a claim of beating cuBLAS/vLLM.

Do this:
1. Confirm the environment with `amk_doctor()` (CLI: `amk doctor`), torch/cuda, device name,
   nvcc, registered targets. A CUDA GPU + nvcc is required to build and measure the megakernel.
2. Compile and verify in one shot via the CLI:
   `amk compile <model> --gpu <gpu> --regime single-stream`
   (equivalently `uv run python amk_cli.py compile <model> --gpu <gpu> --regime single-stream`).
   This lowers, statically validates, verifies correctness vs eager, builds the GPU megakernel,
   and measures it against the `weights / HBM_bandwidth` roofline, emitting the megakernel +
   report.
3. Report the result: correctness PASS (vs eager + ReferenceVM), measured per-token `latency_us`
   + `latency_kind`, `pct_of_roofline`, and the `schedule_id`. If correctness did not PASS, report
   the failure honestly and do NOT report a latency.

Optional next step: to tune the compiled schedule, run `/amk-optimize <model> <gpu>` (interactive
propose -> eval -> keep/revert) or `/amk-autoresearch <model> <gpu>` (unattended). Speedups there
are vs AMK's own default schedule, not vs cuBLAS/vLLM.

Use only the canonical names: CLI `amk compile` and `amk doctor` (MCP `amk_doctor` for the
environment check).
