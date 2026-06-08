# Changelog

All notable changes to AutoMegaKernel. Dates are when the work landed.

## [0.2.0], 2026-06-05

### Added, foundation (M0)
- **Standard megakernel IR** (`schedule/ir.py`): task-DAG, counter sync model, `ScheduleConfig`
  edit surface, `GpuTarget` data records, JSON format, and `validate()`, a real proof of both
  **deadlock-freedom** and **race-freedom** (shared-counter all-join rule, transitive
  happens-before provenance, per-SM queue order, on-chip locality, page-alias safety).
- **Frozen instruction ABI** (`vm/abi.h`) with a drift guard (`tests/test_abi_sync.py`).
- **CPU reference VM** (`vm/reference_vm.py`), bit-exact scheduling semantics + real numerics,
  the GPU-free correctness oracle; `vm/verify_vm.py` proves it.
- **CUDA megakernel VM** (`vm/`): persistent **cooperative** kernel (one block per SM), counter
  sync, page scratchpad. Runs the **full Llama-style decode as one kernel launch**, matching
  eager + the reference to ~1e-7.
- **Instruction library** (`instructions/`): 7 ABI-conformant CUDA micro-kernels (+ a Triton
  path), per-op verify, and an AutoKernel-style generation loop.
- **Scheduler** (`schedule/`): HF/toy graph import, decode lowering, roofline cost model,
  Loop-2 schedule search.
- **Eval** (`eval/`): correctness oracle (logit equivalence + token divergence), correctness-
  **gated** latency bench, roofline reporting, honest baseline stubs.
- **Product**: `compile.py` (`amk compile`) + `amk_cli.py` + the flywheel log.
- **Docs**: `program.md` (autonomous brain), `docs/IR_SPEC.md` (the standard IR spec).

### Added, generality, retargeting, harness
- **HF importer** (`schedule/graph.from_hf`): imports a real `transformers.LlamaForCausalLM`
  (MHA/GQA/tied-embeddings) bit-correct vs HF's own forward; rejects unsupported variants loudly.
- **bf16** path in the VM (matches the bf16 reference bit-identically).
- **Self-retargeting**: the nvcc gencode is derived from the live device, the **same code** built
  and ran correct megakernels on **sm_120 (RTX 5090), sm_80 (A100), sm_90 (H100)**
  (measured via Modal; see `DATACENTER_RESULTS.md`).
- **Coding-agent harness** (`harness.py`, `amk propose/eval/loop`, `HARNESS.md`,
  `schemas/schedule_config.schema.json`): AMK is drivable by any coding agent.
- **Flywheel corpus** (`flywheel/corpus.jsonl`) seeded with real cross-arch measured points.

### Quality
- Two adversarial review passes (a 4-lens panel + Codex) hardened the validator; every finding
  is pinned by a regression in `tests/test_validator_races.py`.
- Honesty enforced in code: no latency without a correctness PASS; roofline distance always
  reported; no unmeasured datacenter numbers.

### Known / in progress
- Latency is currently far from the HBM roofline (correct-but-unoptimized). The lever, a
  bandwidth-efficient tensor-core/vectorized GEMV + coarser sync, is the active production push.
