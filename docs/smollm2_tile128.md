# SmolLM2 GEMV tile128 schedule experiment

`examples/smollm2_gemv_tile128.json` selects the existing `N_tile=128` schedule
knob. The configuration is opt-in; the default lowerer, CUDA VM, ABI, oracle and
kernel knobs are unchanged.

On the measured SmolLM2 workload, default tiling produced 5,286 tasks per step
and tile128 produced 1,716, with 848 buffers and 38,912 bytes of dynamic shared
memory in both arms. A wider tile can reduce schedule/table setup work while
reducing GPU parallelism. The result below measures complete generation; it
is not a GPU-only GEMV claim.

## Using the configuration

The existing evaluator accepts the preset:

```bash
uv run amk eval HuggingFaceTB/SmolLM2-135M-Instruct --gpu rtx5090 \
  --config examples/smollm2_gemv_tile128.json
```

That command evaluates one forward using the registered target. It does not
reproduce the complete-generation experiment or substitute an RTX 5090 target
for a PRO 4000. The archived generation runner reads the live SM120 device
limits and passes the same `ScheduleConfig` to the unmodified lowerer at each
step. `generate` currently passes `config=None`, so reproducing the full request
requires that external adapter; this PR adds no new generation API.

## Recorded complete-generation experiment

Frozen upstream: `a514bbc20a03bbf698a17443f8f14a27a617fc10`.
Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, all 30 layers and 134,515,008
parameters, FP32 weights/activations, no quantization. Six checkpoint file hashes
are included in the evidence. Runtime: Python 3.12, Torch 2.13.0+cu130,
Transformers 5.16.1, CUDA 13.0, one RTX PRO 4000 Blackwell 24 GB (SM120).

Four independent processes ran default/tile128/tile128/default, with independent
first-build directories. Context lengths were 8 and 64 tokens; each request
produced 8 new tokens. All 344 step-level logits comparisons passed the existing
FP32 `atol=rtol=1e-4` oracle against both CPU ReferenceVM and full eager.
All 64 correctness-stage generated tokens matched exactly. Only after both
workloads passed did each process perform one warmup and three timed requests
per context; timed and warmup sequences also matched.

| Context | Process pair | Default median seconds | Tile128 median seconds | Latency reduction |
|---|---|---:|---:|---:|
| 8 | A0/P1 | 3.486624 | 1.778946 | 48.98% |
| 8 | A3/P2 | 3.515794 | 1.755405 | 50.07% |
| 64 | A0/P1 | 16.170751 | 7.911105 | 51.08% |
| 64 | A3/P2 | 15.953324 | 7.895794 | 50.51% |

Allocator peak: 1,320,648,704 bytes in each process. The two process-pair
geometric means are 49.53% and 50.79% latency reduction for contexts 8 and 64.
The baseline last/first median changes are +0.84% and -1.34% respectively.
These are exploratory observations from two process pairs, not a confidence
interval or independently confirmed stable speedup.

Timing includes per-step lowering/validation, VM and table construction, weight
H2D, KV transfer and sampling from a CPU-resident model. It excludes disk model
loading, first CUDA compilation and correctness instrumentation. The existing
path reconstructs the VM each step; this is not GPU-resident steady-state
decode, concurrent service throughput, CUDA Graph replay, or a comparison with
cuBLAS/vLLM.

The preset passed the experiment's advancement rule: every context/process
pair improved by at least 1%. Further independent validation is required before
changing a default. A separate `cpa_stages=2` candidate regressed at context64
and is not included.

[All timings, per-step comparisons, checkpoint hashes and the original runner](https://gist.github.com/0z5a/7b6a65fb1899b27411706bb402201016)
are published together. The archive documents the exact directory layout and
commands for replaying all four processes. The compact checked-in
results (record retained locally) retain every recorded request time and
original receipt hashes. No GPU test was rerun for publication.
