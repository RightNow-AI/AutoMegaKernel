# AMK Baselines, apples-to-apples per-token decode (M2 thesis evidence)

This document records a **measured, correctness-gated** comparison of single-stream decode
latency: the AMK megakernel vs eager PyTorch on the **same model + same GPU**, plus an honest
status for vLLM. Everything here is produced by `eval/baselines.py` and reproduced by
`eval/bench_baselines.py`; nothing is fabricated.

## What is compared

| baseline | what it measures | how |
| --- | --- | --- |
| **eager** | the SAME model doing **one per-op KV-cached decode step**, a stream of individual kernel launches (q/k/v/o GEMVs, RMSNorms, RoPE, attention, gate/up/down GEMVs, lm_head). This is exactly the work AMK fuses. | HF path: prime a `past_key_values` cache, time one incremental `model(next_tok, past_key_values=...)` step. Toy path: single-token forward over the matching prefix. CUDA-event timed. |
| **amk** | the AMK **megakernel decode/token at steady state**, ONE cooperative kernel launch per token, persistent device tables already built (per-token cost = counter-zero + new-token H2D copy + relaunch + read-back). | Drive AMK forward through positions `0..pos` to build the real KV cache, **gate** the step at `pos` against eager logits, then CUDA-event time the steady-state `run()`. Also reports the kernel-only `relaunch()` floor. |
| **vllm** | a third datapoint, **not run here**. | Attempted `import vllm`; fails on this Windows/dev box (no Windows wheels, Linux-only CUDA kernels). Recorded `status='not_run'` with the exact reason + the exact Linux command. Never a fabricated number. |

**Honesty rule (enforced in code):** no row reports a latency without a correctness PASS.
`eager` is the oracle (PASS by construction). `amk` reports a latency **only if** its megakernel
logits match eager's at the gate (`eval.oracle.logit_equivalence`, fp32 tolerance). If AMK is
wrong it returns `status='error'` and no latency, see `eval/bench.py`'s correctness gate.

## Measured results (RTX 5090 Laptop, sm_120, 82 SMs, torch 2.11+cu128, fp32)

Captured on this machine, `--context-len 16 --iters 100`. Re-run to regenerate.

### `--model toy` (self-contained toy Llama: 2L, hidden=256, 8h/4kv, head_dim=32, vocab=512)

| baseline | status | correct | ms/token | tokens/s | % of HBM roofline |
| --- | --- | --- | ---: | ---: | ---: |
| eager-decode | ok | PASS | 3.42 | 292 |, |
| **amk-decode** | ok | PASS | **0.72** | **1393** | 11144% (0.9% HBM util) |
| vllm-decode | not_run |, |, |, |, |

- **AMK vs eager: ~4.8x faster** wall-clock per token (0.72 ms vs 3.42 ms).
- AMK kernel-only (`relaunch`, no host marshalling): **0.47 ms/token**, the on-GPU floor; the
  ~0.25 ms gap to the steady-state `run()` number is per-token host overhead (counter reset, the
  new-token copy, output read-back).

### `--model llama` (small from-config `transformers.LlamaForCausalLM`, no download: 4L, hidden=256, 8h/4kv, vocab=512)

This is the **genuine apples-to-apples** point: eager runs HF's real incremental KV-cached decode
step (`past_key_values`), not a re-prefill.

| baseline | status | correct | ms/token | tokens/s | % of HBM roofline |
| --- | --- | --- | ---: | ---: | ---: |
| eager-decode | ok | PASS | 7.81 | 128 |, |
| **amk-decode** | ok | PASS | **1.32** | **759** | 11247% (0.9% HBM util) |
| vllm-decode | not_run |, |, |, |, |

- **AMK vs eager: ~5.9x faster** wall-clock per token (1.32 ms vs 7.81 ms).
- AMK kernel-only: **0.88 ms/token**.

(Exact numbers drift a few percent run-to-run on a WDDM display GPU; the ratios are stable.)

## Honest interpretation

**Where AMK wins (today, measured):** at this scale, single-stream decode in eager PyTorch is
dominated by **per-op kernel-launch overhead and Python dispatch**, dozens of tiny launches per
token, each with launch latency and an HBM round-trip for activations between ops. AMK collapses
the whole forward pass into **one cooperative launch** with activations kept on-chip / in a single
arena and no inter-op host dispatch. That is precisely the overhead the megakernel exists to
remove, and it shows up as a **~5x wall-clock win** on these small models. The win is real and
correctness-gated.

**Where AMK does NOT yet win / the caveat we report truthfully:**

1. **% of HBM roofline is low here (≈0.9%, ~8 GB/s of 896 GB/s peak).** This is *not* a regression
   from the project's "~30% of roofline" figure, that figure was on a much larger model
   (SmolLM2-135M). The roofline floor is `weight_bytes / HBM_bandwidth`; these *tiny* from-config
   models have tiny weights, so the floor is a fraction of a microsecond and the decode is
   **launch/sync-overhead-bound, not bandwidth-bound**. The big `%-of-bound` number honestly says
   "absolute latency is far above the bandwidth floor for a model this small," which is expected:
   you only approach the roofline when the weights are big enough that streaming them dominates.

2. **The eager number is the un-optimized per-op path, not an optimized server.** The ~5x win is
   over eager PyTorch decode, which is the thing AMK fuses, a fair and meaningful baseline, but it
   is *not* a claim of beating a CUDA-graph-captured / batched / paged-attention serving engine on
   absolute throughput. A larger model would shrink the eager-overhead advantage (eager's per-op
   launches amortize better when each op does more work) and simultaneously push AMK toward the
   bandwidth roofline, both effects matter and we do not extrapolate them here.

3. **AMK's GEMV is still the known M1→M2 lever.** Per the build state, the GEMV is bandwidth-naive
   (no `cp.async` pipelining, grid-wide sync per tile). The kernel-only floor (~0.5–0.9 ms) is
   where future `cp.async` double-buffered / coarser-sync work will move the roofline %; the
   wall-clock win above is from launch fusion, *separate from* and *additive with* that future
   kernel-efficiency work. Note: batch-1 decode is memory-bound; tensor-core MMA does not move a
   bandwidth-bound GEMV.
4. **AMK int8 (W8A16, near-lossless) BEATS CUDA-graphed cuBLAS bf16 at batch-1 decode across the
   Modal-verified, reproducible inference-class GPUs, L4 (300 GB/s, 1.18–1.33×), L40S (864 GB/s,
   1.25–1.27× @4B/6.7B) and A10G (600 GB/s, crossing at ≥3.5B, up to 1.08×), plus the consumer RTX 5090,
   but NOT the training-class A100/H100. Found by AMK's search and gated argmax-exact. It's an
   inference-vs-training regime split, NOT a clean function of bandwidth (the 864 GB/s L40S beats the
   600 GB/s A10G): the dividing line is inference-class vs training-class regime + the per-tile cross-SM
   sync cost amortized by larger GEMV-dominated models. The
   high-bandwidth training-class A100/H100 never cross even at 13B (0.55–0.79×, ratio declining as the
   model grows). The win is pos-0 / low-context at batch-1. (RTX 5090 (sm_120, consumer): AMK int8 beats
   cuBLAS bf16 by ~1.19–1.23× (4/8/16 layers), measured locally on the dev machine and backed by
   `paper/results/int8_search_multisize.json`. Measured locally, not on Modal, because Modal has no
   RTX 5090 silicon, which is why it is not part of the Modal inference-fleet sweep.)
   See [`DATACENTER_RESULTS.md`](DATACENTER_RESULTS.md).**

## vLLM status (not fabricated)

vLLM is recorded `status='not_run'` because it does not run in this environment:

> vLLM is not runnable in this environment: `ModuleNotFoundError: No module named 'vllm'`.
> vLLM publishes no Windows wheels and its CUDA kernels are Linux-only (`uv pip install vllm`
> fails here).

To get the third datapoint on a **Linux GPU box** with the SAME model:

```bash
uv pip install vllm
python -c "
from vllm import LLM, SamplingParams
import time
llm = LLM(model='<hf-model-id>', max_model_len=512, enforce_eager=False)
p = SamplingParams(max_tokens=128, ignore_eos=True)
_ = llm.generate(['hello'], p)                 # warmup + CUDA-graph capture
t = time.perf_counter(); o = llm.generate(['hello'], p); dt = time.perf_counter() - t
print('ms/token', dt / 128 * 1000)
"
```

When vLLM is run there, flip the record's `status` to `'ok'` and fill `latency_us` from the real
measurement, the `BaselineRecord` schema is already the right shape. **Until then, no vLLM number
is reported.**

## Reproduce

```bash
uv run python eval/bench_baselines.py --model toy   --gpu rtx5090     # toy Llama
uv run python eval/bench_baselines.py --model llama --gpu rtx5090     # small from-config Llama
uv run python tests/test_baselines.py                                  # acceptance tests
```

Flags: `--context-len` (KV context / decode position), `--warmup`, `--iters`, `--dtype {f32,f16,bf16}`,
`--device {auto,cuda,cpu}`. On CPU the `amk` row is `status='error'` (the megakernel is CUDA-only;
CPU has only the bit-exact ReferenceVM oracle, which is a correctness reference, not a perf target),
and the `eager` row is a **labelled CPU reference timing**, never quoted as a GPU number.
