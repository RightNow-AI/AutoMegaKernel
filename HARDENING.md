# AMK Production Hardening

The contract: **so stable it never hangs, crashes, OOMs, or miscomputes - any bad input fails fast
with a clear reason, and nothing is hardcoded to one case.** Correctness is already guaranteed *by
construction* (the reference oracle gates every kernel bit-for-bit vs eager HF); this tracker is about
the *other* failure modes.

Source: a 6-agent production-hardening audit (100 findings). Every fix is a **safe additive guard** -
it only fires on bad input and never touches the happy path. **Rule: add a `tests/test_robustness.py`
case for every guard**, so the guarantee is regression-proof.

## Shipped (verified)
- **HANG** - `threads_per_block=512` deadlocked the cooperative grid-sync. Removed from
  `THREADS_PER_BLOCK_CHOICES` + a loader guard refuses it in 0.02s. (`8a11b42`)
- **DYNAMIC** - `from_hf` now rejects MoE (`num_local_experts`), MLA (`kv_lora_rank`), and
  `hidden_size` not divisible by `num_attention_heads`, with precise reasons. (`1503d63`)
- **OOM** - `_normalize_knobs` range-guards the SMEM/register-footprint knobs (cpa_\*, gemv_max_k,
  cols_per_warp, kunroll). (`1503d63`)
- **Test** - `tests/test_robustness.py` proves each of the above fails fast (33 tests green; it already
  caught a false-rejection in the knob guard). (`1503d63`)

## Backlog - safe additive guards (ship next, each with a robustness test)
Priority = HANG/MISCOMPUTE first.

- **MISCOMPUTE - input shape/dtype validation** (`vm/loader.py` `_bind_external` ~561, steady-state
  ~632): assert bound inputs match the buffer shape+dtype (validate numel+dtype, lenient on
  reshape) so a wrong input fails fast instead of producing wrong logits.
- **MISCOMPUTE - semantic param bounds in `validate()`** (`schedule/ir.py`): per-op checks beyond
  int32 range - GEMV/GEMM `K>0 & N_tile>0`, ATTENTION `kv_len>0 & kv_start>=0`, ROPE `head_dim>0`.
  Catches a degenerate program before the GPU sees it.
- **MISCOMPUTE - token/pos bounds** (`generate.py` ~156 `_argmax_logits`, ~287): assert
  `0<=token<vocab` and `pos<max_seq` so an OOB token can't index the embedding / an over-long decode
  can't wrap the KV cache.
- **MISCOMPUTE - KV_APPEND pos guard** (`schedule/lower.py:372`): raise if `pos>=max_seq`.
- **HANG - zero-tile / split-K guards** (`schedule/lower.py` `_tiled_linear` ~265, `_op_attention`
  ~427): raise if a projection emits 0 tiles (N<=0) or split-KV yields <2 parts.
- **CRASH/MISCOMPUTE - id + arena bounds at load** (`vm/loader.py` ~705/715/732/742): defensively
  assert task/buffer/counter/SM ids in range + arena slices fit, before packing.
- **DYNAMIC - more `from_hf` rejections**: per-head/non-uniform `head_dim`, grouped/per-layer RoPE
  types, tie-embeddings consistency.
- **STATIC - dynamic `num_sms`** (`vm/loader.py:420`): query the live device instead of defaulting 82.

## Backlog - deep work (not a one-line guard)
- **HANG (the big one) - device-side watchdog**: add `MAX_SPIN_ITERS` to `amk_wait_all`
  (`vm/sync.cuh`) so a counter that never signals **traps** instead of spinning forever. This is the
  "kernel can never hang" guarantee. Needs a high threshold (only a true deadlock exceeds it) + a
  build + correctness re-verify.
- **HANG root cause - cooperative `grid.sync()` at tpb>256** (`vm/scheduler.cu`): diagnose so the cap
  can be raised (also the path to the GEMV-bandwidth perf win - more warps).
- **MISCOMPUTE - device bounds + softmax denom + dtype trap** (`vm/ops.cuh`): guard `inv_l` against
  zero/empty attention windows, read KV strides from buffer metadata, trap on an unhandled GEMV dtype.
- **HANG - provenance race-proof scaling** (`schedule/ir.py` `_PROVENANCE_MAX_TASKS`): RAW/WAW proofs
  are skipped above ~8000 tasks; large models lose race detection.

Run `python -m pytest tests/test_robustness.py` after each guard.
