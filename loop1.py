"""
AMK, LOOP 1 AGENT HARNESS (single-instruction AutoKernel-style tuning)
======================================================================

This is the **Loop-1** analogue of ``harness.loop`` (Loop 2, schedule search). It drives the
AutoKernel discipline on ONE ABI micro-kernel at a time:

    read the search surface  ->  propose a kernel variant (-D knob set)
        ->  BUILD it (instructions/_build, nvcc -D flags)
        ->  VERIFY correctness vs instructions/reference.py (the ground truth, isolated)
        ->  MICROBENCH latency with CUDA events (warmup + iters, median-ish)
        ->  KEEP / REVERT: correctness ALWAYS first, then a strict >= 1% latency win
        ->  LOG every trial to results.tsv via flywheel.log
    repeat until the budget is spent.

It is the single-instruction twin of ``amk loop``: same propose/eval/keep-revert/log shape, but
the edit surface is ONE kernel file (``instructions/cuda/<op>.cu``) and its searchable ``-D``
macros (``instructions/gen.SEARCH_SPACE``) instead of a ``ScheduleConfig``. The correctness gate is
the locked per-op reference in ``instructions/reference.py``, a wrong variant fails its OWN unit
test, so (unlike a megakernel schedule) NO GPU hang is possible in this loop.

HONESTY (enforced in code, not comments):
  * Every latency comes from a real CUDA-event measurement (>= warmup, >= iters) on the local GPU.
  * A latency is NEVER recorded for a variant that did not pass the reference correctness check
    (``_eval_variant`` only times a kernel after ``correct=True``; a non-PASS row logs blank
    latency, matching the flywheel honesty rule).
  * Nothing here touches Modal / a cloud GPU, the local device is the only one used.
  * The reference (``instructions/reference.py``) is ground truth; if the CUDA kernel disagrees,
    the CUDA kernel is wrong by definition and is REVERTED.

CLI surface lives in ``amk_cli.py`` as ``amk tune-instruction <op> --gpu <arch> --budget N``.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from flywheel.log import ResultRow, append_result  # noqa: E402
from instructions import gen as _gen  # noqa: E402  (search space + Variant + _eval_variant)
from instructions.verify_inst import CASES  # noqa: E402
from schedule.ir import TARGETS, GpuTarget  # noqa: E402

DEFAULT_RESULTS_TSV = os.path.join("workspace", "results.tsv")

# keep/revert: a variant must be correct AND strictly faster than the incumbent by this margin to
# be KEPT. Mirrors AutoKernel / Loop 2 (program.md §3): correctness first, then a >= 1% gain.
_MIN_GAIN = 0.01

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _resolve_target(gpu: str) -> GpuTarget:
    if gpu not in TARGETS:
        raise KeyError(f"unknown gpu {gpu!r}; known targets: {', '.join(sorted(TARGETS))}")
    return TARGETS[gpu]


def _op_bytes(op: str, dtype: torch.dtype) -> int:
    """Bytes of HBM traffic the op's isolated test case moves (inputs read + outputs written), so we
    can report an honest ``pct_of_roofline`` for the micro-kernel against ``bandwidth_bound_us``.

    Built from the SAME ``Case.build_inputs`` shapes ``verify_inst`` uses, so the byte count matches
    exactly what the benchmarked kernel touches. Returns 0 for an unknown op (-> pct omitted)."""
    es = torch.tensor([], dtype=dtype).element_size()
    i64 = 8  # index tensors (ids/pos) are int64 in the verify_inst cases
    if op == "gemv_tile":
        M, K, N_tile = 1, 512, 512
        # read x[M,K] + the W tile [N_tile,K]; write out tile [M,N_tile]
        return (M * K + N_tile * K + M * N_tile) * es
    if op == "rmsnorm":
        H = 2048
        return (H + H + H) * es                      # x + w read, out written
    if op == "silu_mul":
        n = 4096
        return (n + n + n) * es                       # gate + up read, out written
    if op == "add":
        n = 4096
        return (n + n + n) * es                       # a + b read, out written
    if op == "rope":
        S, n_heads, head_dim = 8, 4, 64
        return (S * n_heads * head_dim * 2) * es + S * i64
    if op == "attention_tile":
        n_heads, n_kv, head_dim, kv_len = 4, 2, 64, 37
        rd = (n_heads * head_dim + 2 * kv_len * n_kv * head_dim)
        wr = n_heads * head_dim
        return (rd + wr) * es
    if op == "embed":
        S, H = 4, 2048
        return (S * H) * es + S * i64                  # gather S rows of the table, write S*H
    return 0


def tune_instruction(op: str, gpu: str, budget: int = 6, *,
                     dtype: str = "fp32",
                     results_path: str = DEFAULT_RESULTS_TSV,
                     tag: str = "",
                     verbose: bool = False) -> dict[str, Any]:
    """Run the Loop-1 keep/revert loop on a single ABI instruction and return a JSON-able summary.

    Args:
      op:           an ABI op name (key of ``instructions.verify_inst.CASES``; e.g. ``gemv_tile``).
      gpu:          a registered ``GpuTarget`` name (e.g. ``rtx5090``), used for the roofline only;
                    the kernel is built+timed on the LOCAL device (must match for honest numbers).
      budget:       max number of variants to try (>= 1). Variant 0 is always the incumbent
                    (built-in defaults == exactly what ``verify_inst`` validates).
      dtype:        element dtype for the isolated test (fp32 | fp16 | bf16).
      results_path: results.tsv to append one row per trial to (flywheel substrate).
      tag:          campaign tag for the results rows (default: ``tune-<op>-<gpu>``).

    Returns ``{op, gpu, dtype, device, trials, best_variant, best_us, baseline_us, speedup,
    all_correct, pct_of_roofline, bound_us, results_tsv, trials_detail}``. ``all_correct`` is True
    iff every BUILT variant passed the reference correctness check (the correctness_preserved gate).
    """
    if op not in CASES:
        raise KeyError(f"unknown op {op!r}; known: {', '.join(sorted(CASES))}")
    if budget < 1:
        raise ValueError("tune-instruction budget must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available, tune-instruction needs the local GPU "
                           "(Loop 1 builds + times a real kernel; reference is the oracle).")
    if dtype not in _DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; choose from {', '.join(_DTYPES)}")

    target = _resolve_target(gpu)
    dt = _DTYPES[dtype]
    dev_name = torch.cuda.get_device_name(0)
    tag = tag or f"tune-{op}-{target.name}"
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    op_bytes = _op_bytes(op, dt)
    bound_us = target.bandwidth_bound_us(op_bytes) if op_bytes else float("nan")

    def _pct(lat_us: float) -> float | None:
        if not (op_bytes and math.isfinite(bound_us) and lat_us and lat_us > 0):
            return None
        return round(bound_us / lat_us * 100.0, 4)

    variants = _gen._proposals(op, budget)  # incumbent (defaults) first, then the capped grid

    baseline_us: float | None = None
    best_us: float | None = None
    best_variant: str | None = None
    all_correct = True            # every BUILT variant matched the reference
    n_correct = 0
    trials_detail: list[dict[str, Any]] = []

    if verbose:
        print(f"AMK Loop-1 | tune-instruction {op} | device={dev_name} | gpu(roofline)={target.name} "
              f"| dtype={dtype} | budget={budget}")

    for i, variant in enumerate(variants):
        tr = _gen._eval_variant(variant, dt, verbose)  # build + correctness(vs reference) + bench

        if tr.built and not tr.correct:
            all_correct = False
        if tr.correct:
            n_correct += 1

        # ---- keep/revert: correctness ALWAYS first, then a strict >= 1% latency win ----
        kept = False
        if tr.built and tr.correct and math.isfinite(tr.latency_us):
            if i == 0:
                baseline_us = tr.latency_us
            if best_us is None or tr.latency_us < best_us * (1.0 - _MIN_GAIN):
                kept = True
                best_us = tr.latency_us
                best_variant = tr.variant
        tr.kept = kept

        # ---- correctness verdict for the row (honesty: blank latency unless PASS) ----
        if not tr.built:
            correctness = "CRASH"          # would-not-compile / build failure
        elif tr.correct:
            correctness = "PASS"
        else:
            correctness = "FAIL"           # ran but mismatched the reference oracle
        lat_field = round(tr.latency_us, 4) if (tr.correct and math.isfinite(tr.latency_us)) else ""
        pct = _pct(tr.latency_us) if (tr.correct and math.isfinite(tr.latency_us)) else None
        row_tag = "kept" if kept else ("rejected" if not tr.built
                                       else ("revert" if not tr.correct else "tried"))
        append_result(ResultRow(
            experiment=i, tag=row_tag, loop="instruction", model=f"inst:{op}",
            gpu=target.name, regime="single-stream", correctness=correctness,
            latency_us=lat_field, pct_of_roofline=(pct if pct is not None else ""),
            schedule_id="", kernel_id=f"{op}[{tr.variant}]",
            description=f"dtype={dtype}; {tr.variant}; rel_err={tr.max_rel_err:.2e}; "
                        f"{tr.note}"[:120].replace("\t", " ")),
            path=results_path)

        trials_detail.append({
            "trial": i, "variant": tr.variant, "built": tr.built, "correct": tr.correct,
            "correctness": correctness, "max_rel_err": (None if math.isnan(tr.max_rel_err)
                                                        else round(tr.max_rel_err, 8)),
            "latency_us": (round(tr.latency_us, 4) if (tr.correct and math.isfinite(tr.latency_us))
                           else None),
            "pct_of_roofline": pct, "kept": kept, "note": tr.note,
        })
        if verbose:
            status = ("KEEP" if kept else ("ok  " if tr.correct else
                      ("XBUILD" if not tr.built else "XNUM ")))
            lat = f"{tr.latency_us:8.2f}us" if (tr.correct and math.isfinite(tr.latency_us)) \
                else "   --   "
            print(f"  [{status}] {tr.variant:24s} {lat}" + (f"  ({tr.note})" if tr.note else ""))

    speedup = (round(baseline_us / best_us, 4)
               if (baseline_us and best_us and best_us > 0) else None)

    summary = {
        "op": op,
        "gpu": target.name,
        "dtype": dtype,
        "device": dev_name,
        "trials": len(trials_detail),
        "n_correct": n_correct,
        "best_variant": best_variant,
        "best_us": round(best_us, 4) if best_us is not None else None,
        "baseline_us": round(baseline_us, 4) if baseline_us is not None else None,
        "speedup": speedup,
        "all_correct": all_correct,
        "pct_of_roofline": _pct(best_us) if best_us is not None else None,
        "bound_us": round(bound_us, 6) if math.isfinite(bound_us) else None,
        "op_bytes": op_bytes,
        "results_tsv": results_path,
        "trials_detail": trials_detail,
    }
    return summary


__all__ = ["tune_instruction", "DEFAULT_RESULTS_TSV"]
