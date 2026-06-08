"""
AMK, THE FIXED END-TO-END LATENCY BENCHMARK (correctness-gated, by construction)
================================================================================

This is the one place a number called "latency" is allowed to exist. The single most important
property of the whole research loop is that **we never report a speed for a wrong kernel**, an
autoresearch flywheel that optimizes latency will gleefully discover that the fastest kernel is
the one that computes garbage, unless the harness physically refuses to print a time without a
correctness proof. So:

    HARD HONESTY RULE (enforced in code, not in comments):
      bench() will NOT return a latency unless ``oracle_verdict.correct is True``.
      If the verdict is FAIL it either raises (strict, the default) or returns a result whose
      ``latency_us is None`` and ``correctness == "FAIL"``. There is no third option.

Timing methodology
------------------
  * CUDA: ``torch.cuda.Event`` start/stop around each iteration with a ``torch.cuda.synchronize``
    bracket, the only correct way to time async GPU work. We warm up first (JIT, allocator,
    clocks), then take ``iters`` samples and report median + mean + p10/p90 + min.
  * CPU: ``time.perf_counter``. This is explicitly labelled ``reference`` (``is_real_perf=False``)
    because a CPU reference-VM time is NOT a GPU performance number and must never be quoted as
    one. It exists so the harness runs end-to-end on a laptop with no GPU.

Output is emitted as **grep-friendly** single lines (``AutoKernel ... correctness:... latency_us:...``)
so the flywheel / CI can scrape runs without parsing Python objects.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from eval.oracle import Verdict


class CorrectnessGateError(RuntimeError):
    """Raised by :func:`bench` (in strict mode) when asked to time a kernel whose oracle verdict
    is not ``correct``. The benchmark physically cannot produce a latency for a wrong kernel."""


@dataclass
class BenchResult:
    """Outcome of a benchmark run. ``latency_us`` is ``None`` whenever ``correctness != 'PASS'``;
    consumers MUST check ``correctness`` before trusting any timing field."""

    correctness: str                      # "PASS" | "FAIL"
    latency_us: float | None = None       # median per-call latency, microseconds (None if FAIL)
    device: str = ""
    is_real_perf: bool = True             # False for the CPU reference path (label, never quote)
    warmup: int = 0
    iters: int = 0
    # distribution (all None on FAIL)
    mean_us: float | None = None
    min_us: float | None = None
    p10_us: float | None = None
    p90_us: float | None = None
    std_us: float | None = None
    samples_us: list[float] = field(default_factory=list)
    verdict: Verdict | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.correctness == "PASS"

    def grep_line(self, tag: str = "AutoKernel") -> str:
        """One scrape-friendly line for CI/flywheel ingestion."""
        lat = "None" if self.latency_us is None else f"{self.latency_us:.3f}"
        kind = "measured" if self.is_real_perf else "reference"
        return (f"{tag} correctness:{self.correctness} latency_us:{lat} "
                f"latency_kind:{kind} device:{self.device} iters:{self.iters} "
                f"warmup:{self.warmup}")

    def report(self) -> str:
        lines = [self.grep_line()]
        if self.verdict is not None:
            lines.append("  " + self.verdict.report().replace("\n", "\n  "))
        if self.latency_us is not None:
            lines.append(
                f"  latency_us median={self.latency_us:.3f} mean={self.mean_us:.3f} "
                f"min={self.min_us:.3f} p10={self.p10_us:.3f} p90={self.p90_us:.3f} "
                f"std={self.std_us:.3f}")
            if not self.is_real_perf:
                lines.append("  (CPU reference timing, NOT a GPU performance number)")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def _time_cuda(call: Callable[[], Any], warmup: int, iters: int,
               device: torch.device) -> list[float]:
    """Event-timed samples (microseconds), one per iteration, with a sync bracket."""
    # set_device needs an explicit index; a bare torch.device('cuda') has device.index is None.
    if device.index is not None:
        torch.cuda.set_device(device)
    for _ in range(max(0, warmup)):
        call()
    torch.cuda.synchronize(device)
    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(max(1, iters)):
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)  # ms -> us
    return samples


def _time_cpu(call: Callable[[], Any], warmup: int, iters: int) -> list[float]:
    """perf_counter samples (microseconds). Reference only, never a GPU perf number."""
    for _ in range(max(0, warmup)):
        call()
    samples: list[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        call()
        samples.append((time.perf_counter() - t0) * 1e6)
    return samples


def bench(run_fn: Callable[[], Any] | Callable[..., Any],
          oracle_verdict: Verdict,
          warmup: int = 10,
          iters: int = 50,
          device: str | torch.device = "cuda",
          strict: bool = True,
          run_args: tuple = (),
          run_kwargs: dict | None = None) -> BenchResult:
    """Time ``run_fn`` end-to-end, but ONLY if ``oracle_verdict.correct``.

    Args:
      run_fn:        the thing to time. Called as ``run_fn(*run_args, **run_kwargs)`` each
                     iteration (default no args). For a megakernel this is one forward pass /
                     one decoded token.
      oracle_verdict: a :class:`Verdict` from :mod:`eval.oracle`. THE GATE. If it is not correct,
                     no latency is produced.
      warmup/iters:  warmup launches discarded, then ``iters`` timed samples.
      device:        "cuda" -> event timing (real perf); "cpu" -> perf_counter (reference only).
      strict:        if True (default) raise :class:`CorrectnessGateError` on a failed verdict;
                     if False return a ``BenchResult`` with ``correctness='FAIL'`` and no latency.

    Returns a :class:`BenchResult`. On a failed verdict (non-strict) every timing field is ``None``.
    """
    run_kwargs = run_kwargs or {}
    dev = torch.device(device)
    # torch.cuda.synchronize/set_device need an explicit index; a bare device('cuda') has none.
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        dev = torch.device("cuda", torch.cuda.current_device())
    is_cuda = dev.type == "cuda"

    # ---- THE GATE -----------------------------------------------------------------------
    if not (oracle_verdict is not None and oracle_verdict.correct):
        msg = ("refusing to report latency: oracle verdict is not correct "
               f"({'no verdict' if oracle_verdict is None else oracle_verdict.check + ' FAIL'}). "
               "A benchmark for an incorrect kernel is a lie.")
        if strict:
            raise CorrectnessGateError(msg)
        return BenchResult(correctness="FAIL", latency_us=None, device=str(dev),
                           is_real_perf=is_cuda, warmup=warmup, iters=iters,
                           verdict=oracle_verdict, notes=[msg])

    if is_cuda and not torch.cuda.is_available():
        # Honesty again: do not silently fake a CUDA timing on a machine without CUDA.
        if strict:
            raise CorrectnessGateError("device='cuda' requested but CUDA is not available")
        return BenchResult(correctness="FAIL", latency_us=None, device=str(dev),
                           is_real_perf=False, warmup=warmup, iters=iters, verdict=oracle_verdict,
                           notes=["CUDA requested but unavailable"])

    def call():
        return run_fn(*run_args, **run_kwargs)

    if is_cuda:
        samples = _time_cuda(call, warmup, iters, dev)
    else:
        samples = _time_cpu(call, warmup, iters)

    samples_sorted = sorted(samples)
    n = len(samples_sorted)

    def pct(q: float) -> float:
        if n == 1:
            return samples_sorted[0]
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        return samples_sorted[idx]

    res = BenchResult(
        correctness="PASS",
        latency_us=statistics.median(samples),
        device=str(dev),
        is_real_perf=is_cuda,
        warmup=warmup, iters=iters,
        mean_us=statistics.fmean(samples),
        min_us=min(samples),
        p10_us=pct(0.10),
        p90_us=pct(0.90),
        std_us=(statistics.pstdev(samples) if n > 1 else 0.0),
        samples_us=samples,
        verdict=oracle_verdict,
    )
    if not is_cuda:
        res.notes.append("CPU reference timing, label only, not a GPU performance number")
    return res


__all__ = ["bench", "BenchResult", "CorrectnessGateError"]
