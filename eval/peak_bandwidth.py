"""
AMK, MEASURED SUSTAINED HBM BANDWIDTH (the honest roofline denominator)
=======================================================================

Single-stream decode is memory-bound, so the roofline denominator is the GPU's *sustained*
HBM bandwidth. The vendor spec figure (e.g. the RTX 5090 desktop part's 896 GB/s) is the
WRONG denominator for THIS machine: this is the **laptop** GB203 whose real sustained HBM
bandwidth is materially lower. Reporting "% of 896 GB/s" therefore understates how close AMK
actually is to the achievable floor. This module measures the real number.

What it measures
----------------
Two independent memory-bound microbenchmarks, each a CUDA-event median over many iters after
warmup (the same timing discipline as :mod:`eval.bench`), each reporting achieved GB/s:

  * **D2D copy**, ``dst.copy_(src)`` over a large fp32 buffer. Traffic = 2*nbytes (1 read +
    1 write). The simplest, highest-achievable sustained-bandwidth probe.
  * **STREAM triad**, ``a = b + alpha*c`` over large fp32 buffers (the classic McCalpin
    triad). Traffic = 4*nbytes (2 reads + 1 write counted by convention; PyTorch fuses the
    read-modify-write of the output, so we count the bytes that MUST cross HBM: read b, read
    c, write a => 3*nbytes). We report it explicitly so the convention is auditable.

The **measured peak** we return per GPU is the MAX of the two probes (the best sustained
number the device demonstrably reaches), which is the fairest denominator: AMK cannot exceed
what a trivial streaming kernel achieves, so dividing by the trivial-kernel peak is the most
honest "% of achievable bandwidth" we can state.

Output is a grep-friendly line plus a dict; run as a script to print the local GPU's number.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class BandwidthResult:
    """Measured sustained HBM bandwidth for one GPU. ``peak_gbs`` is the headline denominator -
    the BEST sustained bandwidth the device demonstrably reaches (the fastest timed iteration,
    i.e. the moment the GPU was best boosted and not throttling). We also keep the median for
    each probe so the boost-state spread is auditable. All numbers are real CUDA-event timings;
    nothing is spec-derived."""

    gpu_name: str
    compute_capability: str
    buffer_bytes: int
    iters: int
    warmup: int
    d2d_copy_gbs: float          # 2*nbytes / best_time  (sustained peak this run reached)
    triad_gbs: float             # 3*nbytes / best_time   (read b, read c, write a)
    peak_gbs: float              # max(d2d_copy_gbs, triad_gbs), the fair roofline denominator
    d2d_copy_median_gbs: float   # median over iters (lower; boost/throttle drags it down)
    triad_median_gbs: float
    d2d_copy_best_ms: float
    triad_best_ms: float
    clock_state: str = ""        # nvidia-smi mem/sm clocks + throttle reasons during timing

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def grep_line(self, tag: str = "AutoKernel") -> str:
        return (f"{tag} peak_bandwidth gpu:{self.gpu_name!r} cc:{self.compute_capability} "
                f"d2d_copy_gbs:{self.d2d_copy_gbs:.1f} triad_gbs:{self.triad_gbs:.1f} "
                f"peak_gbs:{self.peak_gbs:.1f}")

    def report(self) -> str:
        return (f"[peak_bandwidth {self.gpu_name} sm_{self.compute_capability}]\n"
                f"  D2D copy  : {self.d2d_copy_gbs:7.1f} GB/s peak  "
                f"({self.d2d_copy_median_gbs:.1f} median, {self.buffer_bytes/1e6:.0f} MB buf)\n"
                f"  STREAM triad: {self.triad_gbs:7.1f} GB/s peak  "
                f"({self.triad_median_gbs:.1f} median)\n"
                f"  MEASURED PEAK: {self.peak_gbs:.1f} GB/s  (fair roofline denominator)\n"
                f"  clocks during timing: {self.clock_state}")


def _clock_state() -> str:
    """Best-effort mem/sm clocks + throttle reasons via nvidia-smi (annotates every number)."""
    import subprocess
    try:
        q = "clocks.mem,clocks.max.mem,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active"
        p = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15)
        return p.stdout.strip().replace("\n", " | ") if p.returncode == 0 else f"rc={p.returncode}"
    except Exception as e:  # pragma: no cover - nvidia-smi not present
        return f"nvidia-smi error: {type(e).__name__}"


def _event_time_ms(fn, warmup: int, iters: int) -> tuple[float, float]:
    """(median_ms, best_ms) over ``iters`` real CUDA-event-timed runs after ``warmup``. The ONLY
    correct way to time async GPU work, same discipline as eval.bench. ``best_ms`` is the
    sustained-peak sample (GPU best boosted); ``median_ms`` shows the boost/throttle spread."""
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(max(1, iters)):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        samples.append(s.elapsed_time(e))  # ms
    return statistics.median(samples), min(samples)


def measure_peak_bandwidth(buffer_bytes: int = 1 << 30, warmup: int = 30,
                           iters: int = 100, device: str = "cuda") -> BandwidthResult:
    """Measure THIS GPU's sustained HBM bandwidth via a D2D copy and a STREAM triad.

    Args:
      buffer_bytes: size of EACH operand buffer (default 1 GiB), large enough to defeat L2.
      warmup/iters: warmup launches discarded, then ``iters`` CUDA-event-timed samples.
      device:       CUDA device.

    Returns a :class:`BandwidthResult`. ``peak_gbs`` is the max of the two probes, the fairest
    "% of achievable bandwidth" denominator (AMK cannot beat a trivial streaming kernel).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("measure_peak_bandwidth requires a CUDA GPU (no fabrication on CPU)")
    dev = torch.device(device)
    if dev.index is not None:
        torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    n = buffer_bytes // 4  # fp32 elements

    nbytes = n * 4

    # ---- D2D copy: traffic = 1 read + 1 write = 2*nbytes ----
    src = torch.empty(n, device=dev, dtype=torch.float32).uniform_(-1, 1)
    dst = torch.empty(n, device=dev, dtype=torch.float32)
    copy_med_ms, copy_best_ms = _event_time_ms(lambda: dst.copy_(src), warmup, iters)
    d2d_gbs = (2 * nbytes) / (copy_best_ms * 1e-3) / 1e9
    d2d_med_gbs = (2 * nbytes) / (copy_med_ms * 1e-3) / 1e9

    # ---- STREAM triad: a = b + alpha*c ; HBM traffic = read b + read c + write a = 3*nbytes ----
    b = torch.empty(n, device=dev, dtype=torch.float32).uniform_(-1, 1)
    c = torch.empty(n, device=dev, dtype=torch.float32).uniform_(-1, 1)
    a = torch.empty(n, device=dev, dtype=torch.float32)
    alpha = 3.0

    def _triad():
        torch.add(b, c, alpha=alpha, out=a)

    triad_med_ms, triad_best_ms = _event_time_ms(_triad, warmup, iters)
    triad_gbs = (3 * nbytes) / (triad_best_ms * 1e-3) / 1e9
    triad_med_gbs = (3 * nbytes) / (triad_med_ms * 1e-3) / 1e9

    return BandwidthResult(
        gpu_name=props.name,
        compute_capability=f"{props.major}{props.minor}",
        buffer_bytes=nbytes,
        iters=iters,
        warmup=warmup,
        d2d_copy_gbs=d2d_gbs,
        triad_gbs=triad_gbs,
        peak_gbs=max(d2d_gbs, triad_gbs),
        d2d_copy_median_gbs=d2d_med_gbs,
        triad_median_gbs=triad_med_gbs,
        d2d_copy_best_ms=copy_best_ms,
        triad_best_ms=triad_best_ms,
        clock_state=_clock_state(),
    )


def main() -> int:
    if not torch.cuda.is_available():
        print("peak_bandwidth: no CUDA GPU available, cannot measure (refusing to fabricate).")
        return 1
    res = measure_peak_bandwidth()
    print(res.report())
    print(res.grep_line())
    return 0


__all__ = ["BandwidthResult", "measure_peak_bandwidth"]


if __name__ == "__main__":
    raise SystemExit(main())
