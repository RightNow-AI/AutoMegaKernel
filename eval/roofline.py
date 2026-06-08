"""
AMK, THE ROOFLINE (distance to the HBM-bandwidth bound)
========================================================

Single-stream LLM decode is **memory-bound**: each token must stream every weight through HBM
exactly once, so the honest performance floor is ``weight_bytes / HBM_bandwidth``. Beating that
floor is physically impossible on one stream; the entire megakernel thesis is "get as close to
it as possible by removing the inter-op HBM bubbles and launch overhead." This module turns a
measured latency into the only metric that matters for that thesis: **percent of the bandwidth
bound** (and, equivalently, achieved HBM utilization).

``report(weight_bytes, measured_us, target)`` returns:
  * ``bound_us``    , the floor: time to stream the weights once at the target's peak bandwidth.
  * ``measured_us`` , what was measured (passed through).
  * ``pct_of_bound``, ``measured_us / bound_us * 100``. 100% == at the roofline; >100% == above
    the floor (real life: launch overhead, imperfect prefetch, attention/KV traffic, occupancy).
    A value of, say, 140% means "1.4x the theoretical floor", the headroom the search closes.
  * ``hbm_util_pct``, the dual view: achieved bandwidth / peak bandwidth = ``bound_us/measured_us``
    * 100. At the bound this is 100%; a kernel taking 2x the floor sustains 50% of peak HBM.

The target's bandwidth is *data* (a :class:`~schedule.ir.GpuTarget` field), so the roofline
retargets to any GPU by swapping the record, never by editing this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from schedule.ir import GpuTarget, MegakernelProgram, TARGETS


@dataclass
class RooflineReport:
    """Distance-to-bound for one (weights, latency, GPU) triple. ``pct_of_bound`` is the headline
    research metric; ``hbm_util_pct`` is its reciprocal view (achieved fraction of peak HBM).

    Two denominators are reported, both labeled honestly:
      * SPEC peak (``hbm_bandwidth_gbs``), the conventional vendor figure (a desktop/theoretical
        number); ``pct_of_bound`` / ``hbm_util_pct`` use it.
      * MEASURED peak (``measured_bw_gbs``), the real sustained HBM bandwidth this GPU reaches
        (``eval/peak_bandwidth.py``); ``pct_of_measured_bound`` / ``measured_hbm_util_pct`` use it.
        This is the FAIRER denominator: a kernel cannot beat what a trivial streaming kernel
        achieves, and on the laptop 5090 the spec figure (896) overstates the achievable floor."""

    bound_us: float
    measured_us: float
    pct_of_bound: float        # measured/bound * 100  (>= 100 when measured >= bound)  [SPEC]
    hbm_util_pct: float        # bound/measured * 100   (<= 100 when measured >= bound) [SPEC]
    weight_bytes: int
    hbm_bandwidth_gbs: float          # SPEC peak
    achieved_gbs: float        # weight_bytes / measured_us, in GB/s
    target: str = ""
    # ---- MEASURED-peak denominator (the fairer one); falls back to spec if unmeasured ----
    measured_bw_gbs: float = 0.0          # measured sustained peak (0 => fell back to spec)
    measured_bound_us: float = 0.0        # weights / measured peak, in us
    pct_of_measured_bound: float = 0.0    # measured_us / measured_bound_us * 100
    measured_hbm_util_pct: float = 0.0    # measured_bound_us / measured_us * 100
    measured_is_real: bool = False        # True iff a real measured peak was used (not spec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bound_us": self.bound_us,
            "measured_us": self.measured_us,
            "pct_of_bound": self.pct_of_bound,
            "hbm_util_pct": self.hbm_util_pct,
            "weight_bytes": self.weight_bytes,
            "hbm_bandwidth_gbs": self.hbm_bandwidth_gbs,
            "achieved_gbs": self.achieved_gbs,
            "target": self.target,
            "measured_bw_gbs": self.measured_bw_gbs,
            "measured_bound_us": self.measured_bound_us,
            "pct_of_measured_bound": self.pct_of_measured_bound,
            "measured_hbm_util_pct": self.measured_hbm_util_pct,
            "measured_is_real": self.measured_is_real,
        }

    def grep_line(self, tag: str = "AutoKernel") -> str:
        return (f"{tag} roofline target:{self.target} bound_us:{self.bound_us:.3f} "
                f"measured_us:{self.measured_us:.3f} pct_of_bound:{self.pct_of_bound:.1f} "
                f"hbm_util_pct:{self.hbm_util_pct:.1f} achieved_gbs:{self.achieved_gbs:.1f} "
                f"measured_bw_gbs:{self.measured_bw_gbs:.1f} "
                f"pct_of_measured_bound:{self.pct_of_measured_bound:.1f} "
                f"measured_hbm_util_pct:{self.measured_hbm_util_pct:.1f}")

    def report(self) -> str:
        spec = (f"[roofline {self.target}] floor={self.bound_us:.2f}us  "
                f"measured={self.measured_us:.2f}us  "
                f"{self.pct_of_bound:.1f}% of SPEC bound  "
                f"({self.hbm_util_pct:.1f}% of {self.hbm_bandwidth_gbs:.0f} GB/s spec peak; "
                f"achieved {self.achieved_gbs:.1f} GB/s)")
        if self.measured_is_real:
            spec += (f"\n  vs MEASURED peak {self.measured_bw_gbs:.0f} GB/s: "
                     f"{self.pct_of_measured_bound:.1f}% of measured bound "
                     f"({self.measured_hbm_util_pct:.1f}% of measured peak) "
                     f"- the fairer denominator")
        return spec


def _weight_bytes(program_or_weight_bytes: Any) -> int:
    """Accept either an int byte-count or a :class:`MegakernelProgram` (take its weight bytes)."""
    if isinstance(program_or_weight_bytes, MegakernelProgram):
        return int(program_or_weight_bytes.total_weight_bytes())
    if isinstance(program_or_weight_bytes, (int, float)):
        return int(program_or_weight_bytes)
    # last resort: anything exposing total_weight_bytes()
    fn = getattr(program_or_weight_bytes, "total_weight_bytes", None)
    if callable(fn):
        return int(fn())
    raise TypeError(
        "program_or_weight_bytes must be an int byte-count or a MegakernelProgram, "
        f"got {type(program_or_weight_bytes).__name__}")


def _resolve_target(target: GpuTarget | str) -> GpuTarget:
    if isinstance(target, GpuTarget):
        return target
    if isinstance(target, str):
        if target not in TARGETS:
            raise KeyError(f"unknown target '{target}'; known: {sorted(TARGETS)}")
        return TARGETS[target]
    raise TypeError(f"target must be a GpuTarget or a name, got {type(target).__name__}")


def report(program_or_weight_bytes: Any,
           measured_us: float,
           target: GpuTarget | str) -> RooflineReport:
    """Compute the distance from a measured latency to the HBM-bandwidth bound.

    Args:
      program_or_weight_bytes: a :class:`MegakernelProgram` (weights taken from its WEIGHT
        buffers) OR an explicit byte-count of bytes that must be streamed from HBM per pass.
      measured_us: the measured per-pass latency in microseconds (from :func:`eval.bench.bench`,
        which only yields one when the oracle says the kernel is correct).
      target:      a :class:`GpuTarget` or a registry name ("rtx5090", "b200", ...). Its
        ``hbm_bandwidth_gbs`` defines the floor.

    Returns a :class:`RooflineReport`. ``bound_us > 0`` whenever there are weights and a positive
    bandwidth; ``pct_of_bound >= 100`` exactly when ``measured_us >= bound_us``.
    """
    wbytes = _weight_bytes(program_or_weight_bytes)
    tgt = _resolve_target(target)
    if measured_us is None or measured_us <= 0:
        raise ValueError(f"measured_us must be a positive latency, got {measured_us!r} "
                         "(did you forget the correctness gate produced None?)")
    if tgt.hbm_bandwidth_gbs <= 0:
        raise ValueError(f"target {tgt.name} has non-positive HBM bandwidth")

    bound_us = tgt.bandwidth_bound_us(wbytes)   # weight_bytes / SPEC bw, in us
    pct_of_bound = (measured_us / bound_us * 100.0) if bound_us > 0 else float("inf")
    hbm_util_pct = (bound_us / measured_us * 100.0) if measured_us > 0 else 0.0
    achieved_gbs = (wbytes / (measured_us * 1e-6)) / 1e9 if measured_us > 0 else 0.0

    # ---- MEASURED-peak denominator (fairer): weights / measured sustained bandwidth ----
    measured_is_real = tgt.measured_bw_gbs > 0
    measured_bw = tgt.measured_bw_gbs if measured_is_real else tgt.hbm_bandwidth_gbs
    measured_bound_us = tgt.measured_bandwidth_bound_us(wbytes)  # falls back to spec if 0
    pct_of_measured = ((measured_us / measured_bound_us * 100.0)
                       if measured_bound_us > 0 else float("inf"))
    measured_util = (measured_bound_us / measured_us * 100.0) if measured_us > 0 else 0.0

    return RooflineReport(
        bound_us=bound_us,
        measured_us=float(measured_us),
        pct_of_bound=pct_of_bound,
        hbm_util_pct=hbm_util_pct,
        weight_bytes=wbytes,
        hbm_bandwidth_gbs=tgt.hbm_bandwidth_gbs,
        achieved_gbs=achieved_gbs,
        target=tgt.name,
        measured_bw_gbs=measured_bw,
        measured_bound_us=measured_bound_us,
        pct_of_measured_bound=pct_of_measured,
        measured_hbm_util_pct=measured_util,
        measured_is_real=measured_is_real,
    )


__all__ = ["report", "RooflineReport"]
