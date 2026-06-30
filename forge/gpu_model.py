"""AMK Forge -- the serializable hardware model (design-C).

Pure Python + stdlib, torch-free at module scope, so a measured GPU profile can be read,
diffed and versioned on a GPU-less box exactly like :class:`schedule.ir.GpuTarget`. Composes
a ``GpuTarget`` and adds:

  * :class:`SaturationCurve` -- the measured achieved-bandwidth-vs-occupancy curve, fitted to a
    hyperbolic ``peak * W / (W + Wh)``. The shape that says "how many warps/SM saturate HBM."
  * :class:`LatencyModel` -- the Little's-law model (HBM dependent-load latency + bytes-in-flight
    per warp). Used to cross-check the curve knee AND to extrapolate to GPUs we cannot run on.
  * :class:`OccupancyLimits` -- the LIVE device occupancy ceilings (num_sms, threads/SM, smem/SM,
    regs/SM, max blocks/SM). Live GB203 reports 1536 threads/SM (48 warps), the static
    ``rtx5090`` record says 2048 (64 warps); trusting the record inflates the knee, so we keep
    the live value and record the drift.
  * :class:`KernelFootprint` -- a kernel's register/smem/threads footprint, mirroring
    ``vm_ext.kernel_attributes()`` so the real megakernel's occupancy can be attributed.

JSON load/save uses the same additive-compatible ``_filter_known`` pattern as
``MegakernelProgram.from_dict`` so newer writers never break older readers.

The analytic ``min(warp, reg, smem, hw-block)`` occupancy limiter here is used ONLY for
counterfactual "what blocks/SM if I cut regs/smem to X" answers; a driver value
(``cudaOccupancyMaxActiveBlocksPerMultiprocessor``) is ALWAYS ground truth when present.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from schedule.ir import GpuTarget, TARGETS  # torch-free; the retargeting record

WARP = 32  # threads per warp (every NVIDIA GPU to date)


# ======================================================================================
# helpers (mirror schedule/ir patterns)
# ======================================================================================
def _round_up(a: int, g: int) -> int:
    """Round ``a`` up to the next multiple of granularity ``g`` (>=1)."""
    if g <= 1:
        return int(a)
    return ((int(a) + g - 1) // g) * g


def _filter_known(cls, d: dict[str, Any]) -> dict[str, Any]:
    """Additive-compatibility: drop keys a newer writer added that this dataclass lacks."""
    known = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in known}


def fit_hyperbolic(samples: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least-squares fit of a saturating hyperbola ``gbps = peak * W / (W + Wh)`` to measured
    ``(warps_per_sm, gbps)`` samples. Returns ``(peak, warps_half, r2)``.

    Pure Python (no numpy/scipy): grid-search the single non-linear parameter ``Wh`` and solve
    the linear ``peak`` in closed form for each candidate (for fixed Wh, ``r_i = W_i/(W_i+Wh)``
    and ``peak = sum(g_i r_i)/sum(r_i^2)`` minimises SSE). r2 is reported on the ORIGINAL GB/s
    scale (not a linearised 1/gbps scale, which would over-weight the low-occupancy samples)."""
    pts = [(float(w), float(g)) for w, g in samples if w > 0 and g > 0]
    if len(pts) < 3:
        # not enough to fit; degenerate model pinned to the max sample
        gmax = max((g for _, g in pts), default=0.0)
        return (gmax, 1.0, 0.0)
    gs = [g for _, g in pts]
    gbar = sum(gs) / len(gs)
    sst = sum((g - gbar) ** 2 for g in gs) or 1e-12

    best = (float("inf"), 0.0, 0.0)  # (sse, peak, Wh)
    wmax = max(w for w, _ in pts)
    # Wh spans well below the smallest spacing to a few x the largest occupancy probed.
    lo, hi = 1e-3, max(8.0 * wmax, 64.0)
    steps = 4000
    for k in range(steps + 1):
        # geometric sweep: dense near small Wh (where the knee lives), sparse at large Wh
        wh = lo * (hi / lo) ** (k / steps)
        num = den = 0.0
        for w, g in pts:
            r = w / (w + wh)
            num += g * r
            den += r * r
        if den <= 0:
            continue
        peak = num / den
        sse = sum((g - peak * (w / (w + wh))) ** 2 for w, g in pts)
        if sse < best[0]:
            best = (sse, peak, wh)
    sse, peak, wh = best
    r2 = 1.0 - sse / sst
    return (float(peak), float(wh), float(r2))


# ======================================================================================
# design-C dataclasses
# ======================================================================================
@dataclass
class SaturationCurve:
    """Achieved HBM bandwidth as a function of occupancy (warps/SM).

    The smooth interpolant is a saturating hyperbola ``gbps(W) = peak_bw_gbs * W/(W + warps_half)``
    used for ``predict_bw`` and the Little's-law cross-check. The KNEE, however, is the SPEC's
    literal model-free quantity: the min warps/SM whose MEASURED median GB/s first reaches
    ``knee_frac`` of the ACHIEVED peak (``achieved_peak_gbs``, the best measured sample) -- NOT the
    hyperbola's extrapolated asymptote, which for a fast-saturating GPU can sit above every sample
    and push an unphysical knee past the occupancy ceiling. Both knees are exposed; the empirical
    one is the headline."""

    peak_bw_gbs: float                 # hyperbola asymptote (fitted; may exceed any sample)
    warps_half: float
    knee_frac: float = 0.95
    model: str = "hyperbolic"
    samples: list[tuple[float, float]] = field(default_factory=list)  # (warps_per_sm, gbps)
    r2: float = 0.0
    achieved_peak_gbs: float = 0.0     # max measured sample (the honest peak denominator)

    def __post_init__(self):
        if self.achieved_peak_gbs <= 0 and self.samples:
            self.achieved_peak_gbs = max(g for _, g in self.samples)

    # ---- hyperbolic interpolant (smooth predict_bw) -------------------------------
    def achieved_gbs(self, warps_per_sm: float) -> float:
        """Modelled achieved GB/s at a given occupancy (warps/SM), hyperbolic interpolant."""
        W = max(0.0, float(warps_per_sm))
        if W <= 0:
            return 0.0
        return self.peak_bw_gbs * W / (W + self.warps_half)

    def warps_for_fraction(self, frac: float) -> float:
        """Occupancy at which the HYPERBOLA reaches ``frac`` of its asymptote (model knee)."""
        f = min(max(float(frac), 0.0), 0.999999)
        return f * self.warps_half / (1.0 - f)

    @property
    def warps_knee_model(self) -> float:
        """Hyperbolic (model) 95%-of-asymptote knee. Can exceed the occupancy ceiling."""
        return self.warps_for_fraction(self.knee_frac)

    # ---- empirical, model-free knee (the SPEC definition; headline) ----------------
    def per_warp_medians(self) -> list[tuple[float, float]]:
        import statistics as _st
        by_w: dict[int, list[float]] = {}
        for w, g in self.samples:
            by_w.setdefault(int(round(w)), []).append(g)
        return sorted((float(w), float(_st.median(gs))) for w, gs in by_w.items())

    @property
    def peak(self) -> float:
        """The honest peak denominator: the achieved max sample (falls back to fitted asymptote)."""
        return self.achieved_peak_gbs or self.peak_bw_gbs

    @property
    def warps_knee(self) -> float:
        """Headline knee: min warps/SM whose measured median first reaches ``knee_frac`` of the
        ACHIEVED peak. Linearly interpolates between the two bracketing samples. Falls back to the
        model knee if there are no samples."""
        med = self.per_warp_medians()
        if not med:
            return self.warps_knee_model
        thr = self.knee_frac * self.peak
        prev = None
        for w, g in med:
            if g >= thr:
                if prev is None:
                    return w
                pw, pg = prev
                if g == pg:
                    return w
                return pw + (w - pw) * (thr - pg) / (g - pg)  # linear interp to the crossing
            prev = (w, g)
        return med[-1][0]  # never reached frac of peak within the probed range

    def fraction_at(self, warps_per_sm: float) -> float:
        """Modelled %-of-achieved-peak (0..1) at a given occupancy (hyperbolic interpolant)."""
        if self.peak <= 0:
            return 0.0
        return self.achieved_gbs(warps_per_sm) / self.peak

    def empirical_fraction_at(self, warps_per_sm: int) -> float:
        """Measured %-of-achieved-peak at an exact probed occupancy (median of that warp count)."""
        med = dict(self.per_warp_medians())
        g = med.get(int(round(warps_per_sm)))
        if g is None or self.peak <= 0:
            return float("nan")
        return g / self.peak

    def measured_gbs(self, warps_per_sm: float) -> float:
        """MEASURED GB/s at an occupancy by linear interpolation of the per-warp medians (clamped
        to the probed range). Use this -- not the hyperbola -- when reporting the achieved bandwidth
        at a real occupancy point, since the hyperbola systematically under-fits near the knee."""
        med = self.per_warp_medians()
        if not med:
            return self.achieved_gbs(warps_per_sm)
        W = float(warps_per_sm)
        if W <= med[0][0]:
            return med[0][1]
        if W >= med[-1][0]:
            return med[-1][1]
        for (w0, g0), (w1, g1) in zip(med, med[1:]):
            if w0 <= W <= w1:
                return g0 + (g1 - g0) * (W - w0) / (w1 - w0) if w1 > w0 else g0
        return med[-1][1]


@dataclass
class LatencyModel:
    """Little's-law memory model. ``bytes_in_flight_to_saturate = peak_bw * hbm_latency``; the
    occupancy needed to keep that many bytes in flight is ``bytes_in_flight / bytes_per_warp``.

    ``bytes_in_flight_per_warp`` is calibrated on the measured GPU (so ``predicted_knee`` matches
    the measured curve knee) and then assumed roughly hardware-invariant, which is what lets the
    SAME probe extrapolate the knee to a GPU we never ran on (the warp-starvation prediction)."""

    hbm_latency_ns: float
    bytes_in_flight_per_warp: float

    def bytes_in_flight(self, peak_bw_gbs: float) -> float:
        """Total bytes that must be outstanding GPU-wide to saturate ``peak_bw_gbs`` (Little's law)."""
        return peak_bw_gbs * 1e9 * (self.hbm_latency_ns * 1e-9)

    def predicted_knee(self, peak_bw_gbs: float, num_sms: int) -> float:
        """Warps/SM needed to saturate, by Little's law: per-SM (bytes-in-flight) / bytes_per_warp.
        ``(peak_bw_gbs*1e9 / num_sms) * latency_s / bytes_in_flight_per_warp``."""
        if self.bytes_in_flight_per_warp <= 0 or num_sms <= 0:
            return float("nan")
        per_sm_bw = peak_bw_gbs * 1e9 / num_sms
        per_sm_inflight = per_sm_bw * (self.hbm_latency_ns * 1e-9)
        return per_sm_inflight / self.bytes_in_flight_per_warp

    def predicted_fraction(self, warps_per_sm: float, peak_bw_gbs: float, num_sms: int) -> float:
        """Linear-then-clamped Little's-law %-of-peak at a fixed occupancy (the un-saturated
        regime grows linearly with warps until it hits 1.0). Used for cross-GPU extrapolation."""
        knee_lin = self.predicted_knee(peak_bw_gbs, num_sms)  # warps to reach 100% (linear model)
        if not math.isfinite(knee_lin) or knee_lin <= 0:
            return float("nan")
        return min(1.0, float(warps_per_sm) / knee_lin)


@dataclass
class OccupancyLimits:
    """Live device occupancy ceilings, read from the driver (NOT the static GpuTarget record).
    These bound the saturation sweep's x-axis and the analytic limiter."""

    max_warps_per_sm: int
    max_blocks_per_sm: int
    regs_per_sm: int
    smem_bytes_per_sm: int
    reg_alloc_gran: int = 256
    smem_alloc_gran: int = 128
    reserved_smem_per_block: int = 1024

    @property
    def max_threads_per_sm(self) -> int:
        return self.max_warps_per_sm * WARP

    @classmethod
    def from_live(cls) -> "OccupancyLimits":
        """Read the live device occupancy ceilings via torch's cudaDeviceProp (lazy torch import,
        so the module stays torch-free for the GPU-less readers). ``max_blocks_per_sm`` is not in
        torch's props on all versions; fall back to a sm-arch default when absent."""
        import torch  # lazy: keep module import GPU-free
        if not torch.cuda.is_available():
            raise RuntimeError("OccupancyLimits.from_live requires a CUDA GPU")
        p = torch.cuda.get_device_properties(0)
        max_threads = int(getattr(p, "max_threads_per_multi_processor", 0)) or 2048
        smem_sm = int(getattr(p, "shared_memory_per_multiprocessor", 0)) or \
            int(getattr(p, "max_shared_memory_per_multiprocessor", 0)) or 102400
        regs_sm = int(getattr(p, "regs_per_multiprocessor", 0)) or 65536
        # max blocks/SM: torch may not expose it; default by arch (Blackwell/Ada=32, older=16/32)
        mbps = int(getattr(p, "max_blocks_per_multi_processor", 0))
        if mbps <= 0:
            mbps = 32 if (p.major, p.minor) >= (8, 0) else 16
        return cls(max_warps_per_sm=max_threads // WARP, max_blocks_per_sm=mbps,
                   regs_per_sm=regs_sm, smem_bytes_per_sm=smem_sm)


@dataclass
class KernelFootprint:
    """A kernel's per-launch resource footprint, mirroring ``vm_ext.kernel_attributes()`` output.
    Used to attribute (and counterfactually re-derive) a kernel's blocks/SM occupancy."""

    name: str
    num_regs: int
    static_smem_bytes: int
    dynamic_smem_bytes: int
    threads_per_block: int
    measured_blocks_per_sm: int = 0  # driver truth when known (0 = not attributed)

    @property
    def warps_per_block(self) -> int:
        return max(1, (self.threads_per_block + WARP - 1) // WARP)

    @classmethod
    def from_attrs(cls, d: dict[str, Any], name: str = "amk_megakernel",
                   threads_per_block: int = 256, dynamic_smem_bytes: int = 0) -> "KernelFootprint":
        """Wrap a ``vm_ext.kernel_attributes(tpb, dyn_smem)`` dict (driver truth)."""
        return cls(
            name=name,
            num_regs=int(d.get("num_regs", 0)),
            static_smem_bytes=int(d.get("static_smem_bytes", 0)),
            dynamic_smem_bytes=int(dynamic_smem_bytes),
            threads_per_block=int(threads_per_block),
            measured_blocks_per_sm=int(d.get("blocks_per_sm", 0)),
        )


@dataclass
class GpuModel:
    """The composed, serializable hardware model: a GpuTarget + the measured saturation curve +
    the Little's-law latency model + live occupancy limits. ``predict_bw`` maps a kernel footprint
    and a blocks/SM choice to a modelled achieved bandwidth."""

    target: GpuTarget
    occ: OccupancyLimits
    curve: SaturationCurve
    latency: LatencyModel
    probe_access_pattern: str = "float4_stream"
    probe_ilp: int = 1
    measured_on: str = ""
    note: str = ""

    # ---- prediction ---------------------------------------------------------------
    def warps_per_sm(self, fp: KernelFootprint, blocks_per_sm: int) -> int:
        return int(blocks_per_sm) * fp.warps_per_block

    def predict_bw(self, fp: KernelFootprint, blocks_per_sm: int) -> float:
        """Modelled achieved GB/s for ``blocks_per_sm`` co-resident blocks of ``fp``."""
        return self.curve.achieved_gbs(self.warps_per_sm(fp, blocks_per_sm))

    def analytic_blocks_per_sm(self, fp: KernelFootprint) -> tuple[int, str, dict[str, int]]:
        """Counterfactual ``min(warp, reg, smem, hw-block)`` occupancy limiter with binder
        attribution. The driver value (``fp.measured_blocks_per_sm``) is ground truth when
        present; this is only for 'what blocks/SM if regs/smem were X' answers."""
        o = self.occ
        # by threads/warps
        by_warp = o.max_threads_per_sm // fp.threads_per_block if fp.threads_per_block else 0
        # by registers: regs/thread rounded to alloc granularity, x threads/block
        regs_per_thread = max(1, fp.num_regs)
        regs_per_block = _round_up(regs_per_thread * fp.threads_per_block, o.reg_alloc_gran)
        by_reg = o.regs_per_sm // regs_per_block if regs_per_block else 0
        # by smem: static+dynamic+reserved per block rounded to alloc granularity
        smem_per_block = _round_up(fp.static_smem_bytes + fp.dynamic_smem_bytes
                                   + o.reserved_smem_per_block, o.smem_alloc_gran)
        by_smem = o.smem_bytes_per_sm // smem_per_block if smem_per_block else 0
        by_hw = o.max_blocks_per_sm
        limits = {"warp": int(by_warp), "reg": int(by_reg), "smem": int(by_smem), "hw": int(by_hw)}
        blocks = max(0, min(limits.values()))
        binder = min(limits, key=lambda k: limits[k])
        return blocks, binder, limits

    # ---- serialization (GpuTarget-style nested asdict + _filter_known) -------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "forge.GpuModel/0.1",
            "target": asdict(self.target) if self.target else None,
            "occ": asdict(self.occ),
            "curve": {**asdict(self.curve),
                      "samples": [[float(w), float(g)] for w, g in self.curve.samples]},
            "latency": asdict(self.latency),
            "probe_access_pattern": self.probe_access_pattern,
            "probe_ilp": self.probe_ilp,
            "measured_on": self.measured_on,
            "note": self.note,
            # convenience read-outs (derived; readers can ignore)
            "derived": {
                "warps_knee_empirical": round(self.curve.warps_knee, 3),
                "warps_knee_model_hyperbolic": round(self.curve.warps_knee_model, 3),
                "achieved_peak_gbs": round(self.curve.peak, 1),
                "fitted_asymptote_gbs": round(self.curve.peak_bw_gbs, 1),
                "r2": round(self.curve.r2, 4),
                "mlp_bytes_total": round(self.latency.bytes_in_flight(self.curve.peak), 1),
                "max_warps_per_sm": self.occ.max_warps_per_sm,
                "frac_at_8_warps_empirical": round(self.curve.empirical_fraction_at(8), 4),
                "frac_at_8_warps_model": round(self.curve.fraction_at(8), 4),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GpuModel":
        target = (GpuTarget(**_filter_known(GpuTarget, d["target"])) if d.get("target") else None)
        occ = OccupancyLimits(**_filter_known(OccupancyLimits, d.get("occ", {})))
        cd = dict(d.get("curve", {}))
        samples = [(float(w), float(g)) for w, g in cd.get("samples", [])]
        cd = _filter_known(SaturationCurve, cd)
        cd["samples"] = samples
        curve = SaturationCurve(**cd)
        latency = LatencyModel(**_filter_known(LatencyModel, d.get("latency", {})))
        known = _filter_known(cls, d)
        known.update({"target": target, "occ": occ, "curve": curve, "latency": latency})
        return cls(**known)

    @classmethod
    def from_json(cls, s: str) -> "GpuModel":
        return cls.from_dict(json.loads(s))

    @classmethod
    def load(cls, path: str) -> "GpuModel":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(f.read())


__all__ = [
    "WARP", "SaturationCurve", "LatencyModel", "OccupancyLimits", "KernelFootprint", "GpuModel",
    "fit_hyperbolic", "TARGETS",
]
