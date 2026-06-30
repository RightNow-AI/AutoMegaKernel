"""AMK Forge -- the saturate/starve advisor (design-C).

Turns a measured :class:`forge.GpuModel` + a kernel footprint into a SATURATED/STARVED verdict
and a cheapest-first, cooperative-aware action list mapped to concrete loader knobs.

Module scope is torch-free (the ``advise`` path is pure arithmetic on the model). The driver
attribution helpers (``attribute_megakernel_occupancy``, ``cooperative_ceiling_ok``) lazily use
the EXISTING ``vm_ext`` extension and are only called by the runnable scratch driver, never at
import; they degrade gracefully if the megakernel extension is not built.

The cooperative-grid constraint is special-cased: for the persistent megakernel ``grid.sync``
requires every block to be co-resident (``max_coresident_blocks >= num_sms``), so raising
``threads_per_block`` past the point where occupancy drops below 1 block/SM is ILLEGAL -- the
advisor refuses to recommend it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gpu_model import GpuModel, KernelFootprint


@dataclass
class Action:
    """One concrete, loader-actionable change, with its modelled effect."""

    kind: str                       # "raise_blocks_per_sm" | "cut_dynamic_smem" | ...
    target: dict[str, Any]          # the knob change, e.g. {"blocks_per_sm": 2}
    expected_blocks_per_sm: int
    expected_warps_per_sm: int
    expected_bw_gbs: float
    expected_speedup: float
    feasible: bool
    maps_to_knob: str               # the ScheduleConfig / MegakernelVM knob it sets
    constraint: str = ""            # why infeasible / what it costs


@dataclass
class Advice:
    """The verdict for one kernel footprint on one GPU."""

    saturated: bool
    binder: str
    current_warps: int
    predicted_bw_gbs: float
    pct_peak: float
    knee_warps: float
    headroom_speedup: float
    actions: list[Action] = field(default_factory=list)
    verdict: str = ""

    def report(self) -> str:
        head = "SATURATED" if self.saturated else "STARVED"
        lines = [f"[{head}] {self.verdict}",
                 f"  current {self.current_warps} warps/SM -> "
                 f"{self.predicted_bw_gbs:.0f} GB/s ({self.pct_peak*100:.1f}% of achieved peak); "
                 f"knee at {self.knee_warps:.1f} warps/SM; headroom x{self.headroom_speedup:.2f}"]
        for a in self.actions:
            tag = "DO" if a.feasible else "NO"
            lines.append(f"   [{tag}] {a.kind}: {a.target} -> {a.expected_warps_per_sm} warps/SM, "
                         f"{a.expected_bw_gbs:.0f} GB/s (x{a.expected_speedup:.2f}) "
                         f"[{a.maps_to_knob}] {a.constraint}".rstrip())
        return "\n".join(lines)


class KernelAdvisor:
    """Maps a measured GpuModel + a kernel footprint to a saturate/starve verdict + actions."""

    def advise(self, gm: GpuModel, fp: KernelFootprint, cooperative: bool = True,
               num_sms: int | None = None, current_blocks_per_sm: int = 1) -> Advice:
        num_sms = int(num_sms or (gm.target.num_sms if gm.target else 1))
        knee = gm.curve.warps_knee
        wpb = fp.warps_per_block
        # operating point = the ACTUAL launch (cooperative megakernel defaults to 1 block/SM);
        # the driver's measured_blocks_per_sm is the FEASIBILITY CEILING, not the operating point.
        blocks_now = int(current_blocks_per_sm)
        warps_now = blocks_now * wpb
        bw_now = gm.curve.measured_gbs(warps_now)  # measured (interp), not the under-fitting hyperbola
        peak = gm.curve.peak  # achieved peak (honest denominator)
        pct = (bw_now / peak) if peak > 0 else 0.0
        saturated = warps_now >= knee
        headroom = (gm.curve.measured_gbs(min(knee, gm.occ.max_warps_per_sm)) / bw_now
                    if bw_now > 0 else float("inf"))

        actions: list[Action] = []
        # occupancy ceiling for THIS footprint: min(by-warps, driver co-residency ceiling).
        max_blocks_warp = gm.occ.max_warps_per_sm // wpb if wpb else 0
        if fp.measured_blocks_per_sm:
            max_blocks_warp = min(max_blocks_warp, fp.measured_blocks_per_sm)

        if not saturated:
            # (a) cheapest: raise co-resident blocks/SM (MegakernelVM(blocks_per_sm=N)).
            for nb in range(blocks_now + 1, max_blocks_warp + 1):
                warps = nb * wpb
                bw = gm.curve.measured_gbs(warps)
                coop_ok = (not cooperative) or (nb * num_sms <= max_blocks_warp * num_sms)
                actions.append(Action(
                    kind="raise_blocks_per_sm", target={"blocks_per_sm": nb},
                    expected_blocks_per_sm=nb, expected_warps_per_sm=warps,
                    expected_bw_gbs=bw, expected_speedup=(bw / bw_now if bw_now else 1.0),
                    feasible=coop_ok, maps_to_knob="MegakernelVM(blocks_per_sm=N)",
                    constraint="" if coop_ok else "co-residency would drop below num_sms"))
                if warps >= knee:
                    break
            # (b) lighten the footprint (cut dynamic smem / registers) to allow more blocks.
            actions.append(Action(
                kind="cut_dynamic_smem", target={"smem_bytes_per_block": 0},
                expected_blocks_per_sm=min(max_blocks_warp, blocks_now + 1),
                expected_warps_per_sm=min(max_blocks_warp, blocks_now + 1) * wpb,
                expected_bw_gbs=gm.curve.achieved_gbs(min(max_blocks_warp, blocks_now + 1) * wpb),
                expected_speedup=1.0, feasible=True, maps_to_knob="ScheduleConfig.smem_bytes_per_block",
                constraint="reg+smem co-limit: only helps if smem (not registers) is the binder"))
            # (c) raise threads/block -- ILLEGAL for the cooperative persistent megakernel.
            if cooperative:
                actions.append(Action(
                    kind="raise_threads_per_block", target={"threads_per_block": ">256"},
                    expected_blocks_per_sm=0, expected_warps_per_sm=0, expected_bw_gbs=0.0,
                    expected_speedup=0.0, feasible=False,
                    maps_to_knob="ScheduleConfig.threads_per_block",
                    constraint="ILLEGAL for cooperative megakernel: tpb>256 -> "
                               "max_coresident<num_sms breaks grid.sync"))
            else:
                tpb2 = fp.threads_per_block * 2
                w2 = (gm.occ.max_warps_per_sm)  # raising tpb fills threads to the cap
                actions.append(Action(
                    kind="raise_threads_per_block", target={"threads_per_block": tpb2},
                    expected_blocks_per_sm=max(1, gm.occ.max_warps_per_sm // max(1, tpb2 // 32)),
                    expected_warps_per_sm=w2, expected_bw_gbs=gm.curve.achieved_gbs(w2),
                    expected_speedup=(gm.curve.achieved_gbs(w2) / bw_now if bw_now else 1.0),
                    feasible=True, maps_to_knob="ScheduleConfig.threads_per_block",
                    constraint="standalone (non-cooperative) kernel only"))

        binder = "occupancy<knee" if not saturated else "bandwidth (at/above knee)"
        verdict = (f"{warps_now} warps/SM >= {knee:.1f} knee: HBM saturated, occupancy is not the "
                   f"bottleneck" if saturated else
                   f"{warps_now} warps/SM < {knee:.1f} knee: WARP-STARVED, "
                   f"raise occupancy or ILP to saturate HBM")
        return Advice(saturated=saturated, binder=binder, current_warps=warps_now,
                      predicted_bw_gbs=bw_now, pct_peak=pct, knee_warps=knee,
                      headroom_speedup=headroom, actions=actions, verdict=verdict)


# ======================================================================================
# driver-truth attribution (lazy vm_ext; only the scratch runner calls these)
# ======================================================================================
def attribute_megakernel_occupancy(ext: Any, tpb: int = 256, dyn_smem: int = 0,
                                    gm: GpuModel | None = None) -> KernelFootprint:
    """Driver-truth footprint of the REAL megakernel via ``ext.kernel_attributes(tpb, dyn_smem)``.
    If a GpuModel is given, assert its analytic blocks/SM matches the driver and warn on drift."""
    d = ext.kernel_attributes(int(tpb), int(dyn_smem))
    fp = KernelFootprint.from_attrs(dict(d), threads_per_block=tpb, dynamic_smem_bytes=dyn_smem)
    if gm is not None:
        analytic, binder, limits = gm.analytic_blocks_per_sm(fp)
        if fp.measured_blocks_per_sm and analytic != fp.measured_blocks_per_sm:
            fp_note = (f"WARN analytic blocks/SM {analytic} ({binder}-bound {limits}) != driver "
                       f"{fp.measured_blocks_per_sm}; trusting driver")
            print("  [attribute] " + fp_note)
    return fp


def cooperative_ceiling_ok(ext: Any, tpb: int, dyn_smem: int, num_sms: int) -> bool:
    """The grid.sync floor: ``max_coresident_blocks(tpb, dyn_smem) >= num_sms`` (pure driver query,
    no launch, zero hang risk)."""
    return int(ext.max_coresident_blocks(int(tpb), int(dyn_smem))) >= int(num_sms)


__all__ = ["Action", "Advice", "KernelAdvisor",
           "attribute_megakernel_occupancy", "cooperative_ceiling_ok"]
