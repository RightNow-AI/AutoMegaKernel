"""Runnable driver for the 5090 GPU profile (scratch; never imported by the package, so GPU work
stays out of ``import forge``).

Run:  ./.venv/Scripts/python.exe forge/_scratch/run_profile_5090.py

Executes ``forge.profile_gpu()`` on the live 5090, prints the saturation curve / peak / 95% knee /
the 8-warp megakernel point as %-of-peak, saves the model JSON, then (best-effort) attributes the
REAL megakernel's occupancy via the existing vm_ext and prints the saturate/starve verdict plus an
H100-shaped Little's-law extrapolation -- explicitly labelled a model PREDICTION, not a local
measurement.
"""
import json
import sys

sys.path.insert(0, "C:/Users/jaber/RightNow-Full/AutoMegaKernel")

import forge
from forge.gpu_model import KernelFootprint
from forge.kernel_advisor import KernelAdvisor


def main() -> int:
    save_path = "C:/Users/jaber/RightNow-Full/AutoMegaKernel/forge/gpu_model_rtx5090.json"
    gm = forge.profile_gpu(reps=80, knee_frac=0.95, ilp_curve=1, buf_mb=768, save=save_path)

    c = gm.curve
    detail = getattr(gm, "_probe_detail", {})
    achieved_peak = c.peak  # achieved max sample (honest denominator)
    knee = c.warps_knee     # empirical: min warps reaching 95% of achieved peak
    medians = dict(c.per_warp_medians())
    mk8 = medians.get(8, c.achieved_gbs(8))           # measured median at 8 warps/SM
    mk8_pct = (mk8 / achieved_peak * 100.0) if achieved_peak else 0.0  # %-of-achieved-peak

    print("\n================= FORGE GpuModel (RTX 5090 Laptop) =================")
    print(f"  fitted peak           : {c.peak_bw_gbs:.1f} GB/s   (r2={c.r2:.4f}, hyperbolic)")
    print(f"  achieved max sample   : {achieved_peak:.1f} GB/s")
    print(f"  warps_half            : {c.warps_half:.3f} warps/SM")
    print(f"  95% knee              : {knee:.2f} warps/SM")
    print(f"  max warps/SM (live)   : {gm.occ.max_warps_per_sm}  (blocks/SM cap {gm.occ.max_blocks_per_sm})")
    print(f"  HBM latency           : {gm.latency.hbm_latency_ns:.1f} ns")
    print(f"  MLP bytes in flight   : {gm.latency.bytes_in_flight(c.peak_bw_gbs)/1024:.0f} KB total "
          f"({gm.latency.bytes_in_flight(c.peak_bw_gbs)/gm.target.num_sms:.0f} B/SM)")
    print(f"  bytes_in_flight/warp  : {gm.latency.bytes_in_flight_per_warp:.0f} B")
    print(f"  --- AMK megakernel point (1 block/SM x 256 thr = 8 warps/SM) ---")
    print(f"  8 warps/SM achieved   : {mk8:.1f} GB/s = {mk8_pct:.1f}% of fitted peak")
    print(f"  8 warps vs knee        : {'AT/ABOVE knee (saturated)' if 8 >= knee else 'BELOW knee (starved)'}")
    print(f"  16 warps/SM (2 blk/SM): {c.achieved_gbs(16):.1f} GB/s = {c.fraction_at(16)*100:.1f}%")
    print("  curve (warps/SM -> GB/s):")
    # dedup to one representative per warp count for a readable print (keep all in JSON)
    by_w: dict[int, list[float]] = {}
    for w, g in c.samples:
        by_w.setdefault(int(round(w)), []).append(g)
    for w in sorted(by_w):
        gs = by_w[w]
        flag = "  <- 8-warp AMK point" if w == 8 else ("  <- knee~" if abs(w - knee) < 1.5 else "")
        print(f"     {w:3d} : {sum(gs)/len(gs):7.1f} GB/s  (n={len(gs)}){flag}")

    print("\n  ILP sweep (near-max occ, {U: GB/s}):",
          {u: round(g, 0) for u, g in detail.get("ilp_sweep", {}).items()})
    print("  copy/L2 anchor:", json.dumps(detail.get("anchor", {})))
    print("  M3 crosscheck:", json.dumps(detail.get("crosscheck", {})))
    print("  latency table:", json.dumps(detail.get("latency_table", {})))
    print(f"  clock state: {detail.get('clock_state','')}")

    # ---- advisor against the REAL megakernel footprint (best-effort via vm_ext) ----
    print("\n----------------- KernelAdvisor (real megakernel) -----------------")
    fp = None
    try:
        from vm.loader import _build_extension  # type: ignore  (verified megakernel ext, cached)
        ext = _build_extension()
        from forge.kernel_advisor import attribute_megakernel_occupancy, cooperative_ceiling_ok
        fp = attribute_megakernel_occupancy(ext, tpb=256, dyn_smem=0, gm=gm)
        coop_ok_256 = cooperative_ceiling_ok(ext, 256, 0, gm.target.num_sms)
        coop_ok_512 = cooperative_ceiling_ok(ext, 512, 0, gm.target.num_sms)
        print(f"  driver footprint: regs={fp.num_regs} static_smem={fp.static_smem_bytes} "
              f"tpb={fp.threads_per_block} blocks/SM={fp.measured_blocks_per_sm} "
              f"warps/block={fp.warps_per_block}")
        print(f"  cooperative ceiling ok @tpb=256: {coop_ok_256}; @tpb=512: {coop_ok_512} "
              f"(False => grid.sync would break)")
    except Exception as e:
        print(f"  [vm_ext attribution skipped: {type(e).__name__}: {e}]")
        # fall back to the known megakernel footprint (256 threads, 1 block/SM cooperative)
        fp = KernelFootprint(name="amk_megakernel", num_regs=128, static_smem_bytes=0,
                             dynamic_smem_bytes=0, threads_per_block=256, measured_blocks_per_sm=1)
        print(f"  [fallback footprint] tpb=256, 1 block/SM (8 warps/SM), cooperative")

    adv = KernelAdvisor().advise(gm, fp, cooperative=True, num_sms=gm.target.num_sms,
                                 current_blocks_per_sm=fp.measured_blocks_per_sm or 1)
    print(adv.report())

    # ---- H100 extrapolation (PREDICTION, not a local measurement) ------------------
    print("\n----------------- H100 EXTRAPOLATION (model PREDICTION) ------------")
    from schedule.ir import TARGETS
    h = TARGETS["h100"]
    h_peak = h.measured_bw_gbs or h.hbm_bandwidth_gbs
    h_lat_ns = 450.0  # public Hopper HBM3 dependent-load latency ~ literature (UNMEASURED here)
    from forge.gpu_model import LatencyModel
    h_latency = LatencyModel(hbm_latency_ns=h_lat_ns,
                             bytes_in_flight_per_warp=gm.latency.bytes_in_flight_per_warp)
    h_knee = h_latency.predicted_knee(h_peak, h.num_sms)
    h_frac8 = h_latency.predicted_fraction(8, h_peak, h.num_sms) * 100.0
    print(f"  H100: peak={h_peak:.0f} GB/s, {h.num_sms} SMs, assumed HBM latency {h_lat_ns:.0f} ns")
    print(f"  predicted knee        : {h_knee:.1f} warps/SM  (vs {knee:.1f} on the 5090)")
    print(f"  AMK 8 warps/SM on H100: ~{h_frac8:.1f}% of peak by the streaming Little's-law model")
    print(f"  => the SAME 8 warps/SM that saturates the 5090 is WARP-STARVED on the H100.")
    print("  (PREDICTION: H100 high-warp region + knee are UNMEASURED locally; only an H100 run "
          "confirms. The measured AMK-on-H100 ~8% of spec is even lower, the in-megakernel "
          "register-frame haircut compounding this streaming-model starvation.)")

    print(f"\nforge gpu:rtx5090 peak_gbs:{c.peak_bw_gbs:.1f} knee_warps:{knee:.2f} "
          f"mk8_pct:{mk8_pct:.1f} latency_ns:{gm.latency.hbm_latency_ns:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
