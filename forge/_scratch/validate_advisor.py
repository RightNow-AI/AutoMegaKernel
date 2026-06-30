"""Validate the forge KernelAdvisor as a GENERATOR ORACLE on the live RTX 5090.

Three independent validations, all driven off the COMMITTED model JSON
(forge/gpu_model_rtx5090.json) + forge.kernel_advisor.KernelAdvisor, with FRESH
on-GPU measurements:

  (1) PREDICT-vs-MEASURE: advisor predicts achieved BW at several occupancy points
      (incl. the AMK 8 warps/SM default) from the stored saturation curve; we then
      MEASURE each point fresh in ONE interleaved round-robin window (drift-robust).
  (2) GAP-CLOSING DEMO: an 8-warp/SM streaming kernel (AMK's default analog) is given
      the advisor's recommended blocks_per_sm=2 (16 warps/SM); we measure it climb
      toward the knee. Correctness of the occupancy lever is gated on copy_k (the
      higher-occupancy launch must still produce dst==src bit-exactly).
  (3) H100 EXTRAPOLATION: the same Little's-law model + TARGETS['h100'] params predict
      the H100 knee and the %-of-peak an 8-warp/SM kernel reaches.

The laptop's SW power cap drifts the ABSOLUTE GB/s 20-34% run-to-run; the curve SHAPE
is drift-robust (interleaved). So we report BOTH the raw-absolute predict-vs-measure
AND a drift-corrected pair anchored to the freshly-measured in-window peak (the honest,
transferable comparison). Run:
  ./.venv/Scripts/python.exe forge/_scratch/validate_advisor.py
"""
from __future__ import annotations

import json
import statistics
import sys

sys.path.insert(0, "C:/Users/jaber/RightNow-Full/AutoMegaKernel")
from vm.loader import _ensure_msvc_on_path  # noqa: E402

_ensure_msvc_on_path()
import torch  # noqa: E402

from eval.peak_bandwidth import _clock_state  # noqa: E402
from forge.gpu_model import WARP, GpuModel, KernelFootprint, LatencyModel  # noqa: E402
from forge.gpu_profiler import (  # noqa: E402
    _sweep_interleaved,
    build_probe,
    cap_smem_for_blocks,
    dev_attrs,
)
from forge.kernel_advisor import KernelAdvisor  # noqa: E402
from schedule.ir import TARGETS  # noqa: E402

MODEL_PATH = "C:/Users/jaber/RightNow-Full/AutoMegaKernel/forge/gpu_model_rtx5090.json"


def find_point(ext, attrs, target_warps, prefer_tpb=None):
    """Return a DRIVER-CONFIRMED (tpb, blocks, smem, wpb) launch that yields exactly
    ``target_warps`` warps/SM, or None. ``prefer_tpb`` is tried first (to mirror a
    specific block shape, e.g. the megakernel's 256)."""
    optin = attrs["smem_optin"]
    order = ([prefer_tpb] if prefer_tpb else []) + [128, 256, 96, 64, 192, 160, 224]
    seen = set()
    for tpb in order:
        if tpb in seen:
            continue
        seen.add(tpb)
        wpb = tpb // WARP
        if wpb == 0 or target_warps % wpb:
            continue
        b = target_warps // wpb
        if b < 1:
            continue
        if int(ext.occ(tpb, 0)) < b:           # natural occ already below target
            continue
        smem = cap_smem_for_blocks(ext, tpb, b, optin)
        if smem is None:
            continue
        if int(ext.occ(tpb, smem)) != b:        # driver must confirm exact occupancy
            continue
        return (tpb, b, smem, wpb)
    return None


def main() -> int:
    gm = GpuModel.load(MODEL_PATH)
    adv = KernelAdvisor()
    ext = build_probe(1)
    attrs = dev_attrs(ext)
    num_sms = attrs["num_sms"]
    optin = attrs["smem_optin"]
    ext.opt_in(optin)

    l2 = attrs["l2_bytes"]
    buf_mb = 768
    n_float4 = buf_mb * 1024 * 1024 // 16
    buf_bytes = n_float4 * 16
    assert buf_bytes >= 8 * l2, "buffer must exceed 8x L2 to stay HBM-bound"
    buf = torch.empty(n_float4 * 4, dtype=torch.float32, device="cuda").normal_()
    out = torch.zeros(1024, dtype=torch.float32, device="cuda")

    print(f"[validate] GPU attrs: num_sms={num_sms} smem/SM={attrs['smem_per_sm']} "
          f"max_threads/SM={attrs['max_threads_per_sm']} L2={l2/1e6:.0f}MB")
    print(f"[validate] clock state (cur_mem,max_mem,cur_sm,max_sm,throttle): {_clock_state()}")
    print(f"[validate] read_reduce natural occ: occ(256,0)={int(ext.occ(256,0))} "
          f"occ(128,0)={int(ext.occ(128,0))} occ(64,0)={int(ext.occ(64,0))}")
    print(f"[validate] stored model: achieved_peak={gm.curve.peak:.1f} GB/s "
          f"empirical_knee={gm.curve.warps_knee:.2f} w/SM  "
          f"hbm_lat={gm.latency.hbm_latency_ns:.1f}ns  "
          f"bytes_in_flight/warp={gm.latency.bytes_in_flight_per_warp:.1f}")

    # =================================================================================
    # (1) PREDICT vs MEASURE
    # =================================================================================
    targets = [4, 8, 12, 16, 24]
    pts = []
    for tw in targets:
        p = find_point(ext, attrs, tw)
        if p:
            pts.append((tw, *p))
        else:
            print(f"  [pvm] no driver-confirmed point for {tw} warps/SM")
    fns, meta = [], []
    for (tw, tpb, b, smem, wpb) in pts:
        grid = b * num_sms
        fns.append(lambda b_=buf, o=out, n=n_float4, g=grid, t=tpb, s=smem:
                   ext.launch(b_, o, n, g, t, s))
        meta.append((tw, tpb, b, smem))
    med_ms = _sweep_interleaved(fns, reps=120, warmup=12)
    measured = [(tw, buf_bytes / (ms * 1e-3) / 1e9) for (tw, *_), ms in zip(meta, med_ms)]
    fresh_peak = max(g for _, g in measured)          # in-window achieved peak (drift anchor)
    stored_peak = gm.curve.peak
    drift = fresh_peak / stored_peak

    print("\n================= (1) PREDICT vs MEASURE =================")
    print(f"  fresh in-window peak = {fresh_peak:.1f} GB/s  (stored boosted peak {stored_peak:.1f}; "
          f"drift x{drift:.3f} from the SW power cap)")
    print(f"  {'warps':>5} {'tpb':>4} {'blk':>3} | {'pred_abs':>8} {'pred_drift':>10} "
          f"{'measured':>8} | {'abs_err%':>8} {'shape_err%':>9}")
    pvm_rows = []
    for (tw, tpb, b, smem), (_, gbs) in zip(meta, measured):
        pred_abs = gm.curve.measured_gbs(tw)            # advisor-native prediction (stored absolute)
        pred_drift = (pred_abs / stored_peak) * fresh_peak  # drift-corrected (shape x in-window peak)
        abs_err = 100.0 * (gbs - pred_abs) / pred_abs
        shp_err = 100.0 * (gbs - pred_drift) / pred_drift
        flag = "  <- AMK 8w/SM" if tw == 8 else ""
        print(f"  {tw:5d} {tpb:4d} {b:3d} | {pred_abs:8.1f} {pred_drift:10.1f} "
              f"{gbs:8.1f} | {abs_err:+8.1f} {shp_err:+9.1f}{flag}")
        pvm_rows.append({"warps_per_sm": tw, "pred_abs": round(pred_abs, 1),
                         "pred_drift": round(pred_drift, 1), "measured": round(gbs, 1),
                         "abs_err_pct": round(abs_err, 1), "shape_err_pct": round(shp_err, 1)})
    max_shape_err = max(abs(r["shape_err_pct"]) for r in pvm_rows)
    print(f"  => max |shape_err| across points = {max_shape_err:.1f}% "
          f"(drift-robust validation; abs_err carries the uniform {(drift-1)*100:+.0f}% clock drift)")

    # =================================================================================
    # (2) GAP-CLOSING DEMO: blocks_per_sm 1 -> 2 (8 -> 16 warps/SM) on a streaming kernel
    # =================================================================================
    print("\n================= (2) GAP-CLOSING DEMO =================")
    # advisor verdict for the AMK 8-warp/SM footprint (tpb=256, 1 block/SM)
    fp8 = KernelFootprint(name="amk_8warp", num_regs=111, static_smem_bytes=400,
                          dynamic_smem_bytes=0, threads_per_block=256, measured_blocks_per_sm=2)
    advice = adv.advise(gm, fp8, cooperative=True, num_sms=num_sms, current_blocks_per_sm=1)
    print("  advisor verdict for AMK default (8 warps/SM):")
    for line in advice.report().splitlines():
        print("    " + line)
    rec = next((a for a in advice.actions
                if a.kind == "raise_blocks_per_sm" and a.target.get("blocks_per_sm") == 2), None)

    # apply it on the read_reduce streaming kernel: tpb=256, grid = 1*num_sms vs 2*num_sms.
    # natural occ(256,0) >= 2 means grid=2*num_sms runs 2 co-resident blocks/SM in one wave.
    nat256 = int(ext.occ(256, 0))
    tpb = 256
    if nat256 >= 2:
        g8, s8 = num_sms, 0                 # 1 block/SM (faithful AMK launch, no throttle)
        g16, s16 = 2 * num_sms, 0           # 2 blocks/SM (advisor's blocks_per_sm=2)
        method = "natural grid (occ(256,0)>=2)"
    else:                                   # fall back to smem-throttled exact occupancy
        p8 = find_point(ext, attrs, 8, prefer_tpb=256)
        p16 = find_point(ext, attrs, 16, prefer_tpb=256)
        g8, s8 = p8[1] * num_sms, p8[2]
        g16, s16 = p16[1] * num_sms, p16[2]
        method = "smem-throttled exact occupancy"
    fns = [lambda b_=buf, o=out, n=n_float4, g=g8, t=tpb, s=s8: ext.launch(b_, o, n, g, t, s),
           lambda b_=buf, o=out, n=n_float4, g=g16, t=tpb, s=s16: ext.launch(b_, o, n, g, t, s)]
    med = _sweep_interleaved(fns, reps=150, warmup=15)
    base_gbs = buf_bytes / (med[0] * 1e-3) / 1e9
    impr_gbs = buf_bytes / (med[1] * 1e-3) / 1e9
    speedup = impr_gbs / base_gbs

    # correctness gate: copy_k at the SAME two launch configs must produce dst==src bit-exactly
    nf = n_float4 * 4
    dst = torch.empty_like(buf)
    dst.zero_(); ext.copy(buf, dst, nf, g8, tpb); torch.cuda.synchronize()
    ok8 = torch.equal(dst, buf)
    dst.zero_(); ext.copy(buf, dst, nf, g16, tpb); torch.cuda.synchronize()
    ok16 = torch.equal(dst, buf)
    correct = bool(ok8 and ok16)

    print(f"  occupancy method: {method}")
    print(f"  baseline   8 warps/SM (1 block/SM):  {base_gbs:7.1f} GB/s  "
          f"({base_gbs/fresh_peak*100:.1f}% of in-window peak)")
    print(f"  improved  16 warps/SM (2 blocks/SM): {impr_gbs:7.1f} GB/s  "
          f"({impr_gbs/fresh_peak*100:.1f}% of in-window peak)")
    print(f"  measured speedup = x{speedup:.3f}  "
          f"(advisor predicted x{(rec.expected_speedup if rec else float('nan')):.3f})")
    print(f"  correctness (copy_k dst==src bit-exact): 1blk/SM={ok8} 2blk/SM={ok16} "
          f"-> occupancy lever PRESERVES output = {correct}")

    # =================================================================================
    # (3) H100 EXTRAPOLATION (model prediction; H100 not run locally)
    # =================================================================================
    print("\n================= (3) H100 EXTRAPOLATION =================")
    h = TARGETS["h100"]
    h_spec_peak = h.hbm_bandwidth_gbs          # 3350 spec (the SPEC's framing: 265/3350 ~ 8%)
    bpw = gm.latency.bytes_in_flight_per_warp  # 294.5 B, calibrated on the 5090
    measured_h100_amk_pct = 265.0 / h_spec_peak * 100.0
    print(f"  H100 params (TARGETS['h100']): peak={h_spec_peak:.0f} GB/s spec, {h.num_sms} SMs, "
          f"{h.smem_bytes_per_sm} B smem/SM")
    print(f"  bytes_in_flight/warp = {bpw:.1f} B (calibrated on 5090, assumed hw-invariant)")
    print(f"  {'lat_ns':>7} {'knee_w/SM':>10} {'8w_stream%':>11} {'8w_x3.5haircut%':>16}")
    h100_rows = []
    for lat in (327.0, 450.0, 600.0):
        hl = LatencyModel(hbm_latency_ns=lat, bytes_in_flight_per_warp=bpw)
        knee = hl.predicted_knee(h_spec_peak, h.num_sms)
        frac8 = hl.predicted_fraction(8, h_spec_peak, h.num_sms) * 100.0
        haircut = frac8 / 3.5            # measured ~3-4x in-megakernel register-frame haircut
        print(f"  {lat:7.0f} {knee:10.1f} {frac8:11.1f} {haircut:16.1f}")
        h100_rows.append({"lat_ns": lat, "knee": round(knee, 1), "stream8_pct": round(frac8, 1),
                          "haircut8_pct": round(haircut, 1)})
    print(f"  MEASURED AMK-on-H100 (prior, via Modal) = 265/3350 = {measured_h100_amk_pct:.1f}% of peak")
    print(f"  5090 empirical knee = {gm.curve.warps_knee:.1f} warps/SM "
          f"(8 warps/SM = {gm.curve.empirical_fraction_at(8)*100:.1f}% of peak, SATURATED-ish)")

    # machine-readable summary for the orchestrator
    summary = {
        "clock_state": _clock_state(),
        "fresh_peak_gbs": round(fresh_peak, 1),
        "stored_peak_gbs": round(stored_peak, 1),
        "drift": round(drift, 4),
        "predict_vs_measure": pvm_rows,
        "max_shape_err_pct": round(max_shape_err, 1),
        "demo": {"baseline_gbps": round(base_gbs, 1), "improved_gbps": round(impr_gbs, 1),
                 "speedup": round(speedup, 3), "correct": correct,
                 "advisor_pred_speedup": round(rec.expected_speedup, 3) if rec else None},
        "h100": h100_rows,
        "h100_measured_amk_pct": round(measured_h100_amk_pct, 1),
    }
    print("\nVALIDATE_SUMMARY_JSON " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
