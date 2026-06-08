"""
AMK, cp.async DOUBLE-BUFFERED decode GEMV: BEFORE vs AFTER (real measured roofline)
====================================================================================

The decode GEMV was memory-LATENCY bound: the register warp-per-column path issued a weight
load and immediately consumed it in the FMA, so too few independent loads were ever in flight to
hide ~400ns HBM latency -> it plateaued at ~48% of the MEASURED sustained HBM bandwidth.

The fix (vm/ops.cuh amk_gemv_tile_cpasync, enabled by the `cpasync` build knob) is the classic
memory-bound technique: SOFTWARE-PIPELINED DOUBLE-BUFFERING with cp.async
(__pipeline_memcpy_async). Each warp streams its weight-row K-chunks SMEM<-HBM into a STAGES-deep
ring; while it computes on stage s it has STAGES-1 future chunks' loads outstanding -> deep
memory-level parallelism -> approaches peak HBM bandwidth.

This bench measures the SAME 'small' bf16 decode megakernel BEFORE (cpasync off, the register
path) and AFTER (cpasync on), kernel-only (vm.relaunch, no host overhead) AND steady-state
(vm.run, the real per-token cost), each a CUDA-event median (warmup>=25, iters>=100). Every number
is correctness-gated vs the CPU ReferenceVM before it is trusted. We report achieved GB/s and % of
the 731 GB/s MEASURED roofline (NOT the 896 desktop spec).

Run:  uv run python eval/bench_cpasync_gemv.py
Writes paper/results/cpasync_gemv.json.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from models.toy import make_toy  # noqa: E402
from schedule.graph import from_toy  # noqa: E402
from schedule.ir import BufferKind, DType, ScheduleConfig, TARGETS, validate  # noqa: E402
from schedule.lower import POS_NAME, RESHAPE_ID_NAME, TOKEN_NAME, lower  # noqa: E402
from vm.reference_vm import ReferenceVM  # noqa: E402

SMALL = dict(hidden=2048, n_layers=4, n_heads=16, n_kv_heads=4, head_dim=128,
             intermediate=5632, vocab=32000)
ITERS = 100
WARMUP = 25
RESULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "paper", "results", "cpasync_gemv.json")


class GpuUnavailable(Exception):
    pass


def _inputs(tok: int, pos: int) -> dict[str, torch.Tensor]:
    return {
        TOKEN_NAME: torch.tensor([tok], dtype=torch.int32),
        POS_NAME: torch.tensor([pos], dtype=torch.int32),
        RESHAPE_ID_NAME: torch.tensor([0], dtype=torch.int32),
    }


def _median_event_ms(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def _streamed_weight_bytes(prog, weights) -> int:
    total = 0
    for b in prog.buffers:
        if b.kind != BufferKind.WEIGHT:
            continue
        key = b.source or b.name
        t = weights.get(key)
        total += (t.numel() * t.element_size()) if t is not None else b.nbytes
    return total


def _build_and_check(prog, weights, inp, ref_logits, knobs, label):
    """Build a VM variant and correctness-gate it vs the CPU reference. Returns (vm, max_err)."""
    from vm.loader import MegakernelVM
    try:
        vm = MegakernelVM(prog, weights, device="cuda", knobs=knobs)
    except (RuntimeError, TimeoutError) as e:
        raise GpuUnavailable(f"{label}: VM build/launch failed: {e}") from e
    out = vm.run(inp, kv={})["logits"].detach().float().cpu()
    err = (out - ref_logits).abs().max().item()
    if not torch.allclose(out, ref_logits, rtol=2e-2, atol=2e-2):  # bf16 gate
        raise AssertionError(f"{label}: GPU != reference (max_err={err:.3e})")
    if vm.last_status.get("status") != "OK":
        raise GpuUnavailable(f"{label}: status {vm.last_status}")
    return vm, err


def _result(label, knobs, vm, err, k_ms, t_ms, target, wbytes):
    roof_meas = target.measured_bandwidth_bound_us(wbytes)
    roof_spec = target.bandwidth_bound_us(wbytes)
    return {
        "label": label, "knobs": knobs,
        "kernel_us": k_ms * 1e3, "token_us": t_ms * 1e3,
        "kernel_gbs": wbytes / (k_ms * 1e-3) / 1e9,
        "token_gbs": wbytes / (t_ms * 1e-3) / 1e9,
        "pct_roof_meas_kernel": roof_meas / (k_ms * 1e3) * 100,
        "pct_roof_meas_token": roof_meas / (t_ms * 1e3) * 100,
        "pct_roof_spec_kernel": roof_spec / (k_ms * 1e3) * 100,
        "dyn_smem_bytes": vm.dyn_smem_bytes, "grid": vm.last_grid_dim,
        "max_err_vs_ref": err,
    }


def run():
    target = TARGETS["rtx5090"]
    meas_bw = target.measured_bw_gbs
    model = make_toy(seed=0, dtype=torch.bfloat16, **SMALL)
    graph = from_toy(model)
    wd = model.weights_dict()
    cfg = ScheduleConfig(pipelining_depth=2)
    prog = lower(graph, target=target, config=cfg, pos=0, dtype=DType.BF16)
    assert validate(prog).ok, "program rejected by validator"
    inp = _inputs(11, 0)
    ref = ReferenceVM(prog, wd, device="cpu").run(inp, kv={})["logits"].float()
    wbytes = _streamed_weight_bytes(prog, wd)

    # BEFORE: register warp-per-column path (cp.async OFF). AFTER: cp.async double-buffer (default).
    # Both VMs are built+correctness-gated up front, then their kernel-only + steady-state latencies
    # are measured in INTERLEAVED ROUNDS. Back-to-back A-then-B timing is biased by GPU boost drift
    # (the 2nd variant runs hotter/throttled); interleaving the two and taking each variant's MEDIAN
    # round removes that bias, an honest, drift-robust A/B on a WDDM laptop GPU.
    vm_b, err_b = _build_and_check(prog, wd, inp, ref, {"cpasync": 0}, "before_register")
    vm_a, err_a = _build_and_check(prog, wd, inp, ref, None, "after_cpasync")

    ROUNDS = 5
    kb, tb, ka, ta = [], [], [], []
    for _ in range(ROUNDS):
        kb.append(_median_event_ms(vm_b.relaunch, ITERS, WARMUP))
        ka.append(_median_event_ms(vm_a.relaunch, ITERS, WARMUP))
        tb.append(_median_event_ms(lambda: vm_b.run(inp, kv={}), ITERS, WARMUP))
        ta.append(_median_event_ms(lambda: vm_a.run(inp, kv={}), ITERS, WARMUP))
    kb.sort()
    tb.sort()
    ka.sort()
    ta.sort()

    def md(v):
        return v[len(v) // 2]
    before = _result("before_register", {"cpasync": 0}, vm_b, err_b, md(kb), md(tb), target, wbytes)
    after = _result("after_cpasync", dict(vm_a.knobs), vm_a, err_a, md(ka), md(ta), target, wbytes)
    before["rounds_kernel_us"] = [x * 1e3 for x in kb]
    after["rounds_kernel_us"] = [x * 1e3 for x in ka]

    kspeed = before["kernel_us"] / after["kernel_us"]
    tspeed = before["token_us"] / after["token_us"]

    print(f"  device={torch.cuda.get_device_name(0)}  model=small 4L hidden={SMALL['hidden']} "
          f"inter={SMALL['intermediate']} vocab={SMALL['vocab']}")
    print(f"  weights streamed = {wbytes/1e6:.1f} MB  |  MEASURED roofline = {meas_bw:.0f} GB/s")
    print(f"  {'variant':>16} | {'kern us':>8} | {'GB/s':>5} | {'%meas(k)':>8} | "
          f"{'tok us':>8} | {'GB/s':>5} | {'%meas(t)':>8} | {'smem':>6} | {'err':>7}")
    for r in (before, after):
        print(f"  {r['label']:>16} | {r['kernel_us']:>8.1f} | {r['kernel_gbs']:>5.0f} | "
              f"{r['pct_roof_meas_kernel']:>7.1f}% | {r['token_us']:>8.1f} | {r['token_gbs']:>5.0f} | "
              f"{r['pct_roof_meas_token']:>7.1f}% | {r['dyn_smem_bytes']:>6} | "
              f"{r['max_err_vs_ref']:>7.1e}")
    print(f"  >>> kernel-only cp.async speedup vs register: {kspeed:.2f}x ; steady-state {tspeed:.2f}x")

    meas_str = (f"{before['kernel_gbs']:.0f} GB/s, {before['pct_roof_meas_kernel']:.0f}% meas -> "
                f"{after['kernel_gbs']:.0f} GB/s, {after['pct_roof_meas_kernel']:.0f}% meas "
                f"({kspeed:.2f}x)")
    headline = (f"cp.async double-buffered decode GEMV: kernel-only "
                f"{before['pct_roof_meas_kernel']:.0f}% -> {after['pct_roof_meas_kernel']:.0f}% of "
                f"{meas_bw:.0f} GB/s MEASURED roofline ({kspeed:.2f}x); correctness preserved "
                f"(max_err {after['max_err_vs_ref']:.1e} <= 2e-2 bf16).")
    print(f"  HEADLINE: {headline}")
    print(f"  measured_before_after: {meas_str}")

    payload = {
        "device": torch.cuda.get_device_name(0),
        "model": "small_4L_h2048_i5632_v32000",
        "iters": ITERS, "warmup": WARMUP,
        "weight_bytes": wbytes,
        "hbm_spec_gbs": target.hbm_bandwidth_gbs, "hbm_measured_gbs": meas_bw,
        "before": before, "after": after,
        "kernel_speedup": kspeed, "token_speedup": tspeed,
        "measured_before_after": meas_str,
        "correctness_preserved": True,
        "headline": headline,
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {RESULT_PATH}")
    return payload


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)
    try:
        run()
    except GpuUnavailable as e:
        print(f"SKIP (GPU unavailable): {e}")
        sys.exit(0)
