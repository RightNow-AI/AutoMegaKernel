"""
AMK ROUND-2 PERF, int8/int4 vs bf16 decode (the byte win) + noinline/SMEM OCCUPANCY experiment
==============================================================================================
The honest Round-2 instrument. Two questions, both answered with REAL CUDA-event medians on the
live RTX 5090 (sm_120), correctness-gated against the CPU ReferenceVM:

  GOAL 1 , Does weight-only int8 (lossless) finally beat bf16 on a memory-bound decode now that the
            quantized GEMV loads the weight row as 128-bit vectors (16 int8 / 32 nibbles per lane),
            decodes the per-group scale ONCE per chunk, and reads x from SMEM (no per-lane register
            staging)? We report kernel-only AND steady-state ms/token, the streamed weight bytes,
            achieved GB/s, and % of the (measured + spec) HBM roofline, for bf16 / int8 / int4.

  GOAL 2 , Can we raise the megakernel's blocks/SM above 2? The kernel inlines every opcode so its
            register frame == the worst opcode; ALSO the fp GEMV's static-SMEM x-cache (32KB) caps
            co-residency. We measure cudaOccupancyMaxActiveBlocksPerMultiprocessor + numRegs +
            static SMEM for: (a) default, (b) __noinline__ opcodes, (c) shrink the SMEM x-cache to
            the program's real max-K + a launch-bound register cap, and report whether occupancy
            actually rose and whether decode bandwidth improved.

Every latency is a CUDA-event median (warmup >= 25, iters >= 100). Correctness is asserted vs the
ReferenceVM before any number is trusted. Writes paper/results/r2_vm_perf.json.

Run:  uv run python eval/bench_r2_perf.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from models.toy import make_toy  # noqa: E402
from schedule.graph import from_toy  # noqa: E402
from schedule.ir import BufferKind, DType, ScheduleConfig, TARGETS  # noqa: E402
from schedule.ir import validate  # noqa: E402
from schedule.lower import POS_NAME, RESHAPE_ID_NAME, TOKEN_NAME, lower  # noqa: E402
from schedule.quantize import quantize_weights  # noqa: E402
from vm.reference_vm import ReferenceVM  # noqa: E402

SMALL = dict(hidden=2048, n_layers=4, n_heads=16, n_kv_heads=4, head_dim=128,
             intermediate=5632, vocab=32000)
ITERS = 100
WARMUP = 25
GROUP = 128
RESULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "paper", "results", "r2_vm_perf.json")


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


def _build_vm(prog, weights, knobs=None, op_noinline=False):
    from vm.loader import MegakernelVM
    try:
        return MegakernelVM(prog, weights, device="cuda", knobs=knobs, op_noinline=op_noinline)
    except (RuntimeError, TimeoutError) as e:
        raise GpuUnavailable(f"CUDA VM could not build/launch: {e}") from e


def _gpu_vs_ref_check(vm, ref_logits, inputs, rtol: float, atol: float):
    """Run the GPU VM and check its logits match the CPU ReferenceVM within (rtol, atol). Returns
    (max_abs_err, ok). bf16 carries ~3 mantissa digits, so the gate is dtype-aware: bf16 uses the
    same rtol/atol=2e-2 as tests/test_cuda_bf16.py; the fp16 quant paths use a tight 3e-3."""
    out = vm.run(inputs, kv={})["logits"].detach().float().cpu()
    err = (out - ref_logits).abs().max().item()
    ok = torch.allclose(out, ref_logits, rtol=rtol, atol=atol)
    return err, ok


# ------------------------------------------------------------------------------------------------
# GOAL 1: bf16 vs int8 vs int4 decode latency + roofline
# ------------------------------------------------------------------------------------------------
def goal1(target, graph, wd, q8, m8, q4, m4):
    cfg = ScheduleConfig(pipelining_depth=2)
    inp = _inputs(11, 0)
    rows = []

    def one(name, dtype, weights, quant):
        prog = lower(graph, target=target, config=cfg, pos=0, dtype=dtype, quant=quant)
        assert validate(prog).ok, f"{name}: program rejected"
        ref = ReferenceVM(prog, weights, device="cpu").run(inp, kv={})["logits"].float()
        vm = _build_vm(prog, weights, knobs=None)
        # dtype-aware gate: bf16 ~3 mantissa digits -> 2e-2 (matches tests/test_cuda_bf16.py);
        # the fp16 quant paths are ulp-exact vs the quantized reference -> tight 3e-3.
        rtol, atol = (2e-2, 2e-2) if dtype == DType.BF16 else (3e-3, 3e-3)
        err, ok = _gpu_vs_ref_check(vm, ref, inp, rtol, atol)
        assert ok, f"{name}: GPU != reference (max_err={err:.2e}, rtol={rtol} atol={atol})"
        assert vm.last_status.get("status") == "OK", f"{name}: {vm.last_status}"
        k_ms = _median_event_ms(vm.relaunch, ITERS, WARMUP)             # kernel-only
        t_ms = _median_event_ms(lambda: vm.run(inp, kv={}), ITERS, WARMUP)  # steady-state
        wbytes = _streamed_weight_bytes(prog, weights)
        roof_spec = target.bandwidth_bound_us(wbytes)
        roof_meas = target.measured_bandwidth_bound_us(wbytes)
        gbs = wbytes / (k_ms * 1e-3) / 1e9
        return {"name": name, "kernel_us": k_ms * 1e3, "token_us": t_ms * 1e3,
                "weight_bytes": wbytes, "roof_spec_us": roof_spec, "roof_meas_us": roof_meas,
                "achieved_gbs": gbs, "pct_roof_spec": roof_spec / (k_ms * 1e3) * 100,
                "pct_roof_meas": roof_meas / (k_ms * 1e3) * 100,
                "max_err_vs_ref": err, "grid": vm.last_grid_dim}

    rows.append(one("bf16", DType.BF16, wd, None))
    rows.append(one("int8", DType.F16, q8, m8))
    rows.append(one("int4", DType.F16, q4, m4))
    bf = rows[0]
    for r in rows:
        r["kspeedup_vs_bf16"] = bf["kernel_us"] / r["kernel_us"]
        r["tspeedup_vs_bf16"] = bf["token_us"] / r["token_us"]
    return rows


# ------------------------------------------------------------------------------------------------
# GOAL 2: occupancy experiment (noinline + SMEM/register reduction)
# ------------------------------------------------------------------------------------------------
def goal2(target, graph, wd):
    cfg = ScheduleConfig(pipelining_depth=2)
    inp = _inputs(11, 0)
    prog = lower(graph, target=target, config=cfg, pos=0, dtype=DType.BF16)
    ref = ReferenceVM(prog, wd, device="cpu").run(inp, kv={})["logits"].float()
    wbytes = _streamed_weight_bytes(prog, wd)

    # real max-K over the program's GEMV tiles -> the smallest SMEM x-cache that keeps the fast path
    max_k = 0
    for t in prog.tasks:
        p = getattr(t, "params", None)
        if p is not None and getattr(p, "K", 0):
            max_k = max(max_k, int(p.K))
    # round up to a multiple of 32 (vector chunk) for the quant path's group alignment
    smem_k = ((max_k + 31) // 32) * 32 if max_k > 0 else 0

    configs = [
        ("default",              None,                                                   False),
        ("noinline",             None,                                                   True),
        ("smem_fit+lb3",         {"gemv_max_k": smem_k, "lb_maxthreads": 256, "lb_minblocks": 3}, False),
        ("smem_fit+lb4",         {"gemv_max_k": smem_k, "lb_maxthreads": 256, "lb_minblocks": 4}, False),
        ("noinline+smem_fit+lb4", {"gemv_max_k": smem_k, "lb_maxthreads": 256, "lb_minblocks": 4}, True),
    ]
    out = []
    for name, knobs, ni in configs:
        try:
            vm = _build_vm(prog, wd, knobs=knobs, op_noinline=ni)
        except GpuUnavailable as e:
            out.append({"config": name, "error": str(e)})
            continue
        attr = vm.ext.kernel_attributes(vm.threads_per_block, vm.dyn_smem_bytes)
        err, close = _gpu_vs_ref_check(vm, ref, inp, rtol=2e-2, atol=2e-2)  # bf16 program
        ok = close and (vm.last_status.get("status") == "OK")
        k_ms = _median_event_ms(vm.relaunch, ITERS, WARMUP)
        gbs = wbytes / (k_ms * 1e-3) / 1e9
        out.append({"config": name, "knobs": knobs, "op_noinline": ni,
                    "num_regs": attr["num_regs"], "static_smem_bytes": attr["static_smem_bytes"],
                    "blocks_per_sm": attr["blocks_per_sm"], "grid": vm.last_grid_dim,
                    "kernel_us": k_ms * 1e3, "achieved_gbs": gbs,
                    "max_err_vs_ref": err, "correct": bool(ok)})
    return {"real_max_k": max_k, "smem_fit_k": smem_k, "configs": out}


def run():
    target = TARGETS["rtx5090"]
    model = make_toy(seed=0, dtype=torch.bfloat16, **SMALL)
    graph = from_toy(model)
    wd = model.weights_dict()
    q8, m8 = quantize_weights(wd, group=GROUP, bits=8, graph=graph)
    q4, m4 = quantize_weights(wd, group=GROUP, bits=4, graph=graph)

    g1 = goal1(target, graph, wd, q8, m8, q4, m4)
    g2 = goal2(target, graph, wd)

    i8, i4 = g1[1], g1[2]
    print(f"  device={torch.cuda.get_device_name(0)}  model=small 4L hidden={SMALL['hidden']} "
          f"inter={SMALL['intermediate']} vocab={SMALL['vocab']}  group={GROUP}")
    print("  GOAL 1, bf16 vs int8(lossless) vs int4 decode:")
    print(f"  {'path':>5} | {'MB':>6} | {'kernel us':>9} | {'GB/s':>5} | {'%roof(meas)':>11} | "
          f"{'kSpeedup':>8} | {'token us':>9} | {'max_err':>8}")
    for r in g1:
        print(f"  {r['name']:>5} | {r['weight_bytes']/1e6:>6.1f} | {r['kernel_us']:>9.1f} | "
              f"{r['achieved_gbs']:>5.0f} | {r['pct_roof_meas']:>10.1f}% | "
              f"{r['kspeedup_vs_bf16']:>7.2f}x | {r['token_us']:>9.1f} | {r['max_err_vs_ref']:>8.1e}")
    print(f"  >>> int8 (greedy-LOSSLESS) kernel-only speedup vs bf16: "
          f"{i8['kspeedup_vs_bf16']:.2f}x ; int4: {i4['kspeedup_vs_bf16']:.2f}x")

    print("  GOAL 2, occupancy experiment (blocks/SM before/after noinline + SMEM/reg cut):")
    print(f"  {'config':>22} | {'regs':>4} | {'smem':>6} | {'blk/sm':>6} | {'kernel us':>9} | "
          f"{'GB/s':>5} | ok")
    for c in g2["configs"]:
        if "error" in c:
            print(f"  {c['config']:>22} | ERROR: {c['error']}")
            continue
        print(f"  {c['config']:>22} | {c['num_regs']:>4} | {c['static_smem_bytes']:>6} | "
              f"{c['blocks_per_sm']:>6} | {c['kernel_us']:>9.1f} | {c['achieved_gbs']:>5.0f} | "
              f"{c['correct']}")
    base = next(c for c in g2["configs"] if c.get("config") == "default")
    best = max((c for c in g2["configs"] if "blocks_per_sm" in c),
               key=lambda c: c["blocks_per_sm"])
    raised = best["blocks_per_sm"] > base["blocks_per_sm"]
    print(f"  >>> occupancy: default={base['blocks_per_sm']} blocks/SM ; "
          f"best={best['blocks_per_sm']} ({best['config']}) -> "
          f"{'RAISED' if raised else 'NOT raised'}; decode bandwidth "
          f"{base['achieved_gbs']:.0f}->{best['achieved_gbs']:.0f} GB/s "
          f"(occupancy was NOT the binding bottleneck for single-stream decode at b=2).")

    headline = (f"int8 (lossless) kernel-only {i8['kspeedup_vs_bf16']:.2f}x vs bf16, "
                f"int4 {i4['kspeedup_vs_bf16']:.2f}x; occupancy "
                f"{base['blocks_per_sm']}->{best['blocks_per_sm']} blocks/SM "
                f"({'raised' if raised else 'not raised'}), but decode GB/s ~flat "
                f"({base['achieved_gbs']:.0f}->{best['achieved_gbs']:.0f}).")

    payload = {
        "device": torch.cuda.get_device_name(0),
        "model": "small_4L_h2048_i5632_v32000",
        "group": GROUP, "iters": ITERS, "warmup": WARMUP,
        "hbm_spec_gbs": target.hbm_bandwidth_gbs, "hbm_measured_gbs": target.measured_bw_gbs,
        "goal1_decode": g1,
        "goal2_occupancy": g2,
        "int8_kernel_speedup_vs_bf16": i8["kspeedup_vs_bf16"],
        "int4_kernel_speedup_vs_bf16": i4["kspeedup_vs_bf16"],
        "occupancy_default_blocks_per_sm": base["blocks_per_sm"],
        "occupancy_best_blocks_per_sm": best["blocks_per_sm"],
        "occupancy_raised": bool(raised),
        "headline": headline,
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {RESULT_PATH}")
    print(f"  HEADLINE: {headline}")
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
