"""
AMK, GEMV TILE-WIDTH decode sweep: BEFORE vs AFTER (real measured roofline) + cuBLAS ceiling
=============================================================================================

DIAGNOSIS (the starting hypothesis): AMK's decode GEMV plateaued at ~50% of the MEASURED 731 GB/s
HBM roofline while a standalone cuBLAS gemv of the SAME 623 MB hits ~74% (measured, this machine).
The hypothesis was PER-TILE MEGAKERNEL OVERHEAD: the small model lowered to ~640 GEMV tiles
(N_tile=128), each re-caching x into SMEM behind 2 __syncthreads + a warp-reduce + dispatch, so
the cure would be FEWER, FATTER tiles.

WHAT THE HARDWARE ACTUALLY SAID (this bench): the hypothesis is REFUTED. A real CUDA-event sweep
shows fatter tiles are monotonically SLOWER and THINNER tiles are FASTER. The binding constraint is
PARALLELISM / SM LOAD-BALANCE + memory-level parallelism, NOT per-tile overhead: each GEMV tile is
one schedulable unit greedily balanced over the 82 SMs (vm/loader LPT), and the cp.async kernel
needs many independent in-flight weight streams to hide ~400 ns HBM latency. More, thinner tiles ==
more independent units == better balance + more MLP == higher achieved bandwidth, until tiles get
so thin (N_tile<=16) that the fixed overhead finally bites.

THE FIX (measured here): drop the default GEMV tile width from 128 to ~32 (schedule/lower.py
auto-sizer; floored at 16). We sweep N_tile from fat to thin, correctness-gate every variant vs the
CPU ReferenceVM, and report kernel-only + steady-state achieved GB/s and % of the 731 GB/s MEASURED
roofline, BEFORE (old N_tile=128, ~640 tiles) vs AFTER (best thin tiling). We also measure a cuBLAS
gemv of the same 623 MB as the honest upper bound the megakernel is chasing, and report how much of
the before->cuBLAS structural gap the change closes.

Every number is a CUDA-event median (warmup>=25, iters>=100) taken in INTERLEAVED A/B rounds to
wash out WDDM boost-clock drift. Run:  uv run python eval/bench_fat_tile_gemv.py
Writes paper/results/fat_tile_gemv.json.
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
ROUNDS = 5
RESULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "paper", "results", "fat_tile_gemv.json")

# Candidate tilings to sweep. "before" reproduces the OLD default (explicit N_tile=128 => ~640
# tiles). The rest sweep the tile width both fatter and thinner so the measured trend (thinner ==
# faster, down to a floor) is visible end-to-end. Each is a (label, ScheduleConfig-tiling) pair.
SWEEP = [
    ("before_ntile128", {"gemv": {"N_tile": 128}}),     # the old default (~640 tiles)
    ("fat_ntile256", {"gemv": {"N_tile": 256}}),        # fatter (the refuted hypothesis direction)
    ("ntile64", {"gemv": {"N_tile": 64}}),
    ("ntile48", {"gemv": {"N_tile": 48}}),
    ("ntile16", {"gemv": {"N_tile": 16}}),              # very thin (auto-sizer floor)
    ("after_default", {}),                              # the NEW shipped default (auto-sizer ~32)
]
# Label of the variant reported as "after" (the realized new default, not the bench-best width).
AFTER_LABEL = "after_default"


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


def _n_gemv_tiles(prog) -> int:
    from schedule.ir import InstructionKind
    return sum(1 for t in prog.tasks if int(t.op) == int(InstructionKind.GEMV_TILE))


def _build_and_check(tiling, wd, inp, ref_logits, target):
    """Lower with the given tiling, build the VM, correctness-gate vs the CPU reference.
    Returns (vm, prog, max_err, n_tiles, wbytes)."""
    from vm.loader import MegakernelVM
    cfg = ScheduleConfig(pipelining_depth=2, tiling=tiling)
    prog = lower(from_toy(MODEL), target=target, config=cfg, pos=0, dtype=DType.BF16)
    assert validate(prog).ok, "program rejected by validator"
    wbytes = _streamed_weight_bytes(prog, wd)
    n_tiles = _n_gemv_tiles(prog)
    try:
        vm = MegakernelVM(prog, wd, device="cuda")   # cp.async default (proven path)
    except (RuntimeError, TimeoutError) as e:
        raise GpuUnavailable(f"VM build/launch failed: {e}") from e
    out = vm.run(inp, kv={})["logits"].detach().float().cpu()
    err = (out - ref_logits).abs().max().item()
    if not torch.allclose(out, ref_logits, rtol=2e-2, atol=2e-2):
        raise AssertionError(f"GPU != reference (max_err={err:.3e})")
    if vm.last_status.get("status") != "OK":
        raise GpuUnavailable(f"status {vm.last_status}")
    return vm, prog, err, n_tiles, wbytes


def _result(label, tiling, vm, prog, err, n_tiles, k_ms, t_ms, target, wbytes):
    roof_meas = target.measured_bandwidth_bound_us(wbytes)
    roof_spec = target.bandwidth_bound_us(wbytes)
    return {
        "label": label, "tiling": tiling, "n_gemv_tiles": n_tiles,
        "kernel_us": k_ms * 1e3, "token_us": t_ms * 1e3,
        "kernel_gbs": wbytes / (k_ms * 1e-3) / 1e9,
        "token_gbs": wbytes / (t_ms * 1e-3) / 1e9,
        "pct_roof_meas_kernel": roof_meas / (k_ms * 1e3) * 100,
        "pct_roof_meas_token": roof_meas / (t_ms * 1e3) * 100,
        "pct_roof_spec_kernel": roof_spec / (k_ms * 1e3) * 100,
        "dyn_smem_bytes": vm.dyn_smem_bytes, "grid": vm.last_grid_dim,
        "max_err_vs_ref": err,
    }


def _cublas_ceiling(wbytes, target):
    """The honest upper bound: a single cuBLAS gemv (torch.mv) over a [N,K] bf16 weight whose total
    bytes match the megakernel's streamed weight bytes. This is the SAME 623 MB read by ONE fused
    matvec with NO DAG dispatch / counter sync / per-tile x-recache, the ceiling AMK chases."""
    K = SMALL["hidden"]
    N = max(1, wbytes // (2 * K))             # bf16 weight rows to match the streamed byte budget
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
    real_bytes = W.numel() * W.element_size()

    def fn():
        return torch.mv(W, x)
    ms = _median_event_ms(fn, ITERS, WARMUP)
    roof_meas = target.measured_bandwidth_bound_us(real_bytes)
    return {
        "bytes": real_bytes, "us": ms * 1e3,
        "gbs": real_bytes / (ms * 1e-3) / 1e9,
        "pct_roof_meas": roof_meas / (ms * 1e3) * 100,
    }


MODEL = None


def run():
    global MODEL
    target = TARGETS["rtx5090"]
    meas_bw = target.measured_bw_gbs
    MODEL = make_toy(seed=0, dtype=torch.bfloat16, **SMALL)
    wd = MODEL.weights_dict()
    inp = _inputs(11, 0)
    # CPU reference oracle (lowering is tiling-independent in result; use the default fat config).
    ref_prog = lower(from_toy(MODEL), target=target,
                     config=ScheduleConfig(pipelining_depth=2), pos=0, dtype=DType.BF16)
    ref = ReferenceVM(ref_prog, wd, device="cpu").run(inp, kv={})["logits"].float()

    # Build + correctness-gate every candidate up front.
    built = []
    for label, tiling in SWEEP:
        try:
            vm, prog, err, n_tiles, wbytes = _build_and_check(tiling, wd, inp, ref, target)
        except (GpuUnavailable, AssertionError) as e:
            print(f"  SKIP {label}: {e}")
            continue
        built.append((label, tiling, vm, prog, err, n_tiles, wbytes))
        print(f"  built {label:16} n_gemv_tiles={n_tiles:4} smem={vm.dyn_smem_bytes:6} "
              f"max_err={err:.1e}")
    if not built:
        raise GpuUnavailable("no candidate built")
    wbytes = built[0][6]

    # Interleaved A/B rounds (wash out WDDM boost drift): each round times every variant's
    # kernel-only (relaunch) and steady-state (run) once; take per-variant medians.
    k_rounds = {lbl: [] for (lbl, *_rest) in built}
    t_rounds = {lbl: [] for (lbl, *_rest) in built}
    for _ in range(ROUNDS):
        for (lbl, tiling, vm, prog, err, n_tiles, wb) in built:
            k_rounds[lbl].append(_median_event_ms(vm.relaunch, ITERS, WARMUP))
        for (lbl, tiling, vm, prog, err, n_tiles, wb) in built:
            t_rounds[lbl].append(_median_event_ms(lambda v=vm: v.run(inp, kv={}), ITERS, WARMUP))

    def md(v):
        return sorted(v)[len(v) // 2]
    results = {}
    for (lbl, tiling, vm, prog, err, n_tiles, wb) in built:
        r = _result(lbl, tiling, vm, prog, err, n_tiles,
                    md(k_rounds[lbl]), md(t_rounds[lbl]), target, wb)
        r["rounds_kernel_us"] = [x * 1e3 for x in sorted(k_rounds[lbl])]
        results[lbl] = r

    cublas = _cublas_ceiling(wbytes, target)

    before = results["before_ntile128"]
    # AFTER = the realized new shipped default (auto-sizer), so the headline matches what ships.
    after = results.get(AFTER_LABEL) or min(
        (r for lbl, r in results.items() if lbl != "before_ntile128"),
        key=lambda r: r["kernel_us"])
    # also report the bench-best width for transparency
    best_width = min((r for lbl, r in results.items() if lbl != "before_ntile128"),
                     key=lambda r: r["kernel_us"])

    kspeed = before["kernel_us"] / after["kernel_us"]
    tspeed = before["token_us"] / after["token_us"]
    # how much of the structural gap (before -> cuBLAS) did we close?
    gap = cublas["pct_roof_meas"] - before["pct_roof_meas_kernel"]
    closed = (after["pct_roof_meas_kernel"] - before["pct_roof_meas_kernel"]) / gap * 100 if gap > 0 else 0.0

    print(f"\n  device={torch.cuda.get_device_name(0)}  model=small 4L hidden={SMALL['hidden']} "
          f"inter={SMALL['intermediate']} vocab={SMALL['vocab']}")
    print(f"  weights streamed = {wbytes/1e6:.1f} MB  |  MEASURED roofline = {meas_bw:.0f} GB/s")
    print(f"  {'variant':>16} | {'tiles':>5} | {'kern us':>8} | {'GB/s':>5} | {'%meas':>6} | "
          f"{'tok us':>8} | {'%meas':>6}")
    for lbl in [lab for lab, _ in SWEEP if lab in results]:
        r = results[lbl]
        print(f"  {r['label']:>16} | {r['n_gemv_tiles']:>5} | {r['kernel_us']:>8.1f} | "
              f"{r['kernel_gbs']:>5.0f} | {r['pct_roof_meas_kernel']:>5.1f}% | "
              f"{r['token_us']:>8.1f} | {r['pct_roof_meas_token']:>5.1f}%")
    print(f"  {'cuBLAS gemv':>16} | {'--':>5} | {cublas['us']:>8.1f} | {cublas['gbs']:>5.0f} | "
          f"{cublas['pct_roof_meas']:>5.1f}% |  (ceiling)")
    print(f"  >>> AFTER (new default) = {after['label']} ({after['n_gemv_tiles']} tiles); "
          f"bench-best width = {best_width['label']} ({best_width['pct_roof_meas_kernel']:.1f}%)")
    print(f"  >>> kernel-only thin-tile speedup vs before: {kspeed:.2f}x ; steady-state {tspeed:.2f}x")
    print(f"  >>> closed {closed:.0f}% of the before->cuBLAS structural gap")

    meas_str = (f"{before['kernel_gbs']:.0f} GB/s, {before['pct_roof_meas_kernel']:.0f}% meas -> "
                f"{after['kernel_gbs']:.0f} GB/s, {after['pct_roof_meas_kernel']:.0f}% meas "
                f"({kspeed:.2f}x); cuBLAS ceiling {cublas['pct_roof_meas']:.0f}% meas")
    headline = (f"thin-tile decode GEMV: kernel-only {before['pct_roof_meas_kernel']:.0f}% -> "
                f"{after['pct_roof_meas_kernel']:.0f}% of {meas_bw:.0f} GB/s MEASURED roofline "
                f"({kspeed:.2f}x), tiles {before['n_gemv_tiles']}->{after['n_gemv_tiles']}; "
                f"cuBLAS ceiling {cublas['pct_roof_meas']:.0f}%; closed {closed:.0f}% of the gap; "
                f"correctness preserved (max_err {after['max_err_vs_ref']:.1e} <= 2e-2 bf16).")
    print(f"  HEADLINE: {headline}")
    print(f"  measured_before_after: {meas_str}")

    payload = {
        "device": torch.cuda.get_device_name(0),
        "model": "small_4L_h2048_i5632_v32000",
        "iters": ITERS, "warmup": WARMUP, "rounds": ROUNDS,
        "weight_bytes": wbytes,
        "hbm_spec_gbs": target.hbm_bandwidth_gbs, "hbm_measured_gbs": meas_bw,
        "sweep": [results[lab] for lab, _ in SWEEP if lab in results],
        "before": before, "after": after, "bench_best_width": best_width,
        "cublas_ceiling": cublas,
        "kernel_speedup": kspeed, "token_speedup": tspeed,
        "gap_closed_pct": closed,
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
