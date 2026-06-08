"""
AMK, ACCEPTANCE TEST for the cost model + Loop-2 schedule search
================================================================

Proves the two contracts of this module against the toy model:

  (1) ``cost_model.predict_us`` returns a finite, positive number for a *real* lowering, and it
      is ``>=`` the honest weights/bandwidth floor (``target.bandwidth_bound_us``). You cannot
      decode faster than you can stream the weights once.

  (2) ``search`` over >= 20 candidates NEVER returns an invalid schedule, every config it keeps
      lowers to a ``validate().ok`` program, and the best predicted latency is ``<=`` the default
      config's predicted latency (keep/revert can never regress past the baseline).

The lowerer (``schedule/lower.py``) is separately owned and may be mid-build, so this test ships a
self-contained ``lower_fn`` built from the patterns in ``vm/verify_vm.py`` (the ``_tiled_gemv``
shared-counter all-join pattern). The config's ``tiling['gemv']['N_tile']`` and ``sm_assignment``
genuinely drive the emitted task-DAG, so different configs produce different (valid) programs and
different predicted latencies, a faithful stand-in for the real lowerer's edit surface.

Run:  uv run python tests/test_search.py     (also a pytest module)
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.toy import ToyConfig  # noqa: E402
from schedule import cost_model  # noqa: E402
from schedule import search as search_mod  # noqa: E402
from schedule.ir import (  # noqa: E402
    BufferKind, DType, InstructionKind, MegakernelProgram, ScheduleConfig, TARGETS, Wait, validate,
)

DT = DType.F16  # decode weights are fp16, exercises the real roofline (2 bytes/weight)


# ======================================================================================
# A real toy lowerer: ScheduleConfig -> validated MegakernelProgram.
# Lowers one Llama-style decoder layer (RMSNorm -> SwiGLU MLP + residual) plus an lm_head, with
# N_tile and SM assignment driven by the config, mirroring vm/verify_vm.py's _tiled_gemv pattern.
# ======================================================================================
def _ceil_div(a, b):
    return (a + b - 1) // b


def _tiled_gemv(p, x_buf, w_buf, out_buf, K, N, n_tile_size, wait, counter, label, region):
    """Emit ceil(N / n_tile_size) GEMV_TILE tasks over disjoint column ranges, all incrementing
    `counter` (a true all-join). Returns the number of tiles emitted (== the join threshold)."""
    n_tile_size = max(1, min(n_tile_size, N))
    n_tiles = _ceil_div(N, n_tile_size)
    for i in range(n_tiles):
        n_off = i * n_tile_size
        n_t = min(n_tile_size, N - n_off)
        # est_bytes: fp16 weight tile streamed from HBM (the roofline-relevant traffic).
        p.add_task(InstructionKind.GEMV_TILE, [x_buf, w_buf], [out_buf], out_counter=counter,
                   waits=list(wait), params={"K": K, "N_tile": n_t, "n_off": n_off},
                   label=f"{region}.{label}[t{i}]",
                   est_bytes=K * n_t * 2, est_flops=2 * K * n_t)
    return n_tiles


def make_lower_fn(cfg_model: ToyConfig):
    """Build a deterministic lower_fn(graph, config, target) for the given toy model config."""
    H = cfg_model.hidden
    inter = cfg_model.intermediate
    V = cfg_model.vocab

    def lower_fn(graph, sched: ScheduleConfig, target) -> MegakernelProgram:
        n_tile = int(sched.tiling.get("gemv", {}).get("N_tile", 256))
        p = MegakernelProgram(meta={"model": "toy", "gpu": target.name}, target=target,
                              config=sched)

        # --- buffers ---
        bx = p.new_buffer("x", BufferKind.IO_INPUT, DT, (1, H)).id
        b_pn = p.new_buffer("post_norm", BufferKind.WEIGHT, DT, (H,), source="post_norm").id
        b_gw = p.new_buffer("gate.w", BufferKind.WEIGHT, DT, (inter, H), source="mlp.gate_proj.weight").id
        b_uw = p.new_buffer("up.w", BufferKind.WEIGHT, DT, (inter, H), source="mlp.up_proj.weight").id
        b_dw = p.new_buffer("down.w", BufferKind.WEIGHT, DT, (H, inter), source="mlp.down_proj.weight").id
        b_fn = p.new_buffer("final_norm", BufferKind.WEIGHT, DT, (H,), source="final_norm").id
        b_lm = p.new_buffer("lm_head.w", BufferKind.WEIGHT, DT, (V, H), source="lm_head.weight").id

        b_xn = p.new_buffer("xn", BufferKind.ACTIVATION, DT, (1, H)).id
        b_g = p.new_buffer("g", BufferKind.ACTIVATION, DT, (1, inter)).id
        b_u = p.new_buffer("u", BufferKind.ACTIVATION, DT, (1, inter)).id
        b_act = p.new_buffer("act", BufferKind.ACTIVATION, DT, (1, inter)).id
        b_d = p.new_buffer("d", BufferKind.ACTIVATION, DT, (1, H)).id
        b_h = p.new_buffer("h", BufferKind.ACTIVATION, DT, (1, H)).id
        b_hn = p.new_buffer("hn", BufferKind.ACTIVATION, DT, (1, H)).id
        b_logits = p.new_buffer("logits", BufferKind.IO_OUTPUT, DT, (1, V)).id

        # --- counters ---
        c_norm = p.new_counter("post_norm").id
        c_gate = p.new_counter("gate").id
        c_up = p.new_counter("up").id
        c_act = p.new_counter("swiglu").id
        c_down = p.new_counter("down").id
        c_res = p.new_counter("residual").id
        c_fnorm = p.new_counter("final_norm").id
        c_lm = p.new_counter("lm_head").id

        # --- MLP region (post_norm -> gate/up -> swiglu -> down -> residual) ---
        p.add_task(InstructionKind.RMSNORM, [bx, b_pn], [b_xn], out_counter=c_norm,
                   params={"eps": cfg_model.rms_eps, "hidden": H}, label="mlp.post_norm",
                   est_bytes=H * 2, est_flops=2 * H)
        n_gate = _tiled_gemv(p, b_xn, b_gw, b_g, H, inter, n_tile, [Wait(c_norm, 1)], c_gate, "gate", "mlp")
        n_up = _tiled_gemv(p, b_xn, b_uw, b_u, H, inter, n_tile, [Wait(c_norm, 1)], c_up, "up", "mlp")
        p.add_task(InstructionKind.SILU_MUL, [b_g, b_u], [b_act], out_counter=c_act,
                   waits=[Wait(c_gate, n_gate), Wait(c_up, n_up)], label="mlp.swiglu",
                   est_bytes=inter * 2, est_flops=2 * inter)
        n_down = _tiled_gemv(p, b_act, b_dw, b_d, inter, H, n_tile, [Wait(c_act, 1)], c_down, "down", "mlp")
        p.add_task(InstructionKind.ADD, [b_d, bx], [b_h], out_counter=c_res,
                   waits=[Wait(c_down, n_down)], label="mlp.residual",
                   est_bytes=H * 2, est_flops=H)

        # --- lm_head region (final_norm -> logits projection) ---
        p.add_task(InstructionKind.RMSNORM, [b_h, b_fn], [b_hn], out_counter=c_fnorm,
                   waits=[Wait(c_res, 1)], params={"eps": cfg_model.rms_eps, "hidden": H},
                   label="lm_head.final_norm", est_bytes=H * 2, est_flops=2 * H)
        _tiled_gemv(p, b_hn, b_lm, b_logits, H, V, n_tile, [Wait(c_fnorm, 1)], c_lm,
                    "logits", "lm_head")

        # --- SM assignment driven by the config (the lowerer resolves the policy into Task.sm) ---
        _assign_sms(p, sched, target)
        return p

    return lower_fn


def _assign_sms(p: MegakernelProgram, sched: ScheduleConfig, target) -> None:
    """Resolve sched.sm_assignment into each Task.sm, respecting the per-SM serial-queue ordering
    invariant the validator enforces: a producer must precede its consumer in the SAME SM's queue
    (which is task-list order). We assign each task to an SM >= the max SM of its predecessors so
    that constraint always holds for any policy."""
    num_sms = max(1, target.num_sms)
    if isinstance(sched.sm_assignment, dict):
        for t in p.tasks:
            if t.id in sched.sm_assignment:
                t.sm = int(sched.sm_assignment[t.id]) % num_sms
        return

    # Build predecessor map from dependency edges.
    preds: dict[int, list[int]] = {t.id: [] for t in p.tasks}
    for a, b in p.dependency_edges():
        preds[b].append(a)

    policy = sched.sm_assignment
    rr = 0
    for t in p.tasks:
        min_sm = 0
        for pr in preds[t.id]:
            if p.tasks[pr].sm is not None:
                min_sm = max(min_sm, p.tasks[pr].sm)
        if policy == "round_robin":
            cand = rr % num_sms
            rr += 1
        else:  # "load_balance" (and any unknown policy): spread but keep it simple + valid
            cand = (t.id * 2) % num_sms
        # never place a task on an SM earlier than a predecessor's (would violate queue ordering
        # since same-SM queue == task-list order and producer.id < consumer.id here)
        t.sm = max(cand, min_sm) % num_sms if max(cand, min_sm) < num_sms else min_sm


# ======================================================================================
# Tests
# ======================================================================================
def test_predict_us_finite_positive_and_above_floor():
    target = TARGETS["rtx5090"]
    cfg_model = ToyConfig(hidden=512, intermediate=1376, vocab=4096)
    lower_fn = make_lower_fn(cfg_model)

    prog = lower_fn(None, search_mod.default_config(target), target)
    res = validate(prog)
    assert res.ok, "default lowering must be valid:\n" + res.report()

    us = cost_model.predict_us(prog, target)
    assert math.isfinite(us), f"predict_us must be finite, got {us}"
    assert us > 0, f"predict_us must be positive, got {us}"

    floor = target.bandwidth_bound_us(prog.total_weight_bytes())
    assert math.isfinite(floor) and floor > 0, f"bandwidth floor must be positive, got {floor}"
    # The whole point of the roofline: you can't decode faster than streaming weights once.
    assert us >= floor - 1e-9, (
        f"predicted {us:.3f}us is below the weights/bandwidth floor {floor:.3f}us, "
        f"the cost model is claiming impossible bandwidth")

    bd = cost_model.estimate(prog, target)
    assert bd.distance_to_bandwidth_bound >= 1.0 - 1e-9
    assert abs(sum(bd.region_us.values()) - bd.makespan_us) < 1e-6 or bd.makespan_us > 0
    print(f"[1] predict_us OK: {bd.summary()}")


def test_search_never_returns_invalid_and_beats_or_ties_default():
    target = TARGETS["rtx5090"]
    cfg_model = ToyConfig(hidden=512, intermediate=1376, vocab=4096)
    lower_fn = make_lower_fn(cfg_model)

    budget = 30  # >= 20 as required
    result = search_mod.search(graph=None, target=target, budget=budget,
                               lower_fn=lower_fn, measure_fn=None, seed=7)

    assert len(result.trials) == budget, f"expected {budget} trials, got {len(result.trials)}"
    assert result.n_valid >= 1, "search produced no valid schedules at all"

    # (2a) Every KEPT config must lower to a validate().ok program, never an invalid schedule.
    kept = [t for t in result.trials if t.kept]
    assert kept, "search kept nothing"
    for t in kept:
        prog = lower_fn(None, t.config, target)
        r = validate(prog)
        assert r.ok, f"search KEPT an invalid config (trial {t.index}):\n" + r.report()

    # The returned best_config must itself be valid.
    assert result.best_config is not None, "search returned no best_config"
    best_prog = lower_fn(None, result.best_config, target)
    assert validate(best_prog).ok, "best_config does not lower to a valid program"

    # (2b) ALSO assert NO valid trial was ever marked kept with an invalid program, and that every
    # single valid trial's program actually validates (the safety contract: invalid => rejected).
    for t in result.trials:
        prog = lower_fn(None, t.config, target)
        ok = validate(prog).ok
        assert ok == t.valid, (
            f"trial {t.index}: search recorded valid={t.valid} but validate()={ok} "
            f"(the validity gate disagrees with the validator)")

    # (2c) Best predicted latency <= default predicted latency (keep/revert never regresses).
    assert result.default_score_us is not None, "default config failed to lower/validate"
    assert result.best_score_us is not None
    assert result.best_score_us <= result.default_score_us + 1e-9, (
        f"best {result.best_score_us:.3f}us > default {result.default_score_us:.3f}us, "
        f"keep/revert regressed past the baseline")

    print(f"[2] search OK: {result.summary()}")
    # A glimpse of the flywheel rows.
    tsv = search_mod.results_tsv(result)
    assert tsv.count("\n") == budget + 1  # header + one row per trial
    print("    sample row:", result.rows()[0])


def test_search_with_measure_fn_falls_back_and_exploits():
    """measure_fn (on-hardware exploit) is optional; supplying a stub must still produce a valid,
    non-regressing best, and the stub's measured latency must drive keep/revert when present."""
    target = TARGETS["rtx5090"]
    cfg_model = ToyConfig(hidden=256, intermediate=688, vocab=2048)
    lower_fn = make_lower_fn(cfg_model)

    # A deterministic "hardware" stub: measured ~ predicted with a small config-dependent wobble.
    def measure_fn(prog, tgt):
        bd = cost_model.estimate(prog, tgt)
        depth = prog.config.pipelining_depth if prog.config else 0
        return bd.predicted_us * (1.0 + 0.01 * ((depth % 3) - 1))

    result = search_mod.search(graph=None, target=target, budget=24, lower_fn=lower_fn,
                               measure_fn=measure_fn, seed=3)
    assert result.best_config is not None
    assert validate(lower_fn(None, result.best_config, target)).ok
    # at least one trial should carry a measured number
    assert any(t.measured_us is not None for t in result.trials), "measure_fn was never exercised"
    assert result.best_score_us is not None and result.default_score_us is not None
    assert result.best_score_us <= result.default_score_us + 1e-9
    print(f"[3] search+measure OK: {result.summary()}")


def test_search_rejects_invalid_configs_without_crashing():
    """A lower_fn that emits a clearly invalid program for certain configs must be tolerated: those
    trials are logged as rejected (valid=0) and NEVER kept, and the search still returns a good
    valid best. This is the agent-safety contract end to end."""
    target = TARGETS["rtx5090"]
    cfg_model = ToyConfig(hidden=128, intermediate=256, vocab=512)
    good = make_lower_fn(cfg_model)

    def flaky_lower(graph, sched, tgt):
        prog = good(graph, sched, tgt)
        # Sabotage configs with a small N_tile: inject a partial wait on a shared counter (a
        # which-producer RACE the validator must reject) so we KNOW some trials are invalid.
        if int(sched.tiling.get("gemv", {}).get("N_tile", 256)) <= 64:
            for t in prog.tasks:
                for w in t.waits:
                    if w.threshold > 1:
                        w.threshold = 1  # partial wait on a multi-producer counter -> RACE
                        return prog
            # if no multi-producer wait existed, force an unsatisfiable wait instead
            prog.tasks[-1].waits.append(Wait(prog.counters[0].id, 999))
        return prog

    result = search_mod.search(graph=None, target=target, budget=40, lower_fn=flaky_lower, seed=11)

    # There must be BOTH valid and rejected trials given the sabotage + N_tile sweep.
    assert any(not t.valid for t in result.trials), "sabotage never triggered (test is vacuous)"
    assert result.n_valid >= 1, "everything was rejected; search found no valid schedule"
    # No rejected trial was ever kept.
    for t in result.trials:
        if not t.valid:
            assert not t.kept, f"trial {t.index} is invalid yet was kept"
            assert t.reject_reason, "rejected trial must carry a reason"
    # The best is valid and non-regressing.
    assert result.best_config is not None
    assert validate(flaky_lower(None, result.best_config, target)).ok
    assert result.best_score_us is not None and result.default_score_us is not None
    assert result.best_score_us <= result.default_score_us + 1e-9
    print(f"[4] reject-without-crash OK: {result.summary()} "
          f"({sum(1 for t in result.trials if not t.valid)} rejected)")


if __name__ == "__main__":
    test_predict_us_finite_positive_and_above_floor()
    test_search_never_returns_invalid_and_beats_or_ties_default()
    test_search_with_measure_fn_falls_back_and_exploits()
    test_search_rejects_invalid_configs_without_crashing()
    print("\nALL SEARCH + COST-MODEL ACCEPTANCE TESTS PASSED")
