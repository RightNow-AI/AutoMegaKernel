"""
AMK, CPU-only acceptance test for the OVERNIGHT autoresearch driver (TCG-OVERNIGHT).

``autoresearch(..., overnight=True)`` is the "run it and sleep" mode: it is built to grind for
hours WITHOUT stopping at the first plateau. Its advertised invariants:

  * it does NOT stop on the consecutive-revert plateau, only the iters/minutes budget ends it;
  * after ``restart_after`` consecutive non-improvements it BASIN-HOPS: it jumps the exploration
    incumbent to a fresh random point and resets the plateau counter (escaping a stuck region);
  * the GLOBAL best is PRESERVED across those restarts, a basin-hop can only ever discover
    something better, never lose the best found so far (monotonically non-worsening).

This test runs the driver on the toy model with device='cpu' (the analytic cost model, so it is
GPU-free and deterministic) into a tmp workspace/state, and asserts it ran the full iteration
budget, performed >= 1 basin-hop restart, and the recorded global best never worsens.

Run:  uv run python -m pytest tests/test_overnight.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import autoresearch as ar  # noqa: E402

GPU = "rtx5090"
MODEL = "toy"
ITERS = 24
RESTART_AFTER = 3


def test_overnight_basin_hops_and_preserves_global_best(tmp_path):
    out = ar.autoresearch(
        MODEL, GPU, iters=ITERS, device="cpu", overnight=True, restart_after=RESTART_AFTER,
        cold=True, seed=0,
        corpus_path=str(tmp_path / "corpus.jsonl"),
        results_path=str(tmp_path / "results.tsv"),
        state_path=str(tmp_path / "state.json"),
        verbose=False)

    # ---- ran the full unattended budget; no crashes on the cpu cost-model path ----
    assert out["iters_run"] == ITERS, f"overnight run should complete all {ITERS} iters"
    assert out["n_crash"] == 0, "no crashes/timeouts on the deterministic cpu path"
    assert out["best_us"] is not None, "found a correct schedule (a global best exists)"

    # ---- performed >= 1 basin-hop restart (the plateau never stopped the night) ----
    trajectory = out["trajectory"]
    restarts = sum(1 for t in trajectory if t.get("source") == "restart")
    assert restarts >= 1, (
        f"overnight mode must basin-hop at least once with restart_after={RESTART_AFTER} "
        f"over {ITERS} iters (saw {restarts})")

    # the overnight wake-up report was written (the morning summary).
    assert os.path.exists(tmp_path / "state.json"), "campaign checkpoint written"

    # ---- the recorded GLOBAL best is monotonically non-worsening across the whole run ----
    # (the measured/predicted evaluator only ever keeps/reverts vs the resident global-best, so a
    #  basin-hop into a fresh region can never raise the recorded best, it is preserved.)
    bests = [t["best_us"] for t in trajectory if t.get("best_us") is not None]
    assert bests, "trajectory records a running global best"
    for prev, cur in zip(bests, bests[1:]):
        assert cur <= prev + 1e-9, (
            f"global best worsened across a restart: {prev} -> {cur} (must be preserved)")

    # the final reported best equals the best ever seen in the trajectory (no regression at the end).
    assert abs(out["best_us"] - min(bests)) <= 1e-9, \
        "final best must equal the minimum global best seen during the run"


def test_overnight_does_not_stop_on_plateau(tmp_path):
    """A long plateau (the toy cost-model landscape is flat, so almost every iter reverts) must NOT
    end an overnight run early, only the iters budget does. This is the 'never stop on a plateau'
    contract that distinguishes overnight from the normal move-on-on-plateau loop."""
    out = ar.autoresearch(
        MODEL, GPU, iters=ITERS, device="cpu", overnight=True, restart_after=RESTART_AFTER,
        cold=True, seed=7,
        corpus_path=str(tmp_path / "corpus.jsonl"),
        results_path=str(tmp_path / "results.tsv"),
        state_path=str(tmp_path / "state.json"),
        verbose=False)
    assert out["iters_run"] == ITERS, "overnight run ignored the plateau and used the full budget"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
