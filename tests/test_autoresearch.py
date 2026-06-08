"""
AMK, ACCEPTANCE TEST for the unattended autoresearch driver + orchestrator (Loop 2).

Asserts, for the toy model on the analytic cost model (device=cpu, deterministic):
  * autoresearch() runs unattended for N iters, keeps only valid+correct schedules, logs every
    experiment to results.tsv with a non-blank correctness, and writes kept points to the corpus.
  * the orchestrator state is checkpointed and RESUMES on a re-run (experiment count grows).
  * the keep/revert rule never keeps an incorrect schedule, and a latency only exists with PASS.
  * the orchestrator move-on (consecutive-reverts plateau) fires and sets status=done.
  * the flywheel makes a warm run start from a corpus seed (warm_start), not the default.

GPU-free: device=cpu uses the predicted cost model, so this runs anywhere pytest does.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import amk_orchestrate as orch  # noqa: E402
import autoresearch as ar  # noqa: E402
from flywheel.log import read_corpus, read_results  # noqa: E402

GPU = "rtx5090"
MODEL = "toy"


def _check(cond: bool, msg: str) -> None:
    assert cond, msg


def test_autoresearch_runs_logs_and_checkpoints(tmp_path):
    corpus = str(tmp_path / "corpus.jsonl")
    results = str(tmp_path / "results.tsv")
    state = str(tmp_path / "state.json")

    out = ar.autoresearch(MODEL, GPU, iters=10, device="cpu", cold=True, seed=0,
                          corpus_path=corpus, results_path=results, state_path=state,
                          verbose=False)

    # ran unattended, found a correct schedule, kept >= 1 incumbent (the default at least).
    _check(out["iters_run"] >= 1, "autoresearch ran at least one iteration")
    _check(out["best_us"] is not None, "autoresearch found a best (correct) schedule")
    _check(out["n_crash"] == 0, "no crashes in the cpu cost-model run")
    _check(out["n_kept"] >= 1, "kept at least the starting incumbent")

    # every results.tsv row carries a non-blank correctness (the honesty rule).
    rows = read_results(results)
    _check(len(rows) >= 1, "results.tsv has experiment rows")
    for r in rows:
        _check(bool(r.get("correctness")), "every results.tsv row has a correctness verdict")
        # a latency only ever appears with a PASS
        if r.get("latency_us"):
            _check(r.get("correctness") == "PASS", "latency only on a PASS row")

    # kept correct points entered the corpus.
    cps = read_corpus(corpus)
    _check(len(cps) >= 1, "kept points written to the flywheel corpus")
    for c in cps:
        _check(c.get("correctness") == "PASS", "only PASS points in the corpus")

    # checkpoint exists and is resumable: a second run continues the SAME campaign.
    _check(os.path.exists(state), "orchestrator checkpoint written")
    st1 = orch.load_state(state)
    runs1 = st1["experiments_run"]
    ar.autoresearch(MODEL, GPU, iters=5, device="cpu", cold=True, seed=1,
                    corpus_path=corpus, results_path=results, state_path=state,
                    verbose=False)
    st2 = orch.load_state(state)
    _check(st2["experiments_run"] > runs1 or st2["status"] == orch.STATUS_DONE,
           "re-run resumes the campaign (experiment count grows or it had already finished)")


def test_orchestrator_move_on_plateau(tmp_path):
    state = orch._fresh_state(MODEL, GPU, str(tmp_path / "s.json"))
    results = str(tmp_path / "r.tsv")
    # first a kept baseline, then enough reverts to trip the plateau.
    orch.record(state, latency_us=100.0, status="kept", config={}, correctness="PASS",
                pct_of_roofline=500.0, latency_kind="predicted", results_path=results)
    for _ in range(orch.MOVE_ON_CRITERIA["consecutive_reverts"]):
        orch.record(state, latency_us=100.0, status="revert", config={}, correctness="PASS",
                    pct_of_roofline=500.0, latency_kind="predicted", results_path=results)
    _check(state["status"] == orch.STATUS_DONE, "plateau move-on set status=done")
    stop, reason = orch.should_move_on(state)
    _check(stop and "plateau" in reason, "should_move_on reports the plateau")


def test_warm_starts_from_corpus_seed(tmp_path):
    corpus = str(tmp_path / "corpus.jsonl")
    results = str(tmp_path / "results.tsv")
    # cold run seeds the corpus
    ar.autoresearch(MODEL, GPU, iters=12, device="cpu", cold=True, seed=0, min_gain=0.0005,
                    corpus_path=corpus, results_path=results,
                    state_path=str(tmp_path / "cold.json"), verbose=False)
    # warm run must pick up >= 1 warm seed from that corpus
    warm = ar.autoresearch(MODEL, GPU, iters=3, device="cpu", cold=False, seed=0, min_gain=0.0005,
                           corpus_path=corpus, results_path=results,
                           state_path=str(tmp_path / "warm.json"), verbose=False)
    _check(warm["warm_seeds"] >= 1, "warm run seeded from the corpus via warm_start")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
