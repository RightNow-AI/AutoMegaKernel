"""Tests for the AMK MCP server tool layer (amk_mcp).

These import amk_mcp directly and exercise the plain module-level tool_* functions, they do NOT
require the 'mcp' SDK to be installed (the SDK is only lazy-imported inside amk_mcp.main()). They
run on CPU (device='cpu') so no GPU is needed: correctness is the CPU ReferenceVM and latency is
the analytic prediction.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import amk_mcp


def test_doctor_keys():
    info = amk_mcp.tool_doctor()
    assert isinstance(info, dict)
    # environment + capability surface
    for key in ("python", "torch", "cuda_available", "nvcc", "targets", "status"):
        assert key in info, f"tool_doctor() missing key {key!r}"
    assert isinstance(info["targets"], list)
    assert "rtx5090" in info["targets"]
    assert isinstance(info["cuda_available"], bool)


def test_propose_returns_config_and_search_space():
    out = amk_mcp.tool_propose("toy", "rtx5090")
    assert isinstance(out, dict)
    assert "schedule_config" in out and isinstance(out["schedule_config"], dict)
    assert "search_space" in out and isinstance(out["search_space"], dict)
    # the kernel-knob sub-surface must be advertised
    assert any(k.startswith("kernel_knobs.") for k in out["search_space"])


def test_eval_default_config_cpu():
    # use the incumbent/default config that propose advertises
    cfg = amk_mcp.tool_propose("toy", "rtx5090")["schedule_config"]
    verdict = amk_mcp.tool_eval("toy", "rtx5090", cfg, device="cpu")
    assert isinstance(verdict, dict)
    for key in ("valid", "correct", "latency_us", "latency_kind",
                "pct_of_roofline", "schedule_id"):
        assert key in verdict, f"verdict missing key {key!r}"
    # the default config must be valid + correct, and on CPU the latency is the analytic prediction.
    assert verdict["valid"] is True
    assert verdict["correct"] is True
    assert verdict["latency_kind"] == "predicted"


def test_eval_accepts_json_string_config():
    cfg = amk_mcp.tool_propose("toy", "rtx5090")["schedule_config"]
    import json

    verdict = amk_mcp.tool_eval("toy", "rtx5090", json.dumps(cfg), device="cpu")
    assert verdict["valid"] is True and verdict["correct"] is True


# ======================================================================================
# The remaining wrappers: loop / autoresearch / orchestrate {status,next,report,record}.
# All exercised on device='cpu' (analytic cost model, no GPU) with tmp paths so the suite
# never touches the real workspace/ state, corpus, or results.tsv.
# ======================================================================================
def _redirect_orchestrate_paths(monkeypatch, tmp_path):
    """Point the orchestrator's module-level paths at a tmp dir. The orchestrate tool_* wrappers
    read amk_orchestrate.STATE_PATH / RESULTS_TSV / REPORT_PATH directly, so redirecting them keeps
    the campaign state out of the real workspace/."""
    import amk_orchestrate as orch

    state = str(tmp_path / "state.json")
    monkeypatch.setattr(orch, "STATE_PATH", state)
    monkeypatch.setattr(orch, "RESULTS_TSV", str(tmp_path / "results.tsv"))
    monkeypatch.setattr(orch, "REPORT_PATH", str(tmp_path / "report.md"))
    return orch, state


def test_tool_loop_cpu_keys():
    """tool_loop runs the keep/revert loop on CPU and returns the LoopResult shape. We monkeypatch
    harness.loop's paths to a tmp dir (the wrapper has no path args) so the real results.tsv/corpus
    are untouched, then assert the documented keys + an honest best verdict."""
    import tempfile

    import harness

    td = tempfile.mkdtemp()
    orig = harness.loop

    def patched(model, gpu, budget=8, device="auto", **kw):
        kw.setdefault("results_path", os.path.join(td, "results.tsv"))
        kw.setdefault("corpus_path", os.path.join(td, "corpus.jsonl"))
        return orig(model, gpu, budget=budget, device=device, **kw)

    harness.loop = patched
    try:
        out = amk_mcp.tool_loop("toy", "rtx5090", budget=2, device="cpu")
    finally:
        harness.loop = orig

    assert isinstance(out, dict)
    for key in ("best_verdict", "best_config", "rows", "n_trials", "n_valid",
                "n_correct", "results_tsv"):
        assert key in out, f"tool_loop missing key {key!r}"
    assert out["n_trials"] == 2
    assert isinstance(out["rows"], list) and len(out["rows"]) == 2
    # a returned best on CPU is correctness-gated (a latency only ever rides a correct verdict).
    assert out["best_verdict"] is not None and out["best_verdict"]["correct"] is True


def test_tool_autoresearch_cpu_keys(monkeypatch, tmp_path):
    """tool_autoresearch runs the unattended campaign on CPU and returns the AutoresearchResult
    shape. The wrapper has no path args, so we monkeypatch autoresearch.autoresearch's defaults to
    tmp paths (and redirect the orchestrator state) to keep the real campaign untouched."""
    import autoresearch

    _redirect_orchestrate_paths(monkeypatch, tmp_path)
    orig = autoresearch.autoresearch

    def patched(model, gpu, **kw):
        kw.setdefault("corpus_path", str(tmp_path / "corpus.jsonl"))
        kw.setdefault("results_path", str(tmp_path / "results.tsv"))
        kw.setdefault("state_path", str(tmp_path / "state.json"))
        return orig(model, gpu, **kw)

    monkeypatch.setattr(autoresearch, "autoresearch", patched)
    out = amk_mcp.tool_autoresearch("toy", "rtx5090", iters=3, device="cpu")

    assert isinstance(out, dict)
    for key in ("model", "gpu", "device", "iters_run", "best_us", "best_config",
                "trajectory", "n_kept", "n_correct"):
        assert key in out, f"tool_autoresearch missing key {key!r}"
    assert out["iters_run"] == 3
    assert out["device"] == "cpu"


def test_tool_orchestrate_status_next_report_no_campaign(monkeypatch, tmp_path):
    """With NO campaign state yet, the three read-only orchestrate tools return clean 'no-campaign'
    dicts rather than crashing (the documented empty-state behaviour)."""
    _redirect_orchestrate_paths(monkeypatch, tmp_path)

    status = amk_mcp.tool_orchestrate_status()
    assert status["campaign"] is None and status["status"] == "no-campaign"

    nxt = amk_mcp.tool_orchestrate_next()
    assert nxt["campaign"] is None and nxt["decision"] == "no-campaign"

    report = amk_mcp.tool_orchestrate_report()
    assert report["campaign"] is None and report["status"] == "no-campaign"


def test_tool_orchestrate_record_mutates_state(monkeypatch, tmp_path):
    """tool_orchestrate_record writes ONE experiment outcome into the campaign state and the
    returned view reflects the mutation. We start a campaign, record a 'kept' baseline, and assert
    experiments_run / best_us advanced; status + next + report then see that state."""
    orch, state_path = _redirect_orchestrate_paths(monkeypatch, tmp_path)

    # start a campaign (the wrappers require an existing state file).
    orch.get_or_create_state("toy", "rtx5090", state_path)

    before = amk_mcp.tool_orchestrate_status()
    assert before["experiments_run"] == 0 and before["best_us"] is None

    # record a kept baseline via the wrapper (config as a JSON string to exercise that path).
    out = amk_mcp.tool_orchestrate_record(
        "kept", latency_us=123.0, pct_roofline=250.0, kind="predicted",
        config='{"threads_per_block": 256}', description="baseline")
    for key in ("campaign", "experiments_run", "experiments_kept", "best_us",
                "best_config", "speedup", "consecutive_reverts"):
        assert key in out, f"tool_orchestrate_record missing key {key!r}"
    assert out["experiments_run"] == 1, "record incremented experiments_run"
    assert out["experiments_kept"] == 1, "a 'kept' outcome incremented experiments_kept"
    assert out["best_us"] == 123.0, "kept latency became the new best"
    assert out["best_config"] == {"threads_per_block": 256}, "kept config recorded"

    # the mutation is persisted: a fresh status read sees the same numbers.
    after = amk_mcp.tool_orchestrate_status()
    assert after["experiments_run"] == 1 and after["best_us"] == 123.0
    assert after["campaign"] == "toy/rtx5090"

    # next now decides on a live campaign (CONTINUE or STOP, never 'no-campaign').
    nxt = amk_mcp.tool_orchestrate_next()
    assert nxt["decision"] in ("CONTINUE", "STOP")
    assert nxt["experiments_run"] == 1

    # report renders the markdown summary for the live campaign.
    report = amk_mcp.tool_orchestrate_report()
    assert isinstance(report.get("report_markdown"), str) and report["report_markdown"]
    assert report["campaign"] == "toy/rtx5090"


def test_tool_orchestrate_record_no_campaign(monkeypatch, tmp_path):
    """Recording with no campaign state is a clean no-campaign dict (it does not start one or
    crash), recording requires a campaign started via autoresearch."""
    _redirect_orchestrate_paths(monkeypatch, tmp_path)
    out = amk_mcp.tool_orchestrate_record("kept", latency_us=1.0)
    assert out["campaign"] is None and out["status"] == "no-campaign"
