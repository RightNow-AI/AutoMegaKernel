#!/usr/bin/env python3
"""
AMK, MCP SERVER (the native coding-agent integration surface)
==============================================================

This module exposes AutoMegaKernel's EXISTING, verified substrate (the harness, propose / eval /
loop -, the unattended autoresearch driver, and the campaign orchestrator state machine) to coding
agents (Claude Code, Codex) over the Model Context Protocol. It changes NOTHING about the
substrate's behaviour: every tool is a thin, JSON-serializing wrapper over the EXISTING functions in
``harness.py`` / ``autoresearch.py`` / ``amk_orchestrate.py`` / ``amk_cli.py``. No number is
invented here.

DESIGN (importable + testable WITHOUT the 'mcp' SDK):
  * All tool LOGIC lives in plain module-level ``tool_*`` functions that import ONLY the existing AMK
    modules. They are importable and unit-testable with no MCP dependency at all.
  * The actual MCP server (FastMCP, stdio transport) is wired inside :func:`main`, which lazy-imports
    the official 'mcp' SDK and prints a helpful install hint if it is missing.

HONESTY CONTRACT (inherited verbatim from the substrate, surfaced, never weakened):
  * Correctness FIRST: a latency is NEVER reported without a correctness PASS vs the CPU ReferenceVM.
    A kept candidate is correct AND >= 1% faster than the incumbent.
  * validate-before-launch: an unsafe ScheduleConfig is a clean REJECTED (deadlock/race-free proof),
    never a hung GPU.
  * The edit surface is ScheduleConfig + kernel_knobs ONLY, never raw kernel code, never vm/ or the
    frozen ABI.
  * Measured-gpu latency is drift-robust; physically-impossible sub-roofline latencies are withheld.
  * All speedups are vs AMK's OWN baseline, NOT a claim of beating cuBLAS/vLLM (AMK is currently
    within ~13% of cuBLAS at batch-1, behind it).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ======================================================================================
# doctor, environment + registered GpuTargets (mirrors amk_cli._doctor, as a dict)
# ======================================================================================
def tool_doctor() -> dict[str, Any]:
    """Report the AMK runtime environment as a JSON-serializable dict: python/torch versions,
    CUDA availability + device name/compute-capability/SM count/memory, whether nvcc is on PATH
    (required to BUILD the CUDA megakernel), and the registered GpuTarget names. Mirrors the
    ``amk doctor`` CLI verb (amk_cli._doctor) but returns structured data instead of printing."""
    from shutil import which

    from schedule.ir import TARGETS

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": None,
        "cuda_available": False,
        "device_name": None,
        "sm_arch": None,
        "num_sms": None,
        "total_memory_gb": None,
        "nvcc": which("nvcc"),
        "targets": list(TARGETS),
        "status": "cpu-only",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        cuda = bool(torch.cuda.is_available())
        info["cuda_available"] = cuda
        if cuda:
            cap = torch.cuda.get_device_capability(0)
            props = torch.cuda.get_device_properties(0)
            info["device_name"] = torch.cuda.get_device_name(0)
            info["sm_arch"] = f"sm_{cap[0]}{cap[1]}"
            info["num_sms"] = int(props.multi_processor_count)
            info["total_memory_gb"] = round(props.total_memory / 1024 ** 3, 1)
    except Exception as e:  # torch import / device query must never crash the tool
        info["error"] = f"{type(e).__name__}: {e}"

    info["status"] = ("ready" if (info["cuda_available"] and info["nvcc"]) else
                      "cpu-only (reference VM works; GPU megakernel needs CUDA+nvcc)")
    return info


# ======================================================================================
# propose, the incumbent ScheduleConfig + the editable search space
# ======================================================================================
def tool_propose(model: str, gpu: str = "rtx5090") -> dict[str, Any]:
    """Return the current incumbent ScheduleConfig (as a dict) plus the documented, editable
    ``search_space`` (the knobs + ranges the agent may move, including the ``kernel_knobs``
    sub-surface). This is the 'read program.md' step: it tells the agent exactly what surface it
    owns. Thin wrapper over ``harness.propose(model, gpu)``."""
    import harness

    return harness.propose(model, gpu)


# ======================================================================================
# eval, the structured JSON verdict for one ScheduleConfig
# ======================================================================================
def tool_eval(model: str, gpu: str, config: Any, device: str = "auto") -> dict[str, Any]:
    """Evaluate ONE ScheduleConfig and return a structured honest verdict
    ``{valid, correct, latency_us, latency_kind, pct_of_roofline, bound_us, schedule_id, ...}``.

    ``config`` may be a JSON object (a ScheduleConfig dict, optionally with a ``"kernel_knobs"``
    object) OR a JSON string of one. A bad config is a clean ``valid=False`` + ``rejected_reason``
    (never a crash, never a latency); a latency is only ever emitted with a correctness PASS vs the
    CPU ReferenceVM. Use ``device='cpu'`` for the analytic prediction with no GPU. Thin wrapper over
    ``harness.evaluate(model, gpu, config, device)``."""
    import harness

    if isinstance(config, str):
        config = json.loads(config)
    return harness.evaluate(model, gpu, config, device=device)


# ======================================================================================
# loop, the keep/revert autoresearch loop over proposed configs
# ======================================================================================
def tool_loop(model: str, gpu: str, budget: int = 8, device: str = "auto") -> dict[str, Any]:
    """Run the Loop-2 keep/revert autoresearch loop for ``budget`` trials and return the best
    verdict + best config + the per-trial log rows + counts. Every trial passes the same honest
    lower -> validate -> correctness -> keep/revert gate; the best VALID + CORRECT schedule that is
    >= 1% faster than the incumbent is kept. Thin wrapper over
    ``harness.loop(model, gpu, budget, device)``."""
    import harness

    return harness.loop(model, gpu, budget=budget, device=device)


# ======================================================================================
# autoresearch, the unattended campaign driver
# ======================================================================================
def tool_autoresearch(model: str, gpu: str, minutes: float | None = None,
                      iters: int | None = None, device: str = "auto",
                      overnight: bool = False, cold: bool = False) -> dict[str, Any]:
    """Run the unattended keep/revert autoresearch campaign for ``iters`` iterations OR ``minutes``
    wall-clock (whichever is given/hit first; defaults to 20 iters if neither). Returns the
    AutoresearchResult dict (baseline_us, best_us, best_config, speedup_vs_baseline, trajectory,
    ...). It is resumable + crash-proof: a per-iteration failure is logged and the loop continues.
    ``overnight=True`` never stops on a plateau (basin-hops, preserves the global best);
    ``cold=True`` ignores the flywheel prior (pure exploration). All speedups are vs AMK's OWN
    default schedule. Thin wrapper over ``autoresearch.autoresearch(...)``."""
    import autoresearch

    return autoresearch.autoresearch(model, gpu, iters=iters, minutes=minutes,
                                     device=device, overnight=overnight, cold=cold,
                                     verbose=False)


# ======================================================================================
# orchestrate, the campaign state machine (status / next / record / report)
# ======================================================================================
def _orch_state() -> tuple[Any, dict[str, Any] | None]:
    """Load the campaign state dict directly (do NOT use the print-only cmd_* funcs). Returns
    (orch_module, state_or_None)."""
    import amk_orchestrate as orch

    state = orch.load_state(orch.STATE_PATH)
    return orch, state


def _state_view(orch: Any, state: dict[str, Any]) -> dict[str, Any]:
    """A clean, JSON-serializable view of the campaign state (drop private keys)."""
    return {k: v for k, v in state.items() if not k.startswith("_")}


def tool_orchestrate_status() -> dict[str, Any]:
    """Return the current campaign state as a structured dict: model/gpu, status, experiments_run /
    experiments_kept, consecutive_reverts, baseline_us, best_us, speedup, best_pct_roofline,
    best_kind, region_breakdown, move_on_reason. Built directly from
    ``amk_orchestrate.load_state(...)`` (not the print-only cmd_status). Returns
    ``{"campaign": None, ...}`` when no campaign has been started yet."""
    orch, state = _orch_state()
    if state is None:
        return {"campaign": None, "status": "no-campaign",
                "message": "no campaign state yet; run autoresearch or orchestrate_record first."}
    view = _state_view(orch, state)
    view["campaign"] = f"{state.get('model')}/{state.get('gpu')}"
    return view


def tool_orchestrate_next() -> dict[str, Any]:
    """Return the continue-or-stop DECISION from the move-on criteria (plateau / near-roofline /
    time-budget / speedup-target ladder) as a structured dict: ``{decision, reason,
    experiments_run, speedup, consecutive_reverts, hottest_region}``. Built from
    ``amk_orchestrate.should_move_on(state)`` (not the print-only cmd_next)."""
    orch, state = _orch_state()
    if state is None:
        return {"campaign": None, "decision": "no-campaign",
                "message": "no campaign state yet; nothing to decide."}
    stop, reason = orch.should_move_on(state)
    decision = "STOP" if (stop or state.get("status") == orch.STATUS_DONE) else "CONTINUE"
    rb = state.get("region_breakdown") or {}
    hot = max((r for r in orch.REGIONS if rb.get(r) is not None),
              key=lambda r: rb.get(r, 0.0), default=None)
    return {
        "campaign": f"{state.get('model')}/{state.get('gpu')}",
        "decision": decision,
        "reason": state.get("move_on_reason") or reason,
        "experiments_run": state.get("experiments_run", 0),
        "speedup": state.get("speedup"),
        "consecutive_reverts": state.get("consecutive_reverts", 0),
        "consecutive_reverts_threshold": orch.MOVE_ON_CRITERIA["consecutive_reverts"],
        "hottest_region": hot,
        "hottest_region_share": rb.get(hot) if hot else None,
    }


def tool_orchestrate_report() -> dict[str, Any]:
    """Write + return the aggregate campaign report. Calls ``amk_orchestrate.cmd_report(state)``
    (which writes ``workspace/amk_aggregate_report.md``) and returns a structured dict with the
    markdown report text plus the campaign summary fields."""
    orch, state = _orch_state()
    if state is None:
        return {"campaign": None, "status": "no-campaign",
                "message": "no campaign state yet; nothing to report."}
    # cmd_report writes the md file + prints; capture its returned markdown string.
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_md = orch.cmd_report(state)
    view = _state_view(orch, state)
    view["campaign"] = f"{state.get('model')}/{state.get('gpu')}"
    view["report_markdown"] = report_md
    view["report_path"] = orch.REPORT_PATH
    return view


def tool_orchestrate_record(status: str, latency_us: float | None = None,
                           pct_roofline: float | None = None, kind: str | None = None,
                           config: Any = None, description: str = "") -> dict[str, Any]:
    """Record ONE experiment outcome into the campaign state (and into results.tsv). ``status`` is
    one of: ``kept | revert | failed | crash | timeout | rejected`` (the AutoKernel vocabulary):
    ``kept`` resets the plateau counter and may set a new best; everything else increments
    ``consecutive_reverts``. ``config`` may be a dict or a JSON string. Requires an existing
    campaign (start one via autoresearch). Thin wrapper over ``amk_orchestrate.record(...)``;
    returns the updated state view."""
    orch, state = _orch_state()
    if state is None:
        return {"campaign": None, "status": "no-campaign",
                "message": "no campaign state yet; start one via autoresearch before recording."}
    if isinstance(config, str):
        config = json.loads(config)
    orch.record(state, latency_us=latency_us, status=status, config=config,
                pct_of_roofline=pct_roofline, latency_kind=kind, description=description)
    view = _state_view(orch, state)
    view["campaign"] = f"{state.get('model')}/{state.get('gpu')}"
    return view


# ======================================================================================
# main, wire the official MCP SDK (FastMCP, stdio). Lazy import so the tools above stay
#        importable + testable WITHOUT the 'mcp' package installed.
# ======================================================================================
def main() -> int:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        sys.stderr.write(
            "The 'mcp' Python SDK is not installed. Install the AMK agent extra:\n"
            "    pip install 'automegakernel[agent]'\n"
            "  or, in this repo:\n"
            "    uv sync --extra agent\n"
            "then run:  amk-mcp   (or:  uv run python amk_mcp.py)\n")
        return 1

    mcp = FastMCP("automegakernel")

    # Register each tool_* as an MCP tool. FastMCP reads the signature + docstring for the schema.
    @mcp.tool()
    def amk_doctor() -> dict[str, Any]:
        """AMK environment report: python/torch, CUDA availability + device, nvcc, GpuTargets."""
        return tool_doctor()

    @mcp.tool()
    def amk_propose(model: str, gpu: str = "rtx5090") -> dict[str, Any]:
        """Incumbent ScheduleConfig + the editable search_space (the agent's edit surface)."""
        return tool_propose(model, gpu)

    @mcp.tool()
    def amk_eval(model: str, gpu: str, config: Any, device: str = "auto") -> dict[str, Any]:
        """Evaluate ONE ScheduleConfig -> structured honest verdict (correctness-gated latency).
        config is a JSON object (a ScheduleConfig, optionally with a 'kernel_knobs' object) or a
        JSON string. Use device='cpu' to get the analytic prediction with no GPU."""
        return tool_eval(model, gpu, config, device)

    @mcp.tool()
    def amk_loop(model: str, gpu: str, budget: int = 8, device: str = "auto") -> dict[str, Any]:
        """Keep/revert autoresearch loop over proposed configs -> best verdict + config + rows."""
        return tool_loop(model, gpu, budget, device)

    @mcp.tool()
    def amk_autoresearch(model: str, gpu: str, minutes: float | None = None,
                         iters: int | None = None, device: str = "auto",
                         overnight: bool = False, cold: bool = False) -> dict[str, Any]:
        """Unattended, resumable, crash-proof autoresearch campaign (iters OR minutes)."""
        return tool_autoresearch(model, gpu, minutes, iters, device, overnight, cold)

    @mcp.tool()
    def amk_orchestrate_status() -> dict[str, Any]:
        """Current campaign state (baseline/best/speedup/region split/move-on reason)."""
        return tool_orchestrate_status()

    @mcp.tool()
    def amk_orchestrate_next() -> dict[str, Any]:
        """Continue-or-stop decision from the move-on criteria ladder."""
        return tool_orchestrate_next()

    @mcp.tool()
    def amk_orchestrate_report() -> dict[str, Any]:
        """Write + return the aggregate campaign report (markdown + summary fields)."""
        return tool_orchestrate_report()

    @mcp.tool()
    def amk_orchestrate_record(status: str, latency_us: float | None = None,
                              pct_roofline: float | None = None, kind: str | None = None,
                              config: Any = None, description: str = "") -> dict[str, Any]:
        """Record one experiment outcome (kept|revert|failed|crash|timeout|rejected)."""
        return tool_orchestrate_record(status, latency_us, pct_roofline, kind, config, description)

    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
