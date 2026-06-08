#!/usr/bin/env python3
"""
AMK, AUTORESEARCH ORCHESTRATOR (the AutoKernel-style campaign state machine)
=============================================================================

This is the AMK analogue of AutoKernel's ``orchestrate.py``: a small, persistent state machine
that tracks ONE ``(model, gpu)`` schedule-search campaign, the baseline, the running best, the
per-region (attention / mlp / lm_head) breakdown, how many experiments ran, the consecutive-revert
plateau counter, and the MOVE_ON_CRITERIA that decide when a region is done. It is the bookkeeping
brain the unattended driver (:mod:`autoresearch`) and a coding agent both talk to.

FAITHFUL TO the AutoKernel autoresearch orchestrator design:
  * The same four verbs, ``status`` / ``next`` / ``record`` / ``report``.
  * The same keep/revert accounting on ``record`` (kept resets the plateau counter + may set a new
    best; revert/failure increments ``consecutive_reverts``).
  * The same MOVE_ON_CRITERIA family, ``consecutive_reverts`` plateau, near-roofline target,
    a max-minutes wall-clock budget, and a speedup-vs-baseline target, but expressed in AMK's
    *latency / pct-of-roofline* units instead of TFLOPS (decode is bandwidth-bound, so the natural
    objective is minimize latency == approach the weights/bandwidth floor).
  * State persisted as JSON (``workspace/amk_orchestration_state.json``), atomically.

HONESTY: every experiment the orchestrator records is ALSO appended to ``results.tsv`` via
``flywheel.log`` with its real correctness verdict and latency_kind, the orchestrator never
invents a number; it only aggregates what the eval measured/predicted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flywheel.log import ResultRow, append_result  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.join(SCRIPT_DIR, "workspace")
STATE_PATH = os.path.join(WORKSPACE, "amk_orchestration_state.json")
RESULTS_TSV = os.path.join(WORKSPACE, "results.tsv")
REPORT_PATH = os.path.join(WORKSPACE, "amk_aggregate_report.md")

# ---------------------------------------------------------------------------
# Move-on criteria, the AutoKernel family, in AMK's latency / roofline units.
# ---------------------------------------------------------------------------
MOVE_ON_CRITERIA = {
    "consecutive_reverts": 8,        # last N experiments all reverted -> plateau, stop grinding
    "pct_roofline_target": 110.0,    # within 10% of the weights/bandwidth floor (100% == the floor)
    "max_minutes": 600,              # 10h wall-clock budget for one campaign
    "speedup_target": 3.0,           # 3x faster than the default/baseline schedule -> done
}

# Region labels (mirror schedule.cost_model.REGIONS minus 'other').
REGIONS = ("attention", "mlp", "lm_head")

STATUS_OPTIMIZING = "optimizing"
STATUS_DONE = "done"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _ensure_workspace() -> None:
    os.makedirs(WORKSPACE, exist_ok=True)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
def _fresh_state(model: str, gpu: str, state_path: str = STATE_PATH) -> dict[str, Any]:
    return {
        "model": model,
        "gpu": gpu,
        "status": STATUS_OPTIMIZING,
        "started_at": _now_iso(),
        "baseline_us": None,        # the default-config latency (the bar to beat)
        "best_us": None,            # best correct latency so far
        "best_config": None,        # the ScheduleConfig dict of the best
        "best_pct_roofline": None,  # pct_of_roofline of the best (100 == the floor)
        "best_kind": None,          # 'measured-gpu' | 'predicted'
        "speedup": None,            # baseline_us / best_us
        "experiments_run": 0,
        "experiments_kept": 0,
        "consecutive_reverts": 0,
        "region_breakdown": {r: None for r in REGIONS},  # pct of critical path in each region
        "move_on_reason": None,
        "_state_path": state_path,
    }


def load_state(state_path: str = STATE_PATH) -> dict[str, Any] | None:
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "experiments_run" not in data:
            raise ValueError("state file missing required keys")
        data.setdefault("_state_path", state_path)
        return data
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        print(f"WARNING: orchestration state corrupted ({exc}); re-initializing.")
        return None


def save_state(state: dict[str, Any]) -> None:
    state_path = state.get("_state_path", STATE_PATH)
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    tmp = state_path + ".tmp"
    to_write = {k: v for k, v in state.items() if not k.startswith("_")}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_write, f, indent=2)
    os.replace(tmp, state_path)


def get_or_create_state(model: str, gpu: str, state_path: str = STATE_PATH) -> dict[str, Any]:
    """Load an existing campaign for (model, gpu), or start a fresh one. If the existing state is
    for a DIFFERENT (model, gpu), a fresh campaign is started (one state file == one campaign)."""
    state = load_state(state_path)
    if state is not None and state.get("model") == model and state.get("gpu") == gpu:
        return state
    state = _fresh_state(model, gpu, state_path)
    save_state(state)
    return state


# ---------------------------------------------------------------------------
# Move-on logic (AutoKernel _should_move_on, in latency/roofline units)
# ---------------------------------------------------------------------------
def should_move_on(state: dict[str, Any], elapsed_minutes: float = 0.0) -> tuple[bool, str]:
    """Evaluate the move-on / done criteria. Returns (should_stop, reason). Mirrors AutoKernel's
    plateau / near-peak / time-budget / speedup-target ladder."""
    consec = state.get("consecutive_reverts", 0)
    if consec >= MOVE_ON_CRITERIA["consecutive_reverts"]:
        return True, (f"plateau: {consec} consecutive reverts "
                      f"(threshold {MOVE_ON_CRITERIA['consecutive_reverts']})")

    pct = state.get("best_pct_roofline")
    if pct is not None and pct <= MOVE_ON_CRITERIA["pct_roofline_target"]:
        return True, (f"near roofline: best at {pct:.1f}% of the weights/bandwidth floor "
                      f"(target <= {MOVE_ON_CRITERIA['pct_roofline_target']:.0f}%)")

    if elapsed_minutes >= MOVE_ON_CRITERIA["max_minutes"]:
        return True, (f"time budget exhausted: {elapsed_minutes:.0f} min "
                      f"(max {MOVE_ON_CRITERIA['max_minutes']} min)")

    speedup = state.get("speedup")
    if speedup is not None and speedup >= MOVE_ON_CRITERIA["speedup_target"]:
        return True, (f"strong speedup achieved: {speedup:.2f}x vs baseline "
                      f"(target {MOVE_ON_CRITERIA['speedup_target']:.1f}x)")

    return False, "headroom remains: keep optimizing"


# ---------------------------------------------------------------------------
# record, the keep/revert accounting (AutoKernel cmd_record, AMK units)
# ---------------------------------------------------------------------------
def record(state: dict[str, Any], *,
           latency_us: float | None,
           status: str,
           config: dict[str, Any] | None = None,
           correctness: str = "PASS",
           pct_of_roofline: float | None = None,
           latency_kind: str | None = None,
           region_breakdown: dict[str, float] | None = None,
           schedule_id: str = "",
           description: str = "",
           loop: str = "schedule",
           regime: str = "single-stream",
           results_path: str = RESULTS_TSV,
           elapsed_minutes: float = 0.0,
           log_tsv: bool = True) -> dict[str, Any]:
    """Record ONE experiment outcome into the campaign state (and, by default, into results.tsv).

    ``status`` is one of: ``kept`` | ``revert`` | ``failed`` | ``crash`` | ``timeout`` |
    ``rejected``, exactly the AutoKernel vocabulary. ``kept`` resets the plateau counter and may
    set a new best; everything else increments ``consecutive_reverts`` (a plateau signal).
    """
    s = status.strip().lower()
    is_kept = s in ("kept", "keep", "improved")
    is_revert = s in ("revert", "reverted", "slower", "same")
    is_failure = s in ("failed", "fail", "crash", "error", "timeout", "rejected")

    state["experiments_run"] = state.get("experiments_run", 0) + 1

    # The default-config trial establishes the baseline (the bar everything else must beat).
    if state.get("baseline_us") is None and latency_us is not None and correctness == "PASS":
        state["baseline_us"] = float(latency_us)

    if is_kept and latency_us is not None:
        state["experiments_kept"] = state.get("experiments_kept", 0) + 1
        state["consecutive_reverts"] = 0
        if state.get("best_us") is None or float(latency_us) < state["best_us"]:
            state["best_us"] = float(latency_us)
            state["best_config"] = config
            state["best_pct_roofline"] = (float(pct_of_roofline)
                                          if pct_of_roofline is not None else None)
            state["best_kind"] = latency_kind
            if region_breakdown:
                state["region_breakdown"] = {r: region_breakdown.get(r) for r in REGIONS}
    elif is_revert:
        state["consecutive_reverts"] = state.get("consecutive_reverts", 0) + 1
    elif is_failure:
        state["consecutive_reverts"] = state.get("consecutive_reverts", 0) + 1
    else:
        state["consecutive_reverts"] = state.get("consecutive_reverts", 0) + 1

    # speedup vs baseline
    if state.get("baseline_us") and state.get("best_us") and state["best_us"] > 0:
        state["speedup"] = round(state["baseline_us"] / state["best_us"], 3)

    # ---- log EVERY experiment to results.tsv (the flywheel substrate, honest correctness) ----
    if log_tsv:
        tag = "kept" if is_kept else ("revert" if is_revert else s or "revert")
        try:
            append_result(ResultRow(
                experiment=state["experiments_run"], tag=tag, loop=loop,
                model=state.get("model", ""), gpu=state.get("gpu", ""), regime=regime,
                correctness=correctness or ("PASS" if not is_failure else "FAIL"),
                latency_us=latency_us if latency_us is not None else "",
                pct_of_roofline=pct_of_roofline if pct_of_roofline is not None else "",
                schedule_id=schedule_id, kernel_id="",
                description=f"{description}; kind={latency_kind}"[:200]),
                path=results_path)
        except Exception as e:  # results.tsv write must never break the campaign
            print(f"  (results.tsv write skipped: {type(e).__name__}: {e})")

    # ---- move-on / done check ----
    stop, reason = should_move_on(state, elapsed_minutes=elapsed_minutes)
    if stop:
        state["status"] = STATUS_DONE
        state["move_on_reason"] = reason

    save_state(state)
    return state


# ---------------------------------------------------------------------------
# Commands (status / next / report)
# ---------------------------------------------------------------------------
def cmd_status(state: dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print("  AMK Autoresearch Orchestration Status")
    print("=" * 60)
    print(f"  campaign     : {state.get('model')} / {state.get('gpu')}")
    print(f"  status       : {state.get('status', '?').upper()}")
    print(f"  experiments  : {state.get('experiments_run', 0)} "
          f"({state.get('experiments_kept', 0)} kept), "
          f"{state.get('consecutive_reverts', 0)} consecutive reverts")
    base = state.get("baseline_us")
    best = state.get("best_us")
    sp = state.get("speedup")
    if base is not None and best is not None:
        print(f"  baseline     : {base:.3f}us -> best {best:.3f}us"
              + (f"  ({sp:.2f}x faster)" if sp else ""))
    elif base is not None:
        print(f"  baseline     : {base:.3f}us (no improvement yet)")
    pct = state.get("best_pct_roofline")
    if pct is not None:
        print(f"  best roofline: {pct:.1f}% of the weights/bandwidth floor "
              f"(100% == floor; lower is better)  kind={state.get('best_kind')}")
    rb = state.get("region_breakdown") or {}
    shown = ", ".join(f"{r}={rb[r]*100:.0f}%" for r in REGIONS
                      if rb.get(r) is not None)
    if shown:
        print(f"  region split : {shown}  (share of critical path -> where to spend budget)")
    if state.get("move_on_reason"):
        print(f"  move-on      : {state['move_on_reason']}")
    print()


def cmd_next(state: dict[str, Any]) -> None:
    stop, reason = should_move_on(state)
    if state.get("status") == STATUS_DONE or stop:
        if stop and state.get("status") != STATUS_DONE:
            state["status"] = STATUS_DONE
            state["move_on_reason"] = reason
            save_state(state)
        print(f"DECISION: STOP optimizing {state.get('model')}/{state.get('gpu')}")
        print(f"  Reason: {state.get('move_on_reason') or reason}")
        print("  Campaign complete. Run `amk_orchestrate.py report` for the summary.")
        return
    # Suggest where to spend the next experiment: the region with the largest critical-path share.
    rb = state.get("region_breakdown") or {}
    hot = max((r for r in REGIONS if rb.get(r) is not None),
              key=lambda r: rb.get(r, 0.0), default=None)
    print(f"DECISION: CONTINUE optimizing {state.get('model')}/{state.get('gpu')}")
    print(f"  Reason: {reason}")
    print(f"  Experiments run: {state.get('experiments_run', 0)} | "
          f"speedup: {state.get('speedup') or 'N/A'} | "
          f"consecutive reverts: {state.get('consecutive_reverts', 0)}"
          f"/{MOVE_ON_CRITERIA['consecutive_reverts']}")
    if hot is not None:
        print(f"  Hottest region: {hot} ({rb[hot]*100:.0f}% of the critical path) "
              f"-> mutate knobs that touch it (e.g. tiling/fusion/pipelining).")


def cmd_report(state: dict[str, Any]) -> str:
    _ensure_workspace()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    base = state.get("baseline_us")
    best = state.get("best_us")
    sp = state.get("speedup")
    pct = state.get("best_pct_roofline")
    lines = [
        "# AMK, Autoresearch Campaign Report",
        "",
        f"Generated: {ts}",
        "",
        f"- Campaign: **{state.get('model')} / {state.get('gpu')}**",
        f"- Status: **{state.get('status', '?').upper()}**"
        + (f" ({state['move_on_reason']})" if state.get("move_on_reason") else ""),
        f"- Experiments run: {state.get('experiments_run', 0)} "
        f"({state.get('experiments_kept', 0)} kept)",
        f"- Consecutive reverts at stop: {state.get('consecutive_reverts', 0)}",
        "",
        "## Result",
        "",
    ]
    if base is not None and best is not None:
        lines.append(f"- Baseline (default schedule): **{base:.3f} us**")
        lines.append(f"- Best schedule: **{best:.3f} us**"
                     + (f", **{sp:.2f}x** faster than baseline" if sp else ""))
        lines.append(f"- Best latency kind: {state.get('best_kind')}")
        if pct is not None:
            lines.append(f"- Distance to roofline: **{pct:.1f}%** of the "
                         f"weights/bandwidth floor (100% == the floor)")
    else:
        lines.append("- No correct schedule recorded yet.")
    lines.append("")
    rb = state.get("region_breakdown") or {}
    if any(rb.get(r) is not None for r in REGIONS):
        lines.append("## Region breakdown (best schedule's critical path)")
        lines.append("")
        for r in REGIONS:
            if rb.get(r) is not None:
                lines.append(f"- {r}: {rb[r]*100:.1f}%")
        lines.append("")
    lines.append("## Best config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(state.get("best_config"), indent=2))
    lines.append("```")
    lines.append("")
    report = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {REPORT_PATH}")
    print()
    cmd_status(state)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="amk_orchestrate",
                                description="AMK autoresearch campaign orchestrator")
    p.add_argument("--model", default=None, help="model id (for status/next/report on a fresh state)")
    p.add_argument("--gpu", default=None, help="gpu target (for a fresh state)")
    p.add_argument("--state", default=STATE_PATH, help="path to the campaign state json")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show the current campaign state")
    sub.add_parser("next", help="continue-or-stop decision (move-on criteria)")
    rec = sub.add_parser("record", help="record one experiment outcome")
    rec.add_argument("status", help="kept | revert | failed | crash | timeout | rejected")
    rec.add_argument("--latency-us", type=float, default=None)
    rec.add_argument("--pct-roofline", type=float, default=None)
    rec.add_argument("--kind", default=None, help="measured-gpu | predicted")
    rec.add_argument("--correctness", default="PASS")
    rec.add_argument("--schedule-id", default="")
    rec.add_argument("--config", default=None, help="ScheduleConfig JSON file")
    rec.add_argument("--description", default="")
    sub.add_parser("report", help="write + print the aggregate report")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = load_state(args.state)
    if state is None:
        if not (args.model and args.gpu):
            print("ERROR: no campaign state at "
                  f"{args.state}. Start one with --model <m> --gpu <g>.")
            return 1
        state = get_or_create_state(args.model, args.gpu, args.state)
    state["_state_path"] = args.state

    if args.command == "status":
        cmd_status(state)
    elif args.command == "next":
        cmd_next(state)
    elif args.command == "report":
        cmd_report(state)
    elif args.command == "record":
        config = None
        if args.config:
            with open(args.config, encoding="utf-8") as f:
                config = json.load(f)
        record(state, latency_us=args.latency_us, status=args.status, config=config,
               correctness=args.correctness, pct_of_roofline=args.pct_roofline,
               latency_kind=args.kind, schedule_id=args.schedule_id,
               description=args.description)
        cmd_status(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
