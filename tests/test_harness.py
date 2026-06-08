"""
AMK, ACCEPTANCE TEST for the coding-agent harness (Loop 2: schedule search).

Run:  uv run python tests/test_harness.py

Asserts, for the toy model:
  * propose() returns a config dict + a non-empty search_space.
  * evaluate() with the default config -> valid=True, correct=True, finite latency + pct_of_roofline.
  * evaluate() with a deliberately BROKEN config (impossible tiling) -> valid=False with a
    rejected_reason, DOES NOT crash, and emits NO latency.
  * loop(budget=8) runs, logs >= 1 results.tsv row, and returns a best verdict with correct=True.
  * CLI smoke: `uv run python amk_cli.py eval toy --gpu rtx5090` prints valid JSON.

This test is GPU-aware but GPU-optional: on CUDA the latency_kind may be 'measured-gpu'; on CPU
(or when the end-to-end GPU path is unavailable) it falls back to 'predicted'. Either way a
latency only ever exists alongside correct=True.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness  # noqa: E402

GPU = "rtx5090"
MODEL = "toy"


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_propose() -> None:
    print("[test_propose]")
    out = harness.propose(MODEL, GPU)
    _check(isinstance(out, dict), "propose returns a dict")
    _check(isinstance(out.get("schedule_config"), dict) and out["schedule_config"],
           "propose returns a non-empty schedule_config dict")
    _check(isinstance(out.get("search_space"), dict) and len(out["search_space"]) > 0,
           f"propose returns a non-empty search_space ({len(out.get('search_space', {}))} knobs)")
    # the config must round-trip through the IR edit surface
    for k in ("tiling", "fusion_grouping", "sm_assignment", "pipelining_depth",
              "page_allocation", "threads_per_block", "smem_bytes_per_block"):
        _check(k in out["schedule_config"], f"schedule_config has knob '{k}'")
    _check(out["schedule_id"].startswith("sch_"), "propose returns a schedule_id")


def test_evaluate_default() -> None:
    print("[test_evaluate_default]")
    default_cfg = harness.propose(MODEL, GPU)["schedule_config"]
    v = harness.evaluate(MODEL, GPU, default_cfg)
    print("  verdict:", json.dumps({k: v[k] for k in
          ("valid", "correct", "max_abs_err", "latency_us", "latency_kind",
           "pct_of_roofline", "bound_us", "tasks", "weight_mb")}, default=str))
    _check(v["valid"] is True, "default config is VALID")
    _check(v["correct"] is True, "default config is CORRECT vs eager")
    _check(v["latency_us"] is not None and math.isfinite(v["latency_us"]) and v["latency_us"] > 0,
           f"finite positive latency ({v['latency_us']}us, kind={v['latency_kind']})")
    _check(v["latency_kind"] in ("measured-gpu", "predicted"),
           f"latency_kind is honest ({v['latency_kind']})")
    _check(v["pct_of_roofline"] is not None and math.isfinite(v["pct_of_roofline"]),
           f"finite pct_of_roofline ({v['pct_of_roofline']}%)")
    _check(v["tasks"] > 0 and v["weight_mb"] > 0, "program has tasks + weights")


def test_evaluate_broken() -> None:
    print("[test_evaluate_broken]")
    # A deliberately impossible config: a non-multiple-of-32, zero-ish threads_per_block plus an
    # absurd SMEM opt-in far above the target cap. The lowerer/loader+validate must reject cleanly.
    broken = harness.propose(MODEL, GPU)["schedule_config"]
    broken["threads_per_block"] = 7                  # not a valid block size
    broken["smem_bytes_per_block"] = 10 ** 9         # way over any target opt-in cap
    broken["tiling"] = {"gemv": {"N_tile": -16}}     # impossible (negative) tile
    v = harness.evaluate(MODEL, GPU, broken)
    print("  verdict:", json.dumps({k: v.get(k) for k in
          ("valid", "rejected_reason", "correct", "latency_us", "latency_kind")}, default=str))
    _check(v["valid"] is False, "broken config is REJECTED (valid=False)")
    _check(bool(v.get("rejected_reason")), f"rejected_reason is set: {v.get('rejected_reason')!r}")
    _check(v.get("latency_us") is None, "NO latency emitted for a rejected config")
    _check(v.get("latency_kind") is None, "NO latency_kind for a rejected config")
    _check(v.get("correct") in (False, None), "no correctness claim for a rejected config")


def test_evaluate_never_crashes() -> None:
    print("[test_evaluate_never_crashes]")
    # garbage shapes / types in the edit surface must come back as clean verdicts, not exceptions.
    for bad in (
        {"tiling": "not-a-dict"},
        {"pipelining_depth": "deep"},
        {"sm_assignment": {"99999": 9999}},          # explicit map onto out-of-range SMs
        {"threads_per_block": 0},
        {},
    ):
        try:
            v = harness.evaluate(MODEL, GPU, bad)
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"evaluate crashed on {bad!r}: {type(e).__name__}: {e}")
        _check(isinstance(v, dict) and "valid" in v,
               f"evaluate returns a clean verdict for {bad!r} (valid={v.get('valid')})")
        if not v["valid"]:
            _check(v.get("latency_us") is None, f"no latency for invalid {bad!r}")


def test_loop() -> None:
    print("[test_loop]")
    with tempfile.TemporaryDirectory() as td:
        results = os.path.join(td, "results.tsv")
        corpus = os.path.join(td, "corpus.jsonl")
        out = harness.loop(MODEL, GPU, budget=8, results_path=results, corpus_path=corpus,
                           seed=0, verbose=True)
        _check(out["n_trials"] == 8, "loop ran 8 trials")
        _check(out["n_valid"] >= 1, f"loop found >= 1 valid config ({out['n_valid']})")
        _check(out["best_verdict"] is not None, "loop returned a best verdict")
        _check(out["best_verdict"]["correct"] is True, "best verdict is CORRECT")
        _check(out["best_verdict"]["latency_us"] is not None,
               f"best verdict has a latency ({out['best_verdict']['latency_us']}us)")
        # results.tsv must have a header + >= 1 data row
        _check(os.path.exists(results), "loop wrote results.tsv")
        with open(results, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        _check(len(lines) >= 2, f"results.tsv has a header + >= 1 row ({len(lines) - 1} rows)")
        # every logged row carries a non-blank correctness (the honesty rule)
        header = lines[0].split("\t")
        ci = header.index("correctness")
        for ln in lines[1:]:
            cells = ln.split("\t")
            _check(bool(cells[ci]), "every results.tsv row has a correctness verdict")


def test_cli_eval_smoke() -> None:
    print("[test_cli_eval_smoke]")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [sys.executable, os.path.join(root, "amk_cli.py"), "eval", MODEL, "--gpu", GPU],
        cwd=root, capture_output=True, text=True, timeout=600)
    print("  exit:", proc.returncode)
    if proc.returncode not in (0, 1):  # 0=correct, 1=incorrect/rejected; both are "ran ok"
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
    _check(proc.returncode in (0, 1), f"CLI eval ran (exit {proc.returncode})")
    # stdout must be valid JSON with the verdict schema
    verdict = json.loads(proc.stdout)
    _check(isinstance(verdict, dict) and "valid" in verdict and "latency_kind" in verdict,
           "CLI eval prints a valid JSON verdict")
    _check(verdict["valid"] is True and verdict["correct"] is True,
           "CLI eval default verdict is valid + correct")


def main() -> int:
    test_propose()
    test_evaluate_default()
    test_evaluate_broken()
    test_evaluate_never_crashes()
    test_loop()
    test_cli_eval_smoke()
    print("\nALL HARNESS ACCEPTANCE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
