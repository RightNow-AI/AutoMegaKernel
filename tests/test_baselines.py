"""
ACCEPTANCE TEST for the AMK per-token DECODE baselines (eval/baselines per-token comparison).
=============================================================================================

The M2 thesis needs apples-to-apples evidence: the SAME model + SAME GPU, AMK's megakernel
decode/token vs eager PyTorch's per-op decode step, both CORRECTNESS-GATED (no latency without a
PASS). This file proves the harness does exactly that, and that vLLM is recorded honestly (it is
not runnable on this Windows dev box, so status='not_run' with a real Linux command).

Asserts (the module's acceptance criteria):
  1. eager_decode_baseline RUNS and returns a real ms/token (status='ok', correctness='PASS',
     ms_per_token > 0).
  2. amk_decode_baseline RUNS on CUDA and returns a real ms/token, correctness-gated against eager
     (status='ok' only if the megakernel logits matched eager). On CPU it is status='error' with a
     clear reason (the megakernel is CUDA-only; CPU has only the ReferenceVM oracle).
  3. NO latency without a pass: both baselines have latency_us is None unless correctness == 'PASS'.
  4. decode_comparison() returns the {eager, amk, vllm} table; vllm is 'not_run' with a command.
  5. The comparison table prints (bench_baselines.print_table) without raising.

Run:  uv run python tests/test_baselines.py     (also a pytest module)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from eval.baselines import (  # noqa: E402
    BaselineRecord, amk_decode_baseline, decode_comparison, eager_decode_baseline,
    vllm_decode_baseline,
)
from models.toy import make_toy  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Small but non-trivial toy: 2 layers, real GQA, so the GEMV/attention work is meaningful while
# staying well under the WDDM TDR watchdog. Same shape the CLI's --model toy uses.
TOY_KW = dict(n_layers=2, hidden=256, intermediate=512, n_heads=8, n_kv_heads=4,
              head_dim=32, vocab=512, max_seq=512)
CTX = 8
WARMUP, ITERS = 3, 10


def _toy():
    return make_toy(seed=0, dtype=torch.float32, **TOY_KW)


# ======================================================================================
# (1) eager decode baseline runs and returns a real ms/token
# ======================================================================================
def test_eager_decode_runs_and_returns_ms_per_token():
    rec = eager_decode_baseline(_toy(), device=DEVICE, context_len=CTX, warmup=WARMUP, iters=ITERS)
    assert isinstance(rec, BaselineRecord)
    assert rec.status == "ok", rec.report()
    assert rec.correctness == "PASS", rec.report()
    assert rec.latency_us is not None and rec.latency_us > 0, rec.report()
    assert rec.ms_per_token is not None and rec.ms_per_token > 0
    assert rec.tokens_per_s is not None and rec.tokens_per_s > 0
    # is_real_perf is True iff we measured on CUDA (CPU is a labelled reference timing).
    assert rec.is_real_perf == (DEVICE == "cuda")
    print("[1] eager-decode ...", rec.grep_line())


# ======================================================================================
# (2) AMK decode baseline runs (CUDA) / is honest on CPU, correctness-gated
# ======================================================================================
def test_amk_decode_runs_or_is_honest_on_cpu():
    rec = amk_decode_baseline(_toy(), "rtx5090", device=DEVICE, context_len=CTX,
                              warmup=WARMUP, iters=ITERS)
    assert isinstance(rec, BaselineRecord)
    if DEVICE == "cuda":
        # On GPU the megakernel must run, match eager (the gate), and yield a real ms/token.
        assert rec.status == "ok", rec.report()
        assert rec.correctness == "PASS", rec.report()
        assert rec.latency_us is not None and rec.latency_us > 0, rec.report()
        assert rec.ms_per_token is not None and rec.ms_per_token > 0
        assert rec.tokens_per_s is not None and rec.tokens_per_s > 0
        # roofline % is attached (steady-state latency vs HBM weight-streaming bound).
        assert rec.pct_of_roofline is not None and rec.pct_of_roofline > 0, rec.report()
        assert rec.extra.get("weight_bytes", 0) > 0
        print("[2] amk-decode (cuda) ...", rec.grep_line(),
              f"| %roofline={rec.pct_of_roofline:.1f} kernel_only_us={rec.extra.get('kernel_only_us')}")
    else:
        # On CPU there is NO megakernel perf number, must be an honest error, never a fake latency.
        assert rec.status == "error", rec.report()
        assert rec.latency_us is None
        assert "CPU" in rec.note or "cuda" in rec.note.lower()
        print("[2] amk-decode (cpu) honestly refuses ...", rec.grep_line())


# ======================================================================================
# (3) NO latency without a correctness pass (the cardinal honesty rule)
# ======================================================================================
def test_no_latency_without_pass():
    # Every record we produce: a non-PASS correctness => latency_us is None. (We cannot easily
    # force a wrong megakernel here, so we assert the invariant holds on the records we DO produce.)
    for rec in (eager_decode_baseline(_toy(), device=DEVICE, context_len=CTX,
                                      warmup=WARMUP, iters=ITERS),
                amk_decode_baseline(_toy(), "rtx5090", device=DEVICE, context_len=CTX,
                                    warmup=WARMUP, iters=ITERS)):
        if rec.correctness != "PASS":
            assert rec.latency_us is None, (
                f"{rec.name}: reported a latency for a non-PASS verdict, honesty violation\n"
                + rec.report())
        if rec.status == "ok":
            assert rec.correctness == "PASS" and rec.latency_us is not None
    print("[3] no latency without a correctness pass ... OK")


# ======================================================================================
# (4) decode_comparison table + vLLM honest not_run
# ======================================================================================
def test_decode_comparison_table_and_vllm_not_run():
    table = decode_comparison(_toy(), "rtx5090", device=DEVICE, context_len=CTX,
                              warmup=WARMUP, iters=ITERS)
    assert set(table) == {"eager", "amk", "vllm"}
    assert table["eager"].status == "ok" and table["eager"].latency_us is not None

    if DEVICE == "cuda":
        assert table["amk"].status == "ok" and table["amk"].latency_us is not None
        # the honest wall-clock ratio is attached when both ran.
        assert "speedup_vs_eager" in table["amk"].extra
        assert table["amk"].extra["speedup_vs_eager"] > 0

    # vLLM: NEVER fabricated. Not runnable on this Windows box => not_run with a real command.
    vllm = table["vllm"]
    assert vllm.status == "not_run", vllm.report()
    assert vllm.latency_us is None
    assert vllm.command, "a not_run vLLM record must carry the exact command to run it on Linux"
    assert "vllm" in vllm.command.lower()

    # The standalone vllm helper agrees.
    v2 = vllm_decode_baseline("toy")
    assert v2.status == "not_run" and v2.latency_us is None and v2.command
    print("[4] decode_comparison table + vLLM not_run ...", table["eager"].grep_line())


# ======================================================================================
# (5) the comparison table actually prints
# ======================================================================================
def test_table_prints():
    from eval.bench_baselines import print_table
    table = decode_comparison(_toy(), "rtx5090", device=DEVICE, context_len=CTX,
                              warmup=WARMUP, iters=ITERS)
    # must not raise; renders eager/amk/vllm rows + interpretation + grep lines.
    print_table("toy", "rtx5090", DEVICE, table)
    print("[5] comparison table prints ... OK")


# ======================================================================================
if __name__ == "__main__":
    print(f"device = {DEVICE}\n")
    test_eager_decode_runs_and_returns_ms_per_token()
    test_amk_decode_runs_or_is_honest_on_cpu()
    test_no_latency_without_pass()
    test_decode_comparison_table_and_vllm_not_run()
    test_table_prints()
    print("\nALL BASELINE ACCEPTANCE TESTS PASSED")
