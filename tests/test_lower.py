"""
AMK, M0 GENERALITY PROOF (graph import + lowering, CPU oracle)
==============================================================

Builds a toy Llama, imports it to a graph, lowers ONE decode step (pos=0, fresh/empty KV cache,
kv_len=1 so the token attends to itself), runs the resulting megakernel under the reference VM,
and asserts the logits equal eager ``ToyLlama.forward`` within tolerance. This exercises the
WHOLE decode path end to end on CPU: embed -> RMSNorm -> q/k/v GEMV tiles -> RoPE -> KV append ->
GQA attention -> o_proj -> residual -> post-norm -> SwiGLU -> down -> residual -> final norm ->
lm_head tiles. It also asserts the program validates and has no RACE/CYCLE warning.

A 2-layer variant proves the per-layer wiring composes (residual threading, KV per layer).

Run:  uv run python tests/test_lower.py     (also a pytest module)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from models.toy import make_toy  # noqa: E402
from schedule.graph import from_toy  # noqa: E402
from schedule.ir import DType, TARGETS, ScheduleConfig, validate  # noqa: E402
from schedule.lower import POS_NAME, RESHAPE_ID_NAME, TOKEN_NAME, lower  # noqa: E402
from vm.reference_vm import ReferenceVM  # noqa: E402

RTOL = ATOL = 2e-4


def _run_decode(model, tok: int, pos: int, config: ScheduleConfig | None = None):
    """Lower a decode step at `pos` and run it in the reference VM; return (logits, program)."""
    graph = from_toy(model)
    prog = lower(graph, target=TARGETS["rtx5090"], config=config, pos=pos, dtype=DType.F32)

    res = validate(prog)
    assert res.ok, "lowered program must validate:\n" + res.report()
    bad = [w for w in res.warnings if "RACE" in w or "CYCLE" in w]
    assert not bad, f"validator emitted RACE/CYCLE warnings: {bad}"
    # Dynamic backstop: no stuck tasks (deadlock-freedom).
    _, stuck = prog.simulate_counters()
    assert stuck == [], f"deadlock: stuck tasks {stuck}"
    # NOTE: we intentionally do NOT assert simulate_adversarial()==[] here. That heuristic flags a
    # KV_APPEND reading its OWN cache buffer (the legitimate in-place read of prior-step state) as
    # a "read before write", because it does not special-case the writer reading its own KV cache.
    # The authoritative race-freedom proof is validate() (checked above), which DOES special-case
    # this (schedule.ir line ~838: a KV read by its own writer needs no happens-before edge).

    inputs = {
        TOKEN_NAME: torch.tensor([tok], dtype=torch.int32),
        POS_NAME: torch.tensor([pos], dtype=torch.int32),
        RESHAPE_ID_NAME: torch.tensor([0], dtype=torch.int32),
    }
    out = ReferenceVM(prog, model.weights_dict(), device="cpu").run(inputs, kv={})
    return out["logits"], prog


def _eager_decode_logits(model, tok: int) -> torch.Tensor:
    """Eager oracle: the first-token logits == decode at pos=0 with an empty cache."""
    with torch.no_grad():
        logits = model.forward(torch.tensor([tok]))     # [S=1, vocab]
    return logits[-1].view(1, -1)


def test_lower_one_layer_decode_matches_eager():
    model = make_toy(seed=0, dtype=torch.float32, n_layers=1)
    tok = 7
    logits, prog = _run_decode(model, tok, pos=0)
    eager = _eager_decode_logits(model, tok)
    torch.testing.assert_close(logits, eager, rtol=RTOL, atol=ATOL)
    # sanity on the emitted structure
    assert any(t.op.name == "ATTENTION_TILE" for t in prog.tasks)
    assert any(t.op.name == "ROPE" for t in prog.tasks)
    assert any(t.op.name == "KV_APPEND" for t in prog.tasks)
    assert any(t.op.name == "SILU_MUL" for t in prog.tasks)
    assert sum(t.op.name == "GEMV_TILE" for t in prog.tasks) >= 7  # q,k,v,o,gate,up,down,lm_head


def test_lower_two_layer_decode_matches_eager():
    model = make_toy(seed=0, dtype=torch.float32, n_layers=2)
    tok = 19
    logits, prog = _run_decode(model, tok, pos=0)
    eager = _eager_decode_logits(model, tok)
    torch.testing.assert_close(logits, eager, rtol=RTOL, atol=ATOL)
    # two layers => two attention tiles, two SwiGLUs, two of each norm pair
    assert sum(t.op.name == "ATTENTION_TILE" for t in prog.tasks) == 2
    assert sum(t.op.name == "SILU_MUL" for t in prog.tasks) == 2


def test_tiling_config_changes_task_count_but_not_result():
    """The same graph re-lowered with a different GEMV tile width must stay correct (search-safe)."""
    model = make_toy(seed=1, dtype=torch.float32, n_layers=1)
    tok = 3
    eager = _eager_decode_logits(model, tok)
    cfg_coarse = ScheduleConfig(tiling={"gemv": {"N_tile": 256}})
    cfg_fine = ScheduleConfig(tiling={"gemv": {"N_tile": 16}})
    l_coarse, p_coarse = _run_decode(model, tok, pos=0, config=cfg_coarse)
    l_fine, p_fine = _run_decode(model, tok, pos=0, config=cfg_fine)
    torch.testing.assert_close(l_coarse, eager, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(l_fine, eager, rtol=RTOL, atol=ATOL)
    assert len(p_fine.tasks) > len(p_coarse.tasks), "finer tiling should emit more GEMV tiles"


def test_decode_at_nonzero_pos_matches_prefill_last_token():
    """Generality beyond pos=0: lower a decode at pos=p with a cache pre-filled from a prefill of
    the preceding tokens, and assert it equals the eager prefill's last-token logits. This proves
    RoPE (non-identity at pos>0), the KV window (kv_len=p+1), and GQA all fire correctly."""
    model = make_toy(seed=2, dtype=torch.float32, n_layers=2)
    seq = [5, 42, 200, 13]                 # prefill these; decode predicts from the last one
    p = len(seq) - 1                       # decode position of the last token
    with torch.no_grad():
        eager = model.forward(torch.tensor(seq))[-1].view(1, -1)

    # Build the KV cache by eager-prefilling the first p tokens, replicating the toy's per-layer
    # roped k / v exactly (this is the host's job between decode steps).
    kv = _prefill_kv(model, seq[:p])       # {f'L{L}.kcache': tensor, f'L{L}.vcache': tensor}

    graph = from_toy(model)
    prog = lower(graph, target=TARGETS["rtx5090"], pos=p, dtype=DType.F32)
    assert validate(prog).ok
    inputs = {
        TOKEN_NAME: torch.tensor([seq[p]], dtype=torch.int32),
        POS_NAME: torch.tensor([p], dtype=torch.int32),
        RESHAPE_ID_NAME: torch.tensor([0], dtype=torch.int32),
    }
    out = ReferenceVM(prog, model.weights_dict(), device="cpu").run(inputs, kv=kv)
    torch.testing.assert_close(out["logits"], eager, rtol=RTOL, atol=ATOL)


def _prefill_kv(model, tokens: list[int]) -> dict[str, torch.Tensor]:
    """Compute the per-layer roped-K / V caches for `tokens` by running the toy layer-by-layer,
    mirroring ToyAttention exactly. Returns a kv dict keyed to match the lowered cache buffers
    (``L{L}.kcache`` / ``L{L}.vcache``)."""
    from models.toy import _rope, _rmsnorm
    cfg = model.cfg
    S = len(tokens)
    pos = torch.arange(S)
    x = model.embed(torch.tensor(tokens))
    kv: dict[str, torch.Tensor] = {}
    for L, layer in enumerate(model.layers):
        a = layer.attn
        xn = _rmsnorm(x, layer.input_norm, cfg.rms_eps)
        k = _rope(a.k_proj(xn).view(S, cfg.n_kv_heads, cfg.head_dim), pos, cfg.head_dim, cfg.rope_theta)
        v = a.v_proj(xn).view(S, cfg.n_kv_heads, cfg.head_dim)
        kc = torch.zeros(cfg.max_seq, cfg.n_kv_heads, cfg.head_dim)
        vc = torch.zeros(cfg.max_seq, cfg.n_kv_heads, cfg.head_dim)
        kc[:S] = k
        vc[:S] = v
        kv[f"L{L}.kcache"] = kc
        kv[f"L{L}.vcache"] = vc
        # advance the residual stream through the FULL layer so the next layer's cache is right
        x = layer(x, pos)
    return kv


if __name__ == "__main__":
    test_lower_one_layer_decode_matches_eager()
    print("[1/4] 1-layer decode (pos=0) == eager ToyLlama ............. OK")
    test_lower_two_layer_decode_matches_eager()
    print("[2/4] 2-layer decode (pos=0) == eager ToyLlama ............. OK")
    test_tiling_config_changes_task_count_but_not_result()
    print("[3/4] tiling config re-lowers correctly (search-safe) ..... OK")
    test_decode_at_nonzero_pos_matches_prefill_last_token()
    print("[4/4] decode at pos>0 with prefilled KV == eager .......... OK")
    print("\nM0 GENERALITY PROOF PASSED (attention+rope+kv+swiglu+tiling, CPU oracle).")
