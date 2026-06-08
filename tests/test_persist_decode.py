"""
AMK, SINGLE-LAUNCH K-TOKEN PERSISTENT DECODE CONFORMANCE + BENCH
================================================================

THE NOVEL CAPABILITY UNDER TEST: an ENTIRE K-token greedy decode loop run inside ONE persistent
cooperative kernel launch (``vm/scheduler_persist.cu`` :: ``amk_megakernel_persist``, driven by
``vm/loader_persist.py`` :: ``PersistentDecodeVM``), no per-token host relaunch.

ACCEPTANCE (the hard bar): the K tokens the single launch produces are token-for-token IDENTICAL
to the existing per-token-relaunch path (``generate.generate`` driving the baseline
``MegakernelVM``) for the SAME model, prompt, and K. We assert exact equality for K>=8.

We also print single-launch vs per-token-relaunch per-token latency so the overhead ELIMINATED
(host marshalling + launch + sync, paid K times by the relaunch path, ONCE by the persistent loop)
is measured, not asserted (it is hardware/driver dependent; the win is largest where per-token host
overhead dominates and on a no-TDR GPU where K can be large).

Run:  uv run python tests/test_persist_decode.py     (also a pytest module)
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
import torch  # noqa: E402

from generate import generate  # noqa: E402
from schedule.ir import DType, TARGETS  # noqa: E402
from schedule.lower import POS_NAME, TOKEN_NAME, lower  # noqa: E402

_HAS_CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not _HAS_CUDA, reason="persistent decode requires a CUDA device")

K_TOKENS = 12          # >= 8, small enough to stay under the WDDM ~2s TDR locally


# --------------------------------------------------------------------------------------
# Helpers: build a model + its graph/weights, and the per-step program at a base position.
# --------------------------------------------------------------------------------------
def _toy(n_layers: int):
    from models.toy import make_toy
    return make_toy(seed=0, dtype=torch.float32, n_layers=n_layers)


def _graph_weights(model):
    from schedule.graph import from_toy
    return from_toy(model), model.weights_dict()


def _relaunch_tokens_gpu(model, prompt_ids, k):
    """The per-token-relaunch reference on the GPU baseline MegakernelVM, ONE cooperative launch
    PER token (the path the single-launch kernel must equal). Mirrors generate.py's loop exactly,
    but forces the proven register/coalesced GEMV (knobs cpasync=0) so the reference is stable and
    runs the SAME GEMV the persistent kernel does (apples-to-apples)."""
    from schedule.graph import from_toy
    from vm.loader import MegakernelVM
    graph = from_toy(model)
    weights = model.weights_dict()
    target = TARGETS["rtx5090"]
    n_prompt = len(prompt_ids)
    n_steps = n_prompt + k - 1
    kv: dict = {}
    tokens = list(prompt_ids)
    generated: list[int] = []
    pos, cur = 0, tokens[0]
    for step in range(n_steps):
        prog = lower(graph, target=target, config=None, pos=pos, dtype=DType.F32)
        vm = MegakernelVM(prog, weights, device="cuda", knobs={"cpasync": 0})
        ins = {
            TOKEN_NAME: torch.tensor([cur], dtype=torch.int32, device="cuda"),
            POS_NAME: torch.tensor([pos], dtype=torch.int32, device="cuda"),
            "reshape_id0": torch.tensor([0], dtype=torch.int32, device="cuda"),
        }
        out = vm.run(ins, kv=kv)
        kv = {b.name: out[b.name] for b in prog.buffers if b.name in out and "cache" in b.name}
        if step < n_prompt - 1:
            cur = tokens[pos + 1]
        else:
            nxt = int(out["logits"].detach().to(torch.float32).reshape(-1).argmax().item())
            generated.append(nxt)
            tokens.append(nxt)
            cur = nxt
        pos += 1
    return generated


def _relaunch_tokens_cpu(model, prompt_ids, k):
    """The bit-exact ReferenceVM oracle (CPU), driven by the per-token relaunch loop. An
    independent, GPU-free reference for the exact-match acceptance assertion."""
    out = generate(model, "rtx5090", prompt_ids, k, device="cpu", dtype=DType.F32, verify=False)
    return out["generated"]


def _baseline_tokens(model, prompt_ids, k):
    """Reference tokens for the exact-match acceptance test: the GPU per-token-relaunch baseline
    (the path the spec names), with the proven GEMV. The single-launch kernel must equal it."""
    return _relaunch_tokens_gpu(model, prompt_ids, k)


def _prefill_kv(base_vm, prompt_ids, graph, target):
    """Run the baseline per-token path for prompt positions 0..n_prompt-2 to build the KV cache,
    returning (kv_dict, base_pos, first_token) for the generation phase. Mirrors generate.py's
    prefill: the last prompt token is consumed by the FIRST generation step (base_pos=n_prompt-1)."""
    from vm.loader import MegakernelVM
    n_prompt = len(prompt_ids)
    kv: dict[str, torch.Tensor] = {}
    for pos in range(n_prompt - 1):
        prog = lower(graph, target=target, config=None, pos=pos, dtype=DType.F32)
        vm = MegakernelVM(prog, base_vm.weights, device="cuda")
        ins = {
            TOKEN_NAME: torch.tensor([prompt_ids[pos]], dtype=torch.int32, device="cuda"),
            POS_NAME: torch.tensor([pos], dtype=torch.int32, device="cuda"),
            "reshape_id0": torch.tensor([0], dtype=torch.int32, device="cuda"),
        }
        out = vm.run(ins, kv=kv)
        kv = {b.name: out[b.name] for b in prog.buffers
              if b.name in out and "cache" in b.name}
    return kv, n_prompt - 1, prompt_ids[-1]


def _persist_tokens(model, prompt_ids, k):
    """The single-launch path: prefill the prompt KV via baseline, then one persist_launch for K."""
    from vm.loader_persist import PersistentDecodeVM
    graph, weights = _graph_weights(model)
    target = TARGETS["rtx5090"]
    base_pos = len(prompt_ids) - 1
    prog = lower(graph, target=target, config=None, pos=base_pos, dtype=DType.F32)

    pdvm = PersistentDecodeVM(prog, weights, device="cuda")
    kv, base_pos2, first_token = _prefill_kv(pdvm.base_vm, prompt_ids, graph, target)
    assert base_pos2 == base_pos
    res = pdvm.decode(first_token, base_pos, k, kv=kv)
    return res["tokens"]


# --------------------------------------------------------------------------------------
# THE ACCEPTANCE TEST: single-launch K tokens == per-token-relaunch K tokens (exact).
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_layers,prompt", [(1, [7, 11, 3]), (2, [19, 5])])
def test_persist_matches_relaunch_exact(n_layers, prompt):
    model = _toy(n_layers)
    relaunch = _baseline_tokens(model, prompt, K_TOKENS)
    single = _persist_tokens(model, prompt, K_TOKENS)
    assert len(single) == K_TOKENS
    assert single == relaunch, (
        f"toy-{n_layers}L: single-launch tokens != per-token-relaunch tokens.\n"
        f"single   = {single}\nrelaunch = {relaunch}")


# --------------------------------------------------------------------------------------
# BENCH: single-launch vs per-token-relaunch per-token latency (printed, not asserted).
# --------------------------------------------------------------------------------------
def _bench(model, prompt, k, warmup=5, iters=20):
    from schedule.graph import from_toy
    from vm.loader_persist import PersistentDecodeVM
    graph = from_toy(model)
    weights = model.weights_dict()
    target = TARGETS["rtx5090"]
    base_pos = len(prompt) - 1
    prog = lower(graph, target=target, config=None, pos=base_pos, dtype=DType.F32)
    pdvm = PersistentDecodeVM(prog, weights, device="cuda")
    kv, _, first_token = _prefill_kv(pdvm.base_vm, prompt, graph, target)

    # ---- single-launch (K tokens, ONE launch) ----
    for _ in range(warmup):
        pdvm.decode(first_token, base_pos, k, kv=kv)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        pdvm.decode(first_token, base_pos, k, kv=kv)
    torch.cuda.synchronize()
    single_total_us = (time.perf_counter() - t0) * 1e6 / iters
    single_per_tok = single_total_us / k

    # ---- per-token relaunch: K cooperative launches, one per token (STEADY-STATE) ----
    # Build the per-step program + VM ONCE at the generation base position, then relaunch the
    # cooperative kernel K times reusing the persistent device tables (MegakernelVM.relaunch / run
    # steady-state). This is the FAIR lower bound for the relaunch path: it pays only the per-token
    # host counter-zero + launch + sync K times, NOT a re-lower/rebuild per token. The single-launch
    # kernel eliminates exactly that residual per-token host+launch+sync cost.
    from vm.loader import MegakernelVM
    base_vm2 = MegakernelVM(prog, weights, device="cuda", knobs={"cpasync": 0})
    ins = {
        TOKEN_NAME: torch.tensor([first_token], dtype=torch.int32, device="cuda"),
        POS_NAME: torch.tensor([base_pos], dtype=torch.int32, device="cuda"),
        "reshape_id0": torch.tensor([0], dtype=torch.int32, device="cuda"),
    }
    base_vm2.run(ins, kv=kv)            # build persistent tables (also primes steady-state)
    for _ in range(warmup):
        for _s in range(k):
            base_vm2.relaunch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for _s in range(k):
            base_vm2.relaunch()         # one cooperative launch per token, tables reused
    torch.cuda.synchronize()
    relaunch_total_us = (time.perf_counter() - t0) * 1e6 / iters
    relaunch_per_tok = relaunch_total_us / k

    # ---- per-token FULL relaunch (the realistic production-style path): re-lower + rebuild + launch
    # EACH token, exactly as generate.py does. This pays the full host marshalling K times, the cost
    # the single launch eliminates entirely. (Stable GEMV via _relaunch_tokens_gpu's cpasync=0 VM.) --
    fr_iters = max(2, iters // 5)
    for _ in range(2):
        _relaunch_tokens_gpu(model, prompt, k)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(fr_iters):
        _relaunch_tokens_gpu(model, prompt, k)
    torch.cuda.synchronize()
    full_relaunch_per_tok = (time.perf_counter() - t0) * 1e6 / fr_iters / k

    return (single_per_tok, relaunch_per_tok, full_relaunch_per_tok,
            single_total_us, relaunch_total_us)


def test_bench_prints():
    model = _toy(1)
    prompt = [7, 11, 3]
    s_pt, r_pt, fr_pt, s_tot, r_tot = _bench(model, prompt, K_TOKENS)
    print(f"\n== SINGLE-LAUNCH vs PER-TOKEN-RELAUNCH (toy-1L, K={K_TOKENS}) ==")
    print(f"  single-launch (ONE launch, K steps in-kernel) : {s_pt:8.1f} us/tok")
    print(f"  relaunch, steady-state (tables reused)        : {r_pt:8.1f} us/tok  (K cooperative launches)")
    print(f"  relaunch, full (re-lower+rebuild/token, gen)  : {fr_pt:8.1f} us/tok  (K launches, production-style)")
    saved_ss = r_pt - s_pt
    saved_fr = fr_pt - s_pt
    print(f"  vs steady-state relaunch : {saved_ss:+7.1f} us/tok "
          f"({(saved_ss/r_pt*100 if r_pt else 0):+.1f}%)  [launch+sync only, already-marshalled]")
    print(f"  vs full relaunch         : {saved_fr:+7.1f} us/tok "
          f"({(saved_fr/fr_pt*100 if fr_pt else 0):+.1f}%)  [full per-token host overhead ELIMINATED]")
    assert s_pt > 0 and r_pt > 0 and fr_pt > 0


if __name__ == "__main__":
    if not _HAS_CUDA:
        print("no CUDA device, persistent decode test skipped")
        sys.exit(0)
    print("== AMK single-launch K-token persistent decode ==")
    for nl, prompt in [(1, [7, 11, 3]), (2, [19, 5])]:
        m = _toy(nl)
        rel = _baseline_tokens(m, prompt, K_TOKENS)
        sg = _persist_tokens(m, prompt, K_TOKENS)
        ok = sg == rel
        print(f"[toy-{nl}L] single == relaunch : {ok}  "
              f"({'EXACT MATCH' if ok else 'MISMATCH'})")
        print(f"          single   = {sg}")
        print(f"          relaunch = {rel}")
        assert ok, "single-launch tokens must equal per-token-relaunch tokens"

    m = _toy(1)
    s_pt, r_pt, fr_pt, s_tot, r_tot = _bench(m, [7, 11, 3], K_TOKENS)
    print(f"\n== BENCH (toy-1L, K={K_TOKENS}) ==")
    print(f"  single-launch (ONE launch)              : {s_pt:8.1f} us/tok")
    print(f"  relaunch, steady-state (tables reused)  : {r_pt:8.1f} us/tok")
    print(f"  relaunch, full re-lower/token (gen.py)  : {fr_pt:8.1f} us/tok")
    print(f"  eliminated vs steady-state : {r_pt - s_pt:+.1f} us/tok (launch+sync only)")
    print(f"  eliminated vs full relaunch: {fr_pt - s_pt:+.1f} us/tok (full per-token host overhead)")
    print("\nSINGLE-LAUNCH K-TOKEN DECODE VERIFIED: one kernel, K tokens, identical to relaunch.")
