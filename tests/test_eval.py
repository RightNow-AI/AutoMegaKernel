"""
ACCEPTANCE TEST for the AMK evaluation harness (eval/oracle, bench, roofline, baselines).
========================================================================================

The harness is the *fixed eval* of the autoresearch loop, so it must be exercised end-to-end
with a real ``run_fn`` that is "the toy model + ReferenceVM", not a mock. To get that without
depending on the (separate) graph lowerer, this file contains a small, self-contained, **valid**
lowering of one ToyLlama forward pass (single token, S=1) into a :class:`MegakernelProgram`, then
runs it under :class:`ReferenceVM`. That gives a genuine counter-driven, validator-accepted
megakernel whose logits we compare against eager ``ToyLlama.forward``, exactly the contract the
real compiler will satisfy.

Asserts (the spec's acceptance criteria):
  1. oracle.logit_equivalence on IDENTICAL tensors => correct=True, top1=1.0;
     on PERTURBED tensors => sensible (larger) errors and a FAIL at fp32 tolerance.
     Plus: the ReferenceVM-lowered toy logits equal eager (the real run_fn is correct), and
     token_divergence reports a perfect (n_tokens) match for the correct run_fn.
  2. bench REFUSES a latency when the verdict is FAIL (raises in strict mode AND marks FAIL with
     latency=None in non-strict mode), and DOES return a latency when correct.
  3. roofline.report yields bound_us > 0 and pct_of_bound >= 100 for measured >= bound.
  4. the eager baseline RUNS and returns a real latency; vLLM / SGLang / MPK return status='not_run'.

Run:  uv run python tests/test_eval.py     (also a pytest module)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from eval.baselines import (  # noqa: E402
    all_baselines, eager_baseline, mpk_baseline, sglang_baseline, vllm_baseline,
)
from eval.bench import CorrectnessGateError, bench  # noqa: E402
from eval.oracle import logit_equivalence, token_divergence  # noqa: E402
from eval.roofline import report as roofline_report  # noqa: E402
from models.toy import ToyConfig, make_toy  # noqa: E402
from schedule.ir import (  # noqa: E402
    BufferKind, DType, InstructionKind, MegakernelProgram, TARGETS, Wait,
)
from vm.reference_vm import ReferenceVM  # noqa: E402

DT = DType.F32
TDT = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Single attention head so the flat projection row [1, q_dim] coincides with the head-structured
# query [n_heads, head_dim] the reference ATTENTION_TILE op consumes (see build_toy_program).
TOY_KW = dict(n_layers=1, n_heads=1, n_kv_heads=1, head_dim=64, hidden=64, intermediate=128)


# ======================================================================================
# A self-contained, VALID single-token lowering of one ToyLlama forward pass.
# (This is a test fixture, NOT the production lowerer, it only needs to be correct & valid.)
# ======================================================================================
def _tiled_gemv(p, x_buf, w_buf, out_buf, K, N, n_tiles, wait, counter, label):
    """n_tiles GEMV_TILE tasks over disjoint output columns, all incrementing `counter`
    (a true all-join: the consumer waits threshold == n_tiles). Mirrors verify_vm._tiled_gemv."""
    tile = (N + n_tiles - 1) // n_tiles
    emitted = 0
    for i in range(n_tiles):
        n_off = i * tile
        n_tile = min(tile, N - n_off)
        if n_tile <= 0:
            break
        p.add_task(InstructionKind.GEMV_TILE, [x_buf, w_buf], [out_buf], out_counter=counter,
                   waits=list(wait), params={"K": K, "N_tile": n_tile, "n_off": n_off},
                   label=f"{label}[t{i}]", est_bytes=K * n_tile * 4, est_flops=2 * K * n_tile)
        emitted += 1
    return emitted


def build_toy_program(cfg: ToyConfig, position: int = 0, gemv_tiles: int = 2) -> MegakernelProgram:
    """Lower a single ToyLlama layer + embed + final norm + lm_head for ONE token at position
    ``position`` (S=1) into a validator-accepted MegakernelProgram. Weight buffer `source` keys
    match ``ToyLlama.weights_dict()`` so a ReferenceVM can bind them directly.

    NOTE (fixture, not the production lowerer): this single-token decode lowering uses the
    single-query reference ATTENTION_TILE op, which consumes the query as ``[n_heads, head_dim]``.
    The q/k projection GEMV emits a flat ``[1, q_dim]`` row, and ``[1, q_dim] == [n_heads, head_dim]``
    only when ``n_heads == 1`` (so the layouts coincide with no reshape op needed). The test
    therefore configures the toy model with a single attention head; that exercises the full
    RMSNorm+RoPE+attention+SwiGLU+lm_head pipeline through the counter-driven VM without needing a
    reshape/gather primitive the reference op library does not expose. ``pos`` lets the harness
    advance RoPE across decode steps so a real multi-step greedy decode is position-correct.
    """
    assert cfg.n_heads == 1 and cfg.n_kv_heads == 1, (
        "this single-token fixture requires n_heads==n_kv_heads==1 so [1,q_dim]==[n_heads,head_dim]")
    H, V = cfg.hidden, cfg.vocab
    NH, NKV, HD, I = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate  # noqa: E741
    QD, KVD = cfg.q_dim, cfg.kv_dim
    p = MegakernelProgram(meta={"model": "toy", "gpu": "rtx5090"}, target=TARGETS["rtx5090"])

    def B(name, kind, shape, source=None):
        return p.new_buffer(name, kind, DT, tuple(shape), source=source).id

    # ---- IO + weights ----
    ids = B("input_ids", BufferKind.IO_INPUT, (1,))
    pos = B("pos", BufferKind.IO_INPUT, (1,))  # position id buffer for RoPE (value supplied at run)
    embed = B("embed.weight", BufferKind.WEIGHT, (V, H), source="embed.weight")
    in_norm = B("layers.0.input_norm", BufferKind.WEIGHT, (H,), source="layers.0.input_norm")
    post_norm = B("layers.0.post_norm", BufferKind.WEIGHT, (H,), source="layers.0.post_norm")
    qw = B("q", BufferKind.WEIGHT, (QD, H), source="layers.0.attn.q_proj.weight")
    kw = B("k", BufferKind.WEIGHT, (KVD, H), source="layers.0.attn.k_proj.weight")
    vw = B("v", BufferKind.WEIGHT, (KVD, H), source="layers.0.attn.v_proj.weight")
    ow = B("o", BufferKind.WEIGHT, (H, QD), source="layers.0.attn.o_proj.weight")
    gw = B("gate", BufferKind.WEIGHT, (I, H), source="layers.0.mlp.gate_proj.weight")
    uw = B("up", BufferKind.WEIGHT, (I, H), source="layers.0.mlp.up_proj.weight")
    dw = B("down", BufferKind.WEIGHT, (H, I), source="layers.0.mlp.down_proj.weight")
    fnorm = B("final_norm", BufferKind.WEIGHT, (H,), source="final_norm")
    lmw = B("lm_head.weight", BufferKind.WEIGHT, (V, H), source="lm_head.weight")

    # ---- activations / KV / output ----
    x = B("x", BufferKind.ACTIVATION, (1, H))
    xn = B("xn", BufferKind.ACTIVATION, (1, H))
    q = B("q_act", BufferKind.ACTIVATION, (1, QD))
    k = B("k_act", BufferKind.ACTIVATION, (1, KVD))
    v = B("v_act", BufferKind.ACTIVATION, (1, KVD))
    qr = B("qr", BufferKind.ACTIVATION, (NH, HD))      # NH==1 => (1, head_dim) == flat q row
    kr = B("kr", BufferKind.ACTIVATION, (1, NKV, HD))  # one new key row, head-structured
    kcache = B("kcache", BufferKind.KV_CACHE, (cfg.max_seq, NKV, HD))
    vcache = B("vcache", BufferKind.KV_CACHE, (cfg.max_seq, NKV, HD))
    attn = B("attn", BufferKind.ACTIVATION, (1, QD))
    ao = B("ao", BufferKind.ACTIVATION, (1, H))
    h1 = B("h1", BufferKind.ACTIVATION, (1, H))     # residual after attn
    h1n = B("h1n", BufferKind.ACTIVATION, (1, H))
    g = B("g", BufferKind.ACTIVATION, (1, I))
    u = B("u", BufferKind.ACTIVATION, (1, I))
    act = B("act", BufferKind.ACTIVATION, (1, I))
    d = B("d", BufferKind.ACTIVATION, (1, H))
    h2 = B("h2", BufferKind.ACTIVATION, (1, H))     # residual after mlp
    h2n = B("h2n", BufferKind.ACTIVATION, (1, H))
    logits = B("logits", BufferKind.IO_OUTPUT, (1, V))

    C = lambda note="": p.new_counter(note).id  # noqa: E731

    # ---- embed ----
    c_emb = C("embed")
    p.add_task(InstructionKind.EMBED, [ids, embed], [x], out_counter=c_emb,
               params={"hidden": H}, label="embed")

    # ---- attn: rmsnorm ----
    c_inorm = C("input_norm")
    p.add_task(InstructionKind.RMSNORM, [x, in_norm], [xn], out_counter=c_inorm,
               waits=[Wait(c_emb, 1)], params={"eps": cfg.rms_eps, "hidden": H}, label="input_norm")

    # ---- q/k/v projections (tiled gemv) ----
    c_q = C("q_proj")
    nq = _tiled_gemv(p, xn, qw, q, K=H, N=QD, n_tiles=gemv_tiles, wait=[Wait(c_inorm, 1)],
                     counter=c_q, label="q_proj")
    c_k = C("k_proj")
    nk = _tiled_gemv(p, xn, kw, k, K=H, N=KVD, n_tiles=min(gemv_tiles, KVD), wait=[Wait(c_inorm, 1)],
                     counter=c_k, label="k_proj")
    c_v = C("v_proj")
    nv = _tiled_gemv(p, xn, vw, v, K=H, N=KVD, n_tiles=min(gemv_tiles, KVD), wait=[Wait(c_inorm, 1)],
                     counter=c_v, label="v_proj")

    # ---- rope q & k ----
    c_qr = C("rope_q")
    p.add_task(InstructionKind.ROPE, [q, pos], [qr], out_counter=c_qr, waits=[Wait(c_q, nq)],
               params={"head_dim": HD, "theta": cfg.rope_theta}, label="rope_q")
    c_kr = C("rope_k")
    p.add_task(InstructionKind.ROPE, [k, pos], [kr], out_counter=c_kr, waits=[Wait(c_k, nk)],
               params={"head_dim": HD, "theta": cfg.rope_theta}, label="rope_k")

    # ---- kv append at `position`, writers of the caches this pass ----
    c_ka = C("kv_append_k")
    p.add_task(InstructionKind.KV_APPEND, [kr, kcache], [kcache], out_counter=c_ka,
               waits=[Wait(c_kr, 1)], params={"pos": position}, label="kv_append_k")
    c_va = C("kv_append_v")
    p.add_task(InstructionKind.KV_APPEND, [v, vcache], [vcache], out_counter=c_va,
               waits=[Wait(c_v, nv)], params={"pos": position}, label="kv_append_v")

    # ---- attention over the live KV window [0, position] (causal: this query sees all cached keys
    # up to and including its own freshly-appended row). Reads qr (happens-after rope_q) and the
    # caches (happens-after their KV_APPENDs). All three happens-before edges wired. ----
    c_attn = C("attention")
    p.add_task(InstructionKind.ATTENTION_TILE, [qr, kcache, vcache], [attn], out_counter=c_attn,
               waits=[Wait(c_qr, 1), Wait(c_ka, 1), Wait(c_va, 1)],
               params={"head_dim": HD, "kv_start": 0, "kv_len": position + 1, "scale": HD ** -0.5,
                       "n_heads": NH, "n_kv_heads": NKV, "flags": 1}, label="attention")

    # ---- o_proj + residual ----
    c_o = C("o_proj")
    no = _tiled_gemv(p, attn, ow, ao, K=QD, N=H, n_tiles=gemv_tiles, wait=[Wait(c_attn, 1)],
                     counter=c_o, label="o_proj")
    c_res1 = C("attn_residual")
    p.add_task(InstructionKind.ADD, [ao, x], [h1], out_counter=c_res1,
               waits=[Wait(c_o, no), Wait(c_emb, 1)], label="attn_residual")

    # ---- mlp ----
    c_pnorm = C("post_norm")
    p.add_task(InstructionKind.RMSNORM, [h1, post_norm], [h1n], out_counter=c_pnorm,
               waits=[Wait(c_res1, 1)], params={"eps": cfg.rms_eps, "hidden": H}, label="post_norm")
    c_g = C("gate")
    ng = _tiled_gemv(p, h1n, gw, g, K=H, N=I, n_tiles=gemv_tiles * 2, wait=[Wait(c_pnorm, 1)],
                     counter=c_g, label="gate")
    c_u = C("up")
    nu = _tiled_gemv(p, h1n, uw, u, K=H, N=I, n_tiles=gemv_tiles * 2, wait=[Wait(c_pnorm, 1)],
                     counter=c_u, label="up")
    c_act = C("swiglu")
    p.add_task(InstructionKind.SILU_MUL, [g, u], [act], out_counter=c_act,
               waits=[Wait(c_g, ng), Wait(c_u, nu)], label="swiglu")
    c_d = C("down")
    nd = _tiled_gemv(p, act, dw, d, K=I, N=H, n_tiles=gemv_tiles * 2, wait=[Wait(c_act, 1)],
                     counter=c_d, label="down")
    c_res2 = C("mlp_residual")
    p.add_task(InstructionKind.ADD, [d, h1], [h2], out_counter=c_res2,
               waits=[Wait(c_d, nd), Wait(c_res1, 1)], label="mlp_residual")

    # ---- final norm + lm_head ----
    c_fnorm = C("final_norm")
    p.add_task(InstructionKind.RMSNORM, [h2, fnorm], [h2n], out_counter=c_fnorm,
               waits=[Wait(c_res2, 1)], params={"eps": cfg.rms_eps, "hidden": H}, label="final_norm")
    c_lm = C("lm_head")
    nl = _tiled_gemv(p, h2n, lmw, logits, K=H, N=V, n_tiles=gemv_tiles * 4, wait=[Wait(c_fnorm, 1)],
                     counter=c_lm, label="lm_head")
    # final all-join is implicit: logits is IO_OUTPUT produced by nl tiles on c_lm.
    assert nl > 0
    return p


def make_run_fn(model, device=DEVICE):
    """Return run_fn(input_ids[S]) -> logits[1, V] using ReferenceVM over the lowered toy program.

    Faithful to the frozen decode model ("one program == one forward pass == one token; KV persists
    across passes in HBM"): the run_fn streams the WHOLE input sequence through the VM token by
    token, appending each token's k/v into a persistent KV cache and advancing the RoPE/append
    position, then returns the logits at the final position. For the single-head causal toy model
    this sequential decode is numerically identical to eager ``ToyLlama.forward(seq)``, which is
    exactly what makes this a legitimate ReferenceVM-backed run_fn for the oracle.

    One MegakernelProgram is built per position (the only thing that changes is the RoPE position
    and the attention window length ``kv_len = pos+1``); they are cached by position.
    """
    cfg = model.cfg
    weights = model.weights_dict()
    prog_cache: dict[int, MegakernelProgram] = {}
    vm_cache: dict[int, ReferenceVM] = {}

    def vm_for(pos: int) -> ReferenceVM:
        if pos not in vm_cache:
            prog = build_toy_program(cfg, position=pos)
            prog_cache[pos] = prog
            vm_cache[pos] = ReferenceVM(prog, weights, device=device)
        return vm_cache[pos]

    def run_fn(input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids.detach().to(torch.long).view(-1)
        kv: dict[str, torch.Tensor] = {}            # persistent KV across the sequence
        logits = None
        for pos in range(ids.shape[0]):
            tok = ids[pos:pos + 1].to(device)
            out = vm_for(pos).run(
                {"input_ids": tok,
                 "pos": torch.tensor([pos], dtype=torch.long, device=device)},
                kv=kv)
            kv = {"kcache": out["kcache"], "vcache": out["vcache"]}  # thread state forward
            logits = out["logits"]
        return logits

    return run_fn


# ======================================================================================
# (1) oracle.logit_equivalence + token_divergence
# ======================================================================================
def test_oracle_identical_and_perturbed():
    torch.manual_seed(0)
    ref = torch.randn(8, 256, dtype=TDT)

    v_id = logit_equivalence(ref.clone(), ref, dtype=TDT)
    assert v_id.correct is True, v_id.report()
    assert v_id.top1_agreement == 1.0
    assert v_id.max_abs_err == 0.0
    assert v_id.kl <= 1e-9

    # perturbed: a real, large numerical error must FAIL at fp32 tolerance with sensible metrics.
    bad = ref + torch.randn_like(ref) * 0.5
    v_bad = logit_equivalence(bad, ref, dtype=TDT)
    assert v_bad.correct is False, v_bad.report()
    assert v_bad.max_abs_err > 1e-2
    assert v_bad.top1_agreement < 1.0

    # a tiny fp32-level perturbation (reduction-order noise) must still PASS.
    tiny = ref + torch.randn_like(ref) * 1e-6
    v_tiny = logit_equivalence(tiny, ref, dtype=TDT)
    assert v_tiny.correct is True, v_tiny.report()

    # dtype tolerance widens: the same mid perturbation that fails fp32 passes bf16.
    mid = ref + torch.randn_like(ref) * 5e-3
    assert logit_equivalence(mid, ref, dtype=TDT).correct is False
    assert logit_equivalence(mid, ref, dtype=torch.bfloat16).correct is True

    # shape mismatch is a clean FAIL, not a crash.
    assert logit_equivalence(ref[:, :128], ref, dtype=TDT).correct is False
    print("[1a] oracle identical/perturbed/dtype/shape ... OK")


def test_oracle_on_real_referencevm():
    """The real run_fn (ToyLlama + ReferenceVM lowering) must equal eager logits, and
    token_divergence must report a perfect match over the decode horizon."""
    model = make_toy(seed=0, dtype=TDT, **TOY_KW)
    run_fn = make_run_fn(model, device="cpu")  # CPU for determinism in the correctness compare
    ids = torch.tensor([7], dtype=torch.long)

    test_logits = run_fn(ids)
    ref_logits = model.forward(ids)
    v = logit_equivalence(test_logits, ref_logits, dtype=TDT)
    assert v.correct is True, v.report()
    assert v.top1_agreement == 1.0

    # behavioral: greedy decode several tokens; correct run_fn never diverges.
    prompt = torch.tensor([3, 14, 159], dtype=torch.long)
    td = token_divergence(run_fn, model, prompt, n_tokens=8, dtype=TDT)
    assert td.correct is True, td.report()
    assert td.first_divergence == td.n_tokens == 8

    # a deliberately wrong run_fn (scaled logits flip the argmax distribution) must diverge early.
    def wrong_fn(input_ids):
        lg = run_fn(input_ids).clone()
        lg[..., :] = lg.flip(-1)  # reverse the vocab axis -> different argmax
        return lg
    td_bad = token_divergence(wrong_fn, model, prompt, n_tokens=8, dtype=TDT)
    assert td_bad.correct is False and 0 <= td_bad.first_divergence < 8, td_bad.report()
    print("[1b] oracle vs real ReferenceVM run_fn + token_divergence ... OK")


# ======================================================================================
# (2) bench correctness gate
# ======================================================================================
def test_bench_refuses_on_fail_and_times_on_pass():
    model = make_toy(seed=0, dtype=TDT, **TOY_KW)
    run_fn = make_run_fn(model, device=DEVICE)
    ids = torch.tensor([7], dtype=torch.long)

    good = logit_equivalence(run_fn(ids), model.forward(ids), dtype=TDT)
    assert good.correct is True, good.report()

    bad = logit_equivalence(run_fn(ids) + 1.0, model.forward(ids), dtype=TDT)
    assert bad.correct is False

    # strict (default): MUST raise rather than return a latency for a failing verdict.
    raised = False
    try:
        bench(lambda: run_fn(ids), bad, warmup=1, iters=2, device=DEVICE, strict=True)
    except CorrectnessGateError:
        raised = True
    assert raised, "bench must REFUSE (raise) to time a kernel with a FAIL verdict"

    # non-strict: MUST mark FAIL and return NO latency.
    r_fail = bench(lambda: run_fn(ids), bad, warmup=1, iters=2, device=DEVICE, strict=False)
    assert r_fail.correctness == "FAIL" and r_fail.latency_us is None, r_fail.report()

    # correct verdict: returns a real latency.
    r_ok = bench(lambda: run_fn(ids), good, warmup=3, iters=10, device=DEVICE)
    assert r_ok.correctness == "PASS" and r_ok.latency_us is not None and r_ok.latency_us > 0, r_ok.report()
    assert r_ok.is_real_perf == (DEVICE == "cuda")
    print("[2] bench refuses on FAIL / times on PASS ...", r_ok.grep_line())


# ======================================================================================
# (3) roofline
# ======================================================================================
def test_roofline_bound_and_pct():
    model = make_toy(seed=0, dtype=TDT, **TOY_KW)
    prog = build_toy_program(model.cfg)
    wbytes = prog.total_weight_bytes()
    assert wbytes > 0
    target = TARGETS["rtx5090"]
    bound = target.bandwidth_bound_us(wbytes)
    assert bound > 0

    # measured exactly at the bound -> 100% of bound, 100% HBM util.
    r_at = roofline_report(prog, bound, target)
    assert r_at.bound_us > 0
    assert abs(r_at.pct_of_bound - 100.0) < 1e-6, r_at.report()
    assert abs(r_at.hbm_util_pct - 100.0) < 1e-6

    # measured ABOVE the bound (realistic) -> pct_of_bound >= 100, util <= 100.
    r_above = roofline_report(prog, bound * 2.5, target)
    assert r_above.pct_of_bound >= 100.0, r_above.report()
    assert r_above.pct_of_bound > 200.0
    assert r_above.hbm_util_pct <= 100.0

    # accepts a raw byte count and a target name too.
    r_bytes = roofline_report(wbytes, bound, "rtx5090")
    assert r_bytes.bound_us == r_at.bound_us
    print("[3] roofline ...", r_above.grep_line())


def test_roofline_measured_peak_denominator():
    """The MEASURED-peak denominator (eval/peak_bandwidth.py -> GpuTarget.measured_bw_gbs) is the
    fairer roofline: since measured sustained bandwidth on the laptop 5090 is BELOW the desktop
    spec figure, the measured bound is LARGER (slower floor) and the % of measured bound is LOWER
    (i.e. closer to the floor / better) than the spec figure for the same latency. This test pins
    that relationship and the fallback-to-spec behaviour for an unmeasured target."""
    model = make_toy(seed=0, dtype=TDT, **TOY_KW)
    prog = build_toy_program(model.cfg)
    wbytes = prog.total_weight_bytes()
    target = TARGETS["rtx5090"]

    # the 5090 record carries a real measured peak that is strictly below the spec figure.
    assert target.measured_bw_gbs > 0
    assert target.measured_bw_gbs < target.hbm_bandwidth_gbs

    measured_bound = target.measured_bandwidth_bound_us(wbytes)
    spec_bound = target.bandwidth_bound_us(wbytes)
    # smaller (measured) bandwidth => larger (slower) bound.
    assert measured_bound > spec_bound

    r = roofline_report(prog, spec_bound * 2.5, target)
    assert r.measured_is_real is True
    assert r.measured_bw_gbs == target.measured_bw_gbs
    assert abs(r.measured_bound_us - measured_bound) < 1e-6
    # same latency, fairer (larger) denominator => closer to the floor: pct_of_measured < pct_of_spec
    assert r.pct_of_measured_bound < r.pct_of_bound, r.report()
    # and the measured HBM utilisation is correspondingly HIGHER than the spec utilisation.
    assert r.measured_hbm_util_pct > r.hbm_util_pct
    # the dict view carries the measured fields for the flywheel/paper.
    d = r.to_dict()
    for key in ("measured_bw_gbs", "measured_bound_us", "pct_of_measured_bound",
                "measured_hbm_util_pct", "measured_is_real"):
        assert key in d
    # an unmeasured target falls back to spec (measured == spec, flagged not-real).
    from schedule.ir import replace as _replace
    unmeasured = _replace(target, measured_bw_gbs=0.0)
    r0 = roofline_report(prog, spec_bound * 2.5, unmeasured)
    assert r0.measured_is_real is False
    assert abs(r0.measured_bound_us - spec_bound) < 1e-6
    assert abs(r0.pct_of_measured_bound - r0.pct_of_bound) < 1e-6
    print("[3b] roofline measured-peak denominator ...", r.grep_line())


# ======================================================================================
# (4) baselines
# ======================================================================================
def test_baselines_eager_runs_competitors_stub():
    model = make_toy(seed=0, dtype=TDT, **TOY_KW)
    ids = torch.tensor([7], dtype=torch.long)

    eager = eager_baseline(model, ids, device=DEVICE, warmup=3, iters=10)
    assert eager.status == "ok", eager.report()
    assert eager.latency_us is not None and eager.latency_us > 0
    assert eager.correctness == "PASS"

    for stub in (vllm_baseline("toy"), sglang_baseline("toy"), mpk_baseline()):
        assert stub.status == "not_run", stub.report()
        assert stub.latency_us is None
        assert stub.command, "a not_run stub must say how to run it"

    table = all_baselines(model, ids, device=DEVICE, warmup=2, iters=5)
    assert table["eager"].status == "ok" and table["eager"].latency_us is not None
    assert table["vllm"].status == "not_run" and table["vllm"].latency_us is None
    assert table["mpk"].status == "not_run" and table["mpk"].latency_us is None
    print("[4] baselines eager runs / competitors not_run ...", table["eager"].grep_line())


# ======================================================================================
if __name__ == "__main__":
    print(f"device = {DEVICE}\n")
    test_oracle_identical_and_perturbed()
    test_oracle_on_real_referencevm()
    test_bench_refuses_on_fail_and_times_on_pass()
    test_roofline_bound_and_pct()
    test_roofline_measured_peak_denominator()
    test_baselines_eager_runs_competitors_stub()
    print("\nALL EVAL ACCEPTANCE TESTS PASSED")
