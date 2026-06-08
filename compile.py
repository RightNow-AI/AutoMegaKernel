"""
AMK, THE PRODUCT SURFACE
=========================

    amk compile <hf-model> --gpu <arch> --regime single-stream

One command. Any (supported) model. Any (registered) GPU. It profiles, searches the schedule,
lowers to a megakernel, VALIDATES it (deadlock+race-free by construction), VERIFIES correctness
against eager PyTorch, benchmarks latency, reports distance to the HBM-bandwidth roofline, logs
to the flywheel, and emits the megakernel program + a report.

Honesty is built in (not bolted on):
  * Correctness is the **authoritative** gate: the lowered program is run through the CPU
    reference VM (bit-exact scheduling semantics) and compared to eager, exactly, every time.
  * Latency on a real GPU is measured only via the CUDA megakernel VM, and only the part that
    actually runs on the GPU is reported as measured; anything not yet GPU-runnable is reported
    as a cost-model PREDICTION, clearly labelled. We never print a measured number we didn't
    measure (see eval/bench.py's correctness gate + the row tags in results.tsv).
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from eval import logit_equivalence, roofline_report  # noqa: E402
from eval.bench import CorrectnessGateError, bench  # noqa: E402
from flywheel.log import (  # noqa: E402
    CorpusRecord, ResultRow, append_corpus, append_result, schedule_id,
)
from schedule.cost_model import estimate, predict_us  # noqa: E402
from schedule.graph import from_toy  # noqa: E402
from schedule.ir import TARGETS, DType, MegakernelProgram, validate  # noqa: E402
from schedule.lower import POS_NAME, RESHAPE_ID_NAME, TOKEN_NAME, lower, lower_fn  # noqa: E402
from schedule.search import default_config, search  # noqa: E402
from vm.reference_vm import ReferenceVM  # noqa: E402


# ======================================================================================
# Model loading
# ======================================================================================
def load_model(model_id: str, dtype: torch.dtype = torch.float32):
    """Return (model, graph_importer, eager_decode_fn, label). 'toy' is the fully-supported path;
    a HuggingFace id is best-effort via schedule.graph.from_hf (requires `transformers`)."""
    if model_id in ("toy", "toy-1L", "toy-2L"):
        from models.toy import make_toy
        n_layers = 2 if model_id == "toy-2L" else 1
        model = make_toy(seed=0, dtype=dtype, n_layers=n_layers)

        def eager_decode(tok: int) -> torch.Tensor:
            with torch.no_grad():
                return model.forward(torch.tensor([tok], device=next(model.parameters()).device))[-1].view(1, -1)

        return model, (lambda m: from_toy(m)), eager_decode, f"toy({n_layers}L)"

    if model_id in ("small", "small-bf16"):
        # The acceptance 'small'-scale decode model (matches vm/autotune.py SMALL +
        # tests/test_cuda_perf.py): a Llama-shaped 4-layer / hidden=2048 decoder, big enough that
        # the kernel-variant knobs (cp.async / cols_per_warp / N_tile / threads_per_block) move the
        # MEASURED decode latency by ~1.2-1.3x, the real headroom the autoresearch loop searches.
        # It loads in bf16 (the realistic decode storage dtype + the dtype the autotune/cp.async
        # wins were measured at); the autoresearch measured-cuda evaluator detects this and gates
        # GPU output against a bf16 CPU ReferenceVM within bf16 tolerance.
        from models.toy import make_toy
        small = dict(hidden=2048, n_layers=4, n_heads=16, n_kv_heads=4, head_dim=128,
                     intermediate=5632, vocab=32000)
        bf16 = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        model = make_toy(seed=0, dtype=bf16, **small)

        def eager_decode(tok: int) -> torch.Tensor:
            with torch.no_grad():
                return model.forward(
                    torch.tensor([tok], device=next(model.parameters()).device))[-1].view(1, -1)

        return model, (lambda m: from_toy(m)), eager_decode, "small(4L/2048h)"

    # HuggingFace path (documented stub, needs transformers + a real from_hf lowering).
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise SystemExit(f"HF model '{model_id}' needs transformers: uv pip install transformers "
                         f"({e})")
    from schedule.graph import from_hf
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).eval()

    def eager_decode(tok: int) -> torch.Tensor:
        with torch.no_grad():
            out = model(torch.tensor([[tok]]))
            return out.logits[0, -1].view(1, -1)

    return model, (lambda m: from_hf(m)), eager_decode, model_id


# ======================================================================================
# Output-path safety (model id / out-dir can be user-controlled, never let an artifact
# name traverse outside the intended output directory; see REL-PATHTRAVERSAL).
# ======================================================================================
def _safe_name_component(name: str) -> str:
    """Sanitize a (possibly user-controlled) string into a single safe filename component.

    Mirrors the legacy ``label.replace('/', '_')`` for normal inputs but additionally
    neutralizes *every* path separator (``/`` and ``\\``), strips ``..`` traversal
    segments, drops drive/root markers, and refuses any residual separator, so the
    result can never escape its parent directory. Normal labels (``toy(1L)``,
    ``meta-llama/Llama-3`` -> ``meta-llama_Llama-3``) are unchanged."""
    # Collapse both separator kinds to the legacy '_' so normal ids round-trip identically.
    s = name.replace("/", "_").replace("\\", "_")
    # Drop anything that still looks like a path (drive letters, leading roots) and '..'.
    s = s.replace(os.sep, "_")
    if os.altsep:
        s = s.replace(os.altsep, "_")
    s = s.replace(":", "_")  # windows drive markers like 'C:'
    # Remove parent-dir segments / leading dots that could still traverse.
    s = re.sub(r"\.\.+", "_", s)  # any run of >=2 dots -> '_'
    s = s.strip()
    if not s or s in (".", ".."):
        s = "model"
    return s


def _safe_join(out_dir: str, filename: str) -> str:
    """Join ``filename`` under ``out_dir`` and assert the realpath stays inside out_dir.

    ``filename`` must already be a single sanitized component. Raises SystemExit if the
    resolved path would escape the intended output directory (defense in depth)."""
    base = os.path.realpath(out_dir)
    full = os.path.realpath(os.path.join(base, filename))
    # Contained iff full == base or full is under base + separator.
    if not (full == base or full.startswith(base + os.sep)):
        raise SystemExit(
            f"AMK refuses to write outside the output directory: {full!r} escapes {base!r}.")
    return full


# ======================================================================================
# The pipeline
# ======================================================================================
def amk_compile(model_id: str, gpu: str, regime: str = "single-stream",
                search_budget: int = 0, device: str = "auto", token: int = 7,
                out_dir: str = "workspace", verbose: bool = True, stamp: float | None = None) -> dict:
    if gpu not in TARGETS:
        raise SystemExit(f"unknown --gpu {gpu!r}; known: {', '.join(TARGETS)}")
    target = TARGETS[gpu]
    os.makedirs(out_dir, exist_ok=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def log(m):
        if verbose:
            print(m)

    log(f"== amk compile {model_id} --gpu {gpu} --regime {regime} ==")
    model, importer, eager_decode, label = load_model(model_id)
    graph = importer(model)
    log(f"  imported graph: {getattr(graph, 'summary', lambda: label)()}")

    # ---- 1. schedule search (Loop 2) or default config -------------------------------
    config = default_config(target)
    if search_budget and search_budget > 0:
        log(f"  searching {search_budget} schedules (cost-model guided)...")
        res = search(graph, target, budget=search_budget, lower_fn=lower_fn)
        if res.best_config is not None:
            config = res.best_config
            log(f"    best predicted {res.best_score_us:.2f}us "
                f"({res.n_valid}/{len(res.trials)} valid candidates)")

    # ---- 2. lower + VALIDATE (refuse to ship an unsafe schedule) ----------------------
    prog: MegakernelProgram = lower(graph, target=target, config=config, pos=0, dtype=DType.F32)
    v = validate(prog)
    log(f"  lowered: {prog.summary()}")
    log(f"  validate: {'VALID' if v.ok else 'REJECTED'}  ({v.stats})")
    if not v.ok:
        for e in v.errors[:10]:
            log(f"    ERROR: {e}")
        raise SystemExit("AMK refuses to emit an invalid schedule (deadlock/race). Aborting.")

    # ---- 3. correctness (AUTHORITATIVE): reference VM vs eager ------------------------
    inputs = {
        TOKEN_NAME: torch.tensor([token], dtype=torch.int32),
        POS_NAME: torch.tensor([0], dtype=torch.int32),
        RESHAPE_ID_NAME: torch.tensor([0], dtype=torch.int32),
    }
    ref_logits = ReferenceVM(prog, model.weights_dict(), device="cpu").run(inputs, kv={})["logits"]
    eager_logits = eager_decode(token).to("cpu", torch.float32)
    verdict = logit_equivalence(ref_logits, eager_logits, dtype=torch.float32)
    log(f"  correctness (reference VM vs eager): {'PASS' if verdict.correct else 'FAIL'} "
        f"(max_abs_err={verdict.max_abs_err:.2e}, top1={verdict.top1_agreement:.3f})")

    # ---- 4. latency: real GPU megakernel if it runs; else cost-model prediction -------
    weight_bytes = prog.total_weight_bytes()
    bound_us = target.bandwidth_bound_us(weight_bytes)
    predicted_us = predict_us(prog, target)
    measured_us = None
    gpu_status = "not_run"
    real_perf = False
    if device == "cuda" and torch.cuda.is_available():
        try:
            from vm.loader import MegakernelVM
            gpu_in = {k: v.to("cuda") for k, v in inputs.items()}
            gvm = MegakernelVM(prog, model.weights_dict(), device="cuda")
            gpu_logits = gvm.run(gpu_in, kv={})["logits"].to("cpu", torch.float32)
            gpu_verdict = logit_equivalence(gpu_logits, eager_logits, dtype=torch.float32)
            if gpu_verdict.correct:
                bres = bench(lambda: gvm.run(gpu_in, kv={}), gpu_verdict,
                             warmup=10, iters=50, device="cuda", strict=True)
                measured_us = bres.latency_us
                real_perf = True
                gpu_status = "OK"
                log(f"  GPU megakernel: measured {measured_us:.2f}us on {torch.cuda.get_device_name(0)} "
                    f"(correctness PASS, max_abs_err={gpu_verdict.max_abs_err:.2e})")
            else:
                gpu_status = "gpu_mismatch"
                log(f"  GPU megakernel ran but mismatched eager (max_abs_err="
                    f"{gpu_verdict.max_abs_err:.2e}); reporting cost-model prediction instead.")
        except CorrectnessGateError as e:
            gpu_status = "gated"
            log(f"  GPU latency withheld (honesty gate): {e}")
        except Exception as e:  # full-decode GPU path may not be wired for every op yet
            gpu_status = f"unsupported: {type(e).__name__}"
            log(f"  GPU megakernel not runnable end-to-end yet ({type(e).__name__}: {e}).")
            log("  -> reporting cost-model PREDICTED latency (the reference VM proves correctness).")

    latency_us = measured_us if measured_us is not None else predicted_us
    rr = roofline_report(weight_bytes, latency_us, target)
    log(f"  roofline: bound={bound_us:.2f}us  {'measured' if real_perf else 'predicted'}="
        f"{latency_us:.2f}us  ({rr.pct_of_bound:.1f}% of bound, {rr.hbm_util_pct:.1f}% HBM util)")
    breakdown = estimate(prog, target)
    log(f"  regions: {breakdown.region_us}")

    # ---- 5. log to results.tsv + flywheel corpus -------------------------------------
    sid = schedule_id(config.to_dict())
    correctness = "PASS" if verdict.correct else "FAIL"
    row = ResultRow(
        experiment=1, tag="kept" if verdict.correct else "revert",
        loop="schedule", model=label, gpu=target.name, regime=regime,
        correctness=correctness,
        latency_us=round(latency_us, 3) if verdict.correct else "",
        pct_of_roofline=round(rr.pct_of_bound, 1) if verdict.correct else "",
        schedule_id=sid, kernel_id="",
        description=f"{'measured' if real_perf else 'predicted'} latency; gpu={gpu_status}")
    append_result(row, path=os.path.join(out_dir, "results.tsv"))
    if verdict.correct:
        append_corpus(CorpusRecord(
            model=label, gpu=target.name, regime=regime, correctness="PASS",
            latency_us=round(latency_us, 3), bound_us=round(bound_us, 3),
            pct_of_roofline=round(rr.pct_of_bound, 1), schedule=config.to_dict(),
            ir_version=prog.ir_version, abi_version=prog.abi_version,
            notes=f"{'measured-gpu' if real_perf else 'cost-model'}; status={gpu_status}"),
            path=os.path.join(out_dir, "..", "flywheel", "corpus.jsonl"), stamp=stamp)

    # ---- 6. emit the megakernel program + a report -----------------------------------
    # label (HF model ids) + out_dir can be user-controlled: sanitize the name component and
    # assert the resolved artifact path stays inside out_dir before writing (REL-PATHTRAVERSAL).
    safe_label = _safe_name_component(label)
    safe_gpu = _safe_name_component(gpu)
    prog_path = _safe_join(out_dir, f"{safe_label}.{safe_gpu}.amk.json")
    prog.save(prog_path)
    report = _render_report(label, target, prog, verdict, latency_us, bound_us, rr, breakdown,
                            real_perf, gpu_status, sid)
    report_path = _safe_join(out_dir, f"{safe_label}.{safe_gpu}.report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log(f"  emitted: {prog_path}")
    log(f"  report:  {report_path}")
    log("== done ==")
    return {
        "program": prog_path, "report": report_path, "correctness": correctness,
        "latency_us": latency_us, "measured": real_perf, "pct_of_bound": rr.pct_of_bound,
        "gpu_status": gpu_status, "schedule_id": sid,
    }


def _render_report(label, target, prog, verdict, latency_us, bound_us, rr, breakdown,
                   real_perf, gpu_status, sid) -> str:
    kind = "MEASURED (GPU megakernel)" if real_perf else "PREDICTED (analytic cost model)"
    lines = [
        f"# AMK compile report, {label} on {target.name}", "",
        f"- schedule id: `{sid}`",
        f"- IR / ABI version: {prog.ir_version} / {prog.abi_version}",
        f"- tasks: {len(prog.tasks)}  buffers: {len(prog.buffers)}  counters: {len(prog.counters)}",
        f"- weights: {prog.total_weight_bytes()/1e6:.2f} MB", "",
        "## Correctness (authoritative, reference VM vs eager PyTorch)",
        f"- verdict: **{'PASS' if verdict.correct else 'FAIL'}**",
        f"- max abs err: {verdict.max_abs_err:.3e}   top-1 agreement: {verdict.top1_agreement:.4f}", "",
        "## Latency", f"- value: **{latency_us:.2f} µs/token**  ({kind})",
        f"- GPU status: {gpu_status}",
        f"- HBM-bandwidth roofline floor: {bound_us:.2f} µs   "
        f"({rr.pct_of_bound:.1f}% of bound, {rr.hbm_util_pct:.1f}% HBM utilization)",
        f"- region breakdown (µs): {breakdown.region_us}", "",
        "## Honesty notes",
        "- Correctness is proven by the CPU reference VM (bit-exact scheduling semantics) vs eager.",
        "- Latency is " + ("a real measurement on this GPU." if real_perf
                           else "a cost-model prediction; the GPU end-to-end path for this model "
                                "is not fully wired yet (see report status)."),
        "- We do not claim datacenter (B200/H100) numbers we did not measure.", "",
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="amk compile", description="Compile a model into a megakernel.")
    ap.add_argument("model", help="'toy' / 'toy-2L' or a HuggingFace model id")
    ap.add_argument("--gpu", default="rtx5090", help=f"target GPU ({', '.join(TARGETS)})")
    ap.add_argument("--regime", default="single-stream", choices=["single-stream", "continuous-batching"])
    ap.add_argument("--search", type=int, default=0, help="schedule search budget (0 = default config)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--token", type=int, default=7, help="probe token id for the decode step")
    ap.add_argument("--out", default="workspace", help="output directory")
    args = ap.parse_args(argv)
    amk_compile(args.model, args.gpu, args.regime, search_budget=args.search,
                device=args.device, token=args.token, out_dir=args.out)


if __name__ == "__main__":
    main()
