"""
AMK, HONEST BASELINE HARNESS (eager runs; competitors are stubs, never faked)
=============================================================================

To claim a speedup you must compare against something real. The cardinal sin of kernel/inference
papers is fabricated or cherry-picked competitor numbers. This module makes that *impossible by
construction*:

  * The **eager PyTorch** baseline ACTUALLY RUNS. It times ``model.forward`` on the toy/real model
    with the same correctness-gated machinery (CUDA events on GPU, perf_counter on CPU) so AMK has
    a legitimate apples-to-apples reference on the same hardware. Eager is also, by definition,
    the correctness oracle (it IS the reference), so its verdict is trivially PASS.

  * **vLLM / SGLang / MPK (Mirage Persistent Kernel)** are returned as ``status='not_run'``
    structured records. Each carries (a) a one-line note on what it is and why it is not run here,
    (b) the exact shell command to run it yourself, and (c) ``latency_us=None``. We NEVER invent a
    number for a system we did not execute. If/when someone wires these up, they flip ``status`` to
    ``'ok'`` and fill ``latency_us`` from a real run, the schema is already the right shape.

Every record is a :class:`BaselineRecord`, so the flywheel can store a uniform competitor table
where the only rows with a latency are rows that were actually measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from eval.bench import BenchResult, bench
from eval.oracle import Verdict, logit_equivalence


@dataclass
class BaselineRecord:
    """One competitor/baseline row. ``status`` is the honesty flag:
      'ok'      -> latency_us is a real measured number on this machine,
      'not_run' -> a stub; latency_us is None and ``how_to_run``/``command`` say how to get it,
      'error'   -> attempted but failed (note has the reason); latency_us is None.
    """

    name: str
    status: str                       # 'ok' | 'not_run' | 'error'
    latency_us: float | None = None
    device: str = ""
    is_real_perf: bool = False
    correctness: str = "unknown"      # 'PASS' | 'FAIL' | 'unknown' (n/a for un-run stubs)
    note: str = ""
    how_to_run: str = ""
    command: str = ""
    # decode-comparison fields (filled by the per-token decode baselines; None otherwise).
    ms_per_token: float | None = None     # latency_us / 1000 (per decoded token); the headline
    tokens_per_s: float | None = None     # 1e6 / latency_us
    pct_of_roofline: float | None = None  # measured/HBM-bound * 100 (AMK only); see eval.roofline
    extra: dict[str, Any] = field(default_factory=dict)

    def grep_line(self, tag: str = "AutoKernel baseline") -> str:
        lat = "None" if self.latency_us is None else f"{self.latency_us:.3f}"
        mspt = "None" if self.ms_per_token is None else f"{self.ms_per_token:.4f}"
        tps = "None" if self.tokens_per_s is None else f"{self.tokens_per_s:.1f}"
        return (f"{tag} name:{self.name} status:{self.status} latency_us:{lat} "
                f"ms_per_token:{mspt} tokens_per_s:{tps} "
                f"correctness:{self.correctness} device:{self.device}")

    def report(self) -> str:
        lines = [self.grep_line()]
        if self.note:
            lines.append(f"  note: {self.note}")
        if self.status == "not_run" and self.command:
            lines.append(f"  run it: {self.command}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# The eager baseline, this one ACTUALLY RUNS.
# ----------------------------------------------------------------------------------------
def eager_baseline(model: Any,
                   input_ids: torch.Tensor,
                   device: str | torch.device = "cuda",
                   warmup: int = 10,
                   iters: int = 50) -> BaselineRecord:
    """Time eager ``model.forward(input_ids)`` on ``device`` and return a real :class:`BaselineRecord`.

    Eager *is* the correctness oracle, so its verdict is PASS by definition (we still construct a
    real :class:`Verdict` by comparing eager against itself, which is the honest representation:
    identical tensors => correct=True). This lets eager flow through the exact same correctness gate
    as the AMK path, no special-casing, no shortcut around the honesty rule.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return BaselineRecord(name="eager", status="error", device=str(dev),
                              note="device='cuda' requested but CUDA is unavailable")

    model = model.to(dev) if hasattr(model, "to") else model
    forward = getattr(model, "forward", model)
    ids = input_ids.to(dev)

    with torch.no_grad():
        ref_logits = forward(ids)
        # eager vs itself: a real Verdict that is trivially PASS (identical tensors).
        verdict: Verdict = logit_equivalence(ref_logits, ref_logits, dtype=ref_logits.dtype)

    def run():
        with torch.no_grad():
            return forward(ids)

    try:
        res: BenchResult = bench(run, verdict, warmup=warmup, iters=iters, device=dev, strict=True)
    except Exception as e:  # pragma: no cover - defensive; eager should always time
        return BaselineRecord(name="eager", status="error", device=str(dev),
                              correctness="PASS", note=f"timing failed: {e}")

    return BaselineRecord(
        name="eager",
        status="ok",
        latency_us=res.latency_us,
        device=str(dev),
        is_real_perf=res.is_real_perf,
        correctness=res.correctness,
        note=("eager PyTorch forward on this machine"
              + ("" if res.is_real_perf else " (CPU reference timing, not a GPU perf number)")),
        extra={"mean_us": res.mean_us, "min_us": res.min_us,
               "p10_us": res.p10_us, "p90_us": res.p90_us, "iters": res.iters},
    )


# ========================================================================================
# APPLES-TO-APPLES PER-TOKEN DECODE COMPARISON (the M2 thesis evidence)
# ========================================================================================
# The single-token decode step is what the megakernel fuses. To compare AMK against eager
# fairly we measure the SAME work on the SAME model + SAME GPU:
#   * eager , the model doing one KV-cached decode step as a stream of per-op kernel launches
#              (the thing AMK collapses into ONE cooperative launch),
#   * AMK   , the megakernel decode/token, measured at steady state (persistent device tables
#              already built; only counter-zero + new-token copy + relaunch per token).
# Both are CUDA-event timed and CORRECTNESS-GATED: no latency is reported without a PASS.
# ----------------------------------------------------------------------------------------
def _is_hf(model: Any) -> bool:
    return hasattr(model, "config") and not hasattr(model, "cfg")


def _model_label(model: Any) -> str:
    if _is_hf(model):
        return type(model).__name__
    if hasattr(model, "cfg"):
        cfg = model.cfg
        return f"toy({cfg.n_layers}L,h{cfg.hidden},v{cfg.vocab})"
    return type(model).__name__


def _eager_logits_at(model: Any, seq: list[int], device: torch.device) -> torch.Tensor:
    """Eager logits for the LAST position of ``seq`` (the decode target). Returns [1, vocab]."""
    if _is_hf(model):
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        return model(input_ids=ids).logits[0, -1].reshape(1, -1)
    ids = torch.tensor(seq, dtype=torch.long, device=device)
    return model.forward(ids)[-1].reshape(1, -1)


def eager_decode_baseline(model: Any,
                          *,
                          device: str | torch.device = "cuda",
                          context_len: int = 16,
                          warmup: int = 10,
                          iters: int = 50) -> BaselineRecord:
    """Time ONE eager per-op decode step (the work AMK fuses), on ``device``.

    Eager runs each op of the layer stack as its own kernel launch with a KV cache, for HF models
    via ``past_key_values``/``use_cache`` (the genuine incremental-decode path), for the toy via a
    single-token forward over a length-``context_len`` prefix (the toy has no KV-cache module, so we
    time the equivalent single-new-token forward, which is the same per-op launch stream the
    megakernel collapses). The result is per-token decode latency in eager PyTorch.

    Correctness-gated like everything else: eager IS the oracle, so the gate is trivially PASS
    (logit_equivalence of eager vs itself). No latency is produced without that PASS.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return BaselineRecord(name="eager-decode", status="error", device=str(dev),
                              note="device='cuda' requested but CUDA is unavailable")
    if dev.index is None and dev.type == "cuda":
        dev = torch.device("cuda", torch.cuda.current_device())

    model = model.to(dev) if hasattr(model, "to") else model
    model_eval = getattr(model, "eval", None)
    if callable(model_eval):
        model.eval()

    label = _model_label(model)
    # AMK measures the step at position `context_len` over a length-(context_len+1) window; match it:
    # the HF path primes `context_len` tokens then does 1 incremental step (total context_len+1),
    # the toy path forwards a length-(context_len+1) prefix whose last position is the decode target.
    ctx = max(1, context_len)
    vocab = (model.config.vocab_size if _is_hf(model) else model.cfg.vocab)
    prompt = [(i * 7 + 1) % vocab for i in range(ctx + 1)]

    try:
        with torch.no_grad():
            if _is_hf(model):
                # Build the KV cache for the first `ctx` tokens, then time ONE incremental step on
                # the last token (total window ctx+1, matching the AMK step at position ctx).
                ids = torch.tensor([prompt[:-1]], dtype=torch.long, device=dev)
                primed = model(input_ids=ids, use_cache=True)
                past = primed.past_key_values
                next_tok = torch.tensor([[prompt[-1]]], dtype=torch.long, device=dev)

                def step():
                    # one decode step: a single new token against the cached context. This is the
                    # per-op kernel-launch stream (q/k/v/o/gate/up/down GEMVs + norms + attention).
                    return model(input_ids=next_tok, past_key_values=past, use_cache=True).logits

                ref = step()
                ref_logits = ref[0, -1].reshape(1, -1)
            else:
                ids = torch.tensor(prompt, dtype=torch.long, device=dev)

                def step():
                    return model.forward(ids)

                ref_logits = step()[-1].reshape(1, -1)

            verdict = logit_equivalence(ref_logits, ref_logits, dtype=ref_logits.dtype)
    except Exception as e:  # pragma: no cover - defensive
        return BaselineRecord(name="eager-decode", status="error", device=str(dev),
                              note=f"eager decode step failed to run: {e}")

    try:
        res: BenchResult = bench(step, verdict, warmup=warmup, iters=iters, device=dev, strict=True)
    except Exception as e:  # pragma: no cover
        return BaselineRecord(name="eager-decode", status="error", device=str(dev),
                              correctness="PASS", note=f"timing failed: {e}")

    lat = res.latency_us
    return BaselineRecord(
        name="eager-decode",
        status="ok",
        latency_us=lat,
        device=str(dev),
        is_real_perf=res.is_real_perf,
        correctness=res.correctness,
        ms_per_token=(lat / 1e3) if lat else None,
        tokens_per_s=(1e6 / lat) if lat else None,
        note=("eager PyTorch per-op decode step (per-token, KV-cached) on this machine"
              + ("" if res.is_real_perf else " (CPU reference timing, NOT a GPU perf number)")),
        extra={"model": label, "context_len": ctx, "backend": "eager-perop",
               "mean_us": res.mean_us, "min_us": res.min_us, "iters": res.iters},
    )


def amk_decode_baseline(model: Any,
                        gpu: str = "rtx5090",
                        *,
                        device: str | torch.device = "cuda",
                        context_len: int = 16,
                        warmup: int = 20,
                        iters: int = 100,
                        dtype: Any = None) -> BaselineRecord:
    """Time the AMK megakernel decode/token at STEADY STATE on ``gpu``, correctness-gated vs eager.

    Builds the model graph, lowers ONE decode step at position ``pos=context_len`` (so the attention
    window matches the eager baseline's context), constructs a :class:`MegakernelVM`, primes it once
    (building the persistent device tables), and:
      1. CORRECTNESS GATE: compares the megakernel logits against eager logits at that position with
         :func:`eval.oracle.logit_equivalence`, no latency is reported unless it PASSes.
      2. Measures the per-token wall latency two ways with CUDA events:
           * ``run()`` steady-state , the realistic per-token path (counter-zero + new-token H2D
             copy + cooperative relaunch + output read-back); this is what a decode loop pays.
           * ``relaunch()`` kernel  , JUST the cooperative megakernel (no host marshalling), the
             pure on-GPU cost. Reported in ``extra`` as the kernel-only floor.
      The headline ``latency_us`` / ``ms_per_token`` is the steady-state ``run()`` path (honest
      per-token cost). ``pct_of_roofline`` is the steady-state latency vs the HBM weight-streaming
      bound from :mod:`eval.roofline`.

    Requires CUDA (the megakernel is a real cooperative CUDA launch). On CPU there is no megakernel;
    returns status='error' with that reason (the ReferenceVM is the oracle, not a perf target).
    """
    from schedule.ir import DType, TARGETS, validate
    from schedule.lower import POS_NAME, RESHAPE_ID_NAME, TOKEN_NAME, lower

    dev = torch.device(device)
    if dev.type != "cuda":
        return BaselineRecord(name="amk-decode", status="error", device=str(dev),
                              note=("AMK decode is a CUDA cooperative megakernel; there is no "
                                    "megakernel perf number on CPU (CPU has only the ReferenceVM "
                                    "oracle). Run with device='cuda'."))
    if not torch.cuda.is_available():
        return BaselineRecord(name="amk-decode", status="error", device=str(dev),
                              note="device='cuda' requested but CUDA is unavailable")
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if gpu not in TARGETS:
        return BaselineRecord(name="amk-decode", status="error", device=str(dev),
                              note=f"unknown gpu {gpu!r}; known: {sorted(TARGETS)}")
    target = TARGETS[gpu]
    dt = dtype if dtype is not None else DType.F32

    # Move the eager model onto the device so the correctness-gate comparison runs there (the
    # megakernel logits come back on `dev`; eager must match device for logit_equivalence inputs).
    model = model.to(dev) if hasattr(model, "to") else model
    if callable(getattr(model, "eval", None)):
        model.eval()

    # ---- import the graph + bind weights (same object eager used, so weights are identical) ----
    try:
        if _is_hf(model):
            from schedule.graph import from_hf, weights_from_hf
            graph = from_hf(model)
            weights = weights_from_hf(model)
            vocab = model.config.vocab_size
        elif hasattr(model, "cfg"):
            from schedule.graph import from_toy
            graph = from_toy(model)
            weights = model.weights_dict()
            vocab = model.cfg.vocab
        else:
            return BaselineRecord(name="amk-decode", status="error", device=str(dev),
                                  note=f"cannot import model of type {type(model).__name__}")
    except Exception as e:  # pragma: no cover
        return BaselineRecord(name="amk-decode", status="error", device=str(dev),
                              note=f"graph import failed: {e}")

    pos = max(0, context_len)
    if pos >= graph.config.max_seq:
        pos = graph.config.max_seq - 1

    label = _model_label(model)

    try:
        from vm.loader import MegakernelVM
    except Exception as e:  # pragma: no cover
        return BaselineRecord(name="amk-decode", status="error", device=str(dev),
                              note=f"could not import MegakernelVM: {e}")

    # A deterministic prompt of length pos+1 (positions 0..pos). The decode step we MEASURE is the
    # one at position `pos`; positions 0..pos-1 only prime the KV cache (the realistic decode state).
    prompt = [(i * 7 + 1) % max(1, vocab) for i in range(pos + 1)]

    def _kv_names(p):
        from schedule.ir import BufferKind
        return [b.name for b in p.buffers if b.kind == BufferKind.KV_CACHE]

    def _ins(token: int, position: int) -> dict[str, torch.Tensor]:
        return {
            TOKEN_NAME: torch.tensor([token], dtype=torch.int32, device=dev),
            POS_NAME: torch.tensor([position], dtype=torch.int32, device=dev),
            RESHAPE_ID_NAME: torch.tensor([0], dtype=torch.int32, device=dev),
        }

    # ---- drive AMK forward 0..pos building the REAL KV cache (the proven-correct decode path) ----
    # Each position re-lowers at its own pos (attention window kv_len=pos+1, KV_APPEND at pos), so
    # the cache the MEASURED step at `pos` attends over is exactly a pos+1 context, apples-to-apples
    # with the eager KV-cached step over the same pos+1 prefix.
    try:
        kv: dict[str, torch.Tensor] = {}
        vm = None
        prog = None
        for p in range(pos + 1):
            prog = lower(graph, target=target, config=None, pos=p, dtype=dt)
            if p == 0:
                vres = validate(prog)
                if not vres.ok:
                    return BaselineRecord(name="amk-decode", status="error", device=str(dev),
                                          note="AMK refuses an invalid schedule:\n" + vres.report())
            vm = MegakernelVM(prog, weights, device=str(dev))
            out = vm.run(_ins(prompt[p], p), kv=kv)
            kv = {n: out[n] for n in _kv_names(prog) if n in out}
        amk_logits = out["logits"].detach().reshape(1, -1)
    except Exception as e:
        return BaselineRecord(name="amk-decode", status="error", device=str(dev),
                              note=f"megakernel decode-to-pos failed: {e}")

    # ---- CORRECTNESS GATE: AMK logits at `pos` vs eager forward over the SAME pos+1 prefix ----
    try:
        eager_logits = _eager_logits_at(model, prompt, dev)
        verdict = logit_equivalence(amk_logits, eager_logits, dtype=amk_logits.dtype)
    except Exception as e:
        return BaselineRecord(name="amk-decode", status="error", device=str(dev),
                              note=f"correctness comparison failed: {e}")

    if not verdict.correct:
        # Honesty: refuse to report any AMK latency if it is not correct.
        return BaselineRecord(name="amk-decode", status="error", device=str(dev),
                              correctness="FAIL",
                              note=("AMK decode logits did not match eager at the gate; refusing to "
                                    "report a latency for an incorrect kernel. " + verdict.report()))

    # ---- steady-state per-token timing of the step at `pos` (tables already built; relaunch path).
    # We re-fire the SAME (token, pos) step against the populated cache; this is the per-token cost a
    # decode loop pays at steady state (counter-zero + new-token copy + cooperative relaunch + read).
    ins = _ins(prompt[pos], pos)

    def run_step():
        return vm.run(ins, kv=kv)

    try:
        res: BenchResult = bench(run_step, verdict, warmup=warmup, iters=iters,
                                 device=dev, strict=True)
    except Exception as e:
        return BaselineRecord(name="amk-decode", status="error", device=str(dev),
                              correctness="PASS", note=f"steady-state timing failed: {e}")

    # ---- kernel-only timing via relaunch() (no host marshalling) for the on-GPU floor ----
    kernel_us = None
    try:
        for _ in range(max(1, warmup)):
            vm.relaunch()
        torch.cuda.synchronize(dev)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ksamples = []
        for _ in range(max(1, iters)):
            ev0.record()
            vm.relaunch()
            ev1.record()
            ev1.synchronize()
            ksamples.append(ev0.elapsed_time(ev1) * 1e3)  # ms -> us
        ksamples.sort()
        kernel_us = ksamples[len(ksamples) // 2]
    except Exception:
        kernel_us = None  # kernel-only is a bonus datapoint; never fail the baseline over it.

    lat = res.latency_us
    # ---- roofline: steady-state latency vs the HBM weight-streaming bound ----
    pct_roof = None
    achieved_gbs = None
    bound_us = None
    try:
        from eval.roofline import report as roofline_report
        rr = roofline_report(prog, lat, target)
        pct_roof = rr.pct_of_bound
        achieved_gbs = rr.achieved_gbs
        bound_us = rr.bound_us
    except Exception:
        pass

    return BaselineRecord(
        name="amk-decode",
        status="ok",
        latency_us=lat,
        device=str(dev),
        is_real_perf=res.is_real_perf,
        correctness=res.correctness,
        ms_per_token=(lat / 1e3) if lat else None,
        tokens_per_s=(1e6 / lat) if lat else None,
        pct_of_roofline=pct_roof,
        note=("AMK megakernel decode/token at steady state (one cooperative launch per token); "
              "correctness-gated vs eager logits"),
        extra={"model": label, "gpu": target.name, "pos": pos, "context_len": context_len,
               "backend": "MegakernelVM", "n_tasks": len(prog.tasks),
               "weight_bytes": prog.total_weight_bytes(),
               "kernel_only_us": kernel_us, "roofline_bound_us": bound_us,
               "achieved_gbs": achieved_gbs,
               "mean_us": res.mean_us, "min_us": res.min_us, "iters": res.iters},
    )


def decode_comparison(model: Any,
                      gpu: str = "rtx5090",
                      *,
                      device: str | torch.device = "cuda",
                      context_len: int = 16,
                      warmup: int = 20,
                      iters: int = 100) -> dict[str, BaselineRecord]:
    """The apples-to-apples per-token decode table on ONE model + ONE GPU.

    Returns ``{"eager": ..., "amk": ..., "vllm": ...}`` of :class:`BaselineRecord`:
      * eager, real per-op KV-cached decode step (measured),
      * amk  , megakernel decode/token at steady state (measured, correctness-gated),
      * vllm , attempted, else status='not_run' with the exact reason + a Linux command.
    Only rows that actually ran carry a latency. ``amk.extra['speedup_vs_eager']`` is the honest
    wall-clock ratio (>1 == AMK faster; may be <1 today on tiny models where eager's optimized
    per-op kernels beat AMK's absolute latency, we report the truth either way)."""
    eager = eager_decode_baseline(model, device=device, context_len=context_len,
                                  warmup=warmup, iters=iters)
    amk = amk_decode_baseline(model, gpu, device=device, context_len=context_len,
                              warmup=warmup, iters=iters)
    vllm = vllm_decode_baseline(_model_label(model))

    if (eager.status == "ok" and amk.status == "ok"
            and eager.latency_us and amk.latency_us):
        amk.extra["eager_ms_per_token"] = eager.ms_per_token
        amk.extra["speedup_vs_eager"] = eager.latency_us / amk.latency_us
    return {"eager": eager, "amk": amk, "vllm": vllm}


def vllm_decode_baseline(model_id: str = "<hf-model-id>") -> BaselineRecord:
    """ATTEMPT to import vLLM for a third decode datapoint; record honestly if it can't run here.

    vLLM does not ship Windows wheels and its CUDA kernels are Linux-only, so on this dev machine
    the import will fail. We try the import (never fabricate a number); on failure we return
    status='not_run' carrying the exact ImportError and the exact command to get the datapoint on a
    Linux box with the SAME model. If vLLM ever imports here, we still mark it 'not_run' with a note
    that wiring its decode latency is left to the Linux path (we refuse to print a half-measured
    number from an environment vLLM does not officially support)."""
    cmd = ("uv pip install vllm && "
           "python -c \"from vllm import LLM, SamplingParams; "
           "llm=LLM(model='<hf-model-id>', max_model_len=512, enforce_eager=False); "
           "import time; "
           "p=SamplingParams(max_tokens=128, ignore_eos=True); "
           "_=llm.generate(['hello'], p);  # warmup + JIT/cudagraph\n"
           "t=time.perf_counter(); o=llm.generate(['hello'], p); dt=time.perf_counter()-t; "
           "print('ms/token', dt/128*1000)\"")
    try:
        import importlib
        importlib.import_module("vllm")
        installed = True
        reason = ("vLLM imported, but its single-stream decode timing is only wired on the Linux "
                  "path; this harness refuses to print a number from an unsupported/partial env.")
    except Exception as e:
        installed = False
        reason = (f"vLLM is not runnable in this environment: {type(e).__name__}: {e}. "
                  "vLLM publishes no Windows wheels and its CUDA kernels are Linux-only "
                  "(`uv pip install vllm` fails here). Run the command on a Linux GPU box.")
    rec = _stub("vllm-decode", note=reason,
                how_to_run=("On Linux: install vLLM, load the SAME model, warm up (CUDA graphs), "
                            "then time single-stream decode and divide by tokens for ms/token."),
                command=cmd)
    rec.extra["vllm_importable"] = installed
    return rec


# ----------------------------------------------------------------------------------------
# Competitor stubs, NEVER fabricated. status='not_run' with a real command to reproduce.
# ----------------------------------------------------------------------------------------
def _stub(name: str, note: str, how_to_run: str, command: str) -> BaselineRecord:
    return BaselineRecord(name=name, status="not_run", latency_us=None, correctness="unknown",
                          note=note, how_to_run=how_to_run, command=command)


def vllm_baseline(model_id: str = "<hf-model-id>", **_: Any) -> BaselineRecord:
    """vLLM paged-attention serving engine. Not run here (separate heavy install + server)."""
    return _stub(
        "vllm",
        note=("vLLM is a separate serving engine (paged-attention, continuous batching). "
              "Not executed by this harness; would require a running vLLM server and a "
              "matching HF checkpoint. No number is reported until it is actually measured."),
        how_to_run=("pip install vllm; serve the model, then time single-stream decode latency "
                    "against the same prompt and dtype as AMK."),
        command=(f"vllm serve {model_id} --max-num-seqs 1 && "
                 "python -m vllm.entrypoints.benchmarks.latency "
                 f"--model {model_id} --input-len 1 --output-len 128 --batch-size 1"),
    )


def sglang_baseline(model_id: str = "<hf-model-id>", **_: Any) -> BaselineRecord:
    """SGLang runtime. Not run here."""
    return _stub(
        "sglang",
        note=("SGLang is a separate LLM serving runtime (RadixAttention). Not executed by this "
              "harness; requires its own install and server. No number is reported until measured."),
        how_to_run=("pip install 'sglang[all]'; launch the server, then bench single-stream "
                    "decode latency on the identical workload."),
        command=(f"python -m sglang.launch_server --model-path {model_id} --disable-cuda-graph && "
                 f"python -m sglang.bench_one_batch --model-path {model_id} "
                 "--batch-size 1 --input-len 1 --output-len 128"),
    )


def mpk_baseline(**_: Any) -> BaselineRecord:
    """MPK = Mirage Persistent Kernel (the prior-art megakernel compiler we are compared against)."""
    return _stub(
        "mpk",
        note=("MPK (Mirage Persistent Kernel) is the prior-art megakernel/persistent-kernel "
              "compiler. Not executed by this harness (separate toolchain + supported-model set). "
              "No number is reported until MPK is actually run on the same model/GPU."),
        how_to_run=("Install Mirage (https://github.com/mirage-project/mirage), compile the model "
                    "to its persistent kernel, and time one decode step on this GPU."),
        command=("git clone https://github.com/mirage-project/mirage && "
                 "python mirage/demo/megakernel/decode.py --model <model> --device cuda:0"),
    )


def all_baselines(model: Any,
                  input_ids: torch.Tensor,
                  device: str | torch.device = "cuda",
                  warmup: int = 10,
                  iters: int = 50,
                  model_id: str = "<hf-model-id>") -> dict[str, BaselineRecord]:
    """Convenience: the full competitor table. Only ``eager`` carries a real latency; the rest are
    honest ``not_run`` stubs with reproduction commands."""
    return {
        "eager": eager_baseline(model, input_ids, device=device, warmup=warmup, iters=iters),
        "vllm": vllm_baseline(model_id),
        "sglang": sglang_baseline(model_id),
        "mpk": mpk_baseline(),
    }


__all__ = [
    "BaselineRecord", "eager_baseline", "vllm_baseline", "sglang_baseline", "mpk_baseline",
    "all_baselines",
    # per-token decode comparison (the M2 thesis evidence)
    "eager_decode_baseline", "amk_decode_baseline", "vllm_decode_baseline", "decode_comparison",
]
