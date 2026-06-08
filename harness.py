"""
AMK, THE CODING-AGENT HARNESS (the NVIDIA-facing product surface)
==================================================================

This is the programmatic + CLI-able API a coding agent (Claude Code, Codex, an NVIDIA/Google
engineer's tooling) drives to generate megakernels. It is the AMK analogue of AutoKernel's
read-program.md / propose / eval / keep-revert loop, but specialized to **Loop 2, schedule
search**: the agent edits ONE structured object (:class:`schedule.ir.ScheduleConfig`), and the
frozen, validated VM lowers it into a runnable megakernel. The agent never writes kernel code,
never touches ``Task.sm`` or the ABI; a bad schedule is turned into a clean ``REJECTED`` by
:func:`schedule.ir.validate`, never a hung GPU.

THREE VERBS (see HARNESS.md for the full contract + a copy-pasteable agent loop):

  * :func:`propose`, return the current incumbent ``ScheduleConfig`` (as a dict) plus the
    documented, editable ``search_space`` (the knobs + ranges the agent may move). This is the
    "read program.md" step: it tells the agent what surface it owns.

  * :func:`evaluate`, take an edited config dict and return a STRUCTURED JSON verdict. It lowers
    the config, ``validate()``s it (a bad config is a clean ``valid=False`` + ``rejected_reason``,
    never a crash, never a latency), proves correctness with the CPU :class:`ReferenceVM` vs eager
    PyTorch (AUTHORITATIVE), then, only if correct, measures GPU latency via
    :class:`vm.loader.MegakernelVM` through the correctness-gated :func:`eval.bench.bench`, or
    falls back to the analytic ``cost_model.predict_us`` when CUDA / the GPU path is unavailable.
    A latency is NEVER emitted without a correctness PASS.

  * :func:`loop`, the keep/revert autoresearch loop over proposed configs. Every trial is logged
    to ``results.tsv`` via :mod:`flywheel.log`; the best VALID + CORRECT schedule is kept (mirrors
    AutoKernel: correctness first, then a >= 1% latency gain, simplicity as the tie-break). Returns
    the best verdict + the log rows.

HONESTY (inherited from the fixed eval, enforced in code not comments):
  * Correctness is the reference VM vs eager, every time, authoritative.
  * ``latency_kind`` is always one of 'measured-gpu' (real CUDA event timing, correctness-gated)
    or 'predicted' (analytic cost model). We never label a prediction as a measurement.
  * Nothing in here touches Modal / a cloud GPU: the local device is the only one used.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from eval import logit_equivalence, roofline_report  # noqa: E402
from eval.bench import CorrectnessGateError, bench  # noqa: E402
from generate import GenerateResult, generate  # noqa: E402  (multi-token autoregressive decode)
from flywheel.log import (  # noqa: E402
    CorpusRecord, ResultRow, append_corpus, append_result, schedule_id,
)
from flywheel.prior import (  # noqa: E402
    SEARCHABLE_KNOB_CHOICES as _KNOB_CHOICES_REF,
    SEARCHABLE_KNOB_DEFAULTS as _KNOB_DEFAULTS_REF,
)
from schedule.cost_model import predict_us  # noqa: E402
from schedule.ir import (  # noqa: E402
    TARGETS, DType, GpuTarget, MegakernelProgram, ScheduleConfig, validate,
)
from schedule.lower import POS_NAME, RESHAPE_ID_NAME, TOKEN_NAME, lower  # noqa: E402
from schedule.search import (  # noqa: E402
    KV_BLOCK_CHOICES, N_TILE_CHOICES, PAGE_POLICY_CHOICES, PIPELINING_CHOICES,
    SM_POLICY_CHOICES, THREADS_PER_BLOCK_CHOICES, default_config, mutate_config,
    random_config,
)

# Default location of the flywheel results log (consistent with compile.py / amk_cli corpus).
DEFAULT_RESULTS_TSV = os.path.join("workspace", "results.tsv")
DEFAULT_CORPUS = os.path.join("flywheel", "corpus.jsonl")

# Probe token for the single-step decode correctness/latency check (matches compile.py).
_PROBE_TOKEN = 7

# ---- kernel-knob edit surface (the MegakernelVM compile-time GEMV build knobs) -------------------
# These are the knobs that actually move MEASURED decode latency on this VM (the ScheduleConfig's
# GEMV tile is auto-sized, so the schedule landscape alone is near-flat, see autoresearch.py).
# An agent edits them via a reserved ``"kernel_knobs"`` key in the config dict; they are threaded to
# vm.loader.MegakernelVM(prog, weights, knobs=...) and realised as ``-D`` macros. The default set is
# byte-identical to the production VM build (so a config WITHOUT kernel_knobs is the exact incumbent).
# SINGLE SOURCE OF TRUTH: choices + defaults live in flywheel.prior.SEARCHABLE_KNOB_CHOICES /
# SEARCHABLE_KNOB_DEFAULTS, imported above as _KNOB_CHOICES_REF / _KNOB_DEFAULTS_REF, and are
# aliased here so all internal helpers keep their existing _KERNEL_KNOB_* names unchanged.
_KERNEL_KNOB_CHOICES: dict[str, tuple[int, ...]] = _KNOB_CHOICES_REF
_KERNEL_KNOB_DEFAULTS: dict[str, int] = dict(_KNOB_DEFAULTS_REF)


def _normalize_kernel_knobs(knobs: dict[str, Any] | None) -> dict[str, int]:
    """Coerce an agent-supplied kernel_knobs dict to the documented int knobs (unknown keys dropped,
    values cast to int, missing keys filled with the production default). The result is what gets
    hashed into the schedule_id and passed to MegakernelVM(knobs=...)."""
    out = dict(_KERNEL_KNOB_DEFAULTS)
    if knobs:
        for k, v in knobs.items():
            if k in _KERNEL_KNOB_DEFAULTS:
                try:
                    out[k] = int(v)
                except (TypeError, ValueError):
                    pass
    return out


def _knobs_are_default(knobs: dict[str, int]) -> bool:
    return all(knobs.get(k) == d for k, d in _KERNEL_KNOB_DEFAULTS.items())


def _sid_with_knobs(cfg_dict: dict[str, Any], knobs: dict[str, int] | None) -> str:
    """Flywheel key for the COMBINED candidate. Knob-free configs keep their exact historical
    schedule_id (so existing corpus keys are unchanged); knobs only perturb the hash when present."""
    if not knobs or _knobs_are_default(knobs):
        return schedule_id(cfg_dict)
    return schedule_id({**cfg_dict, "kernel_knobs": knobs})


def _random_kernel_knobs(rng) -> dict[str, int]:
    """A fresh random point in the kernel-knob space (explore)."""
    return {k: rng.choice(v) for k, v in _KERNEL_KNOB_CHOICES.items()}


def _mutate_kernel_knobs(knobs: dict[str, int] | None, rng) -> dict[str, int]:
    """Change ONE kernel knob (local exploit move; keeps the keep/revert hill-climb smooth)."""
    out = dict(_KERNEL_KNOB_DEFAULTS)
    out.update(knobs or {})
    k = rng.choice(list(_KERNEL_KNOB_CHOICES))
    out[k] = rng.choice(_KERNEL_KNOB_CHOICES[k])
    return out


# ======================================================================================
# Model loading (the toy path is the fully-supported one; HF is best-effort via from_hf)
# ======================================================================================
def _load_model(model_id: str, dtype: torch.dtype = torch.float32):
    """Return (model, importer, eager_decode_fn, label). Mirrors compile.load_model so the
    harness and the one-shot compiler agree on what 'toy' means."""
    import compile as _compile
    return _compile.load_model(model_id, dtype=dtype)


def _resolve_target(gpu: str) -> GpuTarget:
    if gpu not in TARGETS:
        raise KeyError(f"unknown gpu {gpu!r}; known targets: {', '.join(sorted(TARGETS))}")
    return TARGETS[gpu]


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# Map a torch dtype to the IR DType the lowerer/VM use, so correctness is judged at the precision
# the model ACTUALLY runs in (a bf16 model is lowered bf16 and gated at bf16 tolerance, judging a
# correct bf16 megakernel by fp32 tolerances would falsely FAIL it on ordinary bf16 rounding).
_TORCH_TO_DTYPE = {
    torch.float32: DType.F32, torch.float16: DType.F16, torch.bfloat16: DType.BF16,
}


def _infer_model_dtype(model) -> tuple[DType, torch.dtype]:
    """Infer (IR DType, torch dtype) from the model's floating-point weights. Defaults to F32 for
    an all-integer/quantized state dict (the lowerer's historical default)."""
    try:
        for t in model.weights_dict().values():
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                return _TORCH_TO_DTYPE.get(t.dtype, DType.F32), t.dtype
    except Exception:
        pass
    return DType.F32, torch.float32


def _config_type_errors(cfg: ScheduleConfig) -> list[str]:
    """Type-sanity the edit-surface knobs (mirrors schemas/schedule_config.schema.json) so a
    malformed agent edit (e.g. pipelining_depth='deep', tiling not a dict, a negative N_tile) is a
    clean REJECTED rather than a deep crash in the lowerer/cost-model. Structural safety is still
    proven by schedule.ir.validate; this just guards the typed knobs the search space defines."""
    errs: list[str] = []

    def _is_int(x: Any) -> bool:
        return isinstance(x, int) and not isinstance(x, bool)

    if not isinstance(cfg.tiling, dict):
        errs.append(f"tiling must be an object, got {type(cfg.tiling).__name__}")
    else:
        for arch, knobs in cfg.tiling.items():
            if not isinstance(knobs, dict):
                errs.append(f"tiling['{arch}'] must be an object")
                continue
            for k, val in knobs.items():
                if not _is_int(val) or val < 1:
                    errs.append(f"tiling['{arch}']['{k}']={val!r} must be a positive int")
    if not _is_int(cfg.pipelining_depth) or cfg.pipelining_depth < 0:
        errs.append(f"pipelining_depth={cfg.pipelining_depth!r} must be a non-negative int")
    if not _is_int(cfg.threads_per_block):
        errs.append(f"threads_per_block={cfg.threads_per_block!r} must be an int")
    if not _is_int(cfg.smem_bytes_per_block):
        errs.append(f"smem_bytes_per_block={cfg.smem_bytes_per_block!r} must be an int")
    if not isinstance(cfg.page_allocation, str):
        errs.append(f"page_allocation={cfg.page_allocation!r} must be a string")
    if not isinstance(cfg.fusion_grouping, list):
        errs.append(f"fusion_grouping must be a list of groups, got "
                    f"{type(cfg.fusion_grouping).__name__}")
    if not isinstance(cfg.sm_assignment, (str, dict)):
        errs.append(f"sm_assignment={cfg.sm_assignment!r} must be a policy string or a "
                    f"{{task_id: sm}} map")
    return errs


def _launch_config_errors(cfg: ScheduleConfig, target: GpuTarget) -> list[str]:
    """Validate the LAUNCH-config knobs against the target BEFORE launch, the loader-level gate
    that schedule.ir.validate (a pure structural IR check) does not cover. These mirror the
    documented constraints in vm/loader.py + vm/abi.h: block size must be a positive multiple of
    the 32-thread warp and within [32, 1024]; dynamic SMEM opt-in must be within the target's
    per-block opt-in cap. A violation is a clean REJECTED here, never a hung/failed launch."""
    errs: list[str] = []
    tpb = cfg.threads_per_block
    if not isinstance(tpb, int) or isinstance(tpb, bool) or tpb < 32 or tpb > 1024:
        errs.append(f"threads_per_block={tpb!r} out of range [32,1024]")
    elif tpb % 32 != 0:
        errs.append(f"threads_per_block={tpb} is not a multiple of the warp size (32)")
    smem = cfg.smem_bytes_per_block
    if not isinstance(smem, int) or isinstance(smem, bool) or smem < 0:
        errs.append(f"smem_bytes_per_block={smem!r} must be a non-negative int")
    elif smem > target.smem_bytes_per_block_optin:
        errs.append(f"smem_bytes_per_block={smem} exceeds target '{target.name}' opt-in cap "
                    f"{target.smem_bytes_per_block_optin}")
    return errs


def _config_from_dict(config_dict: dict[str, Any] | ScheduleConfig | None,
                      target: GpuTarget) -> ScheduleConfig:
    """Build a ScheduleConfig from an agent-supplied dict (the Loop-2 edit surface). Goes through
    MegakernelProgram.from_dict's filter so unknown keys are dropped (additive compatibility) and
    a string-keyed explicit sm_assignment map is coerced to int keys."""
    if config_dict is None:
        return default_config(target)
    if isinstance(config_dict, ScheduleConfig):
        return config_dict
    # Reuse the IR's tolerant deserializer (drops unknown keys, fixes sm_assignment key types).
    known = MegakernelProgram._filter_known(ScheduleConfig, dict(config_dict))
    sa = known.get("sm_assignment")
    if isinstance(sa, dict):
        known["sm_assignment"] = {int(k): int(v) for k, v in sa.items()}
    return ScheduleConfig(**known)


# ======================================================================================
# 1) propose, the incumbent config + the editable search space
# ======================================================================================
def search_space(target: GpuTarget | None = None) -> dict[str, Any]:
    """The documented Loop-2 edit surface: every knob the agent may move, its choices/ranges, and
    a one-line description. This is the machine-readable companion to schemas/schedule_config.schema.json
    and HARNESS.md, an agent reads it to know exactly what it owns."""
    smem_cap = target.smem_bytes_per_block_optin if target else 101376
    return {
        "tiling.gemv.N_tile": {
            "type": "int", "choices": list(N_TILE_CHOICES), "default": 256,
            "desc": "GEMV output-column tile width (the one tiling knob the frozen lowerer "
                    "consumes). Wider = fewer tiles/less sync; narrower = more parallelism.",
        },
        "tiling.attention.kv_block": {
            "type": "int", "choices": list(KV_BLOCK_CHOICES), "default": 128,
            "desc": "KV window block size. Reserved/searchable; recorded but not yet lowered "
                    "(whole-window attention today).",
        },
        "fusion_grouping": {
            "type": "list[list[str]]",
            "choices": [[], [["gate", "up"]], [["gate", "up", "silu"]], [["rmsnorm", "gemv"]]],
            "default": [],
            "desc": "Op-name groups to co-resident into one fused task group. RESERVED: "
                    "recorded on the schedule and searchable, but NOT yet consumed by the frozen "
                    "lowerer (no effect on the emitted program today).",
        },
        "sm_assignment": {
            "type": "str|dict", "choices": list(SM_POLICY_CHOICES), "default": "load_balance",
            "desc": "SM placement policy ('round_robin'|'load_balance') or an explicit "
                    "{task_id: sm} map. RESERVED: recorded and searchable, but NOT yet consumed by "
                    "the frozen lowerer/loader (no effect on the emitted program today).",
        },
        "pipelining_depth": {
            "type": "int", "choices": list(PIPELINING_CHOICES), "default": 2,
            "desc": "Instructions ahead to prefetch weights (hides the inter-op HBM bubble). "
                    "0 = no prefetch.",
        },
        "page_allocation": {
            "type": "str", "choices": list(PAGE_POLICY_CHOICES), "default": "graph_color",
            "desc": "Activation page reuse policy ('graph_color'|'linear'|'none'). RESERVED: "
                    "recorded and searchable, but NOT yet consumed by the frozen lowerer (no "
                    "effect on the emitted program today).",
        },
        "threads_per_block": {
            "type": "int", "choices": list(THREADS_PER_BLOCK_CHOICES), "default": 256,
            "desc": "Persistent VM kernel block size (multiple of 32). Loader proves occupancy.",
        },
        "smem_bytes_per_block": {
            "type": "int", "min": 0, "max": smem_cap, "default": 0,
            "choices": [0, min(16384, smem_cap), min(49152, smem_cap)],
            "desc": f"Dynamic SMEM opt-in per block, bytes. Must be <= target cap ({smem_cap}). "
                    "Loader rejects an over-cap value before launch.",
        },
        # ---- kernel-knob sub-surface: GEMV build knobs, threaded to MegakernelVM(knobs=...) ----
        # Edit via a reserved "kernel_knobs" object in the config dict. Default set == the production
        # VM build (a config without kernel_knobs is byte-identical to the incumbent). These move
        # MEASURED latency (device=cuda); the analytic/CPU 'predicted' path does not model them.
        "kernel_knobs.cols_per_warp": {
            "type": "int", "choices": list(_KERNEL_KNOB_CHOICES["cols_per_warp"]), "default": 1,
            "desc": "Output columns one warp computes, x-reuse / memory-level parallelism in GEMV.",
        },
        "kernel_knobs.cpasync": {
            "type": "int", "choices": list(_KERNEL_KNOB_CHOICES["cpasync"]), "default": 1,
            "desc": "1 = cp.async double-buffered GEMV (latency-hiding); 0 = register/coalesced path.",
        },
        "kernel_knobs.cpa_stages": {
            "type": "int", "choices": list(_KERNEL_KNOB_CHOICES["cpa_stages"]), "default": 4,
            "desc": "cp.async ring depth, deeper pipeline hides more HBM latency (must fit SMEM).",
        },
        "kernel_knobs.cpa_cols": {
            "type": "int", "choices": list(_KERNEL_KNOB_CHOICES["cpa_cols"]), "default": 2,
            "desc": "Columns one warp streams at once under cp.async.",
        },
    }


def propose(model_id: str, gpu: str, *,
            incumbent: dict[str, Any] | ScheduleConfig | None = None) -> dict[str, Any]:
    """The 'read program.md' step. Return the current incumbent ScheduleConfig (as a dict) and the
    documented, editable search_space.

    Args:
      model_id:  'toy' / 'toy-2L' or a HuggingFace id (toy is the fully-supported path).
      gpu:       a registered GpuTarget name (e.g. 'rtx5090').
      incumbent: an optional starting config (dict or ScheduleConfig); defaults to the neutral
                 default_config(target) a non-searching compiler would emit.

    Returns ``{"model": ..., "gpu": ..., "schedule_config": <ScheduleConfig.to_dict>,
    "schedule_id": ..., "search_space": <knobs+ranges>, "schema": "schemas/schedule_config.schema.json"}``.
    """
    target = _resolve_target(gpu)
    cfg = _config_from_dict(incumbent, target) if incumbent is not None else default_config(target)
    cfg_dict = cfg.to_dict()
    return {
        "model": model_id,
        "gpu": target.name,
        "schedule_config": cfg_dict,
        "schedule_id": schedule_id(cfg_dict),
        "search_space": search_space(target),
        "schema": "schemas/schedule_config.schema.json",
        "edit_surface": "ScheduleConfig only, the frozen VM lowers it; validate() gates launch.",
    }


# ======================================================================================
# 2) evaluate, the structured JSON verdict for one config
# ======================================================================================
@dataclass
class Verdict:
    """The structured result of evaluating one ScheduleConfig. Serialized to JSON for the agent.

    The honesty contract lives in the field combination:
      * ``valid=False`` => the schedule was REJECTED by validate() (or failed to lower); a
        ``rejected_reason`` is set, and there is NO correctness claim and NO latency.
      * ``correct`` is the reference-VM-vs-eager verdict (authoritative). A latency only exists
        when ``correct is True``.
      * ``latency_kind`` distinguishes a real GPU measurement ('measured-gpu') from an analytic
        prediction ('predicted'). We never label a prediction as a measurement.
    """

    valid: bool
    rejected_reason: str | None = None
    correct: bool = False
    max_abs_err: float | None = None
    top1_agreement: float | None = None
    latency_us: float | None = None
    latency_kind: str | None = None          # 'measured-gpu' | 'predicted' | None
    pct_of_roofline: float | None = None
    bound_us: float | None = None
    schedule_id: str = ""
    tasks: int = 0
    weight_mb: float = 0.0
    # extra context (not part of the minimal schema but useful to an agent)
    gpu: str = ""
    model: str = ""
    device: str = ""
    n_buffers: int = 0
    n_counters: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "rejected_reason": self.rejected_reason,
            "correct": self.correct,
            "max_abs_err": self.max_abs_err,
            "top1_agreement": self.top1_agreement,
            "latency_us": self.latency_us,
            "latency_kind": self.latency_kind,
            "pct_of_roofline": self.pct_of_roofline,
            "bound_us": self.bound_us,
            "schedule_id": self.schedule_id,
            "tasks": self.tasks,
            "weight_mb": self.weight_mb,
            "gpu": self.gpu,
            "model": self.model,
            "device": self.device,
            "n_buffers": self.n_buffers,
            "n_counters": self.n_counters,
            "notes": self.notes,
        }


def _probe_inputs(token: int, device: str) -> dict[str, torch.Tensor]:
    return {
        TOKEN_NAME: torch.tensor([token], dtype=torch.int32),
        POS_NAME: torch.tensor([0], dtype=torch.int32),
        RESHAPE_ID_NAME: torch.tensor([0], dtype=torch.int32),
    }


def _evaluate_program(prog: MegakernelProgram, target: GpuTarget, model, eager_decode,
                      cfg_dict: dict[str, Any], device: str, token: int,
                      notes: list[str], kernel_knobs: dict[str, int] | None = None,
                      gate_dtype: torch.dtype = torch.float32) -> Verdict:
    """Shared correctness + latency core: assumes ``prog`` already validated OK. Used by both
    :func:`evaluate` and :func:`loop` so they share one honest implementation.

    ``kernel_knobs`` (the documented GEMV build knobs) are threaded to MegakernelVM(knobs=...) for
    the measured-gpu path only; correctness is the CPU ReferenceVM (knob-independent by construction),
    and the analytic prediction does not model build knobs (a CPU/predicted latency ignores them)."""
    from vm.reference_vm import ReferenceVM

    kk = _normalize_kernel_knobs(kernel_knobs)
    sid = _sid_with_knobs(cfg_dict, kk)
    weight_bytes = prog.total_weight_bytes()
    bound_us = target.bandwidth_bound_us(weight_bytes)

    # ---- correctness (AUTHORITATIVE): CPU reference VM vs eager PyTorch ----
    inputs = _probe_inputs(token, device)
    ref_logits = ReferenceVM(prog, model.weights_dict(), device="cpu").run(inputs, kv={})["logits"]
    eager_logits = eager_decode(token).to("cpu", torch.float32)
    cverdict = logit_equivalence(ref_logits, eager_logits, dtype=gate_dtype)

    v = Verdict(
        valid=True, rejected_reason=None,
        correct=bool(cverdict.correct),
        max_abs_err=float(cverdict.max_abs_err),
        top1_agreement=float(cverdict.top1_agreement),
        schedule_id=sid, tasks=len(prog.tasks),
        weight_mb=round(weight_bytes / 1e6, 4),
        bound_us=round(bound_us, 4) if math.isfinite(bound_us) else None,
        gpu=target.name, model=str(prog.meta.get("model", "")), device=device,
        n_buffers=len(prog.buffers), n_counters=len(prog.counters), notes=list(notes),
    )

    if not cverdict.correct:
        # No latency for a wrong kernel, the whole point of the honesty gate.
        v.notes.append("correctness FAIL vs eager, no latency emitted")
        return v

    # ---- latency: real GPU megakernel (correctness-gated) else analytic prediction ----
    measured_us: float | None = None
    if device == "cuda" and torch.cuda.is_available():
        try:
            from vm.loader import MegakernelVM
            gpu_in = {k: val.to("cuda") for k, val in inputs.items()}
            gvm = MegakernelVM(prog, model.weights_dict(), device="cuda", knobs=dict(kk))
            if not _knobs_are_default(kk):
                notes.append(f"kernel_knobs={kk}")
            gpu_logits = gvm.run(gpu_in, kv={})["logits"].to("cpu", torch.float32)
            gpu_verdict = logit_equivalence(gpu_logits, eager_logits, dtype=gate_dtype)
            if gpu_verdict.correct:
                bres = bench(lambda: gvm.run(gpu_in, kv={}), gpu_verdict,
                             warmup=10, iters=50, device="cuda", strict=True)
                measured_us = bres.latency_us
            else:
                v.notes.append(f"GPU megakernel ran but mismatched eager "
                               f"(max_abs_err={gpu_verdict.max_abs_err:.2e}); using prediction")
        except CorrectnessGateError as e:
            v.notes.append(f"GPU latency withheld (honesty gate): {e}")
        except Exception as e:  # GPU end-to-end path may not be wired for every op yet
            v.notes.append(f"GPU megakernel not runnable end-to-end ({type(e).__name__}: {e}); "
                           f"using cost-model prediction")

    if measured_us is not None and measured_us > 0:
        v.latency_us = round(float(measured_us), 4)
        v.latency_kind = "measured-gpu"
    else:
        v.latency_us = round(float(predict_us(prog, target)), 4)
        v.latency_kind = "predicted"

    rr = roofline_report(weight_bytes, v.latency_us, target)
    v.pct_of_roofline = round(rr.pct_of_bound, 3)

    # ---- PHYSICAL-FLOOR GUARD (honesty): a batch-1 decode must read every weight once, so its
    # latency CANNOT fall below the weights/bandwidth roofline floor (bound_us). A sub-floor number
    # is physically impossible, either a measured artifact (e.g. a silently-failed launch returning
    # a stale-but-correct buffer) or an over-optimistic cost-model fallback. We WITHHOLD it (null the
    # latency) rather than present an impossible win, the same discipline as the correctness gate. ----
    if (v.latency_us is not None and v.bound_us is not None and v.bound_us > 0
            and v.latency_us < v.bound_us):
        v.notes.append(
            f"latency {v.latency_us}us is BELOW the HBM roofline floor {v.bound_us}us "
            f"({v.latency_kind}), physically impossible for this memory-bound decode; "
            f"withheld as unreliable (likely an infeasible-launch fallback or a measurement artifact)")
        v.latency_us = None
        v.latency_kind = None
        v.pct_of_roofline = None
    return v


def evaluate(model_id: str, gpu: str, config_dict: dict[str, Any] | ScheduleConfig | None,
             device: str = "auto", *, token: int = _PROBE_TOKEN,
             _prepared: tuple | None = None) -> dict[str, Any]:
    """Evaluate ONE ScheduleConfig and return a structured JSON verdict (see :class:`Verdict`).

    Pipeline (never crashes on a bad config; a bad config is a clean ``valid=False``):
      1. lower(graph, target, config)               -- realize the config into a program
      2. validate(program)                          -- REJECT cleanly on deadlock/race/arity/...
      3. ReferenceVM correctness vs eager PyTorch   -- authoritative
      4. if correct & CUDA: MegakernelVM measured latency via eval.bench (correctness-gated)
         else: cost_model.predict_us (labelled 'predicted')
      5. roofline -> pct_of_roofline

    Args:
      model_id:    'toy' / 'toy-2L' or a HuggingFace id.
      gpu:         a registered GpuTarget name.
      config_dict: the agent-edited ScheduleConfig as a dict (or a ScheduleConfig, or None for the
                   default). Unknown keys are dropped (additive compatibility).
      device:      'auto' (cuda if available else cpu) | 'cuda' | 'cpu'. On cpu, latency is always
                   the analytic prediction (a CPU reference time is never a GPU perf number).
      _prepared:   internal fast-path (model, importer, eager_decode, graph, target) reused by loop().

    Returns the :class:`Verdict` as a plain dict.
    """
    target = _resolve_target(gpu)
    dev = _resolve_device(device)

    if _prepared is not None:
        model, _importer, eager_decode, graph = _prepared
    else:
        model, importer, eager_decode, _label = _load_model(model_id)
        graph = importer(model)

    # ---- extract the optional kernel-knob edit surface (reserved "kernel_knobs" key) ----
    kernel_knobs: dict[str, int] | None = None
    if isinstance(config_dict, dict) and "kernel_knobs" in config_dict:
        raw_kk = config_dict["kernel_knobs"]
        if raw_kk is not None and not isinstance(raw_kk, dict):
            return Verdict(valid=False,
                           rejected_reason=f"kernel_knobs must be an object, got "
                                           f"{type(raw_kk).__name__}",
                           gpu=target.name, model=model_id, device=dev,
                           notes=["malformed kernel_knobs, REJECTED before lowering"]).to_dict()
        kernel_knobs = _normalize_kernel_knobs(raw_kk)

    try:
        cfg = _config_from_dict(config_dict, target)
    except Exception as e:
        return Verdict(valid=False,
                       rejected_reason=f"malformed ScheduleConfig: {type(e).__name__}: {e}",
                       gpu=target.name, model=model_id, device=dev,
                       notes=["config dict could not be parsed, rejected, no launch"]).to_dict()
    cfg_dict = cfg.to_dict()
    sid = _sid_with_knobs(cfg_dict, kernel_knobs)

    # ---- 0. edit-surface type sanity (malformed knob -> clean REJECTED, never a deep crash) ----
    terrs = _config_type_errors(cfg)
    if terrs:
        return Verdict(valid=False, rejected_reason="; ".join(terrs), schedule_id=sid,
                       gpu=target.name, model=model_id, device=dev,
                       notes=["malformed config knob(s), REJECTED before lowering"]).to_dict()

    # ---- 1+2. lower + VALIDATE (a bad config is a clean REJECTED, never a crash) ----
    sched_dtype, torch_dtype = _infer_model_dtype(model)
    try:
        prog = lower(graph, target=target, config=cfg, pos=0, dtype=sched_dtype)
    except Exception as e:
        return Verdict(valid=False,
                       rejected_reason=f"lowering failed: {type(e).__name__}: {e}",
                       schedule_id=sid, gpu=target.name, model=model_id, device=dev,
                       notes=["lower() raised, config rejected, no launch"]).to_dict()

    vres = validate(prog)
    if not vres.ok:
        reason = "; ".join(vres.errors[:4]) or "validate() rejected the lowered program"
        return Verdict(valid=False, rejected_reason=reason, schedule_id=sid,
                       tasks=len(prog.tasks), weight_mb=round(prog.total_weight_bytes() / 1e6, 4),
                       gpu=target.name, model=model_id, device=dev,
                       n_buffers=len(prog.buffers), n_counters=len(prog.counters),
                       notes=["validate() REJECTED, schedule never launches (agent-safety)"]
                       ).to_dict()

    # ---- 2b. launch-config feasibility (the loader-level gate, surfaced as a clean REJECTED) ----
    lerrs = _launch_config_errors(cfg, target)
    if lerrs:
        return Verdict(valid=False, rejected_reason="; ".join(lerrs), schedule_id=sid,
                       tasks=len(prog.tasks), weight_mb=round(prog.total_weight_bytes() / 1e6, 4),
                       gpu=target.name, model=model_id, device=dev,
                       n_buffers=len(prog.buffers), n_counters=len(prog.counters),
                       notes=["launch config infeasible, REJECTED before launch (agent-safety)"]
                       ).to_dict()

    # ---- 3-5. correctness + latency + roofline ----
    v = _evaluate_program(prog, target, model, eager_decode, cfg_dict, dev, token, notes=[],
                          kernel_knobs=kernel_knobs, gate_dtype=torch_dtype)
    return v.to_dict()


# ======================================================================================
# 3) loop, the keep/revert autoresearch loop over proposed configs
# ======================================================================================
@dataclass
class LoopResult:
    best_verdict: dict[str, Any] | None
    best_config: dict[str, Any] | None
    rows: list[dict[str, Any]]
    n_trials: int
    n_valid: int
    n_correct: int
    results_tsv: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_verdict": self.best_verdict,
            "best_config": self.best_config,
            "rows": self.rows,
            "n_trials": self.n_trials,
            "n_valid": self.n_valid,
            "n_correct": self.n_correct,
            "results_tsv": self.results_tsv,
        }


# keep/revert: a candidate must be (a) valid + correct, and (b) strictly faster by >= this margin
# to dethrone the incumbent. Mirrors AutoKernel: correctness first, then a >=1% latency gain.
_MIN_GAIN = 0.01            # 1%, the AutoKernel move-on threshold
_KIND_RANK = {"measured-gpu": 0, "predicted": 1, None: 2}  # prefer a measured incumbent


def _config_complexity(cfg_dict: dict[str, Any]) -> int:
    """Simplicity tie-break score (lower = simpler). Counts non-default knob activity so that two
    schedules with equal latency resolve to the simpler one (the AutoKernel tie-break)."""
    score = 0
    score += len(cfg_dict.get("fusion_grouping", []) or [])
    score += int(cfg_dict.get("pipelining_depth", 0) or 0)
    score += 1 if cfg_dict.get("smem_bytes_per_block", 0) else 0
    score += 1 if cfg_dict.get("page_allocation") not in ("graph_color", None) else 0
    score += 1 if isinstance(cfg_dict.get("sm_assignment"), dict) else 0
    return score


def _better(cand: dict[str, Any], cand_cfg: dict[str, Any],
            best: dict[str, Any] | None, best_cfg: dict[str, Any] | None) -> bool:
    """keep/revert decision. A candidate replaces the incumbent iff:
      1. correctness first: incumbent must always be correct; a correct candidate beats none.
      2. then latency: strictly faster by >= _MIN_GAIN (1%). A measured number outranks a
         prediction at equal latency (we trust hardware over the model).
      3. simplicity tie-break: at ~equal latency + same kind, the simpler config wins.
    """
    if not (cand.get("valid") and cand.get("correct")):
        return False
    if best is None or not (best.get("valid") and best.get("correct")):
        return True
    cl, bl = cand.get("latency_us"), best.get("latency_us")
    if cl is None:
        return False
    if bl is None:
        return True
    # strict >=1% improvement dethrones outright
    if cl < bl * (1.0 - _MIN_GAIN):
        return True
    if cl > bl * (1.0 - _MIN_GAIN):
        # not enough of a gain; only a measured-over-predicted upgrade at ~equal latency, or a
        # simplicity win within 1%, may pass.
        if cl <= bl * (1.0 + _MIN_GAIN):
            ck = _KIND_RANK.get(cand.get("latency_kind"), 2)
            bk = _KIND_RANK.get(best.get("latency_kind"), 2)
            if ck < bk:
                return True
            if ck == bk and _config_complexity(cand_cfg) < _config_complexity(best_cfg or {}):
                return True
        return False
    return False


def loop(model_id: str, gpu: str, budget: int = 8, device: str = "auto", *,
         seed: int = 0, token: int = _PROBE_TOKEN,
         results_path: str = DEFAULT_RESULTS_TSV,
         corpus_path: str | None = DEFAULT_CORPUS,
         explore_fraction: float = 0.35,
         on_trial: Callable[[dict[str, Any]], None] | None = None,
         verbose: bool = False) -> dict[str, Any]:
    """Run the Loop-2 keep/revert autoresearch loop and return the best verdict + the log rows.

    The loop proposes ScheduleConfigs (default -> a slice of fresh random configs -> mutations of
    the running best), evaluates each via :func:`evaluate`'s honest pipeline, logs EVERY trial to
    ``results.tsv`` (via flywheel.log), and keeps the best VALID + CORRECT schedule by the
    keep/revert rule (correctness first, then a >= 1% latency gain, simplicity tie-break). Kept
    correct points additionally enter the flywheel corpus.

    Args:
      model_id/gpu/device: as :func:`evaluate`.
      budget:        number of configs to try (>= 1). budget=1 evaluates only the default.
      seed:          RNG seed for reproducible proposals.
      results_path:  the results.tsv to append to (default workspace/results.tsv).
      corpus_path:   the flywheel corpus.jsonl for kept points (None to skip corpus writes).
      explore_fraction: share of the budget spent on fresh random configs (rest mutates the best).
      on_trial:      optional callback invoked with each trial's verdict dict (live streaming).

    Returns a :class:`LoopResult` as a dict: best_verdict, best_config, rows, counts, results_tsv.
    """
    import random as _random

    if budget < 1:
        raise ValueError("loop budget must be >= 1")
    target = _resolve_target(gpu)
    dev = _resolve_device(device)
    rng = _random.Random(seed)

    # Prepare the model + graph ONCE and reuse across trials (the only expensive setup).
    model, importer, eager_decode, label = _load_model(model_id)
    graph = importer(model)
    prepared = (model, importer, eager_decode, graph)

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    rows: list[dict[str, Any]] = []
    best_verdict: dict[str, Any] | None = None
    best_cfg_dict: dict[str, Any] | None = None
    incumbent_cfg: ScheduleConfig = default_config(target)
    incumbent_knobs: dict[str, int] = dict(_KERNEL_KNOB_DEFAULTS)
    n_valid = n_correct = 0
    n_explore = max(0, int(round((budget - 1) * explore_fraction)))
    # kernel knobs only move MEASURED latency, so only search them on the GPU path (on cpu the
    # cost-model fitness ignores them, searching them there would be noise).
    search_knobs = (dev == "cuda")

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    for i in range(budget):
        # ---- propose a COMBINED candidate (schedule + kernel_knobs): trial 0 = default; then
        #      explore (random) then exploit (mutate one knob, schedule or kernel). ----
        if i == 0:
            cfg = default_config(target)
            knobs = dict(_KERNEL_KNOB_DEFAULTS)
            source = "default"
        elif i <= n_explore or best_verdict is None:
            cfg = random_config(rng, target)
            knobs = _random_kernel_knobs(rng) if search_knobs else dict(_KERNEL_KNOB_DEFAULTS)
            source = "random"
        else:
            # exploit: flip a coin to mutate either the schedule or one kernel knob (single-knob
            # local moves keep the hill-climb smooth across the joint space).
            if search_knobs and rng.random() < 0.5:
                cfg = incumbent_cfg
                knobs = _mutate_kernel_knobs(incumbent_knobs, rng)
            else:
                cfg = mutate_config(incumbent_cfg, rng, target)
                knobs = dict(incumbent_knobs)
            source = "mutate"
        cfg_dict = cfg.to_dict()
        # the candidate dict carries kernel_knobs only when non-default (keeps knob-free runs and
        # their schedule_ids byte-identical to the historical behaviour).
        cand_dict = dict(cfg_dict)
        if not _knobs_are_default(knobs):
            cand_dict["kernel_knobs"] = knobs

        verdict = evaluate(model_id, gpu, cand_dict, device=dev, token=token, _prepared=prepared)
        verdict["source"] = source
        verdict["trial"] = i
        rows.append(verdict)
        if on_trial:
            on_trial(verdict)

        if verdict.get("valid"):
            n_valid += 1
        if verdict.get("correct"):
            n_correct += 1

        # ---- keep/revert ----
        kept = _better(verdict, cand_dict, best_verdict, best_cfg_dict)
        if kept:
            best_verdict = verdict
            best_cfg_dict = cand_dict
            incumbent_cfg = cfg
            incumbent_knobs = dict(knobs)

        # ---- log EVERY trial to results.tsv (the flywheel substrate) ----
        if verdict.get("valid") and verdict.get("correct"):
            correctness = "PASS"
        elif not verdict.get("valid"):
            correctness = "REJECTED"
        else:
            correctness = "FAIL"
        row = ResultRow(
            experiment=i, tag="kept" if kept else ("rejected" if not verdict["valid"]
                                                   else ("revert" if correctness != "PASS" else "tried")),
            loop="schedule", model=label, gpu=target.name, regime="single-stream",
            correctness=correctness,
            latency_us=verdict["latency_us"] if verdict.get("latency_us") is not None else "",
            pct_of_roofline=(verdict["pct_of_roofline"]
                             if verdict.get("pct_of_roofline") is not None else ""),
            schedule_id=verdict.get("schedule_id", ""), kernel_id="",
            description=f"{source}; kind={verdict.get('latency_kind')}; "
                        f"{(verdict.get('rejected_reason') or '')[:80]}")
        append_result(row, path=results_path)
        _log(f"  [{i}] {source} valid={verdict['valid']} correct={verdict['correct']} "
             f"lat={verdict.get('latency_us')}us ({verdict.get('latency_kind')}) kept={kept}")

        # kept correct points enter the flywheel corpus (the learned-prior moat)
        if kept and corpus_path and verdict.get("correct") and verdict.get("latency_us") is not None:
            try:
                append_corpus(CorpusRecord(
                    model=label, gpu=target.name, regime="single-stream", correctness="PASS",
                    latency_us=float(verdict["latency_us"]),
                    bound_us=float(verdict["bound_us"]) if verdict.get("bound_us") else 0.0,
                    pct_of_roofline=float(verdict["pct_of_roofline"] or 0.0),
                    schedule=cand_dict, ir_version="", abi_version="",
                    notes=f"loop kept; kind={verdict.get('latency_kind')}"),
                    path=corpus_path)
            except Exception as e:  # corpus write must never break the loop
                _log(f"  (corpus write skipped: {type(e).__name__}: {e})")

    return LoopResult(
        best_verdict=best_verdict, best_config=best_cfg_dict, rows=rows,
        n_trials=budget, n_valid=n_valid, n_correct=n_correct,
        results_tsv=results_path,
    ).to_dict()


__all__ = ["propose", "evaluate", "loop", "search_space", "Verdict", "LoopResult",
           "generate", "GenerateResult"]
