#!/usr/bin/env python3
"""
AMK, UNATTENDED AUTORESEARCH DRIVER (the "point it and sleep" loop)
===================================================================

``autoresearch(model, gpu, iters|minutes, ...)`` is the headless driver you point at a
``(model, gpu)`` and walk away from. It proposes a :class:`~schedule.ir.ScheduleConfig`, lowers +
validates it, correctness-gated-evaluates it (measured GPU latency via the megakernel VM, or the
analytic cost model on CPU, always honestly labelled), keeps it only if it is CORRECT *and*
>= 1% faster than the incumbent (else reverts), records the outcome to the
:mod:`amk_orchestrate` campaign + ``results.tsv`` + the flywheel corpus, and CHECKPOINTS the state
every iteration, for hours, unattended, resumable, and crash-proof.

THE FLYWHEEL MAKES IT START SMARTER (the cross-run moat)
-------------------------------------------------------
Unless run ``cold``, each iteration's proposal is biased by :mod:`flywheel.prior`:
  * the FIRST incumbent is the best ``warm_start`` seed from the corpus for the nearest
    ``(model_shape, gpu)`` shape, a warm run begins from accumulated knowledge, not the textbook
    default; and
  * exploitation proposals are RANKED best-first by the learned/kNN prior (``prior.rank``), so the
    loop spends its budget on the configs the corpus predicts are good. Exploration (mutation /
    fresh random) is mixed in epsilon-greedy so the search never collapses onto the prior.

The prior is ADVISORY only: every proposal still passes the same lower -> validate -> correctness
-> keep/revert gate, so a bad suggestion can only ever lose, never corrupt the search or emit a
dishonest number.

ROBUSTNESS (it must survive 10 hours of unattended running)
----------------------------------------------------------
Every iteration is wrapped in try/except. A crash / CUDA error / timeout in one iteration is
logged as ``CRASH`` / ``TIMEOUT`` and the loop CONTINUES, one failure never stops the run. The
state is checkpointed every iteration, so a re-run resumes the same campaign from where it left
off (the orchestrator state file is the single source of truth).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
import amk_orchestrate as orch  # noqa: E402
from flywheel import prior  # noqa: E402
from flywheel.log import CorpusRecord, append_corpus, schedule_id  # noqa: E402
from flywheel.prior import SEARCHABLE_KNOB_CHOICES as _KNOB_CHOICES_REF  # noqa: E402
from flywheel.prior import SEARCHABLE_KNOB_DEFAULTS as _KNOB_DEFAULTS_REF  # noqa: E402
from schedule.ir import TARGETS, ScheduleConfig  # noqa: E402
from schedule.search import default_config, mutate_config, random_config  # noqa: E402

DEFAULT_CORPUS = os.path.join("flywheel", "corpus.jsonl")
DEFAULT_RESULTS = os.path.join("workspace", "results.tsv")

# keep/revert: a candidate must be correct AND strictly faster by this margin to dethrone the
# incumbent. Mirrors AutoKernel / harness.loop: correctness first, then a >= 1% latency gain.
MIN_GAIN = 0.01
# epsilon-greedy: probability an exploitation iteration is replaced by a fresh-random EXPLORE move
# even when the prior could guide it (keeps the search from collapsing onto the corpus).
DEFAULT_EPSILON = 0.30
# how many fresh prior-ranked candidates to draw + rank per exploitation iteration (we evaluate the
# single best, ranking is the cheap part; evaluation is the expensive part).
RANK_POOL = 12

# ============================================================================================
# COMBINED CANDIDATE = (ScheduleConfig, kernel_knobs).
# ============================================================================================
# The ScheduleConfig alone has a near-flat measured landscape on this VM (its biggest GEMV lever,
# N_tile, is auto-sized in schedule/lower.py). The REAL measured headroom (the autotune work proved
# 1.2-1.3x) is the MegakernelVM compile-time BUILD knobs in vm/loader.py. So the autoresearch
# candidate carries BOTH: the schedule + a kernel_knobs dict the cuda eval passes to
# MegakernelVM(prog, weights, knobs=...). These are the knobs the loader accepts; we search the
# subset that the autotune / fat-tile / cp.async experiments showed actually move measured latency.
#   cols_per_warp : output columns a warp computes (x-reuse -> memory-level parallelism)
#   cpasync       : 1 -> cp.async double-buffered GEMV (the production latency-hiding path); 0 -> the
#                   register/coalesced path
#   cpa_stages    : cp.async ring depth (deeper pipeline hides more HBM latency; must fit SMEM)
#   cpa_cols      : columns one warp streams at once under cp.async
# N_tile (GEMV tile width) and threads_per_block live in the ScheduleConfig (tiling.gemv.N_tile /
# threads_per_block) and are searched there; they are the schedule-side levers that ALSO move
# measured latency (thinner tiles -> more parallelism, proven 1.23x).
#
# SINGLE SOURCE OF TRUTH: choices + defaults live in flywheel.prior.SEARCHABLE_KNOB_CHOICES /
# SEARCHABLE_KNOB_DEFAULTS, imported above as _KNOB_CHOICES_REF / _KNOB_DEFAULTS_REF.
# The public names KNOB_CHOICES / KNOB_DEFAULTS below are aliases kept for callers.
KNOB_CHOICES: dict[str, tuple[int, ...]] = _KNOB_CHOICES_REF
# N_tile override choices (explicit tile width; smaller => more tiles => more parallelism). 0 means
# "let the lowerer auto-size" (the proven ~32 default). We include the auto path plus a few widths.
N_TILE_KNOB_CHOICES = (0, 16, 32, 64, 128)
# default kernel knobs (the prior production VM build); mirrors flywheel.prior.SEARCHABLE_KNOB_DEFAULTS.
KNOB_DEFAULTS: dict[str, int] = dict(_KNOB_DEFAULTS_REF)


def default_knobs() -> dict[str, int]:
    return dict(KNOB_DEFAULTS)


def random_knobs(rng: random.Random) -> dict[str, int]:
    return {k: rng.choice(v) for k, v in KNOB_CHOICES.items()}


def mutate_knobs(knobs: dict[str, int], rng: random.Random) -> dict[str, int]:
    """Change ONE kernel knob (local move; keeps the keep/revert hill-climb smooth)."""
    out = dict(KNOB_DEFAULTS)
    out.update(knobs or {})
    k = rng.choice(list(KNOB_CHOICES))
    out[k] = rng.choice(KNOB_CHOICES[k])
    return out


def _norm_knobs(knobs: dict[str, int] | None) -> dict[str, int]:
    out = dict(KNOB_DEFAULTS)
    if knobs:
        for k, v in knobs.items():
            if k in KNOB_DEFAULTS:
                try:
                    out[k] = int(v)
                except (TypeError, ValueError):
                    pass
    return out


def _candidate_dict(cfg: ScheduleConfig, knobs: dict[str, int]) -> dict[str, Any]:
    """The JSON-serializable combined candidate: the ScheduleConfig dict with the kernel_knobs
    embedded under a reserved key (so it round-trips through the corpus + the schedule_id hash)."""
    d = _cfg_to_dict(cfg)
    d["kernel_knobs"] = _norm_knobs(knobs)
    return d


def _split_candidate(cand: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Inverse of _candidate_dict: (schedule_dict_without_knobs, kernel_knobs)."""
    d = dict(cand)
    knobs = _norm_knobs(d.pop("kernel_knobs", None))
    return d, knobs


@dataclass
class AutoresearchResult:
    model: str
    gpu: str
    device: str
    cold: bool
    iters_run: int = 0
    baseline_us: float | None = None
    start_incumbent_us: float | None = None   # the first kept incumbent (warm seed or default)
    best_us: float | None = None
    best_config: dict[str, Any] | None = None
    best_pct_roofline: float | None = None
    best_kind: str | None = None
    speedup_vs_baseline: float | None = None
    iters_to_best: int | None = None
    n_kept: int = 0
    n_correct: int = 0
    n_rejected: int = 0
    n_crash: int = 0
    warm_seeds: int = 0
    ranker_trained: bool = False
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    state_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def _cfg_to_dict(cfg: ScheduleConfig | dict[str, Any]) -> dict[str, Any]:
    return cfg.to_dict() if isinstance(cfg, ScheduleConfig) else dict(cfg)


def _region_breakdown(model_id: str, gpu: str, cfg_dict: dict[str, Any],
                      prepared: tuple) -> dict[str, float] | None:
    """Best-effort per-region (attention/mlp/lm_head) share of the critical path for a config, via
    the analytic cost model. Advisory only (drives orchestrator targeting); never raises."""
    try:
        from schedule.cost_model import estimate
        from schedule.ir import DType
        from schedule.lower import lower
        model, _importer, _eager, graph = prepared
        target = TARGETS[gpu]
        cfg = harness._config_from_dict(cfg_dict, target)
        prog = lower(graph, target=target, config=cfg, pos=0, dtype=DType.F32)
        bd = estimate(prog, target)
        fr = bd.region_fractions()
        return {r: float(fr.get(r, 0.0)) for r in orch.REGIONS}
    except Exception:
        return None


def _mutate_combined(cfg: ScheduleConfig, knobs: dict[str, int], rng: random.Random,
                     target) -> tuple[ScheduleConfig, dict[str, int]]:
    """Local move on the COMBINED candidate: flip a coin to mutate either the ScheduleConfig (its
    levers: N_tile/threads_per_block/...) or a kernel knob (cp.async/cols_per_warp/...). Single-knob
    moves keep the keep/revert hill-climb smooth across the joint space."""
    if rng.random() < 0.5:
        # mutate a kernel knob; sometimes also flip the schedule's N_tile (the schedule-side lever
        # that also moves measured latency).
        new_knobs = mutate_knobs(knobs, rng)
        if rng.random() < 0.35:
            cfg = _with_n_tile(cfg, rng.choice(N_TILE_KNOB_CHOICES))
        return cfg, new_knobs
    # mutate the schedule (incl. its own N_tile/threads_per_block sweep).
    cfg2 = mutate_config(cfg, rng, target)
    if rng.random() < 0.35:
        cfg2 = _with_n_tile(cfg2, rng.choice(N_TILE_KNOB_CHOICES))
    return cfg2, dict(knobs)


def _random_combined(rng: random.Random, target) -> tuple[ScheduleConfig, dict[str, int]]:
    cfg = random_config(rng, target)
    cfg = _with_n_tile(cfg, rng.choice(N_TILE_KNOB_CHOICES))
    return cfg, random_knobs(rng)


def _with_n_tile(cfg: ScheduleConfig, n_tile: int) -> ScheduleConfig:
    """Return cfg with the GEMV tile width set (n_tile==0 -> drop the explicit override so the
    lowerer auto-sizes via base_width)."""
    from schedule.ir import replace
    tiling = {k: dict(v) for k, v in cfg.tiling.items()}
    gemv = tiling.setdefault("gemv", {})
    if n_tile and n_tile > 0:
        gemv["N_tile"] = int(n_tile)
    else:
        gemv.pop("N_tile", None)
    return replace(cfg, tiling=tiling)


def _propose(rng: random.Random, iter_idx: int, *, cold: bool, epsilon: float,
             incumbent_cfg: ScheduleConfig, incumbent_knobs: dict[str, int],
             target, model_shape, gpu: str, corpus_path: str, ranker,
             warm_seed_queue: list[tuple[ScheduleConfig, dict[str, int]]],
             tried: set[str]) -> tuple[ScheduleConfig, dict[str, int], str]:
    """Pick the next COMBINED candidate (ScheduleConfig, kernel_knobs) + a one-word source tag.

    Cold run: default -> fresh random (pure exploration; grows the corpus honestly).
    Warm run: drain warm_start seeds first (exploitation of the best known), then epsilon-greedy
    between prior-ranked fresh candidates (exploit) and mutation/random (explore).
    """
    # Iteration 0 is always the starting incumbent (default or the top warm seed), handled by caller.
    if cold:
        if rng.random() < 0.5:
            c, k = _mutate_combined(incumbent_cfg, incumbent_knobs, rng, target)
            return c, k, "mutate"
        c, k = _random_combined(rng, target)
        return c, k, "random"

    # warm: consume any remaining warm_start seeds (best-known candidates) before searching.
    while warm_seed_queue:
        scfg, sknobs = warm_seed_queue.pop(0)
        if schedule_id(_candidate_dict(scfg, sknobs)) not in tried:
            return scfg, sknobs, "warm_seed"

    # epsilon-greedy explore.
    if rng.random() < epsilon or ranker is None or not getattr(ranker, "trained", False):
        if rng.random() < 0.5:
            c, k = _mutate_combined(incumbent_cfg, incumbent_knobs, rng, target)
            return c, k, "mutate"
        c, k = _random_combined(rng, target)
        return c, k, "random"

    # exploit: draw a pool of fresh COMBINED candidates, rank best-first by the learned prior (which
    # featurizes the kernel_knobs too), take the best one we have not already tried.
    pool: list[tuple[ScheduleConfig, dict[str, int]]] = [
        _mutate_combined(incumbent_cfg, incumbent_knobs, rng, target) for _ in range(RANK_POOL // 2)]
    pool += [_random_combined(rng, target) for _ in range(RANK_POOL - len(pool))]
    pool_dicts = [_candidate_dict(c, k) for c, k in pool]
    ranked = prior.rank(pool_dicts, model_shape, gpu, corpus_path, ranker=ranker)
    for cand_dict in ranked:
        d = cand_dict if isinstance(cand_dict, dict) else cand_dict.to_dict()
        if schedule_id(d) in tried:
            continue
        sd, kn = _split_candidate(d)
        return harness._config_from_dict(sd, target), kn, "prior"
    c, k = _random_combined(rng, target)
    return c, k, "random"


# ============================================================================================
# MEASURED CUDA EVAL with kernel knobs (noise-robust, correctness-gated).
# ============================================================================================
class MeasuredCudaEvaluator:
    """Builds a MegakernelVM with the candidate's kernel_knobs and MEASURES decode latency on the
    GPU, correctness-gated against the CPU ReferenceVM (== eager) golden logits.

    NOISE ROBUSTNESS (WDDM clock drift is real on this laptop GPU):
      * each measurement is warmup>=25, iters>=100 CUDA-event medians on the persistent
        steady-state run() path (tables built once, then re-fired);
      * the keep/revert decision is made by re-measuring the candidate AND the current incumbent
        BACK-TO-BACK over ``rounds`` interleaved passes (so both see the same clock state) and
        comparing their medians, apples-to-apples. A candidate is kept only if it is faster than
        the incumbent by ``margin`` (a fraction beating measurement noise).

    Distinct knob-sets trigger an nvcc rebuild (cached in vm.loader after the first build of that
    variant), which is expensive but happens once per distinct variant, budget accordingly."""

    def __init__(self, model_obj, graph, target, *, dtype, warmup: int, iters: int,
                 rounds: int, rtol: float = 2e-2, atol: float = 2e-2):
        import torch
        from schedule.lower import lower, POS_NAME, RESHAPE_ID_NAME, TOKEN_NAME, required_inputs
        from schedule.ir import validate
        from vm.reference_vm import ReferenceVM
        self._torch = torch
        self._lower = lower
        self._validate = validate
        self.model = model_obj
        self.graph = graph
        self.target = target
        self.dtype = dtype
        self.warmup = warmup
        self.iters = iters
        self.rounds = rounds
        self.rtol, self.atol = rtol, atol
        self.weights = model_obj.weights_dict()
        # one probe token / pos=0 decode step (matches vm/autotune.py + harness probe).
        contract = required_inputs(0)
        self._inputs = {
            TOKEN_NAME: torch.tensor([11], dtype=torch.int32),
            POS_NAME: torch.tensor([0], dtype=torch.int32),
            RESHAPE_ID_NAME: torch.tensor([int(contract[RESHAPE_ID_NAME][0])], dtype=torch.int32),
        }
        # GOLDEN correctness oracle: CPU ReferenceVM (== eager) on the default-config program.
        from schedule.search import default_config
        ref_prog = self._lower_cfg(default_config(target))
        self._golden = ReferenceVM(ref_prog, self.weights, device="cpu").run(
            self._inputs, kv={})["logits"].detach().cpu().to(torch.float32)
        # PHYSICAL FLOOR (honesty guard): a batch-1 decode must read every weight once, so its
        # latency CANNOT fall below weights_bytes / HBM_bandwidth. A measured value below this floor
        # is an ARTIFACT (e.g. a silently-failed launch returning a stale-but-correct buffer) and is
        # NEVER banked as a win, the same discipline as the correctness gate. Computed once (weights
        # are fixed). A tiny epsilon (0.5%) absorbs timer granularity at the floor.
        self._floor_us = target.bandwidth_bound_us(ref_prog.total_weight_bytes()) * 0.995
        # incumbent (best) VM kept resident for interleaved back-to-back comparison.
        self._best_vm = None
        self._best_lat: float | None = None

    def _lower_cfg(self, cfg):
        prog = self._lower(self.graph, target=self.target, config=cfg, pos=0, dtype=self.dtype)
        vres = self._validate(prog)
        if not vres.ok:
            raise ValueError("lowered program rejected: " + "; ".join(vres.errors[:3]))
        return prog

    def _build_vm(self, cfg, knobs):
        from vm.loader import MegakernelVM
        prog = self._lower_cfg(cfg)
        vm = MegakernelVM(prog, self.weights, device="cuda", knobs=dict(knobs))
        return vm

    def _check_correct(self, vm) -> tuple[bool, float]:
        torch = self._torch
        out = vm.run(self._inputs, kv={})
        if vm.last_status.get("status") != "OK":
            return False, float("inf")
        gpu_f = out["logits"].detach().cpu().to(torch.float32)
        max_err = (gpu_f - self._golden).abs().max().item()
        ok = torch.allclose(gpu_f, self._golden, rtol=self.rtol, atol=self.atol)
        return ok, max_err

    def _median_latency(self, vm) -> float:
        """Steady-state per-token latency (us) via CUDA events; tables already built by run()."""
        torch = self._torch
        for _ in range(self.warmup):
            vm.run(self._inputs, kv={})
        torch.cuda.synchronize()
        samples = []
        for _ in range(self.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            vm.run(self._inputs, kv={})
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1e3)  # ms -> us
        samples.sort()
        return samples[len(samples) // 2]

    def _interleaved(self, cand_vm, best_vm) -> tuple[float, float]:
        """Measure candidate and incumbent BACK-TO-BACK over self.rounds passes and return the
        (median-of-round-medians) for each, so both see the same WDDM clock state (apples-to-apples).
        """
        torch = self._torch
        # one warmup pass on each to settle clocks/tables before the timed rounds.
        cand_vm.run(self._inputs, kv={})
        best_vm.run(self._inputs, kv={})
        torch.cuda.synchronize()
        cand_lats, best_lats = [], []
        for _ in range(self.rounds):
            cand_lats.append(self._median_latency(cand_vm))
            best_lats.append(self._median_latency(best_vm))
        cand_lats.sort()
        best_lats.sort()
        return cand_lats[len(cand_lats) // 2], best_lats[len(best_lats) // 2]

    def evaluate(self, cfg, knobs, *, is_incumbent_seed: bool, margin: float) -> dict:
        """Build + correctness-gate + measure the candidate. Returns a verdict dict:
          {valid, correct, latency_us, latency_kind, max_abs_err, improved, best_us}
        On the FIRST kept candidate (is_incumbent_seed or no current best), the candidate becomes
        the resident incumbent and its measured latency is the bar. Otherwise the candidate and the
        resident incumbent are re-measured interleaved and the candidate is kept iff it beats the
        incumbent by ``margin``."""
        v = {"valid": True, "correct": False, "latency_us": None, "latency_kind": "measured-gpu",
             "max_abs_err": None, "improved": False, "best_us": self._best_lat,
             "rejected_reason": None}
        try:
            cand_vm = self._build_vm(cfg, knobs)
        except (ValueError,) as e:
            v["valid"] = False
            v["rejected_reason"] = str(e)[:160]
            return v
        ok, max_err = self._check_correct(cand_vm)
        v["max_abs_err"] = max_err
        if not ok:
            v["correct"] = False
            return v
        v["correct"] = True

        if self._best_vm is None:
            # first correct candidate establishes the incumbent + the bar.
            cand_lat = self._median_latency(cand_vm)
            if cand_lat < self._floor_us:
                # physically impossible -> artifact, never seed the incumbent with it.
                v["valid"] = False
                v["rejected_reason"] = (f"latency {cand_lat:.1f}us below HBM floor "
                                        f"{self._floor_us:.1f}us, artifact, withheld")
                return v
            self._best_vm = cand_vm
            self._best_lat = cand_lat
            v["latency_us"] = round(cand_lat, 4)
            v["improved"] = True
            v["best_us"] = round(cand_lat, 4)
            return v

        # interleaved apples-to-apples comparison vs the resident incumbent.
        cand_lat, best_lat = self._interleaved(cand_vm, self._best_vm)
        if cand_lat < self._floor_us:
            # physically impossible candidate reading -> artifact; keep the incumbent, never bank it.
            v["valid"] = False
            v["rejected_reason"] = (f"latency {cand_lat:.1f}us below HBM floor "
                                    f"{self._floor_us:.1f}us, artifact, withheld")
            v["best_us"] = round(self._best_lat, 4) if self._best_lat else None
            return v
        self._best_lat = best_lat  # refresh the incumbent's measured value at the current clock
        v["latency_us"] = round(cand_lat, 4)
        if cand_lat < best_lat * (1.0 - margin):
            self._best_vm = cand_vm
            self._best_lat = cand_lat
            v["improved"] = True
        v["best_us"] = round(self._best_lat, 4)
        return v


def autoresearch(model: str, gpu: str, *,
                 iters: int | None = None, minutes: float | None = None,
                 device: str = "auto", cold: bool = False, seed: int = 0,
                 epsilon: float = DEFAULT_EPSILON,
                 corpus_path: str = DEFAULT_CORPUS,
                 results_path: str = DEFAULT_RESULTS,
                 state_path: str | None = None,
                 min_gain: float = MIN_GAIN,
                 overnight: bool = False,
                 restart_after: int = 6,
                 verbose: bool = True) -> dict[str, Any]:
    """Run the unattended keep/revert autoresearch loop for ``iters`` iterations OR ``minutes``
    wall-clock (whichever is given; if both, whichever is hit first). Returns an
    :class:`AutoresearchResult` as a dict. Never raises on a per-iteration failure; resumable from
    the orchestrator checkpoint.

    OVERNIGHT MODE (``overnight=True``, intended with a long ``--minutes``): the loop is built to
    run for hours and keep improving instead of stopping at the first plateau -
      * it does NOT stop on the consecutive-revert plateau (only ``minutes``/``iters`` end it);
      * after ``restart_after`` consecutive non-improvements it BASIN-HOPS: it jumps the exploration
        incumbent to a fresh random ``(schedule, kernel_knobs)`` point and resets the plateau
        counter, so the search escapes a stuck region, while the GLOBAL best is preserved (the
        measured evaluator always keeps/reverts against the resident global-best VM, so a restart can
        only ever discover something better, never lose the best found so far);
      * it frees reverted candidate VMs and trims the CUDA cache periodically (bounded memory over
        thousands of iterations);
      * it writes a wake-up report (``workspace/amk_overnight_report.{json,md}``) periodically and at
        the end, so you can ``run it and sleep`` and read the morning summary.
    The morning best is always a correct, measured, drift-robust (interleaved) win over the default;
    it is faster than AMK's own baseline, not a claim of beating cuBLAS/vLLM (see HARNESS.md)."""
    if iters is None and minutes is None:
        iters = 20
    if gpu not in TARGETS:
        raise SystemExit(f"unknown --gpu {gpu!r}; known: {', '.join(sorted(TARGETS))}")
    target = TARGETS[gpu]
    dev = _resolve_device(device)
    rng = random.Random(seed)
    state_path = state_path or orch.STATE_PATH

    def log(m: str) -> None:
        if verbose:
            print(m, flush=True)

    log(f"== amk autoresearch {model} --gpu {gpu} "
        f"({'COLD' if cold else 'WARM'}, device={dev}, "
        f"iters={iters}, minutes={minutes}, seed={seed}) ==")

    # ---- prepare the model + graph ONCE (the only expensive setup), reused across iters ----
    model_obj, importer, eager_decode, label = harness._load_model(model)
    graph = importer(model_obj)
    prepared = (model_obj, importer, eager_decode, graph)
    model_shape = prior.ModelShape.from_graph(graph)

    # ---- campaign state (resumable): one state file per (model, gpu) campaign ----
    state = orch.get_or_create_state(label, target.name, state_path)
    state["_state_path"] = state_path
    started_run = time.time()
    base_experiments = state.get("experiments_run", 0)
    if base_experiments:
        log(f"  RESUMING campaign: {base_experiments} prior experiments, "
            f"best so far {state.get('best_us')}us")

    # ---- the flywheel prior (warm only): seeds + a learned ranker over the corpus ----
    result = AutoresearchResult(model=label, gpu=target.name, device=dev, cold=cold,
                                state_path=state_path)
    result.baseline_us = state.get("baseline_us")
    result.best_us = state.get("best_us")
    result.best_config = state.get("best_config")

    warm_seeds: list[tuple[ScheduleConfig, dict[str, int]]] = []
    ranker = None
    if not cold:
        try:
            warm_seeds = prior.warm_start_combined(model_shape, target.name, corpus_path, k=3)
            ranker = prior.make_ranker(target.name, corpus_path)
        except Exception as e:
            log(f"  (prior unavailable, continuing cold-style: {type(e).__name__}: {e})")
    result.warm_seeds = len(warm_seeds)
    result.ranker_trained = bool(getattr(ranker, "trained", False))
    log(f"  flywheel: {len(warm_seeds)} warm seed(s), "
        f"ranker {'trained (' + getattr(ranker, 'backend', '?') + ')' if result.ranker_trained else 'cold (pure exploration)'}")

    # ---- the running incumbent for THIS run (start from warm seed if available, else default) ----
    if warm_seeds:
        incumbent_cfg, incumbent_knobs = warm_seeds[0]
        start_source = "warm_seed"
    else:
        incumbent_cfg = default_config(target)
        incumbent_knobs = default_knobs()
        start_source = "default"
    best_cfg_dict: dict[str, Any] | None = state.get("best_config")
    best_us: float | None = state.get("best_us")
    warm_seed_queue = list(warm_seeds[1:])  # seed[0] is the iter-0 incumbent
    tried: set[str] = set()

    # ---- the MEASURED cuda evaluator (built lazily on the cuda path): builds the VM with the
    #      candidate kernel_knobs + measures decode latency interleaved vs the resident incumbent. --
    use_measured = (dev == "cuda")
    cuda_eval: MeasuredCudaEvaluator | None = None
    if use_measured:
        try:
            import torch
            from schedule.ir import DType
            torch_dtype = next(model_obj.parameters()).dtype
            ir_dtype = (DType.BF16 if torch_dtype == torch.bfloat16
                        else DType.F16 if torch_dtype == torch.float16 else DType.F32)
            cuda_eval = MeasuredCudaEvaluator(
                model_obj, graph, target, dtype=ir_dtype,
                warmup=25, iters=100, rounds=2)
            log(f"  measured-cuda eval: dtype={ir_dtype.name}, warmup=25 iters=100 rounds=2, "
                f"correctness-gated vs CPU ReferenceVM, interleaved keep/revert")
        except Exception as e:
            log(f"  (measured-cuda eval unavailable: {type(e).__name__}: {e}; "
                f"falling back to cost-model)")
            use_measured = False
            cuda_eval = None
    # margin that must beat measurement noise to KEEP a candidate (>2-3% on this WDDM GPU).
    measured_margin = max(min_gain, 0.02)

    def _budget_left(i: int) -> bool:
        if iters is not None and i >= iters:
            return False
        if minutes is not None and (time.time() - started_run) / 60.0 >= minutes:
            return False
        return True

    consec_no_improve = 0   # for overnight basin-hopping (escape a plateaued region)
    n_restarts = 0
    if overnight:
        log(f"  OVERNIGHT mode: no plateau-stop; basin-hop after {restart_after} "
            f"non-improvements; global best always preserved.")

    i = 0
    while _budget_left(i):
        elapsed_min = (time.time() - started_run) / 60.0
        try:
            # ---- propose the COMBINED candidate (schedule + kernel_knobs) ----
            if i == 0:
                cfg, knobs, source = incumbent_cfg, incumbent_knobs, start_source
            elif overnight and consec_no_improve >= restart_after:
                # BASIN-HOP: jump to a fresh random region so the night isn't spent grinding a
                # stuck basin. The global best is held by the evaluator's resident incumbent, so a
                # restart can only ever find something better. Reset the plateau accounting.
                cfg, knobs, source = random_config(rng, target), random_knobs(rng), "restart"
                incumbent_cfg, incumbent_knobs = cfg, dict(knobs)
                consec_no_improve = 0
                state["consecutive_reverts"] = 0
                n_restarts += 1
                log(f"  [{i}] === basin-hop restart #{n_restarts} (escaping plateau) ===")
            else:
                cfg, knobs, source = _propose(
                    rng, i, cold=cold, epsilon=epsilon, incumbent_cfg=incumbent_cfg,
                    incumbent_knobs=incumbent_knobs,
                    target=target, model_shape=model_shape, gpu=target.name,
                    corpus_path=corpus_path, ranker=ranker,
                    warm_seed_queue=warm_seed_queue, tried=tried)
            knobs = _norm_knobs(knobs)
            cfg_dict = _candidate_dict(cfg, knobs)   # JSON candidate WITH kernel_knobs embedded
            sid = schedule_id(cfg_dict)
            tried.add(sid)

            # ---- evaluate: MEASURED cuda (build VM w/ knobs + interleaved keep/revert) or the
            #      cost-model fallback (cpu path; harness owns the honesty gate). -----------------
            if use_measured and cuda_eval is not None:
                mv = cuda_eval.evaluate(cfg, knobs, is_incumbent_seed=(i == 0),
                                        margin=measured_margin)
                valid = bool(mv.get("valid"))
                correct = bool(mv.get("correct"))
                lat = mv.get("latency_us")
                kind = mv.get("latency_kind")
                # pct of roofline from the measured latency (honest: only when we have a latency).
                pct = None
                if lat is not None:
                    try:
                        from schedule.lower import lower as _lwr
                        wb = _lwr(graph, target=target, config=cfg, pos=0,
                                  dtype=cuda_eval.dtype).total_weight_bytes()
                        bound = target.bandwidth_bound_us(wb)
                        # SAME convention as eval.roofline.pct_of_bound: measured/bound*100. 100% ==
                        # at the HBM floor; > 100% == above it (slower than the weight-streaming
                        # lower bound). Matches the cost-model path so the orchestrator move-on
                        # criteria mean the same thing on cuda and cpu.
                        pct = round(100.0 * lat / bound, 3) if bound > 0 else None
                    except Exception:
                        pct = None
                # the measured evaluator made its OWN interleaved keep/revert decision; honor it.
                # ``incumbent_us`` is the evaluator's resident-incumbent latency re-measured at the
                # CURRENT clock (apples-to-apples), which is the honest "best so far" to report, the
                # candidate's standalone ``lat`` can read lower/higher purely from WDDM clock drift.
                verdict = {"valid": valid, "correct": correct, "latency_us": lat,
                           "pct_of_roofline": pct, "latency_kind": kind, "bound_us": None,
                           "rejected_reason": mv.get("rejected_reason"),
                           "_measured_improved": bool(mv.get("improved")),
                           "_incumbent_us": mv.get("best_us")}
            else:
                verdict = harness.evaluate(label, target.name, _cfg_to_dict(cfg), device=dev,
                                           _prepared=prepared)
                valid = bool(verdict.get("valid"))
                correct = bool(verdict.get("correct"))
                lat = verdict.get("latency_us")
                pct = verdict.get("pct_of_roofline")
                kind = verdict.get("latency_kind")

            if not valid:
                # clean REJECT: log, continue (a bad config can only ever lose).
                result.n_rejected += 1
                orch.record(state, latency_us=None, status="rejected", config=cfg_dict,
                            correctness="REJECTED", schedule_id=sid,
                            description=f"{source}: {(verdict.get('rejected_reason') or '')[:80]}",
                            results_path=results_path, elapsed_minutes=elapsed_min)
                log(f"  [{i}] {source:9s} REJECTED  {(verdict.get('rejected_reason') or '')[:60]}")
                consec_no_improve += 1
                i += 1
                continue

            if not correct:
                result.n_rejected += 1
                orch.record(state, latency_us=None, status="failed", config=cfg_dict,
                            correctness="FAIL", schedule_id=sid,
                            description=f"{source}: correctness FAIL", results_path=results_path,
                            elapsed_minutes=elapsed_min)
                log(f"  [{i}] {source:9s} FAIL (incorrect), no latency emitted")
                consec_no_improve += 1
                i += 1
                continue

            result.n_correct += 1

            # ---- keep/revert ----
            # MEASURED path: the interleaved evaluator already made the apples-to-apples decision
            # (kept iff faster than the resident incumbent by the noise-beating margin); honor it,
            # and use the evaluator's tracked incumbent latency as best_us.
            # COST-MODEL path: correct AND (first incumbent OR >= min_gain faster than best).
            is_first = best_us is None
            if use_measured and cuda_eval is not None:
                improved = bool(verdict.get("_measured_improved"))
            else:
                improved = (lat is not None and
                            (is_first or lat < best_us * (1.0 - min_gain)))
            if improved and lat is not None:
                # the kept latency we RECORD is the candidate's measured latency (it is now the
                # resident incumbent). For the measured path use the evaluator's incumbent value
                # when available (apples-to-apples vs the clock); else the standalone median.
                kept_lat = float(verdict.get("_incumbent_us") or lat)
                best_us = kept_lat
                best_cfg_dict = cfg_dict
                incumbent_cfg = cfg
                incumbent_knobs = dict(knobs)
                result.n_kept += 1
                if result.start_incumbent_us is None:
                    result.start_incumbent_us = kept_lat
                result.iters_to_best = i
                region = _region_breakdown(label, target.name, cfg_dict, prepared)
                orch.record(state, latency_us=kept_lat, status="kept", config=cfg_dict,
                            correctness="PASS", pct_of_roofline=pct, latency_kind=kind,
                            region_breakdown=region, schedule_id=sid,
                            description=f"{source}: kept", results_path=results_path,
                            elapsed_minutes=elapsed_min)
                # kept correct points enter the flywheel corpus (grows the cross-run moat).
                try:
                    append_corpus(CorpusRecord(
                        model=label, gpu=target.name, regime="single-stream",
                        correctness="PASS", latency_us=kept_lat,
                        bound_us=float(verdict.get("bound_us") or 0.0),
                        pct_of_roofline=float(pct or 0.0),
                        schedule=cfg_dict, ir_version="", abi_version="",
                        notes=f"autoresearch kept; {source}; kind={kind}"),
                        path=corpus_path)
                except Exception as e:
                    log(f"      (corpus write skipped: {type(e).__name__}: {e})")
                _pcts = f"{pct:.0f}% roofline" if pct is not None else "roofline n/a"
                log(f"  [{i}] {source:9s} KEPT  {kept_lat:.3f}us ({kind}, {_pcts})  "
                    f"best={best_us:.3f}us")
            else:
                orch.record(state, latency_us=lat, status="revert", config=cfg_dict,
                            correctness="PASS", pct_of_roofline=pct, latency_kind=kind,
                            schedule_id=sid, description=f"{source}: revert (not faster)",
                            results_path=results_path, elapsed_minutes=elapsed_min)
                _latr = f"{lat:.3f}us" if lat is not None else "n/a"
                _bestr = f"{best_us:.3f}us" if best_us is not None else "n/a"
                log(f"  [{i}] {source:9s} revert {_latr} (best stays {_bestr})")

            # overnight basin-hop accounting: a kept win resets the plateau; otherwise it grows.
            consec_no_improve = 0 if (improved and lat is not None) else consec_no_improve + 1

            result.trajectory.append({
                "iter": i, "source": source, "valid": valid, "correct": correct,
                "latency_us": lat, "kind": kind, "pct_of_roofline": pct,
                "kept": bool(improved), "best_us": best_us, "schedule_id": sid,
            })

        except KeyboardInterrupt:
            log("  interrupted by user; checkpoint is saved, run is resumable.")
            break
        except Exception as e:  # ROBUST: a crash/CUDA-error in ONE iter never stops the run.
            result.n_crash += 1
            kind_tag = "TIMEOUT" if "timeout" in str(e).lower() else "CRASH"
            tb = traceback.format_exc(limit=2)
            try:
                orch.record(state, latency_us=None, status=kind_tag.lower(), config=None,
                            correctness=kind_tag, schedule_id="",
                            description=f"iter {i} {kind_tag}: {type(e).__name__}: {e}"[:160],
                            results_path=results_path, elapsed_minutes=elapsed_min)
            except Exception:
                pass
            log(f"  [{i}] {kind_tag}: {type(e).__name__}: {e}  (continuing)\n{tb}")

        # ---- CHECKPOINT every iteration (resumable) ----
        result.iters_run = i + 1
        result.best_us = best_us
        result.best_config = best_cfg_dict
        result.baseline_us = state.get("baseline_us")
        result.best_pct_roofline = state.get("best_pct_roofline")
        result.best_kind = state.get("best_kind")
        result.speedup_vs_baseline = state.get("speedup")
        i += 1

        # ---- overnight hygiene: trim CUDA cache + write a wake-up report periodically ----
        if overnight and i % 25 == 0:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            _write_overnight_report(result, state, n_restarts, elapsed_min)
            log(f"  [{i}] checkpoint: best={best_us}us, {n_restarts} restarts, "
                f"{elapsed_min:.0f} min elapsed, wake-up report updated.")

        # stop early if the campaign hit a move-on / done criterion this iter, UNLESS overnight,
        # where only the minutes/iters budget ends the run (we basin-hop past plateaus instead).
        if state.get("status") == orch.STATUS_DONE and not overnight:
            log(f"  move-on: {state.get('move_on_reason')}, stopping the run.")
            break

    # ---- final report ----
    result.iters_run = i
    result.best_us = best_us
    result.best_config = best_cfg_dict
    result.baseline_us = state.get("baseline_us")
    result.best_pct_roofline = state.get("best_pct_roofline")
    result.best_kind = state.get("best_kind")
    result.speedup_vs_baseline = state.get("speedup")

    if overnight:
        _write_overnight_report(result, state, n_restarts,
                                (time.time() - started_run) / 60.0)
        log(f"  wake-up report: {os.path.join(orch.WORKSPACE, 'amk_overnight_report.md')}")
    if verbose:
        _print_final(result, state)
    return result.to_dict()


def _write_overnight_report(result: "AutoresearchResult", state: dict[str, Any],
                            n_restarts: int, elapsed_min: float) -> None:
    """Write the 'run it and sleep' wake-up summary (JSON + markdown) to the workspace. Called
    periodically during an overnight run and once at the end, always reflects the latest best."""
    orch._ensure_workspace()
    base = state.get("baseline_us")
    best = result.best_us
    sp = (base / best) if (base and best) else None
    # top kept milestones from the trajectory (the improvement story).
    kept = [t for t in result.trajectory if t.get("kept") and t.get("latency_us")]
    summary = {
        "campaign": f"{result.model} / {result.gpu}",
        "device": result.device,
        "elapsed_minutes": round(elapsed_min, 1),
        "iterations_this_run": result.iters_run,
        "total_experiments": state.get("experiments_run", 0),
        "kept": result.n_kept, "correct": result.n_correct,
        "rejected_or_failed": result.n_rejected, "crash_or_timeout": result.n_crash,
        "basin_hop_restarts": n_restarts,
        "baseline_us": base,
        "best_us": best,
        "best_kind": result.best_kind,
        "best_pct_of_roofline": result.best_pct_roofline,
        "speedup_vs_baseline": round(sp, 3) if sp else None,
        "best_config": result.best_config,
        "improvement_milestones": [
            {"iter": t["iter"], "best_us": t["best_us"], "source": t["source"]} for t in kept
        ],
        "honesty": ("All latencies are CUDA-event medians, correctness-gated vs the CPU ReferenceVM, "
                    "and decided by interleaved (drift-robust) keep/revert; sub-roofline artifacts are "
                    "withheld. The speedup is vs AMK's own default schedule, NOT a claim of beating "
                    "cuBLAS/vLLM (AMK is honestly behind those at batch-1; see HARNESS.md)."),
    }
    jpath = os.path.join(orch.WORKSPACE, "amk_overnight_report.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # markdown wake-up note
    lines = [
        "# AMK overnight run, wake-up report", "",
        f"- Campaign: **{summary['campaign']}** (device={summary['device']})",
        f"- Ran: **{summary['elapsed_minutes']} min**, {summary['iterations_this_run']} iters this run "
        f"({summary['total_experiments']} total in campaign)",
        f"- Kept {summary['kept']} / correct {summary['correct']} / "
        f"rejected+failed {summary['rejected_or_failed']} / crash+timeout {summary['crash_or_timeout']}",
        f"- Basin-hop restarts: {summary['basin_hop_restarts']}",
        "",
        "## Result",
        "",
    ]
    if best and base:
        lines += [
            f"- Baseline (default schedule): **{base:.1f} us**",
            f"- Best found: **{best:.1f} us** ({result.best_kind}), **{sp:.3f}x** faster than baseline"
            + (f", {result.best_pct_roofline:.0f}% of the HBM floor" if result.best_pct_roofline else ""),
        ]
    elif best:
        lines.append(f"- Best found: **{best:.1f} us** ({result.best_kind})")
    else:
        lines.append("- No correct schedule found yet.")
    lines += ["", "## Best config", "", "```json", json.dumps(result.best_config, indent=2), "```",
              "", "## Honesty", "", summary["honesty"], ""]
    with open(os.path.join(orch.WORKSPACE, "amk_overnight_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _print_final(result: AutoresearchResult, state: dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print("  AMK autoresearch, final report")
    print("=" * 60)
    print(f"  campaign      : {result.model} / {result.gpu}  "
          f"({'cold' if result.cold else 'warm'}, device={result.device})")
    print(f"  iterations    : {result.iters_run} this run "
          f"({state.get('experiments_run', 0)} total in campaign)")
    print(f"  correct/kept  : {result.n_correct} correct, {result.n_kept} kept, "
          f"{result.n_rejected} rejected/failed, {result.n_crash} crash/timeout")
    if result.warm_seeds or result.ranker_trained:
        print(f"  flywheel      : {result.warm_seeds} warm seed(s), "
              f"ranker {'trained' if result.ranker_trained else 'cold'}")
    if result.start_incumbent_us is not None:
        print(f"  start incumbent: {result.start_incumbent_us:.3f}us "
              f"({'warm seed' if result.warm_seeds else 'default'})")
    if result.best_us is not None:
        line = f"  BEST          : {result.best_us:.3f}us"
        if result.best_kind:
            line += f" ({result.best_kind})"
        if result.best_pct_roofline is not None:
            line += f", {result.best_pct_roofline:.0f}% of roofline"
        if result.iters_to_best is not None:
            line += f", reached at iter {result.iters_to_best}"
        print(line)
        if result.baseline_us:
            sp = result.baseline_us / result.best_us if result.best_us else float("nan")
            print(f"  vs baseline   : {result.baseline_us:.3f}us default "
                  f"-> {sp:.3f}x faster")
    else:
        print("  BEST          : (no correct schedule found)")
    if state.get("move_on_reason"):
        print(f"  stopped       : {state['move_on_reason']}")
    print(f"  state         : {result.state_path}")
    print()


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="amk autoresearch")
    ap.add_argument("model")
    ap.add_argument("--gpu", default="rtx5090")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--cold", action="store_true",
                    help="ignore the flywheel prior (pure exploration; grows the corpus)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--state", default=None)
    ap.add_argument("--overnight", action="store_true",
                    help="long-run mode: never stop on a plateau; basin-hop to fresh regions and "
                         "keep the global best; write a wake-up report. Use with a long --minutes.")
    ap.add_argument("--restart-after", type=int, default=6,
                    help="overnight: basin-hop after this many consecutive non-improvements")
    args = ap.parse_args(argv)
    autoresearch(args.model, args.gpu, iters=args.iters, minutes=args.minutes,
                 device=args.device, cold=args.cold, seed=args.seed, epsilon=args.epsilon,
                 corpus_path=args.corpus, results_path=args.results, state_path=args.state,
                 overnight=args.overnight, restart_after=args.restart_after,
                 verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
