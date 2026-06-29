"""
AMK, LOOP-2 AUTORESEARCH SCHEDULE SEARCH (the research core)
============================================================

The cost-model-guided *explore* + on-hardware *exploit* loop that turns a model graph into the
best :class:`~schedule.ir.ScheduleConfig` for a given :class:`~schedule.ir.GpuTarget`. This is
the AMK analogue of Ansor's search × AutoKernel's keep/revert loop, operating purely on the
**locked** edit surface (:class:`ScheduleConfig`): tiling, fusion grouping, SM assignment policy,
software-pipelining depth, page-allocation policy, and launch config. The search never writes
kernel code, it only *picks a point in this space*, hands it to a frozen ``lower_fn`` that
deterministically realizes a runnable, **validated** megakernel, and scores it.

THE LOOP (per the spec)
-----------------------
1. **Explore (cost model).** Propose diverse candidate configs (seeded from a default, then
   mutated/evolved around the running best). For each candidate:
       lower(config) -> validate() -> cost_model.predict_us
   An invalid lowering is *rejected* (logged, never crashes the loop, never kept). This is the
   agent-safety mechanism: a bad config can only ever lose, never corrupt the search.
2. **Exploit (optional on-hardware).** If a ``measure_fn`` is supplied, the top candidates are
   measured on the real GPU and ranked by *measured* latency; otherwise the cost model's
   prediction is the fitness. Either way we **keep/revert**: a trial replaces the incumbent only
   if it is strictly better, and we evolve around whichever config is currently best (move-on
   discipline, when a region stops improving, mutation naturally explores elsewhere).
3. **Log everything.** Every trial, kept or rejected, is recorded as a results.tsv-style row
   (returned to the caller and writable via :func:`write_results_tsv`), the flywheel substrate.

CLEAN API
---------
``search(graph, target, budget, lower_fn, measure_fn=None, ...) -> SearchResult`` is the single
entry point a coding agent or ``compile.py`` drives. ``lower_fn(graph, config, target) ->
MegakernelProgram`` is the only coupling to the (separately-owned) lowerer; if it is mid-build,
callers pass a toy ``lower_fn`` built from the ``vm/verify_vm.py`` patterns (see tests).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from schedule.cost_model import CostModel
from schedule.ir import (
    GpuTarget,
    MegakernelProgram,
    ScheduleConfig,
    ValidationResult,
    replace,
    validate,
)

# A lower_fn realizes a config into a runnable program. Signature is (graph, config, target).
LowerFn = Callable[[Any, ScheduleConfig, GpuTarget], MegakernelProgram]
# A measure_fn returns *measured* single-stream decode latency in microseconds for a validated
# program (the on-hardware exploit). It may raise / return None if measurement fails; the loop
# then falls back to the predicted latency for that trial.
MeasureFn = Callable[[MegakernelProgram, GpuTarget], float | None]


# ----------------------------------------------------------------------------------------
# Search-space definition (the knobs we vary on ScheduleConfig)
# ----------------------------------------------------------------------------------------
# Candidate values per knob. Deliberately small + interpretable so the search is reproducible.
# Each dimension maps onto a ScheduleConfig field. NOTE: sm_assignment, page_allocation, and
# fusion_grouping are RESERVED, they are recorded on the schedule and searchable, but NOT yet
# consumed by the frozen lowerer/loader (no effect on the emitted program today); see
# harness.search_space and HARNESS.md. They are kept in the space for backward-compatible configs.
N_TILE_CHOICES = (64, 128, 256, 512)
PIPELINING_CHOICES = (0, 1, 2, 3, 4)
SM_POLICY_CHOICES = ("round_robin", "load_balance")     # RESERVED knob (not yet consumed)
PAGE_POLICY_CHOICES = ("linear", "graph_color", "none")  # RESERVED knob (not yet consumed)
THREADS_PER_BLOCK_CHOICES = (128, 256)  # 512 deadlocks the cooperative grid-sync kernel (measured);
# restore only after the barrier is fixed + re-validated for higher occupancy. A search must never
# propose a config that can hang.
KV_BLOCK_CHOICES = (64, 128, 256)
# RESERVED knob (recorded + searchable, not yet consumed by the frozen lowerer):
FUSION_CHOICES: tuple[tuple[list[str], ...], ...] = (
    (),                                   # no fusion
    (["gate", "up"],),                    # fuse the two MLP up-projections
    (["gate", "up", "silu"],),            # fuse up-projections with the SwiGLU activation
    (["rmsnorm", "gemv"],),               # fuse norm into the following projection
)


def default_config(target: GpuTarget | None = None) -> ScheduleConfig:
    """The neutral starting point: the config a non-searching compiler would emit. Search must
    never return something predicted *worse* than this (the acceptance contract)."""
    return ScheduleConfig(
        tiling={"gemv": {"N_tile": 64}, "attention": {"kv_block": 128}},
        fusion_grouping=[],
        sm_assignment="load_balance",
        pipelining_depth=2,
        page_allocation="graph_color",
        threads_per_block=256,
        smem_bytes_per_block=0,
    )


def random_config(rng: random.Random, target: GpuTarget | None = None) -> ScheduleConfig:
    """Sample a fresh, independent point in the search space (the *explore* diversity source)."""
    tpb = rng.choice(THREADS_PER_BLOCK_CHOICES)
    smem_cap = target.smem_bytes_per_block_optin if target else 0
    return ScheduleConfig(
        tiling={
            "gemv": {"N_tile": rng.choice(N_TILE_CHOICES)},
            "attention": {"kv_block": rng.choice(KV_BLOCK_CHOICES)},
        },
        fusion_grouping=[list(g) for g in rng.choice(FUSION_CHOICES)],
        sm_assignment=rng.choice(SM_POLICY_CHOICES),
        pipelining_depth=rng.choice(PIPELINING_CHOICES),
        page_allocation=rng.choice(PAGE_POLICY_CHOICES),
        threads_per_block=tpb,
        # opt into a little SMEM sometimes, always within the target's cap.
        smem_bytes_per_block=rng.choice((0, min(16384, smem_cap), min(49152, smem_cap)))
        if smem_cap else 0,
    )


def mutate_config(cfg: ScheduleConfig, rng: random.Random,
                  target: GpuTarget | None = None) -> ScheduleConfig:
    """Evolve around an incumbent: change ONE knob (local move). Small steps keep the cost-model
    landscape smooth so keep/revert hill-climbs reliably; occasional knob swaps escape plateaus."""
    knob = rng.choice((
        "n_tile", "pipelining", "sm", "page", "fusion", "threads", "kv_block", "smem",
    ))
    tiling = {k: dict(v) for k, v in cfg.tiling.items()}
    if knob == "n_tile":
        tiling.setdefault("gemv", {})["N_tile"] = rng.choice(N_TILE_CHOICES)
        return replace(cfg, tiling=tiling)
    if knob == "kv_block":
        tiling.setdefault("attention", {})["kv_block"] = rng.choice(KV_BLOCK_CHOICES)
        return replace(cfg, tiling=tiling)
    if knob == "pipelining":
        return replace(cfg, pipelining_depth=rng.choice(PIPELINING_CHOICES))
    if knob == "sm":
        return replace(cfg, sm_assignment=rng.choice(SM_POLICY_CHOICES))
    if knob == "page":
        return replace(cfg, page_allocation=rng.choice(PAGE_POLICY_CHOICES))
    if knob == "fusion":
        return replace(cfg, fusion_grouping=[list(g) for g in rng.choice(FUSION_CHOICES)])
    if knob == "threads":
        return replace(cfg, threads_per_block=rng.choice(THREADS_PER_BLOCK_CHOICES))
    # smem
    cap = target.smem_bytes_per_block_optin if target else 65536
    return replace(cfg, smem_bytes_per_block=rng.choice((0, min(16384, cap), min(49152, cap))))


# ----------------------------------------------------------------------------------------
# Trial / result records
# ----------------------------------------------------------------------------------------
@dataclass
class Trial:
    """One evaluated candidate, the unit logged to the flywheel (results.tsv row)."""

    index: int                      # trial number in the search
    source: str                     # "default" | "random" | "mutate", how it was proposed
    config: ScheduleConfig
    valid: bool
    predicted_us: float | None      # cost-model prediction (None if lowering failed)
    measured_us: float | None       # on-hardware latency (None if no measure_fn / measure failed)
    score_us: float | None          # the fitness used for keep/revert (measured if available)
    kept: bool                      # did this trial become / stay the incumbent best?
    reject_reason: str = ""         # why invalid/failed (empty if valid)
    bandwidth_bound_us: float | None = None
    distance_to_bw: float | None = None
    region_us: dict[str, float] = field(default_factory=dict)
    wall_s: float = 0.0
    # Lowered program for the measure() helper.  Set to a valid MegakernelProgram when the trial
    # passed validate(); None when lowering failed or the trial was invalid.  repr=False keeps
    # Trial's __repr__ readable (programs are large objects).
    program: MegakernelProgram | None = field(default=None, repr=False)

    def to_row(self) -> dict[str, Any]:
        """Flat dict for a results.tsv-style row."""
        cfg = self.config
        sm = cfg.sm_assignment if isinstance(cfg.sm_assignment, str) else "explicit"
        return {
            "trial": self.index,
            "source": self.source,
            "valid": int(self.valid),
            "kept": int(self.kept),
            "predicted_us": _r(self.predicted_us),
            "measured_us": _r(self.measured_us),
            "score_us": _r(self.score_us),
            "bandwidth_bound_us": _r(self.bandwidth_bound_us),
            "distance_to_bw": _r(self.distance_to_bw),
            "N_tile": cfg.tiling.get("gemv", {}).get("N_tile", ""),
            "kv_block": cfg.tiling.get("attention", {}).get("kv_block", ""),
            "pipelining_depth": cfg.pipelining_depth,
            "sm_assignment": sm,
            "page_allocation": cfg.page_allocation,
            "threads_per_block": cfg.threads_per_block,
            "smem_bytes_per_block": cfg.smem_bytes_per_block,
            "fusion": ";".join("+".join(g) for g in cfg.fusion_grouping),
            "reject_reason": self.reject_reason,
            "wall_s": round(self.wall_s, 5),
        }


@dataclass
class SearchResult:
    """The outcome of a search run: the best validated config + program, plus every trial."""

    best_config: ScheduleConfig | None
    best_program: MegakernelProgram | None
    best_score_us: float | None
    default_score_us: float | None        # the baseline the best must beat-or-tie
    trials: list[Trial]
    target: GpuTarget

    @property
    def n_valid(self) -> int:
        return sum(1 for t in self.trials if t.valid)

    @property
    def improvement(self) -> float:
        """Speedup of best vs default (>= 1.0 means we improved or tied)."""
        if not self.best_score_us or not self.default_score_us or self.best_score_us <= 0:
            return float("nan")
        return self.default_score_us / self.best_score_us

    def rows(self) -> list[dict[str, Any]]:
        return [t.to_row() for t in self.trials]

    def summary(self) -> str:
        best = f"{self.best_score_us:.2f}us" if self.best_score_us else "n/a"
        base = f"{self.default_score_us:.2f}us" if self.default_score_us else "n/a"
        return (f"SearchResult(best={best} vs default={base}, "
                f"x{self.improvement:.3f} faster, {self.n_valid}/{len(self.trials)} valid)")


def _r(x: float | None) -> Any:
    return round(x, 3) if isinstance(x, (int, float)) else ""


# ----------------------------------------------------------------------------------------
# The search loop
# ----------------------------------------------------------------------------------------
def search(graph: Any,
           target: GpuTarget,
           budget: int,
           lower_fn: LowerFn,
           measure_fn: MeasureFn | None = None,
           *,
           cost_model: CostModel | None = None,
           seed: int = 0,
           explore_fraction: float = 0.35,
           measure_top_k: int = 0,
           start_config: ScheduleConfig | None = None,
           on_trial: Callable[[Trial], None] | None = None) -> SearchResult:
    """Run the Loop-2 autoresearch search and return the best **validated** schedule.

    Parameters
    ----------
    graph        : opaque model-graph handle passed straight through to ``lower_fn``.
    target       : the :class:`GpuTarget` to optimize for (drives the cost model + launch checks).
    budget       : number of candidate configs to evaluate (>= 1). The acceptance test uses >=20.
    lower_fn     : ``(graph, config, target) -> MegakernelProgram``. The frozen, deterministic
                   realization of a config. May raise; a raise is caught and logged as a rejection.
    measure_fn   : optional ``(program, target) -> us`` on-hardware exploit. When given, the top
                   ``measure_top_k`` predicted candidates are measured and ranked by measured time;
                   when ``None`` the cost model is the sole fitness (still correct, just predicted).
    cost_model   : override the default :class:`CostModel` (e.g. a re-tuned one).
    seed         : RNG seed for reproducible proposals.
    explore_fraction : share of the budget spent on fresh random configs (the rest evolves the
                   incumbent). Balances exploration vs exploitation.
    measure_top_k: how many of the best-predicted *valid* configs to additionally measure on
                   hardware at the end (only used when ``measure_fn`` is provided). 0 => measure
                   each kept incumbent inline instead.
    start_config : seed config to begin from (defaults to :func:`default_config`).
    on_trial     : optional callback invoked with each :class:`Trial` (live logging / streaming).

    Returns
    -------
    :class:`SearchResult` whose ``best_config`` is guaranteed to lower to a ``validate().ok``
    program (or ``None`` if *every* candidate was invalid), and whose ``best_score_us`` is
    ``<=`` the default config's score (keep/revert guarantees we never regress past the baseline).
    """
    if budget < 1:
        raise ValueError("search budget must be >= 1")
    cm = cost_model or CostModel()
    rng = random.Random(seed)
    trials: list[Trial] = []

    # ---- helper: lower + validate + cost a single config -----------------------------
    def evaluate(cfg: ScheduleConfig, index: int, source: str) -> Trial:
        t0 = time.time()
        prog: MegakernelProgram | None = None
        valid = False
        reason = ""
        predicted = None
        bw = None
        dist = None
        region_us: dict[str, float] = {}
        try:
            prog = lower_fn(graph, cfg, target)
        except Exception as e:  # a buggy/over-aggressive config must never crash the search
            reason = f"lower_fn raised: {type(e).__name__}: {e}"
            prog = None
        if prog is not None:
            res: ValidationResult = validate(prog)
            if res.ok:
                valid = True
                try:
                    bd = cm.estimate(prog, target)
                    predicted = bd.predicted_us
                    bw = bd.bandwidth_bound_us
                    dist = bd.distance_to_bandwidth_bound
                    region_us = dict(bd.region_us)
                except Exception as e:
                    valid = False
                    reason = f"cost_model raised: {type(e).__name__}: {e}"
            else:
                reason = "; ".join(res.errors[:3]) or "validate() rejected"
        tr = Trial(
            index=index, source=source, config=cfg, valid=valid,
            predicted_us=predicted, measured_us=None,
            score_us=predicted, kept=False, reject_reason=reason,
            bandwidth_bound_us=bw, distance_to_bw=dist, region_us=region_us,
            wall_s=time.time() - t0,
            program=prog if valid else None,
        )
        return tr

    # ---- helper: measure a validated trial on hardware (exploit) ---------------------
    def measure(tr: Trial) -> None:
        if measure_fn is None or not tr.valid:
            return
        prog = tr.program
        if prog is None:
            return
        try:
            m = measure_fn(prog, target)
        except Exception as e:
            tr.reject_reason = (tr.reject_reason + " | " if tr.reject_reason else "") \
                + f"measure_fn raised: {type(e).__name__}: {e}"
            m = None
        if m is not None and m > 0:
            tr.measured_us = float(m)
            tr.score_us = float(m)  # measured latency wins as the fitness when available

    # ---- 0) baseline: the default config (the bar best must clear) -------------------
    base_cfg = start_config or default_config(target)
    base_trial = evaluate(base_cfg, 0, "default")
    measure(base_trial)
    base_trial.kept = base_trial.valid
    trials.append(base_trial)
    if on_trial:
        on_trial(base_trial)

    default_score = base_trial.score_us if base_trial.valid else None
    best_trial: Trial | None = base_trial if base_trial.valid else None
    incumbent_cfg = base_cfg  # what we evolve around (best valid config seen)

    # ---- 1) explore + exploit for the remaining budget -------------------------------
    n_explore = max(0, int(round((budget - 1) * explore_fraction)))
    for i in range(1, budget):
        # Decide proposal source: a fixed slice of fresh random configs, the rest evolve incumbent.
        if i <= n_explore or best_trial is None:
            cfg = random_config(rng, target)
            source = "random"
        else:
            cfg = mutate_config(incumbent_cfg, rng, target)
            source = "mutate"
        tr = evaluate(cfg, i, source)

        # Inline on-hardware measurement of promising candidates: only measure if it could plausibly
        # become the incumbent (predicted at-or-below current best) to spend GPU time wisely.
        if measure_fn is not None and measure_top_k == 0 and tr.valid:
            if best_trial is None or (tr.predicted_us is not None and best_trial.score_us is not None
                                      and tr.predicted_us <= best_trial.score_us * 1.05):
                measure(tr)

        # keep/revert: replace the incumbent only on a strict improvement.
        if tr.valid and tr.score_us is not None:
            if best_trial is None or best_trial.score_us is None or tr.score_us < best_trial.score_us:
                tr.kept = True
                best_trial = tr
                incumbent_cfg = tr.config  # evolve around the new best (move-on discipline)
        trials.append(tr)
        if on_trial:
            on_trial(tr)

    # ---- 2) optional batched exploit pass: measure the top-K predicted valid configs -
    if measure_fn is not None and measure_top_k > 0:
        valid_sorted = sorted(
            (t for t in trials if t.valid and t.predicted_us is not None),
            key=lambda t: t.predicted_us)  # type: ignore[arg-type]
        for tr in valid_sorted[:measure_top_k]:
            if tr.measured_us is None:
                measure(tr)
        # Re-pick the best by the (now possibly measured) score.
        for tr in trials:
            if tr.valid and tr.score_us is not None:
                if best_trial is None or best_trial.score_us is None \
                        or tr.score_us < best_trial.score_us:
                    if best_trial is not None:
                        best_trial.kept = False
                    tr.kept = True
                    best_trial = tr

    # ---- 3) finalize -----------------------------------------------------------------
    # Guarantee the returned best is <= default (keep/revert): if somehow the chosen best scores
    # worse than the validated default, revert to the default (the baseline can never lose).
    if best_trial is not None and base_trial.valid and best_trial.score_us is not None \
            and default_score is not None and best_trial.score_us > default_score:
        for t in trials:
            t.kept = False
        base_trial.kept = True
        best_trial = base_trial

    best_program = best_trial.program if best_trial else None
    return SearchResult(
        best_config=best_trial.config if best_trial else None,
        best_program=best_program,
        best_score_us=best_trial.score_us if best_trial else None,
        default_score_us=default_score,
        trials=trials,
        target=target,
    )


# ----------------------------------------------------------------------------------------
# results.tsv-style logging (the flywheel substrate)
# ----------------------------------------------------------------------------------------
_TSV_COLUMNS = [
    "trial", "source", "valid", "kept", "predicted_us", "measured_us", "score_us",
    "bandwidth_bound_us", "distance_to_bw", "N_tile", "kv_block", "pipelining_depth",
    "sm_assignment", "page_allocation", "threads_per_block", "smem_bytes_per_block",
    "fusion", "reject_reason", "wall_s",
]


def results_tsv(result: SearchResult) -> str:
    """Render every trial as a results.tsv string (header + one row per trial)."""
    lines = ["\t".join(_TSV_COLUMNS)]
    for row in result.rows():
        lines.append("\t".join(str(row.get(c, "")) for c in _TSV_COLUMNS))
    return "\n".join(lines) + "\n"


def write_results_tsv(result: SearchResult, path: str) -> None:
    """Persist the trial log to ``path`` as a results.tsv (the flywheel row store)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(results_tsv(result))


__all__ = [
    "search", "SearchResult", "Trial", "LowerFn", "MeasureFn",
    "default_config", "random_config", "mutate_config",
    "results_tsv", "write_results_tsv",
    "N_TILE_CHOICES", "PIPELINING_CHOICES", "SM_POLICY_CHOICES", "PAGE_POLICY_CHOICES",
]
