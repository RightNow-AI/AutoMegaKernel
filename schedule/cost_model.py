"""
AMK, ANALYTIC ROOFLINE COST MODEL (Layer 2)
============================================

A GPU-free, *analytic* cost model that predicts the single-stream decode latency of a
:class:`~schedule.ir.MegakernelProgram` in microseconds, without compiling or launching a
kernel. This is the cheap fitness function the Loop-2 search (``schedule/search.py``) uses to
rank thousands of candidate :class:`~schedule.ir.ScheduleConfig`\\s before spending real GPU
time on the few best ones (the on-hardware *exploit* step).

WHY ANALYTIC, NOT LEARNED
-------------------------
Decode is overwhelmingly **memory bound**: at batch=1 each weight is read once and almost no
arithmetic is reused, so the honest floor is ``weight_bytes / hbm_bandwidth`` (the roofline
"bandwidth bound"; see :meth:`GpuTarget.bandwidth_bound_us`). A transparent roofline model is
both accurate enough to *rank* schedules and fully explainable, every microsecond is traceable
to bytes, flops, or scheduler overhead. No training data, no black box, no GPU required.

THE MODEL (three terms, composed over the DAG)
----------------------------------------------
1. **Per-task time.** Each task is the max of its two roofline legs plus a fixed scheduler
   transition cost::

       t_task = max(est_bytes / hbm_bw, est_flops / fp16_flops) + TRANSITION_US

   ``TRANSITION_US`` (~1-2us, per the spec's "task->task transition <= ~1-2us" target) models
   the per-SM dequeue + counter wait/signal bubble that the megakernel exists to amortize.
   Software pipelining (``pipelining_depth``) hides part of the *memory* leg of the next task
   behind the current one, so we discount the bytes leg by a pipelining efficiency factor.

2. **Makespan = critical path under SM parallelism.** Tasks assigned to the *same* SM
   serialize (one persistent worker drains a serial queue); tasks on *different* SMs overlap.
   We therefore compute the program makespan as the longest weighted path through a graph that
   honors BOTH the counter dependency edges AND the per-SM serial-queue order. When SMs are not
   yet assigned, we fall back to the pure dependency critical path scaled by an achievable
   parallelism estimate (work / min(num_sms, width)).

3. **Per-region breakdown (Amdahl targeting).** Every task is bucketed into a region
   (``attention`` / ``mlp`` / ``lm_head`` / ``other``) from its op and label so search can spend
   its budget where it actually moves end-to-end latency, and so we can report distance to the
   weights/bandwidth roofline.

The output is a :class:`CostBreakdown` (a finite, positive ``predicted_us`` plus the regional
and roofline detail). ``predict_us(program, target)`` is the one-call convenience used by search.

NOTE ON UNITS: bandwidth is GB/s (1e9 bytes/s), tflops is 1e12 flop/s; we convert both to
per-microsecond rates once and work in microseconds throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from schedule.ir import (
    BufferKind,
    GpuTarget,
    InstructionKind,
    MegakernelProgram,
)

# ----------------------------------------------------------------------------------------
# Tunables (documented constants, NOT magic numbers).
# ----------------------------------------------------------------------------------------
# Fixed per-task scheduler transition: dequeue the next instruction, satisfy its waits, run the
# release fence, signal the out_counter. The spec targets task->task transition <= ~1-2us; we use
# the midpoint as the analytic default. This is what the megakernel-within-a-step design exists to
# amortize away (vs ~5-10us per *kernel launch* in a launch-per-op baseline).
DEFAULT_TRANSITION_US: float = 1.5

# Extra FIXED overhead charged to each GEMV_TILE on top of the generic transition: a GEMV tile, in
# addition to dequeue/wait/signal, re-caches the full activation row x into SMEM behind 2
# __syncthreads and does a warp-shuffle reduction (vm/ops.cuh). That per-tile fixed cost is paid
# once per tile regardless of tile width, so it favors FEWER, wider tiles, the counter-pressure to
# the parallel-floor term (which favors MORE, finer tiles for SM load-balance). The two together
# give the model a non-degenerate interior optimum in N_tile instead of pushing to 1-column tiles.
# Calibrated against eval/bench_fat_tile_gemv.py (RTX 5090): the MEASURED curve is FLAT-and-good in
# the thin band (N_tile 16..32 ~= 61% of roofline) and falls off sharply toward fat tiles (N_tile
# 128..256 ~= 50%..30%). This term reproduces that shape, it makes the analytic optimum a thin
# tile (matching the measured win that thinner == more parallel == faster) while charging the fat
# end the per-tile overhead it deserves, so Loop-2 search is steered AWAY from the old fat default.
# (The auto-sizer floors the realized width at _MIN_AUTO_TILE=16 so search/default never go so thin
# that the measured curve turns back up.)
GEMV_TILE_FIXED_US: float = 0.05

# Software pipelining hides the *memory* leg of a task behind the previous task's compute/transit.
# pipelining_depth=0 hides nothing; deeper prefetch asymptotically hides most of the inter-op HBM
# bubble (the megakernel's single biggest win). We model the *fraction of the bytes leg that is
# still exposed* as 1/(1+depth*PIPELINE_GAIN), clamped so we never claim free bandwidth.
PIPELINE_GAIN: float = 0.6
MIN_EXPOSED_BYTES_FRACTION: float = 0.5  # even infinite depth cannot hide >50% of HBM traffic

# An achievable-occupancy haircut: a real persistent kernel never reaches 100% of peak HBM/FLOP
# (tail effects, partial waves, imperfect coalescing). These scale the *peak* numbers in the
# target down to a sustainable rate used by the roofline. Conservative, target-agnostic.
ACHIEVABLE_BW_FRACTION: float = 0.80
ACHIEVABLE_FLOP_FRACTION: float = 0.70

# Region tags used for the Amdahl-style breakdown.
REGION_ATTENTION = "attention"
REGION_MLP = "mlp"
REGION_LM_HEAD = "lm_head"
REGION_OTHER = "other"
REGIONS = (REGION_ATTENTION, REGION_MLP, REGION_LM_HEAD, REGION_OTHER)


# ----------------------------------------------------------------------------------------
# Result records
# ----------------------------------------------------------------------------------------
@dataclass
class CostBreakdown:
    """The full analytic cost report for one program on one target.

    ``predicted_us`` is the headline number search optimizes. The rest explains it: where the
    time goes by region (for Amdahl targeting) and how close we are to the unavoidable
    weights/bandwidth floor (for honest roofline reporting)."""

    predicted_us: float                       # the headline: estimated single-stream decode latency
    makespan_us: float                        # critical-path latency (== predicted_us; kept explicit)
    total_work_us: float                      # serial sum of all task times (no parallelism)
    bandwidth_bound_us: float                 # weights / hbm_bw, the honest floor
    region_us: dict[str, float] = field(default_factory=dict)        # region -> critical-path-share us
    region_work_us: dict[str, float] = field(default_factory=dict)   # region -> serial work us
    n_tasks: int = 0
    n_sms_used: int = 0
    transition_us_total: float = 0.0          # scheduler overhead component (all tasks)
    bytes_bound_us: float = 0.0               # total HBM-traffic leg (exposed, after pipelining)
    flops_bound_us: float = 0.0               # total compute leg
    notes: list[str] = field(default_factory=list)

    @property
    def distance_to_bandwidth_bound(self) -> float:
        """How far above the unavoidable weights/bandwidth floor we are (ratio >= 1.0 ideally).
        1.0 == we are streaming weights at full achievable bandwidth with zero overhead exposed.
        Larger == overhead/compute/serialization is costing us. Useful for the roofline report."""
        if self.bandwidth_bound_us <= 0:
            return float("nan")
        return self.predicted_us / self.bandwidth_bound_us

    def region_fractions(self) -> dict[str, float]:
        """Fraction of the *critical path* each region accounts for (Amdahl targeting)."""
        tot = sum(self.region_us.values()) or 1.0
        return {r: self.region_us.get(r, 0.0) / tot for r in REGIONS}

    def to_row(self) -> dict[str, Any]:
        """Flat, TSV/flywheel-friendly dict of the salient numbers."""
        return {
            "predicted_us": round(self.predicted_us, 3),
            "makespan_us": round(self.makespan_us, 3),
            "total_work_us": round(self.total_work_us, 3),
            "bandwidth_bound_us": round(self.bandwidth_bound_us, 3),
            "distance_to_bw": round(self.distance_to_bandwidth_bound, 3),
            "n_tasks": self.n_tasks,
            "n_sms_used": self.n_sms_used,
            "transition_us_total": round(self.transition_us_total, 3),
            "bytes_bound_us": round(self.bytes_bound_us, 3),
            "flops_bound_us": round(self.flops_bound_us, 3),
            **{f"region_{r}_us": round(self.region_us.get(r, 0.0), 3) for r in REGIONS},
        }

    def summary(self) -> str:
        rf = self.region_fractions()
        regs = ", ".join(f"{r}={rf[r] * 100:.0f}%" for r in REGIONS if rf[r] > 0.005)
        return (f"predicted={self.predicted_us:.2f}us  floor={self.bandwidth_bound_us:.2f}us  "
                f"x{self.distance_to_bandwidth_bound:.2f}  sms={self.n_sms_used}  "
                f"tasks={self.n_tasks}  [{regs}]")


# ----------------------------------------------------------------------------------------
# Region classification
# ----------------------------------------------------------------------------------------
# Ops that, regardless of label, unambiguously belong to attention.
_ATTENTION_OPS = frozenset({
    InstructionKind.ATTENTION_TILE, InstructionKind.ATTENTION_COMBINE,
    InstructionKind.ROPE, InstructionKind.KV_APPEND, InstructionKind.SOFTMAX,
})
# Ops that signal the final classifier head.
_LM_HEAD_OPS = frozenset({InstructionKind.SAMPLE_ARGMAX})


def classify_region(task) -> str:
    """Bucket a task into a region for the Amdahl breakdown.

    Priority: explicit label keywords (the lowerer tags tasks like ``L0.attn.q[t0]`` /
    ``L0.mlp.down[t3]`` / ``lm_head[t1]``) win, because a GEMV tile is used in attention, MLP,
    AND the head, only the label disambiguates. We fall back to op archetype when unlabeled."""
    label = (task.label or "").lower()
    # Label-driven (the lowerer's region tags). Order matters: lm_head before generic gemv.
    if "lm_head" in label or "lmhead" in label or "logits" in label or "classifier" in label:
        return REGION_LM_HEAD
    if "attn" in label or "attention" in label or "rope" in label or "kv" in label \
            or label.startswith("q") or ".q" in label or ".k" in label or ".v" in label \
            or ".o[" in label or "o_proj" in label:
        return REGION_ATTENTION
    if "mlp" in label or "ffn" in label or "gate" in label or "up" in label \
            or "down" in label or "swiglu" in label or "silu" in label:
        return REGION_MLP
    # Op-driven fallback.
    if task.op in _LM_HEAD_OPS:
        return REGION_LM_HEAD
    if task.op in _ATTENTION_OPS:
        return REGION_ATTENTION
    if task.op in (InstructionKind.SILU_MUL, InstructionKind.GELU):
        return REGION_MLP
    return REGION_OTHER


# ----------------------------------------------------------------------------------------
# Per-task byte/flop estimation
# ----------------------------------------------------------------------------------------
def _estimate_task_bytes_flops(prog: MegakernelProgram, task) -> tuple[int, int]:
    """Return (bytes, flops) for a task, trusting the lowerer's ``est_bytes``/``est_flops`` when
    present and otherwise deriving them from buffer shapes + params.

    The HBM byte count that matters for the roofline is the *weight/KV traffic* (decode reads
    each weight once; activations are tiny and largely on-chip). We therefore derive bytes from
    the WEIGHT/KV_CACHE inputs of the task when no estimate was supplied."""
    if task.est_bytes or task.est_flops:
        return int(task.est_bytes), int(task.est_flops)

    # Fallback derivation from the IR structure.
    weight_bytes = 0
    for bid in task.inputs:
        if 0 <= bid < len(prog.buffers):
            b = prog.buffers[bid]
            if b.kind in (BufferKind.WEIGHT, BufferKind.KV_CACHE):
                weight_bytes += b.nbytes

    flops = 0
    p = task.params
    if task.op in (InstructionKind.GEMV_TILE, InstructionKind.GEMM_TILE):
        K = int(p.get("K", 0))
        n_tile = int(p.get("N_tile", 0))
        m_tile = int(p.get("M_tile", 1))
        flops = 2 * m_tile * K * n_tile  # multiply-add per output element
        if weight_bytes == 0 and K and n_tile:
            weight_bytes = K * n_tile * 2  # assume fp16 weight tile if untagged
    elif task.op == InstructionKind.ATTENTION_TILE:
        head_dim = int(p.get("head_dim", 0))
        kv_len = int(p.get("kv_len", 0))
        n_heads = int(p.get("n_heads", 1))
        # QK^T + softmax*V, two matmuls of [n_heads, kv_len, head_dim]
        flops = 4 * n_heads * kv_len * head_dim
    # Norms / elementwise are negligible flops; their cost is the (tiny) transition + bytes.
    return int(weight_bytes), int(flops)


# ----------------------------------------------------------------------------------------
# The cost model
# ----------------------------------------------------------------------------------------
class CostModel:
    """Analytic roofline cost model. Stateless apart from its tunables, so one instance can score
    an entire search. Construct once, call :meth:`estimate` per candidate program."""

    def __init__(self, transition_us: float = DEFAULT_TRANSITION_US,
                 pipeline_gain: float = PIPELINE_GAIN,
                 achievable_bw_fraction: float = ACHIEVABLE_BW_FRACTION,
                 achievable_flop_fraction: float = ACHIEVABLE_FLOP_FRACTION,
                 gemv_tile_fixed_us: float = GEMV_TILE_FIXED_US):
        self.transition_us = float(transition_us)
        self.pipeline_gain = float(pipeline_gain)
        self.achievable_bw_fraction = float(achievable_bw_fraction)
        self.achievable_flop_fraction = float(achievable_flop_fraction)
        self.gemv_tile_fixed_us = float(gemv_tile_fixed_us)

    def _fixed_overhead_us(self, task) -> float:
        """Per-task fixed overhead: the generic scheduler transition plus, for a GEMV tile, the
        x-cache + 2 __syncthreads + warp-reduce cost paid once per tile (see GEMV_TILE_FIXED_US)."""
        extra = self.gemv_tile_fixed_us if task.op == InstructionKind.GEMV_TILE else 0.0
        return self.transition_us + extra

    # ---- rates -----------------------------------------------------------------------
    def _bytes_per_us(self, target: GpuTarget) -> float:
        # GB/s -> bytes/us : *1e9 bytes/s / 1e6 us/s = *1e3
        return max(target.hbm_bandwidth_gbs, 1e-9) * 1e3 * self.achievable_bw_fraction

    def _flops_per_us(self, target: GpuTarget) -> float:
        # TFLOP/s -> flop/us : *1e12 / 1e6 = *1e6
        return max(target.fp16_tflops, 1e-9) * 1e6 * self.achievable_flop_fraction

    def _pipeline_exposed_fraction(self, prog: MegakernelProgram) -> float:
        """Fraction of each task's HBM-byte leg still exposed after software pipelining. Deeper
        prefetch hides more of the inter-op bubble, down to MIN_EXPOSED_BYTES_FRACTION."""
        depth = 0
        if prog.config is not None:
            depth = max(0, int(prog.config.pipelining_depth))
        exposed = 1.0 / (1.0 + depth * self.pipeline_gain)
        return max(MIN_EXPOSED_BYTES_FRACTION, exposed)

    # ---- per-task time ---------------------------------------------------------------
    def task_time_us(self, prog: MegakernelProgram, task, target: GpuTarget,
                     exposed_frac: float) -> float:
        """max(bytes-leg, flops-leg) + transition, with the bytes leg discounted by pipelining."""
        b, f = _estimate_task_bytes_flops(prog, task)
        bytes_us = (b / self._bytes_per_us(target)) * exposed_frac
        flops_us = f / self._flops_per_us(target)
        return max(bytes_us, flops_us) + self._fixed_overhead_us(task)

    # ---- the main entry point --------------------------------------------------------
    def estimate(self, prog: MegakernelProgram, target: GpuTarget | None = None) -> CostBreakdown:
        """Predict decode latency for ``prog`` on ``target`` (defaults to ``prog.target``).

        Computes per-task times, the critical-path makespan honoring counter deps + per-SM serial
        queues, the regional breakdown, and the distance to the weights/bandwidth floor. Always
        returns a finite, positive ``predicted_us`` (degenerate empty programs cost one transition).
        """
        target = target or prog.target
        if target is None:
            raise ValueError("CostModel.estimate needs a GpuTarget (prog.target is None)")

        notes: list[str] = []
        exposed_frac = self._pipeline_exposed_fraction(prog)

        # 1) per-task time + region + byte/flop accumulation
        t_time: dict[int, float] = {}
        region_of: dict[int, str] = {}
        region_work: dict[str, float] = {r: 0.0 for r in REGIONS}
        total_work = 0.0
        total_bytes_us = 0.0
        total_flops_us = 0.0
        for task in prog.tasks:
            b, f = _estimate_task_bytes_flops(prog, task)
            bytes_us = (b / self._bytes_per_us(target)) * exposed_frac
            flops_us = f / self._flops_per_us(target)
            tt = max(bytes_us, flops_us) + self._fixed_overhead_us(task)
            t_time[task.id] = tt
            reg = classify_region(task)
            region_of[task.id] = reg
            region_work[reg] += tt
            total_work += tt
            total_bytes_us += bytes_us
            total_flops_us += flops_us

        n_tasks = len(prog.tasks)
        transition_total = n_tasks * self.transition_us

        # 2) makespan = longest weighted path through (deps ∪ per-SM serial queue)
        makespan, region_cp, n_sms_used = self._critical_path(prog, target, t_time, region_of, notes)

        # Degenerate guard: an empty / pathological program still costs at least one transition.
        if makespan <= 0:
            makespan = self.transition_us
            notes.append("empty/degenerate program; charged one transition as floor")

        weight_bytes = prog.total_weight_bytes()
        bw_floor = target.bandwidth_bound_us(weight_bytes)
        # If the lowering carried no WEIGHT buffers (e.g. a synthetic toy DAG), fall back to the
        # task-level HBM traffic so the floor is still a meaningful, positive roofline.
        if not (bw_floor > 0):
            bw_floor = total_bytes_us / max(exposed_frac, 1e-9) if total_bytes_us > 0 else 0.0
            if bw_floor > 0:
                notes.append("no WEIGHT buffers; bandwidth floor derived from task HBM traffic")

        return CostBreakdown(
            predicted_us=makespan,
            makespan_us=makespan,
            total_work_us=total_work,
            bandwidth_bound_us=bw_floor,
            region_us=region_cp,
            region_work_us=region_work,
            n_tasks=n_tasks,
            n_sms_used=n_sms_used,
            transition_us_total=transition_total,
            bytes_bound_us=total_bytes_us,
            flops_bound_us=total_flops_us,
            notes=notes,
        )

    # ---- critical-path engine --------------------------------------------------------
    def _critical_path(self, prog: MegakernelProgram, target: GpuTarget,
                       t_time: dict[int, float], region_of: dict[int, str],
                       notes: list[str]) -> tuple[float, dict[str, float], int]:
        """Longest weighted path (the makespan) through the union of:
          * counter dependency edges (producer -> consumer), and
          * per-SM serial-queue edges (consecutive tasks on the same SM, in task-list order),
        which is exactly what the persistent VM realizes: same-SM tasks serialize, cross-SM
        tasks overlap. Also returns each region's share of the critical path and #SMs used.

        If SMs are unassigned we still honor the counter DAG, but additionally cap parallelism at
        ``num_sms`` by serializing within the cheapest balancing that respects the topo order -
        modeled analytically as max(dependency_cp, total_work / num_sms)."""
        adj, indeg = prog._adjacency()
        order = prog.topological_order(adj, indeg)
        if order is None:
            # Cyclic (validator would reject). Don't crash search: report serial work as a
            # pessimistic finite cost so the candidate is simply ranked poorly / discarded upstream.
            notes.append("graph has a cycle; cost falls back to serial work (will be rejected)")
            total = sum(t_time.values())
            rw = {r: 0.0 for r in REGIONS}
            for tid, tt in t_time.items():
                rw[region_of[tid]] += tt
            return total, rw, 0

        sm_assigned = any(t.sm is not None for t in prog.tasks)

        # Build the effective edge set.
        succ: dict[int, list[int]] = {t.id: list(adj[t.id]) for t in prog.tasks}
        if sm_assigned:
            # Add serial-queue edges: within each SM, in task-list order, each task precedes the
            # next (the loader queue == task-list order per SM, matching validate()'s assumption).
            by_sm: dict[int, list[int]] = {}
            for t in prog.tasks:
                if t.sm is not None:
                    by_sm.setdefault(t.sm, []).append(t.id)
            for sm, ids in by_sm.items():
                ids_sorted = sorted(ids, key=lambda i: i)  # task-list order == id order at emit
                for a, b in zip(ids_sorted, ids_sorted[1:]):
                    if b not in succ[a]:
                        succ[a].append(b)
            n_sms_used = len(by_sm)
        else:
            n_sms_used = 0

        # Longest path via DP over a topo order of the *combined* graph. Adding serial-queue edges
        # preserves acyclicity (they follow emit order, which is a linear extension the validator
        # requires), but recompute the topo order on the combined graph to be safe.
        combined_indeg = {t.id: 0 for t in prog.tasks}
        for a, outs in succ.items():
            for b in outs:
                combined_indeg[b] += 1
        topo = self._topo(prog, succ, combined_indeg)
        if topo is None:
            notes.append("combined graph (deps+SM queue) is cyclic; using serial work")
            total = sum(t_time.values())
            rw = {r: 0.0 for r in REGIONS}
            for tid, tt in t_time.items():
                rw[region_of[tid]] += tt
            return total, rw, n_sms_used

        # finish[t] = longest completion time of any path ending at t (inclusive of t).
        # Build predecessor lists once so the DP is O(V+E) rather than scanning succ per node.
        preds: dict[int, list[int]] = {t.id: [] for t in prog.tasks}
        for a, outs in succ.items():
            for b in outs:
                preds[b].append(a)
        finish: dict[int, float] = {}
        best_pred: dict[int, int | None] = {}
        for tid in topo:
            start = 0.0
            bp: int | None = None
            for p in preds[tid]:
                if finish[p] > start:
                    start = finish[p]
                    bp = p
            finish[tid] = start + t_time[tid]
            best_pred[tid] = bp

        if not finish:
            return 0.0, {r: 0.0 for r in REGIONS}, n_sms_used

        # Makespan and the critical chain (for the region breakdown).
        end = max(finish, key=lambda k: finish[k])
        makespan = finish[end]

        region_cp = {r: 0.0 for r in REGIONS}
        node: int | None = end
        while node is not None:
            region_cp[region_of[node]] += t_time[node]
            node = best_pred[node]

        # When SMs are unassigned, the pure dependency critical path is an *optimistic* (infinite-SM)
        # bound. Real hardware has only num_sms workers, so the makespan can't beat work/num_sms.
        if not sm_assigned:
            num_sms = max(1, int(target.num_sms))
            work = sum(t_time.values())
            parallel_floor = work / num_sms
            if parallel_floor > makespan:
                # Scale the region shares up proportionally so the breakdown still sums sensibly.
                scale = parallel_floor / makespan if makespan > 0 else 1.0
                region_cp = {r: v * scale for r, v in region_cp.items()}
                makespan = parallel_floor
                notes.append(f"SMs unassigned; makespan raised to work/num_sms ({num_sms} SMs)")
            n_sms_used = min(num_sms, self._dag_width(prog, adj))

        return makespan, region_cp, n_sms_used

    @staticmethod
    def _topo(prog: MegakernelProgram, succ: dict[int, list[int]],
              indeg: dict[int, int]) -> list[int] | None:
        from collections import deque
        indeg = dict(indeg)
        ready = deque(sorted(tid for tid, d in indeg.items() if d == 0))
        order: list[int] = []
        while ready:
            n = ready.popleft()
            order.append(n)
            for m in succ[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
        return order if len(order) == len(prog.tasks) else None

    @staticmethod
    def _dag_width(prog: MegakernelProgram, adj: dict[int, list[int]]) -> int:
        """Rough max antichain width = the largest number of tasks at one dependency 'level'
        (longest-distance-from-source layering). A cheap parallelism proxy for #SMs-usable."""
        _, indeg = prog._adjacency()
        order = prog.topological_order(adj, indeg)
        if order is None:
            return 1
        level: dict[int, int] = {}
        preds: dict[int, list[int]] = {t.id: [] for t in prog.tasks}
        for a, outs in adj.items():
            for b in outs:
                preds[b].append(a)
        for tid in order:
            level[tid] = (max((level[p] for p in preds[tid]), default=-1) + 1)
        counts: dict[int, int] = {}
        for lv in level.values():
            counts[lv] = counts.get(lv, 0) + 1
        return max(counts.values(), default=1)


# ----------------------------------------------------------------------------------------
# Module-level convenience (the API search and compile.py call)
# ----------------------------------------------------------------------------------------
_DEFAULT_MODEL = CostModel()


def predict_us(program: MegakernelProgram, target: GpuTarget | None = None) -> float:
    """One-call analytic latency prediction in microseconds (single-stream decode).

    Convenience wrapper around the default :class:`CostModel`. Always returns a finite, positive
    float for any well-formed program. See :func:`estimate` for the full breakdown."""
    return _DEFAULT_MODEL.estimate(program, target).predicted_us


def estimate(program: MegakernelProgram, target: GpuTarget | None = None) -> CostBreakdown:
    """Full analytic :class:`CostBreakdown` (predicted_us + regional + roofline detail)."""
    return _DEFAULT_MODEL.estimate(program, target)


__all__ = [
    "CostModel", "CostBreakdown", "predict_us", "estimate", "classify_region",
    "DEFAULT_TRANSITION_US", "REGIONS",
    "REGION_ATTENTION", "REGION_MLP", "REGION_LM_HEAD", "REGION_OTHER",
]
