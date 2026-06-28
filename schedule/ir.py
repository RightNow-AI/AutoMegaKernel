"""
AutoMegaKernel, THE STANDARD MEGAKERNEL IR
============================================

This module defines the canonical intermediate representation for a megakernel: the
SM-level **task-DAG**, the **instruction ABI** mapping, the **counter-based sync** model,
and the **schedule format**. It is the asset others standardize on (the "DWG format"), so
it is deliberately:

  * **Dependency-free.** Pure Python + stdlib only. No torch, no numpy, no CUDA. You can read,
    validate, diff, and version a schedule on a laptop with no GPU. Numerics live in the VM
    and the instruction library; *structure* lives here.
  * **Stable & documented.** Every field is explained. Serialization is explicit, human-readable,
    additive-compatible JSON (git-friendly), versioned by ``IR_VERSION``/``ABI_VERSION``.
  * **Correct by construction.** The single most important function in the repo,
    :func:`validate`, statically proves the **deadlock-freedom invariant** AND the
    **race-freedom invariant**. The VM must refuse to load any program it rejects.

------------------------------------------------------------------------------------
THE SYNC MODEL (read this, several subtleties were load-bearing bugs in v0)
------------------------------------------------------------------------------------
Each task, on completion, performs exactly one ``out_counter += 1`` (producers only
*increment*) after a release fence that orders ALL of its output-buffer writes before the
increment. So ``out_counter`` means "every output of this task is fully written and visible."

Each task, before executing, *waits* on a set of ``(counter, threshold)`` pairs with
**statically known** thresholds (consumers only *wait*, never signal).

Two invariants make this safe, both enforced by :func:`validate`:

  1. **Deadlock-freedom.** Every wait threshold ``t`` satisfies ``1 <= t <= #producers(counter)``
     and the producer→consumer graph is acyclic ⇒ a topological order exists ⇒ every wait is
     eventually satisfiable. (When SMs are assigned, each SM's serial queue must additionally be
     a linear extension of the DAG, else an SM can block on a counter only its own later queue
     entry could signal, also checked.)

  2. **Race-freedom (the subtle one).** A counter only carries a *count*, not *which* producer
     finished. So a ``Wait(c, k)`` with ``1 < k < #producers(c)`` is a "first-k-of-N" race: the
     wrong producers may satisfy it. We therefore require: **a counter with >1 producer is a
     true join, every wait on it must use threshold == #producers** (wait for ALL). And every
     activation/KV read must be backed by a *transitive* producer edge (real happens-before),
     not merely "earlier in some order." Both are enforced.

------------------------------------------------------------------------------------
DECODE-LOOP MODEL (frozen)
------------------------------------------------------------------------------------
**One kernel launch == one forward pass == one decoded token.** Counters are host-memset to
zero before each launch; KV_CACHE persists in HBM across launches; positions advance via params
between launches. The host drives the autoregressive loop. This keeps each launch well under the
Windows WDDM TDR ~2s watchdog (the dev GPU is a laptop on WDDM) while preserving the
megakernel-within-a-step win. See ``vm/abi.h`` for the on-device contract.

------------------------------------------------------------------------------------
ABI MAPPING
------------------------------------------------------------------------------------
Each :class:`Task` maps 1:1 onto an instruction invocation with the frozen ABI in ``vm/abi.h``::

    instruction(input_page_ptrs, output_page_ptrs, params, in_counters, out_counter)
                 ^inputs (buffers)  ^outputs        ^params  ^waits        ^out_counter

The enum values here (``DType``, ``MemSpace``, ``InstructionKind``) and the capacity/version
constants are CANONICAL; ``vm/abi.h`` must match. ``tests/test_abi_sync.py`` enforces it.
"""

from __future__ import annotations

import dataclasses
import json
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from enum import IntEnum
from typing import Any

# Bump on any breaking change to the on-disk format. Minor = additive/compatible.
IR_VERSION = "0.2.0"
# Mirrors AMK_ABI_VERSION_{MAJOR,MINOR} in vm/abi.h. A program records the ABI it targets;
# from_dict gates on a major-version mismatch.
ABI_VERSION = "0.2"

# Fixed POD capacities, MUST equal the #defines in vm/abi.h (enforced by test_abi_sync.py).
ABI_MAX_INPUTS = 8
ABI_MAX_OUTPUTS = 4
ABI_MAX_WAITS = 8
ABI_MAX_RANK = 4


# ======================================================================================
# Element types
# ======================================================================================
class DType(IntEnum):
    """Element types. Numeric value is the canonical ABI code (mirrored in vm/abi.h)."""

    F32 = 0
    F16 = 1
    BF16 = 2
    F8E4M3 = 3
    F8E5M2 = 4
    I32 = 5
    I8 = 6
    I4 = 7  # packed 4-bit (two per byte), quantized weights
    U8 = 8
    BOOL = 9

    @property
    def bits(self) -> int:
        return {
            DType.F32: 32, DType.F16: 16, DType.BF16: 16, DType.F8E4M3: 8, DType.F8E5M2: 8,
            DType.I32: 32, DType.I8: 8, DType.I4: 4, DType.U8: 8, DType.BOOL: 8,
        }[self]

    def nbytes(self, count: int) -> int:
        """Bytes for ``count`` elements (ceil for sub-byte packed types)."""
        return (count * self.bits + 7) // 8


# ======================================================================================
# Memory spaces & buffer roles
# ======================================================================================
class MemSpace(IntEnum):
    """Where a buffer physically lives. The page allocator assigns activations to on-chip
    spaces so they do not round-trip to HBM between ops."""

    HBM = 0             # device global memory (weights, KV cache, model IO)
    GLOBAL_SCRATCH = 1  # VM-managed global scratchpad pages (large activations)
    SMEM = 2            # shared-memory pages (hot activations, the megakernel win)
    REGISTER = 3        # register-resident (tiny, lowering hint only)


class BufferKind(IntEnum):
    """Semantic role of a buffer in the model."""

    WEIGHT = 0      # read-only parameter in HBM; prefetchable (the bandwidth bound is on these)
    ACTIVATION = 1  # transient; lives in pages; reused across non-overlapping live ranges
    KV_CACHE = 2    # read/write persistent state in HBM (written by KV_APPEND, read by attention)
    IO_INPUT = 3    # model input (token ids / input embeds) in HBM
    IO_OUTPUT = 4   # model output (logits / sampled token) in HBM
    CONST = 5       # small constants (rope tables, scales, biases)


# Read-only external kinds: reading them never needs a producer edge.
_READONLY_KINDS = frozenset({BufferKind.WEIGHT, BufferKind.CONST, BufferKind.IO_INPUT})


# ======================================================================================
# Instruction set (opcodes), the Layer-1 ABI archetypes
# ======================================================================================
class InstructionKind(IntEnum):
    """Opcode of an ABI-conformant micro-kernel. Numeric value is the canonical ABI code
    (mirrored in vm/abi.h). Extend by appending; never renumber existing entries."""

    NOP = 0
    COPY = 1             # page -> page move (materialize residual stream segments)
    EMBED = 2            # token id -> embedding row gather
    RMSNORM = 3          # RMSNorm(x) * weight
    LAYERNORM = 4        # LayerNorm(x, weight, bias)
    GEMV_TILE = 5        # [1,K] x [K,N_tile] -> [1,N_tile]  (decode matvec tile)
    GEMM_TILE = 6        # [M_tile,K] x [K,N_tile] -> [M_tile,N_tile] (prefill / batch)
    ATTENTION_TILE = 7   # attention over a KV window (whole-window; multi-block uses COMBINE)
    ROPE = 8             # rotary positional embedding (in place on q/k)
    SILU_MUL = 9         # SwiGLU: silu(gate) * up
    GELU = 10
    ADD = 11             # residual add
    MUL = 12             # elementwise / scale
    DEQUANT = 13         # int4/int8 + scales -> fp16/bf16 tile (fused into gemv/gemm normally)
    SOFTMAX = 14
    ALLREDUCE_SHARD = 15  # tensor-parallel shard reduce (multi-GPU; M6)
    KV_APPEND = 16       # append new k/v into the KV cache page
    SAMPLE_ARGMAX = 17   # logits -> next token (greedy); decode loop terminator
    ATTENTION_COMBINE = 18  # merge per-KV-block (out, m, l) partials (flash online softmax)
    FUSED = 19          # a recipe of primitive ops run with on-chip scratch (kernel fusion). The
                        # CPU reference interprets the recipe; a future slice code-gens it to CUDA.


@dataclass(frozen=True)
class OpSpec:
    """Static description of an opcode: arity and required param keys. Used by the lowerer to
    emit well-formed tasks and by :func:`validate` to sanity-check them."""

    kind: InstructionKind
    min_inputs: int
    max_inputs: int          # -1 = variadic (still capped at ABI_MAX_INPUTS by validate)
    n_outputs: int           # -1 = variable (1..ABI_MAX_OUTPUTS)
    required_params: tuple[str, ...] = ()
    note: str = ""


# Canonical opcode registry. required_params pin the scalars that make the op mathematically
# correct for a real decoder (e.g. RoPE needs theta; attention needs scale + GQA grouping).
OP_REGISTRY: dict[InstructionKind, OpSpec] = {
    InstructionKind.NOP: OpSpec(InstructionKind.NOP, 0, 0, 0),
    InstructionKind.COPY: OpSpec(InstructionKind.COPY, 1, 1, 1),
    InstructionKind.EMBED: OpSpec(InstructionKind.EMBED, 2, 2, 1, ("hidden",)),
    InstructionKind.RMSNORM: OpSpec(InstructionKind.RMSNORM, 2, 2, 1, ("eps", "hidden")),
    InstructionKind.LAYERNORM: OpSpec(InstructionKind.LAYERNORM, 2, 3, 1, ("eps", "hidden")),
    InstructionKind.GEMV_TILE: OpSpec(InstructionKind.GEMV_TILE, 2, 3, 1, ("K", "N_tile", "n_off")),
    InstructionKind.GEMM_TILE: OpSpec(
        InstructionKind.GEMM_TILE, 2, 3, 1, ("M_tile", "K", "N_tile", "n_off")),
    InstructionKind.ATTENTION_TILE: OpSpec(
        InstructionKind.ATTENTION_TILE, 3, 4, 1,
        ("head_dim", "kv_start", "kv_len", "scale", "n_heads", "n_kv_heads")),
    InstructionKind.ROPE: OpSpec(InstructionKind.ROPE, 2, 2, 1, ("head_dim", "theta")),
    InstructionKind.SILU_MUL: OpSpec(InstructionKind.SILU_MUL, 2, 2, 1),
    InstructionKind.GELU: OpSpec(InstructionKind.GELU, 1, 1, 1),
    InstructionKind.ADD: OpSpec(InstructionKind.ADD, 2, 2, 1),
    InstructionKind.MUL: OpSpec(InstructionKind.MUL, 1, 2, 1),
    InstructionKind.DEQUANT: OpSpec(InstructionKind.DEQUANT, 2, 3, 1, ("qdtype", "group")),
    InstructionKind.SOFTMAX: OpSpec(InstructionKind.SOFTMAX, 1, 1, 1),
    InstructionKind.ALLREDUCE_SHARD: OpSpec(InstructionKind.ALLREDUCE_SHARD, 1, 8, 1),
    InstructionKind.KV_APPEND: OpSpec(InstructionKind.KV_APPEND, 2, 2, 1, ("pos",)),
    InstructionKind.SAMPLE_ARGMAX: OpSpec(InstructionKind.SAMPLE_ARGMAX, 1, 1, 1),
    InstructionKind.ATTENTION_COMBINE: OpSpec(InstructionKind.ATTENTION_COMBINE, 2, 8, 1),
    # A fused instruction: 1..N net inputs, 1 output, the fusion described by the "recipe" param
    # (a list of primitive steps run over on-chip scratch). Variadic inputs (the union of the
    # fused subgraph's external reads).
    InstructionKind.FUSED: OpSpec(InstructionKind.FUSED, 1, -1, 1, ("recipe",)),
}

# Known scalar param keys and their marshalled type ('i' -> int32, 'f' -> float in amk_params_t).
# Anything not here is flagged (warning) by validate so IR<->ABI param drift is caught early.
PARAM_FIELDS: dict[str, str] = {
    "K": "i", "N": "i", "M": "i", "N_tile": "i", "M_tile": "i", "n_off": "i", "m_off": "i",
    "hidden": "i", "vocab": "i", "head_dim": "i", "n_heads": "i", "n_kv_heads": "i",
    "kv_start": "i", "kv_len": "i", "pos": "i", "qdtype": "i", "flags": "i", "group": "i",
    "dim": "i", "eps": "f", "scale": "f", "theta": "f",
}
_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1

# Opcodes that do NOT yet have a reference oracle (instructions/reference.py) and therefore
# cannot be executed/verified. validate() rejects any task using one so a schedule can never be
# built on an op whose correctness we cannot check. Remove an entry once its reference lands.
UNIMPLEMENTED_OPS = frozenset()   # ATTENTION_COMBINE now has a reference oracle + CUDA kernel (flash-decoding split-KV)

# Buffers in these spaces are on-chip and per-block: a producer and consumer of one such buffer
# MUST run on the same SM (counters give ordering, not cross-SM data movement).
_ONCHIP_SPACES = frozenset({MemSpace.SMEM, MemSpace.REGISTER})

# Above this task count the exact transitive-provenance check is skipped (and logged) to keep
# validation tractable; smaller schedules get the full O(V·E)-ish proof.
_PROVENANCE_MAX_TASKS = 8000

# Opcodes whose write footprint into the output buffer is a disjoint TILE described by params:
# the column range [n_off, n_off + N_tile) and (GEMM only) the row range [m_off, m_off + M_tile).
# Any other opcode is treated as writing the WHOLE buffer (footprint overlaps everything) for the
# purposes of the multi-writer WAW disjointness proof.
_TILED_WRITE_OPS = frozenset({InstructionKind.GEMV_TILE, InstructionKind.GEMM_TILE})


def _write_region(task: "Task") -> tuple[tuple[int, int], tuple[int, int]] | None:
    """The (row_range, col_range) a writer task covers in its output buffer, as half-open
    [lo, hi) element intervals, or ``None`` for a whole-buffer (overlap-everything) writer.

    GEMV/GEMM tiles cover columns [n_off, n_off + N_tile); GEMM also covers rows
    [m_off, m_off + M_tile). A GEMV row range is the full (unbounded) row dimension, modelled as
    the sentinel ``(0, -1)`` ('all rows') so two GEMV tiles are compared on columns alone."""
    if task.op not in _TILED_WRITE_OPS:
        return None
    p = task.params
    n_off, n_tile = p.get("n_off"), p.get("N_tile")
    if not isinstance(n_off, int) or isinstance(n_off, bool) or n_off < 0:
        return None
    if not isinstance(n_tile, int) or isinstance(n_tile, bool) or n_tile <= 0:
        return None
    col = (n_off, n_off + n_tile)
    if task.op is InstructionKind.GEMM_TILE:
        m_off, m_tile = p.get("m_off"), p.get("M_tile")
        if (isinstance(m_off, int) and not isinstance(m_off, bool) and m_off >= 0
                and isinstance(m_tile, int) and not isinstance(m_tile, bool) and m_tile > 0):
            return ((m_off, m_off + m_tile), col)
    return ((0, -1), col)  # (0, -1) == "all rows" sentinel: a full, unbounded row range


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Half-open [lo, hi) overlap. ``hi == -1`` means an unbounded ('all') range on that axis."""
    a_all = a[1] == -1
    b_all = b[1] == -1
    if a_all or b_all:
        return True
    return a[0] < b[1] and b[0] < a[1]


def _regions_disjoint(ra, rb) -> bool:
    """Two tile footprints are DISJOINT iff their row ranges are disjoint OR their column ranges
    are disjoint (separable along either axis). A ``None`` footprint is a whole-buffer writer and
    overlaps everything, so it is never disjoint from another writer."""
    if ra is None or rb is None:
        return False
    (ar, ac), (br, bc) = ra, rb
    return (not _ranges_overlap(ar, br)) or (not _ranges_overlap(ac, bc))


# ======================================================================================
# Core records
# ======================================================================================
@dataclass
class Buffer:
    """A named tensor in the program. Weights/KV/IO are physical HBM tensors; activations are
    logical and bound to physical pages by the allocator (see :class:`PageAllocation`)."""

    id: int
    name: str
    kind: BufferKind
    dtype: DType
    shape: tuple[int, ...]
    space: MemSpace = MemSpace.HBM
    # For WEIGHT/CONST buffers: dotted key into the model state_dict this maps to. None otherwise.
    source: str | None = None

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return self.dtype.nbytes(self.numel)

    def contiguous_strides(self) -> tuple[int, ...]:
        """Row-major element strides (what the loader writes into amk_buffer_t.stride)."""
        strides = [1] * len(self.shape)
        for i in range(len(self.shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * self.shape[i + 1]
        return tuple(strides)


@dataclass
class Wait:
    """A consumer's precondition: do not execute until ``counter >= threshold``.
    ``threshold`` MUST be statically known; for a counter with >1 producer it MUST equal the
    producer count (a true join, see the race-freedom invariant)."""

    counter: int
    threshold: int


@dataclass
class Task:
    """A node in the task-DAG == one ABI instruction invocation on one SM.

    ``out_counter`` is incremented by 1 on completion and means "all my outputs are written
    and visible." ``sm`` is the lowering OUTPUT (assigned by the scheduler from
    ``ScheduleConfig.sm_assignment``); agents/search edit the config, never this field."""

    id: int
    op: InstructionKind
    inputs: list[int]                       # buffer ids read
    outputs: list[int]                      # buffer ids written
    out_counter: int                        # incremented by 1 on completion
    waits: list[Wait] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    sm: int | None = None                   # SM/worker assignment (lowering output; None = unassigned)
    est_bytes: int = 0                      # HBM bytes this task moves (weights dominate)
    est_flops: int = 0
    label: str = ""                         # human-readable, e.g. "L0.mlp.down[tile3]"


@dataclass
class Counter:
    """A synchronization counter. Starts at ``init`` (always 0 for a fresh forward pass),
    monotonically incremented by its producer tasks. Never decremented."""

    id: int
    init: int = 0
    note: str = ""


@dataclass
class Page:
    """A physical scratchpad slot the allocator hands out. An activation buffer is bound to a
    page for its live range; non-overlapping live ranges share a page (graph-coloring reuse)."""

    id: int
    space: MemSpace
    nbytes: int
    live_start: int = -1
    live_end: int = -1


@dataclass
class PageAllocation:
    """Output of page allocation: which physical page backs each activation buffer, and the
    page table. Produced by Layer 2; ``None`` until allocation runs."""

    buffer_to_page: dict[int, int] = field(default_factory=dict)  # buffer id -> page id
    pages: list[Page] = field(default_factory=list)

    @property
    def total_scratch_bytes(self) -> int:
        return sum(p.nbytes for p in self.pages)


@dataclass
class ScheduleConfig:
    """THE LOOP-2 EDIT SURFACE. The structured object a coding agent proposes; the frozen VM
    deterministically lowers it into a runnable megakernel. The agent never writes kernel code -
    it only chooses a point in this search space, which the VM knows how to realize safely. Also
    exactly what gets logged to the flywheel as the "schedule" column."""

    # Tile sizes per op archetype, e.g. {"gemv": {"N_tile": 256}, "attention": {"kv_block": 128}}.
    tiling: dict[str, dict[str, int]] = field(default_factory=dict)
    # How to fuse adjacent ops into a single resident task group (list of op-name groups).
    fusion_grouping: list[list[str]] = field(default_factory=list)
    # SM assignment policy: "round_robin" | "load_balance" | explicit {task_id: sm}. INPUT only;
    # the lowerer resolves this into each Task.sm.
    sm_assignment: str | dict[int, int] = "load_balance"
    # Software-pipelining depth: instructions ahead to prefetch weights. The biggest megakernel
    # win (hides the inter-op HBM bubble). 0 = no prefetch.
    pipelining_depth: int = 2
    # Page-allocation policy: "linear" | "graph_color" | "none".
    page_allocation: str = "graph_color"
    # ---- launch config (searchable; the loader proves occupancy against the target) ----
    threads_per_block: int = 256        # block size for the persistent VM kernel
    smem_bytes_per_block: int = 0       # dynamic SMEM opt-in per block (<= target opt-in cap)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if isinstance(self.sm_assignment, dict):
            d["sm_assignment"] = {str(k): v for k, v in self.sm_assignment.items()}
        return d


@dataclass
class GpuTarget:
    """A GPU described as *data*, never as branches in code. This is the retargeting surface:
    add a new chip by adding a record, not by editing the scheduler. Search consumes these
    numbers; the cost model and roofline use them; the loader proves the launch config fits."""

    name: str               # "rtx5090", "b200", "h100", "mi300x"
    sm_arch: int            # compute capability * 10, e.g. 120 for sm_120 (Blackwell)
    num_sms: int
    smem_bytes_per_sm: int          # physical SMEM partition per SM
    smem_bytes_per_block_optin: int  # MAX dynamic SMEM a single block may opt into (< per_sm!)
    regs_per_sm: int
    max_threads_per_sm: int
    max_regs_per_thread: int
    l2_bytes: int
    hbm_bytes: int
    hbm_bandwidth_gbs: float        # the bound: weights_bytes / bw is the decode floor (SPEC peak)
    fp16_tflops: float
    clock_ghz: float = 0.0
    supports_cooperative: bool = True   # cudaLaunchCooperativeKernel available (grid.sync)
    wddm_tdr: bool = False              # display GPU with a ~2s OS watchdog (per-step relaunch!)
    # MEASURED sustained HBM bandwidth (GB/s) from eval/peak_bandwidth.py (D2D copy / STREAM triad,
    # CUDA-event median, run on real silicon). The FAIRER roofline denominator than the spec figure:
    # the spec is the *desktop* part / theoretical peak, while real sustained (esp. on this laptop
    # GB203 held in a power-limited clock state) is lower. 0.0 = not yet measured (fall back to spec).
    measured_bw_gbs: float = 0.0
    note: str = ""

    def bandwidth_bound_us(self, weight_bytes: int) -> float:
        """The honest performance floor for single-stream decode: time to stream all weights
        through HBM once, in microseconds. Uses the SPEC ``hbm_bandwidth_gbs`` (the conventional
        roofline denominator); see :meth:`measured_bandwidth_bound_us` for the measured-peak floor."""
        if self.hbm_bandwidth_gbs <= 0:
            return float("nan")
        return (weight_bytes / (self.hbm_bandwidth_gbs * 1e9)) * 1e6

    def measured_bandwidth_bound_us(self, weight_bytes: int) -> float:
        """The fairer decode floor: weights / MEASURED sustained bandwidth, in microseconds.
        Falls back to the spec floor when no measured peak has been recorded (``measured_bw_gbs``
        is 0). This is the denominator that does not penalize us for the spec/laptop gap."""
        bw = self.measured_bw_gbs if self.measured_bw_gbs > 0 else self.hbm_bandwidth_gbs
        if bw <= 0:
            return float("nan")
        return (weight_bytes / (bw * 1e9)) * 1e6


# Built-in target registry. RTX 5090 (Laptop) numbers are MEASURED on this machine via
# torch.cuda.get_device_properties (SM count / smem / L2 / regs exact); bandwidth + tflops are
# spec-derived estimates (labeled). Datacenter entries are public spec sheets used only by the
# cost model/roofline, we never report *measured* numbers for a chip we did not run on.
TARGETS: dict[str, GpuTarget] = {
    "rtx5090": GpuTarget(
        name="rtx5090", sm_arch=120, num_sms=82, smem_bytes_per_sm=102400,
        smem_bytes_per_block_optin=101376, regs_per_sm=65536, max_threads_per_sm=2048,
        max_regs_per_thread=255, l2_bytes=64 * 1024 * 1024, hbm_bytes=24 * 1024**3,
        hbm_bandwidth_gbs=896.0, fp16_tflops=210.0, clock_ghz=1.597,
        supports_cooperative=True, wddm_tdr=True, measured_bw_gbs=731.0,
        note="Blackwell Laptop GB203, sm_120, 24GB, WDDM display GPU. THIS machine. "
             "measured_bw_gbs=731 via eval/peak_bandwidth.py (D2D copy, best-boosted CUDA-event "
             "sample; median of three runs that read 753/731/633 GB/s as the SW power cap "
             "(throttle 0x4) modulates the 14001 MHz mem clock down to ~11001). The laptop part is "
             "power-limited so sustained HBM (~647-753 GB/s, ~731 representative) is below the 896 "
             "desktop-spec figure, measured is the fair floor."),
    "b200": GpuTarget(
        name="b200", sm_arch=100, num_sms=148, smem_bytes_per_sm=233472,
        smem_bytes_per_block_optin=232448, regs_per_sm=65536, max_threads_per_sm=2048,
        max_regs_per_thread=255, l2_bytes=50 * 1024 * 1024, hbm_bytes=192 * 1024**3,
        hbm_bandwidth_gbs=8000.0, fp16_tflops=2250.0, clock_ghz=1.86,
        supports_cooperative=True, wddm_tdr=False, note="Datacenter Blackwell, HBM3e. Spec only."),
    "h100": GpuTarget(
        name="h100", sm_arch=90, num_sms=132, smem_bytes_per_sm=233472,
        smem_bytes_per_block_optin=232448, regs_per_sm=65536, max_threads_per_sm=2048,
        max_regs_per_thread=255, l2_bytes=50 * 1024 * 1024, hbm_bytes=80 * 1024**3,
        hbm_bandwidth_gbs=3350.0, fp16_tflops=989.0, clock_ghz=1.98,
        supports_cooperative=True, wddm_tdr=False, measured_bw_gbs=3089.0,
        note="Hopper SXM5, HBM3. MEASURED via Modal. measured_bw_gbs=3089 (STREAM triad, "
             "CUDA-event median≈peak, clocks at max 1980/2619 MHz) = 92% of the 3350 spec."),
    "a100": GpuTarget(
        name="a100", sm_arch=80, num_sms=108, smem_bytes_per_sm=167936,
        smem_bytes_per_block_optin=166912, regs_per_sm=65536, max_threads_per_sm=2048,
        max_regs_per_thread=255, l2_bytes=40 * 1024 * 1024, hbm_bytes=40 * 1024**3,
        hbm_bandwidth_gbs=1555.0, fp16_tflops=312.0, clock_ghz=1.41,
        supports_cooperative=True, wddm_tdr=False, measured_bw_gbs=1382.0,
        note="Ampere SXM4 40GB, HBM2e. MEASURED via Modal. measured_bw_gbs=1382 (D2D copy, "
             "CUDA-event median≈peak, clocks at max 1410/1215 MHz) = 89% of the 1555 spec."),
    # Inference-class server GPUs (lower bandwidth, GDDR6), the GPUs that actually run batch-1 LLM
    # inference in production cloud. Spec sheets only (cost-model/roofline); measured_bw filled when run.
    "l4": GpuTarget(
        name="l4", sm_arch=89, num_sms=58, smem_bytes_per_sm=102400,
        smem_bytes_per_block_optin=101376, regs_per_sm=65536, max_threads_per_sm=1536,
        max_regs_per_thread=255, l2_bytes=48 * 1024 * 1024, hbm_bytes=24 * 1024**3,
        hbm_bandwidth_gbs=300.0, fp16_tflops=242.0, clock_ghz=2.04,
        supports_cooperative=True, wddm_tdr=False,
        note="Ada L4 (sm_89), 24GB GDDR6, the dominant cloud INFERENCE GPU. Spec 300 GB/s. Low "
             "bandwidth => the megakernel's per-tile cross-SM sync is a small fraction of the byte "
             "stream, so int8's ~0.5x weight read beats cuBLAS bf16 here (measured via Modal)."),
    "t4": GpuTarget(
        name="t4", sm_arch=75, num_sms=40, smem_bytes_per_sm=65536,
        smem_bytes_per_block_optin=65536, regs_per_sm=65536, max_threads_per_sm=1024,
        max_regs_per_thread=255, l2_bytes=4 * 1024 * 1024, hbm_bytes=16 * 1024**3,
        hbm_bandwidth_gbs=320.0, fp16_tflops=65.0, clock_ghz=1.59,
        supports_cooperative=True, wddm_tdr=False,
        note="Turing T4 (sm_75), 16GB GDDR6, the most-deployed cloud inference GPU. Spec 320 GB/s."),
    "a10g": GpuTarget(
        name="a10g", sm_arch=86, num_sms=80, smem_bytes_per_sm=102400,
        smem_bytes_per_block_optin=101376, regs_per_sm=65536, max_threads_per_sm=1536,
        max_regs_per_thread=255, l2_bytes=6 * 1024 * 1024, hbm_bytes=24 * 1024**3,
        hbm_bandwidth_gbs=600.0, fp16_tflops=125.0, clock_ghz=1.71,
        supports_cooperative=True, wddm_tdr=False,
        note="Ampere A10G (sm_86, AWS), 24GB GDDR6, inference GPU. Spec 600 GB/s."),
    "l40s": GpuTarget(
        name="l40s", sm_arch=89, num_sms=142, smem_bytes_per_sm=102400,
        smem_bytes_per_block_optin=101376, regs_per_sm=65536, max_threads_per_sm=1536,
        max_regs_per_thread=255, l2_bytes=96 * 1024 * 1024, hbm_bytes=48 * 1024**3,
        hbm_bandwidth_gbs=864.0, fp16_tflops=362.0, clock_ghz=2.52,
        supports_cooperative=True, wddm_tdr=False,
        note="Ada L40S (sm_89), 48GB GDDR6, current-gen datacenter INFERENCE flagship. Spec 864 GB/s "
             "(mid-bandwidth: the bandwidth x size crossover law predicts int8 crosses cuBLAS only at "
             "larger models ~7-13B, between A10G@600 and A100@1382). Shares sm_89 with L4 -> name-detected."),
}


# ======================================================================================
# The program (the schedule == the IR root)
# ======================================================================================
@dataclass
class MegakernelProgram:
    """A complete, runnable (after validation) megakernel schedule. This is the canonical
    artifact ``compile.py`` emits, the flywheel stores, and the VM loads."""

    meta: dict[str, Any] = field(default_factory=dict)   # model, gpu, regime, dtype, notes...
    target: GpuTarget | None = None
    buffers: list[Buffer] = field(default_factory=list)
    counters: list[Counter] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    pages: PageAllocation | None = None
    config: ScheduleConfig | None = None
    ir_version: str = IR_VERSION
    abi_version: str = ABI_VERSION

    # ---- builder helpers ----------------------------------------------------------
    def new_buffer(self, name: str, kind: BufferKind, dtype: DType, shape: tuple[int, ...],
                   space: MemSpace = MemSpace.HBM, source: str | None = None) -> Buffer:
        b = Buffer(len(self.buffers), name, kind, dtype, tuple(shape), space, source)
        self.buffers.append(b)
        return b

    def new_counter(self, note: str = "") -> Counter:
        c = Counter(len(self.counters), 0, note)
        self.counters.append(c)
        return c

    def add_task(self, op: InstructionKind, inputs: list[int], outputs: list[int],
                 out_counter: int, waits: list[Wait] | None = None,
                 params: dict[str, Any] | None = None, label: str = "",
                 est_bytes: int = 0, est_flops: int = 0, sm: int | None = None) -> Task:
        t = Task(id=len(self.tasks), op=op, inputs=list(inputs), outputs=list(outputs),
                 out_counter=out_counter, waits=list(waits or []), params=dict(params or {}),
                 sm=sm, est_bytes=est_bytes, est_flops=est_flops, label=label)
        self.tasks.append(t)
        return t

    # ---- indexing ------------------------------------------------------------------
    def buffer(self, bid: int) -> Buffer:
        return self.buffers[bid]

    def producers_by_counter(self) -> dict[int, list[int]]:
        d: dict[int, list[int]] = {}
        for t in self.tasks:
            d.setdefault(t.out_counter, []).append(t.id)
        return d

    def writers_by_buffer(self) -> dict[int, list[int]]:
        d: dict[int, list[int]] = {}
        for t in self.tasks:
            for b in t.outputs:
                d.setdefault(b, []).append(t.id)
        return d

    def producers_of(self, counter: int) -> list[Task]:
        return [t for t in self.tasks if t.out_counter == counter]

    def consumers_of(self, counter: int) -> list[Task]:
        return [t for t in self.tasks if any(w.counter == counter for w in t.waits)]

    def dependency_edges(self) -> list[tuple[int, int]]:
        """Producer→consumer edges: for every counter, each producer task precedes every task
        that waits on it. Acyclicity of this graph guarantees deadlock-freedom."""
        prod = self.producers_by_counter()
        edges: set[tuple[int, int]] = set()
        for t in self.tasks:
            for w in t.waits:
                for p in prod.get(w.counter, ()):
                    # Keep self-edges (p == t.id): a task waiting on its OWN out_counter is a
                    # self-deadlock, and the self-edge makes topological_order() reject it.
                    edges.add((p, t.id))
        return sorted(edges)

    def _adjacency(self) -> tuple[dict[int, list[int]], dict[int, int]]:
        adj: dict[int, list[int]] = {t.id: [] for t in self.tasks}
        indeg: dict[int, int] = {t.id: 0 for t in self.tasks}
        for a, b in self.dependency_edges():
            adj[a].append(b)
            indeg[b] += 1
        return adj, indeg

    def topological_order(self, adj=None, indeg=None) -> list[int] | None:
        """Kahn's algorithm. Returns task ids in a valid execution order, or ``None`` if the
        graph has a cycle (i.e. would deadlock). Deterministic (lowest id first)."""
        if adj is None or indeg is None:
            adj, indeg = self._adjacency()
        indeg = dict(indeg)
        ready = deque(sorted(tid for tid, d in indeg.items() if d == 0))
        order: list[int] = []
        while ready:
            n = ready.popleft()
            order.append(n)
            for m in adj[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
        # ready was filled in sorted bursts; for strict determinism re-sort each pop is overkill -
        # the lowest-id invariant matters only for reproducibility, which append-order preserves
        # well enough for a valid topo order. Return None on cycle.
        return order if len(order) == len(self.tasks) else None

    def simulate_counters(self) -> tuple[list[int], list[int]]:
        """STATIC reachability cross-check (NOT a faithful concurrent execution). Fires every
        ready task in list order, incrementing counters; reports (order, stuck_tasks).
        ``stuck`` is non-empty iff some wait is permanently unsatisfiable. For genuine race
        detection use :meth:`simulate_adversarial`."""
        cval = {c.id: c.init for c in self.counters}
        done = [False] * len(self.tasks)
        order: list[int] = []
        progress = True
        while progress:
            progress = False
            for t in self.tasks:
                if done[t.id]:
                    continue
                if all(cval.get(w.counter, 0) >= w.threshold for w in t.waits):
                    done[t.id] = True
                    cval[t.out_counter] = cval.get(t.out_counter, 0) + 1
                    order.append(t.id)
                    progress = True
        return order, [t.id for t in self.tasks if not done[t.id]]

    def simulate_adversarial(self, seeds: int = 16) -> list[str]:
        """Genuine race hunt: across many worst-case interleavings, fire ready tasks in varied
        orders and assert each task's transient (ACTIVATION/IO_OUTPUT/KV_CACHE) inputs were
        actually written by a *prior fired* task. Any read-before-write is a race the static
        checks should also catch, this is the dynamic backstop. Returns a list of violations."""
        violations: list[str] = []
        transient = {b.id for b in self.buffers if b.kind not in _READONLY_KINDS}
        # KV_CACHE pre-exists with prior-step state, so a read is only racy if THIS pass also
        # writes it before the read; treat a KV buffer as "ready" unless some task writes it.
        kv_written_this_pass = {b for b in transient
                                if self.buffers[b].kind == BufferKind.KV_CACHE
                                and any(b in t.outputs for t in self.tasks)}
        for s in range(max(1, seeds)):
            cval = {c.id: c.init for c in self.counters}
            written = {b.id for b in self.buffers if b.id not in transient}
            written |= {b for b in transient
                        if self.buffers[b].kind == BufferKind.KV_CACHE and b not in kv_written_this_pass}
            done = [False] * len(self.tasks)
            remaining = len(self.tasks)
            while remaining > 0:
                ready = [t for t in self.tasks if not done[t.id]
                         and all(cval.get(w.counter, 0) >= w.threshold for w in t.waits)]
                if not ready:
                    break  # deadlock, reported by simulate_counters/validate
                # vary the pick: rotate by seed to explore different "winners"
                t = ready[(s * 2654435761) % len(ready)]
                for b in t.inputs:
                    if b in transient and b not in written:
                        violations.append(
                            f"[seed {s}] task {t.id} ({t.label or t.op.name}) reads buffer "
                            f"#{b} ('{self.buffers[b].name}') before any producer wrote it")
                for b in t.outputs:
                    written.add(b)
                cval[t.out_counter] = cval.get(t.out_counter, 0) + 1
                done[t.id] = True
                remaining -= 1
            if violations:
                break
        # de-dup
        return sorted(set(violations))

    # ---- cost ----------------------------------------------------------------------
    def total_weight_bytes(self) -> int:
        return sum(b.nbytes for b in self.buffers if b.kind == BufferKind.WEIGHT)

    def total_est_bytes(self) -> int:
        return sum(t.est_bytes for t in self.tasks)

    def total_est_flops(self) -> int:
        return sum(t.est_flops for t in self.tasks)

    def gpu_name(self) -> str:
        return self.target.name if self.target else self.meta.get("gpu", "?")

    def summary(self) -> str:
        return (f"MegakernelProgram(model={self.meta.get('model', '?')}, gpu={self.gpu_name()}, "
                f"tasks={len(self.tasks)}, buffers={len(self.buffers)}, "
                f"counters={len(self.counters)}, weight_MB={self.total_weight_bytes() / 1e6:.1f})")

    # ---- serialization (the standard on-disk format) -------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "ir_version": self.ir_version,
            "abi_version": self.abi_version,
            "meta": self.meta,
            "target": asdict(self.target) if self.target else None,
            "buffers": [
                {"id": b.id, "name": b.name, "kind": b.kind.name, "dtype": b.dtype.name,
                 "shape": list(b.shape), "space": b.space.name, "source": b.source}
                for b in self.buffers
            ],
            "counters": [asdict(c) for c in self.counters],
            "tasks": [
                {"id": t.id, "op": t.op.name, "inputs": t.inputs, "outputs": t.outputs,
                 "out_counter": t.out_counter,
                 "waits": [{"counter": w.counter, "threshold": w.threshold} for w in t.waits],
                 "params": t.params, "sm": t.sm, "est_bytes": t.est_bytes,
                 "est_flops": t.est_flops, "label": t.label}
                for t in self.tasks
            ],
            "pages": (
                {"buffer_to_page": {str(k): v for k, v in self.pages.buffer_to_page.items()},
                 "pages": [{"id": p.id, "space": p.space.name, "nbytes": p.nbytes,
                            "live_start": p.live_start, "live_end": p.live_end}
                           for p in self.pages.pages]}
                if self.pages else None
            ),
            "config": self.config.to_dict() if self.config else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @staticmethod
    def _filter_known(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Additive-compatibility: drop keys a newer writer added that this dataclass lacks."""
        known = {f.name for f in dataclasses.fields(cls)}
        return {k: v for k, v in d.items() if k in known}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MegakernelProgram:
        ver = str(d.get("ir_version", IR_VERSION))
        if ver.split(".")[0] != IR_VERSION.split(".")[0]:
            raise ValueError(f"IR major version mismatch: file {ver} vs runtime {IR_VERSION}")
        abiver = str(d.get("abi_version", ABI_VERSION))
        if abiver.split(".")[0] != ABI_VERSION.split(".")[0]:
            raise ValueError(f"ABI major version mismatch: file {abiver} vs runtime {ABI_VERSION}")
        target = (GpuTarget(**cls._filter_known(GpuTarget, d["target"]))
                  if d.get("target") else None)
        buffers = [
            Buffer(id=b["id"], name=b["name"], kind=BufferKind[b["kind"]], dtype=DType[b["dtype"]],
                   shape=tuple(b["shape"]), space=MemSpace[b["space"]], source=b.get("source"))
            for b in d.get("buffers", [])
        ]
        counters = [Counter(**cls._filter_known(Counter, c)) for c in d.get("counters", [])]
        tasks = [
            Task(id=t["id"], op=InstructionKind[t["op"]], inputs=list(t["inputs"]),
                 outputs=list(t["outputs"]), out_counter=t["out_counter"],
                 waits=[Wait(w["counter"], w["threshold"]) for w in t.get("waits", [])],
                 params=t.get("params", {}), sm=t.get("sm"),
                 est_bytes=t.get("est_bytes", 0), est_flops=t.get("est_flops", 0),
                 label=t.get("label", ""))
            for t in d.get("tasks", [])
        ]
        pages = None
        if d.get("pages"):
            pages = PageAllocation(
                buffer_to_page={int(k): v for k, v in d["pages"]["buffer_to_page"].items()},
                pages=[Page(id=p["id"], space=MemSpace[p["space"]], nbytes=p["nbytes"],
                            live_start=p.get("live_start", -1), live_end=p.get("live_end", -1))
                       for p in d["pages"]["pages"]])
        config = None
        if d.get("config"):
            c = cls._filter_known(ScheduleConfig, dict(d["config"]))
            sa = c.get("sm_assignment")
            if isinstance(sa, dict):
                c["sm_assignment"] = {int(k): v for k, v in sa.items()}
            config = ScheduleConfig(**c)
        return cls(meta=d.get("meta", {}), target=target, buffers=buffers, counters=counters,
                   tasks=tasks, pages=pages, config=config, ir_version=ver,
                   abi_version=str(d.get("abi_version", ABI_VERSION)))

    @classmethod
    def from_json(cls, s: str) -> MegakernelProgram:
        return cls.from_dict(json.loads(s))

    @classmethod
    def load(cls, path: str) -> MegakernelProgram:
        with open(path, encoding="utf-8") as f:
            return cls.from_json(f.read())


# ======================================================================================
# VALIDATION, the load-bearing deadlock- AND race-freedom proof
# ======================================================================================
@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok

    def report(self) -> str:
        head = "VALID" if self.ok else "REJECTED"
        lines = [f"[{head}] {self.stats}"]
        lines += [f"  ERROR: {e}" for e in self.errors]
        lines += [f"  warn:  {w}" for w in self.warnings]
        return "\n".join(lines)


def validate(prog: MegakernelProgram) -> ValidationResult:  # noqa: C901
    """Statically prove the program is safe to load. A ``REJECTED`` result MUST prevent launch.
    Never raises on a malformed program, always returns a result (the clean-signal contract).

    Checks: referential integrity + arity + required params; ABI fixed-capacity limits; wait
    satisfiability (1<=t<=#producers); shared-counter true-join rule (race-freedom); acyclicity;
    transitive happens-before provenance for every activation/KV read (race-freedom); per-SM
    queue ordering when SMs are assigned (deadlock-freedom for serial queues); page-aliasing
    ordering when an allocation is present; output reachability; param type sanity; GPU labeling.
    """
    errors: list[str] = []
    warnings: list[str] = []

    buf_ids = {b.id for b in prog.buffers}
    ctr_ids = {c.id for c in prog.counters}
    producers = prog.producers_by_counter()

    # ---- 1. referential integrity + arity + required params + ABI caps + param types ----
    for t in prog.tasks:
        for bid in t.inputs:
            if bid not in buf_ids:
                errors.append(f"task {t.id} ({t.label}) reads missing buffer {bid}")
        for bid in t.outputs:
            if bid not in buf_ids:
                errors.append(f"task {t.id} ({t.label}) writes missing buffer {bid}")
        if t.out_counter not in ctr_ids:
            errors.append(f"task {t.id} ({t.label}) increments missing counter {t.out_counter}")
        if len(t.inputs) > ABI_MAX_INPUTS:
            errors.append(f"task {t.id} has {len(t.inputs)} inputs > ABI_MAX_INPUTS={ABI_MAX_INPUTS}")
        if len(t.outputs) > ABI_MAX_OUTPUTS:
            errors.append(f"task {t.id} has {len(t.outputs)} outputs > ABI_MAX_OUTPUTS={ABI_MAX_OUTPUTS}")
        if len(t.waits) > ABI_MAX_WAITS:
            errors.append(f"task {t.id} has {len(t.waits)} waits > ABI_MAX_WAITS={ABI_MAX_WAITS}")
        for b in (*t.inputs, *t.outputs):
            if b in buf_ids and len(prog.buffers[b].shape) > ABI_MAX_RANK:
                errors.append(f"task {t.id} buffer {b} rank {len(prog.buffers[b].shape)} "
                              f"> ABI_MAX_RANK={ABI_MAX_RANK}")
        for w in t.waits:
            if w.counter not in ctr_ids:
                errors.append(f"task {t.id} ({t.label}) waits on missing counter {w.counter}")
            if not isinstance(w.threshold, int) or isinstance(w.threshold, bool) or w.threshold < 1:
                errors.append(f"task {t.id} ({t.label}) threshold {w.threshold!r} on counter "
                              f"{w.counter} is not a positive static int")
        if t.op in UNIMPLEMENTED_OPS:
            errors.append(f"task {t.id} uses opcode {t.op.name} which has no reference oracle yet "
                          f"(unimplemented), cannot verify, rejected")
        for w in t.waits:
            if w.counter == t.out_counter:
                errors.append(f"task {t.id} ({t.label}) waits on its OWN out_counter "
                              f"{t.out_counter}, self-deadlock")
        spec = OP_REGISTRY.get(t.op)
        if spec is None:
            errors.append(f"task {t.id} uses unregistered opcode {t.op}")
        else:
            n_in = len(t.inputs)
            if n_in < spec.min_inputs or (spec.max_inputs != -1 and n_in > spec.max_inputs):
                errors.append(f"task {t.id} ({t.op.name}) arity: {n_in} inputs, expected "
                              f"[{spec.min_inputs},{spec.max_inputs}]")
            if spec.n_outputs != -1 and len(t.outputs) != spec.n_outputs:
                errors.append(f"task {t.id} ({t.op.name}) expected {spec.n_outputs} outputs, "
                              f"got {len(t.outputs)}")
            for k in spec.required_params:
                if k not in t.params:
                    errors.append(f"task {t.id} ({t.op.name}) missing required param '{k}'")
        # param type/key sanity (IR<->amk_params_t marshalling)
        for k, v in t.params.items():
            kind = PARAM_FIELDS.get(k)
            if kind is None:
                warnings.append(f"task {t.id} ({t.op.name}) param '{k}' is not a known "
                                f"amk_params_t field (will not marshal to the device)")
            elif kind == "i" and (not isinstance(v, int) or isinstance(v, bool)
                                  or not (_INT32_MIN <= v <= _INT32_MAX)):
                errors.append(f"task {t.id} param '{k}'={v!r} must be an int32")
            elif kind == "f" and not isinstance(v, (int, float)):
                errors.append(f"task {t.id} param '{k}'={v!r} must be a real number")

    # ---- 2. wait satisfiability + 3. coverage + race-freedom (shared-counter true-join) ----
    for t in prog.tasks:
        for w in t.waits:
            np = len(producers.get(w.counter, ()))
            if np == 0:
                errors.append(f"task {t.id} ({t.label}) waits on counter {w.counter} with NO "
                              f"producer, unsatisfiable (deadlock)")
            elif isinstance(w.threshold, int):
                if w.threshold > np:
                    errors.append(f"task {t.id} ({t.label}) threshold {w.threshold} > {np} "
                                  f"producers of counter {w.counter}, unsatisfiable (deadlock)")
                elif np > 1 and w.threshold != np:
                    # A counter only counts; a partial wait on a multi-producer counter is a
                    # "first-k-of-N" race (the wrong producers can satisfy it).
                    errors.append(f"task {t.id} ({t.label}) partial wait threshold {w.threshold} "
                                  f"on shared counter {w.counter} ({np} producers), ambiguous "
                                  f"which-producer RACE; a shared counter must be an all-join "
                                  f"(threshold == {np})")

    # ---- 4. acyclicity ----
    adj, indeg = prog._adjacency()
    order = prog.topological_order(adj, indeg)
    if order is None:
        errors.append("producer→consumer graph has a CYCLE, structural deadlock. "
                      + _describe_cycle(prog, adj))

    # ---- 5. transitive happens-before provenance (race-freedom) ----
    # anc[t] = bitmask of task ids that are transitive PREDECESSORS of t. A read of a written
    # buffer B is race-free iff EVERY writer of B (other than t itself) is in anc[t], i.e. ALL
    # producers of B are ordered before the read. (Buffer-level "some writer wrote it" is NOT
    # enough: a partial multi-writer read would slip through.)
    anc: dict[int, int] = {}
    edges = prog.dependency_edges()
    writers = prog.writers_by_buffer()
    if order is not None and len(prog.tasks) <= _PROVENANCE_MAX_TASKS:
        pred: dict[int, list[int]] = {t.id: [] for t in prog.tasks}
        for a, b in edges:
            pred[b].append(a)
        for tid in order:
            m = 0
            for p in pred[tid]:
                m |= anc[p] | (1 << p)
            anc[tid] = m
            t = prog.tasks[tid]
            for b in t.inputs:
                if b not in buf_ids:
                    continue
                bk = prog.buffers[b].kind
                if bk in _READONLY_KINDS:
                    continue
                if bk == BufferKind.KV_CACHE and b not in writers:
                    continue  # pure prior-step state; safe to read
                ws = writers.get(b, [])
                others = [w for w in ws if w != tid]
                if not ws:
                    errors.append(f"task {tid} ({t.label}) reads buffer #{b} "
                                  f"('{prog.buffers[b].name}') that is never produced")
                    continue
                missing = [w for w in others if not (m >> w) & 1]
                if missing:
                    word = "KV cache" if bk == BufferKind.KV_CACHE else "buffer"
                    errors.append(f"task {tid} ({t.label}) reads {word} #{b} "
                                  f"('{prog.buffers[b].name}') but writer task(s) {missing} are "
                                  f"NOT ordered before it, data RACE")
                elif not others and tid in ws and bk != BufferKind.KV_CACHE:
                    warnings.append(f"task {tid} ({t.label}) reads buffer #{b} "
                                    f"('{prog.buffers[b].name}') in place with no external "
                                    f"producer (reads its own uninitialized output?)")

        # ---- 5a. multi-writer WAW race-freedom ----
        # RAW (above) orders writers before READERS, but two tasks WRITING the same buffer with no
        # happens-before edge and overlapping footprints are a write-after-write race. For every
        # buffer with >1 writer require EITHER (a) all writers share one out_counter AND every pair
        # of written regions is provably DISJOINT (GEMV/GEMM tiles via n_off+N_tile / m_off+M_tile;
        # any other writer covers the whole buffer), OR (b) a TOTAL happens-before order among the
        # writers (each later writer has all earlier writers in its ancestor bitmask anc[]). This is
        # exactly the disjoint-tiled-under-one-counter pattern the lowerer emits, that stays VALID.
        for b, ws in writers.items():
            if len(ws) < 2 or b not in buf_ids:
                continue
            bk = prog.buffers[b].kind
            if bk in _READONLY_KINDS:
                continue  # external inputs/weights/consts are not written by the schedule
            # (a) disjoint tiles under ONE shared counter.
            one_counter = len({prog.tasks[w].out_counter for w in ws}) == 1
            regions = {w: _write_region(prog.tasks[w]) for w in ws}
            pairwise_disjoint = one_counter and all(
                _regions_disjoint(regions[ws[i]], regions[ws[j]])
                for i in range(len(ws)) for j in range(i + 1, len(ws)))
            if pairwise_disjoint:
                continue
            # (b) total happens-before order: some writer is an ancestor of / equal-ordered against
            # every other (each pair is comparable, one strictly before the other).
            def _before(u: int, v: int) -> bool:  # noqa: B023 (b/ws are loop-stable here)
                return bool((anc[v] >> u) & 1)
            total_order = all(
                _before(ws[i], ws[j]) or _before(ws[j], ws[i])
                for i in range(len(ws)) for j in range(i + 1, len(ws)))
            if total_order:
                continue
            # Neither proof holds: report the offending writer pair concretely.
            culprits = next(
                ((ws[i], ws[j]) for i in range(len(ws)) for j in range(i + 1, len(ws))
                 if not _regions_disjoint(regions[ws[i]], regions[ws[j]])
                 and not _before(ws[i], ws[j]) and not _before(ws[j], ws[i])),
                (ws[0], ws[1]))
            ti, tj = prog.tasks[culprits[0]], prog.tasks[culprits[1]]
            reason = ("share no counter; " if not one_counter
                      else "share a counter but their tiles OVERLAP; ")
            errors.append(
                f"multi-writer WAW race on buffer #{b} ('{prog.buffers[b].name}'): writers "
                f"{culprits[0]} ('{ti.label}') and {culprits[1]} ('{tj.label}') {reason}"
                f"neither is ordered happens-before the other, write-after-write RACE")
    elif order is not None:
        warnings.append(
            f"transitive RAW-provenance and write-after-write (WAW) race checks SKIPPED: "
            f"{len(prog.tasks)} tasks > {_PROVENANCE_MAX_TASKS} limit. "
            f"Deadlock-freedom is still proven (per-SM queue ordering verified above); "
            f"only the cross-task data-hazard checks are capped at this task count."
        )

    # ---- 5b. on-chip (SMEM/REGISTER) buffers must be single-SM (counters ≠ data movement) ----
    for b in prog.buffers:
        if b.space in _ONCHIP_SPACES:
            users = [t for t in prog.tasks if b.id in t.inputs or b.id in t.outputs]
            sms = {t.sm for t in users if t.sm is not None}
            if len(sms) > 1:
                errors.append(f"on-chip buffer '{b.name}'(#{b.id}, {b.space.name}) is used across "
                              f"SMs {sorted(sms)}, counters give ordering, not cross-SM data; "
                              f"promote to GLOBAL_SCRATCH/HBM or co-locate on one SM")
            elif users and not sms:
                warnings.append(f"on-chip buffer '{b.name}'(#{b.id}) requires same-SM placement "
                                f"(unverified until sm_assignment runs)")

    # ---- 6. per-SM queue ordering (full deadlock-freedom once SMs assigned) ----
    if any(t.sm is not None for t in prog.tasks):
        pos = {t.id: i for i, t in enumerate(prog.tasks)}  # loader queue = task-list order per SM
        for a, b in edges:
            sa, sb = prog.tasks[a].sm, prog.tasks[b].sm
            if sa is not None and sa == sb and pos[a] >= pos[b]:
                errors.append(f"SM {sa} queue order violates dependency {a}->{b} "
                              f"(producer must precede consumer in the same SM's serial queue)")
        for t in prog.tasks:
            if t.sm is not None and prog.target and not (0 <= t.sm < prog.target.num_sms):
                errors.append(f"task {t.id} assigned to SM {t.sm} out of range "
                              f"[0,{prog.target.num_sms})")

    # ---- 7. page-aliasing ordering (ERROR; sound full live-range separation) ----
    # Two buffers sharing a physical page are safe ONLY if one's entire live range (all reads and
    # writes) happens-before the other's first write. Anything else is a clobber race.
    if prog.pages is not None and order is not None and anc:
        page_of = prog.pages.buffer_to_page
        occupants: dict[int, list[int]] = {}
        for bid, pid in page_of.items():
            occupants.setdefault(pid, []).append(bid)
        uses: dict[int, list[int]] = {}
        for t in prog.tasks:
            for b in (*t.inputs, *t.outputs):
                uses.setdefault(b, []).append(t.id)

        def _before(x: int, y: int) -> bool:        # x happens-before y
            return bool((anc.get(y, 0) >> x) & 1)

        def _all_before(buf_a: int, buf_b: int) -> bool:  # every use of a precedes every write of b
            return all(_before(u, w) for w in writers.get(buf_b, []) for u in uses.get(buf_a, []))

        for pid, occ in occupants.items():
            for i in range(len(occ)):
                for j in range(i + 1, len(occ)):
                    a_buf, b_buf = occ[i], occ[j]
                    if not (_all_before(a_buf, b_buf) or _all_before(b_buf, a_buf)):
                        errors.append(f"page {pid} reused by buffers #{a_buf} ('"
                                      f"{prog.buffers[a_buf].name}') and #{b_buf} ('"
                                      f"{prog.buffers[b_buf].name}') whose live ranges are NOT "
                                      f"ordered, clobber RACE (page reuse needs separated ranges)")

    # ---- 8. output reachability + GPU labeling ----
    produced = {bid for t in prog.tasks for bid in t.outputs}
    for b in prog.buffers:
        if b.kind == BufferKind.IO_OUTPUT and b.id not in produced:
            errors.append(f"IO_OUTPUT buffer '{b.name}'(#{b.id}) is never produced")
    if prog.target and prog.meta.get("gpu") and prog.meta["gpu"] != prog.target.name:
        warnings.append(f"meta['gpu']={prog.meta['gpu']!r} != target.name={prog.target.name!r} "
                        f"- flywheel rows derive GPU from target.name")

    stats = {
        "tasks": len(prog.tasks), "buffers": len(prog.buffers), "counters": len(prog.counters),
        "edges": sum(len(v) for v in adj.values()), "acyclic": order is not None,
        "weight_MB": round(prog.total_weight_bytes() / 1e6, 2),
        "sm_assigned": any(t.sm is not None for t in prog.tasks),
    }
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, stats=stats)


def _describe_cycle(prog: MegakernelProgram, adj: dict[int, list[int]]) -> str:
    """Best-effort cycle witness using an ITERATIVE (stack-safe) DFS so a huge cyclic schedule
    can never crash the validator with RecursionError."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {t.id: WHITE for t in prog.tasks}
    for root in (t.id for t in prog.tasks):
        if color[root] != WHITE:
            continue
        stack = [(root, iter(adj[root]))]
        path = [root]
        color[root] = GRAY
        while stack:
            node, it = stack[-1]
            advanced = False
            for v in it:
                if color[v] == GRAY:
                    i = path.index(v)
                    cyc = path[i:] + [v]
                    labels = [f"{x}:{prog.tasks[x].label or prog.tasks[x].op.name}" for x in cyc]
                    return "cycle: " + " -> ".join(labels)
                if color[v] == WHITE:
                    color[v] = GRAY
                    stack.append((v, iter(adj[v])))
                    path.append(v)
                    advanced = True
                    break
            if not advanced:
                color[node] = BLACK
                stack.pop()
                path.pop()
    return "cycle: <not localized>"


__all__ = [
    "IR_VERSION", "ABI_VERSION", "ABI_MAX_INPUTS", "ABI_MAX_OUTPUTS", "ABI_MAX_WAITS",
    "ABI_MAX_RANK", "DType", "MemSpace", "BufferKind", "InstructionKind", "OpSpec", "OP_REGISTRY",
    "PARAM_FIELDS", "Buffer", "Wait", "Task", "Counter", "Page", "PageAllocation",
    "ScheduleConfig", "GpuTarget", "TARGETS", "MegakernelProgram", "ValidationResult",
    "validate", "replace",
]
