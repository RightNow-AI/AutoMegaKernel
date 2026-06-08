"""
AMK, GRAPH LOWERER (Layer 2, front half)
=========================================

``lower(graph, target, config) -> MegakernelProgram`` turns a :class:`schedule.graph.ModelGraph`
into a runnable (after :func:`schedule.ir.validate`) megakernel for a **single decode step**:
one token at position ``pos`` against a persistent KV cache. The emitted task-DAG is the whole
forward pass; the host loop relaunches it per token (KV persists in HBM).

WHAT IT EMITS (per Llama layer):
    RMSNORM -> q/k/v LINEAR (tiled GEMV) -> reshape -> ROPE(q), ROPE(k)
            -> KV_APPEND(k), KV_APPEND(v) -> ATTENTION over the whole window (kv_len = pos+1)
            -> o_proj (tiled GEMV) -> residual ADD -> post RMSNORM
            -> gate/up (tiled GEMV) -> SILU_MUL -> down (tiled GEMV) -> residual ADD
then final RMSNORM -> lm_head (tiled GEMV) -> logits.

THE SHAPE BRIDGE (the load-bearing design choice, read this):
The reference GEMV writes a *flat* ``[1, dim]`` output (column-tiled), but RoPE and ATTENTION
require *head-shaped* tensors (``[n_heads, head_dim]`` for q, ``[1, n_kv, head_dim]`` for k).
The frozen op set has no generic reshape op, and COPY/ADD cannot reshape. The one primitive that
reshapes a transient via ``view_as`` is EMBED (``table.index_select(0, ids).view_as(out)``). We
therefore reshape ``flat -> head-shaped`` with an EMBED whose ``ids = [0]`` selects row 0 of the
flat ``[1, dim]`` activation and views it into the head shape, a correct, validator-clean,
zero-FLOP gather. (V needs no reshape: KV_APPEND already reshapes flat ``[1, kv_dim]`` into the
cache via its own ``view``.) The constant ``[0]`` id is wired as a single IO_INPUT
(:data:`RESHAPE_ID_NAME`) the caller supplies once per run. This is the only "extra" input.

COUNTER WIRING (per the locked invariants in schedule.ir):
  * Every fully-written buffer gets exactly one counter; the producer increments it once.
  * Tiled GEMV producers SHARE one counter; every consumer ALL-JOINs (threshold == #tiles).
    Never a partial threshold on a shared counter (that is a which-producer race -> REJECTED).
  * ATTENTION waits on rope(q) AND both KV_APPENDs (the KV happens-before the validator demands).
  * The DAG is acyclic and every activation/KV read has a transitive producer edge.

COST ANNOTATION: ``est_bytes`` is dominated by weights (decode is bandwidth-bound), so a GEMV
tile bills ``K * N_tile * dtype_bytes``; ``est_flops`` bills the matvec ``2*K*N_tile``. The cost
model / roofline read these.

SEARCHABILITY: the only knob consumed here is ``config.tiling`` (GEMV ``N_tile`` and the cap on
tiles per projection). The same graph re-lowers under any :class:`ScheduleConfig`; SM assignment,
pages, and pipelining are downstream passes (left ``None``/unset so later layers fill them).
"""
from __future__ import annotations

from typing import Any

from schedule.graph import ModelGraph, Node
from schedule.ir import (
    BufferKind, DType, GpuTarget, InstructionKind, MegakernelProgram, ScheduleConfig, Wait,
)

# Name of the constant-zero id IO_INPUT used by the EMBED-reshape bridge (see module docstring).
RESHAPE_ID_NAME = "reshape_id0"
# Name of the position IO_INPUT (this step's absolute position, length-1 int tensor).
POS_NAME = "pos"
# Name of the token-id IO_INPUT.
TOKEN_NAME = "token_id"

_DEFAULT_N_TILE = 128          # legacy default tile width (pre-auto-sizing; kept for reference)
_MAX_TILES = 64                # safety cap on tiles per projection (keeps task counts sane)

# ---- GEMV tile-width auto-sizing (the MEASURED decode-bandwidth win) --------------------------
# Decode GEMV plateaued at ~50% of MEASURED HBM bandwidth with the old N_tile=128 default (640
# tiles), while a standalone cuBLAS gemv of the same 623 MB hits ~74% (this machine, measured).
# The original hypothesis was PER-TILE FIXED OVERHEAD (x-cache + 2 syncs + dispatch over ~640
# tiles) -> "fewer, fatter tiles". A real CUDA-event sweep on the RTX 5090 (eval/bench_fat_tile_gemv
# .py) REFUTED that: fatter tiles are monotonically SLOWER, thinner tiles are FASTER. The binding
# constraint is PARALLELISM / SM LOAD-BALANCE + memory-level parallelism, not per-tile overhead:
# each GEMV tile is one schedulable unit greedily balanced over the 82 SMs (vm/loader LPT), and the
# cp.async kernel needs many independent in-flight weight streams to hide ~400 ns HBM latency. More,
# thinner tiles == more independent units == better balance + more MLP == higher achieved bandwidth,
# until tiles get so thin (~N_tile<=16) that fixed overhead finally bites. Measured sweet spot on
# this model is N_tile ~= 32 (1472 tiles): 368 GB/s/50% -> 449 GB/s/61.5% of the 731 GB/s roofline,
# a real, correctness-preserved 1.22x. The SMEM budget is INDEPENDENT of N_tile (x-cache sized by K,
# cp.async ring by warps/stages), so this costs no extra SMEM and never reduces blocks/SM.
_DEFAULT_N_TILE_AUTO = 32      # measured parallelism-optimal default tile width (was 128)
_MIN_AUTO_TILE = 16            # below this, per-tile overhead starts to regress the bandwidth
_FALLBACK_NUM_SMS = 82         # SM count used when no target is known (this dev machine; RTX 5090)


def _auto_n_tile(N: int, base_width: int) -> int:
    """Auto-pick the GEMV tile width when the config supplies no explicit ``N_tile``.

    The measured win (see the module notes above) is THIN tiles for parallelism + SM load-balance,
    so the default is a small ``base_width`` (~32). We clamp to ``[_MIN_AUTO_TILE, N]``: never go so
    thin that fixed overhead dominates, and never exceed the projection width."""
    width = max(_MIN_AUTO_TILE, base_width)
    return max(1, min(width, N))


def _gemv_n_tile(config: ScheduleConfig | None, N: int, num_sms: int = _FALLBACK_NUM_SMS) -> int:
    """Pick the GEMV output-column tile width.

    Search surface (``config.tiling['gemv']``):
      * ``N_tile``    , explicit tile width; honored (clamped) so a search can sweep it and the
                         tile count moves monotonically (smaller N_tile => more tiles).
      * ``base_width``, when ``N_tile`` is absent, the target width for the auto-sizer (default
                         ``_DEFAULT_N_TILE_AUTO`` == the measured parallelism-optimal ~32). Thinner
                         tiles give more independent, load-balanceable units; see :func:`_auto_n_tile`.

    ``num_sms`` is accepted for ABI stability (callers may pass the target SM count); the measured
    optimum is width-driven so it is not used by the auto-sizer.

    Always clamped to ``_MAX_TILES`` tiles per projection and a non-zero width."""
    gemv_cfg = config.tiling.get("gemv", {}) if config is not None else {}
    if "N_tile" in gemv_cfg:
        n_tile = max(1, min(int(gemv_cfg["N_tile"]), N))
    else:
        base_width = int(gemv_cfg.get("base_width", _DEFAULT_N_TILE_AUTO))
        n_tile = _auto_n_tile(N, base_width)
    # Respect the tile-count cap: widen the tile if we'd otherwise exceed it.
    if (N + n_tile - 1) // n_tile > _MAX_TILES:
        n_tile = (N + _MAX_TILES - 1) // _MAX_TILES
    return max(1, n_tile)


class _Lowerer:
    """Stateful builder. One instance lowers one graph; holds the symbolic-tensor -> buffer-id
    map and the per-tensor 'ready counter' (the counter a consumer waits on to read that tensor,
    with the matching threshold)."""

    def __init__(self, graph: ModelGraph, target: GpuTarget | None, config: ScheduleConfig | None,
                 pos: int, dtype: DType, quant: Any = None):
        self.g = graph
        self.cfg = graph.config
        self.config = config
        # SM count is available to the GEMV tile-width auto-sizer (measured optimum is width-driven).
        self.num_sms = int(target.num_sms) if target is not None else _FALLBACK_NUM_SMS
        self.pos = pos
        self.kv_len = pos + 1
        self.dtype = dtype
        self.dbytes = max(1, dtype.bits // 8)
        # Optional weight-only quantization metadata (schedule.quantize.QuantMeta). When present,
        # projection GEMVs whose weight key is in quant.keys are lowered as dequant-fused quantized
        # GEMV tiles (packed int4/int8 weight buffer + fp16 scales [+ zeros]); everything else and
        # the whole non-quantized path is byte-identical to before.
        self.quant = quant
        self.p = MegakernelProgram(
            meta={"model": graph.meta.get("source", "model"),
                  "gpu": target.name if target else "?",
                  "regime": "decode", "pos": pos, "kv_len": self.kv_len,
                  "dtype": dtype.name, "n_layers": self.cfg.n_layers},
            target=target, config=config)
        # tensor name -> buffer id
        self.buf: dict[str, int] = {}
        # tensor name -> (counter_id, threshold) that signals "this tensor is fully written"
        self.ready: dict[str, tuple[int, int]] = {}
        # weight key -> buffer id (dedup; each weight emitted once)
        self.wbuf: dict[str, int] = {}

    # ---- buffer / counter helpers --------------------------------------------------
    def _new(self, name: str, kind: BufferKind, shape: tuple[int, ...],
             dt: DType | None = None, source: str | None = None) -> int:
        return self.p.new_buffer(name, kind, dt or self.dtype, tuple(shape), source=source).id

    def _weight(self, key: str) -> int:
        if key not in self.wbuf:
            w = self.g.weights[key]
            self.wbuf[key] = self._new(key, BufferKind.WEIGHT, w.shape, source=key)
        return self.wbuf[key]

    def _is_quant_weight(self, key: str) -> bool:
        return self.quant is not None and self.quant.is_quantized(key)

    def _quant_weight_buffers(self, key: str) -> tuple[int, int, int | None]:
        """Emit (and dedup) the packed-int4/int8 weight buffer, the fp16 scales buffer, and the
        optional fp16 zeros buffer for a quantized projection ``key``. Returns their buffer ids.

        The packed weight buffer is dtype I4/I8 with the LOGICAL ``[N, K]`` shape (the GPU kernel
        reads ``W.shape[0]`` rows and unpacks K columns from the packed storage that the loader
        binds). Its ``source`` is ``key``; scales/zeros source the suffixed keys the quantizer
        produced (``key.scales`` / ``key.zeros``)."""
        info = self.quant.keys[key]
        N, K, n_groups = info["N"], info["K"], info["n_groups"]
        qdt = DType.I4 if self.quant.bits == 4 else DType.I8
        wkey = f"{key}::q"
        if wkey not in self.wbuf:
            wb = self._new(key, BufferKind.WEIGHT, (N, K), dt=qdt, source=key)
            sb = self._new(key + ".scales", BufferKind.WEIGHT, (N, n_groups),
                           dt=DType.F16, source=key + ".scales")
            zb = None
            if info.get("has_zeros"):
                zb = self._new(key + ".zeros", BufferKind.WEIGHT, (N, n_groups),
                               dt=DType.F16, source=key + ".zeros")
            self.wbuf[wkey] = (wb, sb, zb)
        return self.wbuf[wkey]

    def _quant_weight_bytes(self, key: str) -> int:
        """HBM bytes of the quantized weight tile per output column (packed weight + its scale),
        for the cost model / roofline. int4 = K/2 bytes + 2-byte scale per group."""
        info = self.quant.keys[key]
        K, n_groups = info["K"], info["n_groups"]
        wbytes = (K * self.quant.bits + 7) // 8
        return wbytes + n_groups * 2  # fp16 scale per group (+zeros ignored; tiny)

    def _counter(self, note: str) -> int:
        return self.p.new_counter(note).id

    def _waits_for(self, *tensors: str) -> list[Wait]:
        """Build the wait list to read the given input tensors. Read-only externals (weights,
        token id, pos, reshape id) have no ready counter and contribute no wait."""
        waits: list[Wait] = []
        for t in tensors:
            rc = self.ready.get(t)
            if rc is not None:
                waits.append(Wait(rc[0], rc[1]))
        return waits

    # ---- IO / constant buffers -----------------------------------------------------
    def _ensure_io(self) -> None:
        H = self.cfg.hidden
        self.buf[TOKEN_NAME] = self._new(TOKEN_NAME, BufferKind.IO_INPUT, (1,), dt=DType.I32)
        self.buf[POS_NAME] = self._new(POS_NAME, BufferKind.IO_INPUT, (1,), dt=DType.I32)
        self.buf[RESHAPE_ID_NAME] = self._new(RESHAPE_ID_NAME, BufferKind.IO_INPUT, (1,), dt=DType.I32)
        # logits IO_OUTPUT is created when we lower the lm_head node.
        _ = H

    # ---- tiled GEMV ----------------------------------------------------------------
    def _gemv_tiles(self, x: str, ob: int, weight_key: str, K: int, N: int, label: str) -> int:
        """Emit the column-tiled GEMV tasks for ``out = x @ W.T`` into the already-created output
        buffer ``ob``, reading input tensor ``x``, sharing one counter. Returns (counter, #tiles).
        Picks the QUANTIZED form (inputs = [x, qW, scales(, zeros)], params qdtype+group) when
        ``weight_key`` is a quantized projection, else the fp form (inputs = [x, W])."""
        xb = self.buf[x]
        n_tile = _gemv_n_tile(self.config, N, self.num_sms)
        n_tiles = (N + n_tile - 1) // n_tile
        ctr = self._counter(f"{label} ({n_tiles} tiles)")
        wait = self._waits_for(x)
        quant = self._is_quant_weight(weight_key)
        if quant:
            wb, sb, zb = self._quant_weight_buffers(weight_key)
            ins = [xb, wb, sb] + ([zb] if zb is not None else [])
            qdtype = int(DType.I4 if self.quant.bits == 4 else DType.I8)
            group = int(self.quant.group)
            per_col_bytes = self._quant_weight_bytes(weight_key)
        else:
            wb = self._weight(weight_key)
            ins = [xb, wb]
            qdtype = group = None
            per_col_bytes = K * self.dbytes
        emitted = 0
        for i in range(n_tiles):
            n_off = i * n_tile
            this = min(n_tile, N - n_off)
            if this <= 0:
                break
            params = {"K": K, "N_tile": this, "n_off": n_off}
            if quant:
                params["qdtype"] = qdtype
                params["group"] = group
            self.p.add_task(
                InstructionKind.GEMV_TILE, list(ins), [ob], out_counter=ctr, waits=list(wait),
                params=params, label=f"{label}[t{i}]",
                est_bytes=per_col_bytes * this, est_flops=2 * K * this)
            emitted += 1
        return ctr, emitted

    def _tiled_linear(self, x: str, weight_key: str, out: str, K: int, N: int,
                      out_shape: tuple[int, ...], label: str) -> None:
        """Emit a column-tiled GEMV for ``out = x @ W.T`` (W is [N, K]). All tiles share one
        counter; the consumer ALL-JOINs on (counter, #tiles). Registers ``out``'s ready edge."""
        ob = self._new(out, BufferKind.ACTIVATION, out_shape)
        self.buf[out] = ob
        ctr, emitted = self._gemv_tiles(x, ob, weight_key, K, N, label)
        self.ready[out] = (ctr, emitted)

    # ---- reshape (flat [1, dim] -> head-shaped) via EMBED view_as ------------------
    def _reshape(self, flat: str, out: str, out_shape: tuple[int, ...], head_dim: int,
                 label: str) -> None:
        ob = self._new(out, BufferKind.ACTIVATION, out_shape)
        self.buf[out] = ob
        ctr = self._counter(label)
        # EMBED(ids=[0], table=flat) -> flat[0:1].view_as(out). ids is read-only; table is the
        # transient flat activation (needs its producer edge).
        self.p.add_task(
            InstructionKind.EMBED, [self.buf[RESHAPE_ID_NAME], self.buf[flat]], [ob],
            out_counter=ctr, waits=self._waits_for(flat),
            params={"hidden": head_dim}, label=label, est_bytes=0, est_flops=0)
        self.ready[out] = (ctr, 1)

    # ---- node lowering -------------------------------------------------------------
    def lower(self) -> MegakernelProgram:
        self._ensure_io()
        for node in self.g.nodes:
            getattr(self, f"_op_{node.op}")(node)
        # Tag the on-disk meta with the weight byte total for the roofline.
        self.p.meta["weight_bytes"] = self.p.total_weight_bytes()
        if self.quant is not None:
            self.p.meta["quantized"] = True
            self.p.meta["quant"] = {"group": self.quant.group, "bits": self.quant.bits,
                                    "symmetric": self.quant.symmetric,
                                    "n_keys": len(self.quant.keys)}
        return self.p

    # -- embed --
    def _op_embed(self, n: Node) -> None:
        H = self.cfg.hidden
        out = n.outputs[0]
        ob = self._new(out, BufferKind.ACTIVATION, (1, H))
        self.buf[out] = ob
        ctr = self._counter("embed")
        table = self._weight(n.weights[0])
        self.p.add_task(
            InstructionKind.EMBED, [self.buf[TOKEN_NAME], table], [ob], out_counter=ctr,
            waits=[], params={"hidden": H}, label=n.name,
            est_bytes=H * self.dbytes, est_flops=0)
        self.ready[out] = (ctr, 1)

    # -- rmsnorm --
    def _op_rmsnorm(self, n: Node) -> None:
        H = int(n.attrs["hidden"])
        x = n.inputs[0]
        out = n.outputs[0]
        ob = self._new(out, BufferKind.ACTIVATION, (1, H))
        self.buf[out] = ob
        ctr = self._counter(n.name)
        w = self._weight(n.weights[0])
        self.p.add_task(
            InstructionKind.RMSNORM, [self.buf[x], w], [ob], out_counter=ctr,
            waits=self._waits_for(x),
            params={"eps": float(n.attrs["eps"]), "hidden": H}, label=n.name,
            est_bytes=H * self.dbytes, est_flops=4 * H)
        self.ready[out] = (ctr, 1)

    # -- linear (tiled GEMV) --
    def _op_linear(self, n: Node) -> None:
        K = int(n.attrs["K"])
        N = int(n.attrs["N"])
        x = n.inputs[0]
        out = n.outputs[0]
        kind = n.attrs.get("kind", "linear")
        # lm_head writes the IO_OUTPUT 'logits'; everything else is a flat activation.
        if kind == "lm_head":
            ob = self._new(out, BufferKind.IO_OUTPUT, (1, N))
            self.buf[out] = ob
            ctr, emitted = self._gemv_tiles(x, ob, n.weights[0], K, N, n.name)
            self.ready[out] = (ctr, emitted)
        else:
            self._tiled_linear(x, n.weights[0], out, K, N, (1, N), n.name)

    # -- rope --
    def _op_rope(self, n: Node) -> None:
        head_dim = int(n.attrs["head_dim"])
        theta = float(n.attrs["theta"])
        which = n.attrs["which"]
        n_h = int(n.attrs["n_heads"])
        flat = n.inputs[0]            # [1, n_h*head_dim] flat projection
        out = n.outputs[0]
        # Reshape flat -> head-shaped, then rope in place on the head shape.
        if which == "q":
            head_shape = (n_h, head_dim)          # rank-2 for ATTENTION
        else:                                     # k -> rank-3 [1, n_kv, head_dim] for KV_APPEND
            head_shape = (1, n_h, head_dim)
        reshaped = f"{out}.hs"
        self._reshape(flat, reshaped, head_shape, head_dim, f"{n.name}.reshape")
        ob = self._new(out, BufferKind.ACTIVATION, head_shape)
        self.buf[out] = ob
        ctr = self._counter(n.name)
        self.p.add_task(
            InstructionKind.ROPE, [self.buf[reshaped], self.buf[POS_NAME]], [ob], out_counter=ctr,
            waits=self._waits_for(reshaped),
            params={"head_dim": head_dim, "theta": theta}, label=n.name,
            est_bytes=0, est_flops=6 * n_h * head_dim)
        self.ready[out] = (ctr, 1)

    # -- kv_append --
    def _op_kv_append(self, n: Node) -> None:
        n_kv = int(n.attrs["n_kv_heads"])
        head_dim = int(n.attrs["head_dim"])
        src = n.inputs[0]            # k: roped [1, n_kv, head_dim]; v: flat [1, kv_dim]
        cache_name = n.outputs[0]
        cache = self._new(cache_name, BufferKind.KV_CACHE, (self.cfg.max_seq, n_kv, head_dim))
        self.buf[cache_name] = cache
        ctr = self._counter(n.name)
        # KV_APPEND inputs = [new_kv, cache]; output = cache (in place). It waits on the source's
        # producer; the cache itself is prior-step state (no edge needed to read it here).
        self.p.add_task(
            InstructionKind.KV_APPEND, [self.buf[src], cache], [cache], out_counter=ctr,
            waits=self._waits_for(src), params={"pos": self.pos}, label=n.name,
            est_bytes=n_kv * head_dim * self.dbytes, est_flops=0)
        self.ready[cache_name] = (ctr, 1)

    # -- attention (flash-decoding: split the KV window across SMs when it is long enough) --
    def _op_attention(self, n: Node) -> None:
        head_dim = int(n.attrs["head_dim"])
        n_heads = int(n.attrs["n_heads"])
        n_kv = int(n.attrs["n_kv_heads"])
        scale = float(n.attrs["scale"])
        q, kc, vc = n.inputs           # q=[n_heads,head_dim], kc/vc = caches
        out = n.outputs[0]
        qd = n_heads * head_dim
        # ATTENTION reads q (after rope) and BOTH caches (after their appends), the KV
        # happens-before the validator requires. All three are distinct ready counters.
        waits = self._waits_for(q, kc, vc)
        _ATTN_PMAX = 8                  # combine input arity cap (== AMK_MAX_INPUTS)
        _ATTN_MIN_KV_PER_SHARD = 16     # don't split below this many keys per shard
        # DEFAULT: ONE warp-parallel ATTENTION_TILE (the measured-best decode attention, heads across
        # warps, no per-token barriers; see vm/ops.cuh). The split-KV + ATTENTION_COMBINE path below is
        # correct and opt-in via config.tiling['attention']['n_partitions']>1, but it is OFF by default:
        # at decode KV sizes its per-shard cross-SM sync + combine overhead EXCEEDS the parallelism gain
        # (measured on L4, it regressed pos128/512 vs single-tile). It is intended for long-context
        # prefill, where the large per-shard work amortizes the split overhead.
        npart = 1
        _atile = (self.config.tiling.get("attention") if (self.config and self.config.tiling) else None)
        if isinstance(_atile, dict):
            npart = int(_atile.get("n_partitions", 1))
        P = 1 if npart <= 1 else max(1, min(_ATTN_PMAX, self.num_sms, npart,
                       (self.kv_len + _ATTN_MIN_KV_PER_SHARD - 1) // _ATTN_MIN_KV_PER_SHARD))
        if P <= 1:
            ob = self._new(out, BufferKind.ACTIVATION, (1, qd))   # flat via attention's view_as
            self.buf[out] = ob
            ctr = self._counter(n.name)
            self.p.add_task(
                InstructionKind.ATTENTION_TILE, [self.buf[q], self.buf[kc], self.buf[vc]], [ob],
                out_counter=ctr, waits=waits,
                params={"head_dim": head_dim, "kv_start": 0, "kv_len": self.kv_len, "scale": scale,
                        "n_heads": n_heads, "n_kv_heads": n_kv}, label=n.name,
                est_bytes=2 * self.kv_len * n_kv * head_dim * self.dbytes,
                est_flops=4 * n_heads * head_dim * self.kv_len)
            self.ready[out] = (ctr, 1)
            return
        # split-KV: P shards (pinned to distinct SMs) each write a flash partial [n_heads, head_dim+2]
        # = {acc | m | l}; ONE ATTENTION_COMBINE all-joins them (each shard has its own threshold-1
        # counter -> no shared-counter race) and writes the normalized output.
        B = (self.kv_len + P - 1) // P
        part_names: list[str] = []
        for p in range(P):
            ks = p * B
            kl = min(B, self.kv_len - ks)
            if kl <= 0:
                break
            pname = f"{out}.part{p}"
            part = self._new(pname, BufferKind.ACTIVATION, (n_heads, head_dim + 2))
            self.buf[pname] = part
            pc = self._counter(f"{n.name}.s{p}")
            self.p.add_task(
                InstructionKind.ATTENTION_TILE, [self.buf[q], self.buf[kc], self.buf[vc]], [part],
                out_counter=pc, waits=waits, sm=(p % self.num_sms),
                params={"head_dim": head_dim, "kv_start": ks, "kv_len": kl, "scale": scale,
                        "n_heads": n_heads, "n_kv_heads": n_kv, "flags": 2}, label=f"{n.name}.s{p}",
                est_bytes=2 * kl * n_kv * head_dim * self.dbytes,
                est_flops=4 * n_heads * head_dim * kl)
            self.ready[pname] = (pc, 1)
            part_names.append(pname)
        ob = self._new(out, BufferKind.ACTIVATION, (1, qd))
        self.buf[out] = ob
        cc = self._counter(n.name)
        self.p.add_task(
            InstructionKind.ATTENTION_COMBINE, [self.buf[nm] for nm in part_names], [ob],
            out_counter=cc, waits=self._waits_for(*part_names),
            params={"head_dim": head_dim, "n_heads": n_heads, "n_kv_heads": n_kv}, label=f"{n.name}.comb",
            est_bytes=len(part_names) * n_heads * (head_dim + 2) * self.dbytes,
            est_flops=len(part_names) * n_heads * head_dim)
        self.ready[out] = (cc, 1)

    # -- swiglu --
    def _op_swiglu(self, n: Node) -> None:
        inter = int(n.attrs["inter"])
        gate, up = n.inputs
        out = n.outputs[0]
        ob = self._new(out, BufferKind.ACTIVATION, (1, inter))
        self.buf[out] = ob
        ctr = self._counter(n.name)
        self.p.add_task(
            InstructionKind.SILU_MUL, [self.buf[gate], self.buf[up]], [ob], out_counter=ctr,
            waits=self._waits_for(gate, up), params={}, label=n.name,
            est_bytes=inter * self.dbytes, est_flops=3 * inter)
        self.ready[out] = (ctr, 1)

    # -- add (residual) --
    def _op_add(self, n: Node) -> None:
        H = int(n.attrs["hidden"])
        a, b = n.inputs
        out = n.outputs[0]
        ob = self._new(out, BufferKind.ACTIVATION, (1, H))
        self.buf[out] = ob
        ctr = self._counter(n.name)
        self.p.add_task(
            InstructionKind.ADD, [self.buf[a], self.buf[b]], [ob], out_counter=ctr,
            waits=self._waits_for(a, b), params={}, label=n.name,
            est_bytes=H * self.dbytes, est_flops=H)
        self.ready[out] = (ctr, 1)


def lower(graph: ModelGraph, target: GpuTarget | None = None,
          config: ScheduleConfig | None = None, *, pos: int = 0,
          dtype: DType = DType.F32, quant: Any = None) -> MegakernelProgram:
    """Lower a model graph into a single-decode-step :class:`MegakernelProgram`.

    Args:
        graph:  a :class:`schedule.graph.ModelGraph` (e.g. from :func:`schedule.graph.from_toy`).
        target: the :class:`GpuTarget` to tag the program with (drives the cost model / roofline).
                Optional so the lowering is testable GPU-free.
        config: a :class:`ScheduleConfig`; only ``config.tiling['gemv']['N_tile']`` is consumed
                here. Other knobs (SM assignment, pages, pipelining) are downstream passes.
        pos:    the decode position of THIS token (0 for the first token / empty cache). The
                attention window is ``kv_len = pos + 1`` and KV_APPEND writes index ``pos``.
        dtype:  activation/weight element type for the emitted buffers (F32 for the CPU oracle).
        quant:  optional :class:`schedule.quantize.QuantMeta`. When given, projection GEMVs whose
                weight key it marks quantized are lowered as dequant-fused int4/int8 GEMV tiles
                (packed weight buffer + fp16 scales [+ zeros]); the caller binds the quantized
                weights dict (from :func:`schedule.quantize.quantize_weights`). The non-quantized
                path is byte-identical when ``quant`` is None.

    Returns a program that, after the caller binds ``model.weights_dict()`` and supplies the
    inputs (token id, pos, the constant reshape id), runs in the reference VM and equals eager.
    The result is guaranteed to pass :func:`schedule.ir.validate` (the VM refuses anything else).
    """
    return _Lowerer(graph, target, config, pos=pos, dtype=dtype, quant=quant).lower()


# ----------------------------------------------------------------------------------------
# search.py adapter: it calls lower_fn(graph, config, target). Provide that argument order too.
# ----------------------------------------------------------------------------------------
def lower_fn(graph: ModelGraph, config: ScheduleConfig, target: GpuTarget) -> MegakernelProgram:
    """Adapter matching :data:`schedule.search.LowerFn` = ``(graph, config, target) -> program``.
    Lowers a fresh decode step at pos=0 (the search/cost-model probe point)."""
    return lower(graph, target=target, config=config, pos=0, dtype=DType.F32)


def required_inputs(pos: int = 0) -> dict[str, Any]:
    """Document the run() input contract for a lowered decode program: the names the caller must
    provide to :meth:`vm.reference_vm.ReferenceVM.run`. The constant reshape id is always ``[0]``.
    (Values are placeholders; the caller fills token_id/pos with real tensors.)"""
    return {TOKEN_NAME: None, POS_NAME: [pos], RESHAPE_ID_NAME: [0]}


__all__ = ["lower", "lower_fn", "required_inputs", "RESHAPE_ID_NAME", "POS_NAME", "TOKEN_NAME"]
