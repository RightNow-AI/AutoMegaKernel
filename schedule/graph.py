"""
AMK, MODEL GRAPH IMPORTER (Layer 2, front half)
================================================

This module turns a *trained model* into a small, explicit **graph IR** that the lowerer
(:mod:`schedule.lower`) consumes. It is deliberately tiny and structural:

  * A :class:`ModelGraph` is a flat list of :class:`Node` ops plus a :class:`ModelConfig`
    (the decoder shape: hidden / heads / kv-heads / head_dim / intermediate / vocab / layers /
    rms_eps / rope_theta). Every node carries its input/output *symbolic tensor names*, an op
    name, and an ``attrs`` dict; weight nodes additionally record the exact ``state_dict`` key
    so the lowerer can wire a ``WEIGHT`` buffer with the right ``source``.
  * :func:`from_toy` reads a :class:`models.toy.ToyLlama` and emits a REAL, complete graph: every
    weight name is the verbatim key from ``model.weights_dict()`` so the reference VM binds it.
  * :func:`from_hf` is a **documented stub** that maps a HuggingFace Llama-style module via its
    named modules / state_dict. The structural mapping (which keys exist, how they name) is
    written out; the parts that need a live ``transformers`` install to verify are marked TODO.

WHY A GRAPH (and not torch.fx): for a known decoder family the structure is *fixed*, embed,
N×(norm, qkv, rope, attn, o, norm, swiglu), final-norm, lm_head. Deriving the graph from the
config + state_dict is exact and robust; fx tracing a real HF model is fragile (control flow,
cache objects, rotary buffers) and would couple us to private internals. We therefore build the
graph from the *known* decoder template and the *verified* weight keys. This is the standard
"model card -> graph" front end; the lowerer never sees torch.

The graph is intentionally backend-free: it names tensors, not buffers, and has no counters,
tiles, or pages. All of that is the lowerer's job. Keeping the two layers separate means search
(:mod:`schedule.search`) can re-lower the same graph under many :class:`ScheduleConfig`s.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ======================================================================================
# Op vocabulary of the graph IR (decoder-family archetypes). These are *graph* ops, not the
# ABI opcodes in schedule.ir.InstructionKind, the lowerer maps graph ops -> ABI tasks. Keeping
# the two vocabularies distinct lets the graph stay close to the math while the IR stays close
# to the machine.
# ======================================================================================
EMBED = "embed"
RMSNORM = "rmsnorm"
LINEAR = "linear"          # y = x @ W.T  (the four attention projections + the two-of-three MLP)
ROPE = "rope"              # rotary position embedding on q or k
KV_APPEND = "kv_append"    # write this step's k/v into the cache at `pos`
ATTENTION = "attention"    # GQA over the whole cached window
SWIGLU = "swiglu"          # silu(gate) * up
ADD = "add"                # residual add
LM_HEAD = "linear"         # the output projection is just a LINEAR (alias for readability)

GRAPH_OPS = frozenset({EMBED, RMSNORM, LINEAR, ROPE, KV_APPEND, ATTENTION, SWIGLU, ADD})


# ======================================================================================
# Config
# ======================================================================================
@dataclass
class ModelConfig:
    """The decoder shape. These are exactly the numbers the lowerer needs to size every buffer
    and pick GQA grouping / rope tables. Field names mirror :class:`models.toy.ToyConfig`."""

    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    n_layers: int
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_seq: int = 256

    @property
    def q_dim(self) -> int:
        """Total query width = n_heads * head_dim (== hidden for square attention, but not in
        general; the toy uses head_dim=16, n_heads=4 -> q_dim=64=hidden by coincidence)."""
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def rep(self) -> int:
        """GQA replication factor: how many query heads share one kv head."""
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        return self.n_heads // self.n_kv_heads

    def validate(self) -> None:
        assert self.head_dim % 2 == 0, "head_dim must be even for rotate-half RoPE"
        assert self.q_dim == self.n_heads * self.head_dim
        assert self.n_heads % self.n_kv_heads == 0


# ======================================================================================
# Graph node + weight records
# ======================================================================================
@dataclass
class Weight:
    """A parameter the graph references. ``key`` is the exact ``state_dict`` name so the lowerer
    can emit a ``BufferKind.WEIGHT`` with ``source=key`` and the VM binds it verbatim."""

    key: str                 # exact state_dict key (e.g. "layers.0.attn.q_proj.weight")
    shape: tuple[int, ...]   # logical shape (torch Linear layout [N_out, K_in] for projections)
    role: str = ""           # human tag: "q_proj", "input_norm", "embed", ...


@dataclass
class Node:
    """One graph op. ``inputs``/``outputs`` are symbolic tensor names (strings). ``weights`` are
    the parameter keys this op reads. ``attrs`` carries op-specific scalars (eps, theta, which
    projection, GQA dims, kv position role, etc.). The lowerer is a pure function of these."""

    op: str
    inputs: list[str]
    outputs: list[str]
    weights: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    name: str = ""           # human label, e.g. "L0.attn.q_proj"

    def __post_init__(self) -> None:
        if self.op not in GRAPH_OPS:
            raise ValueError(f"unknown graph op {self.op!r} (expected one of {sorted(GRAPH_OPS)})")


@dataclass
class ModelGraph:
    """The importer's output: a flat, ordered op list + the model config + a weight table.

    Ordered so a naive reader can follow the forward pass top-to-bottom; the lowerer does NOT
    rely on the order for correctness (it wires dependencies by tensor name), only for stable
    labelling. ``tensor_shapes`` records the symbolic shape of every named tensor for the lowerer
    to size activation buffers without re-deriving them."""

    config: ModelConfig
    nodes: list[Node] = field(default_factory=list)
    weights: dict[str, Weight] = field(default_factory=dict)
    tensor_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- builder helpers ----------------------------------------------------------
    def add_weight(self, key: str, shape: tuple[int, ...], role: str = "") -> str:
        self.weights[key] = Weight(key=key, shape=tuple(shape), role=role)
        return key

    def add(self, op: str, inputs: list[str], outputs: list[str],
            weights: list[str] | None = None, attrs: dict[str, Any] | None = None,
            name: str = "") -> Node:
        n = Node(op=op, inputs=list(inputs), outputs=list(outputs),
                 weights=list(weights or []), attrs=dict(attrs or {}), name=name)
        self.nodes.append(n)
        return n

    def set_shape(self, tensor: str, shape: tuple[int, ...]) -> None:
        self.tensor_shapes[tensor] = tuple(shape)

    # ---- introspection -------------------------------------------------------------
    def weight_keys(self) -> list[str]:
        return list(self.weights.keys())

    def check_weight_keys(self, available: dict[str, Any]) -> list[str]:
        """Return weight keys referenced by the graph that are MISSING from ``available`` (e.g.
        ``model.weights_dict()``). Empty list == the graph is bindable. The lowerer/test call
        this to fail fast with a clear message rather than a deep KeyError in the VM."""
        return [k for k in self.weights if k not in available]

    def summary(self) -> str:
        c = self.config
        return (f"ModelGraph(layers={c.n_layers}, hidden={c.hidden}, heads={c.n_heads}/"
                f"{c.n_kv_heads}kv x{c.head_dim}, inter={c.intermediate}, vocab={c.vocab}, "
                f"nodes={len(self.nodes)}, weights={len(self.weights)})")


# ======================================================================================
# Decoder template, the single source of structure shared by from_toy and from_hf
# ======================================================================================
def _build_decoder_graph(cfg: ModelConfig, keys: "_KeyMap") -> ModelGraph:
    """Emit the canonical Llama-style decode graph for one token.

    The structure is fixed for the whole family; only the *weight key naming* differs between
    the toy and HF, which is abstracted behind :class:`_KeyMap`. This is the heart of the
    importer: get this template right once and every Llama-shaped model imports correctly.
    """
    cfg.validate()
    g = ModelGraph(config=cfg, meta={"family": "llama-decode", "key_scheme": keys.scheme})
    H, qd, kd = cfg.hidden, cfg.q_dim, cfg.kv_dim

    # --- embedding table + lm head + final norm (model-level params) ---
    g.add_weight(keys.embed(), (cfg.vocab, H), role="embed")
    g.add_weight(keys.final_norm(), (H,), role="final_norm")
    g.add_weight(keys.lm_head(), (cfg.vocab, H), role="lm_head")

    # token id -> hidden
    g.add("embed", inputs=["token_id"], outputs=["h0"], weights=[keys.embed()],
          attrs={"hidden": H}, name="embed")
    g.set_shape("token_id", (1,))
    g.set_shape("h0", (1, H))

    resid = "h0"
    for L in range(cfg.n_layers):
        p = f"L{L}"
        # ---- attention sub-block ----
        in_norm = keys.input_norm(L)
        g.add_weight(in_norm, (H,), role="input_norm")
        g.add("rmsnorm", [resid], [f"{p}.an"], weights=[in_norm],
              attrs={"eps": cfg.rms_eps, "hidden": H}, name=f"{p}.input_norm")
        g.set_shape(f"{p}.an", (1, H))

        for proj, dim, role in ((keys.q_proj(L), qd, "q_proj"),
                                (keys.k_proj(L), kd, "k_proj"),
                                (keys.v_proj(L), kd, "v_proj")):
            g.add_weight(proj, (dim, H), role=role)
        g.add("linear", [f"{p}.an"], [f"{p}.q"], weights=[keys.q_proj(L)],
              attrs={"K": H, "N": qd, "kind": "q_proj"}, name=f"{p}.q_proj")
        g.add("linear", [f"{p}.an"], [f"{p}.k"], weights=[keys.k_proj(L)],
              attrs={"K": H, "N": kd, "kind": "k_proj"}, name=f"{p}.k_proj")
        g.add("linear", [f"{p}.an"], [f"{p}.v"], weights=[keys.v_proj(L)],
              attrs={"K": H, "N": kd, "kind": "v_proj"}, name=f"{p}.v_proj")
        g.set_shape(f"{p}.q", (1, qd))
        g.set_shape(f"{p}.k", (1, kd))
        g.set_shape(f"{p}.v", (1, kd))

        g.add("rope", [f"{p}.q", "pos"], [f"{p}.qr"],
              attrs={"head_dim": cfg.head_dim, "theta": cfg.rope_theta, "n_heads": cfg.n_heads,
                     "which": "q"}, name=f"{p}.rope_q")
        g.add("rope", [f"{p}.k", "pos"], [f"{p}.kr"],
              attrs={"head_dim": cfg.head_dim, "theta": cfg.rope_theta, "n_heads": cfg.n_kv_heads,
                     "which": "k"}, name=f"{p}.rope_k")
        g.set_shape(f"{p}.qr", (cfg.n_heads, cfg.head_dim))
        g.set_shape(f"{p}.kr", (1, cfg.n_kv_heads, cfg.head_dim))

        g.add("kv_append", [f"{p}.kr"], [f"{p}.kcache"],
              attrs={"which": "k", "n_kv_heads": cfg.n_kv_heads, "head_dim": cfg.head_dim},
              name=f"{p}.k_append")
        g.add("kv_append", [f"{p}.v"], [f"{p}.vcache"],
              attrs={"which": "v", "n_kv_heads": cfg.n_kv_heads, "head_dim": cfg.head_dim},
              name=f"{p}.v_append")

        g.add("attention", [f"{p}.qr", f"{p}.kcache", f"{p}.vcache"], [f"{p}.attn"],
              attrs={"head_dim": cfg.head_dim, "n_heads": cfg.n_heads,
                     "n_kv_heads": cfg.n_kv_heads, "scale": cfg.head_dim ** -0.5},
              name=f"{p}.attention")
        g.set_shape(f"{p}.attn", (1, qd))

        o_proj = keys.o_proj(L)
        g.add_weight(o_proj, (H, qd), role="o_proj")
        g.add("linear", [f"{p}.attn"], [f"{p}.o"], weights=[o_proj],
              attrs={"K": qd, "N": H, "kind": "o_proj"}, name=f"{p}.o_proj")
        g.set_shape(f"{p}.o", (1, H))

        g.add("add", [resid, f"{p}.o"], [f"{p}.h_attn"], attrs={"hidden": H},
              name=f"{p}.attn_residual")
        g.set_shape(f"{p}.h_attn", (1, H))
        resid = f"{p}.h_attn"

        # ---- MLP (SwiGLU) sub-block ----
        post_norm = keys.post_norm(L)
        g.add_weight(post_norm, (H,), role="post_norm")
        g.add("rmsnorm", [resid], [f"{p}.mn"], weights=[post_norm],
              attrs={"eps": cfg.rms_eps, "hidden": H}, name=f"{p}.post_norm")
        g.set_shape(f"{p}.mn", (1, H))

        gate, up, down = keys.gate(L), keys.up(L), keys.down(L)
        g.add_weight(gate, (cfg.intermediate, H), role="gate_proj")
        g.add_weight(up, (cfg.intermediate, H), role="up_proj")
        g.add_weight(down, (H, cfg.intermediate), role="down_proj")
        g.add("linear", [f"{p}.mn"], [f"{p}.gate"], weights=[gate],
              attrs={"K": H, "N": cfg.intermediate, "kind": "gate_proj"}, name=f"{p}.gate_proj")
        g.add("linear", [f"{p}.mn"], [f"{p}.up"], weights=[up],
              attrs={"K": H, "N": cfg.intermediate, "kind": "up_proj"}, name=f"{p}.up_proj")
        g.set_shape(f"{p}.gate", (1, cfg.intermediate))
        g.set_shape(f"{p}.up", (1, cfg.intermediate))

        g.add("swiglu", [f"{p}.gate", f"{p}.up"], [f"{p}.act"], attrs={"inter": cfg.intermediate},
              name=f"{p}.swiglu")
        g.set_shape(f"{p}.act", (1, cfg.intermediate))

        g.add("linear", [f"{p}.act"], [f"{p}.d"], weights=[down],
              attrs={"K": cfg.intermediate, "N": H, "kind": "down_proj"}, name=f"{p}.down_proj")
        g.set_shape(f"{p}.d", (1, H))

        g.add("add", [resid, f"{p}.d"], [f"{p}.h_mlp"], attrs={"hidden": H},
              name=f"{p}.mlp_residual")
        g.set_shape(f"{p}.h_mlp", (1, H))
        resid = f"{p}.h_mlp"

    # ---- final norm + lm head ----
    g.add("rmsnorm", [resid], ["hf"], weights=[keys.final_norm()],
          attrs={"eps": cfg.rms_eps, "hidden": H}, name="final_norm")
    g.set_shape("hf", (1, H))
    g.add("linear", ["hf"], ["logits"], weights=[keys.lm_head()],
          attrs={"K": H, "N": cfg.vocab, "kind": "lm_head"}, name="lm_head")
    g.set_shape("logits", (1, cfg.vocab))
    g.set_shape("pos", (1,))
    return g


@dataclass
class _KeyMap:
    """Abstracts the *naming* of a model's parameters so the same decoder template serves both
    the toy and HF. Subclasses fill in the dotted ``state_dict`` keys."""

    scheme: str

    def embed(self) -> str: raise NotImplementedError
    def final_norm(self) -> str: raise NotImplementedError
    def lm_head(self) -> str: raise NotImplementedError
    def input_norm(self, L: int) -> str: raise NotImplementedError
    def post_norm(self, L: int) -> str: raise NotImplementedError
    def q_proj(self, L: int) -> str: raise NotImplementedError
    def k_proj(self, L: int) -> str: raise NotImplementedError
    def v_proj(self, L: int) -> str: raise NotImplementedError
    def o_proj(self, L: int) -> str: raise NotImplementedError
    def gate(self, L: int) -> str: raise NotImplementedError
    def up(self, L: int) -> str: raise NotImplementedError
    def down(self, L: int) -> str: raise NotImplementedError


class _ToyKeys(_KeyMap):
    """state_dict keys produced by :meth:`models.toy.ToyLlama.weights_dict`.

    Verified against the live module: ``embed.weight``, ``layers.{L}.input_norm``,
    ``layers.{L}.post_norm``, ``layers.{L}.attn.{q,k,v,o}_proj.weight``,
    ``layers.{L}.mlp.{gate,up,down}_proj.weight``, ``final_norm``, ``lm_head.weight``."""

    def __init__(self) -> None:
        super().__init__(scheme="toy")

    def embed(self) -> str: return "embed.weight"
    def final_norm(self) -> str: return "final_norm"
    def lm_head(self) -> str: return "lm_head.weight"
    def input_norm(self, L: int) -> str: return f"layers.{L}.input_norm"
    def post_norm(self, L: int) -> str: return f"layers.{L}.post_norm"
    def q_proj(self, L: int) -> str: return f"layers.{L}.attn.q_proj.weight"
    def k_proj(self, L: int) -> str: return f"layers.{L}.attn.k_proj.weight"
    def v_proj(self, L: int) -> str: return f"layers.{L}.attn.v_proj.weight"
    def o_proj(self, L: int) -> str: return f"layers.{L}.attn.o_proj.weight"
    def gate(self, L: int) -> str: return f"layers.{L}.mlp.gate_proj.weight"
    def up(self, L: int) -> str: return f"layers.{L}.mlp.up_proj.weight"
    def down(self, L: int) -> str: return f"layers.{L}.mlp.down_proj.weight"


class _HFLlamaKeys(_KeyMap):
    """state_dict keys for a HuggingFace ``LlamaForCausalLM`` (and its many clones: Mistral,
    Qwen2, TinyLlama, ...). The naming is stable across these:

        model.embed_tokens.weight
        model.layers.{L}.input_layernorm.weight
        model.layers.{L}.post_attention_layernorm.weight
        model.layers.{L}.self_attn.{q,k,v,o}_proj.weight
        model.layers.{L}.mlp.{gate,up,down}_proj.weight
        model.norm.weight
        lm_head.weight                       (tied to embed_tokens for some checkpoints)
    """

    def __init__(self, *, tied: bool = False) -> None:
        super().__init__(scheme="hf-llama")
        # When the checkpoint ties the output projection to the input embedding, HF's
        # state_dict omits ``lm_head.weight`` entirely. The math is identical to using the
        # embedding table as the lm_head, so we point the lm_head WEIGHT buffer's source at
        # ``model.embed_tokens.weight``, the VM then binds the same tensor and the output is
        # exactly HF's tied logits. (HF does NOT scale tied logits, matching the toy lm_head.)
        self._tied = bool(tied)

    def embed(self) -> str: return "model.embed_tokens.weight"
    def final_norm(self) -> str: return "model.norm.weight"
    def lm_head(self) -> str:
        return "model.embed_tokens.weight" if self._tied else "lm_head.weight"
    def input_norm(self, L: int) -> str: return f"model.layers.{L}.input_layernorm.weight"
    def post_norm(self, L: int) -> str: return f"model.layers.{L}.post_attention_layernorm.weight"
    def q_proj(self, L: int) -> str: return f"model.layers.{L}.self_attn.q_proj.weight"
    def k_proj(self, L: int) -> str: return f"model.layers.{L}.self_attn.k_proj.weight"
    def v_proj(self, L: int) -> str: return f"model.layers.{L}.self_attn.v_proj.weight"
    def o_proj(self, L: int) -> str: return f"model.layers.{L}.self_attn.o_proj.weight"
    def gate(self, L: int) -> str: return f"model.layers.{L}.mlp.gate_proj.weight"
    def up(self, L: int) -> str: return f"model.layers.{L}.mlp.up_proj.weight"
    def down(self, L: int) -> str: return f"model.layers.{L}.mlp.down_proj.weight"


# ======================================================================================
# Public importers
# ======================================================================================
def from_toy(model: Any) -> ModelGraph:
    """Import a :class:`models.toy.ToyLlama` into a :class:`ModelGraph`. REAL and complete.

    Reads the model's ``cfg`` for shapes and uses the verified ``weights_dict`` key scheme, so
    the resulting graph lowers to a program the reference VM binds and runs end-to-end. This is
    the M0 generality path the acceptance test exercises.
    """
    cfg_src = model.cfg
    cfg = ModelConfig(
        hidden=cfg_src.hidden, n_heads=cfg_src.n_heads, n_kv_heads=cfg_src.n_kv_heads,
        head_dim=cfg_src.head_dim, intermediate=cfg_src.intermediate, vocab=cfg_src.vocab,
        n_layers=cfg_src.n_layers, rms_eps=cfg_src.rms_eps, rope_theta=cfg_src.rope_theta,
        max_seq=getattr(cfg_src, "max_seq", 256),
    )
    g = _build_decoder_graph(cfg, _ToyKeys())
    g.meta["source"] = "models.toy.ToyLlama"
    # Best-effort sanity: every key the graph references should exist in the model. We do not
    # require torch here (graph stays light), but if the model exposes weights_dict we check.
    if hasattr(model, "weights_dict"):
        missing = g.check_weight_keys(model.weights_dict())
        if missing:
            raise KeyError(f"from_toy: graph references weight keys absent from weights_dict: "
                           f"{missing[:8]}")
    return g


def from_hf(model: Any, *, n_layers: int | None = None) -> ModelGraph:
    """Import a HuggingFace Llama-style ``*ForCausalLM`` into a :class:`ModelGraph`.

    IMPLEMENTED (real, no GPU needed):
      * Read the shape from ``model.config`` (hidden_size, num_attention_heads,
        num_key_value_heads, head_dim or hidden//heads, intermediate_size, vocab_size,
        num_hidden_layers, rms_norm_eps, rope_theta). These attribute names are stable across the
        Llama/Mistral/Qwen2 family.
      * Build the canonical decode graph via the shared template with HF key naming
        (:class:`_HFLlamaKeys`). The emitted weight keys match a standard HF ``state_dict``.
      * Validate every referenced key against ``model.state_dict()`` so binding can't silently
        drift; raises a clear error listing missing keys.

    TODO (need a live ``transformers`` install / a real checkpoint to verify, hence stubbed):
      * Tied embeddings: when ``config.tie_word_embeddings`` is True some checkpoints omit
        ``lm_head.weight``; the lowerer should fall back to ``model.embed_tokens.weight``. We
        record the flag in ``meta['tie_word_embeddings']`` but do not yet rewrite the key.
      * Per-head ``head_dim`` override models (e.g. some Qwen variants) and partial-rotary
        (``rotary_pct < 1``), the template assumes full-rotary head_dim, the common case.
      * QKV-fused checkpoints (single ``Wqkv``) and biased projections (Falcon-style), the toy
        path and standard Llama have separate, bias-free projections; fused/biased variants need
        a split/bias-aware template (not implemented).
      * Sliding-window / non-causal-window attention metadata is not threaded through (decode
        attends to the whole cached window here).

    The toy path (:func:`from_toy`) is the fully-exercised one; this gives the real structural
    map for HF so wiring a checkpoint is a config read + a key check, not a rewrite.
    """
    hf = model.config
    hidden = int(hf.hidden_size)
    n_heads = int(hf.num_attention_heads)
    n_kv = int(getattr(hf, "num_key_value_heads", n_heads))
    head_dim = int(getattr(hf, "head_dim", hidden // n_heads))

    # ---- guard the variants this template does NOT model (fail loud, never silently wrong) ----
    # The shared decode template + instructions/reference.py implement exactly: bias-free
    # projections, full-rotary rotate-half RoPE with a single theta, RMSNorm, GQA, SiLU SwiGLU.
    # Any HF config that deviates would produce numerically wrong logits, so we reject it with a
    # precise message rather than emit a graph that quietly disagrees with the HF forward.
    unsupported: list[str] = []
    if bool(getattr(hf, "attention_bias", False)):
        unsupported.append("attention_bias=True (Falcon-style biased q/k/v/o projections)")
    if bool(getattr(hf, "mlp_bias", False)):
        unsupported.append("mlp_bias=True (biased MLP projections)")
    act = getattr(hf, "hidden_act", "silu")
    if act not in ("silu", "swish"):
        unsupported.append(f"hidden_act={act!r} (only SiLU/SwiGLU is modeled)")
    if int(getattr(hf, "num_local_experts", 0) or 0) > 0 or int(getattr(hf, "num_experts", 0) or 0) > 0:
        unsupported.append("mixture-of-experts (num_local_experts/num_experts > 0); MoE routing + "
                           "per-expert FFN are not modeled")
    if getattr(hf, "kv_lora_rank", None) is not None:
        unsupported.append("MLA / latent attention (kv_lora_rank set, DeepSeek-style); the dense "
                           "GQA template does not model compressed-KV attention")
    if getattr(hf, "head_dim", None) is None and n_heads > 0 and hidden % n_heads != 0:
        unsupported.append(f"hidden_size={hidden} is not divisible by num_attention_heads={n_heads} "
                           f"and no explicit head_dim is given (the implicit head_dim would be wrong)")
    rope_theta, rope_unsupported = _resolve_rope(hf)
    if rope_unsupported is not None:
        unsupported.append(rope_unsupported)
    if unsupported:
        raise NotImplementedError(
            "from_hf: this HuggingFace config uses feature(s) the AMK Llama template does not "
            "model yet: " + "; ".join(unsupported) + ". Narrow the config to a standard "
            "bias-free, full-rotary, SiLU Llama (the verified family) or extend the template "
            "+ instructions/reference.py first.")

    cfg = ModelConfig(
        hidden=hidden, n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim,
        intermediate=int(hf.intermediate_size), vocab=int(hf.vocab_size),
        n_layers=int(n_layers if n_layers is not None else hf.num_hidden_layers),
        rms_eps=float(getattr(hf, "rms_norm_eps", 1e-6)),
        rope_theta=float(rope_theta),
        max_seq=int(getattr(hf, "max_position_embeddings", 4096)),
    )

    tied = bool(getattr(hf, "tie_word_embeddings", False))
    g = _build_decoder_graph(cfg, _HFLlamaKeys(tied=tied))
    g.meta["source"] = type(model).__name__
    g.meta["tie_word_embeddings"] = tied
    g.meta["rope_theta"] = float(rope_theta)
    if tied:
        # The lm_head WEIGHT buffer now sources from model.embed_tokens.weight (see _HFLlamaKeys).
        g.meta["lm_head_tied_to_embed"] = True
    if hasattr(model, "state_dict"):
        sd = model.state_dict()
        missing = g.check_weight_keys(sd)
        if missing:
            raise KeyError(
                f"from_hf: graph references state_dict keys not present in the model: "
                f"{missing[:8]} (tie_word_embeddings={tied})")
        # Some checkpoints (e.g. Qwen2) carry HARDCODED q/k/v (or MLP) biases that no config flag
        # advertises, so the config-only check above misses them. The bias-free decode template
        # cannot represent them, scan the actual tensors and reject loudly rather than emit a
        # silently-wrong megakernel. (Found by paper/exp_coverage.py.)
        bias_keys = [k for k in sd
                     if k.endswith(".bias")
                     and any(p in k for p in ("q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj"))]
        if bias_keys:
            raise NotImplementedError(
                "from_hf: the checkpoint has projection bias tensors the bias-free AMK template "
                f"cannot model (e.g. {bias_keys[:4]}). This is the Qwen2-style hardcoded-bias case; "
                "a bias-aware template is needed. Refusing to emit a silently-wrong megakernel.")
    return g


def _resolve_rope(hf: Any) -> tuple[float, str | None]:
    """Resolve the RoPE base ``theta`` across transformers versions and reject scaled-rope.

    Returns ``(theta, unsupported_reason_or_None)``. In transformers v5 the base moved into
    ``config.rope_scaling['rope_theta']`` with a ``rope_type`` discriminator; older versions
    expose a top-level ``config.rope_theta`` and ``rope_scaling=None`` for plain RoPE. The AMK
    template models only *default* (full, unscaled) rotate-half RoPE, so any non-default
    ``rope_type`` (linear / dynamic / llama3 / yarn ...) is reported as unsupported. ``theta``
    defaults to 10000.0 when absent, matching the Llama family default.
    """
    rs = getattr(hf, "rope_scaling", None)
    theta = float(getattr(hf, "rope_theta", 10000.0) or 10000.0)
    if isinstance(rs, dict):
        rope_type = rs.get("rope_type", rs.get("type", "default"))
        if "rope_theta" in rs and rs["rope_theta"] is not None:
            theta = float(rs["rope_theta"])
        if rope_type not in ("default", "llama", None):
            return theta, f"rope_scaling rope_type={rope_type!r} (only default/full RoPE modeled)"
    return theta, None


def weights_from_hf(model: Any) -> dict[str, Any]:
    """Adapter: read a HuggingFace module's ``state_dict()`` into the ``{source_key: tensor}``
    dict :class:`vm.reference_vm.ReferenceVM` binds against (float32, detached, cloned).

    The keys are the verbatim HF ``state_dict`` names, which is exactly what
    :func:`from_hf` records as each WEIGHT buffer's ``source``. Tied checkpoints route the
    lm_head buffer to ``model.embed_tokens.weight``, which is present here, so no special case
    is needed at bind time. Float32 makes this the exact correctness oracle.
    """
    sd = model.state_dict()
    out: dict[str, Any] = {}
    for k, v in sd.items():
        t = v.detach()
        out[k] = t.float().clone() if hasattr(t, "float") else t
    return out


__all__ = [
    "ModelConfig", "Weight", "Node", "ModelGraph",
    "from_toy", "from_hf", "weights_from_hf",
    "EMBED", "RMSNORM", "LINEAR", "ROPE", "KV_APPEND", "ATTENTION", "SWIGLU", "ADD", "GRAPH_OPS",
]
