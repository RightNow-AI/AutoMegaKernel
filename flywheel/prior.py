"""
AMK, THE FLYWHEEL LEARNING PRIOR (the cross-run moat)
=====================================================

This module is *the* thing that makes every AMK autoresearch run start smarter than the last. It
turns the append-only ``flywheel/corpus.jsonl`` (one record per KEPT, correctness-PASS,
measured/predicted (model, gpu, schedule, latency) point) into two cheap, robust functions the
autoresearch loop consumes:

  * :func:`warm_start`, given a target ``(model_shape, gpu)``, return a *seed list* of the
    best-known :class:`~schedule.ir.ScheduleConfig`\\s from the corpus for the same or the NEAREST
    model shape on the same GPU. A new run begins from accumulated knowledge instead of the neutral
    default. This is the single biggest cross-run win: a warm run's *first* incumbent is already a
    good config, not the textbook default.

  * :func:`rank`, a learned ranking prior. Featurize every (model_shape, ScheduleConfig) pair in
    the corpus, target = ``pct_of_roofline`` (lower is better, closer to the floor; we minimise),
    train a lightweight predictor (sklearn ``GradientBoostingRegressor`` if available, else a robust
    k-NN-in-feature-space fallback, NO new heavy deps), and SCORE candidate configs so the loop
    evaluates the most promising ones first. Retrains as the corpus grows; falls back to pure
    exploration (returns candidates unchanged) when the corpus is too sparse to learn anything.

HONESTY / ROBUSTNESS
--------------------
  * The prior is ADVISORY only. It re-orders/seeds proposals; it NEVER replaces the keep/revert
    gate. Every config the loop tries is still lowered, validated, correctness-checked, and
    benchmarked exactly as before, a bad suggestion can only ever lose, never corrupt the search.
  * Everything degrades gracefully: an empty corpus, a corpus with one row, sklearn missing, a
    malformed record, each yields a safe fallback (default seeds / identity ranking), never a
    crash. The autoresearch loop must run unattended for hours; the prior never stops it.
  * No new dependencies. sklearn is used ONLY if already importable; otherwise the pure-numpy/stdlib
    k-NN fallback is used. torch is not required here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from flywheel.log import read_corpus
from schedule.ir import GpuTarget, ScheduleConfig

# A run is "warm-startable" / the ranker is "trainable" only with at least this many usable corpus
# rows for the (gpu) in question. Below this we fall back to pure exploration (honest: a 2-row
# corpus cannot teach a prior anything; pretending otherwise would be fabrication).
MIN_ROWS_TO_RANK = 6
MIN_ROWS_TO_WARM = 1


# ======================================================================================
# Model-shape descriptor + distance (nearest-shape transfer)
# ======================================================================================
@dataclass(frozen=True)
class ModelShape:
    """The shape features that determine which past schedules transfer. Two models with the same
    (hidden, n_layers, GQA grouping, intermediate ratio) optimize almost identically, so the
    flywheel keys schedules on this, NOT on the model *name*, that is what lets a Llama-1B run
    benefit from a prior Qwen-0.6B run of the same shape family."""

    hidden: int = 0
    n_layers: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    intermediate: int = 0
    vocab: int = 0

    @staticmethod
    def from_config(cfg: Any) -> "ModelShape":
        """Build from a ModelConfig / ToyConfig / ModelGraph.config (anything with the fields)."""
        g = lambda n, d=0: int(getattr(cfg, n, d) or 0)  # noqa: E731
        return ModelShape(
            hidden=g("hidden"), n_layers=g("n_layers"), n_heads=g("n_heads"),
            n_kv_heads=g("n_kv_heads", g("n_heads")), head_dim=g("head_dim"),
            intermediate=g("intermediate"), vocab=g("vocab"),
        )

    @staticmethod
    def from_graph(graph: Any) -> "ModelShape":
        cfg = getattr(graph, "config", graph)
        return ModelShape.from_config(cfg)

    def feature_vec(self) -> list[float]:
        """Log-scaled shape features (sizes span orders of magnitude; log keeps distance sane)."""
        def lg(x: int) -> float:
            return math.log1p(max(0, x))
        gqa = self.n_heads / self.n_kv_heads if self.n_kv_heads else 1.0
        ffn_ratio = self.intermediate / self.hidden if self.hidden else 0.0
        return [lg(self.hidden), lg(self.n_layers), lg(self.n_heads), gqa,
                lg(self.head_dim), ffn_ratio, lg(self.vocab)]

    def distance(self, other: "ModelShape") -> float:
        """Euclidean distance in the log-feature space; 0 == identical shape family."""
        a, b = self.feature_vec(), other.feature_vec()
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ======================================================================================
# Config featurization (the ScheduleConfig knobs -> a fixed numeric vector)
# ======================================================================================
_SM_POLICIES = ("round_robin", "load_balance", "explicit")
_PAGE_POLICIES = ("none", "linear", "graph_color")


def config_features(cfg_dict: dict[str, Any]) -> list[float]:
    """Map a ScheduleConfig dict to a fixed-length numeric feature vector for the ranker. Robust to
    missing/odd keys (the corpus spans schema versions). Categorical knobs are one-hot-ish ordinals.

    The combined autoresearch candidate carries a ``kernel_knobs`` sub-dict (the MegakernelVM
    compile-time build knobs: cp.async on/off, cols_per_warp, cpa_stages, ...). Those are the lever
    that actually moves the MEASURED decode latency, so the ranker MUST see them, we append them to
    the feature vector. A schedule-only dict (no kernel_knobs) falls back to the knob DEFAULTS so the
    feature length is fixed across corpus-schema versions."""
    tiling = cfg_dict.get("tiling", {}) or {}
    gemv = tiling.get("gemv", {}) if isinstance(tiling, dict) else {}
    attn = tiling.get("attention", {}) if isinstance(tiling, dict) else {}
    n_tile = float(gemv.get("N_tile", gemv.get("base_width", 0)) or 0)
    kv_block = float(attn.get("kv_block", 0) or 0)
    pipe = float(cfg_dict.get("pipelining_depth", 0) or 0)
    tpb = float(cfg_dict.get("threads_per_block", 0) or 0)
    smem = float(cfg_dict.get("smem_bytes_per_block", 0) or 0)
    sm = cfg_dict.get("sm_assignment", "load_balance")
    sm_name = "explicit" if isinstance(sm, dict) else str(sm)
    sm_ord = float(_SM_POLICIES.index(sm_name)) if sm_name in _SM_POLICIES else 1.0
    page = str(cfg_dict.get("page_allocation", "graph_color"))
    page_ord = float(_PAGE_POLICIES.index(page)) if page in _PAGE_POLICIES else 2.0
    n_fusion = float(len(cfg_dict.get("fusion_grouping", []) or []))
    kk = cfg_dict.get("kernel_knobs") or {}
    cpw = float(kk.get("cols_per_warp", 1) or 1)
    cpa = float(kk.get("cpasync", 1) if kk else 1)
    cpa_stages = float(kk.get("cpa_stages", 4) or 4)
    cpa_cols = float(kk.get("cpa_cols", 2) or 2)
    return [n_tile, kv_block, pipe, tpb, smem / 1024.0, sm_ord, page_ord, n_fusion,
            cpw, cpa, cpa_stages, cpa_cols]


def _full_features(shape: ModelShape, cfg_dict: dict[str, Any]) -> list[float]:
    return shape.feature_vec() + config_features(cfg_dict)


# ======================================================================================
# Corpus loading + filtering
# ======================================================================================
@dataclass
class CorpusPoint:
    shape: ModelShape
    gpu: str
    schedule: dict[str, Any]
    pct_of_roofline: float    # the training target (LOWER is better, closer to the floor)
    latency_us: float


def _shape_from_record(rec: dict[str, Any]) -> ModelShape:
    """Best-effort recover a ModelShape from a corpus record. Older records may only carry a model
    name + a 'schedule' (with optional tasks/weight_mb); we parse the name's shape hint when present
    (e.g. 'llama-small-shape(2048h/4L/GQA)') and otherwise return a zeroed shape (still usable for
    same-gpu ranking, just a far nearest-shape distance)."""
    sch = rec.get("schedule", {}) or {}
    # Newer records may embed explicit shape fields in the schedule meta.
    shape_keys = ("hidden", "n_layers", "n_heads", "n_kv_heads", "head_dim", "intermediate", "vocab")
    if any(k in sch for k in shape_keys):
        return ModelShape(**{k: int(sch.get(k, 0) or 0) for k in shape_keys})
    if "model_shape" in rec and isinstance(rec["model_shape"], dict):
        ms = rec["model_shape"]
        return ModelShape(**{k: int(ms.get(k, 0) or 0) for k in shape_keys})
    # Parse a "(2048h/4L/...)" hint from the model name.
    name = str(rec.get("model", ""))
    hidden = n_layers = 0
    import re
    mh = re.search(r"(\d+)\s*h", name)
    ml = re.search(r"(\d+)\s*L", name)
    if mh:
        hidden = int(mh.group(1))
    if ml:
        n_layers = int(ml.group(1))
    return ModelShape(hidden=hidden, n_layers=n_layers)


def load_points(corpus: list[dict[str, Any]] | str | None = None,
                gpu: str | None = None) -> list[CorpusPoint]:
    """Load + clean corpus rows into CorpusPoints. ``corpus`` may be a path, a pre-read list, or
    None (default corpus path). Filters to ``gpu`` when given. Drops malformed/incorrect rows."""
    if corpus is None or isinstance(corpus, str):
        rows = read_corpus(corpus) if isinstance(corpus, str) else read_corpus()
    else:
        rows = corpus
    points: list[CorpusPoint] = []
    for rec in rows:
        try:
            if rec.get("correctness", "PASS") != "PASS":
                continue
            if gpu is not None and rec.get("gpu") != gpu:
                continue
            sch = rec.get("schedule")
            if not isinstance(sch, dict):
                continue
            pct = rec.get("pct_of_roofline")
            lat = rec.get("latency_us")
            if lat is None:
                continue
            pct_f = float(pct) if pct is not None else float("inf")
            points.append(CorpusPoint(
                shape=_shape_from_record(rec), gpu=str(rec.get("gpu", "")),
                schedule=sch, pct_of_roofline=pct_f, latency_us=float(lat)))
        except (TypeError, ValueError):
            continue  # one bad row never breaks the prior
    return points


# ======================================================================================
# warm_start, seed the run from the best-known configs for the nearest shape
# ======================================================================================
def _clean_schedule_for_config(sch: dict[str, Any]) -> dict[str, Any]:
    """Strip non-ScheduleConfig keys a corpus record may carry (dtype/arch/tasks/weight_mb) so the
    dict round-trips cleanly through ScheduleConfig."""
    keep = {"tiling", "fusion_grouping", "sm_assignment", "pipelining_depth",
            "page_allocation", "threads_per_block", "smem_bytes_per_block",
            "kernel_knobs"}  # carry the VM build knobs through warm_start (the combined candidate)
    return {k: v for k, v in sch.items() if k in keep}


def warm_start(model_shape: ModelShape, gpu: str,
               corpus: list[dict[str, Any]] | str | None = None,
               *, target: GpuTarget | None = None, k: int = 3,
               shape_radius: float = 1.0) -> list[ScheduleConfig]:
    """Return up to ``k`` seed :class:`ScheduleConfig`\\s, the best-known schedules from the corpus
    for ``(model_shape, gpu)`` (same shape preferred, else the nearest shape within ``shape_radius``
    in log-feature space). The list is ordered best-first by measured/predicted ``pct_of_roofline``.

    Returns an EMPTY list when the corpus has nothing usable for this gpu, the caller then begins
    from the neutral default (honest cold start; we never invent a seed). The autoresearch driver
    treats a non-empty return as "warm" (start from corpus knowledge) and an empty one as "cold".
    """
    points = load_points(corpus, gpu=gpu)
    if len(points) < MIN_ROWS_TO_WARM:
        return []

    # Rank candidates: prefer identical shape (distance 0), then nearest within the radius, then
    # any same-gpu point (radius=inf safety net so a brand-new shape still gets *some* seed).
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for p in points:
        dist = model_shape.distance(p.shape)
        scored.append((dist, p.pct_of_roofline, p.schedule))
    # near = same/closest shape; pick by (shape distance, then pct_of_roofline asc).
    near = [s for s in scored if s[0] <= shape_radius]
    pool = near if near else scored
    pool.sort(key=lambda s: (round(s[0], 3), s[1]))

    seeds: list[ScheduleConfig] = []
    seen: set[str] = set()
    for _dist, _pct, sch in pool:
        clean = _clean_schedule_for_config(sch)
        key = repr(sorted(clean.items()))
        if key in seen:
            continue
        seen.add(key)
        try:
            from schedule.ir import MegakernelProgram
            known = MegakernelProgram._filter_known(ScheduleConfig, dict(clean))
            sa = known.get("sm_assignment")
            if isinstance(sa, dict):
                known["sm_assignment"] = {int(kk): int(vv) for kk, vv in sa.items()}
            seeds.append(ScheduleConfig(**known))
        except Exception:
            continue
        if len(seeds) >= k:
            break
    return seeds


# ---------------------------------------------------------------------------
# CANONICAL kernel-knob choices + defaults (SINGLE SOURCE OF TRUTH)
# ---------------------------------------------------------------------------
# The subset of vm.loader._KNOB_MACRO knobs that the autoresearch loop actually
# searches.  harness.py and autoresearch.py both import from here so there is
# exactly one definition and no drift risk.  Kept in prior.py (not in
# schedule/ or vm/) to avoid circular imports: prior.py has no hard dependency
# on vm.loader (which needs torch + CUDA), while harness.py and autoresearch.py
# both already import flywheel.prior.
SEARCHABLE_KNOB_CHOICES: dict[str, tuple[int, ...]] = {
    "cols_per_warp": (1, 2, 4),    # output columns a warp computes -> x-reuse / MLP
    "cpasync":       (0, 1),        # 1 -> cp.async double-buffered GEMV; 0 -> register path
    "cpa_stages":    (2, 3, 4),     # cp.async ring depth (deeper hides more HBM latency)
    "cpa_cols":      (2, 4),        # columns one warp streams at once under cp.async
}
SEARCHABLE_KNOB_DEFAULTS: dict[str, int] = {
    "cols_per_warp": 1,
    "cpasync":       1,
    "cpa_stages":    4,
    "cpa_cols":      2,
}

# Backward-compat alias used by warm_start_combined + config_features internally.
KNOB_DEFAULTS = SEARCHABLE_KNOB_DEFAULTS


def warm_start_combined(model_shape: ModelShape, gpu: str,
                        corpus: list[dict[str, Any]] | str | None = None,
                        *, target: GpuTarget | None = None, k: int = 3,
                        shape_radius: float = 1.0
                        ) -> list[tuple[ScheduleConfig, dict[str, Any]]]:
    """Combined-candidate warm start: like :func:`warm_start` but each seed is a
    ``(ScheduleConfig, kernel_knobs)`` pair recovered from the corpus, so a warm run starts from the
    best-known SCHEDULE *and* the best-known VM build knobs (cp.async/cols_per_warp/...). The
    kernel_knobs sub-dict was embedded in the corpus ``schedule`` record by the autoresearch loop;
    a schedule-only (older) record yields the knob DEFAULTS. Ordered best-first by pct_of_roofline."""
    points = load_points(corpus, gpu=gpu)
    if len(points) < MIN_ROWS_TO_WARM:
        return []
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for p in points:
        dist = model_shape.distance(p.shape)
        scored.append((dist, p.pct_of_roofline, p.schedule))
    near = [s for s in scored if s[0] <= shape_radius]
    pool = near if near else scored
    pool.sort(key=lambda s: (round(s[0], 3), s[1]))

    out: list[tuple[ScheduleConfig, dict[str, Any]]] = []
    seen: set[str] = set()
    from schedule.ir import MegakernelProgram
    for _dist, _pct, sch in pool:
        clean = _clean_schedule_for_config(sch)
        key = repr(sorted((k2, repr(v2)) for k2, v2 in clean.items()))
        if key in seen:
            continue
        seen.add(key)
        knobs = dict(KNOB_DEFAULTS)
        raw_knobs = clean.pop("kernel_knobs", None)
        if isinstance(raw_knobs, dict):
            for kk2, vv2 in raw_knobs.items():
                try:
                    knobs[kk2] = int(vv2)
                except (TypeError, ValueError):
                    pass
        try:
            known = MegakernelProgram._filter_known(ScheduleConfig, dict(clean))
            sa = known.get("sm_assignment")
            if isinstance(sa, dict):
                known["sm_assignment"] = {int(kk2): int(vv2) for kk2, vv2 in sa.items()}
            out.append((ScheduleConfig(**known), knobs))
        except Exception:
            continue
        if len(out) >= k:
            break
    return out


# ======================================================================================
# rank, the learned ranking prior
# ======================================================================================
class _KNNRanker:
    """Robust fallback ranker: predict a candidate's pct_of_roofline as the inverse-distance
    weighted mean of its k nearest neighbours in the full (shape+config) feature space, with
    per-dimension standardization. Pure numpy/stdlib, no sklearn, no new deps."""

    def __init__(self, X: list[list[float]], y: list[float], k: int = 5):
        import numpy as np
        self._np = np
        self.Xraw = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.mu = self.Xraw.mean(axis=0)
        self.sd = self.Xraw.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        self.X = (self.Xraw - self.mu) / self.sd
        self.k = max(1, min(k, len(y)))

    def predict(self, feats: list[list[float]]) -> list[float]:
        np = self._np
        F = (np.asarray(feats, dtype=float) - self.mu) / self.sd
        out: list[float] = []
        for f in F:
            d = np.sqrt(((self.X - f) ** 2).sum(axis=1))
            idx = np.argsort(d)[: self.k]
            w = 1.0 / (d[idx] + 1e-6)
            out.append(float((w * self.y[idx]).sum() / w.sum()))
        return out


class FlywheelRanker:
    """A trained latency/roofline predictor over the corpus. ``score(shape, cfg_dict)`` returns a
    predicted ``pct_of_roofline`` (LOWER == better == closer to the floor). Uses sklearn's
    GradientBoostingRegressor when importable, else the k-NN fallback. ``trained`` is False when the
    corpus is too sparse, then :func:`rank` leaves candidate order untouched (pure exploration)."""

    def __init__(self, points: Sequence[CorpusPoint]):
        self.trained = False
        self.backend = "none"
        self._model = None
        usable = [p for p in points if math.isfinite(p.pct_of_roofline)]
        if len(usable) < MIN_ROWS_TO_RANK:
            return
        X = [_full_features(p.shape, p.schedule) for p in usable]
        y = [p.pct_of_roofline for p in usable]
        # Try sklearn (only if already installed, never a new dep).
        try:
            from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
            m = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08,
                                          subsample=0.9, random_state=0)
            m.fit(X, y)
            self._model = m
            self.backend = "sklearn-gbr"
            self.trained = True
            return
        except Exception:
            pass
        try:
            self._model = _KNNRanker(X, y, k=5)
            self.backend = "knn"
            self.trained = True
        except Exception:
            self.trained = False

    def score(self, shape: ModelShape, cfg_dict: dict[str, Any]) -> float:
        """Predicted pct_of_roofline for one candidate (lower better). +inf if untrained."""
        if not self.trained or self._model is None:
            return float("inf")
        feats = [_full_features(shape, cfg_dict)]
        try:
            pred = self._model.predict(feats)
            return float(pred[0])
        except Exception:
            return float("inf")


def make_ranker(gpu: str, corpus: list[dict[str, Any]] | str | None = None) -> FlywheelRanker:
    """Build (train) a :class:`FlywheelRanker` from the corpus for ``gpu``. Cheap; call once per
    loop and re-call to retrain as the corpus grows. Never raises."""
    try:
        points = load_points(corpus, gpu=gpu)
    except Exception:
        points = []
    return FlywheelRanker(points)


def rank(candidates: Sequence[ScheduleConfig | dict[str, Any]],
         model_shape: ModelShape, gpu: str,
         corpus: list[dict[str, Any]] | str | None = None,
         *, ranker: FlywheelRanker | None = None) -> list[ScheduleConfig | dict[str, Any]]:
    """Order ``candidates`` best-first by the learned prior's predicted ``pct_of_roofline`` so the
    autoresearch loop evaluates the most promising configs first (exploitation). When the corpus is
    too sparse to train (``ranker.trained`` is False), returns the candidates UNCHANGED, pure
    exploration, the honest behaviour when there is nothing to learn from yet.

    Accepts ScheduleConfigs or plain config dicts; returns the same objects, reordered."""
    r = ranker or make_ranker(gpu, corpus)
    if not r.trained or not candidates:
        return list(candidates)

    def _as_dict(c: ScheduleConfig | dict[str, Any]) -> dict[str, Any]:
        return c.to_dict() if isinstance(c, ScheduleConfig) else dict(c)

    scored = [(r.score(model_shape, _as_dict(c)), i, c) for i, c in enumerate(candidates)]
    # Sort by predicted pct_of_roofline asc; ties keep original order (stable via the index).
    scored.sort(key=lambda t: (t[0], t[1]))
    return [c for _s, _i, c in scored]


__all__ = [
    "ModelShape", "CorpusPoint", "FlywheelRanker",
    "warm_start", "warm_start_combined", "rank", "make_ranker", "load_points",
    "config_features", "KNOB_DEFAULTS",
    "SEARCHABLE_KNOB_CHOICES", "SEARCHABLE_KNOB_DEFAULTS",
    "MIN_ROWS_TO_RANK", "MIN_ROWS_TO_WARM",
]
