"""
AMK proposer seam (public contract)
===================================

The propose -> eval -> keep loop (``autoresearch.py``) emits the next candidate
``(ScheduleConfig, kernel_knobs)`` to evaluate. By default that candidate comes
from AMK's built-in epsilon-greedy search over the public schedule x kernel-knob
space. This module defines an OPTIONAL hook so an external package can supply a
smarter candidate generator WITHOUT forking AMK::

    from autoresearch import autoresearch
    autoresearch(model, gpu, proposer=MyProposer())

The proposer is consulted once per NORMAL search iteration; AMK still owns the
iteration-0 seed candidate and the overnight basin-hop restarts, so a proposer does
not own 100% of candidate generation. Returning None defers that iteration to AMK.

A proposer ONLY chooses which point in the (already public) search space to try
next. It cannot bypass the gate: every candidate it returns still flows through
the same ``lower -> validate -> reference-oracle correctness check -> measured
keep/revert -> roofline floor``. A bad proposal can only ever lose the
keep/revert; it can never corrupt the search or produce a dishonest number.

The default is ``proposer=None``, which preserves AMK's built-in behavior
exactly (byte-identical), so this seam adds no behavior and no risk unless a
caller opts in.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from schedule.ir import GpuTarget, ScheduleConfig


@dataclass(frozen=True)
class ProposalContext:
    """Everything a proposer needs to choose the next candidate.

    Every field is an already-public concept: the current incumbent, the target
    GPU spec, the public knob space, the RNG, and the set of already-tried
    candidate ids. No proprietary content crosses this boundary.
    """

    iter_idx: int
    cold: bool
    incumbent_cfg: ScheduleConfig
    incumbent_knobs: dict[str, int]
    target: GpuTarget
    model_shape: Any
    gpu: str
    corpus_path: str
    tried: frozenset[str]
    rng: random.Random
    knob_choices: dict[str, tuple[int, ...]]
    knob_defaults: dict[str, int]
    # Extra schedule-side lever the loop also sweeps (GEMV tile width; 0 == auto).
    n_tile_choices: tuple[int, ...] = ()
    # Canonical id of a (cfg, knobs) candidate matching the loop's dedupe key, so
    # a proposer can avoid re-emitting an already-tried point. None outside the loop.
    candidate_id: Callable[[ScheduleConfig, dict[str, int]], str] | None = None


@dataclass(frozen=True)
class Proposal:
    """A single candidate point in the public (schedule x kernel_knobs) space."""

    config: ScheduleConfig
    kernel_knobs: dict[str, int]
    source: str = "external"


@runtime_checkable
class Proposer(Protocol):
    """Callable returning the next :class:`Proposal`, or ``None`` to defer to
    AMK's built-in search for this iteration."""

    def __call__(self, ctx: ProposalContext) -> Proposal | None: ...


__all__ = ["ProposalContext", "Proposal", "Proposer"]
