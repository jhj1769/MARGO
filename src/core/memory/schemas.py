"""Per-agent typed memory event schemas + memory store wrappers.

Each stakeholder has its own event types and memory store. The shared
``AgentMemory`` protocol (``base.py``) keeps the orchestrator side
uniform; this module provides the typed wrappers for caller convenience
plus per-agent retrieval filters.

Design decisions
----------------

* **Helper functions** (``make_*_event``) wrap typed payloads into
  ``MemoryEvent`` envelopes so callers never write event_type/timestamp
  literals by hand — typos can't silently misclassify an event.

* **Optional verification fields** on prediction events (Item audience
  claims, Trend predictions). Events are written *without* verification
  data; a separate offline job fills them in later. This is what makes
  the "Self-Correction Precision" and "Trend Predictive Validity"
  metrics computable from the same log.

* **Cohort-aware filter** on ItemMemory is the *one* schema-specific
  convenience that lives here — the User Agent always passes the
  viewer's cohort_signature when retrieving an item's reception slice.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.memory.base import AgentMemory, MemoryEvent  # noqa: F401  (re-export)
from core.memory.persistence import JSONLMemoryStore


# --------------------------------------------------------------------------- #
# User Agent — Trajectory + Dual-Layer Calibration                            #
# --------------------------------------------------------------------------- #


class UserMemoryEvent(BaseModel):
    """Typed payload for a User Agent memory event.

    Per-event subset of fields is populated; ``event_type`` (set on the
    enclosing :class:`MemoryEvent`) tells the reader which subset to expect:

    * ``axis_snapshot`` — periodic snapshot of the 4 axes. Populates
      ``snapshot_axes``.
    * ``evaluation_calibration`` — a past evaluation + its actual outcome
      (purchased / not). Populates ``candidate_item_id``,
      ``predicted_score``, ``actual_outcome``.
    * ``rejection_pattern_inferred`` — summary of derived rejected pattern.
      Populates ``rejection_pattern_summary``.
    """

    user_id: str
    snapshot_axes: Optional[dict[str, str]] = None      # axis_name → value
    candidate_item_id: Optional[str] = None
    predicted_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    actual_outcome: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rejection_pattern_summary: Optional[dict[str, Any]] = None


def make_user_event(event_type: str, *, user_id: str, **payload: Any) -> MemoryEvent:
    """Wrap a typed ``UserMemoryEvent`` into a ``MemoryEvent`` envelope."""
    typed = UserMemoryEvent(user_id=user_id, **payload)
    return MemoryEvent(
        event_type=event_type,
        timestamp=time.time(),
        payload=typed.model_dump(exclude_none=True),
    )


class UserMemory(JSONLMemoryStore):
    """File-backed UserMemory at ``<root>/user/<user_id>.jsonl``."""

    def __init__(self, user_id: str, root: Path | str) -> None:
        path = Path(root) / "user" / f"{user_id}.jsonl"
        super().__init__(path)
        self.user_id = user_id


# --------------------------------------------------------------------------- #
# Item Agent — Autobiographical Memory                                         #
# --------------------------------------------------------------------------- #


class ItemReceptionEvent(BaseModel):
    """One reception received by an item (after a recommend call).

    This is the building block of Item autobiography. Aggregated by
    (viewer_cohort × month) in compaction (see
    ``scripts/compact_item_memory.py``) so memory stays bounded.
    """

    item_id: str
    viewer_user_id: str
    viewer_cohort: str
    score: float = Field(ge=0.0, le=1.0)
    rationale_summary: Optional[str] = None
    turn_id: Optional[str] = None


class ItemAudienceClaimEvent(BaseModel):
    """An audience claim made by an item + later verification.

    Written *unverified* during ``self_describe``. The verification fields
    are filled in by a post-hoc evaluator that compares the claim's target
    cohort against later receptions. ``Self-Correction Precision`` is then
    derived from the rolling accuracy.
    """

    item_id: str
    claim_nl: str
    claim_target_cohort: str
    issued_turn_id: Optional[str] = None
    # Filled in later by post-hoc verification:
    verified_at: Optional[float] = None
    verified_avg_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    verified_n: Optional[int] = Field(default=None, ge=0)


def make_item_reception_event(**fields: Any) -> MemoryEvent:
    typed = ItemReceptionEvent(**fields)
    return MemoryEvent(
        event_type="item_reception",
        timestamp=time.time(),
        payload=typed.model_dump(exclude_none=True),
    )


def make_item_audience_claim_event(**fields: Any) -> MemoryEvent:
    typed = ItemAudienceClaimEvent(**fields)
    return MemoryEvent(
        event_type="item_audience_claim",
        timestamp=time.time(),
        payload=typed.model_dump(exclude_none=True),
    )


def _item_cohort_filter(event: MemoryEvent, query: dict[str, Any]) -> bool:
    """Keep events whose target cohort matches the viewer's cohort.

    Applies to both reception (where the cohort is the viewer's) and
    audience-claim events (where the cohort is the claim's target).
    A missing ``viewer_cohort`` in the query disables the filter so the
    caller can also retrieve unconditional slices.
    """
    viewer_cohort = query.get("viewer_cohort")
    if viewer_cohort is None:
        return True
    payload = event.payload
    return (
        payload.get("viewer_cohort") == viewer_cohort
        or payload.get("claim_target_cohort") == viewer_cohort
    )


class ItemMemory(JSONLMemoryStore):
    """File-backed ItemMemory at ``<root>/item/<item_id>.jsonl``.

    The cohort-aware filter is wired so that
    ``retrieve(query={"viewer_cohort": "..."})`` returns only the slice
    relevant to the current viewer — exactly what the Item Agent's
    self_describe needs (and nothing more, to keep LLM context tight).
    """

    def __init__(self, item_id: str, root: Path | str) -> None:
        path = Path(root) / "item" / f"{item_id}.jsonl"
        super().__init__(path, filter_fn=_item_cohort_filter)
        self.item_id = item_id


# --------------------------------------------------------------------------- #
# Trend Agent — Prediction History (Phase E uses this for Predictive Validity) #
# --------------------------------------------------------------------------- #


class TrendPredictionEvent(BaseModel):
    """A trend lifecycle prediction made at time t.

    Verified later by comparing against catalog activity in
    ``[t, t+N weeks]`` — that comparison populates ``actual_lift_pct``,
    which Phase E rolls up into Trend Predictive Validity (TPC).
    """

    snapshot_id: str
    keyword: str
    predicted_lifecycle: str  # 'rising' | 'stable' | 'declining' | 'niche'
    catalog_attributes: list[str] = Field(default_factory=list)
    n_matched_items: int = 0
    # Filled in later by post-hoc evaluation:
    verified_at: Optional[float] = None
    actual_lift_pct: Optional[float] = None  # +X% positive rating count in next N weeks


def make_trend_prediction_event(**fields: Any) -> MemoryEvent:
    typed = TrendPredictionEvent(**fields)
    return MemoryEvent(
        event_type="trend_prediction",
        timestamp=time.time(),
        payload=typed.model_dump(exclude_none=True),
    )


class TrendMemory(JSONLMemoryStore):
    """File-backed TrendMemory at ``<root>/trend/<domain>.jsonl``."""

    def __init__(self, domain: str, root: Path | str) -> None:
        path = Path(root) / "trend" / f"{domain}.jsonl"
        super().__init__(path)
        self.domain = domain


# --------------------------------------------------------------------------- #
# Expert Agent — Directive Outcome Pattern Learning                            #
# --------------------------------------------------------------------------- #


class ExpertDirectiveOutcomeEvent(BaseModel):
    """One Directive's outcome — used for similar-brief retrieval.

    ``brief_summary`` is what later retrievals match against (cosine
    similarity over TF-IDF). Keep it short and dense so similar briefs
    actually surface in retrieval.
    """

    brief_summary: str
    directive_goal: str
    directive_constraints_summary: dict[str, Any] = Field(default_factory=dict)
    n_refinements: int = Field(ge=0)
    final_compliance_score: float = Field(ge=0.0, le=1.0)
    passed: bool


def make_expert_outcome_event(**fields: Any) -> MemoryEvent:
    typed = ExpertDirectiveOutcomeEvent(**fields)
    return MemoryEvent(
        event_type="directive_outcome",
        timestamp=time.time(),
        payload=typed.model_dump(exclude_none=True),
    )


class ExpertMemory(JSONLMemoryStore):
    """File-backed ExpertMemory at ``<root>/expert/<persona_id>.jsonl``."""

    def __init__(self, persona_id: str, root: Path | str) -> None:
        path = Path(root) / "expert" / f"{persona_id}.jsonl"
        super().__init__(path)
        self.persona_id = persona_id
