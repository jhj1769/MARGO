"""Pydantic schemas that constrain every inter-agent message in MARGO.

All agent outputs MUST validate against one of these models. Anything that
fails validation is logged toward the *Schema Violation Rate* (SVR) metric.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Message envelope                                                            #
# --------------------------------------------------------------------------- #


class MessageType(str, Enum):
    """Inter-agent message types used by the LangGraph router."""

    DIRECTIVE = "directive"
    BROADCAST = "broadcast"
    NEGOTIATION = "negotiation"
    CONSULTATION = "consultation"
    COORDINATION = "coordination"
    RESULT = "result"
    VALIDATION = "validation"


class Message(BaseModel):
    """Envelope carried on the message bus.

    The ``payload`` should itself be a Pydantic model serialized via
    ``model_dump()`` so the receiver can re-validate before consumption.
    """

    type: MessageType
    sender: str
    receivers: list[str]
    payload: dict[str, Any]
    timestamp: float = Field(default_factory=time.time)
    trace_id: Optional[str] = None


# --------------------------------------------------------------------------- #
# Domain payloads                                                             #
# --------------------------------------------------------------------------- #


PolicyHint = Literal["daily", "trend_push", "cohort_expansion"]


class Directive(BaseModel):
    """Expert agent output (hybrid: structured + natural language).

    A ``Directive`` is the single source of truth that downstream agents
    must comply with during Phase 3 reasoning and that the Expert agent
    re-checks during Phase 4 validation.
    """

    goal: str = Field(..., description="High-level NL goal, e.g. 'casual→formal upsell'.")
    structured_constraints: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Machine-checkable constraints. Examples: "
            "{'price_diff_pct_max': 30, 'boost_category': 'trench-coat', "
            "'forbid_category': ['lingerie']}"
        ),
    )
    natural_language: str = Field(
        ..., description="Free-form NL phrasing of the directive."
    )
    # Phase C — drives the User Agent's dual-layer (Realistic vs Rejected)
    # weighting at evaluate time:
    #   'daily'             → strict avoidance of past rejected patterns (safe)
    #   'trend_push'        → soft avoidance; trend signal weighs more (riskier)
    #   'cohort_expansion'  → light avoidance; cohort peer signal dominates
    # ``None`` ⇒ legacy v3 behaviour (Rejected layer not applied), preserving
    # backward compatibility for callers that don't set the hint.
    policy_hint: Optional[PolicyHint] = None
    issued_at: float = Field(default_factory=time.time)
    iteration: int = 0
    # Enhancement 5: persistent record of any Expert↔Trend back-and-forth that
    # happened while issuing this directive. ``None`` when negotiation was
    # disabled or no tension was detected.
    negotiation_log: Optional["NegotiationLog"] = None


class TrendInterpretation(BaseModel):
    """Trend Agent output — already interpreted for the recommendation context."""

    domain: str
    time_window: str  # e.g. "2026-Q2"
    summary: str  # 1–2 sentence NL summary
    keywords: list[str] = Field(default_factory=list)
    rising_attributes: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-attribute rising values: {'color': [...], 'silhouette': [...]}",
    )
    raw_sources: list[str] = Field(default_factory=list)
    issued_at: float = Field(default_factory=time.time)


class Rationale(BaseModel):
    """3-layer rationale attached to every recommended item."""

    personal: str
    directive: str
    trend: str


class RankedItem(BaseModel):
    item_id: str
    score: float
    rationale: Rationale

    @model_validator(mode="after")
    def _bounded_score(self) -> "RankedItem":
        # Soft sanity check; scores from LLM judges sometimes drift.
        if not (-10.0 <= self.score <= 10.0):
            raise ValueError(f"RankedItem.score out of plausible range: {self.score}")
        return self


class RecommendationResult(BaseModel):
    """End-to-end recommendation output for a single (user, directive) pair."""

    user_id: str
    directive: Directive
    trend: Optional[TrendInterpretation] = None
    top_k: list[RankedItem]
    candidate_pool_size: int
    phase4_passed: bool
    iterations: int = 1
    trace: list[Message] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Expert agent's validation verdict over a Top-K list."""

    passed: bool
    compliance_score: float  # in [0, 1]
    violations: list[str] = Field(default_factory=list)
    suggested_refinement: Optional[str] = None


# --------------------------------------------------------------------------- #
# User preference state (Enhancement 1)                                       #
# --------------------------------------------------------------------------- #


AxisName = Literal[
    "style_preference",
    "price_preference",
    "category_preference",
    "brand_preference",
]


class PreferenceAxis(BaseModel):
    """One of the 4 explicit axes of a user's preference state.

    Price/category/brand axes are computed deterministically from history
    (``derived_from="statistical"``). Style is LLM-inferred from item
    descriptions (``derived_from="llm_inferred"``).
    """

    name: AxisName
    value: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    secondary_values: list[str] = Field(default_factory=list)
    derived_from: Literal["statistical", "llm_inferred"]
    stability: float = Field(default=1.0, ge=0.0, le=1.0)


class UserPreferenceState(BaseModel):
    """Backward-compatible extension of the user's profile.

    The legacy NL ``profile`` string stays in ``UserState``; this object adds
    structured axes and the cohort signature used by Enhancement 1.5.
    """

    user_id: str
    profile_nl: str
    axes: list[PreferenceAxis] = Field(default_factory=list)
    cohort_signature: str = ""
    last_updated_at: float = Field(default_factory=time.time)

    def get_axis(self, name: AxisName) -> Optional[PreferenceAxis]:
        return next((a for a in self.axes if a.name == name), None)


def compute_cohort_signature(state: "UserPreferenceState") -> str:
    """Order-invariant cohort signature derived from all known axes.

    All axes with a non-empty, non-``"unknown"`` value contribute — including
    LLM-inferred style. Style is a meaningful behavioral dimension and
    dropping it from the signature merges users with very different aesthetic
    preferences into the same cohort, which dilutes the peer signal.

    Consistency requirement: the offline cohort builder MUST populate the
    same axes the runtime UserAgent populates. If runtime adds style but the
    offline build was statistical-only, runtime signatures will never match
    any saved cohort. ``scripts/build_cohort_stats.py`` accepts a
    pre-computed user-states JSONL so the offline pipeline can mirror
    runtime's 4-axis output.

    Example output::

        "bra:brand-diverse|cat:balanced|pri:mid-tier|sty:minimal-casual"
    """
    parts: list[tuple[str, str]] = []
    for axis in state.axes:
        if not axis.value or axis.value == "unknown":
            continue
        parts.append((axis.name[:3], axis.value))
    parts.sort()
    return "|".join(f"{prefix}:{value}" for prefix, value in parts)


class CohortStats(BaseModel):
    """Aggregated peer-level statistics for one cohort signature.

    Populated offline by ``scripts/build_cohort_stats.py``. Cohorts with fewer
    than ``MIN_COHORT_SIZE`` users are not emitted — the loader returns
    ``None`` for them, which the agent treats as "no reliable peer signal".
    """

    signature: str
    size: int
    user_ids: list[str] = Field(default_factory=list)
    item_buy_ratios: dict[str, float] = Field(default_factory=dict)
    top_categories: list[tuple[str, float]] = Field(default_factory=list)
    top_brands: list[tuple[str, float]] = Field(default_factory=list)

    def peer_signal_for(self, item_id: str) -> float:
        """Ratio (0.0 if not in cohort) of cohort members who bought ``item_id``."""
        return self.item_buy_ratios.get(item_id, 0.0)


# --------------------------------------------------------------------------- #
# Item audience + trend position (Enhancements 2 + 3)                         #
# --------------------------------------------------------------------------- #


class ItemAudienceProfile(BaseModel):
    """Aggregated buyer-side facts for one catalog item.

    Built offline by ``scripts/build_buyer_aggregate.py`` from past positive
    interactions (rating ≥ 4). When ``buyer_cohort_distribution`` is populated
    the Item Agent can claim "who I serve well" with cohort grounding.
    """

    item_id: str
    buyer_count: int
    avg_price_history: Optional[float] = None
    median_history_length: Optional[int] = None
    category_distribution: dict[str, float] = Field(default_factory=dict)
    brand_distribution: dict[str, float] = Field(default_factory=dict)
    # Populated only when user states (with cohort signatures) were available
    # during offline build. Empty dict otherwise.
    buyer_cohort_distribution: dict[str, int] = Field(default_factory=dict)
    dominant_cohorts: list[str] = Field(default_factory=list)
    evidence_buyer_ids: list[str] = Field(default_factory=list)
    dominant_pattern_nl: Optional[str] = None
    outlier_pattern_nl: Optional[str] = None


TrendLifecycle = Literal["rising", "stable", "declining", "niche"]
TrendAlignment = Literal["aligned", "orthogonal", "counter"]


class TrendPosition(BaseModel):
    """How an item positions itself relative to the current trend signal.

    Emitted by the Item Agent in Phase 3. The optional NL value proposition
    is REQUIRED when the item is on the wrong side of the trend
    (declining lifecycle or counter alignment) — that's where the agent has
    to argue its case explicitly.
    """

    lifecycle: TrendLifecycle
    alignment: TrendAlignment
    value_proposition_nl: Optional[str] = None

    @model_validator(mode="after")
    def _require_justification_for_counter_signals(self) -> "TrendPosition":
        if self.lifecycle == "declining" or self.alignment == "counter":
            if not self.value_proposition_nl:
                raise ValueError(
                    "value_proposition_nl is required when "
                    "lifecycle='declining' or alignment='counter'"
                )
        return self


class ItemDescription(BaseModel):
    """Item Agent's Phase-3 output, fed into User Agent evaluation.

    The legacy ``description`` field is the single NL string the User Agent
    actually reads. The richer audience-grounded and trend-grounded fields
    are *additions* the prompt is encouraged to fill in when context permits —
    they do not replace the headline description.
    """

    description: str
    audience_fit: float = Field(ge=0.0, le=1.0)
    anchored_attributes: list[str] = Field(default_factory=list)
    # Enhancement 2 — audience grounding
    audience_fit_claim_nl: Optional[str] = None
    outlier_note_nl: Optional[str] = None
    # Enhancement 3 — trend self-positioning
    trend_position: Optional[TrendPosition] = None


# --------------------------------------------------------------------------- #
# Expert ↔ Trend negotiation (Enhancement 5)                                  #
# --------------------------------------------------------------------------- #


TensionType = Literal["attribute_conflict", "price_conflict", "audience_conflict"]
NegotiationActor = Literal["expert", "trend"]
NegotiationMessageType = Literal["challenge", "accept", "reject", "counter"]
NegotiationOutcome = Literal["consensus", "expert_held", "max_turns_reached"]


class TrendDirectiveTension(BaseModel):
    """A single point of disagreement between trend interpretation and directive."""

    type: TensionType
    severity: float = Field(..., ge=0.0, le=1.0)
    description_nl: str
    directive_element: str  # which part of the directive is in tension
    trend_element: str      # which trend signal contests it


class NegotiationMessage(BaseModel):
    """One turn in the Expert↔Trend back-and-forth.

    ``proposed_directive_delta`` is a JSON-patch-like dict applied via
    :meth:`ExpertAgent.apply_directive_delta`. Keys map to ``Directive``
    fields; values are the replacement (NOT a merge).
    """

    turn: int
    from_agent: NegotiationActor
    to_agent: NegotiationActor
    message_type: NegotiationMessageType
    content_nl: str
    tensions: list[TrendDirectiveTension] = Field(default_factory=list)
    proposed_directive_delta: Optional[dict] = None


class NegotiationLog(BaseModel):
    """Audit trail of one negotiation pass — empty list when no tension fired."""

    messages: list[NegotiationMessage] = Field(default_factory=list)
    final_outcome: NegotiationOutcome = "consensus"


# Resolve forward-reference: Directive declared ``negotiation_log: Optional["NegotiationLog"]``
# before NegotiationLog was defined, so we rebuild the model now.
Directive.model_rebuild()
