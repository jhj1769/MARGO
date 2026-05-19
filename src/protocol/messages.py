"""Pydantic schemas that constrain every inter-agent message in MARGO.

All agent outputs MUST validate against one of these models. Anything that
fails validation is logged toward the *Schema Violation Rate* (SVR) metric.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

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
    issued_at: float = Field(default_factory=time.time)
    iteration: int = 0


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
