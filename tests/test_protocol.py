"""Schema sanity for the inter-agent message protocol."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.protocol import (
    Directive,
    Message,
    MessageType,
    RankedItem,
    Rationale,
    RecommendationResult,
    TrendInterpretation,
)


def test_directive_roundtrip():
    d = Directive(
        goal="casual→formal upsell",
        structured_constraints={"price_diff_pct_max": 30, "boost_category": "trench"},
        natural_language="boost trench coats while keeping price gap under 30%.",
    )
    again = Directive.model_validate(d.model_dump())
    assert again.goal == d.goal
    assert again.structured_constraints["price_diff_pct_max"] == 30


def test_ranked_item_rejects_absurd_score():
    with pytest.raises(ValidationError):
        RankedItem(
            item_id="X",
            score=999.0,
            rationale=Rationale(personal="p", directive="d", trend="t"),
        )


def test_message_envelope_carries_payload():
    payload = TrendInterpretation(
        domain="fashion", time_window="2026-Q2", summary="x"
    ).model_dump()
    msg = Message(
        type=MessageType.BROADCAST,
        sender="trend",
        receivers=["user:*"],
        payload=payload,
    )
    assert msg.payload["domain"] == "fashion"


def test_recommendation_result_assembly():
    d = Directive(goal="g", natural_language="x")
    item = RankedItem(item_id="A", score=0.5, rationale=Rationale(personal="p", directive="d", trend="t"))
    res = RecommendationResult(
        user_id="u1",
        directive=d,
        top_k=[item],
        candidate_pool_size=100,
        phase4_passed=True,
    )
    assert res.top_k[0].item_id == "A"
    assert res.candidate_pool_size == 100
