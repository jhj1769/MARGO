"""Enhancements 2 + 3 — audience profile + trend self-positioning."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MARGO_LLM_BACKEND", "dummy")

import pytest
from pydantic import ValidationError

from adapters.llm import LLMClient
from core.agents.item_agent import ItemAgent, ItemFacts
from core.protocol.messages import (
    Directive,
    ItemAudienceProfile,
    ItemDescription,
    TrendInterpretation,
    TrendPosition,
)
from data.fashion.audience_loader import (
    MIN_BUYERS,
    build_audience_profiles,
    load_audience_profile,
    save_audience_profiles,
)


# --------------------------------------------------------------------------- #
# Schema-level tests                                                          #
# --------------------------------------------------------------------------- #


def test_trend_position_requires_justification_for_declining():
    """A declining or counter position must justify itself."""
    with pytest.raises(ValidationError):
        TrendPosition(lifecycle="declining", alignment="aligned")
    with pytest.raises(ValidationError):
        TrendPosition(lifecycle="rising", alignment="counter")
    # With justification → OK
    ok = TrendPosition(
        lifecycle="declining", alignment="aligned",
        value_proposition_nl="Still valued by nostalgic buyers",
    )
    assert ok.value_proposition_nl


def test_trend_position_enum_values_only():
    """Invalid lifecycle / alignment strings are rejected."""
    with pytest.raises(ValidationError):
        TrendPosition(lifecycle="exploding", alignment="aligned")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TrendPosition(lifecycle="rising", alignment="sideways")  # type: ignore[arg-type]


def test_item_description_is_backward_compatible():
    """The legacy core fields (description / audience_fit / anchored) must still parse."""
    d = ItemDescription(description="a tee", audience_fit=0.5)
    assert d.description == "a tee"
    assert d.trend_position is None
    assert d.audience_fit_claim_nl is None


# --------------------------------------------------------------------------- #
# Builder tests                                                               #
# --------------------------------------------------------------------------- #


def _toy_inputs():
    """Minimal corpus: 2 items, 4 buyers each (above MIN_BUYERS)."""
    buyers_by_item = {
        "A": ["u1", "u2", "u3", "u4"],
        "B": ["u1", "u2", "u5", "u6"],
        "Z": ["u7"],  # below threshold → dropped
    }
    user_histories = {
        "u1": ["A", "B"],
        "u2": ["A", "B"],
        "u3": ["A"],
        "u4": ["A"],
        "u5": ["B"],
        "u6": ["B"],
        "u7": ["Z"],
    }
    item_prices = {"A": 50.0, "B": 80.0, "Z": 120.0}
    item_categories = {"A": "t-shirts", "B": "t-shirts", "Z": "coats"}
    item_brands = {"A": "BrandX", "B": "BrandY", "Z": "BrandZ"}
    return buyers_by_item, user_histories, item_prices, item_categories, item_brands


def test_build_audience_profiles_drops_below_threshold():
    buyers, hist, prices, cats, brands = _toy_inputs()
    profiles = build_audience_profiles(
        item_ids=list(buyers.keys()),
        buyers_by_item=buyers,
        user_histories=hist,
        item_prices=prices,
        item_categories=cats,
        item_brands=brands,
        min_buyers=MIN_BUYERS,
    )
    assert set(profiles) == {"A", "B"}
    assert "Z" not in profiles


def test_audience_profile_contains_expected_aggregates():
    buyers, hist, prices, cats, brands = _toy_inputs()
    profiles = build_audience_profiles(
        item_ids=list(buyers.keys()),
        buyers_by_item=buyers,
        user_histories=hist,
        item_prices=prices,
        item_categories=cats,
        item_brands=brands,
        min_buyers=MIN_BUYERS,
    )
    a = profiles["A"]
    assert a.buyer_count == 4
    # All buyer histories contain only t-shirts in their seen items
    assert "t-shirts" in a.category_distribution
    assert a.category_distribution["t-shirts"] == pytest.approx(1.0)
    # Brand distribution mixes X (from A) and Y (from B) since buyers also bought B
    assert "BrandX" in a.brand_distribution
    # avg_price_history should be in the toy price range
    assert 0 < (a.avg_price_history or 0) < 200


def test_audience_profile_with_cohort_signatures():
    """When user cohort signatures are supplied, dominant_cohorts is populated."""
    buyers, hist, prices, cats, brands = _toy_inputs()
    user_cohort_signatures = {
        "u1": "sig-mid",
        "u2": "sig-mid",
        "u3": "sig-mid",
        "u4": "sig-premium",
        "u5": "sig-mid",
        "u6": "sig-mid",
    }
    profiles = build_audience_profiles(
        item_ids=["A"],
        buyers_by_item=buyers,
        user_histories=hist,
        item_prices=prices,
        item_categories=cats,
        item_brands=brands,
        user_cohort_signatures=user_cohort_signatures,
        min_buyers=MIN_BUYERS,
    )
    a = profiles["A"]
    assert "sig-mid" in a.dominant_cohorts
    assert a.buyer_cohort_distribution["sig-mid"] == 3
    assert a.buyer_cohort_distribution["sig-premium"] == 1


def test_audience_save_load_roundtrip(tmp_path: Path):
    buyers, hist, prices, cats, brands = _toy_inputs()
    profiles = build_audience_profiles(
        item_ids=["A", "B"],
        buyers_by_item=buyers,
        user_histories=hist,
        item_prices=prices,
        item_categories=cats,
        item_brands=brands,
        min_buyers=MIN_BUYERS,
    )
    save_audience_profiles(profiles, tmp_path)
    loaded = load_audience_profile("A", tmp_path)
    assert loaded is not None
    assert loaded.buyer_count == 4
    assert load_audience_profile("UNKNOWN_ID", tmp_path) is None


# --------------------------------------------------------------------------- #
# ItemAgent integration                                                       #
# --------------------------------------------------------------------------- #


def _directive() -> Directive:
    return Directive(
        goal="t",
        structured_constraints={},
        natural_language="upsell to dressy",
    )


def _trend() -> TrendInterpretation:
    return TrendInterpretation(
        domain="fashion",
        time_window="2026-Q2",
        summary="minimalism rising",
        keywords=["minimal", "tailored"],
    )


def test_item_agent_self_describes_without_audience_or_trend():
    """Backward-compat: no audience, no trend → description still produced."""
    facts = ItemFacts(item_id="A", title="white tee", attributes={"category": "tops", "price": 25.0})
    agent = ItemAgent(facts, llm=LLMClient(backend="dummy"))
    out = agent.self_describe(
        user_profile="casual shopper",
        directive=_directive(),
        trend=None,
    )
    assert isinstance(out, ItemDescription)
    assert out.description
    # No trend signal → no trend_position required; dummy may emit one but it should validate.


def test_item_agent_logs_audience_grounding_flag():
    """When audience is provided the agent records that it was available."""
    facts = ItemFacts(item_id="A", title="white tee", attributes={"category": "tops", "price": 25.0})
    audience = ItemAudienceProfile(
        item_id="A", buyer_count=10,
        avg_price_history=30.0, median_history_length=8,
        category_distribution={"t-shirts": 0.8, "jeans": 0.2},
        brand_distribution={"BrandX": 0.6},
        dominant_cohorts=["sig-mid"],
    )
    agent = ItemAgent(facts, llm=LLMClient(backend="dummy"))
    agent.self_describe(
        user_profile="casual shopper",
        directive=_directive(),
        trend=_trend(),
        audience_profile=audience,
    )
    assert agent.audience_log
    assert agent.audience_log[-1]["audience_grounded"] is True


def test_item_agent_self_describes_with_full_context():
    """When all inputs are present the call must succeed and return a valid ItemDescription."""
    facts = ItemFacts(item_id="A", title="white tee", attributes={"category": "tops", "price": 25.0})
    audience = ItemAudienceProfile(
        item_id="A", buyer_count=10,
        category_distribution={"t-shirts": 0.8},
        brand_distribution={"BrandX": 0.6},
        dominant_cohorts=["sig-mid"],
    )
    agent = ItemAgent(facts, llm=LLMClient(backend="dummy"))
    out = agent.self_describe(
        user_profile="casual shopper",
        directive=_directive(),
        trend=_trend(),
        audience_profile=audience,
    )
    assert isinstance(out, ItemDescription)
    # Validates the schema as a whole — even with dummy backend the
    # round-trip through pydantic must work.
    assert 0.0 <= out.audience_fit <= 1.0
