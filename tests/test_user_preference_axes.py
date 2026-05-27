"""Enhancement 1 — deterministic axes + UserAgent integration."""

from __future__ import annotations

import os

os.environ.setdefault("MARGO_LLM_BACKEND", "dummy")

import pandas as pd
import pytest
from pydantic import ValidationError

from adapters.llm import LLMClient
from core.agents.user_agent import UserAgent
from core.protocol.messages import PreferenceAxis, UserPreferenceState
from data.fashion.preference_stats import (
    compute_brand_axis,
    compute_category_axis,
    compute_deterministic_axes,
    compute_price_axis,
    tier_for_price,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _focused_catalog() -> pd.DataFrame:
    """8 t-shirts from BrandX at $25-50 — should yield t-shirts-focused/brand-loyal/mid-tier."""
    return pd.DataFrame({
        "parent_asin": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "price": [25.0, 28.0, 32.0, 35.0, 40.0, 45.0, 28.0, 50.0],
        "brand": ["BrandX"] * 7 + ["BrandY"],
        "categories": [
            ["Clothing", "Women", "Tops", "T-shirts"],
        ] * 6 + [
            ["Clothing", "Women", "Tops", "Blouses"],
            ["Clothing", "Women", "Dresses", "Mini-Dresses"],
        ],
    })


def _diverse_catalog() -> pd.DataFrame:
    """12 items: 4 brands × 3 categories with very wide price band.

    Brand share = 3/12 = 25% each (under 30% diversity threshold).
    Category share = 4/12 ≈ 33% (no single category exceeds 50% focus threshold).
    Price spans $15–$250 → high coefficient of variation → low confidence.
    """
    rows = []
    brands = ["BrandA", "BrandB", "BrandC", "BrandD"]
    cats = [
        ["Clothing", "Women", "Tops", "T-shirts"],
        ["Clothing", "Women", "Bottoms", "Jeans"],
        ["Clothing", "Women", "Outerwear", "Coats"],
    ]
    prices = [15.0, 80.0, 200.0, 25.0, 120.0, 18.0, 250.0, 65.0, 35.0, 220.0, 22.0, 95.0]
    for i in range(12):
        rows.append({
            "parent_asin": f"D{i:02d}",
            "price": prices[i],
            "brand": brands[i % 4],
            "categories": cats[i % 3],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_deterministic_axes_computation():
    """Focused catalogue → expected statistical tiers and labels."""
    items = _focused_catalog()
    axes = compute_deterministic_axes("u1", list(items["parent_asin"]), items)

    assert axes["price_preference"]["value"] == "mid-tier"
    assert axes["price_preference"]["derived_from"] == "statistical"
    assert axes["price_preference"]["confidence"] > 0.5

    assert axes["category_preference"]["value"].endswith("-focused")
    assert "t-shirts" in axes["category_preference"]["value"]

    assert axes["brand_preference"]["value"].startswith("brand-loyal:")
    assert "BrandX" in axes["brand_preference"]["value"]


def test_diverse_user_lower_confidence():
    """High variance / spread-out users should NOT get a single-label forced on them."""
    items = _diverse_catalog()
    axes = compute_deterministic_axes("u2", list(items["parent_asin"]), items)

    # Price spans $15–$250 → confidence should be visibly lower than a focused user.
    assert axes["price_preference"]["confidence"] < 0.5

    # Three categories at 3/3/2 → mix or balanced, not single-focused.
    cat_value = axes["category_preference"]["value"]
    assert "-focused" not in cat_value
    assert cat_value in {"balanced", "t-shirts-jeans-mix", "t-shirts-jeans-coats-mix"} or "mix" in cat_value

    # Three brands evenly → brand-diverse.
    assert axes["brand_preference"]["value"] == "brand-diverse"


def test_axis_schema_validation():
    """PreferenceAxis must reject confidence outside [0, 1] and unknown axis names."""
    with pytest.raises(ValidationError):
        PreferenceAxis(
            name="price_preference",
            value="mid-tier",
            confidence=1.5,
            derived_from="statistical",
        )
    with pytest.raises(ValidationError):
        PreferenceAxis(
            name="not_an_axis",  # type: ignore[arg-type]
            value="x",
            confidence=0.5,
            derived_from="statistical",
        )


def test_update_preference_state_picks_up_new_interaction():
    """update_preference_state must re-run deterministic stats on the new history."""
    items = _focused_catalog()
    llm = LLMClient(backend="dummy")
    user = UserAgent(
        user_id="u1",
        history=["t-shirt $25", "t-shirt $28"],
        history_item_ids=["A", "B"],
        items_df=items,
        llm=llm,
    )
    user.build_profile()
    assert user.state.preference_state is not None
    initial = user.state.preference_state.get_axis("brand_preference")
    assert initial is not None
    assert "BrandX" in initial.value  # both items are BrandX

    # Add a BrandY item (id H). Should now show some BrandY influence
    # (loyalty stays since 2/3 = BrandX, but evidence updates).
    user.update_preference_state(new_item_id="H", rating=5.0)
    updated = user.state.preference_state.get_axis("brand_preference")
    assert updated is not None
    # Evidence list should now mention BrandY too.
    evidence = " ".join(updated.evidence)
    assert "BrandY" in evidence


def test_stability_score_when_recent_shifts():
    """A history whose full-window label differs from the recent-window label
    should yield brand stability < 1.0.

    Construction: 20 items. First 12 are BrandY, last 8 (= the recent window)
    are BrandX. Full → loyal BrandY (12/20=60%). Recent 10 → 8 BrandX + 2 BrandY → loyal BrandX.
    """
    brands_in_order = ["BrandY"] * 12 + ["BrandX"] * 8
    rows = [
        {
            "parent_asin": f"I{i:02d}",
            "price": 30.0 + (i % 3),
            "brand": b,
            "categories": ["Clothing", "Women", "Tops", "T-shirts"],
        }
        for i, b in enumerate(brands_in_order)
    ]
    items = pd.DataFrame(rows)
    history_ids = list(items["parent_asin"])

    axes = compute_deterministic_axes("u3", history_ids, items)
    brand = axes["brand_preference"]
    # Sanity-check the labels really do disagree.
    assert "BrandY" in brand["value"], brand["value"]
    # Recent-window label is BrandX (we verified above the construction).
    assert brand["stability"] < 1.0, f"expected drift, got stability={brand['stability']}"


def test_price_tier_boundaries():
    """tier_for_price is a step function — boundaries are inclusive on the low side."""
    assert tier_for_price(0) == "budget"
    assert tier_for_price(29.99) == "budget"
    assert tier_for_price(30.0) == "mid-tier"
    assert tier_for_price(99.99) == "mid-tier"
    assert tier_for_price(100.0) == "premium-open"
    assert tier_for_price(299.99) == "premium-open"
    assert tier_for_price(300.0) == "luxury-aware"
    assert tier_for_price(9999.0) == "luxury-aware"


def test_empty_history_returns_unknown_axes():
    """No purchase history → axes labelled 'unknown' with confidence 0."""
    empty_items = pd.DataFrame(columns=["parent_asin", "price", "brand", "categories"])
    axes = compute_deterministic_axes("u4", [], empty_items)
    for name in ("price_preference", "category_preference", "brand_preference"):
        assert axes[name]["value"] == "unknown"
        assert axes[name]["confidence"] == 0.0


def test_user_agent_falls_back_without_catalog():
    """Old call-site without items_df should still build a profile and skip axes."""
    llm = LLMClient(backend="dummy")
    user = UserAgent(
        user_id="u_legacy",
        history=["bought tee | $25"],
        llm=llm,
    )
    user.build_profile()
    # Legacy path: NL profile present, but structured state remains None.
    assert user.state.profile
    assert user.state.preference_state is None
