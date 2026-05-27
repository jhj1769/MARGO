"""Phase C — User Agent Dual-Layer (Trajectory + Rejected) tests.

Verifies:
  * Directive.policy_hint accepts valid Literals + None
  * load_rejected_history: missing file → {}, populated → user_id mapping
  * summarise_rejection_pattern: cold start (None), populated (dict)
  * UserAgent.get_rejection_pattern: cached, None for users with no rejected
  * UserAgent backward compat (no rejected_item_ids → behaves as v3)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

os.environ.setdefault("MARGO_LLM_BACKEND", "dummy")

from core.agents.user_agent import UserAgent
from core.protocol.messages import Directive
from data.fashion.rejected_loader import load_rejected_history
from data.fashion.rejection_pattern import summarise_rejection_pattern


# --------------------------------------------------------------------------- #
# Directive.policy_hint                                                       #
# --------------------------------------------------------------------------- #


def test_directive_policy_hint_defaults_to_none():
    d = Directive(goal="g", natural_language="nl")
    assert d.policy_hint is None  # backward compat — legacy callers unaffected


def test_directive_policy_hint_accepts_all_three_modes():
    for hint in ("daily", "trend_push", "cohort_expansion"):
        d = Directive(goal="g", natural_language="nl", policy_hint=hint)
        assert d.policy_hint == hint


def test_directive_policy_hint_rejects_unknown_values():
    with pytest.raises(ValidationError):
        Directive(goal="g", natural_language="nl", policy_hint="strict")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# load_rejected_history                                                       #
# --------------------------------------------------------------------------- #


def test_load_rejected_history_missing_file_returns_empty(tmp_path: Path):
    """No rejected.parquet → empty dict (graceful degrade, not an error)."""
    result = load_rejected_history(tmp_path)
    assert result == {}


def test_load_rejected_history_chronological_per_user(tmp_path: Path):
    """Items are returned oldest-first per user so recency weighting works."""
    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u1"],
        "item_id": ["A", "B", "X", "C"],
        "rating": [1.0, 2.0, 1.0, 2.0],
        "timestamp": [100, 50, 200, 150],   # u1 chronological: B(50), A(100), C(150)
    })
    df.to_parquet(tmp_path / "rejected.parquet")
    got = load_rejected_history(tmp_path)
    assert got["u1"] == ["B", "A", "C"]
    assert got["u2"] == ["X"]


def test_load_rejected_history_handles_empty_parquet(tmp_path: Path):
    df = pd.DataFrame({"user_id": [], "item_id": [], "rating": [], "timestamp": []})
    df.to_parquet(tmp_path / "rejected.parquet")
    assert load_rejected_history(tmp_path) == {}


# --------------------------------------------------------------------------- #
# summarise_rejection_pattern                                                 #
# --------------------------------------------------------------------------- #


def _items_df() -> pd.DataFrame:
    """Synthetic catalog mirroring the columns the real loader exposes."""
    return pd.DataFrame([
        {"parent_asin": "A", "title": "thin polyester shirt",
         "categories": [["Clothing", "Women", "Shirts"]], "brand": "ZARA", "price": 25.0},
        {"parent_asin": "B", "title": "thin polyester dress",
         "categories": [["Clothing", "Women", "Dresses"]], "brand": "ZARA", "price": 30.0},
        {"parent_asin": "C", "title": "polyester pants tight fit",
         "categories": [["Clothing", "Women", "Pants"]], "brand": "ZARA", "price": 35.0},
        {"parent_asin": "D", "title": "leather boots",
         "categories": [["Shoes", "Women", "Boots"]], "brand": "Other", "price": 100.0},
    ])


def test_summarise_returns_none_when_too_few_rejections():
    df = _items_df()
    # < 3 items → no signal
    assert summarise_rejection_pattern([], df) is None
    assert summarise_rejection_pattern(["A"], df) is None
    assert summarise_rejection_pattern(["A", "B"], df) is None


def test_summarise_returns_none_when_catalog_mismatch():
    """If almost none of the rejected IDs are in the catalog → no signal."""
    df = _items_df()
    assert summarise_rejection_pattern(["unknown1", "unknown2", "unknown3"], df) is None


def test_summarise_extracts_top_categories_brands_hints():
    df = _items_df()
    got = summarise_rejection_pattern(["A", "B", "C"], df)
    assert got is not None
    assert got["n_rejected"] == 3
    assert got["n_with_attrs"] == 3
    # All three are ZARA → top brand
    assert "ZARA" in got["top_brands"]
    # Categories surfaced
    assert len(got["top_categories"]) >= 1
    # 'polyester' appears in all 3 titles → strong hint
    assert "polyester" in got["style_hints"]


def test_summarise_respects_recency_top_n():
    df = _items_df()
    # Pass 4 items oldest-first; ask for recent 3 only
    got = summarise_rejection_pattern(["D", "A", "B", "C"], df, recency_top_n=3)
    assert got["n_rejected"] == 3  # recent slice only
    # 'D' (leather boots) was dropped → top brand is ZARA exclusively
    assert "Other" not in got["top_brands"]


# --------------------------------------------------------------------------- #
# UserAgent — Rejected layer wiring                                           #
# --------------------------------------------------------------------------- #


def test_user_agent_without_rejected_history_returns_none_pattern():
    """Backward compat: legacy callers (no rejected_item_ids) get no pattern."""
    from adapters.llm import LLMClient
    agent = UserAgent(
        user_id="u1", history=["item"], llm=LLMClient(backend="dummy"),
    )
    assert agent.get_rejection_pattern() is None


def test_user_agent_with_rejected_history_summarises_lazily():
    """Pattern is computed on first access and cached."""
    from adapters.llm import LLMClient
    df = _items_df()
    agent = UserAgent(
        user_id="u1", history=["item"], llm=LLMClient(backend="dummy"),
        items_df=df,
        rejected_item_ids=["A", "B", "C"],
    )
    pat1 = agent.get_rejection_pattern()
    assert pat1 is not None
    assert pat1["n_rejected"] == 3
    # Second call returns the same cached object
    assert agent.get_rejection_pattern() is pat1


def test_user_agent_rejected_history_below_threshold_caches_none():
    """Users with rejected items but not enough → cached None (not recomputed)."""
    from adapters.llm import LLMClient
    df = _items_df()
    agent = UserAgent(
        user_id="u1", history=["item"], llm=LLMClient(backend="dummy"),
        items_df=df,
        rejected_item_ids=["A"],  # below MIN_REJECTED_FOR_SUMMARY
    )
    assert agent.get_rejection_pattern() is None
    # Should not re-attempt; the second call uses the cached None.
    # We verify by mutating _rejected_item_ids — cached None should win.
    agent._rejected_item_ids = ["A", "B", "C", "D"]
    assert agent.get_rejection_pattern() is None  # cache kept


def test_user_agent_rejected_history_failure_is_swallowed():
    """An unexpected pattern-building exception must not break evaluate."""
    from adapters.llm import LLMClient
    # Pass a malformed items_df (missing required column) to force failure
    # inside summarise_rejection_pattern.
    bad_df = pd.DataFrame({"wrong_column": [1, 2, 3]})
    agent = UserAgent(
        user_id="u1", history=["item"], llm=LLMClient(backend="dummy"),
        items_df=bad_df,
        rejected_item_ids=["A", "B", "C"],
    )
    # Returns None instead of raising — robustness contract for evaluate path.
    assert agent.get_rejection_pattern() is None
