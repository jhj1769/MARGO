"""Enhancement 1.5 — cohort signature, peer signal, coverage diagnostic."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MARGO_LLM_BACKEND", "dummy")

import pandas as pd
import pytest

from adapters.llm import LLMClient
from core.agents.user_agent import UserAgent
from core.protocol.messages import (
    PreferenceAxis,
    UserPreferenceState,
    compute_cohort_signature,
)
from data.fashion.cohort_loader import (
    MIN_COHORT_SIZE,
    build_cohorts,
    cohort_coverage,
    load_cohort_stats,
    save_cohorts,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _stat_axis(name: str, value: str) -> PreferenceAxis:
    return PreferenceAxis(
        name=name,  # type: ignore[arg-type]
        value=value,
        confidence=0.9,
        evidence=[],
        secondary_values=[],
        derived_from="statistical",
        stability=1.0,
    )


def _style_axis(value: str = "minimal-casual") -> PreferenceAxis:
    return PreferenceAxis(
        name="style_preference",
        value=value,
        confidence=0.7,
        evidence=[],
        secondary_values=[],
        derived_from="llm_inferred",
        stability=1.0,
    )


def _state(uid: str, *, price="mid-tier", category="balanced", brand="brand-diverse",
           style="minimal-casual") -> UserPreferenceState:
    axes = [
        _stat_axis("price_preference", price),
        _stat_axis("category_preference", category),
        _stat_axis("brand_preference", brand),
        _style_axis(style),
    ]
    sig = compute_cohort_signature(UserPreferenceState(user_id=uid, profile_nl="", axes=axes))
    return UserPreferenceState(user_id=uid, profile_nl="", axes=axes, cohort_signature=sig)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_cohort_signature_deterministic():
    """Identical axes (in any order) → identical signature.

    Style IS part of the signature: users with the same statistical axes
    but different styles belong to different cohorts.
    """
    sig_a = compute_cohort_signature(_state("a"))
    sig_b = compute_cohort_signature(_state("b"))
    assert sig_a == sig_b
    # Different style → different signature.
    sig_diff_style = compute_cohort_signature(_state("c", style="streetwear"))
    assert sig_diff_style != sig_a
    assert "sty:streetwear" in sig_diff_style
    assert "sty:minimal-casual" in sig_a


def test_signature_order_invariance():
    """Same axes inserted in reversed order produce the same signature."""
    state_fwd = _state("u1")
    reversed_axes = list(reversed(state_fwd.axes))
    state_rev = UserPreferenceState(user_id="u1", profile_nl="", axes=reversed_axes)
    assert compute_cohort_signature(state_fwd) == compute_cohort_signature(state_rev)


def test_signature_excludes_unknown_axes():
    """Axes with value 'unknown' must not contaminate the signature."""
    axes = [
        _stat_axis("price_preference", "mid-tier"),
        _stat_axis("category_preference", "unknown"),
        _stat_axis("brand_preference", "brand-diverse"),
    ]
    state = UserPreferenceState(user_id="u1", profile_nl="", axes=axes)
    sig = compute_cohort_signature(state)
    assert "unknown" not in sig
    assert "cat" not in sig.split("|")[0]  # category prefix absent


def test_build_cohorts_drops_small_groups():
    """Cohorts under MIN_COHORT_SIZE must be filtered out."""
    states = [_state(f"u{i}") for i in range(3)]  # only 3 users → below threshold
    cohorts = build_cohorts(
        states,
        user_purchases={s.user_id: ["A", "B"] for s in states},
        min_cohort_size=MIN_COHORT_SIZE,
    )
    assert cohorts == {}


def test_build_cohorts_computes_peer_ratios():
    """item_buy_ratios = (members who bought item) / cohort size."""
    states = [_state(f"u{i}") for i in range(6)]
    purchases = {
        "u0": ["A", "B"],
        "u1": ["A", "C"],
        "u2": ["A"],
        "u3": ["B"],
        "u4": ["B"],
        "u5": ["A", "B", "C"],
    }
    cohorts = build_cohorts(states, purchases, min_cohort_size=MIN_COHORT_SIZE)
    assert len(cohorts) == 1
    cohort = next(iter(cohorts.values()))
    assert cohort.size == 6
    # 4/6 bought A, 4/6 bought B, 2/6 bought C
    assert cohort.peer_signal_for("A") == pytest.approx(4 / 6)
    assert cohort.peer_signal_for("B") == pytest.approx(4 / 6)
    assert cohort.peer_signal_for("C") == pytest.approx(2 / 6)
    # Items not in cohort → 0.0
    assert cohort.peer_signal_for("Z") == 0.0


def test_save_and_load_cohort_roundtrip(tmp_path: Path):
    """Cohort persistence: write → read should restore the same stats."""
    states = [_state(f"u{i}") for i in range(5)]
    cohorts = build_cohorts(
        states,
        user_purchases={s.user_id: ["A"] for s in states},
        min_cohort_size=MIN_COHORT_SIZE,
    )
    sig = next(iter(cohorts.keys()))
    save_cohorts(cohorts, tmp_path)
    loaded = load_cohort_stats(sig, tmp_path)
    assert loaded is not None
    assert loaded.size == 5
    assert loaded.peer_signal_for("A") == 1.0


def test_load_missing_cohort_returns_none(tmp_path: Path):
    """Unknown signature → load returns None (no fallback / no exception)."""
    assert load_cohort_stats("bra:nope|cat:nope|pri:nope", tmp_path) is None


def test_cohort_coverage_report(tmp_path: Path):
    """Coverage report: histogram and fallback rate."""
    states = []
    # 6 users in one cohort (qualified)
    states += [_state(f"qual{i}", price="mid-tier") for i in range(6)]
    # 2 users in another cohort (below threshold)
    states += [_state(f"small{i}", price="luxury-aware") for i in range(2)]

    report = cohort_coverage(states, min_cohort_size=MIN_COHORT_SIZE)
    assert report["total_users"] == 8
    assert report["users_in_qualified_cohorts"] == 6
    assert report["fallback_rate"] == pytest.approx(2 / 8)
    assert report["total_cohorts"] == 2


# --------------------------------------------------------------------------- #
# Agent integration                                                           #
# --------------------------------------------------------------------------- #


def test_user_agent_get_peer_signal_with_cohort(tmp_path: Path):
    """UserAgent loads the cohort and surfaces a non-zero ratio for known items."""
    # Build a 5-user cohort offline, save it to tmp_path.
    states = [_state(f"u{i}") for i in range(5)]
    purchases = {
        "u0": ["X", "Y"],
        "u1": ["X", "Y"],
        "u2": ["X"],
        "u3": ["X"],
        "u4": ["Y"],
    }
    cohorts = build_cohorts(states, purchases, min_cohort_size=MIN_COHORT_SIZE)
    save_cohorts(cohorts, tmp_path)

    # Construct an items_df so UserAgent.build_profile populates axes that
    # match the cohort signature. All purchases are mid-tier balanced/diverse.
    items_df = pd.DataFrame({
        "parent_asin": ["X", "Y"],
        "price": [50.0, 60.0],
        "brand": ["BrandA", "BrandB"],
        "categories": [
            ["Clothing", "Women", "Tops", "T-shirts"],
            ["Clothing", "Women", "Bottoms", "Jeans"],
        ],
    })
    llm = LLMClient(backend="dummy")
    user = UserAgent(
        user_id="u_new",
        history=["item X", "item Y"],
        history_item_ids=["X", "Y"],
        items_df=items_df,
        processed_dir=tmp_path,
        llm=llm,
    )
    user.build_profile()
    assert user.state.preference_state is not None
    # Hand-set the signature to match the offline-built cohort so the
    # lookup hits regardless of how the dummy synthesises axis values.
    matching_sig = next(iter(cohorts.keys()))
    user.state.preference_state.cohort_signature = matching_sig

    ratio, explanation = user.get_peer_signal("X")
    # 4/5 of the cohort bought X.
    assert ratio == pytest.approx(4 / 5)
    assert "4 users" not in explanation  # sanity: explanation references full size
    assert "5 users" in explanation

    # Unknown item: ratio 0, helpful explanation.
    ratio_z, expl_z = user.get_peer_signal("Z_unknown")
    assert ratio_z == 0.0
    assert "cohort" in expl_z.lower()


def test_user_agent_no_processed_dir_returns_default():
    """Without processed_dir, peer signal degrades gracefully (no exception)."""
    items_df = pd.DataFrame({
        "parent_asin": ["X"],
        "price": [50.0],
        "brand": ["BrandA"],
        "categories": [["Clothing", "Women", "Tops", "T-shirts"]],
    })
    user = UserAgent(
        user_id="u1",
        history=["item X"],
        history_item_ids=["X"],
        items_df=items_df,
        llm=LLMClient(backend="dummy"),
        # processed_dir intentionally omitted
    )
    user.build_profile()
    ratio, explanation = user.get_peer_signal("X")
    assert ratio == 0.0
    assert "cohort" in explanation.lower()
