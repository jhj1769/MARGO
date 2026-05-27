"""Tests for the Agentic Memory layer (``src/core/memory/``).

Covers:
  * AgentMemory protocol semantics (NullMemory and JSONLMemoryStore)
  * Each agent's typed event helpers (make_*_event)
  * ItemMemory's cohort-aware retrieval filter (the one schema-specific
    convenience baked into the store)
  * Cache invalidation across re-instantiation (the persistence contract
    that lets the same file be read consistently after another process
    appends to it)
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from core.memory.base import AgentMemory, MemoryEvent, NullMemory
from core.memory.persistence import JSONLMemoryStore
from core.memory.schemas import (
    ExpertDirectiveOutcomeEvent,
    ExpertMemory,
    ItemAudienceClaimEvent,
    ItemMemory,
    ItemReceptionEvent,
    TrendMemory,
    TrendPredictionEvent,
    UserMemory,
    UserMemoryEvent,
    make_expert_outcome_event,
    make_item_audience_claim_event,
    make_item_reception_event,
    make_trend_prediction_event,
    make_user_event,
)


# --------------------------------------------------------------------------- #
# Protocol — NullMemory                                                       #
# --------------------------------------------------------------------------- #


def test_null_memory_is_inert():
    mem = NullMemory()
    assert mem.size() == 0
    assert mem.retrieve() == []
    mem.append(MemoryEvent(event_type="x", timestamp=time.time()))
    # Still 0 — NullMemory swallows writes (this is how agents without
    # memory wired in stay backward-compatible).
    assert mem.size() == 0
    assert mem.retrieve() == []


# --------------------------------------------------------------------------- #
# Persistence — JSONLMemoryStore                                              #
# --------------------------------------------------------------------------- #


def test_jsonl_store_append_then_retrieve(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    store = JSONLMemoryStore(path)
    e1 = MemoryEvent(event_type="t", timestamp=1.0, payload={"i": 1})
    e2 = MemoryEvent(event_type="t", timestamp=2.0, payload={"i": 2})
    store.append(e1)
    store.append(e2)
    assert store.size() == 2
    got = store.retrieve(top_k=10)
    # Most recent first
    assert [g.payload["i"] for g in got] == [2, 1]


def test_jsonl_store_reload_after_external_write(tmp_path: Path):
    """Cache must invalidate when the file changes underneath us.

    Models the "rebuild script appended new events" workflow — the
    long-lived engine instance shouldn't return stale data.
    """
    path = tmp_path / "events.jsonl"
    store = JSONLMemoryStore(path)
    store.append(MemoryEvent(event_type="t", timestamp=1.0, payload={"i": 1}))
    assert store.size() == 1

    # External process appends a line and bumps mtime.
    with path.open("a", encoding="utf-8") as f:
        f.write(MemoryEvent(event_type="t", timestamp=2.0, payload={"i": 2}).model_dump_json() + "\n")
    # Force mtime change (some FS have second-level granularity)
    import os
    later = path.stat().st_mtime + 1
    os.utime(path, (later, later))

    # New store sees both lines
    store2 = JSONLMemoryStore(path)
    assert store2.size() == 2


def test_jsonl_store_clear(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    store = JSONLMemoryStore(path)
    store.append(MemoryEvent(event_type="t", timestamp=1.0))
    store.clear()
    assert store.size() == 0
    assert not path.exists()


def test_jsonl_store_top_k_recency(tmp_path: Path):
    store = JSONLMemoryStore(tmp_path / "events.jsonl")
    for i in range(5):
        store.append(MemoryEvent(event_type="t", timestamp=float(i), payload={"i": i}))
    got = store.retrieve(top_k=3)
    # Most recent 3, newest first
    assert [g.payload["i"] for g in got] == [4, 3, 2]


# --------------------------------------------------------------------------- #
# Typed event helpers                                                          #
# --------------------------------------------------------------------------- #


def test_user_event_roundtrip():
    ev = make_user_event(
        "axis_snapshot",
        user_id="u1",
        snapshot_axes={"style_preference": "minimal-casual", "price_preference": "mid-tier"},
    )
    assert ev.event_type == "axis_snapshot"
    assert ev.payload["user_id"] == "u1"
    assert ev.payload["snapshot_axes"]["style_preference"] == "minimal-casual"
    # ``exclude_none`` keeps the payload tight
    assert "candidate_item_id" not in ev.payload

    # Parse back into the typed object
    parsed = UserMemoryEvent.model_validate(ev.payload)
    assert parsed.snapshot_axes["price_preference"] == "mid-tier"


def test_item_reception_event_roundtrip():
    ev = make_item_reception_event(
        item_id="B001",
        viewer_user_id="u1",
        viewer_cohort="bra:brand-diverse|cat:balanced|pri:mid-tier|sty:minimal-casual",
        score=0.78,
        rationale_summary="strong cohort fit",
    )
    assert ev.event_type == "item_reception"
    typed = ItemReceptionEvent.model_validate(ev.payload)
    assert typed.score == 0.78
    assert typed.viewer_cohort.startswith("bra:")


def test_item_audience_claim_event_unverified():
    ev = make_item_audience_claim_event(
        item_id="B001",
        claim_nl="I serve mid-tier balanced cohorts well",
        claim_target_cohort="bra:brand-diverse|cat:balanced|pri:mid-tier|sty:minimal-casual",
    )
    typed = ItemAudienceClaimEvent.model_validate(ev.payload)
    assert typed.verified_at is None  # written before verification
    assert typed.verified_avg_score is None


def test_trend_prediction_event_unverified():
    ev = make_trend_prediction_event(
        snapshot_id="fashion_2025-04",
        keyword="y2k revival",
        predicted_lifecycle="rising",
        catalog_attributes=["low-rise", "cargo", "metallic"],
        n_matched_items=87,
    )
    typed = TrendPredictionEvent.model_validate(ev.payload)
    assert typed.predicted_lifecycle == "rising"
    assert typed.actual_lift_pct is None  # awaiting verification


def test_expert_outcome_event_roundtrip():
    ev = make_expert_outcome_event(
        brief_summary="casual to formal upsell, exclude over $150",
        directive_goal="upsell-formal",
        directive_constraints_summary={"forbid_category": ["lingerie"], "price_max": 150},
        n_refinements=1,
        final_compliance_score=0.92,
        passed=True,
    )
    typed = ExpertDirectiveOutcomeEvent.model_validate(ev.payload)
    assert typed.passed is True
    assert typed.final_compliance_score == 0.92


# --------------------------------------------------------------------------- #
# Schema validation — invalid payloads should raise                           #
# --------------------------------------------------------------------------- #


def test_item_reception_event_rejects_out_of_range_score():
    with pytest.raises(Exception):
        make_item_reception_event(
            item_id="B001",
            viewer_user_id="u1",
            viewer_cohort="x",
            score=1.5,  # > 1.0 not allowed
        )


def test_expert_outcome_event_rejects_negative_refinements():
    with pytest.raises(Exception):
        make_expert_outcome_event(
            brief_summary="x",
            directive_goal="y",
            n_refinements=-1,
            final_compliance_score=0.5,
            passed=True,
        )


# --------------------------------------------------------------------------- #
# Per-agent memory wrappers                                                    #
# --------------------------------------------------------------------------- #


def test_user_memory_writes_to_per_user_path(tmp_path: Path):
    mem = UserMemory("u1", root=tmp_path)
    mem.append(make_user_event("axis_snapshot", user_id="u1", snapshot_axes={"style_preference": "preppy"}))
    expected = tmp_path / "user" / "u1.jsonl"
    assert expected.exists()
    assert mem.size() == 1


def test_trend_memory_writes_to_per_domain_path(tmp_path: Path):
    mem = TrendMemory("fashion", root=tmp_path)
    mem.append(make_trend_prediction_event(
        snapshot_id="s1", keyword="y2k", predicted_lifecycle="rising",
    ))
    expected = tmp_path / "trend" / "fashion.jsonl"
    assert expected.exists()


def test_expert_memory_writes_to_per_persona_path(tmp_path: Path):
    mem = ExpertMemory("MD-persona-1", root=tmp_path)
    mem.append(make_expert_outcome_event(
        brief_summary="b", directive_goal="g",
        n_refinements=0, final_compliance_score=1.0, passed=True,
    ))
    expected = tmp_path / "expert" / "MD-persona-1.jsonl"
    assert expected.exists()


# --------------------------------------------------------------------------- #
# Item Memory — cohort-aware filter (the schema-specific behaviour)            #
# --------------------------------------------------------------------------- #


def test_item_memory_cohort_filter_isolates_viewer_cohort(tmp_path: Path):
    """retrieve(viewer_cohort=X) must return only X's events, not others."""
    mem = ItemMemory("B001", root=tmp_path)
    cohort_a = "bra:brand-diverse|cat:balanced|pri:mid-tier|sty:minimal-casual"
    cohort_b = "bra:brand-loyal:nike|cat:athleisure-focused|pri:budget|sty:streetwear"

    for cohort, score in [(cohort_a, 0.8), (cohort_b, 0.4), (cohort_a, 0.75)]:
        mem.append(make_item_reception_event(
            item_id="B001", viewer_user_id="u", viewer_cohort=cohort, score=score,
        ))

    slice_a = mem.retrieve(query={"viewer_cohort": cohort_a}, top_k=10)
    slice_b = mem.retrieve(query={"viewer_cohort": cohort_b}, top_k=10)
    assert len(slice_a) == 2
    assert len(slice_b) == 1
    assert all(e.payload["viewer_cohort"] == cohort_a for e in slice_a)


def test_item_memory_cohort_filter_includes_audience_claims_for_target_cohort(tmp_path: Path):
    """A retrieve(viewer_cohort=X) call must also surface audience claims
    that *targeted* X — because the User Agent wants both reception
    evidence AND past claims about this cohort.
    """
    mem = ItemMemory("B001", root=tmp_path)
    cohort = "bra:brand-diverse|cat:balanced|pri:mid-tier|sty:minimal-casual"

    mem.append(make_item_reception_event(
        item_id="B001", viewer_user_id="u1", viewer_cohort=cohort, score=0.8,
    ))
    mem.append(make_item_audience_claim_event(
        item_id="B001",
        claim_nl="I serve mid-tier balanced cohorts well",
        claim_target_cohort=cohort,
    ))
    # Unrelated claim (different cohort) — should NOT surface
    mem.append(make_item_audience_claim_event(
        item_id="B001",
        claim_nl="I serve luxury cohorts well",
        claim_target_cohort="other-cohort",
    ))

    got = mem.retrieve(query={"viewer_cohort": cohort}, top_k=10)
    assert len(got) == 2  # 1 reception + 1 matching claim
    types = {e.event_type for e in got}
    assert types == {"item_reception", "item_audience_claim"}


def test_item_memory_no_cohort_query_returns_all(tmp_path: Path):
    """retrieve() with no query → unfiltered slice (used for analytics)."""
    mem = ItemMemory("B001", root=tmp_path)
    for cohort in ["a", "b", "c"]:
        mem.append(make_item_reception_event(
            item_id="B001", viewer_user_id="u", viewer_cohort=cohort, score=0.5,
        ))
    assert len(mem.retrieve(top_k=10)) == 3
