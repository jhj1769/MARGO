"""Phase B — Item Agent Autobiographical Memory tests.

Verifies:
  * Memory wiring: memory_root → ItemMemory, no args → NullMemory,
    explicit memory= → injected store
  * ``_summarise_past_reception`` helper: empty / Null / populated → expected
  * ``_self_correction_warning`` helper: only fires for verified-below-threshold
    claims with enough sample, never for cold start
  * End-to-end: dummy LLM + memory wire doesn't break self_describe (smoke)
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MARGO_LLM_BACKEND", "dummy")

from core.agents.item_agent import (
    ItemAgent,
    ItemFacts,
    _self_correction_warning,
    _summarise_past_reception,
)
from core.memory.base import MemoryEvent, NullMemory
from core.memory.schemas import (
    ItemMemory,
    make_item_audience_claim_event,
    make_item_reception_event,
)
from core.protocol.messages import Directive


COHORT = "bra:brand-diverse|cat:balanced|pri:mid-tier|sty:minimal-casual"
OTHER_COHORT = "bra:brand-loyal:nike|cat:athleisure-focused|pri:budget|sty:streetwear"


# --------------------------------------------------------------------------- #
# Memory wiring                                                                #
# --------------------------------------------------------------------------- #


def test_item_agent_defaults_to_null_memory():
    agent = ItemAgent(ItemFacts(item_id="B001", title="Tee"))
    assert isinstance(agent.memory, NullMemory)


def test_item_agent_memory_root_opens_item_memory(tmp_path: Path):
    agent = ItemAgent(ItemFacts(item_id="B001", title="Tee"), memory_root=tmp_path)
    assert isinstance(agent.memory, ItemMemory)
    assert agent.memory.item_id == "B001"
    # Initially zero events, path doesn't exist yet (lazy)
    assert agent.memory.size() == 0


def test_item_agent_explicit_memory_overrides_root(tmp_path: Path):
    """An explicit ``memory=`` always wins over ``memory_root=`` — useful
    for tests that want to inject a pre-populated store."""
    explicit = ItemMemory("B001", root=tmp_path / "explicit")
    agent = ItemAgent(
        ItemFacts(item_id="B001", title="Tee"),
        memory_root=tmp_path / "ignored",
        memory=explicit,
    )
    assert agent.memory is explicit


# --------------------------------------------------------------------------- #
# _summarise_past_reception                                                    #
# --------------------------------------------------------------------------- #


def test_summarise_returns_none_for_null_memory():
    assert _summarise_past_reception(NullMemory(), COHORT) is None


def test_summarise_returns_none_when_cohort_missing(tmp_path: Path):
    mem = ItemMemory("B001", root=tmp_path)
    mem.append(make_item_reception_event(
        item_id="B001", viewer_user_id="u", viewer_cohort=COHORT, score=0.8,
    ))
    assert _summarise_past_reception(mem, None) is None
    assert _summarise_past_reception(mem, "") is None


def test_summarise_returns_none_for_empty_cohort_slice(tmp_path: Path):
    mem = ItemMemory("B001", root=tmp_path)
    # Reception exists, but for a DIFFERENT cohort
    mem.append(make_item_reception_event(
        item_id="B001", viewer_user_id="u", viewer_cohort=OTHER_COHORT, score=0.8,
    ))
    assert _summarise_past_reception(mem, COHORT) is None


def test_summarise_aggregates_cohort_receptions(tmp_path: Path):
    mem = ItemMemory("B001", root=tmp_path)
    for score in [0.8, 0.7, 0.9]:
        mem.append(make_item_reception_event(
            item_id="B001", viewer_user_id="u", viewer_cohort=COHORT,
            score=score, rationale_summary=f"score was {score}",
        ))
    s = _summarise_past_reception(mem, COHORT)
    assert s is not None
    assert s["n"] == 3
    assert s["avg_score"] == 0.8  # (0.8+0.7+0.9)/3
    assert s["viewer_cohort"] == COHORT
    # Up to 3 most recent rationales
    assert len(s["recent_rationales"]) == 3


# --------------------------------------------------------------------------- #
# _self_correction_warning                                                     #
# --------------------------------------------------------------------------- #


def test_warning_none_for_null_memory():
    assert _self_correction_warning(NullMemory(), COHORT) is None


def test_warning_none_when_no_claims(tmp_path: Path):
    mem = ItemMemory("B001", root=tmp_path)
    assert _self_correction_warning(mem, COHORT) is None


def test_warning_none_for_unverified_claims(tmp_path: Path):
    """Cold-start safety: claims exist but verification job hasn't run yet → no warning."""
    mem = ItemMemory("B001", root=tmp_path)
    for _ in range(5):
        mem.append(make_item_audience_claim_event(
            item_id="B001",
            claim_nl="I serve this cohort well",
            claim_target_cohort=COHORT,
        ))
    assert _self_correction_warning(mem, COHORT) is None


def test_warning_none_when_too_few_verified_claims(tmp_path: Path):
    """One verified claim isn't enough sample for a warning — need ≥ 2."""
    mem = ItemMemory("B001", root=tmp_path)
    mem.append(make_item_audience_claim_event(
        item_id="B001",
        claim_nl="I serve this cohort well",
        claim_target_cohort=COHORT,
        verified_at=123.0,
        verified_avg_score=0.2,  # well below threshold but lone sample
        verified_n=1,
    ))
    assert _self_correction_warning(mem, COHORT) is None


def test_warning_none_when_verified_accuracy_above_threshold(tmp_path: Path):
    mem = ItemMemory("B001", root=tmp_path)
    for _ in range(3):
        mem.append(make_item_audience_claim_event(
            item_id="B001",
            claim_nl="I serve this cohort well",
            claim_target_cohort=COHORT,
            verified_at=123.0,
            verified_avg_score=0.75,  # > 0.5 threshold
            verified_n=10,
        ))
    assert _self_correction_warning(mem, COHORT) is None


def test_warning_fires_when_verified_accuracy_below_threshold(tmp_path: Path):
    mem = ItemMemory("B001", root=tmp_path)
    for s in [0.30, 0.35, 0.40]:
        mem.append(make_item_audience_claim_event(
            item_id="B001",
            claim_nl="I serve this cohort well",
            claim_target_cohort=COHORT,
            verified_at=123.0,
            verified_avg_score=s,
            verified_n=8,
        ))
    warning = _self_correction_warning(mem, COHORT)
    assert warning is not None
    # Warning mentions the cohort and the threshold so the LLM has actionable evidence
    assert COHORT in warning
    assert "threshold" in warning.lower()


def test_warning_ignores_claims_for_other_cohorts(tmp_path: Path):
    """A bad track record on cohort A must NOT suppress claims about cohort B."""
    mem = ItemMemory("B001", root=tmp_path)
    for _ in range(3):
        mem.append(make_item_audience_claim_event(
            item_id="B001",
            claim_nl="I serve other cohort well",
            claim_target_cohort=OTHER_COHORT,
            verified_at=123.0,
            verified_avg_score=0.2,  # bad accuracy for OTHER_COHORT
            verified_n=8,
        ))
    # No warning for COHORT — track record is for a different cohort
    assert _self_correction_warning(mem, COHORT) is None


# --------------------------------------------------------------------------- #
# End-to-end smoke (dummy LLM)                                                #
# --------------------------------------------------------------------------- #


def test_self_describe_with_memory_wired_does_not_crash(tmp_path: Path):
    """Smoke: an Item Agent with memory wired runs self_describe end-to-end
    against the dummy LLM and writes the audience claim event when present.
    Backward compat: a NullMemory agent runs unchanged.
    """
    from adapters.llm import LLMClient
    llm = LLMClient(backend="dummy")

    facts = ItemFacts(item_id="B001", title="Beige Linen Shirt",
                      attributes={"price": 45, "category": "shirt"})
    directive = Directive(
        goal="daily push",
        natural_language="Casual daily basics.",
    )

    # Agent with real memory
    agent = ItemAgent(facts, llm=llm, memory_root=tmp_path)
    out = agent.self_describe(
        user_profile="user likes minimal casual",
        directive=directive,
        viewer_cohort=COHORT,
        turn_id="u1@123",
    )
    assert out.description  # dummy LLM produced something
    # If the dummy LLM populated audience_fit_claim_nl, it must have been appended.
    # If it didn't (which is fine), memory size is 0 — both paths are valid here.
    events = agent.memory.retrieve(top_k=100)
    if out.audience_fit_claim_nl:
        assert any(e.event_type == "item_audience_claim" for e in events)

    # Agent with NullMemory (backward compat) — never writes regardless of output
    agent2 = ItemAgent(facts, llm=llm)
    out2 = agent2.self_describe(
        user_profile="user likes minimal casual",
        directive=directive,
        viewer_cohort=COHORT,
    )
    assert out2.description
    assert isinstance(agent2.memory, NullMemory)
