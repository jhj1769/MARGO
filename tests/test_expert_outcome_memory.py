"""Phase D — Expert Agent outcome memory + similar-brief retrieval tests.

Verifies:
  * Memory wiring: memory_root → ExpertMemory, no args → NullMemory
  * record_outcome appends a typed event (and swallows errors)
  * Cosine retrieval: empty memory → [], populated → ranked by similarity,
    below threshold → omitted
  * Helper tokeniser / cosine math are stable
  * Backward compat: an ExpertAgent without memory issues directives
    exactly as before (the new prompt section just stays empty)
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MARGO_LLM_BACKEND", "dummy")

from core.agents.expert_agent import (
    ExpertAgent,
    _cosine,
    _tokenise_brief,
    _vectorise,
    retrieve_similar_briefs,
)
from core.memory.base import NullMemory
from core.memory.schemas import ExpertMemory, make_expert_outcome_event
from core.protocol.messages import Directive, ValidationReport


# --------------------------------------------------------------------------- #
# Tokeniser + cosine math                                                     #
# --------------------------------------------------------------------------- #


def test_tokenise_drops_stopwords_and_short_tokens():
    tokens = _tokenise_brief("The casual to formal upsell for autumn outerwear")
    # stopwords ('the', 'for'), short tokens ('to') removed
    assert "the" not in tokens
    assert "for" not in tokens
    assert "to" not in tokens
    # surviving content tokens
    assert "casual" in tokens and "formal" in tokens and "outerwear" in tokens


def test_cosine_identical_vectors_is_one():
    v = _vectorise(["a", "b", "c", "a"])
    assert abs(_cosine(v, v) - 1.0) < 1e-9


def test_cosine_disjoint_vectors_is_zero():
    v1 = _vectorise(["a", "b"])
    v2 = _vectorise(["c", "d"])
    assert _cosine(v1, v2) == 0.0


def test_cosine_with_empty_input_is_zero():
    assert _cosine({}, _vectorise(["a"])) == 0.0
    assert _cosine(_vectorise(["a"]), {}) == 0.0


# --------------------------------------------------------------------------- #
# retrieve_similar_briefs                                                     #
# --------------------------------------------------------------------------- #


def test_retrieve_empty_for_null_memory():
    assert retrieve_similar_briefs(NullMemory(), "any brief") == []


def test_retrieve_empty_when_no_past_briefs(tmp_path: Path):
    mem = ExpertMemory("default", root=tmp_path)
    assert retrieve_similar_briefs(mem, "any brief") == []


def test_retrieve_ranks_by_similarity(tmp_path: Path):
    mem = ExpertMemory("default", root=tmp_path)
    # Populate a few past outcomes
    for brief, goal, passed in [
        ("casual to formal upsell autumn outerwear",
         "casual-to-formal", True),
        ("luxury knitwear push winter cashmere", "luxury-push", True),
        ("trench coat boost campaign", "trench-boost", False),
        ("autumn outerwear seasonal campaign trench",
         "autumn-trench", True),
    ]:
        mem.append(make_expert_outcome_event(
            brief_summary=brief,
            directive_goal=goal,
            n_refinements=1 if passed else 3,
            final_compliance_score=0.9 if passed else 0.4,
            passed=passed,
        ))

    got = retrieve_similar_briefs(
        mem, "autumn outerwear trench campaign", top_k=3,
    )
    assert len(got) >= 2
    # Most-similar should be the autumn-trench brief
    assert got[0]["directive_goal"] == "autumn-trench"
    # Results are ranked descending by similarity
    sims = [r["similarity"] for r in got]
    assert sims == sorted(sims, reverse=True)


def test_retrieve_filters_below_min_similarity(tmp_path: Path):
    mem = ExpertMemory("default", root=tmp_path)
    mem.append(make_expert_outcome_event(
        brief_summary="completely unrelated topic about kitchenware",
        directive_goal="kitchen-push",
        n_refinements=0,
        final_compliance_score=1.0,
        passed=True,
    ))
    # Brief shares no content tokens → cosine well below 0.15 default
    assert retrieve_similar_briefs(mem, "winter outerwear cashmere") == []


def test_retrieve_ignores_non_outcome_events(tmp_path: Path):
    """A foreign event_type in the same store must be skipped, not crash."""
    from core.memory.base import MemoryEvent
    mem = ExpertMemory("default", root=tmp_path)
    mem.append(MemoryEvent(event_type="other", timestamp=1.0,
                           payload={"brief_summary": "noise here"}))
    mem.append(make_expert_outcome_event(
        brief_summary="real outcome about outerwear",
        directive_goal="ok",
        n_refinements=0,
        final_compliance_score=1.0, passed=True,
    ))
    got = retrieve_similar_briefs(mem, "outerwear push")
    assert len(got) == 1
    assert got[0]["directive_goal"] == "ok"


# --------------------------------------------------------------------------- #
# ExpertAgent wiring                                                          #
# --------------------------------------------------------------------------- #


def test_expert_agent_defaults_to_null_memory():
    agent = ExpertAgent(persona="MD")
    assert isinstance(agent.memory, NullMemory)


def test_expert_agent_memory_root_opens_expert_memory(tmp_path: Path):
    agent = ExpertAgent(persona="MD", persona_id="fashion-md", memory_root=tmp_path)
    assert isinstance(agent.memory, ExpertMemory)
    assert agent.memory.persona_id == "fashion-md"
    assert agent.memory.size() == 0  # empty until record_outcome is called


def test_expert_agent_explicit_memory_overrides_root(tmp_path: Path):
    explicit = ExpertMemory("custom", root=tmp_path / "explicit")
    agent = ExpertAgent(
        persona="MD", memory_root=tmp_path / "ignored", memory=explicit,
    )
    assert agent.memory is explicit


def test_record_outcome_appends_typed_event(tmp_path: Path):
    agent = ExpertAgent(persona="MD", persona_id="fashion-md", memory_root=tmp_path)
    directive = Directive(
        goal="autumn-trench",
        natural_language="Push autumn trench coats.",
        structured_constraints={
            "boost_category": "trench-coat",
            "price_max": 500,
            "irrelevant_field": "should be filtered out",
        },
    )
    report = ValidationReport(passed=True, compliance_score=0.92, violations=[])
    agent.record_outcome(
        brief="Push autumn outerwear, emphasise trench coats",
        directive=directive,
        report=report,
        n_refinements=1,
    )
    events = agent.memory.retrieve(top_k=10)
    assert len(events) == 1
    payload = events[0].payload
    assert payload["directive_goal"] == "autumn-trench"
    assert payload["passed"] is True
    assert payload["n_refinements"] == 1
    # Constraint summary is filtered to validator-known keys
    assert "boost_category" in payload["directive_constraints_summary"]
    assert "price_max" in payload["directive_constraints_summary"]
    assert "irrelevant_field" not in payload["directive_constraints_summary"]


def test_record_outcome_clamps_compliance_to_unit_range(tmp_path: Path):
    agent = ExpertAgent(persona="MD", memory_root=tmp_path)
    directive = Directive(goal="g", natural_language="nl")
    # Out-of-range score must be clamped, not crash
    report = ValidationReport(passed=False, compliance_score=1.5, violations=["x"])
    agent.record_outcome(brief="b", directive=directive, report=report, n_refinements=0)
    events = agent.memory.retrieve(top_k=10)
    assert events[0].payload["final_compliance_score"] == 1.0


def test_record_outcome_no_op_for_null_memory():
    """Without memory wired, record_outcome must silently no-op."""
    agent = ExpertAgent(persona="MD")  # NullMemory
    agent.record_outcome(
        brief="b",
        directive=Directive(goal="g", natural_language="nl"),
        report=ValidationReport(passed=True, compliance_score=1.0, violations=[]),
        n_refinements=0,
    )
    # Nothing to assert — the no-op is the contract. We're just verifying
    # no exception is raised on the cold-start path.


def test_issue_directive_renders_past_briefs_when_memory_populated(tmp_path: Path):
    """End-to-end smoke: with a populated memory, issue_directive runs
    against the dummy LLM and renders the past-briefs section without
    error. (We can't assert on dummy LLM output, but we can assert the
    code path completes.)"""
    from adapters.llm import LLMClient
    agent = ExpertAgent(
        persona="MD",
        vocabulary={"category": ["trench-coat"], "color": ["beige"]},
        persona_id="fashion-md",
        memory_root=tmp_path,
        llm=LLMClient(backend="dummy"),
    )
    # Seed memory with a similar past brief
    agent.record_outcome(
        brief="autumn outerwear push trench coats",
        directive=Directive(goal="autumn-trench", natural_language="nl"),
        report=ValidationReport(passed=True, compliance_score=0.95, violations=[]),
        n_refinements=1,
    )
    out = agent.issue_directive("Push autumn outerwear emphasis on trench")
    assert isinstance(out, Directive)
