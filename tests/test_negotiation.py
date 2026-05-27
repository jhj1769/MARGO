"""Enhancement 5 — Expert↔Trend negotiation loop tests.

The LLM is replaced with hand-injected outputs (via monkey-patching of
``detect_tensions`` / ``challenge_directive`` / ``respond_to_challenge``)
so we test the lifecycle logic, not the dummy LLM behaviour.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

os.environ.setdefault("MARGO_LLM_BACKEND", "dummy")

import pytest

from adapters.llm import LLMClient
from core.agents.expert_agent import ExpertAgent
from core.agents.trend_agent import TrendAgent
from core.lifecycle.phase2_directive import run_phase2
from core.protocol.messages import (
    Directive,
    NegotiationMessage,
    TrendDirectiveTension,
    TrendInterpretation,
)
from core.protocol.router import MessageBus


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_agents():
    llm = LLMClient(backend="dummy")
    bus = MessageBus()
    expert = ExpertAgent(
        persona="MD",
        vocabulary={"category": ["outerwear"], "color": ["beige"]},
        llm=llm, bus=bus,
    )
    trend = TrendAgent(domain="fashion", time_window="2023", llm=llm, bus=bus)
    return expert, trend, bus


def _directive(**overrides) -> Directive:
    base = dict(
        goal="upsell outerwear",
        structured_constraints={"boost_category": "outerwear"},
        natural_language="Boost outerwear within +30% price.",
    )
    base.update(overrides)
    return Directive(**base)


def _trend_interp(summary: str = "Y2K minimalism rising") -> TrendInterpretation:
    return TrendInterpretation(
        domain="fashion", time_window="2023",
        summary=summary, keywords=["minimal", "tailored"],
    )


def _tension(severity: float, type_: str = "attribute_conflict") -> TrendDirectiveTension:
    return TrendDirectiveTension(
        type=type_,  # type: ignore[arg-type]
        severity=severity,
        description_nl="example tension",
        directive_element="boost_category: outerwear",
        trend_element="rising: minimal",
    )


# --------------------------------------------------------------------------- #
# Schema-level                                                                #
# --------------------------------------------------------------------------- #


def test_apply_directive_delta_merges_constraints():
    expert, _, _ = _make_agents()
    d = _directive()
    new = expert.apply_directive_delta(d, {
        "structured_constraints": {
            "boost_category": "tailored-pants",  # overwrite
            "price_diff_pct_max": 50,            # add
        },
    })
    assert new.structured_constraints["boost_category"] == "tailored-pants"
    assert new.structured_constraints["price_diff_pct_max"] == 50
    assert new.iteration == d.iteration + 1
    assert new.goal == d.goal  # unchanged


def test_apply_directive_delta_drops_key_on_null():
    expert, _, _ = _make_agents()
    d = _directive()
    new = expert.apply_directive_delta(d, {
        "structured_constraints": {"boost_category": None},
    })
    assert "boost_category" not in new.structured_constraints


def test_apply_directive_delta_empty_delta_returns_same():
    expert, _, _ = _make_agents()
    d = _directive()
    assert expert.apply_directive_delta(d, None) is d
    assert expert.apply_directive_delta(d, {}) is d


# --------------------------------------------------------------------------- #
# Lifecycle paths                                                             #
# --------------------------------------------------------------------------- #


def _patch_expert_issue(expert: ExpertAgent, directive: Directive) -> None:
    expert.issue_directive = MagicMock(return_value=directive)  # type: ignore[assignment]


def _patch_trend_interpret(trend: TrendAgent, interp: TrendInterpretation) -> None:
    trend.interpret_trend = MagicMock(return_value=interp)  # type: ignore[assignment]


def test_negotiation_consensus_when_no_tension():
    """No tensions → outcome=consensus, no challenge/response messages."""
    expert, trend, bus = _make_agents()
    _patch_expert_issue(expert, _directive())
    _patch_trend_interpret(trend, _trend_interp())
    trend.detect_tensions = MagicMock(return_value=[])  # type: ignore[assignment]

    directive, _ = run_phase2(
        expert, trend, brief="brief", bus=bus,
        enable_negotiation=True, max_negotiation_turns=2, tension_threshold=0.5,
    )
    assert directive.negotiation_log is not None
    assert directive.negotiation_log.final_outcome == "consensus"
    assert directive.negotiation_log.messages == []


def test_negotiation_below_threshold_does_not_fire():
    """Tensions exist but max severity < threshold → still consensus."""
    expert, trend, bus = _make_agents()
    _patch_expert_issue(expert, _directive())
    _patch_trend_interpret(trend, _trend_interp())
    trend.detect_tensions = MagicMock(return_value=[_tension(severity=0.4)])  # type: ignore[assignment]

    directive, _ = run_phase2(
        expert, trend, brief="brief", bus=bus,
        enable_negotiation=True, max_negotiation_turns=2, tension_threshold=0.7,
    )
    assert directive.negotiation_log.final_outcome == "consensus"
    assert directive.negotiation_log.messages == []


def test_negotiation_accept_path_applies_delta():
    expert, trend, bus = _make_agents()
    init = _directive()
    _patch_expert_issue(expert, init)
    _patch_trend_interpret(trend, _trend_interp())

    challenge = NegotiationMessage(
        turn=0, from_agent="trend", to_agent="expert",
        message_type="challenge", content_nl="raise price ceiling",
        tensions=[_tension(severity=0.8)],
        proposed_directive_delta={
            "structured_constraints": {"price_diff_pct_max": 50},
        },
    )
    response = NegotiationMessage(
        turn=0, from_agent="expert", to_agent="trend",
        message_type="accept", content_nl="agreed with brief",
        tensions=[],
        proposed_directive_delta=None,
    )
    trend.detect_tensions = MagicMock(return_value=[_tension(severity=0.8)])  # type: ignore[assignment]
    trend.challenge_directive = MagicMock(return_value=challenge)  # type: ignore[assignment]
    expert.respond_to_challenge = MagicMock(return_value=response)  # type: ignore[assignment]

    directive, _ = run_phase2(
        expert, trend, brief="brief", bus=bus,
        enable_negotiation=True, max_negotiation_turns=2, tension_threshold=0.5,
    )
    assert directive.negotiation_log.final_outcome == "consensus"
    assert len(directive.negotiation_log.messages) == 2
    # Challenge's delta was applied (accept path uses challenge's delta).
    assert directive.structured_constraints["price_diff_pct_max"] == 50
    assert directive.iteration == init.iteration + 1


def test_negotiation_reject_path_holds_directive():
    expert, trend, bus = _make_agents()
    init = _directive()
    _patch_expert_issue(expert, init)
    _patch_trend_interpret(trend, _trend_interp())

    challenge = NegotiationMessage(
        turn=0, from_agent="trend", to_agent="expert",
        message_type="challenge", content_nl="drop forbid",
        tensions=[_tension(severity=0.9)],
        proposed_directive_delta={"structured_constraints": {"boost_category": "lingerie"}},
    )
    response = NegotiationMessage(
        turn=0, from_agent="expert", to_agent="trend",
        message_type="reject", content_nl="off-brief",
        tensions=[], proposed_directive_delta=None,
    )
    trend.detect_tensions = MagicMock(return_value=[_tension(severity=0.9)])
    trend.challenge_directive = MagicMock(return_value=challenge)
    expert.respond_to_challenge = MagicMock(return_value=response)

    directive, _ = run_phase2(
        expert, trend, brief="brief", bus=bus,
        enable_negotiation=True, max_negotiation_turns=2, tension_threshold=0.5,
    )
    assert directive.negotiation_log.final_outcome == "expert_held"
    # No delta was applied → constraints unchanged.
    assert directive.structured_constraints == init.structured_constraints
    assert directive.iteration == init.iteration


def test_negotiation_counter_then_consensus_in_second_turn():
    """Turn 0: counter (Expert proposes own delta). Turn 1: no more tensions → consensus."""
    expert, trend, bus = _make_agents()
    init = _directive()
    _patch_expert_issue(expert, init)
    _patch_trend_interpret(trend, _trend_interp())

    challenge_0 = NegotiationMessage(
        turn=0, from_agent="trend", to_agent="expert",
        message_type="challenge", content_nl="raise ceiling to 50%",
        tensions=[_tension(severity=0.9)],
        proposed_directive_delta={"structured_constraints": {"price_diff_pct_max": 50}},
    )
    counter_0 = NegotiationMessage(
        turn=0, from_agent="expert", to_agent="trend",
        message_type="counter", content_nl="40% is fair",
        tensions=[],
        proposed_directive_delta={"structured_constraints": {"price_diff_pct_max": 40}},
    )

    detect_calls = {"n": 0}
    def _detect(directive, interp):
        detect_calls["n"] += 1
        # Turn 0: tension. Turn 1: no tension → consensus.
        return [_tension(severity=0.9)] if detect_calls["n"] == 1 else []

    trend.detect_tensions = MagicMock(side_effect=_detect)  # type: ignore[assignment]
    trend.challenge_directive = MagicMock(return_value=challenge_0)
    expert.respond_to_challenge = MagicMock(return_value=counter_0)

    directive, _ = run_phase2(
        expert, trend, brief="brief", bus=bus,
        enable_negotiation=True, max_negotiation_turns=2, tension_threshold=0.5,
    )
    assert directive.negotiation_log.final_outcome == "consensus"
    # Expert's counter delta is what got applied.
    assert directive.structured_constraints["price_diff_pct_max"] == 40
    # detect_tensions was called twice (once per turn).
    assert detect_calls["n"] == 2


def test_negotiation_max_turns_reached_when_loop_keeps_countering():
    """Both sides counter forever — loop terminates at max_turns."""
    expert, trend, bus = _make_agents()
    init = _directive()
    _patch_expert_issue(expert, init)
    _patch_trend_interpret(trend, _trend_interp())

    trend.detect_tensions = MagicMock(return_value=[_tension(severity=0.9)])
    trend.challenge_directive = MagicMock(return_value=NegotiationMessage(
        turn=0, from_agent="trend", to_agent="expert",
        message_type="challenge", content_nl="raise",
        tensions=[_tension(severity=0.9)],
        proposed_directive_delta={"structured_constraints": {"price_diff_pct_max": 50}},
    ))
    expert.respond_to_challenge = MagicMock(return_value=NegotiationMessage(
        turn=0, from_agent="expert", to_agent="trend",
        message_type="counter", content_nl="40 is fair",
        tensions=[],
        proposed_directive_delta={"structured_constraints": {"price_diff_pct_max": 40}},
    ))

    directive, _ = run_phase2(
        expert, trend, brief="brief", bus=bus,
        enable_negotiation=True, max_negotiation_turns=2, tension_threshold=0.5,
    )
    assert directive.negotiation_log.final_outcome == "max_turns_reached"
    assert len(directive.negotiation_log.messages) == 4  # 2 turns × (challenge + response)


def test_negotiation_disabled_via_flag():
    """When ``enable_negotiation=False`` the loop never runs (no log, no detect call)."""
    expert, trend, bus = _make_agents()
    _patch_expert_issue(expert, _directive())
    _patch_trend_interpret(trend, _trend_interp())
    trend.detect_tensions = MagicMock(return_value=[_tension(severity=0.99)])  # would fire if checked

    directive, _ = run_phase2(
        expert, trend, brief="brief", bus=bus,
        enable_negotiation=False,
    )
    # detect_tensions never called.
    assert trend.detect_tensions.call_count == 0
    # No log attached — disable flag should leave directive untouched.
    assert directive.negotiation_log is None
