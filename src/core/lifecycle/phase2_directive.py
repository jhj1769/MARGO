"""Phase 2 — Directive Generation (+ Enhancement 5 negotiation loop).

Bare flow:
    Expert.issue_directive  →  Trend.interpret_trend  →  publish

With negotiation enabled:
    Expert.issue_directive
      → Trend.interpret_trend
      → Trend.detect_tensions
          (if max-severity ≥ threshold) →
              Trend.challenge_directive
              Expert.respond_to_challenge
                  accept   → apply delta, exit
                  reject   → keep directive, exit
                  counter  → apply Expert's delta, continue another turn (≤ max_turns)
      → publish final directive + the negotiation_log
"""

from __future__ import annotations

import logging
from typing import Optional

from core.agents.expert_agent import ExpertAgent
from core.agents.trend_agent import TrendAgent
from core.protocol.messages import (
    Directive,
    Message,
    MessageType,
    NegotiationLog,
    TrendInterpretation,
)
from core.protocol.router import MessageBus

log = logging.getLogger(__name__)


def run_phase2(
    expert: ExpertAgent,
    trend: TrendAgent,
    brief: str,
    bus: MessageBus,
    *,
    refined_directive: Optional[Directive] = None,
    use_cache: bool = True,
    # Enhancement 5 — negotiation knobs (defaults match v3 §14)
    enable_negotiation: bool = True,
    max_negotiation_turns: int = 1,
    tension_threshold: float = 0.7,
) -> tuple[Directive, TrendInterpretation]:
    """Produce / refresh the directive and the trend interpretation.

    When ``enable_negotiation`` is True, the Trend Agent may raise tensions
    against the directive and the Expert may respond. The final directive
    carries the audit trail in ``directive.negotiation_log``.
    """
    if refined_directive is not None:
        directive = refined_directive
        expert.state.current_directive = directive
        expert.state.history.append(directive)
    else:
        directive = expert.issue_directive(brief=brief)

    directive_hint = directive.natural_language or directive.goal or None
    interpretation = trend.interpret_trend(
        use_cache=use_cache,
        directive_hint=directive_hint,
    )

    # ---------- Negotiation loop ---------------------------------------
    if enable_negotiation and max_negotiation_turns > 0:
        directive, interpretation = _negotiate(
            expert=expert,
            trend=trend,
            brief=brief,
            directive=directive,
            interpretation=interpretation,
            max_turns=max_negotiation_turns,
            threshold=tension_threshold,
            use_cache=use_cache,
            directive_hint_fn=lambda d: d.natural_language or d.goal,
        )

    bus.publish(
        Message(
            type=MessageType.DIRECTIVE,
            sender=expert.id,
            receivers=["user:*", "item:*", "trend"],
            payload=directive.model_dump(),
        )
    )
    trend.broadcast(interpretation, receivers=["user:*", "item:*"])
    return directive, interpretation


def _negotiate(
    *,
    expert: ExpertAgent,
    trend: TrendAgent,
    brief: str,
    directive: Directive,
    interpretation: TrendInterpretation,
    max_turns: int,
    threshold: float,
    use_cache: bool,
    directive_hint_fn,
) -> tuple[Directive, TrendInterpretation]:
    log_obj = NegotiationLog()
    outcome = "consensus"

    for turn in range(max_turns):
        try:
            tensions = trend.detect_tensions(directive, interpretation)
        except Exception as e:  # noqa: BLE001
            log.warning("detect_tensions failed (turn=%d): %s — bailing on negotiation", turn, e)
            outcome = "consensus"
            break

        if not tensions or max((t.severity for t in tensions), default=0.0) < threshold:
            log.info("turn=%d no qualifying tension (max severity < %.2f) — done",
                     turn, threshold)
            outcome = "consensus"
            break

        try:
            challenge = trend.challenge_directive(directive, tensions, turn=turn)
        except Exception as e:  # noqa: BLE001
            log.warning("challenge_directive failed (turn=%d): %s — bailing", turn, e)
            outcome = "consensus"
            break
        log_obj.messages.append(challenge)

        try:
            response = expert.respond_to_challenge(directive, challenge, brief, turn=turn)
        except Exception as e:  # noqa: BLE001
            log.warning("respond_to_challenge failed (turn=%d): %s — bailing", turn, e)
            outcome = "consensus"
            break
        log_obj.messages.append(response)

        if response.message_type == "accept":
            directive = expert.apply_directive_delta(directive, challenge.proposed_directive_delta)
            outcome = "consensus"
            break
        elif response.message_type == "reject":
            outcome = "expert_held"
            break
        elif response.message_type == "counter":
            directive = expert.apply_directive_delta(directive, response.proposed_directive_delta)
            # Re-interpret trend against the new directive before next turn.
            interpretation = trend.interpret_trend(
                use_cache=use_cache,
                directive_hint=directive_hint_fn(directive),
            )
            # Fall through to next turn.
    else:
        outcome = "max_turns_reached"

    log_obj.final_outcome = outcome
    # Attach audit trail.
    directive = directive.model_copy(update={"negotiation_log": log_obj})
    log.info(
        "negotiation finished: outcome=%s, turns=%d",
        outcome, len(log_obj.messages),
    )
    return directive, interpretation
