"""Phase 2 — Directive Generation.

Single LangGraph node that fans the Expert's directive + Trend
interpretation out to the rest of the agent ensemble.
"""

from __future__ import annotations

from typing import Optional

from core.agents.expert_agent import ExpertAgent
from core.agents.trend_agent import TrendAgent
from core.protocol.messages import (
    Directive,
    Message,
    MessageType,
    TrendInterpretation,
)
from core.protocol.router import MessageBus


def run_phase2(
    expert: ExpertAgent,
    trend: TrendAgent,
    brief: str,
    bus: MessageBus,
    *,
    refined_directive: Optional[Directive] = None,
    use_cache: bool = True,
) -> tuple[Directive, TrendInterpretation]:
    """Produce / refresh the directive and the trend broadcast."""
    if refined_directive is not None:
        directive = refined_directive
        expert.state.current_directive = directive
        expert.state.history.append(directive)
    else:
        directive = expert.issue_directive(brief=brief)

    bus.publish(
        Message(
            type=MessageType.DIRECTIVE,
            sender=expert.id,
            receivers=["user:*", "item:*", "trend"],
            payload=directive.model_dump(),
        )
    )

    directive_hint = directive.natural_language or directive.goal or None
    interpretation = trend.interpret_trend(
        use_cache=use_cache,
        directive_hint=directive_hint,
    )
    trend.broadcast(interpretation, receivers=["user:*", "item:*"])
    return directive, interpretation
