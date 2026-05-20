"""Item Agent — context-aware self-description for one catalog item."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.agents.base import BaseAgent
from core.protocol.messages import Directive, TrendInterpretation

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Schemas                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class ItemFacts:
    """Immutable facts that an Item Agent is allowed to reason about."""

    item_id: str
    title: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def base_text(self) -> str:
        """Plain text used by the retriever and as a fallback description."""
        attrs = " ".join(f"{k}={v}" for k, v in self.attributes.items() if v)
        return f"{self.title}. {attrs}".strip()


class _DescribeOut(BaseModel):
    description: str
    audience_fit: float = Field(ge=0.0, le=1.0)
    anchored_attributes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #


class ItemAgent(BaseAgent):
    """A lightweight agent — one instance per *retrieved* item per query.

    Instantiating a hundred of these for every recommendation is fine because
    they share the global LLM client and only run a single LLM call per
    Phase-3 invocation.
    """

    def __init__(self, facts: ItemFacts, **kw) -> None:
        super().__init__(agent_id=f"item:{facts.item_id}", prompt_namespace="item", **kw)
        self.facts = facts
        self.audience_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Skill — self-description (Phase 3)                                  #
    # ------------------------------------------------------------------ #

    def self_describe(
        self,
        *,
        user_profile: str,
        directive: Directive,
        trend: Optional[TrendInterpretation] = None,
    ) -> _DescribeOut:
        prompt = self.render(
            "describe",
            directive=directive,
            trend=trend,
            user_profile=user_profile,
        )
        out = self._ask_structured(
            prompt,
            _DescribeOut,
            system=self._system(item=self.facts.__dict__),
        )
        self.audience_log.append(
            {
                "fit": out.audience_fit,
                "anchored": out.anchored_attributes,
            }
        )
        return out
