"""Item Agent — context-aware self-description for one catalog item.

Phase B (Heterogeneous Stakeholder Reasoning):
    The Item Agent is no longer a stateless wrapper. Each instance can be
    constructed with an :class:`ItemMemory` that accumulates
    *receptions* (what scores this item received from which cohorts) and
    *audience claims* (what cohorts this item claimed to serve well).

    Before each ``self_describe`` we retrieve the slice relevant to the
    current viewer's cohort and inject it as prompt evidence — the agent
    can then ground its audience_fit_claim_nl in past receptions instead
    of guessing.

    After each ``self_describe`` we append any audience_fit_claim_nl as
    an unverified ``ItemAudienceClaimEvent``. A separate offline job
    fills in verification fields (Phase E), at which point
    ``self-correction`` warnings start flowing into future prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.agents.base import BaseAgent
from core.memory.base import AgentMemory, NullMemory
from core.memory.schemas import (
    ItemMemory,
    make_item_audience_claim_event,
)
from core.protocol.messages import (
    Directive,
    ItemAudienceProfile,
    ItemDescription,
    TrendInterpretation,
)

log = logging.getLogger(__name__)


# Threshold below which past audience claims for a cohort are deemed
# unreliable and the agent is told to avoid that claim. Verified scores
# below this trigger a prompt-level warning. Conservative default (0.5
# on the 0-1 score scale) — tunable per dataset.
_CLAIM_ACCURACY_THRESHOLD = 0.5
# Minimum number of *verified* claims for the warning to fire. Below
# this, sample is too small to draw a conclusion (cold-start safe).
_MIN_VERIFIED_CLAIMS_FOR_WARNING = 2


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


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #


class ItemAgent(BaseAgent):
    """A lightweight agent — one instance per *retrieved* item per query.

    Instantiating a hundred of these for every recommendation is fine because
    they share the global LLM client and only run a single LLM call per
    Phase-3 invocation.

    Memory wiring (Phase B):
        Pass ``memory_root`` to attach a per-item :class:`ItemMemory` that
        accumulates reception + audience-claim events across recommend
        turns. Pass ``None`` (default) to keep the agent stateless — same
        behaviour as the pre-Phase-B v3 implementation.
    """

    def __init__(
        self,
        facts: ItemFacts,
        *,
        memory_root: Optional[Path] = None,
        memory: Optional[AgentMemory] = None,
        **kw,
    ) -> None:
        super().__init__(agent_id=f"item:{facts.item_id}", prompt_namespace="item", **kw)
        self.facts = facts
        self.audience_log: list[dict[str, Any]] = []
        # Explicit ``memory=`` overrides ``memory_root`` (useful in tests
        # for injecting an in-memory store). Otherwise we open a per-item
        # JSONL store rooted at memory_root, or fall back to NullMemory.
        if memory is not None:
            self.memory: AgentMemory = memory
        elif memory_root is not None:
            self.memory = ItemMemory(facts.item_id, root=memory_root)
        else:
            self.memory = NullMemory()

    # ------------------------------------------------------------------ #
    # Skill — self-description (Phase 3)                                  #
    # ------------------------------------------------------------------ #

    def self_describe(
        self,
        *,
        user_profile: str,
        directive: Directive,
        trend: Optional[TrendInterpretation] = None,
        audience_profile: Optional[ItemAudienceProfile] = None,
        viewer_cohort: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> ItemDescription:
        """Produce the headline NL description plus optional grounding fields.

        Parameters
        ----------
        viewer_cohort
            The current viewer's ``cohort_signature``. When provided AND
            this agent has a non-Null memory, we retrieve past reception
            + claim events for THIS cohort and inject them as prompt
            evidence. Also used as the ``claim_target_cohort`` on any
            audience claim we append after the LLM call.
        turn_id
            Optional opaque identifier (e.g. ``"{user_id}@{timestamp}"``)
            recorded on the audience-claim event so post-hoc verification
            can stitch the claim back to its originating turn.
        """
        # Phase B retrieval — only when we have both a viewer cohort AND
        # a real memory. Cold-start (no memory file yet) returns [] which
        # the prompt treats as "no past evidence available".
        past_reception = _summarise_past_reception(self.memory, viewer_cohort)
        claim_warning = _self_correction_warning(self.memory, viewer_cohort)

        prompt = self.render(
            "describe",
            directive=directive,
            trend=trend,
            user_profile=user_profile,
            audience=audience_profile,
            has_trend=trend is not None,
            past_reception=past_reception,
            claim_warning=claim_warning,
        )
        out = self._ask_structured(
            prompt,
            ItemDescription,
            system=self._system(item=self.facts.__dict__),
        )

        # Phase B append — record the audience claim (unverified) so a
        # post-hoc verification job can later fill in actual cohort scores.
        # No-op when the agent has NullMemory (default) so we don't pay
        # disk IO for agents the engine didn't wire memory for.
        if out.audience_fit_claim_nl and viewer_cohort:
            try:
                self.memory.append(make_item_audience_claim_event(
                    item_id=self.facts.item_id,
                    claim_nl=out.audience_fit_claim_nl,
                    claim_target_cohort=viewer_cohort,
                    issued_turn_id=turn_id,
                ))
            except Exception:  # noqa: BLE001 — memory write must not fail self_describe
                log.exception("ItemAgent.self_describe memory append failed for %s",
                              self.facts.item_id)

        self.audience_log.append(
            {
                "fit": out.audience_fit,
                "anchored": out.anchored_attributes,
                "audience_grounded": audience_profile is not None,
                "trend_positioned": out.trend_position is not None,
                "memory_used": past_reception is not None,
            }
        )
        return out


# --------------------------------------------------------------------------- #
# Memory summarisers (kept module-level so they're easy to unit-test)         #
# --------------------------------------------------------------------------- #


def _summarise_past_reception(
    memory: AgentMemory,
    viewer_cohort: Optional[str],
) -> Optional[dict[str, Any]]:
    """Return a compact summary of past reception events for ``viewer_cohort``.

    None when there's no memory, no cohort, or no matching events — the
    prompt template renders this as "no past reception evidence".
    """
    if isinstance(memory, NullMemory) or not viewer_cohort:
        return None
    events = memory.retrieve(query={"viewer_cohort": viewer_cohort}, top_k=20)
    receptions = [e for e in events if e.event_type == "item_reception"]
    if not receptions:
        return None
    scores = [float(e.payload.get("score", 0.0)) for e in receptions]
    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "viewer_cohort": viewer_cohort,
        "n": len(receptions),
        "avg_score": round(avg, 3),
        # The 3 most recent rationale summaries (if any) — useful for the
        # agent to see *why* it scored well/poorly in this cohort before.
        "recent_rationales": [
            e.payload.get("rationale_summary")
            for e in receptions[:3]
            if e.payload.get("rationale_summary")
        ],
    }


def _self_correction_warning(
    memory: AgentMemory,
    viewer_cohort: Optional[str],
) -> Optional[str]:
    """Return a prompt-ready warning when past claims about ``viewer_cohort``
    were verified to be inaccurate. None means *no warning* — either the
    agent has never made a claim about this cohort, or those claims were
    verified accurate, or no verification has run yet.
    """
    if isinstance(memory, NullMemory) or not viewer_cohort:
        return None
    events = memory.retrieve(query={"viewer_cohort": viewer_cohort}, top_k=100)
    claims = [
        e for e in events
        if e.event_type == "item_audience_claim"
        and e.payload.get("claim_target_cohort") == viewer_cohort
    ]
    verified = [
        c for c in claims
        if c.payload.get("verified_avg_score") is not None
    ]
    if len(verified) < _MIN_VERIFIED_CLAIMS_FOR_WARNING:
        return None
    avg_verified = sum(
        float(c.payload["verified_avg_score"]) for c in verified
    ) / len(verified)
    if avg_verified >= _CLAIM_ACCURACY_THRESHOLD:
        return None
    return (
        f"Past audience claims targeting cohort '{viewer_cohort}' had a "
        f"verified average score of {avg_verified:.2f} (over {len(verified)} "
        f"verified claims), below the reliability threshold of "
        f"{_CLAIM_ACCURACY_THRESHOLD}. Avoid asserting strong appeal to this "
        f"cohort unless your current evidence is unusually strong."
    )
