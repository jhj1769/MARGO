"""User Agent — the shopper's NL representative.

Skills (paper §2.User Agent):
    * ``build_profile``       — Phase 1, derives an NL persona from history.
    * ``query_preference``    — Express NL preference given context.
    * ``evaluate_candidate``  — Rank items using preference + directive + trend.
    * ``update_profile``      — Reflect on interaction outcome.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from agents.base import BaseAgent
from protocol.messages import (
    Directive,
    RankedItem,
    Rationale,
    TrendInterpretation,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Schemas                                                                      #
# --------------------------------------------------------------------------- #


class _ProfileOut(BaseModel):
    profile: str
    preferred_price_band: str = ""


class _RankedItemRaw(BaseModel):
    item_id: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: Rationale


class _EvalOut(BaseModel):
    ranked: list[_RankedItemRaw]


@dataclass
class CandidateView:
    """What the User Agent sees about a candidate during evaluation.

    The ``description`` is populated by the Item Agent in Phase 3.
    """

    item_id: str
    description: str


@dataclass
class UserState:
    user_id: str
    history: list[str] = field(default_factory=list)
    profile: str = ""
    preferred_price_band: str = ""
    memory: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #


class UserAgent(BaseAgent):
    """Single-user NL reasoner."""

    def __init__(self, user_id: str, history: list[str], **kw) -> None:
        super().__init__(agent_id=f"user:{user_id}", prompt_namespace="user_agent", **kw)
        self.state = UserState(user_id=user_id, history=list(history))

    # ------------------------------------------------------------------ #
    # Skill 1 — profile generation (Phase 1)                              #
    # ------------------------------------------------------------------ #

    def build_profile(self) -> str:
        prompt = self.render("profile", history=self.state.history)
        out = self._ask_structured(prompt, _ProfileOut, system=self._system(profile=""))
        self.state.profile = out.profile
        self.state.preferred_price_band = out.preferred_price_band
        log.info("user %s profile built (%d chars)", self.state.user_id, len(out.profile))
        return out.profile

    # ------------------------------------------------------------------ #
    # Skill 2 — candidate evaluation (Phase 3)                            #
    # ------------------------------------------------------------------ #

    def evaluate_candidates(
        self,
        candidates: list[CandidateView],
        *,
        directive: Directive,
        trend: Optional[TrendInterpretation] = None,
    ) -> list[RankedItem]:
        if not self.state.profile:
            self.build_profile()
        prompt = self.render(
            "evaluate",
            history=self.state.history,
            candidates=candidates,
            directive=directive,
            trend=trend,
        )
        out = self._ask_structured(
            prompt,
            _EvalOut,
            system=self._system(profile=self.state.profile),
        )
        # Map numeric IDs back to real item_ids (prompt uses [1], [2], ...)
        idx_to_id = {str(i + 1): c.item_id for i, c in enumerate(candidates)}
        valid_ids = {c.item_id for c in candidates}

        def _resolve_id(raw: str) -> Optional[str]:
            """Accept '1', '#1', '[1]', 'item 1', 'Item_1', etc."""
            import re
            cleaned = re.sub(r"[^0-9]", "", raw)
            if cleaned and cleaned in idx_to_id:
                return idx_to_id[cleaned]
            if raw in valid_ids:
                return raw
            return None

        ranked: list[RankedItem] = []
        seen: set[str] = set()
        for r in out.ranked:
            real_id = _resolve_id(r.item_id)
            if real_id is None:
                log.warning("user %s could not resolve item_id=%r; dropping", self.id, r.item_id)
                continue
            if real_id in seen:
                continue
            seen.add(real_id)
            ranked.append(RankedItem(item_id=real_id, score=r.score, rationale=r.rationale))
        ranked.sort(key=lambda r: r.score, reverse=True)

        # Fallback: if too many dropped, fill with remaining candidates
        if len(ranked) < len(candidates) // 2:
            log.warning(
                "user %s: only %d/%d resolved; backfilling rest",
                self.id, len(ranked), len(candidates),
            )
            default_rationale = Rationale(
                personal="(auto-filled due to LLM output issues)",
                directive="(auto-filled)",
                trend="(auto-filled)",
            )
            for c in candidates:
                if c.item_id not in seen:
                    seen.add(c.item_id)
                    ranked.append(RankedItem(
                        item_id=c.item_id, score=0.3, rationale=default_rationale,
                    ))
        return ranked

    # ------------------------------------------------------------------ #
    # Skill 3 — reflection (Phase 4 outcome)                              #
    # ------------------------------------------------------------------ #

    def update_profile(self, note: str) -> None:
        """Append a reflective note. Heavier rewrites happen offline."""
        self.state.memory.append(note)
