"""User Agent — the shopper's NL representative.

Skills (paper §2.User Agent):
    * ``build_profile``       — Phase 1, derives an NL persona from history.
    * ``query_preference``    — Express NL preference given context.
    * ``evaluate_candidate``  — Rank items using preference + directive + trend.
    * ``update_preference_state`` — Reflect on a new interaction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from core.agents.base import BaseAgent
from core.protocol.messages import (
    AxisName,
    CohortStats,
    Directive,
    PreferenceAxis,
    RankedItem,
    Rationale,
    TrendInterpretation,
    UserPreferenceState,
    compute_cohort_signature,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Schemas                                                                      #
# --------------------------------------------------------------------------- #


class _ProfileOut(BaseModel):
    """LLM-side of profile generation.

    The four axes (style/price/category/brand) live on :class:`PreferenceAxis`,
    but only ``style`` is LLM-inferred. The other three are passed into the
    prompt as context and we never read them back from the LLM.
    """

    profile: str
    preferred_price_band: str = ""
    style_value: str = ""
    style_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    style_evidence: list[str] = Field(default_factory=list)
    style_secondary_values: list[str] = Field(default_factory=list)


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
    # New in Enhancement 1: structured 4-axis preference state.
    # ``None`` until ``build_profile`` runs with an items catalogue available.
    preference_state: Optional[UserPreferenceState] = None


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #


class UserAgent(BaseAgent):
    """Single-user NL reasoner."""

    def __init__(
        self,
        user_id: str,
        history: list[str],
        *,
        history_item_ids: Optional[list[str]] = None,
        items_df: Optional[pd.DataFrame] = None,
        processed_dir: Optional[Path] = None,
        rejected_item_ids: Optional[list[str]] = None,
        **kw,
    ) -> None:
        super().__init__(agent_id=f"user:{user_id}", prompt_namespace="user", **kw)
        self.state = UserState(user_id=user_id, history=list(history))
        # Inputs for Enhancement 1 deterministic axes. Both are required to
        # build the structured preference state; either being None means we
        # fall back to legacy NL-only behaviour.
        self._history_item_ids: list[str] = list(history_item_ids or [])
        self._items_df: Optional[pd.DataFrame] = items_df
        # Enhancement 1.5: directory that holds ``cohort_stats/*.json``.
        # When None, ``get_peer_signal`` returns the no-signal default.
        self._processed_dir: Optional[Path] = Path(processed_dir) if processed_dir else None
        self._cohort_cache: Optional[CohortStats] = None
        # Phase C — Dual-layer Rejected signal (rating 1-2 history for THIS user).
        # ``None`` ⇒ no rejected history available (37.2% of users have it).
        # Summarisation is lazy + cached so we pay it at most once per agent.
        self._rejected_item_ids: Optional[list[str]] = (
            list(rejected_item_ids) if rejected_item_ids else None
        )
        self._rejection_summary_cache: Optional[dict] = None
        self._rejection_summary_built: bool = False

    # ------------------------------------------------------------------ #
    # Skill 1 — profile generation (Phase 1)                              #
    # ------------------------------------------------------------------ #

    def build_profile(self) -> str:
        """Build the NL persona. Also populates ``preference_state`` if
        deterministic inputs (history_item_ids + items_df) are available.
        """
        det_axes_dicts = self._compute_deterministic_axes_or_empty()

        prompt = self.render(
            "profile",
            history=self.state.history,
            deterministic_axes=list(det_axes_dicts.values()),
        )
        out = self._ask_structured(prompt, _ProfileOut, system=self._system(profile=""))
        self.state.profile = out.profile
        self.state.preferred_price_band = out.preferred_price_band

        # Promote to structured state when we have deterministic axes.
        if det_axes_dicts:
            axes: list[PreferenceAxis] = []
            for axis_dict in det_axes_dicts.values():
                axes.append(PreferenceAxis(**axis_dict))
            axes.append(self._style_axis_from_llm(out))
            state = UserPreferenceState(
                user_id=self.state.user_id,
                profile_nl=out.profile,
                axes=axes,
                cohort_signature="",
                last_updated_at=time.time(),
            )
            state.cohort_signature = compute_cohort_signature(state)
            self.state.preference_state = state

        log.info("user %s profile built (%d chars)", self.state.user_id, len(out.profile))
        return out.profile

    def _compute_deterministic_axes_or_empty(self) -> dict[str, dict]:
        """Return {axis_name: axis_dict} or {} when inputs are missing."""
        if not self._history_item_ids or self._items_df is None or self._items_df.empty:
            return {}
        from data.fashion.preference_stats import compute_deterministic_axes  # local import keeps base-agent fast
        return compute_deterministic_axes(
            self.state.user_id, self._history_item_ids, self._items_df
        )

    def _style_axis_from_llm(self, out: _ProfileOut) -> PreferenceAxis:
        return PreferenceAxis(
            name="style_preference",
            value=out.style_value or "mixed",
            confidence=out.style_confidence,
            evidence=out.style_evidence,
            secondary_values=out.style_secondary_values,
            derived_from="llm_inferred",
            stability=1.0,
        )

    # ------------------------------------------------------------------ #
    # Skill 1.5 — peer signal (Enhancement 1.5)                           #
    # ------------------------------------------------------------------ #

    def get_peer_signal(self, candidate_item_id: str) -> tuple[float, str]:
        """Return ``(buy_ratio, explanation_nl)`` for a candidate.

        ``buy_ratio`` is the fraction of users in this user's cohort that
        purchased ``candidate_item_id`` (rating ≥ 4) in the training set.
        Returns ``(0.0, "<reason>")`` if no reliable cohort signal exists —
        either because we have no structured preference state, no cohort
        store, or the cohort is below ``MIN_COHORT_SIZE``.
        """
        if (
            self.state.preference_state is None
            or not self.state.preference_state.cohort_signature
            or self._processed_dir is None
        ):
            return 0.0, "No cohort context available for this user"

        from data.fashion.cohort_loader import MIN_COHORT_SIZE, load_cohort_stats

        if self._cohort_cache is None:
            self._cohort_cache = load_cohort_stats(
                self.state.preference_state.cohort_signature, self._processed_dir
            )
        cohort = self._cohort_cache
        if cohort is None or cohort.size < MIN_COHORT_SIZE:
            return 0.0, "Cohort too small for reliable peer signal"

        ratio = cohort.peer_signal_for(candidate_item_id)
        if ratio == 0.0:
            explanation = (
                f"No one in your cohort ({cohort.size} users) has bought this item"
            )
        else:
            explanation = (
                f"{ratio:.0%} of users in your cohort ({cohort.size} users) "
                f"purchased this item"
            )
        return ratio, explanation

    # ------------------------------------------------------------------ #
    # Skill 1.6 — Rejected layer summary (Phase C)                        #
    # ------------------------------------------------------------------ #

    def get_rejection_pattern(self) -> Optional[dict]:
        """Return a compact summary of the user's Rejected (rating 1-2)
        history, or ``None`` when there's no signal.

        Cached on the agent — summarisation involves a parquet join that
        we shouldn't repeat per evaluate call. ``None`` is a valid cached
        result (for users without rejected history).
        """
        if self._rejection_summary_built:
            return self._rejection_summary_cache
        self._rejection_summary_built = True
        if not self._rejected_item_ids or self._items_df is None:
            return None
        from data.fashion.rejection_pattern import summarise_rejection_pattern
        try:
            self._rejection_summary_cache = summarise_rejection_pattern(
                self._rejected_item_ids, self._items_df,
            )
        except Exception:  # noqa: BLE001 — never let pattern building break evaluate
            log.exception(
                "rejection pattern build failed for user %s", self.state.user_id,
            )
            self._rejection_summary_cache = None
        return self._rejection_summary_cache

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
        axes_for_prompt = (
            self.state.preference_state.axes if self.state.preference_state else []
        )
        # Peer signal per candidate (Enhancement 1.5). When the user lacks
        # cohort context, every entry's ratio is 0.0 with an explanatory note,
        # which the prompt template renders as "no peer signal available".
        peer_signals = {c.item_id: self.get_peer_signal(c.item_id) for c in candidates}
        # Phase C — Dual-layer Rejected pattern. None when the user has
        # no rejected interactions (or too few) → prompt omits the section.
        rejection_pattern = self.get_rejection_pattern()
        prompt = self.render(
            "evaluate",
            history=self.state.history,
            candidates=candidates,
            directive=directive,
            trend=trend,
            axes=axes_for_prompt,
            peer_signals=peer_signals,
            rejection_pattern=rejection_pattern,
            policy_hint=directive.policy_hint,
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

    def update_preference_state(
        self,
        new_item_id: str,
        rating: Optional[float] = None,
    ) -> Optional[UserPreferenceState]:
        """Re-run deterministic axes after a new interaction.

        Only the statistical axes (price/category/brand) and their stability
        are recomputed. Style is kept from the previous LLM-inferred value
        until ``build_profile`` is called again.

        Returns the updated state, or ``None`` if no structured state exists
        yet (i.e. deterministic inputs were never provided).
        """
        if self._items_df is None or not self.state.preference_state:
            return None
        self._history_item_ids.append(new_item_id)
        if rating is not None:
            # We don't gate on rating today, but record it in evidence later.
            log.debug("rating=%s recorded for %s", rating, new_item_id)

        from data.fashion.preference_stats import compute_deterministic_axes
        new_det = compute_deterministic_axes(
            self.state.user_id, self._history_item_ids, self._items_df
        )

        # Keep the existing style axis; replace the statistical ones.
        old_style = self.state.preference_state.get_axis("style_preference")
        new_axes: list[PreferenceAxis] = [PreferenceAxis(**a) for a in new_det.values()]
        if old_style is not None:
            new_axes.append(old_style)

        updated = UserPreferenceState(
            user_id=self.state.user_id,
            profile_nl=self.state.preference_state.profile_nl,
            axes=new_axes,
            cohort_signature="",
            last_updated_at=time.time(),
        )
        updated.cohort_signature = compute_cohort_signature(updated)
        # Re-running the deterministic axes can change the signature; bust the cohort cache.
        if updated.cohort_signature != self.state.preference_state.cohort_signature:
            self._cohort_cache = None
        self.state.preference_state = updated
        return self.state.preference_state
