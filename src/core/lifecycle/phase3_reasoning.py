"""Phase 3 — Multi-Agent Reasoning.

1. Retriever (BGE-M3 by default) builds a directive-aware candidate pool.
2. Each retrieved item is wrapped in an Item Agent that produces a
   ``self_describe`` string in the current context.
3. The User Agent evaluates the augmented candidate views and emits a
   Top-K ranking with a 3-layer rationale.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol, Sequence

import time

import re
from dataclasses import replace

from core.agents.item_agent import ItemAgent, ItemFacts
from core.agents.trend_agent import apply_cohort_conditioning
from core.agents.user_agent import CandidateView, UserAgent
from core.memory.schemas import ItemMemory, make_item_reception_event
from core.protocol.messages import Directive, RankedItem, TrendInterpretation
from data.fashion.gender import compatible as gender_compatible


# Trend boost — bounded additive bonus applied to retrieval hits whose item
# text overlaps with v5 trend keywords. Kept small by design so the
# directive-shaped retrieval pool still dominates ordering (Asymmetric
# Authority: Expert > Trend). With BGE scores in the 0.55-0.70 range, a
# 0.10-cap boost is a tiebreaker among similar-rank candidates, not a
# re-ordering of the pool.
_TREND_BOOST_PER_KW = 0.02
_TREND_BOOST_MAX = 0.10


def _build_trend_patterns(keywords: list[str]) -> list[tuple[str, "re.Pattern"]]:
    """Pre-compile patterns for trend matching at the keyword level.

    Matching is *strict* on purpose:
      * Multi-word keywords ("sheer fabrics") match only as full phrases.
      * Single-word keywords ("lime") use ``\\b`` anchoring so we don't
        spuriously credit "limestone" or "sublime".

    Why not relax to token-level matching? Because credit for a token
    fragment (e.g. "sheer" -> any sheer hosiery) is no longer a *trend*
    signal — it's token noise. Empty intersection between a narrow
    directive (e.g. graduation) and general-season trends is a legitimate
    result: the Asymmetric Authority design says trends are advisory only,
    so the trend channel correctly goes inert when no overlap exists.
    """
    out: list[tuple[str, re.Pattern]] = []
    for kw in keywords or []:
        k = kw.strip().lower()
        if not k:
            continue
        if " " in k:
            out.append((k, re.compile(re.escape(k))))
        else:
            out.append((k, re.compile(rf"\b{re.escape(k)}\b")))
    return out


def _item_text_for_trend(facts: "ItemFacts") -> str:
    parts: list[str] = []
    if facts.title:
        parts.append(facts.title)
    if facts.attributes:
        for v in facts.attributes.values():
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                parts.extend(str(x) for x in v if x)
            else:
                parts.append(str(v))
    return " ".join(parts).lower()

log = logging.getLogger(__name__)


class _RetrieverProto(Protocol):
    def retrieve_with_directive(
        self, user_profile: str, directive_nl: str, k: int = 100, **kw
    ) -> Sequence: ...


def run_phase3(
    *,
    user: UserAgent,
    retriever: _RetrieverProto,
    catalog: dict[str, ItemFacts],
    directive: Directive,
    trend: Optional[TrendInterpretation],
    top_k: int,
    candidate_size: int = 100,
    rerank_window: int = 30,
    use_item_self_describe: bool = True,
    processed_dir: Optional[Path] = None,
    enable_trend_position: bool = True,
    enable_cohort_conditional_trend: bool = True,
    memory_root: Optional[Path] = None,
    enable_item_memory: bool = True,
    user_gender_focus: Optional[str] = None,
    item_gender_lookup: Optional[dict[str, str]] = None,
    **agent_kw,
) -> tuple[list[RankedItem], int]:
    """Run reasoning and return (ranked, candidate_pool_size).

    ``rerank_window`` caps how many candidates the User Agent has to
    score — Phase 3 cost is linear in this number.

    When ``processed_dir`` is provided we load each candidate's
    :class:`ItemAudienceProfile` from
    ``<processed_dir>/buyer_aggregate/<item_id>.json`` (Enhancement 2) and
    pass it to the Item Agent's self-description. Missing profiles cause a
    silent fall-through — the agent describes the item without buyer
    grounding instead of failing.
    """
    if not user.state.profile:
        user.build_profile()

    retrieve_kw: dict = {}
    if trend and trend.keywords:
        retrieve_kw["trend_keywords"] = trend.keywords
    hits = list(
        retriever.retrieve_with_directive(
            user.state.profile,
            directive.natural_language,
            k=candidate_size,
            **retrieve_kw,
        )
    )
    raw_pool_size = len(hits)

    # Gender filter (post-retrieve, pre-rerank). A women-focused user keeps
    # women's + unisex items; a male-focused user keeps men's + unisex; mixed
    # / unknown users see everything. This is a type-error guard, not a
    # taste preference — see data/fashion/gender.py for the policy.
    gender_dropped = 0
    if (
        user_gender_focus
        and user_gender_focus not in ("mixed", "unknown")
        and item_gender_lookup is not None
    ):
        before = len(hits)
        hits = [
            h for h in hits
            if gender_compatible(
                user_gender_focus,
                item_gender_lookup.get(h.item_id, "unknown"),
            )
        ]
        gender_dropped = before - len(hits)
        log.info(
            "Phase 3: gender filter (%s) dropped %d/%d candidates",
            user_gender_focus, gender_dropped, before,
        )

    # Trend-aware light boost (post gender filter, pre window cap).
    #
    # Goal: among directive-compliant candidates, lift those that also
    # exhibit v5 trend keywords in their title/attributes so the final
    # Top-K reflects the trend without breaking the directive's category
    # boost. Bounded by _TREND_BOOST_MAX so a directive-aligned candidate
    # cannot be unseated by a trend-only candidate.
    trend_boosted_count = 0
    if trend and trend.keywords:
        patterns = _build_trend_patterns(list(trend.keywords))
        if patterns:
            boosted_hits = []
            for h in hits:
                facts = catalog.get(h.item_id)
                if facts is None:
                    boosted_hits.append(h)
                    continue
                text = _item_text_for_trend(facts)
                overlap = sum(1 for _, pat in patterns if pat.search(text))
                if overlap > 0:
                    bonus = min(overlap * _TREND_BOOST_PER_KW, _TREND_BOOST_MAX)
                    boosted_hits.append(replace(h, score=h.score + bonus))
                    trend_boosted_count += 1
                else:
                    boosted_hits.append(h)
            # Re-sort so the boost actually changes ordering.
            hits = sorted(boosted_hits, key=lambda x: -x.score)
            log.info(
                "Phase 3: trend boost applied to %d/%d candidates (max=+%.2f, per-kw=+%.2f)",
                trend_boosted_count, len(hits), _TREND_BOOST_MAX, _TREND_BOOST_PER_KW,
            )

    log.info(
        "Phase 3: retrieved %d candidates (raw=%d, gender_dropped=%d, "
        "trend_boosted=%d, window=%d, top_k=%d)",
        len(hits), raw_pool_size, gender_dropped, trend_boosted_count,
        rerank_window, top_k,
    )

    if processed_dir is not None:
        from data.fashion.audience_loader import load_audience_profile
    else:
        load_audience_profile = None  # type: ignore[assignment]

    # Phase B — viewer cohort drives both Item Agent memory retrieval and
    # (later) the reception event append. None for users without a
    # structured preference state ⇒ Item Agent falls back to v3 behaviour.
    viewer_cohort: Optional[str] = (
        user.state.preference_state.cohort_signature
        if user.state.preference_state is not None
        else None
    )
    # Single turn_id for this recommend call — stitches the audience
    # claim event back to its reception events for post-hoc verification.
    turn_id = f"{user.state.user_id}@{int(time.time() * 1000)}"
    use_item_memory = enable_item_memory and memory_root is not None

    window = [h for h in hits if h.item_id in catalog][:rerank_window]
    candidates: list[CandidateView] = []
    for h in window:
        facts = catalog[h.item_id]
        if use_item_self_describe:
            try:
                audience = (
                    load_audience_profile(h.item_id, processed_dir)
                    if load_audience_profile is not None
                    else None
                )
                desc = ItemAgent(
                    facts,
                    llm=user.llm,
                    prompts=user.prompts,
                    schema_validator=user.schema,
                    memory_root=memory_root if use_item_memory else None,
                )
                # ``enable_trend_position`` False → don't surface the trend to
                # the Item Agent so the describe prompt skips its
                # ``trend_position`` section. The User Agent still sees the
                # global trend in its evaluation call.
                trend_for_item = trend if enable_trend_position else None
                ds = desc.self_describe(
                    user_profile=user.state.profile,
                    directive=directive,
                    trend=trend_for_item,
                    audience_profile=audience,
                    viewer_cohort=viewer_cohort,
                    turn_id=turn_id,
                )
                description = ds.description
            except Exception:
                log.exception("ItemAgent failure for %s; falling back to facts", h.item_id)
                description = facts.base_text()
        else:
            description = facts.base_text()
        candidates.append(CandidateView(item_id=h.item_id, description=description))

    # Phase A — Cohort-Conditional Trend Application.
    # The Trend Agent's interpretation is global per-directive. Before the
    # User Agent evaluates, we reweight it for THIS user's cohort by
    # dropping cohort-conflicting attributes/keywords. Rule-based, no
    # learning, no extra LLM call. No-op when the user lacks a structured
    # cohort signature (legacy callers behave exactly as before).
    trend_for_user = trend
    if (
        enable_cohort_conditional_trend
        and trend is not None
        and user.state.preference_state is not None
        and user.state.preference_state.cohort_signature
    ):
        trend_for_user = apply_cohort_conditioning(
            trend, user.state.preference_state.cohort_signature
        )
        log.debug(
            "Phase 3: cohort-conditioned trend for user %s — "
            "rising kept=%d, keywords kept=%d",
            user.state.user_id,
            sum(len(v) for v in trend_for_user.rising_attributes.values()),
            len(trend_for_user.keywords),
        )

    ranked = user.evaluate_candidates(candidates, directive=directive, trend=trend_for_user)

    # Phase B — append a reception event to each ranked item's memory.
    # This is what makes Item Agent self-description *evolve* across
    # turns: future self_describe calls retrieve THESE events. No-op
    # when memory isn't wired (preserves v3 behaviour).
    if use_item_memory:
        for rk in ranked:
            try:
                ItemMemory(rk.item_id, root=memory_root).append(
                    make_item_reception_event(
                        item_id=rk.item_id,
                        viewer_user_id=user.state.user_id,
                        viewer_cohort=viewer_cohort or "unknown",
                        score=max(0.0, min(1.0, float(rk.score))),
                        rationale_summary=(rk.rationale.personal or "")[:240] or None,
                        turn_id=turn_id,
                    )
                )
            except Exception:  # noqa: BLE001 — memory write must not break recommend
                log.exception(
                    "Phase 3: reception_event append failed for %s",
                    rk.item_id,
                )

    return ranked[:top_k], len(hits)
