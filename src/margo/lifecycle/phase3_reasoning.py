"""Phase 3 — Multi-Agent Reasoning.

1. Retriever (BGE-M3 by default) builds a directive-aware candidate pool.
2. Each retrieved item is wrapped in an Item Agent that produces a
   ``self_describe`` string in the current context.
3. The User Agent evaluates the augmented candidate views and emits a
   Top-K ranking with a 3-layer rationale.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, Sequence

from margo.agents.item_agent import ItemAgent, ItemFacts
from margo.agents.user_agent import CandidateView, UserAgent
from margo.protocol.messages import Directive, RankedItem, TrendInterpretation

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
    **agent_kw,
) -> tuple[list[RankedItem], int]:
    """Run reasoning and return (ranked, candidate_pool_size).

    ``rerank_window`` caps how many candidates the User Agent has to
    score — Phase 3 cost is linear in this number.
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
    log.info(
        "Phase 3: retrieved %d candidates (window=%d, top_k=%d)",
        len(hits),
        rerank_window,
        top_k,
    )

    window = [h for h in hits if h.item_id in catalog][:rerank_window]
    candidates: list[CandidateView] = []
    for h in window:
        facts = catalog[h.item_id]
        if use_item_self_describe:
            try:
                desc = ItemAgent(facts, llm=user.llm, prompts=user.prompts, schema_validator=user.schema)
                ds = desc.self_describe(
                    user_profile=user.state.profile,
                    directive=directive,
                    trend=trend,
                )
                description = ds.description
            except Exception:
                log.exception("ItemAgent failure for %s; falling back to facts", h.item_id)
                description = facts.base_text()
        else:
            description = facts.base_text()
        candidates.append(CandidateView(item_id=h.item_id, description=description))

    ranked = user.evaluate_candidates(candidates, directive=directive, trend=trend)
    return ranked[:top_k], len(hits)
