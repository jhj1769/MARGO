"""End-to-end orchestrator: stitches Phase 2 → 3 → 4 with refine-loop.

The orchestrator is intentionally *imperative* — it's easier to debug than
a LangGraph state machine, and the lifecycle is small (3 active phases).
The behaviour, however, mirrors the diagram in the paper exactly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from core.agents.expert_agent import ExpertAgent
from core.agents.item_agent import ItemFacts
from core.agents.trend_agent import TrendAgent
from core.agents.user_agent import UserAgent
from core.lifecycle.phase2_directive import run_phase2
from core.lifecycle.phase3_reasoning import run_phase3
from core.lifecycle.phase4_validation import run_phase4
from core.protocol.messages import Directive, RecommendationResult
from core.protocol.router import MessageBus

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class MargoRunConfig:
    top_k: int = 10
    candidate_size: int = 100
    rerank_window: int = 30
    max_iterations: int = 3
    convergence_score: float = 0.85
    use_trend_cache: bool = True
    use_item_self_describe: bool = True


class _RetrieverProto(Protocol):
    def retrieve_with_directive(self, *a, **kw): ...


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class MargoOrchestrator:
    """Wires the four agents into a single ``recommend`` call.

    The User Agent + catalog facts are usually loaded once and reused;
    only the brief / directive change per request.
    """

    expert: ExpertAgent
    trend: TrendAgent
    retriever: _RetrieverProto
    catalog: dict[str, ItemFacts]
    item_attrs: dict[str, dict[str, Any]]
    bus: MessageBus = field(default_factory=MessageBus)

    def recommend(
        self,
        user: UserAgent,
        *,
        brief: str,
        config: Optional[MargoRunConfig] = None,
        user_attrs: Optional[dict[str, Any]] = None,
    ) -> RecommendationResult:
        cfg = config or MargoRunConfig()
        t0 = time.time()
        refined: Optional[Directive] = None

        for iteration in range(1, cfg.max_iterations + 1):
            directive, trend = run_phase2(
                self.expert,
                self.trend,
                brief=brief,
                bus=self.bus,
                refined_directive=refined,
                use_cache=cfg.use_trend_cache,
            )
            ranked, pool_size = run_phase3(
                user=user,
                retriever=self.retriever,
                catalog=self.catalog,
                directive=directive,
                trend=trend,
                top_k=cfg.top_k,
                candidate_size=cfg.candidate_size,
                rerank_window=cfg.rerank_window,
                use_item_self_describe=cfg.use_item_self_describe,
            )
            report = run_phase4(
                self.expert,
                ranked,
                item_attrs=self.item_attrs,
                user_attrs=user_attrs,
                directive=directive,
            )
            log.info(
                "iter=%d compliance=%.3f passed=%s violations=%d",
                iteration,
                report.compliance_score,
                report.passed,
                len(report.violations),
            )
            if report.passed or report.compliance_score >= cfg.convergence_score:
                break
            if iteration == cfg.max_iterations:
                log.warning(
                    "max_iterations=%d reached without convergence; returning last attempt",
                    cfg.max_iterations,
                )
                break
            refined = self.expert.refine_directive(report)

        result = RecommendationResult(
            user_id=user.state.user_id,
            directive=directive,
            trend=trend,
            top_k=ranked,
            candidate_pool_size=pool_size,
            phase4_passed=report.passed,
            iterations=iteration,
            trace=list(self.bus.history()),
        )
        log.info("recommend completed in %.1fs", time.time() - t0)
        return result
