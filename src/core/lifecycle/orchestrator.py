"""End-to-end orchestrator: stitches Phase 2 → 3 → 4 with refine-loop.

The orchestrator is intentionally *imperative* — it's easier to debug than
a LangGraph state machine, and the lifecycle is small (3 active phases).
The behaviour, however, mirrors the diagram in the paper exactly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
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
    # Enhancement 5 — Expert↔Trend negotiation. Implemented and tested but
    # default-off: the MD-intent vs trend negotiation semantics are not yet
    # well-grounded enough to ship by default. Flip to True (and tune
    # max_negotiation_turns / tension_threshold) to re-enable.
    enable_negotiation: bool = False
    max_negotiation_turns: int = 1
    tension_threshold: float = 0.7
    # ----------------------------------------------------------------
    # Ablation flags (v3 §9.4). Each toggles one Enhancement *at runtime*
    # so we can A/B that contribution without redeploying code. Defaults
    # turn everything on; set False to ablate.
    # ----------------------------------------------------------------
    enable_multi_axis: bool = True          # Enhancement 1
    enable_peer_signal: bool = True         # Enhancement 1.5
    enable_audience_profile: bool = True    # Enhancement 2
    enable_trend_position: bool = True      # Enhancement 3
    enable_trend_snapshot: bool = True      # v5 season snapshot path
    # Phase A (Heterogeneous Stakeholder Reasoning):
    # Reweight the global trend interpretation per user cohort before the
    # User Agent evaluates candidates. Off → v3 uniform-broadcast behaviour.
    enable_cohort_conditional_trend: bool = True
    # Phase B — Item Agent autobiographical memory.
    # On + memory_root set on the orchestrator ⇒ each Item Agent gets a
    # per-item JSONL store; receptions append after evaluate, audience
    # claims append inside self_describe. Off ⇒ NullMemory (v3 behaviour).
    enable_item_memory: bool = True
    # Phase C — User Agent Rejected layer (rating 1-2 history).
    # When False, even users with rejected history are evaluated as if
    # they had none (clean ablation). The Directive.policy_hint only
    # takes effect when this flag is on AND the user has rejected items.
    enable_rejected_layer: bool = True
    # Phase D — Expert Agent outcome logging + similar-brief retrieval.
    # On + memory_root set ⇒ each Directive's outcome is appended after
    # Phase 4; subsequent issue_directive calls retrieve similar past
    # briefs as evidence. Off ⇒ NullMemory, v3 behaviour.
    enable_expert_memory: bool = True


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
    # When set, Phase 3 will load per-item ``ItemAudienceProfile`` from
    # ``<processed_dir>/buyer_aggregate/*.json`` (Enhancement 2).
    processed_dir: Optional[Path] = None
    # When set, agentic memory (Phase B/C/D) writes per-entity JSONL
    # stores under this root (memory/item/, memory/user/, memory/trend/,
    # memory/expert/). None ⇒ NullMemory everywhere ⇒ v3 behaviour.
    memory_root: Optional[Path] = None
    # Per-item gender label (women / men / kids / unisex / unknown). Used
    # as a Phase 3 candidate filter — see data/fashion/gender.py. ``None``
    # disables the filter (back-compat).
    item_gender_lookup: Optional[dict[str, str]] = None

    def recommend(
        self,
        user: UserAgent,
        *,
        brief: str,
        config: Optional[MargoRunConfig] = None,
        user_attrs: Optional[dict[str, Any]] = None,
        user_gender_focus: Optional[str] = None,
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
                enable_negotiation=cfg.enable_negotiation,
                max_negotiation_turns=cfg.max_negotiation_turns,
                tension_threshold=cfg.tension_threshold,
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
                processed_dir=self.processed_dir if cfg.enable_audience_profile else None,
                enable_trend_position=cfg.enable_trend_position,
                enable_cohort_conditional_trend=cfg.enable_cohort_conditional_trend,
                memory_root=self.memory_root,
                enable_item_memory=cfg.enable_item_memory,
                user_gender_focus=user_gender_focus,
                item_gender_lookup=self.item_gender_lookup,
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
                # Phase D — record the *final* (passing) outcome for this
                # brief so future issue_directive calls can use it as
                # experiential evidence. n_refinements = iterations-1
                # because the first attempt is iteration 1.
                if cfg.enable_expert_memory:
                    self.expert.record_outcome(
                        brief, directive, report,
                        n_refinements=iteration - 1,
                    )
                break
            if iteration == cfg.max_iterations:
                log.warning(
                    "max_iterations=%d reached without convergence; returning last attempt",
                    cfg.max_iterations,
                )
                # Also record non-converging outcomes — these are the
                # most valuable signal for "this brief shape needs more
                # care next time".
                if cfg.enable_expert_memory:
                    self.expert.record_outcome(
                        brief, directive, report,
                        n_refinements=iteration - 1,
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
