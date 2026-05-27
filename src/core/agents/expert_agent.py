"""Expert Agent — Governance authority.

The Expert Agent owns three skills (paper §2.Expert Agent):
    * ``issue_directive``         — Phase 2.
    * ``validate_recommendation`` — Phase 4, checks Top-K against the directive.
    * ``refine_directive``        — Phase 4 → 2 loop on failure.

Hard structured constraints (e.g. ``price_diff_pct_max``) are checked
*programmatically* by :func:`_check_structured` so the LLM doesn't have
to do arithmetic. Soft NL intent is judged by the LLM.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from core.agents.base import BaseAgent
from core.memory.base import AgentMemory, NullMemory
from core.memory.schemas import (
    ExpertDirectiveOutcomeEvent,
    ExpertMemory,
    make_expert_outcome_event,
)
from core.protocol.messages import (
    Directive,
    NegotiationMessage,
    RankedItem,
    ValidationReport,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Phase D — Similar-brief retrieval (cosine over TF-IDF-lite vectors)         #
# --------------------------------------------------------------------------- #
# We deliberately avoid scikit-learn: a 60-line in-house tokeniser keeps the
# baseline path import-light and makes the retrieval logic auditable in one
# place. The text corpus here is in the dozens-to-hundreds of past briefs,
# so even O(N²) cosine is cheap.

_TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{2,}")
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "these", "those",
    "into", "onto", "over", "under", "more", "less", "very", "much",
    "should", "would", "could", "must", "have", "has", "had",
    "are", "was", "were", "been", "being",
    "but", "not", "any", "all", "some", "each", "every",
    "also", "still", "even", "only", "just",
})


def _tokenise_brief(text: str) -> list[str]:
    """Tokens: lowercase a-z words ≥3 chars, stopwords removed."""
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _vectorise(tokens: list[str]) -> dict[str, float]:
    """Term-count vector (raw TF). Good enough for short briefs."""
    return dict(Counter(tokens))


def _cosine(v1: dict[str, float], v2: dict[str, float]) -> float:
    """Standard cosine over term-count vectors; 0 when either is empty."""
    if not v1 or not v2:
        return 0.0
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[t] * v2[t] for t in common)
    n1 = math.sqrt(sum(v * v for v in v1.values()))
    n2 = math.sqrt(sum(v * v for v in v2.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0


def retrieve_similar_briefs(
    memory: AgentMemory,
    brief: str,
    *,
    top_k: int = 3,
    min_similarity: float = 0.15,
) -> list[dict[str, Any]]:
    """Return up to ``top_k`` past Directive outcomes whose brief_summary is
    closest (cosine) to ``brief``. Empty when memory is empty or no past
    brief clears ``min_similarity``.

    Each result dict contains: brief_summary, directive_goal,
    n_refinements, final_compliance_score, passed, similarity.
    """
    if isinstance(memory, NullMemory):
        return []
    query_vec = _vectorise(_tokenise_brief(brief))
    if not query_vec:
        return []
    events = memory.retrieve(top_k=200)  # enough to score
    candidates: list[tuple[float, dict[str, Any]]] = []
    for ev in events:
        if ev.event_type != "directive_outcome":
            continue
        past_brief = ev.payload.get("brief_summary", "")
        sim = _cosine(query_vec, _vectorise(_tokenise_brief(past_brief)))
        if sim < min_similarity:
            continue
        candidates.append((sim, {
            "brief_summary": past_brief,
            "directive_goal": ev.payload.get("directive_goal", ""),
            "n_refinements": ev.payload.get("n_refinements", 0),
            "final_compliance_score": ev.payload.get("final_compliance_score", 0.0),
            "passed": ev.payload.get("passed", False),
            "similarity": round(sim, 3),
        }))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in candidates[:top_k]]


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class ExpertState:
    persona: str
    current_directive: Optional[Directive] = None
    history: list[Directive] = field(default_factory=list)


class ExpertAgent(BaseAgent):
    """The MD / editor / curator persona.

    Phase D — Memory wiring:
        Pass ``memory_root`` to attach a per-persona :class:`ExpertMemory`
        that accumulates one ``directive_outcome`` event per Phase-4
        validation pass. Subsequent ``issue_directive`` calls then
        retrieve similar past briefs (cosine over TF-lite) and inject
        their outcomes as prompt evidence — turning the loop's history
        into actual experiential learning.
    """

    def __init__(
        self,
        persona: str,
        vocabulary: Optional[dict[str, list[str]]] = None,
        *,
        persona_id: Optional[str] = None,
        memory_root: Optional[Path] = None,
        memory: Optional[AgentMemory] = None,
        **kw,
    ) -> None:
        super().__init__(agent_id="expert", prompt_namespace="expert", **kw)
        self.state = ExpertState(persona=persona)
        self.vocabulary = vocabulary or {}
        self.persona_id = persona_id or "default"
        # Same pattern as ItemAgent: explicit memory= wins for tests,
        # otherwise memory_root spawns a JSONL store, otherwise NullMemory.
        if memory is not None:
            self.memory: AgentMemory = memory
        elif memory_root is not None:
            self.memory = ExpertMemory(self.persona_id, root=memory_root)
        else:
            self.memory = NullMemory()

    # ------------------------------------------------------------------ #
    # Phase 2 — directive issuance                                        #
    # ------------------------------------------------------------------ #

    def issue_directive(self, brief: str) -> Directive:
        # Phase D — retrieve up to 3 past briefs most similar to this one
        # and surface their outcomes as evidence. Cold start (NullMemory
        # or no similar past brief) → empty list → prompt section skipped.
        past_briefs = retrieve_similar_briefs(self.memory, brief, top_k=3)
        prompt = self.render(
            "directive",
            brief=brief,
            vocabulary=self.vocabulary,
            past_similar_briefs=past_briefs,
        )
        directive = self._ask_structured(
            prompt,
            Directive,
            system=self._system(persona=self.state.persona),
        )
        self.state.current_directive = directive
        self.state.history.append(directive)
        return directive

    # ------------------------------------------------------------------ #
    # Phase D — Outcome logging (Phase 4 caller hook)                     #
    # ------------------------------------------------------------------ #

    def record_outcome(
        self,
        brief: str,
        directive: Directive,
        report: ValidationReport,
        *,
        n_refinements: int,
    ) -> None:
        """Append an unverified outcome event to memory.

        Called by the orchestrator after Phase 4 with the *final* directive
        + the *final* report for this brief. ``n_refinements`` is the
        number of refine cycles consumed before convergence (0 = passed
        on first try).
        """
        if isinstance(self.memory, NullMemory):
            return
        # Keep brief_summary short and dense — it's the field future
        # retrievals will match against. Use the goal as a fallback when
        # the brief itself is too long for evidence rendering.
        brief_summary = (brief or directive.goal or "")[:280]
        # Slim the constraint dict — only keys the validator inspects,
        # so the stored summary stays small.
        constraint_keys = {
            "forbid_category", "boost_category",
            "price_min", "price_max", "price_diff_pct_max",
        }
        summary = {
            k: v for k, v in (directive.structured_constraints or {}).items()
            if k in constraint_keys
        }
        try:
            self.memory.append(make_expert_outcome_event(
                brief_summary=brief_summary,
                directive_goal=directive.goal,
                directive_constraints_summary=summary,
                n_refinements=max(0, int(n_refinements)),
                final_compliance_score=float(
                    max(0.0, min(1.0, report.compliance_score))
                ),
                passed=bool(report.passed),
            ))
        except Exception:  # noqa: BLE001 — never let bookkeeping break recommend
            log.exception("ExpertAgent.record_outcome append failed")

    # ------------------------------------------------------------------ #
    # Phase 4 — validation                                                #
    # ------------------------------------------------------------------ #

    def validate_recommendation(
        self,
        ranked: list[RankedItem],
        *,
        item_attrs: dict[str, dict[str, Any]],
        user_attrs: Optional[dict[str, Any]] = None,
        directive: Optional[Directive] = None,
    ) -> ValidationReport:
        """Hybrid validator: structured arithmetic + LLM judgement on NL."""
        directive = directive or self.state.current_directive
        if directive is None:
            raise ValueError("ExpertAgent.validate_recommendation called with no directive")

        structured_violations = _check_structured(directive, ranked, item_attrs, user_attrs or {})
        prompt = self.render(
            "validate",
            directive=directive,
            ranked=ranked,
            structured_violations=structured_violations,
        )
        report = self._ask_structured(
            prompt,
            ValidationReport,
            system=self._system(persona=self.state.persona),
        )
        # The LLM cannot lie its way out of hard violations.
        if structured_violations and report.passed:
            log.info("Overriding LLM 'passed' verdict due to structured violations")
            report = ValidationReport(
                passed=False,
                compliance_score=min(report.compliance_score, 0.5),
                violations=list(report.violations) + structured_violations,
                suggested_refinement=report.suggested_refinement
                or "Tighten constraints to forbid the violating items.",
            )
        return report

    # ------------------------------------------------------------------ #
    # Phase 2 — negotiation (Enhancement 5)                               #
    # ------------------------------------------------------------------ #

    def respond_to_challenge(
        self,
        directive: Directive,
        challenge: NegotiationMessage,
        brief: str,
        turn: int,
    ) -> NegotiationMessage:
        """Decide accept / reject / counter on a challenge from the Trend Agent."""
        prompt = self.render(
            "respond_negotiation",
            directive=directive.model_dump(),
            challenge=challenge.model_dump(),
            brief=brief,
            turn=turn,
        )
        msg = self._ask_structured(
            prompt,
            NegotiationMessage,
            system=self._system(persona=self.state.persona),
        )
        # Defensive: enforce direction so a confused LLM doesn't impersonate Trend.
        valid_types = {"accept", "reject", "counter"}
        return msg.model_copy(update={
            "from_agent": "expert",
            "to_agent": "trend",
            "message_type": msg.message_type if msg.message_type in valid_types else "reject",
            "turn": turn,
        })

    def apply_directive_delta(
        self,
        directive: Directive,
        delta: Optional[dict],
    ) -> Directive:
        """Return a NEW Directive with ``delta`` applied.

        Delta semantics:
            * ``structured_constraints`` is *merged* (set a key to ``None`` to drop it).
            * Other top-level fields (``natural_language``, ``goal``) are *replaced*
              when the delta value is non-null.

        The iteration counter is bumped so downstream consumers can tell that
        the directive moved.
        """
        if not delta:
            return directive

        merged_constraints = dict(directive.structured_constraints or {})
        delta_constraints = delta.get("structured_constraints")
        if isinstance(delta_constraints, dict):
            for k, v in delta_constraints.items():
                if v is None:
                    merged_constraints.pop(k, None)
                else:
                    merged_constraints[k] = v

        return directive.model_copy(update={
            "structured_constraints": merged_constraints,
            "goal": delta.get("goal") or directive.goal,
            "natural_language": delta.get("natural_language") or directive.natural_language,
            "iteration": directive.iteration + 1,
        })

    # ------------------------------------------------------------------ #
    # Phase 4 → 2 — refinement                                            #
    # ------------------------------------------------------------------ #

    def refine_directive(self, report: ValidationReport) -> Directive:
        if self.state.current_directive is None:
            raise ValueError("Nothing to refine — no current directive")
        prompt = self.render(
            "refine",
            directive=self.state.current_directive,
            report=report,
            vocabulary=self.vocabulary,
        )
        new_directive = self._ask_structured(
            prompt,
            Directive,
            system=self._system(persona=self.state.persona),
        )
        new_directive.iteration = self.state.current_directive.iteration + 1
        self.state.current_directive = new_directive
        self.state.history.append(new_directive)
        return new_directive


# --------------------------------------------------------------------------- #
# Structured constraint checker                                                #
# --------------------------------------------------------------------------- #


def _check_structured(
    directive: Directive,
    ranked: list[RankedItem],
    item_attrs: dict[str, dict[str, Any]],
    user_attrs: dict[str, Any],
) -> list[str]:
    """Mechanically verify hard rules. Returns a list of human-readable strings."""
    violations: list[str] = []
    constraints = directive.structured_constraints or {}

    forbid = set(_as_list(constraints.get("forbid_category")))
    boost = set(_as_list(constraints.get("boost_category")))
    price_diff_max = constraints.get("price_diff_pct_max")
    price_min = constraints.get("price_min")
    price_max = constraints.get("price_max")
    user_avg_price = user_attrs.get("avg_price")

    boost_seen = False
    for rank in ranked:
        attrs = item_attrs.get(rank.item_id)
        if attrs is None:
            violations.append(f"item {rank.item_id} not in catalog (IHR)")
            continue
        cats = {str(c).lower() for c in _as_list(attrs.get("category"))}

        if forbid and (cats & {c.lower() for c in forbid}):
            violations.append(f"item {rank.item_id} in forbidden category {cats & forbid}")
        if boost and (cats & {c.lower() for c in boost}):
            boost_seen = True

        price = attrs.get("price")
        if price is not None and price_min is not None and price < price_min:
            violations.append(f"item {rank.item_id} below price_min ({price} < {price_min})")
        if price is not None and price_max is not None and price > price_max:
            violations.append(f"item {rank.item_id} above price_max ({price} > {price_max})")
        if (
            price is not None
            and price_diff_max is not None
            and user_avg_price
            and user_avg_price > 0
        ):
            diff_pct = (price - user_avg_price) / user_avg_price * 100
            if diff_pct > price_diff_max:
                violations.append(
                    f"item {rank.item_id} price gap {diff_pct:.1f}% > "
                    f"price_diff_pct_max={price_diff_max}"
                )

    if boost and not boost_seen:
        violations.append(f"no Top-K item from boost_category={sorted(boost)}")

    return violations


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]
