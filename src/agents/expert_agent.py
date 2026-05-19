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
from dataclasses import dataclass, field
from typing import Any, Optional

from agents.base import BaseAgent
from protocol.messages import Directive, RankedItem, ValidationReport

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class ExpertState:
    persona: str
    current_directive: Optional[Directive] = None
    history: list[Directive] = field(default_factory=list)


class ExpertAgent(BaseAgent):
    """The MD / editor / curator persona."""

    def __init__(
        self,
        persona: str,
        vocabulary: Optional[dict[str, list[str]]] = None,
        **kw,
    ) -> None:
        super().__init__(agent_id="expert", prompt_namespace="expert_agent", **kw)
        self.state = ExpertState(persona=persona)
        self.vocabulary = vocabulary or {}

    # ------------------------------------------------------------------ #
    # Phase 2 — directive issuance                                        #
    # ------------------------------------------------------------------ #

    def issue_directive(self, brief: str) -> Directive:
        prompt = self.render("directive", brief=brief, vocabulary=self.vocabulary)
        directive = self._ask_structured(
            prompt,
            Directive,
            system=self._system(persona=self.state.persona),
        )
        self.state.current_directive = directive
        self.state.history.append(directive)
        return directive

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
