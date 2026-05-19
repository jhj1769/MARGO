"""Phase 4 — Validation & Refinement loop."""

from __future__ import annotations

import logging
from typing import Any, Optional

from margo.agents.expert_agent import ExpertAgent
from margo.protocol.messages import Directive, RankedItem, ValidationReport

log = logging.getLogger(__name__)


def run_phase4(
    expert: ExpertAgent,
    ranked: list[RankedItem],
    *,
    item_attrs: dict[str, dict[str, Any]],
    user_attrs: Optional[dict[str, Any]] = None,
    directive: Optional[Directive] = None,
) -> ValidationReport:
    return expert.validate_recommendation(
        ranked, item_attrs=item_attrs, user_attrs=user_attrs, directive=directive
    )
