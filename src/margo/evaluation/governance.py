"""Governance metrics — DCR (Directive Compliance) & TAS (Trend Alignment)."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from margo.agents.expert_agent import _check_structured
from margo.protocol.messages import Directive, RankedItem, TrendInterpretation


# --------------------------------------------------------------------------- #
# DCR                                                                          #
# --------------------------------------------------------------------------- #


def dcr(
    ranked: list[RankedItem],
    *,
    directive: Directive,
    item_attrs: dict[str, dict[str, Any]],
    user_attrs: Optional[dict[str, Any]] = None,
) -> float:
    """Fraction of Top-K items that violate **zero** structured constraints.

    Only the *automatically checkable* part of the directive is counted
    here; the soft NL component is left to the Expert Agent's compliance
    score (which is reported alongside DCR in :mod:`scripts.evaluate`).
    """
    if not ranked:
        return 0.0
    violations = _check_structured(directive, ranked, item_attrs, user_attrs or {})
    bad_ids: set[str] = set()
    for v in violations:
        # Violation strings start with 'item <id>' – cheap parsing.
        if v.startswith("item "):
            bad_ids.add(v.split(" ", 2)[1])
    good = sum(1 for r in ranked if r.item_id not in bad_ids)
    return good / len(ranked)


# --------------------------------------------------------------------------- #
# TAS                                                                          #
# --------------------------------------------------------------------------- #


def tas(
    ranked: list[RankedItem],
    *,
    trend: TrendInterpretation,
    item_attrs: dict[str, dict[str, Any]],
) -> float:
    """Fraction of Top-K items whose attributes intersect the trend keywords."""
    if not ranked or not trend.keywords:
        return 0.0
    trend_tokens = {t.lower() for t in trend.keywords}
    # Also fold rising_attributes into the comparison set.
    for vs in trend.rising_attributes.values():
        trend_tokens.update(v.lower() for v in vs)

    matches = 0
    for r in ranked:
        attrs = item_attrs.get(r.item_id, {})
        haystack = " ".join(str(v).lower() for v in _flatten_values(attrs.values()))
        if any(tok in haystack for tok in trend_tokens):
            matches += 1
    return matches / len(ranked)


def _flatten_values(values: Iterable[Any]) -> Iterable[Any]:
    for v in values:
        if isinstance(v, (list, tuple, set)):
            yield from v
        elif v is not None:
            yield v
