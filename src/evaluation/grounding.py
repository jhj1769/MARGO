"""Grounding metrics — IHR / VDR / CADR / SVR."""

from __future__ import annotations

from typing import Iterable

from grounding.schema_validator import SchemaValidator
from grounding.vocabulary import Vocabulary
from protocol.messages import Message, MessageType, RankedItem


# --------------------------------------------------------------------------- #
# IHR — Item Hallucination Rate                                                #
# --------------------------------------------------------------------------- #


def ihr(ranked: Iterable[RankedItem], catalog_ids: set[str]) -> float:
    """Fraction of recommended IDs that do not appear in the catalog."""
    ranked_list = list(ranked)
    if not ranked_list:
        return 0.0
    out = sum(1 for r in ranked_list if r.item_id not in catalog_ids)
    return out / len(ranked_list)


# --------------------------------------------------------------------------- #
# VDR — Vocabulary Drift Rate                                                  #
# --------------------------------------------------------------------------- #


def vdr(texts: Iterable[str], vocab: Vocabulary) -> float:
    """Out-of-vocab token rate across the supplied texts."""
    total_in = total_out = 0
    for t in texts:
        i, o, _ = vocab.check_text(t)
        total_in += i
        total_out += o
    denom = total_in + total_out
    return total_out / denom if denom else 0.0


# --------------------------------------------------------------------------- #
# CADR — Cross-Agent Disagreement Rate                                         #
# --------------------------------------------------------------------------- #


def cadr(trace: Iterable[Message]) -> float:
    """Fraction of validation rounds where Expert disagrees with Trend.

    Concretely: out of all VALIDATION messages, how many have
    ``payload['passed'] == False`` while a TREND broadcast was active.
    Useful as an early-warning that Trend hallucinations are driving the
    refinement loop.
    """
    trend_active = False
    validations = 0
    disagreements = 0
    for msg in trace:
        if msg.type == MessageType.BROADCAST and msg.sender == "trend":
            trend_active = True
        if msg.type == MessageType.VALIDATION:
            validations += 1
            if trend_active and not msg.payload.get("passed", True):
                disagreements += 1
    return disagreements / validations if validations else 0.0


# --------------------------------------------------------------------------- #
# SVR — Schema Violation Rate                                                  #
# --------------------------------------------------------------------------- #


def svr_from_validator(validator: SchemaValidator) -> float:
    return validator.counter.rate()
