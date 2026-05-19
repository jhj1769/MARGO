"""Evaluation suite — standard + governance + grounding metrics."""

from margo.evaluation.governance import dcr, tas
from margo.evaluation.grounding import cadr, ihr, svr_from_validator, vdr
from margo.evaluation.standard import hit_rate, ndcg

__all__ = [
    "ndcg",
    "hit_rate",
    "dcr",
    "tas",
    "ihr",
    "vdr",
    "cadr",
    "svr_from_validator",
]
