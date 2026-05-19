"""Classic top-K recommendation metrics — leave-one-out variants."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _binary_relevance(ranked_ids: Sequence[str], positive_id: str) -> list[int]:
    return [1 if i == positive_id else 0 for i in ranked_ids]


def hit_rate(ranked_ids: Sequence[str], positive_id: str, k: int) -> float:
    """HR@k for a single user."""
    return float(positive_id in list(ranked_ids)[:k])


def ndcg(ranked_ids: Sequence[str], positive_id: str, k: int) -> float:
    """NDCG@k for a single user (leave-one-out → IDCG = 1)."""
    rel = _binary_relevance(list(ranked_ids)[:k], positive_id)
    dcg = sum(r / math.log2(idx + 2) for idx, r in enumerate(rel))
    return dcg  # IDCG = 1 because there's only one positive.


def mean(iterable: Iterable[float]) -> float:
    values = list(iterable)
    return sum(values) / len(values) if values else 0.0
