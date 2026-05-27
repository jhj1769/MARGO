"""L3 — Reliability-weighted aggregation across sources (Enhancement 4).

Per-keyword we collect one :class:`SourceSignal` from each source that
returned data. The aggregator answers two questions:

1. **Aggregated lifecycle** — what does the weighted majority of sources say?
   Weights are each source's ``reliability_prior`` × its own ``confidence``
   on this keyword. GDELT (0.85) outranks Google (0.70) outranks Wiki (0.60),
   so a single editorial "rising" can beat two weak "stable" votes.

2. **Disagreement flag** — do any sources actively contradict the aggregate
   on direction (rising vs declining)? If yes, we keep the dissent visible
   in ``disagreement_nl`` and let the LLM hedge instead of silencing it.

Niche / stable disagreements are *not* surfaced as a flag because they
represent uncertainty, not contradiction.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Mapping

from adapters.trends.multisource_schema import (
    Lifecycle,
    MultiSourceTrendSignal,
    SourceSignal,
)

log = logging.getLogger(__name__)


# Directional axis. ``stable`` and ``niche`` are non-directional and don't
# count toward "disagreement".
_DIRECTIONAL = {"rising", "declining"}
_OPPOSITE = {"rising": "declining", "declining": "rising"}


def compute_consensus(
    per_source: Mapping[str, SourceSignal],
    *,
    reliability_priors: Mapping[str, float],
) -> tuple[float, Lifecycle, bool, str]:
    """Return ``(consensus_score, aggregated_lifecycle, disagreement, nl)``.

    ``consensus_score`` is the fraction of weighted votes that landed on the
    winning lifecycle (in [0, 1]). 1.0 means unanimous (all sources agree),
    0.5 means a perfect split, etc.
    """
    if not per_source:
        return 0.0, "stable", False, "no sources returned data"

    # Sum the (prior × confidence) weight that each source pushes onto its
    # own lifecycle vote.
    weights: dict[Lifecycle, float] = defaultdict(float)
    total = 0.0
    for name, sig in per_source.items():
        w = reliability_priors.get(name, 0.5) * sig.confidence
        weights[sig.lifecycle] += w
        total += w

    if total <= 0:
        return 0.0, "stable", False, "all sources reported zero confidence"

    aggregated_lifecycle = max(weights.items(), key=lambda kv: kv[1])[0]
    consensus_score = weights[aggregated_lifecycle] / total

    # Disagreement: any source voted opposite to the aggregate?
    disagreement = False
    dissent_notes: list[str] = []
    if aggregated_lifecycle in _DIRECTIONAL:
        opp = _OPPOSITE[aggregated_lifecycle]
        for name, sig in per_source.items():
            if sig.lifecycle == opp:
                disagreement = True
                dissent_notes.append(f"{name} reports {opp}")

    if disagreement:
        nl = (
            f"weighted majority → {aggregated_lifecycle}, but "
            + "; ".join(dissent_notes)
        )
    else:
        nl = f"weighted majority → {aggregated_lifecycle}, no source contradicts"

    return round(consensus_score, 4), aggregated_lifecycle, disagreement, nl


def aggregate_keyword(
    keyword: str,
    per_source: Mapping[str, SourceSignal],
    *,
    reliability_priors: Mapping[str, float],
) -> MultiSourceTrendSignal:
    """Construct the per-keyword L3 record."""
    score, lifecycle, disagreement, nl = compute_consensus(
        per_source, reliability_priors=reliability_priors,
    )
    return MultiSourceTrendSignal(
        keyword=keyword,
        sources=dict(per_source),
        consensus_score=score,
        aggregated_lifecycle=lifecycle,
        disagreement_flag=disagreement,
        disagreement_nl=nl if disagreement else None,
    )
