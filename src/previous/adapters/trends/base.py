"""L1 — Trend source adapter abstract base (Enhancement 4).

A source adapter has one job: take a keyword + time window and return a
:class:`RawSourceData` time-series. Everything that converts that time-series
into a normalised lifecycle label happens downstream in L2.

Reliability prior conventions (used by L3 consensus voting):

    editorial (GDELT)     : 0.85 — closest to MD-facing trade press
    search    (Pinterest) : 0.75 — fashion-native discovery search
    search    (Google)    : 0.70 — mass attention, noisy in long tail
    media     (YouTube)   : 0.65 — creator/haul/lookbook activity

Numbers are starting points and can be tuned by re-running the offline
pipeline. The values are stored on the adapter class so the snapshot can
record what priors actually shaped the consensus.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from adapters.trends.multisource_schema import RawSourceData, SourceType

log = logging.getLogger(__name__)


class TrendSourceAdapter(ABC):
    """Each subclass is responsible for one external source."""

    source_name: str = ""
    source_type: SourceType = "media"
    reliability_prior: float = 0.5

    @abstractmethod
    def fetch(
        self,
        keyword: str,
        time_window: tuple[str, str],
        *,
        region: str = "US",
    ) -> Optional[RawSourceData]:
        """Return the time-series for ``keyword`` over ``time_window``.

        Implementations MUST:
          * return ``None`` on hard failure (network down, missing local data) —
            never raise. The pipeline interprets ``None`` as "this source has
            no opinion on this keyword".
          * return an empty ``time_series`` when the source genuinely has zero
            mentions, not ``None``. Zero-mention is a real signal.
          * return ISO-8601 date strings (no times) at native granularity;
            L2 handles resampling.
        """
        ...

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} name={self.source_name!r} "
            f"type={self.source_type!r} prior={self.reliability_prior}>"
        )
