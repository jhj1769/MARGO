"""Google Trends source adapter (L1, Enhancement 4).

Wraps the existing :class:`GoogleTrendsClient` (which already handles the
pytrends batching / throttling / urllib3 compatibility shim) into the
``TrendSourceAdapter`` interface. We only fetch one keyword at a time here
so the upstream batching is bypassed — the multisource pipeline trades
some bandwidth for a uniform call shape across sources.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from adapters.trends.base import TrendSourceAdapter
from adapters.trends.multisource_schema import RawSourceData

log = logging.getLogger(__name__)


class GoogleTrendsAdapter(TrendSourceAdapter):
    """Mass-search-attention signal."""

    source_name = "google_trends"
    source_type = "search"
    reliability_prior = 0.70

    def __init__(self, client: Optional[Any] = None) -> None:
        # Lazy-load pytrends so unit tests + Reddit/GDELT-only runs don't
        # need the dependency. ``client`` is a ``GoogleTrendsClient`` from
        # ``adapters.trends.google_trends`` but we only require duck-typing
        # for ``fetch_interest_over_time``.
        if client is None:
            from adapters.trends.google_trends import GoogleTrendsClient
            client = GoogleTrendsClient()
        self._client = client

    def fetch(
        self,
        keyword: str,
        time_window: tuple[str, str],
        *,
        region: str = "US",
    ) -> Optional[RawSourceData]:
        timeframe = f"{time_window[0]} {time_window[1]}"
        try:
            iot = self._client.fetch_interest_over_time(
                [keyword], timeframe=timeframe, geo=region, batch_size=1,
            )
        except Exception as e:  # noqa: BLE001 — pytrends raises a zoo of errors
            log.warning("GoogleTrendsAdapter fetch failed for %r: %s", keyword, e)
            return None

        if iot is None or iot.empty or keyword not in iot.columns:
            log.info("GoogleTrendsAdapter: no data for %r in %s", keyword, timeframe)
            return RawSourceData(
                keyword=keyword,
                source_name=self.source_name,
                source_type=self.source_type,
                time_window=time_window,
                time_series=[],
                raw_payload={"empty": True},
            )

        series = iot[keyword].astype(float)
        time_series: list[tuple[str, float]] = [
            (ts.strftime("%Y-%m-%d"), float(v))
            for ts, v in series.items()
        ]
        return RawSourceData(
            keyword=keyword,
            source_name=self.source_name,
            source_type=self.source_type,
            time_window=time_window,
            time_series=time_series,
            raw_payload={"native_unit": "0-100 normalised within batch"},
        )
