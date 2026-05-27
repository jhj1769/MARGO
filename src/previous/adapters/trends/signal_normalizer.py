"""L2 — Convert RawSourceData time-series into a SourceSignal (Enhancement 4).

The normaliser is intentionally simple: a windowed moving-average ratio
(short MA / long MA) is the metric MDs informally describe as "is this
heating up vs the season". Single-point growth percentages are too jumpy
on long-tail keywords — the MA smoothes one-off spikes.

Tunable thresholds (see ``MARGO_Enhancement_Directive_v3.md §7.5``):

    short window  : 4 weeks
    long window   : 12 weeks
    rising        : growth_ratio >= 1.20
    declining     : growth_ratio <= 0.85
    niche cutoff  : volume_percentile < 0.20

Everything else is ``stable``. ``volume_percentile`` is computed at the
pipeline level (across all keywords in the snapshot) because no single
keyword can measure its own loudness.
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime, timedelta

from adapters.trends.multisource_schema import (
    Lifecycle,
    RawSourceData,
    SourceSignal,
)

log = logging.getLogger(__name__)


SHORT_WINDOW_DAYS = 28          # 4 weeks
LONG_WINDOW_DAYS = 84           # 12 weeks
RISING_THRESHOLD = 1.20
DECLINING_THRESHOLD = 0.85
NICHE_VOLUME_PCT = 0.20

# Numeric guardrails: when long_ma is effectively zero we cannot compute a
# ratio. Treat this as either "cold start → rising" or "dead → declining"
# depending on the short_ma sign.
EFFECTIVELY_ZERO = 1e-6


def _parse_iso(d: str) -> datetime:
    return datetime.fromisoformat(d)


def _moving_average(
    time_series: list[tuple[str, float]],
    end_iso: str,
    window_days: int,
) -> float:
    """Mean of points within ``[end - window_days, end]``. Zero if empty."""
    if not time_series:
        return 0.0
    end = _parse_iso(end_iso)
    start = end - timedelta(days=window_days)
    values = [v for d, v in time_series if start <= _parse_iso(d) <= end]
    return statistics.mean(values) if values else 0.0


def normalize_to_signal(
    raw: RawSourceData,
    *,
    volume_percentile: float = 0.5,
    short_window_days: int = SHORT_WINDOW_DAYS,
    long_window_days: int = LONG_WINDOW_DAYS,
) -> SourceSignal:
    """Reduce a time-series to a lifecycle label.

    ``volume_percentile`` is supplied by the caller because the percentile
    has to be ranked against the rest of the keyword pool — it isn't a
    property of a single time-series.
    """
    series = raw.time_series
    window_end = raw.time_window[1]

    short_ma = _moving_average(series, window_end, short_window_days)
    long_ma = _moving_average(series, window_end, long_window_days)

    if long_ma <= EFFECTIVELY_ZERO and short_ma <= EFFECTIVELY_ZERO:
        # No signal at all.
        lifecycle: Lifecycle = "niche"
        growth_ratio = 0.0
        confidence = 0.1
    elif long_ma <= EFFECTIVELY_ZERO:
        # Cold-start surge — short_ma > 0, long_ma ≈ 0.
        lifecycle = "rising"
        growth_ratio = 999.0
        confidence = 0.6
    else:
        growth_ratio = short_ma / long_ma
        if growth_ratio >= RISING_THRESHOLD:
            lifecycle = "rising"
        elif growth_ratio <= DECLINING_THRESHOLD:
            lifecycle = "declining"
        elif volume_percentile < NICHE_VOLUME_PCT:
            lifecycle = "niche"
        else:
            lifecycle = "stable"
        # Confidence: higher when more data points sit inside the long window.
        long_count = sum(
            1 for d, _ in series
            if _parse_iso(window_end) - timedelta(days=long_window_days)
            <= _parse_iso(d) <= _parse_iso(window_end)
        )
        # 12 weeks worth of *some* signal → full confidence.
        confidence = min(1.0, long_count / 12.0)

    return SourceSignal(
        keyword=raw.keyword,
        source_name=raw.source_name,
        source_type=raw.source_type,
        lifecycle=lifecycle,
        growth_ratio=round(growth_ratio, 4),
        short_ma=round(short_ma, 4),
        long_ma=round(long_ma, 4),
        volume_percentile=round(volume_percentile, 4),
        confidence=round(confidence, 4),
    )


def rank_volume_percentiles(raws: list[RawSourceData]) -> dict[tuple[str, str], float]:
    """Compute volume percentile per (keyword, source) across the snapshot.

    Returns a lookup table ``{(keyword, source_name): percentile_in_[0,1]}``.
    "Volume" here is the mean of the entire long window.

    Percentile is computed *per source* so we don't punish GDELT for having
    fewer mentions than Google Trends has search points — different sources
    measure different magnitudes.
    """
    grouped: dict[str, list[tuple[str, float]]] = {}
    for raw in raws:
        vol = (
            statistics.mean(v for _, v in raw.time_series)
            if raw.time_series else 0.0
        )
        grouped.setdefault(raw.source_name, []).append((raw.keyword, vol))

    out: dict[tuple[str, str], float] = {}
    for source_name, pairs in grouped.items():
        sorted_pairs = sorted(pairs, key=lambda p: p[1])
        n = len(sorted_pairs)
        for rank, (kw, _) in enumerate(sorted_pairs):
            pct = (rank + 1) / n if n else 0.0
            out[(kw, source_name)] = pct
    return out
