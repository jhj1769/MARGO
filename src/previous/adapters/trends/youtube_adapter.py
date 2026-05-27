"""YouTube Data API v3 source adapter (L1, Enhancement 4).

Creator-side media signal. YouTube haul / lookbook / "GRWM" videos are the
clearest indicator that a trend has crossed from editorial coverage
(GDELT) into community adoption — a different beat from Pinterest's
discovery-search signal and from Google's mass-search noise.

Strategy
--------

The ``search.list`` endpoint costs **100 quota units per call**, with
10,000 units/day on the default free tier (≈100 keyword-windows/day).
We trade some recall for time-resolved signal by partitioning each
keyword's ``time_window`` into **monthly buckets** and calling
``search.list`` once per (keyword, month) with ``publishedAfter`` /
``publishedBefore``. The resulting time-series is one point per month:
``video_count_capped_at_50`` (the API maxes ``maxResults`` at 50 and
deeper pagination doubles the quota cost for diminishing returns).

A monthly cadence is the right granularity for fashion trend lifecycle
anyway — a "rising" creator wave doesn't flip in a week.

Caching
-------

Per-call JSON responses are persisted under
``<cache_dir>/youtube/<sha1(keyword|YYYY-MM)>.json`` so re-runs of the
same snapshot are free. Set ``cache_dir=None`` to disable.

Failure modes
-------------

* Missing/invalid ``YOUTUBE_API_KEY`` → adapter returns ``None`` for every
  fetch and logs once. The pipeline records this in ``source_metadata``
  and just runs without YouTube; it does NOT crash.
* HTTP 403 quotaExceeded → propagated as ``None`` for the affected
  keyword; subsequent fetches in the same run keep trying (the operator
  decides whether to abort).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

from adapters.trends.base import TrendSourceAdapter
from adapters.trends.multisource_schema import RawSourceData

log = logging.getLogger(__name__)


_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
_DEFAULT_REGION = "US"
# Per-call ``maxResults`` is capped at 50 by the API. We never paginate —
# the quota cost doubles per page and 50 hits/month is plenty of signal.
_MAX_RESULTS = 50


def _month_buckets(start_iso: str, end_iso: str) -> list[tuple[str, str, str]]:
    """Yield ``(label, after_rfc3339, before_rfc3339)`` per calendar month.

    Each bucket spans ``[YYYY-MM-01T00:00:00Z, YYYY-MM+1-01T00:00:00Z)``
    clipped to ``[start_iso, end_iso]``. The label is the bucket's first
    day, which becomes the time-series timestamp downstream.
    """
    start = datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(end_iso).replace(tzinfo=timezone.utc)
    if end <= start:
        return []

    out: list[tuple[str, str, str]] = []
    cur = start.replace(day=1)
    while cur <= end:
        nxt_year = cur.year + (cur.month // 12)
        nxt_month = (cur.month % 12) + 1
        nxt = cur.replace(year=nxt_year, month=nxt_month, day=1)

        bucket_start = max(cur, start)
        bucket_end = min(nxt, end + timedelta(seconds=1))
        # Skip degenerate buckets (clip at the tail of the window).
        if bucket_end <= bucket_start:
            cur = nxt
            continue

        label = bucket_start.strftime("%Y-%m-%d")
        out.append((
            label,
            bucket_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            bucket_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ))
        cur = nxt
    return out


def _cache_path(cache_dir: Path, keyword: str, bucket_label: str, region: str) -> Path:
    key = f"{keyword.lower()}|{bucket_label}|{region}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / "youtube" / f"{digest}.json"


class YouTubeAdapter(TrendSourceAdapter):
    """Creator-activity signal sourced from YouTube Data API v3."""

    source_name = "youtube"
    source_type = "media"
    reliability_prior = 0.65

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        request_timeout: float = 30.0,
        sleep_between: float = 1.0,
        session: Optional[requests.Session] = None,
        query_suffix: str = "fashion",
    ) -> None:
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = request_timeout
        self.sleep_between = sleep_between
        self._session = session or requests.Session()
        # YouTube's relevance ranking on the raw keyword pulls in too much
        # cross-domain noise (e.g. "y2k" matches gaming/music). Appending
        # the domain anchor keeps the signal in-vertical without spending
        # extra quota on a category filter call.
        self.query_suffix = query_suffix
        self._warned_missing_key = False

    # ------------------------------------------------------------------ #
    # Public                                                              #
    # ------------------------------------------------------------------ #

    def fetch(
        self,
        keyword: str,
        time_window: tuple[str, str],
        *,
        region: str = _DEFAULT_REGION,
    ) -> Optional[RawSourceData]:
        if not self.api_key:
            if not self._warned_missing_key:
                log.warning(
                    "YouTubeAdapter: no YOUTUBE_API_KEY set — returning None for all fetches"
                )
                self._warned_missing_key = True
            return None

        buckets = _month_buckets(time_window[0], time_window[1])
        if not buckets:
            return RawSourceData(
                keyword=keyword,
                source_name=self.source_name,
                source_type=self.source_type,
                time_window=time_window,
                time_series=[],
                raw_payload={"reason": "empty window"},
            )

        series: list[tuple[str, float]] = []
        fetched_buckets = 0
        cached_buckets = 0
        for label, after, before in buckets:
            payload = self._fetch_bucket(keyword, label, after, before, region)
            if payload is None:
                # Hard failure on this bucket — record zero and keep going.
                series.append((label, 0.0))
                continue
            if payload.get("_cache_hit"):
                cached_buckets += 1
            else:
                fetched_buckets += 1
            # ``pageInfo.totalResults`` is YouTube's own estimate of the
            # match count and is more informative than counting the page
            # we received — but the API is notorious for inflating it on
            # broad queries, so we cap at ``maxResults`` to stay honest.
            page_info = payload.get("pageInfo", {}) or {}
            total = int(page_info.get("totalResults", 0))
            count = float(min(total, _MAX_RESULTS))
            series.append((label, count))

        return RawSourceData(
            keyword=keyword,
            source_name=self.source_name,
            source_type=self.source_type,
            time_window=time_window,
            time_series=series,
            raw_payload={
                "native_unit": f"monthly video count capped at {_MAX_RESULTS}",
                "query_suffix": self.query_suffix,
                "buckets_fetched": fetched_buckets,
                "buckets_from_cache": cached_buckets,
            },
        )

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _fetch_bucket(
        self,
        keyword: str,
        bucket_label: str,
        published_after: str,
        published_before: str,
        region: str,
    ) -> Optional[dict]:
        """Single ``search.list`` call for one month bucket, with disk cache."""
        cache_file: Optional[Path] = None
        if self.cache_dir is not None:
            cache_file = _cache_path(self.cache_dir, keyword, bucket_label, region)
            if cache_file.exists():
                try:
                    cached = json.loads(cache_file.read_text(encoding="utf-8"))
                    cached["_cache_hit"] = True
                    return cached
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "YouTubeAdapter: cache read failed for %s: %s — refetching",
                        cache_file, e,
                    )

        query = f"{keyword} {self.query_suffix}".strip() if self.query_suffix else keyword
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": _MAX_RESULTS,
            "publishedAfter": published_after,
            "publishedBefore": published_before,
            "regionCode": region,
            "relevanceLanguage": "en",
            "key": self.api_key,
        }
        try:
            r = self._session.get(_SEARCH_URL, params=params, timeout=self.timeout)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "YouTubeAdapter: HTTP failure for %r bucket %s: %s",
                keyword, bucket_label, e,
            )
            return None

        if r.status_code == 403:
            # Most commonly: quota exhausted, or referer/IP restriction on the key.
            log.warning(
                "YouTubeAdapter: 403 on %r bucket %s — likely quota or key restriction",
                keyword, bucket_label,
            )
            return None
        if r.status_code >= 400:
            log.warning(
                "YouTubeAdapter: HTTP %d on %r bucket %s — body=%s",
                r.status_code, keyword, bucket_label, r.text[:200],
            )
            return None

        try:
            payload = r.json()
        except Exception as e:  # noqa: BLE001
            log.warning(
                "YouTubeAdapter: JSON parse failed for %r bucket %s: %s",
                keyword, bucket_label, e,
            )
            return None

        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(payload), encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                log.warning("YouTubeAdapter: cache write failed for %s: %s", cache_file, e)

        if self.sleep_between > 0:
            time.sleep(self.sleep_between)
        return payload
