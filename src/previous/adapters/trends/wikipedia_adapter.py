"""Wikipedia pageview source adapter (L1, Enhancement 4).

Wikipedia is the encyclopedic-lag signal in our 3-source consensus —
trends usually show up here AFTER the press has reported them, so
agreement between GDELT (editorial) and Wikipedia confirms a trend is
truly mainstream while disagreement (e.g. press is loud, Wikipedia is
quiet) suggests a hyped-but-shallow press cycle.

Implementation:

1. ``opensearch`` resolves a keyword to a canonical article title.
2. ``per-article`` pageview endpoint returns daily views inside the window.

Both endpoints are public, key-less, and free — Wikimedia only asks that
you set a descriptive User-Agent.

Cache is keyed on ``(article_title, time_window)`` after resolution so
keyword aliases that map to the same article share the same payload.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from adapters.trends.base import TrendSourceAdapter
from adapters.trends.multisource_schema import RawSourceData

log = logging.getLogger(__name__)


_OPENSEARCH_URL = "https://en.wikipedia.org/w/api.php"
_PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}"
)
_DEFAULT_UA = (
    "MARGO-research-trend-builder/0.1 (academic research; "
    "https://github.com/anthropics; contact: jung@postech.ac.kr)"
)


def _yyyymmdd(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%Y%m%d")


def _cache_path(cache_dir: Path, article: str, start: str, end: str) -> Path:
    key = f"{article}|{start}|{end}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / "wikipedia" / f"{digest}.json"


class WikipediaAdapter(TrendSourceAdapter):
    """Encyclopedic lag-side trend signal sourced from Wikipedia pageviews."""

    source_name = "wikipedia_pageview"
    source_type = "media"
    reliability_prior = 0.6

    def __init__(
        self,
        *,
        cache_dir: Optional[Path] = None,
        user_agent: str = _DEFAULT_UA,
        request_timeout: float = 30.0,
        sleep_between: float = 0.2,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = request_timeout
        self.sleep_between = sleep_between
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._resolved_cache: dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------ #
    # Public                                                              #
    # ------------------------------------------------------------------ #

    def fetch(
        self,
        keyword: str,
        time_window: tuple[str, str],
        *,
        region: str = "US",  # ignored; Wikipedia API is per-language not per-region
    ) -> Optional[RawSourceData]:
        article = self._resolve_article(keyword)
        if not article:
            return RawSourceData(
                keyword=keyword,
                source_name=self.source_name,
                source_type=self.source_type,
                time_window=time_window,
                time_series=[],
                raw_payload={"reason": "no article match"},
            )

        start, end = time_window
        try:
            yyyymmdd_start = _yyyymmdd(start)
            yyyymmdd_end = _yyyymmdd(end)
        except ValueError:
            log.warning("WikipediaAdapter: bad time window %s..%s", start, end)
            return None

        payload = self._fetch_pageviews(article, yyyymmdd_start, yyyymmdd_end)
        if payload is None:
            return None

        items = payload.get("items") or []
        # items[i] = {"timestamp": "20230301", "views": N, ...}
        series: list[tuple[str, float]] = []
        for it in items:
            ts = it.get("timestamp", "")
            if len(ts) < 8:
                continue
            iso = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"
            series.append((iso, float(it.get("views", 0))))

        return RawSourceData(
            keyword=keyword,
            source_name=self.source_name,
            source_type=self.source_type,
            time_window=time_window,
            time_series=series,
            raw_payload={
                "native_unit": "daily pageviews (en.wikipedia, all-access)",
                "article": article,
                "from_cache": payload.get("_cache_hit", False),
            },
        )

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _resolve_article(self, keyword: str) -> Optional[str]:
        kw = keyword.strip().lower()
        if kw in self._resolved_cache:
            return self._resolved_cache[kw]
        try:
            r = self._session.get(
                _OPENSEARCH_URL,
                params={
                    "action": "opensearch",
                    "search": keyword,
                    "limit": 1,
                    "namespace": 0,
                    "format": "json",
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            # opensearch returns [search, [titles], [descriptions], [urls]]
            titles = data[1] if len(data) > 1 else []
            article = titles[0].replace(" ", "_") if titles else None
        except Exception as e:  # noqa: BLE001
            log.warning("WikipediaAdapter: opensearch failed for %r: %s", keyword, e)
            article = None
        self._resolved_cache[kw] = article
        if self.sleep_between:
            time.sleep(self.sleep_between)
        return article

    def _fetch_pageviews(
        self, article: str, start_yyyymmdd: str, end_yyyymmdd: str
    ) -> Optional[dict]:
        cache_file = (
            _cache_path(self.cache_dir, article, start_yyyymmdd, end_yyyymmdd)
            if self.cache_dir
            else None
        )
        if cache_file and cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                payload["_cache_hit"] = True
                return payload
            except Exception:  # noqa: BLE001
                pass  # fall through and refetch

        url = _PAGEVIEWS_URL.format(
            article=article, start=start_yyyymmdd, end=end_yyyymmdd
        )
        try:
            r = self._session.get(url, timeout=self.timeout)
            if r.status_code == 404:
                # No pageviews for this article in the window — record empty.
                payload = {"items": [], "_not_found": True}
            else:
                r.raise_for_status()
                payload = r.json()
        except Exception as e:  # noqa: BLE001
            log.warning(
                "WikipediaAdapter: pageviews failed for %s %s..%s: %s",
                article, start_yyyymmdd, end_yyyymmdd, e,
            )
            return None

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(payload), encoding="utf-8")

        if self.sleep_between:
            time.sleep(self.sleep_between)
        payload["_cache_hit"] = False
        return payload
