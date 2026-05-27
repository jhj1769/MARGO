"""Pinterest API v5 source adapter (L1, Enhancement 4).

Fashion-native discovery-search signal. Pinterest's audience uses the
platform with explicit visual-shopping intent, which makes it a stronger
fashion-aware signal than Google Trends without the editorial bias of
GDELT. Reliability prior sits between the two.

Endpoint
--------

``GET /v5/trends/keywords/{region}/top/{trend_type}`` returns the top
trending keywords for a region with a weekly 0-100 time-series for each
keyword (one year history). The endpoint requires only the
``user_accounts:read`` OAuth scope — i.e. it is reachable from any
standard developer app, not enterprise-only.

Strategy
--------

The endpoint returns *region's top trends as a list*, not per-keyword
queries. So per snapshot we make ONE call per region (cached for the
day) and look up each MARGO keyword in the response. Keywords missing
from the response are reported as ``time_series=[]`` (= zero presence
in Pinterest's top-trends list, which is itself signal).

Trend-type rotation
-------------------

We pull the **union of multiple trend_type slices** (``top``,
``top_monthly``, ``growing``, ``seasonal``) because no single slice
covers the lifecycle states MARGO cares about. Each slice costs one
request and the response is small.

Token sourcing
--------------

The adapter reads the access token from (in order):

    1. ``access_token`` constructor argument
    2. ``PINTEREST_ACCESS_TOKEN`` env var
    3. Token file path (``PINTEREST_TOKEN_FILE``, default
       ``~/.margo/pinterest_token.json``) written by
       ``scripts.pinterest_oauth.py``

If none is available the adapter logs once and returns ``None`` for every
fetch — the pipeline records this in ``source_metadata`` and continues
with the remaining sources.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import requests

from adapters.trends.base import TrendSourceAdapter
from adapters.trends.multisource_schema import RawSourceData

log = logging.getLogger(__name__)


_TRENDS_URL = "https://api.pinterest.com/v5/trends/keywords/{region}/top/{trend_type}"
_DEFAULT_TREND_TYPES: tuple[str, ...] = ("top", "top_monthly", "growing", "seasonal")
_DEFAULT_TOKEN_FILE = "~/.margo/pinterest_token.json"
_DEFAULT_INTERESTS = ("women's fashion", "men's fashion", "beauty")


def _resolve_token_file() -> Path:
    return Path(os.path.expanduser(
        os.getenv("PINTEREST_TOKEN_FILE", _DEFAULT_TOKEN_FILE)
    ))


def _load_token_from_file(path: Path) -> Optional[str]:
    """Return the access token from a JSON file written by the OAuth helper."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("PinterestAdapter: token file %s unreadable: %s", path, e)
        return None
    token = data.get("access_token")
    if not isinstance(token, str) or not token:
        return None
    # Best-effort expiry check. We trust the helper to refresh proactively.
    expires_at = data.get("expires_at_iso")
    if expires_at:
        try:
            if datetime.fromisoformat(expires_at) < datetime.utcnow():
                log.warning(
                    "PinterestAdapter: token in %s appears expired (expires_at=%s); "
                    "run scripts.pinterest_oauth.py refresh",
                    path, expires_at,
                )
        except ValueError:
            pass
    return token


def _cache_path(cache_dir: Path, region: str, trend_type: str, day: str) -> Path:
    key = f"{region}|{trend_type}|{day}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / "pinterest" / f"{digest}.json"


def _parse_weekly_series(time_series_payload) -> list[tuple[str, float]]:
    """Convert Pinterest's weekly time-series object into ``[(iso_date, val)]``.

    The v5 schema returns either ``[{"date": "2024-05-13", "value": 87.4}, ...]``
    or a parallel ``{"dates": [...], "values": [...]}`` shape depending on
    endpoint version. Handle both, gracefully ignore malformed entries.
    """
    out: list[tuple[str, float]] = []
    if time_series_payload is None:
        return out
    if isinstance(time_series_payload, dict):
        dates = time_series_payload.get("dates") or []
        values = time_series_payload.get("values") or []
        for d, v in zip(dates, values):
            try:
                out.append((str(d), float(v)))
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(time_series_payload, list):
        for entry in time_series_payload:
            if not isinstance(entry, dict):
                continue
            d = entry.get("date") or entry.get("week_starting") or entry.get("timestamp")
            v = entry.get("value")
            if v is None:
                v = entry.get("relative_search_volume")
            if d is None or v is None:
                continue
            try:
                out.append((str(d), float(v)))
            except (TypeError, ValueError):
                continue
    return out


class PinterestAdapter(TrendSourceAdapter):
    """Fashion-native discovery-search signal."""

    source_name = "pinterest"
    source_type = "search"
    reliability_prior = 0.75

    def __init__(
        self,
        *,
        access_token: Optional[str] = None,
        token_file: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl_days: int = 1,
        request_timeout: float = 30.0,
        sleep_between: float = 0.2,
        session: Optional[requests.Session] = None,
        trend_types: Iterable[str] = _DEFAULT_TREND_TYPES,
        interests: Iterable[str] = _DEFAULT_INTERESTS,
    ) -> None:
        if access_token is None:
            access_token = os.getenv("PINTEREST_ACCESS_TOKEN")
        if access_token is None:
            path = Path(token_file) if token_file else _resolve_token_file()
            access_token = _load_token_from_file(path)

        self.access_token = access_token
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl_days = cache_ttl_days
        self.timeout = request_timeout
        self.sleep_between = sleep_between
        self._session = session or requests.Session()
        self.trend_types = tuple(trend_types)
        self.interests = tuple(interests)
        # In-memory per-(region) merged keyword index, computed once per fetch
        # of a given region; the adapter is single-threaded per pipeline run.
        self._region_index: dict[str, dict[str, dict]] = {}
        self._warned_missing_token = False

    # ------------------------------------------------------------------ #
    # Public                                                              #
    # ------------------------------------------------------------------ #

    def fetch(
        self,
        keyword: str,
        time_window: tuple[str, str],
        *,
        region: str = "US",
    ) -> Optional[RawSourceData]:
        if not self.access_token:
            if not self._warned_missing_token:
                log.warning(
                    "PinterestAdapter: no access token (env PINTEREST_ACCESS_TOKEN or %s) — "
                    "returning None for all fetches. Run scripts.pinterest_oauth.py to "
                    "authorise.",
                    _resolve_token_file(),
                )
                self._warned_missing_token = True
            return None

        index = self._load_region_index(region)
        if index is None:
            return None

        entry = index.get(keyword.lower())
        if entry is None:
            # Pinterest didn't surface this keyword in any pulled slice —
            # report empty series. This is real signal: "not trending here".
            return RawSourceData(
                keyword=keyword,
                source_name=self.source_name,
                source_type=self.source_type,
                time_window=time_window,
                time_series=[],
                raw_payload={"reason": "not in top trends slices", "region": region},
            )

        full_series = _parse_weekly_series(entry.get("time_series"))
        # Clip to the requested window so the L2 MA windows compute against
        # the right reference period.
        clipped = [
            (d, v) for d, v in full_series
            if time_window[0] <= d <= time_window[1]
        ]
        return RawSourceData(
            keyword=keyword,
            source_name=self.source_name,
            source_type=self.source_type,
            time_window=time_window,
            time_series=clipped,
            raw_payload={
                "native_unit": "weekly 0-100 normalised relative search volume",
                "region": region,
                "matched_trend_types": entry.get("_matched_trend_types", []),
                "pct_growth_wow": entry.get("pct_growth_wow"),
                "pct_growth_mom": entry.get("pct_growth_mom"),
                "pct_growth_yoy": entry.get("pct_growth_yoy"),
            },
        )

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _load_region_index(self, region: str) -> Optional[dict[str, dict]]:
        """Pull every trend_type slice for ``region`` and merge into a keyword index."""
        if region in self._region_index:
            return self._region_index[region]

        merged: dict[str, dict] = {}
        any_success = False
        for trend_type in self.trend_types:
            payload = self._fetch_trends(region=region, trend_type=trend_type)
            if payload is None:
                continue
            any_success = True
            for tk in self._iter_trending_keywords(payload):
                kw = (tk.get("keyword") or "").lower().strip()
                if not kw:
                    continue
                existing = merged.get(kw)
                if existing is None:
                    tk["_matched_trend_types"] = [trend_type]
                    merged[kw] = tk
                else:
                    existing.setdefault("_matched_trend_types", []).append(trend_type)
                    # Prefer the entry with the longer time series — different
                    # slices may differ in history depth.
                    new_len = len(_parse_weekly_series(tk.get("time_series")))
                    old_len = len(_parse_weekly_series(existing.get("time_series")))
                    if new_len > old_len:
                        # Keep the merged matched-trend-types list intact.
                        types = existing["_matched_trend_types"]
                        merged[kw] = tk
                        merged[kw]["_matched_trend_types"] = types

        if not any_success:
            return None

        self._region_index[region] = merged
        log.info(
            "PinterestAdapter: region=%s merged %d keywords across %d trend slices",
            region, len(merged), len(self.trend_types),
        )
        return merged

    @staticmethod
    def _iter_trending_keywords(payload: dict) -> Iterable[dict]:
        """The endpoint nests results under ``trends`` (legacy: ``items``)."""
        for key in ("trends", "items", "data"):
            block = payload.get(key)
            if isinstance(block, list):
                for entry in block:
                    if isinstance(entry, dict):
                        yield entry
                return

    def _fetch_trends(self, *, region: str, trend_type: str) -> Optional[dict]:
        """One ``trends/keywords`` call; honours disk cache with TTL."""
        cache_file: Optional[Path] = None
        if self.cache_dir is not None:
            cache_file = _cache_path(
                self.cache_dir, region, trend_type, _today_isoutc()
            )
            if cache_file.exists() and self._cache_is_fresh(cache_file):
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "PinterestAdapter: cache read failed for %s: %s",
                        cache_file, e,
                    )

        url = _TRENDS_URL.format(region=region, trend_type=trend_type)
        params = {"limit": 50}
        # ``interests`` is a repeated query parameter on the v5 endpoint.
        if self.interests:
            params["interests"] = list(self.interests)
        headers = {"Authorization": f"Bearer {self.access_token}"}

        try:
            r = self._session.get(
                url, params=params, headers=headers, timeout=self.timeout,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "PinterestAdapter: HTTP failure for region=%s type=%s: %s",
                region, trend_type, e,
            )
            return None

        if r.status_code == 401:
            log.warning(
                "PinterestAdapter: 401 — token rejected. Refresh via scripts.pinterest_oauth.py."
            )
            return None
        if r.status_code == 403:
            log.warning(
                "PinterestAdapter: 403 on region=%s type=%s — likely needs Standard access tier",
                region, trend_type,
            )
            return None
        if r.status_code >= 400:
            log.warning(
                "PinterestAdapter: HTTP %d on region=%s type=%s — body=%s",
                r.status_code, region, trend_type, r.text[:200],
            )
            return None

        try:
            payload = r.json()
        except Exception as e:  # noqa: BLE001
            log.warning(
                "PinterestAdapter: JSON parse failed (region=%s type=%s): %s",
                region, trend_type, e,
            )
            return None

        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(payload), encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "PinterestAdapter: cache write failed for %s: %s", cache_file, e
                )

        if self.sleep_between > 0:
            time.sleep(self.sleep_between)
        return payload

    def _cache_is_fresh(self, cache_file: Path) -> bool:
        try:
            mtime = datetime.utcfromtimestamp(cache_file.stat().st_mtime)
        except OSError:
            return False
        return datetime.utcnow() - mtime < timedelta(days=self.cache_ttl_days)


def _today_isoutc() -> str:
    return date.today().isoformat()
