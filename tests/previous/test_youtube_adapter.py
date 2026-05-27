"""Unit tests for :class:`adapters.trends.youtube_adapter.YouTubeAdapter`.

No network. A fake ``requests.Session`` returns canned JSON so we can
assert on quota partitioning, caching, and graceful failure modes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from adapters.trends.youtube_adapter import YouTubeAdapter, _month_buckets


# --------------------------------------------------------------------------- #
# _month_buckets — pure helper                                                #
# --------------------------------------------------------------------------- #


def test_month_buckets_partitions_calendar_months():
    buckets = _month_buckets("2023-01-15", "2023-03-10")
    # Three buckets: Jan (clipped), Feb (full), Mar (clipped)
    assert [b[0] for b in buckets] == ["2023-01-15", "2023-02-01", "2023-03-01"]
    # Each tuple is (label, after, before) and after < before.
    for label, after, before in buckets:
        assert after < before


def test_month_buckets_empty_window():
    assert _month_buckets("2023-05-01", "2023-04-01") == []


# --------------------------------------------------------------------------- #
# Fake session                                                                #
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


def _ok(total_results: int) -> _FakeResponse:
    """Mimic the shape we care about from search.list."""
    return _FakeResponse(200, {
        "kind": "youtube#searchListResponse",
        "pageInfo": {"totalResults": total_results, "resultsPerPage": 50},
        "items": [],  # we only read pageInfo for the count
    })


# --------------------------------------------------------------------------- #
# Adapter behaviour                                                           #
# --------------------------------------------------------------------------- #


def test_returns_none_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    adapter = YouTubeAdapter(api_key=None)
    assert adapter.fetch("y2k", ("2023-01-01", "2023-03-01")) is None


def test_fetch_partitions_by_month_and_extracts_counts():
    session = MagicMock()
    session.get.side_effect = [_ok(120), _ok(30), _ok(0)]
    adapter = YouTubeAdapter(api_key="fake-key", session=session, sleep_between=0)

    raw = adapter.fetch("low-rise jeans", ("2023-01-01", "2023-03-01"))
    assert raw is not None
    # Three monthly buckets → three points; counts capped at 50.
    assert [v for _, v in raw.time_series] == [50.0, 30.0, 0.0]
    assert raw.source_name == "youtube"
    assert raw.source_type == "media"
    assert raw.raw_payload["buckets_fetched"] == 3


def test_fetch_handles_403_quota_exhaustion_gracefully():
    session = MagicMock()
    session.get.side_effect = [_ok(50), _FakeResponse(403, "quotaExceeded"), _ok(10)]
    adapter = YouTubeAdapter(api_key="fake-key", session=session, sleep_between=0)

    raw = adapter.fetch("kw", ("2023-01-01", "2023-03-01"))
    assert raw is not None
    # 403 bucket records a zero so the time-series stays aligned.
    counts = [v for _, v in raw.time_series]
    assert counts == [50.0, 0.0, 10.0]


def test_cache_hit_skips_network(tmp_path: Path):
    """Second call with same (keyword, bucket, region) must not call the API."""
    session = MagicMock()
    session.get.side_effect = [_ok(42)]
    adapter = YouTubeAdapter(
        api_key="fake-key",
        session=session,
        cache_dir=tmp_path,
        sleep_between=0,
    )
    # Single-month window so we make exactly one HTTP call on the first fetch.
    window = ("2023-01-01", "2023-01-20")

    first = adapter.fetch("kw", window)
    assert first.raw_payload["buckets_fetched"] == 1
    assert first.raw_payload["buckets_from_cache"] == 0

    # Second call: same args. The fake session would raise StopIteration if
    # called again because side_effect has been exhausted.
    second = adapter.fetch("kw", window)
    assert second.raw_payload["buckets_fetched"] == 0
    assert second.raw_payload["buckets_from_cache"] == 1
    # Session still called only once.
    assert session.get.call_count == 1


def test_query_suffix_anchors_to_fashion_vertical():
    """``q`` parameter must include the suffix to keep signal in-vertical."""
    session = MagicMock()
    session.get.side_effect = [_ok(0)]
    adapter = YouTubeAdapter(api_key="fake-key", session=session, sleep_between=0)
    adapter.fetch("y2k", ("2023-01-01", "2023-01-15"))

    call_kwargs = session.get.call_args.kwargs
    assert call_kwargs["params"]["q"] == "y2k fashion"
