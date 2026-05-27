"""Unit tests for :class:`adapters.trends.pinterest_adapter.PinterestAdapter`.

No network. We exercise three properties:

1. No token → ``fetch`` returns None (pipeline tolerant).
2. With token → adapter fans out across trend_type slices and merges into
   a region-keyword index that ``fetch`` indexes into.
3. Time-series parsing handles both list- and parallel-array shapes the
   Pinterest API has used across versions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from adapters.trends.pinterest_adapter import (
    PinterestAdapter,
    _parse_weekly_series,
)


# --------------------------------------------------------------------------- #
# Pure helpers                                                                #
# --------------------------------------------------------------------------- #


def test_parse_weekly_series_handles_list_shape():
    payload = [
        {"date": "2024-05-13", "value": 87.4},
        {"date": "2024-05-20", "value": 92.1},
    ]
    assert _parse_weekly_series(payload) == [
        ("2024-05-13", 87.4),
        ("2024-05-20", 92.1),
    ]


def test_parse_weekly_series_handles_parallel_array_shape():
    payload = {"dates": ["2024-05-13", "2024-05-20"], "values": [87.4, 92.1]}
    assert _parse_weekly_series(payload) == [
        ("2024-05-13", 87.4),
        ("2024-05-20", 92.1),
    ]


def test_parse_weekly_series_skips_malformed():
    payload = [{"date": "ok", "value": "nan_string"}, {"date": "ok2"}]
    assert _parse_weekly_series(payload) == []


# --------------------------------------------------------------------------- #
# Fake response                                                               #
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return self._body


def _trends_payload(*keywords_with_series: tuple[str, list[tuple[str, float]]]) -> dict:
    return {
        "trends": [
            {
                "keyword": kw,
                "time_series": [{"date": d, "value": v} for d, v in series],
                "pct_growth_wow": 0.1,
            }
            for kw, series in keywords_with_series
        ]
    }


# --------------------------------------------------------------------------- #
# Adapter behaviour                                                           #
# --------------------------------------------------------------------------- #


def test_no_token_returns_none(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("PINTEREST_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("PINTEREST_TOKEN_FILE", str(tmp_path / "missing.json"))
    adapter = PinterestAdapter()
    assert adapter.access_token is None
    assert adapter.fetch("y2k", ("2023-01-01", "2023-12-31")) is None


def test_fetch_indexes_into_merged_region_response():
    """Single region call fans out across trend_type slices then indexes by keyword."""
    series_a = [("2024-05-13", 10.0), ("2024-05-20", 20.0)]
    series_b = [("2024-05-13", 5.0)]
    session = MagicMock()
    # 4 default trend_types → 4 calls; we vary which keyword appears where.
    session.get.side_effect = [
        _FakeResponse(200, _trends_payload(("y2k", series_a))),
        _FakeResponse(200, _trends_payload(("minimalism", series_b))),
        _FakeResponse(200, {"trends": []}),
        _FakeResponse(200, {"trends": []}),
    ]
    adapter = PinterestAdapter(
        access_token="fake-token", session=session, sleep_between=0,
    )

    raw = adapter.fetch("Y2K", ("2024-05-01", "2024-05-31"))
    assert raw is not None
    assert raw.source_name == "pinterest"
    # Series clipped to window; case-insensitive lookup against the index.
    assert raw.time_series == series_a
    assert "top" in raw.raw_payload["matched_trend_types"]

    # Second keyword reuses cached region index — no additional HTTP calls.
    prev_call_count = session.get.call_count
    raw2 = adapter.fetch("minimalism", ("2024-05-01", "2024-05-31"))
    assert raw2 is not None
    assert raw2.time_series == series_b
    assert session.get.call_count == prev_call_count


def test_keyword_absent_from_response_returns_empty_series():
    """A MARGO keyword that Pinterest doesn't surface → empty series, not None."""
    session = MagicMock()
    session.get.side_effect = [
        _FakeResponse(200, _trends_payload(("something_else", [("2024-05-13", 1.0)]))),
        _FakeResponse(200, {"trends": []}),
        _FakeResponse(200, {"trends": []}),
        _FakeResponse(200, {"trends": []}),
    ]
    adapter = PinterestAdapter(
        access_token="fake-token", session=session, sleep_between=0,
    )
    raw = adapter.fetch("y2k", ("2024-05-01", "2024-05-31"))
    assert raw is not None
    assert raw.time_series == []
    assert raw.raw_payload["reason"] == "not in top trends slices"


def test_401_drops_all_slices_and_returns_none():
    session = MagicMock()
    session.get.side_effect = [
        _FakeResponse(401, "unauthorized"),
    ] * 4
    adapter = PinterestAdapter(
        access_token="bad-token", session=session, sleep_between=0,
    )
    assert adapter.fetch("y2k", ("2024-05-01", "2024-05-31")) is None


def test_window_clipping():
    """Series points outside the requested window must be filtered out."""
    full = [
        ("2024-01-01", 1.0),
        ("2024-05-13", 10.0),
        ("2024-05-20", 20.0),
        ("2024-12-30", 50.0),
    ]
    session = MagicMock()
    session.get.side_effect = [
        _FakeResponse(200, _trends_payload(("y2k", full))),
    ] + [_FakeResponse(200, {"trends": []})] * 3
    adapter = PinterestAdapter(
        access_token="fake-token", session=session, sleep_between=0,
    )
    raw = adapter.fetch("y2k", ("2024-05-01", "2024-05-31"))
    assert raw is not None
    assert raw.time_series == [("2024-05-13", 10.0), ("2024-05-20", 20.0)]


def test_token_file_loading(tmp_path: Path, monkeypatch):
    """Adapter reads token from JSON file the OAuth helper writes."""
    monkeypatch.delenv("PINTEREST_ACCESS_TOKEN", raising=False)
    token_file = tmp_path / "tok.json"
    token_file.write_text(json.dumps({
        "access_token": "from-file",
        "refresh_token": "r",
        "expires_at_iso": "2099-01-01T00:00:00",
    }))
    monkeypatch.setenv("PINTEREST_TOKEN_FILE", str(token_file))

    adapter = PinterestAdapter()
    assert adapter.access_token == "from-file"
