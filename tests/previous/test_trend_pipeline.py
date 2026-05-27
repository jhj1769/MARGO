"""Enhancement 4 — 5-layer trend pipeline tests.

Adapters are mocked everywhere; we never hit a real GDELT / Pinterest /
Google / YouTube endpoint from the test suite. Each layer
(L2 / L3 / L4 / L5 / end-to-end) has its own focused test plus an
integration test through the orchestrator.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("MARGO_LLM_BACKEND", "dummy")

import pytest

from adapters.trends.base import TrendSourceAdapter
from adapters.trends.consensus import aggregate_keyword, compute_consensus
from adapters.trends.multisource_schema import (
    MultiSourceTrendSnapshot,
    RawSourceData,
    SourceSignal,
)
from adapters.trends.pipeline import (
    build_multisource_snapshot,
    load_snapshot,
    save_snapshot,
)
from adapters.trends.semantic_mapper import (
    _LLMSemanticOut,
    build_semantic_mapping,
    load_semantic_from_cache,
)
from adapters.trends.signal_normalizer import (
    normalize_to_signal,
    rank_volume_percentiles,
)


# --------------------------------------------------------------------------- #
# Mock adapters                                                               #
# --------------------------------------------------------------------------- #


def _daily_series(start_iso: str, end_iso: str, values: list[float]) -> list[tuple[str, float]]:
    """Build [(date, value), ...] from a list of consecutive daily values."""
    start = datetime.fromisoformat(start_iso)
    out: list[tuple[str, float]] = []
    for i, v in enumerate(values):
        out.append(((start + timedelta(days=i)).strftime("%Y-%m-%d"), float(v)))
    return out


class _MockAdapter(TrendSourceAdapter):
    """Returns canned time-series per (keyword, this adapter)."""

    def __init__(self, name: str, source_type: str, prior: float,
                 series_by_keyword: dict):
        self.source_name = name
        self.source_type = source_type  # type: ignore[assignment]
        self.reliability_prior = prior
        self._series = series_by_keyword

    def fetch(self, keyword, time_window, *, region="US"):
        series = self._series.get(keyword)
        if series is None:
            return None
        return RawSourceData(
            keyword=keyword,
            source_name=self.source_name,
            source_type=self.source_type,  # type: ignore[arg-type]
            time_window=time_window,
            time_series=series,
        )


# --------------------------------------------------------------------------- #
# L2 signal normalisation                                                     #
# --------------------------------------------------------------------------- #


def test_l2_rising_when_short_ma_exceeds_long_ma():
    """A clearly accelerating series should classify as rising."""
    # 84 days: first 56 days steady at 5, last 28 days at 25 → short ≈ 25, long ≈ ~11.7
    series = _daily_series("2023-06-09", "2023-08-31", [5.0] * 56 + [25.0] * 28)
    raw = RawSourceData(
        keyword="y2k", source_name="m1", source_type="search",
        time_window=("2023-06-09", "2023-08-31"), time_series=series,
    )
    sig = normalize_to_signal(raw, volume_percentile=0.8)
    assert sig.lifecycle == "rising"
    assert sig.growth_ratio > 1.20


def test_l2_declining_when_short_ma_below_long():
    series = _daily_series("2023-06-09", "2023-08-31", [25.0] * 56 + [5.0] * 28)
    raw = RawSourceData(
        keyword="bootcut", source_name="m1", source_type="search",
        time_window=("2023-06-09", "2023-08-31"), time_series=series,
    )
    sig = normalize_to_signal(raw, volume_percentile=0.8)
    assert sig.lifecycle == "declining"
    assert sig.growth_ratio < 0.85


def test_l2_niche_when_volume_low():
    """Flat low-volume series at the bottom percentile → niche."""
    series = _daily_series("2023-06-09", "2023-08-31", [1.0] * 84)
    raw = RawSourceData(
        keyword="archive-fashion", source_name="m1", source_type="media",
        time_window=("2023-06-09", "2023-08-31"), time_series=series,
    )
    sig = normalize_to_signal(raw, volume_percentile=0.1)
    assert sig.lifecycle == "niche"


def test_l2_volume_percentile_ranks_per_source():
    """rank_volume_percentiles returns a per-source ranking, not cross-source."""
    raws = [
        RawSourceData(keyword="a", source_name="src1", source_type="search",
                      time_window=("2023-01-01", "2023-03-01"),
                      time_series=_daily_series("2023-01-01", "2023-01-03", [1.0, 1.0, 1.0])),
        RawSourceData(keyword="b", source_name="src1", source_type="search",
                      time_window=("2023-01-01", "2023-03-01"),
                      time_series=_daily_series("2023-01-01", "2023-01-03", [10.0, 10.0, 10.0])),
        RawSourceData(keyword="a", source_name="src2", source_type="media",
                      time_window=("2023-01-01", "2023-03-01"),
                      time_series=_daily_series("2023-01-01", "2023-01-03", [100.0, 100.0, 100.0])),
    ]
    pcts = rank_volume_percentiles(raws)
    # Within src1: a (vol 1) is lower percentile than b (vol 10).
    assert pcts[("a", "src1")] < pcts[("b", "src1")]
    # Cross-source: a in src2 (vol 100) shouldn't be punished for src1's existence.
    assert pcts[("a", "src2")] == 1.0


# --------------------------------------------------------------------------- #
# L3 consensus                                                                #
# --------------------------------------------------------------------------- #


def _signal(kw: str, source: str, lifecycle: str, conf: float = 0.9) -> SourceSignal:
    return SourceSignal(
        keyword=kw, source_name=source,
        source_type={
            "gdelt": "editorial",
            "pinterest": "search",
            "google": "search",
            "youtube": "media",
        }.get(source, "media"),  # type: ignore[arg-type]
        lifecycle=lifecycle,  # type: ignore[arg-type]
        growth_ratio=1.5 if lifecycle == "rising" else 0.7 if lifecycle == "declining" else 1.0,
        short_ma=10.0, long_ma=10.0,
        volume_percentile=0.5, confidence=conf,
    )


def test_l3_consensus_unanimous():
    priors = {"gdelt": 0.85, "pinterest": 0.75, "google": 0.70, "youtube": 0.65}
    sources = {
        "gdelt": _signal("y2k", "gdelt", "rising"),
        "pinterest": _signal("y2k", "pinterest", "rising"),
        "google": _signal("y2k", "google", "rising"),
        "youtube": _signal("y2k", "youtube", "rising"),
    }
    score, lifecycle, dis, _ = compute_consensus(sources, reliability_priors=priors)
    assert lifecycle == "rising"
    assert score == 1.0
    assert dis is False


def test_l3_consensus_disagreement_flagged_when_opposite():
    """GDELT/Pinterest/Google rising, YouTube declining → flag the dissent."""
    priors = {"gdelt": 0.85, "pinterest": 0.75, "google": 0.70, "youtube": 0.65}
    sources = {
        "gdelt": _signal("y2k", "gdelt", "rising"),
        "pinterest": _signal("y2k", "pinterest", "rising"),
        "google": _signal("y2k", "google", "rising"),
        "youtube": _signal("y2k", "youtube", "declining"),
    }
    score, lifecycle, dis, nl = compute_consensus(sources, reliability_priors=priors)
    # Three higher-priors vote rising → still rising.
    assert lifecycle == "rising"
    assert dis is True
    assert nl is not None and "youtube" in nl.lower()
    assert "declining" in nl.lower()


def test_l3_gdelt_outweighs_two_weaker_stables():
    """A primary "rising" vote beats two weaker "stable" votes."""
    priors = {"gdelt": 0.85, "google": 0.3, "youtube": 0.3}  # secondary/tertiary weakened
    sources = {
        "gdelt": _signal("y2k", "gdelt", "rising", conf=0.95),
        "google": _signal("y2k", "google", "stable", conf=0.4),
        "youtube": _signal("y2k", "youtube", "stable", conf=0.4),
    }
    _, lifecycle, _, _ = compute_consensus(sources, reliability_priors=priors)
    assert lifecycle == "rising"


def test_l3_aggregate_keyword_returns_multi_source_signal():
    priors = {"gdelt": 0.85, "google": 0.7}
    sources = {
        "gdelt": _signal("y2k", "gdelt", "rising"),
        "google": _signal("y2k", "google", "rising"),
    }
    mst = aggregate_keyword("y2k", sources, reliability_priors=priors)
    assert mst.keyword == "y2k"
    assert mst.aggregated_lifecycle == "rising"
    assert mst.disagreement_flag is False
    assert set(mst.sources) == {"gdelt", "google"}


# --------------------------------------------------------------------------- #
# L4 semantic mapper                                                          #
# --------------------------------------------------------------------------- #


def test_l4_drops_ungrounded_terms():
    """A proposed term with zero catalog matches must be filtered out."""
    def _matcher(query: str, k: int) -> list[tuple[str, float]]:
        if query == "low-rise":
            return [("J1", 0.8), ("J2", 0.7)]
        if query == "metallics":
            return [("M1", 0.6), ("M2", 0.55)]
        # "hallucinated-term" gets nothing.
        return []

    def _propose(prompt: str) -> _LLMSemanticOut:
        return _LLMSemanticOut(
            style=["low-rise", "hallucinated-term"],
            category=["jeans"],
            material_or_color=["metallics"],
            era_reference="2000s",
        )

    # Provide jeans matches.
    def _matcher_jeans_too(query, k):
        if query == "jeans":
            return [("J3", 0.85), ("J4", 0.82)]
        return _matcher(query, k)

    sem = build_semantic_mapping(
        "y2k revival",
        matcher=_matcher_jeans_too,
        llm_propose=_propose,
    )
    assert "low-rise" in sem.fashion_attributes["style"]
    assert "hallucinated-term" not in sem.fashion_attributes.get("style", [])
    assert sem.fashion_attributes["category"] == ["jeans"]
    assert sem.fashion_attributes["material_or_color"] == ["metallics"]
    assert sem.fashion_attributes["era_reference"] == ["2000s"]
    assert sem.confidence < 1.0  # one term dropped


def test_l4_cache_roundtrip(tmp_path: Path):
    """First call builds + caches; second call returns cached object without LLM."""
    call_count = {"n": 0}

    def _matcher(query, k):
        return [("X", 0.9)] * 5

    def _propose(prompt):
        call_count["n"] += 1
        return _LLMSemanticOut(style=["foo"], category=["bar"])

    s1 = build_semantic_mapping("kw", matcher=_matcher, llm_propose=_propose, processed_dir=tmp_path)
    s2 = build_semantic_mapping("kw", matcher=_matcher, llm_propose=_propose, processed_dir=tmp_path)
    assert call_count["n"] == 1  # second call hit cache
    assert s1.keyword == s2.keyword == "kw"
    cached = load_semantic_from_cache("kw", tmp_path)
    assert cached is not None
    assert cached.fashion_attributes == s1.fashion_attributes


def test_l4_fallback_when_llm_raises():
    def _matcher(query, k):
        return []

    def _broken_propose(prompt):
        raise RuntimeError("LLM offline")

    sem = build_semantic_mapping("kw", matcher=_matcher, llm_propose=_broken_propose)
    assert sem.catalog_match_method == "fallback"
    assert sem.confidence == 0.0


# --------------------------------------------------------------------------- #
# End-to-end pipeline                                                         #
# --------------------------------------------------------------------------- #


def test_pipeline_end_to_end_with_mock_adapters(tmp_path: Path):
    """Three mock adapters → keywords get fetched, normalised, aggregated."""
    keywords = ["rising_kw", "declining_kw", "agree_kw"]

    # rising_kw: GDELT + Google rising, Wikipedia stable
    rising_series = _daily_series("2023-06-09", "2023-08-31", [5.0] * 56 + [25.0] * 28)
    declining_series = _daily_series("2023-06-09", "2023-08-31", [25.0] * 56 + [5.0] * 28)
    stable_series = _daily_series("2023-06-09", "2023-08-31", [10.0] * 84)

    gdelt = _MockAdapter("gdelt", "editorial", 0.85, {
        "rising_kw": rising_series,
        "declining_kw": declining_series,
        "agree_kw": rising_series,
    })
    pinterest = _MockAdapter("pinterest", "search", 0.75, {
        "rising_kw": rising_series,
        "declining_kw": declining_series,
        "agree_kw": rising_series,
    })
    google = _MockAdapter("google_trends", "search", 0.70, {
        "rising_kw": rising_series,
        "declining_kw": declining_series,
        "agree_kw": rising_series,
    })
    youtube = _MockAdapter("youtube", "media", 0.65, {
        "rising_kw": stable_series,  # YouTube disagrees (stable, not rising)
        "declining_kw": declining_series,
        "agree_kw": rising_series,
    })

    snapshot = build_multisource_snapshot(
        keyword_pool={"catalog_derived": keywords, "brief_derived": [], "merged": keywords},
        adapters=[gdelt, pinterest, google, youtube],
        time_window=("2023-06-09", "2023-08-31"),
        time_window_label="2023-Q3",
        domain="fashion",
        skip_semantic=True,  # L4 tested separately
    )
    assert isinstance(snapshot, MultiSourceTrendSnapshot)
    assert len(snapshot.signals) == 3
    by_kw = {s.keyword: s for s in snapshot.signals}

    assert by_kw["rising_kw"].aggregated_lifecycle == "rising"
    # rising_kw should NOT flag disagreement: YouTube voted "stable", not opposite.
    assert by_kw["rising_kw"].disagreement_flag is False

    assert by_kw["declining_kw"].aggregated_lifecycle == "declining"
    assert by_kw["agree_kw"].aggregated_lifecycle == "rising"
    assert by_kw["agree_kw"].consensus_score == 1.0  # unanimous

    # Snapshot persistence roundtrip.
    save_snapshot(snapshot, tmp_path)
    from adapters.trends.pipeline import snapshot_path
    loaded = load_snapshot(snapshot_path(tmp_path, snapshot.snapshot_id))
    assert loaded is not None
    assert len(loaded.signals) == 3


def test_pipeline_records_fetch_errors_per_source():
    """When an adapter returns None for a keyword it should appear in source_metadata.fetch_errors."""
    gdelt = _MockAdapter("gdelt", "editorial", 0.85, {})  # returns None for everything
    google = _MockAdapter("google_trends", "search", 0.70, {
        "kw1": _daily_series("2023-06-09", "2023-08-31", [10.0] * 84),
    })
    snapshot = build_multisource_snapshot(
        keyword_pool={"merged": ["kw1"]},
        adapters=[gdelt, google],
        time_window=("2023-06-09", "2023-08-31"),
        time_window_label="t",
        skip_semantic=True,
    )
    assert "kw1" in snapshot.source_metadata["gdelt"]["fetch_errors"]
    assert "kw1" not in snapshot.source_metadata["google_trends"]["fetch_errors"]
    # kw1 still produces a signal (from google only)
    assert len(snapshot.signals) == 1
    assert "google_trends" in snapshot.signals[0].sources
    assert "gdelt" not in snapshot.signals[0].sources


# --------------------------------------------------------------------------- #
# L5 brief extraction                                                          #
# --------------------------------------------------------------------------- #


def test_l5_brief_extraction_empty_returns_empty():
    """No brief text → empty list, no LLM call needed."""
    from adapters.trends.keyword_pool import extract_brief_keywords

    class _NeverCalled:
        def complete_structured(self, *a, **kw):
            raise AssertionError("should not be called")

    assert extract_brief_keywords(_NeverCalled(), "") == []
    assert extract_brief_keywords(_NeverCalled(), "   ") == []


def test_l5_brief_extraction_cleans_output():
    from adapters.trends.keyword_pool import extract_brief_keywords, _BriefKeywords

    class _StubLLM:
        def complete_structured(self, prompt, schema, **kw):
            return _BriefKeywords(keywords=[
                "Y2K revival", "y2k revival",      # dup after lowercase
                "  metallics  ",                    # whitespace
                "a" * 50,                           # too long
                "minimal-casual",
            ])

    out = extract_brief_keywords(_StubLLM(), "any brief")
    assert "y2k revival" in out
    # dedup
    assert out.count("y2k revival") == 1
    assert "metallics" in out
    # length filter dropped the 50-char one
    assert "a" * 50 not in out
    assert "minimal-casual" in out
