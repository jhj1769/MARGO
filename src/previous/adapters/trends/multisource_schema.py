"""Schemas for the 5-layer multi-source trend pipeline (Enhancement 4).

Layer ownership:
    L1 Source          → RawSourceData
    L2 Signal          → SourceSignal
    L3 Consensus       → MultiSourceTrendSignal
    L4 Semantic        → TrendSemantic
    L5 Pool            → (see adapters/trends/keyword_pool.py — pool stays there)
    Persistent output  → MultiSourceTrendSnapshot (one JSON per time window)

These models live separately from the legacy ``snapshot_schema.py`` so the
old single-source Google Trends pipeline keeps working untouched. The Trend
Agent prefers the multi-source snapshot when present and falls back to the
legacy one otherwise.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# Source typology shared across the pipeline.
SourceType = Literal["editorial", "search", "media", "community"]
Lifecycle = Literal["rising", "stable", "declining", "niche"]


# --------------------------------------------------------------------------- #
# L1 — Raw source data                                                        #
# --------------------------------------------------------------------------- #


class RawSourceData(BaseModel):
    """Per-source raw time-series for one keyword.

    Adapters return this from ``fetch()``. ``time_series`` is a sorted list
    of ``(ISO-date, value)`` pairs at whatever granularity the source
    natively emits (weekly for GT, daily for Wiki, daily for GDELT). The
    L2 normaliser handles resampling.
    """

    keyword: str
    source_name: str
    source_type: SourceType
    time_window: tuple[str, str]                         # (start_iso, end_iso)
    time_series: list[tuple[str, float]] = Field(default_factory=list)
    raw_payload: dict = Field(default_factory=dict)      # adapter-specific, debug only


# --------------------------------------------------------------------------- #
# L2 — Normalised signal                                                      #
# --------------------------------------------------------------------------- #


class SourceSignal(BaseModel):
    """One source's verdict on one keyword for one window."""

    keyword: str
    source_name: str
    source_type: SourceType
    lifecycle: Lifecycle
    growth_ratio: float                # short_ma / long_ma
    short_ma: float
    long_ma: float
    volume_percentile: float = Field(ge=0.0, le=1.0)  # how loud overall
    confidence: float = Field(ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# L3 — Consensus across sources                                               #
# --------------------------------------------------------------------------- #


class MultiSourceTrendSignal(BaseModel):
    """Per-keyword aggregate. Multiple sources collapsed but disagreement preserved.

    ``aggregated_lifecycle`` is the reliability-weighted majority verdict.
    ``disagreement_flag`` is true when sources actively contradict each other
    on lifecycle direction (rising vs declining); ``disagreement_nl`` is a
    short human-readable explanation passed to the LLM so it can hedge.
    """

    keyword: str
    sources: dict[str, SourceSignal] = Field(default_factory=dict)
    consensus_score: float = Field(default=0.0, ge=0.0, le=1.0)
    aggregated_lifecycle: Lifecycle = "stable"
    disagreement_flag: bool = False
    disagreement_nl: Optional[str] = None


# --------------------------------------------------------------------------- #
# L4 — Semantic mapping                                                       #
# --------------------------------------------------------------------------- #


class TrendSemantic(BaseModel):
    """Keyword grounded in structured fashion attributes.

    LLM proposes the attributes; BGE-M3 (or any embedder) validates by
    finding the K nearest items in the catalogue for each attribute.
    Attributes with zero catalogue matches are dropped — that is how
    hallucinated terms get filtered out.
    """

    keyword: str
    fashion_attributes: dict[str, list[str]] = Field(default_factory=dict)
    # E.g. {"style": ["low-rise", "metallics"], "category": ["jeans"], "era": "2000s"}
    catalog_match_method: Literal["llm_proposed", "bge_validated", "hybrid", "fallback"] = "hybrid"
    matched_item_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# Full snapshot                                                               #
# --------------------------------------------------------------------------- #


class MultiSourceTrendSnapshot(BaseModel):
    """End-to-end persisted output. One JSON per (domain, time_window)."""

    snapshot_id: str
    domain: str
    time_window: str  # logical key (e.g. "2023-full" or "2023-Q3")
    window_start_iso: str
    window_end_iso: str
    region: str = "US"
    snapshot_date: str

    # L5 pool that produced the analysed keywords.
    keyword_pool: dict[str, list[str]] = Field(default_factory=dict)
    # ^ {"catalog_derived": [...], "brief_derived": [...], "merged": [...]}

    # L3 outputs — one per analysed keyword.
    signals: list[MultiSourceTrendSignal] = Field(default_factory=list)

    # L4 outputs — keyed by keyword for fast lookup.
    semantics: dict[str, TrendSemantic] = Field(default_factory=dict)

    # Per-source meta (reliability priors actually used, fetch errors, ...).
    source_metadata: dict[str, dict] = Field(default_factory=dict)
    notes: Optional[str] = None
