"""v5 season-trend snapshot schema — trade-press centric, minimal fields.

v5 design (post-iteration):

* Discover step uses 4 axes — RUNWAY / PRESS / STREET / POP — chosen because
  these correspond to *who the signal comes from* (designer / fashion press /
  ordinary consumer / pop culture & celebrity). The four are near-orthogonal
  and align with the natural lifecycle of a fashion trend.

* Synthesize judges ``trend_stage`` from the *language* trade-press uses
  about the term, not from quantitative pageview/mention curves. 3-tier
  trend stage (the natural growth arc — niche → rising → mainstream):
  ``niche``      (비주류) — only one axis or small coverage, future uncertain;
  ``rising``     (떠오르는) — momentum building, multiple axes start to pick it up;
  ``mainstream`` (주류) — picked up across multiple axes, season's hero.

* Schema is intentionally minimal — variance-prone v4 fields
  (SourceEvidence, attributes dict, matched_items ASIN list) are removed.
  The signal that survives:
    keyword + lifecycle + confidence + rationale + citations

  Catalog matching (matched_items) and attribute axis matching are handled
  at recommendation-time instead, where they can be computed deterministically
  from keyword + retriever/cohort filter.

* Asymmetric authority preserved: this object is *evidence* for the Expert,
  not a directive. Phase 2 prompts treat it as advisory; Phase 3 weights it
  below the directive.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


TrendStage = Literal["niche", "rising", "mainstream"]
Confidence = Literal["high", "medium", "low"]


# --------------------------------------------------------------------------- #
# Citation — one trade-press article cited in a trend's rationale             #
# --------------------------------------------------------------------------- #


class Citation(BaseModel):
    """A single article the Synthesize step cites as evidence for a trend."""

    title: str
    url: str
    domain: str                       # e.g. "vogue.com"
    publish_date: Optional[str] = None
    excerpt: str = ""                 # 1-2 sentences from the article body


# --------------------------------------------------------------------------- #
# Per-trend analysis                                                          #
# --------------------------------------------------------------------------- #


class TrendItem(BaseModel):
    """LLM's structured verdict on one trend term for one season.

    The output of Step 2 (Synthesize), one per trend the LLM surfaced from
    the collected trade-press articles. ``trend_stage`` reflects how the
    language across the 4 axes (RUNWAY / PRESS / STREET / POP) treats the
    term — niche (small coverage), rising (momentum building across axes),
    or mainstream (the season's hero).
    """

    keyword: str
    trend_stage: TrendStage
    confidence: Confidence
    rationale: str                    # natural-language reasoning, cites publications
    citations: list[Citation] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# The persisted snapshot                                                      #
# --------------------------------------------------------------------------- #


class SeasonTrendSnapshot(BaseModel):
    """One JSON per (domain, season). v5 output."""

    snapshot_id: str                  # "fashion_trend_2023_SS"
    domain: str                       # "fashion"
    season: str                       # "2023-SS" (internal label; uses AW form)
    window_start_iso: str
    window_end_iso: str
    region: str = "US"
    snapshot_date: str                # ISO date of build

    # Season-level narrative
    summary: str = ""
    headline_themes: list[str] = Field(default_factory=list)

    # Per-trend analysis (variable length)
    trends: list[TrendItem] = Field(default_factory=list)

    # Audit / provenance
    sources_used: list[str] = Field(default_factory=list)   # domains actually hit
    article_count: int = 0                                  # articles read by Synthesize
    notes: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Recommendation-pipeline helpers                                    #
    # ------------------------------------------------------------------ #

    def mainstream_trends(self) -> list[TrendItem]:
        return [t for t in self.trends if t.trend_stage == "mainstream"]

    def rising_trends(self) -> list[TrendItem]:
        return [t for t in self.trends if t.trend_stage == "rising"]

    def niche_trends(self) -> list[TrendItem]:
        return [t for t in self.trends if t.trend_stage == "niche"]

    def keyword_index(self) -> dict[str, TrendItem]:
        """Lookup by lowercase keyword. Used by item-self-describe to ask
        'is this candidate's keyword in any trend? what trend_stage?'.
        """
        return {t.keyword.lower(): t for t in self.trends}

    def to_interpretation(self):
        """Convert to the legacy ``TrendInterpretation`` shape so existing
        Phase 2/3 consumers keep working unchanged.

        Mapping decisions:
          * ``keywords``           → mainstream + rising trends (boost candidates).
                                     Niche stays out — too thin to drive retrieval.
          * ``rising_attributes``  → empty (no longer in v5 schema). Phase 3 cohort
                                     filter falls back to keyword-based heuristic.
          * ``summary``            → season-level summary verbatim.
          * ``raw_sources``        → snapshot id pointer for audit.
        """
        from core.protocol.messages import TrendInterpretation

        boost_trends = self.mainstream_trends() + self.rising_trends()
        return TrendInterpretation(
            domain=self.domain,
            time_window=self.season,
            summary=self.summary,
            keywords=[t.keyword for t in boost_trends],
            rising_attributes={},                             # v5: handled at runtime
            raw_sources=[f"season_snapshot://{self.snapshot_id}"],
        )
