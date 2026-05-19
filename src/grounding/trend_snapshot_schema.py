"""Typed snapshot for the Google-Trends-grounded MARGO Trend Agent.

The schema follows the design in the MARGO notes — see ``MARGO/docs`` — and
mirrors the JSON layout one-to-one. We keep the snapshot *plain JSON*
(serialisable via ``model_dump``) so it can be checked into the cache
directory and replayed offline.

Fields are deliberately *flat* to make LLM prompt rendering trivial.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class KeywordSignal(BaseModel):
    """One keyword and its trend signal in a fixed window."""

    keyword: str
    absolute_score: float = Field(
        default=0.0,
        description="Mean Google-Trends interest (0–100) over the window.",
    )
    growth_pct: Optional[float] = Field(
        default=None,
        description="Relative growth between the last and first half of the window.",
    )
    peak_score: Optional[float] = None
    source: str = Field(default="catalog", description="catalog | llm_seed | rising_feedback")


class RisingQuery(BaseModel):
    """A 'breakout' query Google reported as rising for a seed."""

    seed: str
    query: str
    growth_pct: float


class CategoryTrend(BaseModel):
    """Aggregated direction for a coarse catalog category."""

    category: str
    direction: str = Field(description="rising | stable | declining | mixed")
    mean_score: float = 0.0
    mean_growth_pct: float = 0.0
    key_terms: list[str] = Field(default_factory=list)


class TrendSnapshot(BaseModel):
    """Top-level Google-Trends-grounded snapshot for one (domain, window)."""

    domain: str
    time_window: str
    region: str = "US"
    snapshot_date: str  # ISO date when the snapshot was built
    source: str = "Google Trends via pytrends"

    rising_keywords: list[KeywordSignal] = Field(default_factory=list)
    stable_top_keywords: list[KeywordSignal] = Field(default_factory=list)
    declining_keywords: list[KeywordSignal] = Field(default_factory=list)

    rising_queries: list[RisingQuery] = Field(default_factory=list)
    category_trends: list[CategoryTrend] = Field(default_factory=list)

    keyword_pool_stats: dict[str, int] = Field(
        default_factory=dict,
        description="counts of how many keywords came from each source",
    )
    notes: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Convenience                                                         #
    # ------------------------------------------------------------------ #

    def short_summary(self) -> str:
        """Compact one-liner for trace messages and logs."""
        return (
            f"{self.domain} · {self.time_window} · {self.region} — "
            f"{len(self.rising_keywords)} rising / "
            f"{len(self.stable_top_keywords)} stable / "
            f"{len(self.declining_keywords)} declining"
        )
