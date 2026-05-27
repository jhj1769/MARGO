"""External evidence for the Trend Agent (v5 — season-snapshot only).

Active pipeline:

    build_season_snapshot.py
        -> fashion_trend_<year>_<SS|FW>.json  (curated by web_search.py)
        -> TrendAgent loads & narrates via season_pipeline.load_snapshot

The earlier v3 (google-trends-only) and v4 (multi-source consensus)
pipelines have been retired and live under ``src/previous/adapters/trends/``
for reference. See RESEARCH_OVERVIEW.md sec 4.3.
"""

from adapters.trends.web_search import WebSearcher, WebSnippet

__all__ = ["WebSearcher", "WebSnippet"]
