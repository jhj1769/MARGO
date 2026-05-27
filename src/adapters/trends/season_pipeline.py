"""v5 season-trend pipeline orchestrator: Collect → Synthesize.

Single public entry: :func:`build_season_snapshot`. Composes the two
modules in this package into one persisted ``SeasonTrendSnapshot``.

File naming convention (v5):
  ``<processed_dir>/trend_cache/fashion_trend_<year>_<SS|FW>.json``

  Internal Season labels remain ``2023-SS`` / ``2017-AW`` (AW = industry
  standard for Autumn/Winter). The file name uses ``FW`` (Fall/Winter) for
  readability — the two are equivalent and the mapping happens here only.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from adapters.trends.season_collect import SeasonCollector
from adapters.trends.season_schema import SeasonTrendSnapshot
from adapters.trends.season_synthesize import synthesize
from adapters.trends.seasons import Season
from adapters.trends.web_search import WebSearcher

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# File-name convention                                                         #
# --------------------------------------------------------------------------- #


_SNAPSHOT_DIR_NAME = "trend_cache"


def _season_file_half(season: Season) -> str:
    """Internal 'AW' (industry standard) → file-name 'FW' (Fall/Winter)."""
    return "FW" if season.half == "AW" else "SS"


def snapshot_filename(domain: str, season: Season) -> str:
    """`fashion_trend_2023_SS.json` / `fashion_trend_2017_FW.json`.

    Note: ``domain`` is unused in the filename for the standard fashion case
    (kept as a parameter for future multi-domain support). The fixed prefix
    ``fashion_trend_`` reflects current scope.
    """
    return f"fashion_trend_{season.year}_{_season_file_half(season)}.json"


def snapshot_id_for(domain: str, season: Season) -> str:
    """`fashion_trend_2023_SS` — matches filename without extension."""
    return f"fashion_trend_{season.year}_{_season_file_half(season)}"


def snapshot_path(processed_dir: Path, domain: str, season: Season) -> Path:
    return (
        Path(processed_dir) / _SNAPSHOT_DIR_NAME
        / snapshot_filename(domain, season)
    )


# --------------------------------------------------------------------------- #
# Public entry                                                                 #
# --------------------------------------------------------------------------- #


def build_season_snapshot(
    *,
    season: Season,
    llm,                                              # LLMClient
    searcher: WebSearcher,
    domain: str = "fashion",
    region: str = "US",
    processed_dir: Optional[Path] = None,
    per_search_max_results: int = 12,
) -> SeasonTrendSnapshot:
    """End-to-end v5 build: 4-axis Tavily collect → single Synthesize LLM call."""
    t0 = time.time()
    start_iso, end_iso = season.date_range()
    snap_id = snapshot_id_for(domain, season)
    notes_lines: list[str] = []

    # ────────────────────────── Step 1 — Collect ───────────────────────────
    log.info("Step 1: Collect (4-axis Tavily) for %s …", season.label)
    collector = SeasonCollector(
        searcher=searcher,
        per_search_max_results=per_search_max_results,
    )
    collect = collector.collect(season)
    log.info(
        "Step 1 done: %d articles from %d domains (%.1fs)",
        collect.article_count, len(collect.sources_used), collect.elapsed_s,
    )
    notes_lines.append(
        f"collect: {collect.article_count} articles, "
        f"{len(collect.sources_used)} domains, {collect.elapsed_s:.1f}s"
    )
    if collect.article_count == 0:
        log.warning("No articles collected — Synthesize will return empty.")

    # ────────────────────────── Step 2 — Synthesize ────────────────────────
    log.info("Step 2: Synthesize …")
    synth = synthesize(
        llm=llm,
        season=season,
        collect=collect,
        domain=domain,
        region=region,
    )
    log.info(
        "Step 2 done: %d trends, %d themes (%.1fs)",
        len(synth.trends), len(synth.headline_themes), synth.elapsed_s,
    )
    notes_lines.append(f"synth: {len(synth.trends)} trends, {synth.elapsed_s:.1f}s")

    # ────────────────────────── Assemble ───────────────────────────────────
    snap = SeasonTrendSnapshot(
        snapshot_id=snap_id,
        domain=domain,
        season=season.label,
        window_start_iso=start_iso,
        window_end_iso=end_iso,
        region=region,
        snapshot_date=datetime.now(timezone.utc).date().isoformat(),
        summary=synth.summary,
        headline_themes=synth.headline_themes,
        trends=synth.trends,
        sources_used=collect.sources_used,
        article_count=collect.article_count,
        notes=" | ".join(notes_lines) + f" | total={time.time() - t0:.1f}s",
    )
    return snap


def save_snapshot(
    snap: SeasonTrendSnapshot, processed_dir: Path, domain: str, season: Season,
) -> Path:
    path = snapshot_path(processed_dir, domain, season)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snap.model_dump_json(indent=2), encoding="utf-8")
    log.info("season snapshot saved → %s", path)
    return path


def load_snapshot(path: Path) -> Optional[SeasonTrendSnapshot]:
    if not path.exists():
        return None
    try:
        return SeasonTrendSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as e:                            # noqa: BLE001
        log.warning("failed to load season snapshot %s: %s", path, e)
        return None
