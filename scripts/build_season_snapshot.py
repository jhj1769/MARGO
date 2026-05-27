"""Build a v5 season-trend snapshot for ONE season.

Usage:

    python -m scripts.build_season_snapshot \
        --season 2023-SS \
        --processed-dir "data/Amazon Fashion/processed"

v5 pipeline (Collect → Synthesize):
  1. Tavily 4-axis search (RUNWAY/PRESS/STREET/POP), 47-domain allowlist,
     publish-date filtered to the season window.
  2. Single LLM call (gpt-4o-mini, temperature=0) extracts TrendItems with
     keyword + trend_stage (niche/rising/mainstream) + confidence +
     rationale + citations.

Output: `<processed_dir>/trend_cache/fashion_trend_<year>_<SS|FW>.json`
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Make `src/...` importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Load .env so TAVILY_API_KEY / OPENAI_API_KEY are picked up
_ENV = _REPO_ROOT / ".env"
if _ENV.exists():
    for line in _ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from adapters.llm.client import get_default_client                       # noqa: E402
from adapters.trends.season_pipeline import (                             # noqa: E402
    build_season_snapshot,
    save_snapshot,
    snapshot_path,
)
from adapters.trends.seasons import parse_season                          # noqa: E402
from adapters.trends.web_search import WebSearcher                        # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v5 season-trend snapshot.")
    p.add_argument("--season", required=True, help="e.g. 2023-SS or 2022-AW")
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--domain", default="fashion")
    p.add_argument("--region", default="US")
    p.add_argument("--force", action="store_true", help="overwrite existing snapshot")
    p.add_argument("--per-search-max-results", type=int, default=12,
                   help="Tavily max_results per axis query (default 12)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    season = parse_season(args.season)
    start, end = season.date_range()
    print(f"[season] {season.label} -> {start} .. {end}")

    out_path = snapshot_path(args.processed_dir, args.domain, season)
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path.name} already exists (use --force to overwrite).")
        sys.exit(0)

    t0 = time.time()

    # LLM
    llm = get_default_client()
    print(f"[llm] backend={llm.backend} model={llm.model}")

    # Tavily search
    searcher = WebSearcher()
    print(f"[search] backend={searcher.backend}")
    if searcher.backend != "tavily":
        print(
            "[search] ERROR: v5 pipeline requires Tavily. Set TAVILY_API_KEY "
            "(or override MARGO_WEB_SEARCH_BACKEND=tavily)."
        )
        sys.exit(2)

    snap = build_season_snapshot(
        season=season,
        llm=llm,
        searcher=searcher,
        domain=args.domain,
        region=args.region,
        processed_dir=args.processed_dir,
        per_search_max_results=args.per_search_max_results,
    )

    path = save_snapshot(snap, args.processed_dir, args.domain, season)
    print(
        f"[done] {path.name}  trends={len(snap.trends)}  "
        f"articles={snap.article_count}  domains={len(snap.sources_used)}  "
        f"total={time.time() - t0:.1f}s"
    )
    print(f"       notes: {snap.notes}")


if __name__ == "__main__":
    main()
