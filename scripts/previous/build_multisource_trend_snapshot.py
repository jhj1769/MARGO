"""Build a multi-source trend snapshot (Enhancement 4 — 5-layer pipeline).

Pipeline:
    L5 Keyword Pool  : catalog ∪ optional brief-extracted keywords
    L1 Source        : GDELT (primary) + Pinterest + Google Trends + YouTube
    L2 Signal        : 4w/12w MA → rising/stable/declining/niche per source
    L3 Consensus     : reliability-weighted aggregation; disagreement preserved
    L4 Semantic      : LLM proposes fashion attributes; BGE-M3 validates against catalog

Usage::

    # Full 4-source pipeline
    MARGO_LLM_BACKEND=vllm python -m scripts.build_multisource_trend_snapshot \\
        --processed-dir "data/Amazon Fashion/processed" \\
        --time-start 2023-01-01 --time-end 2023-09-10 \\
        --time-window-label "2023" \\
        --gdelt-data-dir data/external_trends/gdelt \\
        --brief "Y2K revival; minimalism rising; quiet luxury"

    # Skip individual sources explicitly
    python -m scripts.build_multisource_trend_snapshot \\
        --processed-dir "data/Amazon Fashion/processed" \\
        --time-start 2023-01-01 --time-end 2023-09-10 \\
        --no-google --no-gdelt --no-youtube --no-pinterest

Source toggles auto-degrade: a source whose credentials/data are missing
will log once and contribute zero signals — the snapshot still builds.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from adapters.llm import get_default_client
from adapters.trends.base import TrendSourceAdapter
from adapters.trends.keyword_pool import (
    FASHION_TREND_LEXICON,
    extract_brief_keywords,
    extract_catalog_keywords,
)
from adapters.trends.pipeline import (
    build_multisource_snapshot,
    save_snapshot,
    snapshot_path,
)
from adapters.trends.semantic_mapper import make_bge_matcher


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--season", default=None,
                   help="Fashion season identifier — e.g. '2023-SS' (Feb-Jul) or "
                        "'2022-AW' (Aug 2022 - Jan 2023). When given, fills in "
                        "--time-start / --time-end / --time-window-label.")
    p.add_argument("--time-start", default=None, help="ISO date, e.g. 2023-01-01")
    p.add_argument("--time-end", default=None, help="ISO date, e.g. 2023-09-10")
    p.add_argument("--time-window-label", default=None,
                   help="Logical key used in snapshot_id (defaults to start__end).")
    p.add_argument("--domain", default="fashion")
    p.add_argument("--region", default="US")
    p.add_argument("--brief", default=None,
                   help="Operator brief text (LLM extracts brief-derived keywords).")

    # Source toggles + data locations
    p.add_argument("--no-gdelt", action="store_true")
    p.add_argument("--gdelt-data-dir", type=Path, default=None,
                   help="Directory containing GDELT GKG CSV (or .zip / .gz) files.")
    p.add_argument("--gdelt-data-path", type=Path, default=None,
                   help="Single GDELT CSV (e.g. BigQuery export).")
    p.add_argument("--no-google", action="store_true")
    p.add_argument("--no-youtube", action="store_true")
    p.add_argument("--youtube-cache-dir", type=Path, default=None,
                   help="Directory for cached YouTube search.list responses "
                        "(default: <processed-dir>/trend_cache).")
    p.add_argument("--no-wikipedia", action="store_true")
    p.add_argument("--wikipedia-cache-dir", type=Path, default=None,
                   help="Directory for cached Wikimedia pageview responses "
                        "(default: <processed-dir>/trend_cache).")
    p.add_argument("--no-pinterest", action="store_true")
    p.add_argument("--pinterest-cache-dir", type=Path, default=None,
                   help="Directory for cached Pinterest /trends responses "
                        "(default: <processed-dir>/trend_cache).")

    # Pool size
    p.add_argument("--n-catalog", type=int, default=30,
                   help="Top-N catalog-derived keywords by frequency.")
    p.add_argument("--max-keywords", type=int, default=60,
                   help="Cap on the total analysed keyword count.")
    p.add_argument("--use-trend-lexicon", action="store_true",
                   help="Prepend the curated FASHION_TREND_LEXICON to the pool.")

    # L4 toggles
    p.add_argument("--no-semantic", action="store_true",
                   help="Skip L4 semantic mapping (LLM + BGE-M3).")
    p.add_argument("--no-bge", action="store_true",
                   help="Skip BGE-M3 retriever (fall back to no validation).")
    return p.parse_args()


def build_adapters(args: argparse.Namespace) -> list[TrendSourceAdapter]:
    adapters: list[TrendSourceAdapter] = []
    cache_default = args.processed_dir / "trend_cache"

    # ORDER MATTERS for snapshot.source_metadata only — math uses priors.
    if not args.no_gdelt:
        if not args.gdelt_data_dir and not args.gdelt_data_path:
            print("[gdelt] no --gdelt-data-dir / --gdelt-data-path given — skipping GDELT.")
        else:
            from adapters.trends.gdelt_adapter import GDELTAdapter
            adapters.append(GDELTAdapter(
                data_dir=args.gdelt_data_dir,
                data_path=args.gdelt_data_path,
            ))
            print(f"[gdelt] enabled (prior={GDELTAdapter.reliability_prior})")

    if not args.no_wikipedia:
        from adapters.trends.wikipedia_adapter import WikipediaAdapter
        adapter = WikipediaAdapter(
            cache_dir=args.wikipedia_cache_dir or cache_default,
        )
        adapters.append(adapter)
        print(f"[wikipedia] enabled (prior={WikipediaAdapter.reliability_prior})")

    if not args.no_pinterest:
        from adapters.trends.pinterest_adapter import PinterestAdapter, _resolve_token_file
        adapter = PinterestAdapter(
            cache_dir=args.pinterest_cache_dir or cache_default,
        )
        adapters.append(adapter)
        if adapter.access_token:
            print(f"[pinterest] enabled (prior={PinterestAdapter.reliability_prior})")
        else:
            print(
                f"[pinterest] enabled but NO TOKEN — set PINTEREST_ACCESS_TOKEN or run "
                f"`python -m scripts.pinterest_oauth authorize` (writes to {_resolve_token_file()}). "
                f"Pipeline will continue and record empty contributions."
            )

    if not args.no_google:
        try:
            from adapters.trends.google_trends_adapter import GoogleTrendsAdapter
            adapters.append(GoogleTrendsAdapter())
            print(f"[google] enabled (prior={GoogleTrendsAdapter.reliability_prior})")
        except Exception as e:
            print(f"[google] disabled: {e}")

    if not args.no_youtube:
        from adapters.trends.youtube_adapter import YouTubeAdapter
        adapter = YouTubeAdapter(
            cache_dir=args.youtube_cache_dir or cache_default,
        )
        adapters.append(adapter)
        if adapter.api_key:
            print(f"[youtube] enabled (prior={YouTubeAdapter.reliability_prior})")
        else:
            print(
                "[youtube] enabled but NO API KEY — set YOUTUBE_API_KEY. "
                "Pipeline will continue and record empty contributions."
            )

    return adapters


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()

    # --season fills in time_start / time_end / time_window_label if given.
    if args.season:
        from adapters.trends.seasons import parse_season
        season = parse_season(args.season)
        start, end = season.date_range()
        args.time_start = args.time_start or start
        args.time_end = args.time_end or end
        args.time_window_label = args.time_window_label or season.label
        print(f"[season] {season.label} -> {start} .. {end}")

    if not (args.time_start and args.time_end):
        sys.exit("error: must supply --season or both --time-start and --time-end")

    t0 = time.time()
    items_df = pd.read_parquet(args.processed_dir / "items.parquet")
    print(f"[load] items={len(items_df):,} ({time.time() - t0:.1f}s)")

    # ----- LLM (for brief extraction + L4 semantic proposal) ---------------
    llm_client = None
    try:
        llm_client = get_default_client()
        print(f"[llm ] backend={llm_client.backend} model={llm_client.model}")
    except Exception as e:
        print(f"[llm ] disabled ({e})")

    # ----- L5 keyword pool -------------------------------------------------
    catalog_kws = extract_catalog_keywords(items_df, top_n=args.n_catalog)
    brief_kws: list[str] = []
    if args.brief and llm_client is not None:
        brief_kws = extract_brief_keywords(llm_client, args.brief)
        print(f"[brief] extracted {len(brief_kws)} keywords from brief")
    lexicon_kws = list(FASHION_TREND_LEXICON) if args.use_trend_lexicon else []
    if lexicon_kws:
        print(f"[lex ] curated fashion trend lexicon: {len(lexicon_kws)} terms")

    # Priority order: brief (operator intent) → lexicon (curated trends) → catalog (taxonomy).
    seen: set[str] = set()
    merged: list[str] = []
    for src in (brief_kws, lexicon_kws, catalog_kws):
        for kw in src:
            kw = kw.lower()
            if kw not in seen:
                seen.add(kw)
                merged.append(kw)
            if len(merged) >= args.max_keywords:
                break
        if len(merged) >= args.max_keywords:
            break

    keyword_pool = {
        "catalog_derived": catalog_kws,
        "brief_derived": brief_kws,
        "lexicon_derived": lexicon_kws,
        "merged": merged,
    }
    print(f"[pool] catalog={len(catalog_kws)} brief={len(brief_kws)} "
          f"lexicon={len(lexicon_kws)} merged={len(merged)}")

    # ----- L1 adapters -----------------------------------------------------
    adapters = build_adapters(args)
    if not adapters:
        print("[error] no adapters enabled — aborting.")
        sys.exit(2)

    # ----- L4 retriever (for BGE-M3 catalog validation) --------------------
    matcher = None
    if not args.no_bge and not args.no_semantic:
        try:
            from adapters.retrieval.bge_retriever import BGERetriever
            index_path = args.processed_dir / "faiss_index.bin"
            ids_path = args.processed_dir / "item_ids.txt"
            if index_path.exists() and ids_path.exists():
                item_ids = ids_path.read_text(encoding="utf-8").splitlines()
                import os
                retriever = BGERetriever(
                    index_path=index_path,
                    item_ids=item_ids,
                    device=os.getenv("MARGO_BGE_DEVICE", "cuda:0"),
                )
                matcher = make_bge_matcher(retriever)
                print(f"[bge ] semantic-mapping matcher ready ({len(item_ids)} items)")
            else:
                print("[bge ] index missing — L4 semantic mapping will run UNVALIDATED.")
        except Exception as e:
            print(f"[bge ] disabled: {e}")

    # ----- Build snapshot --------------------------------------------------
    snap = build_multisource_snapshot(
        keyword_pool=keyword_pool,
        adapters=adapters,
        time_window=(args.time_start, args.time_end),
        time_window_label=args.time_window_label or f"{args.time_start}__{args.time_end}",
        domain=args.domain,
        region=args.region,
        llm_client=llm_client,
        catalog_matcher=matcher,
        processed_dir=args.processed_dir,
        skip_semantic=args.no_semantic,
    )

    path = save_snapshot(snap, args.processed_dir)
    print(f"\n[save] {path}  ({path.stat().st_size:,} bytes)")
    print(f"  signals    : {len(snap.signals)}")
    rising = [s for s in snap.signals if s.aggregated_lifecycle == "rising"]
    declining = [s for s in snap.signals if s.aggregated_lifecycle == "declining"]
    disagreed = [s for s in snap.signals if s.disagreement_flag]
    print(f"  rising     : {len(rising)}")
    print(f"  declining  : {len(declining)}")
    print(f"  with_disagreement : {len(disagreed)}")
    print(f"  semantics  : {len(snap.semantics)}")

    if rising[:5]:
        print(f"\n  top 5 rising  : {[s.keyword for s in rising[:5]]}")
    if disagreed[:3]:
        print("\n  example disagreements:")
        for s in disagreed[:3]:
            print(f"    {s.keyword}: {s.disagreement_nl}")

    print("\nOK ✓")


if __name__ == "__main__":
    main()
