"""Build a Google-Trends-grounded snapshot for the MARGO Trend Agent.

Pipeline
--------
1. Read the processed items table.
2. Build a keyword pool from
   (a) catalog tail-categories,
   (b) LLM-expanded trend variants of those seeds,
   (c) Google Trends ``related_queries.rising`` discoveries.
3. Fetch ``interest_over_time`` for the whole pool in batches of 5.
4. Classify each keyword into rising / stable / declining.
5. Aggregate per coarse category using the keyword→category mapping.
6. Persist the snapshot as JSON next to the processed dataset
   (``data/<dataset>/processed/trend_cache/google_trends_<window>.json``).

Usage::

    MARGO_LLM_BACKEND=vllm \\
    MARGO_VLLM_BASE_URL=http://localhost:8000/v1 \\
    MARGO_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \\
    margo/bin/python -m scripts.build_trend_snapshot \\
        --processed-dir "data/Amazon Fashion/processed" \\
        --domain fashion --time-window "2023-Q2" \\
        --gtrends-timeframe "2023-03-01 2023-05-31" \\
        --region US --max-keywords 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from grounding.trend_snapshot_schema import (
    CategoryTrend,
    RisingQuery,
    TrendSnapshot,
)
from llm import get_default_client
from trend_sources.google_trends import GoogleTrendsClient, classify_keywords
from trend_sources.keyword_pool import (
    build_keyword_pool,
    map_keywords_to_categories,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--domain", default="fashion")
    p.add_argument("--time-window", default="2023-Q2",
                   help="Logical key the Trend Agent will use to look this up.")
    p.add_argument("--gtrends-timeframe", default="2023-03-01 2023-05-31",
                   help="The actual date range pytrends queries.")
    p.add_argument("--region", default="US")
    p.add_argument("--max-keywords", type=int, default=50)
    p.add_argument("--n-llm-seeds", type=int, default=8)
    p.add_argument("--n-llm-per-seed", type=int, default=2)
    p.add_argument("--n-rising-seeds", type=int, default=6)
    p.add_argument("--n-rising-per-seed", type=int, default=4)
    p.add_argument("--no-llm", action="store_true", help="Skip LLM seed expansion.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path. Default = <processed>/trend_cache/google_trends_<window>.json")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()

    items_path = args.processed_dir / "items.parquet"
    print(f"\n== Building trend snapshot ==")
    print(f"  processed_dir   : {args.processed_dir}")
    print(f"  domain          : {args.domain}")
    print(f"  time_window     : {args.time_window}  (Google Trends timeframe={args.gtrends_timeframe!r})")
    print(f"  region          : {args.region}")
    print(f"  max_keywords    : {args.max_keywords}\n")

    t0 = time.time()
    items_df = pd.read_parquet(items_path)
    print(f"[load] items={len(items_df):,} in {time.time() - t0:.1f}s")

    # ── LLM (optional) ─────────────────────────────────────────────────
    llm = None
    if not args.no_llm:
        try:
            llm = get_default_client()
            print(f"[llm ] backend={llm.backend}  model={llm.model}")
        except Exception as e:
            print(f"[llm ] disabled ({e})")

    # ── Google Trends ──────────────────────────────────────────────────
    gtc = GoogleTrendsClient()
    print(f"[gtc ] pytrends client ready (sleep_between={gtc.sleep_between}s)")

    # ── Pool build ─────────────────────────────────────────────────────
    print(f"\n[pool] building keyword pool…")
    pool = build_keyword_pool(
        items_df,
        llm=llm,
        gtc=gtc,
        timeframe=args.gtrends_timeframe,
        geo=args.region,
        n_llm_seeds=args.n_llm_seeds,
        n_llm_per_seed=args.n_llm_per_seed,
        n_rising_seeds=args.n_rising_seeds,
        n_rising_per_seed=args.n_rising_per_seed,
        max_keywords=args.max_keywords,
    )
    print(f"[pool] catalog={len(pool.catalog)} | llm={len(pool.llm_seed)} | "
          f"rising={len(pool.rising_feedback)} | total={len(pool.all)}")

    # ── Bulk interest-over-time ────────────────────────────────────────
    print(f"\n[gtrends] fetching interest_over_time for {len(pool.all)} keywords (batches of 5)…")
    iot = gtc.fetch_interest_over_time(pool.all, timeframe=args.gtrends_timeframe, geo=args.region)
    print(f"[gtrends] iot matrix shape={iot.shape if not iot.empty else 'empty'}")

    rising, stable, declining = classify_keywords(iot, sources=pool.source_lookup())
    print(f"[class] rising={len(rising)} | stable={len(stable)} | declining={len(declining)}")

    # ── Rising-query records (for the snapshot) ────────────────────────
    print(f"\n[rising] re-fetching rising queries for snapshot (top {args.n_rising_seeds} seeds)…")
    rising_records = gtc.fetch_rising_queries(
        pool.catalog[: args.n_rising_seeds],
        timeframe=args.gtrends_timeframe,
        geo=args.region,
        top_per_seed=args.n_rising_per_seed,
    )
    rising_queries = [
        RisingQuery(seed=r.seed, query=r.query, growth_pct=r.growth_pct)
        for r in rising_records
    ]

    # ── Category aggregation ───────────────────────────────────────────
    kw_to_cat = map_keywords_to_categories(pool.all, items_df)
    print(f"[cat ] mapped {len(kw_to_cat)} / {len(pool.all)} keywords to a catalog category")

    bucket: dict[str, list] = defaultdict(list)
    for sig in [*rising, *stable, *declining]:
        cat = kw_to_cat.get(sig.keyword)
        if cat:
            bucket[cat].append(sig)

    category_trends: list[CategoryTrend] = []
    for cat, sigs in bucket.items():
        mean_score = sum(s.absolute_score for s in sigs) / len(sigs)
        mean_growth = sum((s.growth_pct or 0.0) for s in sigs) / len(sigs)
        if mean_growth >= 20:
            direction = "rising"
        elif mean_growth <= -15:
            direction = "declining"
        elif mean_score >= 35:
            direction = "stable"
        else:
            direction = "mixed"
        key_terms = [s.keyword for s in sorted(sigs, key=lambda x: (x.growth_pct or 0), reverse=True)[:4]]
        category_trends.append(CategoryTrend(
            category=cat,
            direction=direction,
            mean_score=round(mean_score, 2),
            mean_growth_pct=round(mean_growth, 1),
            key_terms=key_terms,
        ))
    category_trends.sort(key=lambda c: c.mean_growth_pct, reverse=True)

    # ── Assemble snapshot ─────────────────────────────────────────────
    snapshot = TrendSnapshot(
        domain=args.domain,
        time_window=args.time_window,
        region=args.region,
        snapshot_date=datetime.utcnow().strftime("%Y-%m-%d"),
        rising_keywords=rising[:20],
        stable_top_keywords=stable[:15],
        declining_keywords=declining[:10],
        rising_queries=rising_queries[:25],
        category_trends=category_trends[:20],
        keyword_pool_stats={
            "catalog": len(pool.catalog),
            "llm_seed": len(pool.llm_seed),
            "rising_feedback": len(pool.rising_feedback),
            "total": len(pool.all),
        },
        notes=f"Built by scripts/build_trend_snapshot.py in {time.time() - t0:.1f}s",
    )

    out_path = args.out or (
        args.processed_dir / "trend_cache" / f"google_trends_{args.time_window}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")

    print(f"\n[save] {out_path}  ({out_path.stat().st_size:,} bytes)")
    print(f"\n=== Snapshot summary ===")
    print(f"  {snapshot.short_summary()}")
    print(f"  rising keywords  : {[s.keyword for s in snapshot.rising_keywords[:10]]}")
    print(f"  stable keywords  : {[s.keyword for s in snapshot.stable_top_keywords[:5]]}")
    print(f"  declining        : {[s.keyword for s in snapshot.declining_keywords[:5]]}")
    print(f"  rising queries   : {[(q.seed, q.query) for q in snapshot.rising_queries[:6]]}")
    print(f"  category trends  : {[(c.category, c.direction) for c in snapshot.category_trends[:8]]}")
    print(f"\nOK ✓\n")


if __name__ == "__main__":
    main()
