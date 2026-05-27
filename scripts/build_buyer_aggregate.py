"""Offline buyer-aggregate builder (Enhancement 2).

For every item with ≥ MIN_BUYERS positive buyers, materialise an
``ItemAudienceProfile`` and write it to
``<processed-dir>/buyer_aggregate/<item_id>.json``.

Usage::

    # Without cohort distribution (no LLM-inferred user states)
    python -m scripts.build_buyer_aggregate \\
        --processed-dir "data/Amazon Fashion/processed"

    # With cohort distribution (recommended) — requires the same JSONL as build_cohort_stats
    python -m scripts.build_buyer_aggregate \\
        --processed-dir "data/Amazon Fashion/processed" \\
        --user-states-jsonl user_states.jsonl
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from core.protocol.messages import UserPreferenceState, compute_cohort_signature
from data.fashion.audience_loader import (
    MIN_BUYERS,
    build_audience_profiles,
    save_audience_profiles,
)
from data.fashion.preference_stats import _extract_primary_category


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--user-states-jsonl", type=Path, default=None)
    p.add_argument("--positive-rating", type=float, default=4.0)
    p.add_argument("--min-buyers", type=int, default=MIN_BUYERS)
    p.add_argument("--max-items", type=int, default=None, help="Cap items for fast smoke runs.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    t0 = time.time()
    train = pd.read_parquet(args.processed_dir / "train.parquet")
    items = pd.read_parquet(args.processed_dir / "items.parquet")
    log.info("loaded train=%d, items=%d (%.1fs)", len(train), len(items), time.time() - t0)

    positive = train[train["rating"] >= args.positive_rating].copy()
    positive["user_id"] = positive["user_id"].astype(str)
    positive["item_id"] = positive["item_id"].astype(str)

    # buyers_by_item and user_histories share the positive-interaction split.
    buyers_by_item: dict[str, list[str]] = defaultdict(list)
    user_histories: dict[str, list[str]] = defaultdict(list)
    for uid, iid in zip(positive["user_id"].to_numpy(), positive["item_id"].to_numpy()):
        buyers_by_item[iid].append(uid)
        user_histories[uid].append(iid)
    log.info(
        "indexed %d items × %d users from %d positives",
        len(buyers_by_item), len(user_histories), len(positive),
    )

    # Per-item attributes for the aggregate.
    item_prices: dict[str, float] = {}
    item_categories: dict[str, str] = {}
    item_brands: dict[str, str] = {}
    for _, row in items.iterrows():
        iid = str(row.get("parent_asin"))
        price = row.get("price")
        if price is not None and price == price and price > 0:  # not NaN, positive
            item_prices[iid] = float(price)
        cat = _extract_primary_category(row.get("categories"))
        if cat:
            item_categories[iid] = cat
        brand = row.get("brand")
        if brand and str(brand).strip():
            item_brands[iid] = str(brand).strip()

    user_cohort_signatures: dict[str, str] = {}
    if args.user_states_jsonl and args.user_states_jsonl.exists():
        log.info("loading user states from %s", args.user_states_jsonl)
        with args.user_states_jsonl.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    state = UserPreferenceState.model_validate_json(line)
                except Exception as e:  # noqa: BLE001
                    log.warning("skip line: %s", e)
                    continue
                sig = state.cohort_signature or compute_cohort_signature(state)
                if sig:
                    user_cohort_signatures[state.user_id] = sig
        log.info("loaded %d user→signature mappings", len(user_cohort_signatures))

    # Filter to items that pass the threshold so we don't iterate the whole catalog.
    eligible_items = [iid for iid, buyers in buyers_by_item.items() if len(buyers) >= args.min_buyers]
    if args.max_items:
        eligible_items = eligible_items[: args.max_items]
    log.info("eligible items: %d (min buyers=%d)", len(eligible_items), args.min_buyers)

    t0 = time.time()
    profiles = build_audience_profiles(
        item_ids=eligible_items,
        buyers_by_item=buyers_by_item,
        user_histories=user_histories,
        item_prices=item_prices,
        item_categories=item_categories,
        item_brands=item_brands,
        user_cohort_signatures=user_cohort_signatures or None,
        min_buyers=args.min_buyers,
    )
    log.info("built %d profiles (%.1fs)", len(profiles), time.time() - t0)

    save_audience_profiles(profiles, args.processed_dir)


if __name__ == "__main__":
    main()
