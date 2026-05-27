"""Offline cohort builder (Enhancement 1.5).

Groups users by cohort signature and writes per-cohort JSON stats to
``<processed-dir>/cohort_stats/``. Cohorts smaller than ``MIN_COHORT_SIZE``
(5) are dropped — peer signal from < 5 users is too noisy.

Two state-input modes:

* **Statistical-only (default)** — computes price/category/brand axes from
  ``train.parquet``. Fast (no LLM), but cohorts ignore style direction.
  Runtime ``UserAgent.build_profile`` will produce 4-axis signatures that
  CANNOT match these 3-axis cohorts, so this mode only makes sense when
  you also drop style at runtime.

* **From pre-computed states (recommended)** — read user states (with
  style) from a JSONL file produced by an offline UserAgent batch pass.
  This is the runtime-matching mode: signatures here include style so
  runtime lookups hit the same cohorts.

Usage::

    # Mode 1 — fast iteration, statistical-only
    python -m scripts.build_cohort_stats \\
        --processed-dir "data/Amazon Fashion/processed" \\
        --min-history 5

    # Mode 2 — match runtime
    python -m scripts.build_cohort_stats \\
        --processed-dir "data/Amazon Fashion/processed" \\
        --user-states-jsonl user_states.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from core.protocol.messages import (
    PreferenceAxis,
    UserPreferenceState,
    compute_cohort_signature,
)
from data.fashion.cohort_loader import (
    MIN_COHORT_SIZE,
    build_cohorts,
    save_cohorts,
)
from data.fashion.preference_stats import compute_deterministic_axes


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument(
        "--user-states-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL of UserPreferenceState records (one per line). "
            "When provided, axes are taken from this file (including style); "
            "otherwise statistical axes are computed from train.parquet."
        ),
    )
    p.add_argument(
        "--min-history",
        type=int,
        default=5,
        help="Skip users with fewer than this many positive interactions.",
    )
    p.add_argument(
        "--min-cohort-size",
        type=int,
        default=MIN_COHORT_SIZE,
        help="Cohort emission threshold.",
    )
    p.add_argument(
        "--positive-rating",
        type=float,
        default=4.0,
        help="Rating threshold for 'positive' interactions.",
    )
    p.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Cap users for fast smoke runs (None = all).",
    )
    return p.parse_args()


def _load_states_from_jsonl(path: Path) -> list[UserPreferenceState]:
    states: list[UserPreferenceState] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                states.append(UserPreferenceState.model_validate_json(line))
            except Exception as e:  # noqa: BLE001
                log.warning("skip line %d in %s: %s", line_no, path, e)
    return states


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    t0 = time.time()
    train = pd.read_parquet(args.processed_dir / "train.parquet")
    items = pd.read_parquet(args.processed_dir / "items.parquet")
    log.info("loaded train=%d rows, items=%d (%.1fs)", len(train), len(items), time.time() - t0)

    positive = train[train["rating"] >= args.positive_rating].copy()
    positive["item_id"] = positive["item_id"].astype(str)
    positive["user_id"] = positive["user_id"].astype(str)
    log.info("positive interactions (rating >= %.1f): %d", args.positive_rating, len(positive))

    user_purchases: dict[str, list[str]] = defaultdict(list)
    for uid, iid in zip(positive["user_id"].to_numpy(), positive["item_id"].to_numpy()):
        user_purchases[uid].append(iid)

    if args.user_states_jsonl is not None:
        # Mode 2: pre-computed states (with style) — matches runtime signatures.
        log.info("loading pre-computed user states from %s", args.user_states_jsonl)
        all_states = _load_states_from_jsonl(args.user_states_jsonl)
        # Filter to users who also meet the min-history bar in this split.
        states = [
            s for s in all_states
            if len(user_purchases.get(s.user_id, [])) >= args.min_history
        ]
        if args.max_users:
            states = states[: args.max_users]
        log.info(
            "loaded %d states from JSONL, %d retained after history filter",
            len(all_states), len(states),
        )
    else:
        # Mode 1: statistical-only computation on the fly.
        eligible_users = [u for u, p in user_purchases.items() if len(p) >= args.min_history]
        if args.max_users:
            eligible_users = eligible_users[: args.max_users]
        log.info("eligible users (>= %d positives): %d", args.min_history, len(eligible_users))

        t0 = time.time()
        states = []
        for i, uid in enumerate(eligible_users):
            axes_dict = compute_deterministic_axes(uid, user_purchases[uid], items)
            axes = [PreferenceAxis(**a) for a in axes_dict.values()]
            sig = compute_cohort_signature(
                UserPreferenceState(user_id=uid, profile_nl="", axes=axes)
            )
            states.append(UserPreferenceState(
                user_id=uid, profile_nl="", axes=axes, cohort_signature=sig,
            ))
            if (i + 1) % 50_000 == 0:
                log.info("computed axes for %d / %d users (%.1fs)", i + 1, len(eligible_users), time.time() - t0)
        log.info(
            "computed %d statistical-only states (%.1fs) — "
            "these signatures will NOT match runtime 4-axis signatures",
            len(states), time.time() - t0,
        )

    # Lightweight per-item brand/category lookup for cohort attribute tallies.
    items_indexed = items.set_index("parent_asin")
    item_brands: dict[str, str] = {}
    item_categories: dict[str, str] = {}
    for iid, row in items_indexed.iterrows():
        brand = row.get("brand")
        if brand and str(brand).strip():
            item_brands[str(iid)] = str(brand).strip()
        # Reuse the primary-category extractor from preference_stats.
        from data.fashion.preference_stats import _extract_primary_category
        cat = _extract_primary_category(row.get("categories"))
        if cat:
            item_categories[str(iid)] = cat

    t0 = time.time()
    cohorts = build_cohorts(
        states,
        user_purchases,
        min_cohort_size=args.min_cohort_size,
        item_brands=item_brands,
        item_categories=item_categories,
    )
    log.info("built %d qualified cohorts (%.1fs)", len(cohorts), time.time() - t0)

    save_cohorts(cohorts, args.processed_dir)


if __name__ == "__main__":
    main()
