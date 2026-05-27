"""Cohort coverage diagnostic (Enhancement 1.5).

Reports how the user population partitions into cohorts under the current
axis discretisation. If ``fallback_rate`` is above 50% the directive
prescribes coarsening axis values (e.g. fewer price tiers) before relying
on peer signal.

Usage::

    python -m scripts.diagnose_cohort_coverage \\
        --processed-dir "data/Amazon Fashion/processed" \\
        --min-history 5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from core.protocol.messages import PreferenceAxis, UserPreferenceState
from data.fashion.cohort_loader import MIN_COHORT_SIZE, cohort_coverage
from data.fashion.preference_stats import compute_deterministic_axes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--min-history", type=int, default=5)
    p.add_argument("--positive-rating", type=float, default=4.0)
    p.add_argument("--max-users", type=int, default=None)
    p.add_argument(
        "--min-cohort-size",
        type=int,
        default=MIN_COHORT_SIZE,
        help="Threshold used to compute the fallback rate.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    t0 = time.time()
    train = pd.read_parquet(args.processed_dir / "train.parquet")
    items = pd.read_parquet(args.processed_dir / "items.parquet")
    logging.info("loaded train=%d, items=%d (%.1fs)", len(train), len(items), time.time() - t0)

    positive = train[train["rating"] >= args.positive_rating].copy()
    positive["user_id"] = positive["user_id"].astype(str)
    positive["item_id"] = positive["item_id"].astype(str)

    purchases: dict[str, list[str]] = defaultdict(list)
    for uid, iid in zip(positive["user_id"].to_numpy(), positive["item_id"].to_numpy()):
        purchases[uid].append(iid)

    eligible = [u for u, p in purchases.items() if len(p) >= args.min_history]
    if args.max_users:
        eligible = eligible[: args.max_users]

    t0 = time.time()
    states: list[UserPreferenceState] = []
    for uid in eligible:
        axes_dict = compute_deterministic_axes(uid, purchases[uid], items)
        states.append(UserPreferenceState(
            user_id=uid,
            profile_nl="",
            axes=[PreferenceAxis(**a) for a in axes_dict.values()],
        ))
    logging.info("computed %d states (%.1fs)", len(states), time.time() - t0)

    report = cohort_coverage(states, min_cohort_size=args.min_cohort_size)

    print("\n=== Cohort Coverage ===")
    print(f"total_users:               {report['total_users']}")
    print(f"users_with_empty_signature: {report['users_with_empty_signature']}")
    print(f"total_cohorts:             {report['total_cohorts']}")
    print(f"users_in_qualified_cohorts: {report['users_in_qualified_cohorts']}  (size >= {args.min_cohort_size})")
    print(f"fallback_rate:             {report['fallback_rate']:.1%}")
    print(f"\nsize_histogram:")
    for bucket, count in report["size_histogram"].items():
        print(f"  cohorts of size {bucket}: {count}")
    print(f"\ntop_10_cohorts (signature → user count):")
    for sig, cnt in report["top_10_cohorts"]:
        print(f"  {cnt:>6d}  {sig}")

    out_path = args.processed_dir / "cohort_coverage_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport written to {out_path}")

    if report["fallback_rate"] > 0.50:
        print(
            "\n⚠  fallback_rate > 50% — directive prescribes coarsening axis values "
            "(e.g. collapse price into 3 tiers) before relying on peer signal."
        )


if __name__ == "__main__":
    main()
