"""Batch-build :class:`UserPreferenceState` records for offline cohort building.

The cohort-stats and buyer-aggregate scripts both depend on user states whose
signatures match what the runtime ``UserAgent.build_profile`` produces. That
means every state needs the LLM-inferred style axis, not just the 3
deterministic axes.

This script runs ``UserAgent.build_profile`` on every selected user, captures
the resulting :class:`UserPreferenceState`, and writes one JSON record per
line to a JSONL file. Downstream scripts pass that JSONL via
``--user-states-jsonl``.

Usage::

    python -m scripts.build_user_states \\
        --processed-dir "data/Amazon Fashion/processed" \\
        --out user_states_2023.jsonl \\
        --filter-2023-test \\
        --max-users 5000 \\
        --include-user AE22EYWG7T6P6C642J5QIL4QHS4A
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from adapters.llm import get_default_client
from core.agents.user_agent import UserAgent

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path,
                   help="Output JSONL path.")
    p.add_argument("--filter-2023-test", action="store_true",
                   help="Restrict to users whose last interaction is in 2023.")
    p.add_argument("--min-history", type=int, default=5,
                   help="Skip users with fewer positive interactions.")
    p.add_argument("--max-users", type=int, default=None,
                   help="Cap on users processed (random sample).")
    p.add_argument("--include-user", action="append", default=[],
                   help="User IDs to include unconditionally (repeatable).")
    p.add_argument("--positive-rating", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--log-every", type=int, default=200)
    return p.parse_args()


def _build_user_histories(
    train: pd.DataFrame, items: pd.DataFrame
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Render each user's history (mirror of api._build_user_history_table)."""
    item_text_lookup = {
        str(r["parent_asin"]): f"{r.get('title', '')} | price=${r.get('price', '?')}"
        for _, r in items.iterrows()
    }
    texts: dict[str, list[str]] = {}
    ids: dict[str, list[str]] = {}
    for uid, group in train.sort_values("timestamp").groupby("user_id"):
        seq = [str(i) for i in group["item_id"]]
        ids[str(uid)] = seq
        texts[str(uid)] = [item_text_lookup.get(i, "[unknown item]") for i in seq]
    return texts, ids


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    rng = random.Random(args.seed)
    llm = get_default_client()
    log.info("LLM backend=%s model=%s", llm.backend, llm.model)

    t0 = time.time()
    train = pd.read_parquet(args.processed_dir / "train.parquet")
    items = pd.read_parquet(args.processed_dir / "items.parquet")
    log.info("loaded train=%d items=%d (%.1fs)", len(train), len(items), time.time() - t0)

    # Eligibility filter — positive history >= min-history.
    positive = train[train["rating"] >= args.positive_rating].copy()
    positive["user_id"] = positive["user_id"].astype(str)
    hist_counts = positive["user_id"].value_counts()
    eligible = set(hist_counts[hist_counts >= args.min_history].index)
    log.info("eligible users (>= %d positives): %d", args.min_history, len(eligible))

    # Optional restriction to 2023 last-interaction users.
    if args.filter_2023_test:
        test = pd.read_parquet(args.processed_dir / "test.parquet")
        test["user_id"] = test["user_id"].astype(str)
        test["year"] = pd.to_datetime(test["timestamp"], unit="ms", utc=True).dt.year
        users_2023 = set(test.loc[test["year"] == 2023, "user_id"])
        eligible &= users_2023
        log.info("after 2023-last-interaction filter: %d users", len(eligible))

    pool = sorted(eligible)
    rng.shuffle(pool)
    if args.max_users:
        pool = pool[: args.max_users]
    # Force-include explicit user IDs even if dropped by sampling.
    explicit = [u for u in args.include_user if u in hist_counts.index]
    for uid in explicit:
        if uid not in pool:
            pool.append(uid)
    log.info("processing %d users (incl. %d explicit)", len(pool), len(explicit))

    # Pre-render histories ONCE (the loader is the slow part — we avoid
    # paying it per-user).
    t0 = time.time()
    pool_set = set(pool)
    relevant_positive = positive[positive["user_id"].isin(pool_set)]
    histories_texts, histories_ids = _build_user_histories(relevant_positive, items)
    log.info("rendered histories for %d users (%.1fs)", len(histories_texts), time.time() - t0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    t_loop = time.time()
    with args.out.open("w", encoding="utf-8") as f:
        for i, uid in enumerate(pool):
            history = histories_texts.get(uid, [])
            history_item_ids = histories_ids.get(uid, [])
            if not history:
                skipped += 1
                continue
            agent = UserAgent(
                user_id=uid,
                history=history,
                history_item_ids=history_item_ids,
                items_df=items,
                llm=llm,
            )
            try:
                agent.build_profile()
            except Exception as e:  # noqa: BLE001
                log.warning("build_profile failed for %s: %s", uid, e)
                skipped += 1
                continue
            state = agent.state.preference_state
            if state is None:
                skipped += 1
                continue
            f.write(state.model_dump_json() + "\n")
            written += 1
            if (i + 1) % args.log_every == 0:
                rate = (i + 1) / (time.time() - t_loop)
                eta_s = (len(pool) - (i + 1)) / max(rate, 1e-3)
                log.info("progress %d/%d (%.1f users/s, eta=%.1fm) written=%d skipped=%d",
                         i + 1, len(pool), rate, eta_s / 60, written, skipped)

    log.info("DONE — wrote %d states (skipped %d) to %s", written, skipped, args.out)


if __name__ == "__main__":
    main()
