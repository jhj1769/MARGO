"""Build the User Agent dual-layer 'Rejected' signal + qualitative review text.

Why this script exists
----------------------
``scripts/preprocess.py`` aggressively drops everything except rating ≥ 4 to
keep the positive interaction matrix small. The Heterogeneous Stakeholder
Reasoning thesis (see ``RESEARCH_OVERVIEW.md``) needs two additional signals
that the legacy pipeline throws away:

    1. **Rejected interactions (rating 1-2)** — User Agent's dual-layer
       reasoning needs the "actively disliked" signal, not just the positive
       one. We restrict to the 5-core entity set so every row joins cleanly
       with train/valid/test.

    2. **Review text + helpful_vote + verified_purchase** — Item Agent
       autobiography uses these as qualitative evidence per cohort.

We stream the 27 GB raw JSONL **once** and write two new parquet files into
the existing processed dir. Nothing else changes — train/valid/test/items
stay byte-identical so downstream artifacts (buyer_aggregate, cohort_stats,
trend_cache, faiss_index) remain valid.

Usage
-----
::

    python -m scripts.build_negative_signals \\
        --processed-dir "data/Amazon Fashion/processed" \\
        --reviews-path  "data/Amazon Fashion/Clothing_Shoes_and_Jewelry.jsonl"

For a fast sanity check, pass ``--max-reviews 500000`` first.

Outputs
-------
- ``<processed-dir>/rejected.parquet`` — rating ∈ {1, 2} on 5-core entities
- ``<processed-dir>/reviews_text.parquet`` — text + helpful_vote +
  verified_purchase for **every** rating in {1, 2, 3, 4, 5} on 5-core entities
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from data.fashion.loader import iter_review_rows, normalise_review_full

log = logging.getLogger(__name__)


# Ambivalent rating 3 is kept in reviews_text (for full qualitative coverage)
# but is NOT part of the rejected layer. The User Agent only treats rating
# 1-2 as "actively disliked"; 3 is treated as ambivalent and contributes
# qualitatively only.
REJECTED_RATINGS = {1.0, 2.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the Rejected layer + qualitative review text for the User "
            "Agent dual-layer reasoning. Streams raw reviews once, writes two "
            "new parquet files into the existing processed dir."
        ),
    )
    p.add_argument("--processed-dir", required=True, type=Path,
                   help="Existing processed dir (containing train/valid/test/items.parquet).")
    p.add_argument("--reviews-path", required=True, type=Path,
                   help="Raw reviews JSONL (or .jsonl.gz). 27 GB for the full Amazon Fashion file.")
    p.add_argument("--max-reviews", type=int, default=0,
                   help="0 = no cap. Pass a small number for sanity-check runs.")
    p.add_argument("--max-text-len", type=int, default=1000,
                   help="Truncate per-review text at this length. Median raw text ≈ 135.")
    p.add_argument("--rejected-out", type=Path, default=None,
                   help="Override path for rejected.parquet (defaults to <processed-dir>/rejected.parquet).")
    p.add_argument("--text-out", type=Path, default=None,
                   help="Override path for reviews_text.parquet.")
    p.add_argument("--no-text", action="store_true",
                   help="Skip the reviews_text.parquet output (rejected.parquet only).")
    p.add_argument("--log-every", type=int, default=2_000_000,
                   help="Heartbeat log every N raw rows scanned.")
    return p.parse_args()


def load_5core_entities(processed_dir: Path) -> tuple[set[str], set[str]]:
    """Load the 5-core user_id and item_id sets from existing splits.

    Restricting rejected/text to these entity sets is critical:
      * Keeps the new files joinable with train/valid/test
      * Avoids ballooning the rejected pool with cold users/items that the
        rest of the pipeline can't reason about
    """
    train = pd.read_parquet(processed_dir / "train.parquet")
    valid = pd.read_parquet(processed_dir / "valid.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")
    users = (
        pd.concat([train["user_id"], valid["user_id"], test["user_id"]])
        .astype(str).unique()
    )
    items = (
        pd.concat([train["item_id"], valid["item_id"], test["item_id"]])
        .astype(str).unique()
    )
    log.info("5-core entities: %d users, %d items", len(users), len(items))
    return set(users), set(items)


def stream_and_collect(
    reviews_path: Path,
    *,
    users_5core: set[str],
    items_5core: set[str],
    max_rows: Optional[int],
    max_text_len: int,
    keep_text: bool,
    log_every: int,
) -> tuple[list[dict], list[dict]]:
    """Single-pass stream over raw reviews.

    Returns (rejected_records, text_records). When ``keep_text=False`` the
    second list is empty.

    The filter is applied row-by-row to avoid loading 100M+ rows into RAM.
    """
    rejected: list[dict] = []
    text_records: list[dict] = []

    t0 = time.time()
    kept_rejected = 0
    kept_text = 0
    scanned = 0
    skipped_unknown_entity = 0

    for i, row in enumerate(iter_review_rows(reviews_path, max_rows=max_rows or None, log_every=log_every)):
        scanned += 1
        rec = normalise_review_full(row, max_text_len=max_text_len)
        uid = rec.get("user_id")
        iid = rec.get("item_id")
        if uid is None or iid is None:
            continue
        uid_s = str(uid)
        iid_s = str(iid)
        # Restrict to 5-core entities so the new files are joinable.
        if uid_s not in users_5core or iid_s not in items_5core:
            skipped_unknown_entity += 1
            continue

        if rec["rating"] in REJECTED_RATINGS:
            # rejected.parquet — small schema, no text (saves ~75% size)
            rejected.append({
                "user_id": uid_s,
                "item_id": iid_s,
                "rating": rec["rating"],
                "timestamp": rec["timestamp"],
            })
            kept_rejected += 1

        if keep_text:
            text_records.append({
                "user_id": uid_s,
                "item_id": iid_s,
                "rating": rec["rating"],
                "timestamp": rec["timestamp"],
                "text": rec["text"],
                "helpful_vote": rec["helpful_vote"],
                "verified_purchase": rec["verified_purchase"],
            })
            kept_text += 1

    elapsed = time.time() - t0
    log.info(
        "stream done: scanned=%d, in_5core=%d, rejected=%d, with_text=%d, took=%.1fs",
        scanned,
        scanned - skipped_unknown_entity,
        kept_rejected,
        kept_text,
        elapsed,
    )
    return rejected, text_records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    processed_dir = args.processed_dir
    if not (processed_dir / "train.parquet").exists():
        raise FileNotFoundError(
            f"{processed_dir / 'train.parquet'} not found — run scripts/preprocess.py first."
        )

    users_5core, items_5core = load_5core_entities(processed_dir)

    rejected, text_records = stream_and_collect(
        args.reviews_path,
        users_5core=users_5core,
        items_5core=items_5core,
        max_rows=args.max_reviews or None,
        max_text_len=args.max_text_len,
        keep_text=not args.no_text,
        log_every=args.log_every,
    )

    # ---------- Write rejected.parquet ----------
    rejected_out = args.rejected_out or (processed_dir / "rejected.parquet")
    df_rejected = pd.DataFrame.from_records(rejected)
    df_rejected.to_parquet(rejected_out)
    log.info("rejected.parquet → %s (%d rows)", rejected_out, len(df_rejected))
    if not df_rejected.empty:
        log.info(
            "rejected distribution: %s",
            df_rejected["rating"].value_counts().sort_index().to_dict(),
        )
        log.info(
            "rejected coverage: %d users (%.1f%%), %d items (%.1f%%)",
            df_rejected["user_id"].nunique(),
            df_rejected["user_id"].nunique() / max(1, len(users_5core)) * 100,
            df_rejected["item_id"].nunique(),
            df_rejected["item_id"].nunique() / max(1, len(items_5core)) * 100,
        )

    # ---------- Write reviews_text.parquet (optional) ----------
    if not args.no_text:
        text_out = args.text_out or (processed_dir / "reviews_text.parquet")
        df_text = pd.DataFrame.from_records(text_records)
        df_text.to_parquet(text_out)
        log.info("reviews_text.parquet → %s (%d rows)", text_out, len(df_text))
        if not df_text.empty:
            log.info(
                "text length: mean=%.0f, median=%.0f, with_text=%.1f%%",
                df_text["text"].str.len().mean(),
                df_text["text"].str.len().median(),
                (df_text["text"].str.len() > 0).mean() * 100,
            )


if __name__ == "__main__":
    main()
