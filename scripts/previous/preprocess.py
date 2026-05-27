"""Step 1 — Amazon Fashion preprocessing pipeline.

Usage::

    python -m scripts.preprocess \
        --data-dir "data/Amazon Fashion" \
        --out-dir  "data/Amazon Fashion/processed" \
        --max-reviews 0    # 0 = no cap (full file ≈ 100M+ rows)

For initial sanity checks pass ``--max-reviews 200000``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from data.fashion.loader import (
    load_raw_pair,
    resume_items_and_vocab,
    standard_pipeline,
    write_processed,
)
from data.fashion.vocabulary import build_fashion_vocabulary

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess Amazon Fashion (Clothing/Shoes/Jewelry).")
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--reviews-name", default="Clothing_Shoes_and_Jewelry.jsonl")
    p.add_argument("--meta-name", default="meta_Clothing_Shoes_and_Jewelry.jsonl")
    p.add_argument("--min-rating", type=float, default=4.0)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--max-reviews", type=int, default=0,
                   help="0 = full file. Use small numbers for fast iteration.")
    p.add_argument("--max-meta", type=int, default=0)
    p.add_argument(
        "--resume-items",
        action="store_true",
        help="Only write items.parquet + vocab from existing train/valid/test splits "
        "(skips re-reading reviews).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reviews_path = args.data_dir / args.reviews_name
    meta_path = args.data_dir / args.meta_name

    if args.resume_items:
        log.info("resume-items: meta=%s out=%s", meta_path, args.out_dir)
        tables = resume_items_and_vocab(
            args.out_dir,
            meta_path,
            max_meta_rows=args.max_meta or None,
        )
        vocab = build_fashion_vocabulary(tables.items)
        vocab_path = args.out_dir / "vocab_fashion.json"
        vocab.save(vocab_path)
        log.info("vocabulary → %s (%d tokens)", vocab_path, len(vocab.all_tokens))
        return

    log.info("reading reviews=%s meta=%s", reviews_path, meta_path)

    reviews, items = load_raw_pair(
        str(reviews_path),
        str(meta_path),
        max_review_rows=args.max_reviews or None,
        max_meta_rows=args.max_meta or None,
    )
    log.info("raw reviews=%d items=%d", len(reviews), len(items))

    tables = standard_pipeline(
        reviews, items, min_rating=args.min_rating, min_count=args.min_count
    )
    write_processed(tables, args.out_dir)

    vocab = build_fashion_vocabulary(tables.items)
    vocab_path = args.out_dir / "vocab_fashion.json"
    vocab.save(vocab_path)
    log.info("vocabulary → %s (%d tokens)", vocab_path, len(vocab.all_tokens))


if __name__ == "__main__":
    main()
