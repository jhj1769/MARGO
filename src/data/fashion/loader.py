"""Loader & preprocessing utilities for Amazon Reviews 2023 (Clothing/Shoes/Jewelry).

Schema reference: https://amazon-reviews-2023.github.io/
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Iteration                                                                    #
# --------------------------------------------------------------------------- #


def _open_jsonl(path: Path) -> io.TextIOBase:
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, encoding="utf-8")


def iter_review_rows(
    path: str | Path,
    max_rows: Optional[int] = None,
    *,
    log_every: int = 1_000_000,
    label: str = "rows",
) -> Iterator[dict]:
    """Yield review rows. Stops at ``max_rows`` for fast iteration.

    Emits a log line every ``log_every`` rows so long-running preprocess jobs
    are observable from a tail-able file.
    """
    with _open_jsonl(Path(path)) as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                return
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
            if log_every and (i + 1) % log_every == 0:
                log.info("read %d %s from %s", i + 1, label, Path(path).name)


def iter_meta_rows(path: str | Path, max_rows: Optional[int] = None) -> Iterator[dict]:
    """Yield item metadata rows."""
    yield from iter_review_rows(path, max_rows=max_rows, label="meta rows")


# --------------------------------------------------------------------------- #
# Containers                                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class InteractionTables:
    """Output of ``leave_one_out_split``."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    items: pd.DataFrame


# --------------------------------------------------------------------------- #
# Preprocessing pipeline                                                       #
# --------------------------------------------------------------------------- #


def filter_5_core(df: pd.DataFrame, user_col: str, item_col: str, min_count: int = 5) -> pd.DataFrame:
    """Iteratively drop users/items below the activity threshold."""
    prev = -1
    iteration = 0
    while len(df) != prev:
        iteration += 1
        prev = len(df)
        df = df[df.groupby(user_col)[item_col].transform("count") >= min_count]
        df = df[df.groupby(item_col)[user_col].transform("count") >= min_count]
        log.info(
            "5-core iter %d: %d → %d rows (users=%d, items=%d)",
            iteration, prev, len(df),
            df[user_col].nunique(), df[item_col].nunique(),
        )
    return df.reset_index(drop=True)


def leave_one_out_split(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Most recent per user → test; second most recent → valid; rest → train."""
    if "timestamp" not in interactions.columns:
        raise ValueError("leave_one_out_split expects a 'timestamp' column")
    df = interactions.sort_values(["user_id", "timestamp"]).copy()
    df["_rank"] = df.groupby("user_id").cumcount(ascending=False)
    test = df[df["_rank"] == 0].drop(columns="_rank")
    valid = df[df["_rank"] == 1].drop(columns="_rank")
    train = df[df["_rank"] >= 2].drop(columns="_rank")
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


_PLACEHOLDER_PRICES = frozenset({"", "—", "-", "–", "n/a", "na", "none", "null", "unknown"})


def coerce_price(value) -> float | None:
    """Normalise Amazon meta ``price`` to a float or ``None``.

    Raw values include currency strings (``"$29.99"``), em/en dashes (``"—"``),
    and occasional free-text placeholders that break Parquet float columns.
    """
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(f) else f
    s = str(value).strip()
    if s.lower() in _PLACEHOLDER_PRICES:
        return None
    cleaned = re.sub(r"[^\d.]", "", s)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def sanitize_items_frame(items: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``items`` is Parquet-safe (notably the ``price`` column)."""
    if items.empty or "price" not in items.columns:
        return items
    out = items.copy()
    out["price"] = out["price"].map(coerce_price)
    return out


def _is_missing(value) -> bool:
    """Truth-test safe for None, NaN, empty ndarray/list."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in ("nan", "none")
    return False


def _as_text(value) -> str:
    """Coerce parquet cell values (list/ndarray/NaN) to a single string."""
    if _is_missing(value):
        return ""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _as_text(value.item())
        return " ".join(_as_text(x) for x in value.tolist())
    if isinstance(value, (list, tuple)):
        return " ".join(_as_text(x) for x in value)
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none") else text


def _flatten_categories(value) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        flat: list[str] = []
        for c in value:
            if isinstance(c, (list, tuple, np.ndarray)):
                seq = c.tolist() if isinstance(c, np.ndarray) else c
                flat.extend(_as_text(x) for x in seq if not _is_missing(x))
            elif not _is_missing(c):
                flat.append(_as_text(c))
        return ", ".join(flat)
    return _as_text(value)


def build_item_text(row: dict | pd.Series) -> str:
    """Compose the BGE-M3 input string for a single item."""
    title = _as_text(row.get("title"))
    description = _as_text(row.get("description"))
    cat_text = _flatten_categories(row.get("categories"))

    details = row.get("details")
    details_dict = details if isinstance(details, dict) and not _is_missing(details) else {}
    color = _as_text(details_dict.get("Color")) or _as_text(row.get("color"))
    material = _as_text(details_dict.get("Material")) or _as_text(row.get("material"))

    price = coerce_price(row.get("price"))
    price_text = f"${price:.2f}" if price is not None else "n/a"

    parts = [
        title,
        description,
        f"Category: {cat_text}" if cat_text else "",
        f"Color: {color}" if color else "",
        f"Material: {material}" if material else "",
        f"Price: {price_text}",
    ]
    return ". ".join(p for p in parts if p and p != "n/a")


# --------------------------------------------------------------------------- #
# I/O                                                                          #
# --------------------------------------------------------------------------- #


def write_processed(
    tables: InteractionTables,
    out_dir: str | Path,
    *,
    skip_interactions: bool = False,
) -> None:
    """Persist processed tables. Set ``skip_interactions`` when splits already exist."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = sanitize_items_frame(tables.items)
    if not skip_interactions:
        tables.train.to_parquet(out / "train.parquet")
        tables.valid.to_parquet(out / "valid.parquet")
        tables.test.to_parquet(out / "test.parquet")
    items.to_parquet(out / "items.parquet")
    log.info(
        "wrote %s interactions and %d items to %s",
        "skipped" if skip_interactions else f"{len(tables.train)}/{len(tables.valid)}/{len(tables.test)}",
        len(items),
        out,
    )


def load_meta_table(meta_path: str | Path, *, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load item metadata only (no reviews)."""
    log.info("loading meta from %s (max=%s)", meta_path, max_rows)
    meta_records = [normalise_meta(r) for r in iter_meta_rows(meta_path, max_rows=max_rows)]
    log.info("building items DataFrame (%d records)", len(meta_records))
    items = pd.DataFrame.from_records(meta_records)
    return sanitize_items_frame(items)


def resume_items_and_vocab(
    out_dir: str | Path,
    meta_path: str | Path,
    *,
    max_meta_rows: Optional[int] = None,
) -> InteractionTables:
    """Finish a preprocess run that wrote splits but failed on ``items.parquet``.

    Reads existing ``train/valid/test.parquet``, filters meta to catalogue IDs
    present in the splits, then writes ``items.parquet``.
    """
    out = Path(out_dir)
    for name in ("train.parquet", "valid.parquet", "test.parquet"):
        if not (out / name).exists():
            raise FileNotFoundError(f"resume requires {out / name}")

    train = pd.read_parquet(out / "train.parquet")
    valid = pd.read_parquet(out / "valid.parquet")
    test = pd.read_parquet(out / "test.parquet")
    used_ids = pd.Index(
        pd.concat([train["item_id"], valid["item_id"], test["item_id"]]).astype(str).unique()
    )
    log.info("resume: %d unique item_ids from existing splits", len(used_ids))

    catalog = load_meta_table(meta_path, max_rows=max_meta_rows)
    items_used = catalog[catalog["parent_asin"].astype(str).isin(used_ids)].reset_index(drop=True)
    missing = len(used_ids) - items_used["parent_asin"].nunique()
    if missing:
        log.warning("resume: %d item_ids in splits have no meta row", missing)

    tables = InteractionTables(train=train, valid=valid, test=test, items=items_used)
    write_processed(tables, out, skip_interactions=True)
    return tables


def load_processed(in_dir: str | Path) -> InteractionTables:
    in_dir = Path(in_dir)
    return InteractionTables(
        train=pd.read_parquet(in_dir / "train.parquet"),
        valid=pd.read_parquet(in_dir / "valid.parquet"),
        test=pd.read_parquet(in_dir / "test.parquet"),
        items=pd.read_parquet(in_dir / "items.parquet"),
    )


# --------------------------------------------------------------------------- #
# Helpers used by tests / dry-runs                                             #
# --------------------------------------------------------------------------- #


def normalise_review(row: dict) -> dict:
    """Pluck the columns we actually need from one raw review row."""
    return {
        "user_id": row.get("user_id"),
        "item_id": row.get("parent_asin") or row.get("asin"),
        "rating": float(row.get("rating", 0) or 0),
        "timestamp": int(row.get("timestamp", 0) or 0),
    }


def _first_image_url(images) -> str | None:
    """Extract a usable image URL from the Amazon Reviews 2023 `images` field.

    The raw schema is::

        images = [{"thumb": str, "large": str, "variant": "MAIN"|..., "hi_res": str?}, ...]

    We prefer the MAIN variant's large image, fall back to the first row's
    large/hi_res/thumb in that order.
    """
    if not images:
        return None
    if not isinstance(images, list):
        return None
    main = next((im for im in images if isinstance(im, dict) and im.get("variant") == "MAIN"), None)
    candidate = main or images[0]
    if not isinstance(candidate, dict):
        return None
    for key in ("large", "hi_res", "thumb"):
        url = candidate.get(key)
        if url:
            return url
    return None


def normalise_meta(row: dict) -> dict:
    """Pluck the columns we actually need from one raw metadata row."""
    return {
        "parent_asin": row.get("parent_asin"),
        "title": row.get("title"),
        "description": row.get("description"),
        "categories": row.get("categories"),
        "details": row.get("details"),
        "price": coerce_price(row.get("price")),
        "brand": row.get("store") or row.get("brand"),
        "image_url": _first_image_url(row.get("images")),
    }


def load_raw_pair(
    reviews_path: str,
    meta_path: str,
    *,
    max_review_rows: Optional[int] = None,
    max_meta_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream both Amazon-Reviews JSONL files into pandas DataFrames."""
    log.info("loading reviews from %s (max=%s)", reviews_path, max_review_rows)
    review_records = [
        normalise_review(r)
        for r in iter_review_rows(reviews_path, max_rows=max_review_rows, label="reviews")
    ]
    log.info("building reviews DataFrame (%d records)", len(review_records))
    reviews = pd.DataFrame.from_records(review_records)
    del review_records

    log.info("loading meta from %s (max=%s)", meta_path, max_meta_rows)
    meta_records = [
        normalise_meta(r)
        for r in iter_meta_rows(meta_path, max_rows=max_meta_rows)
    ]
    log.info("building items DataFrame (%d records)", len(meta_records))
    items = pd.DataFrame.from_records(meta_records)
    del meta_records

    log.info("load_raw_pair done: reviews=%d items=%d", len(reviews), len(items))
    return reviews, items


def standard_pipeline(
    reviews: pd.DataFrame,
    items: pd.DataFrame,
    *,
    min_rating: float = 4.0,
    min_count: int = 5,
) -> InteractionTables:
    """Implement Steps 1.2 of the implementation plan."""
    log.info("filtering rating >= %.1f (input rows=%d)", min_rating, len(reviews))
    pos = reviews[reviews["rating"] >= min_rating].copy()
    pos = pos.dropna(subset=["user_id", "item_id"])
    log.info("after rating+dropna: %d positive interactions", len(pos))

    pos = pos[pos["item_id"].isin(items["parent_asin"])]
    log.info("after item-meta join: %d interactions (catalog=%d)", len(pos), len(items))

    log.info("running 5-core filter (min_count=%d)", min_count)
    filt = filter_5_core(pos, "user_id", "item_id", min_count=min_count)
    log.info(
        "5-core done: %d interactions, %d users, %d items",
        len(filt), filt["user_id"].nunique(), filt["item_id"].nunique(),
    )

    log.info("leave-one-out split")
    train, valid, test = leave_one_out_split(filt)
    log.info("split: train=%d valid=%d test=%d", len(train), len(valid), len(test))

    used_items = filt["item_id"].unique()
    items_used = sanitize_items_frame(
        items[items["parent_asin"].isin(used_items)].reset_index(drop=True)
    )
    log.info("items_used: %d", len(items_used))
    return InteractionTables(train=train, valid=valid, test=test, items=items_used)
