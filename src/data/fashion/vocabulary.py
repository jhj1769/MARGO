"""Extract a MARGO :class:`Vocabulary` from a fashion item table.

The Amazon Reviews 2023 meta file exposes a ``categories`` list plus a
``details`` dict (Color, Material, Department …). We map those onto the
canonical MARGO buckets (``category``, ``color``, ``material``, …) and add a
small set of well-known fashion silhouettes that the metadata frequently
omits.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from core.validation.vocabulary import Vocabulary


# Canonical silhouettes we always want available to the Expert / Trend agents,
# even if the catalogue misses them.
_BASE_SILHOUETTES = {
    "trench-coat", "blazer", "chinos", "midi-dress", "mini-dress", "maxi-dress",
    "puffer", "parka", "oxford", "loafers", "sneakers", "boots", "heels",
    "knit-sweater", "cardigan", "hoodie", "polo", "tee-shirt", "denim-jacket",
    "tailored-pants", "wide-leg-pants", "skirt", "shorts", "jumpsuit",
}


def _flatten_categories(value) -> list[str]:
    """Amazon metadata's ``categories`` field is a list of lists/strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    flat: list[str] = []
    for item in value:
        if isinstance(item, list):
            flat.extend([str(x) for x in item if x])
        elif item:
            flat.append(str(item))
    return flat


def _detail_value(row_details, key: str) -> str | None:
    if not isinstance(row_details, dict):
        return None
    for k in (key, key.lower(), key.upper(), key.capitalize()):
        v = row_details.get(k)
        if v:
            return str(v)
    return None


def build_fashion_vocabulary(items_df: pd.DataFrame) -> Vocabulary:
    """Construct a fashion-domain :class:`Vocabulary` from an items table.

    Expected columns (any subset works):
        - ``categories``  : list of strings or lists of strings
        - ``details``     : dict (Color / Material / Department / ...)
        - ``brand``       : str
    """
    buckets: dict[str, set[str]] = {
        "category": set(),
        "color": set(),
        "material": set(),
        "department": set(),
        "brand": set(),
        "silhouette": set(_BASE_SILHOUETTES),
    }

    if "categories" in items_df.columns:
        for v in items_df["categories"]:
            for cat in _flatten_categories(v):
                buckets["category"].add(cat.lower())

    if "details" in items_df.columns:
        for d in items_df["details"]:
            for key, bucket in (
                ("Color", "color"),
                ("Material", "material"),
                ("Fabric Type", "material"),
                ("Department", "department"),
                ("Brand", "brand"),
            ):
                val = _detail_value(d, key)
                if val:
                    buckets[bucket].add(val.lower())

    if "brand" in items_df.columns:
        for b in items_df["brand"].dropna().astype(str):
            buckets["brand"].add(b.lower())

    return Vocabulary(buckets)


def build_attribute_table(items_df: pd.DataFrame) -> dict[str, dict]:
    """Compact attribute lookup used by ExpertAgent structured validation."""
    out: dict[str, dict] = {}
    for _, row in items_df.iterrows():
        attrs = {
            "category": _flatten_categories(row.get("categories")),
            "color": _detail_value(row.get("details"), "Color"),
            "material": _detail_value(row.get("details"), "Material"),
            "price": _coerce_float(row.get("price")),
            "brand": row.get("brand"),
            # Image URL is not used by validation, but the web demo pulls it
            # from this same dict — keep it next to the rest of the metadata.
            "image_url": row.get("image_url"),
            "title": row.get("title"),
        }
        out[str(row.get("parent_asin") or row.get("item_id"))] = attrs
    return out


def _coerce_float(v) -> float | None:
    try:
        if v is None or v != v:  # NaN
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def iter_known_categories(buckets: Iterable[str]) -> list[str]:
    """Sugar helper for prompt rendering — flatten a single bucket to sorted list."""
    return sorted(set(buckets))
