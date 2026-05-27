"""Build the fashion-trend keyword pool used to query Google Trends.

Three sources, in order:

(a) Catalog-driven       — high-frequency tail categories + departments + brands.
(b) LLM seed-expansion   — Qwen / GPT expands base seeds into trend variants
                           that the catalog may not yet contain.
(c) Rising-query feedback — Google Trends 'breakout' queries discovered from
                           the seeds; these are the closest signal to a *real*
                           trend.

The final pool is deduplicated and capped at ``max_keywords``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Iterable, Optional

import pandas as pd
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

_FASHION_STOPWORDS = {
    "clothing", "shoes", "jewelry", "women", "men", "girls", "boys", "kids",
    "novelty", "&", "and", "the", "for", "with", "more", "fashion",
}

_TAIL_OK = re.compile(r"^[a-zA-Z][a-zA-Z\-\s]{2,}$")


def _flatten_categories(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    flat: list[str] = []
    try:
        for item in value:
            if isinstance(item, list):
                flat.extend([str(x) for x in item if x])
            elif item:
                flat.append(str(item))
    except TypeError:
        return []
    return flat


# --------------------------------------------------------------------------- #
# (a) Catalog                                                                  #
# --------------------------------------------------------------------------- #


def extract_catalog_keywords(items_df: pd.DataFrame, top_n: int = 30) -> list[str]:
    """Pull the most frequent *tail* category terms from the catalogue.

    We deliberately keep only the last 1–2 segments of each Amazon category
    path because the leading segments are taxonomic boilerplate
    ("Clothing, Shoes & Jewelry / Women / Clothing").
    """
    counter: Counter[str] = Counter()
    for v in items_df.get("categories", []):
        parts = _flatten_categories(v)
        if not parts:
            continue
        # Take last two segments — that's where the fashion-meaningful terms live.
        for token in parts[-2:]:
            term = token.strip().lower()
            if not _TAIL_OK.match(term):
                continue
            if term in _FASHION_STOPWORDS:
                continue
            if len(term) > 30:
                continue
            counter[term] += 1

    # Drop coarse "umbrella" terms that always appear in the path.
    for noise in ("clothing", "shoes", "jewelry", "accessories"):
        counter.pop(noise, None)

    return [kw for kw, _ in counter.most_common(top_n)]


# --------------------------------------------------------------------------- #
# (b) LLM seed-expansion                                                       #
# --------------------------------------------------------------------------- #


class _SeedExpansion(BaseModel):
    keywords: list[str] = Field(default_factory=list)


def expand_seeds_with_llm(
    llm,  # sage.llm.client.LLMClient
    seeds: list[str],
    *,
    domain: str = "fashion",
    time_window: str = "2023 SS",
    region: str = "US",
    n_per_seed: int = 2,
    max_total: int = 30,
) -> list[str]:
    """Ask the LLM for trend-variant keywords for each seed.

    Returns a deduplicated list of variants the model believes are
    actually trending in the given window. We deliberately ask for short
    search-engine-friendly phrases (2–4 words).
    """
    if not seeds:
        return []

    prompt = (
        f"You are a {domain} trend analyst working with {region} consumers in {time_window}. "
        f"For each seed below, propose {n_per_seed} short search-engine-friendly "
        "keywords (2–4 words each) that are realistically *rising* trend variants. "
        "Stay close to the seed's category — do not invent unrelated terms. "
        "Use lowercase. Avoid brand names. Avoid duplicates. "
        "Return ONLY a JSON object {\"keywords\": [\"…\", \"…\"]}.\n\n"
        f"Seeds: {seeds}"
    )
    try:
        out = llm.complete_structured(
            prompt,
            _SeedExpansion,
            system="You output strict JSON. No prose.",
        )
    except Exception as e:
        log.warning("LLM seed expansion failed: %s — skipping", e)
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for kw in out.keywords:
        term = re.sub(r"\s+", " ", str(kw).strip().lower())
        if not term or term in seen:
            continue
        if len(term.split()) > 5 or len(term) > 40:
            continue
        seen.add(term)
        cleaned.append(term)
        if len(cleaned) >= max_total:
            break
    return cleaned


# --------------------------------------------------------------------------- #
# (c) Rising-query feedback                                                    #
# --------------------------------------------------------------------------- #


def harvest_rising_keywords(
    gtc,                            # GoogleTrendsClient
    seeds: list[str],
    *,
    timeframe: str,
    geo: str = "US",
    top_per_seed: int = 5,
    max_total: int = 20,
) -> tuple[list[str], list]:
    """Return (new_keywords, raw_rising_records).

    ``new_keywords`` is a deduplicated list ready to feed into the next
    ``fetch_interest_over_time`` pass. ``raw_rising_records`` is the full
    pytrends payload (kept for the snapshot's ``rising_queries`` field).
    """
    if not seeds:
        return [], []

    records = gtc.fetch_rising_queries(seeds, timeframe=timeframe, geo=geo, top_per_seed=top_per_seed)

    cleaned: list[str] = []
    seen: set[str] = set()
    for r in records:
        term = re.sub(r"\s+", " ", r.query.strip().lower())
        # Filter junk: too long, off-topic literal Breakout phrases.
        if len(term.split()) > 6 or len(term) > 50:
            continue
        if term in seen:
            continue
        seen.add(term)
        cleaned.append(term)
        if len(cleaned) >= max_total:
            break
    return cleaned, records


# --------------------------------------------------------------------------- #
# Curated fashion trend lexicon (Enhancement 4 — L5 priming set)              #
# --------------------------------------------------------------------------- #
#
# Hand-picked, contemporary trend-bearing terms that catalog tail-categories
# usually miss. Editorial trade press tracks these explicitly; including them
# in the pool is what surfaces *micro-trend* signal (rising/declining) in the
# snapshot. Sourced from 2022-2023 Vogue / WWD / BoF / Highsnobiety coverage.

FASHION_TREND_LEXICON: list[str] = [
    # Aesthetics / cores (2022-2023 wave)
    "y2k", "y2k revival", "quiet luxury", "old money", "stealth wealth",
    "coastal grandmother", "tomato girl", "mob wife", "blokecore", "balletcore",
    "gorpcore", "cottagecore", "dark academia", "preppy", "soft girl",
    "indie sleaze", "regencycore", "office siren", "clean girl",
    # Silhouettes / cuts
    "low rise", "high waist", "wide leg", "cargo pants", "parachute pants",
    "barrel jeans", "baggy jeans", "skinny jeans", "boyfriend jeans",
    "corset", "bustier", "crop top", "tube top", "mini skirt", "maxi skirt",
    "midi dress", "slip dress", "tea length", "bodycon", "bias cut",
    # Outerwear
    "oversized blazer", "tailored blazer", "trench coat", "puffer jacket",
    "barn jacket", "varsity jacket", "leather jacket",
    # Materials / details
    "satin", "velvet", "denim", "tweed", "knit", "crochet", "mesh", "sheer",
    "metallic", "chrome", "lace", "linen", "shearling", "faux fur",
    # Footwear
    "ballet flats", "kitten heels", "platform shoes", "chunky sneakers",
    "dad sneakers", "samba", "loafers", "mary janes", "knee high boots",
    "cowboy boots", "uggs", "flip flops",
    # Accessories
    "shoulder bag", "tote bag", "micro bag", "baguette bag", "bucket hat",
    "claw clip", "statement necklace", "stack rings", "pearl necklace",
    "chunky gold",
    # Brands / motifs in the trade press
    "miu miu", "the row", "alaia", "khaite", "loewe", "bottega", "prada",
    "gucci", "balenciaga", "celine", "saint laurent", "chanel",
    "supreme", "stussy", "carhartt", "patagonia", "uniqlo", "zara",
    # Movements
    "slow fashion", "secondhand", "vintage", "archive", "deadstock",
    "upcycle", "circular fashion",
    # Tech / cross-over
    "smart watch", "techwear", "athleisure",
]


class _BriefKeywords(BaseModel):
    keywords: list[str] = Field(default_factory=list)


def extract_brief_keywords(
    llm,
    brief_text: str,
    *,
    max_keywords: int = 20,
    domain: str = "fashion",
) -> list[str]:
    """LLM-extract fashion-relevant keywords from an operator brief.

    The brief is short free-form text from the operator (e.g. "Casual to
    formal upsell, boost outerwear, lean into Y2K revival"). We want the
    concrete trend-bearing terms — not abstract marketing phrases.

    Returns lower-cased, deduplicated, length-capped phrases. Empty list
    on failure (brief still works without brief-derived keywords).
    """
    if not brief_text or not brief_text.strip():
        return []

    prompt = (
        f"You are a {domain} trend analyst. Extract concrete trend-bearing "
        "keywords from the operator brief below.\n\n"
        f"Brief: {brief_text}\n\n"
        "Rules:\n"
        "- Lowercase, 2-4 words each, search-engine-friendly.\n"
        "- Prefer concrete fashion terms (silhouettes, materials, eras, "
        "styles, brands) over abstract marketing language.\n"
        "- Drop generic words like 'trendy' or 'fresh'.\n"
        f"- Cap at {max_keywords} keywords.\n"
        "Return ONLY {\"keywords\": [...]}.\n"
    )
    try:
        out = llm.complete_structured(
            prompt,
            _BriefKeywords,
            system="You output strict JSON. No prose.",
            agent_id="trend-brief-keywords",
        )
    except Exception as e:
        log.warning("brief keyword extraction failed: %s", e)
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for kw in out.keywords:
        term = re.sub(r"\s+", " ", str(kw).strip().lower())
        if not term or term in seen:
            continue
        if len(term.split()) > 5 or len(term) > 40:
            continue
        seen.add(term)
        cleaned.append(term)
        if len(cleaned) >= max_keywords:
            break
    return cleaned


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #


class KeywordPool(BaseModel):
    catalog: list[str] = Field(default_factory=list)
    llm_seed: list[str] = Field(default_factory=list)
    rising_feedback: list[str] = Field(default_factory=list)
    # Enhancement 4: operator brief drives part of the pool. Stays empty
    # for the legacy single-source pipeline.
    brief_derived: list[str] = Field(default_factory=list)

    @property
    def all(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        # Brief-derived keywords come first — they capture operator intent
        # which is the most directly relevant signal for downstream Phase 2.
        for src in (self.brief_derived, self.catalog, self.llm_seed, self.rising_feedback):
            for kw in src:
                if kw not in seen:
                    seen.add(kw)
                    out.append(kw)
        return out

    def source_lookup(self) -> dict[str, str]:
        """Map keyword → its origin source."""
        out: dict[str, str] = {}
        for kw in self.brief_derived:
            out.setdefault(kw, "brief_derived")
        for kw in self.catalog:
            out.setdefault(kw, "catalog")
        for kw in self.llm_seed:
            out.setdefault(kw, "llm_seed")
        for kw in self.rising_feedback:
            out.setdefault(kw, "rising_feedback")
        return out


def build_keyword_pool(
    items_df: pd.DataFrame,
    *,
    llm: Optional[object] = None,
    gtc: Optional[object] = None,
    timeframe: str = "2023-03-01 2023-05-31",
    geo: str = "US",
    n_catalog: int = 25,
    n_llm_per_seed: int = 2,
    n_llm_seeds: int = 8,
    n_rising_seeds: int = 6,
    n_rising_per_seed: int = 4,
    max_keywords: int = 50,
) -> KeywordPool:
    """End-to-end keyword pool builder. See module docstring."""
    catalog_kws = extract_catalog_keywords(items_df, top_n=n_catalog)
    log.info("(a) catalog keywords: %d -> %s", len(catalog_kws), catalog_kws[:8])

    llm_kws: list[str] = []
    if llm is not None and catalog_kws:
        seeds = catalog_kws[:n_llm_seeds]
        llm_kws = expand_seeds_with_llm(
            llm, seeds, n_per_seed=n_llm_per_seed, max_total=n_llm_seeds * n_llm_per_seed
        )
        log.info("(b) LLM-expanded keywords: %d -> %s", len(llm_kws), llm_kws[:8])

    rising_kws: list[str] = []
    if gtc is not None and catalog_kws:
        seeds = catalog_kws[:n_rising_seeds]
        rising_kws, _ = harvest_rising_keywords(
            gtc, seeds, timeframe=timeframe, geo=geo,
            top_per_seed=n_rising_per_seed,
            max_total=n_rising_seeds * n_rising_per_seed,
        )
        log.info("(c) rising-feedback keywords: %d -> %s", len(rising_kws), rising_kws[:8])

    pool = KeywordPool(catalog=catalog_kws, llm_seed=llm_kws, rising_feedback=rising_kws)
    # Cap while preserving source-order priority.
    capped: list[str] = pool.all[:max_keywords]
    cap_set = set(capped)
    pool.catalog = [k for k in pool.catalog if k in cap_set]
    pool.llm_seed = [k for k in pool.llm_seed if k in cap_set]
    pool.rising_feedback = [k for k in pool.rising_feedback if k in cap_set]
    return pool


# --------------------------------------------------------------------------- #
# Category mapping                                                             #
# --------------------------------------------------------------------------- #


def map_keywords_to_categories(
    keywords: Iterable[str],
    items_df: pd.DataFrame,
    *,
    min_matches: int = 3,
) -> dict[str, str]:
    """Map a keyword to its dominant catalogue category (lowercased tail).

    The matching is intentionally simple: a keyword maps to the most
    frequent tail-category among items whose title contains the keyword as
    a substring (case-insensitive). Categories with fewer than
    ``min_matches`` matches are dropped.

    Returns ``{keyword: category}`` for keywords with a confident mapping.
    """
    titles = items_df.get("title", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
    cat_seq = items_df.get("categories")
    if cat_seq is None:
        return {}

    cat_tail: list[str] = []
    for v in cat_seq:
        parts = _flatten_categories(v)
        cat_tail.append(parts[-1].lower() if parts else "")

    cat_series = pd.Series(cat_tail, index=items_df.index)
    out: dict[str, str] = {}
    for kw in keywords:
        kw_l = kw.lower()
        mask = titles.str.contains(re.escape(kw_l), regex=True, na=False)
        if not mask.any():
            continue
        counts = cat_series[mask].value_counts()
        if counts.empty:
            continue
        top_cat, top_count = counts.index[0], int(counts.iloc[0])
        if top_count < min_matches or not top_cat:
            continue
        out[kw] = top_cat
    return out
