"""GDELT 2.0 GKG source adapter (L1, Enhancement 4) — PRIMARY signal.

The reasoning for primacy:

    Editorial trade press (Vogue / WWD / Elle / Business of Fashion /
    GQ / Harper's Bazaar / etc.) is the closest source to what a
    real-life MD actually consumes when forming next-season hypotheses.
    Mass search volume (Google Trends) and cultural pageviews
    (Wikipedia) are *consequences* of editorial waves, not their cause.

Implementation notes:

* GDELT 2.0 GKG (Global Knowledge Graph) rows have a ``SourceCommonName``
  field — the publisher domain. We filter to a curated fashion-media
  allowlist and treat keyword mentions in ``ALLNAMES`` / ``THEMES`` /
  ``V2THEMES`` / ``V2ORGANIZATIONS`` / ``V2PERSONS`` as a hit.
* Granularity: GDELT publishes every 15 minutes. For trend analysis we
  aggregate to daily counts in the time-series output (a "spike day"
  is the unit MDs actually think about).
* Two input modes:
    1. Single CSV (e.g. a BigQuery export filtered to 2023 fashion-media).
       Pass ``data_path`` pointing at the file.
    2. Bulk directory of GDELT-shaped CSVs (e.g. ``YYYY/MM/*.gkg.csv``).
       Pass ``data_dir`` and the adapter walks the directory.

Either way the adapter reads the data ONCE on construction (lazy on first
``fetch``) and builds a keyword-mention index in memory so repeated
``fetch()`` calls are cheap.
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Optional

from adapters.trends.base import TrendSourceAdapter
from adapters.trends.multisource_schema import RawSourceData

log = logging.getLogger(__name__)


# Curated allowlist of fashion-trade-press domains as they appear in the
# GDELT ``SourceCommonName`` column. We keep this short and explicit so the
# editorial signal isn't drowned out by general retail news.
FASHION_MEDIA_DOMAINS: frozenset[str] = frozenset({
    "vogue.com",
    "vogue.co.uk",
    "voguebusiness.com",
    "wwd.com",
    "elle.com",
    "harpersbazaar.com",
    "gq.com",
    "businessoffashion.com",
    "highsnobiety.com",
    "hypebeast.com",
    "whowhatwear.com",
    "fashionista.com",
    "fashionunited.com",
    "fashionweekdaily.com",
    "thefashionspot.com",
    "esquire.com",
    "vanityfair.com",
    "papermag.com",
    "nylon.com",
    "popsugar.com",   # fashion vertical
    "instyle.com",
    "marieclaire.com",
    "cosmopolitan.com",
    "refinery29.com",
})


# GDELT GKG 2.0 CSV column order (per the documentation at
# https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/).
# We only read the columns we need.
_GKG_COLS = (
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
    "DocumentIdentifier", "Counts", "V2Counts", "Themes", "V2Themes",
    "Locations", "V2Locations", "Persons", "V2Persons", "Organizations",
    "V2Organizations", "V2Tone", "Dates", "GCAM", "SharingImage",
    "RelatedImages", "SocialImageEmbeds", "SocialVideoEmbeds", "Quotations",
    "AllNames", "Amounts", "TranslationInfo", "Extras",
)
_COL_IDX = {name: i for i, name in enumerate(_GKG_COLS)}


def _iter_csv_rows(path: Path) -> Iterator[list[str]]:
    """Yield CSV rows, transparently un-zipping ``.zip`` / ``.gz`` containers."""
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                with zf.open(name) as f:
                    text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                    reader = csv.reader(text, delimiter="\t")
                    yield from reader
    elif suffix.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            yield from reader
    else:
        with path.open(encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            yield from reader


def _normalise_date(raw: str) -> Optional[str]:
    """GDELT dates can be ``YYYYMMDDHHMMSS`` (GKG 2.0) or ``YYYYMMDD`` (1.0)."""
    raw = raw.strip()
    if not raw or not raw.isdigit():
        return None
    if len(raw) >= 8:
        try:
            return datetime.strptime(raw[:8], "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


class GDELTAdapter(TrendSourceAdapter):
    """Editorial fashion-media trend signal."""

    source_name = "gdelt"
    source_type = "editorial"
    reliability_prior = 0.85

    def __init__(
        self,
        *,
        data_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        domain_allowlist: Optional[Iterable[str]] = None,
        # Filter rows to those carrying at least one fashion-relevant theme.
        # Substring match against the THEMES / V2THEMES columns. Empty disables.
        theme_filter: Iterable[str] = ("FASHION", "RETAIL", "APPAREL", "LUXURY"),
        max_files: Optional[int] = None,
    ) -> None:
        if not data_path and not data_dir:
            raise ValueError("GDELTAdapter needs either data_path or data_dir")
        self.data_path = Path(data_path) if data_path else None
        self.data_dir = Path(data_dir) if data_dir else None
        self.domain_allowlist = frozenset(
            d.lower() for d in (domain_allowlist or FASHION_MEDIA_DOMAINS)
        )
        self.theme_filter = tuple(t.upper() for t in theme_filter)
        self.max_files = max_files
        # Lazy: keyword → {iso_date: count}
        self._index: Optional[dict[str, dict[str, int]]] = None

    # ------------------------------------------------------------------ #
    # Public                                                              #
    # ------------------------------------------------------------------ #

    def fetch(
        self,
        keyword: str,
        time_window: tuple[str, str],
        *,
        region: str = "US",
    ) -> Optional[RawSourceData]:
        if self._index is None:
            try:
                self._build_index()
            except Exception as e:  # noqa: BLE001
                log.warning("GDELT index build failed: %s", e)
                return None

        keyword_norm = keyword.lower().strip()
        daily_counts = (self._index or {}).get(keyword_norm, {})
        # Filter to the requested window.
        start, end = time_window
        series = [
            (iso, float(cnt))
            for iso, cnt in sorted(daily_counts.items())
            if start <= iso <= end
        ]
        return RawSourceData(
            keyword=keyword,
            source_name=self.source_name,
            source_type=self.source_type,
            time_window=time_window,
            time_series=series,
            raw_payload={
                "native_unit": "daily mentions in fashion-media allowlist",
                "allowlist_size": len(self.domain_allowlist),
            },
        )

    # ------------------------------------------------------------------ #
    # Index construction                                                  #
    # ------------------------------------------------------------------ #

    def _iter_files(self) -> Iterator[Path]:
        if self.data_path is not None:
            yield self.data_path
            return
        # Walk data_dir; pick up any .csv / .csv.zip / .csv.gz
        assert self.data_dir is not None
        for ext in ("*.csv", "*.csv.zip", "*.csv.gz", "*.gkg.csv", "*.gkg.csv.zip"):
            yield from sorted(self.data_dir.rglob(ext))

    def _row_passes_filters(self, row: list[str]) -> bool:
        if len(row) <= max(_COL_IDX["SourceCommonName"], _COL_IDX["Themes"]):
            return False
        source = row[_COL_IDX["SourceCommonName"]].strip().lower()
        if source not in self.domain_allowlist:
            return False
        if self.theme_filter:
            themes_blob = (
                row[_COL_IDX["Themes"]] + ";" + row[_COL_IDX["V2Themes"]]
            ).upper()
            if not any(t in themes_blob for t in self.theme_filter):
                return False
        return True

    def _extract_text_for_mention_match(self, row: list[str]) -> str:
        """Concatenate the text-bearing columns we scan for keyword mentions."""
        parts = []
        for col in ("AllNames", "Persons", "V2Persons", "Organizations",
                    "V2Organizations", "Themes", "V2Themes"):
            idx = _COL_IDX[col]
            if idx < len(row):
                parts.append(row[idx])
        return " | ".join(parts).lower()

    def _build_index(self) -> None:
        index: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        rows_scanned = 0
        rows_kept = 0
        files_scanned = 0
        for path in self._iter_files():
            if self.max_files is not None and files_scanned >= self.max_files:
                break
            files_scanned += 1
            try:
                for row in _iter_csv_rows(path):
                    rows_scanned += 1
                    if not row or len(row) < len(_GKG_COLS) - 4:
                        # Trailing columns can be missing in some snapshots; tolerate.
                        continue
                    if not self._row_passes_filters(row):
                        continue
                    iso = _normalise_date(row[_COL_IDX["DATE"]])
                    if not iso:
                        continue
                    blob = self._extract_text_for_mention_match(row)
                    if not blob:
                        continue
                    rows_kept += 1
                    # Build an inverted index lazily: we don't know the
                    # keyword pool yet, so store the text-blob → date mapping
                    # in a way that lets fetch() ask for any keyword cheaply.
                    # Strategy: tokenise the blob into terms and bigrams.
                    for token in _tokenise(blob):
                        index[token][iso] += 1
            except Exception as e:  # noqa: BLE001
                log.warning("GDELT file %s failed: %s", path, e)
                continue
        self._index = {k: dict(v) for k, v in index.items()}
        log.info(
            "GDELT indexed %d files / %d rows scanned / %d kept → %d distinct terms",
            files_scanned, rows_scanned, rows_kept, len(self._index),
        )


_TOKEN_RE = None  # compiled lazily to keep module import light


def _tokenise(text: str) -> list[str]:
    """Lowercase unigrams + bigrams from a GDELT text blob.

    GDELT fields are semi-structured (entity lists separated by ``;`` and
    ``,``). We split on these plus whitespace, then form 2-grams for
    multi-word keywords like "low rise" or "y2k revival".
    """
    global _TOKEN_RE
    if _TOKEN_RE is None:
        import re
        _TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
    tokens = _TOKEN_RE.findall(text.lower())
    out = list(tokens)
    for i in range(len(tokens) - 1):
        out.append(f"{tokens[i]} {tokens[i + 1]}")
    return out
