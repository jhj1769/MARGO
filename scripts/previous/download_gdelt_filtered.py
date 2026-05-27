"""Download + filter GDELT 2.0 GKG files on-the-fly (Enhancement 4).

GDELT publishes a 15-minute granularity GKG file. A full year of 2023 is
~35,000 files and ~150-300 GB raw. We don't need most of it — we only care
about rows whose ``SourceCommonName`` is in our fashion-media allowlist
(Vogue / WWD / Elle / BoF / GQ / Harper's Bazaar / etc).

Strategy:
    1. Fetch the 15-min file in memory.
    2. Unzip in memory.
    3. Filter rows to the allowlist (and to fashion themes).
    4. Append the kept rows to a single output CSV per month.
    5. Discard the raw download.

Net result: ~150 GB raw → ~few GB filtered.

Usage::

    # One-week smoke test
    python -m scripts.download_gdelt_filtered \\
        --out-dir data/external_trends/gdelt \\
        --start 2023-06-01 --end 2023-06-07 \\
        --parallel 4

    # Full year 2023
    python -m scripts.download_gdelt_filtered \\
        --out-dir data/external_trends/gdelt \\
        --start 2023-01-01 --end 2023-12-31 \\
        --parallel 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import logging
import re
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

import requests

# GDELT GKG rows can carry very long V2Themes / V2Persons strings (one row
# we hit was > 128 KB on the THEMES column alone). Lift the default
# csv field limit so the reader doesn't choke on legitimate large rows.
csv.field_size_limit(sys.maxsize)

from adapters.trends.gdelt_adapter import (
    FASHION_MEDIA_DOMAINS,
    _COL_IDX,
    _GKG_COLS,
)

log = logging.getLogger(__name__)


_GDELT_BASE = "http://data.gdeltproject.org/gdeltv2"
# Each timestamp is YYYYMMDDHHMMSS at the start of a 15-min interval.
_INTERVAL_MINUTES = 15


# Domain tiering — each tier has a different keep policy.
#
# Tier 1: pure fashion / industry trade press. Almost every article is
#   fashion-relevant by construction. Keep everything.
PURE_FASHION = frozenset({
    "vogue.com", "vogue.co.uk", "voguebusiness.com",
    "wwd.com",
    "businessoffashion.com",
    "fashionunited.com", "fashionweekdaily.com", "fashionista.com",
    "thefashionspot.com",
    "whowhatwear.com",
})

# Tier 2: streetwear / sneaker / hype-culture press. Mostly drops & releases;
#   little political/celebrity noise. Keep everything.
STREETWEAR = frozenset({
    "hypebeast.com", "highsnobiety.com",
})

# Tier 3: women's/men's-lifestyle magazines with strong fashion vertical but
#   also entertainment, beauty, dating, etc. Restrict to fashion-relevant
#   URL segments.
MIXED_LIFESTYLE = frozenset({
    "elle.com", "harpersbazaar.com", "cosmopolitan.com",
    "marieclaire.com", "instyle.com", "nylon.com", "popsugar.com",
    "papermag.com", "refinery29.com",
})

# Tier 4: men's general-lifestyle. Fashion is a tiny vertical. Restrict to
#   fashion / style / grooming / watches.
MENS_LIFESTYLE = frozenset({
    "gq.com", "esquire.com",
})

# Tier 5: general culture mags occasionally covering fashion. Strict.
CULTURE_MAGS = frozenset({
    "vanityfair.com",
})


# Positive URL segments — at least one match required for tier-3/4/5.
_FASHION_PATH_RE = re.compile(
    r"/(?:fashion|style|beauty|shopping|outfit|runway|designer"
    r"|clothing|accessor|sneaker|sneakers|jewelry|jewellery|watches"
    r"|streetwear|grooming|fragrance|denim|trench|hoodie)/",
    re.IGNORECASE,
)


def keep_url(domain: str, url: str) -> bool:
    """Apply the per-tier keep policy. ``True`` means the row is kept."""
    domain = (domain or "").lower()
    url = url or ""
    if domain in PURE_FASHION or domain in STREETWEAR:
        return True
    if domain in MIXED_LIFESTYLE or domain in MENS_LIFESTYLE or domain in CULTURE_MAGS:
        return bool(_FASHION_PATH_RE.search(url))
    # Unknown domain: be conservative — caller's allowlist already passed,
    # so default keep is safe.
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Where filtered monthly CSVs are written.")
    p.add_argument("--start", required=True,
                   help="ISO date (inclusive), e.g. 2023-01-01")
    p.add_argument("--end", required=True,
                   help="ISO date (inclusive), e.g. 2023-12-31")
    p.add_argument("--parallel", type=int, default=4,
                   help="Concurrent downloads (raise carefully; GDELT is unhappy with > 8).")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--theme-filter", nargs="*",
                   default=[],
                   help="Substrings to require in THEMES/V2THEMES (case-insensitive). "
                        "Disabled by default — too strict for editorial signal.")
    p.add_argument("--no-drop-non-fashion-paths", action="store_true",
                   help="Disable the URL-path filter that drops politics / "
                        "entertainment / sports / celebrity-gossip articles "
                        "from fashion-media domains. By default these are dropped.")
    p.add_argument("--max-files", type=int, default=None,
                   help="Cap on total files downloaded — for debugging.")
    return p.parse_args()


def _iter_timestamps(start_iso: str, end_iso: str) -> Iterator[str]:
    """Yield YYYYMMDDHHMMSS strings every 15 min between start and end (inclusive end day)."""
    start = datetime.strptime(start_iso, "%Y-%m-%d")
    end = datetime.strptime(end_iso, "%Y-%m-%d") + timedelta(days=1)  # inclusive end day
    cur = start
    while cur < end:
        yield cur.strftime("%Y%m%d%H%M%S")
        cur += timedelta(minutes=_INTERVAL_MINUTES)


def _gkg_url(ts: str) -> str:
    return f"{_GDELT_BASE}/{ts}.gkg.csv.zip"


def _passes_filters(
    row: list[str],
    allowlist: frozenset[str],
    theme_filter: tuple[str, ...],
    drop_non_fashion_paths: bool = True,
) -> bool:
    if len(row) <= max(_COL_IDX["SourceCommonName"], _COL_IDX["V2Themes"]):
        return False
    source = row[_COL_IDX["SourceCommonName"]].strip().lower()
    if source not in allowlist:
        return False
    if drop_non_fashion_paths:
        url = row[_COL_IDX["DocumentIdentifier"]]
        if not keep_url(source, url):
            return False
    if theme_filter:
        themes_blob = (
            row[_COL_IDX["Themes"]] + ";" + row[_COL_IDX["V2Themes"]]
        ).upper()
        if not any(t in themes_blob for t in theme_filter):
            return False
    return True


def _download_and_filter(
    ts: str,
    session: requests.Session,
    timeout: float,
    allowlist: frozenset[str],
    theme_filter: tuple[str, ...],
    drop_non_fashion_paths: bool = True,
) -> tuple[str, list[list[str]]]:
    """Return (ts, kept_rows). Empty rows list means nothing matched."""
    url = _gkg_url(ts)
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code != 200:
            return ts, []
        zf = zipfile.ZipFile(io.BytesIO(r.content))
    except Exception as e:  # noqa: BLE001 — network / corrupt zip
        log.warning("download failed for %s: %s", ts, e)
        return ts, []

    kept: list[list[str]] = []
    for name in zf.namelist():
        try:
            with zf.open(name) as f:
                text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                reader = csv.reader(text, delimiter="\t")
                for row in reader:
                    try:
                        if _passes_filters(row, allowlist, theme_filter, drop_non_fashion_paths):
                            kept.append(row)
                    except (csv.Error, IndexError) as e:
                        # Single malformed row — keep going.
                        log.debug("row skipped in %s: %s", ts, e)
                        continue
        except (csv.Error, UnicodeError, zipfile.BadZipFile) as e:
            log.warning("file %s in %s unreadable: %s", name, ts, e)
            continue
    return ts, kept


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    allowlist = FASHION_MEDIA_DOMAINS
    theme_filter = tuple(t.upper() for t in args.theme_filter)

    timestamps = list(_iter_timestamps(args.start, args.end))
    if args.max_files is not None:
        timestamps = timestamps[: args.max_files]
    total = len(timestamps)
    log.info(
        "GDELT GKG download window: %s → %s (%d 15-min files, %d allowlist domains)",
        args.start, args.end, total, len(allowlist),
    )

    # Output: one CSV per YYYY-MM, append mode.
    out_writers: dict[str, csv.writer] = {}
    out_handles: dict[str, object] = {}

    def writer_for(ts: str) -> csv.writer:
        key = ts[:6]  # YYYYMM
        if key not in out_writers:
            iso = f"{key[:4]}-{key[4:]}"
            # NOTE: GDELT's native format is tab-delimited but named .csv —
            # the adapter expects .csv so the discovery glob matches.
            path = args.out_dir / f"gdelt_gkg_{iso}.csv"
            new_file = not path.exists()
            handle = path.open("a", encoding="utf-8", newline="")
            w = csv.writer(handle, delimiter="\t")
            if new_file:
                w.writerow(_GKG_COLS)
            out_writers[key] = w
            out_handles[key] = handle
        return out_writers[key]

    session = requests.Session()
    session.headers["User-Agent"] = "MARGO-trend-pipeline/0.1 (research)"

    t0 = time.time()
    kept_total = 0
    completed = 0
    drop_non_fashion_paths = not args.no_drop_non_fashion_paths
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [
            pool.submit(
                _download_and_filter, ts, session, args.timeout,
                allowlist, theme_filter, drop_non_fashion_paths,
            )
            for ts in timestamps
        ]
        for fut in cf.as_completed(futures):
            ts, rows = fut.result()
            completed += 1
            if rows:
                w = writer_for(ts)
                for row in rows:
                    w.writerow(row)
                kept_total += len(rows)
            if completed % 100 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-3)
                eta_min = (total - completed) / max(rate, 1e-3) / 60
                log.info(
                    "progress %d/%d files (%.1f f/s, eta=%.1fm) kept_rows=%d",
                    completed, total, rate, eta_min, kept_total,
                )

    # Close all output handles.
    for handle in out_handles.values():
        handle.close()  # type: ignore[attr-defined]

    log.info(
        "DONE — %d files processed → %d filtered rows kept in %s (%.1fm)",
        completed, kept_total, args.out_dir, (time.time() - t0) / 60,
    )


if __name__ == "__main__":
    main()
