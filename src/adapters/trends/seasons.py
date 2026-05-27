"""Fashion-season time abstraction.

The industry sells in two buying seasons per year:

* ``S/S YYYY`` (Spring/Summer) — buying window **Feb–Jul YYYY**.
* ``A/W YYYY`` (Autumn/Winter) — buying window **Aug YYYY – Jan (YYYY+1)**.

These are the cycles fashion houses report against, and they also match
what Amazon Fashion shoppers actually buy when (you buy a wool coat in
October, not in March). So a trend snapshot keyed to a *season* is more
meaningful than one keyed to a single month.

This module is the single source of truth for converting between season
identifiers (``"2023-SS"``, ``"2022-AW"``) and absolute (start, end) date
windows used by the trend adapters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Iterator

_SEASON_RE = re.compile(r"^(\d{4})-(SS|AW)$", re.IGNORECASE)

# Buying-season boundaries (inclusive months). Choose calendar boundaries so
# month-bucketing in adapters lines up trivially.
SS_MONTHS = (2, 3, 4, 5, 6, 7)        # Feb–Jul
AW_MONTHS = (8, 9, 10, 11, 12, 1)     # Aug–Jan (crosses year)


@dataclass(frozen=True)
class Season:
    year: int
    half: str  # "SS" or "AW"

    @property
    def label(self) -> str:
        return f"{self.year}-{self.half}"

    def date_range(self) -> tuple[str, str]:
        """Return inclusive (start, end) ISO date strings for this season."""
        if self.half == "SS":
            start = date(self.year, 2, 1)
            end = date(self.year, 7, 31)
        else:  # AW: Aug YYYY – Jan (YYYY+1)
            start = date(self.year, 8, 1)
            end = date(self.year + 1, 1, 31)
        return start.isoformat(), end.isoformat()

    def __str__(self) -> str:
        return self.label


def parse_season(s: str) -> Season:
    m = _SEASON_RE.match(s.strip())
    if not m:
        raise ValueError(f"Bad season string {s!r}; expected e.g. '2023-SS' or '2022-AW'")
    return Season(year=int(m.group(1)), half=m.group(2).upper())


def season_of(d: date) -> Season:
    """Map a calendar date to the buying season it belongs to."""
    if d.month in SS_MONTHS:
        return Season(year=d.year, half="SS")
    # A/W spans Aug-Dec of year Y and Jan of Y+1; both belong to A/W Y.
    if d.month == 1:
        return Season(year=d.year - 1, half="AW")
    return Season(year=d.year, half="AW")


def iter_seasons(start: Season, end: Season) -> Iterator[Season]:
    """Yield seasons inclusively from start to end in chronological order."""
    order = ("SS", "AW")
    y, h = start.year, start.half
    while True:
        yield Season(year=y, half=h)
        if y == end.year and h == end.half:
            return
        # Advance one half-step.
        if h == "SS":
            h = "AW"
        else:
            h = "SS"
            y += 1


def nearest_season(target: Season, available: list[Season]) -> Season | None:
    """Pick the available season closest in time to ``target``.

    Seasons are ordered by ``(year, SS<AW)``; distance is in half-year steps.
    Ties resolve to the *later* season (more recent data tends to age better
    than older data for fashion trends).
    """
    if not available:
        return None
    order = {"SS": 0, "AW": 1}

    def _ord(s: Season) -> int:
        return 2 * s.year + order[s.half]

    t = _ord(target)
    best = min(available, key=lambda s: (abs(_ord(s) - t), -_ord(s)))
    return best
