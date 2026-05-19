"""Thin pytrends wrapper used by the MARGO Trend Agent.

Why a wrapper instead of using pytrends directly:

* **Compatibility** — pytrends 4.9.2 calls ``urllib3.util.Retry`` with the
  long-removed ``method_whitelist`` keyword. We monkey-patch ``Retry``
  once at import time so the rest of pytrends works against urllib3 ≥ 2.
* **Batching** — Google Trends accepts ≤ 5 keywords per ``build_payload``
  call. We hide that limit behind ``fetch_interest_over_time``.
* **Throttling** — bursty calls get the IP soft-banned. We sleep between
  calls with optional jitter.

Public API:
    GoogleTrendsClient.fetch_interest_over_time(keywords, timeframe, geo) -> pd.DataFrame
    GoogleTrendsClient.fetch_rising_queries(seeds, timeframe, geo)         -> dict[str, list[RisingQuery]]
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# --- urllib3 compat shim — must run before importing pytrends ---------
from urllib3.util.retry import Retry as _Retry  # noqa: E402

_orig_retry_init = _Retry.__init__


def _patched_retry_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    if "method_whitelist" in kwargs:
        kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
    _orig_retry_init(self, *args, **kwargs)


_Retry.__init__ = _patched_retry_init  # type: ignore[assignment]

from pytrends.request import TrendReq  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class RisingQueryRecord:
    seed: str
    query: str
    growth_pct: float


@dataclass
class GoogleTrendsClient:
    """Batched, throttled pytrends frontend."""

    hl: str = "en-US"
    tz: int = 540
    timeout: tuple[int, int] = (10, 30)
    retries: int = 2
    backoff: float = 1.5
    sleep_between: float = 2.0
    jitter: float = 0.5
    cat: int = 0  # Google Trends category id (0 = all)
    _pt: TrendReq = field(init=False)

    def __post_init__(self) -> None:
        self._pt = TrendReq(
            hl=self.hl, tz=self.tz, timeout=self.timeout,
            retries=self.retries, backoff_factor=self.backoff,
        )

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _sleep(self) -> None:
        if self.sleep_between > 0:
            time.sleep(self.sleep_between + random.random() * self.jitter)

    # ------------------------------------------------------------------ #
    # Public — interest over time                                        #
    # ------------------------------------------------------------------ #

    def fetch_interest_over_time(
        self,
        keywords: list[str],
        timeframe: str,
        geo: str = "US",
        batch_size: int = 5,
    ) -> pd.DataFrame:
        """Return a date-indexed dataframe with one column per keyword.

        Google normalises interest to [0..100] *per build_payload call*, so
        scores from different batches are not strictly comparable. For our
        purposes (relative within-window classification) that is fine.

        Empty / failed batches are skipped with a warning rather than
        raising — the caller can decide what an empty pool means.
        """
        if not keywords:
            return pd.DataFrame()

        out: list[pd.DataFrame] = []
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i : i + batch_size]
            try:
                self._pt.build_payload(batch, cat=self.cat, timeframe=timeframe, geo=geo)
                df = self._pt.interest_over_time()
            except Exception as e:  # pragma: no cover - network
                log.warning("interest_over_time batch %s failed: %s", batch, e)
                self._sleep()
                continue
            if df is None or df.empty:
                log.info("interest_over_time batch %s returned empty", batch)
            else:
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])
                out.append(df)
            self._sleep()

        if not out:
            return pd.DataFrame()
        merged = pd.concat(out, axis=1)
        # keep only first occurrence if a keyword was duplicated across batches
        merged = merged.loc[:, ~merged.columns.duplicated()]
        return merged

    # ------------------------------------------------------------------ #
    # Public — related queries (rising)                                  #
    # ------------------------------------------------------------------ #

    def fetch_rising_queries(
        self,
        seeds: list[str],
        timeframe: str,
        geo: str = "US",
        top_per_seed: int = 8,
    ) -> list[RisingQueryRecord]:
        """For each seed, return its rising 'breakout' queries.

        Google sometimes returns growth values as the literal ``"Breakout"``
        string instead of a number. We coerce those to a large positive
        sentinel so they still rank near the top.
        """
        out: list[RisingQueryRecord] = []
        for seed in seeds:
            try:
                self._pt.build_payload([seed], cat=self.cat, timeframe=timeframe, geo=geo)
                rq = self._pt.related_queries()
            except Exception as e:  # pragma: no cover - network
                log.warning("related_queries for seed=%r failed: %s", seed, e)
                self._sleep()
                continue
            payload = (rq or {}).get(seed) or {}
            rising = payload.get("rising")
            if rising is None or len(rising) == 0:
                self._sleep()
                continue
            for _, row in rising.head(top_per_seed).iterrows():
                val = row["value"]
                growth = float(val) if isinstance(val, (int, float)) else 10000.0
                out.append(RisingQueryRecord(seed=seed, query=str(row["query"]), growth_pct=growth))
            self._sleep()
        return out


def classify_keywords(
    iot: pd.DataFrame,
    sources: Optional[dict[str, str]] = None,
    *,
    rising_growth_thresh: float = 30.0,
    declining_growth_thresh: float = -25.0,
    stable_min_score: float = 35.0,
    halves_min_len: int = 2,
) -> tuple[list, list, list]:
    """Split keywords into rising / stable_top / declining lists.

    Returns three lists of :class:`KeywordSignal` (imported lazily to keep
    this module dependency-light).
    """
    from margo.grounding.trend_snapshot_schema import KeywordSignal

    sources = sources or {}
    rising: list = []
    stable: list = []
    declining: list = []

    if iot is None or iot.empty:
        return rising, stable, declining

    n = len(iot)
    half = max(halves_min_len, n // 2)
    first_half = iot.iloc[:half]
    last_half = iot.iloc[-half:]

    for kw in iot.columns:
        series = iot[kw].astype(float)
        first_mean = float(first_half[kw].mean() or 0.0)
        last_mean = float(last_half[kw].mean() or 0.0)
        abs_mean = float(series.mean() or 0.0)
        peak = float(series.max() or 0.0)

        if first_mean <= 1.0 and last_mean <= 1.0:
            # Effectively no signal — drop.
            continue

        if first_mean < 1.0:
            growth = 999.0  # cold start → treat as breakout
        else:
            growth = (last_mean - first_mean) / first_mean * 100.0

        sig = KeywordSignal(
            keyword=str(kw),
            absolute_score=round(abs_mean, 2),
            growth_pct=round(growth, 1),
            peak_score=round(peak, 1),
            source=sources.get(str(kw), "catalog"),
        )

        if growth >= rising_growth_thresh:
            rising.append(sig)
        elif growth <= declining_growth_thresh:
            declining.append(sig)
        elif abs_mean >= stable_min_score:
            stable.append(sig)
        # else: weak / noisy — drop.

    rising.sort(key=lambda s: s.growth_pct or 0.0, reverse=True)
    stable.sort(key=lambda s: s.absolute_score, reverse=True)
    declining.sort(key=lambda s: s.growth_pct or 0.0)
    return rising, stable, declining
