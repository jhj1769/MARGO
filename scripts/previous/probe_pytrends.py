"""Probe pytrends connectivity and quota for fashion keywords.

We test the three call shapes we actually need for the MARGO Trend Agent:
1. ``interest_over_time``  — for stable / rising / declining classification.
2. ``related_queries``     — for rising-query discovery (closest to true trend).
3. ``trending_searches``   — global discovery, no seed needed.

Prints a verdict + small dump so we know whether Google Trends is usable
from this machine right now.
"""

from __future__ import annotations

import time
import traceback

# --- compat shim: pytrends 4.9.2 uses `method_whitelist=`, removed in urllib3>=2 ---
from urllib3.util.retry import Retry as _Retry
_orig_retry_init = _Retry.__init__
def _patched_retry_init(self, *args, **kwargs):
    if "method_whitelist" in kwargs:
        kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
    _orig_retry_init(self, *args, **kwargs)
_Retry.__init__ = _patched_retry_init

from pytrends.request import TrendReq  # noqa: E402


SEEDS = ["linen blazer", "wide leg pants", "trench coat", "skinny jeans", "white sneakers"]
TIMEFRAME = "2023-03-01 2023-05-31"  # SS 2023 window
GEO = "US"


def main() -> None:
    print(f"== pytrends probe ==")
    print(f"  seeds:     {SEEDS}")
    print(f"  timeframe: {TIMEFRAME}")
    print(f"  geo:       {GEO}\n")

    pt = TrendReq(hl="en-US", tz=540, timeout=(10, 30), retries=2, backoff_factor=1.5)

    # --- 1) interest_over_time ----------------------------------------------
    print("[1] interest_over_time (5 keywords in one batch)")
    try:
        pt.build_payload(SEEDS, cat=0, timeframe=TIMEFRAME, geo=GEO)
        iot = pt.interest_over_time()
        if iot.empty:
            print("    EMPTY result")
        else:
            print(f"    shape={iot.shape}, weeks={len(iot)}, kw_cols={[c for c in iot.columns if c != 'isPartial']}")
            print(iot.tail(3).to_string())
        print("    OK ✓\n")
    except Exception as e:
        print(f"    FAILED ✗ — {type(e).__name__}: {e}\n")
        traceback.print_exc()

    time.sleep(2)

    # --- 2) related_queries (rising) ----------------------------------------
    print("[2] related_queries · rising  (per-seed)")
    try:
        pt.build_payload(SEEDS[:3], cat=0, timeframe=TIMEFRAME, geo=GEO)
        rq = pt.related_queries()
        for kw, payload in rq.items():
            rising = payload.get("rising")
            top = payload.get("top")
            print(f"  · {kw}: "
                  f"{0 if rising is None else len(rising)} rising / "
                  f"{0 if top is None else len(top)} top")
            if rising is not None and len(rising) > 0:
                for _, row in rising.head(5).iterrows():
                    print(f"      rising → {row['query']!r}  (+{row['value']}%)")
        print("  OK ✓\n")
    except Exception as e:
        print(f"  FAILED ✗ — {type(e).__name__}: {e}\n")
        traceback.print_exc()

    time.sleep(2)

    # --- 3) trending_searches (global discovery) -----------------------------
    print("[3] trending_searches (today, US — discovery without seeds)")
    try:
        ts = pt.trending_searches(pn="united_states")
        print(f"  rows={len(ts)} — sample:")
        print(ts.head(5).to_string())
        print("  OK ✓\n")
    except Exception as e:
        print(f"  FAILED ✗ — {type(e).__name__}: {e}\n")
        traceback.print_exc()

    print("== probe done ==")


if __name__ == "__main__":
    main()
