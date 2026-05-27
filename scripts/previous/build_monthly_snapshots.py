"""Build per-month multi-source trend snapshots for 2023.

Each monthly snapshot uses a **90-day sliding window** that ends at the
month's last day. That way:

  * No future leakage: a test point in month M only sees data up to
    end-of-M.
  * Smoothing: 90 days of mentions per signal keeps the per-keyword
    short_ma / long_ma ratios stable.
  * Per-month granularity: the system can pick the snapshot whose
    window ends at the test point's month, so a January user sees a
    different trend picture than a September user.

For months 1-2 the 90-day window reaches into Q4 2022 — currently we
only have 2023 GDELT data, so the GDELT side of those windows is
partial. Wikipedia pageviews cover the full window for every month, so
the multi-source consensus compensates.

Usage::

    MARGO_LLM_BACKEND=vllm \\
    python -m scripts.build_monthly_snapshots \\
        --processed-dir "data/Amazon Fashion/processed" \\
        --gdelt-data-dir data/external_trends/gdelt \\
        --year 2023 --months 1-9
"""

from __future__ import annotations

import argparse
import calendar
import logging
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--gdelt-data-dir", type=Path, default=None)
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--months", default="1-9",
                   help="Inclusive month range, e.g. '1-9' or '3-6'.")
    p.add_argument("--lookback-days", type=int, default=90)
    p.add_argument("--max-keywords", type=int, default=200)
    p.add_argument("--n-catalog", type=int, default=80)
    p.add_argument("--use-trend-lexicon", action="store_true", default=True)
    p.add_argument("--no-google", action="store_true", default=True)
    p.add_argument("--no-semantic", action="store_true", default=True)
    p.add_argument("--no-bge", action="store_true", default=True)
    return p.parse_args()


def month_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def build_one(
    *,
    year: int,
    month: int,
    lookback_days: int,
    processed_dir: Path,
    gdelt_data_dir: Path | None,
    max_keywords: int,
    n_catalog: int,
    use_trend_lexicon: bool,
    no_google: bool,
    no_semantic: bool,
    no_bge: bool,
) -> tuple[bool, float]:
    """Invoke the underlying single-window builder for one month. Returns (ok, elapsed_s)."""
    last_day = calendar.monthrange(year, month)[1]
    end = date(year, month, last_day)
    start = end - timedelta(days=lookback_days - 1)
    label = f"{year}-{month:02d}"

    cmd = [
        sys.executable, "-m", "scripts.build_multisource_trend_snapshot",
        "--processed-dir", str(processed_dir),
        "--time-start", start.isoformat(),
        "--time-end", end.isoformat(),
        "--time-window-label", label,
        "--max-keywords", str(max_keywords),
        "--n-catalog", str(n_catalog),
    ]
    if gdelt_data_dir is not None:
        cmd += ["--gdelt-data-dir", str(gdelt_data_dir)]
    if use_trend_lexicon:
        cmd.append("--use-trend-lexicon")
    if no_google:
        cmd.append("--no-google")
    if no_semantic:
        cmd.append("--no-semantic")
    if no_bge:
        cmd.append("--no-bge")

    t0 = time.time()
    print(f"\n{'='*72}\n[{label}] window {start} -> {end} ({lookback_days}d lookback)\n{'='*72}")
    rc = subprocess.call(cmd)
    elapsed = time.time() - t0
    return rc == 0, elapsed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    months = month_range(args.months)
    print(f"Building {len(months)} monthly snapshots for {args.year}: months {months}")

    results: list[tuple[str, bool, float]] = []
    for m in months:
        ok, elapsed = build_one(
            year=args.year, month=m, lookback_days=args.lookback_days,
            processed_dir=args.processed_dir, gdelt_data_dir=args.gdelt_data_dir,
            max_keywords=args.max_keywords, n_catalog=args.n_catalog,
            use_trend_lexicon=args.use_trend_lexicon,
            no_google=args.no_google, no_semantic=args.no_semantic, no_bge=args.no_bge,
        )
        results.append((f"{args.year}-{m:02d}", ok, elapsed))

    print("\n" + "="*72)
    print("Summary")
    print("="*72)
    for label, ok, elapsed in results:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {label}  ({elapsed:.1f}s)")
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} succeeded")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
