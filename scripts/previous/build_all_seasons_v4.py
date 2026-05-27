"""Orchestrate the full bulk build of v5 season-trend snapshots.

Wraps ``scripts.build_season_snapshot`` and iterates over fashion seasons.
Skips seasons whose snapshot file already exists, so re-running picks up
exactly where it left off.

Default range covers Amazon Fashion's meaningful data span
(2015-SS to 2023-SS, 17 seasons).

Usage::

    python -m scripts.build_all_seasons_v4 \\
        --processed-dir "data/Amazon Fashion/processed"

To resume after an interruption: just run again. Finished seasons are skipped.

File naming: ``fashion_trend_<year>_<SS|FW>.json`` under ``trend_cache/``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from adapters.trends.season_pipeline import snapshot_path                # noqa: E402
from adapters.trends.seasons import iter_seasons, parse_season           # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build all 17 v5 season snapshots.")
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--start-season", default="2015-SS")
    p.add_argument("--end-season", default="2023-SS")
    p.add_argument("--domain", default="fashion")
    p.add_argument("--region", default="US")
    p.add_argument("--per-search-max-results", type=int, default=12)
    p.add_argument("--force", action="store_true",
                   help="Re-build even if snapshot exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing.")
    p.add_argument("--inter-season-sleep", type=float, default=3.0,
                   help="Seconds between season builds (API politeness).")
    return p.parse_args()


def build_command(args: argparse.Namespace, season_label: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "scripts.build_season_snapshot",
        "--season", season_label,
        "--processed-dir", str(args.processed_dir),
        "--domain", args.domain,
        "--region", args.region,
        "--per-search-max-results", str(args.per_search_max_results),
    ]
    if args.force:
        cmd.append("--force")
    return cmd


def main() -> None:
    args = parse_args()
    start = parse_season(args.start_season)
    end = parse_season(args.end_season)
    seasons = list(iter_seasons(start, end))

    print(f"[plan] {len(seasons)} seasons: {seasons[0].label} → {seasons[-1].label}")
    t0 = time.time()
    built = 0
    skipped = 0
    failed: list[str] = []

    for i, season in enumerate(seasons, 1):
        prefix = f"[{i:2d}/{len(seasons)}] {season.label:8s}"
        out_path = snapshot_path(args.processed_dir, args.domain, season)
        if not args.force and out_path.exists():
            print(f"{prefix} SKIP ({out_path.name} exists)")
            skipped += 1
            continue
        cmd = build_command(args, season.label)
        print(f"{prefix} BUILD → {out_path.name}")
        if args.dry_run:
            print(f"    {' '.join(cmd)}")
            continue
        season_t0 = time.time()
        try:
            result = subprocess.run(cmd, check=False)
            elapsed = time.time() - season_t0
            if result.returncode == 0:
                print(f"    OK ({elapsed:.0f}s)")
                built += 1
            else:
                print(f"    FAIL rc={result.returncode} ({elapsed:.0f}s)")
                failed.append(season.label)
        except KeyboardInterrupt:
            print("\n[interrupted] partial progress preserved on disk.")
            sys.exit(130)
        if i < len(seasons) and args.inter_season_sleep > 0:
            time.sleep(args.inter_season_sleep)

    total = time.time() - t0
    print(
        f"\n[done] built={built} skipped={skipped} failed={len(failed)} "
        f"elapsed={total:.0f}s ({total / 60:.1f}min)"
    )
    if failed:
        print(f"       failed seasons: {failed}")
        sys.exit(2)


if __name__ == "__main__":
    main()
