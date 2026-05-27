"""Orchestrate the full 17-season bulk build of multisource trend snapshots.

Wraps ``scripts.build_multisource_trend_snapshot`` and iterates over fashion
seasons. Skips seasons whose snapshot file already exists, so re-running
after a YouTube quota stall picks up exactly where it left off.

Default range covers Amazon Fashion's meaningful data span (2015-SS to
2023-SS, 17 seasons). GDELT is enabled when ``--gdelt-data-dir`` points to a
directory with files for the relevant period; seasons outside that range
simply contribute 0 GDELT signal (graceful degradation — Wikipedia and
YouTube still cover them).

Usage::

    python -m scripts.build_all_seasons \\
        --processed-dir "data/Amazon Fashion/processed" \\
        --gdelt-data-dir data/external_trends/gdelt \\
        --max-keywords 30 \\
        --brief "Y2K revival; minimalism; quiet luxury"

To resume after a quota stall: just run again. Finished seasons are skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from adapters.trends.seasons import iter_seasons, parse_season


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--start-season", default="2015-SS",
                   help="First season to build (inclusive).")
    p.add_argument("--end-season", default="2023-SS",
                   help="Last season to build (inclusive).")
    p.add_argument("--gdelt-data-dir", type=Path, default=None,
                   help="If set, GDELT contributes to seasons whose data is in this dir.")
    p.add_argument("--domain", default="fashion")
    p.add_argument("--region", default="US")
    p.add_argument("--max-keywords", type=int, default=30)
    p.add_argument("--brief", default="Y2K revival; minimalism rising; quiet luxury")
    p.add_argument("--use-trend-lexicon", action="store_true", default=True)
    p.add_argument("--no-youtube", action="store_true",
                   help="Disable YouTube globally (e.g. when quota is exhausted).")
    p.add_argument("--no-wikipedia", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Re-build seasons even if snapshot file already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the build commands without executing them.")
    p.add_argument("--inter-season-sleep", type=float, default=5.0,
                   help="Seconds to sleep between season builds (be nice to APIs).")
    return p.parse_args()


def snapshot_exists(processed_dir: Path, domain: str, season_label: str) -> bool:
    path = (
        processed_dir / "trend_cache" / f"multisource_{domain}_{season_label}.json"
    )
    return path.exists()


def build_command(args: argparse.Namespace, season_label: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "scripts.build_multisource_trend_snapshot",
        "--processed-dir", str(args.processed_dir),
        "--season", season_label,
        "--domain", args.domain,
        "--region", args.region,
        "--max-keywords", str(args.max_keywords),
        "--brief", args.brief,
        "--no-pinterest",  # always (no token)
        "--no-google",     # always (pytrends IP-block risk)
    ]
    if args.use_trend_lexicon:
        cmd.append("--use-trend-lexicon")
    if args.gdelt_data_dir:
        cmd.extend(["--gdelt-data-dir", str(args.gdelt_data_dir)])
    else:
        cmd.append("--no-gdelt")
    if args.no_youtube:
        cmd.append("--no-youtube")
    if args.no_wikipedia:
        cmd.append("--no-wikipedia")
    return cmd


def main() -> int:
    args = parse_args()
    seasons = list(iter_seasons(parse_season(args.start_season),
                                parse_season(args.end_season)))
    print(f"[plan] {len(seasons)} seasons: {seasons[0].label} → {seasons[-1].label}")

    skipped, built, failed = 0, 0, 0
    failed_seasons: list[str] = []
    t_start = time.time()

    for i, season in enumerate(seasons, 1):
        if not args.force and snapshot_exists(args.processed_dir, args.domain, season.label):
            print(f"[{i:>2}/{len(seasons)}] {season.label:9s}  SKIP (snapshot exists)")
            skipped += 1
            continue

        cmd = build_command(args, season.label)
        print(f"\n[{i:>2}/{len(seasons)}] {season.label:9s}  BUILD")
        print(f"    {' '.join(cmd)}")
        if args.dry_run:
            continue

        t0 = time.time()
        rc = subprocess.call(cmd)
        elapsed = time.time() - t0
        if rc == 0:
            print(f"    OK ({elapsed:.0f}s)")
            built += 1
        else:
            print(f"    FAIL rc={rc} ({elapsed:.0f}s)")
            failed += 1
            failed_seasons.append(season.label)

        if i < len(seasons) and args.inter_season_sleep > 0:
            time.sleep(args.inter_season_sleep)

    total_elapsed = time.time() - t_start
    print(f"\n[done] built={built} skipped={skipped} failed={failed} "
          f"elapsed={total_elapsed:.0f}s")
    if failed_seasons:
        print(f"  failed seasons: {failed_seasons}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
