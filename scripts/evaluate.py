"""End-to-end evaluation harness — produces the metric table in the paper.

It samples N test users, runs MARGO recommend, then reports NDCG / HR /
DCR / TAS / IHR / VDR / SVR / CADR on that sample. The expensive parts
(LLM calls, BGE encoding) are batched at the engine level.

Usage::

    python -m scripts.evaluate \
        --processed-dir "data/Amazon Fashion/processed" \
        --brief "Casual→formal upsell. price_diff_pct ≤ 30%. Boost trench coats." \
        --num-users 50 --k 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from api import MargoEngine, MargoEngineConfig
from data.fashion.loader import load_processed
from eval.governance import dcr, tas
from eval.grounding import cadr, ihr, svr_from_validator, vdr
from eval.standard import hit_rate, mean, ndcg
from core.lifecycle.orchestrator import MargoRunConfig

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--snapshot-dir", type=Path, default=None)
    p.add_argument("--brief", required=True, type=str)
    p.add_argument("--num-users", type=int, default=20)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--candidate-size", type=int, default=100)
    p.add_argument("--rerank-window", type=int, default=20)
    p.add_argument("--max-iterations", type=int, default=2)
    p.add_argument("--bm25-only", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    cfg = MargoEngineConfig(
        processed_dir=args.processed_dir,
        snapshot_dir=args.snapshot_dir,
        bm25_only=args.bm25_only,
    )
    engine = MargoEngine(cfg)
    tables = load_processed(args.processed_dir)
    test_users = tables.test.dropna(subset=["user_id", "item_id"])
    sample = test_users.sample(min(args.num_users, len(test_users)), random_state=42)
    catalog_ids: set[str] = set(engine.catalog.keys())

    rows: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        uid = str(row["user_id"])
        positive = str(row["item_id"])
        try:
            res = engine.recommend(
                uid,
                args.brief,
                config=MargoRunConfig(
                    top_k=args.k,
                    candidate_size=args.candidate_size,
                    rerank_window=args.rerank_window,
                    max_iterations=args.max_iterations,
                ),
            )
        except Exception:
            log.exception("recommend failed for user %s", uid)
            continue

        ranked_ids = [r.item_id for r in res.top_k]
        descriptions = [r.rationale.personal + " " + r.rationale.directive + " " + r.rationale.trend
                        for r in res.top_k]
        row_metrics = {
            "user_id": uid,
            "positive": positive,
            "ndcg": ndcg(ranked_ids, positive, args.k),
            "hr": hit_rate(ranked_ids, positive, args.k),
            "dcr": dcr(res.top_k, directive=res.directive, item_attrs=engine.item_attrs),
            "tas": tas(res.top_k, trend=res.trend, item_attrs=engine.item_attrs) if res.trend else 0.0,
            "ihr": ihr(res.top_k, catalog_ids=catalog_ids),
            "vdr": vdr(descriptions, engine.vocabulary),
            "iterations": res.iterations,
            "passed": res.phase4_passed,
        }
        rows.append(row_metrics)

    df = pd.DataFrame(rows)
    if df.empty:
        log.error("No successful evaluations")
        return

    summary = {
        "n": int(len(df)),
        "NDCG@k":  float(mean(df["ndcg"])),
        "HR@k":    float(mean(df["hr"])),
        "DCR":     float(mean(df["dcr"])),
        "TAS":     float(mean(df["tas"])),
        "IHR":     float(mean(df["ihr"])),
        "VDR":     float(mean(df["vdr"])),
        "SVR":     float(svr_from_validator(engine.validator)),
        "CADR":    float(cadr(engine.bus.history())),
        "avg_iter":float(mean(df["iterations"])),
        "pass_rate":float(mean(df["passed"].astype(float))),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        with open(args.out.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
