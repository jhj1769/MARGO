"""End-to-end sanity check against the real MargoEngine for a single user.

Loads the processed Amazon Fashion artefacts, picks the user with the
richest history, runs the full 4-phase pipeline with a default brief,
and prints the resulting Top-K + 3-layer rationale + validation summary.

Usage::

    # vLLM (Qwen2.5-7B) — start the server first
    MARGO_LLM_BACKEND=vllm \\
    MARGO_VLLM_BASE_URL=http://localhost:8000/v1 \\
    MARGO_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \\
    python -m scripts.sanity_one_user \\
        --processed-dir "data/Amazon Fashion/processed" \\
        --brief "Casual-to-formal upsell. Boost outerwear; keep within 30% of user's typical price."
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from api import MargoEngine, MargoEngineConfig
from lifecycle.orchestrator import MargoRunConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument(
        "--brief",
        default="Casual-to-formal upsell. Boost outerwear; keep within 30% of user's typical price.",
    )
    p.add_argument("--user-id", default=None, help="Override; default = pick a representative user.")
    p.add_argument(
        "--min-history",
        type=int,
        default=15,
        help="When auto-selecting, require at least this much history.",
    )
    p.add_argument(
        "--max-history",
        type=int,
        default=60,
        help="When auto-selecting, avoid power users (prompt blow-up).",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--candidate-size", type=int, default=50)
    p.add_argument("--rerank-window", type=int, default=10)
    p.add_argument("--max-iterations", type=int, default=1)
    p.add_argument("--bm25-only", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()

    print(f"\n=== MARGO sanity (1 user) ===")
    print(f"processed_dir = {args.processed_dir}")
    print(f"brief         = {args.brief!r}\n")

    cfg = MargoEngineConfig(
        processed_dir=args.processed_dir,
        snapshot_dir=args.processed_dir / "trend_cache",
        bm25_only=args.bm25_only,
    )
    t0 = time.time()
    engine = MargoEngine(cfg)
    print(f"[engine ready] {time.time() - t0:.1f}s — "
          f"{len(engine.catalog):,} items / {len(engine._user_histories):,} users\n")

    if args.user_id:
        user_id = args.user_id
        history = engine._user_histories.get(user_id, [])
    else:
        candidates = [
            (uid, h) for uid, h in engine._user_histories.items()
            if args.min_history <= len(h) <= args.max_history
        ]
        if not candidates:
            user_id, history = max(engine._user_histories.items(), key=lambda kv: len(kv[1]))
            print(f"[warn] no user matched min/max history; falling back to richest user")
        else:
            user_id, history = max(candidates, key=lambda kv: len(kv[1]))

    print(f"[user] {user_id}  (history={len(history)})")
    for h in history[-5:]:
        print(f"   · {h}")
    print()

    run_cfg = MargoRunConfig(
        top_k=args.top_k,
        candidate_size=args.candidate_size,
        rerank_window=args.rerank_window,
        max_iterations=args.max_iterations,
    )

    t0 = time.time()
    result = engine.recommend(user_id, args.brief, config=run_cfg)
    elapsed = time.time() - t0

    print(f"\n[recommend done] {elapsed:.1f}s — "
          f"candidate_pool={result.candidate_pool_size}, "
          f"iterations={result.iterations}, "
          f"phase4_passed={result.phase4_passed}\n")

    print(f"=== Directive ===")
    print(f"  goal:        {result.directive.goal}")
    print(f"  NL:          {result.directive.natural_language}")
    print(f"  structured:  {result.directive.structured_constraints}\n")

    if result.trend:
        print(f"=== Trend ===")
        print(f"  summary:  {result.trend.summary}")
        print(f"  keywords: {result.trend.keywords}\n")

    print(f"=== Top-{len(result.top_k)} ===")
    for i, r in enumerate(result.top_k, 1):
        title = engine.catalog.get(r.item_id).title if engine.catalog.get(r.item_id) else "(unknown)"
        attrs = engine.item_attrs.get(r.item_id, {})
        print(f"\n#{i}  score={r.score:.3f}  [{r.item_id}]")
        print(f"   {title[:90]}")
        print(f"   price={attrs.get('price')}  brand={attrs.get('brand')}  category={attrs.get('category')}")
        print(f"   ─ Personal : {r.rationale.personal}")
        print(f"   ─ Directive: {r.rationale.directive}")
        print(f"   ─ Trend    : {r.rationale.trend}")

    usage = engine.llm.usage_log
    print(f"\n=== LLM usage ===")
    print(f"  calls={usage.calls}  prompt_tokens={usage.prompt_tokens}  "
          f"completion_tokens={usage.completion_tokens}")
    print(f"  per-agent: {usage.calls_by_agent}")

    print("\nOK ✓\n")


if __name__ == "__main__":
    main()
