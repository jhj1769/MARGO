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
from core.agents.user_agent import UserAgent
from core.lifecycle.orchestrator import MargoRunConfig
from core.protocol.messages import UserPreferenceState


_USER_STATE_CACHE: dict[str, UserPreferenceState | None] = {}


def engine_user_state(engine: MargoEngine, user_id: str) -> UserPreferenceState | None:
    """Build a one-off UserAgent solely to extract its preference state.

    Memoised per user_id so multiple call sites in the sanity printout
    don't trigger redundant LLM calls on the same user.
    """
    if user_id in _USER_STATE_CACHE:
        return _USER_STATE_CACHE[user_id]
    history = engine._user_histories.get(user_id, [])
    history_item_ids = engine._user_history_items.get(user_id, [])
    if not history:
        _USER_STATE_CACHE[user_id] = None
        return None
    user = UserAgent(
        user_id=user_id,
        history=history,
        history_item_ids=history_item_ids,
        items_df=engine._items_df,
        processed_dir=engine.cfg.processed_dir,
        llm=engine.llm,
        prompts=engine.prompts,
        bus=engine.bus,
        schema_validator=engine.validator,
    )
    user.build_profile()
    _USER_STATE_CACHE[user_id] = user.state.preference_state
    return user.state.preference_state


def _load_audience(processed_dir, item_id):
    from data.fashion.audience_loader import load_audience_profile
    return load_audience_profile(item_id, processed_dir)


def _last_interaction_date(processed_dir: Path, user_id: str) -> str | None:
    """Find the user's last interaction date (ISO YYYY-MM-DD) across train + test.

    We look at BOTH splits because the "current point in time" for the trend
    anchor is the most recent interaction we know about — test or train.
    """
    import pandas as pd
    latest_ts = None
    for split in ("test", "train"):
        path = processed_dir / f"{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["user_id", "timestamp"])
        df = df[df["user_id"].astype(str) == user_id]
        if df.empty:
            continue
        ts = int(df["timestamp"].max())
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
    if latest_ts is None:
        return None
    return pd.Timestamp(latest_ts, unit="ms", tz="UTC").date().isoformat()


def _peer_signal_or_none(engine: MargoEngine, user_id: str, item_id: str) -> float | None:
    """Compute the peer signal lazily on demand for the sanity printout."""
    state = engine_user_state(engine, user_id)
    if state is None or not state.cohort_signature:
        return None
    from data.fashion.cohort_loader import MIN_COHORT_SIZE, load_cohort_stats
    cohort = load_cohort_stats(state.cohort_signature, engine.cfg.processed_dir)
    if cohort is None or cohort.size < MIN_COHORT_SIZE:
        return None
    return cohort.peer_signal_for(item_id)


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
    p.add_argument(
        "--time-window",
        default="2023",
        help="Trend snapshot time window label (must match the snapshot built offline).",
    )
    p.add_argument(
        "--memory-root",
        type=Path,
        default=None,
        help="When set, Phase B (Item) / Phase D (Expert) agentic memory writes here.",
    )
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
        time_window=args.time_window,
        bm25_only=args.bm25_only,
        memory_root=args.memory_root,
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

    # Anchor the trend snapshot to the test user's actual last-interaction
    # month so the monthly snapshot picker grabs the right window.
    as_of = _last_interaction_date(args.processed_dir, user_id)
    if as_of:
        print(f"[anchor] test user's last interaction: {as_of} -> snapshot month {as_of[:7]}")

    t0 = time.time()
    result = engine.recommend(user_id, args.brief, config=run_cfg, as_of_date=as_of)
    elapsed = time.time() - t0

    print(f"\n[recommend done] {elapsed:.1f}s — "
          f"candidate_pool={result.candidate_pool_size}, "
          f"iterations={result.iterations}, "
          f"phase4_passed={result.phase4_passed}\n")

    print(f"=== Directive ===")
    print(f"  goal:        {result.directive.goal}")
    print(f"  NL:          {result.directive.natural_language}")
    print(f"  structured:  {result.directive.structured_constraints}")
    nlog = result.directive.negotiation_log
    if nlog and nlog.messages:
        print(f"  negotiation: outcome={nlog.final_outcome}, {len(nlog.messages)} message(s)")
        for m in nlog.messages:
            print(f"    [turn {m.turn}] {m.from_agent}→{m.to_agent} ({m.message_type}): "
                  f"{m.content_nl[:120]}")
            if m.proposed_directive_delta:
                print(f"      delta: {m.proposed_directive_delta}")
    elif nlog:
        print(f"  negotiation: outcome={nlog.final_outcome} (no tension)")
    print()

    if result.trend:
        print(f"=== Trend ===")
        print(f"  summary:    {result.trend.summary}")
        print(f"  keywords:   {result.trend.keywords}")
        if result.trend.rising_attributes:
            print(f"  rising:     {result.trend.rising_attributes}")
        if result.trend.raw_sources:
            print(f"  sources:    {result.trend.raw_sources}")
        print()

    # Enhancement 1 / 1.5 — show the user's structured preference state.
    user_state = engine_user_state(engine, user_id)
    if user_state is not None:
        print("=== User preference axes ===")
        print(f"  cohort_signature: {user_state.cohort_signature or '(none)'}")
        for axis in user_state.axes:
            evidence = "; ".join(axis.evidence[:3])
            print(f"  • {axis.name:>22s} = {axis.value!r}  "
                  f"(conf={axis.confidence:.2f}, src={axis.derived_from}, stab={axis.stability:.2f})")
            if evidence:
                print(f"      evidence: {evidence}")
            if axis.secondary_values:
                print(f"      secondary: {axis.secondary_values}")
        print()

    print(f"=== Top-{len(result.top_k)} ===")
    for i, r in enumerate(result.top_k, 1):
        title = engine.catalog.get(r.item_id).title if engine.catalog.get(r.item_id) else "(unknown)"
        attrs = engine.item_attrs.get(r.item_id, {})
        print(f"\n#{i}  score={r.score:.3f}  [{r.item_id}]")
        print(f"   {title[:90]}")
        print(f"   price={attrs.get('price')}  brand={attrs.get('brand')}  category={attrs.get('category')}")
        # Enhancement 2 — audience grounding (if a buyer aggregate exists)
        audience = _load_audience(engine.cfg.processed_dir, r.item_id)
        if audience is not None:
            print(f"   audience: {audience.buyer_count} buyers, "
                  f"top cohorts={audience.dominant_cohorts[:2] or 'n/a'}")
        # Enhancement 1.5 — peer signal for THIS user × item
        peer_ratio = _peer_signal_or_none(engine, user_id, r.item_id)
        if peer_ratio is not None:
            print(f"   peer_signal: {peer_ratio:.0%} of cohort bought this")
        print(f"   ─ Personal : {r.rationale.personal}")
        print(f"   ─ Directive: {r.rationale.directive}")
        print(f"   ─ Trend    : {r.rationale.trend}")

    usage = engine.llm.usage_log
    print(f"\n=== LLM usage ===")
    print(f"  calls={usage.calls}  prompt_tokens={usage.prompt_tokens}  "
          f"completion_tokens={usage.completion_tokens}")
    print(f"  per-agent: {usage.calls_by_agent}")

    # ------------------------------------------------------------------ #
    # Phase A-D explicit working evidence                                 #
    # ------------------------------------------------------------------ #
    print(f"\n=== Phase A-D working evidence ===")

    # Phase A — Cohort-Conditional Trend Application
    if user_state and user_state.cohort_signature and result.trend:
        from core.agents.trend_agent import apply_cohort_conditioning
        before = sum(len(v) for v in result.trend.rising_attributes.values())
        before_kw = len(result.trend.keywords)
        filtered = apply_cohort_conditioning(result.trend, user_state.cohort_signature)
        after = sum(len(v) for v in filtered.rising_attributes.values())
        after_kw = len(filtered.keywords)
        print(f"  [Phase A] Cohort-Conditional Trend:")
        print(f"    cohort = {user_state.cohort_signature}")
        print(f"    rising_attributes: {before} → {after} kept ({before - after} dropped)")
        print(f"    keywords:          {before_kw} → {after_kw} kept ({before_kw - after_kw} dropped)")
    else:
        print(f"  [Phase A] cohort signature unavailable — skipped")

    # Phase C — Rejected layer summary
    rejected_history = engine._rejected_history.get(user_id, [])
    if rejected_history:
        from data.fashion.rejection_pattern import summarise_rejection_pattern
        pat = summarise_rejection_pattern(rejected_history, engine._items_df)
        print(f"  [Phase C] Rejected layer:")
        print(f"    {len(rejected_history)} rejected interaction(s) on file for this user")
        if pat:
            print(f"    top rejected categories: {pat['top_categories']}")
            print(f"    top rejected brands    : {pat['top_brands']}")
            if pat.get("style_hints"):
                print(f"    recurring style hints  : {pat['style_hints']}")
        else:
            print(f"    (too few to summarise)")
    else:
        print(f"  [Phase C] no rejected interactions for this user — Rejected layer inert")

    # Directive policy_hint (Phase C ↔ Expert linkage)
    policy = getattr(result.directive, "policy_hint", None)
    print(f"  [Phase C] Directive.policy_hint = {policy!r}  "
          f"(drives Rejected-layer avoidance strength)")

    # Phase B / D — memory file event counts
    if args.memory_root and args.memory_root.exists():
        item_dir = args.memory_root / "item"
        expert_dir = args.memory_root / "expert"
        def _count(d: Path) -> tuple[int, int]:
            if not d.exists():
                return 0, 0
            n_files = 0
            n_events = 0
            for p in d.glob("*.jsonl"):
                n_files += 1
                try:
                    with p.open("r", encoding="utf-8") as f:
                        n_events += sum(1 for line in f if line.strip())
                except OSError:
                    continue
            return n_files, n_events
        it_files, it_events = _count(item_dir)
        ex_files, ex_events = _count(expert_dir)
        print(f"  [Phase B] Item memory:   {it_events} events across {it_files} item file(s)")
        print(f"  [Phase D] Expert memory: {ex_events} events across {ex_files} persona file(s)")
    else:
        print(f"  [Phase B/D] memory_root not set — agentic memory inert")

    print("\nOK ✓\n")


if __name__ == "__main__":
    main()
