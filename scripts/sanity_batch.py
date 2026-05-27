"""Batch sanity — run a small set of *diverse* users through MARGO end-to-end
and print a compact summary so multiple working-evidence cases fit on one
screen.

Selection strategy: pick N users from test split such that the set
spans (a) users with rejected history (Phase C active),
(b) users without rejected history (Phase C inert — baseline-like),
(c) varying cohort signatures so we see Phase A both trigger and not.

Per user we print:
  * cohort signature + key axes
  * directive policy_hint
  * trend keywords (top 3)
  * top-3 recommended items (id, brand, 1-line personal rationale)
  * Phase A drops / Phase C usage / Phase B,D memory deltas
  * elapsed seconds + LLM calls

At the end a compact 1-line-per-user table for quick scanning.

Usage::

    MARGO_LLM_BACKEND=vllm \\
    MARGO_VLLM_BASE_URL=http://localhost:8000/v1 \\
    MARGO_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \\
    python -m scripts.sanity_batch \\
        --processed-dir "data/Amazon Fashion/processed" \\
        --memory-root "memory" --bm25-only \\
        --num-users 5 --max-iterations 1
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Any

from api import MargoEngine, MargoEngineConfig
from core.agents.trend_agent import apply_cohort_conditioning
from core.agents.user_agent import UserAgent
from core.lifecycle.orchestrator import MargoRunConfig
from data.fashion.rejection_pattern import summarise_rejection_pattern


SCENARIO_PRESETS: dict[str, str] = {
    "light_layering":
        "Wind down winter coats and lead shoppers into spring with light "
        "trench coats, cardigans, and field jackets. Skip down-filled or "
        "puffy outerwear.",
    "private_label_first":
        "Lift the share of private-label items in this week's feed. Stay "
        "close to each shopper's usual price band so the push feels "
        "natural rather than pushy.",
    "graduation_dress":
        "Graduation season is here. Recommend dresses, heels, and clutches "
        "that fit a graduation ceremony or after-party — but stay within "
        "each shopper's familiar color and silhouette range.",
    "workout_onboarding":
        "Help shoppers who have very little activewear in their history "
        "take a first step. Recommend entry-level leggings, sports bras, "
        "training tees, and supportive sneakers at an approachable price.",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--memory-root", type=Path, required=True)
    p.add_argument(
        "--direction",
        default=None,
        help="Operator's natural-language direction. If omitted, --scenario picks "
             "one of the web demo presets.",
    )
    p.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_PRESETS.keys()),
        default="graduation_dress",
        help="Preset direction from web demo mock. Ignored when --direction is set.",
    )
    p.add_argument("--num-users", type=int, default=5)
    p.add_argument("--min-history", type=int, default=15)
    p.add_argument("--max-history", type=int, default=60)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--candidate-size", type=int, default=50)
    p.add_argument("--rerank-window", type=int, default=10)
    p.add_argument("--max-iterations", type=int, default=1)
    p.add_argument("--bm25-only", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="ISO date (YYYY-MM-DD). Anchors trend snapshot to the season "
             "containing this date (e.g. 2023-03-15 -> 2023-SS).",
    )
    p.add_argument(
        "--show-pool",
        type=int,
        default=8,
        help="How many retrieval pool items to print per user (after gender filter, "
             "before rerank-window cap). Helps diagnose retrieval composition.",
    )
    return p.parse_args()


def count_memory(memory_root: Path, sub: str) -> int:
    d = memory_root / sub
    if not d.exists():
        return 0
    total = 0
    for p in d.glob("*.jsonl"):
        try:
            with p.open("r", encoding="utf-8") as f:
                total += sum(1 for line in f if line.strip())
        except OSError:
            continue
    return total


def pick_diverse_users(
    engine: MargoEngine,
    *,
    n: int,
    min_history: int,
    max_history: int,
    seed: int,
) -> list[str]:
    """Pick a small spread: half with rejected history (Phase C active),
    half without (Phase C inert). Filter by history size to keep prompts
    sane and to avoid power users blowing context."""
    rng = random.Random(seed)
    eligible_with_rej: list[str] = []
    eligible_without:  list[str] = []
    for uid, hist in engine._user_histories.items():
        if not (min_history <= len(hist) <= max_history):
            continue
        if uid in engine._rejected_history:
            eligible_with_rej.append(uid)
        else:
            eligible_without.append(uid)
    rng.shuffle(eligible_with_rej)
    rng.shuffle(eligible_without)
    half = n // 2
    chosen = eligible_with_rej[:half] + eligible_without[: n - half]
    if len(chosen) < n:
        chosen += (eligible_with_rej + eligible_without)[: n - len(chosen)]
    return chosen[:n]


def run_one(
    engine: MargoEngine,
    user_id: str,
    *,
    direction: str,
    run_cfg: MargoRunConfig,
    memory_root: Path,
    as_of_date: str | None = None,
    show_pool: int = 8,
) -> dict[str, Any]:
    """Run one user end-to-end and return a compact summary dict."""
    print(f"\n{'=' * 100}")
    history_size = len(engine._user_histories.get(user_id, []))
    rej_count = len(engine._rejected_history.get(user_id, []))
    user_gender = engine._user_gender_focus_cache.get(user_id)
    if user_gender is None:
        # Pre-warm so we can print it before recommend() runs.
        from data.fashion.gender import aggregate_user_gender
        user_gender = aggregate_user_gender(
            engine._user_history_items.get(user_id, []),
            engine._item_gender_lookup,
        )
        engine._user_gender_focus_cache[user_id] = user_gender

    print(f"USER {user_id}  |  history={history_size}  |  rejected={rej_count}  |  "
          f"gender_focus={user_gender}")
    print(f"{'=' * 100}")

    # Memory snapshot BEFORE this recommend (we'll diff after)
    item_events_before  = count_memory(memory_root, "item")
    expert_events_before = count_memory(memory_root, "expert")

    t0 = time.time()
    result = engine.recommend(user_id, direction, config=run_cfg, as_of_date=as_of_date)
    elapsed = time.time() - t0

    item_events_after  = count_memory(memory_root, "item")
    expert_events_after = count_memory(memory_root, "expert")

    # Re-fetch the user agent's preference state (engine creates a fresh
    # UserAgent per recommend; we recreate one to read the same axes).
    history = engine._user_histories.get(user_id, [])
    history_ids = engine._user_history_items.get(user_id, [])
    user_agent_probe = UserAgent(
        user_id=user_id,
        history=history,
        history_item_ids=history_ids,
        items_df=engine._items_df,
        processed_dir=engine.cfg.processed_dir,
        llm=engine.llm, prompts=engine.prompts, bus=engine.bus,
        schema_validator=engine.validator,
    )
    user_agent_probe.build_profile()
    state = user_agent_probe.state.preference_state

    # ---- Phase 1 / 1.5 ----
    if state:
        cohort = state.cohort_signature
        axes_summary = " | ".join(
            f"{a.name.split('_')[0]}={a.value}" for a in state.axes
        )
        print(f"\n[Phase 1] cohort = {cohort}")
        print(f"          axes   = {axes_summary}")
    else:
        cohort = ""
        axes_summary = "(no structured state)"
        print(f"\n[Phase 1] no structured state available")

    # ---- Phase 2 ----
    print(f"\n[Phase 2] Directive:")
    print(f"          goal        = {result.directive.goal[:80]}")
    print(f"          policy_hint = {getattr(result.directive, 'policy_hint', None)!r}")
    print(f"          constraints = {result.directive.structured_constraints}")
    if result.trend:
        print(f"          trend       = {result.trend.summary[:80]}...")
        print(f"          keywords    = {result.trend.keywords[:5]}")
        print(f"          rising      = {result.trend.rising_attributes}")
        print(f"          raw_sources = {result.trend.raw_sources}")
        print(f"          time_window = {result.trend.time_window}")

    # ---- Retrieval pool inspection ----
    # Reproduce the same retrieve + gender filter + trend boost the engine
    # made so we can show what the User Agent sees. Diagnostic only (no LLM).
    from data.fashion.gender import compatible as gender_compatible
    from core.lifecycle.phase3_reasoning import (
        _build_trend_patterns,
        _item_text_for_trend,
        _TREND_BOOST_PER_KW,
        _TREND_BOOST_MAX,
    )
    from dataclasses import replace
    retrieve_kw: dict = {}
    if result.trend and result.trend.keywords:
        retrieve_kw["trend_keywords"] = result.trend.keywords
    raw_hits = list(
        engine.retriever.retrieve_with_directive(
            user_agent_probe.state.profile,
            result.directive.natural_language,
            k=run_cfg.candidate_size,
            **retrieve_kw,
        )
    )
    gendered_hits = [
        h for h in raw_hits
        if gender_compatible(
            user_gender, engine._item_gender_lookup.get(h.item_id, "unknown"),
        )
    ]
    # Apply trend boost (must mirror phase3_reasoning).
    trend_kw_list = list(result.trend.keywords) if result.trend and result.trend.keywords else []
    patterns = _build_trend_patterns(trend_kw_list)
    boosted_hits = []
    trend_match_count = 0
    for h in gendered_hits:
        facts = engine.catalog.get(h.item_id)
        if facts is None or not patterns:
            boosted_hits.append((h, 0))
            continue
        text = _item_text_for_trend(facts)
        overlap = sum(1 for _, pat in patterns if pat.search(text))
        if overlap > 0:
            bonus = min(overlap * _TREND_BOOST_PER_KW, _TREND_BOOST_MAX)
            boosted_hits.append((replace(h, score=h.score + bonus), overlap))
            trend_match_count += 1
        else:
            boosted_hits.append((h, 0))
    boosted_hits.sort(key=lambda t: -t[0].score)

    print(f"\n[Retrieval] raw_pool={len(raw_hits)}  "
          f"gender_kept={len(gendered_hits)}  "
          f"gender_dropped={len(raw_hits) - len(gendered_hits)}  "
          f"trend_kw_match={trend_match_count}")
    if show_pool > 0:
        print(f"            top-{show_pool} candidates (after gender filter + trend boost):")
        for i, (h, ov) in enumerate(boosted_hits[:show_pool], 1):
            facts = engine.catalog.get(h.item_id)
            title = (facts.title if facts else "(unknown)")[:50]
            brand = (engine.item_attrs.get(h.item_id, {}).get("brand") or "-")[:14]
            ig = engine._item_gender_lookup.get(h.item_id, "?")
            trend_mark = f"+T{ov}" if ov > 0 else "    "
            print(f"            {i:>2}. score={h.score:.3f} [{ig:<7}] "
                  f"{trend_mark}  {brand:<14} | {title}")
        if not boosted_hits:
            print("            (empty pool after gender filter)")

    # ---- Phase A ----
    a_drops = 0
    if state and cohort and result.trend:
        before = sum(len(v) for v in result.trend.rising_attributes.values())
        filtered = apply_cohort_conditioning(result.trend, cohort)
        after = sum(len(v) for v in filtered.rising_attributes.values())
        a_drops = before - after
        print(f"\n[Phase A] cohort-conditional drop: {before} → {after} kept ({a_drops} dropped)")
    else:
        print(f"\n[Phase A] skipped (no cohort or no trend)")

    # ---- Phase C ----
    rej_count = len(engine._rejected_history.get(user_id, []))
    if rej_count >= 3:
        pat = summarise_rejection_pattern(
            engine._rejected_history[user_id], engine._items_df,
        )
        print(f"\n[Phase C] Rejected layer:")
        print(f"          {rej_count} rejected interaction(s) on file")
        if pat:
            print(f"          top categories: {pat['top_categories']}")
            print(f"          top brands    : {pat['top_brands']}")
            if pat.get("style_hints"):
                print(f"          style hints   : {pat['style_hints']}")
    else:
        print(f"\n[Phase C] {rej_count} rejected interaction(s) — layer inert")

    # ---- Top-K with 1-line rationale ----
    print(f"\n[Phase 3] Top-{len(result.top_k)}:")
    for i, r in enumerate(result.top_k, 1):
        facts = engine.catalog.get(r.item_id)
        title = (facts.title if facts else "(unknown)")[:70]
        attrs = engine.item_attrs.get(r.item_id, {})
        brand = attrs.get("brand") or "—"
        print(f"  #{i}  score={r.score:.3f}  [{r.item_id}]  brand={brand}")
        print(f"      title: {title}")
        print(f"      personal: {r.rationale.personal[:130]}")

    # ---- Phase B/D delta ----
    print(f"\n[Phase B] Item memory:   +{item_events_after - item_events_before} "
          f"events (total {item_events_after})")
    print(f"[Phase D] Expert memory: +{expert_events_after - expert_events_before} "
          f"events (total {expert_events_after})")

    print(f"\n[runtime] {elapsed:.1f}s  /  LLM calls={engine.llm.usage_log.calls}  /  "
          f"iter={result.iterations}  passed={result.phase4_passed}")

    return {
        "user_id": user_id,
        "history_size": len(history),
        "rejected_count": rej_count,
        "gender_focus": user_gender,
        "raw_pool": len(raw_hits),
        "gender_dropped": len(raw_hits) - len(gendered_hits),
        "cohort_short": (cohort or "")[:60],
        "policy_hint": getattr(result.directive, "policy_hint", None),
        "trend_keywords": (result.trend.keywords[:3] if result.trend else []),
        "phase_a_drops": a_drops,
        "phase_c_active": rej_count >= 3,
        "item_mem_delta": item_events_after - item_events_before,
        "expert_mem_delta": expert_events_after - expert_events_before,
        "top1_brand": (
            engine.item_attrs.get(result.top_k[0].item_id, {}).get("brand")
            if result.top_k else None
        ),
        "top1_score": result.top_k[0].score if result.top_k else 0.0,
        "elapsed_sec": round(elapsed, 1),
        "passed":   result.phase4_passed,
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()

    cfg = MargoEngineConfig(
        processed_dir=args.processed_dir,
        snapshot_dir=args.processed_dir / "trend_cache",
        bm25_only=args.bm25_only,
        memory_root=args.memory_root,
    )

    # Resolve the operator's direction: explicit --direction wins, otherwise
    # the --scenario preset (mirrors the web demo mock.json directives).
    direction = args.direction or SCENARIO_PRESETS[args.scenario]
    direction_label = "custom" if args.direction else args.scenario

    print(f"\n=== MARGO sanity-batch ({args.num_users} users) ===")
    print(f"Direction ({direction_label}): {direction}")
    if args.as_of_date:
        print(f"as-of-date  : {args.as_of_date}")
    t0 = time.time()
    engine = MargoEngine(cfg)
    print(f"[engine ready] {time.time() - t0:.1f}s — {len(engine.catalog):,} items, "
          f"{len(engine._user_histories):,} users, "
          f"{len(engine._rejected_history):,} users with rejected history")
    # Quick gender-lookup stats so we can see catalog gender distribution.
    from collections import Counter
    gender_dist = Counter(engine._item_gender_lookup.values())
    print(f"[catalog gender mix] " + ", ".join(
        f"{g}={n:,}" for g, n in gender_dist.most_common()))

    users = pick_diverse_users(
        engine,
        n=args.num_users,
        min_history=args.min_history,
        max_history=args.max_history,
        seed=args.seed,
    )
    print(f"[selected] {len(users)} users (mix of with/without rejected history)")

    run_cfg = MargoRunConfig(
        top_k=args.top_k,
        candidate_size=args.candidate_size,
        rerank_window=args.rerank_window,
        max_iterations=args.max_iterations,
    )

    summaries: list[dict[str, Any]] = []
    for uid in users:
        try:
            s = run_one(
                engine,
                uid,
                direction=direction,
                run_cfg=run_cfg,
                memory_root=args.memory_root,
                as_of_date=args.as_of_date,
                show_pool=args.show_pool,
            )
            summaries.append(s)
        except Exception:
            logging.exception("recommend failed for user %s", uid)

    # ---- Final compact table ----
    print(f"\n\n{'=' * 100}")
    print(f"SANITY SUMMARY  (n={len(summaries)})")
    print(f"{'=' * 100}")
    print(f"{'user_id':<32} {'hist':>5} {'rej':>4} {'gender':<8} "
          f"{'pool':>5} {'g_drop':>6} {'A_drop':>7} {'sec':>6} {'top1_brand':<14}")
    print("-" * 100)
    for s in summaries:
        print(f"{s['user_id']:<32} {s['history_size']:>5} {s['rejected_count']:>4} "
              f"{str(s.get('gender_focus','?'))[:7]:<8} "
              f"{s.get('raw_pool', 0):>5} {s.get('gender_dropped', 0):>6} "
              f"{s['phase_a_drops']:>7} "
              f"{s['elapsed_sec']:>6} {str(s['top1_brand'] or '-')[:13]:<14}")
    print()


if __name__ == "__main__":
    main()
