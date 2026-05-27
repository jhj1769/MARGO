"""High-level façade used by the web routes.

In *engine* mode this wraps :class:`api.MargoEngine` and translates
its outputs into JSON-serialisable dicts. In *mock* mode it serves
hand-crafted demo data so the UI can be developed against a live
backend even without a GPU.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Make the in-repo `src/` packages importable without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

log = logging.getLogger("margo.web.runner")


# --------------------------------------------------------------------------- #
# Public dataclasses                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class TraceEvent:
    phase: str  # "phase1" | "phase2" | "phase3" | "phase4"
    sender: str
    receivers: list[str]
    type: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "sender": self.sender,
            "receivers": self.receivers,
            "type": self.type,
            "summary": self.summary,
            "payload": self.payload,
            "ts": self.ts,
        }


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #


class MargoRunner:
    """Mode-switching façade."""

    def __init__(self, mode: str, engine=None, mock_data: Optional[dict] = None) -> None:
        self.mode = mode  # "engine" | "mock"
        self.engine = engine
        self.mock = mock_data or _load_mock_data()
        # In-process active directive (per process; not multi-tenant).
        self.active_directive: Optional[dict] = self.mock["scenarios"][0]["directive"]
        self.directive_history: list[dict] = []
        # Lazily populated in engine mode: per-user gender label and the
        # cached "demo roster" of 10 women + 10 men used by Step 1.
        self._gender_cache: dict[str, str] = {}
        self._user_roster: Optional[list[dict]] = None

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls) -> "MargoRunner":
        if os.getenv("MARGO_DEMO_MODE", "").lower() == "mock":
            log.info("Forced mock mode via MARGO_DEMO_MODE=mock")
            return cls(mode="mock")
        try:
            from api import MargoEngine, MargoEngineConfig  # type: ignore

            processed = os.getenv("MARGO_PROCESSED_DIR")
            if not processed:
                raise RuntimeError("MARGO_PROCESSED_DIR not set — defaulting to mock mode")
            # Phase B/D — agentic memory root. When set, Item & Expert agents
            # persist their typed events under <memory_root>/{item,expert}/.
            # Unset ⇒ NullMemory (legacy v3 behaviour, deployment back-compat).
            memory_root_env = os.getenv("MARGO_MEMORY_ROOT")
            memory_root = Path(memory_root_env) if memory_root_env else None
            cfg = MargoEngineConfig(
                processed_dir=Path(processed),
                snapshot_dir=Path(os.getenv("MARGO_SNAPSHOT_DIR", processed)) / "trend_cache",
                bm25_only=os.getenv("MARGO_BM25_ONLY", "0") == "1",
                time_window=os.getenv("MARGO_TREND_WINDOW", "2023-Q2"),
                memory_root=memory_root,
            )
            if memory_root:
                log.info("agentic memory enabled — root=%s", memory_root)
            engine = MargoEngine(cfg)
            return cls(mode="engine", engine=engine)
        except Exception as e:
            log.warning("Falling back to mock mode (%s).", e)
            return cls(mode="mock")

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def list_users(self) -> list[dict]:
        if self.mode != "engine":
            return self.mock["users"]
        if self._user_roster is None:
            self._user_roster = self._build_demo_roster()
        return self._user_roster

    def get_user_history(self, user_id: str) -> dict:
        """Structured per-user history for the Step 1 expandable panel.

        Returns the item-by-item record the User Agent's `query_preference`
        ultimately consumes — but in a UI-friendly shape (image, category
        breadcrumb, price). Restricted to the most recent ``limit`` entries
        so the panel stays scannable.
        """
        if self.mode != "engine":
            user = next(
                (u for u in self.mock["users"] if u["user_id"] == user_id),
                None,
            )
            if not user:
                return {"user_id": user_id, "items": [], "summary": ""}
            return {
                "user_id": user_id,
                "items": [],
                "summary": user.get("summary", ""),
            }

        ids = self.engine._user_history_items.get(user_id, [])
        if not ids:
            return {"user_id": user_id, "items": [], "summary": ""}

        limit = int(os.getenv("MARGO_DEMO_HISTORY_VIEW_LIMIT", "30"))
        # Show the most recent N (history is chronological).
        recent_ids = ids[-limit:][::-1]  # newest first
        items: list[dict] = []
        for iid in recent_ids:
            attrs = self.engine.item_attrs.get(iid) or {}
            items.append({
                "item_id": iid,
                "title": attrs.get("title") or "",
                "image_url": attrs.get("image_url"),
                "category": attrs.get("category") or [],
                "price": attrs.get("price"),
                "brand": attrs.get("brand"),
            })
        return {
            "user_id": user_id,
            "items": items,
            "summary": _summarise_history(self.engine._user_histories.get(user_id, [])),
            "history_size": len(ids),
        }

    # ------------------------------------------------------------------ #
    # Demo roster (10 women + 10 men)                                     #
    # ------------------------------------------------------------------ #

    def _classify_gender(self, user_id: str) -> Optional[str]:
        """Return 'women' | 'men' | None based on a user's history."""
        cached = self._gender_cache.get(user_id)
        if cached is not None:
            return cached if cached != "_unknown_" else None
        ids = self.engine._user_history_items.get(user_id, [])
        women = men = 0
        for iid in ids:
            cats = self.engine.item_attrs.get(iid, {}).get("category") or []
            lower = {str(c).lower() for c in cats}
            if "women" in lower or "girls" in lower:
                women += 1
            elif "men" in lower or "boys" in lower:
                men += 1
        total = women + men
        if total < 5:
            self._gender_cache[user_id] = "_unknown_"
            return None
        if women / total >= 0.7:
            label = "women"
        elif men / total >= 0.7:
            label = "men"
        else:
            label = "_unknown_"
        self._gender_cache[user_id] = label
        return None if label == "_unknown_" else label

    def _build_demo_roster(self) -> list[dict]:
        min_h = int(os.getenv("MARGO_DEMO_MIN_HISTORY", "30"))
        max_h = int(os.getenv("MARGO_DEMO_MAX_HISTORY", "200"))
        per_side = int(os.getenv("MARGO_DEMO_USERS_PER_GENDER", "10"))

        candidates = [
            (uid, hist) for uid, hist in self.engine._user_histories.items()
            if min_h <= len(hist) <= max_h
        ]
        # Larger histories first → richer demo runs.
        candidates.sort(key=lambda kv: len(kv[1]), reverse=True)

        women: list[tuple[str, list]] = []
        men: list[tuple[str, list]] = []
        # Walk down the list until both buckets are full; cap scan to keep
        # the first call fast.
        scan_cap = int(os.getenv("MARGO_DEMO_ROSTER_SCAN", "2000"))
        for uid, hist in candidates[:scan_cap]:
            if len(women) >= per_side and len(men) >= per_side:
                break
            g = self._classify_gender(uid)
            if g == "women" and len(women) < per_side:
                women.append((uid, hist))
            elif g == "men" and len(men) < per_side:
                men.append((uid, hist))

        if len(women) < per_side or len(men) < per_side:
            log.warning(
                "Demo roster underfilled (women=%d, men=%d); consider raising "
                "MARGO_DEMO_ROSTER_SCAN.", len(women), len(men),
            )

        def _row(uid: str, hist: list, gender: str) -> dict:
            return {
                "user_id": uid,
                "name": f"User {uid[:8]}",
                "gender": gender,
                "history_size": len(hist),
                "summary": _summarise_history(hist),
            }

        roster = [_row(uid, h, "women") for uid, h in women]
        roster += [_row(uid, h, "men") for uid, h in men]
        log.info(
            "Demo roster built: %d women + %d men (gender cache=%d).",
            len(women), len(men), len(self._gender_cache),
        )
        return roster

    def list_scenarios(self) -> list[dict]:
        return self.mock["scenarios"]

    def get_active_directive(self) -> Optional[dict]:
        return self.active_directive

    def set_active_directive(self, directive: dict) -> dict:
        directive["issued_at"] = time.time()
        directive["iteration"] = directive.get("iteration", 0)
        self.directive_history.append(directive)
        self.active_directive = directive
        return directive

    async def recommend(
        self,
        user_id: str,
        *,
        directive: Optional[dict] = None,
        k: int = 6,
        emit: Optional[callable] = None,  # type: ignore
    ) -> dict:
        directive = directive or self.active_directive or self.mock["scenarios"][0]["directive"]
        if self.mode == "engine":
            if user_id.startswith("u_"):
                raise ValueError(
                    f"Mock user '{user_id}' cannot run in engine mode. "
                    "Please select a real user from the roster."
                )
            return await self._run_engine(user_id, directive, k, emit)
        return await self._run_mock(user_id, directive, k, emit)

    # ------------------------------------------------------------------ #
    # Engine path                                                          #
    # ------------------------------------------------------------------ #

    async def _run_engine(
        self,
        user_id: str,
        directive: dict,
        k: int,
        emit,
    ) -> dict:
        from core.lifecycle.orchestrator import MargoRunConfig  # type: ignore

        loop = asyncio.get_event_loop()

        def _blocking():
            brief = directive.get("natural_language") or directive.get("goal", "")
            rerank = int(os.getenv("MARGO_DEMO_RERANK", "30"))
            return self.engine.recommend(
                user_id,
                brief,
                config=MargoRunConfig(
                    top_k=k,
                    candidate_size=int(os.getenv("MARGO_DEMO_CANDIDATES", "40")),
                    rerank_window=max(rerank, k),
                    max_iterations=int(os.getenv("MARGO_DEMO_MAX_ITER", "1")),
                ),
            )

        if emit is not None:
            await emit(TraceEvent(phase="phase2", sender="expert", receivers=["user", "item", "trend"],
                                  type="directive", summary=directive.get("natural_language", ""),
                                  payload=directive))

        result = await loop.run_in_executor(None, _blocking)
        out = {
            "user_id": result.user_id,
            "directive": result.directive.model_dump(),
            "trend": result.trend.model_dump() if result.trend else None,
            "candidate_pool_size": result.candidate_pool_size,
            "iterations": result.iterations,
            "phase4_passed": result.phase4_passed,
            "insights": _engine_insights(result),
            "top_k": [
                {
                    "item_id": r.item_id,
                    "score": r.score,
                    "rationale": r.rationale.model_dump(),
                    "title": (self.engine.catalog.get(r.item_id).title
                              if self.engine.catalog.get(r.item_id) else r.item_id),
                    "image_url": self._image_for(r.item_id),
                    "price": self.engine.item_attrs.get(r.item_id, {}).get("price"),
                    "category": self.engine.item_attrs.get(r.item_id, {}).get("category"),
                    "brand": self.engine.item_attrs.get(r.item_id, {}).get("brand"),
                    "color": self.engine.item_attrs.get(r.item_id, {}).get("color"),
                    "material": self.engine.item_attrs.get(r.item_id, {}).get("material"),
                }
                for r in result.top_k
            ],
        }
        if emit is not None:
            await emit(TraceEvent(
                phase="phase3", sender="user-agent", receivers=["result"], type="ranked",
                summary=f"User Agent produced Top-{k}",
                payload={"ranked": [r.item_id for r in result.top_k]},
            ))
            await emit(TraceEvent(
                phase="phase4", sender="expert", receivers=["result"], type="validation",
                summary=("passed" if result.phase4_passed else "refined") + f" · iter={result.iterations}",
                payload={"passed": result.phase4_passed, "iterations": result.iterations},
            ))
        return out

    # ------------------------------------------------------------------ #
    # Mock path                                                            #
    # ------------------------------------------------------------------ #

    async def _run_mock(
        self,
        user_id: str,
        directive: dict,
        k: int,
        emit,
    ) -> dict:
        user = next((u for u in self.mock["users"] if u["user_id"] == user_id), self.mock["users"][0])
        catalog: list[dict] = self.mock["catalog"]
        trend = self._pick_trend(directive)

        # ── Phase 2 — Directive + Trend broadcast ──────────────────────────
        if emit:
            await emit(TraceEvent(
                phase="phase2", sender="expert-agent", receivers=["user-agent", "item-agent", "trend-agent"],
                type="directive", summary=directive["natural_language"],
                payload={"goal": directive["goal"],
                         "structured": directive.get("structured_constraints", {})},
            ))
            await asyncio.sleep(0.35)
            await emit(TraceEvent(
                phase="phase2", sender="trend-agent", receivers=["user-agent", "item-agent"],
                type="broadcast", summary=trend["summary"],
                payload=trend,
            ))
            await asyncio.sleep(0.35)
            await emit(TraceEvent(
                phase="phase3", sender="retriever", receivers=["item-agent"],
                type="candidates",
                summary=f"BGE-M3 retrieved {len(catalog)} candidates (directive-aware)",
                payload={"pool_size": len(catalog)},
            ))
            await asyncio.sleep(0.4)

        # ── Phase 3 — Rank with rich NL outputs ────────────────────────────
        ranked = self._mock_rank(user, directive, catalog, k)

        # Inject Item Agent NL self-description per recommended item.
        for r in ranked:
            r["self_description"] = _phrase_self_describe(r, directive, trend)

        # The User Agent's NL reaction to the directive — used as an insight card.
        user_reaction = _phrase_user_reaction(user, directive, trend, ranked)
        expert_check = _phrase_expert_check(directive, ranked)
        passed = expert_check["passed"]
        compliance = expert_check["compliance"]

        if emit:
            # Highlight one item the Item Agent rewrote, for the trace timeline.
            if ranked:
                top = ranked[0]
                await emit(TraceEvent(
                    phase="phase3", sender="item-agent", receivers=["user-agent"],
                    type="self-describe",
                    summary=f"“{top['self_description']}”",
                    payload={"item_id": top["item_id"], "description": top["self_description"]},
                ))
                await asyncio.sleep(0.35)

            await emit(TraceEvent(
                phase="phase3", sender="user-agent", receivers=["result"],
                type="ranked",
                summary=user_reaction,
                payload={"ranked": [r["item_id"] for r in ranked], "reaction": user_reaction},
            ))
            await asyncio.sleep(0.35)

            await emit(TraceEvent(
                phase="phase4", sender="expert-agent", receivers=["result"],
                type="validation",
                summary=expert_check["summary"],
                payload={"passed": passed, "compliance": compliance,
                         "violations": expert_check["violations"]},
            ))

        insights = {
            "trend_card": {
                "title": "Trend Agent · context broadcast",
                "summary": trend["summary"],
                "keywords": trend.get("keywords", []),
                "domain": trend.get("domain", "fashion"),
                "time_window": trend.get("time_window", "2026-Q2"),
            },
            "directive_card": {
                "title": "Expert Agent · directive issued",
                "goal": directive.get("goal", ""),
                "natural_language": directive.get("natural_language", ""),
                "structured": directive.get("structured_constraints", {}),
            },
            "user_card": {
                "title": "User Agent · reaction",
                "summary": user_reaction,
                "user": user["name"],
            },
            "validation_card": {
                "title": "Expert Agent · validation",
                "summary": expert_check["summary"],
                "passed": passed,
                "compliance": compliance,
                "violations": expert_check["violations"],
            },
        }

        return {
            "user_id": user_id,
            "directive": directive,
            "trend": trend,
            "candidate_pool_size": len(catalog),
            "iterations": 1,
            "phase4_passed": passed,
            "insights": insights,
            "top_k": ranked,
        }

    # ------------------------------------------------------------------ #
    # Mock helpers                                                         #
    # ------------------------------------------------------------------ #

    def _mock_rank(self, user: dict, directive: dict, catalog: list[dict], k: int) -> list[dict]:
        c = directive.get("structured_constraints", {}) or {}
        boost = {x.lower() for x in _as_list(c.get("boost_category"))}
        forbid = {x.lower() for x in _as_list(c.get("forbid_category"))}
        price_max = c.get("price_max")
        price_min = c.get("price_min")
        diff_max = c.get("price_diff_pct_max")
        user_avg = user.get("avg_price", 60.0)
        nl = (directive.get("natural_language") or "").lower()

        trend = self._pick_trend(directive)
        trend_kw = {k.lower() for k in trend.get("keywords", [])}

        scored = []
        for item in catalog:
            categories = {c.lower() for c in (item.get("category") or [])}
            if categories & forbid:
                continue
            price = item.get("price", 60.0)
            if price_max is not None and price > price_max:
                continue
            if price_min is not None and price < price_min:
                continue
            if diff_max is not None and user_avg > 0:
                diff = (price - user_avg) / user_avg * 100
                if diff > diff_max:
                    continue

            score = 0.55
            if categories & boost:
                score += 0.25
            if any(kw in item["title"].lower() for kw in trend_kw):
                score += 0.10
            if any(t in item["title"].lower() for t in user.get("affinity_tokens", [])):
                score += 0.08
            if any(t in nl for t in item.get("intent_tokens", [])):
                score += 0.05

            score += random.uniform(-0.05, 0.05)
            score = min(0.99, max(0.05, score))

            rationale = {
                "personal": _phrase_personal(item, user),
                "directive": _phrase_directive(item, directive, score, user_avg),
                "trend": _phrase_trend(item, trend),
            }
            scored.append({
                **item,
                "score": round(score, 3),
                "rationale": rationale,
            })

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:k]

    def _pick_trend(self, directive: dict) -> dict:
        nl = (directive.get("natural_language") or "").lower()
        for t in self.mock["trends"]:
            if any(c.lower() in nl for c in t["match_cues"]):
                return t
        return self.mock["trends"][0]

    def _image_for(self, item_id: str) -> Optional[str]:
        attrs = self.engine.item_attrs.get(item_id, {}) if self.engine else {}
        return attrs.get("image_url")


# --------------------------------------------------------------------------- #
# Mock data loading                                                            #
# --------------------------------------------------------------------------- #


_MOCK_PATH = Path(__file__).resolve().parents[1] / "data" / "mock.json"


def _load_mock_data() -> dict:
    if _MOCK_PATH.exists():
        return json.loads(_MOCK_PATH.read_text(encoding="utf-8"))
    log.warning("Mock data file %s missing; using empty stub", _MOCK_PATH)
    return {"users": [], "catalog": [], "trends": [], "scenarios": []}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]


def _summarise_history(hist: list[str]) -> str:
    if not hist:
        return "(no history)"
    return " · ".join(hist[:3])


def _phrase_personal(item: dict, user: dict) -> str:
    band = user.get("price_band", "₩50–100k")
    style = user.get("style", "casual-leaning")
    return (
        f"{item['title']} sits inside your usual {band} range and aligns with "
        f"a {style} wardrobe — close to items you've already chosen."
    )


def _phrase_directive(item: dict, directive: dict, score: float, user_avg: float) -> str:
    goal = directive.get("goal", "")
    boost = ", ".join(directive.get("structured_constraints", {}).get("boost_category", []) or [])
    return (
        f"Honors the operator goal “{goal}”."
        + (f" Belongs to boosted category ({boost})." if boost else "")
        + f" Price gap vs. your average: {((item['price'] - user_avg) / max(1, user_avg)) * 100:+.0f}%."
    )


def _phrase_trend(item: dict, trend: dict) -> str:
    cue = next(
        (kw for kw in trend.get("keywords", []) if kw.lower() in item["title"].lower()),
        None,
    )
    if cue:
        return f"Matches the rising signal “{cue}” — {trend['summary']}"
    return f"Coherent with current context — {trend['summary']}"


# --------------------------------------------------------------------------- #
# NL outputs used by the demo's insight cards                                 #
# --------------------------------------------------------------------------- #


def _phrase_self_describe(item: dict, directive: dict, trend: dict) -> str:
    """Mimic what an Item Agent would emit under the current directive + trend."""
    boost = directive.get("structured_constraints", {}).get("boost_category") or []
    boost_hit = any(b.lower() in (item.get("category") or [""])[0].lower() for b in boost)
    trend_hit = next(
        (kw for kw in trend.get("keywords", []) if kw.lower() in item["title"].lower()),
        None,
    )
    base = item.get("description") or item["title"]
    parts = [base]
    if boost_hit and boost:
        parts.append(f"Sits inside the boosted “{boost[0]}” lane for this campaign.")
    if trend_hit:
        parts.append(f"Carries the rising “{trend_hit}” signal of SS26.")
    if item.get("price"):
        parts.append(f"At ${int(item['price'])}, it lands within the operator's acceptable range.")
    return " ".join(parts)


def _phrase_user_reaction(user: dict, directive: dict, trend: dict, ranked: list[dict]) -> str:
    """One short paragraph in the user's voice — used as the User Agent card."""
    if not ranked:
        return (
            f"Given the operator's intent “{directive.get('goal', '')}”, "
            f"nothing in the pool fits {user['name']}'s usual {user.get('price_band', '')} "
            f"band without breaking trust."
        )
    top = ranked[0]
    extras = []
    if directive.get("structured_constraints", {}).get("boost_category"):
        extras.append("the operator wants me to lean toward "
                      + ", ".join(directive["structured_constraints"]["boost_category"]))
    if trend.get("keywords"):
        extras.append("and the season is leaning " + ", ".join(trend["keywords"][:3]))
    extra_text = " and ".join(extras)
    return (
        f"{user['name']} — “{top['title']} is the gentlest step forward from my usual {user.get('style', 'wardrobe')}; "
        f"{extra_text}.”"
        if extras
        else f"{user['name']} — “{top['title']} is the gentlest step forward from my usual {user.get('style', 'wardrobe')}.”"
    )


def _phrase_expert_check(directive: dict, ranked: list[dict]) -> dict:
    """Mock Expert validation — checks the same hard constraints MARGO does."""
    sc = directive.get("structured_constraints", {}) or {}
    violations: list[str] = []
    boost = {x.lower() for x in _as_list(sc.get("boost_category"))}

    if boost:
        boost_seen = any(
            boost & {c.lower() for c in (r.get("category") or [])} for r in ranked
        )
        if not boost_seen and ranked:
            violations.append(
                f"None of the Top-{len(ranked)} items belongs to the boosted "
                f"category ({', '.join(sorted(boost))})."
            )

    passed = not violations
    compliance = round(0.93 if passed else random.uniform(0.55, 0.78), 3)
    summary = (
        f"✓ Top-{len(ranked)} passed validation · compliance {compliance:.2f}"
        if passed
        else f"✗ {len(violations)} violation(s) — refined directive recommended"
    )
    return {"passed": passed, "compliance": compliance, "violations": violations, "summary": summary}


# --------------------------------------------------------------------------- #
# Engine-mode insight cards                                                    #
# --------------------------------------------------------------------------- #


def _engine_insights(result) -> dict:
    """Translate a live MargoEngine `RecommendationResult` into the same
    insight-card shape the mock produces. Keeps the frontend code uniform."""
    directive = result.directive
    trend = result.trend
    ranked = result.top_k
    return {
        "trend_card": (
            {
                "title": "Trend Agent · context broadcast",
                "summary": trend.summary,
                "keywords": trend.keywords,
                "domain": trend.domain,
                "time_window": trend.time_window,
            }
            if trend is not None
            else None
        ),
        "directive_card": {
            "title": "Expert Agent · directive issued",
            "goal": directive.goal,
            "natural_language": directive.natural_language,
            "structured": directive.structured_constraints,
        },
        "user_card": {
            "title": "User Agent · reaction",
            "summary": (
                f"Top-{len(ranked)} produced under directive “{directive.goal}”."
            ),
            "user": result.user_id,
        },
        "validation_card": {
            "title": "Expert Agent · validation",
            "summary": ("✓ passed validation"
                        if result.phase4_passed
                        else f"✗ refined after {result.iterations} iter"),
            "passed": result.phase4_passed,
            "compliance": 1.0 if result.phase4_passed else 0.6,
            "violations": [],
        },
    }
