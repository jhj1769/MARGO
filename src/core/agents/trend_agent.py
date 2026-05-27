"""Trend Agent — turns season-level editorial evidence into a TrendInterpretation.

Evidence comes from a **v5 season snapshot** built by
``scripts/build_season_snapshot.py`` and stored at
``<processed_dir>/trend_cache/fashion_trend_<year>_<SS|FW>.json``. Each
snapshot carries TrendItems with lifecycle, confidence, rationale,
attributes, matched ASINs, and per-source evidence — so the LLM only
*interprets* this curated record, never the raw web.

When no season snapshot is available (e.g. unit tests with no data),
the agent falls back to live web search snippets via
:class:`adapters.trends.web_search.WebSearcher`. There is no v3/v4
fallback chain — those snapshot styles were retired with the v5 build
(see ``data/previous/trend_cache_legacy/`` and ``src/previous/``).

An interpretation cache (see :class:`adapters.trends.snapshot.TrendSnapshotStore`)
keeps re-runs cheap and reproducible.

Asymmetric authority: the snapshot is *evidence* for Expert reasoning,
not a directive. Phase 2 prompts surface its tensions as advisory only;
Phase 3 weights its keywords below the Expert directive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from core.agents.base import BaseAgent
from adapters.trends.snapshot import TrendSnapshotStore
from adapters.trends.season_pipeline import (
    load_snapshot as load_season_snapshot,
    snapshot_path as season_snapshot_path,
)
from adapters.trends.season_schema import SeasonTrendSnapshot
from adapters.trends.seasons import (
    Season,
    nearest_season,
    parse_season,
    season_of,
)
from core.validation.vocabulary import Vocabulary
from core.protocol.messages import (
    Directive,
    Message,
    MessageType,
    NegotiationMessage,
    TrendDirectiveTension,
    TrendInterpretation,
)
from adapters.trends.web_search import WebSearcher, WebSnippet

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Phase A — Cohort-Conditional Trend Application                              #
# --------------------------------------------------------------------------- #
# v3에서 trend는 모든 user에 균일 적용했지만, Heterogeneous Stakeholder
# Reasoning thesis는 *Trend도 first-class agent라면 user cohort에 맞게
# 차별 적용*되어야 한다고 주장한다. 여기서 *rule-based drop*으로 가장
# light한 구현을 한다 — cohort axis 값과 명시적으로 충돌하는 trend
# attribute는 그 user의 evaluate 단계에서만 제외된다. 학습은 없다.
#
# 의도적 단순함:
# * Drop-only (boost X) — schema 변경 없이 rising_attributes의 list만 줄임
# * 명시적 conflict lookup — reviewer가 "왜 drop했냐"에 한 줄로 답 가능
# * Cohort signature 미설정 시 no-op — backward compatible

# Cohort axis value → 명시적으로 충돌하는 attribute substring (lowercase)
# *대척점이 분명한* 페어만 등록. 애매한 케이스는 보존(over-aggressive drop 회피).
_COHORT_CONFLICT_TABLE: dict[str, set[str]] = {
    # Price tier 충돌
    "budget":         {"luxury", "premium", "high-end", "exclusive"},
    "luxury-aware":   {"fast-fashion", "ultra-budget"},
    # Style 충돌 (대척점이 분명한 페어)
    "minimal-casual": {"maximalist", "flashy", "gaudy", "baroque"},
    "streetwear":     {"preppy", "formal-traditional", "evening-wear"},
    "preppy":         {"streetwear", "punk", "grunge"},
    "feminine-romantic": {"hyper-masculine"},
    "athleisure":     {"evening-wear", "ball-gown", "tuxedo"},
}


def _parse_cohort_signature(sig: str) -> dict[str, str]:
    """``"bra:brand-diverse|cat:balanced|pri:mid-tier|sty:minimal-casual"``
    → ``{"bra": "brand-diverse", "cat": "balanced", "pri": "mid-tier", "sty": "minimal-casual"}``.
    """
    out: dict[str, str] = {}
    for part in (sig or "").split("|"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        if k and v:
            out[k.strip()] = v.strip()
    return out


def _conflicts_with_cohort(attribute: str, cohort_values: dict[str, str]) -> bool:
    """True iff ``attribute`` (case-insensitive) contains a substring that
    the conflict table marks as incompatible with any of the cohort's
    axis values.
    """
    attr_low = (attribute or "").lower()
    if not attr_low:
        return False
    for axis_value in cohort_values.values():
        conflicts = _COHORT_CONFLICT_TABLE.get(axis_value)
        if not conflicts:
            continue
        if any(c in attr_low for c in conflicts):
            return True
    return False


def apply_cohort_conditioning(
    interp: TrendInterpretation,
    cohort_signature: str,
) -> TrendInterpretation:
    """Return a cohort-filtered copy of ``interp``.

    *Rule-based*: attributes / keywords that explicitly conflict with the
    cohort's axis values (per ``_COHORT_CONFLICT_TABLE``) are dropped.
    All other content is preserved. ``summary`` and ``raw_sources`` are
    untouched — they describe the *evidence*, not what to apply.

    No-ops when ``cohort_signature`` is empty (preserves backward
    compatibility for callers without structured preference state).
    """
    if not cohort_signature:
        return interp
    cohort_values = _parse_cohort_signature(cohort_signature)
    if not cohort_values:
        return interp

    filtered_rising: dict[str, list[str]] = {}
    for axis, attrs in (interp.rising_attributes or {}).items():
        kept = [a for a in attrs if not _conflicts_with_cohort(a, cohort_values)]
        if kept:
            filtered_rising[axis] = kept

    filtered_keywords = [
        kw for kw in (interp.keywords or [])
        if not _conflicts_with_cohort(kw, cohort_values)
    ]

    return interp.model_copy(update={
        "rising_attributes": filtered_rising,
        "keywords": filtered_keywords,
    })


# --------------------------------------------------------------------------- #
# LLM output schema (private)                                                 #
# --------------------------------------------------------------------------- #


class _RawInterp(BaseModel):
    domain: str
    time_window: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    rising_attributes: dict[str, list[str]] = Field(default_factory=dict)
    raw_sources: list[str] = Field(default_factory=list)


@dataclass
class TrendState:
    domain: str
    time_window: str
    persona: str = "trend analyst"
    log: list[TrendInterpretation] = field(default_factory=list)


class TrendAgent(BaseAgent):
    """4th stakeholder — external editorial context speaks back into the framework."""

    def __init__(
        self,
        domain: str,
        time_window: str,
        *,
        vocabulary: Optional[Vocabulary] = None,
        searcher: Optional[WebSearcher] = None,
        snapshot_store: Optional[TrendSnapshotStore] = None,
        processed_dir: Optional[Path] = None,
        **kw,
    ) -> None:
        super().__init__(agent_id="trend", prompt_namespace="trend", **kw)
        self.state = TrendState(domain=domain, time_window=time_window)
        self.vocabulary = vocabulary or Vocabulary({})
        self.searcher = searcher or WebSearcher()
        self.snapshot = snapshot_store
        # Parent of ``trend_cache/`` containing fashion_trend_<year>_<SS|FW>.json
        # produced by scripts/build_season_snapshot.py.
        self.processed_dir = Path(processed_dir) if processed_dir else None
        # Per-call ablation toggle set by MargoEngine.recommend when
        # ``config.enable_trend_snapshot`` is False. Defaults to off so the
        # snapshot is consulted unless an ablation explicitly disables it.
        self._ablate_snapshot: bool = False

    # ------------------------------------------------------------------ #
    # Skill 1 — query_trend                                               #
    # ------------------------------------------------------------------ #

    def query_trend(self, query_hint: Optional[str] = None) -> list[WebSnippet]:
        query = self._compose_query(query_hint)
        return self.searcher.search(query)

    # ------------------------------------------------------------------ #
    # Skill 2 — interpret_trend                                           #
    # ------------------------------------------------------------------ #

    def interpret_trend(
        self,
        sources: Optional[Sequence[WebSnippet]] = None,
        *,
        use_cache: bool = True,
        directive_hint: Optional[str] = None,
    ) -> TrendInterpretation:
        # Per-call runtime window (set by engine from as_of_date) takes precedence
        # over the agent's default state.time_window. This matters because each
        # test user has its own anchor month — without using the override here,
        # the on-disk snapshot cache silently returns the wrong month's interp.
        effective_window = (
            getattr(self, "_runtime_time_window", None) or self.state.time_window
        )
        # Interpretation cache hit short-circuits LLM + web entirely.
        if use_cache and self.snapshot is not None:
            hit = self.snapshot.get(self.state.domain, effective_window)
            if hit is not None:
                log.info("trend cache HIT (%s, %s)", self.state.domain, effective_window)
                self.state.log.append(hit)
                return hit

        # Primary path — v5 SeasonTrendSnapshot. Carries TrendItems with
        # lifecycle, confidence, rationale, attributes, matched ASINs, and
        # per-source evidence. Convert via .to_interpretation() so downstream
        # Phase 2/3 code keeps working unchanged.
        season_snapshot = self._load_season_snapshot()
        if season_snapshot is not None:
            interp = season_snapshot.to_interpretation()
            # Override time_window with the caller's expected key so the
            # interpretation cache stays consistent across users that anchor
            # to different months within the same season.
            interp = interp.model_copy(update={"time_window": effective_window})
            self.state.log.append(interp)
            if self.snapshot is not None:
                self.snapshot.put(interp)
            # Stash the rich snapshot for any downstream consumer that wants
            # more than TrendInterpretation (e.g. matched_items ASIN inject).
            self._last_season_snapshot = season_snapshot
            return interp

        # Fallback — live web search snippets, narrated by the LLM.
        snippets = list(sources) if sources is not None else self.query_trend()
        if not snippets:
            interp = TrendInterpretation(
                domain=self.state.domain,
                time_window=effective_window,
                summary="No external evidence available.",
            )
        else:
            prompt = self.render(
                "interpret",
                sources=[{"url": s.url, "snippet": s.snippet} for s in snippets],
                domain=self.state.domain,
                time_window=effective_window,
            )
            raw = self._ask_structured(
                prompt,
                _RawInterp,
                system=self._system(
                    domain=self.state.domain,
                    time_window=effective_window,
                    vocabulary={k: sorted(v) for k, v in self.vocabulary.buckets.items()},
                ),
            )
            interp = TrendInterpretation(**raw.model_dump())

        self.state.log.append(interp)
        if self.snapshot is not None:
            self.snapshot.put(interp)
        return interp

    # ------------------------------------------------------------------ #
    # Season snapshot path (v5)                                          #
    # ------------------------------------------------------------------ #

    def _load_season_snapshot(self) -> Optional[SeasonTrendSnapshot]:
        """Load the v5 season snapshot for the active time window.

        Lookup order:
          1. Exact season match for the runtime time_window.
          2. If the runtime time_window is YYYY-MM, map to its season and
             try exact match for that season.
          3. Nearest available season (over the season files on disk).

        ``None`` when no season snapshot is present — caller falls back to
        live web search.
        """
        if self.processed_dir is None or self._ablate_snapshot:
            return None
        time_window = (
            getattr(self, "_runtime_time_window", None) or self.state.time_window
        )

        def _try(season_label: str) -> Optional[SeasonTrendSnapshot]:
            try:
                season_obj = parse_season(season_label)
            except ValueError:
                return None
            path = season_snapshot_path(
                self.processed_dir, self.state.domain, season_obj,
            )
            return load_season_snapshot(path)

        # (1) Exact season hit
        snap = _try(time_window)
        if snap is not None:
            log.info("season snapshot HIT exact (%s) — %d trends",
                     time_window, len(snap.trends))
            return snap

        # (2) YYYY-MM → containing season
        try:
            from datetime import date
            y, m = (int(x) for x in time_window.split("-"))
            containing = season_of(date(y, m, 1))
            snap = _try(containing.label)
            if snap is not None:
                log.info("season snapshot HIT containing-season (%s used for %s) — %d trends",
                         containing.label, time_window, len(snap.trends))
                return snap
        except (ValueError, AttributeError):
            pass

        # (3) Nearest available season
        nearest = self._find_nearest_season_file(time_window)
        if nearest is not None:
            snap = load_season_snapshot(nearest)
            if snap is not None:
                log.info("season snapshot NEAREST (%s used for %s) — %d trends",
                         nearest.name, time_window, len(snap.trends))
                return snap

        log.info("season snapshot MISS for %s", time_window)
        return None

    def _find_nearest_season_file(self, requested_window: str) -> Optional[Path]:
        """Pick the nearest v5 season file on disk.

        v5 file naming: ``fashion_trend_<year>_<SS|FW>.json`` where FW = AW.
        """
        if self.processed_dir is None:
            return None
        try:
            requested = parse_season(requested_window)
        except ValueError:
            try:
                from datetime import date
                y, m = (int(x) for x in requested_window.split("-"))
                requested = season_of(date(y, m, 1))
            except (ValueError, AttributeError):
                return None

        snap_dir = Path(self.processed_dir) / "trend_cache"
        if not snap_dir.exists():
            return None
        import re
        pattern = re.compile(r"^fashion_trend_(\d{4})_(SS|FW)\.json$")
        available: list[tuple[Season, Path]] = []
        for p in snap_dir.glob("fashion_trend_*.json"):
            m = pattern.match(p.name)
            if not m:
                continue
            year = int(m.group(1))
            half_file = m.group(2)
            half = "AW" if half_file == "FW" else "SS"
            available.append((Season(year=year, half=half), p))
        if not available:
            return None
        best = nearest_season(requested, [s for s, _ in available])
        if best is None:
            return None
        for s, p in available:
            if s.label == best.label:
                return p
        return None

    def set_runtime_time_window(self, time_window: Optional[str]) -> None:
        """Per-call override for the snapshot lookup key.

        Set by the orchestrator/engine when an ``as_of_date`` is passed so
        the season snapshot loaded for THIS recommend call matches the
        test point's month/quarter. ``None`` clears the override.
        """
        self._runtime_time_window = time_window

    # ------------------------------------------------------------------ #
    # Skill 3 — negotiation (Enhancement 5)                                #
    # ------------------------------------------------------------------ #

    def detect_tensions(
        self,
        directive: Directive,
        trend_interpretation: TrendInterpretation,
    ) -> list[TrendDirectiveTension]:
        """LLM-driven tension detection. Returns ``[]`` when nothing is worth raising."""
        class _Out(BaseModel):
            tensions: list[TrendDirectiveTension] = Field(default_factory=list)

        prompt = self.render(
            "detect_tension",
            directive=directive.model_dump(),
            trend_interpretation=trend_interpretation.model_dump(),
        )
        out = self._ask_structured(
            prompt,
            _Out,
            system=self._system(
                domain=self.state.domain,
                time_window=self.state.time_window,
                vocabulary={k: sorted(v) for k, v in self.vocabulary.buckets.items()},
            ),
        )
        return list(out.tensions)

    def challenge_directive(
        self,
        directive: Directive,
        tensions: list[TrendDirectiveTension],
        turn: int,
    ) -> NegotiationMessage:
        """Generate a single challenge message proposing a directive delta."""
        prompt = self.render(
            "challenge",
            directive=directive.model_dump(),
            tensions=[t.model_dump() for t in tensions],
            turn=turn,
        )
        msg = self._ask_structured(
            prompt,
            NegotiationMessage,
            system=self._system(
                domain=self.state.domain,
                time_window=self.state.time_window,
                vocabulary={k: sorted(v) for k, v in self.vocabulary.buckets.items()},
            ),
        )
        # Defensive: enforce direction in case the LLM scrambled the actor fields.
        return msg.model_copy(update={
            "from_agent": "trend",
            "to_agent": "expert",
            "message_type": "challenge" if msg.message_type not in {"challenge", "counter"} else msg.message_type,
            "turn": turn,
            "tensions": tensions,
        })

    # ------------------------------------------------------------------ #
    # Skill 4 — broadcast                                                  #
    # ------------------------------------------------------------------ #

    def broadcast(self, interp: TrendInterpretation, receivers: Sequence[str]) -> None:
        msg = Message(
            type=MessageType.BROADCAST,
            sender=self.id,
            receivers=list(receivers),
            payload=interp.model_dump(),
        )
        self.emit(msg)

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _compose_query(self, hint: Optional[str]) -> str:
        base = f"{self.state.domain} trend {self.state.time_window}"
        return f"{base} {hint}" if hint else base
