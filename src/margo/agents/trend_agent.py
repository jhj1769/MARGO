"""Trend Agent — turns raw evidence into a TrendInterpretation.

Two evidence sources are supported and tried in order:

1. **Pre-built Google Trends snapshot** — JSON dropped under
   ``processed/trend_cache/google_trends_<time_window>.json`` by
   ``scripts/build_trend_snapshot.py``. Loaded deterministically; the
   LLM only *interprets* the numbers, it does not invent them.

2. **Live web search snippets** — the original fallback when no
   snapshot is available (Tavily / SerpAPI / stub).

A persistent interpretation cache (separate from the raw snapshot)
keeps re-runs cheap and reproducible (see :mod:`sage.grounding.snapshot`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from margo.agents.base import BaseAgent
from margo.grounding.snapshot import TrendSnapshotStore
from margo.grounding.trend_snapshot_schema import TrendSnapshot
from margo.grounding.vocabulary import Vocabulary
from margo.protocol.messages import Message, MessageType, TrendInterpretation
from margo.trend_sources.web_search import WebSearcher, WebSnippet

log = logging.getLogger(__name__)


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
    """4th stakeholder — external context speaks back into the framework."""

    def __init__(
        self,
        domain: str,
        time_window: str,
        *,
        vocabulary: Optional[Vocabulary] = None,
        searcher: Optional[WebSearcher] = None,
        snapshot_store: Optional[TrendSnapshotStore] = None,
        gtrends_snapshot_dir: Optional[Path] = None,
        **kw,
    ) -> None:
        super().__init__(agent_id="trend", prompt_namespace="trend_agent", **kw)
        self.state = TrendState(domain=domain, time_window=time_window)
        self.vocabulary = vocabulary or Vocabulary({})
        self.searcher = searcher or WebSearcher()
        self.snapshot = snapshot_store
        self.gtrends_snapshot_dir = Path(gtrends_snapshot_dir) if gtrends_snapshot_dir else None

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
        # Interpretation cache hit short-circuits LLM + web entirely.
        if use_cache and self.snapshot is not None:
            hit = self.snapshot.get(self.state.domain, self.state.time_window)
            if hit is not None:
                log.info("trend cache HIT (%s, %s)", self.state.domain, self.state.time_window)
                self.state.log.append(hit)
                return hit

        # Preferred path: Google-Trends snapshot built offline by
        # ``scripts/build_trend_snapshot.py``. Deterministic, no hallucination.
        gt_snapshot = self._load_gtrends_snapshot()
        if gt_snapshot is not None:
            interp = self._interpret_gtrends_snapshot(gt_snapshot, directive_hint=directive_hint)
        else:
            snippets = list(sources) if sources is not None else self.query_trend()
            if not snippets:
                interp = TrendInterpretation(
                    domain=self.state.domain,
                    time_window=self.state.time_window,
                    summary="No external evidence available.",
                )
            else:
                prompt = self.render(
                    "interpret",
                    sources=[{"url": s.url, "snippet": s.snippet} for s in snippets],
                    domain=self.state.domain,
                    time_window=self.state.time_window,
                )
                raw = self._ask_structured(
                    prompt,
                    _RawInterp,
                    system=self._system(
                        domain=self.state.domain,
                        time_window=self.state.time_window,
                        vocabulary={k: sorted(v) for k, v in self.vocabulary.buckets.items()},
                    ),
                )
                interp = TrendInterpretation(**raw.model_dump())

        self.state.log.append(interp)
        if self.snapshot is not None:
            self.snapshot.put(interp)
        return interp

    # ------------------------------------------------------------------ #
    # Google Trends snapshot path                                        #
    # ------------------------------------------------------------------ #

    def _load_gtrends_snapshot(self) -> Optional[TrendSnapshot]:
        if self.gtrends_snapshot_dir is None:
            return None
        path = self.gtrends_snapshot_dir / f"google_trends_{self.state.time_window}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            snap = TrendSnapshot.model_validate(data)
            log.info("gtrends snapshot HIT (%s) — %s", path.name, snap.short_summary())
            return snap
        except Exception as e:
            log.warning("failed to load gtrends snapshot %s: %s", path, e)
            return None

    def _interpret_gtrends_snapshot(
        self,
        snap: TrendSnapshot,
        *,
        directive_hint: Optional[str] = None,
    ) -> TrendInterpretation:
        """Let the LLM narrate a deterministic Google Trends snapshot."""
        prompt = self.render(
            "interpret_gtrends",
            snapshot=snap.model_dump(),
            directive_hint=directive_hint or "",
        )
        raw = self._ask_structured(
            prompt,
            _RawInterp,
            system=self._system(
                domain=self.state.domain,
                time_window=self.state.time_window,
                vocabulary={k: sorted(v) for k, v in self.vocabulary.buckets.items()},
            ),
        )
        # Force the raw_sources field to carry the snapshot pointer so the
        # interpretation stays auditable back to its source.
        sources = [f"google_trends://{snap.domain}/{snap.time_window}/{snap.region}/{snap.snapshot_date}"]
        return TrendInterpretation(
            domain=raw.domain,
            time_window=raw.time_window,
            summary=raw.summary,
            keywords=raw.keywords,
            rising_attributes=raw.rising_attributes,
            raw_sources=sources,
        )

    # ------------------------------------------------------------------ #
    # Skill 3 — broadcast                                                  #
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
