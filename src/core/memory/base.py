"""AgentMemory protocol — the *operational* interface shared by all 4 stakeholders.

Each ``BaseAgent`` instance optionally holds an ``AgentMemory``. The two
operations the orchestrator cares about are:

    * ``append(event)`` — record an event after an agent call completes (with
      enough context to be useful as evidence later).
    * ``retrieve(query)`` — pull a relevant slice **before** the next agent
      call, so the LLM never has to read full history.

The heterogeneous schemas (UserMemory, ItemMemory, etc.) live in
``schemas.py``. This module only defines the interface and a no-op
implementation so that agents work with or without memory wired up.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


class MemoryEvent(BaseModel):
    """Base envelope for an event written to an AgentMemory.

    Subclasses in ``schemas.py`` add their own typed fields. Keeping a
    common envelope means downstream tooling (TSV exports, audit logs, the
    web /trace endpoint) can treat all four memories uniformly.
    """

    event_type: str
    timestamp: float = Field(..., description="Unix epoch seconds when the event was created")
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentMemory(ABC):
    """Per-agent memory store.

    Implementations are responsible for both *append* (after each agent
    call) and *retrieve* (a relevant slice for the next call). Retrieval is
    intentionally schema-aware: the query dict's shape is decided by each
    subclass, so the User memory's ``retrieve(cohort_signature=...)`` and
    the Item memory's ``retrieve(viewer_cohort=...)`` can ask different
    questions of the same protocol.
    """

    @abstractmethod
    def append(self, event: MemoryEvent) -> None:
        """Persist one event. Must be cheap; called once per agent invocation."""

    @abstractmethod
    def retrieve(self, query: Optional[dict[str, Any]] = None, *, top_k: int = 20) -> list[MemoryEvent]:
        """Return at most ``top_k`` events relevant to ``query`` (most recent first).

        ``query=None`` means "give me the most recent slice with no
        additional filter" — useful for cold start.
        """

    @abstractmethod
    def size(self) -> int:
        """Total event count. Used for ablation diagnostics and growth curves."""


class NullMemory(AgentMemory):
    """No-op memory. Used when an agent is constructed without memory wired in.

    This is what makes memory backward-compatible — existing 87 tests
    instantiate agents without memory and continue to work because every
    memory call short-circuits here.
    """

    def append(self, event: MemoryEvent) -> None:  # pragma: no cover — trivial
        return

    def retrieve(self, query: Optional[dict[str, Any]] = None, *, top_k: int = 20) -> list[MemoryEvent]:
        return []

    def size(self) -> int:
        return 0
