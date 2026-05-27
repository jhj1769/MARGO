"""JSON Lines persistence for AgentMemory.

Design choices:

    * **Append-only JSONL** — every event is one line. Concurrency-safe with
      simple ``a`` mode open; no locks needed for the single-process
      research workflow. Easy to git-ignore, easy to ``wc -l`` for growth
      curves, easy to ``jq`` for ad-hoc inspection.

    * **One file per entity** — ``memory/<agent_type>/<entity_id>.jsonl``.
      Lazy-loads on first read; keeps RAM bounded even with 50K items.

    * **In-memory cache per instance** — the in-process ``list`` mirrors
      the file so ``retrieve`` doesn't disk-roundtrip on every call. The
      cache is invalidated when the file timestamp changes (handles the
      "rebuild script ran, restart the engine" workflow without surprises).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from core.memory.base import AgentMemory, MemoryEvent

log = logging.getLogger(__name__)


class JSONLMemoryStore(AgentMemory):
    """File-backed append-only memory.

    Parameters
    ----------
    path
        File path (will be created if missing). Caller is responsible for
        choosing a per-entity path, e.g.
        ``memory/item/<item_id>.jsonl``.
    filter_fn
        Optional callable applied to each event during ``retrieve``. When
        ``None``, ``retrieve`` returns the most-recent ``top_k`` events.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        filter_fn=None,
    ) -> None:
        self.path = Path(path)
        self._cache: list[MemoryEvent] = []
        self._loaded_mtime: Optional[float] = None
        self._lock = threading.Lock()
        self._filter_fn = filter_fn

    # ------------------------------------------------------------------ #
    # Internal cache management                                          #
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        """Lazy-load cache from disk if file changed or never loaded."""
        if not self.path.exists():
            self._cache = []
            self._loaded_mtime = None
            return
        mtime = self.path.stat().st_mtime
        if self._loaded_mtime == mtime and self._cache:
            return
        events: list[MemoryEvent] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(MemoryEvent.model_validate_json(line))
                except Exception as e:  # noqa: BLE001
                    log.warning("skipping malformed memory line in %s: %s", self.path, e)
        self._cache = events
        self._loaded_mtime = mtime

    # ------------------------------------------------------------------ #
    # AgentMemory interface                                              #
    # ------------------------------------------------------------------ #

    def append(self, event: MemoryEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = event.model_dump_json()
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            # Keep cache hot
            self._cache.append(event)
            try:
                self._loaded_mtime = self.path.stat().st_mtime
            except FileNotFoundError:  # pragma: no cover
                self._loaded_mtime = None

    def retrieve(
        self,
        query: Optional[dict[str, Any]] = None,
        *,
        top_k: int = 20,
    ) -> list[MemoryEvent]:
        self._ensure_loaded()
        events = self._cache
        if self._filter_fn is not None and query is not None:
            events = [e for e in events if self._filter_fn(e, query)]
        # Most recent first
        return list(reversed(events[-top_k:])) if events else []

    def size(self) -> int:
        self._ensure_loaded()
        return len(self._cache)

    # ------------------------------------------------------------------ #
    # Convenience for tests / scripts                                    #
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        """Delete the file and empty cache. Used by tests."""
        with self._lock:
            if self.path.exists():
                self.path.unlink()
            self._cache = []
            self._loaded_mtime = None
