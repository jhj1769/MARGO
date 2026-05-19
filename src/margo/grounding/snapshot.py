"""Snapshot cache for Trend Agent outputs.

The Trend Agent is the only MARGO agent that touches the live web — which
makes its outputs non-reproducible and (occasionally) noisy. To keep the
research pipeline reproducible we cache its interpretation per
``(domain, time_window)`` key and reuse it on subsequent runs.

Manual editing of cache files is encouraged when a hallucination is
spotted — the file format is plain JSON.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from margo.protocol.messages import TrendInterpretation

log = logging.getLogger(__name__)


class TrendSnapshotStore:
    """On-disk cache of TrendInterpretation objects."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, domain: str, time_window: str) -> Path:
        safe = "".join(c if c.isalnum() else "_" for c in time_window)
        return self.cache_dir / f"{domain}_{safe}.json"

    def get(self, domain: str, time_window: str) -> Optional[TrendInterpretation]:
        p = self._path(domain, time_window)
        if not p.exists():
            return None
        try:
            return TrendInterpretation.model_validate(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            log.exception("Corrupt trend snapshot at %s; ignoring.", p)
            return None

    def put(self, interp: TrendInterpretation) -> Path:
        p = self._path(interp.domain, interp.time_window)
        payload = interp.model_dump()
        payload["_cached_at"] = datetime.now(timezone.utc).isoformat()
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("trend snapshot cached → %s", p)
        return p
