"""Phase 1 — Agent Initialization.

This phase runs OFFLINE and is therefore exposed as a callable rather than
a node in the runtime lifecycle. For a research pipeline you would call
``initialize_user_agents`` once after preprocessing and persist the
resulting NL profiles to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from core.agents.user_agent import UserAgent
from adapters.llm import LLMClient

log = logging.getLogger(__name__)


def initialize_user_agents(
    user_histories: dict[str, list[str]],
    *,
    llm: LLMClient | None = None,
    out_path: Path | None = None,
) -> dict[str, str]:
    """Generate an NL profile per user.

    Parameters
    ----------
    user_histories:
        ``user_id → list[item_text]``.
    out_path:
        Optional JSON path to persist the result.
    """
    import json

    profiles: dict[str, str] = {}
    for uid, history in user_histories.items():
        agent = UserAgent(user_id=uid, history=history, llm=llm)
        try:
            profiles[uid] = agent.build_profile()
        except Exception as e:
            log.exception("profile build failed for user %s", uid)
            profiles[uid] = f"[fallback] history={history[:5]}; err={e}"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("user profiles → %s", out_path)
    return profiles


def batch_iter(items: Iterable, n: int) -> Iterable[list]:
    """Tiny helper to chunk an iterable (kept here to avoid an extra import)."""
    buf: list = []
    for x in items:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
