"""Agentic Memory layer for MARGO (Heterogeneous Stakeholder Reasoning thesis).

Each of the 4 stakeholders gets its own memory shape, but they all share the
same protocol: ``AgentMemory`` — ``append`` an event after each agent call,
``retrieve`` a relevant slice before the next call.

Why a shared protocol with heterogeneous schemas?
    The thesis asserts that each stakeholder needs a *different* kind of
    reasoning. The memory schema reflects that (UserMemory tracks trajectory
    + dual-layer; ItemMemory tracks reception + claims; TrendMemory tracks
    predictions + reliability; ExpertMemory tracks directive outcomes).
    But the *operational interface* is uniform so the orchestrator and tests
    don't grow per-agent branches.
"""

from core.memory.base import AgentMemory, NullMemory, MemoryEvent
from core.memory.persistence import JSONLMemoryStore
from core.memory.schemas import (
    UserMemory,
    ItemMemory,
    TrendMemory,
    ExpertMemory,
    UserMemoryEvent,
    ItemReceptionEvent,
    TrendPredictionEvent,
    ExpertDirectiveOutcomeEvent,
)

__all__ = [
    "AgentMemory",
    "NullMemory",
    "MemoryEvent",
    "JSONLMemoryStore",
    "UserMemory",
    "ItemMemory",
    "TrendMemory",
    "ExpertMemory",
    "UserMemoryEvent",
    "ItemReceptionEvent",
    "TrendPredictionEvent",
    "ExpertDirectiveOutcomeEvent",
]
