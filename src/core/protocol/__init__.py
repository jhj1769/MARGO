"""Message protocol primitives shared by every MARGO agent."""

from core.protocol.messages import (
    Directive,
    Message,
    MessageType,
    RankedItem,
    Rationale,
    RecommendationResult,
    TrendInterpretation,
    ValidationReport,
)
from core.protocol.router import MessageBus

__all__ = [
    "Directive",
    "Message",
    "MessageType",
    "RankedItem",
    "Rationale",
    "RecommendationResult",
    "TrendInterpretation",
    "ValidationReport",
    "MessageBus",
]
