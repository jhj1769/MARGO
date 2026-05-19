"""Message protocol primitives shared by every MARGO agent."""

from protocol.messages import (
    Directive,
    Message,
    MessageType,
    RankedItem,
    Rationale,
    RecommendationResult,
    TrendInterpretation,
    ValidationReport,
)
from protocol.router import MessageBus

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
